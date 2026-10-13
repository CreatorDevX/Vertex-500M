"""Train vertex500m: backbone pretraining, then DSpark post-training.

Two stages, per spec 12.4.

Stage A -- backbone pretraining. DSpark does not exist. Trains everything else.

Stage B -- DSpark post-training. Rebuild with num_nextn_predict_layers=1, load the
    backbone, attach DSpark, FREEZE the backbone, train only mtp.*. The backbone's
    logits must be bit-identical before and after a B step; that is asserted in
    test_checkpoint.py::test_dspark_freeze_leaves_backbone_bit_identical and NOT here,
    because the freeze happens before the optimizer is built and a B step in this
    script would need a teacher forward to compare against.

Mixed precision (`--amp fp16|bf16|off`)
---------------------------------------
Parameters, gradients and optimizer state stay fp32 in every mode; only compute is
lowered, via autocast. Two reasons that is not negotiable here: AdamW is built with
`eps=1e-20`, which is below fp16's smallest subnormal, and Muon's Newton-Schulz plus
SinkhornBalanced's 11 balancing rounds are fp32-reasoning loops.

`--amp fp16` attaches a `GradScaler`; `--amp bf16` does not (bf16's exponent range is
fp32's, so gradients cannot overflow). On an NVIDIA T4 -- amp bf16 is a trap: sm_75 has
FP16 tensor cores but no BF16 ones, so bf16 computes at roughly FP32 speed with doubled
activation bytes. `amp.warn_if_unsupported` says so at startup.

The scaler's state is saved in the checkpoint and restored by `--resume`, alongside the
optimizer state. Without that a resumed run restarts at scale 65536 and can burn several
hundred inf-skipped steps re-converging on a loss scale that its weights were not
trained at.

Usage:
    python -m vertex500m.train --smoke                       # tiny end-to-end check
    python -m vertex500m.train --data data/ --steps 3000 --out backbone.pt
    python -m vertex500m.train --amp fp16 --data data/ --steps 3000 --out backbone.pt
    python -m vertex500m.train --stage dspark --init backbone.pt --data data/ \
        --steps 400 --out full.pt

Multi-GPU (2xT4, no NVLink -- the allreduce runs over PCIe):
    torchrun --nproc_per_node 2 -m vertex500m.train --data data/ --steps 3000 \
        --out backbone.pt
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F

from . import amp as amp_mod
from .config import PRESET
from .engram import EngramLayout, NgramHashState, identity_token_map
from .init_weights import init_from_scratch
from .optim import build_optimizer
from .quant import set_quant_mode
from .params import count


def lr_at(step: int, total: int, peak: float, warmup: int = 2000,
          min_ratio: float = 0.1) -> float:
    """Linear warmup to `peak`, then cosine decay to `peak * min_ratio`."""
    if step < warmup:
        return peak * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    p = min(max(p, 0.0), 1.0)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def set_lr(optimizers, lr: float):
    """Scale every group's LR by its `lr_mult` relative to `lr`.

    `build_optimizer` assigns each group an absolute LR (Muon 0.02, AdamW 2.6e-4,
    embed 0.02, Engram table 0.1). The schedule must PRESERVE that partition -- a
    single scalar applied to every group collapses all four onto one value and hands
    AdamW's norm/sink/hc parameters a 77x-too-large step. So each group carries an
    `lr_mult` (its build-time LR relative to the schedule's base) and the scheduler
    multiplies by that, rather than overwriting the group's own value.
    """
    for o in optimizers:
        for g in o.param_groups:
            g["lr"] = lr * g.get("lr_mult", 1.0)


# --------------------------------------------------------------------------- #
# Timing / memory instrumentation
#
# train.py historically never synchronised the device, so its reported step time was
# however long it took to ENQUEUE the work, not to do it. Every mixed-precision number
# below -- tokens/s, peak memory -- is meaningless without this, which is why it comes
# before the AMP work rather than after.
# --------------------------------------------------------------------------- #
def sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def peak_mem_gb(device) -> float:
    if torch.device(device).type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024 ** 3


def reset_peak_mem(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()


# --------------------------------------------------------------------------- #
# Distributed
# --------------------------------------------------------------------------- #
class Dist:
    """Distributed context, or a single-process stand-in with the same surface.

    Rank 0 is the only rank that prints or saves; every rank runs the same number of
    steps so the process group never desynchronises at a collective.
    """

    def __init__(self, enabled=False, rank=0, world_size=1, local_rank=0):
        self.enabled, self.rank = enabled, rank
        self.world_size, self.local_rank = world_size, local_rank

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.enabled:
            torch.distributed.barrier()

    def cleanup(self) -> None:
        if self.enabled:
            torch.distributed.destroy_process_group()


def setup_distributed(args) -> Dist:
    """Join a torchrun process group if one is launched, else run single-process."""
    if getattr(args, "cpu", False) or os.environ.get("WORLD_SIZE", "1") == "1":
        return Dist()
    import torch.distributed as dist
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return Dist(True, dist.get_rank(), dist.get_world_size(), local_rank)


def wrap_ddp(model, dist_ctx: Dist, args):
    """DDP with `static_graph=True`, which this model needs and the default breaks.

    `embed.weight` is TIED: `VertexHead` is a view onto the same `nn.Embedding` that
    reads the input ids, so one Parameter is read in two places inside a single forward.
    That produces two autograd contributions. DDP's reducer fires its allreduce on the
    FIRST one it sees and then marks the parameter ready; the second contribution can
    arrive after the collective and be silently dropped, which trains a subtly wrong
    model that looks perfectly healthy. `static_graph=True` promises the graph does not
    change between iterations, which is true here, and lets the reducer accumulate every
    contribution before it fires. tests.py::test_ddp_tied_embedding_reduces_once pins
    this against a single-process reference.
    """
    if not dist_ctx.enabled:
        return model
    import torch.nn as nn
    device_ids = [dist_ctx.local_rank] if torch.cuda.is_available() else None
    return nn.parallel.DistributedDataParallel(
        model, device_ids=device_ids,
        # Stage B freezes the backbone, so every remaining trainable tensor is mtp.* and
        # every one of them is used. Turning this on would mask real wiring bugs.
        find_unused_parameters=False,
        static_graph=True,
    )


def unwrap(model):
    """The underlying Transformer, whether or not DDP wrapped it."""
    return model.module if hasattr(model, "module") else model


def chunked_ce(logits: torch.Tensor, targets: torch.Tensor, chunk: int = 1024,
               reduction: str = "mean") -> torch.Tensor:
    """Cross-entropy over positions without materialising a [B*S, V] fp32 log_softmax.

    `forward_full` returns [B, S, V] and the naive `logits.float().view(-1, V)` makes
    TWO more copies of it (the fp32 cast, then log_softmax inside cross_entropy). At
    S=8192, V=32000 that is ~3.1 GB of logits alone, on a 12 GB card. Chunking over
    positions bounds the transient to `chunk` rows; the backward is unchanged because
    every chunk still contributes to the same graph.

    Loss is the mean over all positions, identical to the unchunked call.
    """
    b, s, v = logits.shape
    flat = logits.reshape(b * s, v)
    tgt = targets.reshape(-1)
    if reduction == "sum":
        total = flat.new_zeros(())
        for i in range(0, flat.size(0), chunk):
            total = total + F.cross_entropy(flat[i:i + chunk].float(), tgt[i:i + chunk],
                                            reduction="sum")
        return total
    # 'mean': sum of per-chunk MEANS weighted by chunk size == one global mean, and
    # avoids dividing by a count that differs on the final partial chunk.
    total = flat.new_zeros(())
    n = 0
    for i in range(0, flat.size(0), chunk):
        end = min(i + chunk, flat.size(0))
        total = total + F.cross_entropy(flat[i:end].float(), tgt[i:end],
                                        reduction="sum")
        n += end - i
    return total / max(n, 1)


def _head_ce_chunk(h_c: torch.Tensor, t_c: torch.Tensor,
                   head_weight: torch.Tensor, amp_mode, device_type):
    """One position-chunk of head matmul + cross-entropy (see chunked_head_ce).

    The autocast region lives INSIDE here (not at the caller) so checkpoint replay
    re-executes the identical function: replay runs during backward, outside any
    ambient autocast, and without this the replay would evaluate the head in fp32
    while forward ran it in fp16 -- the sparse-attention forward/backward agreement
    lesson, applied to the head.
    """
    with amp_mod.amp_context(amp_mode, device_type):
        logits_c = F.linear(h_c, head_weight.float())
        return F.cross_entropy(logits_c.float(), t_c, reduction="sum")


def chunked_head_ce(hidden: torch.Tensor, targets: torch.Tensor,
                    head_weight: torch.Tensor, chunk: int = 1024,
                    amp_mode: str | None = None,
                    device_type: str = "cuda") -> torch.Tensor:
    """Cross-entropy over positions with the head matmul fused in per chunk.

    `forward_full` returns [B, S, V] and `chunked_ce` chunks only the loss, so the
    full logit tensor is still materialised -- at B=4, S=8192, V=32000 that is
    ~2.1 GB fp16 for a tensor the loss consumes chunk by chunk. This takes the
    normed hidden states [B, S, d] instead (see `forward_full(return_hidden=True)`)
    and runs head + cross-entropy one position-chunk at a time. The only [B,S,*]
    tensor retained is hidden itself (24x smaller than logits at V=32000, d=1344).

    Each chunk runs under gradient checkpointing (`use_reentrant=False`): forward
    keeps only the chunk's inputs and rebuilds the chunk during backward, so the
    transient stays bounded by one chunk instead of all of them. The backbone
    still replays exactly once per step (a single backward traverses it), and the
    per-chunk head recompute costs one extra head matmul (~1.6% of step FLOPs).
    DDP and the GradScaler see one loss and one backward, exactly as before -- no
    `no_sync` juggling, no per-chunk scaling.

    Bit-identical loss to `chunked_ce(forward_full(...))`: a linear layer over row
    slices computes the same per-row dots in the same order, and the per-chunk
    fp32 upcast before cross_entropy matches chunked_ce exactly.
    """
    from torch.utils.checkpoint import checkpoint as _ckpt
    b, s, d = hidden.shape
    flat_h = hidden.reshape(b * s, d)
    flat_t = targets.reshape(-1)
    n = flat_t.numel()
    total = flat_h.new_zeros(())
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        total = total + _ckpt(_head_ce_chunk, flat_h[lo:hi], flat_t[lo:hi],
                              head_weight, amp_mode, device_type,
                              use_reentrant=False)
    return total / max(n, 1)


def finish_optimizer_step(optimizers, scaler, before):
    """Optimizer step + zero_grad after the backward already ran.

    Split from `optimizer_step` for the fused head+CE path, whose backward runs
    chunk by chunk inside `chunked_head_ce`. `before` is the GradScaler scale
    captured before any backward ran. Returns (skipped, scale) like
    `optimizer_step`.
    """
    if scaler is not None:
        for o in optimizers:              # each step() unscales its own groups
            scaler.step(o)
        scaler.update()                   # exactly ONE update per iteration
    else:
        for o in optimizers:
            o.step()
    for o in optimizers:
        o.zero_grad(set_to_none=True)
    after = scaler.get_scale() if scaler is not None else None
    return (before is not None and after is not None and after < before), after


def optimizer_step(loss, optimizers, scaler):
    """Backward + step for one loss, honouring the GradScaler if there is one.

    Returns (skipped, scale). `skipped` is True when the scaler found inf/NaN in the
    gradients and threw the step away -- the whole point of scaling fp16 gradients, and
    a number that must be watched: if more than a few percent of steps skip, the loss
    scale is wrong for this model and `--amp fp16` is the wrong tool (bench.py --train
    reports the rate).
    """
    before = scaler.get_scale() if scaler is not None else None
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    return finish_optimizer_step(optimizers, scaler, before)


def _run_dspark(args, model, cfg, device, loader, optimizers, scaler=None,
                dist_ctx=None):
    """Stage B. The backbone's own loss is not differentiable here -- every backbone
    parameter is frozen, so nothing in its graph requires grad. DSpark is trained with
    its own objective: draft a block, score it against the FROZEN backbone. See
    dspark_loss.py for the derivation and for what is simplified."""
    from .dspark_loss import dspark_loss
    model.eval()                      # freeze BN/dropout-like behaviour; backbone frozen
    steps = args.steps
    it = iter(loader)
    t0 = time.time()
    peak = args.mtp_lr
    skips = 0
    say = (lambda *a: None) if (dist_ctx and not dist_ctx.is_main) else print
    say(f"[stage B] {steps} steps, DSpark only (backbone frozen)")
    for step in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        ids = batch["input_ids"].to(device)
        set_lr(optimizers, lr_at(step, steps, peak, warmup=args.warmup))
        with amp_mod.amp_context(device_type=torch.device(device).type):
            loss, conf, stats = dspark_loss(model, ids, temperature=args.temperature)
        if loss is None:
            continue
        reset_peak_mem(device)
        skipped, scale = optimizer_step(loss, optimizers, scaler)
        skips += int(skipped)
        if step % args.log_every == 0:
            sync(device)
            extra = f"  scale {scale:.3g}" if scale is not None else ""
            say(f"  step {step:5d}/{steps}  dspark_loss {loss.item():.4f}  "
                f"accept {stats.get('accept_now', 0):.2f}  {time.time()-t0:.0f}s{extra}")

    # the acceptance rate is the number that decides whether DSpark earns its params
    accs = []
    model.eval()
    # Must be inside the autocast region for the same reason the training loop is: the
    # caches were allocated in the AMP dtype by init_caches, and running the forwards in
    # fp32 against an fp16 cache makes attention see q as fp16 and kv as fp16 while the
    # surrounding activations are fp32 -- a dtype mismatch inside the einsum.
    with torch.no_grad(), amp_mod.amp_context(device_type=torch.device(device).type):
        for i, batch in enumerate(loader):
            if i >= args.eval_batches:
                break
            ids = batch["input_ids"].to(device)
            _, _, st = dspark_loss(model, ids, temperature=args.temperature)
            if st:
                accs.append(st["accept_now"])
    if accs:
        mean_acc = sum(accs) / len(accs)
        say(f"\n[acceptance] {mean_acc*100:.1f}% over {len(accs)} batches")
        say(f"  DSpark pays only above the measured break-even (~22% at 503.8M on the")
        say(f"  CPU bench; re-measure on your GPU). Below that, drop it.")
    if scaler is not None and skips:
        say(f"[scaler] {skips}/{steps} steps skipped as inf/NaN "
            f"(final scale {scaler.get_scale():.3g})")
    return mean_acc if accs else 0.0


def _save(model, args, optimizers, extra: dict | None = None, scaler=None) -> None:
    """Save weights + optimizer state.

    Optimizer state matters: Muon's momentum buffer and the tables' Sinkhorn momentum
    are not reconstructible, so a checkpoint without them cannot resume training
    faithfully -- it restarts both. `freqs_cis` is excluded because it is a pure
    function of the config and is rebuilt on load.

    The GradScaler's state matters for the same reason and is saved for the same
    reason: a resumed run that restarts at scale 65536 spends its first hundreds of steps
    stepping on the floor waiting for the scale to back off, and if the model turned out
    to need a much smaller scale than 65536 it never gets there.
    """
    if not args.out or (getattr(args, "_dist", None) and not args._dist.is_main):
        return
    sd = {k: v for k, v in model.state_dict().items() if "freqs_cis" not in k}
    payload = {"model": sd,
               "optimizers": [{"state": o.state_dict(), "param_groups": o.param_groups}
                              for o in optimizers],
               "config": {"hc_variant": getattr(model, "hc_variant", "mhc"),
                          "amp": amp_mod.get_amp_mode()},
               **(extra or {})}
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, args.out)
    n = sum(v.numel() for v in sd.values())
    print(f"[save] {args.out} ({n/1e6:.1f}M weights"
          f"{', optimizer state included' if optimizers else ''}"
          f"{', scaler state included' if scaler is not None else ''})")


def run(args):
    dist_ctx = setup_distributed(args)
    args._dist = dist_ctx
    say = (lambda *a: None) if not dist_ctx.is_main else print

    use_cuda = torch.cuda.is_available() and not args.cpu
    device = (f"cuda:{dist_ctx.local_rank}" if use_cuda else "cpu")
    device_t = torch.device(device)

    set_quant_mode(args.quant_mode)
    amp_mod.set_amp_mode(args.amp)
    amp_mod.warn_if_unsupported(args.amp, device_t.type)

    cfg = PRESET.tiny() if args.smoke else PRESET
    if args.stage == "dspark":
        cfg = cfg.with_dspark(1)

    layout = EngramLayout(cfg)
    # The Engram compressed token map is REQUIRED, not optional: with the identity map
    # the n-gram hashes address a table whose semantics do not match the tokenizer, and
    # every collision silently degrades retrieval. Produce it with
    #   python -m vertex500m.tokenize --map-only --save-map token_map.json
    token_map = identity_token_map(cfg.vocab_size)
    if args.token_map:
        import json
        d = json.loads(open(args.token_map, encoding="utf-8").read())
        token_map = d["token_map"]
        cfg = cfg.with_(engram_compressed_vocab_size=d["engram_compressed_vocab_size"])
        say(f"[engram] compressed_vocab_size={d['engram_compressed_vocab_size']} "
            f"collisions={d.get('collisions', '?')}")
    else:
        say("[engram] WARNING: no --token-map given; using the IDENTITY map. The n-gram")
        say("         table will not match the tokenizer's strings. This is fine for a")
        say("         smoke test and WRONG for a real run.")
    hashes = NgramHashState(cfg, layout, token_map, cfg.engram_compressed_vocab_size)
    # Per-rank seed. Sharing a seed across ranks would make every rank generate the SAME
    # synthetic corpus AND apply the same dropout-like perturbations; the model would
    # train on identical data N times and call it a speedup.
    model = init_from_scratch(cfg, engram_layout=layout, engram_hashes=hashes,
                              hc_variant=args.hc_variant, seed=args.seed + dist_ctx.rank)
    model.to(device)

    optimizers = []
    pending_opt_state = None
    pending_scaler_state = None
    if args.init:
        ck = torch.load(args.init, map_location=device, weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        # A checkpoint saved with a different hc variant has a DIFFERENT projection
        # width (24 vs 32 at hc=4), so load_state_dict raises a wall of per-tensor size
        # mismatches. Check the recorded variant FIRST and say what actually went wrong.
        saved_variant = (ck.get("config", {}) or {}).get("hc_variant") \
            if isinstance(ck, dict) and "model" in ck else None
        if saved_variant and saved_variant != args.hc_variant:
            raise SystemExit(
                f"checkpoint {args.init} was trained with hc_variant={saved_variant!r} "
                f"but --hc-variant is {args.hc_variant!r}. The two weight spaces are not "
                f"related by any invertible map -- mHC-lite is a reparameterization of "
                f"H_res, not a conversion (mHC-lite.md 1), so there is no bridge. "
                f"Retrain, or pass --hc-variant {saved_variant}.")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if unexpected:
            say(f"  [warn] unexpected keys: {list(unexpected)[:5]}")
        say(f"  loaded {args.init} (missing={len(missing)} unexpected={len(unexpected)})")
        if args.resume and isinstance(ck, dict) and "optimizers" in ck:
            pending_opt_state = ck["optimizers"]
        pending_scaler_state = ck.get("scaler") if args.resume and isinstance(ck, dict) else None
    if args.stage == "dspark":
        model.init_caches(1, cfg.max_seq_len, device=device)
        model.attach_dspark().to(device)
        # attach_dspark() creates main_proj / markov_head / confidence_head AFTER
        # init_from_scratch() has already run, so they hold PyTorch module defaults --
        # and nn.Embedding defaults to N(0,1), which dspark.py adds straight to the
        # logits. Initialize them now, before the freeze and before the optimizer sees
        # them. See init_weights.init_new_dspark_params for why this is not a second
        # call to init_weights.
        from .init_weights import init_new_dspark_params
        n_new = init_new_dspark_params(model, seed=args.seed)
        say(f"[stage B] initialized {n_new} newly-attached DSpark tensors")

    c = count(cfg, args.hc_variant)
    say(f"[cfg] {c.total/1e6:.2f}M total, {c.active/1e6:.1f}M active, "
        f"{c.gflop_fwd_bwd:.3f} GFLOP/token, hc={args.hc_variant}")

    if args.stage == "dspark":
        for n, p in model.named_parameters():
            p.requires_grad_(n.startswith("mtp."))
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        say(f"[stage B] trainable (mtp only): {trainable/1e6:.2f}M")

    optimizers = build_optimizer(model, muon_lr=args.muon_lr, adamw_lr=args.adamw_lr,
                                 embed_lr=args.embed_lr, weight_decay=args.weight_decay,
                                 embed_optimizer=args.embed_optimizer)
    if not optimizers:
        raise RuntimeError("no trainable parameters -- check the freeze logic")
    # Freeze first, THEN build the optimizer (correct order) and record each group's
    # build-time LR as an `lr_mult` so set_lr scales the partition instead of
    # collapsing it.
    # build_optimizer already set each group's `lr_mult`; stage B rescales it against
    # the mtp schedule base because only mtp.* parameters exist in that stage.
    if args.stage == "dspark":
        for o in optimizers:
            for g in o.param_groups:
                g["lr_mult"] = g["lr"] / args.mtp_lr
    if pending_opt_state:
        for o, st in zip(optimizers, pending_opt_state):
            o.load_state_dict(st)
        say(f"  restored optimizer state for {len(optimizers)} optimizer(s)")

    scaler = amp_mod.make_scaler(args.amp, device_t.type, init_scale=args.init_scale)
    if scaler is not None and pending_scaler_state:
        scaler.load_state_dict(pending_scaler_state)
        say(f"  restored GradScaler state (scale {scaler.get_scale():.3g})")

    if args.compile:
        # Compile the module, THEN wrap in DDP: compiling the DDP wrapper would trace
        # the reducer's bookkeeping into the graph.
        model = torch.compile(model, dynamic=False)
        say("[compile] torch.compile(dynamic=False)")

    model = wrap_ddp(model, dist_ctx, args)
    step_model = model
    core = unwrap(model)

    # data
    data_dir = args.data or "/tmp/vertex500m_synth"
    if not args.data:
        from .data import synthetic_corpus
        synthetic_corpus(data_dir, n_tokens=args.synthetic_tokens,
                         vocab_size=cfg.vocab_size)
    from .data import build_dataloader
    seq_len = 64 if args.smoke else args.seq_len
    loader = build_dataloader(data_dir, seq_len, batch_size=args.batch_size)
    say(f"[data] {len(loader.dataset)} windows of {args.batch_size}x{seq_len} tokens")
    say(f"[amp] {args.amp}"
        f"{' + GradScaler(init=' + format(args.init_scale, 'g') + ')' if scaler else ''}"
        f"  device={device}")

    steps = args.steps
    it = iter(loader)
    t0 = time.time()
    tokens_per_step = args.batch_size * seq_len
    say(f"[train] {steps} steps, {tokens_per_step} tok/step, "
        f"{steps*tokens_per_step/1e6:.1f}M tokens")
    if dist_ctx.enabled:
        say(f"[ddp] world_size={dist_ctx.world_size}  (gradients are averaged)")

    if args.stage == "dspark":
        acc = _run_dspark(args, step_model, cfg, device, loader, optimizers, scaler,
                          dist_ctx)
        _save(core, args, optimizers, extra={"stage": "dspark", "accept": acc},
              scaler=scaler)
        dist_ctx.cleanup()
        return

    model.train()
    skips = 0
    log_every = max(1, args.log_every)
    for step in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        lr = lr_at(step, steps, args.muon_lr, warmup=args.warmup)
        set_lr(optimizers, lr)
        # non_blocking: the loader pins memory (see data.py), so these copies run
        # async on the copy engine and overlap the previous step's tail instead of
        # stalling the compute stream.
        ids = batch["input_ids"].to(device, non_blocking=True)
        tgt = batch["targets"].to(device, non_blocking=True)
        reset_peak_mem(device)
        with amp_mod.amp_context(args.amp, device_t.type):
            core.init_caches(ids.size(0), ids.size(1), device=device)
            # Hidden states, not logits: chunked_head_ce runs the head itself one
            # position-chunk at a time, so [B,S,V] is never materialised (see
            # chunked_head_ce). DDP-safe: still routed through the wrapper's forward.
            hidden, _ = step_model(ids, use_checkpoint=not args.no_ckpt,
                                   full_logits=True, return_hidden=True)
            loss = chunked_head_ce(hidden, tgt, core.embed.weight,
                                   chunk=args.loss_chunk, amp_mode=args.amp,
                                   device_type=device_t.type)
        del hidden
        skipped, scale = optimizer_step(loss, optimizers, scaler)
        skips += int(skipped)
        if step % log_every == 0:
            sync(device)
            el = time.time() - t0
            done = step + 1
            extra = f"  scale {scale:.3g}" if scale is not None else ""
            mem = peak_mem_gb(device)
            say(f"  step {step:5d}/{steps}  loss {loss.item():.4f}  "
                f"lr {lr:.2e}  {el:.0f}s  "
                f"{done*tokens_per_step/max(el,1e-9)/1e3:.1f}k tok/s"
                f"{f'  peak {mem:.2f} GB' if mem else ''}{extra}")
    if scaler is not None:
        say(f"[scaler] final scale {scaler.get_scale():.3g}, {skips}/{steps} steps "
            f"skipped as inf/NaN"
            + ("  <-- TOO MANY, lower --init-scale" if skips > max(2, steps // 20) else ""))

    _save(core, args, optimizers, extra={"stage": "pre", "steps": steps}, scaler=scaler)
    dist_ctx.cleanup()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["pre", "dspark"], default="pre")
    ap.add_argument("--data", default=None, help="dir of uint16 .bin shards")
    ap.add_argument("--init", default=None, help="checkpoint to load")
    ap.add_argument("--resume", action="store_true",
                    help="also restore optimizer state from --init (it is saved; this "
                         "opts into loading it)")
    ap.add_argument("--token-map", default=None,
                    help="JSON from tokenize.py --save-map (REQUIRED for a real run)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny CPU end-to-end")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", default=None, choices=["off", "fp16", "bf16"],
                    help="autocast dtype for COMPUTE only; weights, gradients and "
                         "optimizer state stay fp32 in every mode. fp16 also attaches a "
                         "GradScaler. On a T4 (sm_75) use fp16 -- bf16 has no tensor "
                         "cores there. Default: fp16 when CUDA is available (and --cpu "
                         "is not given), else off, so the documented recommendation "
                         "is what you get without passing a flag.")
    ap.add_argument("--init-scale", type=float, default=65536.0,
                    help="GradScaler starting loss scale (fp16 only). Lower it if "
                         "bench.py --train reports a high inf-skip rate.")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the module (dynamic=False). The forward already "
                         "traces into a single graph; the backward's per-layer graph "
                         "breaks are fixed, but verify loss before trusting a run.")
    ap.add_argument("--hc-variant", default="mhc", choices=["mhc", "mhc-lite"],
                    help="hyper-connection parameterization; mhc-lite is exact by "
                         "construction (Birkhoff) but has no checkpoint bridge to mHC")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=2048,
                    help="training window length. Defaults to 2048, not cfg.max_seq_len "
                         "(32768): the fp32 logit tensor scales with S*V and 32768 does "
                         "not fit. Longer contexts need checkpointing and a bigger card.")
    ap.add_argument("--loss-chunk", type=int, default=1024,
                    help="positions per CE chunk; bounds the transient fp32 log_softmax")
    ap.add_argument("--synthetic-tokens", type=int, default=200_000)
    ap.add_argument("--quant-mode", default="FAKE", choices=["FAKE", "NONE"])
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--adamw-lr", type=float, default=2.6e-4)
    ap.add_argument("--mtp-lr", type=float, default=0.02,
                    help="stage-B schedule base; scales the mtp groups' lr_mult")
    ap.add_argument("--embed-lr", type=float, default=0.02)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--embed-optimizer", default="adamw", choices=["adamw", "sinkhorn"])
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--no-ckpt", action="store_true")
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--eval-batches", type=int, default=8)
    args = ap.parse_args()
    if args.amp is None:
        args.amp = "fp16" if (torch.cuda.is_available() and not args.cpu) else "off"
        if args.amp == "fp16":
            print("[amp] defaulting to --amp fp16 (CUDA present; pass --amp "
                  "explicitly to override)")
    if args.smoke:
        args.steps = min(args.steps, 5)
        args.cpu = True
        args.seq_len = 64
    run(args)


if __name__ == "__main__":
    main()
