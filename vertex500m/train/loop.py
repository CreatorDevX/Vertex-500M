"""Stage A / stage B training loops.

Phase-1 moves (from train.py) plus the training-QoL work:

* `--token-map` is REQUIRED for a real run (refuses instead of warning).
* Synthetic fallback goes to a fresh tempdir, not `/tmp/vertex500m_synth`.
* Stage B attaches DSpark BEFORE loading `--init` and initialises only the
  mtp.* parameters the checkpoint did not supply (stage-B resume keeps its
  trained drafter).
* EXACT RESUME (`--init ckpt --resume`): step counter, LR-schedule position
  (total/warmup restored from the checkpoint), all RNG streams, and the
  dataloader position (epoch number + in-epoch offset, rebuilt deterministically
  via per-epoch reseeding). A resumed run continues the identical data order --
  see `_BatchCursor`.
* PERIODIC CHECKPOINTS (`--save-every`, `--keep-last`): per-stage files
  `{stage}.step_{n}.pt` + `{stage}.latest.pt` (atomic writes), rotation of old
  step files, `{stage}.best.pt` on validation improvement. Signal/NaN/deadline
  saves land in `{stage}.latest.pt` whenever a checkpoint location exists.
* IN-LOOP VALIDATION (`--eval-every` + `--val-data`): held-out CE during stage
  A via eval.evaluate; best-val tracking.
* `--grad-clip` (global norm, GradScaler-aware: unscale every optimizer, clip
  once over the union, then step -- the documented multi-optimizer pattern),
  `--accum-steps` (micro-batches, loss scaled by 1/N).
* EARLY EXITS: `--target-loss` (EMA-based) and `--deadline-s` (wall clock).
  `run()` returns "done" | "target" | "deadline" so callers (and the pipeline)
  can tell a completed stage from a stopped-early one.
* NaN-loss gate: a non-finite loss saves `latest.pt` and fails fast instead of
  training on NaNs for hours.
* SIGINT/SIGTERM: save `latest.pt` (when a checkpoint location exists) and
  exit 128+signum, so an interrupted run is resumable.
* Every step record goes to `MetricsSink` (metrics.jsonl) as well as console.
"""

from __future__ import annotations

import os
import shutil
import signal
import tempfile
import time

import torch

from .. import amp as amp_mod
from ..config import PRESET
from ..factory import build_model, load_checkpoint_into, load_token_map_json
from ..optim import build_optimizer
from ..params import count
from ..quant import set_quant_mode
from .checkpoint import (
    ckpt_paths,
    resolve_ckpt_dir,
    restore_optimizer_state,
    restore_rng_state,
    save_checkpoint,
)
from .dist import seed_everything, setup_distributed, unwrap, wrap_ddp
from .losses import chunked_head_ce
from .metrics import MetricsSink
from .schedule import lr_at, set_lr


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
# Optimizer stepping, split for clipping/accumulation
# --------------------------------------------------------------------------- #
def backward_loss(loss, scaler) -> None:
    """Backward for one (possibly 1/N-scaled micro-batch) loss."""
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()


def finalize_step(optimizers, scaler, grad_clip: float | None = None):
    """Clip (optionally) + step + zero_grad. Returns (skipped, scale, grad_norm).

    With a GradScaler the clip MUST happen after `unscale_` and before `step`:
    clipping scaled gradients would clip at `max_norm * scale`. Each optimizer
    is unscaled individually (the documented multi-optimizer pattern); the norm
    is global over the union of grad-carrying params, which is exact because
    `build_optimizer` partitions parameters disjointly across optimizers.
    `grad_norm` is the pre-clip total norm (None when clipping is off).
    """
    before = scaler.get_scale() if scaler is not None else None
    grad_norm = None
    if grad_clip is not None:
        if scaler is not None:
            for o in optimizers:
                scaler.unscale_(o)
        params = [p for o in optimizers for g in o.param_groups
                  for p in g["params"] if p.grad is not None]
        if params:
            grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
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
    skipped = before is not None and after is not None and after < before
    return skipped, after, grad_norm


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


def optimizer_step(loss, optimizers, scaler, grad_clip: float | None = None):
    """Backward + step for one loss, honouring the GradScaler if there is one.

    Returns (skipped, scale). `skipped` is True when the scaler found inf/NaN in the
    gradients and threw the step away -- the whole point of scaling fp16 gradients, and
    a number that must be watched: if more than a few percent of steps skip, the loss
    scale is wrong for this model and `--amp fp16` is the wrong tool (bench.py --train
    reports the rate).
    """
    backward_loss(loss, scaler)
    skipped, scale, _ = finalize_step(optimizers, scaler, grad_clip)
    return skipped, scale


# --------------------------------------------------------------------------- #
# Batches with epoch tracking + resume fast-forward
# --------------------------------------------------------------------------- #
class _BatchCursor:
    """`next()` over a loader with epoch tracking for exact resume.

    `pos` counts batches consumed from the CURRENT iterator; `epoch` counts
    rollovers. Each epoch's iterator is created from a generator freshly seeded
    with `seed + epoch`, so the epoch's data order is a pure function of
    (seed, epoch) -- a resumed run rebuilds the identical order by restoring
    the epoch number, with no dependence on opaque generator internals.

    Why not save/restore the generator state: `iter(loader)` draws worker
    seeds eagerly while the shuffle permutation is drawn lazily on first
    `next()`, so a saved mid-epoch generator state can never be re-aligned by
    just calling `iter()` again -- the restored iterator draws a DIFFERENT
    permutation (pinned by a test after this bit exactly that). Reseeding per
    epoch sidesteps the DataLoader internals entirely: both sides perform the
    identical draw sequence from a known seed.

    `fast_forward(skip)` replays `skip` batches of the restored epoch. A
    mid-skip exhaustion (data changed size between runs) restarts the epoch
    loudly instead of hanging or crashing.
    """

    def __init__(self, loader, say, skip: int = 0, seed: int = 0,
                 start_epoch: int = 0, sampler=None):
        if len(loader.dataset) == 0:
            raise SystemExit(
                "[data] dataloader is EMPTY (no full windows). The corpus is "
                "too small for this --seq-len/--batch-size, or the split "
                "selected no shards.")
        self.loader = loader
        self.say = say
        self.seed = seed
        self.epoch = start_epoch
        self.sampler = sampler
        self.pos = 0
        self._new_iter()
        if skip:
            self.fast_forward(skip)

    @property
    def sampler_epoch(self):
        """This epoch, when a rank-aware sampler is in use (else None)."""
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            return self.epoch
        return None

    def _new_iter(self) -> None:
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            # DistributedSampler derives its per-epoch shuffle from
            # (seed, epoch) -- advancing it in lockstep with cursor.epoch
            # keeps every rank's Resume order identical to its original order.
            self.sampler.set_epoch(self.epoch)
        gen = getattr(self.loader, "generator", None)
        if gen is not None:
            gen.manual_seed(self.seed + self.epoch)
        self.it = iter(self.loader)
        self.pos = 0

    def fast_forward(self, skip: int) -> None:
        done = 0
        while done < skip:
            try:
                next(self.it)
                done += 1
            except StopIteration:
                self.say(f"[resume] WARNING: dataloader exhausted while "
                         f"fast-forwarding ({done}/{skip} batches) -- the data "
                         f"changed size since the checkpoint. Restarting this "
                         f"epoch from 0.")
                self._new_iter()
                done = 0
                break
        self.pos = done

    def next(self):
        try:
            b = next(self.it)
        except StopIteration:
            self.epoch += 1
            self._new_iter()
            try:
                b = next(self.it)
            except StopIteration:
                raise SystemExit("[data] dataloader became EMPTY mid-run -- "
                                 "this should be unreachable (see __init__).")
        self.pos += 1
        return b


# --------------------------------------------------------------------------- #
# Signals: save-then-exit so an interrupted run is resumable
# --------------------------------------------------------------------------- #
class _StopRequested(Exception):
    def __init__(self, signum):
        super().__init__(f"signal {signum}")
        self.signum = signum


_stop_state: dict = {}


def _poll_stop():
    """Pending signal number, else None. Only set while signals are armed."""
    return _stop_state.get("s")


def _install_signal_handlers() -> object:
    """Arm SIGINT/SIGTERM to record into `_stop_state`. Returns old handlers.

    Out-of-main-thread (or missing-signal platforms): leave everything alone.
    """
    try:
        old = (signal.signal(signal.SIGINT, lambda s, f: _stop_state.update(s=s)),
               signal.signal(signal.SIGTERM, lambda s, f: _stop_state.update(s=s)))
        return old
    except (ValueError, OSError, RuntimeError):
        return None


def _restore_signal_handlers(old) -> None:
    if not old:
        return
    try:
        signal.signal(signal.SIGINT, old[0])
        signal.signal(signal.SIGTERM, old[1])
    except (ValueError, OSError, RuntimeError):
        pass


# --------------------------------------------------------------------------- #
def _run_dspark(args, model, cfg, device, loader, optimizers, total: int,
                scaler=None, dist_ctx=None, start_step: int = 0, cursor=None,
                sink=None, save=None):
    """Stage B. The backbone's own loss is not differentiable here -- every backbone
    parameter is frozen, so nothing in its graph requires grad. DSpark is trained with
    its own objective: draft a block, score it against the FROZEN backbone. See
    dspark_loss.py for the derivation and for what is simplified.

    `save` is a `save(why, step_completed, pos)` callable for periodic/signal/nan
    saves (None disables them).
    """
    from ..dspark_loss import dspark_loss
    model.eval()                      # freeze BN/dropout-like behaviour; backbone frozen
    cursor = cursor or _BatchCursor(loader, print,
                                      seed=getattr(args, "seed", 0),
                                      sampler=getattr(loader, "sampler", None))
    t0 = time.time()
    peak = args.mtp_lr
    skips = 0
    say = (lambda *a: None) if (dist_ctx and not dist_ctx.is_main) else print
    say(f"[stage B] {total} steps from step {start_step}, DSpark only "
        f"(backbone frozen)")
    temperature = getattr(args, "temperature", 1.0)
    log_every = max(1, getattr(args, "log_every", 1))
    eval_batches = getattr(args, "eval_batches", 8)
    for step in range(start_step, total):
        sig = _poll_stop()
        if sig is not None:
            if save is not None:
                save("signal", step, cursor)
            raise _StopRequested(sig)
        batch = cursor.next()
        ids = batch["input_ids"].to(device)
        set_lr(optimizers, lr_at(step, total, peak, warmup=args.warmup))
        with amp_mod.amp_context(device_type=torch.device(device).type):
            loss, conf, stats = dspark_loss(model, ids, temperature=temperature)
        if loss is None:
            continue
        if not torch.isfinite(loss):
            if save is not None:
                save("nan", step, cursor)
            raise SystemExit(
                f"[stage B] non-finite dspark loss at step {step}; saved latest "
                f"and stopping. Investigate the data/AMP settings before resuming.")
        reset_peak_mem(device)
        skipped, scale = optimizer_step(loss, optimizers, scaler)
        skips += int(skipped)
        if sink is not None:
            sink.log({"step": step + 1, "split": "train", "stage": "dspark",
                      "loss": float(loss.item()),
                      "accept_now": float(stats.get("accept_now", 0)),
                      "elapsed_s": round(time.time() - t0, 1),
                      **({"scale": float(scale)} if scale is not None else {})})
        if step % log_every == 0:
            sync(device)
            extra = f"  scale {scale:.3g}" if scale is not None else ""
            say(f"  step {step:5d}/{total}  dspark_loss {loss.item():.4f}  "
                f"accept {stats.get('accept_now', 0):.2f}  {time.time()-t0:.0f}s{extra}")
        if save is not None:
            save("periodic", step + 1, cursor)

    if sink is not None:
        sink.log({"event": "stage_complete", "stage": "dspark",
                  "steps": total, "skips": skips})
    # the acceptance rate is the number that decides whether DSpark earns its params
    accs = []
    model.eval()
    # Must be inside the autocast region for the same reason the training loop is: the
    # caches were allocated in the AMP dtype by init_caches, and running the forwards in
    # fp32 against an fp16 cache makes attention see q as fp16 and kv as fp16 while the
    # surrounding activations are fp32 -- a dtype mismatch inside the einsum.
    with torch.no_grad(), amp_mod.amp_context(device_type=torch.device(device).type):
        for i, batch in enumerate(loader):
            if i >= eval_batches:
                break
            ids = batch["input_ids"].to(device)
            _, _, st = dspark_loss(model, ids, temperature=temperature)
            if st:
                accs.append(st["accept_now"])
    if accs:
        mean_acc = sum(accs) / len(accs)
        say(f"\n[acceptance] {mean_acc*100:.1f}% over {len(accs)} batches")
        say(f"  DSpark pays only above the measured break-even (~22% at 503.8M on the")
        say(f"  CPU bench; re-measure on your GPU). Below that, drop it.")
    if scaler is not None and skips:
        say(f"[scaler] {skips}/{total} steps skipped as inf/NaN "
            f"(final scale {scaler.get_scale():.3g})")
    return mean_acc if accs else 0.0


def _run_val_eval(args, core, data_dir, device, amp_mode, seq_len, say,
                  dist_ctx=None) -> tuple | None:
    """One held-out CE pass over the val split. Returns (ce, ntok) or None.

    Under DDP every rank evaluates the full val set (it is small) and the
    (nll, ntok) pair is all-reduced, so all ranks agree on the CE -- no rank
    idles at a barrier and no collective desynchronises.
    """
    from ..data import build_dataloader
    from ..eval import evaluate
    val_dir = getattr(args, "val_data", None) or data_dir
    val_split = getattr(args, "val_split", "val")
    pack = getattr(args, "pack", "fixed")
    try:
        vloader = build_dataloader(val_dir, seq_len,
                                   batch_size=args.batch_size, split=val_split,
                                   pack=pack, seed=args.seed, shuffle=False,
                                   num_workers=0)
    except FileNotFoundError as e:
        say(f"[eval] WARNING: {e} Skipping in-loop validation.")
        return None
    was_training = core.training
    core.eval()
    try:
        with torch.no_grad():
            ce, nll, ntok = evaluate(core, vloader, device,
                                     max_batches=getattr(args, "eval_batches", 8) or None,
                                     log_every=0, seq_desc="val", amp_mode=amp_mode)
    finally:
        if was_training:
            core.train()
    if dist_ctx is not None and dist_ctx.enabled:
        import torch.distributed as dist
        t = torch.tensor([nll, float(ntok)], dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        ce = float(t[0] / max(t[1], 1.0))
        ntok = int(t[1].item())
    return ce, ntok


def run(args):
    """Full setup + stage. Returns "done" | "target" | "deadline"."""
    global _stop_state
    _stop_state = {}
    old_handlers = _install_signal_handlers()
    try:
        return _run_inner(args)
    finally:
        _restore_signal_handlers(old_handlers)
        _stop_state = {}


def _run_inner(args):
    dist_ctx = setup_distributed(args)
    args._dist = dist_ctx
    say = (lambda *a: None) if not dist_ctx.is_main else print
    # Fully-seeded, rank-offset RNGs. The old code seeded only the weight init,
    # so every torchrun rank shuffled the data identically and DDP averaged
    # gradients over duplicated batches while looking perfectly healthy.
    seed_everything(args.seed + dist_ctx.rank)

    use_cuda = torch.cuda.is_available() and not args.cpu
    device = (f"cuda:{dist_ctx.local_rank}" if use_cuda else "cpu")
    device_t = torch.device(device)

    set_quant_mode(args.quant_mode)
    amp_mod.set_amp_mode(args.amp)
    amp_mod.warn_if_unsupported(args.amp, device_t.type)

    cfg = PRESET.tiny() if (args.smoke or getattr(args, "tiny", False)) else PRESET
    if args.stage == "dspark":
        cfg = cfg.with_dspark(1)

    # The Engram compressed token map is REQUIRED, not optional: with the identity map
    # the n-gram hashes address a table whose semantics do not match the tokenizer, and
    # every collision silently degrades retrieval. Produce it with
    #   python -m vertex500m.tokenize --map-only --save-map token_map.json
    # On a real run (real --data, not smoke/synthetic) proceeding without it is a
    # silent wrong answer, so refuse -- pass --allow-identity-map to override.
    token_map = None
    if args.token_map:
        token_map, cvs, collisions = load_token_map_json(args.token_map)
        cfg = cfg.with_(engram_compressed_vocab_size=cvs)
        say(f"[engram] compressed_vocab_size={cvs} collisions={collisions}")
    elif args.data and not args.smoke and not getattr(args, "allow_identity_map", False):
        raise SystemExit(
            "[engram] refusing to train on real data without --token-map: the "
            "identity map makes the n-gram table address the wrong strings and the "
            "run would train a silently degraded model. Produce the map with\n"
            "  python -m vertex500m.tokenize --map-only --save-map token_map.json\n"
            "and re-run with --token-map token_map.json (or pass "
            "--allow-identity-map if you really mean it).")
    else:
        say("[engram] WARNING: no --token-map given; using the IDENTITY map. The n-gram")
        say("         table will not match the tokenizer's strings. This is fine for a")
        say("         smoke test and WRONG for a real run.")
    # Per-rank seed. Sharing a seed across ranks would make every rank generate the SAME
    # synthetic corpus AND apply the same dropout-like perturbations; the model would
    # train on identical data N times and call it a speedup.
    model = build_model(cfg, token_map=token_map, device=device,
                        hc_variant=args.hc_variant,
                        seed=args.seed + dist_ctx.rank, say=say)

    optimizers = []
    pending_opt_state = None
    pending_scaler_state = None
    resumed_train = None
    if args.stage == "dspark":
        # Attach BEFORE loading: a stage-B checkpoint carries mtp.* weights, and
        # loading them into a backbone without mtp drops the trained drafter as
        # "unexpected" keys. See factory.load_checkpoint_into.
        model.init_caches(1, cfg.max_seq_len, device=device)
        model.attach_dspark().to(device)
    if args.init:
        ck, missing, unexpected = load_checkpoint_into(
            model, args.init, args.hc_variant, say=say)
        if args.resume and isinstance(ck, dict) and "optimizers" in ck:
            pending_opt_state = ck["optimizers"]
        pending_scaler_state = ck.get("scaler") if args.resume and isinstance(ck, dict) else None
        if args.resume and isinstance(ck, dict) and isinstance(ck.get("train"), dict):
            ts = ck["train"]
            if ts.get("stage", args.stage) != args.stage:
                say(f"[resume] WARNING: checkpoint's train state is for stage "
                    f"{ts.get('stage')!r}, this run is stage {args.stage!r}; "
                    f"starting fresh (weights/optimizer still loaded as before).")
            else:
                resumed_train = ts
                if "rng" in ck and isinstance(ck["rng"], dict):
                    try:
                        restore_rng_state(ck["rng"])
                        say("  restored RNG state")
                    except Exception as e:  # noqa: BLE001
                        say(f"  [resume] WARNING: could not restore RNG state "
                            f"({e}); continuing with fresh RNG.")
    if args.stage == "dspark":
        # Initialise ONLY the drafter parameters the checkpoint did not supply.
        # A fresh stage B (--init backbone.pt) is missing all of mtp.* and inits all
        # of it, exactly as before; a stage-B resume (--init full.pt) is missing none
        # and keeps its trained drafter instead of re-randomising it.
        from ..init_weights import init_new_dspark_params
        if args.init:
            mtp_missing = {m for m in missing if m.startswith("mtp.")}
            n_new = init_new_dspark_params(model, seed=args.seed, names=mtp_missing)
        else:
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
            restore_optimizer_state(o, st)
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

    # ---- QoL knobs (all default-off: without them this is the old loop) ----
    save_every = max(0, getattr(args, "save_every", 0))
    keep_last = max(0, getattr(args, "keep_last", 3))
    eval_every = max(0, getattr(args, "eval_every", 0))
    grad_clip = getattr(args, "grad_clip", None)
    if grad_clip is not None and grad_clip <= 0:
        raise SystemExit(f"--grad-clip must be positive, got {grad_clip}")
    accum = max(1, getattr(args, "accum_steps", 1))
    target_loss = getattr(args, "target_loss", None)
    deadline_s = getattr(args, "deadline_s", None)
    pack = getattr(args, "pack", "fixed")
    train_split = getattr(args, "train_split", "train")
    ckpt_dir = resolve_ckpt_dir(args)
    paths = ckpt_paths(ckpt_dir, args.stage) if ckpt_dir else None
    metrics_path = getattr(args, "metrics", None)
    if metrics_path is None and ckpt_dir:
        metrics_path = os.path.join(ckpt_dir, "metrics.jsonl")
    sink = MetricsSink(metrics_path if dist_ctx.is_main else None)
    if metrics_path and dist_ctx.is_main:
        say(f"[metrics] {metrics_path}")

    # ---- data ----
    data_dir = args.data
    if not data_dir:
        from ..data import synthetic_corpus
        data_dir = tempfile.mkdtemp(prefix="vertex500m_synth_")
        synthetic_corpus(data_dir, n_tokens=args.synthetic_tokens,
                         vocab_size=cfg.vocab_size)
        train_split = None                    # synthetic shards carry no split
    from ..data import build_dataloader, build_dataset
    seq_len = 64 if args.smoke else args.seq_len
    # Exact resume: the epoch number rebuilds the data order deterministically
    # (per-epoch reseeding in _BatchCursor); the cursor then fast-forwards to
    # the saved in-epoch position.
    start_epoch = int((resumed_train or {}).get("epoch", 0) or 0)
    skip_batches = int((resumed_train or {}).get("pos_in_epoch", 0) or 0)
    if resumed_train and (start_epoch or skip_batches):
        say(f"  restored dataloader position (epoch {start_epoch}, "
            f"{skip_batches} batches in)")
    dataset = build_dataset(data_dir, seq_len, split=train_split, pack=pack,
                            seed=args.seed)
    sampler = None
    if dist_ctx.enabled:
        from torch.utils.data.distributed import DistributedSampler
        # Rank-sharded sampling: each rank trains on a DISJOINT slice of every
        # epoch (the old shuffle-everything loader fed every rank the same
        # batches). Epoch advances in lockstep via cursor (set_epoch per
        # epoch), so resume restores each rank's slice identically.
        sampler = DistributedSampler(dataset, num_replicas=dist_ctx.world_size,
                                     rank=dist_ctx.rank, shuffle=True,
                                     seed=args.seed, drop_last=True)
    loader = build_dataloader(dataset=dataset, seq_len=seq_len,
                              batch_size=args.batch_size, seed=args.seed,
                              sampler=sampler)
    say(f"[data] {len(loader.dataset)} windows of {args.batch_size}x{seq_len} tokens"
        + (f"  pack={pack}" if pack != "fixed" else "")
        + (f"  split={train_split}" if train_split else ""))
    if pack == "docs":
        say(f"  [data] packing waste "
            f"{loader.dataset.stats['waste']*100:.2f}% (dropped partial tails)")
    say(f"[amp] {args.amp}"
        f"{' + GradScaler(init=' + format(args.init_scale, 'g') + ')' if scaler else ''}"
        f"  device={device}")
    if accum > 1:
        say(f"[accum] {accum} micro-batches/step "
            f"({accum*args.batch_size*seq_len/1e3:.1f}k tok/optimizer step)")
    if grad_clip is not None:
        say(f"[clip] global grad-norm clip at {grad_clip:g}")

    # ---- schedule position (resume-exactness: the CHECKPOINT owns these) ----
    # A larger --steps EXTENDS the run (the cosine tail continues past the
    # checkpoint's total); a smaller one is ignored with a note, so a typo can
    # never silently truncate a resumed run. Warmup always stays the
    # checkpoint's -- re-warming mid-run would spike the LR.
    total = args.steps
    warmup = args.warmup
    start_step = 0
    best_val = float("inf")
    if resumed_train:
        if resumed_train.get("seed") is not None \
                and resumed_train["seed"] != args.seed:
            say(f"[resume] WARNING: checkpoint was trained with seed "
                f"{resumed_train['seed']}, this run uses --seed {args.seed}: the "
                f"data order WILL differ from the original run.")
        ckpt_total = int(resumed_train.get("total_steps", total))
        if args.steps > ckpt_total:
            say(f"[resume] extending total {ckpt_total} -> {args.steps}")
            total = args.steps
        else:
            total = ckpt_total
        warmup = int(resumed_train.get("warmup", warmup))
        start_step = int(resumed_train.get("step", 0))
        best_val = float(resumed_train.get("best_val", float("inf")))
        if total != args.steps or warmup != args.warmup:
            say(f"[resume] schedule from checkpoint: total={total} warmup={warmup} "
                f"(ignoring --steps {args.steps} --warmup {args.warmup})")
        say(f"[resume] continuing {args.stage} at step {start_step}/{total}"
            + (f"  best_val={best_val:.4f}" if best_val < float("inf") else ""))
        if start_step >= total:
            say("[resume] checkpoint already reached its total; nothing to do.")
            sink.close()
            dist_ctx.cleanup()
            return "done"

    say(f"[train] {total} steps, {args.batch_size * seq_len} tok/micro-batch, "
        f"{total*args.batch_size*seq_len/1e6:.1f}M tokens")
    if dist_ctx.enabled:
        say(f"[ddp] world_size={dist_ctx.world_size}  (gradients are averaged)")

    # ---- checkpoint helpers (closures over loop state) ----
    step_files: list[str] = []

    def snapshot(step_completed: int, cursor) -> dict:
        return {"stage": args.stage, "step": step_completed,
                "total_steps": total, "warmup": warmup,
                "pos_in_epoch": cursor.pos, "epoch": cursor.epoch,
                "best_val": best_val, "seed": args.seed,
                "sampler_epoch": cursor.sampler_epoch}

    def save_step_file(step_completed: int, cursor) -> None:
        if paths is None or not dist_ctx.is_main:
            return
        dest = save_checkpoint(core, args, optimizers,
                               extra={"stage": args.stage, "steps": step_completed},
                               scaler=scaler,
                               train_state=snapshot(step_completed, cursor),
                               path=paths["step"](step_completed))
        if dest:
            step_files.append(dest)
            shutil.copyfile(dest, paths["latest"])
            while keep_last >= 0 and len(step_files) > keep_last:
                old = step_files.pop(0)
                try:
                    os.remove(old)
                except OSError:
                    pass

    def save_latest(step_completed: int, cursor, why: str) -> None:
        if paths is None or not dist_ctx.is_main:
            if dist_ctx.is_main:
                say(f"[{why}] no checkpoint location (--out/--ckpt-dir); "
                    f"progress since start/last save is lost.")
            return
        dest = save_checkpoint(core, args, optimizers,
                               extra={"stage": args.stage, "steps": step_completed},
                               scaler=scaler,
                               train_state=snapshot(step_completed, cursor),
                               path=paths["latest"])
        if dest:
            say(f"[{why}] saved {dest} at step {step_completed}/{total}")

    def save_dispatch(why: str, step_completed: int, cursor) -> None:
        if why == "periodic":
            if save_every and step_completed % save_every == 0:
                save_step_file(step_completed, cursor)
        else:
            save_latest(step_completed, cursor, why)

    def maybe_eval(step_completed: int, cursor) -> None:
        nonlocal best_val
        if not (eval_every and step_completed % eval_every == 0):
            return
        res = _run_val_eval(args, core, data_dir, device, args.amp, seq_len, say,
                            dist_ctx)
        if res is None:
            return
        ce, ntok = res
        improved = ce < best_val
        if improved:
            best_val = ce
        say(f"  [val] step {step_completed}/{total}  CE {ce:.4f}  "
            f"tokens {ntok}" + ("  <-- best" if improved else ""))
        sink.log({"step": step_completed, "split": "val", "stage": args.stage,
                  "val_ce": ce,
                  "val_tokens": ntok, "best_val": best_val})
        if improved and paths is not None and dist_ctx.is_main:
            dest = save_checkpoint(core, args, optimizers,
                                   extra={"stage": args.stage, "steps": step_completed},
                                   scaler=scaler,
                                   train_state=snapshot(step_completed, cursor),
                                   path=paths["best"])
            if dest:
                say(f"  [val] saved best {dest}")

    if args.stage == "dspark":
        cursor = _BatchCursor(loader, say, skip=skip_batches, seed=args.seed,
                              start_epoch=start_epoch, sampler=sampler)
        acc = _run_dspark(args, step_model, cfg, device, loader, optimizers, total,
                          scaler, dist_ctx, start_step=start_step, cursor=cursor,
                          sink=sink, save=save_dispatch)
        save_checkpoint(core, args, optimizers,
                        extra={"stage": "dspark", "accept": acc},
                        scaler=scaler,
                        train_state=snapshot(total, cursor))
        if save_every and paths is not None and args.out and dist_ctx.is_main:
            shutil.copyfile(args.out, paths["latest"])
        sink.close()
        dist_ctx.cleanup()
        return "done"

    t0 = time.time()
    tokens_per_opt_step = accum * args.batch_size * seq_len
    say(f"[train] {tokens_per_opt_step} tok/optimizer step")
    cursor = _BatchCursor(loader, say, skip=skip_batches, seed=args.seed,
                            start_epoch=start_epoch, sampler=sampler)
    model.train()
    skips = 0
    ema = None
    log_every = max(1, args.log_every)
    status = "done"
    for step in range(start_step, total):
        sig = _poll_stop()
        if sig is not None:
            save_latest(step, cursor, "signal")
            sink.close()
            dist_ctx.cleanup()
            raise SystemExit(128 + int(sig))
        if deadline_s is not None and time.time() - t0 > deadline_s:
            save_latest(step, cursor, "deadline")
            say(f"[deadline] wall-clock budget ({deadline_s:.0f}s) reached at step "
                f"{step}/{total}; saved latest. Re-run to resume.")
            status = "deadline"
            break
        lr = lr_at(step, total, args.muon_lr, warmup=warmup)
        set_lr(optimizers, lr)
        micro_sum = 0.0
        for _ in range(accum):
            batch = cursor.next()
            # non_blocking: the loader pins memory (see data/shards.py), so these
            # copies run async on the copy engine and overlap the previous step's
            # tail instead of stalling the compute stream.
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
                                       device_type=device_t.type) / accum
            del hidden
            if not torch.isfinite(loss):
                save_latest(step, cursor, "nan")
                raise SystemExit(
                    f"[train] non-finite loss at step {step}; saved latest and "
                    f"stopping. Lower --init-scale / LR or check the data before "
                    f"resuming.")
            backward_loss(loss, scaler)
            micro_sum += float(loss.item()) * accum
        loss_value = micro_sum / accum
        skipped, scale, gnorm = finalize_step(optimizers, scaler, grad_clip)
        skips += int(skipped)
        ema = loss_value if ema is None else 0.98 * ema + 0.02 * loss_value
        if step % log_every == 0:
            sync(device)
            el = time.time() - t0
            done = step - start_step + 1
            extra = f"  scale {scale:.3g}" if scale is not None else ""
            gn = f"  gnorm {float(gnorm):.2f}" if gnorm is not None else ""
            mem = peak_mem_gb(device)
            say(f"  step {step:5d}/{total}  loss {loss_value:.4f}  "
                f"lr {lr:.2e}  {el:.0f}s  "
                f"{done*tokens_per_opt_step/max(el,1e-9)/1e3:.1f}k tok/s"
                f"{f'  peak {mem:.2f} GB' if mem else ''}{extra}{gn}")
            sink.log({"step": step + 1, "split": "train", "stage": args.stage,
                      "loss": loss_value,
                      "lr": lr, "tok_s": done * tokens_per_opt_step / max(el, 1e-9),
                      "elapsed_s": round(el, 1),
                      **({"peak_gb": mem} if mem else {}),
                      **({"scale": float(scale)} if scale is not None else {}),
                      **({"grad_norm": float(gnorm)} if gnorm is not None else {})})
        if save_every and (step + 1) % save_every == 0:
            save_step_file(step + 1, cursor)
        maybe_eval(step + 1, cursor)
        if target_loss is not None and ema <= target_loss:
            say(f"[target] EMA loss {ema:.4f} <= {target_loss} at step {step + 1}; "
                f"stopping early.")
            status = "target"
            break
    if scaler is not None:
        say(f"[scaler] final scale {scaler.get_scale():.3g}, {skips}/{total} steps "
            f"skipped as inf/NaN"
            + ("  <-- TOO MANY, lower --init-scale" if skips > max(2, total // 20) else ""))

    sink.log({"event": "stage_complete", "stage": args.stage, "status": status,
              "steps": total, "skips": skips,
              **({"final_scale": float(scaler.get_scale())}
                 if scaler is not None else {})})
    save_checkpoint(core, args, optimizers, extra={"stage": "pre", "steps": total},
                    scaler=scaler, train_state=snapshot(total, cursor))
    if save_every and paths is not None and args.out and dist_ctx.is_main:
        shutil.copyfile(args.out, paths["latest"])
    sink.close()
    dist_ctx.cleanup()
    return status
