"""Decode/throughput benchmark: base backbone vs DSpark speculative decoding.

Everything uses RANDOM weights -- the point is throughput, not quality. That matters
for the honesty of the result: an untrained model has ~0% draft acceptance, so a naive
"DSpark is faster" claim would be false. This measures the parts that ARE meaningful
with random weights, and states the rest as a cost model:

  * base   : tokens/s decoding one token per backbone forward.
  * draft  : the drafter's own cost per round.
  * verify : cost of one backbone forward over the whole draft block.
  * break-even acceptance a*: below this, speculative decoding LOSES to plain decoding.

a* is derived from the MEASURED cost ratio, not assumed. Above a*, the projected
throughput table shows what a trained model would need to achieve for DSpark to pay.

Decode is stepped SEQUENTIALLY from the end of the prefill. This matters: the CSA2
compressor carries a mutable `PoolState` whose invariant is count == start_pos % ratio,
so jumping `start_pos` forward (as a naive microbenchmark does) trips the desync assert
in attention.py. That assert is load-bearing -- it is how a replayed segment or a
re-used cache gets caught -- so the benchmark obeys it rather than working around it.

TRAINING BENCHMARK (`--train`)
------------------------------
The decode benchmark above cannot tell you whether mixed precision helped, because
training is a completely different memory regime: it keeps the activation graph for the
backward pass, and at T=8192 that graph dwarfs the weights. `--train` measures
tokens/second and peak allocated memory for a full fwd+bwd+optimizer step, in each AMP
mode, at the shapes you actually intend to train at.

That is also the only place the GradScaler's inf-skip rate is visible. A skip rate above
~5% means the loss scale is wrong for this model and `--amp fp16` is the wrong tool --
watch the `skipped` column before committing to a long run.

Run:  python -m vertex500m.bench --device cpu --tiny
      python -m vertex500m.bench --device cuda --train --seq-len 8192 --batch-size 2
"""

from __future__ import annotations

import argparse
import time

import torch

from .amp import amp_context, make_scaler, set_amp_mode, warn_if_unsupported
from .config import PRESET
from .engram import EngramLayout, NgramHashState, identity_token_map
from .init_weights import init_from_scratch
from .optim import build_optimizer
from .params import count
from .quant import set_quant_mode


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def timeit(fn, device, warmup=2, iters=8):
    for _ in range(warmup):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / iters


def build(cfg, hc_variant="mhc"):
    layout = EngramLayout(cfg)
    hashes = NgramHashState(cfg, layout, identity_token_map(cfg.vocab_size),
                            cfg.engram_compressed_vocab_size)
    return init_from_scratch(cfg, engram_layout=layout, engram_hashes=hashes,
                             hc_variant=hc_variant)


@torch.no_grad()
def bench_base(model, device, prompt_len, decode_tokens, iters):
    """Sequential single-token decode: the honest baseline."""
    cfg = model.cfg
    need = prompt_len + decode_tokens + 2
    assert need <= cfg.max_seq_len, f"need {need} > max_seq_len {cfg.max_seq_len}"
    prompt = torch.randint(0, cfg.vocab_size, (1, prompt_len), device=device)

    def run():
        model.init_caches(1, need, device=device)
        logits, _ = model(prompt, start_pos=0)          # prefill
        for t in range(decode_tokens):
            nxt = logits.argmax(-1, keepdim=True)
            logits, _ = model(nxt, start_pos=prompt_len + t)
        return logits

    t = timeit(run, device, iters=iters)
    return t, decode_tokens / t


@torch.no_grad()
def bench_base_prefill(model, device, prompt_len, iters):
    cfg = model.cfg
    prompt = torch.randint(0, cfg.vocab_size, (1, prompt_len), device=device)

    def run():
        model.init_caches(1, cfg.max_seq_len, device=device)
        model(prompt, start_pos=0)
    return timeit(run, device, iters=iters)


@torch.no_grad()
def bench_dspark_costs(model, device, prompt_len, decode_tokens, iters):
    """Time the draft forward and the block verify forward INDEPENDENTLY.

    Each component is timed from its own freshly-consistent state (a fresh prefill), so
    there is no cross-iteration state threading and no PoolState desync. This measures
    the marginal cost of each piece, which is what the cost model needs; the round cost
    is then draft + verify (they are independent forwards, not sequential state).

    The CSA2 compressor's PoolState invariant (count == start_pos % ratio) means you
    cannot jump start_pos forward arbitrarily -- so each timed closure re-runs the full
    prefill and decodes to its anchor position. The prefill is subtracted out where it
    matters, or amortized by decoding enough tokens that the steady-state per-token cost
    dominates.
    """
    cfg = model.cfg
    block = cfg.dspark_block_size + 1
    prompt = torch.randint(0, cfg.vocab_size, (1, prompt_len), device=device)
    nxt_tok = torch.randint(0, cfg.vocab_size, (1, 1), device=device)

    # ---- draft forward marginal cost ----
    # baseline: prefill + 1 decode (to reach the anchor), no drafting
    def base_step():
        model.init_caches(1, cfg.max_seq_len, device=device)
        logits, _ = model(prompt, start_pos=0)
        model(nxt_tok, start_pos=prompt_len)
        return logits
    base_step_t = timeit(base_step, device, warmup=1, iters=iters)

    # with drafting: same, plus one forward_spec at the anchor
    def draft_step():
        model.init_caches(1, cfg.max_seq_len, device=device)
        logits, _ = model(prompt, start_pos=0)
        logits, mh = model(nxt_tok, start_pos=prompt_len)
        model.forward_spec(logits.argmax(-1, keepdim=True), mh[:, -1:, :],
                           prompt_len + 1)
        return logits
    draft_step_t = timeit(draft_step, device, warmup=1, iters=iters)
    draft_t = max(draft_step_t - base_step_t, 0.0)

    # ---- verify forward: a block of `block` positions ----
    blk_ids = torch.randint(0, cfg.vocab_size, (1, block), device=device)
    def verify_step():
        model.init_caches(1, cfg.max_seq_len, device=device)
        logits, _ = model(prompt, start_pos=0)
        model(nxt_tok, start_pos=prompt_len)
        model(blk_ids, start_pos=prompt_len + 1)      # decode `block` positions
        return logits
    verify_step_t = timeit(verify_step, device, warmup=1, iters=iters)
    verify_t = max(verify_step_t - base_step_t, 0.0)

    return draft_t, verify_t


def break_even_acceptance(draft_t, verify_t, base_t_per_token, block_size):
    """Acceptance fraction a* where speculative == plain, from measured costs.

    Plain decode of (1 + a*B) tokens costs (1 + a*B) * base_t.
    One speculative round costs (draft_t + verify_t) and emits (1 + a*B) tokens.
    Speculative wins when base_t * (1 + a*B) > draft_t + verify_t, i.e.

        a* > ((draft_t + verify_t) / base_t - 1) / B
    """
    cost_ratio = (draft_t + verify_t) / base_t_per_token
    a_star = (cost_ratio - 1.0) / block_size
    return cost_ratio, a_star


# --------------------------------------------------------------------------- #
# Training benchmark
# --------------------------------------------------------------------------- #
def bench_train(model, device, seq_len, batch, iters, warmup, amp_mode, use_ckpt=True,
                use_compile=False):
    """One full fwd+bwd+optimizer step, timed and memory-profiled.

    Returns (tokens_per_second, peak_gb, skipped, scale). `skipped` is the GradScaler's
    inf/NaN skip count over the timed iterations -- the number that decides whether fp16
    is viable for this model at this shape.
    """
    from .train import chunked_head_ce, optimizer_step

    cfg = model.cfg
    opt_list = build_optimizer(model)
    scaler = make_scaler(amp_mode, device.type)
    dev_type = device.type
    ids = torch.randint(0, cfg.vocab_size, (batch, seq_len), device=device)
    tgt = torch.randint(0, cfg.vocab_size, (batch, seq_len), device=device)
    toks = batch * seq_len
    if use_compile:
        # Same order as train.py: compile the module (not any DDP wrapper).
        model = torch.compile(model, dynamic=False)
    model.train()

    def step():
        with amp_context(amp_mode, dev_type):
            model.init_caches(batch, seq_len, device=device)
            hidden, _ = model.forward_full(ids, use_checkpoint=use_ckpt,
                                           return_hidden=True)
            loss = chunked_head_ce(hidden, tgt, model.embed.weight, chunk=1024,
                                   amp_mode=amp_mode, device_type=dev_type)
        del hidden
        return optimizer_step(loss, opt_list, scaler)

    for _ in range(warmup):
        step()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    skips, scale = 0, None
    for _ in range(iters):
        s, scale = step()
        skips += int(s)
    _sync(device)
    dt = (time.perf_counter() - t0) / iters
    peak = (torch.cuda.max_memory_allocated() / 1024 ** 3) if device.type == "cuda" else 0.0
    return toks / dt, peak, skips, scale


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--decode-tokens", type=int, default=16)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--hc-variant", default="mhc", choices=["mhc", "mhc-lite"])
    ap.add_argument("--quant", default="FAKE", choices=["FAKE", "NONE"])
    ap.add_argument("--amp", default="off", choices=["off", "fp16", "bf16"],
                    help="for --train: which modes to sweep (all three are measured)")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--train", action="store_true",
                    help="benchmark a training step (fwd+bwd+optimizer) instead of decode")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--train-iters", type=int, default=5)
    ap.add_argument("--train-warmup", type=int, default=2)
    ap.add_argument("--no-ckpt", action="store_true",
                    help="disable gradient checkpointing in --train (for the no-ckpt "
                         "memory comparison)")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the module in --train (dynamic=False), same "
                         "as train.py --compile, to A/B compiled vs eager step time")
    args = ap.parse_args()

    device = torch.device(args.device)
    set_quant_mode(args.quant)
    cfg = PRESET.tiny() if args.tiny else PRESET
    if args.prompt_len + args.decode_tokens + 16 > cfg.max_seq_len:
        args.prompt_len = max(16, cfg.max_seq_len - args.decode_tokens - 16)

    from .params import count
    if args.train:
        _run_train_bench(args, device, cfg)
        return
    c = count(cfg.with_dspark(1))
    print(f"=== vertex500m throughput ===")
    print(f"  device={device}  QUANT={args.quant}  hc={args.hc_variant}")
    print(f"  {c.total/1e6:.1f}M total, {c.active/1e6:.1f}M active, "
          f"{c.gflop_fwd_bwd:.3f} GFLOP/token, block_size={cfg.dspark_block_size}")
    print(f"  prompt={args.prompt_len}  decode={args.decode_tokens}  "
          f"iters={args.iters}\n")

    set_amp_mode(args.amp)
    warn_if_unsupported(args.amp, device.type)
    base = build(cfg.with_dspark(0), args.hc_variant).to(device).eval()
    with amp_context(args.amp, device.type):
        prefill_t = bench_base_prefill(base, device, args.prompt_len, args.iters)
        total_t, base_tps = bench_base(base, device, args.prompt_len,
                                       args.decode_tokens, args.iters)
    decode_t = (total_t - prefill_t) / args.decode_tokens
    base_t = decode_t

    print("BASE BACKBONE (no DSpark)")
    print(f"  prefill {args.prompt_len:5d} tok : {prefill_t*1e3:8.2f} ms  "
          f"{args.prompt_len/prefill_t:8.0f} tok/s")
    print(f"  decode      1 tok    : {base_t*1e3:8.2f} ms  {base_tps:8.2f} tok/s")
    print(f"  full {args.prompt_len}+{args.decode_tokens}       : {total_t*1e3:8.2f} ms  "
          f"{(args.prompt_len+args.decode_tokens)/total_t:8.0f} tok/s")

    full = build(cfg.with_dspark(1), args.hc_variant).to(device).eval()
    with amp_context(args.amp, device.type):
        full.init_caches(1, cfg.max_seq_len, device=device)
        full.attach_dspark().to(device).eval()
        draft_t, verify_t = bench_dspark_costs(full, device, args.prompt_len,
                                                args.decode_tokens, args.iters)
    block = cfg.dspark_block_size + 1
    round_t = draft_t + verify_t

    print(f"\nWITH DSPARK (measured marginal costs, block={block})")
    print(f"  draft forward            : {draft_t*1e3:8.1f} ms  "
          f"({draft_t/base_t*100:.0f}% of one base decode step)")
    print(f"  verify forward ({block:2d} pos)   : {verify_t*1e3:8.1f} ms  "
          f"({verify_t/base_t*100:.0f}% of one base decode step)")
    print(f"  draft+verify round       : {round_t*1e3:8.1f} ms  "
          f"({round_t/base_t:.2f}x one base step)")

    cost_ratio, a_star = break_even_acceptance(draft_t, verify_t, base_t,
                                               cfg.dspark_block_size)
    print(f"\n  cost ratio (round / 1 base token) : {cost_ratio:.2f}x")
    if a_star <= 0:
        print("  break-even acceptance a*          : <= 0%  (already profitable)")
    else:
        print(f"  break-even acceptance a*          : {a_star*100:.1f}%")
        print(f"\n  >>> RANDOM WEIGHTS: acceptance ~0%, so DSpark runs at "
              f"{1/cost_ratio:.2f}x the base throughput -- SLOWER.")
        print(f"      DSpark only pays if a trained model accepts > {a_star*100:.0f}% "
              f"of the draft block.")

    print(f"\n  projected throughput vs acceptance (cost model only -- random weights "
          f"cannot show this):")
    print(f"    base decode is {base_t*1e3:.0f} ms/token; a round costs "
          f"{round_t*1e3:.0f} ms and emits 1 + a*{cfg.dspark_block_size} tokens")
    print(f"    {'accept':>7}  {'tok/round':>10}  {'tok/s':>9}  {'vs base':>8}")
    base_tps = 1.0 / base_t
    for a in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0):
        toks = 1 + a * cfg.dspark_block_size
        tps = toks / round_t
        print(f"    {a*100:6.0f}%  {toks:10.2f}  {tps:9.2f}  {tps/base_tps:7.2f}x")


def _run_train_bench(args, device, cfg):
    """Sweep AMP modes for a training step and report speed, memory and inf-skips."""
    c = count(cfg)
    modes = ["off", "bf16", "fp16"]
    if args.amp != "off":
        modes = [args.amp] if args.amp in modes else modes
    print(f"=== vertex500m TRAINING step ===")
    print(f"  {c.total/1e6:.1f}M params ({c.active/1e6:.1f}M active/token)  "
          f"device={device}  QUANT={args.quant}")
    print(f"  B={args.batch_size} x T={args.seq_len} = "
          f"{args.batch_size*args.seq_len} tok/step  checkpointing="
          f"{'off' if args.no_ckpt else 'segment-%d' % cfg.segment_size}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(device)}  sm_{''.join(map(str, torch.cuda.get_device_capability(device)))}"
              + ("   [no bf16 tensor cores here -- use fp16]"
                 if torch.cuda.get_device_capability(device)[0] < 8 else ""))
    print()
    print(f"  {'amp':>5}  {'tok/s':>10}  {'speedup':>8}  {'peak GB':>8}  "
          f"{'mem vs fp32':>12}  {'skipped':>8}  {'scale':>9}")
    base_tps = None
    base_mem = None
    for mode in modes:
        set_amp_mode(mode)
        warn_if_unsupported(mode, device.type)
        model = build(cfg, args.hc_variant).to(device)
        try:
            tps, peak, skips, scale = bench_train(
                model, device, args.seq_len, args.batch_size,
                args.train_iters, args.train_warmup, mode, use_ckpt=not args.no_ckpt,
                use_compile=args.compile)
        except torch.cuda.OutOfMemoryError:
            print(f"  {mode:>5}  {'OOM':>10}")
            del model
            torch.cuda.empty_cache() if device.type == "cuda" else None
            continue
        if base_tps is None:
            base_tps, base_mem = tps, peak
        sp = f"{tps/base_tps:.2f}x" if base_tps else ""
        mem = f"{base_mem/peak:.2f}x less" if peak and base_mem else ""
        sc = f"{scale:.3g}" if scale else "-"
        warn = "  <-- TOO MANY, lower --init-scale" if skips > max(2, args.train_iters // 10) else ""
        print(f"  {mode:>5}  {tps:10.1f}  {sp:>8}  {peak:8.2f}  {mem:>12}  "
              f"{skips:8d}  {sc:>9}{warn}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print()
    print("  'skipped' = GradScaler steps discarded for inf/NaN gradients. Above ~5% of")
    print("  steps, fp16 is the wrong tool for this model: lower --init-scale, or use")
    print("  bf16 (which needs no scaler and cannot overflow) if the box has bf16 cores.")
    set_amp_mode("off")


if __name__ == "__main__":
    main()
