"""Held-out evaluation: cross-entropy and perplexity.

You cannot tell whether a training run worked without this. The training loss is
computed on the same distribution you trained on and is dominated by whatever the model
has already memorised; a held-out number is the only honest signal.

Two numbers matter and they mean different things:

  * CE / PPL on held-out text   -- did the model learn the language.
  * CE on the TRAINING shards  -- the gap between the two is the generalisation gap.

Implementation notes that matter for correctness:

  * The loss is computed in fp32 over every position via `forward_full`, CHUNKED over
    positions (`train.chunked_ce`). A full [T, V] fp32 logit tensor is 524 MB at T=4096
    and 2.1 GB at T=16384; materialising it (plus its fp32 cast, plus log_softmax) is the
    single largest allocation in training (vertex500m-dense 18.12).
  * `init_caches` is called per batch. The CSA2 window ring and compressor PoolState
    persist across decode steps but must be reset between independent sequences.
  * The Engram hash cache must also be reset, or n-grams leak across documents and the
    number is silently optimistic.
  * `--split` selects shards by filename substring and FALLS BACK TO ALL SHARDS if
    nothing matches, so check the shard count in the output before trusting a held-out
    number.
  * `--amp` must MATCH how the checkpoint was trained. Evaluating an fp16-trained model
    in fp32 measures the precision gap, not quality; and evaluating in fp16 when the
    caches were allocated for fp32 (or vice versa) is a hard dtype mismatch inside
    attention. Both loops below run inside the autocast region for that reason.

Run:
    python -m vertex500m.eval --ckpt ckpt.pt --data data/ --split val
    python -m vertex500m.eval --ckpt ckpt.pt --data data/ --split train --max-batches 50
"""

from __future__ import annotations

import argparse
import math
import time

import torch

from .amp import amp_context, set_amp_mode, warn_if_unsupported
from .config import PRESET
from .engram import EngramLayout, NgramHashState
from .params import count
from .quant import set_quant_mode
from .train import chunked_ce


def load_token_map(path):
    import json
    d = json.loads(open(path, encoding="utf-8").read())
    return d["token_map"], d["engram_compressed_vocab_size"], d.get("collisions", 0)


def build_model(cfg, ckpt=None, token_map=None, device="cpu", hc_variant="mhc",
                with_dspark=False):
    from .init_weights import init_from_scratch
    layout = EngramLayout(cfg)
    if token_map is None:
        from .engram import identity_token_map
        token_map = identity_token_map(cfg.vocab_size)
    hashes = NgramHashState(cfg, layout, token_map, cfg.engram_compressed_vocab_size)
    model = init_from_scratch(cfg, engram_layout=layout, engram_hashes=hashes,
                              hc_variant=hc_variant)
    if ckpt:
        obj = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
        saved = (obj.get("config", {}) or {}).get("hc_variant") \
            if isinstance(obj, dict) and "model" in obj else None
        if saved and saved != hc_variant:
            raise SystemExit(
                f"checkpoint {ckpt} was trained with hc_variant={saved!r} but "
                f"--hc-variant is {hc_variant!r}. mHC-lite is a reparameterization, not "
                f"a conversion (mHC-lite.md 1): the projection widths differ (24 vs 32 "
                f"at hc=4) and there is no bridge. Retrain or pass --hc-variant {saved}.")
        # Attach BEFORE loading, not after. A stage-B checkpoint carries mtp.* weights;
        # attaching afterwards makes every one of them "unexpected", so the drafter you
        # trained is silently thrown away and acceptance is measured on random weights.
        if with_dspark:
            model.init_caches(1, cfg.max_seq_len)
            model.attach_dspark()
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if unexpected:
            print(f"  [warn] unexpected keys: {list(unexpected)[:5]}")
        print(f"  loaded {ckpt} (missing={len(missing)} unexpected={len(unexpected)})")
    elif with_dspark:
        model.init_caches(1, cfg.max_seq_len)
        model.attach_dspark()
    return model.to(device).eval()


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None, log_every=20, seq_desc="eval",
             amp_mode=None):
    """Token-level cross-entropy over held-out windows. Returns (ce, nll_sum, ntok)."""
    total_nll, total_tok = 0.0, 0
    t0 = time.time()
    dev_type = torch.device(device).type
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        ids = batch["input_ids"].to(device, non_blocking=True)
        tgt = batch["targets"].to(device, non_blocking=True)
        B, S = ids.shape
        # fresh caches per sequence: window ring, compressor pool, engram id history.
        # Must be INSIDE the autocast region -- init_caches reads the global AMP mode to
        # pick the cache dtype, so allocating them outside it hands attention an fp16 KV
        # cache against fp32 activations.
        with amp_context(amp_mode, dev_type):
            model.init_caches(B, S + 8, device=device)
            logits, _ = model.forward_full(ids, start_pos=0)
            # chunked over positions: a full [B*S, V] fp32 log_softmax is the single
            # largest tensor here and it does not fit at realistic S (vertex500m-dense 18.12)
            nll = chunked_ce(logits, tgt, chunk=1024, reduction="sum")
        total_nll += float(nll)
        total_tok += tgt.numel()
        if log_every and i % log_every == 0:
            print(f"  {seq_desc} batch {i:4d}  running CE {total_nll/total_tok:.4f}"
                  f"  ({time.time()-t0:.0f}s)")
    ce = total_nll / max(total_tok, 1)
    return ce, total_nll, total_tok


@torch.no_grad()
def dspark_acceptance(model, loader, device, max_batches=8, temperature=1.0,
                      amp_mode=None):
    """Mean per-slot draft acceptance against the frozen backbone.

    This is what `--with-dspark` promises. It used to be advertised and never computed:
    the flag attached DSpark and then ran plain CE, so the draft quality of the model
    you just trained was never reported anywhere.
    """
    from .dspark_loss import dspark_loss
    accs = []
    dev_type = torch.device(device).type
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        ids = batch["input_ids"].to(device)
        with amp_context(amp_mode, dev_type):
            _, _, st = dspark_loss(model, ids, temperature=temperature)
        if st:
            accs.append(st["accept_now"])
    return (sum(accs) / len(accs) if accs else 0.0, len(accs))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--data", required=True, help="dir of uint16 .bin shards")
    ap.add_argument("--split", default="val",
                    help="substring selecting shards; falls back to ALL shards if no "
                         "shard name matches, so verify the printed count")
    ap.add_argument("--token-map", default=None, help="JSON from tokenize.py --save-map")
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=20,
                    help="print running CE every N batches; set 1 to watch an AMP A/B "
                         "converge, 0 to silence")
    ap.add_argument("--eval-batches", type=int, default=8,
                    help="batches used for the DSpark acceptance estimate")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--hc-variant", default="mhc", choices=["mhc", "mhc-lite"])
    ap.add_argument("--quant", default="FAKE", choices=["FAKE", "NONE"])
    ap.add_argument("--amp", default=None, choices=[None, "off", "fp16", "bf16"],
                    help="default: whatever the checkpoint records, else off. Matching "
                         "the training precision is what makes the CE number comparable; "
                         "a mismatch measures the precision gap instead of the model.")
    ap.add_argument("--with-dspark", action="store_true",
                    help="also report DSpark draft acceptance")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    set_quant_mode(args.quant)
    cfg = PRESET.tiny() if args.tiny else PRESET
    if args.with_dspark:
        cfg = cfg.with_dspark(1)

    token_map, cvs, collisions = (None, 0, 0)
    if args.token_map:
        token_map, cvs, collisions = load_token_map(args.token_map)
        print(f"[token map] compressed_vocab_size={cvs} collisions={collisions}")
        cfg = cfg.with_(engram_compressed_vocab_size=cvs)

    ckpt_amp = None
    if args.ckpt:
        import json
        try:
            ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
            if isinstance(ck, dict):
                ckpt_amp = (ck.get("config", {}) or {}).get("amp")
        except Exception:                      # unreadable here; build_model will say so
            ckpt_amp = None
    mode = args.amp if args.amp is not None else (ckpt_amp or "off")
    set_amp_mode(mode)
    warn_if_unsupported(mode, torch.device(device).type)

    model = build_model(cfg, args.ckpt, token_map, device, args.hc_variant,
                        with_dspark=args.with_dspark)
    if args.with_dspark:
        with amp_context(mode, torch.device(device).type):
            model.init_caches(args.batch_size, cfg.max_seq_len, device=device)
        model = model.eval()

    from .data import list_shards
    from pathlib import Path
    shards = list_shards(args.data)
    if args.split:
        shards = [s for s in shards if args.split in s.name] or shards
    print(f"[eval] {len(shards)} shard(s), seq_len={args.seq_len or cfg.max_seq_len}")

    from .data import PackedTokenDataset
    seq = args.seq_len or min(cfg.max_seq_len, 2048)
    ds = PackedTokenDataset(shards, seq)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                         num_workers=0, drop_last=True)
    c = count(cfg, args.hc_variant)
    print(f"[cfg] {c.total/1e6:.1f}M total  device={device}  quant={args.quant}  "
          f"amp={mode}  hc={args.hc_variant}")

    ce, nll, ntok = evaluate(model, loader, device, args.max_batches,
                         log_every=args.log_every, seq_desc=args.split,
                         amp_mode=mode)
    print(f"\n=== {args.split} ===")
    print(f"  tokens     : {ntok:,}")
    print(f"  mean CE    : {ce:.4f} nats/token")
    print(f"  perplexity : {math.exp(min(ce, 20)):.3f}")
    print(f"  bits/token : {ce / math.log(2):.4f}")

    if args.with_dspark:
        # dspark_loss re-seeds the caches itself at L + block + 4, which is the right
        # length for a training window and also fine here.
        acc, n = dspark_acceptance(model, loader, device, args.eval_batches,
                                   temperature=args.temperature, amp_mode=mode)
        print(f"\n=== DSpark draft acceptance ===")
        print(f"  mean accept: {acc*100:.1f}%  over {n} batches")
        print(f"  DSpark earns its parameters only above the measured break-even")
        print(f"  (run `python -m vertex500m.bench` on your hardware for that number).")


if __name__ == "__main__":
    main()
