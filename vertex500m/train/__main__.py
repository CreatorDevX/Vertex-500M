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

import torch

from .loop import run


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
    ap.add_argument("--allow-identity-map", action="store_true",
                    help="train on real --data with the identity Engram map. Only for "
                         "experiments that explicitly want the degraded configuration.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny CPU end-to-end")
    ap.add_argument("--tiny", action="store_true",
                    help="PRESET.tiny() model for fast debugging/CI (unlike --smoke, "
                         "does not cap steps or force seq_len=64)")
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
    # ---- training QoL: periodic checkpoints, in-loop eval, robustness ----
    ap.add_argument("--save-every", type=int, default=0,
                    help="write a step checkpoint every N completed steps "
                         "({stage}.step_{n}.pt + {stage}.latest.pt in --ckpt-dir); "
                         "0 = only the final --out save, as before")
    ap.add_argument("--ckpt-dir", default=None,
                    help="directory for periodic/best/latest checkpoints and "
                         "metrics.jsonl. Default: directory of --out.")
    ap.add_argument("--keep-last", type=int, default=3,
                    help="how many step checkpoints to retain (older ones are "
                         "deleted; latest/best are never rotated). 0 = latest only.")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="held-out CE every N completed steps (stage A only); "
                         "improvements are saved as {stage}.best.pt. 0 = off.")
    ap.add_argument("--val-data", default=None,
                    help="shard dir for in-loop validation (default: --data)")
    ap.add_argument("--val-split", default="val",
                    help="split selected inside --val-data/--data for validation")
    ap.add_argument("--train-split", default="train",
                    help="split selected inside --data for training "
                         "('all' for legacy unsplit directories)")
    ap.add_argument("--pack", default="fixed", choices=["fixed", "docs"],
                    help="fixed windows, or document-aware packing (needs the "
                         ".idx.npy index tokenize.py writes)")
    ap.add_argument("--grad-clip", type=float, default=None,
                    help="global grad-norm clip (GradScaler-aware ordering). "
                         "Unset = no clipping, as before.")
    ap.add_argument("--accum-steps", type=int, default=1,
                    help="micro-batches per optimizer step (trades step time for "
                         "effective batch size on small GPUs)")
    ap.add_argument("--target-loss", type=float, default=None,
                    help="stop stage A early when the EMA loss reaches this "
                         "(exit 0, final checkpoint saved)")
    ap.add_argument("--deadline-s", type=float, default=None,
                    help="wall-clock budget in seconds; on expiry saves latest "
                         "and exits 10 so the run can be resumed")
    ap.add_argument("--metrics", default=None,
                    help="metrics.jsonl path (default: <ckpt-dir>/metrics.jsonl "
                         "when checkpoints are enabled; '' disables)")
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
    status = run(args)
    # "done"/"target" = stage complete; "deadline" = stopped early, resume it.
    raise SystemExit(0 if status in ("done", "target") else 10)


if __name__ == "__main__":
    main()
