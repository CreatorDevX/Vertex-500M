"""Checkpoint save/load for training.

Payload layout (all keys optional except `model`, so old checkpoints still
load):

    {"model": ..., "optimizers": ..., "scaler": ...,
     "config": {"hc_variant", "amp"},
     "train": {"stage", "step", "total_steps", "warmup", "pos_in_epoch",
               "epoch", "best_val", "seed", "sampler_epoch"},
     "rng": {"torch", "cuda", "numpy", "python"},
     ...extra (stage, steps, accept, ...)}

`train.step` is the NEXT step to run (steps completed). `pos_in_epoch` is
batches consumed from the current dataloader iterator and `epoch` is the
epoch number; the loop rebuilds the epoch's data order deterministically
(per-epoch reseeding in `_BatchCursor`) and fast-forwards `pos_in_epoch`
batches. Together with weights/optimizer/scaler/RNG state they give exact
resume: same weights, same optimizer, same LR position, same data order.

Saves are atomic: `torch.save` to a temp file + `os.replace`, so a crash or
SIGTERM mid-save can never leave a half-written `latest.pt` behind.
"""

from __future__ import annotations

import os
import random
import tempfile

import numpy as np
import torch

from .. import amp as amp_mod


def capture_rng_state() -> dict:
    """Snapshot all RNG streams that affect a run (data order is separate)."""
    st = {"torch": torch.get_rng_state(),
          "numpy": np.random.get_state(),
          "python": random.getstate()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def restore_rng_state(st: dict) -> None:
    """Restore streams captured by `capture_rng_state`."""
    torch.set_rng_state(st["torch"])
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def restore_optimizer_state(optimizer, saved) -> None:
    """`optimizer.load_state_dict`, tolerating the legacy double-wrapped layout.

    Checkpoints written before the phase-3 fix stored
    `{"state": o.state_dict(), "param_groups": o.param_groups}`; passing that
    to `load_state_dict` validates the groups but silently drops every
    per-parameter buffer (momentum restarts from zero while the log claims a
    restore). Detect the nesting and unwrap one level.
    """
    inner = saved.get("state") if isinstance(saved, dict) else None
    if isinstance(inner, dict) and set(inner) == {"state", "param_groups"}:
        saved = inner
    optimizer.load_state_dict(saved)


def atomic_torch_save(obj, path: str) -> None:
    """`torch.save` that never leaves a partial file at `path`."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_ckpt_", suffix=".pt")
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(obj, f)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def resolve_ckpt_dir(args) -> str | None:
    """Where periodic checkpoints live, or None when nothing enables them."""
    if getattr(args, "ckpt_dir", None):
        return args.ckpt_dir
    if getattr(args, "out", None):
        return os.path.dirname(os.path.abspath(args.out)) or "."
    return None


def ckpt_paths(ckpt_dir: str, stage: str) -> dict:
    """Per-stage checkpoint filenames inside `ckpt_dir`."""
    return {"latest": os.path.join(ckpt_dir, f"{stage}.latest.pt"),
            "best": os.path.join(ckpt_dir, f"{stage}.best.pt"),
            "step": lambda step: os.path.join(
                ckpt_dir, f"{stage}.step_{step:07d}.pt")}


def save_checkpoint(model, args, optimizers, extra: dict | None = None,
                    scaler=None, train_state: dict | None = None,
                    path: str | None = None) -> str:
    """Save weights + optimizer state (+ scaler + train/RNG state when given).

    Optimizer state matters: Muon's momentum buffer and the tables' Sinkhorn momentum
    are not reconstructible, so a checkpoint without them cannot resume training
    faithfully -- it restarts both. `freqs_cis` is excluded because it is a pure
    function of the config and is rebuilt on load.

    The GradScaler's state matters for the same reason and is saved for the same
    reason: a resumed run that restarts at scale 65536 spends its first hundreds of steps
    stepping on the floor waiting for the scale to back off, and if the model turned out
    to need a much smaller scale than 65536 it never gets there.

    `path` overrides `args.out` (periodic/best checkpoints). Returns the path
    written. Rank-gated as before: non-main ranks write nothing and get "".
    """
    dest = path or args.out
    if not dest or (getattr(args, "_dist", None) and not args._dist.is_main):
        return ""
    sd = {k: v for k, v in model.state_dict().items() if "freqs_cis" not in k}
    # Optimizer state is `o.state_dict()` VERBATIM -- the documented
    # `{"state", "param_groups"}` format `load_state_dict` expects. An older
    # revision wrapped it once more as {"state": o.state_dict(), ...}, which
    # `load_state_dict` silently ignores (its keys never match a parameter),
    # so resume "restored" the state while momentum actually restarted from
    # zero. `restore_optimizer_state` unwraps that legacy layout.
    payload = {"model": sd,
               "optimizers": [o.state_dict() for o in optimizers],
               "config": {"hc_variant": getattr(model, "hc_variant", "mhc"),
                          "amp": amp_mod.get_amp_mode()},
               **(extra or {})}
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if train_state is not None:
        payload["train"] = train_state
        if "rng" not in payload:
            payload["rng"] = capture_rng_state()
    atomic_torch_save(payload, dest)
    n = sum(v.numel() for v in sd.values())
    print(f"[save] {dest} ({n/1e6:.1f}M weights"
          f"{', optimizer state included' if optimizers else ''}"
          f"{', scaler state included' if scaler is not None else ''}"
          f"{', train state included' if train_state is not None else ''})")
    return dest
