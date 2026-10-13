"""Learning-rate schedule: linear warmup, then cosine decay.

The schedule must PRESERVE the optimizer partition -- see `set_lr`. Moved
verbatim from train.py (phase-1 restructure); behaviour is unchanged.
"""

from __future__ import annotations

import math


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
