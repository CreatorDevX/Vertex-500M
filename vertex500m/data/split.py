"""Train/val splitting: deterministic and streaming-friendly.

`plan_split` returns a boolean mask over doc indices (`True` = validation).
The split is a seeded permutation, so it is reproducible and stable for a
given (n_docs, val_fraction, seed).

Two-pass friendly: count the documents first, plan the split, then stream the
corpus a second time routing each doc by its index -- texts are never
materialised just to split them.
"""

from __future__ import annotations

import numpy as np


def plan_split(n_docs: int, val_fraction: float = 0.005, seed: int = 0):
    """Boolean mask of length `n_docs`; `True` entries are the val set.

    Guarantees at least one val doc when `val_fraction > 0` (and at least one
    train doc), so a small corpus fails loudly at plan time instead of writing
    an empty val split that breaks evaluation later.
    """
    if n_docs < 0:
        raise ValueError(f"n_docs must be >= 0, got {n_docs}")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    if n_docs == 0:
        return np.zeros(0, dtype=bool)
    if val_fraction <= 0.0:
        return np.zeros(n_docs, dtype=bool)
    if n_docs < 2:
        raise ValueError(
            f"cannot hold out a val split from {n_docs} document(s) with "
            f"val_fraction={val_fraction}: need at least 2 documents (or pass "
            f"val_fraction=0 for a train-only corpus).")
    n_val = max(1, int(n_docs * val_fraction))
    n_val = min(n_val, n_docs - 1)          # always keep at least one train doc
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_docs)
    is_val = np.zeros(n_docs, dtype=bool)
    is_val[perm[:n_val]] = True
    return is_val
