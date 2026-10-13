"""Data pipeline. Reads pretokenized binary shards (uint16, for vocab <= 65535).

Format on disk: one or more files named {prefix}_{i:05d}.bin, each a flat sequence of
little-endian uint16 token ids, optionally followed by a .npy index of document offsets.

If you have a HuggingFace tokenizer, use `tokenize.py` to produce these shards first.
A tiny synthetic corpus is generated when no shard path is given, so the smoke test runs
with no data setup.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch


def _load_shard(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint16)


def list_shards(directory: str) -> list[Path]:
    d = Path(directory)
    shards = sorted(d.glob("*.bin"))
    if not shards:
        raise FileNotFoundError(
            f"no .bin shards in {directory}. Run tokenize.py first; omitting --data "
            f"in train.py generates a toy corpus instead.")
    return shards


class PackedTokenDataset(torch.utils.data.Dataset):
    """Fixed-length packed windows over a set of uint16 shards.

    Each item is a contiguous window of `seq_len` tokens. Windows never cross a shard
    boundary (packing is best-fit within each shard). This is the simplest correct
    dataloader; for real pretraining you would prefer a shuffled, packing-efficient
    stream with a <1e-4 pad rate (spec 19.1), but that is an optimization.
    """

    def __init__(self, shards, seq_len: int, batch_size: int = 1):
        self.seq_len = seq_len
        self.shards = [_load_shard(s) for s in shards]
        # count how many full windows each shard holds
        self.windows_per_shard = [max(0, len(a) // (seq_len + 1)) for a in self.shards]
        self.index = []
        for si, n in enumerate(self.windows_per_shard):
            self.index.extend((si, w) for w in range(n))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        si, w = self.index[i]
        arr = self.shards[si]
        start = w * (self.seq_len + 1)
        x = torch.from_numpy(arr[start:start + self.seq_len].astype(np.int64))
        y = torch.from_numpy(arr[start + 1:start + self.seq_len + 1].astype(np.int64))
        return {"input_ids": x, "targets": y}


def synthetic_corpus(path: str, n_tokens: int = 2_000_000, vocab_size: int = 32000,
                     seed: int = 0):
    """Write a small structured-but-not-random corpus for smoke tests.

    Real training needs real data; this exists so `python -m vertex500m.train --smoke`
    runs end-to-end with no setup. The distribution is a simple Markov chain over a small
    alphabet so the loss actually moves.
    """
    d = Path(path)
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    alphabet = 256
    # Markov transition matrix with strong local structure
    P = np.full((alphabet, alphabet), 1e-3)
    P += np.eye(alphabet) * 0.05
    for i in range(alphabet):
        P[i, (i + 1) % alphabet] += 0.05
        P[i, (i + 2) % alphabet] += 0.02
    P /= P.sum(-1, keepdims=True)
    ids = np.zeros(n_tokens, dtype=np.uint16)
    cur = rng.integers(0, alphabet)
    for t in range(n_tokens):
        ids[t] = cur
        cur = rng.choice(alphabet, p=P[cur])
    ids.tofile(d / "synthetic_00000.bin")
    return d


def build_dataloader(data_dir, seq_len: int, batch_size: int = 1,
                     synthetic_tokens: int = 2_000_000) -> torch.utils.data.DataLoader:
    shards = list_shards(data_dir)
    ds = PackedTokenDataset(shards, seq_len)
    # persistent_workers keeps the two loader workers alive across epochs instead of
    # respawning them every pass over the data; prefetch_factor deepens the ready
    # queue so the GPU never waits on the Python-side batch assembly. pin_memory
    # pairs with non_blocking H2D copies in the training loop (see train.py).
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True,
                                       num_workers=2, drop_last=True,
                                       pin_memory=True, persistent_workers=True,
                                       prefetch_factor=4)
