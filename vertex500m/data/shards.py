"""Shard I/O and the fixed-window dataloader.

Format on disk: one or more files named {stem}_{i:05d}.bin, each a flat sequence of
little-endian uint16 token ids, optionally followed by a .idx.npy index of document
offsets (see packing.py). tokenize.py writes `{name}_train_*.bin` /
`{name}_val_*.bin` plus a manifest.json; use `list_shards(..., split=...)` or the
manifest to select a split.

Shard data stays memory-mapped and is loaded lazily per process: the dataset
holds PATHS, not arrays, so DataLoader workers (fork or spawn) never receive a
full corpus copy. `__getstate__` drops the per-process mmap cache for the same
reason.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

TOKEN_DTYPE = np.uint16          # exact for vocab <= 65535
TOKEN_BYTES = 2


def _load_shard(path: Path) -> np.ndarray:
    """Memory-mapped read-only view of one shard. Slices copy on `.astype()`."""
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r")


def shard_num_tokens(path: Path) -> int:
    size = os.path.getsize(path)
    if size % TOKEN_BYTES:
        raise ValueError(
            f"shard {path} is {size} bytes, not a multiple of 2: truncated write?")
    return size // TOKEN_BYTES


def list_shards(directory: str, split: str | None = None) -> list[Path]:
    """All `.bin` shards in `directory`, optionally filtered to one split.

    `split` matches the `{name}_{split}_{i:05d}.bin` stem convention tokenize.py
    writes (`split="train"` / `"val"`). `None`/`""`/`"all"` returns everything.
    A split that matches nothing is an error, never a silent fall-back to all
    shards.
    """
    d = Path(directory)
    shards = sorted(d.glob("*.bin"))
    if not shards:
        raise FileNotFoundError(
            f"no .bin shards in {directory}. Run tokenize.py first; omitting --data "
            f"in train.py generates a toy corpus instead.")
    if split in (None, "", "all"):
        return shards
    matched = [s for s in shards
               if f"_{split}_" in s.stem or s.stem.endswith(f"_{split}")
               or s.stem == split]
    if not matched:
        raise FileNotFoundError(
            f"no .bin shards for split {split!r} in {directory} "
            f"({len(shards)} shard(s) present: "
            + ", ".join(s.name for s in shards[:5])
            + ("..." if len(shards) > 5 else "")
            + "). Tokenize with a val fraction, or select split 'all'.")
    return matched


class PackedTokenDataset(torch.utils.data.Dataset):
    """Fixed-length packed windows over a set of uint16 shards.

    Each item is a contiguous window of `seq_len` tokens. Windows never cross a shard
    boundary (packing is best-fit within each shard). For document-aware packing
    with zero padding see `packing.DocPackedDataset`; this loader stays for
    compatibility and for corpora without a doc index.

    Shard contents are memory-mapped lazily per process (`_arrays` cache);
    `windows_per_shard`/`index` need only file sizes at construction.
    """

    def __init__(self, shards, seq_len: int, batch_size: int = 1):
        self.seq_len = seq_len
        self.paths = [Path(s) for s in shards]
        if not self.paths:
            raise FileNotFoundError("PackedTokenDataset got no shards")
        # count how many full windows each shard holds
        self.windows_per_shard = [shard_num_tokens(p) // (seq_len + 1)
                                  for p in self.paths]
        self.index = []
        for si, n in enumerate(self.windows_per_shard):
            self.index.extend((si, w) for w in range(n))
        self._arrays: dict[int, np.ndarray] = {}

    def __len__(self):
        return len(self.index)

    def _arr(self, si: int) -> np.ndarray:
        a = self._arrays.get(si)
        if a is None:
            a = self._arrays[si] = _load_shard(self.paths[si])
        return a

    def __getstate__(self):
        # mmap arrays pickle BY VALUE (a full corpus copy per worker under spawn);
        # drop the cache so workers re-map read-only instead.
        st = self.__dict__.copy()
        st["_arrays"] = {}
        return st

    def __getitem__(self, i):
        si, w = self.index[i]
        arr = self._arr(si)
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


def build_dataset(data_dir, seq_len: int, *,
                  split: str | None = None, pack: str = "fixed", seed: int = 0):
    """Dataset over one split (no loader). Split out so callers that need the
    dataset first -- e.g. to build a DistributedSampler around it -- don't
    have to construct the loader twice."""
    shards = list_shards(data_dir, split=split)
    if pack == "docs":
        from .packing import DocPackedDataset
        return DocPackedDataset(shards, seq_len, seed=seed)
    elif pack == "fixed":
        return PackedTokenDataset(shards, seq_len)
    else:
        raise ValueError(f"pack must be 'fixed' or 'docs', got {pack!r}")


def build_dataloader(data_dir=None, seq_len: int = 2048, batch_size: int = 1,
                     synthetic_tokens: int = 2_000_000, *,
                     split: str | None = None, pack: str = "fixed", seed: int = 0,
                     shuffle: bool = True, sampler=None, num_workers: int = 2,
                     generator=None, persistent_workers: bool = False,
                     dataset=None,
                     ) -> torch.utils.data.DataLoader:
    """Dataloader over one split of a shard directory.

    `pack="fixed"` cuts fixed windows (PackedTokenDataset); `pack="docs"` packs
    whole documents (packing.DocPackedDataset, needs the .idx.npy index).
    `seed` deterministically seeds the shuffle. `sampler` (e.g. a
    DistributedSampler) disables shuffling, which the DataLoader forbids
    alongside an explicit sampler. `generator` overrides the seeded shuffle
    generator (exact-resume restores a saved generator state through it).

    `persistent_workers` defaults to False ON PURPOSE, at the cost of a worker
    respawn per epoch: worker processes carry state across iterators, which
    breaks the per-epoch reseeding `_BatchCursor` relies on for exact resume
    (pinned by test_cursor_epoch_rollover_is_deterministic). The respawn costs
    seconds per epoch -- negligible next to training steps.

    `dataset` overrides directory construction (the loop builds the dataset
    first when it needs a sampler around it); `data_dir` may then be None.
    """
    if dataset is None:
        if data_dir is None:
            raise ValueError("build_dataloader needs data_dir or dataset")
        ds = build_dataset(data_dir, seq_len, split=split, pack=pack, seed=seed)
    else:
        ds = dataset
    if sampler is not None:
        shuffle = False
    if generator is None and shuffle:
        generator = torch.Generator().manual_seed(seed)
    # persistent_workers / prefetch_factor need live workers; with
    # num_workers=0 both are illegal.
    use_persistent = bool(persistent_workers) and num_workers > 0
    prefetch = 4 if num_workers > 0 else None
    # persistent_workers keeps the two loader workers alive across epochs instead of
    # respawning them every pass over the data; prefetch_factor deepens the ready
    # queue so the GPU never waits on the Python-side batch assembly. pin_memory
    # pairs with non_blocking H2D copies in the training loop (see train/loop.py).
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                       sampler=sampler, generator=generator,
                                       num_workers=num_workers, drop_last=True,
                                       pin_memory=True,
                                       persistent_workers=use_persistent,
                                       prefetch_factor=prefetch)
