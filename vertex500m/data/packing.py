"""Document-aware packing: whole documents into fixed windows.

`PackedTokenDataset` (shards.py) cuts each shard into fixed windows wherever the
cut falls -- documents are split mid-sentence and every shard tail shorter than
`seq_len + 1` is dropped. For real pretraining you want document boundaries
respected and the packing efficient (spec 19.1 wants a <1e-4 pad rate; this
packs with ZERO padding by dropping partial tails instead).

`DocPackedDataset` greedy-packs whole documents (EOS separators are already in
the token stream -- tokenize.py appends one per doc) into `seq_len + 1` windows:

* a document longer than a window is split into consecutive full windows (plus
  a remainder piece that packs with the neighbours, like everything else);
* short documents accumulate next-fit into a window until the next one would
  overflow, at which point the full window is emitted;
* partial tails are dropped (`drop_last` semantics), never padded;
* the window ORDER is shuffled with `seed`; contents are deterministic.

Doc boundaries come from the per-shard `.idx.npy` index tokenize.py writes
(doc-end offsets, exclusive). A shard WITHOUT an index (legacy shards) falls
back to fixed windows for that shard, with a warning -- mixing is allowed so
old data keeps working.

Scaling note: the window table is one small tuple per window (~a 10B-token
corpus at S=8192 is ~1.2M entries). Token data itself stays memory-mapped and
is only copied for the windows actually read.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch


def index_path_for(bin_path: Path) -> Path:
    """Sibling `.idx.npy` for a `.bin` shard (`web_train_00000.bin` ->
    `web_train_00000.idx.npy`)."""
    name = bin_path.name
    if name.endswith(".bin"):
        return bin_path.with_name(name[:-len(".bin")] + ".idx.npy")
    return bin_path.with_suffix(".idx.npy")


class DocPackedDataset(torch.utils.data.Dataset):
    """Greedy document-packed windows over uint16 shards. See module docstring."""

    def __init__(self, shards, seq_len: int, seed: int = 0):
        self.seq_len = seq_len
        self.seed = seed
        self.paths = [Path(s) for s in shards]
        if not self.paths:
            raise FileNotFoundError("DocPackedDataset got no shards")
        self._arrays: dict[int, np.ndarray] = {}
        w = seq_len + 1
        windows: list[tuple] = []          # (shard_idx, ((s, e), ...)) pieces
        tokens_total = 0
        tokens_used = 0
        fallback_shards = 0
        for si, p in enumerate(self.paths):
            n_tokens = os.path.getsize(p) // 2
            tokens_total += n_tokens
            idx = index_path_for(p)
            if idx.exists():
                ends = np.load(idx).astype(np.int64)
                # doc spans, clamped to the shard (a corrupt index must fail
                # loudly here, not produce overlapping windows downstream)
                spans = []
                prev = 0
                for e in ends:
                    e = min(int(e), n_tokens)
                    if e > prev:
                        spans.append((prev, e))
                    prev = e
            else:
                fallback_shards += 1
                # legacy shard: fixed windows, same layout as PackedTokenDataset
                n = n_tokens // w
                windows.extend((si, ((k * w, (k + 1) * w),)) for k in range(n))
                tokens_used += n * w
                continue
            # split overlong docs, then next-fit the pieces
            pieces = []
            for s, e in spans:
                k = s
                while k < e:
                    q = min(k + w, e)
                    pieces.append((k, q))
                    k = q
            cur: list[tuple[int, int]] = []
            cur_len = 0
            for s, e in pieces:
                L = e - s
                if cur_len + L > w:
                    if cur_len == w:
                        windows.append((si, tuple(cur)))
                        tokens_used += w
                    cur, cur_len = [], 0       # partial tail: dropped, not padded
                cur.append((s, e))
                cur_len += L
                if cur_len == w:
                    windows.append((si, tuple(cur)))
                    tokens_used += w
                    cur, cur_len = [], 0
            # final partial tail is dropped (drop_last semantics)
        if fallback_shards:
            import warnings
            warnings.warn(
                f"{fallback_shards}/{len(self.paths)} shard(s) have no .idx.npy "
                f"doc index and were packed as fixed windows. Re-tokenize to get "
                f"document-aware packing everywhere.")
        if not windows:
            raise ValueError(
                f"no full {seq_len + 1}-token windows in {len(self.paths)} shard(s) "
                f"({tokens_total} tokens): corpus too small for this seq_len.")
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(windows))
        self.windows = [windows[i] for i in order]
        self.stats = {"windows": len(self.windows),
                      "tokens_total": tokens_total,
                      "tokens_used": tokens_used,
                      "waste": 1.0 - tokens_used / max(tokens_total, 1)}

    def __len__(self):
        return len(self.windows)

    def _arr(self, si: int) -> np.ndarray:
        a = self._arrays.get(si)
        if a is None:
            a = self._arrays[si] = np.memmap(self.paths[si], dtype=np.uint16,
                                             mode="r")
        return a

    def __getstate__(self):
        # memmap arrays pickle BY VALUE (a full corpus copy per worker under
        # spawn); drop the cache so workers re-map read-only instead.
        st = self.__dict__.copy()
        st["_arrays"] = {}
        return st

    def __getitem__(self, i):
        si, pieces = self.windows[i]
        arr = self._arr(si)
        if len(pieces) == 1:
            s, e = pieces[0]
            win = np.asarray(arr[s:e], dtype=np.int64)
        else:
            win = np.concatenate([np.asarray(arr[s:e], dtype=np.int64)
                                  for s, e in pieces])
        x = torch.from_numpy(win[:self.seq_len].copy())
        y = torch.from_numpy(win[1:self.seq_len + 1].copy())
        return {"input_ids": x, "targets": y}
