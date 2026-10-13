"""Engram: conditional n-gram memory (spec 11).

A hash table of n-gram embeddings, looked up per position, gated on agreement between
the residual stream and the retrieved key, and added to the stream. The hash ids are
computed once for the whole stack, in a no-grad pass, because they are a pure function
of the token ids and their n-gram context.

Layout. Each layer owns `(max_ngram_size - 1) * n_heads` primes drawn upward from
`engram_vocab_size`, one per (n-gram size, head). `engram_num_embeddings[layer]` is
their SUM. Odd multipliers ensure `token * multiplier` cannot overflow int64.

Note on the spec: 11.2's prose says "for each ngram_size in 2..(max_ngram-1)", which
yields 2 sizes, but `n_hash_cols = (max_ngram-1)*n_heads = 12` requires 3 sizes
(2-, 3-, 4-grams). The reference CODE uses `range(max_ngram - 1)` = 3 and is correct;
the prose is wrong. This implementation follows the code.
"""

from __future__ import annotations

import torch

from .rope import RMSNorm, rms_recip  # noqa: F401  (re-exported for symmetry)


def next_prime_above(n: int, seen: set[int]) -> int:
    """Next prime strictly greater than n, not already used."""
    c = n + 1
    while True:
        if c % 2 == 0 and c > 2:
            c += 1
            continue
        d, ok = 3, True
        while d * d <= c:
            if c % d == 0:
                ok = False
                break
            d += 2
        if ok and c not in seen:
            return c
        c += 1


class EngramLayout:
    """Prime layout for the n-gram hash table."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.layer_ids = tuple(cfg.engram_layer_ids)
        self.n_layers = len(self.layer_ids)
        self.max_ngram = cfg.engram_max_ngram_size
        self.n_heads = cfg.engram_n_heads
        # (max_ngram - 1) n-gram SIZES, not (max_ngram - 2). See module docstring.
        self.n_sizes = self.max_ngram - 1
        self.n_hash_cols = self.n_sizes * self.n_heads
        seen: set[int] = set()
        self.primes_3d: list[list[list[int]]] = []
        self.offsets_3d: list[list[list[int]]] = []
        self.rows_per_layer: list[int] = []
        for _ in self.layer_ids:
            layer, cur = [], cfg.engram_vocab_size - 1
            for _ in range(self.n_sizes):
                heads = []
                for _ in range(self.n_heads):
                    cur = next_prime_above(cur, seen)
                    seen.add(cur)
                    heads.append(cur)
                layer.append(heads)
            self.primes_3d.append(layer)
            self.rows_per_layer.append(sum(sum(h) for h in layer))

        # Each (size, head) column gets its OWN disjoint row range, otherwise every
        # column would hash into [0, max(prime)) and the tail of the table -- which is
        # most of it, since rows == sum(primes) >> max(prime) -- would never be read.
        for li, layer in enumerate(self.primes_3d):
            offs, base = [], 0
            for heads in layer:
                row = []
                for p in heads:
                    row.append(base)
                    base += p
                offs.append(row)
            self.offsets_3d.append(offs)

        # Odd multipliers: `token * multiplier` must not overflow int64, so the
        # multiplier must stay below the smallest prime in its row.
        self.multipliers_3d = []
        for layer in self.primes_3d:
            rows = []
            for i, heads in enumerate(layer):
                rows.append([self._odd_multiplier(i * self.n_heads + j, p)
                             for j, p in enumerate(heads)])
            self.multipliers_3d.append(rows)

    @staticmethod
    def _odd_multiplier(i: int, prime: int) -> int:
        m = (2 * (i % 4096) + 1) | 1
        while m >= prime:
            m = (m + 2) | 1
        return m

    @property
    def total_rows(self) -> int:
        return sum(self.rows_per_layer)

    @property
    def primes(self) -> list[int]:
        return [p for layer in self.primes_3d for row in layer for p in row]


class NgramHashState(torch.nn.Module):
    """Maps each position to the hash ids of the n-grams ending there."""

    DEAD = -1

    def __init__(self, cfg, layout: EngramLayout, token_map, compressed_vocab: int):
        super().__init__()
        self.layout, self.cfg = layout, cfg
        self.compressed_vocab = compressed_vocab
        self.pad_id = token_map[cfg.engram_pad_token_id]
        self.register_buffer("token_map",
                             torch.tensor(list(token_map), dtype=torch.long),
                             persistent=False)
        # [n_layers, n_sizes, n_heads] -- one prime per (layer, n-gram size, head)
        self.register_buffer("primes", torch.tensor(layout.primes_3d, dtype=torch.long),
                             persistent=False)
        self.register_buffer("multipliers",
                             torch.tensor(layout.multipliers_3d, dtype=torch.long),
                             persistent=False)
        self.register_buffer("offsets",
                             torch.tensor(layout.offsets_3d, dtype=torch.long),
                             persistent=False)
        self.register_buffer("cache",
                             torch.zeros(cfg.max_batch_size, cfg.max_seq_len,
                                         dtype=torch.long), persistent=False)

    @torch.no_grad()
    def reset(self, max_len: int | None = None, batch: int | None = None) -> None:
        """Resize (and clear) the id history. Called from Transformer.init_caches.

        The cache must be resized whenever the required batch or length changes: it is
        allocated at construction from cfg.max_batch_size / cfg.max_seq_len, and a
        training batch or a benchmark prompt can exceed either. It must be RESET
        whenever a NEW SEQUENCE starts, and never between chunks of the same sequence
        (the n-gram context spans the boundary).
        """
        n = max_len or self.cache.size(1)
        b = batch or self.cache.size(0)
        if (n, b) != tuple(self.cache.shape):
            self.cache = torch.zeros(b, n, dtype=self.cache.dtype,
                                     device=self.cache.device)
        else:
            self.cache.zero_()

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, start_pos: int, token_mask=None):
        """-> [b, S, n_engram_layers, n_hash_cols], offset into the flat table."""
        b, S = input_ids.shape
        compressed = self.token_map[input_ids]
        if token_mask is not None:
            compressed = torch.where(token_mask, compressed, self.DEAD)
        self.cache[:b, start_pos:start_pos + S] = compressed
        pos = torch.arange(start_pos, start_pos + S, device=input_ids.device).expand(b, S)
        tokens, blocked = [], torch.zeros_like(pos, dtype=torch.bool)
        for shift in range(self.layout.max_ngram):
            src = self.cache[:b].gather(1, (pos - shift).clamp_min(0))
            blocked = blocked | (pos < shift) | (src == self.DEAD)
            tokens.append(torch.where(blocked, self.pad_id, src))
        tokens = torch.stack(tokens, dim=-1)                    # [b,S,max_ngram]
        nl, ns, nh = self.layout.n_layers, self.layout.n_sizes, self.layout.n_heads
        # [b,S,n_layers,n_sizes,n_heads,max_ngram]
        products = (tokens[:, :, None, None, None, :]
                    * self.multipliers[None, None, :, :, :, None])
        rolling = products[..., 0]
        for i in range(1, self.layout.max_ngram):
            rolling = torch.bitwise_xor(rolling, products[..., i])
        h = (rolling % self.primes[None, None]) + self.offsets[None, None]
        return h.reshape(b, S, nl, ns * nh)                      # [b,S,nl,n_hash_cols]

    def check_bounds(self, hash_ids: torch.Tensor) -> None:
        """Every index must address a real row. Cheap; call once after building."""
        rows = self.layout.rows_per_layer
        hi = int(hash_ids.max()) + 1
        if hi > rows[0]:
            raise IndexError(
                f"Engram hash id {hi - 1} exceeds layer 0's {rows[0]} rows. Either the "
                f"prime layout and offsets disagree, or engram_num_embeddings is wrong.")


class ParallelEngramEmbedding(torch.nn.Module):
    """fp8 rows with UE8M0 row scales, dequantized per lookup."""

    def __init__(self, rows: int, dim: int, block: int = 32):
        super().__init__()
        assert dim % block == 0, f"engram head_dim {dim} must be a multiple of {block}"
        self.block, self.dim = block, dim
        self.weight = torch.nn.Parameter(torch.zeros(rows, dim, dtype=torch.float32))
        self.scale = torch.nn.Parameter(torch.zeros(rows, dim // block, dtype=torch.float32))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        v = self.weight[idx].float().view(*idx.shape, -1, self.block)
        s = self.scale[idx].float().unsqueeze(-1)
        return (v * s).flatten(-2).to(self.weight.dtype)


class Engram(torch.nn.Module):
    """Writes an n-gram lookup into the residual stream, gated by stream/key agreement."""

    def __init__(self, cfg, layer_index: int):
        super().__init__()
        self.dim, self.hc = cfg.dim, cfg.hc_mult
        self.eps, self.clamp_value = cfg.rms_norm_eps, 1e-6
        self.embed = ParallelEngramEmbedding(cfg.engram_num_embeddings[layer_index],
                                            cfg.engram_head_dim)
        cols = (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads
        self.wkv = torch.nn.Linear(cols * cfg.engram_head_dim,
                                   cfg.dim * (self.hc + 1), bias=False)
        self.q_weight = torch.nn.Parameter(torch.ones(self.hc, cfg.dim))
        self.k_weight = torch.nn.Parameter(torch.ones(self.hc, cfg.dim))

    def forward(self, x: torch.Tensor, hash_ids: torch.Tensor,
                token_mask: torch.Tensor | None = None):
        kv = self.wkv(self.embed(hash_ids).flatten(-2))
        key, value = kv.split([self.hc * self.dim, self.dim], dim=-1)
        key = key.view(*key.shape[:-1], self.hc, self.dim)
        w = self.q_weight.to(x.dtype) * self.k_weight.to(x.dtype)
        # The gate is ONE scalar per stream copy, but the reductions behind it run over
        # `dim` = 1344 terms. Those stay fp32 -- a 1344-term fp16 sum carries a few
        # percent of relative error, which is far too much to feed a sigmoid -- but they
        # are computed with `rms_recip`/`sum(dtype=)`, neither of which materialises an
        # fp32 copy of the stream. The elementwise products then run in the activation
        # dtype, which is what removes the three fp32 [b,s,hc,d] temporaries this used to
        # allocate on every forward (Engram is one layer, so this is the smallest of the
        # fp32 wins -- but it is the one on the critical path).
        rstd = (rms_recip(x, self.eps).squeeze(-1)
                * rms_recip(key, self.eps).squeeze(-1))
        dot = (x * key * w).sum(-1, dtype=torch.float32) * rstd * (self.dim ** -0.5)
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (x + gate.unsqueeze(-1).to(x.dtype) * value.unsqueeze(-2)).to(x.dtype)


def identity_token_map(vocab_size: int) -> list[int]:
    """Fallback token map. Real runs derive this from the tokenizer (see tokenize.py)."""
    return list(range(vocab_size))
