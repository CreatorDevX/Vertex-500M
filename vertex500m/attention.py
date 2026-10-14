"""Attention: shared-KV latent + sliding window + CSA2 compressor and indexer.

This is the port of spec 7 (attention core) and spec 8 (CSA2), with one deliberate
change from the reference: `shared` is an EXPLICIT object rather than a module-level
global singleton.

Why. The reference declares

    class SharedAttentionRuntime:
        compress_kv = index_k = topk_idxs = candidates = None
    shared = SharedAttentionRuntime()

and layers mutate it in place (`self.index_k[...] = k`, `self.compress_kv[...] = latent`)
while also reassigning its attributes. That works for one model instance but shares
state across every instance in the process, which makes tests order-dependent. More
importantly it interacts badly with gradient checkpointing: replaying a segment during
backward re-executes those in-place writes, and `PoolState` accumulates partial groups
across calls. Threading the object explicitly makes the dependency visible and lets
`checkpointing.py` snapshot it around a replay.

Ordering invariant (spec 4.2): layers execute in ascending index order and a producer
must write before any consumer reads. This file relies on that.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .amp import cache_dtype
from .config import Cfg
from .quant import act_quant, fp4_act_quant
from .rope import RMSNorm, precompute_freqs_cis, rope_tail
from .sparse_attn import (get_chunk_window_topk_idxs, get_window_topk_idxs,
                          select_candidate_blocks, sparse_attn)

@dataclass
class SharedAttentionRuntime:
    """What attention layers hand down the stack instead of recomputing.

    Two very different kinds of state live here, and the difference is load-bearing for
    gradient checkpointing:

      * GRAPH-CARRYING: `compress_kv` and `index_k` are buffers that a producing layer
        writes in place from tensors that require grad. The gradient a consumer layer
        sends back through this cache is the ONLY path by which a compressor or an
        indexer's parameters receive signal at all. Never rebind these to a detached
        clone -- that silently severs the graph and every compressor/indexer gradient
        drops to zero or garbage while the loss still looks fine.

      * GRAPH-FREE: `topk_idxs` (int32) and `candidates` (bool). These are pure selection
        metadata, safe to snapshot and restore, and they MUST be restored because
        checkpoint replay runs segments in reverse order: segment N's replay overwrites
        topk_idxs, so segment N-1's replay would otherwise read segment N's value.

      * GRAD-CARRYING PUBLISH (`compress_fresh` / `index_fresh`, below): the producer's
        fresh latent/keys as plain tensors. Consumers read THESE during training so the
        compressor/indexer grad edge never passes through a mutated buffer (which
        dynamo/AOT cannot replay -- see `compressed_branch`). Rebinding discipline is
        the same as the buffers': producers publish before consumers read, rewrites are
        idempotent, so reverse-order checkpoint replay resolves correctly. Never
        snapshot these during TRAINING (that would sever the graph, same as the
        buffers); the speculative DECODE snapshot does clone them, because there the
        values only choose which of two non-equivalent reads `_resolve_fresh` returns.
        See `Transformer.snapshot_decode_state`.

    The per-producer buffers (`self.compress_kv`, `self.index_k`) are written
    idempotently -- same slice, same values -- so they need no value restoration, which
    is precisely why they can keep their graph.
    """

    compress_kv: torch.Tensor | None = None
    index_k: torch.Tensor | None = None
    topk_idxs: torch.Tensor | None = None
    candidates: torch.Tensor | None = None
    # (tensor, first_group, n_groups): the producer's fresh values, grad-carrying.
    # The buffers above are persistence for decode (written under no_grad); these
    # are what training consumers read. See `compressed_branch`.
    compress_fresh: tuple | None = None
    index_fresh: tuple | None = None

    def snapshot_graph_free(self) -> tuple:
        """Save only the graph-free selection metadata. See the class docstring."""
        return (None if self.topk_idxs is None else self.topk_idxs,
                None if self.candidates is None else self.candidates)

    def restore_graph_free(self, snap: tuple) -> None:
        self.topk_idxs, self.candidates = snap

    @staticmethod
    def _resolve_fresh(fresh: tuple | None, buf: torch.Tensor | None,
                       b: int, need: int) -> torch.Tensor:
        """The consumer's read: fresh publish when it covers [0, need), else buffer.

        Training forwards start at position 0, so the producer's fresh write covers
        the whole needed prefix and the consumer takes it directly -- a pure
        functional grad edge that never touches mutated storage (which a compiled
        backward cannot replay). Continuation forwards (start_pos > 0) are
        inference-only and read the persistent buffer. Both paths yield identical
        values; only the grad edge differs. The branch is on ints/shapes, so it
        specializes cleanly under torch.compile.
        """
        if fresh is not None:
            tensor, first, n = fresh
            if first == 0 and n >= need:
                return tensor[:, :need]
        return buf[:b, :need]


class PoolState:
    """Partial-group state for a ratio>1 compressor, in ABSOLUTE position terms.

    Holds the `count` buffered tokens of the current incomplete group, all belonging to
    group floor(next_position / ratio). Invariant: count == next start_pos % ratio.

    `dtype`/`device` follow the caches in `Attention.init_caches`. The compressor
    concatenates these into the incoming `kv`/`score` (see `Compressor.forward`), so a
    dtype mismatch here silently promotes the pooled activation back to fp32.
    """

    def __init__(self, batch: int, ratio: int, dim: int, dtype=torch.float32, device=None):
        self.ratio = ratio
        self.count = 0
        self.kv = torch.zeros(batch, ratio, dim, dtype=dtype, device=device)
        self.score = torch.full((batch, ratio, dim), -torch.inf, dtype=dtype,
                                device=device)

    def clone(self) -> "PoolState":
        n = PoolState.__new__(PoolState)
        n.ratio, n.count = self.ratio, self.count
        n.kv, n.score = self.kv.clone(), self.score.clone()
        return n

    def reset(self) -> None:
        """Clear for a fresh sequence: exactly what a new PoolState would hold.

        `score` refills to -inf, NOT zero -- a fresh PoolState builds it with
        `torch.full(..., -torch.inf)`, and the buffered scores participate in the
        compressor's softmax whenever a partial group is pending.
        """
        self.count = 0
        self.kv.zero_()
        self.score.fill_(float("-inf"))

    def copy_from(self, other: "PoolState") -> None:
        # Cache write under no_grad; see window_branch for why (this runs inside
        # checkpoint replay as well as decode restore).
        with torch.no_grad():
            self.count = other.count
            self.kv.copy_(other.kv)
            self.score.copy_(other.score)


class Compressor(torch.nn.Module):
    """Pools `ratio` consecutive tokens into one KV latent with a learned softmax gate.

    Chunk-general unified forward: prepends the buffered partial group, pools every
    complete absolute group, keeps the remainder. Fresh prefill, 1-token decode and
    chunked prefill are all the same code path.
    Returns (latent | None, n_groups, first_group_index).
    """

    def __init__(self, dim: int, head_dim: int, ratio: int, eps: float):
        super().__init__()
        self.ratio, self.head_dim, self.eps = ratio, head_dim, eps
        self.norm = RMSNorm(head_dim, eps)
        self.wkv = torch.nn.Linear(dim, head_dim, bias=False)
        if ratio > 1:
            self.wgate = torch.nn.Linear(dim, head_dim, bias=False)

    def forward(self, x: torch.Tensor, start_pos: int, state: PoolState):
        r = self.ratio
        b, S, _ = x.shape
        if r == 1:
            # one latent per token: n_groups == S, groups start at absolute start_pos
            return self.norm(self.wkv(x)), S, start_pos

        # x is already in the autocast dtype (or fp32 with AMP off); `wkv`/`wgate` are Linear
        # layers and autocast will take them to the tensor cores as they are. The old
        # `.float()` here upcast a full [b,s,d] activation only for autocast to cast it
        # straight back down. The pooled output is cast to x.dtype below either way.
        kv, score = self.wkv(x), self.wgate(x)
        pend = state.count
        assert pend == start_pos % r, (
            f"PoolState desync: count={pend} but start_pos={start_pos}, ratio={r}. "
            f"A replayed segment or a re-used cache advanced it twice."
        )
        # PoolState is preallocated at max_batch_size, so slice to the real batch here.
        # Every other cache in this file is sliced the same way (spec 17.2).
        if pend:
            kv = torch.cat([state.kv[:b, :pend], kv], dim=1)
            score = torch.cat([state.score[:b, :pend], score], dim=1)
        total = pend + S
        n_groups, rem = total // r, total % r
        out = None
        if n_groups:
            gk = kv[:, :n_groups * r].unflatten(1, (-1, r))
            gs = score[:, :n_groups * r].unflatten(1, (-1, r))
            out = self.norm(((gk * gs.softmax(dim=2)).sum(dim=2)).to(x.dtype))
        if rem:
            # Cache write under no_grad; see window_branch for why.
            with torch.no_grad():
                state.kv[:b, :rem] = kv[:, -rem:]
                state.score[:b, :rem] = score[:, -rem:]
            state.count = rem
        else:
            state.count = 0
        first_group = (start_pos - pend) // r
        return out, n_groups, first_group


class Indexer(torch.nn.Module):
    """Keeps the index_topk best compressed positions per query.

    Scoring is relu(q.k) per head, weighted by a learned per-head weight, summed over
    heads. This is DSA-style, NOT a softmax.
    """

    def __init__(self, cfg: Cfg, layer_id: int, owns_k: bool):
        super().__init__()
        self.cfg, self.owns_k = cfg, owns_k
        self.ratio = cfg.compress_ratios[layer_id]
        self.is_candidate_source = layer_id == cfg.candidate_source_layer_id
        self.uses_candidates = 0 <= cfg.candidate_source_layer_id < layer_id
        self.softmax_scale = cfg.index_head_dim ** -0.5
        # The constant folded into `q` before the dot product; see Indexer.forward.
        self.score_scale = self.softmax_scale * (cfg.index_n_heads ** -0.5)
        self.wq_b = torch.nn.Linear(cfg.q_lora_rank,
                                    cfg.index_n_heads * cfg.index_head_dim, bias=False)
        self.weights_proj = torch.nn.Linear(cfg.dim, cfg.index_n_heads, bias=False)
        if owns_k:
            self.wk = torch.nn.Linear(cfg.head_dim, cfg.index_head_dim, bias=False)
            self.k_norm = RMSNorm(cfg.index_head_dim, cfg.rms_norm_eps)

    def make_keys(self, latent: torch.Tensor, freqs_cis: torch.Tensor,
                  first_group: int) -> torch.Tensor:
        """latent: pre-RoPE, pre-quant compressed latent from this layer's compressor.

        A latent stands for the FIRST token of its group, so group g rotates at g*ratio.
        """
        n = latent.size(1)
        r = self.ratio
        freqs = freqs_cis[first_group * r:(first_group + n) * r:r]
        k = self.k_norm(self.wk(latent))
        k = rope_tail(k, self.cfg.qk_rope_head_dim, freqs)
        k, _ = fp4_act_quant(k, block=32, e4m3_scale=False, inplace=False)
        return k

    def forward(self, x, qr, freqs_cis, start_pos, offset, index_k, shared):
        cfg = self.cfg
        b, S, _ = x.shape
        end = start_pos + S
        q = self.wq_b(qr).view(b, S, cfg.index_n_heads, cfg.index_head_dim)
        q = rope_tail(q, cfg.qk_rope_head_dim, freqs_cis[start_pos:end])
        q, _ = fp4_act_quant(q, block=32, e4m3_scale=False, inplace=False)
        # Fold the constant score scale into q BEFORE the dot product.
        #
        # Exact, not an approximation. The old code computed
        #     final = sum_h relu(dot_h) * w_h * c        with c = softmax_scale * n_heads**-0.5
        # and this computes
        #     final = sum_h relu(dot_h * c) * w_h
        # which are the same number because relu(c*x) == c*relu(x) for every c > 0. One
        # positive scalar, applied once, not per head.
        #
        # It is also the only overflow risk in the model under `--amp fp16`. `score` is a
        # ReLU-weighted sum over 8 heads of 64-dim dots and is O(T) per query -- the
        # largest single tensor in the forward pass. Unscaled it can exceed fp16's
        # 65504, and an overflowed score does NOT raise: it saturates to inf, topk
        # happily ranks it first, and the model retrieves the wrong keys along a
        # perfectly plausible loss curve. Scaling first removes the exposure for free.
        q = q * self.score_scale
        w = self.weights_proj(x)
        ik = index_k[:b, :end // self.ratio]
        # Chunked over queries (cfg.index_chunk): the score tensor is [b,S,H,R] --
        # at S=R=8192 that is ~268 MB fp16 per temporary with several live at once.
        # Every op below is per-query independent (scores, masking, candidate
        # selection, topk), so chunking over S bounds the transient to
        # [b,chunk,H,R] with BIT-IDENTICAL results (the contracted dims are
        # untouched). Under torch.compile the static trip count unrolls.
        R = ik.size(1)
        k = max(1, min(cfg.index_topk, end // self.ratio))
        ar_r = torch.arange(R, device=x.device).view(1, 1, R)
        chunk = max(1, cfg.index_chunk)
        idx_parts, cand_parts = [], []
        for s0 in range(0, S, chunk):
            s1 = min(s0 + chunk, S)
            # chunk-general reachability: query at absolute position p sees groups
            # completed by p, i.e. group j iff j*ratio + (ratio-1) <= p <=> j < cl[p]
            cl = ((start_pos + s0 + torch.arange(1, s1 - s0 + 1, device=x.device))
                  // self.ratio).view(1, s1 - s0, 1).expand(b, s1 - s0, 1)
            score = torch.einsum("bshd,btd->bsht", q[:, s0:s1], ik)
            score = (score.relu() * w[:, s0:s1].unsqueeze(-1)).sum(dim=2)
            score = score.masked_fill(ar_r >= cl, float("-inf"))
            if self.is_candidate_source:
                cand = select_candidate_blocks(
                    score, cl, cfg.candidate_topk_blocks, cfg.candidate_block_size)
                cand_parts.append(cand)
            elif self.uses_candidates and shared.candidates is not None:
                score = score.masked_fill(
                    ~shared.candidates[:, s0:s1], float("-inf"))
            idx = score.topk(k, dim=-1, sorted=False).indices.sort(dim=-1).values
            idx_parts.append(torch.where(idx < cl, idx + offset, -1))
        if self.is_candidate_source:
            shared.candidates = (torch.cat(cand_parts, dim=1) if len(cand_parts) > 1
                                 else cand_parts[0])
        idx = torch.cat(idx_parts, dim=1) if len(idx_parts) > 1 else idx_parts[0]
        return idx.int()


class Attention(torch.nn.Module):
    """Latent attention over two KV sources in one call: sliding window + top-k."""

    def __init__(self, cfg: Cfg, layer_id: int):
        super().__init__()
        self.cfg, self.layer_id = cfg, layer_id
        self.dim, self.n_heads, self.hd = cfg.dim, cfg.n_heads, cfg.head_dim
        self.rd = cfg.qk_rope_head_dim
        self.window = cfg.sliding_window
        self.ratio = cfg.compress_ratios[layer_id] if layer_id < len(cfg.compress_ratios) else 0
        self.softmax_scale = self.hd ** -0.5

        self.wq_a = torch.nn.Linear(cfg.dim, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank, cfg.rms_norm_eps)
        self.wq_b = torch.nn.Linear(cfg.q_lora_rank, cfg.n_heads * self.hd, bias=False)
        self.wkv = torch.nn.Linear(cfg.dim, self.hd, bias=False)
        self.kv_norm = RMSNorm(self.hd, cfg.rms_norm_eps)
        self.wo_a = torch.nn.Linear(cfg.n_heads * self.hd // cfg.o_groups,
                                    cfg.o_groups * cfg.o_lora_rank, bias=False)
        self.wo_b = torch.nn.Linear(cfg.o_groups * cfg.o_lora_rank, cfg.dim, bias=False)
        self.attn_sink = torch.nn.Parameter(torch.zeros(cfg.n_heads, dtype=torch.float32))

        self.is_kv_source = layer_id < cfg.n_layers and layer_id in cfg.kv_source_layer_ids
        self.is_index_source = layer_id < cfg.n_layers and layer_id in cfg.index_source_layer_ids
        self.compressor = (Compressor(cfg.dim, self.hd, self.ratio, cfg.rms_norm_eps)
                           if self.is_kv_source else None)
        self.indexer = (Indexer(cfg, layer_id, self.is_kv_source)
                        if self.is_index_source else None)
        self.pool_state: PoolState | None = None

        orig, theta = ((cfg.rope_original_seq_len, cfg.compress_rope_theta)
                       if self.ratio else (0, cfg.rope_theta))
        self.register_buffer("freqs_cis", precompute_freqs_cis(
            self.rd, cfg.max_seq_len, orig, theta,
            cfg.rope_factor, cfg.beta_fast, cfg.beta_slow), persistent=False)

    # ---- caches -----------------------------------------------------------
    def init_caches(self, batch: int, max_len: int, device=None, dtype=None):
        """Allocate the CSA2 caches.

        `dtype=None` resolves from the active AMP mode: fp32 when mixed precision is off
        (the historical behaviour), the autocast dtype when it is on. This matters more
        than the signature suggests. `Attention.forward` concatenates `window_kv` with the
        freshly computed KV, and one fp32 operand promotes the WHOLE concatenation, so
        caches left in fp32 under `--amp fp16` would keep the gathered chunk fp32 and
        forfeit the tensor-core path on the attention core -- 26% of this model's FLOPs.
        See `amp.cache_dtype`.

        Repeated calls with identical geometry do NOT reallocate: the training loop
        calls `init_caches` every step with the same batch/length, and reallocating
        every layer's buffers every step is allocator churn plus redundant fills.
        When every live buffer already matches the requested shape/dtype/device they
        are cleared in place (one memset each) and the same storage is kept.
        """
        dtype = cache_dtype(device) if dtype is None else dtype
        dev = torch.empty(0, device=device).device  # resolves None -> default device
        shapes = {"window_kv": (batch, self.window, self.hd)}
        if self.is_kv_source:
            shapes["compress_kv"] = (batch, max_len // self.ratio, self.hd)
            shapes["index_k"] = (batch, max_len // self.ratio,
                                 self.cfg.index_head_dim)
        if self._caches_match(batch, shapes, dtype, dev):
            self.window_kv.zero_()
            if self.is_kv_source:
                self.compress_kv.zero_()
                self.index_k.zero_()
                self.pool_state.reset()
            return
        self.window_kv = torch.zeros(batch, self.window, self.hd, dtype=dtype,
                                     device=device)
        if self.is_kv_source:
            self.compress_kv = torch.zeros(batch, max_len // self.ratio, self.hd,
                                           dtype=dtype, device=device)
            self.index_k = torch.zeros(batch, max_len // self.ratio,
                                       self.cfg.index_head_dim, dtype=dtype,
                                       device=device)
            self.pool_state = PoolState(batch, self.ratio, self.hd, dtype=dtype,
                                        device=device)
        else:
            self.compress_kv = None
            self.index_k = None

    def _caches_match(self, batch: int, shapes: dict, dtype: torch.dtype,
                      dev: torch.device) -> bool:
        """True when every live buffer already has the requested geometry."""
        for name, shape in shapes.items():
            buf = getattr(self, name, None)
            if (buf is None or tuple(buf.shape) != tuple(shape)
                    or buf.dtype != dtype or buf.device != dev):
                return False
        if self.is_kv_source:
            ps = self.pool_state
            if ps is None:
                return False
            if (tuple(ps.kv.shape) != (batch, self.ratio, self.hd)
                    or tuple(ps.score.shape) != (batch, self.ratio, self.hd)):
                return False
            if ps.kv.dtype != dtype or ps.kv.device != dev:
                return False
            if ps.score.dtype != dtype or ps.score.device != dev:
                return False
        return True

    # ---- sliding window ---------------------------------------------------
    # NOTE (compile): every ring write below runs under `torch.no_grad()`. The ring
    # buffers never require grad, so no gradient flows through the write itself --
    # kv's grad arrives via the reads (`torch.cat` below) -- but dynamo/AOT's
    # functionalization of in-place mutations bakes first-invocation state into the
    # compiled backward, and the SECOND compiled step then fails with "backward
    # through the graph a second time". no_grad keeps the mutation out of the joint
    # graph and repeated compiled steps work. Verified by rerunning a compiled
    # fwd+bwd twice (see the --compile notes in train.py).
    def window_branch(self, x: torch.Tensor, freqs: torch.Tensor, start_pos: int):
        b, S, _ = x.shape
        kv = self.kv_norm(self.wkv(x))
        kv = rope_tail(kv, self.rd, freqs)
        kv, _ = act_quant(kv, block=32, round_scale=True, inplace=False)
        if start_pos == 0:
            if S <= self.window:
                with torch.no_grad():
                    self.window_kv[:b, :S] = kv
            else:
                cutoff = S % self.window
                tail = kv[:, -self.window:]
                with torch.no_grad():
                    self.window_kv[:b, cutoff:self.window] = tail[:, :self.window - cutoff]
                    self.window_kv[:b, :cutoff] = tail[:, self.window - cutoff:]
            return kv, get_window_topk_idxs(self.window, b, S, start_pos, x.device)
        if S == 1:
            with torch.no_grad():
                self.window_kv[:b, start_pos % self.window] = kv[:, 0]
            return (self.window_kv[:b],
                    get_window_topk_idxs(self.window, b, S, start_pos, x.device))
        # chunk-general (start_pos > 0, S > 1): attend [ring | chunk]
        n = min(S, self.window)
        if n:
            # One indexed write, not n scalar ones. The old loop issued one kernel
            # per position (up to `window` launches per layer per forward on this
            # path); the slots are consecutive mod `window` with n <= window, so
            # they are distinct and the single index_copy_ is exactly equivalent.
            # Under no_grad (see above) kv's grad arrives via the `torch.cat` read
            # below, not via this write.
            slots = (start_pos + S - n + torch.arange(n, device=kv.device)) % self.window
            with torch.no_grad():
                self.window_kv[:b].index_copy_(1, slots, kv[:, S - n:])
        window_kv = torch.cat([self.window_kv[:b], kv], dim=1)
        return window_kv, get_chunk_window_topk_idxs(self.window, b, S, start_pos,
                                                     x.device)

    # ---- compressed KV + indexer -----------------------------------------
    def compressed_branch(self, x, qr, start_pos, offset, shared):
        b, S, _ = x.shape
        r = self.ratio
        compress_len = (start_pos + S) // r
        latent, n_groups, first_group = None, 0, 0
        if self.is_kv_source:
            latent, n_groups, first_group = self.compressor(x, start_pos, self.pool_state)
            shared.compress_kv = self.compress_kv       # publish BEFORE consumers read
        if self.is_index_source:
            if compress_len == 0:
                idxs = torch.empty(b, S, 0, dtype=torch.int32, device=x.device)
            else:
                if self.indexer.owns_k and latent is not None:
                    k = self.indexer.make_keys(latent, self.freqs_cis, first_group)
                    # Persistence write only (no_grad): the indexer grad edge
                    # travels via index_fresh below, never through mutated
                    # storage -- see SharedAttentionRuntime and window_branch.
                    with torch.no_grad():
                        self.index_k[:b, first_group:first_group + n_groups] = k
                    shared.index_k = self.index_k
                    shared.index_fresh = (k, first_group, n_groups)
                ik = SharedAttentionRuntime._resolve_fresh(
                    shared.index_fresh, shared.index_k, b,
                    (start_pos + S) // self.indexer.ratio)
                idxs = self.indexer(x, qr, self.freqs_cis, start_pos, offset,
                                    ik, shared)
            shared.topk_idxs = idxs
        else:
            idxs = shared.topk_idxs
        if latent is not None:
            # group g rotates at absolute position g*ratio (its first token)
            freqs_g = self.freqs_cis[first_group * r:(first_group + n_groups) * r:r]
            latent = rope_tail(latent, self.rd, freqs_g)
            latent, _ = fp4_act_quant(latent, block=16, e4m3_scale=True, inplace=False)
            # Persistence write only (no_grad): the compressor grad edge travels
            # via compress_fresh below. `latent` here is exactly the values the
            # buffer holds, so training consumers see bit-identical values.
            with torch.no_grad():
                self.compress_kv[:b, first_group:first_group + n_groups] = latent
            shared.compress_fresh = (latent, first_group, n_groups)
        return SharedAttentionRuntime._resolve_fresh(
            shared.compress_fresh, shared.compress_kv, b, compress_len), idxs

    # ---- forward ----------------------------------------------------------
    def forward(self, x: torch.Tensor, start_pos: int, shared: SharedAttentionRuntime):
        b, S, _ = x.shape
        cfg = self.cfg
        freqs = self.freqs_cis[start_pos:start_pos + S]
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).view(b, S, cfg.n_heads, self.hd)
        q = rope_tail(q, self.rd, freqs)

        kv, idxs = self.window_branch(x, freqs, start_pos)
        if self.ratio:
            ckv, cidx = self.compressed_branch(x, qr, start_pos, kv.size(1), shared)
            kv = torch.cat([kv, ckv], dim=1)
            idxs = torch.cat([idxs, cidx], dim=-1)

        o = sparse_attn(q, kv, self.attn_sink, idxs.contiguous(),
                        self.softmax_scale, chunk=cfg.attn_chunk)
        o = rope_tail(o, self.rd, freqs, inverse=True)

        g = cfg.o_groups
        # No `.to(self.wo_a.weight.dtype)` here. The weight is always fp32, and the
        # einsum below is autocast-eligible, so under autocast it computes in the
        # activation dtype either way. The old cast upcast the full [b,S,H*hd]
        # attention output fp16->fp32 on every layer just for autocast to cast it
        # straight back down. Outside autocast this is a no-op (o is already fp32).
        o = o.view(b, S, g, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, self.wo_a.weight.view(g, cfg.o_lora_rank, -1))
        return self.wo_b(o.flatten(2))
