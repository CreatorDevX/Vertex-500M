"""Block, Transformer, generation (spec 4.2, 5.6, 12, 21.11).

Three things differ from the spec's reference implementation, each documented at its
definition:

  * `VertexHead` replaces `ParallelHead` because embeddings are tied: there is exactly
    one [V, d] tensor in the model and it appears once in the state dict.
  * DSpark is constructed on demand by `attach_dspark()`, so pretraining pays nothing
    for it and never builds the `main_hiddens` graph edge.
  * The layer loop is optionally wrapped in segment-N gradient checkpointing, with the
    shared attention runtime snapshotted around each replay.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .amp import cache_dtype, current_autocast_dtype
from .attention import Attention, SharedAttentionRuntime
from .config import Cfg
from .engram import Engram, EngramLayout
from .ffn import DenseFFN
from .hc import (hc_mixes, hc_post, hc_pre, init_hc_params, make_identity_pre_mix,
                 mix_hc_for, permutation_basis)
from .quant import act_quant
from .rope import RMSNorm
from .sparse_attn import get_dspark_topk_idxs, sparse_attn


class VertexHead(torch.nn.Module):
    """LM head sharing storage with `embed.weight`. NOT a Parameter of its own.

    The spec's ParallelHead holds `weight` as a separate fp32 Parameter of shape
    [V, d], which is the main reason the released config sets
    `tie_word_embeddings: false`. With tying the head becomes a VIEW.

    Implementation note: the embedding is stored in `object.__setattr__` so it does NOT
    become a child module. Registering it normally (`self.embed = embed`) makes
    `nn.Module.__setattr__` treat it as a submodule, and the tied tensor then appears
    TWICE in the state dict -- `embed.weight` and `head.embed.weight` -- which is the
    double-storage this class exists to avoid, and which also breaks any
    "one [V,d] tensor in the model" assertion.

    Do NOT wrap `embed.weight` in a second Parameter either; that silently un-ties them.
    """

    def __init__(self, embed: torch.nn.Embedding):
        super().__init__()
        # bypass nn.Module's submodule registration
        object.__setattr__(self, "_embed", embed)

    @property
    def embed(self) -> torch.nn.Embedding:
        return self._embed

    def forward(self, x: torch.Tensor, full_logits: bool = False):
        if not full_logits:
            x = x[:, -1]
        # No `x.float()`: F.linear is autocast-eligible, so under autocast the logit
        # matmul runs in the compute dtype either way and the upcast only bought a
        # wasted fp16->fp32->fp16 round trip of [b,s,d]. Outside autocast x is
        # already fp32. (weight.float() is a no-op -- parameters stay fp32 in every
        # AMP mode -- kept to document that the head reads fp32 storage.)
        return F.linear(x, self._embed.weight.float())


class Block(torch.nn.Module):
    """Residual stream is `hc` parallel copies (Hyper-Connections, Single-Pass)."""

    attention_cls = Attention

    def __init__(self, cfg: Cfg, layer_id: int, engram_layout: EngramLayout | None = None,
                 hc_variant: str = "mhc"):
        super().__init__()
        self.layer_id, self.cfg = layer_id, cfg
        self.engram_idx = (engram_layout.layer_ids.index(layer_id)
                           if engram_layout and layer_id in engram_layout.layer_ids else None)
        self.attn = self.attention_cls(cfg, layer_id)
        self.ffn = DenseFFN(cfg, layer_id)
        self.engram = Engram(cfg, self.engram_idx) if self.engram_idx is not None else None
        self.attn_norm = RMSNorm(cfg.dim, cfg.rms_norm_eps)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.rms_norm_eps)

        self.hc_mult = hc = cfg.hc_mult
        self.hc_sinkhorn_iters, self.hc_eps = cfg.hc_sinkhorn_iters, cfg.hc_eps
        self.norm_eps = cfg.rms_norm_eps
        self.hc_variant = hc_variant
        mix_hc = mix_hc_for(hc, hc_variant)
        self.mix_hc = mix_hc
        p = dict(dtype=torch.float32)
        self.hc_attn_fn = torch.nn.Parameter(torch.zeros(mix_hc, hc * cfg.dim, **p))
        self.hc_ffn_fn = torch.nn.Parameter(torch.zeros(mix_hc, hc * cfg.dim, **p))
        self.hc_attn_base = torch.nn.Parameter(torch.zeros(mix_hc, **p))
        self.hc_ffn_base = torch.nn.Parameter(torch.zeros(mix_hc, **p))
        self.hc_attn_scale = torch.nn.Parameter(torch.ones(3, **p))
        self.hc_ffn_scale = torch.nn.Parameter(torch.ones(3, **p))
        init_hc_params(self.hc_attn_fn, self.hc_attn_base, self.hc_attn_scale,
                       hc, hc_variant)
        init_hc_params(self.hc_ffn_fn, self.hc_ffn_base, self.hc_ffn_scale,
                       hc, hc_variant)
        if hc_variant == "mhc-lite":
            self.register_buffer("perm", permutation_basis(hc), persistent=True)

    def mixes(self, x, fn, scale, base):
        return hc_mixes(x, fn, scale, base, self.hc_mult, self.norm_eps,
                        self.hc_sinkhorn_iters, self.hc_eps,
                        getattr(self, "perm", None))

    def forward(self, x, start_pos, pre_mix, shared, *attn_args):
        # attention consumes the PREVIOUS sublayer's `pre` (Single-Pass hand-off)
        residual = x
        attn_pre, attn_post, attn_comb = self.mixes(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm(hc_pre(x, pre_mix))
        x = self.attn(x, start_pos, shared, *attn_args)
        x = hc_post(x, residual, attn_post, attn_comb)

        # FFN consumes THIS block's attention pre
        residual = x
        ffn_pre, ffn_post, ffn_comb = self.mixes(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm(hc_pre(x, attn_pre))
        x = self.ffn(x)
        x = hc_post(x, residual, ffn_post, ffn_comb)
        return x, ffn_pre                    # consumed by the next sublayer


class Transformer(torch.nn.Module):
    def __init__(self, cfg: Cfg, engram_layout: EngramLayout | None = None,
                 engram_hashes=None, hc_variant: str = "mhc"):
        super().__init__()
        self.cfg, self.n_layers = cfg, cfg.n_layers
        self.hc_mult = cfg.hc_mult
        self.hc_variant = hc_variant
        self.engram_layout, self.engram_hash = engram_layout, engram_hashes

        self.embed = torch.nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = torch.nn.ModuleList(
            [Block(cfg, i, engram_layout, hc_variant) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, cfg.rms_norm_eps)
        self.head = VertexHead(self.embed)

        # DSpark lives in a SEPARATE list, not in self.layers, so the layer loop and
        # the checkpoint segmentation stay clean.
        self.mtp = torch.nn.ModuleList()
        self.target_layer_ids: tuple = ()

        self._init_attention_runtime()

    # ------------------------------------------------------------------ #
    def _init_attention_runtime(self) -> None:
        self.shared = SharedAttentionRuntime()

    # ------------------------------------------------------------------ #
    def attach_dspark(self) -> "Transformer":
        """Construct the draft stages. Call AFTER init_caches, or the drafter's
        window ring will never be allocated and forward_spec will fail.

        Weights tie into the already-built backbone, so no copy and no re-tie.
        """
        from .dspark import DSparkBlock
        cfg = self.cfg
        assert not len(self.mtp), "dspark already attached"
        assert cfg.num_nextn_predict_layers >= 1, "set num_nextn_predict_layers first"
        for stage in range(cfg.num_nextn_predict_layers):
            blk = DSparkBlock(cfg, cfg.n_layers + stage)
            if stage == 0:
                blk.main_proj = torch.nn.Linear(
                    cfg.dim * len(cfg.dspark_target_layer_ids), cfg.dim, bias=False)
                blk.main_norm = RMSNorm(cfg.dim, cfg.rms_norm_eps)
            if stage == cfg.num_nextn_predict_layers - 1:
                blk.norm = RMSNorm(cfg.dim, cfg.rms_norm_eps)
                blk.markov_head = DSparkMarkovHead(cfg)
                blk.confidence_head = DSparkConfidenceHead(cfg)
            self.mtp.append(blk)
        for blk in self.mtp:
            blk.embed, blk.head = self.embed, self.head
        # now that mtp exists, capturing target hidden states becomes meaningful
        self.target_layer_ids = tuple(cfg.dspark_target_layer_ids)
        if hasattr(self, "_cached_max_len"):
            self.init_caches(self._cached_batch, self._cached_max_len,
                             self._cached_device, self._cached_dtype)
        return self

    # ------------------------------------------------------------------ #
    def init_caches(self, batch: int, max_len: int | None = None, device=None,
                    dtype=None):
        """Allocate every layer's caches for one sequence.

        `dtype=None` resolves from the active AMP mode (`amp.cache_dtype`): fp32 when
        mixed precision is off, the autocast dtype when it is on. It is remembered and
        replayed by `attach_dspark`, so the drafter's ring matches the backbone's instead
        of being silently reallocated at a different precision.
        """
        max_len = max_len or self.cfg.max_seq_len
        dtype = cache_dtype(device) if dtype is None else dtype
        self._cached_batch, self._cached_max_len = batch, max_len
        self._cached_device, self._cached_dtype = device, dtype
        for layer in list(self.layers) + list(self.mtp):
            layer.attn.init_caches(batch, max_len, device=device, dtype=dtype)
        self._init_attention_runtime()
        if self.engram_hash is not None:
            self.engram_hash.reset(max_len, batch)

    # ------------------------------------------------------------------ #
    # Decode-state snapshot / restore (speculative decoding only)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def snapshot_decode_state(self, start_pos: int, width: int) -> dict:
        """Clone every buffer a `width`-wide forward at `start_pos` is about to write.

        Speculative decoding needs this to undo a verify block that accepted fewer
        tokens than it verified. By the time the accept loop runs, the REJECTED drafts'
        KV has already been committed to the window ring, to `compress_kv`/`index_k`,
        and to the Engram n-gram id history. Nothing downstream complains -- the model
        just quietly attends to tokens that were never emitted, and the corruption
        compounds for the rest of the sequence.

        Only tails are cloned: `compress_kv` and `index_k` are append-only in absolute
        group index, so a forward at `start_pos` of width `width` can touch at most
        `width // ratio + 1` groups at the end. Cloning the full `[B, max_len, hd]`
        buffer every round would cost tens of MB per speculative step.
        """
        snap: dict = {"window": [], "pool": [], "tail": [],
                      "shared": self.shared.snapshot_graph_free(), "engram": None}
        for layer in list(self.layers) + list(self.mtp):
            a = layer.attn
            if getattr(a, "window_kv", None) is not None:
                snap["window"].append((a, a.window_kv.clone()))
            ps = getattr(a, "pool_state", None)
            if ps is not None:
                snap["pool"].append((a, ps.clone()))
            r = max(a.ratio, 1)
            for name in ("compress_kv", "index_k"):
                buf = getattr(a, name, None)
                if buf is None:
                    continue
                lo = max(0, buf.size(1) - (width // r + 2))
                snap["tail"].append((buf, lo, buf[:, lo:].clone()))
        if self.engram_hash is not None:
            # the id history is a plain [B, max_seq_len] store indexed by position
            c = self.engram_hash.cache
            hi = min(c.size(1), start_pos + width)
            snap["engram"] = (c, min(start_pos, hi), c[:, start_pos:hi].clone())
        return snap

    @torch.no_grad()
    def restore_decode_state(self, snap: dict) -> None:
        """Undo a forward, restoring exactly the buffers it wrote. Inverse of
        `snapshot_decode_state`; see that docstring for why this is load-bearing."""
        for a, buf in snap["window"]:
            a.window_kv.copy_(buf)
        for a, ps in snap["pool"]:
            a.pool_state.copy_from(ps)
        for buf, lo, tail in snap["tail"]:
            buf[:, lo:].copy_(tail)
        if snap["engram"] is not None:
            c, lo, seg = snap["engram"]
            c[:, lo:lo + seg.size(1)].copy_(seg)
        self.shared.restore_graph_free(snap["shared"])

    def reset_shared(self) -> None:
        """A no-op by design.

        The reference NEVER resets the shared runtime between forwards: cache OBJECTS
        persist across decode steps so non-publishing steps (e.g. a ratio-2 owner on an
        odd step) still read a valid cache. Per-step activations (topk_idxs,
        candidates) are always republished by the first producer before any consumer
        reads them, so no reset is required.
        """
        return

    # ------------------------------------------------------------------ #
    def _segment(self, x, start_pos, pre_mix, shared, lo, hi, engram_mask, hash_ids,
                 use_ckpt: bool, capture: list | None = None) -> tuple:
        """Run layers [lo, hi). With use_ckpt, under a checkpointed segment.

        Checkpoint replay hazard. The compressor writes into `compress_kv` / `index_k`
        in place and advances `PoolState`. Replaying a segment re-executes those writes,
        so a replay is only correct if it starts from the state that existed at segment
        ENTRY.

        Two rules, and getting either wrong is silent:

        1. Snapshot only GRAPH-FREE state. `topk_idxs` (int32) and `candidates` (bool)
           must be restored, because replay runs segments in reverse: segment N's replay
           overwrites topk_idxs, so segment N-1's replay would read segment N's value.
           `compress_kv` / `index_k` must NOT be rebound -- they carry the gradient path
           back to the compressor and indexer parameters, and detaching them zeroes
           every one of those gradients while the loss still looks correct.

        2. Restore at the TOP of `run`, never after. Restoring after also undoes state
           that later segments legitimately read.
        """
        entry = shared.snapshot_graph_free() if use_ckpt else None
        entry_pools = ([layer.attn.pool_state.clone()
                        for layer in self.layers[lo:hi]
                        if layer.attn.pool_state is not None] if use_ckpt else None)

        def restore_entry():
            shared.restore_graph_free(entry)
            i = 0
            for layer in self.layers[lo:hi]:
                if layer.attn.pool_state is not None:
                    layer.attn.pool_state.copy_from(entry_pools[i])
                    i += 1

        def run(h, pm):
            if use_ckpt:
                restore_entry()
            for i in range(lo, hi):
                layer = self.layers[i]
                if layer.engram is not None and hash_ids is not None:
                    # [b,S,n_engram_layers,n_hash_cols] -> [b,S,n_hash_cols]
                    h = layer.engram(h, hash_ids[:, :, layer.engram_idx, :], engram_mask)
                if capture is not None and i in self.target_layer_ids:
                    capture.append(h.mean(dim=2))   # attention input of the target layer
                h, pm = layer(h, start_pos, pm, shared)
            return h, pm

        if not use_ckpt:
            return run(x, pre_mix)
        return checkpoint(run, x, pre_mix, use_reentrant=False)

    def _embed_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embedding lookup, cast to the autocast dtype when one is active.

        `nn.Embedding` is not an autocast-eligible op, so the lookup stays fp32. That is
        normally fine -- the first Linear casts it down and autocast has already cleaned
        up after it. It is NOT fine here: the embedding output seeds the residual
        stream, which is `hc` copies wide and is consumed by `hc_pre`/`hc_post` -- pure
        elementwise ops, not eligible ops -- BEFORE any Linear sees it. An fp32 seed
        therefore keeps the entire residual stream fp32 for the whole forward, and
        `hc_post`'s `comb * residual` re-widens an fp32 [b,s,hc,d] tensor on every one of
        the 40 calls. That was most of what the AMP traffic reduction was fighting.
        """
        h = self.embed(input_ids)
        # Traceable autocast read (see amp.current_autocast_dtype): this runs inside
        # compiled regions and the strict read would graph-break per call.
        ad = current_autocast_dtype(h.device.type)
        return h if ad is None else h.to(ad)

    # ------------------------------------------------------------------ #
    def forward(self, input_ids: torch.Tensor, token_types=None, start_pos: int = 0,
                pixel_values=None, image_grid_hws=None, use_checkpoint: bool = False,
                full_logits: bool = False, return_hidden: bool = False):
        """Generation entry point: logits at the LAST position only, [B, V].

        `full_logits=True` routes to `forward_full` instead. That is here rather than
        forcing callers to reach past a DistributedDataParallel wrapper (whose `.forward`
        is the only collective-aware entry point) to call `forward_full` by name -- under
        DDP, bypassing the wrapper means no gradient allreduce, so the training loop
        would silently train on one rank's gradients forever.

        `return_hidden=True` (with `full_logits=True`) returns the normed hidden states
        [B, S, d] instead of logits, so the training loop can fuse the head matmul
        with a chunked cross-entropy and never materialise [B, S, V] (see
        train.chunked_head_ce). It rides the same DDP-safe entry point.

        The two are deliberately NOT unified. Generation materialises [B, S, V] here if
        you route it through `forward_full`, and at S=2048, V=32000 that is 262 MB for a
        tensor whose last row is all generation needs.
        """
        if full_logits:
            return self.forward_full(input_ids, token_types=token_types,
                                     start_pos=start_pos, use_checkpoint=use_checkpoint,
                                     return_hidden=return_hidden)
        cfg = self.cfg
        # image_mask survives router removal: it still masks image tokens out of the
        # n-gram lookup. Deleting it here would silently disable Engram masking.
        image_mask = None if token_types is None else token_types >= 0
        engram_mask = None if image_mask is None else ~image_mask

        hash_ids = None
        if self.engram_hash is not None:
            hash_ids = self.engram_hash(input_ids, start_pos, engram_mask)

        h = self._embed_ids(input_ids)
        if cfg.vision_n_layers and pixel_values is not None:
            raise NotImplementedError(
                "vision is disabled for vertex500m (vision_n_layers=0). The merge path "
                "is specified but untrained here; see README.")

        b, S, _ = h.shape
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        pre_mix = make_identity_pre_mix(b, S, self.hc_mult, device=h.device)

        seg = self.cfg.segment_size if use_checkpoint else self.n_layers
        capture: list = [] if (len(self.mtp) and self.target_layer_ids) else None
        for lo in range(0, self.n_layers, seg):
            hi = min(lo + seg, self.n_layers)
            h, pre_mix = self._segment(h, start_pos, pre_mix, self.shared, lo, hi,
                                        engram_mask, hash_ids, use_checkpoint, capture)

        h = hc_pre(h, pre_mix)                       # final collapse with the last ffn pre
        logits = self.head(self.norm(h))
        main_hidden = (torch.cat(capture, -1) if capture else None)
        return logits, main_hidden

    def forward_full(self, input_ids: torch.Tensor, token_types=None, start_pos: int = 0,
                     use_checkpoint: bool = False, return_hidden: bool = False):
        """TRAINING path: return logits at EVERY position, [B, S, V].

        `forward` returns only the last position, which is all generation needs but is
        useless for a loss. Generation never wants [B,S,V] (it would be the single
        largest tensor in memory), so these are deliberately separate entry points.

        With `return_hidden=True`, returns the normed hidden states [B, S, d] instead
        of logits -- the training loop's fused head+CE (train.chunked_head_ce) calls
        the head itself one position-chunk at a time, so [B,S,V] is never
        materialised. Bit-identical loss to the logits path (same per-row matmuls).
        """
        cfg = self.cfg
        image_mask = None if token_types is None else token_types >= 0
        engram_mask = None if image_mask is None else ~image_mask

        hash_ids = None
        if self.engram_hash is not None:
            hash_ids = self.engram_hash(input_ids, start_pos, engram_mask)

        h = self._embed_ids(input_ids)
        b, S, _ = h.shape
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        pre_mix = make_identity_pre_mix(b, S, self.hc_mult, device=h.device)

        seg = self.cfg.segment_size if use_checkpoint else self.n_layers
        capture: list = [] if (len(self.mtp) and self.target_layer_ids) else None
        for lo in range(0, self.n_layers, seg):
            hi = min(lo + seg, self.n_layers)
            h, pre_mix = self._segment(h, start_pos, pre_mix, self.shared, lo, hi,
                                        engram_mask, hash_ids, use_checkpoint, capture)

        h = hc_pre(h, pre_mix)
        main_hidden = (torch.cat(capture, -1) if capture else None)
        hn = self.norm(h)
        if return_hidden:
            return hn, main_hidden
        # No `.float()` on the hidden state; see VertexHead.forward. Under autocast
        # F.linear returns the compute dtype either way, and chunked_ce upcasts each
        # chunk to fp32 for the loss (see train.py), so the full [B,S,V] tensor stays
        # in the activation dtype here.
        logits = F.linear(hn, self.embed.weight.float())
        return logits, main_hidden

    def forward_spec(self, output_ids, main_hidden, start_pos):
        """DSpark pass: draft a block conditioned on the backbone's target layers."""
        if not len(self.mtp):
            raise RuntimeError("DSpark is not attached; call attach_dspark() first")
        blk, last = self.mtp[0], self.mtp[-1]
        main_x = blk.main_norm(blk.main_proj(main_hidden))
        B = output_ids.size(0)
        ids = torch.full((B, self.cfg.dspark_block_size),
                         self.cfg.dspark_noise_token_id,
                         dtype=torch.long, device=output_ids.device)
        ids[:, 0] = output_ids.reshape(B)
        x = self.embed(ids).unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        pre_mix = make_identity_pre_mix(x.size(0), x.size(1), self.hc_mult, device=x.device)
        for m in self.mtp:
            x, pre_mix = m(x, start_pos, pre_mix, self.shared, main_x)
        if start_pos == 0:
            return None                            # prefill only seeded caches
        return last.forward_head(x, pre_mix, output_ids)


# --------------------------------------------------------------------- #
# DSpark pieces. Defined here to avoid a circular import with model.py.
# --------------------------------------------------------------------- #
def DSparkMarkovHead(cfg):
    import torch.nn as nn

    class _MarkovHead(nn.Module):
        """Additive logit bias from the previously emitted token id."""

        def __init__(self, cfg):
            super().__init__()
            self.embed = nn.Embedding(cfg.vocab_size, cfg.dspark_markov_rank)
            self.head = nn.Linear(cfg.dspark_markov_rank, cfg.vocab_size, bias=False)

        def forward(self, ids):
            e = self.embed(ids)
            return self.head(e.float()), e

    return _MarkovHead(cfg)


def DSparkConfidenceHead(cfg):
    import torch.nn as nn

    class _ConfidenceHead(nn.Module):
        """Per-position acceptance probability."""

        def __init__(self, cfg):
            super().__init__()
            self.proj = nn.Linear(cfg.dim + cfg.dspark_markov_rank, 1, bias=False)

        def forward(self, hidden, markov_embeds):
            return self.proj(torch.cat([hidden.float(), markov_embeds.float()], -1)).squeeze(-1)

    return _ConfidenceHead(cfg)


@torch.no_grad()
def sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(-1)
    probs = torch.softmax(logits.float() / max(temperature, 1e-5), dim=-1)
    return torch.multinomial(probs, 1)


@torch.no_grad()
def generate(model: Transformer, prompt_ids: torch.Tensor, max_new: int = 64,
             temperature: float = 1.0, use_dspark: bool = False,
             use_checkpoint: bool = False):
    """Prefill in one shot, then 1-token decode. start_pos is threaded explicitly."""
    model.eval()
    S = prompt_ids.size(1)
    ids = prompt_ids
    logits, main_hidden = model(ids, start_pos=0, use_checkpoint=use_checkpoint)
    if use_dspark:
        model.forward_spec(logits.argmax(-1, keepdim=True), main_hidden, 0)
    for t in range(max_new):
        logits, main_hidden = model(ids[:, -1:], start_pos=S + t,
                                     use_checkpoint=use_checkpoint)
        if use_dspark:
            model.forward_spec(logits.argmax(-1, keepdim=True), main_hidden, S + t)
        nxt = sample(logits, temperature)
        ids = torch.cat([ids, nxt], dim=1)
    return ids
