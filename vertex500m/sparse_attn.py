"""Sparse attention over a shared KV latent (spec 7.4, 21.5).

Memory note. The obvious implementation materializes the gathered KV at
`[b, m, k, hd]` and the attention probabilities at `[b, m, k, h]`. For this config
(k = 640 positions, hd = 128, h = 21) that is

    gathered kv   640 * 128 * 2 B = 160.0 KB/token/layer   (59% of the activation bill)
    attn probs    640 *  21 * 4 B =  52.5 KB/token/layer   (19%)

retained by autograd for the whole backward pass. The gathered KV is the dominant term
in training memory and it is pure overhead: it is a private temporary copy of the cache
that only exists to feed two matmuls.

`SparseAttention` below avoids both by chunking the query dimension, so the gather is
recomputed in backward rather than stored. It retains only `q`, `kv` and the top-k
indices -- everything the recompute needs -- which is the same trick FlashAttention
uses to avoid materializing the probability matrix.

`attn_sink` enters the DENOMINATOR ONLY. It has no value vector and no key (spec
invariant 20).

Semantics preserved from the reference:
  - `topk_idxs` is contiguous int32, -1 masks an entry, and the score floor is the
    dtype's most negative FINITE value (not -inf) so an all-masked row returns exactly 0
     rather than NaN. See the floor discussion in `_forward_chunks`.
  - `topk_idxs` must live on the same device as `q`; the index helpers below take an
    explicit `device` for that reason.
  - `kv` is a SINGLE latent per position, reused as both key and value. There is no
    value projection anywhere in this architecture.

AMP notes (the four things that were wrong under autocast):
   1. The score floor is dtype-dependent. The literal -1e30 that the reference kernel
      hard-codes OVERFLOWS fp16 -- `Tensor.clamp_min(-1e30)` on a half tensor raises
      `value cannot be converted to type at::Half without overflow`, so `--amp fp16` died
      in the first attention layer. The floor is `torch.finfo(score.dtype).min`, which is
      structurally identical in every dtype: it lifts an all-`-inf` masked row's max to a
      finite value so `exp(-inf - floor) == 0`. The specific number never matters; only
      that it is finite does.
   2. Autocast is ENABLED inside `forward` but DISABLED inside `backward` of a bare
      `torch.autograd.Function`. Forward would therefore evaluate the scores in fp16 while
      the recompute differentiated an fp32 evaluation -- a derivative of a function that
      was never run. The kernel below is a `torch.library.custom_op`, not a bare
      Function: forward reads the live autocast state through traceable calls only
      (`torch.is_autocast_enabled` / `torch.get_autocast_dtype`, never the untraceable
      `torch._C._is_autocast_available` builtin that broke compilation into one graph
      per layer), and the resolved compute dtype travels to backward in the autograd
      context -- backward differentiates exactly the function forward ran, compiled or
      eager. tests.py::test_attn_fwd_bwd_dtype_agreement pins the agreement.
   3. `acc` must follow autocast. Promoting fp16 activations to fp32 "for a stable
      softmax" is backwards under autocast: the einsums are autocast-eligible, so the
      round trip is pure waste. See `amp.compute_dtype`.
   4. Gradients accumulate in fp32 regardless of the cache dtype. `gkv` sums one chunk's
      contribution per iteration; accumulating those in fp16 loses precision silently.
"""

from __future__ import annotations

import warnings

import torch

from .amp import compute_dtype, current_autocast_dtype


def _gather_kv(kv: torch.Tensor, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """kv [b,N,d], idx [b,m,k] -> (gathered [b,m,k,d], valid [b,m,k]).

    Invalid (-1) entries are gathered from slot 0 and then zeroed; their scores are
    masked to the floor regardless, so the zeroing is belt-and-braces.
    """
    b, n, d = kv.shape
    safe = idx.clamp(0, n - 1).long()
    g = kv.unsqueeze(1).expand(b, idx.size(1), n, d)
    gathered = torch.gather(g, 2, safe.unsqueeze(-1).expand(-1, -1, -1, d))
    valid = idx >= 0
    return gathered * valid.unsqueeze(-1), valid


def sparse_attn_ref(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor,
                    topk_idxs: torch.Tensor, softmax_scale: float) -> torch.Tensor:
    """Reference implementation, materialized. Correct but memory-hungry.

    q          [b,m,h,d]  already RoPE-rotated
    kv         [b,N,d]    single latent (K == V), already rotated + quantized
    attn_sink  [h]        fp32, denominator only
    topk_idxs  [b,m,k]    int32, -1 = masked

    Kept as the ground truth that `sparse_attn` is tested against.
    """
    b, m, h, d = q.shape
    # Inside an autocast region this is the autocast dtype; outside it, promote to at
    # least fp32 (fp64 when the caller is, so this stays a valid ground truth at any
    # precision). Under autocast the einsum below would downcast anyway, so promoting
    # first would only buy a wasted round trip.
    acc = compute_dtype(q)
    idx = topk_idxs.long()
    valid = idx >= 0
    safe = idx.clamp_min(0).clamp_max(kv.size(1) - 1)
    kvg = torch.gather(kv.unsqueeze(1).expand(b, m, kv.size(1), d), 2,
                       safe.unsqueeze(-1).expand(b, m, safe.size(-1), d))
    kvg = kvg * valid.unsqueeze(-1)
    sc = torch.einsum("bmhd,bmkd->bmkh", q.to(acc), kvg.to(acc)) * softmax_scale
    sc = sc.masked_fill(~valid.unsqueeze(-1), float("-inf"))
    # Finite floor, see the module docstring: -1e30 overflows fp16.
    mx = sc.amax(-2).clamp_min(torch.finfo(sc.dtype).min)   # max over keys
    p = torch.exp(sc - mx.unsqueeze(-2))
    den = p.sum(-2) + torch.exp(attn_sink.to(acc).view(1, 1, h) - mx)
    acc_out = torch.einsum("bmkh,bmkd->bmhd", p, kvg.to(acc))
    return (acc_out / den.unsqueeze(-1)).to(q.dtype)


# --------------------------------------------------------------------------- #
# Chunked kernel: shared math core
#
# `_forward_chunks` / `_backward_chunks` are plain tensor code with the compute
# dtype passed in explicitly -- no live autocast reads inside, so eager and
# compiled execution agree by construction. The custom op below resolves `acc`
# once per call and hands it to both.
# --------------------------------------------------------------------------- #
def _forward_chunks(q, kv, attn_sink, topk_idxs, softmax_scale, chunk, acc):
    """Chunked forward; `acc` is the compute dtype (see `compute_dtype`)."""
    b, m, h, d = q.shape
    out = torch.empty(b, m, h, d, dtype=q.dtype, device=q.device)
    for s0 in range(0, m, chunk):
        s1 = min(s0 + chunk, m)
        qc = q[:, s0:s1]
        kvg, valid = _gather_kv(kv, topk_idxs[:, s0:s1])
        sc = torch.einsum("bchd,bckd->bchk",
                          qc.to(acc), kvg.to(acc)) * softmax_scale
        sc = sc.masked_fill(~valid.unsqueeze(-2), float("-inf"))     # [b,c,h,k]
        # The floor must be finite so an all-masked row yields exp(-inf - floor) == 0
        # instead of exp(-inf - -inf) == NaN. Its VALUE is irrelevant; its finiteness
        # is the whole point, which is why this reads finfo rather than -1e30.
        mx = sc.amax(-1, keepdim=True).clamp_min(torch.finfo(sc.dtype).min)
        p = torch.exp(sc - mx)
        sink = attn_sink.to(acc).view(1, 1, h, 1)
        den = p.sum(-1, keepdim=True) + torch.exp(sink - mx)         # [b,c,h,1]
        out[:, s0:s1] = (torch.einsum("bchk,bckd->bchd", p, kvg.to(acc))
                         / den).to(q.dtype)
    # No logsumexp is retained. It is fully determined by q/kv/idx -- which are all
    # saved -- so keeping a [b,m,h] fp32 tensor alive across the whole backward
    # (and paying for its compute in every forward chunk) buys nothing.
    return out


# Dynamo-safe autocast read (see amp.current_autocast_dtype for why the strict
# read cannot be used inside compiled code).
def _live_compute_dtype(q: torch.Tensor) -> torch.dtype:
    """The kernel's compute dtype, read from live autocast state. See `compute_dtype`
    for the policy; this is its traceable twin (no `is_autocast_available`)."""
    ad = current_autocast_dtype(q.device.type)
    if ad is not None:
        return ad
    return torch.promote_types(q.dtype, torch.float32)


# `torch.library.custom_op` (torch>=2.4) is the supported way to carry a custom
# backward through `torch.compile`: unlike a bare `autograd.Function`, whose `ctx`
# AOTAutograd baked into the first compiled invocation (so the SECOND backward
# ran against freed saved tensors), a registered op recompiles per invocation.
_HAVE_CUSTOM_OP = hasattr(torch.library, "custom_op")


def _backward_chunks(grad_out, q, kv, attn_sink, topk_idxs, scale, chunk, acc,
                     need_q, need_kv, need_sink):
    """Closed-form chunked backward (FlashAttention-style). No autograd rebuild.

    The old code rebuilt each chunk's forward as a small autograd graph and called
    `torch.autograd.grad` on it. Correct, but it pays graph construction plus a
    full extra forward+backward's worth of dispatch per chunk -- and the
    `requires_grad_()` leaf calls are untraceable by dynamo, so a compiled model's
    backward ran eager (see README: `torch.compile`).

    The output is a smooth function of the scores, so the whole backward is a
    handful of einsums per chunk. Per chunk (shapes [b,c,h,k], head dim d):

        p   = exp(sc - max)              (recomputed, exactly as in forward)
        den = sum_k p + exp(sink - max)  (the sink sits in the denominator only)
        o   = (p @ kvg) / den
        A   = go @ kvg                   per (h, j): <go_h, kvg_j>
        G   = <go, o>                    per h

        dL/dsc[h,j] = (p[h,j] / den[h]) * (A[h,j] - G[h])

    The row-max terms cancel exactly: differentiating through the max contributes
    argmax-weighted terms that subtract out, leaving the standard softmax-backward
    form with `den` in place of the plain rowsum (because of the sink term).

        dL/dq[h]    = scale * sum_j w[h,j] * kvg[j]
        dL/dkvg[j]  = sum_h (p[h,j] * go[h] / den[h] + scale * w[h,j] * q[h])
                      -- summed over heads, because kv is a SINGLE latent shared
                      by all of them (spec invariant: no value projection)
        dL/dsink[h] = -(G[h] / den[h]) * exp(sink[h] - max[h])

    Masked entries contribute exactly 0: their p is 0 and their zeroed kvg makes
    A 0, so nothing flows back to slot 0 through the scatter. All-masked rows give
    p = 0 and den = +inf, hence exactly 0 everywhere; the sink gradient is guarded
    to 0 there because 0 * inf would otherwise be NaN (an all-masked row is
    exactly the forward's `nan_to_num`-free 0, so the sink must see no signal).

    Cost: one extra forward's worth of matmuls per chunk -- the same recompute
    FlashAttention already pays -- with no graph to build and nothing dynamo
    cannot trace. Memory stays bounded because only one chunk's tensors are alive
    at a time.     Pinned against the reference by
    tests.py::test_sparse_attn_matches_reference.

    Returns (gq|None, gkv_accumulator|None, gsink|None). The kv accumulator is in
    the widened dtype; the caller casts it to kv.dtype.
    """
    b, m, h, d = q.shape
    if not (need_q or need_kv or need_sink):
        return None, None, None

    gq = torch.zeros_like(q) if need_q else None
    # Accumulate at least fp32 regardless of the cache dtype: this sums one
    # contribution per query chunk, and doing that in fp16 loses precision with no
    # symptom. Cast once on return. `promote_types` keeps the fp64 ground-truth path
    # (tests.py::test_sparse_attn_matches_reference) from being silently truncated to
    # fp32 by an accumulator that is supposed to be a strictly wider home for it.
    gkv = (torch.zeros(kv.shape, dtype=torch.promote_types(kv.dtype, torch.float32),
                       device=kv.device)
           if need_kv else None)
    # attn_sink is an fp32 Parameter; accumulating its gradient in the activation dtype
    # would round on every chunk's in-place +=.
    gsink = (torch.zeros(h, dtype=attn_sink.dtype, device=q.device)
             if need_sink else None)

    need_qkv = need_q or need_kv
    for s0 in range(0, m, chunk):
        s1 = min(s0 + chunk, m)
        idx = topk_idxs[:, s0:s1]
        # Promote exactly as `forward` does (see the module docstring on acc).
        go = grad_out[:, s0:s1].to(acc)
        qc = q[:, s0:s1].to(acc)

        kvg, valid = _gather_kv(kv, idx)
        kvg_a = kvg.to(acc)
        sc = torch.einsum("bchd,bckd->bchk", qc, kvg_a) * scale
        sc = sc.masked_fill(~valid.unsqueeze(-2), float("-inf"))
        mx = sc.amax(-1, keepdim=True).clamp_min(torch.finfo(sc.dtype).min)
        p = torch.exp(sc - mx)
        e = torch.exp(attn_sink.to(acc).view(1, 1, h, 1) - mx)
        den = p.sum(-1, keepdim=True) + e                       # [b,c,h,1]
        o = torch.einsum("bchk,bckd->bchd", p, kvg_a) / den
        # G[h] = <go_h, o_h>; the sink gradient's only path.
        G = (go * o).sum(-1, keepdim=True)

        if need_qkv:
            A = torch.einsum("bchd,bckd->bchk", go, kvg_a)      # <go_h, kvg_j>
            pov = p / den
            w = pov * (A - G)                                   # dL/dsc
        if need_q:
            gq[:, s0:s1] = (torch.einsum("bchk,bckd->bchd", w, kvg_a)
                            * scale).to(q.dtype)
        if need_kv:
            gkvg = (torch.einsum("bchk,bchd->bckd", pov, go)
                    + torch.einsum("bchk,bchd->bckd", w, qc) * scale)
            # scatter_add_ along the position dim: flatten the chunk's (c, k)
            # into one row per gathered entry. Masked entries scattered 0 to
            # slot 0 (their gkvg is exactly 0), matching gather's backward.
            safe = idx.clamp(0, kv.size(1) - 1).long()
            gkv.scatter_add_(1, safe.reshape(b, -1).unsqueeze(-1).expand(-1, -1, d),
                             gkvg.reshape(b, -1, d).to(gkv.dtype))
        if need_sink:
            ds = -(G / den) * e
            # An all-masked row has den == +inf; without the guard this is
            # 0 * inf == NaN for a row the forward defines as exactly 0.
            ds = torch.where(torch.isfinite(den), ds, 0.0)
            gsink += ds.sum(dim=(0, 1)).reshape(h).to(attn_sink.dtype)
    return gq, gkv, gsink


if _HAVE_CUSTOM_OP:
    @torch.library.custom_op("v500m::sparse_attn", mutates_args=())
    def _sparse_attn_op(q: torch.Tensor, kv: torch.Tensor,
                        attn_sink: torch.Tensor, topk_idxs: torch.Tensor,
                        softmax_scale: float, chunk: int) -> torch.Tensor:
        """Chunked, recomputing sparse attention as a compiled-safe custom op.

        Scores are held as [b, c, h, k] (heads before keys) so the mask and the
        logsumexp broadcast along the last axis. Forward saves only q, kv and the
        top-k indices; the gathered KV and the probabilities are rebuilt in
        backward, one query chunk at a time.
        """
        return _forward_chunks(q, kv, attn_sink, topk_idxs, softmax_scale, chunk,
                               _live_compute_dtype(q))

    @torch.library.register_fake("v500m::sparse_attn")
    def _sparse_attn_fake(q, kv, attn_sink, topk_idxs, softmax_scale, chunk):
        return torch.empty_like(q)

    def _sparse_attn_setup(ctx, inputs, output):
        q, kv, attn_sink, topk_idxs, softmax_scale, chunk = inputs
        ctx.q, ctx.kv, ctx.sink, ctx.idx = q, kv, attn_sink, topk_idxs
        ctx.scale, ctx.chunk = softmax_scale, chunk
        # Resolved NOW and saved: backward runs with autocast off, so re-reading
        # live state there would differentiate a different function than forward
        # ran. A plain Python value -- nothing dynamo cannot trace.
        ctx.acc = _live_compute_dtype(q)

    def _sparse_attn_autograd(ctx, grad_out):
        need_q, need_kv, need_sink = ctx.needs_input_grad[:3]
        gq, gkv, gsink = _backward_chunks(
            grad_out, ctx.q, ctx.kv, ctx.sink, ctx.idx,
            ctx.scale, ctx.chunk, ctx.acc, need_q, need_kv, need_sink)
        return (gq, gkv.to(ctx.kv.dtype) if gkv is not None else None, gsink,
                None, None, None)

    torch.library.register_autograd("v500m::sparse_attn", _sparse_attn_autograd,
                                    setup_context=_sparse_attn_setup)


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale, chunk: int = 256):
    """Memory-bounded sparse attention. See the module docstring for the why.

    `chunk` bounds the transient gather at chunk*k*hd elements instead of m*k*hd.
    """
    idx = topk_idxs
    assert idx.dtype == torch.int32, (
        f"sparse_attn requires contiguous int32 topk_idxs, got {idx.dtype} "
        f"(spec invariant 19)"
    )
    # A CPU index against a CUDA tensor raises inside `torch.gather` or the caller's
    # `torch.cat`, with a message about device mismatch that does not point back here.
    # Every producer builds its indices through the helpers above, which take `device`
    # explicitly; this catches the next one that forgets.
    assert idx.device == q.device, (
        f"topk_idxs is on {idx.device} but q is on {q.device}; index producers must "
        f"pass device= (see get_window_topk_idxs / get_chunk_window_topk_idxs / "
        f"get_dspark_topk_idxs)"
    )
    idx = idx.contiguous()
    if not q.requires_grad and not kv.requires_grad:
        # Inference: no graph to keep alive, the reference path is both faster and
        # uses strictly less peak memory because nothing is stashed for backward.
        return sparse_attn_ref(q, kv, attn_sink, idx, softmax_scale)
    if _HAVE_CUSTOM_OP:
        return _sparse_attn_op(q, kv, attn_sink, idx, softmax_scale, chunk)
    # torch<2.4 has no custom-op autograd: fall back to the reference, which is
    # correct but retains the full [b,m,k,d] gather (see the module docstring).
    # A compat path, not a performance path.
    warnings.warn("sparse_attn: torch.library.custom_op needs torch>=2.4; falling "
                  "back to the memory-hungry reference implementation.",
                  RuntimeWarning, stacklevel=2)
    return sparse_attn_ref(q, kv, attn_sink, idx, softmax_scale)


# --------------------------------------------------------------------------- #
# Index helpers (spec 21.5)
# --------------------------------------------------------------------------- #
def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int,
                         device=None):
    """Sliding-window ring slots each query attends to; -1 = empty.

    `device` is threaded explicitly rather than inferred: every `arange` here would
    otherwise land on CPU, and a CPU index tensor fails later -- at the `torch.cat` in
    `Attention.forward`, or inside `torch.gather` -- with a message that does not name
    the real cause. There is no `torch.set_default_device` in this package.
    """
    if start_pos == 0:
        end = torch.arange(seqlen, device=device).unsqueeze(1)
        idxs = ((end - window_size + 1).clamp(0)
                + torch.arange(min(seqlen, window_size), device=device))
        idxs = torch.where(idxs > end, -1, idxs)
    else:
        oldest = start_pos % window_size + 1
        idxs = torch.cat([torch.arange(oldest, window_size, device=device),
                          torch.arange(oldest, device=device)])
        idxs = torch.where(idxs > start_pos, -1, idxs)
    return idxs.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


def get_chunk_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int,
                               device=None):
    """Chunk-general window indices (start_pos > 0, seqlen > 1).

    The KV handed to attention is [ring (win slots) | chunk (seqlen rows)]. Query j at
    absolute position p attends cached q in [max(0, p-win+1), start_pos) at ring slot
    q % win, and chunk positions q in [start_pos, p] at win + (q - start_pos).

    Fully vectorised. This used to be a Python loop over `seqlen` emitting six kernels
    each, which is fine to write and ruinous to run: at seqlen=2048 that is ~12k kernel
    launches PER CALL, times twenty layers, and it is a guaranteed dynamo graph break.
    The loop's exact semantics -- a per-query window of min(win, p+1) real slots followed
    by -1 padding -- is reproduced here as one [seqlen, window_size] computation, and
    tests.py pins it against the loop form.
    """
    p = torch.arange(start_pos, start_pos + seqlen, device=device)
    # a query at absolute p covers [max(0, p-win+1), p], i.e. min(win, p+1) positions
    length = torch.clamp(p + 1, max=window_size)
    t = torch.arange(window_size, device=device)
    # q[j, i] = p_j - length_j + 1 + i, the i-th position of query j's window
    q = p[:, None] - (length[:, None] - 1 - t[None, :])
    slots = torch.where(q < start_pos, q % window_size, window_size + (q - start_pos))
    # columns past this query's window are empty, and -1 is the mask sentinel
    slots = torch.where(t[None, :] < length[:, None], slots, -1)
    return slots.to(torch.int32).unsqueeze(0).expand(bsz, -1, -1).contiguous()


def get_dspark_topk_idxs(window_size: int, bsz: int, block_size: int, start_pos: int,
                          device=None):
    """Draft queries see the whole backbone window plus every draft-block token."""
    matrix = torch.cat([torch.arange(min(window_size, start_pos + 1), device=device),
                        window_size + torch.arange(block_size, device=device)])
    return matrix.int().view(1, 1, -1).expand(bsz, block_size, -1).contiguous()


def select_candidate_blocks(logits: torch.Tensor, compress_lens: torch.Tensor,
                            topk_blocks: int, block_size: int) -> torch.Tensor:
    """Level 1 of the hierarchical indexer -> bool mask shaped like logits."""
    width = logits.size(-1)
    padded = torch.nn.functional.pad(logits, (0, -width % block_size), value=float("-inf"))
    scores = padded.unflatten(-1, (-1, block_size)).amax(-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size          # pin the newest block in
    scores = scores.masked_fill(
        torch.arange(num_blocks, device=logits.device) == last, float("inf"))
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > -float("inf"))
    return keep.repeat_interleave(block_size, -1)[..., :width]
