"""Training losses: chunked cross-entropy over positions.

Moved verbatim from train.py (phase-1 restructure); behaviour is unchanged.
`eval.py` uses `chunked_ce`; the training loop uses `chunked_head_ce`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .. import amp as amp_mod


def chunked_ce(logits: torch.Tensor, targets: torch.Tensor, chunk: int = 1024,
               reduction: str = "mean") -> torch.Tensor:
    """Cross-entropy over positions without materialising a [B*S, V] fp32 log_softmax.

    `forward_full` returns [B, S, V] and the naive `logits.float().view(-1, V)` makes
    TWO more copies of it (the fp32 cast, then log_softmax inside cross_entropy). At
    S=8192, V=32000 that is ~3.1 GB of logits alone, on a 12 GB card. Chunking over
    positions bounds the transient to `chunk` rows; the backward is unchanged because
    every chunk still contributes to the same graph.

    Loss is the mean over all positions, identical to the unchunked call.
    """
    b, s, v = logits.shape
    flat = logits.reshape(b * s, v)
    tgt = targets.reshape(-1)
    if reduction == "sum":
        total = flat.new_zeros(())
        for i in range(0, flat.size(0), chunk):
            total = total + F.cross_entropy(flat[i:i + chunk].float(), tgt[i:i + chunk],
                                            reduction="sum")
        return total
    # 'mean': sum of per-chunk MEANS weighted by chunk size == one global mean, and
    # avoids dividing by a count that differs on the final partial chunk.
    total = flat.new_zeros(())
    n = 0
    for i in range(0, flat.size(0), chunk):
        end = min(i + chunk, flat.size(0))
        total = total + F.cross_entropy(flat[i:end].float(), tgt[i:end],
                                        reduction="sum")
        n += end - i
    return total / max(n, 1)


def _head_ce_chunk(h_c: torch.Tensor, t_c: torch.Tensor,
                   head_weight: torch.Tensor, amp_mode, device_type):
    """One position-chunk of head matmul + cross-entropy (see chunked_head_ce).

    The autocast region lives INSIDE here (not at the caller) so checkpoint replay
    re-executes the identical function: replay runs during backward, outside any
    ambient autocast, and without this the replay would evaluate the head in fp32
    while forward ran it in fp16 -- the sparse-attention forward/backward agreement
    lesson, applied to the head.
    """
    with amp_mod.amp_context(amp_mode, device_type):
        logits_c = F.linear(h_c, head_weight.float())
        return F.cross_entropy(logits_c.float(), t_c, reduction="sum")


def chunked_head_ce(hidden: torch.Tensor, targets: torch.Tensor,
                    head_weight: torch.Tensor, chunk: int = 1024,
                    amp_mode: str | None = None,
                    device_type: str = "cuda") -> torch.Tensor:
    """Cross-entropy over positions with the head matmul fused in per chunk.

    `forward_full` returns [B, S, V] and `chunked_ce` chunks only the loss, so the
    full logit tensor is still materialised -- at B=4, S=8192, V=32000 that is
    ~2.1 GB fp16 for a tensor the loss consumes chunk by chunk. This takes the
    normed hidden states [B, S, d] instead (see `forward_full(return_hidden=True)`)
    and runs head + cross-entropy one position-chunk at a time. The only [B,S,*]
    tensor retained is hidden itself (24x smaller than logits at V=32000, d=1344).

    Each chunk runs under gradient checkpointing (`use_reentrant=False`): forward
    keeps only the chunk's inputs and rebuilds the chunk during backward, so the
    transient stays bounded by one chunk instead of all of them. The backbone
    still replays exactly once per step (a single backward traverses it), and the
    per-chunk head recompute costs one extra head matmul (~1.6% of step FLOPs).
    DDP and the GradScaler see one loss and one backward, exactly as before -- no
    `no_sync` juggling, no per-chunk scaling.

    Bit-identical loss to `chunked_ce(forward_full(...))`: a linear layer over row
    slices computes the same per-row dots in the same order, and the per-chunk
    fp32 upcast before cross_entropy matches chunked_ce exactly.
    """
    from torch.utils.checkpoint import checkpoint as _ckpt
    b, s, d = hidden.shape
    flat_h = hidden.reshape(b * s, d)
    flat_t = targets.reshape(-1)
    n = flat_t.numel()
    total = flat_h.new_zeros(())
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        total = total + _ckpt(_head_ce_chunk, flat_h[lo:hi], flat_t[lo:hi],
                              head_weight, amp_mode, device_type,
                              use_reentrant=False)
    return total / max(n, 1)
