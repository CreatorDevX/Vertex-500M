"""Weight initialization.

The spec's reference implementation uses `torch.empty` for `hc_attn_fn` / `hc_ffn_fn`
and leaves `Gate.weight` uninitialized -- the reference is inference-only and expects a
loaded checkpoint. Training from scratch requires an explicit init, and getting it
wrong is silent: `torch.empty` on a fresh allocation is usually zeros on Linux, which
means every hyper-connection coefficient starts at a degenerate constant rather than
garbage, and the model trains, just badly.

Two inits here are load-bearing rather than cosmetic, and both failed silently before
being caught by `tests.py::test_backward_reaches_everything`:

  * The Engram n-gram table must NOT be zero. It enters the network only through
    wkv(embed(hash_ids)); a zero table makes value == 0 and d(out)/d(table) identically
    zero, so the table is a gradient dead-end.
  * The Engram row scale must start at 1.0, not ~0. It is a per-32-channel block
    exponent, so 1.0 is neutral. Initializing it N(0, 0.02) makes the product v*s ~1e-4
    and the table nearly dead by the same route.

Rules:
  * nn.Linear weights   ~ N(0, initializer_range**2)
  * RMSNorm weights     = 1
  * attn_sink           = 0 (fp32). It enters the denominator only, so 0 means "no
                          extra probability mass".
* hc_*                zeros + the mHC recipe in hc.init_hc_params. This makes step 0 a
                        residual connection with a ~0.538-scaled branch, not a bit-exact
                        one: `post = 2*sigmoid(base)` with base=-1 gives ~0.538. Close
                        enough to train; do not write a test asserting unit gain.
  * Embedding           ~ N(0, initializer_range**2). The table is TIED, so this is both
                          the input embedding and the LM head.
"""

from __future__ import annotations

import math

import torch

from .hc import init_hc_params
from .model import Transformer


def _blocks(model: Transformer) -> list:
    """Backbone + draft blocks. `mtp` is empty until `attach_dspark()`."""
    return list(model.layers) + list(model.mtp)


def init_weights(model: Transformer, initializer_range: float = 0.02,
                 seed: int = 0) -> Transformer:
    g = torch.Generator().manual_seed(seed)
    std = initializer_range
    hcs = getattr(model.cfg, "hc_mult", 4)

    def normal_(t):
        with torch.no_grad():
            t.normal_(0.0, std, generator=g)

    for name, p in model.named_parameters():
        if "hc_" in name:
            continue                                  # handled below, explicitly
        if name.endswith("attn_sink"):
            with torch.no_grad():
                p.zero_()
        elif ".embed.scale" in name:
            with torch.no_grad():                    # Engram row scale: neutral = 1
                p.fill_(1.0)
        elif name.endswith("norm.weight") or "q_weight" in name or "k_weight" in name:
            with torch.no_grad():
                p.fill_(1.0)
        elif p.dim() >= 2:
            normal_(p)                               # every 2-D weight is a Linear kernel

    # hyper-connections: hc_fn zeroed, scales 0.01, base set so step 0 ~ residual
    variant = getattr(model, "hc_variant", "mhc")
    for blk in _blocks(model):
        init_hc_params(blk.hc_attn_fn, blk.hc_attn_base, blk.hc_attn_scale, hcs, variant)
        init_hc_params(blk.hc_ffn_fn, blk.hc_ffn_base, blk.hc_ffn_scale, hcs, variant)

    # residual-scaled depth: scale the output projections by 1/sqrt(2*L) so the
    # residual stream variance does not grow with depth (GPT-2 / PaLM convention).
    # L is the TOTAL block count, so a draft stage counts too.
    n = max(len(_blocks(model)), 1)
    scale = 1.0 / math.sqrt(2 * n)
    with torch.no_grad():
        for blk in _blocks(model):
            blk.attn.wo_b.weight.mul_(scale)
            blk.ffn.expert.w2.weight.mul_(scale)

    return model


def init_new_dspark_params(model: Transformer, seed: int = 0,
                            names: set[str] | None = None) -> int:
    """Initialize ONLY the parameters `attach_dspark()` created, and return their count.

    Why this exists. Stage B builds the drafter AFTER `init_from_scratch()` has already
    run, so `main_proj`, `markov_head.embed`, `markov_head.head` and
    `confidence_head.proj` never saw `init_weights`. They keep PyTorch's module defaults,
    and `nn.Embedding` defaults to N(0,1) -- which `dspark.py`'s `forward_head` adds
    DIRECTLY to the logits. Stage B would then begin with a unit-scale random logit bias
    on top of a backbone whose logits are O(1-10). The drafter's parameter count is
    modest, but the bias is applied at every draft slot, so it swamps the initial loss.

    Why not just call `init_weights(model)` again. It would also re-apply the
    `1/sqrt(2*L)` output-projection downscale at init_weights.py, which is
    MULTIPLICATIVE: a second pass squares it, silently halving the backbone's
    `wo_b`/`w2` on a checkpoint resume. So this touches the new parameters only, and
    scales the draft blocks' projections exactly once.

    Calling this on a model with no `mtp` is a no-op.

    `names` restricts the init to a subset of mtp.* parameter names (a stage-B
    resume passes the names the checkpoint did NOT supply, so the trained
    drafter it did supply is left alone). `None` means all of mtp.*, exactly as
    before. An empty set is a no-op that returns 0.

    The block-level inits (hyper-connection params, output-projection
    downscale) run only for a FRESH drafter -- every mtp.* parameter being
    inited. The downscale is multiplicative, so re-applying it to a resumed
    drafter would silently shrink its projections a second time.
    """
    all_mtp = [n for n, _ in model.named_parameters() if n.startswith("mtp.")]
    if names is not None and not set(names):
        return 0
    fresh = names is None or set(names) >= set(all_mtp)
    new = [p for n, p in model.named_parameters()
           if n.startswith("mtp.") and (names is None or n in names)]
    if not new:
        return 0
    g = torch.Generator().manual_seed(seed)
    std = model.cfg.initializer_range
    fresh_ids = {id(p) for p in new}
    for name, p in model.named_parameters():
        if id(p) not in fresh_ids:
            continue
        if "hc_" in name:
            continue
        with torch.no_grad():
            if name.endswith("norm.weight"):
                p.fill_(1.0)
            elif p.dim() >= 2:
                p.normal_(0.0, std, generator=g)

    if not fresh:
        # Partial init (a resume that recovered some drafter tensors): per-tensor
        # values above, but no block-level re-init -- see the docstring.
        return len(new)
    variant = getattr(model, "hc_variant", "mhc")
    for blk in model.mtp:
        init_hc_params(blk.hc_attn_fn, blk.hc_attn_base, blk.hc_attn_scale,
                       model.cfg.hc_mult, variant)
        init_hc_params(blk.hc_ffn_fn, blk.hc_ffn_base, blk.hc_ffn_scale,
                       model.cfg.hc_mult, variant)

    n = max(len(_blocks(model)), 1)
    scale = 1.0 / math.sqrt(2 * n)
    with torch.no_grad():
        for blk in model.mtp:
            blk.attn.wo_b.weight.mul_(scale)
            blk.ffn.expert.w2.weight.mul_(scale)
    return len(new)


def init_from_scratch(cfg, engram_layout=None, engram_hashes=None,
                      hc_variant: str = "mhc", initializer_range: float = 0.02,
                      seed: int = 0) -> Transformer:
    """Construct a model and initialize it for training from scratch."""
    model = Transformer(cfg, engram_layout=engram_layout,
                        engram_hashes=engram_hashes, hc_variant=hc_variant)
    init_weights(model, initializer_range=cfg.initializer_range, seed=seed)
    return model


def count_trainable(model: Transformer) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
