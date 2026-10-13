"""Hyper-Connections: mHC residual stream, Sinkhorn comb, Single-Pass hand-off (spec 5).

The residual stream is `hc` parallel copies of width `d`. Each block predicts, from its
own input, three sets of coefficients that (a) collapse the stream into one vector for
the sublayer, (b) expand the sublayer output back to `hc` copies, and (c) mix the
residual. The mixing matrix is constrained to the Birkhoff polytope by projecting with
`hc_sinkhorn_iters` Sinkhorn-Knopp passes -- an APPROXIMATION (see mHC-lite.md 2.3 for
the measured 7.47e-02 worst-case deviation after 20 iterations).

`mHCVariant` switches to the exact convex-combination-of-permutations parameterization
(mHC-lite). It is bit-different from the shipped kernel and has no checkpoint bridge;
it is opt-in and off by default.
"""

from __future__ import annotations

import itertools
import math

import torch

from .rope import rms_recip


# --------------------------------------------------------------------------- #
# Exact variant (mHC-lite). Off by default; see module docstring.
# --------------------------------------------------------------------------- #
def permutation_basis(hc: int, device=None) -> torch.Tensor:
    """Constant [hc!, hc*hc] buffer: every permutation matrix flattened row-major.

    Entry 0 IS the identity permutation. `itertools.permutations` yields the identity
    tuple first; do not reorder or shuffle this buffer, or a logit bias of 0 on entry 0
    silently points at a random permutation and training starts from a scrambled
    residual (mHC-lite.md 6).
    """
    rows = []
    for p in itertools.permutations(range(hc)):
        M = torch.zeros(hc, hc, device=device)
        for i, j in enumerate(p):
            M[i, j] = 1.0
        rows.append(M.reshape(-1))
    return torch.stack(rows)


# --------------------------------------------------------------------------- #
# Coefficient split
# --------------------------------------------------------------------------- #
def hc_split_sinkhorn(mixes: torch.Tensor, hc: int, scale: torch.Tensor,
                      base: torch.Tensor, iters: int, eps: float):
    """mixes [b,s,(2+hc)*hc] -> (pre [b,s,hc], post [b,s,hc], comb [b,s,hc,hc]).

    Layout of the (2+hc)*hc outputs:
        pre  [0     : hc  ]
        post [hc    : 2hc ]
        comb [2hc   : 2hc+hc*hc ]  comb[j,k] weights residual[j] into output[k]
    """
    b, s, _ = mixes.shape
    pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2.0 * torch.sigmoid(mixes[..., hc:2 * hc] * scale[1] + base[hc:2 * hc])
    comb = mixes[..., 2 * hc:].view(b, s, hc, hc) * scale[2] + base[2 * hc:].view(hc, hc)
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def hc_split_perm(mixes: torch.Tensor, hc: int, scale: torch.Tensor,
                  base: torch.Tensor, perm: torch.Tensor, eps: float):
    """mHC-lite: softmax over the hc! permutations, then one matmul.

    EXACTLY doubly stochastic by the Birkhoff-von Neumann theorem, and closed under
    composition, so prod_l H_res is also doubly stochastic. `iters` is absent because
    there is nothing to iterate. `eps` survives for `pre` only.
    """
    b, s, _ = mixes.shape
    pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2.0 * torch.sigmoid(mixes[..., hc:2 * hc] * scale[1] + base[hc:2 * hc])
    logits = mixes[..., 2 * hc:] * scale[2] + base[2 * hc:]   # [b,s,hc!]
    a = logits.softmax(-1)
    comb = (a.to(perm.dtype) @ perm).view(b, s, hc, hc)
    return pre, post, comb


# --------------------------------------------------------------------------- #
# Collapse / expand
# --------------------------------------------------------------------------- #
def hc_pre(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """Collapse hc copies: [b,s,hc,d] x [b,s,hc] -> [b,s,d].

    This is a reduction over `hc` (4) terms, not over `d`, so doing it in the activation
    dtype costs one rounding rather than a meaningful accumulation error -- the input is
    already fp16 under autocast, so an fp32 sum of fp16 values buys nothing.

    The `.float()` that used to be here materialised a [b,s,hc,d] fp32 tensor, 21 KB per
    token, on each of the 41 `hc_pre` calls in a forward pass. See `hc_post` and
    `hc_mixes` for the rest of that bill.
    """
    return (pre_mix.unsqueeze(-1).to(x.dtype) * x).sum(2)


def hc_post(y: torch.Tensor, residual: torch.Tensor, post: torch.Tensor,
            comb: torch.Tensor) -> torch.Tensor:
    """Expand a sublayer output back to hc copies.

    `post` and `comb` arrive in fp32 and stay there: `comb` is the Birkhoff/Sinkhorn
    mixing matrix, and 20 alternating normalisation passes over a [b,s,hc,hc] tensor is
    precisely where the doubly-stochastic constraint is enforced. fp16 would violate it
    after a handful of iterations (`hc_eps` is 1e-6, which is also near fp16's resolution
    against coefficients of order 0.25, so the stabiliser stops doing anything).

    What runs in the activation dtype are the two products against the residual stream --
    the large tensors. `hc_post` is called twice per layer, so these were another
    ~1.7 MB/token of fp32 traffic.

    The residual mix is one batched matmul, not an elementwise product plus a sum:
    out[h,d] = sum_k comb[h,k] * residual[k,d], i.e. `comb @ residual`. The old
    broadcast-then-sum materialised a [b,s,hc,hc] intermediate -- 4x the residual
    stream at hc=4 -- to compute exactly that. Same math, different reduction order.
    """
    out = post.to(y.dtype).unsqueeze(-1) * y.unsqueeze(-2)
    # Both operands are explicitly in the activation dtype (never rely on autocast
    # coverage here: on CPU only a subset of ops lower, and the explicit casts are
    # what keep this path low-precision everywhere -- see
    # tests.py::test_hc_path_stays_lowp_under_amp). residual always shares y's
    # dtype: every producer (hc_post itself, the embedding seed, norms, Engram)
    # returns the activation dtype.
    mix = torch.matmul(comb.to(y.dtype), residual.to(y.dtype))
    return (out + mix).to(y.dtype)


def hc_mixes(x: torch.Tensor, hc_fn, hc_scale, hc_base, hc: int, norm_eps: float,
             iters: int, eps: float, perm: torch.Tensor | None = None):
    """Predict this sublayer's coefficients from the current stream state.

    The stream is flattened to [b,s,hc*d] and reduced in fp32 (norm_eps is 1e-20, see
    `rope.rms_recip`), but the projection itself runs on the ORIGINAL activation: under
    autocast `F.linear` is an eligible op, so it lands on the tensor cores in fp16. The
    previous `x = x.flatten(2).float()` upcast the whole stream for the benefit of a
    projection whose output is only (2+hc)*hc = 24 channels wide -- 860 KB/token of
    fp32 traffic for ~1% of the model's FLOPs, twice per layer.
    """
    flat = x.flatten(2)
    rsqrt = rms_recip(flat, norm_eps)
    mixes = torch.nn.functional.linear(flat, hc_fn) * rsqrt
    if perm is None:
        return hc_split_sinkhorn(mixes, hc, hc_scale, hc_base, iters, eps)
    return hc_split_perm(mixes, hc, hc_scale, hc_base, perm, eps)


def make_identity_pre_mix(b: int, s: int, hc: int, device=None) -> torch.Tensor:
    """The first block's `pre` is a one-hot on stream copy 0."""
    m = torch.zeros(b, s, hc, dtype=torch.float32, device=device)
    m[..., 0] = 1.0
    return m


# --------------------------------------------------------------------------- #
# Parameter layout
# --------------------------------------------------------------------------- #
def mix_hc_for(hc: int, variant: str = "mhc") -> int:
    """Width of the fused (fn, base) projection per sublayer."""
    return (2 + hc) * hc if variant == "mhc" else 2 * hc + math.factorial(hc)


def init_hc_params(hc_fn: torch.Tensor, hc_base: torch.Tensor,
                   hc_scale: torch.Tensor, hc: int, variant: str = "mhc") -> None:
    """Initialize the hyper-connection parameters so the block starts as a residual.

    The spec's reference uses torch.empty for hc_*_fn, which is uninitialized garbage.
    This recipe makes step 0 an ordinary residual connection:

      hc_fn    : all zeros  -> constant coefficients regardless of the stream
      hc_scale : 0.01       -> small learnable gate
      base[:2*hc] : -1 everywhere except ONE entry at +1 -> H_pre concentrates on a
                     single stream at init instead of smearing over all hc copies
      base[2*hc:]   : for Sinkhorn this is a bias added to the [hc,hc] logits; for
                     mHC-lite it is a per-permutation logit where entry 0 is the
                     identity permutation.

    H_res at init is APPROXIMATELY the identity, not exactly: under mHC-lite, 0.99234
    of the softmax mass lands on the identity and the mean diagonal is 0.99402, because
    the (hc-1)! - 1 = 5 transpositions that fix a given index also add to its diagonal.
    Do not assert an exact identity.
    """
    with torch.no_grad():
        hc_fn.zero_()
        hc_scale.fill_(0.01)
        hc_base.zero_()
        hc_base[: 2 * hc].fill_(-1.0)
        hc_base[0] = 1.0
        if variant == "mhc-lite":
            # base[2*hc] is the identity permutation's logit (already 0); every other
            # permutation is suppressed so softmax concentrates on the identity.
            hc_base[2 * hc + 1:] = -8.0
        else:
            # Sinkhorn: exp() of the bias approximates I directly, so -8 off-diagonal
            # and 0 on the diagonal (the equivalent recipe in mHC's own parameterization).
            hc = hc
            for i in range(hc):
                for j in range(hc):
                    hc_base[2 * hc + i * hc + j] = 0.0 if i == j else -8.0
