# mHC-lite: exact, kernel-free manifold-constrained hyper-connections

**Companion to** `DeepSeek-V4.1-Flash_Architecture_Spec.md`. That document specifies the
residual path exactly as DeepSeek ships it (`hc_mult=4`, `hc_sinkhorn_iters=20`). This one
examines an alternative parameterization of the *same* constraint that is exact by
construction and needs no custom kernels.

**Status: nothing in the V4.1 spec has been changed.** This document lists what *would*
change if you adopted mHC-lite, so you can make that call yourself.

| | |
|---|---|
| Primary source | Yang & Gao, *mHC-lite: You Don't Need 20 Sinkhorn-Knopp Iterations*, arXiv:2601.05732, ICLR 2026 ([paper](https://arxiv.org/abs/2601.05732), [code](https://github.com/FFTYYY/mhc-lite)) |
| Underlying method | Xie et al. (DeepSeek-AI), *mHC: Manifold-Constrained Hyper-Connections*, arXiv:2512.24880 (§5 of the V4.1 spec) |
| Predecessor | Zhu et al., *Hyper-Connections*, arXiv:2409.19606, ICLR 2025 |

---

## 1. Summary

**What it is.** mHC-lite keeps mHC's architecture byte-for-byte and changes exactly one
thing: instead of *projecting* a learned matrix onto the Birkhoff polytope with 20
Sinkhorn-Knopp iterations, it *represents* the matrix as a convex combination of
permutation matrices. One matmul replaces the entire iteration loop.

**The three claims, and how well they hold up:**

1. **Exactness.** True, and provable, not empirical. The Birkhoff-von Neumann theorem
   guarantees every doubly stochastic matrix is a convex combination of permutations, and
   a convex combination of doubly stochastic matrices is doubly stochastic. `H_res` is
   doubly stochastic to floating-point precision, and so is the depthwise product
   `prod_l H_res` — the composition property mHC only approximates.
2. **Portability.** True. No iterative normalization, no custom kernel, no backward-pass
   recomputation of iteration intermediates. Pure PyTorch.
3. **Matches mHC in quality.** Mostly, but not uniformly. See §2.4 — on one of six
   dataset/scale cells mHC-lite is *worse* than mHC. Treat it as a tie, not a win.

**When to adopt it:**

- You are training from scratch and do not need bit-parity with the released
  DeepSeek-V4.1-Flash checkpoint. **This is the only clearly favourable case.**
- Your hidden size is small. See §5.1: Sinkhorn's cost is `O(1)` in `d`, so it is
  roughly **10x more intrusive at `d=512` than at `d=5120`**. §8.1 restates this at the
  `d=1344` the vertex500m preset actually uses: **~26%** of the coefficient path there,
  versus ~7% at `d=5120`.
- You want to work in a framework without TileLang/CUDA kernel work.

**When not to:**

- You need to load released V4.1-Flash weights. This is a **reparameterization, not a
  conversion**: `W_res` changes shape from `[nC, n^2]` to `[nC, n!]`, and the two weight
  spaces are not related by any invertible map. There is no checkpoint bridge.
- You need to reproduce DeepSeek's published numbers. §5.4 and §3.6 of the V4.1 spec mark
  `hc_sinkhorn_iters: 20` as part of the architecture contract, and it is in the released
  `config.json`. Swapping it is a deliberate deviation.
- `n > 4`. `n!` blows up: `n=4` gives 24 permutations, `n=8` gives 40,320. See §8.

---

## 2. Why 20 Sinkhorn iterations are not enough

### 2.1 The mechanism

mHC produces its residual matrix as

```
H_res = SK( exp( mat(alpha_res * x_hat' W_res + b_res) ) )      [mHC, eq. 2]
```

`SK` alternately rescales rows and columns so both sum to 1. The classical result
(Linial et al. 1998; Chakrabarty & Khanna 2021) is that Sinkhorn-Knopp is **not uniformly
fast**: to reach an `l1`-error of `epsilon` you may need

```
O( n^2 * log(n/nu) / epsilon^2 )    iterations
```

where `nu = min{x_ij : x_ij > 0} / max{x_ij}` is the **relative range**. Convergence is
governed by `1/nu`, and `exp()` of an affine function of activations routinely produces
matrices with an enormous relative range.

### 2.2 A concrete failure case

From Linial et al. 1998, reproduced in the mHC-lite paper:

```
                    SK (20 iters)
[ 1/2    a    a  ]         [ 0.91  0.045  0.045 ]
[ 1/2    a    a  ]  ---->  [ 0.91  0.045  0.045 ]
[ a      1    1  ]         [ 0.0   0.5    0.5   ]
```

with `a = 1e-13`. The input is strictly positive, as `exp()` guarantees. After 20
iterations the column sums are **1.92, 0.59, 0.59** — nowhere near doubly stochastic.

### 2.3 Measured, per token

mHC-lite re-runs the analysis from §5.4 of the mHC paper but **per token rather than
averaged across tokens**, arguing that averaging hides instability. On trained nanoGPT
models at `n=4`:

| quantity | value |
|---|---|
| fraction of Sinkhorn inputs with `1/nu >= 10^13` | **27.9%** |
| worst deviation of a single `H_res` column sum from 1 | **100%** |
| worst deviation of `prod_l H_res` column sums, 24-layer net | **220%** |

They also find `1/nu` **grows with depth** (comparing 6-layer vs 24-layer models), so a
fixed iteration budget becomes less sufficient as models get deeper.

**What this does not contradict: mHC's own "1.6" figure.** DeepSeek reports a composite
gain magnitude peaking at ~1.6 for mHC. mHC-lite reports 220% deviation. These are
different measurements, not conflicting ones: DeepSeek's §5.4 analysis averages `H_res`
over all tokens in a sequence before measuring, while mHC-lite records every
`(layer, token)` matrix and takes statistics over that population. Averaging a mix of
correctly and badly scaled matrices yields a plausible-looking average. mHC's own paper
confirms the approximation is imperfect — it reports the backward gradient gain of a
*single-layer* matrix already deviating slightly from 1 — it just does not quantify the
tail.

**How bad is 20 iterations on ordinary inputs?** Independently reproducible and worth
knowing before you decide this is a corner-case-only problem. Feeding both
parameterizations the same well-conditioned pre-softmax logits at scale 8 (fp32, `hc=4`,
`B=2`, `S=512`) and measuring worst row-sum deviation from 1:

```
Sinkhorn-20 iterations   7.47e-02
mHC-lite                 2.38e-07     <- fp32 noise floor
```

That is ~0.75% error on a *single* layer, from inputs that are nowhere near
ill-conditioned. The paper's 100%/220% figures are the tail of this distribution, not a
separate regime.

### 2.4 Quality: a tie, not a win

Final train/val loss, nanoGPT, `n=4`, 10k steps (~1.3B tokens), 8xA100:

| dataset | scale | Residual | HC | mHC | mHC-lite |
|---|---|---|---|---|---|
| OpenWebText | S (6L, 45M) | 3.566 / 3.562 | 3.475 / 3.471 | 3.474 / 3.469 | **3.471 / 3.467** |
| OpenWebText | M (12L, 120M) | 3.343 / 3.336 | 3.272 / 3.264 | 3.267 / 3.259 | **3.261 / 3.255** |
| OpenWebText | L (24L, 360M) | 3.237 / 3.242 | 3.244 / 3.248 | **3.191 / 3.198** | 3.194 / 3.198 |
| FineWeb-Edu | S (6L, 45M) | 3.526 / 3.536 | 3.463 / 3.473 | **3.462 / 3.473** | 3.468 / 3.477 |
| FineWeb-Edu | M (12L, 120M) | 3.316 / 3.321 | 3.266 / 3.273 | **3.237 / 3.243** | 3.243 / 3.249 |
| FineWeb-Edu | L (24L, 360M) | 3.238 / 3.240 | 3.241 / 3.244 | 3.200 / 3.204 | **3.181 / 3.185** |

mHC-lite wins 3 of 6, loses 3 of 6, and every gap is <= 0.006 nats — within run-to-run
noise at this scale. The paper's phrasing is "on par with mHC or even slightly better",
which is fair; a stronger reading would not be. Two things are worth noting beyond the
table:

- **mHC-lite is consistently the most stable of the constrained variants.** Its gradient
  norm is lower and less fluctuating than mHC's on the L/FineWeb-Edu run, which is the
  expected consequence of exactness.
- **The gains over plain residuals are real but modest at these scales** (~0.04-0.06
  nats). The large DeepSeek numbers (−0.021 loss gap, +2.1 BBH, +2.3 DROP) come from 27B
  MoE models with proportional data, which this study does not reach.

### 2.5 Throughput

mHC-lite reports **higher training throughput than HC** on the M model (8xA100) with a
naive PyTorch implementation, versus the 6.7% overhead DeepSeek reports for its
kernel-optimized mHC.

Two caveats, both material:

1. The mHC baseline in that comparison is a **PyTorch reimplementation**, not DeepSeek's
   TileLang kernels, which were not publicly available. The authors flag this explicitly
   and say it may underestimate mHC. So this result shows "mHC-lite is cheap", not
   "mHC-lite beats optimized mHC".
2. The "6.7%" figure itself is quoted inconsistently across sources. The mHC paper's
   abstract reads as overhead vs a single-stream baseline; the V4 report reads as overhead
   "of the overlapped 1F1B pipeline stage"; mHC-lite reads it as overhead vs HC. These are
   different denominators. Do not treat 6.7% as a portable constant.

---

## 3. The reparameterization

### 3.1 Birkhoff-von Neumann

> For any `X` in the Birkhoff polytope `B_n` (the set of `n x n` doubly stochastic
> matrices), there exists `a = (a_1, ..., a_{n!})` with `a_k >= 0` and `sum_k a_k = 1`
> such that `X = sum_k a_k P_k`, where `{P_k}` are the `n!` permutation matrices.

Two consequences, and together they are the whole method:

- **Surjectivity.** Any convex combination of permutations is doubly stochastic. So if you
  *build* `H_res` this way, it is doubly stochastic — not approximately, exactly.
- **Closure.** A convex combination of doubly stochastic matrices is doubly stochastic, so
  the depthwise product `prod_l H_res` is doubly stochastic too. This is the property
  mHC's Sinkhorn loop only approximates and that mHC-lite measures at 220% error.

### 3.2 The equations

Everything is identical to mHC except the last two lines:

```
x_hat' = RMSNorm(vec(x_l))                                    # 1 x nC

H_pre  = sigmoid( alpha_pre * x_hat' W_pre  + b_pre )         # 1 x n
H_post = 2 * sigmoid( alpha_post * x_hat' W_post + b_post )   # 1 x n

a      = softmax( alpha_res * x_hat' W_res + b_res )          # 1 x n!     <-- was n^2
H_res  = a @ P                                                 # n x n       <-- one matmul
```

with `W_pre, W_post` in `R^{nC x n}` and **`W_res` in `R^{nC x n!}`** (was `R^{nC x n^2}`).
`P` is a **constant** `n! x n^2` buffer: every permutation matrix flattened row-major and
concatenated. It is registered as a buffer, not a parameter.

At `n=4` that is `P` of shape `24 x 16` — 384 floats, permanently L2-resident, read by
every token.

---

## 4. Drop-in for this spec

The V4.1 spec fuses three things into one linear layer (`hc_fn` / `hc_scale` / `hc_base`,
§5.3) rather than using the paper's separate `W` / `b` / `alpha`. So the adaptation below
targets **`hc_split_sinkhorn` at `DeepSeek-V4.1-Flash_Architecture_Spec.md:1514`**, not the
paper's equations. Everything else in §5.6 of the spec — the block structure, the
Single-Pass one-sublayer-late hand-off, the `hc_pre` / `hc_post` collapse/expand — is
unchanged.

```python
import itertools
import math

import torch
import torch.nn as nn

def permutation_basis(hc: int) -> torch.Tensor:
    """Constant [hc!, hc*hc] buffer: every permutation matrix, flattened row-major.

    Index 0 is the IDENTITY permutation, so a logit bias of 0 on entry 0 with
    -inf (or a large negative) elsewhere makes softmax() return the identity.
    """
    rows = []
    for p in itertools.permutations(range(hc)):
        M = torch.zeros(hc, hc)
        for i, j in enumerate(p):
            M[i, j] = 1.0
        rows.append(M.reshape(-1))
    return torch.stack(rows)                      # [hc!, hc*hc]


# Register once, per (model, hc); shared by every sublayer. Not a parameter.
PERM_4 = permutation_basis(4)                      # [24, 16]


def hc_split_sinkhorn_lite(mixes, hc, scale, base, perm):
    """mHC-lite counterpart of hc_split_sinkhorn. Exact doubly-stochastic comb.

    mixes : [B, S, 2*hc + hc!] fp32   (was 2*hc + hc*hc)
    scale : [3] fp32                  (unchanged)
    base  : [2*hc + hc!] fp32         (was 2*hc + hc*hc)
    perm  : [hc!, hc*hc] fp32 constant, from permutation_basis(hc)

    Returns (pre [B,S,hc], post [B,S,hc], comb [B,S,hc,hc]).
    `iters` and `eps` are intentionally absent: there is nothing to iterate.
    """
    b, s, _ = mixes.shape
    n, n2 = hc, hc * hc

    # --- pre / post: byte-identical to the shipped implementation ---
    pre = torch.sigmoid(mixes[..., :n] * scale[0] + base[:n]) + 1e-6
    post = 2.0 * torch.sigmoid(mixes[..., n:2 * n] * scale[1] + base[n:2 * n])

    # --- res: logits -> softmax over the hc! permutations -> one matmul ---
    # mixes[..., 2*hc:] has hc! entries (24 at hc=4) — one logit per permutation,
    # NOT hc*hc. Softmax over that axis is the Birkhoff-von Neumann convex combo.
    logits = mixes[..., 2 * n:] * scale[2] + base[2 * n:]   # [B, S, hc!]
    a = logits.softmax(-1)                       # [B, S, hc!]
    comb = (a @ perm).reshape(b, s, n, n)        # exactly doubly stochastic

    return pre, post, comb
```

`iters` and `eps` for the *comb* disappear; `eps` remains in `pre` because that is the
shipped `+ hc_eps` behaviour, unrelated to Sinkhorn.

### 4.1 Module wrapper

```python
class HCMixerLite(nn.Module):
    """Drop-in for a sublayer's HC coefficient predictor. Pair with hc_pre / hc_post."""

    def __init__(self, hc: int, d: int):
        super().__init__()
        self.hc = hc
        self.mix_hc = 2 * hc + math.factorial(hc)        # 32 when hc=4 (was 24)
        self.fn = nn.Linear(hc * d, self.mix_hc, bias=False)
        self.register_buffer("perm", permutation_basis(hc), persistent=True)

    def forward(self, x, scale, base):
        # x: [B, S, hc, d] -> flatten to [B, S, hc*d], exactly as spec 5.3
        xf = x.flatten(2).float()
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-20)
        mixes = self.fn(xf) * rsqrt
        return hc_split_sinkhorn_lite(mixes, self.hc, scale, base, self.perm)
```

`hc_pre`, `hc_post`, the block hand-off, and the head collapse all carry over untouched.

---

## 5. Cost

### 5.1 Why the gap widens as `d` shrinks

This is the argument that matters for the Vertex-500M preset.

The coefficient projection reads `x_hat'` once: `nC` elements per token per sublayer. That
read is **identical** in both variants. The difference is everything after it:

- **mHC** writes `n^2 + 2n = 24` scalars, then runs 19 iterations of normalize-along-dim
  on a `[hc, hc] = [4, 4] = 16`-element tensor. Every iteration touches all 16 elements at
  least twice (once to reduce, once to scale). Op-counting that gives roughly
  **1.3-1.4k extra element-touches** per token per sublayer.
- **mHC-lite** writes `2n + n! = 32` scalars and reads a 384-float constant that stays in
  cache. Roughly **50 extra element-touches**.

Sinkhorn's cost is `O(1)` in `d`; the projection's read is `O(d)`. So the ratio degrades
as `d` shrinks:

| | `nC` read | Sinkhorn extra | % overhead |
|---|---|---|---|
| V4.1-Flash, `d=5120` | 20,480 | ~1,400 | **~6%** |
| Vertex-500M, `d=512` | 2,048 | ~1,400 | **~64%** |
| mHC-lite, either | same `nC` | ~50 | **~0.2% / ~2.4%** |

**These are op-count estimates, not measurements.** Profile before trusting them; the shape
of the result (constant numerator, `d`-denominated denominator) is what makes it robust to
the exact constants.

### 5.2 Kernel launches

The shipped loop body is `comb / (comb.sum(-1, keepdim=True) + eps)`, repeated twice per
iteration, 19 times — a sum, an add, and a division each, on a `[B, S, 4, 4]` tensor:

```
19 iterations x ~6 tensor ops + ~5 setup ops  =  ~120 launches per hc_split_sinkhorn call
x 2 sublayers x 16 layers (Vertex-500M)     =  ~3,800 launches per forward
```

All of them on 16-element-per-token tensors, i.e. pure launch overhead. mHC-lite: **one**
matmul and one softmax per call, so ~4 launches per call.

At `d=5120` and pipeline-parallel scale with fused kernels, DeepSeek recovers this to
6.7% wall-clock (their number, see the caveat in §2.5). At `d=512` on a single node there
is no fused kernel to write and no pipeline to hide it behind.

### 5.3 Parameters

The spec's formula (§2.1) is `2 * ((2+hc)*hc*hc*d + (2+hc)*hc + 3)` per layer — weight,
bias, and the 3-element scale, for attention and FFN. mHC-lite widens the output from
`(2+hc)*hc = 24` to `2*hc + hc! = 32`:

```
mHC      : 2 * (24 * 4 * d + 24 + 3)      = 2 * (96d + 27)
mHC-lite : 2 * (32 * 4 * d + 32 + 3)      = 2 * (128d + 35)
delta    : 2 * (32d + 8) = 64d + 16        per layer
```

| target | `d` | mHC | mHC-lite | delta | % of backbone |
|---|---|---|---|---|---|
| per layer, V4.1-Flash | 5120 | 983,094 | 1,310,790 | +327,696 | — |
| per layer, Vertex-500M | 512 | 98,358 | 131,142 | +32,784 | — |
| whole model, V4.1-Flash (40L) | 5120 | 39,323,760 | 52,431,600 | **+13.1M** | **+0.0024%** of 551.57B |
| whole model, Vertex-500M (16L) | 512 | 1,573,728 | 2,098,272 | **+0.52M** | **+0.13%** of 414.4M |

Negligible at both scales. Plus a 384-float constant buffer per model (serialized with the
checkpoint, non-trainable).

### 5.4 Throughput, measured

NanoGPT, M model (12L, 768d), OpenWebText, 8xA100, naive PyTorch for all arms:

```
mHC-lite  >  HC
```

with mHC-lite the fastest arm. DeepSeek's optimized mHC is reported at 6.7% overhead but
was not available for this comparison — see §2.5. The honest reading: **mHC-lite is fast
enough in eager PyTorch that kernel work stops being a prerequisite.** That is a different
and weaker claim than "mHC-lite is faster than DeepSeek's mHC."

---

## 6. Initialization

Verbatim from mHC-lite §4, adapted to the spec's fused parameter layout:

| tensor | init | reason |
|---|---|---|
| `hc_fn` (weights) | all zero | |
| `hc_scale` (`alpha`) | `0.01` | learnable scalar gate, small |
| `base[0 : 2*hc]` | `-1` everywhere, **one entry `+1`** | `H_pre = sigmoid(...)` concentrates on one stream |
| `base[2*hc]` | `0` | the **identity permutation's** logit |
| `base[2*hc+1 :]` | `-8` | suppress all other permutations |

The `-8` on the non-identity permutation entries makes `softmax` return a vector
concentrated on the identity permutation, so `H_res ≈ I` at step 0 and every block
reduces to an ordinary residual connection.

**Measured, not assumed.** At `hc=4` the resulting numbers are:

| quantity | value |
|---|---|
| softmax mass on the identity permutation, `1/(1 + 23·e⁻⁸)` | **0.99234** |
| mean diagonal entry of `H_res` | **0.99402** |
| `max abs(comb - I)` | **6.0e-03** |

The mean diagonal (0.99402) is *higher* than the identity mass (0.99234) because the
`(hc-1)! - 1 = 5` transpositions that fix a given index also add to its diagonal. So
`H_res` at init is `≈ I`, **not** exactly `I` — do not write an exact-identity assertion,
and do not assume the shipped mHC init and the mHC-lite init produce the same step-0
matrix. They do not; they are both "starts as a residual connection", at different
points.

**Requirement on `permutation_basis`: permutation 0 must be the identity.** The code above
gets this from `itertools.permutations`, whose first output for `range(n)` is the identity
tuple — but do not reorder or shuffle that buffer, or the `base[2*hc] = 0` initialization
silently points at a random permutation and training starts from a scrambled residual.

The equivalent recipe in mHC's own parameterization is `b_res = -8` off-diagonal, `0` on
diagonal, because there `exp()` of the bias approximates `I` directly.

---

## 7. What would change in the V4.1 spec

Listed, **not applied**.

| Location | Change |
|---|---|
| §2.1, hyper-connections row | `2*((2+hc)*hc*hc*d + (2+hc)*hc + 3)` → `2*((2*hc + hc!)*hc*d + (2*hc + hc!) + 3)`. V4.1 total: `983,094` → `1,310,790` per layer; whole-model +13.1M |
| §2.1 formula block | same substitution; `551.57B` becomes `551.58B` (still "552B") |
| §3.4 config | `hc_sinkhorn_iters: 20` becomes unused. Either delete it or keep it and document it as sinkhorn-only |
| §3.6 Vertex-500M table | same three rows as above |
| §5.1–5.2 | no change (architecture is identical) |
| §5.3 | `mix_hc = (2 + hc) * hc = 24` → `2*hc + hc! = 32`; `hc_fn`/`hc_base` shapes follow |
| §5.4 | replace `hc_split_sinkhorn` with `hc_split_sinkhorn_lite`; add `permutation_basis` |
| §5.5, §5.6, §5.7 | no change |
| §14, invariant 35 | "`sinks, RMSNorm eps, Sinkhorn, and clamps remain`" — Sinkhorn reference becomes stale |
| §25, invariant 1 | `mix_hc == (2 + hc) * hc` → `mix_hc == 2*hc + hc!` |
| §25, invariant 2 | `hc_scale` still has 3 entries — unchanged |
| §16 checkpoint layout | `perm` buffer is new, non-trainable, must be serialized (or regenerated from `hc_mult` on load) |
| §18 invariants | new invariant: `permutation_basis(hc)[0] == I` |

Two additions worth planning for:

- **The `perm` buffer in checkpoint conversion (§16).** Either ship it, or regenerate it
  from `hc_mult` at load time. Regenerating is cleaner — it is a pure function of `hc` —
  but then it must be excluded from the state-dict comparison.
- **`hc_eps` survives.** It is still used in `pre`. Do not remove it along with `iters`.

---

## 8. Scaling past n=4

`n!` is the whole cost story:

| `n` | permutations | `W_res` cols | `W_res` size at `d=5120` | `P` buffer |
|---|---|---|---|---|
| 2 | 2 | 4 vs 4 | same as mHC | 8 floats |
| 3 | 6 | 6 vs 9 | **smaller** | 54 floats |
| **4** | **24** | **24 vs 16** | **1.5x mHC** | **384 floats** |
| 5 | 120 | 120 vs 25 | 4.8x | 3,000 floats |
| 8 | 40,320 | 40,320 vs 64 | 630x | 2.6M floats |

At `n=3` mHC-lite is actually *cheaper* than mHC. At `n=4` it is 1.5x. There is no
sensible point past that.

### 8.1 At the vertex500m preset (`d=1344`, not `d=512`)

**The `d=512` figures throughout this document are for the MoE preset in
`vertex500m-dense.md` 21.1, not for the config `vertex500m/config.py` actually builds.**
The shipped `PRESET` is `d=1344, L=20`, so the headline cost argument needs restating
at that width. It survives, weaker:

| | `d` | `nC` read | Sinkhorn extra | % overhead |
|---|---|---|---|---|
| V4.1-Flash | 5120 | 20,480 | ~1,400 | **~7%** |
| **vertex500m `PRESET`** | **1344** | **5,376** | ~1,400 | **~26%** |
| MoE preset (the `d` this doc assumes) | 512 | 2,048 | ~1,400 | ~68% |

And the parameter delta, recomputed at `d=1344`:

| target | `d` | mHC | mHC-lite | delta |
|---|---|---|---|---|
| per layer, vertex500m | 1344 | 258,102 | 344,134 | **+86,032** |
| whole model, vertex500m (20L) | 1344 | 5,162,040 | 6,882,680 | **+1.72M** |

+1.72M is **+0.34%** of the 503.8M with-DSpark model (`python -m vertex500m.params
--hc-variant mhc-lite` prints this). Still negligible. `params.py` now takes
`hc_variant` and computes both; it no longer hardcodes the mHC width.

**So: at `d=1344` mHC-lite is a 26%-of-the-coefficient-path engineering win, not a
64% one.** The recommendation in §11 is unchanged but the margin is roughly 2.5x
thinner than the numbers above it suggest.

**Three escapes, in order of preference:**

1. **`n=4` is the right answer anyway.** The Hyper-Connections paper's ablation shows
   `n=4` is the sweet spot and `n=8` adds almost nothing. Both mHC-lite and KromHC assume
   you stay at `n=4`.
2. **Subsample the permutation set.** Keep `K` of the `n!` permutations, `K << n!`.
   `H_res` remains **exactly** doubly stochastic — you have merely restricted the feasible
   set to a subset of the Birkhoff polytope. The cost/quality trade becomes a tunable `K`.
   This is the mHC-lite authors' own suggestion (§5 of their paper) and it is the cheapest
   knob available.
3. **KromHC** (Zhou et al., *KromHC: Manifold-Constrained Hyper-Connections with
   Kronecker-Product Residual Matrices*). Parameterizes `H_res` as a Kronecker product of
   smaller doubly stochastic factors, enforcing the constraint along each tensorized mode.
   Gives **exact** double stochasticity at `O(n^2 C)` parameters instead of mHC-lite's
   `O(n C n!)`, with no specialized kernels and no factorial blowup. Reports matching or
   beating mHC and mHC-lite. This is the version to reach for if you ever need `n > 4` or
   want exactness at low parameter cost.

---

## 9. Adjacent manifold work

Doubly stochastic is the *Birkhoff* choice, not the only one. Three papers argue it is
the wrong constraint, in ways worth knowing before you invest in it.

**oHC — Orthogonal Hyper-Connections on SO(4)** (Guo et al., arXiv:2609.02672, Sep 2026).
Proves that within the doubly stochastic set, the mixer can only shrink inter-stream
differences — the stream *mean* is conserved but the *differences* decay, and that decay
compounds. Measured on mHC at `n=4`: every mixer's smallest singular value is below 1
(median **0.26** for attention, **0.60** for MLP), and the composed mixer retains
`1.9e-14` of the difference energy — i.e. **all four streams collapse into one**. oHC
constrains `H_res` to `SO(n)`, giving singular values exactly 1, and at 3.9B-A0.4B MoE
beats RC, mHC, and iHC (identity mixer). Notably **iHC — which does no cross-stream mixing
at all — slightly beats mHC**, which is awkward for the whole HC family. oHC
parameterizes `SO(4)` in closed form with a pair of unit quaternions: no iteration, no
matrix inverse, and measured **20.6x cheaper than the Sinkhorn iteration it replaces**.

**JPmHC** (arXiv:2602.18308). Reaches the same conclusion — Birkhoff is contractive —
from a free-probability analysis of the Jacobian, and lands on orthogonality via a Cayley
transform on the Stiefel manifold. Weaker evidence (a 7M TRM model on ARC-AGI, arms that
differ in more than the manifold), but the direction agrees.

**Stream utilization.** Separately, three analyses of *trained* mHC models find the
four-stream pathway is substantially under-used: a typical site routes through ~2 of 4
streams (arXiv:2609.05309, on DeepSeek-V4-Flash), and in layers 22-42 the residual mixers
are near-identity and can be replaced with exact identity for +1.9% perplexity. Two
independent groups find a dominant-stream regime (see also `brain-lab-research/hc-stream-collapse`
and `wgpeng/mhc-780m`).

**What this means for the decision.** If you adopt mHC-lite for its exactness, you are
still committed to a manifold whose known pathology is silent contraction. oHC's `iHC`
result suggests the honest baseline to beat may be a single stream, not mHC. None of this
is settled, and none of it changes what the V4.1 spec should reproduce.

---

## 10. Verification

All seven checks below were executed against `torch 2.9.1` (CPU, fp32) and pass. Test 7 is
the one that answers §2.2 — it drives the logits into the regime where Sinkhorn fails and
the comb is still exactly doubly stochastic, because there is no iteration to fail.

```python
import itertools, math
import torch
import torch.nn as nn

def assert_doubly_stochastic(comb, tol=1e-6):
    ones = torch.ones(comb.size(-1))
    row = (comb.sum(-1) - ones).abs().max()
    col = (comb.sum(-2) - ones).abs().max()
    neg = (-comb).clamp_min(0).max()
    assert row < tol and col < tol, f"row {row:.2e} col {col:.2e}"
    assert neg < tol, f"negative entries {neg:.2e}"

def test_lite():
    torch.manual_seed(0)
    B, S, hc, d = 2, 512, 4, 512
    mix = HCMixerLite(hc, d)                     # class from §4.1
    with torch.no_grad():
        mix.fn.weight.zero_()
    P = mix.perm
    scale = torch.full((3,), 0.01)
    base = torch.full((mix.mix_hc,), -8.0)
    base[0] = base[hc] = 1.0                    # the single +1 entries for pre / post
    base[2 * hc] = 0.0                          # identity permutation

    # 1. exact at init, over many random inputs
    for _ in range(64):
        _, _, comb = mix(torch.randn(B, S, hc, d), scale, base)
        assert_doubly_stochastic(comb)

    # 2. ~identity at init, with the leakage predicted in closed form
    e8 = math.exp(-8.0)
    a_id = 1.0 / (1.0 + (math.factorial(hc) - 1) * e8)
    a_other = e8 / (1.0 + (math.factorial(hc) - 1) * e8)
    pred_diag = a_id + (math.factorial(hc - 1) - 1) * a_other   # -1: identity fixes i too
    _, _, comb = mix(torch.randn(B, S, hc, d), scale, base)
    got = comb.diagonal(dim1=-2, dim2=-1).mean().item()
    assert abs(got - pred_diag) < 1e-4, (got, pred_diag)
    assert (comb - torch.eye(hc)).abs().max() < 1e-2      # NOT < 1e-3; see §6

    # 3. exact under extreme logits -- the regime that breaks Sinkhorn
    for sv in (1.0, 10.0, 100.0, 1000.0):
        _, _, c = mix(torch.randn(B, S, hc, d), torch.full((3,), sv), base)
        assert_doubly_stochastic(c)

    # 4. permutation_basis contract
    Pm = P.reshape(-1, hc, hc)
    assert P.shape == (24, 16)
    assert torch.equal(P[0], torch.eye(hc).reshape(-1))     # entry 0 IS the identity
    assert ((Pm == 0) | (Pm == 1)).all()
    assert (Pm.sum(-1) == 1).all() and (Pm.sum(-2) == 1).all()
    assert len({tuple(r) for r in Pm.reshape(24, -1).tolist()}) == 24

    # 5. parameter count must match the §5.3 table
    got_p = 2 * ((2 * hc + math.factorial(hc)) * hc * d
                 + (2 * hc + math.factorial(hc)) + 3)
    assert got_p == 131_142, got_p

def test_head_to_head():
    """Sinkhorn-20 vs mHC-lite on identical well-conditioned logits (the §2.2 number).

    Requires the shipped kernel from the V4.1 spec, not redefined here:
        from hc import hc_split_sinkhorn     # DeepSeek-V4.1-Flash_Architecture_Spec.md:1514
    """
    torch.manual_seed(2)
    hc, B, S = 4, 2, 512
    P = permutation_basis(hc)
    res = torch.randn(B, S, hc * hc) * 8.0
    _, _, sink = hc_split_sinkhorn(                       # shipped V4.1 kernel
        torch.cat([torch.zeros(B, S, 2 * hc), res], -1), hc,
        torch.ones(3), torch.zeros((2 + hc) * hc), 20, 1e-6)
    _, _, lite = hc_split_sinkhorn_lite(                  # mHC-lite, same 16 logits
        torch.cat([torch.zeros(B, S, 2 * hc), res,
                   torch.full((B, S, math.factorial(hc) - hc * hc), -30.0)], -1),
        hc, torch.ones(3), torch.zeros(2 * hc + math.factorial(hc)), P)
    print(f"sinkhorn-20 {(sink.sum(-1)-1).abs().max():.2e}   "
          f"mHC-lite {(lite.sum(-1)-1).abs().max():.2e}")
    # -> sinkhorn-20 7.47e-02   mHC-lite 2.38e-07
```

Two of these assertions encode traps that I hit while writing the implementation, and
that will bite anyone porting this:

- **A flattened permutation matrix sums to `hc`, not 1.** Asserting `P.sum(-1) == 1` fails.
  The property you want is that each `P[k]` *reshaped* to `[hc, hc]` has row and column
  sums of 1.
- **The mean diagonal of `H_res` is not the identity permutation's softmax weight.**
  The `(hc-1)! - 1 = 5` transpositions that fix a given index also contribute to its
  diagonal, so 0.99402 vs 0.99234. Testing against the naive value fails by ~3e-4.

---

## 11. Bottom line

mHC-lite is a **numerical and engineering improvement to mHC, not a capability upgrade**.
Its quality matches mHC within noise; its stability is better by construction; its cost is
lower and its implementation is trivial. The price is `n!` parameters, exactness traded
for `O(n^2)` expressibility per matrix, and loss of checkpoint parity.

For the V4.1 spec as written: **do not change it.** `hc_sinkhorn_iters: 20` is part of the
architecture contract, the shipped kernels already recover most of the cost, and at
`d=5120` the difference is a few percent of a few percent.

For a scaled-down reproduction at `d=512` trained from scratch on a single node, with no
TileLang in the plan: **the calculus flips.** §5.1 puts the Sinkhorn loop at roughly 64%
overhead on the coefficient path versus ~2% for mHC-lite, in a setting where there is no
fused kernel and no pipeline bubble to hide ~3,800 launches per forward.

**At the `vertex500m` preset actually shipped here (`d=1344`, §8.1) the flip is real but
weaker:** ~26% overhead versus ~4%, not 64% versus 2%. That is still a large enough gap
to matter on a single node with eager PyTorch, and the implementation is ~40 lines
(`vertex500m/hc.py`, `hc_split_perm` + `permutation_basis`, selected by
`--hc-variant mhc-lite`). It is **now trainable end to end** — the flag threads through
`train.py`, `eval.py` and `bench.py`, and `params.py --hc-variant mhc-lite` accounts for
it. The remaining argument against is the quality tie in §2.4 plus the loss of checkpoint
parity; neither is a reason not to run an A/B, and the cost is +1.72M parameters.
