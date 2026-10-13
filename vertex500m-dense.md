# vertex500m: pure-dense conversion of the DeepSeek-V4.1-Flash architecture

**Companion to** `DeepSeek-V4.1-Flash_Architecture_Spec.md` (the architecture of record) and
`mHC-lite.md` (the hyper-connection variant study).

**This is not DeepSeek's model.** `vertex500m` is an independent reproduction of the V4.1
architecture at 0.5B scale with the feed-forward sublayer made dense. Nothing here is
DeepSeek's; the V4.1 spec remains the description of their released weights, and this
document is a deviation from it in every dimension that matters for checkpoint parity.

**Status.** The V4.1 spec is **not modified by this document.** This document specifies a
derived configuration of the same architecture in which the FFN is a single dense SwiGLU
instead of a DeepSeekMoE routed bank, bound to a specific third-party tokenizer, and shaped
for a particular execution model (§4.6). Use it as the conversion checklist and the sizing
derivation; use the V4.1 spec for everything else.

**One-line summary.** V4.1's parameter budget is 98% routed-expert bank and 1% attention. At
0.5B total that allocation is dubious — 94% of every layer sits in 32 experts of which each
token touches 2, so every expert's gradient is estimated from `B/16` tokens per step instead
of `B` — so the whole FFN becomes a dense SwiGLU, embeddings are tied, the drafter drops to
one stage, and the shape is chosen wide-and-shallow for a `torch.compile` layer loop:
active parameters go **40.1M → 329.5M (8.2×)** at **490.1M total**, in **1.977 GFLOP/token**.

---

## Table of contents

**Part I — The argument**

- §0 Executive summary
- §1 Why dense at 500M: the evidence, for and against
- §2 The load-bearing arithmetic
- §3 The survival map: what changes, what does not, what breaks

**Part II — The conversion**

- §4 `PRESET_VERTEX_500M`: the configuration
- §5 Parameter accounting
- §6 Dense FFN — replacement text for spec §6
- §7 Config delta (exact)
- §8 Reference implementation
- §9 Checkpoint layout and conversion
- §10 Quantization: what survives FP4 and what does not

**Part III — Consequences**

- §11 Training
- §12 CED and the prefill/decode activation split
- §13 Engram at a 17.9% share
- §14 DSpark
- §15 Multimodality after router removal
- §16 The scaling-down guide, and binding to qt-VI.1.3

**Part IV — Execution**

- §17 Verification suite
- §18 Pitfalls
- §19 Invariants checklist
- §20 Open questions and untested choices
- §21 Appendices
- §22 Sources

---

# Part I — The argument

## 0 Executive summary

| | MoE preset (`PRESET_VERTEX_500M`) | dense preset (`PRESET_VERTEX_500M`) |
|---|---|---|
| tokenizer | DeepSeek fast, `V=129280` (unused) | **qt-VI.1.3-32k, `V=32000`** |
| `tie_word_embeddings` | false | **true** |
| `d` / `L` | 512 / 16 | **1408 / 16** (8+8) |
| `H` × `hd` | 8 × 64 (= `d`) | **22 × 128 (= 2d)** |
| FFN | 32 routed (top-2) + 1 shared, `MoEI=416` | **dense SwiGLU, `MoEI=3712`** |
| `engram_head_dim` | 64 | **128** |
| DSpark stages | 2 | **1** |
| backbone | 414.4M | 375.7M |
| engram | 40.4M | 87.7M |
| dspark | 32.1M | 26.7M |
| **total** | **486.9M** | **490.1M** |
| **active / token** | **40.1M (8.2%)** | **329.5M (67.2%)** |
| fwd+bwd FLOPs / token | 0.241 GFLOP | **1.977 GFLOP (8.2×)** |
| Chinchilla tokens (20/param) | 9.7B | 9.8B |
| GPU-bound est., 8×H100 @35% MFU | 0.7 h | **1.9 h** |
| embed share | 13.8% | **9.2%** (tied) |
| ffn : attn : hc per layer | 97.3% : 2.3% : 0.5% | 76.1% : 22.5% : 1.3% |
| eager kernel launches / forward | ~4,400 | **~3,510** |
| prefill / decode active | 20.1M / 40.1M | 164.8M / 329.5M (1:2 preserved) |

**Deleted:** the `Gate`, `sqrtsoftplus` scoring, `noaux_tc` bias-corrected top-k, per-modality
`bias_vl`, the expert dispatch/combine loop, aux-loss-free load balancing, and ten config
keys. **Untouched:** hyper-connections, the shared-KV latent attention core, `sparse_attn`
and `attn_sink`, the entire CSA2 stack, partial RoPE with YaRN and compressed theta, the CED
split, Engram's internals, DSpark's structure, the vision tower, all KV-cache quantization,
and the serving architecture.

**Global KV cache:** 144 B/token → **4.7 MB** for a full 32K prompt, plus a fixed 256 KiB of
window rings.

---

## 1 Why dense at 500M: the evidence, for and against

### 1.1 Evidence that dense wins at or below this scale

**DeepSeek's own upper bound.** The DeepSeekMoE paper (arXiv:2401.06066, the paper the V4.1
FFN is built from) states in its abstract:

> "DeepSeekMoE 2B **nearly approaches the performance of its dense counterpart with the same
> number of total parameters**, which sets the strict upper bound of MoE language models."

At 2B total with 0.3B activated — a *more* favourable sparsity ratio than the MoE preset's
10:1 — MoE merely ties a matched-total dense model. There is no published evidence that it
wins below that.

**Controlled matched-total comparison.** arXiv:2605.13769 (3 seeds, TinyStories, sub-25M):

| arm | val loss | PPL |
|---|---|---|
| dense, **total**-matched | **1.5608 ± 0.0025** | **4.763 ± 0.012** |
| MoE (top-2, z-loss) | 1.5788 ± 0.0020 | 4.849 ± 0.010 |
| dense, active-matched | 1.6545 ± 0.0012 | 5.231 ± 0.006 |

Dense wins matched-total by **0.018 nats**. The paper notes the gap "narrows sharply" with
training but never inverts, and concludes:

> "In this sub-25M-parameter regime, MoE therefore improves validation loss under
> active-parameter matching but **does not surpass dense training at equal total stored
> capacity**."

**Smallest tasks favour dense.** ST-MoE (arXiv:2202.08906):

> "The only two tasks without improvement from the sparse model are **the two smallest**: CB
> with 250 training examples and WSC with 259."

**Data availability decides it.** Nakamura et al., *Optimal Sparsity of MoE Language Models*
(arXiv:2508.18672):

> "In high-compute regimes, the optimal density depends jointly by compute and data: **with
> limited data, denser MoE layers preserve reasoning, while with abundant data, greater
> sparsity remains advantageous.**"

And from the Ling scaling-law study (arXiv:2507.17702):

> "at any given compute budget, the optimal MoE model is computationally smaller (lower
> `M_opt`) but **trained on more data** (larger `D_opt`) than its optimal dense counterpart."

### 1.2 Evidence the other way, stated honestly

**Meta's CoSMoEs (ALVR 2026) reports MoE beating dense at 125M–350M** — right at this scale
— by 2.35% on language modelling. They align on *both* active and total parameters, and
normalize data and recipes.

**Ling scaling laws (arXiv:2507.17702)** fit MoE as more compute-efficient at *all* evaluated
sizes, including a 0.85B-active pilot.

Every paper in §1.1 compares against a dense model of matched **total** parameters. CoSMoEs
explicitly tunes both arms for on-device latency, and its dense baselines are its own — a
genuinely fair comparison that lands the other way. The field is unsettled at 500M.

### 1.3 Why the conversion is defensible even if §1.2 is right

The binding constraints are **training throughput on one node** (§4.6) and quality. MoE at
matched parameter count is not cheaper on either:

- **Memory: identical.** 486.9M and 490.1M parameters cost the same order of VRAM and the
  same checkpoint size. MoE saves *compute*, not memory.
- **Training systems: strictly harder.** Expert-parallel sharding, all-to-all dispatch and
  combine, a router GEMM and top-k per layer, and aux-loss-free load balancing — for a model
  whose gradients are 16× noisier per step (§2.1).
- **Regime: ~10× below the crossover on both axes at once.** DeepSeek's crossover is ≥2B
  parameters on ≥100B tokens. This is 0.49B on 9.8B.

The honest cost statement is §2.2: **dense at matched parameter count costs 8.2× the FLOPs
per token.** If the constraint were FLOPs rather than parameters, the right move would be a
~163M dense model (§4.5, candidate M), not a 490M one.

---

## 2 The load-bearing arithmetic

### 2.1 What the MoE preset actually was

```
PRESET_VERTEX_500M as written (V4.1 spec §3.6)

  stored parameters                486.9M
  active parameters / token         40.1M        (8.2% of stored)
  fwd+bwd FLOPs / token            0.241 GFLOP

  per layer  21,693,758 = ffn 21,102,656 (97.3%) | attn 491,720 (2.3%) | hc 98,358 (0.5%)
  routing    32 experts, top-2 of 32, plus 1 always-on shared
```

Each token sends a gradient into **3 of 33** FFN weight matrices. The consequence that
matters is *not* that the experts are under-updated:

```
at 9.7B tokens
  MoE   each expert receives 606M token-updates over a 0.64M-parameter matrix
        = 949 updates per parameter -- ample by any measure
  dense each FFN weight receives 9.7B updates over a 15.68M-parameter matrix
        = 619 updates per parameter
```

The problem is **per-step gradient batch size**. At a global batch of `B` tokens, a routed
expert sees `B × 2/32 = B/16` of them each step, while a dense FFN weight sees all `B`.
Every expert's gradient is estimated from 16× fewer samples, and from a *different* 16th of
the distribution at every step. This is the "sparse backward update" problem identified by
DefaultMoE (NeurIPS 2025) as a primary source of MoE training instability and suboptimality
— and it is the entire reason the aux-loss-free balancing machinery of spec §19.3 exists.

So the MoE preset's problem is not a starved parameter bank; it is a **noisily-batched** one,
carrying a full normalization, routing, dispatch, combine and load-balancing bill for it.

### 2.2 What densifying buys and costs

```
                          MoE preset    vertex500m     ratio
  stored parameters           486.9M        490.1M      1.01x
  active / token              40.1M        329.5M      8.21x
  fwd+bwd FLOPs / token       0.241         1.977      8.20x
  FFN share of layer          97.3%         76.1%
  attention share of layer     2.3%         22.5%
  embed share                13.8%          9.2%      (tied)
  bf16 w + grad + Muon mom    3.89 GB       3.92 GB      1.01x
```

Both fit on a single node:

```
  MoE   @ 9.7B tokens:  2,336 PFLOP -> 0.7 h on 8xH100 @35% MFU
  dense @ 9.8B tokens: 19,378 PFLOP -> 1.9 h on 8xH100 @35% MFU
  dense @50B tokens:  98,867 PFLOP -> 9.9 h on 8xH100 @35% MFU
```

Wall clock is not the binding constraint either way — even 50B tokens is under a day on
8×H100. What decides it is **which regime each architecture trains in**, and that 8.2× more
FLOPs buys 8.2× more gradient signal per step.

### 2.3 Why parameter count is a memory number, not a quality one

```
                                     MoE preset   vertex500m
  stored params                          486.9M       490.1M
  bf16 weights + bf16 grads + Muon        3.89 GB      3.92 GB   <-- ~identical
  fwd+bwd FLOPs / token                   0.241        1.977     <-- 8.2x apart
```

**MoE saves compute, not memory.** At a matched parameter count — which is what "a 500M
model" means — MoE and dense cost the same VRAM and the same checkpoint file. MoE only buys
something if you let the parameter count rise while holding compute fixed, which is what
DeepSeek does (552B stored, 16B active) and what the MoE preset did *not* do (487M stored,
40M active: it took the memory without the compute).

The comparison this document makes is therefore **"at a fixed ~0.5B parameter budget, dense
or MoE?"** — not "is 0.5B the right size?"

### 2.4 The forks, and how they were resolved

| fork | resolution | effect |
|---|---|---|
| 500M = stored params, or compute-equivalent? | **stored params** | `d=1408, L=16` (§4.6) |
| Engram share: current table size, or ~17%? | **keep ~17%** | `enghd` 64 → **128**; Engram 40.4M → 87.7M |
| `swiglu_limit` clamp in the dense FFN? | **keep it** | clamped SwiGLU survives verbatim (§6.2) |
| which tokenizer? | **qt-VI.1.3-32k** (`V=32000`) | not the spec's 129,280 (§16.4) |
| tie embeddings? | **yes** | embed 83.9M → 45.1M at the same vocab (§2.5) |
| image role tokens | **1 slot**, not 4 | delimiters are learned `[d]` params, not vocab (§16.4) |
| shape: deep-narrow or wide-shallow? | **wide-shallow**, `L/d = 0.0114` | §4.6 |
| DSpark stages | **1** | dspark 32.1M → 26.7M, 5.4% of the model |

### 2.5 Tying buys the vocab back

Tying halves the embed cost, the single largest fixed cost in a small model:

```
  V=32768 untied -> embed+head 83.9M (17.0%)   [the naive choice]
  V=32000 tied   -> embed      45.1M ( 9.2%)   [a real tokenizer for the same cost]
```

9.2% is where tied sub-1B models sit (Pythia-1B, OLMo-2-1B: ~8.7% tied). Tying is mainstream
at this scale and is why the vocab was affordable. The price is a real deviation from the
released config (`tie_word_embeddings: false`) and a known constraint: input and output
embeddings become the same matrix. On a corpus with a very skewed input distribution (heavy
code indentation, whitespace) that is worth an ablation (§20.5).

---

## 3 The survival map

### 3.1 Untouched — do not touch, do not re-derive

| spec section | what | why it is unaffected |
|---|---|---|
| §5 | Hyper-Connections (mHC), Single-Pass hand-off, Sinkhorn split | No dependency on routing. Only `d` changes, so the per-layer HC cost scales. See §4.6 and §21.4 for the execution-model caveats. |
| §7 | Attention core: `wq_a`/`q_norm`/`wq_b`, single shared KV latent, `kv_norm`, grouped `wo_a`/`wo_b`, `attn_sink`, partial RoPE tail, inverse RoPE | Only widths change. The "no value projection" invariant and the sink-in-denominator rule are untouched. |
| §7.4 | `sparse_attn` exact semantics, `-1e30` max floor, all-masked rows → exactly zero | Pure function of `(q, kv, idxs, sink)`. |
| §8 | CSA2 in full: Compressor + softmax gate, cross-layer Full/Reindex/Reuse, Indexer (`relu(q·k)` head-weighted), Hierarchical Sparse Indexer with candidate blocks | The indexer is a *side attention*, not an MoE. It has its own `wq_b`/`weights_proj` and shares the low-rank `qr` — none of which is the FFN router. |
| §9 | Partial RoPE, YaRN, `compress_rope_theta`, per-layer rope selection | Unaffected. |
| §10 | CED: 8+8 split, projected global KV, Decoder SWA Bounded Replay | Unaffected — and the 1:2 prefill/decode ratio is structural (§12). |
| §11 | Engram: prime layout, `NgramHashState`, gated additive write | Unaffected except that `wkv`'s input width grows with `enghd`. |
| §12 | DSpark structure: block drafting, Markov head, confidence head, `DSparkAttention` | Only the stage count and the drafter's FFN change (§14). |
| §13 | ViT, Aligner, image merge, 2D-RoPE | Unaffected. |
| §14 | All KV-cache quantization: FP8 E4M3 window, FP4 E2M1 compressed, FP4 indexer, UE8M0/E4M3 scales, `QUANT_MODE` switch | These are *cache* formats. Only the *weight* formats change (§10). |
| §15 | Serving: KV budget, bounded replay, EPD disaggregation, kernel counts | Recomputed for the new geometry in §4.4 but structurally unchanged. |

### 3.2 Deleted outright

| spec section | content | reason |
|---|---|---|
| §6.1 | `Gate` — `sqrtsoftplus` scoring, `noaux_tc` selection, `norm_topk_prob`, `routed_scaling_factor`, per-modality bias selection | A router with one destination has no decision to make. |
| §6.3 | Per-expert dispatch/combine: the `bincount` / `torch.where(idx == e)` gather-scatter loop | No experts to gather. Also removes a Python loop of `n_experts` iterations **per layer per forward** — 512 iterations at 16 layers, the single biggest eager-PyTorch win in the conversion. |
| §13.4 | "Multimodal routing" — entirely about `bias_vl` | See §15. |
| §19.3 | MoE load balancing: auxiliary-loss-free bias updates, separate text/image balance, sequence-level balance loss (weight 1e-4) | No routing, no balance to maintain. |
| §16.1 | `ffn.gate.{weight,bias,bias_vl}`, `ffn.experts.{e}.{w1,w2,w3}`, `ffn.experts.{e}.*.scale`, `ffn.shared_experts.*` | Replaced by three tensors (§9.1). |
| §25 #20, #21 | Gate invariants | No gate. |

**Config keys deleted (10):**

```
n_routed_experts         n_shared_experts          num_experts_per_tok
scoring_func             topk_method               norm_topk_prob
routed_scaling_factor    dspark_n_routed_experts   dspark_num_experts_per_tok
```

`moe_intermediate_size` is **kept as a key name** even though the block is no longer MoE — it
is still the FFN intermediate width, used by backbone and DSpark. Renaming it to
`ffn_intermediate_size` is cleaner and worth doing only if you are not round-tripping
checkpoints (§7.4).

### 3.3 Rewritten

| spec section | change |
|---|---|
| §1.1 | "Feed-forward" row; "Activated parameters per token"; tokenizer/`tie_word_embeddings` rows |
| §2.1, §2.3 | Parameter tables: delete `routed experts` and `gate` rows |
| §2.2 | 8B/16B → 164.8M/329.5M. Ratio preserved. |
| §3.1 | Delete symbol rows `E` (routed experts) and `k` (experts per token) |
| §3.3 | Delete the "MoE gate scores" fp32 dtype row; keep "SwiGLU gate/up products fp32" |
| §3.4 | Delete 10 keys |
| §3.5, §3.6 | All geometry rows |
| §6 | **Replaced in full by §6 of this document** |
| §12.2 | "each has its own final `norm`, MoE (128 experts, top-3)" → dense FFN, 1 stage |
| §14.1 | "routed expert weights FP4 E2M1" row deleted; the FFN becomes FP8 E4M3 |
| §16.1, §16.2 | Checkpoint layout and disk composition (§9) |
| §17.1 | `dsv41/` → `vertex500m/`; `moe.py` → `ffn.py` |
| §20.1 | Counting script: routed/gate terms optional (§5.1) |
| §20.2 | Activation counts: `3·d·MoEI·(1+6)` → `3·d·MoEI` |
| §23 | **Scaling-down guide rewritten from scratch** — §16 |
| §25 #22 | SwiGLU clamps stay, relabelled from an MoE invariant to a dense one |
| §25 #37 | Gradient contract: `gate.bias` no-grad clause deleted; replaced by a no-dispatch invariant (§19) |

### 3.4 New risks introduced by this conversion

1. **Attention is starved in absolute terms relative to its share.** It is 22.5% of a layer
   here versus 2.3% in the MoE preset, but that share is the *result* of a fixed recipe.
   `wkv` is 0.18M and `index_head_dim` is 64. See §20.1.
2. **`H·hd = 2d` is a judgement call.** Released: `6.4d` at 1M context; MoE preset: `1d`.
   `2d` is a compromise (§16.2, §20.1).
3. **The FFN is now the largest activation source**, so `swiglu_limit` goes from a numerics
   nicety to a load-bearing clamp (§10.3).
4. **The tokenizer is multilingual and NFKC-hostile.** qt-VI.1.3 covers 204 languages; folding
   them into an Engram compressed map will collide (§16.4, §18).
5. **Embeddings are tied** — a deviation from the released config and a real constraint on
   input/output specialization (§2.5, §20.5).
6. **The `torch.compile` fusion assumption is unmeasured.** The entire shape argument in §4.6
   rests on the premise that inductor fuses the mHC Sinkhorn loop but *not* `topk`, `sort`,
   `gather`, or the scattered compressor write. That is a reasoned expectation, not a
   measurement. §17.8 exists to measure it.
7. **Activation memory at long context.** 924 KB/token is 3.78 GB at `T=4096` but **15.1 GB at
   `T=16384`** without checkpointing. Gradient checkpointing is not optional at long context.

---

## 4 `PRESET_VERTEX_500M`: the configuration

### 4.1 Full configuration

```json
{
  "architectures": ["Vertex500MForCausalLM"],
  "model_type": "vertex500m",
  "dtype": "bfloat16",
  "tokenizer": "JamesQuartz/qt-VI.1.3-32k",

  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "image_token_id": 85,

  "text_config": {
    "model_type": "vertex500m_text",
    "vocab_size": 32000,
    "tie_word_embeddings": true,

    "hidden_size": 1408,
    "num_hidden_layers": 16,
    "num_attention_heads": 22,
    "num_key_value_heads": 1,
    "head_dim": 128,
    "qk_rope_head_dim": 32,
    "q_lora_rank": 352,
    "o_lora_rank": 352,
    "o_groups": 4,

    "moe_intermediate_size": 3712,
    "hidden_act": "silu",
    "swiglu_limit": 10.0,
    "rms_norm_eps": 1e-20,
    "attention_bias": false,
    "attention_dropout": 0.0,
    "initializer_range": 0.02,
    "use_cache": true,

    "max_position_embeddings": 32768,
    "rope_theta": 10000,
    "rope_scaling": {
      "rope_type": "yarn", "factor": 16, "beta_fast": 32, "beta_slow": 1,
      "original_max_position_embeddings": 8192
    },

    "sliding_window": 128,
    "compress_ratios": [0,0, 2,2,2,2,2,2, 1,1,1,1,1,1,1,1, 0],
    "compress_rope_theta": 160000,
    "kv_source_layer_ids": [2, 8],
    "index_source_layer_ids": [2, 8, 12],
    "candidate_source_layer_id": 8,
    "candidate_topk_blocks": 512,
    "candidate_block_size": 8,
    "index_n_heads": 8,
    "index_head_dim": 64,
    "index_topk": 512,

    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,

    "engram_layer_ids": [1],
    "engram_num_embeddings": [600766],
    "engram_max_ngram_size": 4,
    "engram_vocab_size": 50021,
    "engram_n_heads": 4,
    "engram_head_dim": 128,
    "engram_pad_token_id": 0,
    "engram_compressed_vocab_size": 0,

    "num_nextn_predict_layers": 1,
    "dspark_block_size": 4,
    "dspark_noise_token_id": 3,
    "dspark_target_layer_ids": [15],
    "dspark_markov_rank": 64,

    "vision_n_layers": 0
  },

  "quantization_config": {
    "quant_method": "fp8", "activation_scheme": "dynamic",
    "weight_block_size": [32, 32], "scale_fmt": "ue8m0"
  }
}
```

`engram_num_embeddings = [600766]` is the sum of the 12 primes drawn upward from 50021 for
`(3 n-gram sizes × 4 heads)`:

```
50021 50023 50033 50047 50051 50053 50069 50077 50087 50093 50101 50111  = 600766
```

Verify at build time and store the sum — spec invariant 30.

### 4.2 Derived geometry

```
H · hd            = 22 · 128 = 2816  =  2d
q  path:  wq_a 1408→352 ; q_norm(352) ; wq_b 352→2816
kv path:  wkv 1408→128 ; kv_norm(128)          # single latent, K == V, no value projection
out path: wo_a (2816/4=704) → (4·352=1408), block-diagonal over 4 groups ; wo_b 1408→1408
RoPE:    last qk_rope_head_dim = 32 channels of each 128-dim head  (96 non-rotary)
softmax_scale = hd ** -0.5 = 2 ** -3.5

MoEI = round_to_nearest_128(8/3 · d) = round_to_nearest_128(3754.67) = 3712
```

**The `MoEI` rule is round-to-nearest, not round-up.** `3712` is *below* `8/3·d = 3754.67`
(29.33 → 29 → 3712). Both the previous preset (3456 vs 3413.3) and this one land below the
raw ratio, because the nearest 128-multiple can fall either side. Implement it as
`round(8*d/3 / 128) * 128` and do not write "round up" anywhere.

### 4.3 Budget

| component | params | share |
|---|---|---|
| embed (**tied**, shared with LM head) | 45,056,000 | 9.19% |
| 16 × layer stack | 329,503,424 | 67.23% |
| 3 indexers (`wq_b` + `weights_proj`) | 574,464 | 0.117% |
| 2 index-key projections (`wk` + `k_norm`) | 16,512 | 0.003% |
| 2 compressors | 540,928 | 0.110% |
| **backbone** | **375,691,328** | **76.66%** |
| Engram (1 layer) | 87,722,752 | 17.90% |
| DSpark (1 stage) | 26,675,309 | 5.44% |
| **total** | **490,089,389** | 100% |

Per layer (20,593,964):

| component | shape | params | share |
|---|---|---|---|
| dense FFN `w1` | [3712, 1408] | 5,226,496 | 25.38% |
| dense FFN `w2` | [1408, 3712] | 5,226,496 | 25.38% |
| dense FFN `w3` | [3712, 1408] | 5,226,496 | 25.38% |
| **FFN subtotal** | | **15,679,488** | **76.13%** |
| `attn.wo_b` | [1408, 1408] | 1,982,464 | 9.63% |
| `attn.wo_a` | [1408, 704] | 991,232 | 4.81% |
| `attn.wq_b` | [2816, 352] | 991,232 | 4.81% |
| `attn.wq_a` | [352, 1408] | 495,616 | 2.41% |
| `attn.wkv` | [128, 1408] | 180,224 | 0.88% |
| `q_norm` + `kv_norm` + `attn_sink` | | 502 | 0.002% |
| **attention subtotal** | | **4,641,270** | **22.54%** |
| hyper-connections (attn + ffn) | 2 × ([24,5632]+[24]+[3]) | 270,390 | 1.31% |
| `attn_norm` + `ffn_norm` | 2 × [1408] | 2,816 | 0.01% |

Attention's largest single tensor is `wo_b` at 42.7% of the attention block. The grouped
low-rank output projection (`wo_a` → `wo_b`) is where the width went when `d` grew — the
released V4.1 structure faithfully scaled, not a new design.

Engram: table `600766×128 = 76,898,048` + `wkv [7040, 1536] = 10,813,440` +
`q/k_weight = 11,264`.

DSpark: `1 × layer = 20,593,964` + `main_proj` and norm `= 1,983,872` +
`Markov (2·V·64) = 4,096,000` + `confidence = 1,473`.

### 4.4 KV cache

| tensor | format | bytes/token | grows? |
|---|---|---|---|
| compressed KV, layer 2 (`ratio=2`) | FP4 E2M1 blk 16 | 128 × 0.5 / 2 = **32** | yes |
| compressed KV, layer 8 (`ratio=1`) | FP4 E2M1 blk 16 | 128 × 0.5 = **64** | yes |
| indexer K, layer 2 (`ratio=2`) | FP4 E2M1 blk 32 | 64 × 0.5 / 2 = **16** | yes |
| indexer K, layer 8 (`ratio=1`) | FP4 E2M1 blk 32 | 64 × 0.5 = **32** | yes |
| **global total (HBM-resident)** | | **144 B/token** | yes |
| window rings, 16 × 128 slots | FP8 E4M3 | 16·128·128 = **256 KiB per sequence** | **no** |

At `max_seq_len = 32768`: **4.72 MB** of global KV for a full-length prompt plus a fixed
256 KiB of window rings. The published 890 B/token is a property of the 552B model's four
compressors and four indexer-key owners; at this size it is 144.

### 4.5 Alternatives considered

All at `hd=128`, `q_lora = o_lora = d/4`, `o_groups=4`, `engram_head_dim=128`,
`index_n_heads=8`, `index_head_dim=64`, **1 DSpark stage**, **tied embeddings**,
`V=32000`. Launch counts are the eager estimate from §4.6 (~214/layer).

| # | `d` | `L` | `H` | `MoEI` | total | active/tok | GFLOP | eager launches | act KB/tok | L/d |
|---|---|---|---|---|---|---|---|---|---|---|
| **A** | 1408 | 16 | 22 | 3712 | **490.1M** | 329.5M | **1.977** | 3,509 | 924 | 0.0114 |
| B | 1280 | 20 | 20 | 3456 | 499.2M | 347.4M | 2.084 | 4,365 | 1,060 | 0.0156 |
| C | 1536 | 14 | 24 | 4096 | 515.7M | 345.5M | 2.073 | 3,081 | 886 | 0.0091 |
| D | 1280 | 16 | 20 | 3456 | 429.8M | 277.9M | 1.668 | 3,509 | 848 | 0.0125 |
| E | 1792 | 12 | 28 | 4736 | 589.2M | 399.2M | 2.395 | 2,653 | 882 | 0.0067 |
| **M** | 512 | 20 | 8 | 1408 | **163.3M** | 58.4M | 0.350 | 4,365 | 430 | 0.0391 |

A is chosen because it is the **FLOP minimum at the ~0.5B target**, which is what matters
once you are GPU-bound (§4.6), and because 16 layers still gives a comfortable 8+8 CED split
with 6 encoder `ratio=2` layers and 14 Reuse layers for the CSA2 hierarchy to show up in.

Candidate **M** is the *compute-matched* arm: same FLOPs/token as the MoE preset, delivered
densely. It is the honest ablation arm if a matched-FLOP dense-vs-MoE comparison is wanted —
but it is not a 0.5B model.

**A caution about citing small-model shapes.** Qwen3-0.6B (28L/d=1024, L/d = 0.027),
SmolLM2-360M (32L/960, 0.033) and Qwen2.5-0.5B (24L/896, 0.027) are all deep-and-narrow — but
all three are *compiled and fused* implementations. Their L/d ratios do **not** transfer to a
`torch.compile`-without-graphs layer loop. See §4.6.

### 4.6 Execution model and what it optimizes

**This section is why the shape is `d=1408, L=16` and not something deeper.** It was absent
from earlier drafts of this document, and the config was originally `d=1280, L=20` on the
(incorrect, for this execution model) reasoning that sub-1B models run deep-and-narrow.

**The target: `torch.compile` on each `Block`, no CUDA graphs.**

Two consequences pull in opposite directions, and both are needed to pick a shape.

**1. Kernel launches scale with `L`, not with `d`.**

The dominant per-layer cost in eager PyTorch is `hc_split_sinkhorn`: 19 iterations of
`comb / (comb.sum(-1) + eps)` on a `[B, S, 4, 4]` fp32 tensor. That tensor is **the same size
regardless of hidden size**, so the launch count is constant in `d`:

```
  eager, per layer                                   compiled, per layer
  --------------------------------------------------------------
  mHC Sinkhorn      ~166   (constant in d)           ~1-3    (fused)
  attention          ~30                             ~15-25  (partially fused)
  hc_pre/hc_post      8                             ~4
  norms               4                             ~2
  FFN                10                             ~6      (fused)
  --------------------------------------------------------------
  total            ~214                             ~30-40  irreducible floor
      ^ 76% of eager launches is mHC                ^ dominated by topk/sort/gather
```

**2. What inductor cannot fuse is data-dependent work.**

| survives compilation | why | per layer |
|---|---|---|
| indexer `topk` + `sort` | output count is data-dependent | 2–3 |
| `sparse_attn` gather | index tensor is data-dependent | 2–4 |
| compressor `pool_state` scattered write | index is data-dependent, ratio-branched | 3–5 |
| compressor `ratio > 1` branch | control flow, not a tensor op | 1–2 |
| window ring slot writes | index is data-dependent | 1–2 |

That ~30–40/layer floor is irreducible and **is** what `L` multiplies.

**Measured effect on the two candidates** (per 8192-token micro-batch, 8×H100 @35% MFU):

```
                        eager launches   compiled launches   GPU ms
  d=1280 L=20                4365             ~700              6.16
  d=1408 L=16                3509             ~550              5.84
                              -20%             -21%             -5%
```

Eager, both are CPU-bound (≈30 ms and ≈25 ms of launch time against ≈6 ms of GPU work).
Compiled, both are GPU-bound — the surviving launch overhead falls below the FLOP time — so
**FLOPs become the objective, and `d=1408 L=16` is the FLOP minimum at the 0.5B target.**

**3. Wide-and-shallow also wins on memory, which is counter-intuitive.**

Total activation is proportional to `L · d`. In a parameter-matched trade, `L` falls faster
than `d` rises:

```
  d=1280 L=20   ->  1060 KB/token   (4.34 GB @ T=4096)
  d=1408 L=16   ->   924 KB/token   (3.78 GB @ T=4096)
  d=1792 L=12   ->   882 KB/token   (3.61 GB @ T=4096)
```

At `T=16384` without gradient checkpointing that is 15.1 GB for `d=1408 L=16` and 16.9 GB for
`d=1280 L=20`. Checkpointing is required at long context either way (§18.13).

**4. The layer loop itself is not compiled away.**

`torch.compile` on `Block` compiles each block; the `for layer in self.layers` loop still
runs in Python, so `L` still multiplies graph-entry cost even after fusion. This is the floor
that makes `L=16` meaningfully better than `L=20` even under compilation.

**5. This whole section is an assumption until measured.** §17.8 exists to measure it.

---

## 5 Parameter accounting

### 5.1 The counting script

A single script handling **both** the MoE preset and the dense preset, so §5.2 is a real
regression test rather than a claim.

**Note the `hcp` line.** The spec's formula is
`2·((2+hc)·hc·(hc·d) + (2+hc)·hc + 3)`, which reduces to `2·(96d + 27)` at `hc=4`. A careless
transcription — `2·(24·4·(4·d) + 27)` — is 4× too large, and was an actual error during the
derivation of this document.

```python
def primes_after(start, seen):
    """Next prime strictly greater than `start`, not already in `seen`."""
    c = start + 1
    while True:
        d, n = 2, c
        while d * d <= n and n % d:
            d += 1
        if d * d > n and c not in seen:
            return c
        c += 1


def engram_rows(base, max_ngram, n_heads, n_layers):
    """Rows per layer = sum of its (max_ngram-1) * n_heads unique primes."""
    seen, per = set(), []
    for _ in range(n_layers):
        s = 0
        for _ in range(max_ngram - 1):        # max_ngram-1 sizes => the 2/3/4-grams
            cur = base - 1
            for _ in range(n_heads):
                cur = primes_after(cur, seen)
                seen.add(cur)
                s += cur
        per.append(s)
    return per


def count(c):
    """c is a config dict. Handles dense and MoE. Returns exact integer counts."""
    d, L, H, hd, V = c['dim'], c['n_layers'], c['n_heads'], c['head_dim'], c['vocab']
    ql, og, ol, hc = c['q_lora_rank'], c['o_groups'], c['o_lora_rank'], c['hc_mult']
    MoEI, ratios = c['ffn_intermediate_size'], c['compress_ratios']
    kv, isrc = c['kv_source'], c['index_src']
    ih, idd = c['index_n_heads'], c['index_head_dim']

    # --- per layer ---------------------------------------------------------
    attn = d*ql + ql + ql*H*hd + d*hd + hd + (H*hd//og)*(og*ol) + og*ol*d + H
    hcp  = 2*((2+hc)*hc*(hc*d) + (2+hc)*hc + 3)      # == 2*(96d + 27) at hc=4
    ffn  = 3*d*MoEI                                   # dense FFN, or the shared expert

    E   = c.get('n_routed_experts', 0)
    K   = c.get('num_experts_per_tok', 0)
    S   = c.get('n_shared_experts', 1)
    dsE = c.get('dspark_experts', 0)
    if E:
        ffn += E*3*d*MoEI + E*d + 2*E                  # routed bank + gate weight + biases

    lay     = attn + ffn + hcp + 2*d                  # stored per layer
    lay_act = attn + ((K + S)*3*d*MoEI if E else ffn) + hcp + 2*d
    lay_ds  = attn + 3*d*MoEI + hcp + 2*d + (dsE*3*d*MoEI if dsE else 0)

    # --- whole model -------------------------------------------------------
    backbone = V*d*(1 if c['tied'] else 2) + L*lay
    backbone += len(isrc)*(ih*idd*ql + ih*d)           # indexer wq_b + weights_proj
    backbone += len(kv)*(hd*idd + idd)                # indexer wk + k_norm
    backbone += sum(d*hd + hd + (d*hd if ratios[l] > 1 else 0) for l in kv)   # compressors

    engram = 0
    for r in engram_rows(c['engram_base'], c['engram_max_ngram'],
                         c['engram_n_heads'], len(c['engram_layers'])):
        engram += (r*c['engram_head_dim']
                   + (c['engram_max_ngram'] - 1)*c['engram_n_heads']*c['engram_head_dim']
                     *d*(hc + 1)
                   + 2*hc*d)

    ns = c['dspark_stages']
    dspark = ns*lay_ds + d*len(c['dspark_targets'])*d + d \
           + V*c['dspark_markov_rank']*2 + (d + c['dspark_markov_rank']) + 1

    return dict(backbone=backbone, engram=engram, dspark=dspark,
                total=backbone + engram + dspark,
                lay=lay, attn=attn, ffn=ffn, hcp=hcp, active=L*lay_act,
                embed=V*d*(1 if c['tied'] else 2))
```

### 5.2 Validation — the script must reproduce the MoE preset first

```
>>> count(MOE_PRESET)          # V4.1 spec §3.6, untied, V=65536, 2 DSpark stages
backbone  414,366,880   (spec §2: 414.4M)   PASS
engram     40,419,200   (spec §2:  40.4M)   PASS
dspark     32,083,517   (spec §2:  31.9M)   PASS
total     486,869,597   (486.9M)
active/token  40,128,480  (8.2% of stored)   0.241 GFLOP
per layer  21,693,758 = ffn 21,102,656 (97.3%) | attn 491,720 (2.3%) | hc 98,358 (0.5%)

>>> count(PRESET_VERTEX_500M)
backbone  375,691,328
engram     87,722,752
dspark     26,675,309
total     490,089,389   (490.09M)
active/token 329,503,424  (67.2% of stored)  1.977 GFLOP
per layer  20,593,964 = ffn 15,679,488 (76.1%) | attn 4,641,270 (22.5%) | hc 270,390 (1.3%)
embed (tied)  45,056,000  (9.19%)
```

If the MoE preset does not reproduce, the script is wrong and every dense number derived from
it is wrong. That is the point of keeping both paths in one function.

### 5.3 Where the 486.9M went

The conversion is a reallocation, not a growth:

```
                          MoE preset      vertex500m      delta
  routed experts (16L)     326.4M              0.0M    -326.4M   deleted
  shared expert (16L)      10.2M              0.0M     -10.2M   absorbed
  dense FFN (16L)            0.0M           250.9M    +250.9M   new
  attention (16L, retuned)  7.9M            74.3M     +66.4M   d 512->1408
  hyper-connections         1.6M             4.3M      +2.8M   d 512->1408
  norms                      0.0M             0.05M     +0.03M
  embed + head              67.1M            45.1M     -22.1M   V 65536->32000, TIED
  indexers + compressors     0.15M            1.13M     +0.98M
  ENGRAM                    40.4M            87.7M     +47.3M   enghd 64->128
  DSPARK                    32.1M            26.7M      -5.4M   2 stages -> 1
  ------------------------------------------------------------------------------
  TOTAL                   486.9M           490.1M      +3.2M
```

Embeddings *shrank* by 22.1M while the vocab dropped only 49% — tying did more work than the
vocab reduction. Engram grew fastest of all: a conditional n-gram memory is a *lookup*, not
compute, so its share costs nothing in FLOPs, and it is arguably the one component of V4.1
that does not shrink when you go dense. DSpark shrank because one stage instead of two.

---

## 6 Dense FFN — replacement text for spec §6

### 6.1 Definition

Every block contains exactly one feed-forward sublayer: a **clamped SwiGLU** with
intermediate width `moe_intermediate_size`. It is the *shared expert* of the released
architecture, promoted to the whole FFN, widened from 2304 to 3712, and stripped of the
router.

```
DenseFFN.forward(x):
    gate = w1(x).float()          # [*, MoEI]
    up   = w3(x).float()          # [*, MoEI]
    y    = silu(clamp(gate)) * clamp(up, -limit, limit)
    return w2(y).to(x.dtype)      # [*, d]
```

There is no `Gate` module, no `bias`, no `bias_vl`, no routing weight, no
`routed_scaling_factor`, and no dispatch. The output is a single matmul chain over the full
token batch.

### 6.2 Clamped SwiGLU — unchanged, and now load-bearing

The `Expert` class carries over **verbatim** from spec §6.2:

```
gate = w1(x)          # [n, MoEI]
up   = w3(x)          # [n, MoEI]
if swiglu_limit > 0:                  # limit = 10.0
    up   = clamp(up,   -limit, limit) # two-sided
    gate = clamp(gate, -inf,  limit) # upper-only
y = silu(gate) * up                   # fp32
out = w2(y)
```

**Why it survives dense conversion:**

- The clamp exists to keep activations inside the FP8 E4M3 / FP4 E2M1 dynamic range so the
  fake-quant round-trip in §14 does not clip. That argument is about *activation magnitude*,
  not routing, and does not weaken when the FFN becomes dense.
- At `MoEI=3712` the FFN intermediate is the single largest activation tensor in the network
  — 8.9× wider than the MoE preset's. If anything the clamp matters more.
- Keeping it means the model lives in the same numerics family as the released one, so the
  `QUANT_MODE` switch and the QAT schedule in §14.5 carry over unchanged.
- Cost: two elementwise ops on a `[B, S, 3712]` tensor. Immaterial next to three GEMMs.

### 6.3 Parameter shapes

```
ffn.w1.weight    [MoEI, d] = [3712, 1408]      # gate
ffn.w3.weight    [MoEI, d] = [3712, 1408]      # up
ffn.w2.weight    [d, MoEI] = [1408, 3712]      # down
```

`3·d·MoEI = 15,679,488` per layer. No `bias`, no `bias_vl`, no expert `ModuleList`, no
per-expert `.scale` tensors — the dense FFN is FP8 with UE8M0 block scales like every other
dense linear (§10.2).

### 6.4 What is deliberately absent

| removed | would have been |
|---|---|
| `gate.weight [E, d]` | 32×1408 = 45,056 |
| `gate.bias [E]`, `gate.bias_vl [E]` | 64 |
| `experts.{0..31}.{w1,w2,w3}` | 326.4M |
| `shared_experts.{w1,w2,w3}` | 10.2M |

Total removed: **336.6M**, of which 250.9M returns as the widened dense FFN and the remainder
funds `d` 512 → 1408 and `enghd` 64 → 128.

### 6.5 Initialization

Nothing new required. `nn.Linear` default init with `initializer_range = 0.02`, which is what
the shared expert already used.

**One genuinely new consideration.** In the MoE model `d` was small (512) because the expert
bank supplied the width; here `d = 1408` and the FFN is `d`-wide in both directions. If Muon
is used (§11.3), the FFN matrices are `[3712, 1408]` — larger and better conditioned than
anything in the MoE preset. No action needed, but do not carry over a Muon update-RMS constant
tuned on the old shapes.

---

## 7 Config delta

### 7.1 Dataclass

```python
@dataclass
class Cfg:
    # shape
    vocab_size: int; dim: int; n_layers: int; n_heads: int; head_dim: int
    q_lora_rank: int; qk_rope_head_dim: int; o_lora_rank: int; o_groups: int
    moe_intermediate_size: int                  # KEEP the key name; it is the FFN width
    tie_word_embeddings: bool = True
    # tokenizer
    bos_token_id: int = 1; eos_token_id: int = 2; pad_token_id: int = 0
    image_token_id: int = 85
    # attention schedule
    sliding_window: int
    compress_ratios: tuple; kv_source_layer_ids: tuple; index_source_layer_ids: tuple
    candidate_source_layer_id: int; candidate_topk_blocks: int; candidate_block_size: int
    index_n_heads: int; index_head_dim: int; index_topk: int
    # ~~ moe block deleted entirely (10 fields) ~~
    # hyper-connections
    hc_mult: int; hc_sinkhorn_iters: int; hc_eps: float
    # engram
    engram_layer_ids: tuple; engram_num_embeddings: tuple; engram_max_ngram_size: int
    engram_vocab_size: int; engram_n_heads: int; engram_head_dim: int
    engram_pad_token_id: int; engram_compressed_vocab_size: int
    # dspark
    num_nextn_predict_layers: int; dspark_block_size: int; dspark_noise_token_id: int
    dspark_target_layer_ids: tuple; dspark_markov_rank: int
    # ~~ dspark_n_routed_experts / dspark_num_experts_per_tok deleted ~~
    # rope / numerics / vision: unchanged
    ...

    # ~~ get_moe_config() DELETED ~~
```

### 7.2 The preset

```python
PRESET_VERTEX_500M = Cfg(
    vocab_size=32000, dim=1408, n_layers=16, n_heads=22, head_dim=128,
    q_lora_rank=352, qk_rope_head_dim=32, o_lora_rank=352, o_groups=4,
    moe_intermediate_size=3712, sliding_window=128,
    tie_word_embeddings=True,
    bos_token_id=1, eos_token_id=2, pad_token_id=0, image_token_id=85,

    compress_ratios=tuple([0, 0] + [2]*6 + [1]*8 + [0]),      # 17 == L + 1 DSpark
    kv_source_layer_ids=(2, 8), index_source_layer_ids=(2, 8, 12),
    candidate_source_layer_id=8, candidate_topk_blocks=512, candidate_block_size=8,
    index_n_heads=8, index_head_dim=64, index_topk=512,

    hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,

    engram_layer_ids=(1,), engram_num_embeddings=(600766,),
    engram_max_ngram_size=4, engram_vocab_size=50_021, engram_n_heads=4,
    engram_head_dim=128, engram_pad_token_id=0,
    engram_compressed_vocab_size=0,              # set to YOUR tokenizer's map size

    num_nextn_predict_layers=1, dspark_block_size=4, dspark_noise_token_id=3,
    dspark_target_layer_ids=(15,), dspark_markov_rank=64,

    rope_theta=10000.0, compress_rope_theta=160000.0, rope_factor=16,
    beta_fast=32, beta_slow=1, rope_original_seq_len=8192, max_seq_len=32768,
    vision_n_layers=0,
)
```

### 7.3 Keys deleted (10)

```
n_routed_experts          n_shared_experts          num_experts_per_tok
scoring_func              topk_method               norm_topk_prob
routed_scaling_factor     dspark_n_routed_experts   dspark_num_experts_per_tok
```

### 7.4 Keys kept but meaning changed

| key | change |
|---|---|
| `moe_intermediate_size` | still the FFN width; used by backbone **and** DSpark. Rename to `ffn_intermediate_size` only if not round-tripping checkpoints. |
| `tie_word_embeddings` | `false` → `true`. Removes `head.weight` from the state dict. |
| `vocab_size` | 129280 (released) / 65536 (old preset) → **32000**. |
| `bos/eos/pad/image_token_id` | qt-VI.1.3's fixed 86-slot map (§16.4), not DeepSeek's 0/1/2/129264. |
| `engram_pad_token_id` | 2 → **0** (`<|padding|>`). |
| `dspark_noise_token_id` | 1 → **3** (`<|unk|>`, never emitted). |
| `num_nextn_predict_layers` | 2 → **1**; `dspark_target_layer_ids` shrinks from 3 ids to 1. |

### 7.5 Keys unchanged in meaning

Everything else. `swiglu_limit`, `rms_norm_eps`, `hc_*`, the whole CSA2 schedule, the RoPE
block, `sliding_window`, `dspark_block_size`/`_markov_rank`, and the quantization config.

---

## 8 Reference implementation

### 8.1 `ffn.py` — replaces `moe.py` in its entirety

```python
import torch
import torch.nn.functional as F


class Expert(torch.nn.Module):
    """Clamped SwiGLU: up two-sided, gate upper-only, fp32 product. UNCHANGED from spec 6.2."""

    def __init__(self, dim, inter, limit):
        super().__init__()
        self.w1 = torch.nn.Linear(dim, inter, bias=False)      # gate
        self.w2 = torch.nn.Linear(inter, dim, bias=False)      # down
        self.w3 = torch.nn.Linear(dim, inter, bias=False)      # up
        self.limit = limit

    def forward(self, x, weights=None):
        gate, up = self.w1(x).float(), self.w3(x).float()
        if self.limit > 0:
            up = up.clamp(-self.limit, self.limit)
            gate = gate.clamp(max=self.limit)
        y = F.silu(gate) * up
        if weights is not None:                # dead here; kept for parity/future routing
            y = weights * y
        return self.w2(y.to(x.dtype))


class DenseFFN(torch.nn.Module):
    """The whole feed-forward sublayer. One SwiGLU, no router, no experts.

    Drop-in for spec 6.3's `MoE`. Backbone and DSpark stages use the identical class;
    the DSpark distinction that used to be `dspark_n_routed_experts` is now purely which
    `compress_ratios` entry and `Attention` subclass the block carries (14.3).
    """

    def __init__(self, cfg, layer_id=None):
        super().__init__()
        self.expert = Expert(cfg.dim, cfg.moe_intermediate_size, cfg.swiglu_limit)

    def forward(self, x):
        return self.expert(x)
```

`layer_id` is accepted and ignored, so `Block.__init__` needs no DSpark special case.

**The `weights` argument on `Expert.forward` is now dead** — nothing passes it. It is kept
because it costs three lines, preserves the ability to weight the FFN output, and means a
future reintroduction of routing does not require editing `Expert`.

### 8.2 `Block`

```python
class Block(torch.nn.Module):
    attention_cls = Attention

    def __init__(self, cfg, layer_id, engram_layout=None):
        super().__init__()
        self.layer_id, self.cfg = layer_id, cfg
        self.engram_idx = (engram_layout.layer_ids.index(layer_id)
                           if engram_layout and layer_id in engram_layout.layer_ids else None)
        self.attn = self.attention_cls(cfg, layer_id)
        self.ffn = DenseFFN(cfg, layer_id)                # <-- was MoE(cfg, layer_id)
        self.engram = Engram(cfg, self.engram_idx) if self.engram_idx is not None else None
        self.attn_norm = RMSNorm(cfg.dim, cfg.rms_norm_eps)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.rms_norm_eps)
        self.hc_mult = hc = cfg.hc_mult
        self.hc_sinkhorn_iters, self.hc_eps = cfg.hc_sinkhorn_iters, cfg.hc_eps
        self.norm_eps = cfg.rms_norm_eps
        mix_hc = (2 + hc) * hc
        self.hc_attn_fn = torch.nn.Parameter(torch.empty(mix_hc, hc * cfg.dim, dtype=torch.float32))
        self.hc_ffn_fn  = torch.nn.Parameter(torch.empty(mix_hc, hc * cfg.dim, dtype=torch.float32))
        self.hc_attn_base = torch.nn.Parameter(torch.zeros(mix_hc, dtype=torch.float32))
        self.hc_ffn_base  = torch.nn.Parameter(torch.zeros(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = torch.nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.hc_ffn_scale  = torch.nn.Parameter(torch.ones(3, dtype=torch.float32))

    def mixes(self, x, fn, scale, base):
        return hc_mixes(x, fn, scale, base, self.hc_mult, self.hc_sinkhorn_iters,
                        self.hc_eps, self.norm_eps)

    def forward(self, x, start_pos, pre_mix, *attn_args):          # <-- image_mask gone
        # attention: uses the PREVIOUS sublayer's pre (Single-Pass shift)
        residual = x
        attn_pre, attn_post, attn_comb = self.mixes(x, self.hc_attn_fn, self.hc_attn_scale,
                                                     self.hc_attn_base)
        x = self.attn_norm(hc_pre(x, pre_mix))
        x = self.attn(x, start_pos, *attn_args)
        x = hc_post(x, residual, attn_post, attn_comb)

        # FFN: uses THIS block's attention pre
        residual = x
        ffn_pre, ffn_post, ffn_comb = self.mixes(x, self.hc_ffn_fn, self.hc_ffn_scale,
                                                 self.hc_ffn_base)
        x = self.ffn_norm(hc_pre(x, attn_pre))
        x = self.ffn(x)                                             # <-- was self.ffn(x, image_mask)
        x = hc_post(x, residual, ffn_post, ffn_comb)
        return x, ffn_pre                     # <- consumed by the next sublayer
```

### 8.3 `Transformer`

```python
    def forward(self, input_ids, token_types=None, start_pos=0,
                pixel_values=None, image_grid_hws=None):
        cfg = self.cfg
        image_mask = None if token_types is None else token_types >= 0
        engram_mask = None if image_mask is None else ~image_mask
        hash_ids = None
        if self.engram_hash is not None:
            hash_ids = self.engram_hash(input_ids, start_pos, engram_mask)

        h = self.embed(input_ids)
        if cfg.vision_n_layers and pixel_values is not None:
            assert start_pos == 0
            feats = self.vision(pixel_values, image_grid_hws)
            emb = self.aligner(feats, image_grid_hws)
            h = merge_image_embeddings(h, token_types, emb, self.image_start,
                                       self.image_end, self.image_newline)

        b, S, _ = h.shape
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        pre_mix = make_identity_pre_mix(b, S, self.hc_mult)

        main_hiddens = []
        for i, layer in enumerate(self.layers):
            if layer.engram is not None and hash_ids is not None:
                h = layer.engram(h, hash_ids[:, :, layer.engram_idx], engram_mask)
            if i in self.target_layer_ids:
                main_hiddens.append(h.mean(dim=2))     # attention input of the target layer
            h, pre_mix = layer(h, start_pos, pre_mix)   # <-- image_mask dropped

        h = hc_pre(h, pre_mix)                         # final collapse with last ffn pre
        return self.head(self.norm(h)), (torch.cat(main_hiddens, -1) if main_hiddens else None)
```

**`image_mask` is not deleted** — it is still computed here, because `engram_mask` derives
from it (§11.3 masks image tokens out of the n-gram lookup). Only its use as a *router* input
is gone.

### 8.4 `VertexHead` — the tied LM head

```python
class VertexHead(torch.nn.Module):
    """LM head sharing storage with `embed.weight`. NOT a Parameter of its own.

    Spec §21.11's ParallelHead holds `weight` as an fp32 Parameter of shape [V, d] and is
    the single biggest reason the released config sets `tie_word_embeddings: false`. With
    tying, the head becomes a *view*: there is exactly one [32000, 1408] tensor in the
    model, and it appears once in the state dict as `embed.weight`.
    """

    def __init__(self, embed):
        super().__init__()
        self.embed = embed                 # torch.nn.Embedding; register, do not duplicate

    def forward(self, x, full_logits=False):
        if not full_logits:
            x = x[:, -1]
        return F.linear(x.float(), self.embed.weight.float())    # run in fp32 regardless of dtype
```

Instantiate as `self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)` then
`self.head = VertexHead(self.embed)`. **Do not** wrap `embed.weight` in another `Parameter`
— that silently un-ties them (and allocates a second copy).

Gradients accumulate correctly: both the embedding lookup and the head projection write into
the same leaf, and autograd sums them.

### 8.5 `DSparkBlock`

```python
class DSparkBlock(Block):
    attention_cls = DSparkAttention

    def forward(self, x, start_pos, pre_mix, main_x):
        if start_pos == 0:
            self.attn(x, start_pos, main_x)           # seed window caches only
            return x, pre_mix
        return Block.forward(self, x, start_pos, pre_mix, main_x)   # no image_mask arg
```

### 8.6 Complete delta summary

```
vertex500m/moe.py  -> deleted, replaced by ffn.py (§8.1)
  class Gate                            -> DELETED
  class Expert                          -> UNCHANGED (moved)
  class MoE                             -> class DenseFFN

config.py
  Cfg.n_routed_experts                  -> DELETED
  Cfg.n_shared_experts                  -> DELETED
  Cfg.num_experts_per_tok               -> DELETED
  Cfg.scoring_func                      -> DELETED
  Cfg.topk_method                       -> DELETED
  Cfg.norm_topk_prob                    -> DELETED
  Cfg.routed_scaling_factor             -> DELETED
  Cfg.dspark_n_routed_experts           -> DELETED
  Cfg.dspark_num_experts_per_tok        -> DELETED
  Cfg.get_moe_config()                  -> DELETED
  Cfg.tie_word_embeddings               -> ADDED (default True)
  PRESET_VERTEX_500M                    -> rewritten (§7.2)

model.py
  Block.__init__:   self.ffn = MoE(...)      -> self.ffn = DenseFFN(cfg, layer_id)
  Block.forward:   (..., image_mask, ...)    -> (..., *attn_args)
  Block.forward:   self.ffn(x, image_mask)   -> self.ffn(x)
  Transformer.forward: layer(h, start_pos, pre_mix, image_mask)
                                     -> layer(h, start_pos, pre_mix)
  ParallelHead                      -> VertexHead (tied view, §8.4)

dspark.py
  DSparkBlock.forward: passes None as image_mask -> signature simplified (§8.5)
  num_nextn_predict_layers          2 -> 1

untouched files: quant.py, rope.py, hc.py, sparse_attn.py, attention.py,
                engram.py, vision.py, generate.py
```

### 8.7 Minimal runnable test

```python
from dataclasses import replace
import sys
import torch

tiny = replace(PRESET_VERTEX_500M,
               vocab_size=512, dim=128, n_layers=8, n_heads=4, head_dim=32,
               q_lora_rank=64, qk_rope_head_dim=8, o_lora_rank=64, o_groups=4,
               moe_intermediate_size=64, sliding_window=8,
               index_n_heads=2, index_head_dim=32, index_topk=8,
               candidate_topk_blocks=4,
               compress_ratios=(0, 0, 2, 2, 1, 1, 1, 1, 0),   # 8 backbone + 1 DSpark
               kv_source_layer_ids=(2, 4), index_source_layer_ids=(2, 4, 6),
               candidate_source_layer_id=4,
               engram_num_embeddings=(600766,), engram_vocab_size=50_021,
               engram_n_heads=2, engram_head_dim=32,
               engram_compressed_vocab_size=512,
               num_nextn_predict_layers=1, dspark_block_size=3,
               dspark_target_layer_ids=(7,), dspark_markov_rank=16,
               max_seq_len=64, max_batch_size=1, vision_n_layers=0)

layout     = EngramLayout(tiny)
token_map  = list(range(tiny.vocab_size))
hash_state = NgramHashState(tiny, layout, token_map, tiny.engram_compressed_vocab_size)

model = Transformer(tiny, engram_layout=layout, engram_hashes=hash_state).eval()

# --- no router survived ---------------------------------------------------
# Gate was deleted from ffn.py, so it must not resolve -- that IS the assertion.
assert "ffn" in sys.modules and "Gate" not in vars(sys.modules["ffn"])
for blk in list(model.layers) + list(model.mtp):
    assert isinstance(blk.ffn, DenseFFN)
    assert sum(p.numel() for p in blk.ffn.parameters()) == 3 * tiny.dim * tiny.moe_intermediate_size

# --- embeddings are genuinely tied -----------------------------------------
names = [n for n, _ in model.named_parameters()]
assert not any(n.endswith("head.weight") for n in names), names
assert sum(1 for n in names if n.endswith("embed.weight")) == 1
assert model.head.embed is model.embed

ids = torch.randint(0, tiny.vocab_size, (1, 17))
logits, _ = model(ids, start_pos=0)
assert logits.shape == (1, tiny.vocab_size)
assert torch.isfinite(logits).all()

# --- decode ----------------------------------------------------------------
cur = logits
for i in range(ids.size(1), ids.size(1) + 23):
    cur = sample(logits, 1.0)
    logits, main_hidden = model(cur[:, :1], start_pos=i)
    assert torch.isfinite(logits).all()
assert model.layers[2].attn.compress_kv is not None

# --- DSpark drafting -------------------------------------------------------
S = ids.size(1) - 1 + 23
spec = model.forward_spec(logits.argmax(-1, keepdim=True), main_hidden, S)
assert spec is not None and spec[0].shape[1] == tiny.dspark_block_size + 1

# --- chunk-general prefill must match single-shot (V4.1 acceptance test) ----
model.init_caches(1)
a, _ = model(ids[:, :9], start_pos=0)
model.init_caches(1)
b, _ = model(ids[:, :5], start_pos=0)
bb, _ = model(ids[:, 5:9], start_pos=5)
assert torch.allclose(a, bb, atol=2e-2, rtol=2e-2), (a - bb).abs().max()

# --- backward reaches the dense FFN, and the tied embed gets BOTH grads ------
model.train()
out, _ = model(ids, start_pos=0)
out.sum().backward()
for i, blk in enumerate(model.layers):
    for n in ("w1", "w2", "w3"):
        g = getattr(blk.ffn.expert, n).weight.grad
        assert g is not None and g.abs().sum() > 0, f"layer {i} ffn.{n} got no gradient"
assert model.embed.weight.grad is not None and model.embed.weight.grad.abs().sum() > 0

print("vertex500m dense conversion: all architecture checks passed")
```

---

## 9 Checkpoint layout and conversion

### 9.1 Tensor naming

```
embed.weight                       [32000, 1408]          bf16   <-- the ONLY [V,d] tensor
layers.{i}.attn.wq_a.weight        [352, 1408]
layers.{i}.attn.q_norm.weight      [352]
layers.{i}.attn.wq_b.weight        [2816, 352]
layers.{i}.attn.wkv.weight         [128, 1408]
layers.{i}.attn.kv_norm.weight     [128]
layers.{i}.attn.wo_a.weight        [1408, 704]            (o_groups=4, o_lora_rank=352)
layers.{i}.attn.wo_b.weight        [1408, 1408]
layers.{i}.attn.attn_sink          [22]                   fp32
layers.{i}.attn.compressor.wkv.weight    [128, 1408]      (kv_source layers: 2, 8)
layers.{i}.attn.compressor.wgate.weight  [128, 1408]      (ratio > 1 only -> layer 2)
layers.{i}.attn.compressor.norm.weight   [128]
layers.{i}.attn.indexer.wq_b.weight         [512, 352]    (3 indexers: 2, 8, 12)
layers.{i}.attn.indexer.weights_proj.weight [8, 1408]
layers.{i}.attn.indexer.wk.weight           [64, 128]     (2 owners: 2, 8)
layers.{i}.attn.indexer.k_norm.weight       [64]
layers.{i}.ffn.w1.weight            [3712, 1408]          <-- NEW: three, was 103
layers.{i}.ffn.w2.weight            [1408, 3712]
layers.{i}.ffn.w3.weight            [3712, 1408]
layers.{i}.ffn.w{1,2,3}.scale       [out/32, in/32]       ue8m0
layers.{i}.attn_norm.weight / ffn_norm.weight  [1408]      fp32
layers.{i}.hc_attn_fn / hc_ffn_fn   [24, 5632]            fp32
layers.{i}.hc_attn_base / hc_ffn_base [24]                fp32
layers.{i}.hc_attn_scale / hc_ffn_scale [3]               fp32
layers.{i}.engram.embed.weight      [600766, 128]  fp8 (+ .scale [600766, 4] ue8m0)
layers.{i}.engram.wkv.weight        [7040, 1536]
layers.{i}.engram.q_weight / k_weight [4, 1408]  bf16
norm.weight                         [1408]                 fp32
mtp.0.*                             (DSpark draft stage; 26.7M)
```

**No `head.weight`.** The head is a view of `embed.weight` (§8.4).

**One shape to watch:** `engram.embed.weight` row scales go from
`[rows, head_dim/32] = [600766, 2]` at `enghd=64` to `[600766, 4]` at `enghd=128`.
`ParallelEngramEmbedding` computes `dim // block`; 128 % 32 = 0, consistent.

### 9.2 Disk composition

Exclusive components — the backbone's 16 layers already contain their own attention, HC and
norm weights, so those are **not** listed again.

| component | params | bytes |
|---|---|---|
| 16 backbone layers (ffn + attn + HC + norms) | 329,503,424 | 330 MB fp8 |
| embed (**tied**, single table) | 45,056,000 | **90 MB** bf16 |
| Engram table (fp8) | 76,898,048 | 77 MB |
| Engram `wkv` + `q/k_weight` | 10,824,704 | 22 MB fp8 |
| 1 DSpark layer | 20,593,964 | 21 MB fp8 |
| DSpark `main_proj` + norm | 1,983,872 | 4 MB fp8 |
| DSpark Markov head (`2·V·64`) | 4,096,000 | 8 MB fp8 |
| DSpark confidence head | 1,473 | — |
| 3 indexers + 2 index-key projections | 590,976 | ~1 MB fp8 |
| 2 compressors | 540,928 | ~1 MB fp8 |
| **total** | **490,089,389** | **≈ 555 MB** |

Compare the MoE preset: its 326.4M-parameter FP4 expert bank would occupy ~163 MB, but as
32 × 16 = 512 separate packed tensors plus 512 scale tensors, each needing a Python-level
`torch.where(idx == e)` dispatch per layer per forward.

### 9.3 Converting the existing MoE preset

No weight-level bridge exists — this is a re-training target. A script should:

1. **Delete** `ffn.gate.{weight,bias,bias_vl}`, `ffn.experts.*`, `ffn.shared_experts.*` for
   every backbone layer and every `mtp.*` stage.
2. **Create** `ffn.{w1,w2,w3}.weight` at `[3712,1408]`/`[1408,3712]`/`[3712,1408]`.
3. **Initialize** with `initializer_range = 0.02`, not by copying `shared_experts`.
   Sparse-upcycling (stacking the old shared expert, filling from routed experts) does **not**
   preserve the function — the routing weights are gone — and is only worth doing as a
   warm-start experiment.
4. **Replace `head.weight` with `embed.weight`** and verify they are the same storage object
   after load (§8.4).
5. **Drop the second DSpark stage** entirely (`mtp.1.*`).
6. **Retokenize.** Every id changes. `engram_compressed_vocab_size` and the Engram
   `token_map` must be rebuilt from qt-VI.1.3 (§16.4).

---

## 10 Quantization: what survives FP4 and what does not

### 10.1 Survives unchanged — all of it

The KV-cache quantization is the load-bearing part of V4.1's serving story and has nothing to
do with MoE:

| tensor | format | block | scale |
|---|---|---|---|
| window KV (RoPE tail included) | FP8 E4M3 | 32 | UE8M0, `2^ceil(log2(amax/448))`, amax floor 1e-4 |
| compressed KV latent | FP4 E2M1 | 16 | E4M3, amax floor `6·2^-9` |
| indexer Q / K | FP4 E2M1 | 32 | UE8M0, amax floor `6·2^-126` |
| Engram table rows | FP8 E4M3 | 32 | UE8M0 |

Block divisibility at the new geometry (invariant 43):

```
window     block 32 | hd = 128  ->  128 % 32 = 0   OK
compressed block 16 | hd = 128  ->  128 % 16 = 0   OK
indexer    block 32 | idd = 64  ->   64 % 32 = 0   OK
```

`QUANT_MODE ∈ {NONE, FAKE, REAL}` and the QAT placement rule (FP4 main-KV QAT enters in
post-training, not pre-training) are unchanged.

### 10.2 Disappears

| format | was | now |
|---|---|---|
| routed expert weights | FP4 E2M1, `[out, in/2]` packed int8 + `[out, in/32]` ue8m0 scales | **gone with the experts** |
| dense FFN weights | (were FP4 as "expert" weights) | **FP8 E4M3**, `ceil(out/32) × ceil(in/32)` UE8M0 — same as every other dense linear |

This is the one place the conversion makes the model *less* aggressively quantized: the FFN
is now 51% of parameters quantized at FP8 rather than FP4, so the checkpoint is ~2× larger
than a naive FP4 expectation. In absolute terms trivial (330 MB vs 165 MB). **Do not**
FP4-quantize the dense FFN to recover the symmetry: the released FP4 expert format exists
because 384 experts × 40 layers of FP8 weights would not fit in memory at 552B. That
constraint does not exist at 0.5B, and FP4 on a single wide FFN costs real accuracy for no
benefit.

### 10.3 A new activation-magnitude problem

`swiglu_limit = 10.0` was calibrated against per-expert FFN activations of width 416 (MoE
preset) or 2304 (released). The dense FFN here is width 3712 and carries **every** token's
full gradient instead of 3/33 of them. Two consequences:

1. **Keep the clamp** (already decided, §2.4) and expect it to bind more often than it did.
   Instrument the fraction of pre-SwiGLU values that saturate during the first 100 steps.
2. **Re-check RMSNorm statistics in fp32.** With `rms_norm_eps = 1e-20` and `d = 1408` (2.75×
   the MoE preset), the mean-of-squares denominator sums over 1408 elements. The architecture
   is unchanged but the magnitudes are not. Do not "fix" a `1e-20` that looks wrong —
   invariant 39; it is deliberate and matched to the released model.

---

## 11 Training

### 11.1 What disappears

| MoE training machinery | replacement |
|---|---|
| Auxiliary-loss-free load balancing (`noaux_tc`) | none |
| Per-modality expert bias updates (§19.3) | none |
| Sequence-level balance loss, weight `1e-4` | none |
| Sample-level attention masking during pre-training | **keep** — CSA2, not MoE (§19.4) |
| `gate.bias` / `gate.bias_vl` updated by the balance rule | gone |
| Expert-parallel all-reduce / dispatch / combine overlap | gone — the FFN is a plain tensor-parallel GEMM |
| Unselected-experts-receive-no-gradient | gone — every FFN parameter gets a gradient on every step |

The last row matters for throughput accounting. The MoE preset needed expert-parallel
sharding and all-to-all overlap to hide dispatch. The dense preset needs only tensor
parallelism, and the `for e in range(32): torch.where(idx == e)` loop — 512 iterations per
forward at 16 layers — is gone.

### 11.2 Token budget

| | MoE preset | vertex500m |
|---|---|---|
| stored params | 486.9M | 490.1M |
| Chinchilla tokens (20/param) | 9.7B | 9.8B |
| tokens DeepSeek's MoE crossover implies (≥2B params, 100B+ tokens) | **50–100B** | n/a |
| `D_opt` direction vs dense (arXiv:2507.17702) | **larger** | baseline |
| FLOPs at that token count | 2,336 PFLOP | 19,378 PFLOP |
| GPU-bound wall clock, 8×H100 @35% MFU | 0.7 h | **1.9 h** |
| FFN gradient batch per step | `B/16` per expert | `B` |
| trainable regime | 0.49B on 9.7B — **~10× below** the crossover on both axes | 0.49B on 9.8B — **inside** every published dense scaling law |

The argument is not that the MoE preset is too slow. It is that the MoE preset is trained in
a regime where MoE has no demonstrated advantage, at 16× the gradient noise per step, while
paying for a router, a dispatch/combine pass, and aux-loss-free load balancing.

**Every wall-clock figure in this document is FLOP-only at 35% MFU and excludes kernel-launch
overhead and `torch.compile` warm-up.** §4.6 argues the launch term is small after
compilation, but that argument is unmeasured — §17.8 measures it.

If you want the matched-FLOP control, candidate M (§4.5) is a 163M dense model at
0.350 GFLOP/token. Train all three arms on the same tokens; the ablation costs a few hours on
8×H100 and settles a genuinely contested question (§20.6).

### 11.3 Optimizers — unchanged in structure

Per spec §19.2, the allocation is orthogonal to MoE:

| parameter class | optimizer |
|---|---|
| LM backbone linear weights, Engram projections, VL projector | **Muon** (momentum 0.95, wd 0.1, update RMS 0.18), head-wise Muon for Q and K |
| norm weights, `attn_sink`, `hc_*` | **AdamW** (β 0.9/0.95, ε 1e-20, wd 0.1) |
| embedding, LM head, Engram tables | **momentum + Sinkhorn balancing** (`K=11`, `τ=1e-3`, `γ=0.18`) |

**Tying interacts with the third row.** The released spec assigns embedding, prediction head
and Engram tables to a momentum + Sinkhorn-balanced optimizer (Algorithm 1) rather than Adam.
With `tie_word_embeddings: true` there is now **one** tensor in that group instead of two, and
it receives gradient from both the input lookup and the output projection. The Sinkhorn
row/column equalization is defined for embedding-like tables; applying it to a tensor that is
simultaneously an input and an output embedding is untested. Either exclude `embed.weight`
from the Sinkhorn optimizer and give it plain AdamW, or implement and validate the combined
update. Do not silently run the released algorithm on a tied tensor (§20.5).

### 11.4 Learning rate

- Muon's LR is expressed through the update-RMS rescale, so it is roughly scale-free. Start
  at the LR you would use for a dense ~0.5B model with Muon — typically 3×–10× the AdamW LR.
  Do **not** use 2.6e-4 from the V4.1 report; that is tuned for a 2048-GPU run.
- The FFN matrices being 8.9× wider than the MoE preset's experts is exactly the situation
  where the standard dense LR heuristic applies. If training is stable, raise before lowering.
- Warmup: 2000 steps is a reasonable carry-over.

### 11.5 QAT schedule

Unchanged: window-KV FP8 fake-quant from the start, compressed-KV FP4 fake-quant from the
start (V4 behavior), main-KV FP4 QAT introduced during post-training (§14.5).

---

## 12 CED and the prefill/decode activation split

The CED claim survives structurally, and this is worth stating carefully because it is easy
to assume it depends on MoE.

**It does not.** The split comes from the encoder/decoder *layer* boundary, not from routing.
Prefill runs `L/2 = 8` layers; decode runs all 16. Under MoE that gave 8B/16B because each
layer activated 7 of 385 FFN matrices. Under dense it gives 164.8M/329.5M because each layer
activates its whole FFN. **The ratio is 1:2 in both cases**, for the same reason.

```
vertex500m:
  prefill    8 layers x 20,593,964  =  164,751,712 active   (encoder only)
  decode    16 layers x 20,593,964  =  329,503,424 active   (full stack)
  ratio                               exactly 1:2
```

What changes is the *value* of the split in wall-clock terms. The 2× prefill saving is real,
but in the MoE model it was competing against dispatch/combine communication that dense does
not have. Dense CED is a cleaner 2× than MoE CED was.

**Spec edits for §2.2 and §20.2:** replace "8B prefill / 16B decode" with
"164.8M / 329.5M", and rewrite the derivation as `L/2 · (attn + ffn + hc + norms)` per side
rather than the `(1 + num_experts_per_tok)` form. Delete §10.3's `[SIZING]` note — it explains
a number that no longer exists.

---

## 13 Engram at a 17.9% share

```
engram_head_dim    64 -> 128
engram rows        600766 (unchanged: the 12 primes from 50021)
engram table       600766 x 64  = 38.4M   ->  600766 x 128 = 76.9M
engram wkv         Linear(12*64,  1408*5) =  5.4M   ->  Linear(12*128, 1408*5) = 10.8M
engram q/k_weight  2 x 4 x 1408  = 11,264   (unchanged)
                   --------------------------------------------------
                   43.4M                              87.7M   (17.9% of total)
```

**Why growing `enghd` rather than the row count.** The two are interchangeable in parameter
count, but not in what the component *does*:

- `enghd` is the width of each hash-table row and of the concatenated lookup vector. Raising
  it increases the capacity of each n-gram cell.
- `rows` is the number of hash buckets, fixed by the prime layout (`sum(primes)`). Raising it
  reduces collisions.

At 600,766 rows × 64 dims = 38.4M cells across 12 prime ranges, collisions are already rare.
Raising `enghd` to 128 gives each cell twice the representation capacity at the same
collision rate — the better trade when the row count is pinned by the prime scheme.

**Consequence to watch.** The Engram forward does `wkv(embed(hash_ids).flatten(-2))` — a GEMM
of `[B·S·12, 128] → [B·S, 7040]`. At `B·S = 4096` tokens the flattened lookup vector is
`4096 × 1536 × 2 bytes = 12.6 MB` in bf16. It is a *lookup*, so it is memory-bound rather than
compute-bound. Prefetch the FP8 rows overlapped with the vision encoder as §11.5 prescribes.

**Pre-existing spec bug found while writing this.** §11.2 says:

> "for each ngram_size in 2..(max_ngram-1)"

That range yields 2 values, but the released model has 24 primes per layer and
`n_hash_cols = (max_ngram-1) * n_heads = 24` requires **3** n-gram sizes (2, 3, 4). The
reference `EngramLayout.__init__` uses `for _ in range(self.max_ngram - 1)`, which is 3 — so
the **code is correct and the prose is wrong**. The script in §5.1 carries the comment
`max_ngram-1 sizes => the 2/3/4-grams`. Correct the doc to `2..max_ngram` while converting.

---

## 14 DSpark

### 14.1 What changes

| element | before | after |
|---|---|---|
| draft stages | 2 | **1** |
| FFN | MoE, 16 routed top-2 + shared, `MoEI=416` | **dense `DenseFFN`, `MoEI=3712`** |
| attention | `DSparkAttention`, `compress_ratio == 0` assert | unchanged |
| `main_proj` | `Linear(d·3, d)` over target layers `[13,14,15]` | **`Linear(d·1, d)` over `[15]`** |
| Markov head | rank 64, `Embedding(V,64)` + `Linear(64,V)` | unchanged — **tied to `V=32000`** |
| confidence head | `Linear(d+64, 1)` | unchanged |
| block size | 4 draft positions | unchanged |
| embedding / LM head | tied to backbone | unchanged (already was) |

DSpark total: 32.1M → **26.7M**, and its share falls from 6.6% (MoE preset) to **5.44%** of
the model.

### 14.2 Why one stage

At two full-width layers (`d=1408`), DSpark was 9.96% of the model — up from 6.6% — and a
higher share of the *active* parameter count, because the drafter runs on every decode step.
That is the point at which a speculative decoder stops being cheap relative to its target.
One stage brings it to 5.44%, which is the right order for a 4-position-block drafter.

Note the Markov head is `2·V·64 = 4.1M` and **scales linearly with the vocabulary**. That is
the concrete reason not to grow `V` further: at `V=131072` it becomes 16.8M, 3% of the model
spent on a one-token-lookahead bias.

### 14.3 The `dspark_n_routed_experts` distinction is gone

Previously the draft stack was distinguished from the backbone by having its own expert count
(128/3 vs 384/6) and its own `MoE`. Now both use the identical `DenseFFN`. The only remaining
structural differences are `attention_cls = DSparkAttention` and
`compress_ratios[n_layers] == 0`. That is sufficient — `DSparkBlock` differentiates itself
through its forward path, not its FFN.

With one stage, `stage 0` is simultaneously the first and the last, so it carries
`main_proj`, `main_norm`, `markov_head` and `confidence_head` at once. Check that the spec
§12.1 forward path handles the single stage without a second iteration — the
`for i in range(dspark_block_size)` sampling loop is unchanged, but `mtp[0]` is now also
`mtp[-1]`.

### 14.4 Training schedule

Unchanged: DSpark is introduced in a dedicated stage after backbone pre-training with the
backbone frozen; during post-training it keeps training but **does not propagate gradients
into the backbone**. This matters more in dense, not less — with no router, a gradient leak
from the drafter into the backbone is a straightforward optimization bug rather than a subtle
routing interaction.

---

## 15 Multimodality after router removal

### 15.1 The loss

V4.1 routes image and text tokens to *different experts* by selecting between `bias` and
`bias_vl` (spec §6.1, §13.4). Spec §1.2 lists "every MoE layer now scores all tokens" as a
V4.1 delta — so in V4.1 the multimodal routing is a full 384-expert score with a per-modality
bias vector, not a hard partition.

Removing MoE removes this entirely. There is no mechanism left that distinguishes modalities
inside the FFN.

### 15.2 The replacement, and why it is not a deviation

1. **The tokenizer already provides it.** qt-VI.1.3 reserves **72 `<|lang:xx|>` tokens** at
   IDs 14–84 for language control. These are 71 free, trained embedding rows in the tied
   table. Using the language tag (or, once vision is enabled, a modality tag occupying the
   same slot) as the conditioning signal is strictly better than inventing a new learned
   embedding: it costs zero parameters and comes with a tokenizer-native convention. With
   `vision_n_layers = 0` this is the whole story for a text model — the model is already
   multilingual at initialization.
2. **Engram already distinguishes modalities.** Spec §11.3 masks image tokens `DEAD` in the
   hash cache and §11.4 forces the write gate to 0 on them. The multimodal signal moved
   from the FFN to the residual stream.
3. **If vision is enabled**, one image token at ID 85 plus the three learned `[d]` delimiter
   vectors (spec §13.3) carry image position. Image spans are already excluded from the
   n-gram hash.

**Do not** reintroduce a gate for this. A `d`-dimensional per-modality bias vector added to
the FFN input is the entire mechanism, and it is a different architecture from `bias_vl`.

### 15.3 What must still be kept

`image_mask` is still computed in `Transformer.forward` — not for routing, but because
`engram_mask = ~image_mask` feeds `NgramHashState`. Deleting the router does **not** mean
deleting `token_types` handling from the forward.

---

## 16 The scaling-down guide, and binding to qt-VI.1.3

### 16.1 The dense procedure (replaces spec §23)

The old procedure exists entirely because the routed-expert term dominated and had to be
*solved* for:

```
OLD (spec §23 step 2):
  "Choose E, MoEI and topk so that L·3·E·d·MoEI lands at ~85% of the budget;
   topk/E should stay in the 1.5%-7% range."

NEW:
  1. Fix the tokenizer first. V is not a free parameter -- it is whatever the tokenizer
     is, and V*d (tied) is a large fixed cost. See §16.4.
  2. Decide the execution model (EAGER vs COMPILED vs GRAPHS) BEFORE fixing d and L.
     It determines whether you minimize kernel launches or FLOPs. See §4.6.
  3. Set MoEI = round_to_nearest_128(8/3 * d). Not a free parameter -- this is the rule.
  4. Pick L for the target compute regime, then solve d to hit the parameter budget.
     Under a compiled layer loop, prefer fewer layers: launches scale with L, not d,
     and FLOPs are what remain once fusion happens.
  5. The FFN/attention split is a RESULT, not a choice. At H*hd = 2d it lands ~76:22.
     To change it, change H*hd (§16.2).
  6. Engram: pick engram_head_dim, then choose engram_vocab_size (the prime seed) so
     rows * engram_head_dim lands at your intended share -- typically 8-18%.
     Verify rows == sum(primes) at build time.
  7. DSpark is 5-10% and its Markov head is 2*V*markov_rank. Budget for it.
  8. Check embed share. If V*d exceeds ~20% of the budget, d is too small or V is too
     large -- go back to step 1.
  9. Check total activation = O(L * d). At long context without checkpointing this
     can exceed the weights. See §4.6 point 3.
 10. Preserve every STRUCTURAL invariant (§16.3). These are cheap and they are the point.
```

### 16.2 If you must change the attention/FFN split

The lever is `H·hd`, since attention scales linearly in it and the FFN does not:

| `H·hd` | attention share of layer | total at `d=1408, L=16, V=32000` tied |
|---|---|---|
| `1d` (H=11, hd=128) | 18.3% | 471.6M |
| `2d` (H=22, hd=128) | **22.5%** | **490.1M** |
| `4d` (H=44, hd=128) | 28.9% | 527.2M |

Or keep `H·hd = 2d` and raise `hd`: `hd = 256, H = 11` gives a similar parameter count but
fewer, wider heads. **Wider heads are probably better here**, because `sparse_attn` and the
indexer both benefit from head width and `index_head_dim = 64` is already narrow. This is an
untested choice (§20.1).

### 16.3 Structural invariants — unchanged, still mandatory

From spec §3.6, surviving the conversion verbatim:

- `compress_ratios[0] == 0` and `compress_ratios[1] == 0`.
- `L/2` is the encoder/decoder boundary; the first decoder layer `L/2 = 8` is a
  `kv_source_layer` **and** the `candidate_source_layer_id`.
- Every non-source CSA2 layer has `compress_ratio > 0` and reads the nearest source above it.
- Encoder CSA2 layers use `m = 2`; decoder CSA2 layers use `m = 1`.
- `len(index_source_layer_ids) > len(kv_source_layer_ids)` — Reindex layers own no keys.
- `o_groups` divides `H·hd`.
- `index_head_dim` need not equal `hd`.

Plus the quantization divisibility invariant (spec #36), re-checked at the new geometry in
§10.1.

### 16.4 Binding to `JamesQuartz/qt-VI.1.3-32k`

**Tokenizer facts (from the model card).** Vocabulary of **32,000** tokens, including **86
reserved special tokens at IDs 0–85**. Two-stage byte-level BPE (subwords, then
cross-word "superwords"), byte-level base, so **no UNK is ever emitted** and round-trip is
lossless on all 204 FLORES-200 languages. No `<unk>` collision is possible by construction.

```
  0  <|padding|>      ->  pad_token_id = 0,  engram_pad_token_id = 0
  1  <|bos|>          ->  bos_token_id = 1
  2  <|endoftext|>     ->  eos_token_id = 2
  3  <|unk|>          ->  dspark_noise_token_id = 3   (structurally never emitted)
  4  <|sep|>
  5-13     9 chat/tools tokens:  <|system|> <|user|> <|assistant|> <|tool_call|>
                               <|tool_result|> <|thinking|> <|/thinking|>
                               <|code|> <|/code|>
  14-84    71 x <|lang:xx|>   language control  (was 72)
  85       <image>             image_token_id   (repurposed one <|lang:xx|> slot)
  86+      unused
```

**These ids are not assignable.** The tokenizer reserves 0–85; the config must match them.
Note this is the **opposite** of the released DeepSeek convention (`bos=0, eos=1, pad=2`),
and a loader that assumes DeepSeek's ordering will silently produce a model whose BOS is a
PAD.

**Why ID 3 for `dspark_noise_token_id`.** The `<|unk|>` token is declared for convention but
the byte-level base means the tokenizer can never emit it, and no corpus contains it. A
draft-block filler that cannot collide with real data is exactly what §12.1 wants.

**Why only ONE image token, not four.** Spec §13.3 fills the three delimiter positions with
**learned `[d]` vectors** (`image_start`, `image_end`, `image_newline` — spec line 1378,
`Transformer.__init__` lines 2662–2664), not vocabulary entries. The delimiter *roles* ride in
the parallel `token_types` tensor (`-1` text; `0..3` IMAGE_START / NEW_LINE / END / IMAGE).
Only `image_token_id` is a real vocabulary id. Repurposing one `<|lang:xx|>` slot preserves
`V = 32000` exactly and costs 71 language tags instead of 72.

**Build-time steps (placeholders in the config, resolved once):**

```
1. engram_compressed_vocab_size
   Apply the spec §11.2 normalizer to each of the 32,000 token strings:
       NFKC -> NFD -> StripAccents -> Lowercase
       -> collapse [\t\r\n]+ to " " -> "^ $" -> sentinel -> Strip -> sentinel -> " "
   Count distinct results. THIS NUMBER IS NOT 32,000 and it is not knowable in advance:
   NFKC folding collapses CJK compatibility ideographs, Arabic presentation forms, and
   full-width Latin into their canonical forms, so distinct tokens collide. On a 204-language
   byte-level vocabulary this is a material reduction and it silently degrades Engram lookup
   quality if unmeasured. Assert:
       compressed_vocab_size < 50021   (the prime seed must exceed it)
       compressed_vocab_size == cfg.engram_compressed_vocab_size

2. engram_num_embeddings
   sum of the 12 primes >= engram_vocab_size = 50021 -> 600766. Assert equality.

3. token_map:  [32000] int64, token id -> compressed id.
   Built from the same normalizer. Identity for tokens whose normal form is themselves.
```

**Cost of the tokenizer choice.** English runs 3.90 chars/token vs Llama-2's 4.24 — an ~8%
token tax on English training, paid for lossless coverage of the long tail (CJK 0.94 vs
Llama-2's 0.70; Arabic 2.11 vs 1.09; Thai 1.80 vs 0.93). At a fixed 32K window that means
~8% more tokens for the same amount of English text, or ~8% less text for the same budget.
Worth knowing before you compare against an English-only baseline.

---

## 17 Verification suite

### 17.1 Parameter count (fails first, catches the most)

```python
v = count(PRESET_VERTEX_500M)
assert v['backbone'] == 375_691_328, v['backbone']
assert v['engram']   ==  87_722_752, v['engram']
assert v['dspark']   ==  26_675_309, v['dspark']
assert v['total']    == 490_089_389, v['total']
assert v['lay']      ==  20_593_964, v['lay']
assert v['active']   == 329_503_424, v['active']

# and the same script must still reproduce the MoE preset, untied:
m = count(MOE_PRESET)
assert abs(m['backbone']/1e6 - 414.4) < 0.1
assert abs(m['engram']/1e6   -  40.4) < 0.1
assert abs(m['dspark']/1e6   -  31.9) < 0.3
```

### 17.2 No router survived, and embeddings are tied

```python
import inspect, sys
assert "Gate" not in vars(sys.modules["ffn"])              # deleted from ffn.py
assert "image_mask" not in inspect.signature(DenseFFN.forward).parameters
for blk in list(model.layers) + list(model.mtp):
    assert isinstance(blk.ffn, DenseFFN)
    assert sum(p.numel() for p in blk.ffn.parameters()) == 3*cfg.dim*cfg.moe_intermediate_size

names = {n for n, _ in model.named_parameters()}
assert not any(("ffn.gate" in n or "ffn.experts" in n or "shared_expert" in n) for n in names)
assert not any(n.endswith("bias_vl") for n in names)
assert not any(n.endswith("head.weight") for n in names)   # tied
assert sum(1 for n in names if n.endswith("embed.weight")) == 1
assert model.head.embed is model.embed
```

### 17.3 The §21 acceptance test, unchanged

Prefill + 23 decode steps + DSpark drafting → finite logits; chunked prefill `[0:9]` vs
`[0:5]+[5:9]` agrees **bit-exactly**; forward deterministic across runs; backward reaches the
dense FFN, Engram projections, indexer queries, HC coefficients and compressors. Full code in
§8.7.

### 17.4 Per-layer composition guard

Catches a partial revert — the most likely mistake, since a half-converted model still runs:

```python
lay = sum(p.numel() for p in model.layers[0].parameters())
ffn = sum(p.numel() for p in model.layers[0].ffn.parameters())
assert 0.70 < ffn/lay < 0.82, ffn/lay        # MoE preset was 0.973; a revert shows up here
```

### 17.5 Attention invariants at the new widths

- `32 | hd`, `16 | hd`, `32 | index_head_dim` — §10.1
- `o_groups | (H·hd)` and `o_groups | o_lora_rank` — (2816 % 4 = 0; 352 % 4 = 0)
- `all-masked` row in `sparse_attn` → exactly `0`, not `NaN`
- `attn_sink` enters the denominator only, no value vector
- inverse RoPE applied to the output tail before `wo_a`
- compressor ratio-2 emits every 2nd decode step and carries `pool_state` on the others
- indexer runs **before** the latent is RoPE'd and quantized

### 17.6 Gradient contract (rewritten from spec #37)

```
  every ffn.w1/w2/w3 receives a dense gradient on every step   <-- the point of the conversion
  indexer wk receives gradients only on layers 2 and 8         (unchanged)
  embed.weight receives gradient from BOTH the lookup and the tied head projection
  gate.bias / bias_vl no-gradient assertion                   DELETED (no gate exists)
  Unselected-experts-receive-no-gradient assertion            DELETED (no experts exist)
```

The second half of the old invariant is the interesting one to explicitly delete: a suite
carried over unchanged from the MoE version would assert that *most* parameters get **no**
gradient, the exact opposite of what a dense model must do.

### 17.7 Numerical parity

With `QUANT_MODE="NONE"` and matched widths, the dense forward should agree with a
straightforward dense implementation to ~1e-6. There is no parity target against the *released*
552B checkpoint — the shapes differ, the tokenizer differs, and the head is tied. What must
be bit-exact:

- chunked vs single-shot prefill (0.0, per the V4.1 validation status)
- determinism across repeated runs
- the FP4/FP8 value grids and block scales (§10.1)

### 17.8 Measure the execution-model assumption (new)

§4.6 is the reason this config is `d=1408 L=16` rather than something deeper, and it rests
on a claim about `torch.compile` that has not been measured. Measure it before committing.

```python
import torch, time
from torch.utils._python_dispatch import TorchDispatchMode

class LaunchCounter(TorchDispatchMode):
    def __init__(self): self.n = 0
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.n += 1
        return func(*args, **(kwargs or {}))

model = Transformer(cfg, ...).cuda().train()
model = torch.compile(model)                  # no mode='reduce-overhead'

B, T = 8, 1024
ids = torch.randint(0, cfg.vocab_size, (B, T), device='cuda')

# 1. warm up / trigger compilation -- measure this separately, it is a real one-time cost
t0 = time.perf_counter(); model(ids, start_pos=0); torch.cuda.synchronize()
compile_s = time.perf_counter() - t0
print(f"compile: {compile_s:.0f}s")

# 2. per-layer launch counts, POST-compile
#    (run one Backward on a single compiled Block to isolate it)
blk = model._orig_mod.layers[0]
cblk = torch.compile(blk)
c = LaunchCounter()
with c:
    cblk(ids[:, :T].view(-1, 1) if False else torch.randn(B, T, cfg.dim, device='cuda'),
         start_pos=0)
print(f"launches/block post-compile: {c.n}")

# 3. is it GPU-bound?  compare step wall time to FLOP-bound estimate.
for _ in range(5): model(ids, start_pos=0)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(20): model(ids, start_pos=0); torch.cuda.synchronize()
step_s = (time.perf_counter() - t0) / 20

active = 329_503_424
achieved = 1.5e15                       # measured, not assumed -- read from your MFU
gpu_s = 6 * active * B * T / achieved
print(f"step {step_s*1e3:.1f}ms  gpu-est {gpu_s*1e3:.1f}ms -> "
      + ("LAUNCH-bound" if step_s > 1.5*gpu_s else "GPU-bound"))
```

Decide from the output:

| result | action |
|---|---|
| GPU-bound, ≲40 launches/layer | **config is right**; mHC-lite is optional (§21.4) |
| launch-bound | adopt mHC-lite (§21.4) and consider `L=14`, `d=1536` (§4.5) |
| mHC still ≳100 launches/layer after compile | inductor did not fuse the Sinkhorn; mHC-lite becomes mandatory again |

---

## 18 Pitfalls

Ordered by how likely they are.

1. **`MoEI` is round-to-nearest, not round-up.** `3712` is *below* `8/3·d = 3754.67`. Both
   this config and the previous one (3456 vs 3413.3) land below the raw ratio. Implement
   `round(8*d/3/128)*128` and never write "round up" (§4.2).
2. **Do not mistranscribe the `hcp` formula.** `2·((2+hc)·hc·(hc·d)+…)` reduces to
   `2·(96d+27)` at `hc=4`. Writing `2·(24·4·(4·d)+27)` gives `2·(384d+27)` — 4× too large.
   This error was made and caught twice during this conversion; at `d=1408` it would
   silently inflate the per-layer count from 270,390 to 1,081,560.
3. **The tokenizer's special-token ids are not yours to assign.** `bos=1`, `eos=2`, `pad=0` —
   the *reverse* of the released DeepSeek ordering. A loader assuming DeepSeek's convention
   produces a model whose BOS is a PAD, which degrades quietly rather than loudly.
4. **`dspark_noise_token_id = 3` is `<|unk|>`.** Never emitted by a byte-level tokenizer,
   which is the entire point. Do not "fix" it to a real token.
5. **`engram_compressed_vocab_size` must be measured, not assumed.** On a 204-language
   byte-level vocabulary, NFKC folding produces real collisions (§16.4). Assert
   `compressed_vocab_size < 50021`.
6. **`image_mask` is not dead.** It still feeds `engram_mask`. Removing it from
   `Block.forward` is correct; removing it from `Transformer.forward` is a bug that silently
   disables image masking in the n-gram hash.
7. **Only ONE image token is needed.** `image_start`/`image_end`/`image_newline` are learned
   `[d]` parameters, not vocabulary entries (spec line 1378). Do not reserve four.
8. **`tie_word_embeddings: true` must be a real tie.** Wrapping `embed.weight` in a second
   `Parameter` silently un-ties and allocates a copy (§8.4).
9. **The tied tensor's optimizer is unspecified in the released spec.** Spec §19.2 assigns
   embedding, head and Engram tables to a momentum + Sinkhorn-balanced optimizer. With tying
   there is one tensor receiving two gradient sources. Decide this explicitly (§11.3) rather
   than letting the released algorithm run on a structure it was not written for.
10. **Do not delete `swiglu_limit`.** It is the only thing keeping the dense FFN's activations
    inside the fake-quant dynamic range (§10.3).
11. **Do not FP4-quantize the dense FFN.** See §10.2.
12. **Chunked cross-entropy is load-bearing.** `T×V` fp32 logits is 524 MB at `T=4096` and
    2.1 GB at `T=16384`. Unchunked it is the largest single tensor in training.
13. **Activation memory needs checkpointing at long context.** 924 KB/token is 3.78 GB at
    `T=4096` but **15.1 GB at `T=16384`** without checkpointing (§4.6 point 3).
14. **The layer loop is not compiled away.** `torch.compile` on `Block` compiles each block;
    the `for layer in self.layers` loop is still Python, so `L` still multiplies graph-entry
    cost. This is why `L=16` beats `L=20` even under compilation.
15. **The `torch.compile` fusion of the Sinkhorn loop is assumed, not verified.** §4.6 is the
    basis for the whole shape choice. Measure it (§17.8) before treating the config as final.
16. **`index_topk = 512` at `max_seq_len = 32768`.** The *released* value from 1M context. At
    32K with `ratio=1` in the decoder, 512 compressed positions is generous; at `ratio=2` in
    the encoder it covers 1024 raw tokens of a 32K prompt. Verify the indexer is not the
    bottleneck before raising it — it is a per-token GEMM against the full cache.
17. **`candidate_topk_blocks = 512` with `block_size = 8` → 4096 candidate positions** per
    query for layers 9–15. At `max_seq_len=32768` with `ratio=1`, `R = 32768` positions =
    4096 blocks, so 512 blocks is an 8:1 reduction. If you raise `max_seq_len`, re-check.
18. **`engram.embed.weight` row scales change shape** from `[600766, 2]` to `[600766, 4]`.
    A loader that hard-codes 2 will break.
19. **Fix the §11.2 n-gram range while you are in there** (§13). Prose says
    `2..(max_ngram-1)`; code says `range(max_ngram-1)`. They disagree and the code is right.
20. **One DSpark stage makes `mtp[0]` also `mtp[-1]`.** It now carries `main_proj`,
    `main_norm`, `markov_head` and `confidence_head` simultaneously. Check the spec §12.1
    forward path handles the single stage without a second iteration.

---

## 19 Invariants checklist

**Construction**

1. `mix_hc == (2 + hc) * hc == 24`; `comb` consumes the last `hc*hc` entries of `mixes`.
2. `hc_*` parameters are fp32; `hc_scale` has exactly 3 entries per sublayer.
3. A block owns **two** sets of HC parameters (attention, FFN) and **two** norms.
4. `compress_ratios` has `n_layers + num_nextn_predict_layers == 17` entries.
5. `Attention` owns a `Compressor` iff `layer < n_layers and layer in kv_source_layer_ids`
   (layers 2, 8); an `Indexer` iff `layer < n_layers and layer in index_source_layer_ids`
   (layers 2, 8, 12).
6. `Indexer.owns_k` iff the layer is also a kv_source. Count of `wq_b`/`weights_proj` == 3;
   `wk` == 2.
7. **NEW.** `ffn.w1/w2/w3` shapes are exactly `[MoEI, d]`, `[d, MoEI]`, `[MoEI, d]`.
8. **NEW.** Every `ffn` has exactly 3 parameters. No `ModuleList`, no `bias`, no `bias_vl`.
9. **NEW.** `head` is not a Parameter; exactly one `[V, d]` tensor exists and
   `model.head.embed is model.embed`. Parameter count uses `V*d`, not `2*V*d`.

**Prefill**

10. Layers execute in ascending order; shared-state slots are never reset between forwards.
11. The indexer runs **before** the latent is rotated and quantized.
12. Compressed indices are offset by the number of window rows; unreachable entries are `-1`.
13. The window ring is seeded with the trailing `n_win` rows, `cutoff = S % n_win`.
14. Ratio-2 compressors move the trailing `rem` tokens into `pool_state`, emit `S // ratio`.

**Decode**

15. `start_pos` is threaded explicitly; every cache index derives from it.
16. Ring slot `start_pos % n_win`; compressed slot `start_pos // ratio`; indexer prefix
    `end // ratio`.
17. Latent RoPE frequency: prefill `freqs[:S - S % ratio : ratio]`, decode
    `freqs[start_pos + 1 - ratio]`.
18. A ratio-2 compressor returns `None` whenever `(start_pos + 1) % 2 != 0`.

**Attention math**

19. `sparse_attn` input `topk_idxs` is contiguous `int32`; `-1` masked; max floor `-1e30`.
20. `attn_sink` enters the denominator only and has no value vector.
21. Inverse RoPE is applied to the output tail before `wo_a`.
22. `wo_a` is block-diagonal over `o_groups`;
    `einsum("bsgd,grd->bsgr", o, W.view(G,R,Dg))`.
23. **NEW.** `o_groups | (H·hd)` and `o_groups | o_lora_rank`. (2816 % 4 = 0; 352 % 4 = 0.)

**FFN**

24. `swiglu_limit == 10.0`; `up` clamped two-sided, `gate` clamped upper-only, product fp32.
25. **NEW.** Every `ffn` parameter receives a gradient on every step. There is no routing
    sparsity to assert.

**Tokenizer**

26. **NEW.** `bos/eos/pad/dspark_noise/image` ids match the qt-VI.1.3 map (§16.4) and are
    *loaded from the tokenizer*, never hard-coded in two places.
27. **NEW.** `dspark_noise_token_id` is an id the tokenizer provably never emits.
28. **NEW.** `engram_compressed_vocab_size` is measured from the tokenizer and satisfies
    `0 < compressed < engram_vocab_size` (50021).

**Engram**

29. Primes are unique across `(layer, n-gram size, head)` and drawn upward from
    `engram_vocab_size`.
30. `sum(primes of layer) == engram_num_embeddings[layer]` → 600766.
31. Multipliers are odd; `token * multiplier` cannot overflow int64.
32. Image tokens are `DEAD` in the hash cache; blocked slots read the compressed pad id; the
    gate is forced to 0 on masked positions.
33. **NEW.** `engram_head_dim % 32 == 0` (the `ParallelEngramEmbedding` row-scale block).

**DSpark**

34. `layer_id = n_layers + stage_id`; draft layers assert `compress_ratio == 0`.
35. Draft block: position 0 is the real token, positions 1.. are `dspark_noise_token_id`.
36. Prefill only seeds the draft window cache; no draft attention, no FFN.
37. DSpark blocks share the backbone `embed` (and therefore the tied head).
38. **NEW.** `len(mtp) == 1`, and `mtp[0]` carries `main_proj`, `main_norm`, `markov_head`
    and `confidence_head` simultaneously.

**Numerics**

39. `rms_norm_eps == 1e-20` everywhere it appears.
40. All RMSNorm statistics computed in fp32.
41. Fake-quantized values are exactly representable in bf16.
42. When `QUANT_MODE == "NONE"`, only quantization is skipped — sinks, RMSNorm eps, Sinkhorn,
    and **SwiGLU clamps** all remain.
43. **Quant block divisibility:** window block 32 | `hd`; compressed block 16 | `hd`;
    indexer block 32 | `index_head_dim`. (128 % 32 = 0, 128 % 16 = 0, 64 % 32 = 0.)
44. **Training vs inference code paths:** the released kernels rotate/quantize in place. A
    training implementation must use out-of-place variants; in-place ops on autograd inputs
    break backward.

---

## 20 Open questions and untested choices

Decisions this document makes that are **not** derived from the released model. Each is cheap
to ablate at this scale.

### 20.1 `H·hd = 2d` — the weakest choice in the preset

Released: `64·512 = 32768 = 6.4d` at 1M context. MoE preset: `1d` (starved on purpose).
`2d` here is a middle guess; alternatives in §16.2. The specific sub-question worth testing
first: **`hd = 256, H = 11`** — a similar parameter count but 2× wider heads. Both
`sparse_attn` and the DSA indexer benefit from head width, and `index_head_dim = 64` is
already narrow for a 512-wide released model.

### 20.2 `index_n_heads = 8, index_head_dim = 64`

Released: `32 × 128` at `d = 5120` (ratios `d/160`, `d/40`). Here: `8 × 64` at `d = 1408`
(`d/176`, `d/22`). The head count roughly follows the `d/160` rule; the head *dim* does not.
With only 3 indexers contributing 0.59M parameters total, there is budget to raise
`index_head_dim` to 128 and double retrieval capacity if retrieval is the bottleneck.

### 20.3 `qk_rope_head_dim = 32` (25% of `hd`)

Released: `64/512 = 12.5%`. MoE preset: `16/64 = 25%`. This preset keeps the MoE preset's
ratio at a larger `hd`, so it is strictly *more* positional than the released model. Not
obviously wrong — DSA compresses positions (`ratio ∈ {1,2}`), so more explicit position
encoding per head may help. Untested.

### 20.4 `sliding_window = 128`, `max_seq_len = 32768`, `index_topk = 512`

All carried from the released 1M-context configuration without rescaling to `L=16`,
`d=1408`. At 32K the window is 0.4% of the sequence, which makes the SWA branch cheap and the
global branch dominant — probably the right regime (§18.16, §18.17).

### 20.5 Tying: which optimizer for `embed.weight`?

Not resolved. See §11.3 — the released momentum + Sinkhorn-balanced algorithm was written for
a table that is *either* an embedding *or* a head, and tying makes it both. This needs a
decision and a test, not a default.

### 20.6 Whether a dense-vs-MoE ablation is worth running

The evidence in §1 is mixed and the scale is exactly where it is unsettled. Three arms, all on
the same token budget with the same tokenizer, data and schedule:

| arm | shape | stored | FLOP/tok | 8×H100 @35% |
|---|---|---|---|---|
| **dense** | `d=1408 L=16 V=32000`, `MoEI=3712` | 490.1M | 1.977 | 1.9 h |
| MoE, matched total | same `d, L, V`, `E=32` top-2, `MoEI` solved to match | ~490M | ~0.24 | ~0.7 h |
| dense, matched FLOP | candidate M (§4.5) | 163.3M | 0.350 | ~1.1 h |

Total ≈ **4 hours on 8×H100**. Note the second arm is not free: at matched total and matched
`d`, its `MoEI` must be far below `8/3·d` to fit the same budget in a routed bank, so it is
not a like-for-like FFN width. That is inherent to matched-total MoE comparisons and worth
being explicit about when reading the result.

### 20.7 Whether `L=16` is really better than `L=20`

§4.6 argues it is, on both launch count and FLOPs. But the launch half of that argument is
eager-mode reasoning, and you are running `torch.compile`. Under compilation the launch
advantage shrinks to whatever survives fusion. The FLOP half of the argument (1.977 vs 2.084
GFLOP/token) is independent of execution model and stands on its own — but it is only 5%.
Measure before treating `L=16` as settled (§17.8).

---

## 21 Appendices

### 21.1 Full parameter script

The script in §5.1 is the runnable version and handles both presets. Config fragments:

```python
MOE_PRESET = dict(
    dim=512, n_layers=16, n_heads=8, head_dim=64, vocab=65536,
    q_lora_rank=128, o_lora_rank=128, o_groups=4, hc_mult=4,
    ffn_intermediate_size=416, tied=False,
    compress_ratios=tuple([0, 0] + [2]*6 + [1]*8 + [0, 0]),
    kv_source=(2, 8), index_src=(2, 8, 12),
    index_n_heads=4, index_head_dim=32,
    engram_base=50021, engram_max_ngram=4, engram_n_heads=4, engram_head_dim=64,
    engram_layers=(1,),
    dspark_stages=2, dspark_targets=(13, 14, 15), dspark_markov_rank=64,
    n_routed_experts=32, num_experts_per_tok=2, n_shared_experts=1, dspark_experts=16,
)

PRESET_VERTEX_500M = dict(
    dim=1408, n_layers=16, n_heads=22, head_dim=128, vocab=32000,
    q_lora_rank=352, o_lora_rank=352, o_groups=4, hc_mult=4,
    ffn_intermediate_size=3712, tied=True,
    compress_ratios=tuple([0, 0] + [2]*6 + [1]*8 + [0]),
    kv_source=(2, 8), index_src=(2, 8, 12),
    index_n_heads=8, index_head_dim=64,
    engram_base=50021, engram_max_ngram=4, engram_n_heads=4, engram_head_dim=128,
    engram_layers=(1,),
    dspark_stages=1, dspark_targets=(15,), dspark_markov_rank=64,
    n_routed_experts=0, dspark_experts=0,
)
```

### 21.2 Verification output

```
>>> count(MOE_PRESET)
backbone  414,366,880   (spec §2: 414.4M)   PASS
engram     40,419,200   (spec §2:  40.4M)   PASS
dspark     32,083,517   (spec §2:  31.9M)   PASS
total     486,869,597   (486.9M)
active/token  40,128,480  (8.2% of stored)   0.241 GFLOP
layer      21,693,758 = ffn 21,102,656 (97.3%) | attn 491,720 (2.3%) | hc 98,358 (0.5%)

>>> count(PRESET_VERTEX_500M)
backbone  375,691,328   (76.66%)
engram     87,722,752   (17.90%)
dspark     26,675,309   ( 5.44%)
total     490,089,389   (490.09M)
embed (tied)  45,056,000  (9.19%)
active/token 329,503,424  (67.2% of stored)  1.977 GFLOP
layer      20,593,964 = ffn 15,679,488 (76.1%) | attn 4,641,270 (22.5%) | hc 270,390 (1.3%)

prefill / decode   164,751,712 / 329,503,424    (exactly 1:2)
budget  9.8B tokens @20/param | 19,378 PFLOP | 1.9 h on 8xH100@35% MFU
KV      144 B/token global + 256 KiB window rings -> 4.72 MB at 32768 context
act     924 KB/token -> 3.78 GB @ T=4096, 15.14 GB @ T=16384
```

### 21.3 Worked trace at the new geometry

Decoder layer 9 (a Reuse layer, `ratio=1`, `hd=128`, `n_win=128`, `index_topk=512`), decode
step at `start_pos = 4096`:

```
window branch:
  kv = kv_norm(wkv(x))                      [B, 1, 128]
  rope tail (last 32 channels) at position 4096
  act_quant FP8 block 32                    amax floor 1e-4
  window_kv_cache[:B, 4096 % 128 == 0] = kv[:, 0]
  attend the whole ring: 128 slots, oldest first from slot 1

compressed branch:
  layer 9 is NOT a kv_source      -> reads shared.compress_kv   (layer 8's)
  layer 9 is NOT an index_source  -> reads shared.topk_idxs    (layer 8's)
  compress_len = (4096 + 1) // 1 = 4097
  topk_idxs already published by layer 8, offset by window length 128

sparse_attn:
  q [B,1,22,128] (last 32 channels rotated at position 4096)
  kv [B, 128 + 4097, 128]
  topk_idxs [B, 1, 128 + 512] contiguous int32, -1 masked
  scores = einsum("bmhd,bmkd->bmkh", q, kv_gathered) * 2**-3.5
  mx = amax(-1).clamp_min(-1e30)
  den = exp(sc - mx).sum(-1) + exp(attn_sink - mx)        # sink: denominator only
  o = einsum(...)/den

output:
  o[..., -32:] inverse-rope'd at position 4096
  view(B,1,4,-1)                                  # o_groups = 4
  einsum("bsgd,grd->bsgr", o, wo_a.view(4, 352, 704))
  wo_b -> [B, 1, 1408]
```

### 21.4 Note on `mHC-lite.md`

**This section was substantially rewritten.** An earlier draft claimed mHC-lite buys a
3.4× reduction in kernel launches and was therefore mandatory. That was **eager-mode
reasoning** and does not apply to your execution model.

`hc_split_sinkhorn` is 19 iterations of `comb / (comb.sum(-1) + eps)` on a `[B, S, 4, 4]`
fp32 buffer — a chain of elementwise and reduction ops on one small buffer. That is the
canonical `torch.compile`/inductor fusion case, and it will very likely collapse to 1–3
kernels. The launch advantage largely evaporates once compiled.

What remains is real but smaller:

| benefit | holds under `torch.compile`? |
|---|---|
| 3.4× fewer eager launches | **mostly no** — inductor fuses them |
| **exactness** — Birkhoff by construction, no 19-iteration approximation | **yes**, and it is now the main argument |
| HLO graph size (~200 ops → ~8 for that function) | **yes** — less to trace |
| compile time across 17 compiled `Block`s | **yes** |

**Recommendation: measure, then decide.** Run §17.8 first. If post-compile mHC launches are
≤3, adopt mHC-lite for exactness and compile time, not throughput. If they are still ≳100,
inductor did not fuse the Sinkhorn and the throughput case is back.

The parameter cost grows slightly: `+64d + 16` per layer, × 16 layers = **+1.47M** (0.30% of
the model) — still negligible. The `hc_sinkhorn_iters: 20` config key becomes unused.

**Recommendation on the rest of `mHC-lite.md`: unchanged** — its §5.3 cost table needs its
`d=512` / `414.4M` figures recomputed for `d=1408` / `375.7M`, and §7's "what would change in
the V4.1 spec" list still applies verbatim if you adopt it.

---

## 22 Sources

1. **Architecture of record.** *DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache
   Compression*, arXiv:2609.19969v1; released `config.json`; reference
   `inference/{model,engram,vision,kernel}.py`; vLLM / SGLang / NeMo recipes. Local copy:
   `DeepSeek-V4.1-Flash_Architecture_Spec.md`.
2. **Tokenizer.** *QT VI.1.3 UltraLingo (32K, Prelude)*, Quartz (AENEA Global), 2026 —
   `JamesQuartz/qt-VI.1.3-32k`. 32,000-token two-stage byte-level BPE, 86 reserved specials
   at ids 0–85, 72 `<|lang:xx|>` tags, FLORES-200 204/204 coverage.
3. **Dense vs sparse at small scale.** Wael, *Dense vs Sparse Pretraining at Tiny Scale:
   Active-Parameter vs Total-Parameter Matching*, arXiv:2605.13769 (May 2026). Matched-total
   dense beats MoE by 0.018 nats at sub-25M.
4. **DeepSeek's own upper bound.** Dai et al., *DeepSeekMoE*, arXiv:2401.06066. "DeepSeekMoE
   2B nearly approaches the performance of its dense counterpart with the same number of
   total parameters, which sets the strict upper bound of MoE language models."
5. **Small-data tasks favour dense.** Zoph et al., *ST-MoE*, arXiv:2202.08906.
6. **Data availability decides sparsity.** Nakamura et al., *Optimal Sparsity of MoE Language
   Models*, arXiv:2508.18672.
7. **MoE wants more data at fixed compute.** Ling et al., *Scaling Laws for Efficient MoE
   Language Models*, arXiv:2507.17702.
8. **Counter-evidence.** Huber et al., *CoSMoEs: Compact Sparse Mixture of Experts*, ALVR 2026
   (Meta Reality Labs) — MoE > dense at 125M–350M on-device.
9. **Sparse attention lineage.** DeepSeek Sparse Attention, arXiv:2512.02556.
10. **Hyper-connections.** Xie et al., arXiv:2512.24880 (mHC); Zhu et al., arXiv:2409.19606
    (Hyper-Connections); Yang & Gao, *mHC-lite*, arXiv:2601.05732 (see `mHC-lite.md`).
11. **Token budget.** Hoffmann et al., arXiv:2203.15556 — 20 tokens/parameter.
12. **Sparse backward update is the real MoE training cost.** Scherrer et al., *Dense
    Backpropagation Improves Training for Sparse Mixture-of-Experts*, NeurIPS 2025
    (arXiv:2411.00666).
