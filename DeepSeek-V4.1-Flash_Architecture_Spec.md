# DeepSeek-V4.1-Flash — Complete Architecture Specification

**Target model:** `deepseek-ai/DeepSeek-V4.1-Flash` (a.k.a. "DeepSeek V4.1 Flash", model_type `deepseek_v41`).
**Purpose of this document:** a single, self-contained, implementation-grade specification of the full architecture, sufficient for an engineer or an offline LLM to re-implement the model from scratch (forward pass, inference cache, quantization, serving architecture, weight layout) without access to the DeepSeek source.

**Status of facts.** Every numeric value, formula, and index convention in this document is sourced from one of: (a) the official technical report *DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression* (arXiv:2609.19969v1); (b) the released `config.json` on HuggingFace; (c) DeepSeek's own reference inference implementation (`inference/model.py`, `inference/engram.py`, `inference/vision.py`, `inference/kernel.py`); (d) the vLLM / SGLang / NeMo serving recipes and an independent port's parity notes. Where the document explains a design choice it is labeled `[RATIONALE]`. Where a value is a free parameter for a scaled-down reproduction it is labeled `[PRESET]`.

**How to use.** Read §1–§3 for scope, notation and configuration. §4 is the end-to-end dataflow. §5–§13 are the module specifications in dependency order. §14–§16 are quantization, serving, and checkpoint layout. §17 is an implementation blueprint with reference code. §18–§20 are pitfalls, parameter accounting, and appendices. Every algorithm that has a non-obvious index or rounding rule is given either exact pseudo-code or an exact code block.

---

## Table of contents

- §1 Model at a glance
- §2 Parameter accounting
- §3 Notation, tensor conventions, configuration
- §4 End-to-end dataflow
- §5 Hyper-Connections (mHC): residual stream, Single-Pass mHC, Sinkhorn comb — plus a note on the exact, kernel-free alternative (mHC-lite, see `mHC-lite.md`)
- §6 Mixture-of-Experts (MoE): gating, clamped SwiGLU, shared expert
- §7 Attention core: shared-KV latent, sliding window, grouped output projection
- §8 Compressed Sparse Attention 2 (CSA2): compressor, cross-layer reuse, indexer, hierarchical sparse indexer, FP4 KV
- §9 Rotary position embedding (partial RoPE, YaRN, compressed theta)
- §10 Causal Encoder–Decoder (CED): projected decoder KV, activation split, SWA bounded replay
- §11 Engram: conditional n-gram memory
- §12 DSpark: semi-autoregressive speculative decoding
- §13 Vision: DeepSeek-ViT and the Aligner
- §14 Quantization: formats, fake-quant QAT semantics, how to disable
- §15 Inference & serving architecture
- §16 Checkpoint layout and weight conversion
- §17 Implementation blueprint (file layout, reference code, verification)
- §18 Pitfalls and invariants checklist
- §19 Training-side algorithms that touch the architecture
- §20 Appendices

**Part II — Complete reference implementation and worked traces**

- §21 Full reference implementation (runnable PyTorch, single file)
- §22 Worked traces (numeric index arithmetic)
- §23 Scaling-down guide (deriving a preset from a parameter budget)
- §24 Bit-level quantization reference
- §25 Expanded invariants checklist
- §26 Reproducing the published numbers

---

## 1 Model at a glance

DeepSeek-V4.1-Flash is a **multimodal, causal Mixture-of-Experts Transformer** that reads images and text and generates text autoregressively. It is the first model in DeepSeek's post-V4 architecture family and is built around aggressive KV-cache compression for input-heavy, long-horizon agentic workloads.

### 1.1 Top-level facts

| Property | Value |
|---|---|
| Architecture | Causal Encoder–Decoder (CED), 40 layers = 20-layer causal encoder + 20-layer decoder |
| Backbone parameters | 552B (551.57B computed in §2) |
| Engram conditional-memory parameters | 196B (196.93B computed; sits outside the 552B backbone) |
| DSpark drafter parameters | ~14B (14.23B computed; sits outside the 552B backbone) |
| Total checkpoint | ~763B parameters, ~511 GB on disk (476 GiB) |
| Activated parameters per token | **8B during prefill, 16B during decode** |
| Context length | 1,048,576 tokens (1M) |
| Global KV cache | **890 bytes/token** (HBM-resident) ≈ 1/4 of DeepSeek-V4-Flash, ≈ 1/437 of V1 |
| Persistent KV cache | ≈ 1/8 of DeepSeek-V4-Flash (SSD/host) |
| Language backbone depth | 40 causal layers (both encoder and decoder halves are causal; there is no bidirectional encoder) |
| Hidden size | d = 5120 |
| Attention | 64 query heads, one shared KV "head" (MLA-like single latent), head_dim 512, partial RoPE on last 64 dims |
| Feed-forward | DeepSeekMoE: 1 shared expert + 384 routed experts, 6 routed active per token, SwiGLU with clamping |
| Quantization of released checkpoint | MoE expert weights FP4 (MX-style, E2M1); other weights FP8 (E4M3, 32×32 blocks, UE8M0 scales); embeddings + LM head BF16 |
| Pretraining corpus | 45T multimodal tokens (7:1 text:multimodal), 64K→1M sequence-length extension at 34T tokens |

### 1.2 What actually changed from DeepSeek-V4 (the "V4.1" delta)

DeepSeek conceptually treats V4 as "an SWA local backbone + a compressed global context." V4.1 keeps the SWA backbone and reworks the global branch:

1. **Causal Encoder–Decoder (CED).** The 40 layers split into a 20-layer causal encoder and a 20-layer decoder. Decoder global KV is projected from the final encoder hidden states (not recomputed per decoder layer). This is what makes prefill activate only 8B params/token while decode activates 16B (§10).
2. **Compressed Sparse Attention 2 (CSA2).** Replaces V4's CSA+HCA hybrid. Adds cross-layer reuse of main KV, indexer K, and Top-K indices via three static per-layer modes (Full / Reindex / Reuse), plus a Hierarchical Sparse Indexer in the decoder that bounds deeper indexer cost independently of context length (§8).
3. **FP4 main KV cache** (E2M1 with one E4M3 scale per 16 channels) instead of FP8, halving HBM cache again (§14).
4. **SWA Bounded Replay** replaces persisting SWA KV to SSD; SWA KV is reconstructed by replaying only the most recent `n_win` tokens (§15).
5. **Single-Pass mHC** + **Mega-mHC** deployment kernel — halves hyper-connection activation memory traffic (§5).
6. **Engram** conditional n-gram memory added at two layers (196B params) (§11).
7. **DSpark** replaces the trained MTP module for speculative decoding (§12).
8. **Native multimodality from the start of pre-training** (DeepSeek-ViT trained from scratch) (§13).
9. Removed from V4: per-head q RMS-normalization, compressor absolute-position (APE) slot embedding, ratio-4 overlapping spans, the indexer Hadamard rotation, and hash routing (`tid2eid`) — every MoE layer now scores all tokens.

### 1.3 Cross-layer sharing map (the CSA2 schedule)

From the released `config.json`:

```
compress_ratios (43 entries, one per layer incl. 3 DSpark stages):
  idx : 0  1 | 2 .................... 19 | 20 .................... 39 | 40 41 42
  val : 0  0 | 2  (x18, encoder CSA2) | 1  (x20, decoder CSA2)     | 0  0  0

kv_source_layer_ids     = [2, 8, 14, 20]          # own a Compressor + compressed-KV cache
index_source_layer_ids  = [2, 8, 14, 20, 24, 28, 32, 36]  # own an Indexer (produce Top-K)
candidate_source_layer_id = 20                      # builds the Hierarchical candidate pool
```

Per-layer behavior:

| layers | compress ratio | compressed KV read from | Top-K selection from | index keys from | CSA2 mode |
|---|---|---|---|---|---|
| 0–1 | 0 | — (window only) | — | — | sliding-only |
| 2–7 | 2 | layer 2 | layer 2 | layer 2 | Full@2, Reuse@3–7 |
| 8–13 | 2 | layer 8 | layer 8 | layer 8 | Full@8, Reuse@9–13 |
| 14–19 | 2 | layer 14 | layer 14 | layer 14 | Full@14, Reuse@15–19 |
| 20–23 | 1 | layer 20 | layer 20 | layer 20 | Full@20, Reuse@21–23 |
| 24–27 | 1 | layer 20 | **layer 24** | layer 20 | Reindex@24, Reuse@25–27 |
| 28–31 | 1 | layer 20 | **layer 28** | layer 20 | Reindex@28, Reuse@29–31 |
| 32–35 | 1 | layer 20 | **layer 32** | layer 20 | Reindex@32, Reuse@33–35 |
| 36–39 | 1 | layer 20 | **layer 36** | layer 20 | Reindex@36, Reuse@37–39 |

Interpretation: there are 4 index-key producers (layers 2, 8, 14, 20) but 8 index-query producers. Layers 24/28/32/36 own an indexer query and weights projection but **no keys** — they rescore layer 20's key cache with their own learned weights, producing fresh Top-K selections over the *same* compressed KV. Hence the checkpoint contains 8 × `indexer.wq_b`/`weights_proj` but only 4 × `indexer.wk`. Retrieval decisions are made twice as often as keys are stored.

---

## 2 Parameter accounting

Derived from the released configuration. Reproduce with the script in §2.3.

### 2.1 DeepSeek-V4.1-Flash (released)

Per-layer (identical across all 40 backbone layers):

| component | formula | count |
|---|---|---|
| attention (q LoRA + shared KV + grouped out) | `d·q_lora + q_lora·H·hd + d·hd + (H·hd/ogr)·(ogr·o_lora) + ogr·o_lora·d` | 126,617,408 |
| routed experts | `384 · 3 · 5120 · 2304` | 13,589,544,960 |
| shared expert | `3 · 5120 · 2304` | 35,389,440 |
| hyper-connections (attn+ffn fn/base/scale) | `2·((2+hc)·hc·hc·d + (2+hc)·hc + 3)` with hc=4 | 983,094 |
| gate | `384·5120 + 384·(bias+bias_vl)` | ~1.97M |

Whole-model:

```
backbone (embed + head + 40 × (attn + routed + shared + HC + norms + gate))
      + 8 indexers (wq_b + weights_proj) + 4 index-key projections (wk + k_norm)
      + 4 compressors (2 ratio-2 wkv+wgate, 2 ratio-1 wkv)
  ≈ 551.57B      ("552B" in the report)
engram tables + projections                                  ≈ 196.93B  ("196B")
DSpark (3 stages × MoE-128 + markov + confidence)            ≈ 14.23B   ("~14B")
                                                            ─────────
total                                                          ≈ 762.7B  (vLLM: 763.2B)
```

### 2.2 Activation split

- **Decode (16B/token):** all 40 layers run for every generated token; each MoE layer activates 1 shared + 6 of 384 routed experts.
- **Prefill (8B/token):** only the 20-layer encoder runs over the prompt to produce global KV; the decoder half is not run over the full prompt. Its global KV is already available (projected from the encoder output), and the decoder's per-layer sliding-window KV is reconstructed on demand by **Decoder SWA Bounded Replay** over only the last `n_win` tokens (§10.3). So the prompt token positions "activate" only the encoder's ≈8B worth of weights.

### 2.3 Reproducing the parameter count

```python
def count_flash():
    d, L, H, hd, V, MoEI = 5120, 40, 64, 512, 129280, 2304
    qlora, ogr, olora, hc = 1280, 8, 1024, 4
    attn = d*qlora + qlora + qlora*H*hd + d*hd + hd + (H*hd//ogr)*(ogr*olora) + ogr*olora*d + H
    routed = 384*3*d*MoEI; shared = 3*d*MoEI
    hc_p = 2*((2+hc)*hc*(hc*d) + (2+hc)*hc + 3)
    idx = 32*128*qlora + 32*d            # per indexer
    comp = 3*(d*hd) + 3*(d*hd)            # 2 ratio-2 (wkv+wgate) + 2 ratio-1
    backbone = V*d + V*d + L*(attn+routed+shared+hc_p+2*d+384*d+2*384) + 8*idx + 4*(512*128) + comp
    # engram: two tables, rows = [384006168, 384016682], dim 256, + wkv + q/k_weight
    # dspark: 3 stages × (128·3·d·2304) + markov(vocab×256×2) + confidence + main_proj
    ...
```
(The full runnable script is reproduced in §20.1.)

---

## 3 Notation, tensor conventions, configuration

### 3.1 Symbols

| symbol | meaning |
|---|---|
| `B` | batch size |
| `T` / `S` | sequence length of the current forward chunk (prefill chunk or 1) |
| `L` | number of backbone layers (40 released) |
| `l` | layer index `0 … L-1` |
| `d` | hidden size (5120) |
| `H` | number of query heads (64) |
| `hd` | head_dim (512) |
| `H_kv` | number of KV heads — always **1** (shared latent) |
| `rd` | rope_head_dim, the number of trailing head channels that carry RoPE (64) |
| `nd = hd − rd` | non-rotary head channels (448) |
| `V` | vocabulary size (129,280) |
| `E` | routed experts per layer (384) |
| `k` | routed experts activated per token (6) |
| `MoEI` | expert intermediate size (2304) |
| `hc` | hyper-connection stream count (4) |
| `n_win` | sliding-window size (128) |
| `m` | compression ratio for a layer (0 / 1 / 2) |
| `start_pos` | absolute position of the first token in the current chunk (0 during prefill, >0 during decode) |
| `ratio` | the layer's `compress_ratios[layer_id]` |

### 3.2 Tensor layout conventions

- Activations are `[B, S, ...]` throughout the reference implementation. Query tensors are `[B, S, H_local, hd]`. The KV tensor is `[B, N_kv, hd]` — a **single** latent per position, reused as both key and value.
- Hyper-connection state `x` is `[B, S, hc, d]`.
- MoE consumes flattened token rows `[B*S, d]`.
- Integer index tensors (`topk_idxs`, window indices) are `int32` and **contiguous**.
- `-1` in any index tensor means "masked / nothing there".

### 3.3 dtype policy

| tensor / operation | dtype | note |
|---|---|---|
| RMSNorm statistics and output scale | fp32 | `x.float()`; eps = `rms_norm_eps` = **1e-20** (effectively zero; matched exactly) |
| MoE gate scores | fp32 | `sqrtsoftplus` applied in fp32 |
| ratio > 1 compressor (`wkv`, `wgate`) and pooling | fp32 | softmax pooling needs the headroom |
| ratio = 1 compressor `wkv` | bf16 | plain projection, no gate |
| SwiGLU gate/up products | fp32 | clamped before the product |
| HC coefficient prediction (`hc_*`) | fp32 | |
| `attn_sink`, `gate.bias`, `gate.bias_vl` | fp32 | |
| index score / top-k / candidate masks | fp32 | `-inf` for unreachable |
| window KV cache | FP8 E4M3, block 32, UE8M0 scale | fake-quantized in place |
| compressed KV cache | FP4 E2M1, block 16, E4M3 scale | fake-quantized in place |
| indexer Q/K | FP4 E2M1, block 32, UE8M0 scale | fake-quantized in place |
| LM head logits | fp32 | weights stored fp32 in the reference even though the checkpoint ships bf16 |

### 3.4 Full released configuration (`text_config`)

Verbatim from the released `config.json`. A reproduction must treat every key below as part of the architecture contract.

```json
{
  "architectures": ["DeepseekV41ForCausalLM"],
  "model_type": "deepseek_v41",
  "dtype": "bfloat16",
  "bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 2,
  "image_token_id": 129264,
  "quantization_config": {
    "quant_method": "fp8", "activation_scheme": "dynamic",
    "weight_block_size": [32, 32], "scale_fmt": "ue8m0", "expert_dtype": "fp4"
  },
  "text_config": {
    "model_type": "deepseek_v41_text",
    "vocab_size": 129280,
    "hidden_size": 5120,
    "moe_intermediate_size": 2304,
    "num_hidden_layers": 40,
    "num_attention_heads": 64,
    "num_key_value_heads": 1,
    "head_dim": 512,
    "qk_rope_head_dim": 64,
    "q_lora_rank": 1280,
    "o_lora_rank": 1024,
    "o_groups": 8,
    "hidden_act": "silu",
    "swiglu_limit": 10.0,
    "rms_norm_eps": 1e-20,
    "attention_bias": false,
    "attention_dropout": 0.0,
    "initializer_range": 0.02,
    "use_cache": true,
    "tie_word_embeddings": false,
    "max_position_embeddings": 1048576,
    "rope_theta": 10000,
    "rope_scaling": {
      "rope_type": "yarn", "factor": 16, "beta_fast": 32, "beta_slow": 1,
      "original_max_position_embeddings": 65536
    },
    "n_routed_experts": 384,
    "n_shared_experts": 1,
    "num_experts_per_tok": 6,
    "scoring_func": "sqrtsoftplus",
    "topk_method": "noaux_tc",
    "norm_topk_prob": true,
    "routed_scaling_factor": 1.5,
    "sliding_window": 128,
    "compress_ratios": [
      0,0,
      2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
      1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
      0,0,0
    ],
    "compress_rope_theta": 160000,
    "kv_source_layer_ids": [2, 8, 14, 20],
    "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
    "index_n_heads": 32,
    "index_head_dim": 128,
    "index_topk": 512,
    "candidate_source_layer_id": 20,
    "candidate_topk_blocks": 2048,
    "candidate_block_size": 8,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
    "engram_layer_ids": [1, 14],
    "engram_num_embeddings": [384006168, 384016682],
    "engram_max_ngram_size": 4,
    "engram_vocab_size": 16000000,
    "engram_n_heads": 8,
    "engram_head_dim": 256,
    "engram_pad_token_id": 2,
    "engram_compressed_vocab_size": 99092,
    "num_nextn_predict_layers": 3,
    "dspark_block_size": 5,
    "dspark_noise_token_id": 128799,
    "dspark_target_layer_ids": [37, 38, 39],
    "dspark_markov_rank": 256,
    "dspark_n_routed_experts": 128,
    "dspark_num_experts_per_tok": 3
  },
  "vision_config": {
    "model_type": "deepseek_v41_vision",
    "num_hidden_layers": 32,
    "hidden_size": 1024,
    "num_attention_heads": 16,
    "intermediate_size": 2816,
    "patch_size": 14,
    "rope_theta": 10000,
    "downsample_ratio": 3,
    "max_image_tokens": 1024,
    "min_pixels": 295936,
    "max_wh_ratio": null
  }
}
```

### 3.5 Attention head geometry derived from the config

```
H · hd            = 64 · 512   = 32768  (q width)
d                 = 5120
q path:   wq_a: 5120 → 1280 ; q_norm(1280) ; wq_b: 1280 → 32768
kv path:  wkv:  5120 → 512   ; kv_norm(512)     # single latent, K == V
out path: wo_a: (32768/8 = 4096) → (8 · 1024 = 8192), block-diagonal over 8 groups
          wo_b: 8192 → 5120
RoPE:     last qk_rope_head_dim = 64 channels of each 512-dim head (448 non-rotary)
softmax_scale = hd ** -0.5 = 2 ** -4.5
```

### 3.6 Vertex-500M preset `[PRESET]`

A scaled reproduction that preserves **every architectural mechanism** (CED split, CSA2 three modes, hierarchical indexer, hyper-connections, Engram, DSpark, FP4/FP8 fake-quant, modality routing) at a size that trains on a single multi-GPU node. Totals: backbone 414.4M, Engram 40.4M, DSpark 31.9M → **≈487M total**. Ratio of routed-expert share stays ≈ the released model's (98%+ of the backbone).

| key | released | Vertex-500M |
|---|---|---|
| `vocab_size` | 129280 | 65536 |
| `hidden_size` | 5120 | 512 |
| `num_hidden_layers` | 40 (20+20) | **16 (8+8)** |
| `num_attention_heads` | 64 | 8 |
| `head_dim` | 512 | 64 (8·64 = 512 = d ✓) |
| `qk_rope_head_dim` | 64 | 16 |
| `q_lora_rank` | 1280 | 128 |
| `o_lora_rank` | 1024 | 128 |
| `o_groups` | 8 | 4 |
| `sliding_window` | 128 | 64 |
| `n_routed_experts` / top-k | 384 / 6 | 32 / 2 |
| `moe_intermediate_size` | 2304 | 416 |
| `n_shared_experts` | 1 | 1 |
| `compress_ratios` | `[0,0, 2×18, 1×20, 0×3]` | `[0,0, 2×6, 1×8, 0×2]` |
| `kv_source_layer_ids` | `[2,8,14,20]` | `[2,8]` |
| `index_source_layer_ids` | `[2,8,14,20,24,28,32,36]` | `[2,8,12]` |
| `candidate_source_layer_id` | 20 | 8 |
| `candidate_topk_blocks` / `candidate_block_size` | 2048 / 8 | 256 / 8 |
| `index_n_heads` / `index_head_dim` / `index_topk` | 32 / 128 / 512 | 4 / 32 / 128 |
| `hc_mult` / `hc_sinkhorn_iters` / `hc_eps` | 4 / 20 / 1e-6 | 4 / 20 / 1e-6 |
| `engram_layer_ids` | `[1,14]` | `[1]` |
| `engram_vocab_size` (prime seed) | 16,000,000 | 50,021 |
| `engram_num_embeddings` | `[384006168, 384016682]` | computed by §11.2 at build time (= sum of the layer's primes) |
| `engram_n_heads` / `engram_head_dim` | 8 / 256 | 4 / 64 |
| `engram_max_ngram_size` | 4 | 4 |
| `engram_compressed_vocab_size` | 99092 | size of *your* tokenizer's compressed map (§11.2) |
| `num_nextn_predict_layers` (DSpark stages) | 3 | 2 |
| `dspark_block_size` | 5 | 4 |
| `dspark_target_layer_ids` | `[37,38,39]` | `[13,14,15]` |
| `dspark_markov_rank` | 256 | 64 |
| `dspark_n_routed_experts` / top-k | 128 / 3 | 16 / 2 |
| `rope_theta` / `compress_rope_theta` | 10000 / 160000 | 10000 / 160000 |
| `max_position_embeddings` | 1048576 | 32768 |
| `rope_scaling.original_max_position_embeddings` | 65536 | 8192 |
| `rope_scaling.factor` | 16 | 16 |
| `rms_norm_eps` / `swiglu_limit` / `routed_scaling_factor` | 1e-20 / 10.0 / 1.5 | identical |
| `swiglu_limit` clamp behaviour | two-sided up, upper-only gate | identical |

Structural invariants that a preset **must** preserve:

- `compress_ratios[0] == 0` and `compress_ratios[1] == 0` (first two encoder layers are pure sliding-window).
- `L/2` is the encoder/decoder boundary; the first decoder layer (`l = L/2`) is always a `kv_source_layer` (it is where decoder global KV is born) **and** the `candidate_source_layer_id` when a hierarchical indexer is used.
- Every non-source CSA2 layer has `compress_ratio > 0` and reads the shared cache produced by the nearest source above it.
- Encoder CSA2 layers use `m = 2`; decoder CSA2 layers use `m = 1`.
- The number of `index_source_layer_ids` is greater than the number of `kv_source_layer_ids` (Reindex layers own no keys).
- `H · hd == d` is **not** required (released model has 64·512 ≫ 5120), but `o_groups` must divide `H · hd`.
- `index_head_dim` need not equal `hd`; released: `hd=512`, `index_head_dim=128`.

---

## 4 End-to-end dataflow

### 4.1 Preprocessing (outside the model)

1. Tokenize text into `input_ids` with the DeepSeek fast tokenizer.
2. For each image, resize so that `pixels ≥ min_pixels` and `pixels/...` respects `max_image_tokens = 1024`, then split into `14×14` patches. Image regions are represented in `input_ids` as a contiguous span of `image_token_id = 129264`; a parallel `token_types` tensor marks each position's role.
3. `token_types` encoding used by the reference `image_processor`:
   - `-1` → text token
   - `0..3` → image role codes: `IMAGE_START`, `IMAGE_NEW_LINE`, `IMAGE_END`, `IMAGE` (the raw image token)
   Any position with `token_types >= 0` is an "image token" and is flagged to the router (`image_mask`).
4. Engram requires the tokenizer to build its compressed token map (§11.2).

### 4.2 Forward pass outline

```
def forward(input_ids [B,S], token_types [B,S], pixel_values, image_grid_hws, start_pos):
    # ---- masks -------------------------------------------------------------
    image_mask  = token_types >= 0                      # [B,S] bool
    engram_mask = ~image_mask                          # image tokens take no n-gram

    # ---- Engram hash ids (whole-stack, computed once) ----------------------
    hash_ids = engram_hash_state(input_ids, start_pos, engram_mask)
        # -> [B, S, n_engram_layers, n_hash_cols]

    # ---- embed + vision merge ---------------------------------------------
    h = embed(input_ids)                               # [B,S,d]
    if vision_enabled:
        feats = vision(pixel_values, grid_h, grid_w)    # ViT  -> [n_patches, d_vis]
        emb   = aligner(feats, grid_h, grid_w)          # -> [n_image_tokens, d]
        h = merge_image_embeddings(h, token_types, emb,
                                   image_start, image_end, image_newline)

    # ---- open the hyper-connection stream ---------------------------------
    h = h.unsqueeze(2).repeat(1, 1, hc, 1)              # [B,S,hc,d]
    pre_mix = identity_pre_mix(B, S, hc)                # [B,S,hc], 1 at index 0

    main_hiddens = []
    for i, layer in enumerate(layers):                  # ascending order is REQUIRED
        if layer.engram is not None:
            h = layer.engram(h, hash_ids[:,:,layer.engram.layer_hash_index,:],
                             engram_mask)              # gated additive write (§11)
        if i in dspark_target_layer_ids:
            main_hiddens.append(h.mean(dim=2))          # *attention input* of target layer
        h, pre_mix = layer(h, start_pos, pre_mix, image_mask)

    # ---- head --------------------------------------------------------------
    h = layers[-1].hc_pre(h, pre_mix)                   # final collapse with last ffn pre
    logits = lm_head(norm(h))                           # [B,S,V] (fp32); last position suffices for decode
    return logits
```

**Ordering invariant.** Layers execute in ascending index order and there is exactly one shared-state slot per quantity (compressed KV, indexer K, Top-K indices, candidate mask). Correctness depends on a producer writing before its consumers read. Any parallel or reordered execution must materialize per-source state instead (§18).

### 4.3 What runs when (activation profile)

| phase | encoder layers 0–19 | decoder layers 20–39 | active params/token |
|---|---|---|---|
| prefill over prompt | full pass | **skipped**; decoder global KV is projected from encoder output, decoder SWA KV obtained by bounded replay of the last `n_win` tokens | ≈8B |
| decode (per token) | full pass | full pass | 16B |

A faithful *reference* implementation may simply run all 40 layers in both phases (this is what DeepSeek's own `inference/model.py` does — it contains no CED fast-path); the 8B/16B split is a **serving** property. See §10 and §15.

---

## 5 Hyper-Connections (mHC): residual stream, Single-Pass mHC, Sinkhorn comb

V4.1 uses DeepSeek's **manifold-constrained hyper-connections** (mHC): the residual stream is not a single vector but `hc` parallel copies mixed by token-wise, learned, doubly-stochastic coefficients.

> **Consider mHC-lite as well.** `hc_split_sinkhorn` (§5.4) enforces the doubly-stochastic constraint by *approximation* — 20 Sinkhorn-Knopp iterations, which do not guarantee the constraint holds. **mHC-lite** (arXiv:2601.05732) enforces the *same* constraint by *construction*, representing `H_res` as a convex combination of the `hc!` permutation matrices (Birkhoff-von Neumann): exact, one matmul, no custom kernels. It is a reparameterization, not a conversion — it breaks checkpoint parity with the released V4.1-Flash weights, since `W_res` goes from `[hc·d, hc²]` to `[hc·d, hc!]`. Its value grows as `d` shrinks, because the Sinkhorn loop costs `O(1)` in `d` while the projection it sits behind costs `O(d)`: ~7% overhead at `d=5120` versus ~26% at the `d=1344` the vertex500m preset actually uses (~68% at the `d=512` MoE preset). See **`mHC-lite.md`** for the full analysis, the exact drop-in implementation, the required spec edits, and the initialization recipe; §8.1 of that document recomputes the cost table at `d=1344`. **Nothing in this document has been changed** — `hc_sinkhorn_iters: 20` remains the architecture of record here.

### 5.1 Formulation

Let `X_l ∈ ℝ^{hc×d}` be the stream entering block `l`, and `F_l(·)` the block's attention or FFN sublayer. mHC computes token-wise coefficients from the stream itself:

```
X_{l+1} = B_l · X_l  +  C_l · F_l(A_l · X_l)          (original mHC, V4)
(A_l, B_l, C_l) = H(X_l)                               with
A_l ∈ ℝ^{1×hc},  C_l ∈ ℝ^{hc×1},  B_l ∈ ℝ^{hc×hc}
```

`H` = normalization + projection.

### 5.2 Single-Pass mHC (V4.1) — the one-block shift

V4.1 shifts the input-mixing coefficients by one block:

```
X_{l+1} = B_l · X_l  +  C_l · F_l(A_{l-1} · X_l)       (Single-Pass mHC, V4.1)
(A_l, B_l, C_l) = H(X_l)
```

so the attention/FFN input mixing uses `A_{l-1}` (the previous sublayer's pre-coefficients) while the residual/expansion uses `B_l, C_l` computed from the current stream. In code, this means the block returns its FFN `pre` coefficients and the **next** sublayer consumes them. Empirically the shift costs negligible degradation but lets the whole hyper-connection computation fuse into fewer kernels (Mega-mHC), halving activation memory traffic.

### 5.3 Coefficient prediction (`hc_mixes`)

Inputs: flattened stream `x = X.flatten(2).float()` of shape `[B, S, hc*d]`.

```
rsqrt  = rsqrt(mean(x^2, dim=-1, keepdim=True) + norm_eps)   # norm_eps = rms_norm_eps = 1e-20
mixes = Linear(x, hc_fn) * rsqrt                              # hc_fn: [mix_hc, hc*d], fp32
pre, post, comb = hc_split_sinkhorn(mixes, hc_scale[3], hc_base[mix_hc], hc_mult, iters, eps)
```

with `mix_hc = (2 + hc) * hc = 24` when `hc = 4`. Layout of the 24 outputs:

```
pre  [0 : hc]                       # [B,S,hc]
post [hc : 2hc]                     # [B,S,hc]
comb [2hc : 2hc + hc*hc]  reshaped [B,S,hc,hc]   # row/col orientation: comb[j,k] weights residual[j] into output[k]
```

Each sublayer has its **own** `hc_fn`, `hc_base`, `hc_scale` (attention and FFN are separate), all in fp32, sized `[mix_hc, hc*d]`, `[mix_hc]`, `[3]`.

### 5.4 `hc_split_sinkhorn` — exact algorithm

This is the kernel `hc_split_sinkhorn_kernel` in the reference. Reproduce exactly:

```
# mixes: [n, (2+hc)*hc] fp32 ; hc_scale: [3] ; hc_base: [(2+hc)*hc]
pre[j]  = sigmoid(mixes[j]          * hc_scale[0] + hc_base[j]) + eps      # eps = hc_eps = 1e-6
post[j] = 2 * sigmoid(mixes[j+hc]    * hc_scale[1] + hc_base[j+hc])
comb[j,k] = mixes[j*hc + k + 2hc]   * hc_scale[2] + hc_base[j*hc + k + 2hc]

# row-softmax then column-normalize:
row_max[j] = max_k comb[j,k]
comb[j,k] = exp(comb[j,k] - row_max[j])
comb[j,k] = comb[j,k] / sum_k comb[j,k] + eps
comb[j,k] = comb[j,k] / (sum_j comb[j,k] + eps)

# Sinkhorn: 20 iterations of alternating row/col normalization:
for _ in range(sinkhorn_iters - 1):        # total = 20 passes => 1 softmax + 1 col-norm + 19*(row+col)
    comb[j,k] = comb[j,k] / (sum_k comb[j,k] + eps)
    comb[j,k] = comb[j,k] / (sum_j comb[j,k] + eps)
```

The result is an approximately doubly-stochastic matrix (rows and columns sum to ≈1). This is the **only** place the "manifold constraint" is enforced.

### 5.5 Collapse / expand

```
hc_pre  (collapse to one sublayer input):
    x = sum_j pre[j] * X[j]                              # [B,S,hc] x [B,S,hc,d] -> [B,S,d]

hc_post (expand sublayer output back to hc copies):
    X_new[k] = post[k] * y  +  sum_j comb[j,k] * X_residual[j]   # y = sublayer output [B,S,d]
```

Both accumulate in fp32 then cast back to the input dtype.

### 5.6 Block structure and the staggered hand-off

Each backbone block (`Block`) does, **in this exact order**:

```
residual = x
attn_pre, attn_post, attn_comb = hc_mixes(x, hc_attn_fn, hc_attn_scale, hc_attn_base)  # predicts THIS block's attention coeffs
x = hc_pre(x, pre_mix)              # <-- uses the PREVIOUS sublayer's pre (Single-Pass shift)
x = attn_norm(x)                    # RMSNorm(d)
x = attention(x, start_pos, ...)    # §7, §8
x = hc_post(x, residual, attn_post, attn_comb)

residual = x
ffn_pre, ffn_post, ffn_comb = hc_mixes(x, hc_ffn_fn, hc_ffn_scale, hc_ffn_base)
x = hc_pre(x, attn_pre)             # <-- FFN uses THIS block's attention pre
x = ffn_norm(x)
x = ffn(x, image_mask)              # MoE, §6
x = hc_post(x, residual, ffn_post, ffn_comb)
return x, ffn_pre                   # <-- returns its pre for the next block
```

Chain of hand-offs (the "one sublayer late" rule):
- attention of block `l` uses `ffn_pre` produced by block `l-1` (identity one-hot at layer 0).
- FFN of block `l` uses `attn_pre` produced by block `l`'s attention.
- the LM head's final collapse uses the last block's `ffn_pre`.
- There is **no separate `hc_head_fn`** (V4 had one).

### 5.7 Mega-mHC (deployment kernel)

Mega-mHC fuses residual update + input mixing + coefficient prediction into one kernel. Memory-traffic analysis from the report:
- Ideal single map requires `(n+1)d` reads and `(n+1)d` writes → lower bound `(2n+2)d`.
- V4's three-kernel implementation, including pre-norm in `F_l`, moves `(4n+4)d` (2× the bound).
- Mega-mHC processes `X_l` in tiles along the hidden dim; each tile computes its mixed input and accumulates the partial sums needed to predict the *next* block's coefficients. It also folds in input pre-norm and FP8 conversion, so the residual is read once and written once, hitting the `(2n+2)d` bound for Single-Pass mHC and halving activation traffic vs. V4.

For a functional PyTorch implementation Mega-mHC is optional (it is a memory-traffic optimization, not a semantic change). Implement it only if you are targeting the published kernel counts.

---

## 6 Mixture-of-Experts (MoE): gating, clamped SwiGLU, shared expert

Every block contains one MoE FFN. Routing is `sqrtsoftplus` scoring with `noaux_tc` selection (auxiliary-loss-free, bias-corrected top-k).

### 6.1 Gate

```
scores = Linear_fp32(x, weight) / gate_temp          # weight: [E, d] (bf16 in ckpt, applied in fp32)
if scoring_func == "sqrtsoftplus":  scores = sqrt(softplus(scores))
elif scoring_func == "sigmoid":     scores = sigmoid(scores)
elif scoring_func == "softmax":     scores = softmax(scores)

# selection uses a learned correction bias; weighting uses the RAW scores
bias = bias_vl if token is an image token else bias      # separate per-modality bias (noaux_tc_for_vl)
indices = topk(scores + bias, k)                         # [n, k]
weights = gather(scores, indices)                        # unbiased scores at the chosen experts
if norm_topk_prob and k > 1:
    weights = weights / (sum(weights) + 1e-20)           # note: literally 1e-20, not rms_norm_eps
weights = weights * routed_scaling_factor                # × 1.5
```

Per-modality balance (training-side, §19): text and image tokens each get their own `bias` updated independently from their own expert loads; selection for an image token uses `bias_vl`, text uses `bias`.

### 6.2 Expert (clamped SwiGLU)

```
gate = w1(x)          # [n, MoEI]
up   = w3(x)          # [n, MoEI]
if swiglu_limit > 0:                  # limit = 10.0
    up   = clamp(up,   -limit, limit) # two-sided
    gate = clamp(gate, -inf,  limit) # upper-only
y = silu(gate) * up                   # fp32
y = weights[:, None] * y              # apply routing weight
out = w2(y)                           # [n, d]
```

The clamps exist to keep fp8/fp4 activations in range during training; they are part of the architecture, not a numerical afterthought.

### 6.3 Shared expert and aggregation

Every token additionally passes through exactly 1 shared expert with the same clamped SwiGLU (no routing weight). Token dispatch is per-expert gather/scatter:

```
y = zeros_like(x, fp32)
counts = bincount(indices)             # for expert-parallel fast paths
for e in local_experts:
    idx, top = where(indices == e)
    y[idx] += expert_e(x[idx]) * weights[idx, top]
all_reduce(y) if tensor-parallel        # experts are tensor-parallel across ranks
y += shared_expert(x)
return y.view_as_input
```

The reference loops over experts in Python (`torch.where(indices == e)`); a production implementation uses grouped GEMM. Semantics are identical.

### 6.4 Parameter shapes (per MoE layer)

```
routed expert e:  w1 [MoEI, d]   w2 [d, MoEI]   w3 [MoEI, d]      (SwiGLU: w1 gate, w3 up, w2 down)
shared expert:    same shapes, 1 instance
gate:             weight [E, d] ; bias [E] fp32 ; bias_vl [E] fp32 (only when vision enabled)
```

Released: 384 routed (top-6) + 1 shared, MoEI = 2304. DSpark draft layers reuse MoE with 128 experts (top-3).

---

## 7 Attention core: shared-KV latent, sliding window, grouped output projection

Attention in V4.1 is **latent attention over two KV sources at once**, concatenated into a single sparse-attention call:

1. a **sliding window** (last `n_win` = 128 tokens) of raw per-layer KV, and
2. when `compress_ratio > 0`, **`index_topk` = 512 compressed positions** reaching further back (selected by the indexer, §8).

Q and the output projection are both low-rank; the output projection is grouped and block-diagonal.

### 7.1 Parameters (identical in every attention layer)

```
wq_a   : [q_lora_rank, d]        = [1280, 5120]      low-rank query down-proj
q_norm : RMSNorm(q_lora_rank)     eps = 1e-20
wq_b   : [H*hd, q_lora_rank]     = [32768, 1280]     query up-proj (column-parallel)
wkv    : [hd, d]                 = [512, 5120]       single shared KV latent
kv_norm: RMSNorm(hd)             eps = 1e-20
wo_a   : [o_groups*o_lora_rank, (H*hd)//o_groups]  = [8192, 4096]  grouped low-rank out (bf16)
wo_b   : [d, o_groups*o_lora_rank] = [5120, 8192]                   (row-parallel)
attn_sink: [H] fp32               per-head learnable sink, enters the softmax DENOMINATOR only
softmax_scale = hd ** -0.5
```

**There is no separate value projection.** The single latent `kv_norm(wkv(x))` is used as both K and V for all `H` query heads (`num_key_value_heads = 1`), broadcast across heads — the MLA-style design of DeepSeek-V3/4.

**Partial RoPE:** only the last `qk_rope_head_dim` = 64 channels of each 512-dim head carry rotary position; the leading 448 are position-free (no positional embedding, no APE slot embedding — V4's APE was removed in V4.1).

**RoPE convention:** adjacent real/imag pairs (`view_as_complex` on the last dim split into `(…, 2)`); inverse RoPE is applied to the attention **output** tail before the output projection, so the cache can stay in one shared rotated form.

### 7.2 Per-layer forward

```
def forward(x [B,S,d], start_pos):
    freqs = freqs_cis[start_pos : start_pos+S]                  # [S, rd/2]
    qr = q_norm(wq_a(x))                                       # [B,S,q_lora_rank]
    q  = wq_b(qr).view(B,S,H,hd)                               # query
    q[..., -rd:] = rope(q[..., -rd:], freqs)                    # rotate only the tail

    kv, win_idxs = window_kv(x, freqs, start_pos)              # sliding-window branch, §7.3
    if compress_ratio > 0:
        ckv, c_idxs = compressed_kv(x, qr, start_pos, offset=kv.size(1))  # §8
        kv = cat([kv, ckv], dim=1)                             # [B, N_kv, hd]
        win_idxs = cat([win_idxs, c_idxs], dim=-1)

    o = sparse_attn(q, kv, attn_sink, win_idxs, softmax_scale)  # §7.4
    o[..., -rd:] = rope_inverse(o[..., -rd:], freqs)           # un-rotate the tail

    # grouped block-diagonal output projection (LoRA over o_groups head groups)
    o = o.view(B,S,n_local_groups,-1)
    wo_a_w = wo_a.weight.view(n_local_groups, o_lora_rank, -1) # [G, R, Dg]
    o = einsum("bsgd,grd->bsgr", o, wo_a_w)
    return wo_b(o.flatten(2))
```

### 7.3 Sliding-window branch (per-layer, unshared, never grows with context)

Every layer keeps its **own** raw-KV ring buffer of `n_win` slots:

```
window_kv(x, freqs, start_pos):
    kv = kv_norm(wkv(x))                                       # [B,S,hd]
    kv[..., -rd:] = rope(kv[..., -rd:], freqs)
    act_quant_fp8(kv)   # block-32 UE8M0, whole vector incl. RoPE tail; stays FP8

    if start_pos == 0:            # prefill: attend the current causal chunk, seed the ring
        if S <= n_win: window_kv_cache[:B, :S] = kv
        else:          # ring holds the last n_win of the chunk
            cutoff = S % n_win
            window_kv_cache[:B, cutoff:n_win] = kv[:, -n_win:][:, :n_win-cutoff]
            window_kv_cache[:B, :cutoff]       = kv[:, -n_win:][:, n_win-cutoff:]
        window_kv = kv                            # attend the chunk's raw KV
    else:                         # decode: one token into the ring, attend the whole ring
        window_kv_cache[:B, start_pos % n_win] = kv[:, 0]
        window_kv = window_kv_cache[:B]
    return window_kv, get_window_topk_idxs(n_win, B, S, start_pos)
```

Window index arithmetic (`get_window_topk_idxs`), materialized as `int32 [B, S_or_1, <=n_win]`:

```
prefill (start_pos == 0):
    end  = arange(S).unsqueeze(1)                          # query positions within chunk
    idxs = (end - n_win + 1).clamp(0) + arange(min(S, n_win))
    idxs = where(idxs > end, -1, idxs)                     # before sequence start
decode:
    oldest = start_pos % n_win + 1
    idxs = cat([arange(oldest, n_win), arange(oldest)])     # ring slots, oldest first
    idxs = where(idxs > start_pos, -1, idxs)                # ring still filling
```

`-1` marks an empty slot and is masked by `sparse_attn`.

### 7.4 `sparse_attn` — exact semantics

`sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)`:
- `q: [B, m, H, hd]` (rotated tail), `kv: [B, N, hd]` (single latent per position, rotated), `topk_idxs: [B, m, topk] int32` (positions into `kv`; `-1` = masked), `attn_sink: [H] fp32`.
- Algorithm (FlashAttention-style online softmax over gathered slots):
  - For each gathered slot `i` with `topk_idxs[i] == -1`: load `kv[i] = 0`, set score to `-inf`.
  - `acc_s = q · kv^T * softmax_scale`; running max/sum online.
  - `scores_max` initialized to **`-1e30`** (finite floor, not `-inf`) so an all-masked row produces `exp(-1e30 - -1e30) = 1` finite terms and yields an **all-zero output** rather than NaN.
  - After the loop, `sum_exp[h] += exp(attn_sink[h] - scores_max[h])` and `acc_o[h] /= sum_exp[h]`. **The sink contributes to the denominator only; it carries no value vector.**
  - Heads are padded to 16 for the kernel and stripped afterward.
- Output `o: [B, m, H, hd]`.

Negative controls (load-bearing behaviors verified in the reference port): the inverse RoPE on the output tail, and the sink-in-denominator-only rule (an all-masked row yields exactly zero). Both must be reproduced.

---

## 8 Compressed Sparse Attention 2 (CSA2)

CSA2 compresses along three multiplicative dimensions simultaneously:

1. **Entry size** — GQA / MLA-style single latent (§7).
2. **Sequence dimension** — every `m` tokens compressed into one entry (the Compressor).
3. **Layer dimension** — some layers reuse other layers' KV and/or Top-K selections (cross-layer reuse + index reuse), decoupled.

V4.1 uses **pure CSA2** (no HCA). Compression ratios `m ∈ {1, 2}` (V4 used {4, 128}).

### 8.1 Compressor (only `kv_source_layer_ids` own one)

Pools `m` consecutive tokens into one KV latent with a **learned softmax gate**:

```
Compressor.forward(x [B,S,d], start_pos) -> latent [B, S//m (completed), hd] | None
    ratio = m
    if ratio == 1:
        return norm(wkv(x))                          # plain projection, bf16 weights, no gate

    # ratio > 1 (weights fp32)
    kv    = wkv(x).float()                            # [B,S,hd]
    score = wgate(x).float()                          # [B,S,hd]
    if start_pos == 0:                               # prefill
        if S >= ratio:
            full, rem = S // ratio, S % ratio
            if rem:                                   # trailing partial group waits in state
                kv,    self.kv_state[:B,:rem]    = kv.split([full*ratio, rem], dim=1)
                score, self.score_state[:B,:rem] = score.split([full*ratio, rem], dim=1)
            kv    = kv.unflatten(1, (-1, ratio))      # [B, full, ratio, hd]
            score = score.unflatten(1, (-1, ratio))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)   # softmax over the group dim, fp32
        return norm(kv)
    else:                                             # decode: one token per step
        should_compress = (start_pos + 1) % ratio == 0
        slot = start_pos % ratio
        self.kv_state[:B, slot]    = kv[:, 0]
        self.score_state[:B, slot] = score[:, 0]       # score_state init -inf
        if should_compress:
            kv = (self.kv_state[:B] * self.score_state[:B].softmax(dim=1)).sum(dim=1, keepdim=True)
            return norm(kv)
        return None                                     # group still filling
```

Notes:
- `norm` is `RMSNorm(hd, eps=1e-20)` applied to the latent **before RoPE**.
- The `kv_state`/`score_state` buffers are `fp32`, shape `[max_batch, ratio, hd]`, `score_state` initialized to `-inf`, `persistent=False` (runtime state, not weights).
- A latent "stands for the **first** token of its group": group `j` carries RoPE position `j*ratio` and is written at cache slot `start_pos // ratio`.

### 8.2 Cross-layer reuse (the Full / Reindex / Reuse modes)

The reference threads one process-global object down the stack:

```
class SharedAttentionRuntime:
    compress_kv : Tensor | None      # published by kv_source layers
    index_k     : Tensor | None      # published by indexer-key owners (subset of kv_source)
    topk_idxs   : Tensor | None      # published by index_source layers
    candidates  : Tensor | None      # published by the candidate-source layer
```

Per-layer mode is a **static** assignment determined by config:

| mode | is_kv_source | is_index_source | behavior |
|---|---|---|---|
| **Full** | yes | yes | compute own compressed KV; project indexer K from it; run indexer → fresh Top-K; (if candidate source) build candidate pool |
| **Reindex** | no | yes | reuse nearest available `compress_kv` **and** its `index_k`; run own indexer query to rescore shared keys → fresh Top-K (selection changes, storage shared) |
| **Reuse** | no | no | reuse nearest `compress_kv` **and** the latest `topk_idxs` from a Full/Reindex layer above; attend directly (no indexer query, no scoring) |

Layers with `compress_ratio == 0` (pure sliding) participate in none of this.

The **critical intra-layer ordering** in the reference `_compress_kv`:

```
latent = compressor(x, start_pos)          # pre-RoPE latent
shared.compress_kv = self.compress_kv_cache
idxs = compute_topk_idxs(x, qr, latent)   # INDEXER RUNS FIRST — consumes pre-RoPE, pre-quant latent
if latent is not None:
    freqs = freqs_cis[ :S - S%ratio : ratio]  (prefill)   # group j -> position j*ratio
            or freqs_cis[start_pos + 1 - ratio].unsqueeze(0)  (decode)
    latent[..., -rd:] = rope(latent[..., -rd:], freqs)
    fp4_act_quant(latent, block=16, scale_dtype=E4M3)     # compressed KV: 16-blocks, E4M3 scales
    compress_kv_cache[:B, start_pos//ratio : +len] = latent
return shared.compress_kv[:B, :compress_len], idxs         # read AFTER the write
```

The indexer must see the latent **before** RoPE and before quantization because indexer K are projected from the unrotated latent.

### 8.3 Indexer (only `index_source_layer_ids` own one)

A small side attention that keeps the `index_topk` = 512 best compressed positions per query. It shares the attention's low-rank query `qr`:

```
parameters:
    wq_b         : [index_n_heads*index_head_dim, q_lora_rank] = [32*128, 1280]  (column-parallel)
    weights_proj : [index_n_heads, d] = [32, 5120]                                (bf16)
    wk           : [index_head_dim, hd] = [128, 512]        (ONLY if this layer owns indexer K)
    k_norm       : RMSNorm(index_head_dim, 1e-20)            (ONLY if owns K)
    softmax_scale = index_head_dim ** -0.5
    owns_k       = layer_id in kv_source_layer_ids
    index_n_heads = 32, index_head_dim = 128

forward(x, qr, latent, start_pos, offset):
    # (A) if owns_k and latent is not None -> produce & publish indexer K
    if owns_k and latent is not None:
        freqs = freqs_cis[:S - S%ratio : ratio]  (prefill)  or  freqs_cis[start_pos+1-ratio][None]  (decode)
        k = k_norm(wk(latent))                    # [B, n_new, index_head_dim]
        k[..., -rd:] = rope(k[..., -rd:], freqs)  # rotate the tail (rd = qk_rope_head_dim)
        fp4_act_quant(k, block=32, scale_dtype=UE8M0)
        k_cache[:B, start_pos//ratio : +len] = k
        shared.index_k = k_cache
    # (B) indexer query from the shared low-rank qr
    q = wq_b(qr).view(B, S, index_n_heads, index_head_dim)
    q[..., -rd:] = rope(q[..., -rd:], freqs_cis[start_pos:end_pos])
    fp4_act_quant(q, block=32, scale_dtype=UE8M0)

    # (C) score against shared indexer K
    index_k = shared.index_k[:B, :end_pos // ratio]
    weights = weights_proj(x) * (softmax_scale * index_n_heads ** -0.5)
    score = einsum("bshd,btd->bsht", q, index_k)          # [B,S,heads,ratio_len]
    score = (score.relu_() * weights.unsqueeze(-1)).sum(dim=2)   # ReLU-then-weight, summed over heads -> [B,S,ratio_len]

    # (D) reachability mask
    prefill: compress_lens = (arange(1,S+1) // ratio).unsqueeze(-1)
             score.masked_fill_(arange(S//ratio) >= compress_lens, -inf)
    decode:  compress_lens = end_pos // ratio             (a plain int)

    # (E) two-level selection (§8.4)
    if is_candidate_source:
        shared.candidates = select_candidate_blocks(score, compress_lens, topk_blocks, block_size)
    elif uses_candidates:
        score = score.masked_fill(~shared.candidates, -inf)   # level-2: search only the pool

    topk = min(index_topk, end_pos // ratio)
    idxs = score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values  # ascending by position
    return where(idxs < compress_lens, idxs + offset, -1).int()
```

Scoring rule: `relu(q·k)` per head, then weighted sum over heads by `weights_proj(x)` (scaled by `softmax_scale · n_heads^-0.5`). This is DSA/DeepSeek-v3.2 style — *not* a softmax.

### 8.4 Hierarchical Sparse Indexer (decoder only)

Used in the decoder to bound deeper indexers' cost. Level 1 (the candidate-source layer, layer 20) pools its scores into a block mask; level 2 (Reindex layers 24/28/32/36) masks its scores by that mask before its own top-512, turning deeper indexer cost from linear-in-context into constant.

```
select_candidate_blocks(logits [B,S,R], compress_lens, topk_blocks=2048, block_size=8) -> bool mask [B,S,R]
    width = R
    padded = pad(logits, (0, -width % block_size), value=-inf)
    block_scores = padded.unflatten(-1, (-1, block_size)).amax(-1)   # best position per block
    last = (compress_lens - 1) // block_size                          # block holding newest visible position
    block_scores = block_scores.masked_fill(arange(num_blocks) == last, +inf)   # pin the newest block in
    top = block_scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = zeros_like(block_scores, bool).scatter_(-1, top.indices, top.values > -inf)  # drop -inf picks
    return keep.repeat_interleave(block_size, -1)[..., :width]
```

- `block_size = 8`, `topk_blocks = 2048` → up to **16,384 candidate positions** per query.
- The candidate pool is larger than the final Top-512 and is shared by all downstream indexers, while each layer's final selection may differ.
- Only the **first** Full-mode decoder layer scans the full causal range; the rest search the pool.

### 8.5 Per-layer role table (implementation view)

Each `Attention` module owns `compressor` iff `layer in kv_source_layer_ids` and `indexer` iff `layer in index_source_layer_ids` (both restricted to backbone layers, i.e., `layer < n_layers`). Layers with `compress_ratio > 0` that own neither still read `shared.compress_kv` and `shared.topk_idxs`.

---

## 9 Rotary position embedding (partial RoPE, YaRN, compressed theta)

### 9.1 Frequency table

```
precompute_freqs_cis(dim=rd, seqlen, original_seq_len, base, factor, beta_fast, beta_slow) -> [seqlen, dim/2] complex
    freqs = 1 / base ** (arange(0, dim, 2) / dim)
    if original_seq_len > 0:                       # YaRN
        def corrected_dim(rotations):
            return dim * log(original_seq_len / (rotations * 2π)) / (2 * log(base))
        low  = max(floor(corrected_dim(beta_fast)), 0)
        high = min(ceil(corrected_dim(beta_slow)), dim - 1)
        ramp   = ((arange(dim//2) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = outer(arange(seqlen), freqs)
    return polar(ones, freqs)
```

### 9.2 Per-layer rope selection

```
if compress_ratio > 0:
    original_seq_len, base = rope_scaling.original_max_position_embeddings, compress_rope_theta  # 160000
else:                                    # pure sliding-window layer
    original_seq_len, base = 0, rope_theta                                                     # YaRN off
freqs_cis = precompute_freqs_cis(rd, max_seq_len, original_seq_len, base, factor, beta_fast, beta_slow)
```

- Compressing layers use **YaRN** (factor 16, original context 65,536) on **theta 160,000**.
- Pure-sliding layers (0, 1 in the released schedule) use plain **theta 10,000** with YaRN disabled.
- One frequency table per layer serves: queries, window KV, compressed latents (group `j` at position `j*ratio`), the indexer, and the inverse on the way out.

### 9.3 Apply / invert

```
rope(x, freqs, inverse=False):     # x: [b,s,d] or [b,s,h,d]; rotates adjacent pairs as complex
    y = x
    xc = view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse: freqs = freqs.conj()
    xc = view_as_real(xc * freqs).flatten(-2)
    y.copy_(xc)                     # in-place
```

Only the **last `rd` channels** of each head are ever passed to `rope`.

---

## 10 Causal Encoder–Decoder (CED)

### 10.1 Definition

`L = 40`; layers `0 … L/2-1` (0–19) are the **causal encoder**; layers `L/2 … L-1` (20–39) are the **decoder**. Both are causal (left-to-right); there is no bidirectional encoder and no cross-attention.

For **global** attention in the decoder, the KV entries are *not* derived from each decoder layer's own hidden state. They are projected directly from the hidden state of layer `L/2` (the last encoder layer) using **layer-dependent** projection weights:

```
C_l = H_{L/2} · W_l^KV ,    Z_l = H_{L/2} · W_l^Z ,    for l > L/2
```

where `C` = KV entries and `Z` = their compression (gate) weights. So all 20 decoder layers share one encoded representation for global KV; prefill computes only the first half of the layers and acquires the upper-layer global KV cache with minimal extra cost.

### 10.2 How this appears in the released checkpoint

The released configuration realizes CED through the CSA2 schedule: the **first decoder layer (20) is the only `kv_source` in the decoder** and its Compressor projects from its input hidden state — which, by construction of the residual stream, is `H_{20} = H_{L/2}`, the encoder output. Layers 21–39 all read layer 20's compressed-KV cache. In ratio-1 mode the "compression weight" `Z` is trivial (identity: no gate, plain projection). This is the concrete embodiment of `C_l = H_{L/2} W^KV` shared across the decoder.

`num_attention_heads` is unchanged across the boundary; what changes is *where the KV comes from*, and *which layers run during prefill*.

### 10.3 Decoder SWA Bounded Replay (the 8B/16B split)

Under CED, decoder **global** KV is already available after the encoder pass. The remaining prefill work for the decoder is the decoder's **sliding-window** KV, which (by design) is computed layer-wise from each decoder layer's own hidden states — so it needs the decoder layers to actually run over the prompt. Doing that exactly costs `L/2 · n_win` extra tokens. Instead:

- **Decoder SWA Bounded Replay:** at every prefill, replay only the **last `n_win` tokens** of the prompt through the decoder layers, under SWA truncation, to seed decoder SWA KV. The global KV for those positions is taken from the encoder projection. The replayed SWA state is used for decoding, **not** stored into the prefix cache (so it is naturally approximate — see the robustness caveat, §15.4).
- Consequently, the token positions of the prompt "activate" only the encoder ≈8B worth of weights during prefill, while each decoded token activates the full stack ≈16B.

[SIZING] The claim "8B prefill / 16B decode" is a count of *activated* parameters. Encoder-only prefill touches half the backbone; the ~8B/16B figures correspond to the encoder's and full-stack's active-parameter totals respectively (each MoE layer activates 1 shared + 6 of 384 routed experts).

### 10.4 CED interaction with the sparse indexer

Because every decoder layer reads the **same** compressed KV (layer 20's), the decoder's global branch is read-only across layers except for re-indexing. That is exactly why Reindex layers (24/28/32/36) own an indexer query but no keys (§8.2).

### 10.5 CED vs. a plain 40-layer decoder

For a from-scratch reproduction you may run all 40 layers every step (the reference does). That yields identical math but not the 8B/16B split nor the prefill speedup; the split is realized in the serving system, not the model graph.

---

## 11 Engram: conditional n-gram memory

Engram adds **196B** parameters of sparsely-accessed conditional memory that is *looked up* (not computed by routed experts). It is not part of the routed MoE; it is a hash table consulted at two points in the network and written into the residual stream through a learned gate.

### 11.1 Concept

Each token position forms the `n`-grams **ending** at it, for `n ∈ {2, 3, 4}` (`engram_max_ngram_size = 4`). Each n-gram is split across `engram_n_heads = 8` heads; each `(n-gram size, head)` pair owns a disjoint prime-sized bucket range in a per-layer table. The fetched 256-d rows are projected to a key (per hc stream copy) and one shared value, and added to the stream through a sigmoid gate driven by a normalized dot product of the stream against the key.

### 11.2 The compressed token map and bucket layout

Deterministic hashing runs over a **compressed token map** so that tokens that normalize alike collide:

```
normalizer = NFKC -> NFD -> StripAccents -> Lowercase
             -> collapse [ \t\r\n]+ to " " -> "^ $" -> sentinel -> Strip -> sentinel -> " "
map every tokenizer id to a new id under this normalizer (partial-UTF8 byte tokens keyed by raw form)
released compressed vocab = 99,092 ids (assert == engram_compressed_vocab_size)
```

Per-layer, odd multipliers (one per look-back) from a fixed RNG, so layers hash differently:

```
multipliers[layer] = 2 * default_rng(10007 * layer_id).integers(0, bound, size=(max_ngram,), int64) + 1
bound = max(1, (2**63 // compressed_vocab) // 2)   # keeps token*multiplier inside int64
```

Per `(n-gram size, head)` a distinct prime modulus, drawn **sequentially upward** from `engram_vocab_size` and never reused; the layer's `engram_num_embeddings` equals the sum of its primes (verified: 384,006,168 and 384,016,682).

```
EngramLayout.from_args:
  for each layer:
    for each ngram_size in 2..(max_ngram-1):
      sizes = []; cur = engram_vocab_size - 1
      for head in 1..n_heads:
        cur = next_prime_above(cur, seen); seen.add(cur); sizes.append(cur)
  offsets[layer][ngram] = cumsum([0, *sizes[:-1]])   # head index -> start of its prime range
```

### 11.3 Hash-id computation (`NgramHashState`)

Produces, per position, the list of `n_hash_cols = (max_ngram-1) * n_heads = 24` table indices (one per (n-gram size, head)):

```
compressed = token_map[input_ids];  compressed[~token_mask] = DEAD(-1)
cache[:B, start_pos : start_pos+S] = compressed              # persists across prefill/decode

for shift in 0..max_ngram-1:                                  # shift = look-back
    source   = cache.gather(positions - shift)               # clamp_min 0 at sequence start
    blocked |= (positions < shift) | (source == DEAD)          # image spans are DEAD
    tokens.append(where(blocked, pad_id, source))            # pad_id = compressed id of pad token (2)
tokens = stack(tokens, dim=-1)                                # [B,S,max_ngram]

products = tokens.unsqueeze(2) * multipliers                  # [B,S,n_layers,max_ngram]
rolling = products[...,0]; hashes = []
for i in 1..max_ngram-1:                                      # running XOR across look-backs
    rolling = bitwise_xor(rolling, products[...,i])           # after step i -> hash of the (i+1)-gram
    hashes.append((rolling.unsqueeze(-1) % primes[:, i-1]))   # reduce into that (ngram,head) prime range
return cat(hashes, -1) + offsets                              # [B,S,n_layers,24]
```

- Look-back stops at the sequence start and at any dead (image) token, so an n-gram never spans an image.
- Blocked slots read the compressed **pad** token; this preserves determinism across the prefill/decode split.
- The cache is `[max_batch, max_seq_len] int64`, `persistent=False`.

### 11.4 Engram module (the gated additive write)

```
Engram(args, layer_id, layout):
    embed   : ParallelEngramEmbedding(num_embeddings[layer], head_dim=256)   # fp8 rows + ue8m0 row scales
    wkv     : Linear(n_hash_cols * 256, d * (hc + 1))
    q_weight, k_weight : [hc, d] each, init ones (bf16)

forward(x [B,S,hc,d], hash_ids [B,S,24], token_mask):
    kv = wkv(embed(hash_ids).flatten(-2))                     # [B,S, d*(hc+1)]
    key, value = kv.split([hc*d, d], dim=-1)
    key   = key.view(B,S,hc,d)
    weight = q_weight.float() * k_weight.float()             # per-stream elementwise gate weights
    h, eps = x.float(), norm_eps
    rstd = rsqrt(mean(h^2,-1)+eps) * rsqrt(mean(key^2,-1)+eps) # normalized per (token, stream) over d
    dot  = (h * weight * key).sum(-1) * rstd * d ** -0.5
    gate = sigmoid(copysign(|dot|.clamp_min(1e-6).sqrt(), dot))  # signed sqrt before sigmoid
    gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)        # image tokens: no write
    return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)
```

- The `embed` lookup dequantizes fp8 rows on access: `rows = fp8_weight[idx] * fp8_scale[idx]` (fp32) → bf16.
- Decode needs only the previous 3 compressed ids; the id cache (§11.3) carries the history.

### 11.5 Placement and sharding

- Modules sit at **layers 1 and 14** (`engram_layer_ids`), balancing memory across pipeline stages.
- Tables are partitioned **by rows** across an `engram_parallel` group; each rank allocates `ceil(rows / world_size)`. Lookups are deterministic given the token ids, so a token's rows live on exactly one rank (masked all-reduce on lookup).
- During RL rollouts the tables are kept GPU-resident; during training, embeddings are stored/fetched in FP8 with prefetch overlapping the vision encoder.
- Engram projections are trained with **Muon** (see §19); the tables use the momentum+Sinkhorn-balanced update (§19.3).

---

## 12 DSpark: semi-autoregressive speculative decoding

DSpark replaces V3/V4's trained MTP module. A small **drafter** proposes a block of tokens; the backbone verifies them.

### 12.1 Structure

- The drafter is **3 Transformer blocks** (released) with a sliding attention window of 128, `layer_id = n_layers + stage_id` (so they live *after* the backbone). `Block.attention_cls` is overridden to `DSparkAttention`.
- Block size = **5** draft positions per round. Stage `s` produces one position per round in an autoregressive chain (`block_size` sequential samples), but all share one forward by attending bidirectionally inside the block.
- `main_proj` (stage 0): `Linear(d * len(target_layer_ids), d)` + norm, applied to the concatenated **attention inputs** of target layers `[37,38,39]` (each captured as `h.mean(dim=2)` right before that layer runs).
- A lightweight **Markov head** on the last stage produces an additive logit bias from the previous token id (`embed: ParallelEmbedding(V, markov_rank=256)`, `head: ParallelHead(V, markov_rank=256)`).
- A **confidence head** predicts a per-position acceptance probability from `cat[x, markov_embed]` → scalar; the scheduler uses prefix-survival estimates plus profiled engine throughput curves to choose the verification length per request.

### 12.2 Forward interfaces

```
forward_embed(main_hidden, input_ids):
    main_x = main_norm(main_proj(main_hidden))
    draft_ids = full([B, block_size], noise_token_id);  draft_ids[:,0] = input_ids
    x = embed(draft_ids).unsqueeze(2).repeat(1,1,hc,1)
    return x, main_x

forward(x, start_pos, pre_mix, main_x):
    if start_pos == 0:                                  # prefill: only seed the window cache
        self.attn(x, 0, main_x);  return x, pre_mix
    return super().forward(x, start_pos, pre_mix, None, main_x)   # drafts are text: no VL bias

forward_head(x, pre_mix, input_ids):
    x = hc_pre(x, pre_mix)
    logits = head(norm(x), full_logits=True)
    out[:,0] = input_ids
    for i in range(block_size):
        logits_bias, markov_embed = markov_head(out[:, i])
        logits[:, i] += logits_bias
        out[:, i+1] = sample(logits[:, i], temperature)
    confidence = confidence_head(x, stack(markov_embeds))
    return out, logits, confidence
```

The DSpark blocks share the backbone's `embed` and `head` (weight-tied: `mtp[-1].embed = embed`, `mtp[-1].head = head`); each has its own final `norm`, MoE (128 experts, top-3), and hyper-connection parameters.

### 12.3 `DSparkAttention`

```
main_kv = kv_norm(wkv(main_x)); rope tail; act_quant fp8         # at backbone positions
if start_pos == 0:  write main_kv into ring; return x             # seed only
# decode: draft block occupies positions start_pos+seqlen .. +block_size
q = wq_b(q_norm(wq_a(x))) at the draft positions; rope
kv_draft = kv_norm(wkv(x)) at the draft positions; rope; act_quant
topk_idxs = cat([ arange(min(win, start_pos+1)),          # backbone window slots, oldest first
                  win + arange(block_size) ])            # the draft block's own KV (no causal mask within block)
ring[start_pos % win] = main_kv[:,0]
kv = cat([window_kv_cache[:B], kv_draft], dim=1)
o = sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale); inverse-rope tail; grouped output proj
```

Draft layers assert `compress_ratio == 0` (pure sliding window).

### 12.4 Training schedule for DSpark

DSpark is introduced in a **dedicated stage after backbone pre-training** (only the drafter trains; backbone frozen). During post-training it keeps training alongside the backbone but **does not propagate DSpark gradients into the backbone**, which keeps it aligned with the evolving policy and accelerates rollout generation.

---

## 13 Vision: DeepSeek-ViT and the Aligner

A vision encoder trained from scratch plus a 2-layer MLP projector, inserted into the text embedding sequence and processed jointly with text from the start of pre-training.

### 13.1 DeepSeek-ViT

Built on ViT with these modifications (all load-bearing):

- **2D-RoPE** instead of absolute positional embeddings, to accept arbitrary resolutions.
- **Linear patch projection** instead of a conv, so the encoder is compatible with the Muon optimizer.
- **RMSNorm** everywhere (no LayerNorm/Bias), **SwiGLU** MLP.

Released geometry: 32 layers, hidden 1024, 16 heads (head_dim 64), patch size 14, MLP intermediate 2816, rope_theta 10000. Vision parameters ≈ 0.4B, bf16.

```
rope_dim = hidden / n_heads / 2 = 1024/16/2 = 32          # per-head rotary dim

get_vision_cos_sin(n_h, n_w, dim=32, theta):
    inv_freq = 1 / theta ** (arange(0, dim, 2)/dim)        # 16 frequencies
    hpos = arange(n_h)[:,None].expand(n_h, n_w)
    wpos = arange(n_w)[None,:].expand(n_h, n_w)
    freqs = stack([hpos, wpos], -1).reshape(-1, 2, 1) * inv_freq ; freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)     # [n_patches, 1, 32]

apply_rotary(x [n, heads, head_dim], cos, sin):   # split head_dim into halves (not adjacent pairs!)
    x1, x2 = x.chunk(2, -1); return cat([x1*cos - x2*sin, x2*cos + x1*sin], -1)

PatchEmbed: Linear(3 * 14^2, 1024)                        # linear, Muon-friendly
Attention:  wqkv: Linear(1024, 3072); q,k = rope(...); SDPA(bidirectional over the image); wo: Linear(1024,1024)
MLP:       w1: Linear(1024, 2*2816); gate,up = chunk(2); w2: Linear(2816, 1024); out = w2(silu(gate)*up)
Block:     x = x + attn(norm1(x), cos, sin);  x = x + mlp(norm2(x))
ViT:       patch_embed -> blocks -> RMSNorm
```

Note the vision rotary uses **half-split** halves (`chunk(2)`), whereas the LLM uses **adjacent pairs** (`view_as_complex`). Keep them distinct.

### 13.2 Aligner (pixel-unshuffle + MLP)

```
Aligner(downsample_ratio=3):
    in_dim = vision_dim * 3^2 = 1024 * 9 = 9216
    w1: Linear(9216, d=5120);  w2: Linear(5120, 5120)
    forward(x [n_h*n_w, 1024], n_h, n_w):
        r = 3
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)         # [C, H, W]
        x = pad(x, (0, -n_w % r, 0, -n_h % r))            # pad to multiple of r
        x = unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)   # [n_h', n_w', C*r*r]
        return w2(gelu(w1(x)))
```

The 3×3 pixel-unshuffle reduces visual token count by 9×, supporting input resolutions up to ~1344×1344 with `max_image_tokens = 1024`, `min_pixels = 295936`.

### 13.3 Merging image tokens into the sequence

Visual embeddings occupy the `image_token_id` positions in `input_ids`:

```
merge_image_embeddings(h, token_types, image_embeds, image_start, image_end, image_newline):
    for each image:
        IMAGE token positions  <- image_embeds (row-major)
        IMAGE_START position   <- learned image_start  [d]
        IMAGE_NEW_LINE pos     <- learned image_newline [d]
        IMAGE_END position     <- learned image_end    [d]
```
All merges happen at `start_pos == 0` (prefill only). The learned delimiter vectors live in the text model and are bf16.

### 13.4 Multimodal routing

Image and text tokens get separate expert-correction biases (`gate.bias_vl`), updated independently after each step (§6.1, §19.5). Vision tokens are excluded from Engram n-grams (masked DEAD, gate forced to 0).

---

## 14 Quantization: formats, fake-quant QAT semantics, how to disable

V4.1 is a **quantization-aware** model: the KV caches and (at inference) the weights hold *fake-quantized* values — the model was trained to see them. A faithful implementation reproduces this exactly.

### 14.1 Weight formats (released checkpoint)

| tensor class | format | scale | packing |
|---|---|---|---|
| dense linear weights (attn, projections, routers) | FP8 E4M3 | UE8M0, per `ceil(out/32) × ceil(in/32)` | `[out, in]` |
| routed expert weights | FP4 E2M1 | UE8M0, per `[out, in/32]` | `[out, in/2]` (two 4-bit values/byte, low nibble = even index) |
| embedding + LM head | BF16 | — | `[V, d]` |
| norms, `attn_sink`, `hc_*`, `gate.bias*` | FP32 | — | — |
| Engram tables | FP8 E4M3 rows | UE8M0 `[rows, head_dim/32]` | `[rows, 256]` |

`wo_a` is stored FP8 but DeepSeek's own `convert.py` **dequantizes it to BF16** for deployment (an FP8 grouped GEMM would halve the memory); reproduce this or keep an FP8 grouped GEMM. Never quantize `wo_a` with a layout-changing (packed) scheme — the grouped reshape would carve packed bytes.

### 14.2 FP8 E4M3 activation/weight quantization (block 32, UE8M0)

```
act_quant(x [M,N], block_size=32, scale_dtype=UE8M0, round_scale=True, inplace):
    for each row, each K-block of 32:
        amax = max(|x_block|, 1e-4)
        s    = 2 ** ceil(log2(amax / 448))          # UE8M0 pow2 scale via IEEE bit trick
        y    = clamp(x / s, -448, 448) -> cast E4M3  (in fake-quant: * s back, stored bf16)
```
`448` is the E4M3 max; the pow2 scale is computed by `fast_log2_ceil` / `fast_pow2` bit manipulation (reproduce bit-exactly if you care about bit parity).

### 14.3 FP4 E2M1 quantization

Two variants are used:
- **Indexer Q/K:** block 32, **UE8M0** pow2 scale: `s = 2 ** ceil(log2(amax / 6))`, `amax = max(amax, 6*2**-126)`.
- **Compressed KV latents:** block **16**, **E4M3** scale: `s = E4M3(amax / 6)`, `amax = max(amax, 6*2**-9)`.

Value grid (E2M1): `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` with round-to-nearest-even ties: at `.25`→0, `1.25`→1, `2.5`→2, `5`→4 (down); at `.75`→1, `1.75`→2, `3.5`→4 (up). Code `8` decodes as `-0` → `+0.0`.

### 14.4 Where quantization is applied in the forward pass

| tensor | format | block | scale dtype |
|---|---|---|---|
| window KV (whole 512-d vector, **RoPE tail included**) | FP8 E4M3 | 32 | UE8M0 |
| compressed KV latents | FP4 E2M1 | 16 | E4M3 |
| indexer Q and K | FP4 E2M1 | 32 | UE8M0 |

All applied **after RoPE** for the cache tensors (the compressed-KV latent is rotated, then quantized, then written; the indexer consumes the pre-RoPE latent for its keys and quantizes its own post-RoPE query/keys). Pre-RoPE quantization gives only marginal accuracy gain and adds decode overhead, so it is **not** used.

### 14.5 QAT placement (training-side)

The FP4 main-KV cache QAT is introduced **during post-training** (not pre-training). Indexer FP4 QAT (queries/keys) already existed in V4. Both non-RoPE and RoPE components use the same quantization format.

### 14.6 How to skip / disable quantization `[implementer note]`

Provide a single global switch (e.g., `QUANT_MODE ∈ {NONE, FAKE, REAL}`) consulted by every quant op:

- **`FAKE` (default, reference-faithful):** apply `act_quant`/`fp4_act_quant` with `inplace=True` — values are round-tripped through the low-bit grid but stored in bf16. This is what the model was trained against and what a correctness/parity build should use.
- **`REAL`:** store the packed low-bit tensors + scales and dequantize on the fly in the GEMM/attention kernels (matches the deployed kernels' bit patterns to within rounding).
- **`NONE`:** skip all cache/activation quantization, run bf16/fp32. This is mathematically cleanest and easiest to get right, but is **not** the trained model — expect degraded quality. Useful for architecture debugging and for training a small reproduction where you control the QAT schedule.

```
def act_quant(x, block_size=32, scale_dtype=UE8M0, round_scale=True, inplace=True, mode=QUANT):
    if mode == NONE: return x if inplace else (x, None)
    ... (the block absmax / pow2-scale logic)
```
When disabled, also skip the `attn_sink`/scale plumbing? No — `attn_sink`, RMSNorm, and the Sinkhorn split are architecture, not quantization; keep them always.

---

## 15 Inference & serving architecture

This section describes the systems that turn the architecture into the published 890 B/token and 1/4, 1/8 ratios.

### 15.1 The KV-cache budget

Two tiers:

- **Global KV** (always in **HBM**): main KV + indexer K. Compressed by CSA2 cross-layer reuse + FP4 → **890 bytes/token** ≈ 1/4 of V4-Flash, ≈ 1/437 of V1. A full 1M-token prompt holds **under 1 GB** of global KV.
- **SWA KV** (per-layer, bounded by `n_win`): cheap to rebuild, high turnover.

### 15.2 Persistent (SSD/host) KV — SWA Bounded Replay

V4-Flash persisted both global and SWA KV to SSD (SWA cached only at two points: end of prompt, end of output). V4.1 **stops persisting SWA KV** and instead reconstructs it:

- **Encoder SWA Bounded Replay:** on a global-KV hit / SWA-KV miss, replay the **last `n_win` tokens of the cached prefix** together with the uncached suffix. Replayed tokens regenerate **only SWA KV**, reusing cached global KV without overwriting it; the uncached suffix generates both. Replayed prefix SWA is approximate by design.
- **Decoder SWA Bounded Replay:** during prefill, run the decoder layers over only the last `n_win` prompt tokens (§10.3) to seed decoder SWA KV, used for decoding but **not** written to the prefix cache.

Net effect: persistent KV (global only, FP4) is **≈ 1/8** of V4-Flash; SWA KV lives in a distributed pool sized from **10% of host DRAM** per machine with minute-scale TTL; global KV keeps a **≥ 72-hour** lifetime. Replay turns a "prohibitive" full `L × n_win` recompute into a graceful `n_win`-token recompute.

### 15.3 Kernel fusion and counts

The deployment fuses RoPE + attention + RoPE + cast (FlashMLA), the Mega-* gate/mHC/MoE ops (DeepGEMM), tile kernels, and the Top-K (DeepSelect). A Reuse-mode layer executes with **15 kernels at prefill, 11 at decode**. The inference system uses **Encoder–Prefill–Decode (EPD) disaggregation** so vision encoding, prefill, and decode scale independently and overlap.

### 15.4 Robustness boundary (architectural caveat)

The report explicitly notes V4.1's simplifications "create robustness boundaries that have yet to be fully characterized": **CSA2 sparse-selection errors** and **approximate SWA-state reconstruction** (both bounded replay flavors) may degrade capability on untested boundary cases. Stress-testing focuses on sparse retrieval over long contexts and SWA reconstruction at cache-resumption boundaries.

### 15.5 Decode-time FLOPs

Because of CED + CSA2 + bounded indexing, single-token decode FLOPs are **nearly constant** in context length: extending 4K→1M (256×) raises decode FLOPs by only ~1/4. Compute precision is weighted BF16/FP8/FP4 as 1/0.5/0.25.

---

## 16 Checkpoint layout and weight conversion

### 16.1 Tensor naming (hub layout)

The hub already uses inference-style names (no `model.` prefix):

```
embed.weight                       [V, d]                 bf16
layers.{i}.attn.wq_a.weight        [q_lora_rank, d]
layers.{i}.attn.q_norm.weight      [q_lora_rank]
layers.{i}.attn.wq_b.weight        [H*hd, q_lora_rank]
layers.{i}.attn.wkv.weight         [hd, d]
layers.{i}.attn.kv_norm.weight     [hd]
layers.{i}.attn.wo_a.weight        [o_groups*o_lora_rank, (H*hd)//o_groups]   bf16 after convert
layers.{i}.attn.wo_b.weight        [d, o_groups*o_lora_rank]
layers.{i}.attn.attn_sink          [H]                    fp32
layers.{i}.attn.compressor.wkv.weight      [hd, d]        (kv_source layers only)
layers.{i}.attn.compressor.wgate.weight    [hd, d]        (ratio > 1 only)
layers.{i}.attn.compressor.norm.weight     [hd]
layers.{i}.attn.indexer.wq_b.weight         [index_n_heads*index_head_dim, q_lora_rank]
layers.{i}.attn.indexer.weights_proj.weight [index_n_heads, d]
layers.{i}.attn.indexer.wk.weight           [index_head_dim, hd]   (index-key owners only)
layers.{i}.attn.indexer.k_norm.weight       [index_head_dim]
layers.{i}.ffn.gate.weight                  [E, d]          bf16
layers.{i}.ffn.gate.bias                    [E]             fp32
layers.{i}.ffn.gate.bias_vl                 [E]             fp32 (vision builds)
layers.{i}.ffn.shared_experts.{w1,w2,w3}.weight
layers.{i}.ffn.experts.{e}.{w1,w2,w3}.weight            fp4 packed [MoEI, d]/[d, MoEI]/...
layers.{i}.ffn.experts.{e}.*.scale          ue8m0
layers.{i}.ffn.shared_experts.*.scale
layers.{i}.attn_norm.weight / ffn_norm.weight  [d]        fp32
layers.{i}.hc_attn_fn   [mix_hc, hc*d]        fp32
layers.{i}.hc_ffn_fn    [mix_hc, hc*d]        fp32
layers.{i}.hc_attn_base [mix_hc]              fp32
layers.{i}.hc_ffn_base  [mix_hc]              fp32
layers.{i}.hc_attn_scale [3] / hc_ffn_scale [3]          fp32
layers.{i}.engram.embed.weight   [rows, 256]  fp8  (+ .scale [rows, 8] ue8m0)
layers.{i}.engram.wkv.weight     [d*(hc+1), 24*256]
layers.{i}.engram.q_weight / k_weight  [hc, d] bf16
norm.weight                          [d]        fp32
head.weight                          [V, d]     bf16 (run in fp32)
image_start / image_end / image_newline.weight  [d]  bf16
vision.*  /  aligner.*  (bf16, ~0.4B, carried through unchanged)
mtp.{0,1,2}.*                        (DSpark draft stack; ~14B)
```

Scale tensors are attached as `<weight_name>.scale` (not `weight_scale_inv`).

### 16.2 Disk composition (~511 GB / 476 GiB)

| component | entries | stored |
|---|---|---|
| routed + DSpark experts (MXFP4) | 557.2B | 259.5 GiB |
| Engram tables (FP8) | 196.6B | 183.1 GiB |
| attention, dense projections, routers (FP8) | 7.4B | 6.9 GiB |
| embedding + LM head (BF16), norms (FP32) | 2.0B | 3.9 GiB |
| UE8M0 block scales | 23.6B | 21.9 GiB |

48 safetensors shards; the two Engram tables ship as two single ~101 GB shards.

### 16.3 Conversion obligations

1. Materialize `float8_e4m3fn` and `float8_e8m0fnu` weights (safetensors `F8_E4M3` / `F8_E8M0`) so the dequant kernels see raw bytes.
2. FP4 experts are stored `int8` `[out, in/2]` + `[out, in/32]` ue8m0 scales (low nibble = even index; code 8 = `+0.0`).
3. Dequantize `wo_a` FP8 → BF16 (or keep FP8 with a grouped GEMM).
4. Strip the `mtp.*` stack if you do not implement DSpark (it is not needed for the backbone).
5. The vision tower and aligner are passed through unchanged.

---

## 17 Implementation blueprint

### 17.1 File layout

```
dsv41/
  config.py       # dataclass + PRESET_552B + PRESET_VERTEX_500M  (§3.4, §3.6)
  quant.py        # act_quant / fp4_act_quant / fast_log2_ceil / fast_pow2 (§14.2, §14.3)
  rope.py         # precompute_freqs_cis, apply_rotary_emb, vision rope (§9, §13.1)
  hc.py           # hc_split_sinkhorn, hc_pre, hc_post (§5)
  sparse_attn.py  # sparse_attn semantics (§7.4)
  attention.py    # Attention, Compressor, Indexer, select_candidate_blocks (§7, §8)
  moe.py          # Gate, Expert, MoE (§6)
  engram.py       # EngramLayout, NgramHashState, Engram (§11)
  dspark.py       # DSparkBlock, DSparkAttention, heads (§12)
  vision.py       # ViT, Aligner (§13)
  model.py        # Block, Transformer, forward, forward_spec (§4, §17.3)
  generate.py     # prefill/decode loop (§17.4)
```

### 17.2 Numerical preconditions

- All RMSNorms normalize in fp32 with `eps = rms_norm_eps` (1e-20); multiply by the scale and cast back.
- `sparse_attn` requires a **contiguous int32** `topk_idxs` `[B, m, topk]`; materialize with `torch.cat(...).contiguous().int()`.
- Caches are preallocated at `max_batch_size`, sliced `[:bsz]`. Ring buffer for the window, flat cache for compressed KV / indexer K.
- `start_pos` is threaded explicitly through the whole forward (not inferred).

### 17.3 Reference implementation (core, single-file PyTorch)

```python
import math
import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------- config
@dataclass
class Cfg:
    vocab: int; dim: int; n_layers: int; n_heads: int; head_dim: int
    q_lora_rank: int; qk_rope_head_dim: int; o_lora_rank: int; o_groups: int
    window: int; hc_mult: int; hc_sinkhorn_iters: int; hc_eps: int
    compress_ratios: tuple; kv_source: tuple; index_source: tuple
    index_n_heads: int; index_head_dim: int; index_topk: int
    candidate_source: int; candidate_topk_blocks: int; candidate_block_size: int
    n_experts: int; topk: int; moe_inter: int; routed_scale: float
    rms_norm_eps: float = 1e-20; swiglu_limit: float = 10.0

# ---------------------------------------------------------------- quant
FP8_MAX, FP4_MAX = 448.0, 6.0

def _fast_log2_ceil(x):                       # bit-exact UE8M0 scale helper
    b = x.view(torch.int32)
    e = ((b >> 23) & 0xFF) - 127
    m = b & ((1 << 23) - 1)
    return e + (m != 0).int()

def _fast_pow2(x):                            # 2**x for integer x
    return ((x + 127) << 23).view(torch.float32)

def act_quant(x, block=32, round_scale=True, inplace=True):
    """Block-32 FP8 E4M3 fake-quant, UE8M0 pow2 scales. Returns (y, s)."""
    M, N = x.shape
    xb = x.view(M, N // block, block).float()
    amax = xb.abs().amax(-1).clamp_min(1e-4)
    s = (_fast_pow2(_fast_log2_ceil(amax / FP8_MAX)) if round_scale
         else amax / FP8_MAX).unsqueeze(-1)          # [M, N/block, 1]
    q = (xb / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * s
    y = q.view(M, N).to(x.dtype)
    if inplace: x.copy_(y)
    return (y, s.view(M, N // block).to(torch.float8_e8m0fnu))

def fp4_act_quant(x, block=32, e4m3_scale=False):
    """Block-wise FP4 E2M1 fake-quant. e4m3_scale=True -> compressed-KV variant (block 16, E4M3 scale)."""
    M, N = x.shape
    xb = x.view(M, N // block, block).float()
    amax = xb.abs().amax(-1)
    if e4m3_scale:
        amax = amax.clamp_min(6 * 2**-9)
        s = (amax / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(6 * 2**-126)
        s = _fast_pow2(_fast_log2_ceil(amax / FP4_MAX))
    grid = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=x.device)
    v = (xb / s.unsqueeze(-1)).clamp(-FP4_MAX, FP4_MAX).abs()
    # bucketize with round-half-to-even on the 8-value E2M1 grid
    q = grid[torch.bucketize(v, torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], device=x.device))]
    q = torch.sign(xb) * q * s.unsqueeze(-1)
    return q.view(M, N).to(x.dtype), s

# ---------------------------------------------------------------- rope
def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:                   # YaRN
        def corr(rot): return dim * math.log(original_seq_len / (rot * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(corr(beta_fast)), 0); high = min(math.ceil(corr(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rope(x, freqs_cis, inverse=False):
    f = freqs_cis.conj() if inverse else freqs_cis
    if x.ndim == 3: f = f.view(1, x.size(1), -1)
    else:           f = f.view(1, x.size(1), 1, -1)
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)) * f)
    return torch.view_as_real(xc).flatten(-2).to(x.dtype)

# ---------------------------------------------------------------- hyper-connections
def hc_split_sinkhorn(mixes, hc, scale, base, iters, eps):
    b, s, _ = mixes.shape
    pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2*hc] * scale[1] + base[hc:2*hc])
    comb = (mixes[..., 2*hc:].view(b, s, hc, hc) * scale[2]
            + base[2*hc:].view(hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb

class RMSNorm(nn.Module):
    def __init__(s, d, eps): super().__init__(); s.eps = eps; s.w = nn.Parameter(torch.ones(d))
    def forward(s, x):
        dt = x.dtype; xf = x.float()
        xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + s.eps)
        return (s.w.float() * xf).to(dt)

# ---------------------------------------------------------------- sparse attention
def sparse_attn(q, kv, attn_sink, topk_idxs, scale, hmin=-1e30):
    """q [B,m,H,d]; kv [B,N,d]; topk_idxs [B,m,K] int32 (-1 masked); sink -> denominator only."""
    B, m, H, d = q.shape
    acc = torch.zeros(B, m, H, d, dtype=torch.float32, device=q.device)
    den = torch.zeros(B, m, H, dtype=torch.float32, device=q.device)
    idx = topk_idxs.long()
    valid = idx >= 0
    safe = idx.clamp_min(0)
    kvg = torch.gather(kv.unsqueeze(1).expand(B, m, kv.size(1), d), 2,
                       safe.unsqueeze(-1).expand(B, m, H if False else safe.size(-1), d))  # [B,m,K,d]
    # scores: [B,m,K,H]
    sc = torch.einsum("bmhd,bmkd->bmkh", q.float(), kvg.float()) * scale
    sc = sc.masked_fill(~valid.unsqueeze(-1), float("-inf"))
    mx = sc.amax(-1).clamp_min(hmin)                                     # finite floor -> zero rows
    p = torch.exp(sc - mx.unsqueeze(-1))
    den = p.sum(-1) + torch.exp(attn_sink.float().view(1, 1, H) - mx)
    acc = torch.einsum("bmkh,bmkd->bmhd", p, kvg.float()) / den.unsqueeze(-1)
    return acc.to(q.dtype)

def get_window_topk_idxs(win, bsz, seqlen, start_pos):
    if start_pos == 0:
        end = torch.arange(seqlen).unsqueeze(1)
        idxs = (end - win + 1).clamp(0) + torch.arange(min(seqlen, win))
        idxs = torch.where(idxs > end, -1, idxs)
    else:
        oldest = start_pos % win + 1
        idxs = torch.cat([torch.arange(oldest, win), torch.arange(oldest)])
        idxs = torch.where(idxs > start_pos, -1, idxs)
    return idxs.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()

def select_candidate_blocks(logits, compress_lens, topk_blocks, block_size):
    width = logits.size(-1)
    padded = F.pad(logits, (0, -width % block_size), value=float("-inf"))
    scores = padded.unflatten(-1, (-1, block_size)).amax(-1)
    num = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num, device=logits.device) == last, float("inf"))
    top = scores.topk(min(topk_blocks, num), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -float("inf"))
    return keep.repeat_interleave(block_size, -1)[..., :width]

# ---------------------------------------------------------------- attention (CSA2)
class Compressor(nn.Module):
    def __init__(s, cfg, layer_id):
        super().__init__(); s.ratio = cfg.compress_ratios[layer_id]; s.hd = cfg.head_dim
        s.norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        dt = torch.float32 if s.ratio > 1 else torch.bfloat16
        s.wkv = nn.Linear(cfg.dim, cfg.head_dim, bias=False).to(dt)
        if s.ratio > 1:
            s.wgate = nn.Linear(cfg.dim, cfg.head_dim, bias=False).to(torch.float32)
    def forward(s, x, start_pos, state):
        r = s.ratio
        if r == 1: return s.norm(s.wkv(x))
        kv, sc = s.wkv(x.float()), s.wgate(x.float())
        if start_pos == 0:
            if x.size(1) >= r:
                full, rem = x.size(1) // r, x.size(1) % r
                if rem: state.store(kv, sc, rem)
                kv = kv[:, :full*r].unflatten(1, (-1, r)); sc = sc[:, :full*r].unflatten(1, (-1, r))
                kv = (kv * sc.softmax(2)).sum(2)
            return s.norm(kv)
        done = (start_pos + 1) % r == 0
        state.push(kv[:, 0], sc[:, 0], start_pos % r)
        if done: return s.norm((state.kv * state.sc.softmax(1)).sum(1, keepdim=True))
        return None

class Indexer(nn.Module):
    def __init__(s, cfg, layer_id, owns_k):
        super().__init__(); s.cfg = cfg; s.owns_k = owns_k
        s.ratio = cfg.compress_ratios[layer_id]
        s.is_cand_src = layer_id == cfg.candidate_source
        s.uses_cand = 0 <= cfg.candidate_source < layer_id
        s.scale = cfg.index_head_dim ** -0.5
        s.wq_b = nn.Linear(cfg.q_lora_rank, cfg.index_n_heads * cfg.index_head_dim, bias=False)
        s.weights_proj = nn.Linear(cfg.dim, cfg.index_n_heads, bias=False)
        if owns_k:
            s.wk = nn.Linear(cfg.head_dim, cfg.index_head_dim, bias=False)
            s.k_norm = RMSNorm(cfg.index_head_dim, cfg.rms_norm_eps)
    def forward(s, x, qr, freqs_cis, start_pos, offset, index_k):
        B, S, _ = x.shape; r = s.ratio; rd = s.cfg.qk_rope_head_dim; end = start_pos + S
        k = s.k_norm(s.wk(x))                    # x here is the pre-RoPE latent (only if owns_k)
        k[..., -rd:] = apply_rope(k[..., -rd:], freqs_cis[:S - S%r : r] if start_pos == 0
                                  else freqs_cis[start_pos + 1 - r][None])
        k, _ = fp4_act_quant(k, 32, e4m3_scale=False)
        q = s.wq_b(qr).view(B, S, s.cfg.index_n_heads, s.cfg.index_head_dim)
        q[..., -rd:] = apply_rope(q[..., -rd:], freqs_cis[start_pos:end])
        q, _ = fp4_act_quant(q, 32, e4m3_scale=False)
        w = s.weights_proj(x).float() * (s.scale * s.cfg.index_n_heads ** -0.5)
        sc = torch.einsum("bshd,btd->bsht", q.float(), index_k[:B, :end//r].float())
        sc = (sc.relu() * w.unsqueeze(-1)).sum(2)                       # [B,S,R]
        if start_pos == 0:
            cl = (torch.arange(1, S+1, device=x.device) // r).unsqueeze(-1)
            sc = sc.masked_fill(torch.arange(S//r, device=x.device) >= cl, float("-inf"))
        else:
            cl = end // r
        if s.is_cand_src:
            shared.candidates = select_candidate_blocks(sc, cl, s.cfg.candidate_topk_blocks,
                                                        s.cfg.candidate_block_size)
        elif s.uses_cand:
            sc = sc.masked_fill(~shared.candidates, float("-inf"))
        topk = min(s.cfg.index_topk, end // r)
        idx = sc.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        return torch.where(idx < cl, idx + offset, -1).int()
```

(`shared` is the `SharedAttentionRuntime` singleton of §8.2; `state` is the per-layer `(kv_state, score_state)` pair of §8.1.)

### 17.4 Generation loop contract

```
prefill:  out, logits, main_hidden = model(x[:, :S], start_pos=0)
          -> encoder+decoder both run in the reference; serving may skip decoder except last n_win
decode:   for i in range(S, S+steps):
              out, logits, main_hidden = model(x[:, i:i+1], start_pos=i)
              draft, logits, conf = model.forward_spec(out, main_hidden, start_pos=i)   # DSpark
```
`start_pos` drives every cache index: window ring slot `start_pos % window`, compressed slot `start_pos // ratio`, indexer slot `end // ratio`, freq slice `[start_pos : start_pos+S]`. The reference supports full prefill + 1-token decode; a chunk-general implementation must carry the window ring, compressor partial-group state, indexer visibility, and Engram id history across arbitrary chunk boundaries (an independent MLX port verified this generalization).

### 17.5 Verification checklist (what to test)

1. **Shapes:** every projection's `[in, out]` matches §3.4; KV latent is `[B, N, hd]` with no value projection.
2. **Index arithmetic:** window indices per §7.3; compressed indices offset by window length; `-1` masking; candidate pool size = `2048*8`.
3. **Sparse attention:** all-masked row → exactly zero output; sink enters denominator only; inverse RoPE on output tail before output projection.
4. **Compressor:** ratio-1 emits every step; ratio-2 emits every 2nd decode step and carries partial groups; latent RoPE at group-first position.
5. **Cross-layer sharing:** layer 24 selects from layer 20's keys but uses its own indexer weights; Reuse layers use the published `topk_idxs`.
6. **Engram hashing:** primes-per-layer sums equal `engram_num_embeddings`; compressed map size equals `engram_compressed_vocab_size`; hashes stable across prefill/decode split.
7. **Quantization:** value grids match §14.3; window KV block-32 UE8M0, compressed KV block-16 E4M3, indexer block-32 UE8M0.
8. **Parity:** fp4 fake-quant values are exactly representable in bf16, so with QAT disabled agreement across implementations should be ~1e-6; with QAT on, occasional single-ULP rounding-midpoint flips are expected.

---

## 18 Pitfalls and invariants checklist

- **`start_pos` discipline.** Every cache write and index is absolute. Forgetting to thread `start_pos` breaks ratio-2 compressors, the window ring, and indexer visibility at once.
- **Index-before-RoPE.** The indexer must consume the compressor's **pre-RoPE, pre-quant** latent. Rotating/quantizing before the indexer silently corrupts indexer K.
- **Compression position convention.** A latent stands for the **first** token of its group (position `j*ratio`), not the last. Both prefill slicing (`freqs[:S-S%r : r]`) and decode (`freqs[start_pos+1-r]`) rely on it.
- **Reuse-mode layers** must not run an indexer; they read the published `topk_idxs`. Reindex layers run an indexer but read shared keys. Full layers do both.
- **Candidate source pins the newest block** to `+inf` before top-k, else the most recent tokens can be outscored by an older full block.
- **Sink in denominator only**; a `-1e30` max floor, not `-inf`, to avoid NaN on empty rows.
- **Inverse RoPE on the output tail** is load-bearing (carried from V4).
- **hc pre-coefficients are consumed one sublayer late** (Single-Pass shift). Using the same block's `pre` for its own sublayer changes the model.
- **`wo_a` must not be packed-quantized** — the grouped reshape would corrupt bytes.
- **Per-modality routing bias.** Image tokens use `bias_vl`; text uses `bias`. Text-only inference always uses `bias` but keep `bias_vl` loaded for round-trip.
- **rmIsNorm eps is literally 1e-20**, not a "reasonable" 1e-6.
- **Encoder SWA replay regenerates only SWA KV** and must not overwrite cached global KV.
- **DSpark gradients do not flow into the backbone** during post-training.
- **Known reference artifact** (decode only): the global `index_k` pointer is only reassigned when a layer publishes keys, so on ratio-2 decode steps a consumer may read a stale producer's cache. A clean implementation reads the layer's own cache (as the MLX port does) and is correct on all steps.

---

## 19 Training-side algorithms that touch the architecture

These are not needed for a forward-pass reproduction but define how the architecture is trained and are part of the "spec" of the model.

### 19.1 Pre-training recipe

- 45T multimodal tokens (7:1 text:multimodal), no instability. Batch fixed at **100.6M tokens**.
- LR: linear warmup 2000 steps to **2.6e-4**, hold to **28T**, cosine decay to **2.6e-5** by **40T**, hold to **45T**.
- Sequence length: sparse attention trained from scratch at **64K**, extended to **1M at 34T tokens**.
- Padding rate ≤ 1e-4 (best-fit packing).

### 19.2 Optimizers (architecture-coupled)

- **AdamW** for norm weights and other non-matrix params: `β1=0.9, β2=0.95, ε=1e-20, wd=0.1`.
- **Muon** for linear weights of the LM backbone, Engram projections, and the VL projector: momentum `0.95`, wd `0.1`, update matrix RMS rescaled to **0.18** (to reuse the AdamW LR). **Head-wise Muon** (per-head preconditioner) for Q and K weights.
- **Momentum + Sinkhorn balancing** (instead of Adam) for embedding, prediction head, and Engram tables: momentum update → Nesterov → row/column equalization by odd-count Sinkhorn (K alternating normalizations), `γ=0.18` LR correction, `ε=1e-20`. Algorithm 1 in the report; `K=11, τ=1e-3`. Only a momentum buffer needed and empirically beats Adam.
- Engram LR scaled ×5.

### 19.3 MoE load balancing

Auxiliary-loss-free balancing with **separate expert-wise correction biases for text and image tokens**, updated independently per step by their own expert loads; plus a small sequence-level balance loss (weight `1e-4`) and sample-level attention masking during pre-training.

### 19.4 Attention-sharing training (CSA2 across pipeline stages)

Layers sharing attention components may live on different pipeline stages. The training infra uses: **shadow indexers** (executable replicas with a single logical parameter owner + param sync / grad aggregation), **pipeline payload extensions** (pass intermediate activations + sparse routing info across stages, partitioned consistently with context parallelism), and **micro-batch shared-state tracking** (retain a shared state until its final consumer completes).

### 19.5 Data & vision training

- DeepSeek-ViT trained in two stages before integration: (1) SigLIP contrastive on ~47B image-text pairs at 224²; (2) autoregressive fine-tune attached to a **4B MoE LLM** on 236B tokens (captions/alt/charts/OCR) at 544²–1344², then the 4B LLM is discarded. The vision encoder is **frozen until the LR-decay stage**, then unfrozen and trained at a smaller LR.

---

## 20 Appendices

### 20.1 Parameter-count script

Reproduce §2 exactly (this is the script that produced the verified numbers):

```python
def primes_after(start, seen):
    from sympy import isprime
    c = start + 1
    while not isprime(c) or c in seen: c += 1
    return c

def engram_rows(base, max_ngram, n_heads, n_layers):
    seen, per = set(), []
    for _ in range(n_layers):
        s = 0
        for _ in range(max_ngram - 1):
            cur = base - 1
            for _ in range(n_heads):
                cur = primes_after(cur, seen); seen.add(cur); s += cur
        per.append(s)
    return per

def count(cfg):
    d,L,H,hd,V,MoEI = cfg['hidden'],cfg['n_layers'],cfg['n_heads'],cfg['head_dim'],cfg['vocab'],cfg['moe_inter']
    ql,og,ol,hc = cfg['q_lora_rank'],cfg['o_groups'],cfg['o_lora_rank'],cfg['hc_mult']
    attn = d*ql + ql + ql*H*hd + d*hd + hd + (H*hd//og)*(og*ol) + og*ol*d + H
    routed = cfg['n_experts']*3*d*MoEI; shared = 3*d*MoEI
    hcp = 2*((2+hc)*hc*(hc*d) + (2+hc)*hc + 3)
    idx = cfg['index_n_heads']*cfg['index_head_dim']*ql + cfg['index_n_heads']*d
    idx *= len(cfg['index_src'])
    idx += len(cfg['kv_source'])*(hd*cfg['index_head_dim'] + cfg['index_head_dim'])
    comp = sum(d*hd + hd + (d*hd if cfg['compress_ratios'][l]>1 else 0) for l in cfg['kv_source'])
    backbone = V*d + V*d + L*(attn+routed+shared+hcp+2*d+cfg['n_experts']*d+2*cfg['n_experts']) + idx + comp
    engram = 0
    for r in engram_rows(cfg['engram_base'],cfg['engram_max_ngram'],cfg['engram_n_heads'],len(cfg['engram_layers'])):
        engram += r*cfg['engram_head_dim'] + (cfg['engram_max_ngram']-1)*cfg['engram_n_heads']*cfg['engram_head_dim']*d*(hc+1) + 2*hc*d
    ds = cfg['dspark_stages']
    dspark = ds*(attn+cfg['dspark_experts']*3*d*MoEI+shared+hcp+2*d) + d*len(cfg['dspark_targets'])*d + d \
             + V*cfg['dspark_markov_rank']*2 + (d+cfg['dspark_markov_rank'])+1
    return backbone, engram, dspark
```
Called with the released config and the Vertex-500M preset (§3.4, §3.6) this yields 551.57B / 196.93B / 14.23B and 414.4M / 40.4M / 31.9M respectively.

### 20.2 Activation counts (8B / 16B)

Each MoE layer activates 1 shared + 6 routed experts (`3*d*MoEI*(1+6) = 3*5120*2304*7 ≈ 248M`) plus attention (~127M) + HC (~1M) ≈ **0.4B per layer**. Forty layers ≈ 16B (decode); the 20-layer encoder ≈ 8B (prefill). This is where the published "8B / 16B" comes from.

### 20.3 Efficiency headline numbers

| property | value |
|---|---|
| Global KV (HBM) | 890 B/token (≈1/4 V4-Flash, ≈1/437 V1) |
| Persistent KV (SSD/host) | ≈1/8 V4-Flash; global guaranteed ≥72 h |
| SWA KV pool | 10% of host DRAM per machine, minute TTL |
| 1M-token global KV | < 1 GB |
| Reuse-layer kernel count | 15 (prefill) / 11 (decode) |
| Decode FLOPs 4K→1M | +≈1/4 only (near-constant) |

### 20.4 Sources

1. Technical report: *DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression*, arXiv:2609.19969v1.
2. Released `config.json` + `inference/{model,engram,vision,kernel}.py` (huggingface `deepseek-ai/DeepSeek-V4.1-Flash`).
3. vLLM recipe `deepseek-ai/DeepSeek-V4.1-Flash`; SGLang cookbook DeepSeek-V4.1; NeMo AutoModel `deepseek_v41`.
4. An independent MLX port's upstream notes (parity findings: QAT formats, HC hand-off, decode artifact).

---

# PART II — Complete reference implementation

## 21 Full reference implementation (runnable PyTorch, single file)

This section is a complete, dependency-light PyTorch implementation of the architecture. It is written to be *read and ported* — it favours clarity and exactness over speed, but it is functionally complete: forward pass, all caches, chunked prefill, decode, DSpark drafting, vision path, and a `QUANT_MODE` switch.

### 21.1 Module map

```
dsv41_model.py
  ├─ QUANT_MODE, fp8/fp4 helpers            (§14)
  ├─ RMSNorm, precompute_freqs_cis, apply_rope, vision_rope   (§5.3, §9, §13.1)
  ├─ hc_split_sinkhorn, hc_pre, hc_post      (§5)
  ├─ sparse_attn, get_window_topk_idxs, get_dspark_topk_idxs, select_candidate_blocks   (§7.4, §8.4)
  ├─ Compressor + PoolState                   (§8.1)
  ├─ Indexer                                 (§8.3)
  ├─ Attention                               (§7.2)
  ├─ Gate, Expert, MoE                       (§6)
  ├─ CompressorState, ParallelEngramEmbedding, EngramLayout, NgramHashState, Engram  (§11)
  ├─ DSparkAttention, DSparkMarkovHead, DSparkConfidenceHead, DSparkBlock    (§12)
  ├─ Block                                   (§5.6)
  ├─ ViT, Aligner                            (§13)
  ├─ SharedAttentionRuntime (shared)         (§8.2)
  ├─ Transformer                             (§4.2)
  └─ generate()                              (§21.11)
```

### 21.2 Quantization primitives (§14)

```python
import torch

QUANT_MODE = "FAKE"          # "NONE" | "FAKE" | "REAL"

FP8_MAX = 448.0              # E4M3 max
FP4_MAX = 6.0                # E2M1 max
FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FP4_EDGES = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])   # RNE-to-even boundaries


def _fast_log2_ceil(x: torch.Tensor) -> torch.Tensor:
    """ceil(log2(x)) by IEEE-754 bit manipulation (matches the tilelang reference)."""
    b = x.view(torch.int32)
    e = ((b >> 23) & 0xFF) - 127
    m = b & ((1 << 23) - 1)
    return e + (m != 0).to(torch.int32)


def _fast_pow2(x: torch.Tensor) -> torch.Tensor:
    """2**x for integer x."""
    return ((x + 127) << 23).view(torch.float32)


def act_quant(x: torch.Tensor, block: int = 32, round_scale: bool = True,
              inplace: bool = True):
    """Block-wise FP8 E4M3 (de)quantization with UE8M0 pow2 scales.

    window KV path: block=32, round_scale=True, scale dtype UE8M0.
    Returns (x_fake_quantized, scales); when inplace, x is overwritten.
    """
    if QUANT_MODE == "NONE":
        return (x, None) if inplace else (x, None)
    lead, N = x.shape[:-1], x.size(-1)
    M = 1
    for s in lead:
        M *= s
    xb = x.reshape(M, N // block, block).float()
    amax = xb.abs().amax(-1).clamp_min(1e-4)
    if round_scale:
        s = _fast_pow2(_fast_log2_ceil(amax / FP8_MAX))
    else:
        s = amax / FP8_MAX
    s = s.unsqueeze(-1)                                    # [M, N/block, 1]
    q = (xb / s).clamp(-FP8_MAX, FP8_MAX)
    q = q.to(torch.float8_e4m3fn).float() * s              # fake-quant round trip
    # straight-through estimator: forward uses quantized values, backward passes to x
    q = xb + (q - xb).detach()
    y = q.view(M, N).view(*lead, N).to(x.dtype)
    if inplace:
        x.copy_(y)
    scales = s.view(M, N // block).to(torch.float8_e8m0fnu)
    return y, scales


def fp4_act_quant(x: torch.Tensor, block: int = 32, e4m3_scale: bool = False,
                  inplace: bool = True):
    """Block-wise FP4 E2M1 fake-quantization.

    indexer q/k  : block=32, e4m3_scale=False  (UE8M0 pow2 scale)
    compressed KV: block=16, e4m3_scale=True   (E4M3 scale, floor 6*2**-9)
    """
    if QUANT_MODE == "NONE":
        return (x, None) if inplace else (x, None)
    lead, N = x.shape[:-1], x.size(-1)
    M = 1
    for s_ in lead:
        M *= s_
    xb = x.reshape(M, N // block, block).float()
    amax = xb.abs().amax(-1)
    if e4m3_scale:
        amax = amax.clamp_min(6 * 2 ** -9)
        s = (amax / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(6 * 2 ** -126)
        s = _fast_pow2(_fast_log2_ceil(amax / FP4_MAX))
    s = s.unsqueeze(-1)
    v = (xb / s).clamp(-FP4_MAX, FP4_MAX).abs()
    # torch.bucketize with right=False maps value v to index of first edge > v;
    # the EDGES array encodes round-half-to-even on the 8-value grid.
    idx = torch.bucketize(v.contiguous(), FP4_EDGES.to(v.device).contiguous())
    q = FP4_GRID.to(v.device)[idx] * torch.sign(xb)
    # straight-through estimator (bucketize/sign are not differentiable)
    q = xb + ((q * s.unsqueeze(-1)) - xb).detach()
    y = q.view(M, N).view(*lead, N).to(x.dtype)
    if inplace:
        x.copy_(y)
    scale_dt = torch.float8_e4m3fn if e4m3_scale else torch.float8_e8m0fnu
    return y, s.view(M, N // block).to(scale_dt)
```

`QUANT_MODE` semantics (§14.6): `"FAKE"` reproduces the trained model exactly (recommended); `"NONE"` runs unquantized bf16/fp32 (architecture debugging / small reproductions); `"REAL"` stores packed low-bit + scales and dequantizes in kernels (deployment parity).

### 21.3 Norm, RoPE (§5.3, §9)

```python
import math


class RMSNorm(torch.nn.Module):
    """Normalize in fp32, multiply scale in fp32, cast back. eps is literally rms_norm_eps."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dt = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * xf).to(dt)


def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    """Rotary frequencies as complex exponentials, one row per position. original_seq_len>0 => YaRN."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def _rope_tail(t, rd, freqs, inverse=False):
    """Out-of-place partial RoPE: rotate the last `rd` channels, keep the rest.

    The training reference never mutates its inputs (the released kernels do the
    rotation in place; an inference-only build may restore `t[..., -rd:] = ...`)."""
    head, tail = t[..., :-rd], t[..., -rd:]
    return torch.cat([head, apply_rope(tail, freqs, inverse)], dim=-1)

def apply_rope(x, freqs_cis, inverse=False):
    """Rotate the last dim's adjacent pairs. x: [b,s,d] or [b,s,h,d]. Pure: returns new tensor."""
    f = freqs_cis.conj() if inverse else freqs_cis
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))     # complex FIRST...
    if xc.ndim == 3:
        f = f.view(1, xc.size(1), xc.size(-1))                     # ...then broadcast to complex width
    else:
        f = f.view(1, xc.size(1), 1, xc.size(-1))
    xc = torch.view_as_real(xc * f).flatten(-2)
    return xc.to(x.dtype)
```

### 21.4 Hyper-connections (§5)

```python
def hc_split_sinkhorn(mixes, hc, scale, base, iters, eps):
    """mixes: [b,s,(2+hc)*hc] -> (pre [b,s,hc], post [b,s,hc], comb [b,s,hc,hc])."""
    b, s, _ = mixes.shape
    pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * scale[1] + base[hc:2 * hc])
    comb = mixes[..., 2 * hc:].view(b, s, hc, hc) * scale[2] + base[2 * hc:].view(hc, hc)
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def hc_pre(x, pre_mix):
    """Collapse hc copies: [b,s,hc,d] x [b,s,hc] -> [b,s,d]."""
    return (pre_mix.unsqueeze(-1) * x.float()).sum(2).to(x.dtype)


def hc_post(y, residual, post, comb):
    """Expand sublayer output: out[k] = post[k]*y + sum_j comb[j,k]*residual[j]."""
    out = post.unsqueeze(-1) * y.unsqueeze(-2)
    out = out + (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(2)
    return out.to(y.dtype)


def hc_mixes(x, hc_fn, hc_scale, hc_base, hc, iters, eps, norm_eps):
    x = x.flatten(2).float()                                       # [b,s,hc*d]
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    mixes = torch.nn.functional.linear(x, hc_fn) * rsqrt
    return hc_split_sinkhorn(mixes, hc, hc_scale, hc_base, iters, eps)


def make_identity_pre_mix(b, s, hc):
    """First block's pre is a one-hot on stream copy 0."""
    m = torch.zeros(b, s, hc, dtype=torch.float32)
    m[..., 0] = 1.0
    return m
```

### 21.5 Sparse attention and index helpers (§7.4, §8.4)

```python
def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """q [b,m,h,d] rotated; kv [b,n,d] single latent (K==V) rotated; attn_sink [h];
    topk_idxs [b,m,k] int32, -1 = masked. Sink enters the denominator only."""
    b, m, h, d = q.shape
    idx = topk_idxs.long()
    valid = idx >= 0
    safe = idx.clamp_min(0).clamp_max(kv.size(1) - 1)
    kvg = torch.gather(kv.unsqueeze(1).expand(b, m, kv.size(1), d), 2,
                       safe.unsqueeze(-1).expand(b, m, safe.size(-1), d))   # [b,m,k,d]
    kvg = kvg * valid.unsqueeze(-1)
    sc = torch.einsum("bmhd,bmkd->bmkh", q.float(), kvg.float()) * softmax_scale   # [b,m,k,h]
    sc = sc.masked_fill(~valid.unsqueeze(-1), float("-inf"))
    mx = sc.amax(-2).clamp_min(-1e30)                  # max over keys; finite floor -> empty rows give 0
    p = torch.exp(sc - mx.unsqueeze(-2))
    den = p.sum(-2) + torch.exp(attn_sink.float().view(1, 1, h) - mx)
    acc = torch.einsum("bmkh,bmkd->bmhd", p, kvg.float())
    return (acc / den.unsqueeze(-1)).to(q.dtype)


def get_window_topk_idxs(window_size, bsz, seqlen, start_pos):
    """Which sliding-window ring slots each query attends to; -1 = empty. int32, materialized."""
    if start_pos == 0:
        end = torch.arange(seqlen).unsqueeze(1)
        idxs = (end - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        idxs = torch.where(idxs > end, -1, idxs)
    else:
        oldest = start_pos % window_size + 1
        idxs = torch.cat([torch.arange(oldest, window_size), torch.arange(oldest)])
        idxs = torch.where(idxs > start_pos, -1, idxs)
    return idxs.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


def get_chunk_window_topk_idxs(window_size, bsz, seqlen, start_pos):
    """Chunk-general sliding-window indices (start_pos > 0, seqlen > 1).

    The window KV handed to attention is [ring (win slots) | chunk (seqlen rows)].
    Query j (absolute position p = start_pos + j) attends cached positions
    q in [max(0, p-win+1), start_pos) at ring slot q % win, and chunk positions
    q in [start_pos, p] at win + (q - start_pos). Rows padded with -1 to width win.
    """
    rows = []
    for j in range(seqlen):
        p = start_pos + j
        lo = max(0, p - window_size + 1)
        qs = torch.arange(lo, p + 1)
        slots = torch.where(qs < start_pos, qs % window_size, window_size + (qs - start_pos))
        pad = torch.full((window_size - slots.numel(),), -1, dtype=torch.int32)
        rows.append(torch.cat([slots.to(torch.int32), pad]))
    return torch.stack(rows).unsqueeze(0).expand(bsz, -1, -1).contiguous()


def get_dspark_topk_idxs(window_size, bsz, block_size, start_pos):
    """Draft queries see the whole backbone window (oldest first) plus all draft-block tokens."""
    matrix = torch.cat([
        torch.arange(min(window_size, start_pos + 1)),
        window_size + torch.arange(block_size),
    ])
    return matrix.int().view(1, 1, -1).expand(bsz, block_size, -1).contiguous()


def select_candidate_blocks(logits, compress_lens, topk_blocks, block_size):
    """Level 1 of the hierarchical indexer. Returns a bool mask shaped like logits."""
    width = logits.size(-1)
    padded = torch.nn.functional.pad(logits, (0, -width % block_size), value=float("-inf"))
    scores = padded.unflatten(-1, (-1, block_size)).amax(-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size                 # pin the newest block in
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last,
                                float("inf"))
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices,
                                                              top.values > -float("inf"))
    return keep.repeat_interleave(block_size, -1)[..., :width]
```

### 21.6 Compressor (§8.1)

```python
class PoolState:
    """Partial-group state for a ratio>1 compressor, in ABSOLUTE position terms.

    Holds the `count` buffered tokens of the current incomplete group (all belonging to
    group `floor(next_position / ratio)`). Invariant: count == (next start_pos) % ratio.
    Carries the group across decode steps AND across chunked-prefill boundaries.
    """

    def __init__(self, batch, ratio, dim):
        self.ratio = ratio
        self.count = 0
        self.kv = torch.zeros(batch, ratio, dim, dtype=torch.float32)
        self.score = torch.full((batch, ratio, dim), -torch.inf, dtype=torch.float32)


class Compressor(torch.nn.Module):
    """Pools `ratio` consecutive tokens into one KV latent with a learned softmax gate.

    Chunk-general unified forward: prepends the buffered partial group, pools every
    complete absolute group, keeps the remainder. Fresh prefill (count==0, start==0),
    1-token decode, and chunked prefill are all the same code path.
    Returns (latent | None, n_groups, first_group_index).
    """

    def __init__(self, dim, head_dim, ratio, eps):
        super().__init__()
        self.ratio, self.head_dim, self.eps = ratio, head_dim, eps
        self.norm = RMSNorm(head_dim, eps)
        dt = torch.float32   # fp32 throughout the runnable reference; ckpt dtypes in §14/§16
        self.wkv = torch.nn.Linear(dim, head_dim, bias=False).to(dt)
        if ratio > 1:
            self.wgate = torch.nn.Linear(dim, head_dim, bias=False).to(torch.float32)

    def forward(self, x, start_pos, state):
        r = self.ratio
        if r == 1:
            # one latent per token: n_groups == S, groups start at absolute start_pos
            return self.norm(self.wkv(x)), x.size(1), start_pos
        kv, score = self.wkv(x.float()), self.wgate(x.float())
        S = x.size(1)
        pend = state.count
        assert pend == start_pos % r, (pend, start_pos, r)    # absolute-alignment invariant
        if pend:
            kv = torch.cat([state.kv[:, :pend], kv], dim=1)
            score = torch.cat([state.score[:, :pend], score], dim=1)
        total = pend + S
        n_groups, rem = total // r, total % r
        out = None
        if n_groups:
            gk = kv[:, :n_groups * r].unflatten(1, (-1, r))
            gs = score[:, :n_groups * r].unflatten(1, (-1, r))
            out = self.norm(((gk * gs.softmax(dim=2)).sum(dim=2)).to(x.dtype))
        if rem:
            state.kv[:, :rem] = kv[:, -rem:]
            state.score[:, :rem] = score[:, -rem:]
            state.count = rem
        else:
            state.count = 0
        first_group = (start_pos - pend) // r
        return out, n_groups, first_group
```

### 21.7 Indexer (§8.3)

```python
class Indexer(torch.nn.Module):
    """Keeps the index_topk best compressed positions per query. Shares the attention's qr."""

    def __init__(self, cfg, layer_id, owns_k):
        super().__init__()
        self.cfg, self.owns_k = cfg, owns_k
        self.ratio = cfg.compress_ratios[layer_id]
        self.is_candidate_source = layer_id == cfg.candidate_source_layer_id
        self.uses_candidates = 0 <= cfg.candidate_source_layer_id < layer_id
        self.softmax_scale = cfg.index_head_dim ** -0.5
        self.wq_b = torch.nn.Linear(cfg.q_lora_rank, cfg.index_n_heads * cfg.index_head_dim,
                                    bias=False)
        self.weights_proj = torch.nn.Linear(cfg.dim, cfg.index_n_heads, bias=False)
        if owns_k:
            self.wk = torch.nn.Linear(cfg.head_dim, cfg.index_head_dim, bias=False)
            self.k_norm = RMSNorm(cfg.index_head_dim, cfg.rms_norm_eps)

    def make_keys(self, latent, freqs_cis, first_group):
        """latent: pre-RoPE, pre-quant compressed latent produced by this layer's compressor.

        A latent stands for the FIRST token of its group, so group g is rotated at g*ratio."""
        n = latent.size(1)
        r = self.ratio
        freqs = freqs_cis[first_group * r: (first_group + n) * r: r]
        k = self.k_norm(self.wk(latent))
        k = _rope_tail(k, self.cfg.qk_rope_head_dim, freqs)
        k, _ = fp4_act_quant(k, block=32, e4m3_scale=False, inplace=False)
        return k

    def forward(self, x, qr, freqs_cis, start_pos, offset, index_k):
        cfg = self.cfg
        b, S, _ = x.shape
        end = start_pos + S
        q = self.wq_b(qr).view(b, S, cfg.index_n_heads, cfg.index_head_dim)
        q = _rope_tail(q, cfg.qk_rope_head_dim, freqs_cis[start_pos:end])
        q, _ = fp4_act_quant(q, block=32, e4m3_scale=False, inplace=False)
        w = self.weights_proj(x).float() * (self.softmax_scale * cfg.index_n_heads ** -0.5)
        score = torch.einsum("bshd,btd->bsht", q.float(), index_k[:b, :end // self.ratio].float())
        score = (score.relu() * w.unsqueeze(-1)).sum(dim=2)          # [b,S,R]
        # chunk-general reachability: query at absolute position p sees groups completed by p,
        # i.e. group j iff j*ratio + (ratio-1) <= p  <=>  j <= (p+1)//ratio - ... == j < cl[p]
        R = score.size(-1)
        cl = ((start_pos + torch.arange(1, S + 1, device=x.device)) // self.ratio
              ).view(1, S, 1).expand(b, S, 1)                        # [b,S,1] visible counts
        score = score.masked_fill(torch.arange(R, device=x.device).view(1, 1, R) >= cl,
                                  float("-inf"))
        if self.is_candidate_source:
            shared.candidates = select_candidate_blocks(score, cl, cfg.candidate_topk_blocks,
                                                        cfg.candidate_block_size)
        elif self.uses_candidates:
            score = score.masked_fill(~shared.candidates, float("-inf"))
        k = min(cfg.index_topk, end // self.ratio)
        idx = score.topk(k, dim=-1, sorted=False).indices.sort(dim=-1).values
        return torch.where(idx < cl, idx + offset, -1).int()
```

### 21.8 Attention (§7.2, §7.3, §8.2)

```python
class SharedAttentionRuntime:
    """What attention layers hand down the stack instead of recomputing. One slot per quantity;
    correctness relies on ascending layer order and source-writes-before-consumers-read."""
    compress_kv = None
    index_k = None
    topk_idxs = None
    candidates = None


shared = SharedAttentionRuntime()


class Attention(torch.nn.Module):
    """Latent attention over two KV sources in one call: sliding window + top-k compressed."""

    def __init__(self, cfg, layer_id):
        super().__init__()
        self.cfg, self.layer_id = cfg, layer_id
        self.dim, self.n_heads, self.hd = cfg.dim, cfg.n_heads, cfg.head_dim
        self.rd, self.window, self.ratio = cfg.qk_rope_head_dim, cfg.sliding_window, cfg.compress_ratios[layer_id]
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
        self.compressor = Compressor(cfg.dim, self.hd, self.ratio, cfg.rms_norm_eps) \
            if self.is_kv_source else None
        self.indexer = Indexer(cfg, layer_id, self.is_kv_source) if self.is_index_source else None
        self.pool_state = None                                   # created lazily in forward
        if self.ratio:
            orig, theta = cfg.rope_original_seq_len, cfg.compress_rope_theta
        else:
            orig, theta = 0, cfg.rope_theta
        self.register_buffer("freqs_cis", precompute_freqs_cis(
            self.rd, cfg.max_seq_len, orig, theta, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow),
            persistent=False)

    # --- caches -------------------------------------------------------------
    def init_caches(self, batch, max_len):
        self.window_kv = torch.zeros(batch, self.window, self.hd, dtype=torch.float32)
        if self.is_kv_source:
            self.compress_kv = torch.zeros(batch, max_len // self.ratio, self.hd,
                                           dtype=torch.float32)
            self.index_k = torch.zeros(batch, max_len // self.ratio, self.cfg.index_head_dim,
                                       dtype=torch.float32)
            self.pool_state = PoolState(batch, self.ratio, self.hd)
        else:
            self.compress_kv = None
            self.index_k = None

    # --- sliding window ------------------------------------------------------
    def window_branch(self, x, freqs, start_pos):
        b, S, _ = x.shape
        kv = self.kv_norm(self.wkv(x))
        kv = _rope_tail(kv, self.rd, freqs)
        kv, _ = act_quant(kv, block=32, round_scale=True, inplace=False)
        if start_pos == 0:
            # reference prefill path, verbatim: attend the chunk, seed the ring with its tail
            if S <= self.window:
                self.window_kv[:b, :S] = kv
            else:
                cutoff = S % self.window
                tail = kv[:, -self.window:]
                self.window_kv[:b, cutoff:self.window] = tail[:, :self.window - cutoff]
                self.window_kv[:b, :cutoff] = tail[:, self.window - cutoff:]
            return kv, get_window_topk_idxs(self.window, b, S, start_pos)
        if S == 1:
            # reference 1-token decode path, verbatim: write the slot, attend the whole ring
            self.window_kv[:b, start_pos % self.window] = kv[:, 0]
            return self.window_kv[:b], get_window_topk_idxs(self.window, b, S, start_pos)
        # chunk-general path (start_pos > 0, S > 1): attend [ring | chunk], one row per query
        n = min(S, self.window)
        for i in range(n):                          # rotate the chunk tail into the ring
            self.window_kv[:b, (start_pos + S - n + i) % self.window] = kv[:, S - n + i]
        window_kv = torch.cat([self.window_kv[:b], kv], dim=1)     # N_w = win + S
        return window_kv, get_chunk_window_topk_idxs(self.window, b, S, start_pos)

    # --- compressed KV + indexer -------------------------------------------
    def compressed_branch(self, x, qr, freqs, start_pos, offset):
        b, S, _ = x.shape
        r = self.ratio
        compress_len = (start_pos + S) // r
        latent, n_groups, first_group = None, 0, 0
        if self.is_kv_source:
            latent, n_groups, first_group = self.compressor(x, start_pos, self.pool_state)
            shared.compress_kv = self.compress_kv             # publish BEFORE consumers read
        # indexer consumes the PRE-RoPE latent -> must run before rotation/quant below
        if self.is_index_source:
            if compress_len == 0:
                idxs = torch.empty(b, S, 0, dtype=torch.int32, device=x.device)
            else:
                if self.indexer.owns_k and latent is not None:
                    k = self.indexer.make_keys(latent, self.freqs_cis, first_group)
                    self.index_k[:b, first_group: first_group + n_groups] = k
                    shared.index_k = self.index_k
                idxs = self.indexer(x, qr, self.freqs_cis, start_pos, offset, shared.index_k)
            shared.topk_idxs = idxs
        else:
            idxs = shared.topk_idxs
        if latent is not None:
            # group g rotates at absolute position g*ratio (== first token of the group)
            freqs_g = self.freqs_cis[first_group * r: (first_group + n_groups) * r: r]
            latent = _rope_tail(latent, self.rd, freqs_g)
            latent, _ = fp4_act_quant(latent, block=16, e4m3_scale=True, inplace=False)
            self.compress_kv[:b, first_group: first_group + n_groups] = latent
        return shared.compress_kv[:b, :compress_len], idxs

    # --- forward ------------------------------------------------------------
    def forward(self, x, start_pos):
        b, S, _ = x.shape
        cfg = self.cfg
        freqs = self.freqs_cis[start_pos: start_pos + S]
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).view(b, S, cfg.n_heads, self.hd)
        q = _rope_tail(q, self.rd, freqs)

        kv, idxs = self.window_branch(x, freqs, start_pos)
        if self.ratio:
            ckv, cidx = self.compressed_branch(x, qr, freqs, start_pos, kv.size(1))
            kv = torch.cat([kv, ckv], dim=1)
            idxs = torch.cat([idxs, cidx], dim=-1)

        o = sparse_attn(q, kv, self.attn_sink, idxs.contiguous(), self.softmax_scale)
        o = _rope_tail(o, self.rd, freqs, inverse=True)

        g = cfg.o_groups
        o = o.view(b, S, g, -1).to(self.wo_a.weight.dtype)
        wo_a_w = self.wo_a.weight.view(g, cfg.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a_w)
        return self.wo_b(o.flatten(2))
```

### 21.9 MoE (§6)

```python
class Gate(torch.nn.Module):
    """sqrtsoftplus scoring; correction bias reorders selection only, not weighting."""

    def __init__(self, cfg, n_experts, topk):
        super().__init__()
        self.n_experts, self.topk = n_experts, topk
        self.score_func = cfg.scoring_func
        self.route_scale = cfg.routed_scaling_factor
        self.gate_temp = 1.0
        self.weight = torch.nn.Parameter(torch.empty(n_experts, cfg.dim))
        self.bias = torch.nn.Parameter(torch.zeros(n_experts, dtype=torch.float32))
        self.bias_vl = torch.nn.Parameter(torch.zeros(n_experts, dtype=torch.float32))

    def forward(self, x, image_mask=None):
        scores = torch.nn.functional.linear(x.float(), self.weight.float()) / self.gate_temp
        if self.score_func == "sqrtsoftplus":
            scores = torch.nn.functional.softplus(scores).sqrt()
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = scores.softmax(-1)
        bias = self.bias
        if image_mask is not None:
            bias = torch.where(image_mask.unsqueeze(-1), self.bias_vl, bias)
        idx = (scores + bias).topk(self.topk, dim=-1)[1]
        w = scores.gather(1, idx)
        if self.topk > 1:
            w = w / (w.sum(-1, keepdim=True) + 1e-20)       # literally 1e-20
        return w * self.route_scale, idx


class Expert(torch.nn.Module):
    """Clamped SwiGLU: up two-sided, gate upper-only, fp32 product."""

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
        y = torch.nn.functional.silu(gate) * up
        if weights is not None:
            y = weights * y
        return self.w2(y.to(x.dtype))


class MoE(torch.nn.Module):
    """top-k routed experts + one shared expert every token passes through."""

    def __init__(self, cfg, layer_id):
        super().__init__()
        n_e, topk = cfg.get_moe_config(layer_id)
        self.n_experts, self.topk = n_e, topk
        self.gate = Gate(cfg, n_e, topk)
        self.experts = torch.nn.ModuleList([Expert(cfg.dim, cfg.moe_intermediate_size,
                                                   cfg.swiglu_limit) for _ in range(n_e)])
        self.shared_experts = Expert(cfg.dim, cfg.moe_intermediate_size, cfg.swiglu_limit)

    def forward(self, x, image_mask=None):
        shape = x.shape
        xf = x.reshape(-1, shape[-1])
        w, idx = self.gate(xf, None if image_mask is None else image_mask.reshape(-1))
        y = torch.zeros_like(xf, dtype=torch.float32)
        for e in range(self.n_experts):
            rows, top = torch.where(idx == e)
            if rows.numel():
                y[rows] += self.experts[e](xf[rows], w[rows, top, None])
        y += self.shared_experts(xf)
        return y.to(x.dtype).view(shape)
```

### 21.10 Engram (§11)

```python
import numpy as np


def _next_prime(start, seen):
    from sympy import isprime
    c = start + 1
    while not isprime(c) or c in seen:
        c += 1
    return c


class EngramLayout:
    """Each (n-gram size, head) owns a disjoint prime-sized bucket range; primes never reused."""

    def __init__(self, cfg):
        self.max_ngram = cfg.engram_max_ngram_size
        self.layer_ids = tuple(cfg.engram_layer_ids)
        self.n_heads = cfg.engram_n_heads
        self.head_dim = cfg.engram_head_dim
        primes, seen = [], set()
        for _ in self.layer_ids:
            per = []
            for _ in range(self.max_ngram - 1):
                sizes, cur = [], cfg.engram_vocab_size - 1
                for _ in range(self.n_heads):
                    cur = _next_prime(cur, seen)
                    seen.add(cur)
                    sizes.append(cur)
                per.append(tuple(sizes))
            primes.append(tuple(per))
        self.primes = primes
        self.num_embeddings = tuple(_sum_primes(cfg, p) for p in primes)   # == engram_num_embeddings
        flat = [[p for per in layer for p in per] for layer in primes]
        self.offsets = [np.cumsum([0, *s[:-1]]) for s in flat]
        bound = max(1, (np.iinfo(np.int64).max // cfg.engram_compressed_vocab_size) // 2)
        rows = []
        for lid in self.layer_ids:
            g = np.random.default_rng(10007 * lid)
            rows.append(torch.tensor(g.integers(0, bound, size=(self.max_ngram,),
                                                dtype=np.int64) * 2 + 1))
        self.multipliers = torch.stack(rows)


def _sum_primes(cfg, per_layer):
    return int(sum(p for per in per_layer for p in per))


class NgramHashState(torch.nn.Module):
    """Maps each position to the hash ids of the n-grams ending there."""

    DEAD = -1

    def __init__(self, cfg, layout, token_map, compressed_vocab):
        super().__init__()
        self.layout = layout
        self.pad_id = token_map[cfg.engram_pad_token_id]
        self.register_buffer("token_map", torch.tensor(token_map), persistent=False)
        self.register_buffer("primes", torch.tensor(layout.primes), persistent=False)
        self.register_buffer("offsets", torch.tensor(np.array(layout.offsets)), persistent=False)
        self.register_buffer("multipliers", layout.multipliers, persistent=False)
        self.register_buffer("cache", torch.zeros(cfg.max_batch_size, cfg.max_seq_len,
                                                  dtype=torch.long), persistent=False)

    @torch.no_grad()
    def forward(self, input_ids, start_pos, token_mask=None):
        b, S = input_ids.shape
        compressed = self.token_map[input_ids]
        if token_mask is not None:
            compressed = torch.where(token_mask, compressed, self.DEAD)
        self.cache[:b, start_pos: start_pos + S] = compressed
        pos = torch.arange(start_pos, start_pos + S, device=input_ids.device).expand(b, S)
        tokens, blocked = [], torch.zeros_like(pos, dtype=torch.bool)
        for shift in range(self.layout.max_ngram):
            src = self.cache[:b].gather(1, (pos - shift).clamp_min(0))
            blocked = blocked | (pos < shift) | (src == self.DEAD)
            tokens.append(torch.where(blocked, self.pad_id, src))
        tokens = torch.stack(tokens, dim=-1)                                   # [b,S,K]
        products = tokens.unsqueeze(2) * self.multipliers.to(tokens.device)      # [b,S,Lyr,K]
        rolling, hashes = products[..., 0], []
        for i in range(1, self.layout.max_ngram):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append((rolling.unsqueeze(-1) % self.primes[:, i - 1]))
        return torch.cat(hashes, dim=-1) + self.offsets


class ParallelEngramEmbedding(torch.nn.Module):
    """fp8 rows + ue8m0 row scales, dequantized per lookup.

    weight: [rows, head_dim] fp8 storage (here fp32 for clarity); scale: [rows, head_dim//32] ue8m0.
    """
    def __init__(self, rows, dim, block=32):
        super().__init__()
        self.block, self.dim = block, dim
        self.weight = torch.nn.Parameter(torch.zeros(rows, dim, dtype=torch.float32))
        self.scale = torch.nn.Parameter(torch.zeros(rows, dim // block, dtype=torch.float32))

    def forward(self, idx):
        v = self.weight[idx].float().view(*idx.shape, -1, self.block)
        s = self.scale[idx].float().unsqueeze(-1)
        return (v * s).flatten(-2).to(self.weight.dtype)


class Engram(torch.nn.Module):
    """Writes an n-gram lookup into the residual stream, gated by stream/key agreement."""

    def __init__(self, cfg, layer_index):
        super().__init__()
        self.dim, self.hc = cfg.dim, cfg.hc_mult
        self.eps, self.clamp_value = cfg.rms_norm_eps, 1e-6
        self.embed = ParallelEngramEmbedding(cfg.engram_num_embeddings[layer_index], cfg.engram_head_dim)
        cols = (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads
        self.wkv = torch.nn.Linear(cols * cfg.engram_head_dim, cfg.dim * (self.hc + 1), bias=False)
        self.q_weight = torch.nn.Parameter(torch.ones(self.hc, cfg.dim))
        self.k_weight = torch.nn.Parameter(torch.ones(self.hc, cfg.dim))

    def forward(self, x, hash_ids, token_mask=None):
        kv = self.wkv(self.embed(hash_ids).flatten(-2))
        key, value = kv.split([self.hc * self.dim, self.dim], dim=-1)
        key = key.view(*key.shape[:-1], self.hc, self.dim)
        w = self.q_weight.float() * self.k_weight.float()
        h, eps = x.float(), self.eps
        rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
        dot = (h * w * key).sum(-1) * rstd * (self.dim ** -0.5)
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)
```

### 21.11 Block and Transformer (§4.2, §5.6)

```python
class Block(torch.nn.Module):
    """Residual stream is hc parallel copies (Hyper-Connections, Single-Pass hand-off)."""

    attention_cls = Attention

    def __init__(self, cfg, layer_id, engram_layout=None):
        super().__init__()
        self.layer_id, self.cfg = layer_id, cfg
        self.engram_idx = (engram_layout.layer_ids.index(layer_id)
                           if engram_layout and layer_id in engram_layout.layer_ids else None)
        self.attn = self.attention_cls(cfg, layer_id)
        self.ffn = MoE(cfg, layer_id)
        self.engram = Engram(cfg, self.engram_idx) if self.engram_idx is not None else None
        self.attn_norm = RMSNorm(cfg.dim, cfg.rms_norm_eps)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.rms_norm_eps)
        self.hc_mult = hc = cfg.hc_mult
        self.hc_sinkhorn_iters, self.hc_eps = cfg.hc_sinkhorn_iters, cfg.hc_eps
        self.norm_eps = cfg.rms_norm_eps
        mix_hc = (2 + hc) * hc
        self.hc_attn_fn = torch.nn.Parameter(torch.empty(mix_hc, hc * cfg.dim, dtype=torch.float32))
        self.hc_ffn_fn = torch.nn.Parameter(torch.empty(mix_hc, hc * cfg.dim, dtype=torch.float32))
        self.hc_attn_base = torch.nn.Parameter(torch.zeros(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = torch.nn.Parameter(torch.zeros(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = torch.nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.hc_ffn_scale = torch.nn.Parameter(torch.ones(3, dtype=torch.float32))

    def mixes(self, x, fn, scale, base):
        return hc_mixes(x, fn, scale, base, self.hc_mult, self.hc_sinkhorn_iters,
                        self.hc_eps, self.norm_eps)

    def forward(self, x, start_pos, pre_mix, image_mask, *attn_args):
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
        x = self.ffn(x, image_mask)
        x = hc_post(x, residual, ffn_post, ffn_comb)
        return x, ffn_pre                     # <- consumed by the next sublayer


class ParallelHead(torch.nn.Module):
    """LM head. Weights run in fp32 even when the checkpoint ships bf16."""

    def __init__(self, vocab, dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(vocab, dim, dtype=torch.float32))

    def forward(self, x, full_logits=False):
        if not full_logits:
            x = x[:, -1]
        return torch.nn.functional.linear(x.float(), self.weight)


class Transformer(torch.nn.Module):
    def __init__(self, cfg, engram_layout=None, engram_hashes=None):
        super().__init__()
        self.cfg, self.n_layers = cfg, cfg.n_layers
        self.hc_mult = cfg.hc_mult
        self.embed = torch.nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = torch.nn.ModuleList([Block(cfg, i, engram_layout)
                                           for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, cfg.rms_norm_eps)
        self.head = ParallelHead(cfg.vocab_size, cfg.dim)
        self.engram_layout, self.engram_hash = engram_layout, engram_hashes
        self.target_layer_ids = tuple(cfg.dspark_target_layer_ids)

        # DSpark draft stages, appended after the backbone (layer ids n_layers..)
        self.mtp = torch.nn.ModuleList()
        for stage in range(cfg.num_nextn_predict_layers):
            blk = DSparkBlock(cfg, cfg.n_layers + stage)
            if stage == 0:
                blk.main_proj = torch.nn.Linear(cfg.dim * len(cfg.dspark_target_layer_ids), cfg.dim,
                                                bias=False)
                blk.main_norm = RMSNorm(cfg.dim, cfg.rms_norm_eps)
            if stage == cfg.num_nextn_predict_layers - 1:
                blk.norm = RMSNorm(cfg.dim, cfg.rms_norm_eps)
                blk.markov_head = DSparkMarkovHead(cfg)
                blk.confidence_head = DSparkConfidenceHead(cfg)
            self.mtp.append(blk)
        for blk in self.mtp:                       # weight-tied to the backbone
            blk.embed, blk.head = self.embed, self.head

        if cfg.vision_n_layers:
            self.vision = ViT(cfg)
            self.aligner = Aligner(cfg)
            self.image_start = torch.nn.Parameter(torch.zeros(cfg.dim))
            self.image_end = torch.nn.Parameter(torch.zeros(cfg.dim))
            self.image_newline = torch.nn.Parameter(torch.zeros(cfg.dim))

        self.init_caches(cfg.max_batch_size)

    def init_caches(self, batch):
        for layer in list(self.layers) + list(self.mtp):
            layer.attn.init_caches(batch, self.cfg.max_seq_len)

    def reset_shared(self):
        # The reference NEVER resets the shared runtime between forwards ("nothing needs
        # resetting between forwards"): cache OBJECTS persist across decode steps so that
        # non-publishing steps still read a valid cache (e.g. ratio-2 owners on odd steps).
        # Per-step activations (topk_idxs, candidates) are always republished by the first
        # producer before any consumer reads them, so no reset is required. This method is a
        # no-op kept only to document the invariant: DO NOT null these slots per step.
        pass

    def forward(self, input_ids, token_types=None, start_pos=0,
                pixel_values=None, image_grid_hws=None):
        cfg = self.cfg
        # NOTE: no reset of the shared runtime here (see reset_shared docstring).
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
            h, pre_mix = layer(h, start_pos, pre_mix, image_mask)

        h = hc_pre(h, pre_mix)                         # final collapse with last ffn pre
        return self.head(self.norm(h)), (torch.cat(main_hiddens, -1) if main_hiddens else None)

    def forward_spec(self, output_ids, main_hidden, start_pos):
        """DSpark pass: drafts a block of tokens conditioned on the backbone's target layers."""
        blk, last = self.mtp[0], self.mtp[-1]
        main_x = blk.main_norm(blk.main_proj(main_hidden))
        ids = output_ids.new_full([output_ids.size(0), self.cfg.dspark_block_size],
                                  self.cfg.dspark_noise_token_id)
        ids[:, 0] = output_ids
        x = self.embed(ids).unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        pre_mix = make_identity_pre_mix(x.size(0), x.size(1), self.hc_mult)
        for m in self.mtp:
            x, pre_mix = m(x, start_pos, pre_mix, main_x)
        if start_pos == 0:
            return None                                # prefill only seeded caches
        return last.forward_head(x, pre_mix, output_ids)
```

### 21.12 DSpark (§12)

```python
class DSparkAttention(Attention):
    """Draft attention over the backbone's window ring + the draft block's own KV."""

    def __init__(self, cfg, layer_id):
        super().__init__(cfg, layer_id)                # compress_ratio must be 0
        assert self.ratio == 0

    def init_caches(self, batch, max_len):
        self.window_kv = torch.zeros(batch, self.window, self.hd, dtype=torch.float32)

    def forward(self, x, start_pos, main_x):
        b, S, _ = main_x.shape
        main_freqs = self.freqs_cis[start_pos: start_pos + S]
        main_kv = self.kv_norm(self.wkv(main_x))
        main_kv = _rope_tail(main_kv, self.rd, main_freqs)
        main_kv, _ = act_quant(main_kv, block=32, round_scale=True, inplace=False)
        if start_pos == 0:                             # prefill: seed the ring only
            if S <= self.window:
                self.window_kv[:b, :S] = main_kv
            else:
                cutoff = S % self.window
                tail = main_kv[:, -self.window:]
                self.window_kv[:b, cutoff:self.window] = tail[:, :self.window - cutoff]
                self.window_kv[:b, :cutoff] = tail[:, self.window - cutoff:]
            return x

        blk = x.size(1)
        freqs = self.freqs_cis[start_pos + S: start_pos + S + blk]
        q = self.wq_b(self.q_norm(self.wq_a(x))).view(b, blk, self.n_heads, self.hd)
        q = _rope_tail(q, self.rd, freqs)
        kv = self.kv_norm(self.wkv(x))
        kv = _rope_tail(kv, self.rd, freqs)
        kv, _ = act_quant(kv, block=32, round_scale=True, inplace=False)
        idxs = get_dspark_topk_idxs(self.window, b, blk, start_pos)
        self.window_kv[:b, start_pos % self.window] = main_kv[:, 0]
        kv = torch.cat([self.window_kv[:b], kv], dim=1)
        o = sparse_attn(q, kv, self.attn_sink, idxs, self.softmax_scale)
        o = _rope_tail(o, self.rd, freqs, inverse=True)
        g = self.cfg.o_groups
        o = o.view(b, blk, g, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, self.wo_a.weight.view(g, self.cfg.o_lora_rank, -1))
        return self.wo_b(o.flatten(2))


class DSparkMarkovHead(torch.nn.Module):
    """Additive logit bias from the previously emitted token id."""

    def __init__(self, cfg):
        super().__init__()
        self.embed = torch.nn.Embedding(cfg.vocab_size, cfg.dspark_markov_rank)
        self.head = torch.nn.Linear(cfg.dspark_markov_rank, cfg.vocab_size, bias=False)

    def forward(self, ids):
        e = self.embed(ids)
        return self.head(e.float()), e


class DSparkConfidenceHead(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.proj = torch.nn.Linear(cfg.dim + cfg.dspark_markov_rank, 1, bias=False)

    def forward(self, hidden, markov_embeds):
        return self.proj(torch.cat([hidden.float(), markov_embeds.float()], -1)).squeeze(-1)


class DSparkBlock(Block):
    attention_cls = DSparkAttention

    def forward(self, x, start_pos, pre_mix, main_x):
        if start_pos == 0:
            self.attn(x, start_pos, main_x)           # seed window caches only
            return x, pre_mix
        return Block.forward(self, x, start_pos, pre_mix, None, main_x)   # drafts are text-only

    def forward_head(self, x, pre_mix, input_ids):
        x = hc_pre(x, pre_mix)
        logits = self.head(self.norm(x), full_logits=True)
        out = input_ids.new_empty([input_ids.size(0), self.cfg.dspark_block_size + 1])
        out[:, 0] = input_ids
        markov = []
        for i in range(self.cfg.dspark_block_size):
            bias, emb = self.markov_head(out[:, i])
            logits[:, i] = logits[:, i] + bias
            markov.append(emb)
            out[:, i + 1] = sample(logits[:, i], self.cfg.temperature)
        conf = self.confidence_head(x, torch.stack(markov, dim=1))
        return out, logits, conf
```

### 21.13 Vision (§13)

```python
def get_vision_cos_sin(n_h, n_w, dim, theta):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def vision_rope(x, cos, sin):
    """Vision uses HALF-SPLIT halves, not the LLM's adjacent-pair convention."""
    dt = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dt)


class ViT(torch.nn.Module):
    """DeepSeek-ViT: bidirectional attention within an image, 2D RoPE, linear patch embed."""

    def __init__(self, cfg):
        super().__init__()
        self.rope_dim = cfg.vision_dim // cfg.vision_n_heads // 2
        self.rope_theta = cfg.vision_rope_theta
        self.patch_embed = torch.nn.Linear(3 * cfg.vision_patch_size ** 2, cfg.vision_dim)
        self.blocks = torch.nn.ModuleList([_ViTBlock(cfg) for _ in range(cfg.vision_n_layers)])
        self.norm = RMSNorm(cfg.vision_dim, 1e-6)

    def forward(self, patches, n_h, n_w):
        x = self.patch_embed(patches)
        cos, sin = get_vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.norm(x)


class _ViTBlock(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads, self.head_dim = cfg.vision_n_heads, cfg.vision_dim // cfg.vision_n_heads
        self.norm1 = RMSNorm(cfg.vision_dim, 1e-6)
        self.wqkv = torch.nn.Linear(cfg.vision_dim, 3 * cfg.vision_dim)
        self.wo = torch.nn.Linear(cfg.vision_dim, cfg.vision_dim)
        self.norm2 = RMSNorm(cfg.vision_dim, 1e-6)
        self.w1 = torch.nn.Linear(cfg.vision_dim, 2 * cfg.vision_inter_dim, bias=False)
        self.w2 = torch.nn.Linear(cfg.vision_inter_dim, cfg.vision_dim, bias=False)

    def forward(self, x, cos, sin):
        h = self.norm1(x)
        n = h.size(0)
        q, k, v = (t.view(n, self.n_heads, self.head_dim)
                   for t in self.wqkv(h).chunk(3, dim=-1))
        q, k = vision_rope(q, cos, sin), vision_rope(k, cos, sin)
        o = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))   # bidirectional
        x = x + self.wo(o.transpose(0, 1).reshape(n, -1))
        gate, up = self.w1(self.norm2(x)).chunk(2, dim=-1)
        return x + self.w2(torch.nn.functional.silu(gate) * up)


class Aligner(torch.nn.Module):
    """3x3 pixel-unshuffle + 2-layer MLP into the LLM hidden size."""

    def __init__(self, cfg):
        super().__init__()
        self.r = cfg.vision_downsample_ratio
        self.w1 = torch.nn.Linear(cfg.vision_dim * self.r ** 2, cfg.dim)
        self.w2 = torch.nn.Linear(cfg.dim, cfg.dim)

    def forward(self, x, n_h, n_w):
        r = self.r
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = torch.nn.functional.pad(x, (0, -n_w % r, 0, -n_h % r))
        x = torch.nn.functional.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return self.w2(torch.nn.functional.gelu(self.w1(x)))


@torch.no_grad()
def merge_image_embeddings(h, token_types, image_embeds, image_start, image_end, image_newline):
    """Scatter ViT outputs into the image_token_id positions; delimiters get learned vectors."""
    h = h.clone()
    IMAGE, START, NL, END = 0, 1, 2, 3
    img, nl, st, en = 0, 0, 0, 0
    for b in range(h.size(0)):
        for i in range(h.size(1)):
            t = int(token_types[b, i])
            if t == IMAGE:
                h[b, i] = image_embeds[img]; img += 1
            elif t == NL:
                h[b, i] = image_newline; nl += 1
            elif t == START:
                h[b, i] = image_start; st += 1
            elif t == END:
                h[b, i] = image_end; en += 1
    return h
```

### 21.14 Config presets (§3.4, §3.6)

```python
@dataclass
class Cfg:
    # shape
    vocab_size: int; dim: int; n_layers: int; n_heads: int; head_dim: int
    q_lora_rank: int; qk_rope_head_dim: int; o_lora_rank: int; o_groups: int
    moe_intermediate_size: int
    # attention schedule
    sliding_window: int
    compress_ratios: tuple; kv_source_layer_ids: tuple; index_source_layer_ids: tuple
    candidate_source_layer_id: int; candidate_topk_blocks: int; candidate_block_size: int
    index_n_heads: int; index_head_dim: int; index_topk: int
    # moe
    n_routed_experts: int; n_shared_experts: int; num_experts_per_tok: int
    scoring_func: str; topk_method: str; norm_topk_prob: bool; routed_scaling_factor: float
    # hyper-connections
    hc_mult: int; hc_sinkhorn_iters: int; hc_eps: float
    # engram
    engram_layer_ids: tuple; engram_num_embeddings: tuple; engram_max_ngram_size: int
    engram_vocab_size: int; engram_n_heads: int; engram_head_dim: int
    engram_pad_token_id: int; engram_compressed_vocab_size: int
    # dspark
    num_nextn_predict_layers: int; dspark_block_size: int; dspark_noise_token_id: int
    dspark_target_layer_ids: tuple; dspark_markov_rank: int
    dspark_n_routed_experts: int; dspark_num_experts_per_tok: int
    # rope
    rope_theta: float; compress_rope_theta: float; rope_factor: float
    beta_fast: int; beta_slow: int; rope_original_seq_len: int; max_seq_len: int
    # numerics / runtime
    rms_norm_eps: float = 1e-20; swiglu_limit: float = 10.0
    max_batch_size: int = 4; temperature: float = 1.0
    image_token_id: int = 129264
    # vision (0 layers disables)
    vision_n_layers: int = 0; vision_dim: int = 1024; vision_n_heads: int = 16
    vision_inter_dim: int = 2816; vision_patch_size: int = 14
    vision_rope_theta: float = 10000.0; vision_downsample_ratio: int = 3

    def get_moe_config(self, layer_id):
        if layer_id < self.n_layers:
            return self.n_routed_experts, self.num_experts_per_tok
        return self.dspark_n_routed_experts, self.dspark_num_experts_per_tok


PRESET_552B = Cfg(
    vocab_size=129280, dim=5120, n_layers=40, n_heads=64, head_dim=512,
    q_lora_rank=1280, qk_rope_head_dim=64, o_lora_rank=1024, o_groups=8,
    moe_intermediate_size=2304, sliding_window=128,
    compress_ratios=tuple([0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]),
    kv_source_layer_ids=(2, 8, 14, 20),
    index_source_layer_ids=(2, 8, 14, 20, 24, 28, 32, 36),
    candidate_source_layer_id=20, candidate_topk_blocks=2048, candidate_block_size=8,
    index_n_heads=32, index_head_dim=128, index_topk=512,
    n_routed_experts=384, n_shared_experts=1, num_experts_per_tok=6,
    scoring_func="sqrtsoftplus", topk_method="noaux_tc", norm_topk_prob=True,
    routed_scaling_factor=1.5, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
    engram_layer_ids=(1, 14), engram_num_embeddings=(384006168, 384016682),
    engram_max_ngram_size=4, engram_vocab_size=16_000_000, engram_n_heads=8,
    engram_head_dim=256, engram_pad_token_id=2, engram_compressed_vocab_size=99092,
    num_nextn_predict_layers=3, dspark_block_size=5, dspark_noise_token_id=128799,
    dspark_target_layer_ids=(37, 38, 39), dspark_markov_rank=256,
    dspark_n_routed_experts=128, dspark_num_experts_per_tok=3,
    rope_theta=10000.0, compress_rope_theta=160000.0, rope_factor=16,
    beta_fast=32, beta_slow=1, rope_original_seq_len=65536, max_seq_len=1048576,
    vision_n_layers=32,
)

PRESET_VERTEX_500M = Cfg(
    vocab_size=65536, dim=512, n_layers=16, n_heads=8, head_dim=64,
    q_lora_rank=128, qk_rope_head_dim=16, o_lora_rank=128, o_groups=4,
    moe_intermediate_size=416, sliding_window=64,
    compress_ratios=tuple([0, 0] + [2] * 6 + [1] * 8 + [0, 0]),
    kv_source_layer_ids=(2, 8), index_source_layer_ids=(2, 8, 12),
    candidate_source_layer_id=8, candidate_topk_blocks=256, candidate_block_size=8,
    index_n_heads=4, index_head_dim=32, index_topk=128,
    n_routed_experts=32, n_shared_experts=1, num_experts_per_tok=2,
    scoring_func="sqrtsoftplus", topk_method="noaux_tc", norm_topk_prob=True,
    routed_scaling_factor=1.5, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
    engram_layer_ids=(1,), engram_num_embeddings=(600766,),   # = sum of the 12 primes
                                                                    # [50021,50023,50033,50047,50051,50053,
                                                                    #  50069,50077,50087,50093,50101,50111]
    engram_max_ngram_size=4, engram_vocab_size=50_021, engram_n_heads=4,
    engram_head_dim=64, engram_pad_token_id=2,
    engram_compressed_vocab_size=0,                            # set to YOUR tokenizer's map size
    num_nextn_predict_layers=2, dspark_block_size=4, dspark_noise_token_id=1,
    dspark_target_layer_ids=(13, 14, 15), dspark_markov_rank=64,
    dspark_n_routed_experts=16, dspark_num_experts_per_tok=2,
    rope_theta=10000.0, compress_rope_theta=160000.0, rope_factor=16,
    beta_fast=32, beta_slow=1, rope_original_seq_len=8192, max_seq_len=32768,
    vision_n_layers=0,
)
```

### 21.15 Generation loop (§21.4 contract)

```python
def sample(logits, temperature):
    if temperature <= 0:
        return logits.argmax(-1)
    probs = torch.softmax(logits.float() / max(temperature, 1e-5), dim=-1)
    return torch.multinomial(probs, 1)                      # [B,1], ready to concat


@torch.no_grad()
def generate(model, prompt_ids, max_new=64, temperature=1.0, use_dspark=False):
    """Prefill in one shot, then 1-token decode. start_pos is threaded explicitly."""
    S = prompt_ids.size(1)
    ids = prompt_ids
    logits, main_hidden = model(ids, start_pos=0)
    if use_dspark:
        model.forward_spec(logits.argmax(-1, keepdim=True), main_hidden, 0)
    for t in range(max_new):
        logits, main_hidden = model(ids[:, -1:], start_pos=S + t)
        nxt = sample(logits[:, -1], temperature)
        ids = torch.cat([ids, nxt], dim=1)
        if use_dspark:
            spec = model.forward_spec(nxt, main_hidden, S + t)
            # spec = (draft_ids [B, block+1], draft_logits, confidence)
            # a production loop accepts the longest prefix whose per-position
            # confidence exceeds the threshold; see §12.1 confidence-scheduled verification
    return ids
```

**Minimal runnable test of the architecture** (shrink `PRESET_VERTEX_500M` further so it runs on CPU in seconds):

```python
tiny = replace(PRESET_VERTEX_500M,
               vocab_size=512, dim=128, n_layers=8, n_heads=4, head_dim=32,
               q_lora_rank=64, qk_rope_head_dim=8, o_lora_rank=64, o_groups=4,
               moe_intermediate_size=64, sliding_window=8, index_n_heads=2,
               index_head_dim=32, index_topk=8, candidate_topk_blocks=4,
               n_routed_experts=8, num_experts_per_tok=2, max_seq_len=64,
               max_batch_size=1, engram_layer_ids=(1,), engram_num_embeddings=(600766,),
               engram_vocab_size=50_021, engram_n_heads=2, engram_head_dim=32,
               engram_compressed_vocab_size=512,
               compress_ratios=(0, 0, 2, 2, 1, 1, 1, 1, 0, 0),   # 8 backbone + 2 DSpark
               kv_source_layer_ids=(2, 4), index_source_layer_ids=(2, 4, 6),
               candidate_source_layer_id=4,
               num_nextn_predict_layers=2, dspark_block_size=3,
               dspark_target_layer_ids=(5, 6, 7), dspark_markov_rank=16,
               dspark_n_routed_experts=4, dspark_num_experts_per_tok=2,
               vision_n_layers=0)

# Engram needs a compressed token map (identity here; real runs derive it from the tokenizer)
layout = EngramLayout(tiny)
token_map = list(range(tiny.vocab_size))
hash_state = NgramHashState(tiny, layout, token_map, tiny.engram_compressed_vocab_size)

model = Transformer(tiny, engram_layout=layout, engram_hashes=hash_state).eval()
ids = torch.randint(0, tiny.vocab_size, (1, 17))

logits, _ = model(ids, start_pos=0)                       # prefill
# the head returns the LAST position only by default (generation never needs the rest)
assert logits.shape == (1, tiny.vocab_size), logits.shape

# decode, one token per step, extending the sequence each time
cur = ids
for i in range(cur.size(1), cur.size(1) + 23):
    nxt = sample(logits, 1.0)
    cur = torch.cat([cur, nxt], dim=1)
    logits, main_hidden = model(cur[:, i:i + 1], start_pos=i)
    assert logits.shape == (1, tiny.vocab_size)
    assert model.layers[2].attn.compress_kv is not None

# DSpark drafting pass at the final position
S = cur.size(1) - 1
spec_out = model.forward_spec(logits.argmax(-1, keepdim=True), main_hidden, S)
assert spec_out is not None and spec_out[0].shape[1] == tiny.dspark_block_size + 1

# chunk-general prefill must match single-shot prefill
# (init_caches once per SEQUENCE, never between chunks of the same sequence)
model.init_caches(1)
a, _ = model(cur[:, :9], start_pos=0)
model.init_caches(1)
b, _ = model(cur[:, :5], start_pos=0)
bb, _ = model(cur[:, 5:9], start_pos=5)     # same caches: pool state, ring, id history carry over
assert torch.allclose(a, bb, atol=2e-2, rtol=2e-2), (a - bb).abs().max()
print("all architecture checks passed")
```

Note on chunk-general prefill: the reference `model.py` implements full prefill and 1-token decode only (its decode branches assume `seqlen == 1`). Chunk-general support requires carrying the window ring, compressor partial-group state, indexer visibility, and the Engram id history across chunk boundaries — an independent MLX port verified this generalization and it is the right acceptance test for any reimplementation.

**Validation status of this reference code.** The implementation above was executed end-to-end on CPU (tiny config, 8 layers, Full/Reuse/Reindex modes, hierarchical indexer, Engram, DSpark): prefill + 23 decode steps + DSpark drafting all produce finite logits; chunked prefill `[0:9]` vs `[0:5]+[5:9]` agrees **bit-exactly (max diff 0.0)**; forward is deterministic across runs; backward flows to routed experts, Engram projections, indexer queries, HC coefficients, and compressors (gate biases correctly receive none — see §25.37).

---

## 22 Worked traces (numeric index arithmetic)

Concrete traces that let an implementer check index conventions without running the 552B model. All use a miniature schedule mirroring the released one:
`compress_ratios = [0,0,2,2,1,1]`, `kv_source = (2,4)`, `index_source = (2,4)`, `candidate_source = 4`, `window = 4`, `index_topk = 3`, `candidate_block_size = 2`, `candidate_topk_blocks = 2`.

### 22.1 Prefill, window indices (`start_pos = 0`, `n_win = 4`, `S = 7`)

```
query i attends slots:  i = 0 -> [-1,-1,-1,0]      (seq start)
                        i = 1 -> [-1,-1,0,1]
                        i = 3 -> [0,1,2,3]
                        i = 6 -> [3,4,5,6]
formula: idxs = (i - win + 1).clamp(0) + arange(min(S,win));  idxs > i -> -1
```
The ring is seeded with the last `n_win` rows: `cutoff = S % win = 3`, so `window_kv[3:4] = kv[-4:][:1]` and `window_kv[:3] = kv[-4:][1:]`.

### 22.2 Prefill, compressor (ratio 2, S = 7)

```
full = 3 groups -> 3 latents from tokens (0,1) (2,3) (4,5)
remainder rem = 1 -> token 6 stored in pool_state (kv_state[:, :1], score_state[:, :1])
latent j is RoPE'd at position j*2 = 0, 2, 4  (freqs[0:7:2] -> rows 0,2,4)
written to compress_kv slots 0,1,2
compress_len = 7 // 2 = 3
```

### 22.3 Prefill, indexer reachability (ratio 2)

```
compress_lens per query i = i // 2   ->   [0,0,1,1,2,2,3]
mask positions j >= compress_lens: query 6 may see j ∈ {0,1,2}
```
A query at position `i` can only see compressed groups fully completed by `i`; because a latent "stands for the first token of its group", group `j` requires `i >= 2j+1`... which is exactly `i//2 > j`.

### 22.4 Prefill, candidate blocks (block_size 2, topk_blocks 2, ratio 2)

```
positions:  0  1 | 2  3 | 4  5 | 6(pad -inf)
block scores = amax per block:  [max(s0,s1), max(s2,s3), max(s4,s5), -inf]
query 6: last = (3-1)//2 = 1 -> pin block 1 to +inf
top-2 blocks -> pool of <= 4 positions -> level-2 indexers score only those
```
Note the *pinned* block guarantees the most recent tokens always remain retrievable even if an older full block scores higher.

### 22.5 Decode step `start_pos = 7`, ratio 2

```
should_compress = (7 + 1) % 2 == 0  -> True, slot = 7 % 2 = 1
pool completes group 3 -> latent RoPE'd at position start_pos + 1 - ratio = 6
written to compress_kv slot start_pos // ratio = 3;  compress_len = 8 // 2 = 4
indexer scores against index_k[:4]  (K published by the kv_source layer for every completed group)
```
Decode step `start_pos = 8` (ratio 2): `should_compress = (9) % 2 == 1 -> False`; the token's `kv`/`score` land in slot 0 and the layer returns `None`; `compress_len` stays 4; the *indexer query* for this step scores the same 4 keys and picks a fresh Top-K.

### 22.6 Decode step, window indices (`start_pos = 7`, `n_win = 4`)

```
oldest = 7 % 4 + 1 = 4      (not a valid slot index; ring wrap already applied)
idxs = cat([arange(4,4) , arange(4)])  = [0,1,2,3]
idxs > start_pos -> none  (ring full)
```
Decode step `start_pos = 2`, `n_win = 4`:
```
oldest = 2 % 4 + 1 = 3
idxs = cat([arange(3,4), arange(3)]) = [3,0,1,2];  idxs > 2 -> [3,-1,-1,-1]
slot 3 has not been written yet (first decode after a 3-token prefill)
```

### 22.7 Cross-layer read offsets

`Attention.forward` concatenates `[window kv (N_w rows), compressed kv (N_c rows)]` and passes `offset = kv_window.size(1)` to `compressed_branch`. Compressed indices therefore read `idxs + N_w`; window indices are already slot indices `< N_w`. At prefill `N_w = S` (the whole chunk is the window); at decode `N_w = n_win`.

### 22.8 Full decode trace of the released model (one token)

```
input: x[:, i:i+1], start_pos = i
layer 0  (ratio 0):  window only; ring slot i % 128 written; attend ring
layer 1  (ratio 0):  window only; Engram gated write before the layer (layer_id == 1)
layer 2  (ratio 2, Full): compress at i odd -> RoPE at i (i+1-2), cache slot i//2;
                           indexer K from latent (fp4, block 32 UE8M0);
                           indexer Q -> scores vs index_k; own Top-512 (Reuse? no, Full)
layers 3-7 (ratio 2, Reuse): read shared compress_kv + shared topk_idxs
layers 8-13 (Reuse, source 8), 14-19 (Reuse, source 14)
layer 20 (ratio 1, Full, candidate source): plain projection (no gate, bf16), RoPE at i,
                           indexer K, candidate pool, own Top-512
layers 21-23 (Reuse of 20)
layer 24 (ratio 1, Reindex): own indexer Q rescoring layer 20's keys -> own Top-512
layers 25-27 (Reuse of 24's topk)
... 28/32/36 likewise ...
head: hc_pre with last block's ffn_pre -> RMSNorm -> Linear -> fp32 logits
```

---

## 23 Scaling-down guide (deriving a preset from a parameter budget)

To pick a new preset, solve for the routed-expert term, which dominates.

```
params ≈ 2·V·d                                    (embed + untied head)
       + L·[ 3·E·d·MoEI + attn(d, hd, qlora, ogr, olora) + hc(4, d) + norms ]
       + (#kv_source)·(d·hd·(1 + [m>1]) + hd)      (compressors)
       + (#index_source)·(index_n_heads·index_head_dim·q_lora_rank + index_n_heads·d)
       + (#kv_source)·(hd·index_head_dim + index_head_dim)   (indexer K)
       + engram_rows·engram_head_dim + (#engram)·(cols·engram_head_dim·d·(hc+1) + 2·hc·d)
       + DSpark terms (§2)
attn ≈ d·q_lora + q_lora·H·hd + d·hd + (H·hd/ogr)·(ogr·olora) + ogr·olora·d
```

Procedure:
1. Fix `d` (multiple of `o_groups·head_dim`) and `L` (even; ≥ 4). Choose `H`, `hd` with `H·hd` a multiple of `o_groups`.
2. Choose `E`, `MoEI` and `topk` so that `L·3·E·d·MoEI` lands at ~85% of the budget; `topk/E` should stay in the 1.5%–7% range.
3. Choose `V` for the tokenizer; `2·V·d` is often 5–10% of the total.
4. Pick `engram_head_dim` and `engram_vocab_size` (prime seed) so `rows·engram_head_dim` is 5–10% of the budget; verify `rows == sum(primes)` at build time and store that number as `engram_num_embeddings`.
5. Keep all *structural* invariants of §3.6 — a smaller model must still have the encoder/decoder split, one decoder `kv_source` that is also the candidate source, at least one Reindex decoder layer, and Engram at a layer other than 0.

---

## 24 Bit-level quantization reference

### 24.1 FP8 E4M3 (FN) value grid

```
max normal   448      0 1111 110
min normal   2^-6     0 0000 001
min subnorm  2^-9     0 0000 000
NaN/Inf      S.1111.111
```
Used for: dense weights (per 32×32 block, UE8M0 scale), window KV (per 32 along the feature axis), Engram table rows (per 32 along `head_dim`).

### 24.2 FP4 E2M1 (FN) value grid

```
code  0    1     2    3     4    5    6    7    8
val   0   0.5   1.0  1.5   2.0  3.0  4.0  6.0  -0.0
```
Rounding is round-to-nearest-**even** on the 8-value grid:
`0.25→0, 0.75→1, 1.25→1, 1.75→2, 2.5→2, 3.5→4, 5→4`.

### 24.3 UE8M0 (power-of-two) scales

`scale = 2^e` where `e = ceil(log2(x))`, stored as an 8-bit unsigned exponent. Computed bit-exactly with:

```
e = ((bits(x) >> 23) & 0xFF) - 127 + (mantissa != 0)
scale = float32((e + 127) << 23)
```

### 24.4 Which scale format where

| quantity | block | scale format | amax floor |
|---|---|---|---|
| dense weight | 32 (out) × 32 (in) | UE8M0 | none |
| routed expert weight | 32 along K | UE8M0 | none |
| window KV activation | 32 along hd | UE8M0 (`2^ceil(log2(amax/448))`) | `1e-4` |
| compressed KV latent | 16 along hd | **E4M3** (`E4M3(amax/6)`) | `6·2^-9` |
| indexer Q / K | 32 | UE8M0 (`2^ceil(log2(amax/6))`) | `6·2^-126` |
| Engram table row | 32 along head_dim | UE8M0 | none |

### 24.5 Known parity caveat

Values near an E2M1 midpoint can flip a code between implementations due to ~1e-7 upstream noise; a single flipped window-KV entry cascades through overlapping windows. Discrete decisions (expert top-k, index sets, candidate blocks) still agree. With QAT simulation disabled, cross-implementation agreement is ~1e-6 everywhere.

---

## 25 Expanded invariants checklist

Consolidated, ordered by the phase of a forward pass.

**Construction**
1. `mix_hc == (2 + hc) * hc`; `comb` consumes the last `hc*hc` entries of `mixes`.
2. `hc_*` parameters are fp32; `hc_scale` has exactly 3 entries per sublayer.
3. A block owns **two** sets of HC parameters (attention, FFN) and **two** norms.
4. `compress_ratios` has `n_layers + num_nextn_predict_layers` entries.
5. `Attention` owns a `Compressor` iff `layer < n_layers and layer in kv_source_layer_ids`; an `Indexer` iff `layer < n_layers and layer in index_source_layer_ids`.
6. `Indexer.owns_k` iff the layer is also a kv_source. The count of `wq_b`/`weights_proj` equals `len(index_source)`; `wk` equals `len(kv_source)`.

**Prefill**
7. Layers execute in ascending order; the shared-state slots are never reset between forwards.
8. The indexer runs **before** the latent is rotated and quantized.
9. Compressed indices are offset by the number of window rows; unreachable entries are `-1`.
10. The window ring is seeded with the trailing `n_win` rows of the chunk, using `cutoff = S % n_win`.
11. Ratio-2 compressors move the trailing `rem` tokens into `pool_state` and emit `S // ratio` latents.

**Decode**
12. `start_pos` is threaded explicitly; every cache index derives from it.
13. Ring slot `start_pos % n_win`; compressed slot `start_pos // ratio`; indexer prefix `end // ratio`.
14. Latent RoPE frequency: prefill `freqs[:S - S % ratio : ratio]`, decode `freqs[start_pos + 1 - ratio]`.
15. A ratio-2 compressor returns `None` on every step where `(start_pos + 1) % 2 != 0`.

**Attention math**
16. `sparse_attn` input `topk_idxs` is contiguous `int32`; `-1` masked; max floor `-1e30`.
17. `attn_sink` enters the denominator only and has no value vector.
18. Inverse RoPE is applied to the output tail before `wo_a`.
19. `wo_a` is block-diagonal over `o_groups`; realized as `einsum("bsgd,grd->bsgr", o, W.view(G,R,Dg))`.

**MoE**
20. Gate scores fp32 `sqrt(softplus(...))`; selection bias adds only for ordering; normalization divides by `sum + 1e-20`; scale `×1.5`.
21. `bias_vl` is used for image tokens, `bias` otherwise.
22. SwiGLU clamps: `up` two-sided, `gate` upper-only, both at `swiglu_limit`.

**Engram**
23. Primes are unique across `(layer, n-gram size, head)` and drawn upward from `engram_vocab_size`.
24. `sum(primes of layer) == engram_num_embeddings[layer]`.
25. Multipliers are odd; `token * multiplier` cannot overflow int64.
26. The compressed token map size equals `engram_compressed_vocab_size`.
27. Image tokens are `DEAD` in the hash cache; blocked slots read the compressed pad id; the gate is forced to 0 on masked positions.

**DSpark**
28. `layer_id = n_layers + stage_id`; draft layers assert `compress_ratio == 0`.
29. Draft block: position 0 is the real token, positions 1.. are `dspark_noise_token_id`.
30. Prefill only seeds the draft window cache; no draft attention, no FFN.
31. DSpark blocks share the backbone `embed` and `head`.

**Numerics**
32. `rms_norm_eps == 1e-20` everywhere it appears.
33. All RMSNorm statistics computed in fp32.
34. Fake-quantized values are exactly representable in bf16.
35. When `QUANT_MODE == "NONE"`, only quantization is skipped — sinks, RMSNorm eps, Sinkhorn, and clamps remain.
36. **Quant block divisibility:** window block 32 must divide `head_dim`; compressed block 16 must divide `head_dim`; indexer block 32 must divide `index_head_dim`. (Released: 512 % 32 = 0, 512 % 16 = 0, 128 % 32 = 0. Violations fail loudly in reshape.)
37. **Gradient contract:** `gate.bias` / `gate.bias_vl` correctly receive **no** gradient (they only order top-k selection; they are updated by the aux-loss-free balance rule, §19.3). Unselected routed experts receive no gradient on a given batch (routing sparsity). Fake-quant uses a straight-through estimator so QAT training flows; without STE, bucketize/sign detach the graph.
38. **Training vs inference code paths:** the released kernels rotate/quantize in place. A training implementation must use out-of-place variants (`_rope_tail`, non-inplace quant returning new tensors); in-place ops on autograd inputs break backward.

---

## 26 Reproducing the published numbers (what to measure)

| claim | how to reproduce |
|---|---|
| 552B / 196B / 14B parameter split | §2 script; must print 551.57B / 196.93B / 14.23B |
| 8B prefill / 16B decode | count activated weights per layer (§20.2) × layers run in each phase |
| ~1/4 HBM global KV | CSA2 sharing (4 KV producers for 40 layers) × FP4 latents; measure `bytes/token` of `compress_kv + index_k` |
| ~1/8 persistent KV | global KV only (SWA dropped from persistence) ÷ V4's global+SWA |
| <1 GB for 1M context | 890 B × 1,048,576 ≈ 0.93 GB |
| ~437× smaller than V1 | 890 B vs V1's per-token global KV |
| decode FLOPs +1/4 from 4K→1M | integrate per-token FLOPs with the compressed-KV length at both context lengths, weighting BF16/FP8/FP4 as 1/0.5/0.25 |
| 15 / 11 kernels per Reuse layer | kernel-launch counting in a fused CUDA/TileLang backend (not reproducible in eager PyTorch) |

---

