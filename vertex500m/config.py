"""Configuration for vertex500m: a dense 503.8M reproduction of DeepSeek-V4.1-Flash.

Derived from `vertex500m-dense.md` (the MoE->dense conversion) with four deliberate
deviations, each recorded in `DEVIATIONS` below and asserted by `params.py`:

  1. q_lora_rank = o_lora_rank = 384, not d/4 = 336. FP8 block-32 quantization requires
     every weight dim to be a multiple of 32; 336 % 32 = 16 would fail the reshape in
     act_quant on wq_a, wq_b and indexer.wq_b. HARD REQUIREMENT, not a preference.
  2. index_topk = 384 (spec uses 512). The top-k gather is the largest single term in
     naive training activation memory, but it is TRANSIENT under chunked recompute, so
     this trades gather FLOPs for retrieval quality rather than memory. At T=8192 it
     attends 512 of 8192 positions (6.2%).
  3. L = 20, d = 1344 (spec-derived dense preset uses L = 16, d = 1408).
  4. engram_head_dim = 64 (spec-derived dense preset uses 128, 17.9% of the model).

`num_nextn_predict_layers` defaults to 0: DSpark is attached post-hoc via
`Transformer.attach_dspark()` so it costs nothing during pretraining.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


@dataclass
class Cfg:
    # ---- shape -------------------------------------------------------------
    vocab_size: int = 32000
    dim: int = 1344
    n_layers: int = 20
    n_heads: int = 21
    head_dim: int = 128
    q_lora_rank: int = 384
    qk_rope_head_dim: int = 32
    o_lora_rank: int = 384
    o_groups: int = 4
    moe_intermediate_size: int = 3584  # key name kept; it is the dense FFN width
    tie_word_embeddings: bool = True

    # ---- tokenizer (JamesQuartz/qt-VI.1.3-32k) ----------------------------
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    dspark_noise_token_id: int = 3
    image_token_id: int = 85

    # ---- attention schedule (CSA2) ----------------------------------------
    sliding_window: int = 128
    # One entry per layer, INCLUDING draft stages. During pretraining DSpark does not
    # exist, so this is n_layers long; `with_dspark()` appends the trailing zeros.
    compress_ratios: tuple = (0, 0) + (2,) * 8 + (1,) * 10
    kv_source_layer_ids: tuple = (2, 10)
    index_source_layer_ids: tuple = (2, 10, 14)
    candidate_source_layer_id: int = 10
    candidate_topk_blocks: int = 512
    candidate_block_size: int = 8
    index_n_heads: int = 8
    index_head_dim: int = 64
    # 384, not the spec's 512. The gathered top-k KV is 59% of naive training activation
    # memory, but it is TRANSIENT (recomputed per chunk in backward), so raising topk
    # costs gather FLOPs rather than retained memory. 384 keeps 128+384 = 512 attended
    # positions, which is 6.2% of an 8K context, and stays well under the 4096-position
    # candidate pool that bounds Reindex layers.
    index_topk: int = 384

    # ---- hyper-connections -------------------------------------------------
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # ---- engram ------------------------------------------------------------
    engram_layer_ids: tuple = (1,)
    engram_num_embeddings: tuple = (600766,)
    engram_max_ngram_size: int = 4
    engram_vocab_size: int = 50021
    engram_n_heads: int = 4
    engram_head_dim: int = 64
    engram_pad_token_id: int = 0
    engram_compressed_vocab_size: int = 0  # MUST be measured from the tokenizer

    # ---- dspark ------------------------------------------------------------
    num_nextn_predict_layers: int = 0
    dspark_block_size: int = 4
    dspark_target_layer_ids: tuple = (19,)
    dspark_markov_rank: int = 64

    # ---- rope --------------------------------------------------------------
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    rope_factor: float = 16
    beta_fast: int = 32
    beta_slow: int = 1
    rope_original_seq_len: int = 8192
    max_seq_len: int = 32768

    # ---- numerics / runtime ------------------------------------------------
    rms_norm_eps: float = 1e-20
    swiglu_limit: float = 10.0
    initializer_range: float = 0.02
    max_batch_size: int = 4
    temperature: float = 1.0

    # ---- vision (0 layers disables the whole path) -------------------------
    vision_n_layers: int = 0
    vision_dim: int = 1024
    vision_n_heads: int = 16
    vision_inter_dim: int = 2816
    vision_patch_size: int = 14
    vision_rope_theta: float = 10000.0
    vision_downsample_ratio: int = 3

    # ---- MoE override, used ONLY by the Phase-0 counting gate ----------------
    # vertex500m itself is dense; these exist so params.py can reproduce the MoE
    # preset from vertex500m-dense.md 21.1 before trusting any derived dense number.
    # All zero => dense FFN.
    n_routed_experts: int = 0
    num_experts_per_tok: int = 0
    n_shared_experts: int = 0
    dspark_n_routed_experts: int = 0
    dspark_num_experts_per_tok: int = 0

    # ---- runtime knobs not part of the architecture contract ---------------
    segment_size: int = 2  # gradient-checkpointing granularity, in layers
    attn_chunk: int = 256  # query-dim chunk for the sparse-attention gather
    index_chunk: int = 256  # query-dim chunk for the indexer score/topk

    # ------------------------------------------------------------------ #
    def __post_init__(self):
        self.validate()

    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """Structural invariants. Raises with a specific message on violation.

        See `DeepSeek-V4.1-Flash_Architecture_Spec.md` 3.6 and
        `vertex500m-dense.md` 16.3 / 19.
        """
        errs = []

        # ---- shape consistency
        # H*hd is a free multiplier (released V4.1: 6.4d; this preset: 2d; the MoE
        # preset used 1d). Only the o_groups divisibility is structural.
        if self.n_heads * self.head_dim < self.dim:
            errs.append("H*hd should be >= d")
        if self.qk_rope_head_dim >= self.head_dim:
            errs.append("qk_rope_head_dim must be < head_dim")
        # NOTE: og * o_lora_rank is the intermediate width feeding wo_b and is NOT
        # required to equal dim. The released model has 8 * 1024 = 8192 != 5120.

        # -- CSA2 schedule
        n_total = self.n_layers + self.num_nextn_predict_layers
        if len(self.compress_ratios) != n_total:
            errs.append(
                f"len(compress_ratios)={len(self.compress_ratios)} != "
                f"n_layers + num_nextn_predict_layers = {n_total}"
            )
        if self.compress_ratios[0] != 0 or self.compress_ratios[1] != 0:
            errs.append("compress_ratios[0] and [1] must both be 0 (pure sliding window)")
        if self.n_layers % 2 != 0:
            errs.append("n_layers must be even for the CED encoder/decoder split")
        boundary = self.n_layers // 2
        if self.kv_source_layer_ids[0] != 2:
            errs.append("kv_source_layer_ids[0] must be 2")
        if boundary not in self.kv_source_layer_ids:
            errs.append(f"the first decoder layer (L/2={boundary}) must be a kv_source")
        if self.candidate_source_layer_id != boundary:
            errs.append("candidate_source_layer_id must equal L/2")
        if len(self.index_source_layer_ids) <= len(self.kv_source_layer_ids):
            errs.append("len(index_source) must exceed len(kv_source) (Reindex owns no keys)")
        for l in range(self.n_layers):
            if self.compress_ratios[l] == 0 or l in self.kv_source_layer_ids:
                continue  # sliding-only layers and producers need no source above
            if not any(s < l for s in self.kv_source_layer_ids):
                errs.append(f"layer {l} has ratio>0 but no kv_source above it")
        for l in self.kv_source_layer_ids:
            if l >= self.n_layers:
                errs.append(f"kv_source layer {l} must be < n_layers")
            if self.compress_ratios[l] == 0:
                errs.append(f"kv_source layer {l} must have compress_ratio > 0")
        enc = {self.compress_ratios[l] for l in range(2, boundary) if self.compress_ratios[l]}
        dec = {self.compress_ratios[l] for l in range(boundary, self.n_layers)
               if self.compress_ratios[l]}
        if enc - {2}:
            errs.append(f"encoder CSA2 layers must all use ratio 2, found {enc}")
        if dec - {1}:
            errs.append(f"decoder CSA2 layers must all use ratio 1, found {dec}")

        # -- quantization block divisibility (spec invariant 36)
        for name, dim, blk in [
            ("head_dim (window block 32)", self.head_dim, 32),
            ("head_dim (compressed block 16)", self.head_dim, 16),
            ("index_head_dim (block 32)", self.index_head_dim, 32),
            ("q_lora_rank", self.q_lora_rank, 32),
            ("o_lora_rank", self.o_lora_rank, 32),
            ("dim", self.dim, 32),
            ("moe_intermediate_size", self.moe_intermediate_size, 32),
            ("(H*hd)//o_groups", (self.n_heads * self.head_dim) // self.o_groups, 32),
            ("H*hd", self.n_heads * self.head_dim, 32),
        ]:
            if dim % blk:
                errs.append(f"FP8 block-{blk} divisibility: {name}={dim} ({dim}%{blk}={dim%blk})")
        if self.o_lora_rank % self.o_groups:
            errs.append("o_groups must divide o_lora_rank")
        if (self.n_heads * self.head_dim) % self.o_groups:
            errs.append("o_groups must divide H*hd")

        # -- engram
        if self.engram_head_dim % 32:
            errs.append("engram_head_dim % 32 must be 0 (ParallelEngramEmbedding row scale)")
        if len(self.engram_num_embeddings) != len(self.engram_layer_ids):
            errs.append("len(engram_num_embeddings) must equal len(engram_layer_ids)")
        if self.engram_compressed_vocab_size:
            if not 0 < self.engram_compressed_vocab_size < self.engram_vocab_size:
                errs.append(
                    f"engram_compressed_vocab_size={self.engram_compressed_vocab_size} "
                    f"must be in (0, engram_vocab_size={self.engram_vocab_size})"
                )

        # -- dspark
        for l in self.dspark_target_layer_ids:
            if not 0 <= l < self.n_layers:
                errs.append(f"dspark_target_layer_id {l} out of range")

        # -- tokenizer ids
        ids = {"pad": self.pad_token_id, "bos": self.bos_token_id, "eos": self.eos_token_id,
               "dspark_noise": self.dspark_noise_token_id}
        if len(set(ids.values())) != len(ids):
            errs.append(f"token ids collide: {ids}")

        if errs:
            raise ValueError("Cfg validation failed:\n  - " + "\n  - ".join(errs))

    # ------------------------------------------------------------------ #
    def with_(self, **kw) -> "Cfg":
        return replace(self, **kw)

    def tiny(self, **over) -> "Cfg":
        """A CPU-runnable shrink of this config, for architecture tests.

        `engram_num_embeddings` is recomputed from the layout rather than copied from the
        full preset: the prime sum depends on `engram_n_heads`, so shrinking the head
        count changes the row count and a copied constant would be wrong.
        """
        n_layers, n_heads = over.get("n_layers", 8), over.get("engram_n_heads", 2)
        seed = over.get("engram_vocab_size", 50_021)
        # same construction as EngramLayout: max_ngram-1 sizes x n_heads unique primes
        seen, cur, rows = set(), seed - 1, 0
        for _ in range(self.engram_max_ngram_size - 1):
            for _ in range(n_heads):
                while True:
                    cur += 1
                    if cur not in seen and _is_prime(cur):
                        seen.add(cur)
                        rows += cur
                        break
        base = dict(
            vocab_size=512, dim=128, n_layers=8, n_heads=4, head_dim=32,
            q_lora_rank=64, qk_rope_head_dim=8, o_lora_rank=64, o_groups=4,
            moe_intermediate_size=64, sliding_window=8,
            index_n_heads=2, index_head_dim=32, index_topk=8,
            candidate_topk_blocks=4, candidate_block_size=4,
            compress_ratios=(0, 0, 2, 2, 1, 1, 1, 1),   # 8 backbone layers
            kv_source_layer_ids=(2, 4), index_source_layer_ids=(2, 4, 6),
            candidate_source_layer_id=4,
            engram_num_embeddings=(rows,), engram_vocab_size=50_021,
            engram_n_heads=2, engram_head_dim=32, engram_compressed_vocab_size=512,
            num_nextn_predict_layers=0, dspark_block_size=3,
            dspark_target_layer_ids=(7,), dspark_markov_rank=16,
            max_seq_len=64, max_batch_size=2, segment_size=2, attn_chunk=64,
        )
        base.update(over)
        return replace(self, **base)

    def with_dspark(self, n_stages: int = 1) -> "Cfg":
        """Attach DSpark: grow compress_ratios by n_stages trailing zeros."""
        ratios = tuple(self.compress_ratios) + (0,) * n_stages
        return replace(self, compress_ratios=ratios,
                       num_nextn_predict_layers=n_stages,
                       dspark_target_layer_ids=(self.n_layers - 1,))


PRESET = Cfg()
PRESET.validate()

# The MoE preset from `vertex500m-dense.md` 21.1, kept so `params.py` can prove the
# counting script reproduces it before trusting any dense number derived from it.
MOE_PRESET = Cfg(
    vocab_size=65536, dim=512, n_layers=16, n_heads=8, head_dim=64,
    q_lora_rank=128, qk_rope_head_dim=16, o_lora_rank=128, o_groups=4,
    moe_intermediate_size=416, sliding_window=64, tie_word_embeddings=False,
    compress_ratios=(0, 0) + (2,) * 6 + (1,) * 8 + (0, 0),
    kv_source_layer_ids=(2, 8), index_source_layer_ids=(2, 8, 12),
    candidate_source_layer_id=8, candidate_topk_blocks=256, candidate_block_size=8,
    index_n_heads=4, index_head_dim=32, index_topk=128,
    engram_layer_ids=(1,), engram_num_embeddings=(600766,),
    engram_max_ngram_size=4, engram_vocab_size=50021, engram_n_heads=4,
    engram_head_dim=64, engram_pad_token_id=2, engram_compressed_vocab_size=0,
    num_nextn_predict_layers=2, dspark_block_size=4,
    dspark_target_layer_ids=(13, 14, 15), dspark_markov_rank=64,
    rope_original_seq_len=8192, max_seq_len=32768,
    n_routed_experts=32, num_experts_per_tok=2, n_shared_experts=1,
    dspark_n_routed_experts=16, dspark_num_experts_per_tok=2,
)


DEVIATIONS = """
1. q_lora_rank = o_lora_rank = 384 (not d/4 = 336). 336 % 32 != 0 breaks FP8 block-32
   quantization on attn.wq_a, attn.wq_b and indexer.wq_b. This is a hard requirement
   of the quantization format, not a design preference.
2. index_topk = 384 (spec: 512). The top-k gather is the largest single term in naive
   training activation memory, but it is transient under chunked recompute, so this
   trades gather FLOPs for retrieval quality rather than memory. At T=8192 this
   attends 512 of 8192 positions (6.2%).
3. L = 20, d = 1344 (dense-conversion doc: L = 16, d = 1408). L=16 was chosen on an
   eager-mode launch-count argument that the doc itself concedes does not hold under
   torch.compile; with compilation, FLOPs are the objective and L=20 is deeper for
   less FLOPs.
4. engram_head_dim = 64 (dense-conversion doc: 128). Caps Engram at 8.7% of the model.
5. num_nextn_predict_layers = 0 during pretraining. DSpark is constructed on demand
   via Transformer.attach_dspark() and trained in a separate frozen-backbone stage,
   per spec 12.4. It contributes zero to pretraining throughput.
"""
