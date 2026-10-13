"""Exact integer parameter accounting.

Phase 0 of the build: this script must reproduce the MoE preset from
`vertex500m-dense.md` 21.1 BEFORE any dense number derived from it is trusted.
If `count(MOE_PRESET)` does not match, the script is wrong and so is everything else.

Run:  python -m vertex500m.params
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from .config import MOE_PRESET, PRESET, Cfg
from .hc import mix_hc_for


def is_prime(n: int) -> bool:
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


def primes_after(start: int, seen: set[int]) -> int:
    """Next prime strictly greater than `start`, not already in `seen`."""
    c = start + 1
    while True:
        if is_prime(c) and c not in seen:
            return c
        c += 1


def engram_primes(cfg: Cfg) -> list[list[int]]:
    """Per engram layer, the list of unique primes drawn upward from engram_vocab_size.

    `max_ngram - 1` n-gram SIZES (the 2/3/4-grams), each with `n_heads` heads.
    Spec 11.2's prose says "2..(max_ngram-1)" which yields 2 values; the reference
    code uses range(max_ngram-1) = 3 values. The CODE is right and the prose is
    wrong (`vertex500m-dense.md` 13). See note in README.
    """
    out, seen = [], set()
    for _ in cfg.engram_layer_ids:
        layer: list[int] = []
        for _ in range(cfg.engram_max_ngram_size - 1):
            cur = cfg.engram_vocab_size - 1
            for _ in range(cfg.engram_n_heads):
                cur = primes_after(cur, seen)
                seen.add(cur)
                layer.append(cur)
        out.append(layer)
    return out


@dataclass
class Counts:
    embed: int
    backbone: int
    engram: int
    dspark: int
    total: int
    layer: int
    attn: int
    ffn: int
    hcp: int
    active: int
    lay_ds: int
    gflop_fwd_bwd: float

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["gflop_fwd_bwd"] = round(self.gflop_fwd_bwd, 4)
        return d


def count(cfg: Cfg, hc_variant: str = "mhc") -> Counts:
    """Exact integer parameter count. No floating point anywhere in the path.

    `hc_variant` selects the hyper-connection parameterization. It matters: mHC-lite
    widens the fused projection from `(2+hc)*hc = 24` to `2*hc + hc! = 32` at hc=4,
    which is +64*d per layer (+86,032 at d=1344, +6.88M over the model). Hardcoding
    the mHC formula under-reports every mHC-lite checkpoint by that amount, silently,
    in the one script whose output is treated as a gate.
    """
    d, L, H, hd = cfg.dim, cfg.n_layers, cfg.n_heads, cfg.head_dim
    ql, ol, og, hc = cfg.q_lora_rank, cfg.o_lora_rank, cfg.o_groups, cfg.hc_mult
    MoEI = cfg.moe_intermediate_size
    kv, isrc = cfg.kv_source_layer_ids, cfg.index_source_layer_ids
    mix_hc = mix_hc_for(hc, hc_variant)

    # ---- per layer --------------------------------------------------------
    # attention: q LoRA + single shared KV latent + grouped output projection + sink.
    # NOTE: there is no value projection and H_kv == 1; K and V are the same tensor.
    attn = (
        d * ql + ql                 # wq_a + q_norm
        + ql * H * hd               # wq_b
        + d * hd + hd               # wkv + kv_norm
        + (H * hd // og) * (og * ol)  # wo_a  (block-diagonal over og groups)
        + og * ol * d               # wo_b
        + H                         # attn_sink
    )
    # hyper-connections, 2 sublayers x (fn [mix_hc, hc*d] + base [mix_hc] + scale [3]).
    # mix_hc comes from hc.mix_hc_for so this follows the ACTUAL parameterization:
    # (2+hc)*hc = 24 for mHC (2*(96d + 27) at hc=4), 2*hc + hc! = 32 for mHC-lite.
    # DO NOT mistranscribe as 2*(mix_hc*hc*(hc*d) + 27) -- that is 4x too large.
    hcp = 2 * (mix_hc * (hc * d) + mix_hc + 3)

    # MoE gate: only the Phase-0 gate path sets these. The base 3*d*MoEI below IS
    # one SwiGLU, which is exactly the always-on shared expert when S == 1 -- so the
    # MoE branch replaces it rather than adding to it.
    E, K, S = cfg.n_routed_experts, cfg.num_experts_per_tok, cfg.n_shared_experts
    ffn = 3 * d * MoEI
    if E or S:
        ffn = (E + S) * 3 * d * MoEI          # routed bank + shared experts
        if E:
            ffn += E * d + 2 * E               # gate.weight [E,d] + bias + bias_vl
    layer = attn + ffn + hcp + 2 * d  # + attn_norm + ffn_norm

    # Active per layer: the FFN part is replaced by (top-k routed + always-on shared).
    ff = ((K + S) * 3 * d * MoEI) if E else ffn
    layer_active = attn + ff + hcp + 2 * d

    # DSpark stages run the same block; with MoE they carry their own expert count.
    dsE = cfg.dspark_n_routed_experts
    ds_ffn = 3 * d * MoEI + (dsE * 3 * d * MoEI if dsE else 0)
    lay_ds = attn + ds_ffn + hcp + 2 * d

    # ---- whole model ------------------------------------------------------
    embed = cfg.vocab_size * d * (1 if cfg.tie_word_embeddings else 2)
    backbone = embed + L * layer
    # 1 indexer (wq_b + weights_proj) per index_source layer
    backbone += len(isrc) * (cfg.index_n_heads * cfg.index_head_dim * ql
                             + cfg.index_n_heads * d)
    # 1 index-key projection (wk + k_norm) per kv_source layer
    backbone += len(kv) * (hd * cfg.index_head_dim + cfg.index_head_dim)
    # compressors: wkv + norm everywhere, plus wgate where ratio > 1
    backbone += sum(d * hd + hd + (d * hd if cfg.compress_ratios[l] > 1 else 0) for l in kv)

    engram = 0
    for i, primes in enumerate(engram_primes(cfg)):
        rows = sum(primes)
        assert rows == cfg.engram_num_embeddings[i], (
            f"engram layer {i}: sum(primes)={rows} != "
            f"engram_num_embeddings[{i}]={cfg.engram_num_embeddings[i]}"
        )
        cols = (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads
        engram += (
            rows * cfg.engram_head_dim                    # the n-gram table
            + cols * cfg.engram_head_dim * d * (hc + 1)   # wkv -> d*(hc+1)
            + 2 * hc * d                                  # q_weight + k_weight
        )

    n_t = len(cfg.dspark_target_layer_ids)
    dspark = (
        cfg.num_nextn_predict_layers * lay_ds
        + d * n_t * d + d                              # main_proj + main_norm
        + cfg.vocab_size * cfg.dspark_markov_rank * 2   # markov embed + head
        + (d + cfg.dspark_markov_rank) + 1              # confidence_head
    )

    total = backbone + engram + dspark
    active = L * layer_active + (embed if cfg.tie_word_embeddings else 0)
    return Counts(
        embed=embed, backbone=backbone, engram=engram, dspark=dspark, total=total,
        layer=layer, attn=attn, ffn=ffn, hcp=hcp, active=active, lay_ds=lay_ds,
        # fwd(2*N) + bwd(4*N) over GEMM only; excludes attention score FLOPs.
        gflop_fwd_bwd=6 * active / 1e9,
    )


def check_engram_rows(cfg: Cfg) -> None:
    """Invariant: sum(primes of layer) == engram_num_embeddings[layer]."""
    for i, primes in enumerate(engram_primes(cfg)):
        rows = sum(primes)
        assert rows == cfg.engram_num_embeddings[i], (
            f"engram layer {i}: sum(primes)={rows} != "
            f"engram_num_embeddings[{i}]={cfg.engram_num_embeddings[i]}"
        )
        assert len(set(primes)) == len(primes), "primes must be unique"
        # Primes are drawn upward starting AT the seed, so min == engram_vocab_size
        # when that seed happens to be prime (it is, for 50021). Every compressed
        # token id must be strictly below the smallest prime to be addressable.
        assert min(primes) >= cfg.engram_vocab_size
        if cfg.engram_compressed_vocab_size:
            assert cfg.engram_compressed_vocab_size <= min(primes)


def _fmt(name: str, c: Counts) -> None:
    t = c.total
    print(f"  {name}")
    print(f"    embed (tied)  {c.embed:>12,}   {100*c.embed/t:6.2f}%")
    print(f"    backbone      {c.backbone:>12,}   {100*c.backbone/t:6.2f}%")
    print(f"    engram        {c.engram:>12,}   {100*c.engram/t:6.2f}%")
    print(f"    dspark        {c.dspark:>12,}   {100*c.dspark/t:6.2f}%")
    print(f"    TOTAL         {c.total:>12,}   ({t/1e6:.2f}M)")
    print(f"    per layer     {c.layer:>12,}")
    print(f"      ffn  {c.ffn:>12,}  {100*c.ffn/c.layer:6.2f}%")
    print(f"      attn {c.attn:>12,}  {100*c.attn/c.layer:6.2f}%")
    print(f"      hc   {c.hcp:>12,}  {100*c.hcp/c.layer:6.2f}%")
    print(f"    active/token  {c.active:>12,}   ({100*c.active/t:.1f}% of stored)")
    print(f"    fwd+bwd       {c.gflop_fwd_bwd:12.3f} GFLOP/token (GEMM only)")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--hc-variant", default="mhc", choices=["mhc", "mhc-lite"],
                    help="hyper-connection parameterization; changes the per-layer "
                         "projection width, so it changes the totals")
    args = ap.parse_args()

    # ---- gate: reproduce the MoE preset first --------------------------------
    print("=== GATE: reproduce the MoE preset from vertex500m-dense.md 21.1 ===")
    check_engram_rows(MOE_PRESET)
    m = count(MOE_PRESET)
    fails = []
    if abs(m.backbone / 1e6 - 414.4) > 0.1:
        fails.append(f"backbone {m.backbone/1e6:.1f}M != 414.4M")
    if abs(m.engram / 1e6 - 40.4) > 0.1:
        fails.append(f"engram {m.engram/1e6:.1f}M != 40.4M")
    if abs(m.dspark / 1e6 - 31.9) > 0.3:
        fails.append(f"dspark {m.dspark/1e6:.1f}M != 31.9M")
    if abs(m.total / 1e6 - 486.9) > 0.1:
        fails.append(f"total {m.total/1e6:.1f}M != 486.9M")
    print(f"  backbone {m.backbone:,}  engram {m.engram:,}  "
          f"dspark {m.dspark:,}  total {m.total:,}")
    for f in fails:
        print(f"    FAIL: {f}")
    if fails:
        print("\nGATE FAILED -- the counting script is wrong; do not trust any "
              "dense number derived from it.")
        return 1
    print("  PASS\n")

    # ---- the real config ------------------------------------------------------
    print("=== vertex500m ===")
    check_engram_rows(PRESET)
    v = count(PRESET, args.hc_variant)
    _fmt(f"PRESET (d={PRESET.dim}, L={PRESET.n_layers}, hc={args.hc_variant})", v)

    print("=== with DSpark attached (1 stage, backbone frozen) ===")
    vd = count(PRESET.with_dspark(1), args.hc_variant)
    _fmt("PRESET + DSpark", vd)

    if args.hc_variant != "mhc":
        base = count(PRESET.with_dspark(1), "mhc")
        print(f"  hc={args.hc_variant} costs +{vd.total - base.total:,} params over mHC "
              f"({100*(vd.total-base.total)/base.total:+.3f}%), "
              f"+{vd.hcp - base.hcp:,} per layer.\n")

    if args.json:
        print(json.dumps({"preset": v.as_dict(), "dspark": vd.as_dict(),
                          "moe_preset": m.as_dict(), "hc_variant": args.hc_variant},
                         indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
