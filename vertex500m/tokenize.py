"""Tokenizer front-end: qt-VI.1.3 -> uint16 shards, plus the Engram compressed token map.

Two outputs:

1. SHARDS. Raw token ids as flat little-endian uint16, written to
   `{out_dir}/{name}_{i:05d}.bin`. V = 32000 < 65536 so uint16 is exact. `data.py`
   reads exactly this format. uint16 REQUIRES a token id < 65536; if you swap in a
   larger-vocab tokenizer, change this to uint32 and update PackedTokenDataset.

2. ENGRAM TOKEN MAP. The compressed-vocab map required by spec 11.2, which
   `NgramHashState` needs at construction time. This is the largest "supply your own
   data" gap in the whole build: the spec's 99,092 is specific to DeepSeek's tokenizer
   and is NOT knowable in advance for qt-VI.1.3.

   The normalizer (spec 11.2), applied to each token STRING:
       NFKC -> NFD -> StripAccents -> Lowercase
       -> collapse [\\t\\r\\n]+ to " " -> "^ $" -> sentinel -> Strip -> sentinel -> " "
   Count distinct results. That count MUST be < engram_vocab_size (50021), or the
   n-gram prime buckets cannot address every compressed id.

   Why this matters and why it is a real risk on a 204-language byte-level vocabulary:
   NFKC folding collapses CJK compatibility ideographs, Arabic presentation forms and
   full-width Latin into canonical forms, so DISTINCT tokens collide. Every collision
   merges two tokens into one Engram lookup cell and silently degrades retrieval. The
   collision count is measured and reported below, not assumed.

Special-token ids are read FROM the tokenizer, never hard-coded (invariant 26). The
qt-VI.1.3 map is the REVERSE of DeepSeek's: bos=1, eos=2, pad=0. A loader assuming
DeepSeek's ordering produces a model whose BOS is a PAD.

Usage:
    python -m vertex500m.tokenize --tokenizer JamesQuartz/qt-VI.1.3-32k \
        --input corpus/*.txt --out data/ --name web
    python -m vertex500m.tokenize --map-only            # just measure the token map
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Engram compressed token map (spec 11.2)
# --------------------------------------------------------------------------- #
def normalize_token(s: str) -> str:
    """The spec 11.2 normalizer. Order matters; each step is cumulative."""
    s = unicodedata.normalize("NFKC", s)
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn")   # StripAccents
    s = s.lower()
    s = " ".join(s.split())                            # collapse whitespace
    s = f"^{s}$"                                       # sentinel
    s = s.strip()                                      # Strip
    s = f"{s} "                                        # sentinel + trailing space
    return s


def build_token_map(vocab_strings, compressed_vocab_size_max: int | None = None):
    """token id -> compressed id, with collisions measured.

    Returns (token_map, compressed_vocab_size, n_collisions).

    `compressed_vocab_size_max` caps the compressed vocab: once `len(seen)` reaches the
    cap, further DISTINCT normalized forms all fold onto one id. It is advisory only --
    the caller must check `compressed_vocab_size < engram_vocab_size` itself, because
    exceeding the cap silently destroys retrieval rather than raising. `--map-only`
    prints both numbers and the caller decides.
    """
    seen: dict[str, int] = {}
    token_map: list[int] = []
    collisions = 0
    overflow = 0
    for i, s in enumerate(vocab_strings):
        norm = normalize_token(s)
        if norm in seen:
            collisions += 1
            token_map.append(seen[norm])
        elif compressed_vocab_size_max is not None and len(seen) >= compressed_vocab_size_max:
            # over the cap: fold onto id 0 rather than growing the table past what the
            # Engram prime layout can address
            overflow += 1
            collisions += 1
            token_map.append(0)
        else:
            cid = len(seen)
            seen[norm] = cid
            token_map.append(cid)
    cvs = len(seen)
    if overflow:
        print(f"  WARNING: compressed vocab hit the cap of {compressed_vocab_size_max}; "
              f"{overflow} distinct forms were folded onto id 0. Retrieval for those "
              f"tokens is destroyed.")
    return token_map, cvs, collisions


# --------------------------------------------------------------------------- #
# Shards
# --------------------------------------------------------------------------- #
DTYPE = np.uint16          # exact for vocab <= 65535


def tokenize_corpus(texts, tokenizer, out_dir: str, name: str,
                    tokens_per_shard: int = 50_000_000, eos_id: int | None = None,
                    verbose: bool = True):
    """Encode `texts` and write uint16 shards. Appends EOS between documents."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    eos = tokenizer.eos_token_id if eos_id is None else eos_id

    buf: list[np.ndarray] = []
    total = 0
    shard = 0
    written = []

    def flush():
        nonlocal buf, total, shard
        if not buf:
            return
        arr = np.concatenate(buf).astype(DTYPE)
        p = out / f"{name}_{shard:05d}.bin"
        arr.tofile(p)
        written.append(p)
        if verbose:
            print(f"  wrote {p.name}  {arr.size/1e6:.1f}M tokens  "
                  f"{p.stat().st_size/2**20:.1f} MiB")
        buf, total, shard = [], 0, shard + 1

    if verbose:
        print(f"[tokenize] encoding {len(texts)} documents -> {out}")
    for i, t in enumerate(texts):
        ids = tokenizer(t, add_special_tokens=False)["input_ids"]
        if eos is not None:
            ids = list(ids) + [eos]
        buf.append(np.asarray(ids, dtype=DTYPE))
        total += len(ids)
        if total >= tokens_per_shard:
            flush()
        if verbose and (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(texts)} docs, {total/1e6:.1f}M tokens buffered")
    flush()
    return written


def read_texts(paths, limit=None):
    texts = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files = sorted(p.rglob("*.txt")) + sorted(p.rglob("*.jsonl"))
            for f in files:
                if f.suffix == ".jsonl":
                    for line in f.open(encoding="utf-8"):
                        obj = json.loads(line)
                        t = obj.get("text") or obj.get("content")
                        if t:
                            texts.append(t)
                            if limit and len(texts) >= limit:
                                return texts
                else:
                    texts.append(f.read_text(encoding="utf-8"))
                    if limit and len(texts) >= limit:
                        return texts
        else:
            texts.append(p.read_text(encoding="utf-8"))
            if limit and len(texts) >= limit:
                return texts
    return texts


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", default="JamesQuartz/qt-VI.1.3-32k")
    ap.add_argument("--input", nargs="*", default=None,
                    help="text files, directories of .txt/.jsonl, or '-' for stdin")
    ap.add_argument("--out", default="data/")
    ap.add_argument("--name", default="web")
    ap.add_argument("--tokens-per-shard", type=int, default=50_000_000)
    ap.add_argument("--map-only", action="store_true",
                    help="measure and print the Engram token map, write nothing")
    ap.add_argument("--save-map", default=None,
                    help="write the token map + compressed_vocab_size as JSON")
    ap.add_argument("--limit-docs", type=int, default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    V = len(tok)
    print(f"[tokenizer] {args.tokenizer}  V={V}")

    # --- special ids: read them, never assume them (invariant 26) ---
    ids = dict(pad=tok.pad_token_id, bos=tok.bos_token_id, eos=tok.eos_token_id,
               unk=tok.unk_token_id)
    print(f"[ids] {ids}")
    print(f"      qt-VI.1.3 is the REVERSE of DeepSeek's ordering "
          f"(DeepSeek: bos=0 eos=1 pad=2).")

    # --- Engram compressed token map ---
    vocab_strings = tok.convert_ids_to_tokens(list(range(V)))
    token_map, cvs, collisions = build_token_map(vocab_strings)
    print(f"\n[engram map]")
    print(f"  distinct normalized forms : {cvs}")
    print(f"  collisions (tokens merged): {collisions} "
          f"({100*collisions/max(V,1):.2f}% of vocab)")
    print(f"  MUST be < engram_vocab_size (50021): "
          f"{'OK' if 0 < cvs < 50021 else 'FAIL -- raise engram_vocab_size'}")
    if collisions:
        print(f"  WARNING: every collision merges two tokens into one Engram cell and")
        print(f"           silently degrades n-gram retrieval. On a 204-language")
        print(f"           byte-level vocab this is expected; the question is whether")
        print(f"           {collisions} is acceptable, not whether it is zero.")
    if args.save_map:
        Path(args.save_map).write_text(json.dumps({
            "tokenizer": args.tokenizer,
            "vocab_size": V,
            "engram_compressed_vocab_size": cvs,
            "collisions": collisions,
            "token_map": token_map,
        }))
        print(f"  saved {args.save_map}")

    if args.map_only:
        return

    # --- shards ---
    if not args.input:
        ap.error("--input required unless --map-only")
    texts = read_texts(args.input, args.limit_docs)
    print(f"\n[corpus] {len(texts)} documents")
    written = tokenize_corpus(texts, tok, args.out, args.name,
                              tokens_per_shard=args.tokens_per_shard, eos_id=ids["eos"])
    total = sum(p.stat().st_size for p in written) // 2
    print(f"\n[done] {len(written)} shard(s), {total/1e6:.1f}M tokens")
    print(f"  set engram_compressed_vocab_size = {cvs} in config before training")


if __name__ == "__main__":
    main()
