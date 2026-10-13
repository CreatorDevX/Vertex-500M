"""Tokenizer front-end: qt-VI.1.3 -> uint16 shards, plus the Engram compressed token map.

Three outputs in the output directory:

1. SHARDS. Raw token ids as flat little-endian uint16, written to
    `{out_dir}/{name}_{split}_{i:05d}.bin` with `split` = `train` or `val`.
    V = 32000 < 65536 so uint16 is exact. `data/shards.py` reads exactly this format.
    uint16 REQUIRES a token id < 65536; if you swap in a larger-vocab tokenizer,
    change this to uint32 and update PackedTokenDataset.

2. DOC INDEX. `{out_dir}/{name}_{split}_{i:05d}.idx.npy`: the exclusive doc-end
    token offset of every document in the shard. `data.packing.DocPackedDataset`
    uses it to pack whole documents into windows with zero padding.

3. MANIFEST. `{out_dir}/manifest.json`: the bill of materials (tokenizer, split
    sizes, sha256 per shard). The pipeline's preflight re-validates it, and
    train/eval select splits through it. `engram_compressed_vocab_size` is read
    from `--save-map` automatically by train.py/eval.py.

4. ENGRAM TOKEN MAP. The compressed-vocab map required by spec 11.2, which
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

Inputs are streamed, not materialised: directories of `.txt` (one doc per file)
and `.jsonl` (one doc per line, `text` or `content` field), single files, or `-'
for stdin lines. The train/val split is planned up front from the doc COUNT
(`data.split.plan_split`: seeded, reproducible), then each doc is routed by its
index -- texts are never held in RAM just to split them.

Usage:
    python -m vertex500m.tokenize --tokenizer JamesQuartz/qt-VI.1.3-32k \
        --input corpus/ --out data/ --name web --save-map token_map.json
    python -m vertex500m.tokenize --map-only            # just measure the token map
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np

from .data import plan_split, shard_record, write_manifest


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
# Inputs: streamed, counted, then routed by a pre-planned split
# --------------------------------------------------------------------------- #
def collect_files(paths) -> list[tuple[str, object]]:
    """Expand `--input` to `[("txt"|"jsonl"|"list", payload)]`, deterministic order.

    `.txt` = one doc per file; `.jsonl` = one doc per non-empty line (`text` or
    `content` field); `-` = stdin lines, read eagerly once into a list (a pipe
    cannot be counted and then re-read).
    """
    files = []
    for p in paths:
        if p == "-":
            docs = [ln for ln in (sys.stdin.read().splitlines()) if ln.strip()]
            files.append(("list", docs))
            continue
        p = Path(p)
        if p.is_dir():
            for f in sorted(p.rglob("*.txt")):
                files.append(("txt", f))
            for f in sorted(p.rglob("*.jsonl")):
                files.append(("jsonl", f))
        elif p.suffix == ".jsonl":
            files.append(("jsonl", p))
        else:
            files.append(("txt", p))
    return files


def _jsonl_doc_count(path: Path, limit: int | None) -> tuple[int, int]:
    n, bad = 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if limit is not None and n >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            t = obj.get("text") or obj.get("content") if isinstance(obj, dict) else None
            if t:
                n += 1
            else:
                bad += 1
    return n, bad


def count_docs(files, limit: int | None = None) -> tuple[int, int]:
    """(n_docs, n_skipped_lines). Respects `limit` (first N docs)."""
    n, skipped = 0, 0
    for kind, payload in files:
        if limit is not None and n >= limit:
            break
        if kind == "txt":
            n += 1
        elif kind == "list":
            take = payload if limit is None else payload[:max(0, limit - n)]
            n += len(take)
        else:
            c, bad = _jsonl_doc_count(payload,
                                      None if limit is None else max(0, limit - n))
            n += c
            skipped += bad
    return n, skipped


def iter_texts(files, limit: int | None = None):
    """Yield doc texts in exactly the order `count_docs` counted them."""
    yielded = 0
    for kind, payload in files:
        if limit is not None and yielded >= limit:
            return
        if kind == "txt":
            yield Path(payload).read_text(encoding="utf-8")
            yielded += 1
        elif kind == "list":
            for t in payload:
                if limit is not None and yielded >= limit:
                    return
                yield t
                yielded += 1
        else:
            with open(payload, encoding="utf-8") as f:
                for line in f:
                    if limit is not None and yielded >= limit:
                        return
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    t = obj.get("text") or obj.get("content") \
                        if isinstance(obj, dict) else None
                    if t:
                        yield t
                        yielded += 1


def read_texts(paths, limit=None):
    """Materialise all input texts (small-corpus convenience; streamed path above
    is what main() uses). Kept for backward compatibility."""
    return list(iter_texts(collect_files(paths), limit))


# --------------------------------------------------------------------------- #
# Shards
# --------------------------------------------------------------------------- #
DTYPE = np.uint16          # exact for vocab <= 65535


class _ShardWriter:
    """Accumulates tokenised docs into `{stem}_{i:05d}.bin` + `.idx.npy`.

    The index holds the exclusive doc-end token offset of every doc in the
    shard (offsets restart at 0 per shard). `close()` returns the (bin, index)
    path pairs written.
    """

    def __init__(self, out_dir: str, stem: str, tokens_per_shard: int = 50_000_000,
                 verbose: bool = True):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.stem = stem
        self.tokens_per_shard = tokens_per_shard
        self.verbose = verbose
        self.buf: list[np.ndarray] = []
        self.ends: list[int] = []
        self.total = 0
        self.shard = 0
        self.n_docs = 0
        self.written: list[tuple[Path, Path]] = []

    def add_doc(self, ids: np.ndarray) -> None:
        ids = np.asarray(ids, dtype=DTYPE)
        self.buf.append(ids)
        self.total += len(ids)
        self.ends.append(self.total)
        self.n_docs += 1
        if self.total >= self.tokens_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        arr = np.concatenate(self.buf).astype(DTYPE)
        bin_p = self.out / f"{self.stem}_{self.shard:05d}.bin"
        arr.tofile(bin_p)
        idx_p = bin_p.with_name(bin_p.stem + ".idx.npy")
        np.save(idx_p, np.asarray(self.ends, dtype=np.uint64))
        self.written.append((bin_p, idx_p))
        if self.verbose:
            print(f"  wrote {bin_p.name}  {arr.size/1e6:.1f}M tokens  "
                  f"{len(self.ends)} docs  {bin_p.stat().st_size/2**20:.1f} MiB")
        self.buf, self.ends, self.total, self.shard = [], [], 0, self.shard + 1

    def close(self) -> list[tuple[Path, Path]]:
        self.flush()
        return self.written


def tokenize_corpus(texts, tokenizer, out_dir: str, name: str,
                    tokens_per_shard: int = 50_000_000, eos_id: int | None = None,
                    verbose: bool = True):
    """Encode `texts` and write uint16 shards (+ doc index). Appends EOS per doc.

    Kept for backward compatibility (single split, stem `{name}`); main() below
    uses one `_ShardWriter` per split instead.
    """
    eos = tokenizer.eos_token_id if eos_id is None else eos_id
    w = _ShardWriter(out_dir, name, tokens_per_shard, verbose=verbose)
    if verbose:
        try:
            n = len(texts)
        except TypeError:
            n = "?"
        print(f"[tokenize] encoding {n} documents -> {w.out}")
    for i, t in enumerate(texts):
        ids = tokenizer(t, add_special_tokens=False)["input_ids"]
        if eos is not None:
            ids = list(ids) + [eos]
        w.add_doc(np.asarray(ids, dtype=DTYPE))
        if verbose and (i + 1) % 1000 == 0:
            print(f"  {i+1} docs, {w.total/1e6:.1f}M tokens buffered")
    w.close()
    return [b for b, _ in w.written]


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
    ap.add_argument("--val-fraction", type=float, default=0.005,
                    help="fraction of documents held out as {name}_val_*.bin. "
                         "0 = train-only corpus (no val split).")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the train/val shuffle (recorded in manifest.json)")
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
    files = collect_files(args.input)
    n_docs, skipped = count_docs(files, args.limit_docs)
    if skipped:
        print(f"  [corpus] skipped {skipped} corrupt/empty jsonl lines")
    if n_docs == 0:
        ap.error("no documents found in --input (empty files or all lines skipped)")
    is_val = plan_split(n_docs, args.val_fraction, args.seed)
    n_val = int(is_val.sum())
    print(f"\n[corpus] {n_docs} documents "
          f"({n_docs - n_val} train / {n_val} val, seed={args.seed})")

    train_w = _ShardWriter(args.out, f"{args.name}_train",
                           tokens_per_shard=args.tokens_per_shard)
    val_w = _ShardWriter(args.out, f"{args.name}_val",
                         tokens_per_shard=args.tokens_per_shard)
    eos = ids["eos"]
    for i, t in enumerate(iter_texts(files, args.limit_docs)):
        enc = tok(t, add_special_tokens=False)["input_ids"]
        if eos is not None:
            enc = list(enc) + [eos]
        (val_w if is_val[i] else train_w).add_doc(np.asarray(enc, dtype=DTYPE))
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{n_docs} docs")
    train_files = train_w.close()
    val_files = val_w.close()
    if n_val and not val_files:
        raise SystemExit("[tokenize] planned a val split but wrote no val shards: "
                         "a val doc must have produced zero tokens. Check the corpus.")
    splits = {"train": [shard_record(b, x) for b, x in train_files]}
    if val_files:
        splits["val"] = [shard_record(b, x) for b, x in val_files]
    manifest = write_manifest(args.out, tokenizer=args.tokenizer, vocab_size=V,
                              compressed_vocab_size=cvs, collisions=collisions,
                              seed=args.seed, val_fraction=args.val_fraction,
                              splits=splits)
    total = sum(manifest["tokens"].values())
    print(f"\n[done] {len(train_files)} train + {len(val_files)} val shard(s), "
          f"{total/1e6:.1f}M tokens; manifest.json written")
    print(f"  train.py and eval.py read engram_compressed_vocab_size={cvs} from the")
    print(f"  map automatically when passed --token-map {args.save_map}")


if __name__ == "__main__":
    main()
