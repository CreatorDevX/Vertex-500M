"""manifest.json: a data directory's bill of materials.

After tokenizing, the output directory contains `{stem}_{i:05d}.bin` shards,
`{stem}_{i:05d}.idx.npy` doc-offset indices, and one `manifest.json`:

    {"tokenizer": ..., "vocab_size": ..., "engram_compressed_vocab_size": ...,
     "collisions": ..., "seed": ..., "val_fraction": ..., "created": ...,
     "splits": {"train": [shard_record, ...], "val": [...]}}

A shard record is `{"file", "sha256", "tokens", "docs"}`. The pipeline's
preflight re-validates the manifest (files present, hashes match) so a
partially-copied data directory fails BEFORE a multi-day run, not 40 hours in.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

MANIFEST_NAME = "manifest.json"
INDEX_SUFFIX = ".idx.npy"


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def shard_record(bin_path: Path, idx_path: Path | None = None) -> dict:
    """Hash + count one shard. The doc count comes from the index when present."""
    tokens = bin_path.stat().st_size // 2        # uint16
    docs = None
    if idx_path is not None and idx_path.exists():
        ends = np.load(idx_path)
        docs = int(len(ends))
        if len(ends) and int(ends[-1]) != tokens:
            raise ValueError(
                f"index {idx_path.name} ends at {ends[-1]} but {bin_path.name} "
                f"holds {tokens} tokens: index and shard disagree.")
    return {"file": bin_path.name,
            "sha256": sha256_file(bin_path),
            "tokens": tokens,
            "docs": docs,
            "index": idx_path.name if idx_path is not None and idx_path.exists()
                     else None}


def write_manifest(out_dir, *, tokenizer: str, vocab_size: int,
                   compressed_vocab_size: int, collisions: int,
                   seed: int, val_fraction: float,
                   splits: dict[str, list[dict]]) -> dict:
    """Write `manifest.json` in `out_dir`. Returns the manifest dict."""
    out = Path(out_dir)
    total = {k: sum(r["tokens"] for r in v) for k, v in splits.items()}
    manifest = {
        "tokenizer": tokenizer,
        "vocab_size": vocab_size,
        "engram_compressed_vocab_size": compressed_vocab_size,
        "collisions": collisions,
        "seed": seed,
        "val_fraction": val_fraction,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tokens": total,
        "splits": splits,
    }
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2),
                                     encoding="utf-8")
    return manifest


def read_manifest(out_dir) -> dict:
    p = Path(out_dir) / MANIFEST_NAME
    if not p.exists():
        raise FileNotFoundError(
            f"no {MANIFEST_NAME} in {out_dir}. Tokenize with the current "
            f"tokenize.py (it writes one) instead of guessing at shard layouts.")
    return json.loads(p.read_text(encoding="utf-8"))


def validate_manifest(manifest: dict, out_dir, check_hashes: bool = True) -> list[str]:
    """Re-check a manifest against the directory. Returns a list of problems."""
    problems = []
    out = Path(out_dir)
    for split, records in manifest.get("splits", {}).items():
        if not records:
            problems.append(f"split {split!r} lists no shards")
        for r in records:
            p = out / r["file"]
            if not p.exists():
                problems.append(f"split {split!r}: missing file {r['file']}")
                continue
            if p.stat().st_size // 2 != r["tokens"]:
                problems.append(
                    f"split {split!r}: {r['file']} is {p.stat().st_size // 2} tokens, "
                    f"manifest says {r['tokens']}")
            if check_hashes and sha256_file(p) != r["sha256"]:
                problems.append(f"split {split!r}: {r['file']} hash mismatch "
                                f"(file changed since tokenizing)")
            if r.get("index") and not (out / r["index"]).exists():
                problems.append(f"split {split!r}: missing index {r['index']}")
    return problems


def split_shard_paths(out_dir, manifest: dict, split: str) -> list[Path]:
    """Absolute shard paths for one manifest split. Unknown split = error."""
    splits = manifest.get("splits", {})
    if split not in splits:
        raise KeyError(
            f"split {split!r} not in manifest (have: {sorted(splits)}). "
            f"Re-tokenize with a val fraction, or train with split 'train'.")
    out = Path(out_dir)
    return [out / r["file"] for r in splits[split]]
