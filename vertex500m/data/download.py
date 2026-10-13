"""Corpus acquisition: HuggingFace datasets -> corpus/*.jsonl.

Each source in the run config is a dict:

    {path: "HuggingFaceFW/fineweb-edu", split: "train", name: null,
     config: null, text_field: "text", limit_docs: 50000, revision: null}

and produces `corpus_dir/<name>.jsonl` with one `{"text": ...}` object per
line. The fetch is idempotent: an existing non-empty output is reused and the
network is never touched for it, so re-running the pipeline does not
re-download. `offline=True` forbids all network access (missing outputs are an
error, not a download).

`datasets` is imported lazily so `import vertex500m.data` never requires it.
"""

from __future__ import annotations

import json
from pathlib import Path


def _slug(path: str, config=None) -> str:
    slug = path.replace("/", "_").replace("-", "_")
    return f"{slug}_{config}" if config else slug


def fetch_corpus(sources, corpus_dir, offline: bool = False,
                 verbose: bool = True) -> list[Path]:
    """Fetch every source into `corpus_dir`. Returns the jsonl paths in order."""
    out_dir = Path(corpus_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    say = print if verbose else (lambda *a: None)
    load_dataset = None            # imported only if a fetch is actually needed
    written = []
    for src in sources:
        src = dict(src)
        path, split = src["path"], src.get("split", "train")
        name = src.get("name") or _slug(path, src.get("config"))
        field = src.get("text_field", "text")
        limit = src.get("limit_docs")
        out = out_dir / f"{name}.jsonl"
        if out.exists() and out.stat().st_size > 0:
            say(f"  [download] cached {out.name} "
                f"({out.stat().st_size / 2**20:.1f} MiB), skipping fetch")
            written.append(out)
            continue
        if offline:
            raise FileNotFoundError(
                f"[download] offline mode and {out} is missing: fetch the corpus "
                f"on a connected machine first (or drop offline mode).")
        if load_dataset is None:
            from datasets import load_dataset
        say(f"  [download] {path} split={split} field={field!r} "
            f"limit={limit} -> {out.name}")
        n = _fetch_one(load_dataset, src, path, split, field, limit, out, say)
        say(f"  [download] wrote {n} documents -> {out.name}")
        written.append(out)
    return written


def _fetch_one(load_dataset, src: dict, path: str, split: str, field: str,
               limit, out: Path, say) -> int:
    kw = {}
    if src.get("config"):
        kw["name"] = src["config"]
    if src.get("revision"):
        kw["revision"] = src["revision"]
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        if limit is not None:
            # Streaming + take: bounded memory AND bounded download. The full
            # split is never materialised locally.
            ds = load_dataset(path, split=split, streaming=True, **kw)
            it = ds.take(limit) if hasattr(ds, "take") else ds
            for row in it:
                t = (row.get(field) or "")
                if isinstance(t, str) and t.strip():
                    f.write(json.dumps({"text": t}) + "\n")
                    n += 1
        else:
            ds = load_dataset(path, split=split, **kw)
            for row in ds:
                t = (row.get(field) or "")
                if isinstance(t, str) and t.strip():
                    f.write(json.dumps({"text": t}) + "\n")
                    n += 1
    if n == 0:
        out.unlink(missing_ok=True)
        raise ValueError(
            f"[download] {path} split={split} yielded no documents with a "
            f"non-empty {field!r} field. Check the dataset name, split, and "
            f"text field.")
    return n
