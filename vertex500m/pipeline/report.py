"""End-of-run reporting: report.md (human) + summary.json (machine).

Inputs are all files (state.json, metrics_*.jsonl, manifest.json, checkpoint
listing, eval stdout) -- `build_summary` is pure file-reading, so the report
can be regenerated any time with `--stages report`.
"""

from __future__ import annotations

import json
import math
import os
import re
import time


def parse_eval_output(text: str) -> dict:
    """Pull the numbers out of `python -m vertex500m.eval` stdout."""
    out: dict = {}
    m = re.search(r"mean CE\s*:\s*([0-9.eE+-]+)", text)
    if m:
        out["val_ce"] = float(m.group(1))
    m = re.search(r"perplexity\s*:\s*([0-9.eE+-]+)", text)
    if m:
        out["perplexity"] = float(m.group(1))
    m = re.search(r"bits/token\s*:\s*([0-9.eE+-]+)", text)
    if m:
        out["bits_per_token"] = float(m.group(1))
    m = re.search(r"tokens\s*:\s*([0-9,]+)", text)
    if m:
        out["eval_tokens"] = int(m.group(1).replace(",", ""))
    m = re.search(r"mean accept:\s*([0-9.eE+-]+)%", text)
    if m:
        out["acceptance_pct"] = float(m.group(1))
    return out


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except ValueError:
                    pass
    return recs


def _ckpt_inventory(ckpt_dir: str) -> list[dict]:
    inv = []
    if not os.path.isdir(ckpt_dir):
        return inv
    for name in sorted(os.listdir(ckpt_dir)):
        if name.endswith(".pt"):
            p = os.path.join(ckpt_dir, name)
            try:
                inv.append({"file": name, "bytes": os.path.getsize(p)})
            except OSError:
                pass
    return inv


def build_summary(run_dir: str, cfg, state: dict,
                  eval_result: dict | None = None) -> dict:
    """Assemble the summary dict from run-dir artefacts."""
    from ..data import read_manifest
    metrics_files = {"stage_a": os.path.join(run_dir, "metrics_pre.jsonl"),
                     "stage_b": os.path.join(run_dir, "metrics_dspark.jsonl")}
    # legacy single-file layout (older runs): fold it in as stage_a
    legacy = os.path.join(run_dir, "metrics.jsonl")
    summary: dict = {
        "run_name": cfg.run_name,
        "config_hash": state.get("config_hash"),
        "created": state.get("created"),
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stages": state.get("stages", {}),
        "wandb": {"mode": cfg.tracking.mode,
                  "run_id": state.get("wandb_run_id")},
        "data": {},
        "train": {},
        "checkpoints": _ckpt_inventory(os.path.join(run_dir, "checkpoints")),
        "eval": eval_result or state.get("eval_result"),
    }
    try:
        m = read_manifest(os.path.join(run_dir, "data"))
        summary["data"] = {
            "tokenizer": m.get("tokenizer"),
            "tokens": m.get("tokens"),
            "compressed_vocab_size": m.get("engram_compressed_vocab_size"),
            "collisions": m.get("collisions"),
            "seed": m.get("seed"),
            "val_fraction": m.get("val_fraction"),
        }
    except (FileNotFoundError, ValueError, KeyError):
        pass
    train_recs = _read_jsonl(metrics_files["stage_a"]) + \
        (_read_jsonl(legacy) if not os.path.exists(metrics_files["stage_a"])
         else [])
    dspark_recs = _read_jsonl(metrics_files["stage_b"])
    steps = [r for r in train_recs if r.get("split") == "train"]
    vals = [r for r in train_recs if r.get("split") == "val"]
    events = [r for r in train_recs + dspark_recs if r.get("event")]
    if steps:
        last = steps[-1]
        summary["train"]["stage_a"] = {
            "steps": last.get("step"),
            "final_loss": last.get("loss"),
            "final_lr": last.get("lr"),
        }
    if vals:
        summary["train"]["val"] = {
            "evaluations": len(vals),
            "best_val_ce": min(r["val_ce"] for r in vals if "val_ce" in r),
            "last_val_ce": vals[-1].get("val_ce"),
        }
    bsteps = [r for r in dspark_recs if r.get("split") == "train"]
    if bsteps:
        summary["train"]["stage_b"] = {
            "steps": bsteps[-1].get("step"),
            "final_loss": bsteps[-1].get("loss"),
            "final_accept": bsteps[-1].get("accept_now"),
        }
    if events:
        summary["train"]["events"] = events
    return summary


def render_report(summary: dict) -> str:
    L = [f"# run report: {summary.get('run_name')}",
         "",
         f"- config hash: `{summary.get('config_hash')}`",
         f"- updated: {summary.get('updated')}",
         f"- wandb: {summary['wandb']['mode']} "
         f"({summary['wandb'].get('run_id') or 'no run'})",
         "",
         "## stages",
         ""]
    for name, st in (summary.get("stages") or {}).items():
        secs = st.get("seconds")
        L.append(f"- **{name}**: {st.get('status')}"
                 + (f" ({secs:.0f}s)" if isinstance(secs, (int, float)) else "")
                 + (f" -- {st.get('detail')}" if st.get("detail") else ""))
    d = summary.get("data") or {}
    if d:
        L += ["", "## data", "",
              f"- tokenizer: `{d.get('tokenizer')}`",
              f"- tokens: {d.get('tokens')}",
              f"- engram cvs: {d.get('compressed_vocab_size')} "
              f"(collisions {d.get('collisions')})",
              f"- split seed: {d.get('seed')}, val fraction: "
              f"{d.get('val_fraction')}"]
    t = summary.get("train") or {}
    if t:
        L += ["", "## training", ""]
        a = t.get("stage_a") or {}
        if a:
            L.append(f"- stage A: {a.get('steps')} steps, final loss "
                     f"{a.get('final_loss')}")
        v = t.get("val") or {}
        if v:
            L.append(f"- val: best CE {v.get('best_val_ce')} over "
                     f"{v.get('evaluations')} evals")
        b = t.get("stage_b") or {}
        if b:
            L.append(f"- stage B: {b.get('steps')} steps, final loss "
                     f"{b.get('final_loss')}, accept {b.get('final_accept')}")
        for ev in t.get("events") or []:
            L.append(f"- event {ev.get('event')}: {ev}")
    ck = summary.get("checkpoints") or []
    if ck:
        L += ["", "## checkpoints", ""]
        for c in ck:
            L.append(f"- {c['file']} ({c['bytes']/2**30:.2f} GiB)")
    e = summary.get("eval") or {}
    if e:
        L += ["", "## final eval", ""]
        for k, v in e.items():
            L.append(f"- {k}: {v}")
    L += ["", "_Regenerate with "
           "`python -m vertex500m.pipeline --config ... --stages report`._",
          ""]
    return "\n".join(L)


def write_report(run_dir: str, cfg, state: dict,
                 eval_result: dict | None = None) -> dict:
    """Write summary.json + report.md. Returns the summary dict."""
    summary = build_summary(run_dir, cfg, state, eval_result)
    if eval_result:
        state["eval_result"] = eval_result
    with open(os.path.join(run_dir, "summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(run_dir, "report.md"), "w",
              encoding="utf-8") as f:
        f.write(render_report(summary))
    return summary
