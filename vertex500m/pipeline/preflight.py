"""Pre-flight: fail fast BEFORE a multi-day run, not 40 hours in.

Every check returns (level, message) with level in {"ok", "warn", "fail"}.
`run_preflight` runs all of them (optionally skipping the slow test suites)
and returns the full list; the orchestrator fails the run on any "fail".

Deliberately paranoid: tokenizer reachability, token-map measurability, disk
budget, wandb auth, manifest integrity and even a no-`input()` audit (the
training device may have no terminal -- anything that prompts is a hang, and
a hang with no terminal is indistinguishable from a crash).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def _have(mod: str) -> tuple[bool, str]:
    try:
        m = __import__(mod)
        return True, getattr(m, "__version__", "?")
    except ImportError:
        return False, ""


@check
def check_python(cfg, ctx):
    import sys as _s
    if _s.version_info < (3, 10):
        return [("fail", f"python {_s.version.split()[0]} < 3.10 "
                         f"(needs X | Y syntax)")]
    return [("ok", f"python {_s.version.split()[0]}")]


@check
def check_dependencies(cfg, ctx):
    out = []
    ok, v = _have("torch")
    out.append(("ok" if ok else "fail", f"torch {v}" if ok else
                "torch is not installed (pip install torch)"))
    ok, v = _have("numpy")
    out.append(("ok" if ok else "fail", f"numpy {v}" if ok else
                "numpy is not installed"))
    if cfg.data.synthetic_tokens is None:
        ok, v = _have("transformers")
        out.append(("ok" if ok else "fail", f"transformers {v}" if ok else
                    "transformers is not installed (needed for tokenizing)"))
    if cfg.data.download:
        ok, v = _have("datasets")
        out.append(("ok" if ok else "fail", f"datasets {v}" if ok else
                    "datasets is not installed (needed for corpus download)"))
    if cfg.tracking.mode != "disabled":
        ok, v = _have("wandb")
        out.append(("ok" if ok else "fail", f"wandb {v}" if ok else
                    "wandb is not installed (or set tracking.mode: disabled)"))
    return out


@check
def check_device_amp(cfg, ctx):
    try:
        import torch
    except ImportError:
        return [("fail", "torch missing (see dependencies)")]
    amp = cfg.train.amp
    if not torch.cuda.is_available():
        if amp != "off":
            return [("fail", f"train.amp={amp!r} but no CUDA device found "
                             f"(this box is CPU-only): set amp: off")]
        return [("ok", "CPU-only box, amp off")]
    cap = torch.cuda.get_device_capability()
    n = torch.cuda.device_count()
    msgs = [(("ok", f"{n}x CUDA, sm_{cap[0]}{cap[1]}"))]
    if amp == "bf16" and cap < (8, 0):
        msgs.append(("fail", f"train.amp=bf16 but sm_{cap[0]}{cap[1]} has no "
                             f"BF16 tensor cores (needs sm_80+): use fp16"))
    return msgs


@check
def check_params_gate(cfg, ctx):
    r = subprocess.run([sys.executable, "-m", "vertex500m.params",
                        "--hc-variant", cfg.model.hc_variant],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return [("fail", f"params gate failed:\n{r.stderr[-2000:]}")]
    line = [ln for ln in r.stdout.splitlines() if "total" in ln.lower()]
    return [("ok", f"params gate ({cfg.model.hc_variant}): "
                   f"{(line[-1].strip() if line else 'passed')}")]


@check
def check_test_suites(cfg, ctx):
    if ctx.get("skip_tests"):
        return [("warn", "test suites skipped (disabled in config or passing "
                         "record in state.json; use --force-tests to re-run)")]
    out = []
    for mod in ("vertex500m.tests", "vertex500m.test_checkpoint",
                "vertex500m.test_pipeline"):
        r = subprocess.run([sys.executable, "-m", mod],
                           capture_output=True, text=True)
        tail = "\n".join((r.stdout + r.stderr).splitlines()[-6:])
        if r.returncode != 0:
            return [("fail", f"{mod} FAILED -- fix before training:\n{tail}")]
        out.append(("ok", f"{mod} passed"))
    return out


@check
def check_tokenizer_and_map(cfg, ctx):
    if cfg.data.synthetic_tokens is not None:
        return [("ok", "synthetic corpus: no tokenizer needed")]
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg.data.tokenizer,
                                            trust_remote_code=False)
    except Exception as e:  # noqa: BLE001
        return [("fail", f"tokenizer {cfg.data.tokenizer!r} unreachable: {e}. "
                         f"Fetch/cache it on a connected machine first.")]
    try:
        from ..tokenize import build_token_map
        from ..config import PRESET
        vocab_strings = tok.convert_ids_to_tokens(list(range(len(tok))))
        _, cvs, collisions = build_token_map(vocab_strings)
    except Exception as e:  # noqa: BLE001
        return [("fail", f"could not measure the Engram token map: {e}")]
    limit = PRESET.engram_vocab_size
    if not 0 < cvs < limit:
        return [("fail", f"compressed vocab {cvs} not in (0, {limit}): raise "
                         f"engram_vocab_size or change tokenizer")]
    ctx["cvs"] = cvs
    return [("ok", f"tokenizer {cfg.data.tokenizer} V={len(tok)} cvs={cvs} "
                   f"collisions={collisions}")]


@check
def check_manifest(cfg, ctx):
    from ..data import read_manifest, validate_manifest
    manifest_path = os.path.join(ctx["data_dir"], "manifest.json")
    if not os.path.exists(manifest_path):
        return [("ok", "no manifest yet (data stage will create one)")]
    try:
        m = read_manifest(ctx["data_dir"])
    except Exception as e:  # noqa: BLE001
        return [("fail", f"unreadable manifest: {e}")]
    problems = validate_manifest(m, ctx["data_dir"])
    if problems:
        return [("fail", "data manifest INVALID:\n" + "\n".join(
            f"  - {p}" for p in problems))]
    toks = m.get("tokens", {})
    if m.get("tokenizer") != cfg.data.tokenizer:
        return [("fail", f"manifest tokenizer {m.get('tokenizer')!r} != config "
                         f"{cfg.data.tokenizer!r}: re-tokenize")]
    return [("ok", f"manifest valid ({toks} tokens) -- data stage will skip")]


@check
def check_disk(cfg, ctx):
    try:
        from ..params import count
        from ..config import PRESET
        c = count(PRESET, cfg.model.hc_variant)
        # weights + grads + optimizer state, fp32, plus margin for step files
        need = c.total * 16 * (cfg.train.keep_last + 3)
    except Exception:  # noqa: BLE001
        need = 12_000_000_000
    try:
        free = shutil.disk_usage(ctx["run_dir"]).free
    except OSError as e:
        return [("fail", f"cannot stat run dir disk: {e}")]
    if free < need:
        return [("fail", f"disk free {free/2**30:.1f} GiB < estimated "
                         f"{need/2**30:.1f} GiB for {(cfg.train.keep_last + 3)} "
                         f"checkpoints: free space or lower keep_last")]
    return [("ok", f"disk free {free/2**30:.1f} GiB "
                   f"(need ~{need/2**30:.1f} GiB)")]


@check
def check_wandb_auth(cfg, ctx):
    if cfg.tracking.mode != "online":
        return [("ok", f"tracking {cfg.tracking.mode}: no auth needed")]
    if os.environ.get("WANDB_API_KEY"):
        return [("ok", "tracking online (WANDB_API_KEY is set)")]
    try:
        import wandb
        key = wandb.api.api_key
    except Exception as e:  # noqa: BLE001
        return [("fail", f"tracking online but wandb auth unreadable ({e})")]
    if not key:
        return [("fail", "tracking online but no wandb API key: run `wandb "
                         "login` on a machine WITH a terminal, or set "
                         "WANDB_API_KEY before deploying (or use mode: offline)")]
    return [("ok", "tracking online (wandb key present)")]


@check
def check_no_prompts(cfg, ctx):
    """No `input()` CALLS in pipeline/train/data (token-based, so docstrings
    and the string "input(" in comments/code don't trip it)."""
    import io
    import pathlib
    import tokenize as _tok
    root = pathlib.Path(__file__).resolve().parent.parent
    offenders = []
    for sub in ("pipeline", "train", "data"):
        for p in sorted((root / sub).glob("*.py")):
            try:
                toks = list(_tok.generate_tokens(
                    io.StringIO(p.read_text(encoding="utf-8")).readline))
            except (OSError, SyntaxError, _tok.TokenError):
                continue
            for a, b in zip(toks, toks[1:]):
                if a.type == _tok.NAME and a.string == "input" \
                        and b.type == _tok.OP and b.string == "(":
                    offenders.append(f"{p.name}:{a.start[0]}")
    if offenders:
        return [("fail", f"interactive input() calls (hang with no terminal): "
                         f"{offenders}")]
    return [("ok", "no interactive prompts in pipeline/train/data")]


@check
def check_paths_writable(cfg, ctx):
    try:
        for d in (ctx["run_dir"], ctx["data_dir"], ctx["ckpt_dir"],
                  ctx["corpus_dir"]):
            os.makedirs(d, exist_ok=True)
        probe = os.path.join(ctx["run_dir"], ".writable")
        with open(probe, "w") as f:
            f.write("ok")
        os.unlink(probe)
    except OSError as e:
        return [("fail", f"run directories not writable: {e}")]
    return [("ok", f"run dir writable: {ctx['run_dir']}")]


def run_preflight(cfg, ctx) -> list[tuple[str, str]]:
    """Run every check. `ctx` carries run_dir/data_dir/ckpt_dir/corpus_dir,
    skip_tests, and anything checks want to stash (e.g. cvs)."""
    out: list[tuple[str, str]] = []
    for fn in CHECKS:
        try:
            out.extend(fn(cfg, ctx))
        except Exception as e:  # noqa: BLE001
            out.append(("fail", f"{fn.__name__} crashed: {e}"))
    return out
