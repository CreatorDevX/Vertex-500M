"""Pipeline orchestration: preflight -> data -> stage A -> stage B -> eval -> report.

Each stage runs as a SUBPROCESS (`python -m vertex500m.*`) for crash
isolation and torchrun-compatibility; the orchestrator streams each child's
output to console + a stage log file. `state.json` records every stage, so
re-running the same command is the resume mechanism: completed stages with
verified artefacts are skipped, interrupted training resumes from its latest
checkpoint. There is deliberately NO auto-retry -- a crash is investigated,
then the same command re-runs.

Exit codes: 0 = all selected stages done; 1 = a stage failed (or preflight);
10 = stopped at a deadline (re-run resumes); 130/143 = interrupted by signal
(child was left to save its own checkpoint first).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

STOP = {"signum": None, "count": 0}


def _on_signal(signum, frame):
    STOP["count"] += 1
    STOP["signum"] = signum


STAGE_ORDER = ("preflight", "data", "stage_a", "stage_b", "eval", "report")
NEEDS = {"preflight": [], "data": ["preflight"], "stage_a": ["data"],
         "stage_b": ["stage_a"], "eval": ["stage_a"], "report": []}


def repo_root() -> str:
    import vertex500m
    return os.path.dirname(os.path.dirname(os.path.abspath(vertex500m.__file__)))


def default_paths(cfg, run_dir: str) -> dict:
    """Resolve every derived path (explicit config wins, else run-contained)."""
    d = cfg.data
    data_dir = d.dir or os.path.join(run_dir, "data")
    return {
        "run_dir": run_dir,
        "data_dir": data_dir,
        "corpus_dir": d.corpus_dir or os.path.join(run_dir, "corpus"),
        "token_map": d.token_map or os.path.join(run_dir, "token_map.json"),
        "ckpt_dir": os.path.join(run_dir, "checkpoints"),
        "backbone": os.path.join(run_dir, "backbone.pt"),
        "full": os.path.join(run_dir, "full.pt"),
        "metrics_pre": os.path.join(run_dir, "metrics_pre.jsonl"),
        "metrics_dspark": os.path.join(run_dir, "metrics_dspark.jsonl"),
    }


def load_state(run_dir: str) -> dict:
    p = os.path.join(run_dir, "state.json")
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except ValueError:
            pass
    return {"config_hash": None, "wandb_run_id": None, "created": None,
            "updated": None, "stages": {}, "tests_passed": False,
            "eval_result": None}


def save_state(run_dir: str, state: dict) -> None:
    state["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp = os.path.join(run_dir, "state.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, os.path.join(run_dir, "state.json"))


def mark(state: dict, stage: str, status: str, detail: str | None = None,
         seconds: float | None = None, extra: dict | None = None) -> None:
    st = state.setdefault("stages", {}).setdefault(stage, {})
    st["status"] = status
    if detail is None:
        pass
    elif detail:
        st["detail"] = detail
    else:
        st.pop("detail", None)      # stale details must not survive a retry
    if seconds is not None:
        st["seconds"] = round(seconds, 1)
    if extra:
        st.update(extra)


def run_subprocess(argv: list[str], log_path: str, extra_env: dict | None = None,
                   say=print) -> int:
    """Run a stage, teeing output to console + log file. Signal-aware wait.

    The child gets every signal directly too (same console), and saves its own
    checkpoint on SIGINT/SIGTERM -- so on the first signal we just keep
    draining until it exits; a second signal kills it. Returns the exit code.
    """
    import queue
    import threading
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)
    say(f"[run] {' '.join(argv)}", flush=True)
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            errors="replace", bufsize=1, env=env,
                            cwd=repo_root())
    q: queue.Queue = queue.Queue()

    def reader():
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                q.put(line)
        finally:
            q.put(None)

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n=== {' '.join(argv)} ===\n")
        log.flush()
        while True:
            if STOP["count"] >= 2 and proc.poll() is None:
                try:
                    proc.kill()
                except OSError:
                    pass
            try:
                item = q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            log.write(item)
            say(item.rstrip("\n"), flush=True)
    t.join(timeout=10)
    return int(proc.wait())


# --------------------------------------------------------------------------- #
# argv builders
# --------------------------------------------------------------------------- #
def train_argv(cfg, paths: dict, stage: str, init: str | None,
               resume: bool) -> list[str]:
    t = cfg.train
    # Multi-GPU: launch the stage under torchrun (each rank is rank-gated for
    # prints/saves/metrics inside the loop, so orchestration is unchanged).
    argv = [sys.executable, "-m", "torch.distributed.run",
            "--nproc_per_node", str(t.dist_nproc),
            "--module", "vertex500m.train"] if t.dist_nproc > 1 else \
        [sys.executable, "-m", "vertex500m.train"]
    if stage == "dspark":
        argv += ["--stage", "dspark"]
    argv += ["--data", paths["data_dir"],
             "--out", paths["full"] if stage == "dspark" else paths["backbone"],
             "--steps", str(t.stage_b_steps if stage == "dspark"
                            else t.stage_a_steps),
             "--seed", str(t.seed), "--amp", t.amp,
             "--hc-variant", cfg.model.hc_variant,
             "--batch-size", str(t.batch_size), "--seq-len", str(t.seq_len),
             "--loss-chunk", str(t.loss_chunk),
             "--accum-steps", str(t.accum_steps),
             "--muon-lr", repr(t.muon_lr), "--adamw-lr", repr(t.adamw_lr),
             "--mtp-lr", repr(t.mtp_lr), "--embed-lr", repr(t.embed_lr),
             "--weight-decay", repr(t.weight_decay),
             "--embed-optimizer", t.embed_optimizer,
             "--warmup", str(t.warmup), "--quant-mode", t.quant_mode,
             "--log-every", str(t.log_every),
             "--pack", cfg.data.pack,
             "--train-split", "train", "--val-split", "val",
             "--save-every", str(t.save_every), "--keep-last", str(t.keep_last),
             "--eval-every", str(t.eval_every),
             "--ckpt-dir", paths["ckpt_dir"],
             "--metrics", paths["metrics_dspark"] if stage == "dspark"
                          else paths["metrics_pre"]]
    if t.eval_batches is None:
        argv += ["--eval-batches", "0"]
    else:
        argv += ["--eval-batches", str(t.eval_batches)]
    if t.grad_clip is not None:
        argv += ["--grad-clip", repr(t.grad_clip)]
    if t.target_loss is not None:
        argv += ["--target-loss", repr(t.target_loss)]
    if t.deadline_s is not None:
        argv += ["--deadline-s", repr(t.deadline_s)]
    if t.compile:
        argv += ["--compile"]
    if t.no_ckpt:
        argv += ["--no-ckpt"]
    if cfg.model.tiny:
        argv += ["--tiny"]
    if t.device == "cpu":
        argv += ["--cpu"]
    if os.path.exists(paths["token_map"]):
        argv += ["--token-map", paths["token_map"]]
    elif t.allow_identity_map:
        argv += ["--allow-identity-map"]
    if init:
        argv += ["--init", init]
        if resume:
            argv += ["--resume"]
    return argv


def tokenize_argv(cfg, paths: dict, inputs: list[str]) -> list[str]:
    d = cfg.data
    return ([sys.executable, "-m", "vertex500m.tokenize",
             "--tokenizer", d.tokenizer,
             "--input"] + inputs + [
             "--out", paths["data_dir"], "--name", d.name,
             "--tokens-per-shard", str(d.tokens_per_shard),
             "--val-fraction", repr(d.val_fraction), "--seed", str(d.seed),
             "--save-map", paths["token_map"]])


def eval_argv(cfg, paths: dict, ckpt: str, with_dspark: bool) -> list[str]:
    e = cfg.eval
    argv = [sys.executable, "-m", "vertex500m.eval",
            "--ckpt", ckpt, "--data", paths["data_dir"],
            "--split", e.split, "--seq-len", str(e.seq_len),
            "--batch-size", str(e.batch_size),
            "--hc-variant", cfg.model.hc_variant]
    if e.batches is None:
        argv += ["--max-batches", "0"]
    else:
        argv += ["--max-batches", str(e.batches)]
    if os.path.exists(paths["token_map"]):
        argv += ["--token-map", paths["token_map"]]
    if with_dspark:
        argv += ["--with-dspark"]
    if cfg.model.tiny:
        argv += ["--tiny"]
    if cfg.train.device == "cpu":
        argv += ["--cpu"]
    return argv


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def stage_preflight(cfg, ctx, say=print) -> bool:
    from .preflight import run_preflight
    say("[preflight] running checks...")
    suites_already_green = bool(ctx["state"].get("tests_passed", False)
                                and not ctx.get("force_tests"))
    skip = (not cfg.pipeline.run_tests) or suites_already_green
    if not cfg.pipeline.run_tests:
        say("[preflight] test suites disabled by config (pipeline.run_tests: false)")
    pctx = dict(ctx["paths"], skip_tests=skip)
    t0 = time.time()
    results = run_preflight(cfg, pctx)
    fails = [m for lvl, m in results if lvl == "fail"]
    for lvl, m in results:
        for line in m.splitlines():
            say(f"  [{'ok' if lvl == 'ok' else lvl.upper()}] {line}")
    if fails:
        mark(ctx["state"], "preflight", "failed",
             f"{len(fails)} check(s) failed", time.time() - t0)
        return False
    if not skip:
        ctx["state"]["tests_passed"] = True
    mark(ctx["state"], "preflight", "done", "", time.time() - t0)
    ctx["cvs"] = pctx.get("cvs")
    return True


def _manifest_matches(m: dict, cfg) -> tuple[bool, str]:
    d = cfg.data
    for k, want in (("tokenizer", d.tokenizer), ("val_fraction", d.val_fraction),
                    ("seed", d.seed)):
        if m.get(k) != want:
            return False, f"manifest {k}={m.get(k)!r} != config {want!r}"
    return True, ""


def stage_data(cfg, ctx, say=print) -> bool:
    from ..data import (fetch_corpus, read_manifest, synthetic_corpus,
                        validate_manifest)
    paths, state, d = ctx["paths"], ctx["state"], cfg.data
    t0 = time.time()
    # idempotency: a valid manifest matching this config means data is done
    try:
        m = read_manifest(paths["data_dir"])
        problems = validate_manifest(m, paths["data_dir"])
        if not problems:
            ok, why = _manifest_matches(m, cfg)
            if ok:
                mark(state, "data", "done",
                     f"up to date ({m.get('tokens')} tokens)",
                     time.time() - t0)
                say("[data] manifest valid and matches config -- skipping")
                return True
            say(f"[data] manifest mismatch ({why}); re-tokenizing")
        else:
            say(f"[data] manifest invalid ({len(problems)} problems); rebuilding")
    except FileNotFoundError:
        pass
    if d.synthetic_tokens is not None:
        os.makedirs(paths["data_dir"], exist_ok=True)
        synthetic_corpus(paths["data_dir"], n_tokens=d.synthetic_tokens,
                         vocab_size=32000)
        # name the shard for its split so split-aware loaders accept it
        src = os.path.join(paths["data_dir"], "synthetic_00000.bin")
        dst = os.path.join(paths["data_dir"], f"{d.name}_train_00000.bin")
        if os.path.exists(src):
            os.replace(src, dst)
        mark(state, "data", "done",
             f"synthetic ({d.synthetic_tokens} tokens)", time.time() - t0)
        return True
    # download (idempotent per source inside fetch_corpus)
    inputs: list[str] = []
    if d.download:
        try:
            got = fetch_corpus(d.download, paths["corpus_dir"],
                               offline=ctx.get("offline", False))
        except Exception as e:  # noqa: BLE001
            mark(state, "data", "failed", f"download: {e}", time.time() - t0)
            return False
        inputs += [str(p) for p in got]
    inputs += list(d.local or [])
    if not inputs:
        mark(state, "data", "failed", "no corpus inputs (download + local empty)",
             time.time() - t0)
        return False
    code = run_subprocess(tokenize_argv(cfg, paths, inputs),
                          os.path.join(ctx["run_dir"], "data.log"))
    if code != 0:
        mark(state, "data", "failed", f"tokenize exit {code}", time.time() - t0)
        return False
    try:
        m = read_manifest(paths["data_dir"])
        problems = validate_manifest(m, paths["data_dir"])
    except Exception as e:  # noqa: BLE001
        mark(state, "data", "failed", f"manifest: {e}", time.time() - t0)
        return False
    if problems:
        mark(state, "data", "failed",
             "manifest invalid after tokenize: " + "; ".join(problems),
             time.time() - t0)
        return False
    mark(state, "data", "done", f"({m.get('tokens')} tokens)", time.time() - t0)
    return True


def _train_status(paths: dict, stage: str, want_steps: int, out_path: str,
                  ) -> tuple[str, str | None, bool]:
    """(action, init_ckpt, resume) for a training stage.

    action: skip (artefact complete) | resume (a checkpoint with train state
    exists) | fresh. The resume point is whichever of latest/final has more
    completed steps (the final save carries full train state too, so extending
    a completed run resumes from it, not from scratch).
    """
    import torch

    def _ckpt_step(p: str) -> int:
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            return int((ck.get("train") or {}).get("step", 0))
        except Exception:  # noqa: BLE001
            return -1

    tag = "dspark" if stage == "dspark" else "pre"
    latest = os.path.join(paths["ckpt_dir"], f"{tag}.latest.pt")
    step_out = _ckpt_step(out_path) if os.path.exists(out_path) else -1
    step_latest = _ckpt_step(latest) if os.path.exists(latest) else -1
    if step_out >= want_steps:
        return "skip", None, False
    if step_latest >= step_out and step_latest >= 0:
        return "resume", latest, True
    if step_out >= 0:
        return "resume", out_path, True
    return "fresh", None, False


def stage_train(cfg, ctx, stage: str, say=print) -> bool:
    paths, state, t = ctx["paths"], ctx["state"], cfg.train
    want = t.stage_b_steps if stage == "dspark" else t.stage_a_steps
    out_path = paths["full"] if stage == "dspark" else paths["backbone"]
    t0 = time.time()
    action, latest, resume_flag = _train_status(paths, stage, want, out_path)
    if action == "skip":
        mark(state, f"stage_{'b' if stage == 'dspark' else 'a'}", "done",
             "artefact complete -- skipping", 0.0)
        say(f"[{stage}] {out_path} complete -- skipping")
        return True
    init = None
    if action == "resume":
        init = latest
        say(f"[{stage}] resuming from {latest}")
    elif stage == "dspark":
        # prefer the best stage-A checkpoint, then latest, then the final
        cands = [os.path.join(paths["ckpt_dir"], "pre.best.pt"),
                 os.path.join(paths["ckpt_dir"], "pre.latest.pt"),
                 paths["backbone"]]
        init = next((c for c in cands if os.path.exists(c)), None)
        if init is None:
            mark(state, "stage_b", "failed", "no stage-A checkpoint found",
                 time.time() - t0)
            return False
        say(f"[stage_b] init from {init}")
    tracker = ctx.get("tracker")
    metrics = paths["metrics_dspark"] if stage == "dspark" else paths["metrics_pre"]
    if tracker is not None:
        tracker.tail_file(metrics, "dspark/" if stage == "dspark" else "")
    code = run_subprocess(train_argv(cfg, paths, stage, init, resume_flag),
                          os.path.join(ctx["run_dir"], f"{stage}.log"))
    tag = "stage_b" if stage == "dspark" else "stage_a"
    if code == 0:
        if not os.path.exists(out_path):
            mark(state, tag, "failed", f"exit 0 but {out_path} missing",
                 time.time() - t0)
            return False
        mark(state, tag, "done", "", time.time() - t0)
        return True
    if code == 10:
        mark(state, tag, "deadline", "wall-clock budget hit; re-run resumes",
             time.time() - t0)
        say(f"[{stage}] deadline exit (10): re-run the same command to resume")
        return False
    mark(state, tag, "failed", f"exit {code} (see {stage}.log)",
         time.time() - t0)
    return False


def stage_eval(cfg, ctx, say=print) -> bool:
    from .report import parse_eval_output
    paths, state = ctx["paths"], ctx["state"]
    t0 = time.time()
    ran_b = state.get("stages", {}).get("stage_b", {}).get("status") == "done"
    ckpt = paths["full"] if (ran_b and os.path.exists(paths["full"])) \
        else paths["backbone"]
    if not os.path.exists(ckpt):
        mark(state, "eval", "failed", f"checkpoint missing: {ckpt}",
             time.time() - t0)
        return False
    prev = state.get("eval_result") or {}
    try:
        if prev.get("ckpt") == ckpt and prev.get("mtime") == os.path.getmtime(ckpt):
            mark(state, "eval", "done", "checkpoint unchanged -- skipping", 0.0)
            say("[eval] checkpoint unchanged -- skipping")
            return True
    except OSError:
        pass
    with_dspark = ckpt == paths["full"]
    code = run_subprocess(eval_argv(cfg, paths, ckpt, with_dspark),
                          os.path.join(ctx["run_dir"], "eval.log"))
    if code != 0:
        mark(state, "eval", "failed", f"exit {code} (see eval.log)",
             time.time() - t0)
        return False
    try:
        with open(os.path.join(ctx["run_dir"], "eval.log"), encoding="utf-8",
                  errors="replace") as f:
            text = f.read()
    except OSError:
        text = ""
    result = parse_eval_output(text)
    result["ckpt"] = ckpt
    try:
        result["mtime"] = os.path.getmtime(ckpt)
    except OSError:
        pass
    state["eval_result"] = result
    tracker = ctx.get("tracker")
    if tracker is not None:
        tracker.summary({f"eval/{k}": v for k, v in result.items()
                         if isinstance(v, (int, float))})
    if not result:
        mark(state, "eval", "done", "completed (numbers unparsed; see eval.log)",
             time.time() - t0)
    else:
        mark(state, "eval", "done",
             "val CE %(val_ce)s" % {"val_ce": result.get("val_ce", "?")},
             time.time() - t0)
    return True


def stage_report(cfg, ctx, say=print) -> bool:
    from .report import write_report
    paths, state = ctx["paths"], ctx["state"]
    t0 = time.time()
    summary = write_report(ctx["run_dir"], cfg, state,
                           state.get("eval_result"))
    tracker = ctx.get("tracker")
    if tracker is not None:
        flat: dict = {}
        for sec in ("train", "eval", "data"):
            v = summary.get(sec)
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, (int, float)):
                        flat[f"{sec}/{k2}"] = v2
        tracker.summary(flat)
    mark(state, "report", "done", "", time.time() - t0)
    say(f"[report] {os.path.join(ctx['run_dir'], 'report.md')}")
    return True


# --------------------------------------------------------------------------- #
def selected_stages(cfg, cli_stages: list[str] | None) -> list[str]:
    sel = list(cli_stages) if cli_stages else list(cfg.pipeline.stages or STAGE_ORDER)
    unknown = [s for s in sel if s not in STAGE_ORDER]
    if unknown:
        raise SystemExit(f"unknown stages {unknown} (known: {list(STAGE_ORDER)})")
    return [s for s in STAGE_ORDER if s in sel]


def check_prereqs(sel: list[str], state: dict) -> None:
    done = {s for s, st in state.get("stages", {}).items()
            if st.get("status") == "done"}
    for s in sel:
        for need in NEEDS[s]:
            if need not in done and need not in sel:
                raise SystemExit(
                    f"stage {s!r} needs {need!r}, which is neither done nor "
                    f"selected. Select it too (--stages ...,{need},...) or run "
                    f"the full pipeline first.")


def run_pipeline(cfg, paths: dict, run_dir: str, sel: list[str],
                 tracker=None, say=print, validate_only: bool = False,
                 force_tests: bool = False, offline: bool = False,
                 allow_config_change: bool = False) -> int:
    """Execute the selected stages in order. Returns the process exit code."""
    from .run_config import config_hash
    state = load_state(run_dir)
    if state["created"] is None:
        state["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    h = config_hash(cfg)
    if state.get("config_hash") and state["config_hash"] != h:
        if not allow_config_change:
            save_state(run_dir, state)
            say("[pipeline] config CHANGED since this run started "
                f"({state['config_hash']} -> {h}). Refusing: a resumed run with "
                f"different hyperparameters trains a different run under the "
                f"same name. Re-run with --allow-config-change to override.")
            return 1
        say("[pipeline] WARNING: config changed; continuing (--allow-config-change)")
    state["config_hash"] = h
    ctx = {"paths": paths, "run_dir": run_dir, "state": state, "tracker": tracker,
           "stages_selected": sel, "force_tests": force_tests,
           "offline": offline or cfg.pipeline.offline,
           "say": say}
    save_state(run_dir, state)
    if validate_only:
        ok = stage_preflight(cfg, ctx, say)
        return 0 if ok else 1
    check_prereqs(sel, state)
    runners = {"preflight": stage_preflight, "data": stage_data,
               "stage_a": lambda c, x, s=say: stage_train(c, x, "pre", s),
               "stage_b": lambda c, x, s=say: stage_train(c, x, "dspark", s),
               "eval": stage_eval, "report": stage_report}
    for s in sel:
        if STOP["count"] >= 2:
            say("[pipeline] interrupted twice; stopping")
            save_state(run_dir, state)
            return 130
        say(f"\n===== stage: {s} =====")
        ok = runners[s](cfg, ctx, say)
        save_state(run_dir, state)
        if not ok:
            if state.get("stages", {}).get(s, {}).get("status") == "deadline":
                return 10
            return 1
    return 0
