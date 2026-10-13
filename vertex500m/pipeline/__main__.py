"""One-command training pipeline.

    python -m vertex500m.pipeline --config runs/base.yaml
    python -m vertex500m.pipeline --config runs/base.yaml --validate
    python -m vertex500m.pipeline --config runs/base.yaml --stages data,stage_a
    python -m vertex500m.pipeline --config runs/base.yaml --set train.batch_size=8

The config file is the ONLY input: the training device may have no terminal,
so nothing here ever prompts. Re-running the same command resumes (completed
stages are skipped after verifying their artefacts; interrupted training
continues from its latest checkpoint). Exit codes: 0 done, 1 failed,
10 deadline (re-run resumes), 130/143 interrupted.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
from dataclasses import asdict

from .orchestrator import STOP, default_paths, run_pipeline, selected_stages
from .run_config import (apply_set_overrides, check_config, dump_config,
                         load_run_config)


def _install():
    try:
        signal.signal(signal.SIGINT, lambda s, f: STOP.update(signum=s, count=STOP["count"] + 1))
        signal.signal(signal.SIGTERM, lambda s, f: STOP.update(signum=s, count=STOP["count"] + 1))
    except (ValueError, OSError, RuntimeError):
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None,
                    help="run YAML (default: ./config.yaml if present)")
    ap.add_argument("--validate", action="store_true",
                    help="run config validation + preflight only (for checking "
                         "a config on a dev machine before deploying)")
    ap.add_argument("--stages", default=None,
                    help="comma-separated subset of "
                         "preflight,data,stage_a,stage_b,eval,report")
    ap.add_argument("--set", dest="sets", action="append", default=[],
                    help="config override dotted.path=value (repeatable)")
    ap.add_argument("--allow-config-change", action="store_true",
                    help="continue even if the config changed since the run "
                         "started (otherwise a changed config refuses)")
    ap.add_argument("--force-tests", action="store_true",
                    help="re-run the test suites even with a passing record")
    ap.add_argument("--offline", action="store_true",
                    help="forbid network in the data stage (cached corpora only)")
    args = ap.parse_args()
    _install()

    cfg_path = args.config or ("config.yaml" if os.path.exists("config.yaml") else None)
    if cfg_path is None:
        ap.error("no --config given and no ./config.yaml found")
    try:
        cfg = load_run_config(cfg_path)
        if args.sets:
            apply_set_overrides(cfg, args.sets)
        print(f"[config] {cfg_path}")
        check_config(cfg)
    except Exception as e:  # noqa: BLE001
        print(f"[config] INVALID:\n  {e}")
        return 1

    run_dir = os.path.abspath(os.path.join(cfg.output_dir, cfg.run_name))
    paths = default_paths(cfg, run_dir)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.source.yaml"), "w",
              encoding="utf-8") as f:
        f.write(open(cfg_path, encoding="utf-8").read())
    with open(os.path.join(run_dir, "config.resolved.yaml"), "w",
              encoding="utf-8") as f:
        f.write(dump_config(cfg))
    print(f"[run] {cfg.run_name} -> {run_dir}")

    sel = selected_stages(cfg, args.stages.split(",") if args.stages else None)

    tracker = None
    if not args.validate:
        from .tracker import Tracker
        from .orchestrator import load_state
        state0 = load_state(run_dir)
        tracker = Tracker(cfg.tracking, run_dir, cfg.run_name,
                          config_dict=json.loads(json.dumps(asdict(cfg))),
                          resume_id=state0.get("wandb_run_id"))
        if tracker.run_id and not state0.get("wandb_run_id"):
            state0["wandb_run_id"] = tracker.run_id
            from .orchestrator import save_state
            save_state(run_dir, state0)
    try:
        code = run_pipeline(cfg, paths, run_dir, sel, tracker=tracker,
                            validate_only=args.validate,
                            force_tests=args.force_tests,
                            offline=args.offline,
                            allow_config_change=args.allow_config_change)
    finally:
        if tracker is not None:
            try:
                tracker.finish(exit_code=0)
            except Exception:  # noqa: BLE001
                pass
    if STOP["count"] and code == 0:
        return 128 + int(STOP["signum"] or 2)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
