"""Experiment tracking: wandb, fed by tailing metrics.jsonl.

Training subprocesses stay wandb-free (no API keys in children, no import
cost per stage, no resume races): the orchestrator tails each stage's
metrics.jsonl and logs new records into ONE wandb run. Stage B keys are
prefixed `dspark/` (its loss is a different objective with its own step
counter -- sharing the step axis would overwrite stage-A rows).

Modes: `online` (needs auth -- verified in preflight, never prompted),
`offline` (writes runs/<name>/wandb for a later `wandb sync`), `disabled`
(everything is a no-op; wandb is never even imported, so the import cost and
any network touch are gone).
"""

from __future__ import annotations

import json
import os
import threading
import time


class _Tailer(threading.Thread):
    """Poll a metrics.jsonl from its EOF-at-start, wandb.log new records."""

    def __init__(self, log_fn, path: str, key_prefix: str = "", interval: float = 2.0):
        super().__init__(daemon=True)
        self._log = log_fn
        self.path = path
        self.prefix = key_prefix
        self.interval = interval
        # NOTE: must NOT be named `_stop` -- threading.Thread already uses
        # that name internally and join() calls it.
        self._stop_event = threading.Event()
        self._last_step = 0

    def stop(self):
        self._stop_event.set()

    def run(self):
        try:
            while not os.path.exists(self.path) and not self._stop_event.is_set():
                time.sleep(self.interval)
            if self._stop_event.is_set():
                return
            with open(self.path, "r", encoding="utf-8") as f:
                f.seek(0, os.SEEK_END)          # only this invocation's rows
                while not self._stop_event.is_set():
                    line = f.readline()
                    if not line:
                        time.sleep(self.interval)
                        continue
                    try:
                        rec = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if isinstance(rec, dict):
                        self._emit(rec)
        except Exception as e:  # noqa: BLE001
            print(f"[tracker] tailer for {self.path} died: {e}", flush=True)

    def _emit(self, rec: dict):
        step = rec.get("step", self._last_step)
        try:
            step = int(step)
        except (TypeError, ValueError):
            step = self._last_step
        self._last_step = max(self._last_step, step)
        out = {}
        for k, v in rec.items():
            if k in ("step",):
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[self.prefix + k] = v
            elif k in ("event", "stage", "split", "status"):
                out[self.prefix + k] = str(v)
        if out:
            self._log(out, step=step)


class Tracker:
    """One wandb run for the whole pipeline. No-op when disabled."""

    def __init__(self, tracking_cfg, run_dir: str, run_name: str,
                 config_dict: dict | None = None, resume_id: str | None = None):
        self.mode = tracking_cfg.mode
        self.run_id = resume_id
        self._run = None
        self._tailers: list[_Tailer] = []
        if self.mode == "disabled":
            return
        try:
            import wandb
        except ImportError:
            raise SystemExit("[tracker] tracking.mode is "
                             f"{self.mode!r} but wandb is not installed "
                             f"(pip install wandb, or set mode: disabled)")
        os.environ.setdefault("WANDB_SILENT", "1")
        self._wandb = wandb
        init_kw: dict = dict(
            project=tracking_cfg.project,
            name=run_name,
            tags=list(tracking_cfg.tags or []),
            dir=os.path.join(run_dir, "wandb"),
            mode="online" if self.mode == "online" else "offline",
            resume="allow",
            settings=wandb.Settings(silent=True),
        )
        if tracking_cfg.entity:
            init_kw["entity"] = tracking_cfg.entity
        if resume_id:
            init_kw["id"] = resume_id
        if config_dict:
            init_kw["config"] = config_dict
        self._run = wandb.init(**init_kw)
        self.run_id = self._run.id
        print(f"[tracker] wandb run {self.run_id} ({self.mode})", flush=True)

    def tail_file(self, path: str, key_prefix: str = "") -> None:
        """Start tailing a metrics.jsonl into this run (no-op when disabled)."""
        if self._run is None:
            return
        t = _Tailer(self._run.log, path, key_prefix)
        t.start()
        self._tailers.append(t)

    def log(self, record: dict, step: int | None = None) -> None:
        if self._run is not None:
            self._run.log(record, step=step)

    def summary(self, record: dict) -> None:
        if self._run is not None:
            for k, v in record.items():
                self._run.summary[k] = v

    def finish(self, exit_code: int = 0) -> None:
        for t in self._tailers:
            t.stop()
        for t in self._tailers:
            t.join(timeout=10)
        self._tailers = []
        if self._run is not None:
            self._run.finish(exit_code=exit_code)
            self._run = None
        if getattr(self, "_wandb", None) is not None:
            # stop the backend service too: otherwise it lingers (holding its
            # log files open -- on Windows that blocks run-dir cleanup) until
            # process exit.
            try:
                tear = getattr(self._wandb, "teardown", None)
                if tear is not None:
                    tear()
            except Exception:  # noqa: BLE001
                pass
            self._wandb = None
