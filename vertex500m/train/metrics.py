"""Structured metrics: every tracked scalar also lands in a JSONL file.

Console prints vanish when the terminal does; `metrics.jsonl` in the checkpoint
directory is the durable record (and what the pipeline's report + wandb sync
read back). One JSON object per line:

    {"step": 250, "split": "train", "loss": 3.21, "lr": 0.019, ...}

`split` is "train" for step records and "val" for in-loop evaluation records.
The sink is dependency-free on purpose -- the pipeline's tracker (phase 5)
tails this file into wandb, so training itself never imports wandb.
"""

from __future__ import annotations

import json
import os


class MetricsSink:
    """Append-only JSONL metrics log. `path=None` disables file output."""

    def __init__(self, path: str | None = None):
        self.path = path
        self._fh = None
        if path:
            d = os.path.dirname(os.path.abspath(path))
            os.makedirs(d, exist_ok=True)
            self._fh = open(path, "a", encoding="utf-8")

    def log(self, record: dict) -> dict:
        """Append one record. Returns it (so callers can chain/print)."""
        if self._fh is not None:
            self._fh.write(json.dumps(record) + "\n")
            self._fh.flush()
        return record

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def read_metrics(path: str) -> list[dict]:
    """Read a metrics.jsonl back (report generation, tests)."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
