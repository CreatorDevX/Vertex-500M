"""Stage A / stage B training. `python -m vertex500m.train` runs `train.__main__`.

Backward compatible: the names other modules import from `train` --
`chunked_ce`, `chunked_head_ce`, `optimizer_step`, `Dist`, `wrap_ddp`,
`unwrap`, `lr_at`, `set_lr`, `run` -- are all re-exported here, so
`from vertex500m.train import chunked_ce` keeps working unchanged.
"""

from .checkpoint import (
    capture_rng_state,
    ckpt_paths,
    resolve_ckpt_dir,
    restore_optimizer_state,
    restore_rng_state,
    save_checkpoint,
)
from .dist import Dist, seed_everything, setup_distributed, unwrap, wrap_ddp
from .losses import chunked_ce, chunked_head_ce
from .loop import backward_loss, finalize_step, finish_optimizer_step, optimizer_step, run
from .metrics import MetricsSink, read_metrics
from .schedule import lr_at, set_lr

__all__ = [
    "Dist",
    "MetricsSink",
    "backward_loss",
    "capture_rng_state",
    "chunked_ce",
    "chunked_head_ce",
    "ckpt_paths",
    "finalize_step",
    "finish_optimizer_step",
    "lr_at",
    "optimizer_step",
    "read_metrics",
    "resolve_ckpt_dir",
    "restore_optimizer_state",
    "restore_rng_state",
    "run",
    "save_checkpoint",
    "seed_everything",
    "set_lr",
    "setup_distributed",
    "unwrap",
    "wrap_ddp",
]
