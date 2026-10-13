"""Automated training pipeline: preflight -> data -> stage A -> stage B -> eval.

One command runs the whole chain; re-running it resumes (idempotent stages).
See run_config.py for the YAML schema.
"""

from .orchestrator import (
    STAGE_ORDER,
    default_paths,
    load_state,
    run_pipeline,
    save_state,
    selected_stages,
)
from .run_config import (
    ConfigError,
    RunConfig,
    apply_set_overrides,
    check_config,
    config_hash,
    dump_config,
    load_run_config,
)

__all__ = [
    "ConfigError",
    "RunConfig",
    "STAGE_ORDER",
    "apply_set_overrides",
    "check_config",
    "config_hash",
    "default_paths",
    "dump_config",
    "load_run_config",
    "load_state",
    "run_pipeline",
    "save_state",
    "selected_stages",
]
