"""Run configuration: one YAML file fully describes a training run.

The training device may have no interactive terminal, so EVERYTHING is settled
before deployment: `python -m vertex500m.pipeline --config run.yaml` takes no
other input (besides optional `--stages` / `--set` / `--validate`). Unknown
YAML keys are an error (a typo'd hyperparameter must never silently fall back
to a default), and the resolved config's hash is recorded in the run dir -- a
resumed run with a changed config refuses to continue unless passed
`--allow-config-change`.

Schema (all sections optional except where noted; defaults shown):

    run_name: my-run            # required; becomes runs/<run_name>/
    output_dir: runs
    data:
      dir: null                 # default <run>/data
      corpus_dir: null          # default <run>/corpus
      token_map: null           # default <run>/token_map.json
      name: web                 # shard stem: {name}_{train|val}_*.bin
      tokenizer: JamesQuartz/qt-VI.1.3-32k
      val_fraction: 0.005
      seed: 0
      tokens_per_shard: 50000000
      pack: fixed               # fixed | docs
      download: []              # [{path, split, name, config, text_field,
                                #   limit_docs, revision}]
      local: []                 # local .txt/.jsonl inputs
      synthetic_tokens: null    # set -> generate toy corpus, skip download+tokenize
    model:
      hc_variant: mhc
      tiny: false
    train: {amp, seed, batch_size, seq_len, loss_chunk, accum_steps, grad_clip,
      muon_lr, adamw_lr, mtp_lr, embed_lr, weight_decay, embed_optimizer,
      warmup, quant_mode, compile, no_ckpt, save_every, keep_last, eval_every,
      eval_batches, target_loss, deadline_s, stage_a_steps, stage_b_steps}
    eval: {batches, seq_len, with_dspark}
    tracking: {mode: online|offline|disabled, project, entity, tags}
    pipeline: {run_tests: true, stages: null (=all), offline: false}
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields


class ConfigError(ValueError):
    """A run config problem. Raised with ALL problems found, not just the first."""


@dataclass
class DataConfig:
    dir: str | None = None
    corpus_dir: str | None = None
    token_map: str | None = None
    name: str = "web"
    tokenizer: str = "JamesQuartz/qt-VI.1.3-32k"
    val_fraction: float = 0.005
    seed: int = 0
    tokens_per_shard: int = 50_000_000
    pack: str = "fixed"
    download: list = field(default_factory=list)
    local: list = field(default_factory=list)
    synthetic_tokens: int | None = None


@dataclass
class ModelConfig:
    hc_variant: str = "mhc"
    tiny: bool = False


@dataclass
class TrainConfig:
    amp: str = "fp16"
    seed: int = 0
    device: str = "auto"            # auto | cpu | cuda
    batch_size: int = 4
    seq_len: int = 8192
    loss_chunk: int = 1024
    accum_steps: int = 1
    grad_clip: float | None = 1.0
    muon_lr: float = 0.02
    adamw_lr: float = 2.6e-4
    mtp_lr: float = 0.02
    embed_lr: float = 0.02
    weight_decay: float = 0.1
    embed_optimizer: str = "adamw"
    dist_nproc: int = 1
    warmup: int = 2000
    quant_mode: str = "FAKE"
    compile: bool = False
    no_ckpt: bool = False
    log_every: int = 20
    save_every: int = 250
    keep_last: int = 3
    eval_every: int = 250
    eval_batches: int | None = 8
    target_loss: float | None = None
    deadline_s: float | None = None
    allow_identity_map: bool = False
    stage_a_steps: int = 3000
    stage_b_steps: int = 400


@dataclass
class EvalConfig:
    batches: int | None = 16
    seq_len: int = 2048
    batch_size: int = 1
    split: str = "val"
    with_dspark: bool = True


@dataclass
class TrackingConfig:
    mode: str = "online"
    project: str = "vertex500m"
    entity: str | None = None
    tags: list = field(default_factory=list)


@dataclass
class PipelineConfig:
    run_tests: bool = True
    stages: list | None = None
    offline: bool = False


@dataclass
class RunConfig:
    run_name: str = ""
    output_dir: str = "runs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


STAGES = ("preflight", "data", "stage_a", "stage_b", "eval", "report")


def _strict(cls, d: dict, where: str, problems: list) -> dict:
    """Reject unknown keys (typo'd hyperparameters must never silently default)."""
    if not isinstance(d, dict):
        problems.append(f"{where}: expected a mapping, got {type(d).__name__}")
        return {}
    known = {f.name for f in fields(cls)}
    for k in d:
        if k not in known:
            problems.append(
                f"{where}.{k}: unknown key (expected one of {sorted(known)})")
    return {k: v for k, v in d.items() if k in known}


def _subsection(cls, d: dict, where: str, problems: list):
    sub = d.get(where, {})
    if sub is None:
        sub = {}
    return cls(**_strict(cls, sub, where, problems))


def load_run_config(path: str) -> RunConfig:
    """Parse a YAML file into a RunConfig (validation happens in `validate`)."""
    import yaml
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    problems: list[str] = []
    top = _strict(RunConfig, raw, "config", problems)
    if problems:
        raise ConfigError("\n".join(problems))
    cfg = RunConfig(
        run_name=raw.get("run_name", ""),
        output_dir=raw.get("output_dir", "runs"),
        data=_subsection(DataConfig, raw, "data", problems),
        model=_subsection(ModelConfig, raw, "model", problems),
        train=_subsection(TrainConfig, raw, "train", problems),
        eval=_subsection(EvalConfig, raw, "eval", problems),
        tracking=_subsection(TrackingConfig, raw, "tracking", problems),
        pipeline=_subsection(PipelineConfig, raw, "pipeline", problems),
    )
    if problems:
        raise ConfigError("\n".join(problems))
    # YAML 1.1 parses unquoted `off` as boolean False -- accept it for amp.
    if cfg.train.amp is False:
        cfg.train.amp = "off"
    return cfg


def apply_set_overrides(cfg: RunConfig, overrides: list[str]) -> RunConfig:
    """Apply `--set a.b.c=value` overrides (values parsed as YAML)."""
    import yaml
    for ov in overrides:
        if "=" not in ov:
            raise ConfigError(f"--set {ov!r}: expected dotted.path=value")
        dotted, raw_val = ov.split("=", 1)
        try:
            val = yaml.safe_load(raw_val)
        except Exception as e:  # noqa: BLE001
            raise ConfigError(f"--set {ov!r}: bad value: {e}")
        parts = dotted.split(".")
        obj = cfg
        for p in parts[:-1]:
            if not hasattr(obj, p):
                raise ConfigError(f"--set {ov!r}: unknown path {dotted!r}")
            obj = getattr(obj, p)
        if not hasattr(obj, parts[-1]):
            raise ConfigError(f"--set {ov!r}: unknown path {dotted!r}")
        setattr(obj, parts[-1], val)
    return cfg


def validate(cfg: RunConfig) -> list[str]:
    """All config problems (errors AND warnings as ("error"/"warn", msg))."""
    out: list[tuple[str, str]] = []

    def err(m):
        out.append(("error", m))

    def warn(m):
        out.append(("warn", m))

    if not cfg.run_name or "/" in cfg.run_name or "\\" in cfg.run_name:
        err("run_name must be a non-empty single path component")
    t, d, m = cfg.train, cfg.data, cfg.model
    if t.amp not in ("off", "fp16", "bf16"):
        err(f"train.amp must be off|fp16|bf16, got {t.amp!r}")
    if t.device not in ("auto", "cpu", "cuda"):
        err(f"train.device must be auto|cpu|cuda, got {t.device!r}")
    if t.device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                err("train.device=cuda but no CUDA device is available")
        except ImportError:
            err("train.device=cuda but torch is not installed")
    if t.batch_size < 1:
        err(f"train.batch_size must be >= 1, got {t.batch_size}")
    if t.seq_len < 8:
        err(f"train.seq_len must be >= 8, got {t.seq_len}")
    try:
        from ..config import PRESET
        if t.seq_len > PRESET.max_seq_len:
            err(f"train.seq_len {t.seq_len} exceeds model max_seq_len "
                f"{PRESET.max_seq_len}")
    except ImportError:
        pass
    if t.loss_chunk < 1:
        err("train.loss_chunk must be >= 1")
    if t.accum_steps < 1:
        err("train.accum_steps must be >= 1")
    if t.grad_clip is not None and t.grad_clip <= 0:
        err("train.grad_clip must be positive")
    if t.stage_a_steps < 1:
        err("train.stage_a_steps must be >= 1")
    if t.stage_b_steps < 1:
        err("train.stage_b_steps must be >= 1")
    if t.warmup < 0:
        err("train.warmup must be >= 0")
    if t.warmup >= t.stage_a_steps:
        warn(f"train.warmup ({t.warmup}) >= stage_a_steps ({t.stage_a_steps}): "
             f"the run never leaves warmup; the cosine never engages")
    for lr in ("muon_lr", "adamw_lr", "mtp_lr", "embed_lr"):
        if getattr(t, lr) <= 0:
            err(f"train.{lr} must be positive")
    if t.weight_decay < 0:
        err("train.weight_decay must be >= 0")
    if t.embed_optimizer not in ("adamw", "sinkhorn"):
        err(f"train.embed_optimizer must be adamw|sinkhorn, got {t.embed_optimizer!r}")
    if t.dist_nproc < 1:
        err(f"train.dist_nproc must be >= 1, got {t.dist_nproc}")
    if t.quant_mode not in ("FAKE", "NONE"):
        err(f"train.quant_mode must be FAKE|NONE, got {t.quant_mode!r}")
    if t.save_every < 0 or t.keep_last < 0 or t.eval_every < 0:
        err("train.save_every/keep_last/eval_every must be >= 0")
    if t.log_every < 1:
        err("train.log_every must be >= 1")
    if t.eval_batches is not None and t.eval_batches < 1:
        err("train.eval_batches must be positive or null (full)")
    if t.target_loss is not None and t.target_loss <= 0:
        err("train.target_loss must be positive")
    if t.deadline_s is not None and t.deadline_s < 0:
        err("train.deadline_s must be >= 0")
    if m.hc_variant not in ("mhc", "mhc-lite"):
        err(f"model.hc_variant must be mhc|mhc-lite, got {m.hc_variant!r}")
    if not 0.0 <= d.val_fraction < 1.0:
        err(f"data.val_fraction must be in [0, 1), got {d.val_fraction}")
    if d.pack not in ("fixed", "docs"):
        err(f"data.pack must be fixed|docs, got {d.pack!r}")
    if d.tokens_per_shard < t.seq_len + 1:
        err(f"data.tokens_per_shard ({d.tokens_per_shard}) must exceed "
            f"train.seq_len + 1 ({t.seq_len + 1}), else every shard is empty")
    if d.synthetic_tokens is not None:
        if d.synthetic_tokens < (t.seq_len + 1) * t.batch_size:
            err("data.synthetic_tokens too small for even one batch")
        if d.download or d.local:
            warn("data.synthetic_tokens is set: download/local sources are ignored")
    for i, src in enumerate(d.download or []):
        if not isinstance(src, dict) or not src.get("path"):
            err(f"data.download[{i}]: each source needs at least a 'path'")
        lim = (src.get("limit_docs") or 0) if isinstance(src, dict) else 0
        if lim is not None and lim < 0:
            err(f"data.download[{i}]: limit_docs must be >= 0")
    e = cfg.eval
    if e.batches is not None and e.batches < 1:
        err("eval.batches must be positive or null (full)")
    if e.seq_len < 8:
        err("eval.seq_len must be >= 8")
    if e.batch_size < 1:
        err("eval.batch_size must be >= 1")
    if not e.split:
        err("eval.split must be non-empty (use 'all' for every shard)")
    tr = cfg.tracking
    if tr.mode not in ("online", "offline", "disabled"):
        err(f"tracking.mode must be online|offline|disabled, got {tr.mode!r}")
    p = cfg.pipeline
    if p.stages:
        unknown = [s for s in p.stages if s not in STAGES]
        if unknown:
            err(f"pipeline.stages has unknown entries {unknown} "
                f"(known: {list(STAGES)})")
    if t.eval_every and not d.synthetic_tokens and d.val_fraction == 0 \
            and not d.local and not d.download:
        warn("train.eval_every is set but no val data is configured "
             "(val_fraction=0 and no sources): in-loop validation will skip")
    return out


def check_config(cfg: RunConfig) -> None:
    """Raise ConfigError listing every problem, or return silently."""
    problems = validate(cfg)
    errors = [m for lvl, m in problems if lvl == "error"]
    for lvl, m in problems:
        print(f"  [{'FAIL' if lvl == 'error' else 'WARN'}] {m}")
    if errors:
        raise ConfigError(f"{len(errors)} config error(s):\n"
                          + "\n".join(f"  - {m}" for m in errors))


def dump_config(cfg: RunConfig) -> str:
    """Canonical YAML dump (hashing + config.resolved.yaml)."""
    import yaml
    return yaml.safe_dump(json.loads(json.dumps(asdict(cfg))),
                          sort_keys=True, default_flow_style=False)


def config_hash(cfg: RunConfig) -> str:
    return hashlib.sha256(dump_config(cfg).encode("utf-8")).hexdigest()[:16]
