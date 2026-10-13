"""Mixed precision: autocast dtype registry, GradScaler factory, cache dtype.

THREE RULES THIS MODULE ENFORCES
---------------------------------

1. WEIGHTS STAY fp32. Only COMPUTE runs in fp16/bf16, via autocast. Parameters,
   gradients and optimizer state are never cast. That is not a stylistic choice:

     * `optim.AdamW` is built with `eps=1e-20` (spec 19.2). In fp16 that is below the
       smallest subnormal (5.96e-8), so the update term collapses and the parameter
       stops moving. Muon's Newton-Schulz (optim.py:41) and SinkhornBalanced's 11 rounds
       of row/column equalization (optim.py:136-140) are both fp32-reasoning loops.
     * Checkpoints stay fp32 and load unchanged, so --amp never changes a file format.

2. DEVICES GIVE YOU DIFFERENT ANSWERS. On an NVIDIA T4 (sm_75, Turing) the tensor cores
   are FP16; BF16 tensor cores arrived with Ampere (sm_80). `torch.cuda.is_bf16_supported`
   returns true only for `major >= 8`. Running `--amp bf16` on a T4 therefore computes in
   bf16 at roughly FP32 throughput -- a ~0% FLOP win and doubled activation bytes, i.e.
   strictly worse than `--amp fp16`. `warn_if_unsupported` says so out loud rather than
   letting the user discover it as a mystery slowdown.

3. fp16 NEEDS A GradScaler, bf16 DOES NOT. bf16 has fp32's exponent range, so unscaled
   gradients cannot overflow. fp16's max is 65504 and this model's activations are large
   (see the indexer score bound in attention.py), so gradients underflow to zero without
   a loss scale. `make_scaler` returns None for off/bf16 and a real scaler for fp16.

The compute dtype is read at CALL time, not imported from here, because autocast's dtype
depends on the device_type that is enabled. That is why sparse_attn.py reads
`torch.get_autocast_dtype(...)` rather than importing AMP_MODE.
"""

from __future__ import annotations

import contextlib
import warnings

import torch

# Module-level mode, mirroring quant.QUANT_MODE. "off" | "fp16" | "bf16"
AMP_MODE = "off"

_DT = {"fp16": torch.float16, "bf16": torch.bfloat16}
_MODES = ("off", "fp16", "bf16")


# --------------------------------------------------------------------------- #
# Mode registry
# --------------------------------------------------------------------------- #
def set_amp_mode(mode: str) -> None:
    """Set the process-wide autocast mode. Raises on an unknown mode."""
    global AMP_MODE
    if mode not in _MODES:
        raise ValueError(f"AMP mode must be one of {_MODES}, got {mode!r}")
    AMP_MODE = mode


def get_amp_mode() -> str:
    return AMP_MODE


def resolve_dtype(mode: str | None = None) -> torch.dtype | None:
    """The autocast dtype for `mode`, or None when autocast is off."""
    m = get_amp_mode() if mode is None else mode
    return _DT.get(m)


def amp_enabled(mode: str | None = None) -> bool:
    return resolve_dtype(mode) is not None


# --------------------------------------------------------------------------- #
# Device-capability advice
# --------------------------------------------------------------------------- #
def warn_if_unsupported(mode: str | None = None, device_type: str = "cuda") -> None:
    """Warn when `mode` cannot actually deliver a speedup on this device.

    A warning, never an error: bf16 is *correct* everywhere, it is just not fast on
    pre-Ampere NVIDIA parts. Refusing to run would be the wrong call.
    """
    m = get_amp_mode() if mode is None else mode
    if m != "bf16" or device_type != "cuda" or not torch.cuda.is_available():
        return
    major, _ = torch.cuda.get_device_capability()
    if major < 8:
        warnings.warn(
            f"--amp bf16 on compute capability {major}.x (pre-Ampere, e.g. T4): BF16 "
            "tensor cores do not exist here, so this runs at roughly FP32 throughput "
            "with doubled activation bytes. Use --amp fp16 for a real speedup.",
            RuntimeWarning, stacklevel=2)


def needs_scaler(mode: str | None = None) -> bool:
    """Only fp16 needs loss scaling; bf16's exponent range is fp32's."""
    return (get_amp_mode() if mode is None else mode) == "fp16"


# --------------------------------------------------------------------------- #
# Contexts and the scaler
# --------------------------------------------------------------------------- #
def amp_context(mode: str | None = None, device_type: str = "cuda"):
    """`torch.autocast` for `mode`, or a no-op context when off.

    `device_type` defaults to cuda because that is the only device where this is worth
    anything; CPU autocast is supported but is a debug/CI aid, not a speedup.
    """
    dt = resolve_dtype(mode)
    if dt is None or not torch.amp.is_autocast_available(device_type):
        return contextlib.nullcontext()
    return torch.autocast(device_type=device_type, dtype=dt)


def make_scaler(mode: str | None = None, device_type: str = "cuda",
                init_scale: float = 65536.0, growth_interval: int = 2000):
    """A `torch.amp.GradScaler` for fp16, or None for off/bf16.

    Returns None rather than a disabled scaler so the training loop reads as
    `if scaler is not None` instead of branching on a flag nobody can see.
    """
    m = get_amp_mode() if mode is None else mode
    if m != "fp16":
        return None
    # The kwarg is `device`, not `device_type`, despite the type annotation saying str.
    return torch.amp.GradScaler(device=device_type, enabled=True,
                                init_scale=init_scale,
                                growth_interval=growth_interval)


# --------------------------------------------------------------------------- #
# Cache dtype
# --------------------------------------------------------------------------- #
def cache_dtype(device=None, mode: str | None = None) -> torch.dtype:
    """Dtype for the KV / compressed-KV / index caches.

    This is the single highest-leverage line in the AMP migration. The caches default to
    fp32, and `Attention.forward` concatenates them with the freshly computed fp16 KV
    (`attention.py`, `kv = torch.cat([kv, ckv], dim=1)`). An fp32 operand promotes the
    concatenation, so the gathered chunk is fp32 and the attention core -- 26% of this
    model's FLOPs -- never reaches the tensor cores, no matter what autocast says.

    Off -> fp32, exactly the pre-AMP behaviour. On -> the autocast dtype.
    """
    dt = resolve_dtype(mode)
    if dt is None:
        return torch.float32
    if device is None:
        return dt
    dev_type = device.type if isinstance(device, torch.device) else str(device)
    # autocast only lowers precision on devices where it is enabled; on anything else
    # (cpu in a CUDA-only run, a private-use accelerator) fall back to fp32.
    return dt if torch.amp.is_autocast_available(dev_type) else torch.float32


# --------------------------------------------------------------------------- #
# Introspection used by the custom autograd Function and by tests
# --------------------------------------------------------------------------- #
def autocast_dtype_for(t: torch.Tensor) -> torch.dtype | None:
    """The dtype autocast is using on `t`'s device, or None if not in a region."""
    dev = t.device.type
    if not torch.amp.is_autocast_available(dev) or not torch.is_autocast_enabled(dev):
        return None
    try:
        return torch.get_autocast_dtype(dev)
    except AttributeError:                       # torch < 2.4
        return torch.get_autocast_gpu_dtype() if dev == "cuda" else None


# Feature flag for the torch<2.4 autocast API (see current_autocast_dtype).
_HAS_GET_AUTOCAST_DTYPE = hasattr(torch, "get_autocast_dtype")


def current_autocast_dtype(device_type: str) -> torch.dtype | None:
    """The autocast dtype active on `device_type`, or None when outside a region.

    Traceable twin of `autocast_dtype_for` for compiled regions: it reads only
    `torch.is_autocast_enabled` / `torch.get_autocast_dtype`, never the
    untraceable `torch._C._is_autocast_available` builtin behind
    `torch.amp.is_autocast_available` (calling that inside compiled code breaks
    dynamo into extra graphs -- one per call site per layer, in practice).
    Identical results on cuda/cpu; on exotic devices without autocast support it
    may report a dtype where `autocast_dtype_for` reports None, so compiled
    regions (cuda/cpu only) use this and everything else keeps the strict read.
    """
    if not torch.is_autocast_enabled(device_type):
        return None
    if _HAS_GET_AUTOCAST_DTYPE:
        return torch.get_autocast_dtype(device_type)
    if device_type == "cuda":                   # torch < 2.4
        return torch.get_autocast_gpu_dtype()
    return None


def compute_dtype(q: torch.Tensor) -> torch.dtype:
    """What the sparse-attention kernel should accumulate in.

    Inside an autocast region: the autocast dtype. The einsums inside the kernel are
    autocast-eligible, so promoting to fp32 first only buys a wasteful
    fp16 -> fp32 -> fp16 round trip of BOTH `q` and the gathered `kv` on every chunk.

    Outside one: `promote_types(q.dtype, float32)`, which preserves the fp64 ground-truth
    path that `tests.py::test_sparse_attn_matches_reference` relies on.
    """
    ad = autocast_dtype_for(q)
    if ad is not None:
        return ad
    return torch.promote_types(q.dtype, torch.float32)


__all__ = ["AMP_MODE", "set_amp_mode", "get_amp_mode", "resolve_dtype", "amp_enabled",
           "warn_if_unsupported", "needs_scaler", "amp_context", "make_scaler",
           "cache_dtype", "autocast_dtype_for", "current_autocast_dtype",
           "compute_dtype"]