"""Weight and activation quantization (spec 14, 24).

QUANT_MODE:
  "NONE" -- skip quantization entirely. Architecture debugging only. Note spec
            invariant 35: NONE removes ONLY quantization. Sinks, rms_norm_eps=1e-20,
            the Sinkhorn iterations and the SwiGLU clamps all remain.
  "FAKE" -- round values through the low-bit grid but store bf16. Default. This is
            what the model is trained against, with a straight-through estimator.
  "REAL" -- not implemented here (deployment kernels dequantize on the fly).
"""

from __future__ import annotations

import torch

QUANT_MODE = "FAKE"  # "NONE" | "FAKE"

FP8_MAX = 448.0   # E4M3 max normal
FP4_MAX = 6.0     # E2M1 max
FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FP4_EDGES = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])  # RNE-to-even edges

# Per-device copies of the two lookup tables above. fp4_act_quant used to move them
# CPU->GPU on EVERY call (`.to(v.device)` per invocation, several times per layer
# per forward) for 7- and 8-element tensors -- pure launch + allocation overhead.
_FP4_CACHE: dict = {}


def _fp4_consts(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """(grid, edges) on `device`, moved once and reused."""
    key = torch.device(device)
    hit = _FP4_CACHE.get(key)
    if hit is None:
        hit = (FP4_GRID.to(key), FP4_EDGES.to(key).contiguous())
        _FP4_CACHE[key] = hit
    return hit


def _fast_log2_ceil(x: torch.Tensor) -> torch.Tensor:
    """ceil(log2(x)) by IEEE-754 bit manipulation. Exact, no transcendental."""
    b = x.view(torch.int32)
    e = ((b >> 23) & 0xFF) - 127
    m = b & ((1 << 23) - 1)
    return e + (m != 0).to(torch.int32)


def _fast_pow2(x: torch.Tensor) -> torch.Tensor:
    """2**x for integer x, constructed in the float32 exponent field."""
    return ((x + 127) << 23).view(torch.float32)


def act_quant(x: torch.Tensor, block: int = 32, round_scale: bool = True,
              inplace: bool = False):
    """Block-wise FP8 E4M3 fake-quantization with UE8M0 pow2 scales.

    Window KV path: block=32, round_scale=True, scale dtype UE8M0.

    `inplace` defaults to False. The released inference kernels quantize in place;
    a training implementation MUST NOT, because in-place mutation of an autograd
    input breaks backward (spec invariant 44).
    """
    if QUANT_MODE == "NONE":
        return (x, None)
    lead, N = x.shape[:-1], x.size(-1)
    M = 1
    for s in lead:
        M *= s
    assert N % block == 0, (
        f"act_quant: last dim {N} not divisible by block {block}. Every weight and "
        f"activation dim must be a multiple of the block size."
    )
    xb = x.reshape(M, N // block, block).float()
    amax = xb.abs().amax(-1).clamp_min(1e-4)
    s = (_fast_pow2(_fast_log2_ceil(amax / FP8_MAX)) if round_scale
         else amax / FP8_MAX)
    s = s.unsqueeze(-1)
    q = (xb / s).clamp(-FP8_MAX, FP8_MAX)
    q = q.to(torch.float8_e4m3fn).float() * s
    # straight-through estimator: forward uses quantized values, backward flows to x
    q = xb + (q - xb).detach()
    y = q.view(M, N).view(*lead, N).to(x.dtype)
    scales = s.view(M, N // block).to(torch.float8_e8m0fnu)
    if inplace:
        x.copy_(y)
    return y, scales


def fp4_act_quant(x: torch.Tensor, block: int = 32, e4m3_scale: bool = False,
                  inplace: bool = False):
    """Block-wise FP4 E2M1 fake-quantization.

    indexer q/k   : block=32, e4m3_scale=False  (UE8M0 pow2 scale, floor 6*2^-126)
    compressed KV : block=16, e4m3_scale=True   (E4M3 scale,    floor 6*2^-9)
    """
    if QUANT_MODE == "NONE":
        return (x, None)
    lead, N = x.shape[:-1], x.size(-1)
    M = 1
    for s_ in lead:
        M *= s_
    assert N % block == 0, f"fp4_act_quant: last dim {N} not divisible by block {block}"
    xb = x.reshape(M, N // block, block).float()
    amax = xb.abs().amax(-1)
    if e4m3_scale:
        amax = amax.clamp_min(6 * 2 ** -9)
        s = (amax / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(6 * 2 ** -126)
        s = _fast_pow2(_fast_log2_ceil(amax / FP4_MAX))
    s = s.unsqueeze(-1)
    v = (xb / s).clamp(-FP4_MAX, FP4_MAX).abs()
    # bucketize(right=False) -> index of the first edge strictly greater than v;
    # FP4_EDGES encodes round-half-to-even on the 8-value grid.
    grid, edges = _fp4_consts(v.device)
    idx = torch.bucketize(v.contiguous(), edges)
    q = grid[idx] * torch.sign(xb)
    q = xb + ((q * s) - xb).detach()  # STE
    y = q.view(M, N).view(*lead, N).to(x.dtype)
    scale_dt = torch.float8_e4m3fn if e4m3_scale else torch.float8_e8m0fnu
    return y, s.view(M, N // block).to(scale_dt)


def set_quant_mode(mode: str) -> None:
    global QUANT_MODE
    if mode not in ("NONE", "FAKE"):
        raise ValueError(f"QUANT_MODE must be NONE or FAKE, got {mode!r}")
    QUANT_MODE = mode
