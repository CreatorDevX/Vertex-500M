"""RMSNorm, partial RoPE and YaRN (spec 5.3, 9, 21.3)."""

from __future__ import annotations

import math

import torch


def rms_recip(x: torch.Tensor, eps: float) -> torch.Tensor:
    """`rsqrt(mean(x^2, dim=-1) + eps)`, reduced in fp32, shape (..., 1).

    The reduction has to be fp32 -- `rms_norm_eps` is 1e-20 (spec invariant 39), which is
    below fp16's smallest subnormal (5.96e-8), so an fp16 mean-of-squares that flushes
    to zero makes `rsqrt(eps)` ~ 1e10. But the INPUT to that reduction should not be
    materialised in fp32, and this is the standard AMP shape: reduce in fp32, apply in
    the activation dtype.

    Two obvious implementations are both wrong:
      * `x.float().square().mean(-1)` -- allocates a full fp32 copy of `x`.
      * `x.pow(2).mean(-1)` -- squares in the INPUT dtype, and fp16 overflows above
        |x| ~ 256 (sqrt(finfo(float16).max)), so this returns inf.
    `linalg.vector_norm(..., dtype=)` accumulates in fp32 without either.
    """
    n2 = torch.linalg.vector_norm(x, dim=-1, keepdim=True, dtype=torch.float32)
    return torch.rsqrt(n2.square() / x.size(-1) + eps)


class RMSNorm(torch.nn.Module):
    """Normalize in fp32, apply the scale in fp32, cast back.

    eps is literally `rms_norm_eps` and that value is 1e-20 by design, matching the
    released model. Do not "fix" it -- spec invariant 39.

    Under autocast the REDUCTION stays fp32 (see `rms_recip`) but the normalisation and
    the gain are applied in the activation dtype. That trades one extra rounding for the
    removal of two fp32 [b, s, d] temporaries on the hottest elementwise op in the
    model -- this runs 41 times per forward, so it is ~440 KB/token of traffic saved. In
    a pure fp32 run nothing changes: `dt` is fp32 and every cast is a no-op.
    """

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        return self.weight.to(dt) * (x * rms_recip(x, self.eps).to(dt))

    def extra_repr(self) -> str:
        return f"dim={self.weight.numel()}, eps={self.eps:g}"


def precompute_freqs_cis(dim: int, seqlen: int, original_seq_len: int, base: float,
                         factor: float, beta_fast: int, beta_slow: int):
    """Rotary frequencies as complex exponentials, one row per position.

    original_seq_len > 0 applies the YaRN ramp (spec 9).
    """
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        def corrected_dim(rotations: float) -> float:
            return (dim * math.log(original_seq_len / (rotations * 2 * math.pi))
                    / (2 * math.log(base)))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low)
                / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False):
    """Rotate the last dim's adjacent pairs. x: [b,s,d] or [b,s,h,d]. Pure."""
    f = freqs_cis.conj() if inverse else freqs_cis
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if xc.ndim == 3:
        f = f.view(1, xc.size(1), xc.size(-1))
    else:
        f = f.view(1, xc.size(1), 1, xc.size(-1))
    xc = torch.view_as_real(xc * f).flatten(-2)
    return xc.to(x.dtype)


def rope_tail(t: torch.Tensor, rd: int, freqs: torch.Tensor, inverse: bool = False):
    """Out-of-place partial RoPE: rotate only the trailing `rd` channels.

    The training path NEVER mutates its input. The released inference kernels do the
    rotation in place; a training implementation must not (spec invariant 44).
    """
    if rd <= 0:
        return t
    head, tail = t[..., :-rd], t[..., -rd:]
    return torch.cat([head, apply_rope(tail, freqs, inverse)], dim=-1)
