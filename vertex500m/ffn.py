"""Dense feed-forward sublayer (replaces spec 6 MoE entirely).

The released architecture routes each token to 1 shared + 6 of 384 experts. At 0.5B
that allocation is dubious: every expert's gradient is estimated from B/16 tokens per
step instead of B, which is the "sparse backward update" problem. The whole FFN is
therefore one clamped SwiGLU -- the shared expert promoted to the entire sublayer,
widened, and stripped of its router.

Nothing here is DeepSeek's; see `vertex500m-dense.md` 6.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class Expert(torch.nn.Module):
    """Clamped SwiGLU: two-sided clamp on up, upper-only on gate, fp32 product.

    Unchanged from spec 6.2. The clamp exists to keep activations inside the FP8/FP4
    dynamic range so the fake-quant round-trip does not clip. That argument is about
    activation magnitude, not routing, so it does not weaken when the FFN goes dense
    -- and at MoEI=3584 the FFN intermediate is the single largest activation tensor
    in the network, so it matters MORE, not less.
    """

    def __init__(self, dim: int, inter: int, limit: float):
        super().__init__()
        self.w1 = torch.nn.Linear(dim, inter, bias=False)   # gate
        self.w2 = torch.nn.Linear(inter, dim, bias=False)   # down
        self.w3 = torch.nn.Linear(dim, inter, bias=False)   # up
        self.limit = limit

    def forward(self, x: torch.Tensor, weights: torch.Tensor | None = None):
        # No explicit `.float()` here. Under autocast `w1`/`w3` already return fp16, so
        # the old `.float()` promoted the two largest activations in the FFN -- MoEI =
        # 3584 channels, twice, on every one of the 20 layers -- straight back out of the
        # tensor-core dtype for arithmetic that has no reason to leave it. That is
        # 860 KB/token of traffic to prepare a silu-multiply.
        #
        # The clamp is what makes fp16 safe here: `swiglu_limit` is 10.0, so `up` is
        # bounded to +-10 and `silu(gate) * up` to about +-10 -- three orders of
        # magnitude inside fp16's 65504. The clamp's stated purpose is keeping
        # activations inside the FP8/FP4 dynamic range (E4M3 max 448), which fp16
        # exceeds comfortably anyway; it is doing the same job either way.
        gate, up = self.w1(x), self.w3(x)
        if self.limit > 0:
            up = up.clamp(-self.limit, self.limit)
            gate = gate.clamp(max=self.limit)
        y = F.silu(gate) * up
        if weights is not None:            # dead in the dense model; kept for parity
            y = weights * y
        return self.w2(y.to(x.dtype))


class DenseFFN(torch.nn.Module):
    """The entire feed-forward sublayer: one SwiGLU, no router, no experts.

    Backbone and DSpark stages use the identical class. What used to distinguish the
    drafter's FFN (`dspark_n_routed_experts`) is now purely which Attention subclass
    the block carries and which compress_ratios entry it has.
    """

    def __init__(self, cfg, layer_id: int | None = None):
        super().__init__()
        # layer_id is accepted and ignored so Block.__init__ needs no DSpark special case.
        self.expert = Expert(cfg.dim, cfg.moe_intermediate_size, cfg.swiglu_limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expert(x)
