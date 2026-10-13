"""Muon optimizer for 2-D weights (spec 19.2).

Muon is a momentum-based optimizer whose update is orthogonalized via Newton-Schulz,
then rescaled so the update has a fixed RMS. It is used for all linear weights in the LM
backbone, Engram projections, and (when enabled) Q/K.

The spec's allocation:
  * Muon              -> LM backbone linears, Engram projections, the hc_* projections
                        (hc_attn_fn / hc_ffn_fn are 2-D [mix_hc, hc*d], so Muon, not AdamW)
  * AdamW             -> norm weights, attn_sink, hc_*_base / hc_*_scale, Engram
                        q_weight / k_weight (all 1-D; eps=1e-20, betas 0.9/0.95)
  * momentum+Sinkhorn -> embedding, LM head, Engram tables AND their fp8 row scales

The tied-embedding caveat (spec 19.2, vertex500m-dense 11.3): `embed.weight` is BOTH the
input embedding and the LM head, so it receives gradient from two sources. The released
momentum+Sinkhorn-balanced algorithm assumes a table that is *either* an embedding *or* a
head. This implementation therefore defaults `embed.weight` to plain AdamW, not the
Sinkhorn-balanced optimizer, because running the released algorithm on a tied tensor is
untested (vertex500m-dense 20.5). Pass embed_optimizer='sinkhorn' to opt in.

Head-wise Muon (a per-head preconditioner for Q and K weights) is in the spec but not
implemented here; the orthogonalization is already applied per matrix.
"""

from __future__ import annotations

import torch


def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate the orthogonal matrix factor via 5 Newton-Schulz iterations.

    This is the standard Muon iteration. The coefficient sequence
    (3.4445, -4.7750, 2.0315) is a contraction, so reduced precision is enough
    for the iteration to converge to the same orthogonal factor that fp32 would
    reach, and the factor matmuls stay off the fp32 path -- which is where Muon's
    speed comes from. The returned tensor is cast back to the input dtype.

    The working dtype is fp16 on CUDA, bf16 on CPU. The training target (T4/sm_75)
    has FP16 tensor cores but NO bf16 ones, so bf16 matmuls run at ~fp32 speed
    there while fp16 hits the tensor cores -- and fp16 carries more mantissa
    (10 vs 7 bits), so it is also the more accurate choice. On Ampere+ both hit
    tensor cores at the same rate and fp16 stays more accurate. On CPU, bf16 keeps
    the historical behaviour (CPU fp16 matmuls are emulated and slow).

    Normalization happens in the INPUT dtype before the cast: casting a large
    update to fp16 first could overflow fp16's 65504 max, while bf16's fp32-grade
    exponent range never had that exposure.
    """
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    low = torch.float16 if G.is_cuda else torch.bfloat16
    X = G / (G.norm() + 1e-7)
    X = X.to(low)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Momentum + Newton-Schulz orthogonalization, update rescaled to a fixed RMS.

    update_RMS controls the size of the update; it is what makes Muon's LR roughly
    scale-free and reusable with the AdamW LR (spec: "to reuse the AdamW LR").
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 weight_decay: float = 0.1, update_rms: float = 0.18,
                 nesterov: bool = True):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        update_rms=update_rms, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            wd = group["weight_decay"]
            rms = group["update_rms"]
            nesterov = group["nesterov"]
            # The elementwise bookkeeping (momentum, weight decay, final add)
            # runs as foreach ops: one kernel per group instead of one per
            # parameter. Newton-Schulz itself stays per matrix (shapes differ, and
            # its matmuls dominate anyway); only the surrounding elementwise work
            # is batched. Semantics are identical to the per-parameter loop: buf
            # is mutated in place, `update` is always a fresh tensor.
            params, grads, bufs = [], [], []
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                params.append(p)
                grads.append(p.grad)
                bufs.append(state["momentum_buffer"])
            if not params:
                continue
            torch._foreach_mul_(bufs, mom)
            torch._foreach_add_(bufs, grads)
            updates = (torch._foreach_add(grads, bufs, alpha=mom) if nesterov
                       else list(bufs))
            scaled = []
            for u in updates:
                if u.ndim == 2:
                    u = _zeropower_via_newtonschulz5(u)
                # rescale to the target update RMS
                u = u * (rms / (u.norm() + 1e-8))
                scaled.append(u)
            if wd:
                torch._foreach_mul_(params, 1 - lr * wd)
            torch._foreach_add_(params, scaled, alpha=-lr)
        return loss


class SinkhornBalanced(torch.optim.Optimizer):
    """Momentum + Nesterov + row/column Sinkhorn equalization for embedding-like tables.

    Used for the Engram n-gram table (and optionally the tied embedding). Row/column
    equalization redistributes update mass so every row/column of a table sees a
    balanced signal, which empirically beats Adam on sparse embedding tables.

    NOTE: applying this to a TIED embedding (simultaneously input embedding and LM head)
    is untested. Default is plain AdamW for embed.weight; opt in explicitly.
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.9,
                 weight_decay: float = 0.0, n_k: int = 11, tau: float = 1e-3,
                 gamma: float = 0.18, eps: float = 1e-20):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        n_k=n_k, tau=tau, gamma=gamma, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            wd = group["weight_decay"]
            n_k, tau, gamma = group["n_k"], group["tau"], group["gamma"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(g)
                update = g.add(buf, alpha=mom)          # Nesterov
                # K rounds of row/column Sinkhorn balancing on the update
                if p.ndim == 2:
                    for _ in range(n_k):
                        update = update - (update.mean(-1, keepdim=True))
                        update = update / (update.abs().mean(-1, keepdim=True) + tau)
                        update = update - update.mean(-2, keepdim=True)
                        update = update / (update.abs().mean(-2, keepdim=True) + tau)
                    # gamma correction sets the update magnitude
                    update = update * (gamma / (update.norm() + group["eps"]))
                update = update.to(p.dtype)
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)
        return loss


def build_param_groups(model, muon_lr: float = 0.02, adamw_lr: float = 2.6e-4,
                       embed_lr: float = 0.02, engram_lr_mult: float = 5.0,
                       embed_optimizer: str = "adamw",
                       weight_decay: float = 0.1, engram_table_optimizer: str = "sinkhorn"):
    """Partition parameters into Muon / AdamW / table groups per spec 19.2.

    embed_optimizer: 'adamw' (default, safe for tied) or 'sinkhorn' (released algorithm,
        untested on a tied tensor -- vertex500m-dense 20.5).
    engram_table_optimizer: 'sinkhorn' (default) or 'adamw'.
    """
    muon_params, adamw_params, embed_params, table_params = [], [], [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Engram n-gram tables (the huge sparse tables) -> table optimizer.
        # `embed.scale` is the fp8 row scale of that SAME table: it is indexed by the
        # same `idx` and multiplied into it in ParallelEngramEmbedding.forward, so it
        # must move with the table at the table's LR. It is deliberately NOT routed to
        # Muon: it is [rows, dim/block], i.e. rank-2, and Newton-Schulz would
        # orthogonalize it into something meaningless.
        if "engram.embed" in name:
            table_params.append(p)
        elif name == "embed.weight":
            # The tied [V, d] table. `VertexHead` stores the Embedding via
            # `object.__setattr__` and exposes it as a property, so there is no
            # `head.weight` in the parameter tree -- do not add one to match the spec's
            # ParallelHead, that would un-tie the model.
            embed_params.append(p)
        elif p.dim() >= 2:
            muon_params.append(p)
        else:
            # 1-D only: RMSNorm gains, attn_sink, hc_*_base/hc_*_scale, Engram
            # q_weight/k_weight. Note hc_attn_fn / hc_ffn_fn are [mix_hc, hc*d] and
            # therefore 2-D, so they go to Muon above, NOT here.
            adamw_params.append(p)

    groups = []
    # `lr_mult` is the group's LR relative to the scheduler's base, so a schedule can
    # scale every group by one scalar WITHOUT collapsing the partition onto a single
    # LR. Without it, `g["lr"] = lr` in a training loop silently hands AdamW's
    # norm/sink/hc parameters (built for 2.6e-4) the Muon LR of 0.02.
    if muon_params:
        groups.append({"params": muon_params, "use_muon": True, "lr": muon_lr,
                       "lr_mult": 1.0, "weight_decay": weight_decay})
    if adamw_params:
        groups.append({"params": adamw_params, "use_muon": False, "lr": adamw_lr,
                       "lr_mult": adamw_lr / muon_lr, "weight_decay": weight_decay})
    if embed_params:
        use_sk = embed_optimizer == "sinkhorn"
        groups.append({"params": embed_params, "use_muon": False,
                       "use_sinkhorn": use_sk, "lr": embed_lr,
                       "lr_mult": embed_lr / muon_lr,
                       "weight_decay": 0.0 if use_sk else weight_decay})
    if table_params:
        use_sk = engram_table_optimizer == "sinkhorn"
        groups.append({"params": table_params, "use_muon": False,
                       "use_sinkhorn": use_sk, "lr": embed_lr * engram_lr_mult,
                       "lr_mult": embed_lr * engram_lr_mult / muon_lr,
                       "weight_decay": 0.0 if use_sk else weight_decay})
    return groups


def build_optimizer(model, muon_lr: float = 0.02, adamw_lr: float = 2.6e-4,
                    embed_lr: float = 0.02, engram_lr_mult: float = 5.0,
                    embed_optimizer: str = "adamw", weight_decay: float = 0.1,
                    engram_table_optimizer: str = "sinkhorn",
                    momentum: float = 0.95, adam_betas=(0.9, 0.95),
                    adam_eps: float = 1e-20, update_rms: float = 0.18):
    """Construct the optimizer from spec 19.2's parameter allocation."""
    groups = build_param_groups(model, muon_lr, adamw_lr, embed_lr, engram_lr_mult,
                                embed_optimizer, weight_decay, engram_table_optimizer)
    muon_groups, adam_groups = [], []
    for g in groups:
        if g["use_muon"]:
            muon_groups.append(g)
        else:
            adam_groups.append(g)

    optims = []
    if muon_groups:
        optims.append(Muon(muon_groups, lr=muon_lr, momentum=momentum,
                           weight_decay=weight_decay, update_rms=update_rms))
    adam_groups = adam_groups or []
    # split adam further by whether it uses sinkhorn
    plain = [g for g in adam_groups if not g.get("use_sinkhorn")]
    sink = [g for g in adam_groups if g.get("use_sinkhorn")]
    if plain:
        optims.append(torch.optim.AdamW(plain, lr=adamw_lr, betas=adam_betas,
                                        eps=adam_eps, weight_decay=weight_decay))
    if sink:
        optims.append(SinkhornBalanced(sink, lr=embed_lr, weight_decay=0.0))
    return optims
