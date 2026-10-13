"""Distributed context: torchrun process group, or a single-process stand-in.

Phase-1 move (from train.py) plus the DDP-correctness work: `seed_everything`
gives every rank fully-seeded, rank-offset RNGs (the old code seeded only the
weight init, so every torchrun rank shuffled the data identically and DDP
averaged gradients over duplicated batches).
"""

from __future__ import annotations

import os
import random

import torch


def seed_everything(seed: int) -> None:
    """Seed every RNG stream torch/CUDA/numpy/python offer, deterministically.

    Call once per process with `seed + rank`. Makes same-seed runs
    bit-reproducible regardless of process history, and -- together with the
    DistributedSampler below -- stops DDP ranks from training on identical
    batches.
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2 ** 32))
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Dist:
    """Distributed context, or a single-process stand-in with the same surface.

    Rank 0 is the only rank that prints or saves; every rank runs the same number of
    steps so the process group never desynchronises at a collective.
    """

    def __init__(self, enabled=False, rank=0, world_size=1, local_rank=0):
        self.enabled, self.rank = enabled, rank
        self.world_size, self.local_rank = world_size, local_rank

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.enabled:
            torch.distributed.barrier()

    def cleanup(self) -> None:
        if self.enabled:
            torch.distributed.destroy_process_group()


def setup_distributed(args) -> Dist:
    """Join a torchrun process group if one is launched, else run single-process."""
    if getattr(args, "cpu", False) or os.environ.get("WORLD_SIZE", "1") == "1":
        return Dist()
    import torch.distributed as dist
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return Dist(True, dist.get_rank(), dist.get_world_size(), local_rank)


def wrap_ddp(model, dist_ctx: Dist, args):
    """DDP with `static_graph=True`, which this model needs and the default breaks.

    `embed.weight` is TIED: `VertexHead` is a view onto the same `nn.Embedding` that
    reads the input ids, so one Parameter is read in two places inside a single forward.
    That produces two autograd contributions. DDP's reducer fires its allreduce on the
    FIRST one it sees and then marks the parameter ready; the second contribution can
    arrive after the collective and be silently dropped, which trains a subtly wrong
    model that looks perfectly healthy. `static_graph=True` promises the graph does not
    change between iterations, which is true here, and lets the reducer accumulate every
    contribution before it fires. tests.py::test_ddp_tied_embedding_reduces_once pins
    this against a single-process reference.
    """
    if not dist_ctx.enabled:
        return model
    import torch.nn as nn
    device_ids = [dist_ctx.local_rank] if torch.cuda.is_available() else None
    return nn.parallel.DistributedDataParallel(
        model, device_ids=device_ids,
        # Stage B freezes the backbone, so every remaining trainable tensor is mtp.* and
        # every one of them is used. Turning this on would mask real wiring bugs.
        find_unused_parameters=False,
        static_graph=True,
    )


def unwrap(model):
    """The underlying Transformer, whether or not DDP wrapped it."""
    return model.module if hasattr(model, "module") else model
