"""Shared model construction + checkpoint loading.

`train.py`, `eval.py`, `bench.py` and `test_checkpoint.py` each grew their own
copy of "build layout, build hashes, init weights, load state dict, check the
hc variant". Four copies of the wiring means four places to get the attach /
load order wrong (eval.py documents one such bug it already hit: attaching
DSpark AFTER loading silently throws the trained drafter away). This module is
the single copy.

Two entry points:

* `load_token_map_json(path)` -- the token map JSON `tokenize.py --save-map`
  writes, as `(token_map, compressed_vocab_size, collisions)`.
* `build_model(...)` -- construct + optionally load a checkpoint.
* `load_checkpoint_into(model, path, ...)` -- the weight-loading half, for
  callers (the training loop) that must interleave their own steps between
  attach and load.
"""

from __future__ import annotations

import torch

from .engram import EngramLayout, NgramHashState, identity_token_map
from .init_weights import init_from_scratch


def load_token_map_json(path):
    """Read a `--save-map` JSON file. Returns (token_map, cvs, collisions)."""
    import json
    d = json.loads(open(path, encoding="utf-8").read())
    return d["token_map"], d["engram_compressed_vocab_size"], d.get("collisions", 0)


def check_hc_variant(path, saved_variant, hc_variant) -> None:
    """Fail fast when a checkpoint was trained under the other hc variant.

    A checkpoint saved with a different hc variant has a DIFFERENT projection
    width (24 vs 32 at hc=4), so load_state_dict raises a wall of per-tensor size
    mismatches. Check the recorded variant FIRST and say what actually went wrong:
    the two weight spaces are not related by any invertible map -- mHC-lite is a
    reparameterization of H_res, not a conversion (mHC-lite.md 1) -- so there is
    no bridge. Retrain, or pass the recorded variant.
    """
    if saved_variant and saved_variant != hc_variant:
        raise SystemExit(
            f"checkpoint {path} was trained with hc_variant={saved_variant!r} "
            f"but --hc-variant is {hc_variant!r}. The two weight spaces are not "
            f"related by any invertible map -- mHC-lite is a reparameterization of "
            f"H_res, not a conversion (mHC-lite.md 1), so there is no bridge. "
            f"Retrain, or pass --hc-variant {saved_variant}.")


def load_checkpoint_into(model, path, hc_variant: str = "mhc", *,
                         with_dspark: bool = False, say=None):
    """Load `path`'s weights into `model`. Returns (ckpt_obj, missing, unexpected).

    When `with_dspark` is set, DSpark is attached BEFORE loading, not after: a
    stage-B checkpoint carries mtp.* weights, and attaching afterwards makes every
    one of them "unexpected", so the drafter you trained is silently thrown away
    and acceptance is measured on random weights (eval.py documented this bug).

    `say` is the print function (rank-gated in distributed training).
    """
    say = say or print
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    saved_variant = (ck.get("config", {}) or {}).get("hc_variant") \
        if isinstance(ck, dict) and "model" in ck else None
    check_hc_variant(path, saved_variant, hc_variant)
    if with_dspark:
        model.init_caches(1, model.cfg.max_seq_len)
        model.attach_dspark()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        say(f"  [warn] unexpected keys: {list(unexpected)[:5]}")
    say(f"  loaded {path} (missing={len(missing)} unexpected={len(unexpected)})")
    return ck, missing, unexpected


def build_model(cfg, ckpt=None, token_map=None, device: str = "cpu",
                hc_variant: str = "mhc", with_dspark: bool = False, seed: int = 0,
                say=None):
    """Construct a model, optionally loading `ckpt` into it, on `device`.

    `token_map=None` selects the identity map (correct for smoke tests and unit
    tests; WRONG for a real run -- the training loop and eval refuse to do that
    silently, see their `--token-map` handling). DSpark, when requested, is
    attached BEFORE the checkpoint load (see `load_checkpoint_into`).
    """
    layout = EngramLayout(cfg)
    if token_map is None:
        token_map = identity_token_map(cfg.vocab_size)
    hashes = NgramHashState(cfg, layout, token_map, cfg.engram_compressed_vocab_size)
    model = init_from_scratch(cfg, engram_layout=layout, engram_hashes=hashes,
                              hc_variant=hc_variant, seed=seed)
    if ckpt:
        load_checkpoint_into(model, ckpt, hc_variant,
                             with_dspark=with_dspark, say=say)
    elif with_dspark:
        model.init_caches(1, cfg.max_seq_len)
        model.attach_dspark()
    return model.to(device)
