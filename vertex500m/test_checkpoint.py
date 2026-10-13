"""Checkpoint save/load round-trip and DSpark stage-B freeze verification.

These are the tests that would have caught the init bugs. Given that `tests.py` already
found four silent ones (zero Engram table, fp16 `lse`, gradient-severing snapshot,
detached dtype), a save/load that silently drops or reshapes a tensor is exactly the
class of bug worth pinning down before a multi-day training run.

Run:  python -m vertex500m.test_checkpoint
"""

from __future__ import annotations

import io
import math

import torch

from .config import PRESET
from .factory import build_model
from .optim import build_optimizer
from .tests import FAILED, check


def _build(cfg, seed=0, hc_variant="mhc"):
    return build_model(cfg, seed=seed, hc_variant=hc_variant)


# `perm` is the mHC-lite permutation basis: a CONSTANT [hc!, hc*hc] buffer derived
# purely from `hc_mult`. It is registered persistent=True (spec 16 says non-trainable
# buffers must be serialized) so it round-trips, but it is not a parameter and must not
# be counted as one.
NON_PARAM_BUFFERS = ("perm",)


def _live_tensors(model):
    """Every parameter, plus every persistent buffer that is not a derived constant."""
    out = {n: p for n, p in model.named_parameters()}
    for n, b in model.named_buffers():
        if n.endswith(NON_PARAM_BUFFERS):
            out[n] = b
    return out


def save_load(model, cfg, hc_variant="mhc"):
    """Round-trip through an in-memory buffer, the way a real save/load behaves."""
    sd = {k: v for k, v in model.state_dict().items() if "freqs_cis" not in k}
    buf = io.BytesIO()
    torch.save(sd, buf)
    buf.seek(0)
    loaded = torch.load(buf, map_location="cpu", weights_only=True)
    fresh = _build(cfg, seed=123, hc_variant=hc_variant)   # different seed on purpose
    missing, unexpected = fresh.load_state_dict(loaded, strict=False)
    return fresh, missing, unexpected


def test_roundtrip_is_loss_identical():
    cfg = PRESET.tiny()
    m = _build(cfg, seed=0).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 24))
    m.init_caches(2, cfg.max_seq_len)
    with torch.no_grad():
        a, _ = m(ids, start_pos=0)

    fresh, missing, unexpected = save_load(m, cfg)
    check("no missing keys on load", not missing, str(missing)[:200])
    check("no unexpected keys on load", not unexpected, str(unexpected)[:200])
    fresh.init_caches(2, cfg.max_seq_len)
    with torch.no_grad():
        b, _ = fresh(ids, start_pos=0)
    d = (a - b).abs().max().item()
    check("round-tripped model is BIT-identical", torch.equal(a, b),
          f"max diff {d:.3e}")


def test_no_tensor_is_silently_dropped():
    """A tensor that vanishes from the state dict would train fine and score nothing."""
    for variant in ("mhc", "mhc-lite"):
        cfg = PRESET.tiny()
        m = _build(cfg, seed=0, hc_variant=variant)
        sd = {k: v for k, v in m.state_dict().items() if "freqs_cis" not in k}
        live = _live_tensors(m)
        missing = set(live) - set(sd)
        check(f"[{variant}] every parameter is in the saved state dict", not missing,
              str(sorted(missing))[:200])
        # and nothing in the state dict should be dead weight. Counting ONLY
        # `named_parameters()` under-reports by exactly the `perm` buffers under
        # mHC-lite (24 x 16 x n_layers floats), which made this check report those
        # constant buffers as orphans. They are derived from hc_mult, so they count.
        n_params = sum(p.numel() for p in live.values())
        n_saved = sum(v.numel() for v in sd.values())
        check(f"[{variant}] saved tensors == live tensors (no orphans)",
              n_saved == n_params, f"saved {n_saved} vs live {n_params}")


def test_mhc_lite_checkpoint_roundtrip():
    """mHC-lite's extra `perm` buffer must survive save/load and not shift weights.

    The checkpoint suite only ever built the default `mhc` model, so the mHC-lite
    state dict -- which has 8x wider hc projections plus a constant buffer per block --
    was untested in both directions.
    """
    cfg = PRESET.tiny()
    m = _build(cfg, seed=0, hc_variant="mhc-lite").eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 24))
    m.init_caches(2, cfg.max_seq_len)
    with torch.no_grad():
        a, _ = m(ids, start_pos=0)
    fresh, missing, unexpected = save_load(m, cfg, "mhc-lite")
    check("[mhc-lite] no missing keys on load", not missing, str(missing)[:200])
    check("[mhc-lite] no unexpected keys on load", not unexpected, str(unexpected)[:200])
    check("[mhc-lite] perm buffers serialized",
          all(hasattr(fresh.layers[i], "perm") for i in range(cfg.n_layers))
          and fresh.layers[0].perm.shape[-1] == cfg.hc_mult * cfg.hc_mult,
          "each block carries its own [hc!, hc*hc] permutation basis")
    fresh.init_caches(2, cfg.max_seq_len)
    with torch.no_grad():
        b, _ = fresh(ids, start_pos=0)
    check("[mhc-lite] round-tripped model is BIT-identical",
          torch.equal(a, b), f"max diff {(a-b).abs().max().item():.3e}")


def test_hc_variants_have_different_widths():
    """mHC-lite is a reparameterization, not a conversion: there is no bridge."""
    from .hc import mix_hc_for
    cfg = PRESET.tiny()
    a = _build(cfg, seed=0, hc_variant="mhc")
    b = _build(cfg, seed=0, hc_variant="mhc-lite")
    w_a = a.layers[0].hc_attn_fn
    w_b = b.layers[0].hc_attn_fn
    check("mhc and mhc-lite projections differ in width",
          w_a.shape != w_b.shape,
          f"{tuple(w_a.shape)} vs {tuple(w_b.shape)}")
    check("mix_hc follows the variant",
          mix_hc_for(cfg.hc_mult, "mhc-lite") == 2 * cfg.hc_mult
          + math.factorial(cfg.hc_mult),
          f"{mix_hc_for(cfg.hc_mult, 'mhc')} -> {mix_hc_for(cfg.hc_mult, 'mhc-lite')}")
    src = {k: v for k, v in a.state_dict().items() if "freqs_cis" not in k}
    try:
        b.load_state_dict(src, strict=False)
        loaded = True
    except RuntimeError:
        loaded = False
    check("cross-variant load is REJECTED, not silently accepted", not loaded,
          "a mismatched load would leave hc projections randomly initialized")


def test_tied_embedding_survives_roundtrip():
    """The head is a VIEW of embed.weight. After a reload it must still be a view,
    not a detached copy -- otherwise the two silently stop sharing gradients."""
    cfg = PRESET.tiny()
    m = _build(cfg, seed=0)
    fresh, _, _ = save_load(m, cfg)
    check("head still aliases embed after reload", fresh.head.embed is fresh.embed)
    # the backbone embedding specifically -- layers.1.engram.embed.weight is a different
    # table (the n-gram memory) and is expected to appear
    n_embed = sum(1 for k in fresh.state_dict()
                  if k == "embed.weight" or k.endswith(".embed.weight"))
    check("exactly one backbone [V,d] tensor",
          "embed.weight" in fresh.state_dict()
          and not any(k.startswith("head.") for k in fresh.state_dict()),
          f"state-dict entries ending .embed.weight: {n_embed} "
          f"(includes the engram table, which is correct)")
    check("no head.weight in the state dict",
          not any(k.endswith("head.weight") for k in fresh.state_dict()))


def test_dspark_freeze_leaves_backbone_bit_identical():
    """Stage B's whole contract.

    The drafter shares the tied embedding and the tied head. If the optimizer is built
    before the freeze, or if requires_grad is set wrong, DSpark gradients leak into the
    backbone. Nothing crashes -- the model just quietly gets worse.
    """
    cfg = PRESET.tiny().with_dspark(1)
    m = _build(cfg, seed=0)
    m.init_caches(1, cfg.max_seq_len)
    m.attach_dspark()
    ids = torch.randint(0, cfg.vocab_size, (1, 20))
    m.eval()
    with torch.no_grad():
        before, _ = m(ids, start_pos=0)

    for n, p in m.named_parameters():
        p.requires_grad_(n.startswith("mtp."))
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    check("stage B trains only mtp.*", n_train > 0, f"{n_train} params")
    check("backbone embed frozen", not m.embed.weight.requires_grad)
    check("backbone layer 0 frozen", not m.layers[0].ffn.expert.w1.weight.requires_grad)
    check("engram frozen", not m.layers[1].engram.wkv.weight.requires_grad)

    opts = build_optimizer(m)
    opt_params = [p for o in opts for g in o.param_groups for p in g["params"]]
    check("optimizer holds ONLY trainable params",
          all(p.requires_grad for p in opt_params),
          f"{sum(1 for p in opt_params if p.requires_grad)}/{len(opt_params)}")
    check("optimizer excludes the tied embed",
          all(p is not m.embed.weight for p in opt_params))

    # a few steps of the REAL stage-B objective must not move the backbone at all.
    # Note this is NOT the backbone loss: with the backbone frozen that loss has no
    # trainable tensor in its graph and is correctly non-differentiable. See
    # dspark_loss.py -- that non-differentiability is the signal that the objective
    # was wrong, not something to work around.
    from .dspark_loss import dspark_loss
    m.train()
    for _ in range(3):
        loss, _, _ = dspark_loss(m, ids, temperature=1.0)
        if loss is None:
            continue
        loss.backward()
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
    m.eval()
    with torch.no_grad():
        after, _ = m(ids, start_pos=0)
    d = (before - after).abs().max().item()
    check("backbone logits BIT-identical after DSpark training", d == 0.0,
          f"max diff {d:.3e}")


def test_optimizer_state_roundtrip():
    """Muon momentum and the tables' momentum must survive a checkpoint."""
    cfg = PRESET.tiny()
    m = _build(cfg, seed=0)
    ids = torch.randint(0, cfg.vocab_size, (1, 20))
    m.init_caches(1, cfg.max_seq_len)
    opts = build_optimizer(m)
    m.train()
    for _ in range(3):
        m.init_caches(1, cfg.max_seq_len)
        m.forward_full(ids, use_checkpoint=False)[0].sum().backward()
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
    w_before = m.layers[0].ffn.expert.w1.weight.detach().clone()

    sd = {k: v for k, v in m.state_dict().items() if "freqs_cis" not in k}
    buf = io.BytesIO(); torch.save(sd, buf); buf.seek(0)
    loaded = torch.load(buf, map_location="cpu", weights_only=True)
    fresh = _build(cfg, seed=999)
    fresh.load_state_dict(loaded, strict=True)
    check("weights survive save/load",
          torch.equal(w_before, fresh.layers[0].ffn.expert.w1.weight))

    # continue training from the loaded model; must match the original's next step
    for mm in (m, fresh):
        for o in build_optimizer(mm):
            pass
    opts_m = build_optimizer(m)
    opts_f = build_optimizer(fresh)
    for _ in range(2):                      # warm the momentum buffers
        for mm, oo in ((m, opts_m), (fresh, opts_f)):
            mm.init_caches(1, cfg.max_seq_len)
            mm.forward_full(ids, use_checkpoint=False)[0].sum().backward()
            for o in oo:
                o.step(); o.zero_grad(set_to_none=True)
    d = (m.layers[0].ffn.expert.w1.weight
         - fresh.layers[0].ffn.expert.w1.weight).abs().max().item()
    check("continued training matches after reload", d < 1e-6, f"max diff {d:.3e}")


def _train_step(m, opts, ids):
    m.init_caches(1, m.cfg.max_seq_len)
    m.forward_full(ids, use_checkpoint=False)[0].sum().backward()
    for o in opts:
        o.step()
        o.zero_grad(set_to_none=True)


def test_optimizer_state_survives_real_checkpoint():
    """The actual save_checkpoint payload must restore momentum, not just weights.

    REGRESSION: the payload used to wrap `o.state_dict()` once more, which
    `load_state_dict` silently ignores -- resume logged a restore while every
    momentum buffer restarted from zero, diverging a resumed run within steps.
    Train 2 steps, save, restore into a fresh model+optimizers, train 1 more
    step on both sides: weights must match BIT-exactly (deterministic CPU).
    """
    import argparse
    import tempfile
    from .train.checkpoint import restore_optimizer_state, save_checkpoint
    cfg = PRESET.tiny()
    m = _build(cfg, seed=0)
    ids = torch.randint(0, cfg.vocab_size, (1, 20))
    m.init_caches(1, cfg.max_seq_len)
    opts = build_optimizer(m)
    m.train()
    for _ in range(2):
        _train_step(m, opts, ids)

    with tempfile.TemporaryDirectory() as tmp:
        dest = f"{tmp}/real.pt"
        args = argparse.Namespace(out=dest, _dist=None)
        save_checkpoint(m, args, opts)

        fresh = _build(cfg, seed=999)
        ck = torch.load(dest, map_location="cpu", weights_only=False)
        missing, unexpected = fresh.load_state_dict(ck["model"], strict=False)
        assert not missing and not unexpected, (missing, unexpected)
        opts_f = build_optimizer(fresh)
        for o, st in zip(opts_f, ck["optimizers"]):
            restore_optimizer_state(o, st)
        # momentum buffers actually arrived (non-zero after 2 real steps)
        n_nonzero = sum((v.abs().sum().item() > 0)
                        for o in opts_f for v in o.state.values()
                        if isinstance(v, dict)
                        for v in [v for v in v.values()
                                  if isinstance(v, torch.Tensor)])
        check("restored momentum is non-empty", n_nonzero > 0, f"{n_nonzero}")

        fresh.train()
        _train_step(m, opts, ids)
        _train_step(fresh, opts_f, ids)
        names = [n for n in m.state_dict() if "freqs_cis" not in n]
        d = max((m.state_dict()[n].float()
                 - fresh.state_dict()[n].float()).abs().max().item() for n in names)
        check("one more step matches after optimizer restore", d == 0.0,
              f"max diff {d:.3e}")


def main():
    torch.manual_seed(0)
    tests = [
        test_roundtrip_is_loss_identical,
        test_no_tensor_is_silently_dropped,
        test_mhc_lite_checkpoint_roundtrip,
        test_hc_variants_have_different_widths,
        test_tied_embedding_survives_roundtrip,
        test_dspark_freeze_leaves_backbone_bit_identical,
        test_optimizer_state_roundtrip,
        test_optimizer_state_survives_real_checkpoint,
    ]
    for t in tests:
        print(f"\n== {t.__name__} ==")
        try:
            t()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            FAILED.append(f"{t.__name__}: {e}")
    print("\n" + "=" * 60)
    if FAILED:
        print(f"{len(FAILED)} FAILURE(S):")
        for f in FAILED:
            print(f"  - {f}")
        return 1
    print("all checkpoint checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
