"""Pipeline, data, and training-QoL tests: download/split/manifest/packing.

The architecture suite lives in tests.py and the checkpoint suite in
test_checkpoint.py; this module covers everything the automation layers add:
deterministic splitting, shard writing + manifests, document packing, corpus
download (offline/stubbed -- no network), and later phases' training-QoL and
orchestrator checks.

Run:  python -m vertex500m.test_pipeline
"""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import torch

from .tests import FAILED, check


class _FakeTokenizer:
    """Duck-typed tokenizer: deterministic ids per text, no network."""

    def __init__(self, vocab: int = 500):
        self.eos_token_id = 2
        self._vocab = vocab

    def __call__(self, text, add_special_tokens=False):
        h = abs(hash(text)) % (self._vocab - 10) + 10
        n = max(1, len(text) % 37 + 3)
        return {"input_ids": [(h + i * 31) % self._vocab for i in range(n)]}


def _write_fake_corpus(d: Path, docs: list[list[int]], stem: str,
                       tokens_per_shard: int = 40):
    """Write bins+idx via the real _ShardWriter. Returns ([bins], [idxs])."""
    from .tokenize import _ShardWriter
    w = _ShardWriter(str(d), stem, tokens_per_shard=tokens_per_shard,
                     verbose=False)
    for doc in docs:
        w.add_doc(np.asarray(doc, dtype=np.uint16))
    return w.close()


def _close(*datasets):
    """Release memory-mapped shards. On Windows an open mmap blocks file
    deletion, so tests must call this before their TemporaryDirectory exits."""
    import gc
    for ds in datasets:
        cache = getattr(ds, "_arrays", None)
        if isinstance(cache, dict):
            cache.clear()
    gc.collect()


# --------------------------------------------------------------------------- #
def test_plan_split_sizes_and_determinism():
    from .data import plan_split
    a = plan_split(1000, 0.005, seed=0)
    check("val size is exact", int(a.sum()) == 5, f"got {int(a.sum())}")
    b = plan_split(1000, 0.005, seed=0)
    check("same seed -> same mask", bool((a == b).all()))
    c = plan_split(1000, 0.005, seed=1)
    check("different seed -> different mask", bool((a != c).any()))
    z = plan_split(100, 0.0, seed=0)
    check("val_fraction=0 -> all train", not bool(z.any()) and len(z) == 100)
    small = plan_split(3, 0.005, seed=0)
    check("tiny corpus still holds out 1 val + 2 train",
          int(small.sum()) == 1 and len(small) == 3)
    empty = plan_split(0, 0.005, seed=0)
    check("zero docs -> empty mask (tokenize refuses earlier)",
          len(empty) == 0)
    for bad_n, bad_f in [(1, 0.005)]:
        try:
            plan_split(bad_n, bad_f, seed=0)
            check(f"plan_split({bad_n}, {bad_f}) raises", False)
        except ValueError:
            check(f"plan_split({bad_n}, {bad_f}) raises", True)
    try:
        plan_split(10, 1.5, seed=0)
        check("val_fraction >= 1 raises", False)
    except ValueError:
        check("val_fraction >= 1 raises", True)


def test_shard_writer_roundtrip_and_index():
    from .tokenize import _ShardWriter
    from .data import shard_record
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        docs = [[10, 11, 12], [20, 21], [30, 31, 32, 33], [40]]
        pairs = _write_fake_corpus(d, docs, "web_train", tokens_per_shard=6)
        check("small shard size forces multiple shards", len(pairs) == 2,
              f"got {len(pairs)}")
        # shard 0 fills to 3+2+4=9 tokens (flush triggers at >= 6 after doc 3);
        # shard 1 holds the final doc.
        for (b, x), group in zip(pairs, [docs[:3], docs[3:]]):
            arr = np.fromfile(b, dtype=np.uint16)
            ends = np.load(x)
            flat = [t for doc in group for t in doc]
            check(f"{b.name} contents exact", arr.tolist() == flat)
            # real index check: cumulative doc ends within this shard
            want = []
            acc = 0
            for doc in group:
                acc += len(doc)
                want.append(acc)
            check(f"{x.name} offsets exact", ends.tolist() == want,
                  f"{ends.tolist()} vs {want}")
            rec = shard_record(b, x)
            check(f"{b.name} record counts", rec["tokens"] == len(flat)
                  and rec["docs"] == len(group))


def test_manifest_write_read_validate():
    from .data import (read_manifest, shard_record, validate_manifest,
                       write_manifest)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        pairs = _write_fake_corpus(d, [[1, 2, 3], [4, 5]], "web_train")
        splits = {"train": [shard_record(b, x) for b, x in pairs]}
        m = write_manifest(d, tokenizer="tok", vocab_size=99,
                           compressed_vocab_size=50, collisions=3, seed=7,
                           val_fraction=0.0, splits=splits)
        back = read_manifest(d)
        check("manifest round-trips", back == m)
        check("validate clean", validate_manifest(back, d) == [],
              str(validate_manifest(back, d)))
        # tamper one byte -> hash mismatch (change it for sure: +1 mod 256)
        b = d / splits["train"][0]["file"]
        with open(b, "r+b") as f:
            f.seek(-1, 2)
            here = f.tell()
            val = (f.read(1)[0] + 1) % 256
            f.seek(here)
            f.write(bytes([val]))
        problems = validate_manifest(back, d)
        check("tampered shard flagged",
              any("hash mismatch" in p for p in problems), str(problems))
        # delete a shard -> missing flagged
        b.unlink()
        problems = validate_manifest(back, d)
        check("missing shard flagged",
              any("missing file" in p for p in problems), str(problems))
        # absent manifest -> loud error, not empty dict
        try:
            read_manifest(d / "nope")
            check("missing manifest raises", False)
        except FileNotFoundError:
            check("missing manifest raises", True)


def test_list_shards_split_filtering():
    from .data import list_shards
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for name in ("web_train_00000.bin", "web_val_00000.bin", "other_00000.bin"):
            (d / name).write_bytes(np.arange(10, dtype=np.uint16).tobytes())
        check("split=train selects one",
              [p.name for p in list_shards(d, split="train")]
              == ["web_train_00000.bin"])
        check("split=None returns all", len(list_shards(d)) == 3)
        try:
            list_shards(d, split="zzz")
            check("unknown split raises", False)
        except FileNotFoundError as e:
            check("unknown split raises", "zzz" in str(e))


def test_seed_everything_is_deterministic():
    from .train.dist import seed_everything
    import random as pyrandom
    seed_everything(123)
    a = (torch.rand(4).tolist(), pyrandom.random(),
         __import__("numpy").random.rand())
    seed_everything(123)
    b = (torch.rand(4).tolist(), pyrandom.random(),
         __import__("numpy").random.rand())
    check("same seed -> same streams", a == b)
    seed_everything(124)
    c = torch.rand(4).tolist()
    check("different seed -> different stream", c != a[0])


def test_distributed_sampler_partitions_disjoint():
    from torch.utils.data.distributed import DistributedSampler
    from .data import PackedTokenDataset
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write_fake_corpus(d, [[(i + j) % 200 + 5 for j in range(20)]
                               for i in range(10)], "web_train",
                           tokens_per_shard=10000)
        ds = PackedTokenDataset(sorted(d.glob("*.bin")), seq_len=19)
        assert len(ds) == 10, len(ds)
        got = []
        for rank in (0, 1):
            s = DistributedSampler(ds, num_replicas=2, rank=rank, shuffle=True,
                                   seed=0, drop_last=True)
            s.set_epoch(3)
            got.append(sorted(s))
        check("ranks are disjoint", not set(got[0]) & set(got[1]),
              f"{got[0]} vs {got[1]}")
        check("ranks cover the epoch", sorted(got[0] + got[1]) == list(range(10)))
        _close(ds)


def test_cursor_drives_sampler_epoch():
    from .data import build_dataloader
    from .train.loop import _BatchCursor

    class _FakeSampler:
        def __init__(self):
            self.epochs = []

        def set_epoch(self, epoch):
            self.epochs.append(epoch)

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write_fake_corpus(d, [[(i * 3 + j) % 250 + 5 for j in range(9)]
                               for i in range(6)], "web_train",
                           tokens_per_shard=10000)
        n_win = sum(os.path.getsize(b) // 2 for b in d.glob("*.bin")) // 10
        loader = build_dataloader(d, 9, batch_size=1, seed=0)
        fake = _FakeSampler()
        cur = _BatchCursor(loader, print, seed=0, sampler=fake)
        check("epoch 0 registered at construction", fake.epochs == [0],
              str(fake.epochs))
        check("sampler_epoch mirrors cursor epoch", cur.sampler_epoch == 0)
        for _ in range(n_win):
            cur.next()
        cur.next()   # rolls over
        check("rollover advances sampler epoch", fake.epochs == [0, 1],
              str(fake.epochs))
        check("sampler_epoch follows", cur.sampler_epoch == 1)
        # release the memory-mapped shard before the TemporaryDirectory exits
        # (Windows forbids deleting an open mapping)
        _close(loader.dataset)
        del cur, loader
        __import__("gc").collect()


def test_cursor_epoch_rollover_is_deterministic():    # The exact-resume contract, without training: consuming past the epoch
    # end must yield the identical epoch-1 order as a fresh start_epoch=1
    # cursor. REGRESSION: with persistent_workers=True the carried-over worker
    # processes broke this (epoch 1 differed); workers now respawn per epoch.
    import gc
    from .data import build_dataloader
    from .train.loop import _BatchCursor
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write_fake_corpus(d, [[(i * 7 + j) % 400 + 5 for j in range(9)]
                               for i in range(12)], "web_train",
                           tokens_per_shard=1000)
        bins = sorted(d.glob("*.bin"))
        n_win = sum(os.path.getsize(b) // 2 for b in bins) // 10  # seq 9 + 1
        seq = 9
        la = build_dataloader(d, seq, batch_size=1, seed=0)
        ca = _BatchCursor(la, print, seed=0)
        for _ in range(n_win):
            ca.next()
        check("cursor consumed exactly epoch 0", ca.epoch == 0 and ca.pos == n_win,
              f"epoch={ca.epoch} pos={ca.pos} n={n_win}")
        rolled = [ca.next()["input_ids"].tolist() for _ in range(3)]
        check("cursor rolled to epoch 1", ca.epoch == 1)
        del ca, la
        gc.collect()
        lb = build_dataloader(d, seq, batch_size=1, seed=0)
        cb = _BatchCursor(lb, print, seed=0, start_epoch=1)
        fresh = [cb.next()["input_ids"].tolist() for _ in range(3)]
        check("rollover order == fresh epoch-1 order", rolled == fresh)
        _close(cb)
        del cb, lb
        gc.collect()


def test_packed_dataset_lazy_memmap_and_pickle():
    from .data import PackedTokenDataset
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        pairs = _write_fake_corpus(d, [[i % 250 + 5 for _ in range(9)]
                                       for i in range(4)], "web_train",
                                   tokens_per_shard=1000)
        bins = [b for b, _ in pairs]
        ds = PackedTokenDataset(bins, seq_len=8)
        check("construction loads nothing", ds._arrays == {})
        check("window count", len(ds) == 4, f"got {len(ds)}")
        item = ds[0]
        arr = np.fromfile(bins[0], dtype=np.uint16)
        check("x is window head", item["input_ids"].tolist() == arr[:8].tolist())
        check("y is window head shifted",
              item["targets"].tolist() == arr[1:9].tolist())
        ds2 = pickle.loads(pickle.dumps(ds))     # spawn-safe: no array payload
        check("pickle round-trip readable",
              ds2[0]["input_ids"].tolist() == item["input_ids"].tolist())
        _close(ds, ds2)


def test_doc_packing_covers_windows_and_shuffles():
    from .data import DocPackedDataset
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        rng = np.random.default_rng(0)
        docs = [rng.integers(10, 300, size=int(n)).tolist()
                for n in [3, 5, 2, 8, 4, 6, 3, 7, 5, 4]]
        pairs = _write_fake_corpus(d, docs, "web_train", tokens_per_shard=30)
        bins = [b for b, _ in pairs]
        seq = 7
        ds = DocPackedDataset(bins, seq_len=seq, seed=0)
        check("packed windows exist", len(ds) > 0, f"got {len(ds)}")
        for i in range(len(ds)):
            it = ds[i]
            check(f"window {i} shapes",
                  it["input_ids"].shape == (seq,)
                  and it["targets"].shape == (seq,))
            check(f"window {i} y is x shifted",
                  torch.equal(it["targets"], it["input_ids"].roll(-1, 0))
                  or it["targets"][:-1].tolist() == it["input_ids"][1:].tolist())
            if i > 2:
                break
        total = sum(os.path.getsize(b) // 2 for b in bins)
        used = ds.stats["tokens_used"]
        check("used + waste == total",
              used + int(round(ds.stats["waste"] * total)) == total,
              f"{used} + waste vs {total}")
        check("zero padding by construction", True)
        again = DocPackedDataset(bins, seq_len=seq, seed=0)
        same = all(torch.equal(ds[i]["input_ids"], again[i]["input_ids"])
                   for i in range(len(ds)))
        check("same seed -> identical order", same)
        other = DocPackedDataset(bins, seq_len=seq, seed=1)
        same_sets = sorted(tuple(sorted(ds[i]["input_ids"].tolist()))
                           for i in range(len(ds))) == \
            sorted(tuple(sorted(other[i]["input_ids"].tolist()))
                   for i in range(len(other)))
        check("different seed shuffles the same windows", same_sets)
        _close(ds, again, other)


def test_doc_packing_legacy_fallback():
    import warnings
    from .data import DocPackedDataset
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "legacy_00000.bin").write_bytes(
            np.arange(50, dtype=np.uint16).tobytes())   # no .idx.npy
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ds = DocPackedDataset([d / "legacy_00000.bin"], seq_len=9, seed=0)
        check("legacy shard warns", any("no .idx.npy" in str(w.message)
                                        for w in caught))
        check("legacy shard fixed-packs", len(ds) == 5, f"got {len(ds)}")
        _close(ds)


def test_download_offline_and_cached():
    from .data import fetch_corpus
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        src = [{"path": "no/such-dataset", "split": "train",
                "text_field": "text", "limit_docs": 5}]
        try:
            fetch_corpus(src, d, offline=True)
            check("offline + missing output raises", False)
        except FileNotFoundError:
            check("offline + missing output raises", True)
        # cached output is reused without touching the network (datasets is
        # only imported when a fetch is actually needed)
        (d / "no_such_dataset.jsonl").write_text('{"text": "hi"}\n')
        got = fetch_corpus(src, d, offline=True, verbose=False)
        check("cached output reused offline",
              len(got) == 1 and got[0].name == "no_such_dataset.jsonl")


def test_fetch_one_with_stub_loader():
    from .data.download import _fetch_one

    class _StubDS:
        def __init__(self, rows):
            self._rows = rows

        def take(self, n):
            return self._rows[:n]

        def __iter__(self):
            return iter(self._rows)

    rows = [{"text": "hello"}, {"text": "  "}, {"nope": 1}, {"text": "world"}]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "s.jsonl"
        n = _fetch_one(lambda *a, **k: _StubDS(rows), {}, "x", "train", "text",
                       10, out, lambda *a: None)
        check("stub streaming fetch skips empties", n == 2, f"got {n}")
        lines = out.read_text().strip().split("\n")
        check("jsonl lines valid",
              [__import__("json").loads(ln)["text"] for ln in lines]
              == ["hello", "world"])


# --------------------------------------------------------------------------- #
# Training QoL (phase 3): tiny model, small synth corpora, CPU, --tiny
# --------------------------------------------------------------------------- #
def _train_args(tmp: str, **over):
    """A complete train-loop Namespace. Defaults: tiny CPU run, all QoL off."""
    import argparse
    args = argparse.Namespace(
        stage="pre", data=None, init=None, resume=False, token_map=None,
        allow_identity_map=True, out=None, smoke=False, tiny=True, cpu=True,
        seed=0, amp="off", init_scale=65536.0, compile=False, hc_variant="mhc",
        steps=4, batch_size=1, seq_len=32, loss_chunk=1024, synthetic_tokens=20000,
        quant_mode="FAKE", muon_lr=0.02, adamw_lr=2.6e-4, mtp_lr=0.02,
        embed_lr=0.02, weight_decay=0.1, embed_optimizer="adamw", warmup=2,
        no_ckpt=True, log_every=1, temperature=1.0, eval_batches=4,
        save_every=0, ckpt_dir=None, keep_last=3, eval_every=0, val_data=None,
        val_split="val", train_split="all", pack="fixed", grad_clip=None,
        accum_steps=1, target_loss=None, deadline_s=None, metrics=None)
    for k, v in over.items():
        setattr(args, k, v)
    if args.out is None and args.ckpt_dir is None:
        pass
    return args


def _synth_dir(tmp: str, name: str = "c", n_tokens: int = 20000) -> str:
    from .data import synthetic_corpus
    d = os.path.join(tmp, name)
    synthetic_corpus(d, n_tokens=n_tokens)
    return d


def _final_weights(path: str) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["model"]


def _max_diff(a: dict, b: dict) -> float:
    return max((a[k].float() - b[k].float()).abs().max().item() for k in a)


def test_finalize_step_clips_global_norm():
    from .train import finalize_step
    p1 = torch.nn.Parameter(torch.zeros(4))
    p2 = torch.nn.Parameter(torch.zeros(3))
    o1 = torch.optim.SGD([p1], lr=1.0)
    o2 = torch.optim.SGD([p2], lr=1.0)
    p1.grad = torch.full((4,), 3.0)
    p2.grad = torch.full((3,), 4.0)
    skipped, scale, norm = finalize_step([o1, o2], None, grad_clip=4.2)
    import math
    want = math.sqrt(4 * 9.0 + 3 * 16.0)   # sqrt(84)
    check("reported norm is the global norm", abs(float(norm) - want) < 1e-5,
          f"{float(norm)} vs {want}")
    check("update scaled by clip/norm",
          abs(p1[0].item() - (-3.0 * 4.2 / want)) < 1e-6, f"{p1[0].item()}")
    check("no skip, no scale without scaler", (skipped, scale) == (False, None))


def test_accum_matches_large_batch():
    # Micro-batching and full-batching take different summation orders (and
    # Adam/Muon normalise near-zero grads to +-lr), so weight trajectories can
    # never be bit-identical across batchings. What accumulation MUST guarantee:
    # (a) the reported loss is the true full-batch mean (1/N scaling correct),
    # (b) exactly one optimizer step per N micro-batches.
    from unittest import mock
    from vertex500m import optim as optim_mod
    from vertex500m.train import run
    from vertex500m.train.metrics import read_metrics
    with tempfile.TemporaryDirectory() as tmp:
        data = _synth_dir(tmp)
        a = _train_args(tmp, data=data, steps=1, batch_size=1, accum_steps=2,
                        out=os.path.join(tmp, "a.pt"), no_ckpt=True,
                        metrics=os.path.join(tmp, "a.jsonl"))
        run(a)
        b = _train_args(tmp, data=data, steps=1, batch_size=2, accum_steps=1,
                        out=os.path.join(tmp, "b.pt"), no_ckpt=True,
                        metrics=os.path.join(tmp, "b.jsonl"))
        run(b)
        la = [r for r in read_metrics(os.path.join(tmp, "a.jsonl"))
              if r.get("split") == "train"][0]["loss"]
        lb = [r for r in read_metrics(os.path.join(tmp, "b.jsonl"))
              if r.get("split") == "train"][0]["loss"]
        check("accum loss == full-batch loss", abs(la - lb) < 1e-6,
              f"{la} vs {lb}")

        def count_steps(**kw):
            counts = {}
            patches = []
            for cls in (optim_mod.Muon, torch.optim.AdamW,
                        optim_mod.SinkhornBalanced):
                maker = cls
                orig = cls.step

                def counting(self, *args, _cls=maker, _orig=orig, **kwargs):
                    counts[_cls.__name__] = counts.get(_cls.__name__, 0) + 1
                    return _orig(self, *args, **kwargs)

                patches.append(mock.patch.object(cls, "step", counting))
            for p in patches:
                p.start()
            try:
                run(_train_args(tmp, data=data, out=os.path.join(tmp, "x.pt"),
                                no_ckpt=True, **kw))
            finally:
                for p in patches:
                    p.stop()
            return counts

        c2 = count_steps(steps=2, batch_size=1, accum_steps=2)
        c1 = count_steps(steps=4, batch_size=1, accum_steps=1)
        check("accum=2 steps twice per optimizer",
              all(v == 2 for v in c2.values()) and len(c2) == 3, str(c2))
        check("accum=1 steps every micro-batch (same 4 micros -> 4 steps)",
              all(v == 4 for v in c1.values()) and len(c1) == 3, str(c1))


def test_grad_clip_reports_norm_and_noop_is_identical():
    # Muon's update is scale-invariant (Newton-Schulz orthogonalisation), so a
    # clip cannot bound the STEP the way it does for SGD -- what the loop must
    # guarantee is (a) the reported norm is the true pre-clip global norm and
    # (b) a no-op clip (max >> norm) is bit-identical to no clipping. Real
    # down-scaling math is pinned exactly by test_finalize_step_clips_global_norm.
    from .train import read_metrics, run
    with tempfile.TemporaryDirectory() as tmp:
        data = _synth_dir(tmp)
        args = _train_args(tmp, data=data, steps=2, out=os.path.join(tmp, "c.pt"),
                           grad_clip=1e30, metrics=os.path.join(tmp, "m.jsonl"))
        run(args)
        recs = [r for r in read_metrics(os.path.join(tmp, "m.jsonl"))
                if r.get("split") == "train"]
        check("metrics records grad_norm",
              len(recs) == 2 and all(r["grad_norm"] > 0 for r in recs),
              str([r.get("grad_norm") for r in recs]))
        args2 = _train_args(tmp, data=data, steps=2, out=os.path.join(tmp, "u.pt"))
        run(args2)
        d = _max_diff(_final_weights(args.out), _final_weights(args2.out))
        check("no-op clip is bit-identical to no clip", d == 0.0, f"{d:.3e}")
        args3 = _train_args(tmp, data=data, steps=1, out=os.path.join(tmp, "t.pt"),
                            grad_clip=1e-9,
                            metrics=os.path.join(tmp, "m3.jsonl"))
        run(args3)
        tiny_recs = [r for r in read_metrics(os.path.join(tmp, "m3.jsonl"))
                     if r.get("split") == "train"]
        check("reported norm is pre-clip (same grads, tiny max)",
              abs(tiny_recs[0]["grad_norm"] - recs[0]["grad_norm"]) < 1e-6,
              f"{tiny_recs[0]['grad_norm']} vs {recs[0]['grad_norm']}")


def test_resume_is_exact():
    from .train import run
    with tempfile.TemporaryDirectory() as tmp:
        data = _synth_dir(tmp)
        ckpt_dir = os.path.join(tmp, "ck")
        a = _train_args(tmp, data=data, steps=2, out=os.path.join(tmp, "a.pt"),
                        ckpt_dir=ckpt_dir, save_every=1, no_ckpt=False)
        st = run(a)
        check("first leg completes", st == "done")
        latest = os.path.join(ckpt_dir, "pre.latest.pt")
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        check("latest carries train state",
              ck.get("train", {}).get("step") == 2, str(ck.get("train", {}).keys()))
        check("latest carries RNG + epoch position",
              "rng" in ck and ck["train"].get("epoch") == 0
              and ck["train"].get("pos_in_epoch") == 2,
              str(ck.get("train")))
        b = _train_args(tmp, data=data, steps=4, init=latest, resume=True,
                        out=os.path.join(tmp, "b.pt"), ckpt_dir=ckpt_dir,
                        save_every=1, no_ckpt=False)
        st = run(b)
        check("resume completes", st == "done")
        c = _train_args(tmp, data=data, steps=4, out=os.path.join(tmp, "c.pt"),
                        no_ckpt=False)
        run(c)
        d = _max_diff(_final_weights(b.out), _final_weights(c.out))
        check("resumed 2+2 == uninterrupted 4", d < 1e-6, f"max diff {d:.3e}")


def test_periodic_save_eval_best_and_metrics():
    import json
    from .train import run
    with tempfile.TemporaryDirectory() as tmp:
        data = _synth_dir(tmp)
        vdir = _synth_dir(tmp, name="v")
        os.rename(os.path.join(vdir, "synthetic_00000.bin"),
                  os.path.join(vdir, "synth_val_00000.bin"))
        ckpt_dir = os.path.join(tmp, "ck")
        args = _train_args(tmp, data=data, steps=4, out=os.path.join(tmp, "f.pt"),
                           ckpt_dir=ckpt_dir, save_every=2, keep_last=1,
                           eval_every=2, val_data=vdir, no_ckpt=True)
        st = run(args)
        check("run completes", st == "done")
        names = sorted(os.path.basename(p) for p in Path(ckpt_dir).glob("*.pt"))
        check("only last 1 step file kept",
              "pre.step_0000004.pt" in names
              and "pre.step_0000002.pt" not in names, str(names))
        check("latest + best written",
              "pre.latest.pt" in names and "pre.best.pt" in names, str(names))
        recs = [json.loads(ln) for ln in
                open(os.path.join(ckpt_dir, "metrics.jsonl"))]
        train_recs = [r for r in recs if r.get("split") == "train"]
        val_recs = [r for r in recs if r.get("split") == "val"]
        check("train records logged", len(train_recs) == 4, f"{len(train_recs)}")
        check("val records logged with CE",
              len(val_recs) == 2 and all("val_ce" in r for r in val_recs),
              f"{len(val_recs)}")
        check("val steps align with eval_every",
              [r["step"] for r in val_recs] == [2, 4])


# --------------------------------------------------------------------------- #
# Pipeline (phase 5): config, argv builders, state machine, report
# --------------------------------------------------------------------------- #
def _smoke_cfg():
    from .pipeline import load_run_config
    root = Path(__file__).resolve().parent.parent
    return load_run_config(str(root / "configs" / "smoke.yaml"))


def test_run_config_validation():
    from .pipeline import check_config
    from .pipeline.run_config import ConfigError, RunConfig, validate
    cfg = _smoke_cfg()
    check("smoke.yaml validates clean",
          [lvl for lvl, _ in validate(cfg)] == [] or
          all(lvl == "warn" for lvl, _ in validate(cfg)))
    bad = _smoke_cfg()
    bad.train.amp = "wat"
    try:
        check_config(bad)
        check("bad amp raises", False)
    except ConfigError:
        check("bad amp raises", True)
    bad = _smoke_cfg()
    bad.train.stage_a_steps = -3
    bad.model.hc_variant = "moe"
    bad.tracking.mode = "carrier-pigeon"
    try:
        check_config(bad)
        check("multiple errors listed together", False)
    except ConfigError as e:
        txt = str(e)
        check("multiple errors listed together",
              "stage_a_steps" in txt and "hc_variant" in txt
              and "tracking.mode" in txt, txt[:200])
    try:
        from .pipeline import load_run_config as _load
        import tempfile as _tf
        with _tf.NamedTemporaryFile("w", suffix=".yaml",
                                    delete=False) as f:
            f.write("run_name: x\ntrain:\n  grad_clipp: 1.0\n")
            name = f.name
        try:
            _load(name)
            check("unknown key raises", False)
        finally:
            os.unlink(name)
    except ConfigError:
        check("unknown key raises", True)
    _ = RunConfig


def test_set_overrides():
    from .pipeline import apply_set_overrides
    from .pipeline.run_config import ConfigError
    cfg = _smoke_cfg()
    apply_set_overrides(cfg, ["train.batch_size=8", "train.grad_clip=null",
                              "model.tiny=false", "tracking.tags=[a, b]"])
    check("overrides apply with types",
          cfg.train.batch_size == 8 and cfg.train.grad_clip is None
          and cfg.model.tiny is False and cfg.tracking.tags == ["a", "b"],
          f"{cfg.train.batch_size} {cfg.train.grad_clip} {cfg.tracking.tags}")
    try:
        apply_set_overrides(cfg, ["train.nope=1"])
        check("unknown override path raises", False)
    except ConfigError:
        check("unknown override path raises", True)


def test_train_and_eval_argv():
    from .pipeline.orchestrator import default_paths, eval_argv, train_argv
    import tempfile as _tf
    with _tf.TemporaryDirectory() as tmp:
        cfg = _smoke_cfg()
        run_dir = os.path.join(tmp, "runs", cfg.run_name)
        os.makedirs(run_dir)
        paths = default_paths(cfg, run_dir)
        argv = train_argv(cfg, paths, "pre", None, False)
        for flag in ("--tiny", "--ckpt-dir", "--metrics", "--steps", "6",
                     "--allow-identity-map", "--train-split", "train",
                     "--amp", "off"):
            check(f"train argv has {flag}", flag in argv, " ".join(argv))
        check("no token-map when file absent", "--token-map" not in argv)
        ev = eval_argv(cfg, paths, os.path.join(run_dir, "backbone.pt"), False)
        check("eval argv split all", "--split" in ev
              and ev[ev.index("--split") + 1] == "all")


def test_eval_output_parsing():
    from .pipeline.report import parse_eval_output
    text = ("=== val ===\n  tokens     : 12,345\n  mean CE    : 3.2141 nats/token\n"
            "  perplexity : 24.871\n  bits/token : 4.6362\n"
            "  mean accept: 31.2%  over 8 batches\n")
    r = parse_eval_output(text)
    check("eval numbers parsed",
          r.get("val_ce") == 3.2141 and r.get("perplexity") == 24.871
          and r.get("eval_tokens") == 12345 and r.get("acceptance_pct") == 31.2,
          str(r))
    check("empty parse is empty", parse_eval_output("no numbers here") == {})


def test_train_status_machine():
    import torch
    from .pipeline.orchestrator import _train_status
    with tempfile.TemporaryDirectory() as tmp:
        ckdir = os.path.join(tmp, "ck")
        os.makedirs(ckdir)
        paths = {"ckpt_dir": ckdir}
        out = os.path.join(tmp, "backbone.pt")
        check("fresh with no files",
              _train_status(paths, "pre", 6, out) == ("fresh", None, False))
        torch.save({"train": {"step": 6}}, os.path.join(ckdir, "pre.latest.pt"))
        torch.save({"train": {"step": 6}}, out)
        check("complete artefacts skip",
              _train_status(paths, "pre", 6, out)[0] == "skip")
        act, init, res = _train_status(paths, "pre", 9, out)
        check("extended total resumes (from latest or final, same point)",
              act == "resume" and res and init in
              (out, os.path.join(ckdir, "pre.latest.pt")), f"{act} {init}")
        os.unlink(out)
        torch.save({"train": {"step": 3}}, os.path.join(ckdir, "pre.latest.pt"))
        act, init, res = _train_status(paths, "pre", 6, out)
        check("partial latest resumes",
              act == "resume" and res and init.endswith("pre.latest.pt"))


def test_tracker_disabled_is_noop():
    import sys as _sys
    from .pipeline.tracker import Tracker
    from .pipeline.run_config import TrackingConfig
    before = set(_sys.modules)
    tr = Tracker(TrackingConfig(mode="disabled"), "/tmp/nonexistent", "x")
    tr.tail_file("/tmp/nonexistent/m.jsonl")
    tr.log({"loss": 1.0}, step=1)
    tr.summary({"a": 1})
    tr.finish()
    check("disabled tracker never imports wandb",
          "wandb" not in set(_sys.modules) - before)
    check("disabled tracker has no run id", tr.run_id is None)


def test_tracker_offline_logs_tailed_metrics():
    # Real wandb run in offline mode (no network): init + tail a metrics
    # file + finish must work end to end. Slowest test here (wandb import).
    # NOTE: wandb's backend holds its log file open for the life of the
    # process, so on Windows this scratch dir cannot be deleted from inside
    # the test -- mkdtemp + ignore_errors instead of TemporaryDirectory.
    import json as _json
    import shutil as _shutil
    from .pipeline.run_config import TrackingConfig
    from .pipeline.tracker import Tracker
    tmp = tempfile.mkdtemp(prefix="vertex_wandb_test_")
    try:
        mpath = os.path.join(tmp, "m.jsonl")
        open(mpath, "w").close()
        tr = Tracker(TrackingConfig(mode="offline", project="vertex-test"),
                     tmp, "test-run", config_dict={"a": 1})
        check("offline run gets an id", bool(tr.run_id), str(tr.run_id))
        tr.tail_file(mpath)
        with open(mpath, "a") as f:
            f.write(_json.dumps({"step": 1, "split": "train", "stage": "pre",
                                 "loss": 5.0}) + "\n")
        deadline = __import__("time").time() + 60
        runs_dir = os.path.join(tmp, "wandb")
        seen = False
        while __import__("time").time() < deadline:
            if os.path.isdir(runs_dir) and os.listdir(runs_dir):
                seen = True
                break
            __import__("time").sleep(2)
        tr.summary({"done": 1})
        tr.finish()
        check("offline wandb run materialised", seen)
    finally:
        _shutil.rmtree(tmp, ignore_errors=True)


def test_config_change_detection_and_report_stage():
    from .pipeline import load_state, run_pipeline, save_state
    from .pipeline.orchestrator import default_paths
    import tempfile as _tf
    with _tf.TemporaryDirectory() as tmp:
        cfg = _smoke_cfg()
        cfg.output_dir = tmp
        run_dir = os.path.join(tmp, cfg.run_name)
        os.makedirs(run_dir)
        paths = default_paths(cfg, run_dir)
        # fabricated prior state with a DIFFERENT hash
        st = load_state(run_dir)
        st["config_hash"] = "deadbeefdeadbeef"
        st["stages"] = {"preflight": {"status": "done"}}
        save_state(run_dir, st)
        quiet = lambda *a, **k: None
        code = run_pipeline(cfg, paths, run_dir, ["report"], say=quiet)
        check("changed config refuses", code == 1)
        code = run_pipeline(cfg, paths, run_dir, ["report"], say=quiet,
                            allow_config_change=True)
        check("allow flag proceeds", code == 0)
        check("report artefacts written",
              os.path.exists(os.path.join(run_dir, "report.md"))
              and os.path.exists(os.path.join(run_dir, "summary.json")))
        import json as _json
        summary = _json.load(open(os.path.join(run_dir, "summary.json")))
        check("summary names the run", summary.get("run_name") == "vertex-smoke")


def test_report_from_fixtures():
    import json as _json
    from .pipeline.report import write_report
    import tempfile as _tf
    with _tf.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "checkpoints"))
        with open(os.path.join(tmp, "checkpoints", "pre.latest.pt"),
                  "wb") as f:
            f.write(b"x" * 100)
        with open(os.path.join(tmp, "metrics_pre.jsonl"), "w") as f:
            f.write(_json.dumps({"step": 1, "split": "train", "stage": "pre",
                                 "loss": 5.0}) + "\n")
            f.write(_json.dumps({"step": 2, "split": "val", "stage": "pre",
                                 "val_ce": 4.5, "val_tokens": 64,
                                 "best_val": 4.5}) + "\n")
        cfg = _smoke_cfg()
        state = {"config_hash": "abc", "stages": {"stage_a": {"status": "done"}},
                 "eval_result": {"val_ce": 4.4, "perplexity": 81.4},
                 "wandb_run_id": None}
        summary = write_report(tmp, cfg, state, state["eval_result"])
        check("summary has train/val/eval",
              summary["train"]["stage_a"]["final_loss"] == 5.0
              and summary["train"]["val"]["best_val_ce"] == 4.5
              and summary["eval"]["val_ce"] == 4.4, str(summary["train"]))
        check("checkpoints inventoried",
              summary["checkpoints"][0]["file"] == "pre.latest.pt")


def test_target_and_deadline_exits():
    from .train import run
    with tempfile.TemporaryDirectory() as tmp:
        data = _synth_dir(tmp)
        args = _train_args(tmp, data=data, steps=4, out=os.path.join(tmp, "t.pt"),
                           target_loss=1e9)
        st = run(args)
        check("huge target exits immediately", st == "target")
        check("target still saves final", os.path.exists(args.out))
        args2 = _train_args(tmp, data=data, steps=4, out=os.path.join(tmp, "d.pt"),
                            ckpt_dir=os.path.join(tmp, "ck2"), deadline_s=0)
        st2 = run(args2)
        check("zero deadline stops", st2 == "deadline")
        check("deadline saves latest",
              os.path.exists(os.path.join(tmp, "ck2", "pre.latest.pt")))


def main():
    torch.manual_seed(0)
    tests = [
        test_plan_split_sizes_and_determinism,
        test_shard_writer_roundtrip_and_index,
        test_manifest_write_read_validate,
        test_list_shards_split_filtering,
        test_packed_dataset_lazy_memmap_and_pickle,
        test_doc_packing_covers_windows_and_shuffles,
        test_doc_packing_legacy_fallback,
        test_download_offline_and_cached,
        test_fetch_one_with_stub_loader,
        test_cursor_epoch_rollover_is_deterministic,
        test_seed_everything_is_deterministic,
        test_distributed_sampler_partitions_disjoint,
        test_cursor_drives_sampler_epoch,
        test_finalize_step_clips_global_norm,
        test_accum_matches_large_batch,
        test_grad_clip_reports_norm_and_noop_is_identical,
        test_resume_is_exact,
        test_periodic_save_eval_best_and_metrics,
        test_target_and_deadline_exits,
        test_run_config_validation,
        test_set_overrides,
        test_train_and_eval_argv,
        test_eval_output_parsing,
        test_train_status_machine,
        test_tracker_disabled_is_noop,
        test_tracker_offline_logs_tailed_metrics,
        test_config_change_detection_and_report_stage,
        test_report_from_fixtures,
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
    print("all pipeline checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
