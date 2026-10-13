# vertex500m

A dense **503.8M** reproduction of the DeepSeek-V4.1-Flash architecture, trained from
scratch on a single consumer GPU.

Architecture of record: `DeepSeek-V4.1-Flash_Architecture_Spec.md`.
MoE→dense conversion: `vertex500m-dense.md`. Hyper-connection study: `mHC-lite.md`.

## Quick start

```bash
pip install torch transformers numpy datasets pyyaml wandb
# (or: pip install -e . from the repo root)
```

### Automated: one command, whole pipeline (recommended)

All configuration lives in a YAML file that is settled BEFORE deployment --
the training device needs no terminal input, ever:

```bash
# 0. check the config on a dev machine (validates everything, runs preflight,
#    touches nothing else). Do this BEFORE deploying.
python -m vertex500m.pipeline --config configs/t4-fp16.yaml --validate

# 1. run the whole chain: preflight -> data (download + tokenize + split)
#    -> stage A -> stage B -> eval -> report. Everything lands in
#    runs/<run_name>/ (checkpoints, metrics.jsonl, report.md, logs).
python -m vertex500m.pipeline --config configs/t4-fp16.yaml

# 2. resume after anything (crash, Ctrl+C, deadline): re-run the SAME command.
#    Completed stages are skipped; interrupted training continues at the exact
#    step, LR position, RNG state and data order. A changed config refuses
#    unless passed --allow-config-change.
python -m vertex500m.pipeline --config configs/t4-fp16.yaml

# 3. try it fast, on CPU, with no network:
python -m vertex500m.pipeline --config configs/smoke.yaml
```

`configs/t4-fp16.yaml` is the documented real-run starting point (single T4,
fp16, FineWeb-Edu sample, wandb online); `configs/smoke.yaml` runs the full
chain on CPU in minutes. Metrics go to `metrics_*.jsonl` AND wandb
(`tracking.mode: offline` records locally for a later `wandb sync`).

### Manual mode: the individual steps

```bash
# 0. verify the parameter accounting (reproduces the MoE preset as a gate)
python -m vertex500m.params
python -m vertex500m.params --hc-variant mhc-lite    # the exact-Birkhoff variant

# 1. verify the architecture
python -m vertex500m.tests              # 90+ checks, all must pass
python -m vertex500m.test_checkpoint    # save/load, both hc variants, stage-B freeze
python -m vertex500m.test_pipeline      # data pipeline, training QoL, orchestrator

# 2. tokenize your corpus and build the Engram token map
#    (writes {name}_{train|val}_*.bin + .idx.npy doc index + manifest.json;
#    --val-fraction 0 for a train-only corpus)
python -m vertex500m.tokenize --tokenizer JamesQuartz/qt-VI1.3-32k \
    --input corpus/ --out data/ --name web --save-map token_map.json

# 3. pretrain (stage A)
python -m vertex500m.train --data data/ --token-map token_map.json \
    --steps 3000 --batch-size 4 --seq-len 8192 --out backbone.pt

# 3b. ...the same thing in mixed precision (RECOMMENDED on any GPU)
python -m vertex500m.train --amp fp16 --data data/ --token-map token_map.json \
    --steps 3000 --batch-size 4 --seq-len 8192 --out backbone.pt

# 4. DSpark (stage B: frozen backbone, drafter only, ~9 min)
python -m vertex500m.train --stage dspark --init backbone.pt \
    --token-map token_map.json --data data/ --steps 400 --out full.pt

# 5. evaluate  (--amp defaults to whatever the checkpoint recorded)
python -m vertex500m.eval --ckpt full.pt --data data/ --split val \
    --token-map token_map.json --with-dspark

# 6. generate / benchmark
python -m vertex500m.bench --device cuda

# 6b. measure what --amp actually bought YOU (tokens/s + peak memory + inf-skips)
python -m vertex500m.bench --device cuda --train --seq-len 8192 --batch-size 2

# A/B the hyper-connection parameterization end to end (see mHC-lite.md §8.1).
# Both variants must be trained from scratch: there is no checkpoint bridge.
python -m vertex500m.train --data data/ --token-map token_map.json \
    --hc-variant mhc-lite --steps 3000 --out backbone_lite.pt
```

Resume an interrupted run with `--init backbone.pt --resume`. Resume is EXACT:
the checkpoint carries the step counter, the LR schedule position
(`total_steps`/`warmup` -- the checkpoint's win over conflicting CLI flags),
all RNG streams, the dataloader epoch + in-epoch position (per-epoch
reseeding rebuilds the identical data order), the optimizer state AND the
GradScaler state. Without `--resume` the weights load but everything else
restarts. Periodic checkpoints (`--save-every N` -> `{stage}.step_{n}.pt` +
`{stage}.latest.pt`, `--keep-last` rotation, `{stage}.best.pt` on validation
improvement), in-loop validation (`--eval-every`), gradient clipping
(`--grad-clip`), accumulation (`--accum-steps`), early exits
(`--target-loss`, `--deadline-s`), NaN-loss fail-fast, SIGINT/SIGTERM
save-then-exit, and a `metrics.jsonl` log round out the training QoL; see
`train/__main__.py` for the full flag list.

Two GPUs (torchrun; note a T4 has no NVLink, so the allreduce runs over PCIe).
Prefer the pipeline (`train.dist_nproc: 2` launches the stages under
torchrun itself); manually:

```bash
torchrun --nproc_per_node 2 -m vertex500m.train --amp fp16 \
    --data data/ --token-map token_map.json --steps 3000 --out backbone.pt
```

DDP is rank-sharded (DistributedSampler, per-epoch `set_epoch`, so resume
restores each rank's slice) with fully-seeded per-rank RNGs -- the old
identical-shuffle-across-ranks bug is fixed and pinned by tests. In-loop
validation all-reduces (nll, ntokens) so all ranks agree on the CE.

## Mixed precision (`--amp fp16|bf16|off`)

Weights, gradients and optimizer state stay **fp32 in every mode**. Only compute is
lowered, via `torch.autocast`. That is not negotiable here: AdamW is built with
`eps=1e-20`, below fp16's smallest subnormal (5.96e-8), and Muon's Newton-Schulz plus
SinkhornBalanced's 11 balancing rounds are fp32-reasoning loops. Checkpoints stay fp32
and load unchanged, so `--amp` never changes a file format.

**`fp16` is the default you want, and on a T4 it is the only one that works.** BF16
tensor cores arrived with Ampere (sm_80); a T4 is sm_75 and has FP16 tensor cores only.
`--amp bf16` on a T4 computes at roughly FP32 speed with doubled activation bytes —
strictly worse than fp16. `amp.warn_if_unsupported` says so at startup.

`--amp fp16` attaches a `torch.amp.GradScaler`; `bf16` does not (its exponent range is
fp32's, so gradients cannot overflow). **Watch the inf-skip rate**
(`bench.py --train` prints it, and `train.py` prints it at the end of a run). Above ~5%
of steps, fp16 is the wrong tool for this model: lower `--init-scale`, or move to bf16
on a box that has bf16 tensor cores.

Measured at `PRESET.tiny()`, checkpointing on, same weights, cosine of every gradient
against an fp32 run:

| mode | loss vs fp32 | grad cosine (median / worst) | tensors with grad lost |
|---|---|---|---|
| `fp16` | 6.21255 → 6.21261 (Δ 1.6e-05) | **+0.99988 / +0.9766** | 0 |
| `bf16` | 6.21255 → 6.21233 (Δ 2.2e-04) | +0.99945 / **+0.9111** | 0 |

fp16's worst case is materially better than bf16's (11 mantissa bits vs 8), which is the
quantitative version of "fp16 is the right default".

fp32 activation **traffic** in one forward pass, real width (d=1344, 4 layers, T=128),
measured with a `TorchDispatchMode` byte counter:

| | total float bytes written | fp32 share |
|---|---|---|
| `--amp off` | 1531 MB | 100% |
| `--amp fp16` | 1014 MB | **14%** |

Note this is *traffic*, not *residency*: it is bytes moved per forward, which is what
bounds throughput on a bandwidth-bound GPU. Resident activation memory (the table below)
is a different quantity and is governed by gradient checkpointing.

## The config

| | value |
|---|---|
| `d` / `L` / `H` × `hd` | 1344 / 20 / 21 × 128 (= 2d) |
| `q_lora_rank` = `o_lora_rank` / `o_groups` | 384 / 4 |
| `MoEI` | 3584 = `round(8·1344/3/128)·128` |
| `index_topk` / `sliding_window` | 384 / 128 |
| `engram_head_dim` / rows | 64 / 600,766 |
| vocab / tied | 32,000 / **yes** |
| **total (with DSpark)** | **503,795,816** |
| **total (stage A, no DSpark)** | **484,266,909** |
| **active/token** | **433.6M (86.1%)** |
| fwd+bwd | **2.602 GFLOP/token** |

Budget: embed 8.5% · backbone 86.3% · Engram 8.7% · DSpark 5.0%.
Per layer 74.0% FFN / 24.7% attention / 1.3% hyper-connections.

`python -m vertex500m.params --hc-variant mhc-lite` accounts for the mHC-lite
parameterization (+1.72M, +0.34%); `mHC-lite.md` §8.1 has the reasoning.

### Memory

| | |
|---|---|
| KV cache (inference, HBM) | 144 B/token + 320 KiB/seq window rings (288 B/token under `--amp`) |
| training activations (resident) | 1,144 KB/token → **220 KB/token with segment-2 checkpointing** |
| weights + grads + momentum | 9.2 GB on a 16 GB card; `--amp` does **not** reduce this |
| **total @ T=8192, fp32** | **5.46 GB** → fits a 12 GB card |
| **total @ T=8192, `--amp fp16`** | roughly half the activation term; same 9.2 GB static bill |
| checkpoint overhead | +33% FLOPs (independent of segment size) |

The T=8192 figure assumes a **chunked** fp32 loss. `train.chunked_ce` and `eval.py` both
chunk over positions; unchunked, `[B,S,V]` fp32 logits plus their cast plus log_softmax is
~3.1 GB at T=8192 and ~12.6 GB at T=32768, before any activation. `--seq-len` therefore
defaults to 2048 rather than `cfg.max_seq_len`.

**Mixed precision buys traffic, not residency.** Halving activation bytes doubles the
bandwidth headroom on a memory-bound GPU; it does not halve the peak allocation, which
segment checkpointing governs. On a T4 (65 TFLOPS fp16, 320 GB/s → a ridge point near
203 FLOP/byte) this model sits well below that ridge, so the traffic reduction is the
win that matters and the GPU will not become FLOP-bound.

## Five deliberate deviations

1. **`q_lora_rank = 384`, not `d/4 = 336`.** `336 % 32 = 16` fails FP8 block-32
   quantization on `wq_a`, `wq_b` and `indexer.wq_b`. A hard requirement of the
   format, not a preference. This is also why the spec's own `d/4` rule silently
   breaks at `d=1344`.
2. **`index_topk = 384`** (spec: 512). The top-k gather is transient under chunked
   recompute, so this trades gather FLOPs for retrieval quality, not memory.
3. **`L = 20`, `d = 1344`** (conversion doc: `L=16`, `d=1408`).
4. **`engram_head_dim = 64`** (conversion doc: 128), capping Engram at 8.7%.
5. **`num_nextn_predict_layers = 0` during pretraining** — DSpark is constructed on
   demand by `attach_dspark()` and trained in a separate frozen-backbone stage. It
   contributes nothing to pretraining throughput.

The 503,795,816 headline is the model **with DSpark attached**. Stage A pretrains
484,266,909 parameters; stage B adds the drafter.

## Layout

```
config.py       503.8M config; validate() enforces ~25 structural invariants at construction
params.py       exact integer accounting; reproduces the MoE preset as a GATE before trusting dense numbers
amp.py          autocast mode registry, dtype resolution, GradScaler factory, cache dtype
quant.py        FP8 E4M3 + FP4 E2M1 fake-quant, straight-through estimator, UE8M0 bit-exact
rope.py         partial RoPE + YaRN; RMSNorm + rms_recip (fp32 reduction, fp32-safe eps)
hc.py           hyper-connections: Sinkhorn (shipped) and exact Birkhoff (mHC-lite)
sparse_attn.py  chunked gather + logsumexp recompute; grads match the reference to 1.3e-15
attention.py    shared-KV latent, sliding window, CSA2 compressor/indexer, hierarchical candidates
engram.py       prime layout, n-gram hash state, gated additive write
ffn.py          clamped SwiGLU (the whole FFN; the router is deleted)
dspark.py       drafter attention, Markov head, confidence head
model.py        Block/Transformer, tied VertexHead, deferred DSpark, segment-2 checkpointing
init_weights.py the initialization the spec omits
optim.py        Muon + AdamW + Sinkhorn, per spec 19.2; per-group lr_mult so a
                schedule scales the partition instead of collapsing it
factory.py      shared model construction + checkpoint loading (train/eval/bench/tests
                used to each carry their own copy, including an attach/load-order bug)
train/          stage A/B loops (schedule, losses, checkpoint+TRAIN state, dist,
                metrics), exact resume, periodic/best checkpoints, in-loop eval,
                grad clip + accumulation, early exits, signal save-then-exit.
                `python -m vertex500m.train` still works via train/__main__.py
dspark_loss.py  stage-B objective (the backbone loss is not differentiable when frozen)
data/           shard I/O (lazy mmap) + document packing + HF-datasets download +
                seeded train/val splitting + manifest.json. `from vertex500m.data
                import build_dataloader` still works; `data.py` moved to data/shards.py
tokenize.py     qt-VI.1.3 → {name}_{train|val}_*.bin + .idx.npy doc index + manifest.json
eval.py         held-out CE / perplexity, chunked; --with-dspark reports acceptance;
                --split now REFUSES non-matching splits instead of silently scoring all shards
generate.py     sampling + speculative accept/reject, with cache rollback
bench.py        decode throughput, DSpark break-even, and `--train` (AMP sweep)
pipeline/       the automation layer: run_config (YAML schema + validation),
                preflight (fail-fast checks), tracker (wandb via metrics tailing),
                orchestrator (idempotent stages, subprocess isolation, state.json),
                report (report.md + summary.json). `python -m vertex500m.pipeline`
tests.py        architecture acceptance suite + mixed-precision group
test_checkpoint.py  save/load round-trip (both hc variants) + stage-B freeze +
                real optimizer-state survival
test_pipeline.py    data pipeline + training QoL + DDP + orchestrator tests
```

## Bugs found in the source docs

Worth fixing upstream; each cost real debugging time here.

1. **`vertex500m-dense.md:499` is wrong by 128.** It lists 540,928 compressor params;
   the §5.1 counting script yields 541,056 — a missing `+hd` on the ratio-1 compressor.
2. **The `swiglu_limit` justification is unsupported.** §10.3 says the clamp keeps FFN
   activations inside the fake-quant dynamic range. In the reference implementation the
   FFN is *never* passed to `act_quant`/`fp4_act_quant` — only the KV cache is. The clamp
   is kept (harmless, one comparison) but the stated reason does not hold.
3. **Spec §11.2's n-gram range is wrong.** "for each ngram_size in 2..(max_ngram-1)"
   yields 2 sizes; `n_hash_cols = (max_ngram-1)*n_heads = 12` requires 3. The reference
   *code* is right and the prose is wrong. This implementation follows the code.
4. **`forward_spec`'s `main_hidden` shape is undocumented.** At decode it must be the
   last position only, `[B,1,d·n_targets]`. Passing the full prefill sequence overruns
   the RoPE table.
5. **Spec §458 (and `mHC-lite.md` throughout) quotes `d=512` for the vertex500m preset.**
   The preset is `d=1344`. The cost conclusion survives — ~26% coefficient-path overhead
   rather than ~64% — but every number in `mHC-lite.md` §5.1/§5.3 was computed for the
   MoE preset in `vertex500m-dense.md` §21.1. Restated in `mHC-lite.md` §8.1.

## Bugs found in this implementation

Recorded so they are not reintroduced; all are covered by a test now.

1. **Speculative decoding corrupted the KV cache.** The verify forward commits
   `verify_w` positions but only `1 + accepted` are emitted, and nothing rolled the
   remainder back — so rejected drafts' KV stayed in the window ring, in
   `compress_kv`/`index_k` and in the Engram n-gram history. Odd accept counts tripped
   a `PoolState desync` assert; even ones silently produced wrong logits (2.08 absolute,
   wrong argmax). Fixed by `snapshot_decode_state` / `restore_decode_state` plus
   re-running the survivors.
2. **The accept loop read the wrong logits.** It used `model()` (last position only) and
   indexed `logits[0, i]` into the *vocab* axis, comparing a token id against a logit
   value. Acceptance was therefore always 0 and the loop was pure overhead. Needs
   `forward_full`.
3. **A full accept re-emitted a duplicate token.** The bonus token was indexed at
   `min(n_keep, B_blk-1)`, but slot `B_blk-1` matched by construction. The verify now runs
   one column wider than the block.
4. **`max_new` bounded rounds, not tokens,** so a partially-accepting drafter overran the
   caches and returned more tokens than requested.
5. **`mhc-lite` could not be trained** — `hc_variant` was threaded through `model`,
   `eval` and `bench` but never through `train`. `params.py` also hardcoded the mHC
   projection width, under-reporting every mHC-lite checkpoint by 1.72M.
6. **The LR schedule collapsed the optimizer partition.** `g["lr"] = lr` overwrote every
   group with the Muon LR, handing AdamW's norm/sink/hc parameters (built for 2.6e-4) a
   77x-too-large step. Groups now carry `lr_mult`.
7. **Stage B ignored `--out`** (it returned before the save block) and no stage saved
   optimizer state, so there was no true resume.
8. **The CE loss was never chunked** despite three docstrings saying it must be. At
   `cfg.max_seq_len` the fp32 logits alone are ~12.6 GB.
9. **DSpark's confidence head never trained** — `survived` was computed and discarded, so
   `confidence_head` gradients were permanently `None`.
10. **`eval.py --with-dspark` reported nothing.** It attached DSpark, then ran plain CE.

## Bugs found during the automation work

Same deal as above: each is a crash or a silent wrong answer, each now has a
test that fails without the fix.

1. **Checkpoints never actually saved the optimizer state.** The payload
    wrapped `o.state_dict()` once more (`{"state": o.state_dict(), ...}`),
    which `load_state_dict` validates past and then silently ignores -- its
    keys never match a parameter. Resume logged "restored optimizer state"
    while every momentum buffer restarted from zero, diverging the run within
    steps. The payload is now `o.state_dict()` verbatim (which also preserves
    the custom `lr_mult` groups), with an unwrapper for legacy checkpoints.
2. **The covering test never tested it.** `test_optimizer_state_roundtrip`
    round-tripped MODEL weights, built FRESH optimizers on both sides, warmed
    both for 2 steps (washing out the momentum difference), and compared --
    so bug 1 passed the suite. `test_optimizer_state_survives_real_checkpoint`
    now saves through the real path and requires a bit-identical next step.
3. **Stage-B resume threw the trained drafter away.** `--init full.pt`
    loaded into a backbone without `mtp.*` (every drafter weight "unexpected")
    and then `init_new_dspark_params` re-randomised all of it. Stage B now
    attaches BEFORE loading and initialises only the `mtp.*` names the
    checkpoint did not supply (a fresh stage B behaves exactly as before).
4. **DDP trained on duplicated data.** Only the weight init was rank-seeded;
    every rank shuffled identically, so the allreduce averaged N copies of
    the same batch. Ranks now get fully-seeded RNGs plus a rank-sharded
    DistributedSampler (advanced per epoch, resume-aware).
5. **Dataloader generator state cannot be resumed.** `iter(loader)` draws
    worker seeds eagerly while the shuffle permutation is drawn lazily on
    first `next()`, so restoring a saved generator state and calling `iter()`
    again draws a DIFFERENT permutation (pinned by a failing test). Resume is
    now epoch-based: each epoch reseeds `seed + epoch`, so both sides perform
    the identical draw sequence.
6. **`persistent_workers=True` breaks epoch determinism.** Worker processes
    carry state across iterators, so a reseeded epoch-1 order did not match a
    fresh one. Workers now respawn per epoch (seconds per epoch, negligible
    next to training).
7. **The pipeline's eval built the wrong model.** `eval_argv` did not forward
    `model.tiny`, so a smoke run evaluated tiny weights with the full 484M
    model (size-mismatch crash). Caught by the smoke run itself.
8. **`threading.Thread` owns `_stop`.** The wandb tailer thread named its stop
    `Event` `_stop`, which shadows the internal method `join()` calls --
    `TypeError` on finish. Renamed.
9. **Preflight ignored `pipeline.run_tests: false`.** The skip logic only
    looked at the passing record, so smoke runs paid for the full suites.
    Caught by the smoke run itself.

## Bugs found during the mixed-precision migration

Recorded for the same reason as the above: each is a crash or a silent wrong answer, and
each is now covered by a test in `tests.py`.

1. **`--amp fp16` died in the first attention layer.** The row-max floor was the literal
   `-1e30`, which overflows fp16 (`clamp_min(-1e30)` on a half tensor raises
   `value cannot be converted to type at::Half without overflow`). The floor is now
   `torch.finfo(score.dtype).min` — finite in every dtype, which is the only property the
   literal ever had.
2. **Autocast is enabled in a custom `autograd.Function`'s forward and DISABLED in its
   backward.** Forward evaluated the attention scores in fp16 while the recompute
   differentiated an fp32 evaluation: a derivative of a function that was never run.
   Fixed with `custom_fwd`/`custom_bwd`.
3. **fp32 KV caches silently demoted the attention core.** `Attention.init_caches`
   hard-coded `dtype=torch.float32`; `Attention.forward` concatenates `window_kv` with
   fresh KV, so an fp32 operand promoted the whole thing and 26% of the model's FLOPs
   never reached the tensor cores.
4. **The fp32 embedding kept the whole residual stream fp32.** `nn.Embedding` is not an
   autocast op, and its output is consumed by `hc_pre`/`hc_post` — pure elementwise ops,
   not eligible ops — before any Linear sees it. Every `comb * residual` re-widened an
   fp32 `[b,s,hc,d]` tensor.
5. **The indexer score could overflow fp16, silently.** It is a ReLU-weighted sum over 8
   heads of 64-dim dots and is `O(T)` per query — the largest tensor in the forward. An
   overflow saturates to inf, `topk` ranks it first, and retrieval goes wrong with no
   NaN and no exception. The constant score scale is now folded into `q` before the dot
   product, which is algebraically exact (`relu(c·x) = c·relu(x)` for `c > 0`) and free.
6. **`get_chunk_window_topk_idxs` was a Python loop over `seqlen`** emitting six kernels
   each — ~12k launches per call at S=2048, times 20 layers, and a guaranteed dynamo
   graph break. Now one vectorised `[seqlen, window_size]` computation, pinned against
   the loop form by a test.

## Performance notes

- **FLOP mix per layer: FFN 73.2%, attention 26.1%, hyper-connections 0.7%.** But the
  hyper-connection residual plumbing (`hc_pre`/`hc_post`/`hc_mixes`) was emitting
  ~3.4 MB/token of fp32 activations for that 0.7% of the arithmetic — more than the
  README's entire uncheckpointed activation budget. That is why autocast alone was not
  enough: it halves the bytes, but you have to stop writing them first.
- **`--amp` must match the cache dtype, everywhere.** Cache dtype is chosen from the
  global AMP mode at `init_caches` time, while compute dtype is chosen by the ambient
  autocast context. Anything that runs the model with the caches already allocated but
  outside an autocast region gets fp16 KV against fp32 activations, which is a hard
  dtype error inside the einsum. `train.py`'s acceptance loop and both `eval.py` loops
  hit exactly this and are now inside the region.
- **`torch.compile` (`--compile`) now traces the whole model into one graph.**
  Previously the forward traced into ~20 graphs and the backward ran eager, because
  `_SparseAttention` was a bare `autograd.Function` (untraceable `requires_grad_()`
  leaves, plus an untraceable `torch._C._is_autocast_available` call in the AMP
  decorator lookup that broke the graph once per layer). Three changes fixed it:
  the kernel is now a `torch.library.custom_op` with a closed-form chunked backward
  (no per-chunk autograd rebuild); autocast state is read through traceable calls
  only (`amp.current_autocast_dtype`); and in-place cache writes run under
  `torch.no_grad()` (a compiled backward cannot replay a grad edge through mutated
  storage -- see `Attention.window_branch`). Measured on `PRESET.tiny()`:
  20 graphs / 19 breaks -> **1 graph / 0 breaks**, with repeated compiled fwd+bwd
  matching eager grads exactly. Two caveats remain: inductor still lowers the custom
  op as an eager fallback (no fusion inside the attention kernel without a custom
  lowering), and `--compile` is validated for repeated steps on CPU here -- A/B it on
  the GPU box with `bench.py --train [--compile]` (which accepts `--compile` for
  exactly that comparison) before trusting a long run.
- **The compressor/indexer grad edge no longer passes through the caches.**
  `compress_kv`/`index_k` are written under `no_grad` (decode persistence only);
  training consumers read the producer's fresh values via `shared.compress_fresh` /
  `shared.index_fresh` instead. Same values, pure functional grad edge, replay-safe
  under compile and checkpointing. Verified: the 5 compressor params keep their
  gradients (the 10 indexer params never had any -- `topk` returns integers -- which
  predates this change).
- **The training loss no longer materialises `[B,S,V]` logits.**
  `forward_full(return_hidden=True)` returns normed hidden states and
  `train.chunked_head_ce` runs head + cross-entropy one position-chunk at a time
  (each chunk checkpointed, so the transient stays bounded and the backbone still
  replays once per step). Bit-identical loss to `chunked_ce(forward_full(...))`.
  At B=4, S=8192 this removes the ~2.1 GB fp16 logit tensor; only hidden
  ([B,S,d], 24x smaller) is retained. `eval.py` keeps the `forward_full` +
  `chunked_ce` path.
- **The indexer score/topk is chunked over queries (`cfg.index_chunk`, default 256).**
  The `[b,S,H,R]` score tensor was the largest transient in the forward (~268 MB
  fp16 per temporary at S=R=8192, several live at once); chunking bounds it to
  `[b,chunk,H,R]` with bit-identical indices (every op is per-query independent).
- **`init_caches` no longer reallocates every step.** Repeated calls with identical
  geometry clear the live buffers in place (`Attention`/`DSparkAttention`, plus a
  `PoolState.reset` that refills the score buffer to `-inf`, not zero). The training
  loop's per-step `init_caches` is now memsets instead of allocator churn.
- **Muon updates batch their elementwise work** (`torch._foreach_*` for momentum,
  Nesterov combine, weight decay and the final add; Newton-Schulz stays per matrix).
  Bit-identical to the loop. Its Newton-Schulz iteration runs in fp16 on CUDA
  instead of bf16: the T4 has no bf16 tensor cores (bf16 ran at ~fp32 speed) and
  fp16 carries more mantissa, so this is faster *and* more accurate there.
- **Smaller traffic fixes:** the attention output projection and the LM head no
  longer upcast to fp32 under autocast (the einsum/`F.linear` lower anyway);
  `hc_post` computes the residual mix as one `comb @ residual` matmul instead of
  materialising a 4x `[b,s,hc,hc,d]` intermediate; the window ring writes go out in
  one `index_copy_` instead of up to 128 scalar writes; the FP4 lookup tables are
  moved to-device once and cached; the dataloader uses `persistent_workers` +
  deeper prefetch with `non_blocking` H2D copies; `--amp` defaults to fp16 when
  CUDA is present.

## Known limitations

- **`token_types` / vision raises `NotImplementedError`.** `vision_n_layers=0`; text-only.
- **`dspark_loss.py` uses a first-order objective.** Every draft slot is scored against
  the same teacher distribution; a full implementation re-scores each slot conditioned on
  the previously accepted drafts. At `block_size=4` this captures the dominant signal but
  is a simplification, and it is documented as one.
- **DSpark's confidence head is trained but not used to gate acceptance by default.**
  Spec §12.1 needs "profiled engine throughput curves" to choose verification length;
  those don't exist. Gating on it is opt-in via `generate(use_confidence=True)`.
- **Speculative decoding runs batch-of-1 semantics per round.** All rows advance by the
  minimum accepted length across the batch, so rows that accepted more tokens are not
  credited for them. Correct, just not maximally efficient at B > 1.
- **32K context will be untrained.** Activations make `T=16384` need checkpointing; plan
  for `T=8192` and treat longer contexts as extrapolation.
- **`embed.weight` uses plain AdamW, not the released momentum+Sinkhorn algorithm.**
  The spec's algorithm assumes a table that is *either* an embedding *or* a head; with
  tying it is both, and that case is untested (`vertex500m-dense.md` §20.5).
- **`--split` used to fall back to all shards** if the substring matched nothing,
  so a "val" evaluation could silently run on training data. It now REFUSES
  (pass `--split all` to select everything explicitly).
- **`--amp bf16` is a trap on a T4.** sm_75 has no bf16 tensor cores; it computes at
  roughly FP32 speed with doubled activation bytes. Correct, just pointless — use fp16.
  (The pipeline turns this into a hard preflight error instead of a warning.)
- **The GradScaler's inf-skip rate for the real model is unmeasured.** What was measured:
  on `PRESET.tiny()` with CPU fp16, 0 skips in 36 steps at `init_scale` 65536 / 8192 /
  2048, but 2 skips in 25 steps in one differently-drawn run with checkpointing on. So
  overflow is marginal and data-dependent at this scale. The scaled fp16 gradients
  flowing through `_SparseAttention.backward` peak around magnitude 12 against fp16's
  65504 — three to four orders of magnitude of headroom, not an order. None of that
  transfers to the 503.8M model. `bench.py --train` prints the rate; if it is above ~5%,
  `--amp fp16` is the wrong tool here and that should be recorded.
- **Multi-GPU is fixed but 2-proc validation still needs a GPU box.**
  `wrap_ddp` uses `static_graph=True`
  specifically because the tied `embed.weight` receives two autograd contributions per
  forward and the default reducer can fire between them, dropping the head's gradient
  with no error. `tests.py::test_ddp_tied_embedding_reduces_once` pins this against a
  single-process reference, but it SKIPs wherever a process group cannot be created —
  which includes the CPU-only build this was developed on (gloo cannot resolve a device).
  The identical-shuffle bug is fixed (rank seeding + DistributedSampler, unit-pinned)
  and the pipeline can launch stages under torchrun itself (`train.dist_nproc`),
  but run the 2-proc path on the GPU box before trusting a 2xT4 run.
- **T4s have no NVLink.** Two of them allreduce ~9 GB of fp32 gradients and parameters
  per step over PCIe. Measure `bench.py --train` single- vs dual-GPU before committing;
  if the allreplace dominates, 1xT4 with a larger batch is the better use of the budget.
- **The rollback acceptance test is marginal on CPU.** `tests.py`'s "rollback +
  re-run matches a clean incremental decode" band (2e-2 relative) was tuned on a
  different BLAS; this CPU box measures 2.098e-02 -- accumulation-order noise on
  random tokens, while the BIT-exact restore check directly above it passes. Since
  the pipeline's preflight gates on green suites, run the suites on the target box,
  or set `pipeline.run_tests: false` after dev verification.
- **Config is settled before deployment, by design.** The pipeline takes exactly
  one input (`--config`), never prompts (preflight audits for `input()`), and a
  resumed run with a changed config refuses unless passed `--allow-config-change`.
  `tracking.mode: online` needs `WANDB_API_KEY` (or `wandb login`) before deploy;
  `offline` records locally for a later `wandb sync`.
