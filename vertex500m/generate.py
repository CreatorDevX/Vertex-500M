"""Text generation, with and without DSpark speculative decoding.

The spec's `generate()` (21.15) only does prefill + 1-token decode and leaves the
speculative acceptance loop as a COMMENT ("a production loop accepts the longest prefix
whose per-position confidence exceeds the threshold; see 12.1"). This implements it.

How speculative decoding actually works, since the spec does not spell it out:

  1. The backbone emits token t and its logits -> the next token's distribution.
  2. The drafter proposes d_1..d_B conditioned on t and the target-layer hidden states.
  3. The backbone verifies ALL B draft positions in ONE forward (this is the whole
     trick: B positions cost barely more than 1, because decode is launch-bound).
  4. Walk the drafts left to right. Accept draft i while it equals the backbone's
     argmax at that position. On the first mismatch, emit the backbone's own token
     there and DISCARD the rest of the block.
  5. Tokens_per_round = 1 + (number accepted).

That last point is why acceptance rate maps directly to speedup, and why bench.py
computes a break-even acceptance from measured costs.

The verify is B+1 columns wide (padded to even), and the caches are ROLLED BACK and
the accepted tokens re-run afterwards. Both are required, and both were wrong before:

  * Step 3 commits `verify_w` positions of KV, but only `1 + accepted` are emitted.
    Skipping the rollback leaves the REJECTED drafts in the window ring, in
    `compress_kv`/`index_k` and in the Engram n-gram history -- every later query then
    attends to tokens that were never emitted. Nothing raises; generation just decays.
    `Transformer.snapshot_decode_state` / `restore_decode_state` implement the undo.
  * The verify needs one column BEYOND the block. On a full accept, the token emitted
    after the block is the backbone's prediction at slot B, not at slot B-1 -- slot B-1
    matched by construction, so indexing it re-emits a duplicate of the token just
    emitted.

The confidence head is trained but NOT used to gate acceptance here. Spec 12.1 says the
scheduler "uses prefix-survival estimates plus profiled engine throughput curves to
choose the verification length per request", and no such curves exist. Gating on an
uncalibrated confidence head would be worse than plain acceptance, so the default is the
standard greedy-prefix rule, with `--use-confidence` available for experiments.
"""

from __future__ import annotations

import torch

from .model import sample


@torch.no_grad()
def generate(model, prompt_ids, max_new=64, temperature=1.0,
             use_dspark=False, use_confidence=False, conf_threshold=0.5,
             eos_id=None, progress=False, amp_mode=None):
    """Returns (token_ids [B, prompt+new], n_rounds, n_accepted).

    `max_new` bounds TOKENS EMITTED, not rounds. A speculative round emits
    `1 + accepted` tokens, so bounding rounds instead lets a partially-accepting
    drafter run `max_new` rounds of up to `block_size + 1` tokens each -- overrunning
    the caches (the Engram id history is sized from `max_new`) and returning more
    tokens than asked for.

    `start_pos` is threaded explicitly everywhere; the CSA2 caches are NOT reset between
    steps because the window ring and compressor PoolState must carry across a single
    generation (they are reset between independent sequences).

    `amp_mode` ("off"/"fp16"/"bf16") must match how the caches were allocated, which is
    decided by the global AMP mode. Passing it here rather than relying on the caller is
    deliberate: `init_caches` reads the mode at allocation time, so a model whose caches
    are fp16 fed fp32 activations raises a dtype mismatch deep inside attention instead of
    quietly producing fp32 logits.
    """
    from .amp import amp_context
    model.eval()
    cfg = model.cfg
    device = prompt_ids.device
    B = prompt_ids.size(0)
    S = prompt_ids.size(1)
    # The verify forward writes `verify_w` positions past `pos`. The final round may
    # start at S + max_new - 1 and still verify `verify_w` columns ahead, so the caches
    # must hold S + max_new + verify_w.
    verify_w = cfg.dspark_block_size + 1 + (cfg.dspark_block_size % 2)

    with amp_context(amp_mode, device.type):
        model.init_caches(B, S + max_new + verify_w + 4, device=device)

        ids = prompt_ids
        logits, main_hidden = model(ids, start_pos=0)          # prefill
        if use_dspark:
            # seed the drafter's window ring from the prefill (spec 12.2)
            model.forward_spec(logits.argmax(-1, keepdim=True), main_hidden, 0)

        n_rounds = n_accepted = 0
        pos = S

        while ids.size(1) - S < max_new:
            n_rounds += 1
            if not use_dspark:
                nxt = sample(logits, temperature).reshape(B, 1)
                ids = torch.cat([ids, nxt], dim=1)
                pos += 1
                logits, main_hidden = model(nxt, start_pos=pos - 1)
                if eos_id is not None and bool((nxt == eos_id).all()):
                    break
                continue

            # ---- speculative round ---------------------------------------
            B_blk = cfg.dspark_block_size
            trigger = logits.argmax(-1, keepdim=True)          # [B,1]
            spec = model.forward_spec(trigger, main_hidden[:, -1:, :], pos)
            if spec is None:                                   # not seeded yet
                nxt = sample(logits, temperature).reshape(B, 1)
                ids = torch.cat([ids, nxt], dim=1); pos += 1
                logits, main_hidden = model(nxt, start_pos=pos - 1)
                continue
            draft, _, conf = spec                              # draft [B, B_blk+1]
            # slot 0 of draft is the trigger itself; slots 1.. are the proposals
            proposals = draft[:, 1:]                            # [B, B_blk]

            # ---- verify the whole block in ONE backbone forward --------
            #
            # Two width constraints, both load-bearing:
            #
            # 1. PoolState parity. A ratio-2 compressor emits on even positions and its
            #    `count` must equal `start_pos % ratio` on entry, so the verified span
            #    must be EVEN-width or the next round desyncs (attention.py asserts).
            #
            # 2. The bonus token. On a FULL accept every draft slot matches the
            #    backbone's argmax, so the token emitted after the block is the
            #    backbone's prediction at slot `B_blk` -- which does not exist unless
            #    the verify ran one column wider than the block. Clamping the index to
            #    `B_blk - 1` instead (the old behaviour) re-emits `proposals[B_blk-1]`,
            #    a duplicate of the token just emitted, because slot B_blk-1 matched by
            #    construction.
            #
            # So: verify B_blk + 1 columns, padded up to an even width. The extra
            # columns are free in an already-batched forward and their inputs are
            # discarded.
            verify_w = B_blk + 1
            if verify_w % 2:
                verify_w += 1
            fillers = proposals[:, -1:].clone().expand(B, verify_w - B_blk)
            verify_in = torch.cat([proposals, fillers], dim=1)

            # Snapshot BEFORE the verify: it commits `verify_w` positions of KV, of
            # which only `emit` survive. See Transformer.snapshot_decode_state.
            snap = model.snapshot_decode_state(pos, verify_w)
            # forward_full, NOT forward: the accept loop needs the backbone's
            # distribution at EVERY verified position, and `forward` returns only the
            # last one.
            logits, main_hidden = model.forward_full(verify_in, start_pos=pos)

            # `logits[:, i]` is the backbone's distribution for the token following
            # draft slot i. Accept the longest prefix that matches; on the first
            # mismatch the backbone's own token is emitted there and the rest of the
            # block is dropped. Column B_blk is NOT part of the block -- it is the
            # post-block bonus prediction, reachable only because the verify ran one
            # column wider.
            greedy_all = logits[:, :B_blk + 1].argmax(-1)          # [B, B_blk+1]
            greedy = greedy_all[:, :B_blk]
            ok = proposals.eq(greedy)
            gate = None
            if use_confidence and conf is not None:
                gate = (conf.sigmoid() > conf_threshold).long().cummin(dim=1).values
                ok = ok & gate.bool()
            # rows may stop at different slots; emit the common prefix so the batch
            # stays rectangular. No row is ever advanced past what it accepted.
            keep_per_row = ok.to(torch.int32).cumprod(-1).sum(-1)        # [B]
            n_keep = int(keep_per_row.min())
            n_accepted += n_keep

            emit = torch.cat([proposals[:, :n_keep], greedy_all[:, [n_keep]]],
                             dim=1)                                       # [B, n_keep+1]
            # the last round may overshoot max_new; truncate the block, keeping the
            # accepted prefix and dropping the trailing bonus token if it does not fit
            room = max_new - (ids.size(1) - S)
            if emit.size(1) > room:
                emit = emit[:, :room]
            if not emit.size(1):                    # nothing left to emit; done
                break
            ids = torch.cat([ids, emit], dim=1)
            pos += emit.size(1)

            if progress:
                print(f"  round {n_rounds}: accepted {n_keep}/{B_blk} (min over batch)")

            if eos_id is not None:
                if bool((emit == eos_id).any()):
                    break

            # Roll the caches back to the pre-verify state and re-run the ACCEPTED
            # tokens in a single forward. This is what advances the caches by exactly
            # `emit` positions and gives the next round its logits -- re-running the
            # survivors (rather than patching the last slot) is what keeps
            # `compress_kv`, the Engram n-gram history and `PoolState` free of the
            # rejected drafts.
            model.restore_decode_state(snap)
            logits, main_hidden = model(emit, start_pos=pos - emit.size(1))
            if ids.size(1) - S >= max_new:
                break
            model.forward_spec(logits.argmax(-1, keepdim=True),
                               main_hidden[:, -1:, :], pos)

    return ids, n_rounds, n_accepted


def decode(model, tokenizer, prompt: str, **kw) -> str:
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
        next(model.parameters()).device)
    out, rounds, acc = generate(model, ids, **kw)
    return tokenizer.decode(out[0], skip_special_tokens=True)
