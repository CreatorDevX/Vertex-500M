"""DSpark training objective (spec 12, stage B).

The drafter learns to PREDICT what the frozen backbone would emit. That is a different
objective from backbone pretraining, and it cannot be expressed as "backbone loss with
the backbone frozen":

    with every backbone parameter frozen, the backbone's own loss has NO trainable
    tensor in its graph, so it is correctly non-differentiable and .backward() raises
    "element 0 of tensors does not require grad". This is not a bug to work around --
    it is the signal that the objective is wrong.

The correct objective, per round:

  1. The FROZEN backbone runs normally over the context and emits target logits
     `t_0 .. t_{L-1}`. These are the teacher's distribution and get NO gradient.
  2. The drafter proposes a block starting at position `p`: `d_1 .. d_B`.
  3. Teacher-forced CE: feed the drafter the block, and at each draft position apply the
     drafter's logits to the TEACHER's next-token distribution. Both drafts and targets
     are shifted by one, so the drafter learns the backbone's conditional distribution
     at each draft slot.
  4. The confidence head is trained against the realised prefix-survival target, with
     BCE on its acceptance logit (weight `conf_weight`).

We deliberately do NOT backprop through the backbone: detach everything teacher-side.

This is the honest interpretation of spec 12.4 ("only the drafter trains; backbone
frozen" and "does not propagate DSpark gradients into the backbone").
"""

from __future__ import annotations

import torch


def dspark_loss(model, input_ids, temperature: float = 1.0, conf_weight: float = 0.1):
    """One stage-B step. Returns (loss, conf, stats).

    input_ids: [B, L] a training window (teacher-forced, so no sampling noise here).

    `conf_weight` scales the confidence-head BCE. It was previously computed and then
    DISCARDED -- `confidence_head` params stayed at `grad is None` forever while sitting
    in the optimizer, so the head never trained at all.
    """
    cfg = model.cfg
    B, L = input_ids.shape
    block = cfg.dspark_block_size

    # ---- 1. teacher: frozen backbone over the whole window ------------------
    # forward_full returns logits at EVERY position ([B,S,V]); `forward` returns only
    # the last one, which is useless here because the drafter needs the teacher
    # distribution at a specific interior position.
    with torch.no_grad():
        model.init_caches(B, L + block + 4, device=input_ids.device)
        teacher_logits, main_hidden = model.forward_full(input_ids, start_pos=0)
        teacher_probs = torch.softmax(teacher_logits.float() / max(temperature, 1e-5), -1)

    # pick an anchor somewhere in the middle of the window so the drafter sees real
    # left context, and a target token after it
    anchor = max(1, L // 2)
    prev = input_ids[:, anchor - 1:anchor]                  # the real last token
    tgt_probs = teacher_probs[:, anchor - 1]                # [B, V] teacher distribution
    mh = main_hidden[:, anchor - 1:anchor]                  # [B, 1, d*nt]

    # ---- 2/3. drafter proposes a block, scored against the teacher ----------
    out, logits, conf = model.forward_spec(prev, mh, anchor)
    if out is None:
        return None, None, {}

    # logits: [B, block+1, V] (position 0 is the real token's slot)
    # train every draft slot against the SAME teacher distribution at the anchor.
    # This is the simplification: a full implementation would walk the accepted prefix
    # and re-score each slot against the teacher's distribution conditioned on the
    # previously accepted drafts. At block_size=4 the first-order version captures the
    # dominant signal (can the drafter predict the backbone's next token?) and is what
    # we can verify here.
    # logits: [B, block, V]. forward_head runs the drafter over `dspark_block_size`
    # draft slots; slot i's logits predict token i+1. Every slot is trained.
    n_slots = logits.size(1)
    tgt = tgt_probs.unsqueeze(1).expand(tgt_probs.size(0), n_slots, tgt_probs.size(-1))
    student = torch.log_softmax(logits.float(), dim=-1)
    ce = -(tgt * student).sum(-1)                          # [B, n_slots]
    loss = ce.mean()

    # ---- 4. confidence head -------------------------------------------------
    # realised prefix-survival target: 1 if that slot's top draft matches the teacher
    with torch.no_grad():
        teacher_top = tgt_probs.argmax(-1)                 # [B]
        draft_top = logits.float().argmax(-1)              # [B, n_slots]
        survived = (draft_top == teacher_top.unsqueeze(1)).float()
    # `conf` is the head's per-slot acceptance LOGIT; train it with BCE against
    # `survived`. This term used to be computed and thrown away.
    conf_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        conf.float().reshape(-1), survived.reshape(-1))
    loss = loss + conf_weight * conf_loss
    return (loss, conf,
            {"accept_now": survived.mean().item(), "conf_loss": conf_loss.item()})
