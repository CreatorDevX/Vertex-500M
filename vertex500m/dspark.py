"""DSpark: semi-autoregressive speculative decoding (spec 12).

A small drafter proposes a block of tokens; the backbone verifies them. Per spec 12.4
it is trained in a DEDICATED stage after backbone pre-training, with the backbone
frozen, because the drafter learns to imitate the backbone's output distribution.
Training it jointly would mean chasing a target that moves every step, and would cost
+6.6% FLOPs per pretraining step for zero pretraining throughput.

Because the backbone's `embed` and `head` are shared with the drafter (weight-tied),
the optimizer MUST be built only over `mtp.*` parameters in stage B. Otherwise
DSpark gradients leak into the tied embedding and silently corrupt the backbone. That
is exactly what `train.py`'s backbone-identity assertion checks.
"""

from __future__ import annotations

import torch

from .amp import cache_dtype
from .attention import Attention
from .config import Cfg
from .quant import act_quant
from .rope import RMSNorm, rope_tail
from .sparse_attn import get_dspark_topk_idxs, sparse_attn
from .model import Block, DSparkConfidenceHead, DSparkMarkovHead


class DSparkAttention(Attention):
    """Draft attention over the backbone's window ring plus the draft block's own KV."""

    def __init__(self, cfg: Cfg, layer_id: int):
        super().__init__(cfg, layer_id)
        assert self.ratio == 0, f"draft layer {layer_id} must have compress_ratio == 0"
        # the drafter carries no compressor or indexer
        self.compressor = None
        self.indexer = None
        self.is_kv_source = False
        self.is_index_source = False
        self.compress_kv = None
        self.index_k = None
        self.pool_state = None

    def init_caches(self, batch: int, max_len: int, device=None, dtype=None):
        # `dtype=None` resolves from the AMP mode; see Attention.init_caches for why the
        # drafter's ring must not be left in fp32 under `--amp`.
        dtype = cache_dtype(device) if dtype is None else dtype
        dev = torch.empty(0, device=device).device
        old = getattr(self, "window_kv", None)
        if (old is not None and tuple(old.shape) == (batch, self.window, self.hd)
                and old.dtype == dtype and old.device == dev):
            # Same geometry as the live ring (stage-B calls this per batch):
            # clear in place instead of reallocating. See Attention.init_caches.
            old.zero_()
            return
        self.window_kv = torch.zeros(batch, self.window, self.hd, dtype=dtype,
                                     device=device)

    def forward(self, x: torch.Tensor, start_pos: int, shared, main_x=None):
        b, S, _ = main_x.shape
        main_freqs = self.freqs_cis[start_pos:start_pos + S]
        main_kv = self.kv_norm(self.wkv(main_x))
        main_kv = rope_tail(main_kv, self.rd, main_freqs)
        main_kv, _ = act_quant(main_kv, block=32, round_scale=True, inplace=False)
        if start_pos == 0:
            # prefill: seed the ring only (cache writes under no_grad; see
            # Attention.window_branch for why compiled replays need that)
            if S <= self.window:
                with torch.no_grad():
                    self.window_kv[:b, :S] = main_kv
            else:
                cutoff = S % self.window
                tail = main_kv[:, -self.window:]
                with torch.no_grad():
                    self.window_kv[:b, cutoff:self.window] = tail[:, :self.window - cutoff]
                    self.window_kv[:b, :cutoff] = tail[:, self.window - cutoff:]
            return x

        blk = x.size(1)
        freqs = self.freqs_cis[start_pos + S:start_pos + S + blk]
        q = self.wq_b(self.q_norm(self.wq_a(x))).view(b, blk, self.n_heads, self.hd)
        q = rope_tail(q, self.rd, freqs)
        kv = self.kv_norm(self.wkv(x))
        kv = rope_tail(kv, self.rd, freqs)
        kv, _ = act_quant(kv, block=32, round_scale=True, inplace=False)
        idxs = get_dspark_topk_idxs(self.window, b, blk, start_pos, x.device)
        with torch.no_grad():           # cache write; see Attention.window_branch
            self.window_kv[:b, start_pos % self.window] = main_kv[:, 0]
        kv = torch.cat([self.window_kv[:b], kv], dim=1)
        o = sparse_attn(q, kv, self.attn_sink, idxs, self.softmax_scale,
                        chunk=self.cfg.attn_chunk)
        o = rope_tail(o, self.rd, freqs, inverse=True)
        g = self.cfg.o_groups
        # No `.to(weight.dtype)` upcast; see Attention.forward for why it is waste.
        o = o.view(b, blk, g, -1)
        o = torch.einsum("bsgd,grd->bsgr", o,
                         self.wo_a.weight.view(g, self.cfg.o_lora_rank, -1))
        return self.wo_b(o.flatten(2))


class DSparkBlock(Block):
    """A Block whose attention_cls is DSparkAttention.

    At start_pos == 0 it only seeds the window caches -- no draft attention, no FFN.
    """

    attention_cls = DSparkAttention

    def forward(self, x, start_pos, pre_mix, shared, main_x):
        if start_pos == 0:
            self.attn(x, start_pos, shared, main_x)
            return x, pre_mix
        return Block.forward(self, x, start_pos, pre_mix, shared, main_x)

    def forward_head(self, x, pre_mix, input_ids):
        """Draft a block and return (ids, logits, confidence).

        Autograd discipline (spec invariant 44). Two things must stay out-of-place:
          * the Markov bias must not be written back into `logits` in place;
          * the sampled ids must not be written back into a tensor that an EARLIER
            markov_head call read. `out[:, i+1] = ...` mutates `out`, and `out[:, i]`
            was already fed to markov_head on a previous iteration -- autograd's version
            counter then rejects the whole graph. Keeping the running ids in a Python
            list and stacking at the end removes the aliasing entirely.
        """
        x = hc_pre_local(x, pre_mix)
        logits = self.head(self.norm(x), full_logits=True)
        B = input_ids.size(0)
        sampled = [input_ids.reshape(B)]
        markov = []
        cur = logits
        for i in range(self.cfg.dspark_block_size):
            bias, emb = self.markov_head(sampled[i])
            cur = torch.cat([cur[:, :i], (cur[:, i] + bias).unsqueeze(1), cur[:, i + 1:]],
                            dim=1)
            markov.append(emb)
            probs = torch.softmax(cur[:, i].float() / max(self.cfg.temperature, 1e-5), -1)
            sampled.append(torch.multinomial(probs, 1).reshape(B))
        out = torch.stack(sampled, dim=1)
        conf = self.confidence_head(x, torch.stack(markov, dim=1))
        return out, cur, conf


def hc_pre_local(x, pre_mix):
    from .hc import hc_pre
    return hc_pre(x, pre_mix)


__all__ = ["DSparkAttention", "DSparkBlock", "DSparkMarkovHead", "DSparkConfidenceHead"]
