"""Beam search that runs entirely on the GPU, inside the decode-step CUDA graph.

Semantics are those of CTranslate2's ``BeamSearch::search`` for Whisper (the
CPU port in ``engine.py`` is the reference): 2*beam candidates per item,
finished hypotheses registered in candidate order and replaced by the next
non-EOS secondary candidate, patience / early-exit rules, and the Whisper
logits processors (suppress tokens, suppress blank, timestamp rules). Hard
prefixes, repetition penalty and no-repeat-ngram are left to the CPU path.

All state lives in persistent device buffers so one captured graph serves every
``generate`` call; per-call options are device scalars written before decoding.

Ported from Whisper-WebUI Premium (SECourses), modules/whisper/convrot at revision 613beba (September 2026).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

NEG = float(torch.finfo(torch.float32).min)


@triton.jit
def _reorder_copy_kernel(SRC_BUF, DST_BUF, SRC_ROWS, LO, HI, stride_lk, stride_b, stride_s,
                         ROW_WIDTH: tl.constexpr, BLOCK_S: tl.constexpr, GATHER: tl.constexpr):
    """Copy positions [LO, HI) of every row whose source row differs.

    GATHER: DST_BUF[lk, r] <- SRC_BUF[lk, src[r]]   (cache -> temp)
    else:   DST_BUF[lk, r] <- SRC_BUF[lk, r]        (temp -> cache)
    """
    lk = tl.program_id(0).to(tl.int64)
    r = tl.program_id(1)
    s0 = tl.program_id(2) * BLOCK_S
    src = tl.load(SRC_ROWS + r)
    lo = tl.load(LO)
    hi = tl.load(HI)
    if src == r or s0 >= hi or s0 + BLOCK_S <= lo:
        return
    offs_s = s0 + tl.arange(0, BLOCK_S)
    smask = (offs_s >= lo) & (offs_s < hi)
    read_row = src if GATHER else r
    base_r = SRC_BUF + lk * stride_lk + read_row.to(tl.int64) * stride_b
    base_w = DST_BUF + lk * stride_lk + tl.cast(r, tl.int64) * stride_b
    for c0 in tl.static_range(0, ROW_WIDTH, 256):
        offs_c = c0 + tl.arange(0, 256)
        ptr_off = offs_s[:, None].to(tl.int64) * stride_s + offs_c[None, :]
        m = smask[:, None] & (offs_c[None, :] < ROW_WIDTH)
        v = tl.load(base_r + ptr_off, mask=m)
        tl.store(base_w + ptr_off, v, mask=m)


class GpuBeamState:
    """Device-side beam search state for one DecodeSession (rows = nb * beam)."""

    def __init__(self, sess, eot: int, no_timestamps: int, timestamp_begin: int, timestamp_end: int,
                 no_speech: int, max_generated: int = 224, max_hypotheses_factor: int = 4):
        rt = sess.rt
        dev = rt.device
        self.sess = sess
        self.nb, self.beam = sess.nb, sess.group
        self.rows = sess.rows
        self.V = rt.dims.n_vocab
        self.eot, self.nt, self.tb, self.te, self.no_speech = eot, no_timestamps, timestamp_begin, timestamp_end, no_speech
        self.S1 = max_generated + 1
        self.maxh = max_hypotheses_factor * self.beam
        nb, beam = self.nb, self.beam
        i64 = torch.long
        self.alive = torch.zeros((nb, beam, self.S1), device=dev, dtype=i64)
        self.scores = torch.zeros((self.rows,), device=dev, dtype=torch.float32)
        self.last_ts = torch.full((self.rows,), -1, device=dev, dtype=i64)
        self.hyp_tok = torch.zeros((nb, self.maxh + 1, self.S1), device=dev, dtype=i64)
        self.hyp_score = torch.zeros((nb, self.maxh + 1), device=dev, dtype=torch.float32)
        self.hyp_len = torch.zeros((nb, self.maxh + 1), device=dev, dtype=i64)
        self.hyp_count = torch.zeros((nb,), device=dev, dtype=i64)
        self.top_finished = torch.zeros((nb,), device=dev, dtype=torch.bool)
        self.done = torch.zeros((nb,), device=dev, dtype=torch.bool)
        self.ns_prob = torch.zeros((self.rows,), device=dev, dtype=torch.float32)
        self.step_t = torch.zeros((), device=dev, dtype=i64)
        # per-call options
        self.max_len = torch.zeros((), device=dev, dtype=i64)
        self.max_cands = torch.zeros((), device=dev, dtype=i64)
        self.num_hyps = torch.zeros((), device=dev, dtype=i64)
        self.early_exit = torch.zeros((), device=dev, dtype=torch.bool)
        self.ts_on = torch.zeros((), device=dev, dtype=torch.bool)
        self.max_init_id = torch.zeros((), device=dev, dtype=i64)
        self.prompt_len = torch.zeros((), device=dev, dtype=i64)
        self.hi_pos = torch.zeros((), device=dev, dtype=i64)
        self.suppress_mask = torch.zeros((self.V,), device=dev, dtype=torch.bool)
        self.begin_mask = torch.zeros((self.V,), device=dev, dtype=torch.bool)
        # constants
        self.arange_v = torch.arange(self.V, device=dev, dtype=i64)
        self.arange_beam = torch.arange(beam, device=dev, dtype=i64)
        self.row_base = (torch.arange(nb, device=dev, dtype=i64) * beam)[:, None]
        self.identity_rows = torch.arange(self.rows, device=dev, dtype=i64)
        self.kv_tmp = torch.empty_like(sess.kv)

    # ------------------------------------------------------------------
    def reset(self, start_ids, max_length: int, max_cands: int, num_hyps: int, length_penalty: float,
              timestamps: bool, max_init_id: int, prompt_len: int, suppress_mask: torch.Tensor,
              begin_mask: torch.Tensor):
        dev = self.alive.device
        beam = self.beam
        self.sess.ids.copy_(torch.as_tensor(start_ids, dtype=torch.long).repeat_interleave(beam).to(dev))
        self.scores.fill_(NEG)
        self.scores[::beam] = 0.0
        self.alive.zero_()
        self.last_ts.fill_(-1)
        self.hyp_count.zero_()
        self.top_finished.zero_()
        self.done.zero_()
        self.step_t.zero_()
        self.max_len.fill_(max_length)
        self.max_cands.fill_(max_cands)
        self.num_hyps.fill_(num_hyps)
        self.early_exit.fill_(length_penalty == 0)
        self.ts_on.fill_(bool(timestamps))
        self.max_init_id.fill_(max_init_id)
        self.prompt_len.fill_(prompt_len)
        self.suppress_mask.copy_(suppress_mask)
        self.begin_mask.copy_(begin_mask)

    # ------------------------------------------------------------------
    def process(self, logits: torch.Tensor):
        """Runs inside the decode graph after the logits of the current step are computed."""
        nb, beam, V, B = self.nb, self.beam, self.V, self.rows
        tb, te, eot, nt = self.tb, self.te, self.eot, self.nt
        step = self.step_t
        lg = logits
        self.ns_prob.copy_(torch.softmax(lg, dim=-1)[:, self.no_speech])
        lg.masked_fill_(self.suppress_mask[None, :], NEG)
        lg.masked_fill_((self.begin_mask & (step == 0))[None, :], NEG)

        # Whisper timestamp rules (CTranslate2 ApplyTimestampRules, no prefix => sample_begin = 0)
        alive_flat = self.alive.view(B, self.S1)
        last = alive_flat.gather(1, (step - 1).clamp(min=0).expand(B, 1)).squeeze(1)
        pen_raw = alive_flat.gather(1, (step - 2).clamp(min=0).expand(B, 1)).squeeze(1)
        pen = torch.where(step >= 2, pen_raw, last)
        has_last = step >= 1
        step0 = step == 0
        last_is_ts = has_last & (last >= tb)
        pen_is_ts = pen >= tb
        case_b = last_is_ts & pen_is_ts
        case_c = last_is_ts & ~pen_is_ts
        case_d = has_last & ~last_is_ts
        case_d_ts = case_d & (self.last_ts >= 0)
        zero = torch.zeros_like(last)
        lo1 = torch.where(step0, zero, torch.where(case_b, zero + tb, torch.where(case_d_ts, zero + tb, zero)))
        hi1 = torch.where(step0, zero + tb, torch.where(case_b, zero + te + 1, torch.where(
            case_c, zero + eot, torch.where(case_d_ts, self.last_ts + 1, zero))))
        lo2 = torch.where(step0, (self.max_init_id + 1).expand(B), torch.where(case_c, zero + tb, zero))
        hi2 = torch.where(step0, zero + te + 1, torch.where(case_c, last, zero))
        ar = self.arange_v[None, :]
        ts_mask = (ar == nt) | ((ar >= lo1[:, None]) & (ar < hi1[:, None])) | ((ar >= lo2[:, None]) & (ar < hi2[:, None]))
        lg.masked_fill_(ts_mask & self.ts_on, NEG)
        check = (case_c | case_d) & self.ts_on
        lp = torch.log_softmax(lg, dim=-1)
        ts_lp = torch.logsumexp(lp[:, tb:], dim=-1)
        max_text = lp[:, :tb].max(dim=-1).values
        force = check & (ts_lp > max_text)
        lg[:, :tb].masked_fill_(force[:, None], NEG)

        # beam expansion
        lp = torch.log_softmax(lg, dim=-1) + self.scores[:, None]
        vals, idx = torch.topk(lp.view(nb, beam * V), 2 * beam, dim=-1)
        words = idx % V
        origin = idx // V

        # register finished hypotheses (top `beam` candidates, candidate order)
        eos = words == eot
        is_last = (step + 1) == self.max_len
        active = ~self.done
        fin = (eos[:, :beam] | is_last) & active[:, None]
        rank = fin.long().cumsum(dim=1) - 1
        slot = torch.where(fin, self.hyp_count[:, None] + rank, torch.full_like(rank, self.maxh))
        hist = self.alive.gather(1, origin[:, :beam, None].expand(nb, beam, self.S1))
        hist.scatter_(2, step.expand(nb, beam, 1), words[:, :beam, None])
        self.hyp_tok.scatter_(1, slot[:, :, None].expand(nb, beam, self.S1), hist)
        self.hyp_score.scatter_(1, slot, vals[:, :beam])
        self.hyp_len.scatter_(1, slot, torch.where(eos[:, :beam], step, step + 1).expand(nb, beam))
        self.top_finished |= fin[:, 0]
        self.hyp_count += fin.long().sum(dim=1)

        # replace finished candidates with the next non-EOS secondary candidates, in order
        sec_ok = ~eos[:, beam:]
        sec_rank = torch.where(sec_ok, sec_ok.long().cumsum(dim=1) - 1, torch.full_like(rank, beam))
        sec_map = torch.full((nb, beam + 1), -1, device=lg.device, dtype=torch.long)
        sec_map.scatter_(1, sec_rank, (self.arange_beam + beam).expand(nb, beam))
        repl = sec_map.gather(1, rank.clamp(min=0, max=beam - 1))
        next_beam = torch.where(fin & (repl >= 0), repl, self.arange_beam.expand(nb, beam))

        # finished items
        done_now = torch.where(self.early_exit, self.top_finished & (self.hyp_count >= self.num_hyps),
                               self.hyp_count >= self.max_cands) | is_last
        self.done |= active & done_now

        # select the active beams for the next step
        w_sel = words.gather(1, next_beam)
        s_sel = vals.gather(1, next_beam)
        o_sel = origin.gather(1, next_beam)
        new_alive = self.alive.gather(1, o_sel[:, :, None].expand(nb, beam, self.S1))
        new_alive.scatter_(2, step.expand(nb, beam, 1), w_sel[:, :, None])
        self.alive.copy_(new_alive)
        prev_ts = self.last_ts.view(nb, beam).gather(1, o_sel)
        self.last_ts.copy_(torch.where(w_sel >= tb, w_sel, prev_ts).view(B))
        self.scores.copy_(s_sel.reshape(B))
        self.sess.ids.copy_(w_sel.reshape(B))

        # reorder the self-attention cache of rows whose parent changed (positions [prompt, prompt+step])
        src = (self.row_base + o_sel).reshape(B)
        src = torch.where(self.done.repeat_interleave(beam), self.identity_rows, src)
        self.hi_pos.copy_(self.prompt_len + step + 1)
        kv = self.sess.kv
        lk, rows, S = kv.shape[0] * kv.shape[1], kv.shape[2], kv.shape[3]
        width = kv.shape[4] * kv.shape[5]
        kv3 = kv.view(lk, rows, S, width)
        tmp3 = self.kv_tmp.view(lk, rows, S, width)
        grid = (lk, rows, triton.cdiv(S, 16))
        _reorder_copy_kernel[grid](kv3, tmp3, src, self.prompt_len, self.hi_pos, kv3.stride(0), kv3.stride(1),
                                   kv3.stride(2), ROW_WIDTH=width, BLOCK_S=16, GATHER=True, num_warps=4)
        _reorder_copy_kernel[grid](tmp3, kv3, src, self.prompt_len, self.hi_pos, kv3.stride(0), kv3.stride(1),
                                   kv3.stride(2), ROW_WIDTH=width, BLOCK_S=16, GATHER=False, num_warps=4)
        self.step_t += 1

    # ------------------------------------------------------------------
    def results(self):
        """Host copy of the registered hypotheses: list per item of (score, tokens)."""
        counts = self.hyp_count.cpu().tolist()
        scores = self.hyp_score.cpu()
        lens = self.hyp_len.cpu()
        toks = self.hyp_tok.cpu()
        out = []
        for i, n in enumerate(counts):
            n = min(int(n), self.maxh)
            out.append([(float(scores[i, j]), toks[i, j, :int(lens[i, j])].tolist()) for j in range(n)])
        return out
