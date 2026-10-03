"""Split-KV attention for a few queries per row (flash-decoding), for the engine's decode steps without flash-attn.

Whisper-WebUI runs its decoder with flash-attn's ``flash_attn_with_kvcache``. This app's PyTorch has no matching
flash-attn build, and PyTorch's SDPA ran a decode step's tiny attentions (one query per beam against a 448-position
cache, five queries against the 1,500 encoder frames) on a general kernel at about 55 us a call: the engine took
885 ms per 30-second window against Whisper-WebUI's 397 ms. These Triton kernels split the keys across programs,
combine the partial softmax results, and read only each row's valid cache length, as flash-attn does.

``decode_attention(q, k, v, lengths)``: q [B, Q, H, D] with Q <= 16, k and v [B, S, H, D] (any strides with a
contiguous last dimension); ``lengths`` [B] (plus ``length_offset``) limits each row's keys. ``write_kv`` stores a
step's new key and value at each row's length. Both keep static shapes, so they are captured in the decode CUDA graph.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

BLOCK_Q = 16
BLOCK_N = 64
MAX_QUERIES = BLOCK_Q
TARGET_PROGRAMS = 256


@triton.jit
def _partial_kernel(Q, K, V, LENGTHS, ACC, M, L,
                    stride_qb, stride_qq, stride_qh, stride_kb, stride_ks, stride_kh, stride_vb, stride_vs, stride_vh,
                    H, NQ, S, SPLIT_LEN, NUM_SPLITS, LENGTH_OFFSET, scale,
                    HAS_LENGTHS: tl.constexpr, D: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_N: tl.constexpr):
    bh = tl.program_id(0)
    split = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_q = tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, D)
    q_ptrs = Q + b.to(tl.int64) * stride_qb + offs_q[:, None] * stride_qq + h * stride_qh + offs_d[None, :]
    q = tl.load(q_ptrs, mask=(offs_q < NQ)[:, None], other=0.0)
    if HAS_LENGTHS:
        length = tl.minimum(tl.load(LENGTHS + b).to(tl.int32) + LENGTH_OFFSET, S)
    else:
        length = S
    start = split * SPLIT_LEN
    end = tl.minimum(start + SPLIT_LEN, length)
    m_i = tl.full([BLOCK_Q], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_Q], tl.float32)
    acc = tl.zeros([BLOCK_Q, D], tl.float32)
    k_base = K + b.to(tl.int64) * stride_kb + h * stride_kh
    v_base = V + b.to(tl.int64) * stride_vb + h * stride_vh
    for n0 in range(start, end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        n_mask = offs_n < end
        k = tl.load(k_base + offs_n[:, None].to(tl.int64) * stride_ks + offs_d[None, :], mask=n_mask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(n_mask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_base + offs_n[:, None].to(tl.int64) * stride_vs + offs_d[None, :], mask=n_mask[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    slot = bh * NUM_SPLITS + split
    tl.store(M + slot * BLOCK_Q + offs_q, m_i)
    tl.store(L + slot * BLOCK_Q + offs_q, l_i)
    tl.store(ACC + slot.to(tl.int64) * BLOCK_Q * D + offs_q[:, None] * D + offs_d[None, :], acc)


@triton.jit
def _combine_kernel(ACC, M, L, OUT, stride_ob, stride_oq, stride_oh, H, NQ,
                    NUM_SPLITS: tl.constexpr, D: tl.constexpr, BLOCK_Q: tl.constexpr):
    bh = tl.program_id(0)
    b = bh // H
    h = bh % H
    offs_q = tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, D)
    base = bh * NUM_SPLITS
    m = tl.full([BLOCK_Q], float("-inf"), tl.float32)
    for split in tl.static_range(NUM_SPLITS):
        m = tl.maximum(m, tl.load(M + (base + split) * BLOCK_Q + offs_q))
    total = tl.zeros([BLOCK_Q], tl.float32)
    acc = tl.zeros([BLOCK_Q, D], tl.float32)
    for split in tl.static_range(NUM_SPLITS):
        weight = tl.exp(tl.load(M + (base + split) * BLOCK_Q + offs_q) - m)
        total += weight * tl.load(L + (base + split) * BLOCK_Q + offs_q)
        acc += weight[:, None] * tl.load(ACC + (base + split).to(tl.int64) * BLOCK_Q * D + offs_q[:, None] * D + offs_d[None, :])
    out = acc / total[:, None]
    out_ptrs = OUT + b.to(tl.int64) * stride_ob + offs_q[:, None] * stride_oq + h * stride_oh + offs_d[None, :]
    tl.store(out_ptrs, out.to(OUT.dtype.element_ty), mask=(offs_q < NQ)[:, None])


@triton.jit
def _write_kernel(KC, VC, KN, VN, POS, stride_cb, stride_cs, stride_ch, stride_kb, stride_kh, stride_vb, stride_vh,
                  H, D: tl.constexpr):
    bh = tl.program_id(0)
    b = bh // H
    h = bh % H
    offs_d = tl.arange(0, D)
    position = tl.load(POS + b).to(tl.int64)
    target = b.to(tl.int64) * stride_cb + position * stride_cs + h * stride_ch + offs_d
    tl.store(KC + target, tl.load(KN + b.to(tl.int64) * stride_kb + h * stride_kh + offs_d))
    tl.store(VC + target, tl.load(VN + b.to(tl.int64) * stride_vb + h * stride_vh + offs_d))


def supported(q: torch.Tensor, k: torch.Tensor) -> bool:
    """CUDA fp16/bf16 tensors with a few queries, a power-of-two head size from 16 and contiguous last dimensions."""
    head = q.shape[-1]
    return (q.is_cuda and q.dtype in (torch.float16, torch.bfloat16) and k.dtype == q.dtype and q.shape[1] <= MAX_QUERIES
            and head >= 16 and head & (head - 1) == 0 and q.stride(-1) == 1 and k.stride(-1) == 1)


def decode_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lengths: torch.Tensor | None = None, *,
                     length_offset: int = 0) -> torch.Tensor:
    """Softmax attention of q [B, Q, H, D] over k, v [B, S, H, D]; ``lengths + length_offset`` keys per row."""
    batch, queries, heads, head = q.shape
    keys = k.shape[1]
    out = torch.empty((batch, queries, heads, head), device=q.device, dtype=q.dtype)
    blocks = triton.cdiv(keys, BLOCK_N)
    splits = max(1, min(blocks, triton.cdiv(TARGET_PROGRAMS, batch * heads)))
    split_len = triton.cdiv(blocks, splits) * BLOCK_N
    splits = triton.cdiv(keys, split_len)
    acc = torch.empty((batch * heads * splits, BLOCK_Q, head), device=q.device, dtype=torch.float32)
    m = torch.empty((batch * heads * splits, BLOCK_Q), device=q.device, dtype=torch.float32)
    total = torch.empty_like(m)
    has_lengths = lengths is not None
    _partial_kernel[(batch * heads, splits)](
        q, k, v, lengths if has_lengths else q, acc, m, total,
        q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.stride(2),
        heads, queries, keys, split_len, splits, int(length_offset), 1.0 / math.sqrt(head),
        HAS_LENGTHS=has_lengths, D=head, BLOCK_Q=BLOCK_Q, BLOCK_N=BLOCK_N, num_warps=4, num_stages=2)
    _combine_kernel[(batch * heads,)](acc, m, total, out, out.stride(0), out.stride(1), out.stride(2), heads, queries,
                                      NUM_SPLITS=splits, D=head, BLOCK_Q=BLOCK_Q, num_warps=4)
    return out


def write_kv(k_cache: torch.Tensor, v_cache: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
             positions: torch.Tensor) -> None:
    """``k_cache[b, positions[b]] = k[b]`` and the same for v; k and v are [B, H, D], the caches [B, S, H, D]."""
    batch, heads, head = k.shape
    _write_kernel[(batch * heads,)](
        k_cache, v_cache, k, v, positions, k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        k.stride(0), k.stride(1), v.stride(0), v.stride(1), heads, D=head, num_warps=1)


__all__ = ["MAX_QUERIES", "decode_attention", "supported", "write_kv"]
