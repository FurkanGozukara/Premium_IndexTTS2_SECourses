"""Triton kernels for INT8 ConvRot Whisper inference.

ConvRot rotates every 256-wide input group with the regular Hadamard matrix
H256 = (H4 kron H4 kron H4 kron H4) / 16, the same matrix Comfy's
``comfy_kitchen`` uses for ``int8_tensorwise`` + ``convrot`` checkpoints.
Weights are stored pre-rotated (W @ H, H is symmetric and orthogonal) with one
fp32 scale per output channel, so ``x @ W.T == (x @ H) @ (W @ H).T``.

Kernels:
  * ``rotate_quantize``: optional LayerNorm or exact GELU, group rotation and
    dynamic INT8 quantization with one scale per row or per (row, 256-group).
  * ``int8_linear_q``: INT8 x INT8 -> INT32 GEMM (W8A8) with a fused epilogue
    (activation scales * channel scale, bias, exact GELU, residual add).
  * ``rotate_act`` + ``w8a16_linear``: weight-only INT8 path (rotated fp16
    activations x INT8 weights) with the same epilogue; swap-AB and split-K
    for decode-sized inputs.

Ported from Whisper-WebUI Premium (SECourses), modules/whisper/convrot at revision 613beba (September 2026).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .triton_status import instrument_autotuner

CONVROT_GROUP = 256

PRE_NONE = 0
PRE_LAYERNORM = 1
PRE_GELU = 2


@triton.jit
def _h4_stage(values, BLOCK: tl.constexpr, STRIDE: tl.constexpr):
    """Apply one H4 factor on the base-4 digit with the given stride."""
    OUTER: tl.constexpr = BLOCK // (4 * STRIDE)
    grouped = tl.reshape(values, (OUTER, 4, STRIDE))
    quartets = tl.permute(grouped, (0, 2, 1))
    paired = tl.reshape(quartets, (OUTER, STRIDE, 2, 2))
    ac, bd = tl.split(paired)
    a, c = tl.split(ac)
    b, d = tl.split(bd)
    p = a + b
    q = a - b
    r = c + d
    s = c - d
    y02 = tl.join(p + s, q + r)
    y13 = tl.join(p - s, r - q)
    out = tl.reshape(tl.join(y02, y13), (OUTER, STRIDE, 4))
    out = tl.permute(out, (0, 2, 1))
    return tl.reshape(out, (BLOCK,))


@triton.jit
def _rotate_groups256(values, BLOCK: tl.constexpr):
    """Unnormalized H256 on every contiguous 256-group of a flat block."""
    values = _h4_stage(values, BLOCK, 1)
    values = _h4_stage(values, BLOCK, 4)
    values = _h4_stage(values, BLOCK, 16)
    values = _h4_stage(values, BLOCK, 64)
    return values


@triton.jit
def _gelu_erf(x):
    return 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))


@triton.jit
def _rotate_quantize_kernel(
    X, stride_xm, Q, S, LNW, LNB, eps,
    K: tl.constexpr, NG_P2: tl.constexpr, PRE: tl.constexpr, GS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    g = tl.arange(0, NG_P2)
    c = tl.arange(0, 256)
    col = g[:, None] * 256 + c[None, :]
    mask = col < K
    x = tl.load(X + row * stride_xm + col, mask=mask, other=0.0).to(tl.float32)
    if PRE == 1:
        mean = tl.sum(tl.sum(x, axis=1), axis=0) / K
        xc = tl.where(mask, x - mean, 0.0)
        var = tl.sum(tl.sum(xc * xc, axis=1), axis=0) / K
        w = tl.load(LNW + col, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(LNB + col, mask=mask, other=0.0).to(tl.float32)
        x = xc * tl.math.rsqrt(var + eps) * w + b
        x = tl.where(mask, x, 0.0)
    elif PRE == 2:
        x = _gelu_erf(x)
    flat = tl.reshape(x, (NG_P2 * 256,))
    flat = _rotate_groups256(flat, NG_P2 * 256) * 0.0625
    x = tl.reshape(flat, (NG_P2, 256))
    if GS:
        NG: tl.constexpr = K // 256
        gscale = tl.maximum(tl.max(tl.abs(x), axis=1) / 127.0, 1e-12)
        qv = tl.extra.cuda.libdevice.rint(x / gscale[:, None])
        qv = tl.minimum(tl.maximum(qv, -127.0), 127.0)
        tl.store(Q + row * K + col, qv.to(tl.int8), mask=mask)
        tl.store(S + row * NG + g, gscale, mask=g < NG)
    else:
        amax = tl.max(tl.max(tl.abs(x), axis=1), axis=0)
        scale = tl.maximum(amax / 127.0, 1e-12)
        qv = tl.extra.cuda.libdevice.rint(x / scale)
        qv = tl.minimum(tl.maximum(qv, -127.0), 127.0)
        tl.store(Q + row * K + col, qv.to(tl.int8), mask=mask)
        tl.store(S + row, scale)


@triton.jit
def _rotate_quantize_group_kernel(
    X, stride_xm, Q, S, LNW, LNB, eps,
    K: tl.constexpr, NG_P2: tl.constexpr, PRE: tl.constexpr, QUANT: tl.constexpr,
):
    """One program per (row, 256-group). QUANT: INT8 + per-group scale; otherwise the rotated
    activation is written as fp16 (weight-only INT8 path)."""
    row = tl.program_id(0).to(tl.int64)
    g = tl.program_id(1)
    NG: tl.constexpr = K // 256
    c = tl.arange(0, 256)
    x_row = X + row * stride_xm
    x = tl.load(x_row + g * 256 + c).to(tl.float32)
    if PRE == 1:
        gg = tl.arange(0, NG_P2)
        col = gg[:, None] * 256 + c[None, :]
        mask = col < K
        xa = tl.load(x_row + col, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(tl.sum(xa, axis=1), axis=0) / K
        xc = tl.where(mask, xa - mean, 0.0)
        var = tl.sum(tl.sum(xc * xc, axis=1), axis=0) / K
        w = tl.load(LNW + g * 256 + c).to(tl.float32)
        b = tl.load(LNB + g * 256 + c).to(tl.float32)
        x = (x - mean) * tl.math.rsqrt(var + eps) * w + b
    elif PRE == 2:
        x = _gelu_erf(x)
    x = _rotate_groups256(x, 256) * 0.0625
    if QUANT:
        scale = tl.maximum(tl.max(tl.abs(x), axis=0) / 127.0, 1e-12)
        qv = tl.extra.cuda.libdevice.rint(x / scale)
        qv = tl.minimum(tl.maximum(qv, -127.0), 127.0)
        tl.store(Q + row * K + g * 256 + c, qv.to(tl.int8))
        tl.store(S + row * NG + g, scale)
    else:
        tl.store(Q + row * K + g * 256 + c, x.to(Q.dtype.element_ty))


GROUP_KERNEL_MAX_M = 64


def _next_pow2(v: int) -> int:
    return 1 << (int(v) - 1).bit_length()


def rotate_quantize(x: torch.Tensor, pre: int = PRE_NONE, ln_weight=None, ln_bias=None,
                    eps: float = 1e-5, out_q: torch.Tensor | None = None,
                    out_s: torch.Tensor | None = None, group_scales: bool = False):
    """Rotate every 256-group of each row and quantize to INT8.

    Returns ``(q [M, K] int8, scale fp32)``; the scale is per row ``[M]`` or,
    with ``group_scales``, per row and 256-group ``[M, K/256]``.
    """
    k = x.shape[-1]
    x2 = x.reshape(-1, k)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    m = x2.shape[0]
    if k % CONVROT_GROUP:
        raise ValueError(f"ConvRot needs K divisible by {CONVROT_GROUP}, got {k}")
    q = out_q if out_q is not None else torch.empty((m, k), device=x.device, dtype=torch.int8)
    s_shape = (m, k // CONVROT_GROUP) if group_scales else (m,)
    s = out_s if out_s is not None else torch.empty(s_shape, device=x.device, dtype=torch.float32)
    if m == 0:
        return q, s
    ng_p2 = _next_pow2(k // CONVROT_GROUP)
    if pre == PRE_LAYERNORM:
        lnw, lnb = ln_weight, ln_bias
    else:
        lnw = lnb = x2
    if group_scales and m <= GROUP_KERNEL_MAX_M:
        _rotate_quantize_group_kernel[(m, k // CONVROT_GROUP)](
            x2, x2.stride(0), q, s, lnw, lnb, eps, K=k, NG_P2=ng_p2, PRE=pre, QUANT=True, num_warps=4,
        )
        return q, s
    num_warps = 4 if ng_p2 <= 8 else 8
    _rotate_quantize_kernel[(m,)](
        x2, x2.stride(0), q, s, lnw, lnb, eps,
        K=k, NG_P2=ng_p2, PRE=pre, GS=group_scales, num_warps=num_warps,
    )
    return q, s


def rotate_act(x: torch.Tensor, pre: int = PRE_NONE, ln_weight=None, ln_bias=None, eps: float = 1e-5):
    """[LayerNorm | GELU] + group rotation, fp16 output (activation of the weight-only INT8 path)."""
    k = x.shape[-1]
    x2 = x.reshape(-1, k)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    m = x2.shape[0]
    out = torch.empty((m, k), device=x.device, dtype=x.dtype)
    if m == 0:
        return out
    lnw, lnb = (ln_weight, ln_bias) if pre == PRE_LAYERNORM else (x2, x2)
    _rotate_quantize_group_kernel[(m, k // CONVROT_GROUP)](
        x2, x2.stride(0), out, out, lnw, lnb, eps, K=k, NG_P2=_next_pow2(k // CONVROT_GROUP), PRE=pre,
        QUANT=False, num_warps=4,
    )
    return out


def _gemm_configs():
    configs = []
    for bm, bn, bk, warps, stages in (
        (128, 128, 64, 4, 4),
        (128, 128, 128, 8, 3),
        (128, 256, 64, 8, 3),
        (64, 128, 128, 4, 4),
        (128, 64, 128, 4, 4),
        (64, 64, 128, 4, 4),
        (64, 64, 256, 4, 3),
        (128, 64, 256, 4, 2),
        (64, 128, 256, 4, 2),
        (128, 64, 256, 8, 2),
    ):
        configs.append(triton.Config({"BM": bm, "BN": bn, "BK": bk}, num_warps=warps, num_stages=stages))
    return configs


def _small_m_configs():
    configs = []
    for bn, bk, warps, stages in (
        (32, 256, 4, 3),
        (64, 256, 4, 3),
        (32, 128, 4, 4),
        (64, 128, 4, 4),
        (16, 256, 4, 3),
        (128, 128, 4, 3),
    ):
        configs.append(triton.Config({"BM": 16, "BN": bn, "BK": bk}, num_warps=warps, num_stages=stages))
    return configs


@triton.jit
def _int8_gemm_body(
    A, B, C, SA, SB, BIAS, RES, M, N, K,
    stride_am, stride_bn, stride_cm, stride_rm,
    pid_m, pid_n,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    HAS_BIAS: tl.constexpr, GELU: tl.constexpr, HAS_RES: tl.constexpr, GS: tl.constexpr,
):
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = A + (offs_m[:, None] % M).to(tl.int64) * stride_am + offs_k[None, :]
    b_ptrs = B + (offs_n[None, :] % N).to(tl.int64) * stride_bn + offs_k[:, None]
    m_mask = offs_m < M
    n_mask = offs_n < N
    sb = tl.load(SB + offs_n, mask=n_mask, other=0.0)
    if GS:
        NG = K // 256
        acc_f = tl.zeros((BM, BN), dtype=tl.float32)
        if BK == 256:
            # one K tile per activation-scale group: a flat loop the pipeliner handles well
            for g in range(0, NG):
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                acc = tl.dot(a, b, out_dtype=tl.int32)
                sa_g = tl.load(SA + offs_m * NG + g, mask=m_mask, other=0.0)
                acc_f += acc.to(tl.float32) * sa_g[:, None]
                a_ptrs += BK
                b_ptrs += BK
        else:
            for g in range(0, NG):
                acc = tl.zeros((BM, BN), dtype=tl.int32)
                for _kk in tl.static_range(256 // BK):
                    a = tl.load(a_ptrs)
                    b = tl.load(b_ptrs)
                    acc = tl.dot(a, b, acc, out_dtype=tl.int32)
                    a_ptrs += BK
                    b_ptrs += BK
                sa_g = tl.load(SA + offs_m * NG + g, mask=m_mask, other=0.0)
                acc_f += acc.to(tl.float32) * sa_g[:, None]
        out = acc_f * sb[None, :]
    else:
        acc = tl.zeros((BM, BN), dtype=tl.int32)
        for _ in range(0, K, BK):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc, out_dtype=tl.int32)
            a_ptrs += BK
            b_ptrs += BK
        sa = tl.load(SA + offs_m, mask=m_mask, other=0.0)
        out = acc.to(tl.float32) * sa[:, None] * sb[None, :]
    if HAS_BIAS:
        out += tl.load(BIAS + offs_n, mask=n_mask, other=0.0).to(tl.float32)[None, :]
    if GELU:
        out = _gelu_erf(out)
    mask = m_mask[:, None] & n_mask[None, :]
    if HAS_RES:
        res = tl.load(RES + offs_m[:, None].to(tl.int64) * stride_rm + offs_n[None, :], mask=mask, other=0.0)
        out += res.to(tl.float32)
    tl.store(C + offs_m[:, None].to(tl.int64) * stride_cm + offs_n[None, :], out.to(C.dtype.element_ty), mask=mask)


@triton.autotune(configs=_gemm_configs(), key=["M_BUCKET", "N", "K", "GS"], restore_value=["C"],
                 warmup=5, rep=25, cache_results=True)
@triton.jit
def _int8_gemm_kernel(
    A, B, C, SA, SB, BIAS, RES, M, N, K, M_BUCKET,
    stride_am, stride_bn, stride_cm, stride_rm,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    HAS_BIAS: tl.constexpr, GELU: tl.constexpr, HAS_RES: tl.constexpr, GS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    group_m = 8
    num_pid_in_group = group_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_m
    group_size_m = min(num_pid_m - first_pid_m, group_m)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    _int8_gemm_body(A, B, C, SA, SB, BIAS, RES, M, N, K,
                    stride_am, stride_bn, stride_cm, stride_rm, pid_m, pid_n,
                    BM, BN, BK, HAS_BIAS, GELU, HAS_RES, GS)


@triton.autotune(configs=_small_m_configs(), key=["M_BUCKET", "N", "K", "GS"], restore_value=["C"],
                 warmup=5, rep=25, cache_results=True)
@triton.jit
def _int8_gemm_small_m_kernel(
    A, B, C, SA, SB, BIAS, RES, M, N, K, M_BUCKET,
    stride_am, stride_bn, stride_cm, stride_rm,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    HAS_BIAS: tl.constexpr, GELU: tl.constexpr, HAS_RES: tl.constexpr, GS: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    _int8_gemm_body(A, B, C, SA, SB, BIAS, RES, M, N, K,
                    stride_am, stride_bn, stride_cm, stride_rm, pid_m, pid_n,
                    BM, BN, BK, HAS_BIAS, GELU, HAS_RES, GS)


SMALL_M_MAX = 64


_M_BUCKETS = (16, 64, 256, 1024, 4096)


def _m_bucket(m: int) -> int:
    """Coarse row-count classes for autotuning: few classes keep first-run tuning short."""
    for bucket in _M_BUCKETS:
        if m <= bucket:
            return bucket
    return 16384


def int8_linear_q(xq: torch.Tensor, x_scale: torch.Tensor, weight: torch.Tensor, w_scale: torch.Tensor,
                  bias: torch.Tensor | None = None, gelu: bool = False,
                  residual: torch.Tensor | None = None, out: torch.Tensor | None = None,
                  out_dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """``out = [residual +] act(xq*xs @ (w*ws).T + bias)`` for pre-quantized activations.

    ``residual`` may alias ``out`` (in-place residual update).
    """
    m, k = xq.shape
    n = weight.shape[0]
    if out is None:
        out = torch.empty((m, n), device=xq.device, dtype=out_dtype)
    if m == 0:
        return out
    has_bias = bias is not None
    has_res = residual is not None
    gs = x_scale.dim() == 2
    bias_arg = bias if has_bias else w_scale
    res_arg = residual if has_res else out
    stride_rm = residual.stride(0) if has_res else 0
    mb = _m_bucket(m)
    if m <= SMALL_M_MAX:
        grid = lambda meta: (triton.cdiv(n, meta["BN"]), triton.cdiv(m, meta["BM"]))
        _int8_gemm_small_m_kernel[grid](
            xq, weight, out, x_scale, w_scale, bias_arg, res_arg, m, n, k, mb,
            xq.stride(0), weight.stride(0), out.stride(0), stride_rm,
            HAS_BIAS=has_bias, GELU=gelu, HAS_RES=has_res, GS=gs,
        )
    else:
        grid = lambda meta: (triton.cdiv(m, meta["BM"]) * triton.cdiv(n, meta["BN"]),)
        _int8_gemm_kernel[grid](
            xq, weight, out, x_scale, w_scale, bias_arg, res_arg, m, n, k, mb,
            xq.stride(0), weight.stride(0), out.stride(0), stride_rm,
            HAS_BIAS=has_bias, GELU=gelu, HAS_RES=has_res, GS=gs,
        )
    return out


@triton.jit
def _w8a16_gemm_body(
    A, B, C, SB, BIAS, RES, M, N, K, stride_am, stride_bn, stride_cm, stride_rm, pid_m, pid_n,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    HAS_BIAS: tl.constexpr, GELU: tl.constexpr, HAS_RES: tl.constexpr,
):
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = A + (offs_m[:, None] % M).to(tl.int64) * stride_am + offs_k[None, :]
    b_ptrs = B + (offs_n[None, :] % N).to(tl.int64) * stride_bn + offs_k[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K, BK):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs).to(a.dtype)  # int8 -> fp16 is exact
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK
    m_mask = offs_m < M
    n_mask = offs_n < N
    out = acc * tl.load(SB + offs_n, mask=n_mask, other=0.0)[None, :]
    if HAS_BIAS:
        out += tl.load(BIAS + offs_n, mask=n_mask, other=0.0).to(tl.float32)[None, :]
    if GELU:
        out = _gelu_erf(out)
    mask = m_mask[:, None] & n_mask[None, :]
    if HAS_RES:
        res = tl.load(RES + offs_m[:, None].to(tl.int64) * stride_rm + offs_n[None, :], mask=mask, other=0.0)
        out += res.to(tl.float32)
    tl.store(C + offs_m[:, None].to(tl.int64) * stride_cm + offs_n[None, :], out.to(C.dtype.element_ty), mask=mask)


def _w8a16_configs():
    configs = []
    for bm, bn, bk, warps, stages in ((128, 128, 64, 8, 3), (128, 64, 64, 4, 4), (64, 128, 64, 4, 4),
                                      (64, 64, 128, 4, 3), (128, 128, 32, 4, 4)):
        configs.append(triton.Config({"BM": bm, "BN": bn, "BK": bk}, num_warps=warps, num_stages=stages))
    return configs


def _w8a16_small_m_configs():
    configs = []
    for bn, bk, warps, stages in ((64, 256, 4, 3), (32, 256, 4, 3), (64, 128, 4, 3), (128, 128, 4, 3),
                                  (32, 128, 4, 4), (128, 64, 4, 4)):
        configs.append(triton.Config({"BN": bn, "BK": bk}, num_warps=warps, num_stages=stages))
    return configs


@triton.autotune(configs=_w8a16_small_m_configs(), key=["M_BUCKET", "N", "K", "SPLIT"], restore_value=["C"],
                 warmup=5, rep=25, cache_results=True)
@triton.jit
def _w8a16_small_m_kernel(
    A, B, C, WS, SB, BIAS, RES, M, N, K, M_BUCKET, stride_am, stride_cm, stride_rm,
    BN: tl.constexpr, BK: tl.constexpr, MX: tl.constexpr, SPLIT: tl.constexpr,
    HAS_BIAS: tl.constexpr, GELU: tl.constexpr, HAS_RES: tl.constexpr,
):
    """Decode-size weight-only GEMM computed as out^T = W @ x^T ("swap AB"): the weight tile is the
    wide MMA operand and the few activation rows the narrow one. With SPLIT > 1 each program
    covers K / SPLIT and writes an fp32 partial to WS; _splitk_reduce_kernel finishes."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_m = tl.arange(0, MX)
    offs_k = tl.arange(0, BK)
    k_span = K // SPLIT
    k0 = pid_k * k_span
    w_ptrs = B + (offs_n[:, None] % N).to(tl.int64) * K + k0 + offs_k[None, :]
    x_ptrs = A + offs_m[None, :].to(tl.int64) * stride_am + k0 + offs_k[:, None]
    m_mask = offs_m < M
    n_mask = offs_n < N
    acc = tl.zeros((BN, MX), dtype=tl.float32)
    for _ in range(0, k_span, BK):
        w = tl.load(w_ptrs).to(tl.float16)
        xt = tl.load(x_ptrs, mask=m_mask[None, :], other=0.0).to(tl.float16)
        acc = tl.dot(w, xt, acc)
        w_ptrs += BK
        x_ptrs += BK
    mask = n_mask[:, None] & m_mask[None, :]
    if SPLIT > 1:
        ws_ptrs = WS + pid_k * M * N + offs_m[None, :].to(tl.int64) * N + offs_n[:, None]
        tl.store(ws_ptrs, acc, mask=mask)
    else:
        out = acc * tl.load(SB + offs_n, mask=n_mask, other=0.0)[:, None]
        if HAS_BIAS:
            out += tl.load(BIAS + offs_n, mask=n_mask, other=0.0).to(tl.float32)[:, None]
        if GELU:
            out = _gelu_erf(out)
        if HAS_RES:
            res = tl.load(RES + offs_m[None, :].to(tl.int64) * stride_rm + offs_n[:, None], mask=mask, other=0.0)
            out += res.to(tl.float32)
        tl.store(C + offs_m[None, :].to(tl.int64) * stride_cm + offs_n[:, None], out.to(C.dtype.element_ty),
                 mask=mask)


@triton.autotune(configs=_w8a16_configs(), key=["M_BUCKET", "N", "K"], restore_value=["C"],
                 warmup=5, rep=25, cache_results=True)
@triton.jit
def _w8a16_gemm_kernel(
    A, B, C, SB, BIAS, RES, M, N, K, M_BUCKET, stride_am, stride_bn, stride_cm, stride_rm,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    HAS_BIAS: tl.constexpr, GELU: tl.constexpr, HAS_RES: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    group_m = 8
    num_pid_in_group = group_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_m
    group_size_m = min(num_pid_m - first_pid_m, group_m)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    _w8a16_gemm_body(A, B, C, SB, BIAS, RES, M, N, K, stride_am, stride_bn, stride_cm, stride_rm, pid_m, pid_n,
                     BM, BN, BK, HAS_BIAS, GELU, HAS_RES)


@triton.jit
def _splitk_reduce_kernel(WS, C, SB, BIAS, RES, M, N, stride_cm, stride_rm,
                          SPLIT: tl.constexpr, BLOCK: tl.constexpr,
                          HAS_BIAS: tl.constexpr, GELU: tl.constexpr, HAS_RES: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M * N
    m = offs // N
    n = offs % N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for sidx in tl.static_range(SPLIT):
        acc += tl.load(WS + sidx * M * N + offs, mask=mask, other=0.0)
    out = acc * tl.load(SB + n, mask=mask, other=0.0)
    if HAS_BIAS:
        out += tl.load(BIAS + n, mask=mask, other=0.0).to(tl.float32)
    if GELU:
        out = _gelu_erf(out)
    if HAS_RES:
        out += tl.load(RES + m.to(tl.int64) * stride_rm + n, mask=mask, other=0.0).to(tl.float32)
    tl.store(C + m.to(tl.int64) * stride_cm + n, out.to(C.dtype.element_ty), mask=mask)


# First-run tuning is reported in the console and the Live Transcription box; Triton keeps
# the winning configurations on disk, so later processes load them instead of re-tuning.
for _kernel, _label in (
    (_int8_gemm_kernel, "INT8 GEMM"),
    (_int8_gemm_small_m_kernel, "INT8 GEMM (small M)"),
    (_w8a16_gemm_kernel, "weight-only INT8 GEMM"),
    (_w8a16_small_m_kernel, "weight-only INT8 GEMM (decode)"),
):
    instrument_autotuner(_kernel, _label)
del _kernel, _label


def w8a16_linear(x_rot: torch.Tensor, weight: torch.Tensor, w_scale: torch.Tensor,
                 bias: torch.Tensor | None = None, gelu: bool = False,
                 residual: torch.Tensor | None = None, out: torch.Tensor | None = None) -> torch.Tensor:
    """Weight-only INT8: rotated fp16 activation x INT8 ConvRot weight, fused epilogue."""
    m, k = x_rot.shape
    n = weight.shape[0]
    if out is None:
        out = torch.empty((m, n), device=x_rot.device, dtype=x_rot.dtype)
    if m == 0:
        return out
    has_bias = bias is not None
    has_res = residual is not None
    bias_arg = bias if has_bias else w_scale
    res_arg = residual if has_res else out
    stride_rm = residual.stride(0) if has_res else 0
    if m <= SMALL_M_MAX:
        split = 4 if k >= 4096 else 1
        ws = torch.empty((split, m, n), device=x_rot.device, dtype=torch.float32) if split > 1 else out
        grid = lambda meta: (triton.cdiv(n, meta["BN"]), split)
        _w8a16_small_m_kernel[grid](
            x_rot, weight, out, ws, w_scale, bias_arg, res_arg, m, n, k, _m_bucket(m), x_rot.stride(0),
            out.stride(0), stride_rm, MX=max(16, triton.next_power_of_2(m)), SPLIT=split,
            HAS_BIAS=has_bias, GELU=gelu, HAS_RES=has_res)
        if split > 1:
            block = 1024
            _splitk_reduce_kernel[(triton.cdiv(m * n, block),)](
                ws, out, w_scale, bias_arg, res_arg, m, n, out.stride(0), stride_rm, SPLIT=split, BLOCK=block,
                HAS_BIAS=has_bias, GELU=gelu, HAS_RES=has_res, num_warps=4)
    else:
        grid = lambda meta: (triton.cdiv(m, meta["BM"]) * triton.cdiv(n, meta["BN"]),)
        _w8a16_gemm_kernel[grid](x_rot, weight, out, w_scale, bias_arg, res_arg, m, n, k, _m_bucket(m),
                                 x_rot.stride(0), weight.stride(0), out.stride(0), stride_rm,
                                 HAS_BIAS=has_bias, GELU=gelu, HAS_RES=has_res)
    return out


def convrot_linear(x: torch.Tensor, weight: torch.Tensor, w_scale: torch.Tensor,
                   bias: torch.Tensor | None = None, pre: int = PRE_NONE,
                   ln_weight=None, ln_bias=None, eps: float = 1e-5, gelu: bool = False,
                   residual: torch.Tensor | None = None, out: torch.Tensor | None = None,
                   group_scales: bool = False) -> torch.Tensor:
    """Full W8A8 ConvRot linear on an fp16 activation of shape [..., K]."""
    lead = x.shape[:-1]
    xq, xs = rotate_quantize(x, pre=pre, ln_weight=ln_weight, ln_bias=ln_bias, eps=eps,
                             group_scales=group_scales)
    res2 = residual.reshape(-1, residual.shape[-1]) if residual is not None else None
    out2 = out.reshape(-1, out.shape[-1]) if out is not None else None
    y = int8_linear_q(xq, xs, weight, w_scale, bias=bias, gelu=gelu, residual=res2, out=out2)
    return y.reshape(*lead, weight.shape[0])


# ---------------------------------------------------------------------------
# Reference helpers (offline weight conversion and testing)
# ---------------------------------------------------------------------------

_HADAMARD_CACHE: dict = {}


def hadamard_matrix(size: int = CONVROT_GROUP, device="cpu", dtype=torch.float64) -> torch.Tensor:
    """Normalized regular Hadamard matrix (power-of-4 size), comfy_kitchen convention."""
    key = (size, str(device), dtype)
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype, device=device)
    h = h4
    while h.shape[0] < size:
        h = torch.kron(h, h4)
    if h.shape[0] != size:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")
    h = h / (size ** 0.5)
    _HADAMARD_CACHE[key] = h
    return h


def rotate_weight(weight: torch.Tensor, group: int = CONVROT_GROUP) -> torch.Tensor:
    """Offline W @ H per input group (fp64 math)."""
    n, k = weight.shape
    h = hadamard_matrix(group, device=weight.device, dtype=torch.float64)
    w = weight.to(torch.float64).reshape(n, k // group, group)
    return (w @ h).reshape(n, k)


def quantize_weight_rowwise(weight_rot: torch.Tensor, clip_ratio: torch.Tensor | float | None = None):
    """Symmetric per-output-channel INT8 quantization of an already rotated weight."""
    w = weight_rot.to(torch.float64)
    amax = w.abs().amax(dim=1).clamp_min(1e-12)
    if clip_ratio is not None:
        amax = amax * clip_ratio
    scale = amax / 127.0
    q = torch.clamp(torch.round(w / scale[:, None]), -127, 127).to(torch.int8)
    return q, scale.to(torch.float32)
