"""Optional fused CUDA inference kernel for token-sized rank-32 adapters."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_lora_kernel(
    x,
    lora_a,
    lora_b,
    base_result,
    scale,
    bias_correction,
    output,
    K: tl.constexpr,
    N: tl.constexpr,
    R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_BIAS_CORRECTION: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offsets_r = tl.arange(0, R)
    rank = tl.zeros((R,), tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offsets_k = k_start + tl.arange(0, BLOCK_K)
        x_value = tl.load(
            x + pid_m * K + offsets_k, mask=offsets_k < K, other=0.0
        )
        a_value = tl.load(
            lora_a + offsets_r[:, None] * K + offsets_k[None, :],
            mask=offsets_k[None, :] < K,
            other=0.0,
        )
        rank += tl.sum(a_value * x_value[None, :], axis=1)

    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offsets_n < N
    b_value = tl.load(
        lora_b + offsets_n[:, None] * R + offsets_r[None, :],
        mask=mask_n[:, None],
        other=0.0,
    )
    lora_value = tl.sum(b_value * rank[None, :], axis=1)
    base_value = tl.load(
        base_result + pid_m * N + offsets_n, mask=mask_n, other=0.0
    ).to(tl.float32)
    if HAS_SCALE:
        scale_value = tl.load(scale + offsets_n, mask=mask_n, other=0.0).to(
            tl.float32
        )
        base_value *= scale_value
    if HAS_BIAS_CORRECTION:
        correction = tl.load(
            bias_correction + offsets_n, mask=mask_n, other=0.0
        ).to(tl.float32)
        base_value += correction
    tl.store(
        output + pid_m * N + offsets_n,
        base_value + lora_value,
        mask=mask_n,
    )


@triton.jit
def _fused_base_lora_kernel(
    x,
    base_weight,
    base_bias,
    lora_a,
    lora_b,
    scale,
    output,
    K: tl.constexpr,
    N: tl.constexpr,
    R: tl.constexpr,
    WEIGHT_STRIDE_N: tl.constexpr,
    WEIGHT_STRIDE_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_r = tl.arange(0, R)
    mask_n = offsets_n < N
    base_value = tl.zeros((BLOCK_N,), tl.float32)
    rank = tl.zeros((R,), tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offsets_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offsets_k < K
        x_value = tl.load(
            x + pid_m * K + offsets_k, mask=mask_k, other=0.0
        )
        weight_value = tl.load(
            base_weight
            + offsets_n[:, None] * WEIGHT_STRIDE_N
            + offsets_k[None, :] * WEIGHT_STRIDE_K,
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
        )
        base_value += tl.sum(weight_value * x_value[None, :], axis=1)
        a_value = tl.load(
            lora_a + offsets_r[:, None] * K + offsets_k[None, :],
            mask=mask_k[None, :],
            other=0.0,
        )
        rank += tl.sum(a_value * x_value[None, :], axis=1)

    b_value = tl.load(
        lora_b + offsets_n[:, None] * R + offsets_r[None, :],
        mask=mask_n[:, None],
        other=0.0,
    )
    lora_value = tl.sum(b_value * rank[None, :], axis=1)
    if HAS_SCALE:
        scale_value = tl.load(scale + offsets_n, mask=mask_n, other=0.0).to(
            tl.float32
        )
        base_value *= scale_value
    if HAS_BIAS:
        base_value += tl.load(
            base_bias + offsets_n, mask=mask_n, other=0.0
        ).to(tl.float32)
    tl.store(
        output + pid_m * N + offsets_n,
        base_value + lora_value,
        mask=mask_n,
    )


def fused_lora(
    x: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    base_result: torch.Tensor,
    scale: torch.Tensor | None,
    bias_correction: torch.Tensor | None,
    *,
    block_n: int = 128,
    block_k: int = 128,
    num_warps: int = 4,
) -> torch.Tensor:
    rows = base_result.numel() // base_result.shape[-1]
    output = torch.empty_like(base_result)
    has_scale = scale is not None
    has_bias_correction = bias_correction is not None
    if scale is None:
        scale = base_result
    if bias_correction is None:
        bias_correction = base_result
    grid = (triton.cdiv(base_result.shape[-1], block_n), rows)
    _fused_lora_kernel[grid](
        x,
        lora_a,
        lora_b,
        base_result,
        scale,
        bias_correction,
        output,
        x.shape[-1],
        base_result.shape[-1],
        lora_a.shape[0],
        block_n,
        block_k,
        has_scale,
        has_bias_correction,
        num_warps=num_warps,
    )
    return output


def fused_base_lora(
    x: torch.Tensor,
    base_weight: torch.Tensor,
    base_bias: torch.Tensor | None,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scale: torch.Tensor | None,
    *,
    weight_stride_n: int,
    weight_stride_k: int,
    block_n: int = 64,
    block_k: int = 128,
    num_warps: int = 4,
    num_stages: int = 4,
) -> torch.Tensor:
    rows = x.numel() // x.shape[-1]
    out_features = lora_b.shape[0]
    output = torch.empty(
        (*x.shape[:-1], out_features), device=x.device, dtype=x.dtype
    )
    has_bias = base_bias is not None
    has_scale = scale is not None
    if base_bias is None:
        base_bias = x
    if scale is None:
        scale = x
    grid = (triton.cdiv(out_features, block_n), rows)
    _fused_base_lora_kernel[grid](
        x,
        base_weight,
        base_bias,
        lora_a,
        lora_b,
        scale,
        output,
        x.shape[-1],
        out_features,
        lora_a.shape[0],
        weight_stride_n,
        weight_stride_k,
        block_n,
        block_k,
        has_bias,
        has_scale,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output
