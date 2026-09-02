"""ComfyUI-compatible INT8 ConvRot conversion and runtime support.

The on-disk layout follows ComfyUI's ``int8_tensorwise`` convention: weights
are stored in Linear layout (out_features, in_features), activations and
weights share a block-Hadamard rotation, and scales are per output row.
"""

from __future__ import annotations

import json
import math
import os
import re
import struct
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn


COMFY_FORMAT = "int8_tensorwise"
DEFAULT_GROUP_SIZES = (256, 64, 16)
_GPT_PROJECTIONS = (
    "attn.c_attn",
    "attn.c_proj",
    "mlp.c_fc",
    "mlp.c_proj",
)
_GPT_WEIGHT_RE = re.compile(
    r"^gpt\.h\.(\d+)\.(attn\.c_attn|attn\.c_proj|mlp\.c_fc|mlp\.c_proj)\.weight$"
)
_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}
_HADAMARD_LOCK = threading.Lock()
_INT8_GEMM_CACHE: dict[int, bool] = {}
_CUDA_TURING_CACHE: dict[int, bool] = {}
_INT8_DEVICE_NAMES: dict[int, str] = {}
_INT8_KERNEL_CHOICES: dict[tuple[str, int, int, str], str] = {}
_INT8_KERNEL_CACHE_LOADED = False
_INT8_KERNEL_CACHE_VERSION = 1
_INT8_KERNEL_CACHE_PATH = (
    Path(__file__).resolve().parents[2] / "models" / ".int8_kernel_cache.json"
)
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def _is_power_of_four(value: int) -> bool:
    if value < 4:
        return False
    while value > 1 and value % 4 == 0:
        value //= 4
    return value == 1


def _build_hadamard(
    size: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the normalized regular (H4 Kronecker) Hadamard matrix."""

    size = int(size)
    if not _is_power_of_four(size):
        raise ValueError(f"Regular Hadamard size must be a power of four, got {size}")
    resolved_device = torch.device(device)
    key = (size, str(resolved_device), dtype)
    with _HADAMARD_LOCK:
        cached = _HADAMARD_CACHE.get(key)
        if cached is not None:
            return cached
        h4 = torch.tensor(
            [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
            dtype=dtype,
            device=resolved_device,
        )
        matrix = h4
        current = 4
        while current < size:
            matrix = torch.kron(matrix, h4)
            current *= 4
        matrix = matrix / math.sqrt(size)
        _HADAMARD_CACHE[key] = matrix
        return matrix


def clear_hadamard_cache() -> int:
    """Clear lazily built Hadamard matrices and return the number removed."""

    with _HADAMARD_LOCK:
        count = len(_HADAMARD_CACHE)
        _HADAMARD_CACHE.clear()
    return count


def _rotate_weight(weight: torch.Tensor, h: torch.Tensor, group_size: int) -> torch.Tensor:
    """Rotate Linear-layout weights along K: ``W_rot = W @ H_block.T``."""

    if weight.ndim != 2:
        raise ValueError(f"Expected a 2-D weight, got shape {tuple(weight.shape)}")
    out_features, in_features = weight.shape
    if in_features % group_size:
        raise ValueError(
            f"in_features {in_features} is not divisible by group size {group_size}"
        )
    groups = in_features // group_size
    matrix = (
        h
        if h.device == weight.device and h.dtype == weight.dtype
        else h.to(device=weight.device, dtype=weight.dtype)
    )
    return torch.matmul(
        weight.reshape(out_features, groups, group_size),
        matrix.T,
    ).reshape(out_features, in_features)


def _rotate_activation(x: torch.Tensor, h: torch.Tensor, group_size: int) -> torch.Tensor:
    """Rotate the last activation dimension: ``x_rot = x @ H_block``."""

    if x.ndim < 1:
        raise ValueError("ConvRot activation must have at least one dimension")
    shape = x.shape
    if shape[-1] % group_size:
        raise ValueError(
            f"features {shape[-1]} is not divisible by group size {group_size}"
        )
    groups = shape[-1] // group_size
    matrix = (
        h
        if h.device == x.device and h.dtype == x.dtype
        else h.to(device=x.device, dtype=x.dtype)
    )
    return torch.matmul(
        x.reshape(-1, groups, group_size),
        matrix,
    ).reshape(shape)


def _comfy_quant_dict(group_size: int) -> dict[str, Any]:
    return {
        "format": COMFY_FORMAT,
        "convrot": True,
        "convrot_groupsize": int(group_size),
    }


def comfy_quant_tensor(group_size: int) -> torch.Tensor:
    """Encode a ComfyUI quantization marker as UTF-8 bytes in a U8 tensor."""

    payload = json.dumps(_comfy_quant_dict(group_size)).encode("utf-8")
    return torch.tensor(list(payload), dtype=torch.uint8)


@torch.no_grad()
def _search_scales(rotated: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the three-stage per-row HQ MSE clip search."""

    absmax = rotated.abs().amax(dim=1, keepdim=True).clamp_min(1.0e-30)
    best_alpha = torch.ones_like(absmax)
    best_mse = torch.full_like(absmax, float("inf"))

    def search(values: torch.Tensor, *, centered: bool) -> None:
        nonlocal best_alpha, best_mse
        center = best_alpha.clone() if centered else torch.zeros_like(best_alpha)
        for delta in values.unbind():
            alpha = (center + delta).clamp(0.5, 1.0)
            scale = (absmax * alpha / 127.0).clamp_min(1.0e-30)
            quantized = (rotated / scale).round().clamp(-127, 127)
            mse = (quantized * scale - rotated).square().mean(dim=1, keepdim=True)
            better = mse < best_mse
            best_mse = torch.where(better, mse, best_mse)
            best_alpha = torch.where(better, alpha, best_alpha)

    search(torch.linspace(0.60, 1.00, 41, device=rotated.device), centered=False)
    search(torch.linspace(-0.012, 0.012, 25, device=rotated.device), centered=True)
    search(torch.linspace(-0.001, 0.001, 21, device=rotated.device), centered=True)
    scale = (absmax * best_alpha / 127.0).clamp_min(1.0e-30)
    quantized = (rotated / scale).round().clamp(-127, 127).to(torch.int8)
    return quantized, scale.to(torch.float32)


@torch.no_grad()
def quantize_convrot(
    weight: torch.Tensor,
    group_size: int,
    *,
    mse_clip: bool = True,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate and per-row quantize one Linear-layout weight matrix."""

    target_device = torch.device(device) if device is not None else weight.device
    value = weight.to(device=target_device, dtype=torch.float32)
    h = _build_hadamard(group_size, device=target_device, dtype=torch.float32)
    rotated = _rotate_weight(value, h, group_size)
    if mse_clip:
        return _search_scales(rotated)
    scale = (rotated.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1.0e-30)
    quantized = (rotated / scale).round().clamp(-127, 127).to(torch.int8)
    return quantized, scale.to(torch.float32)


@torch.no_grad()
def reconstruction_metrics(
    weight_int8: torch.Tensor,
    weight_scale: torch.Tensor,
    reference: torch.Tensor,
    group_size: int,
    *,
    device: str | torch.device | None = None,
) -> tuple[float, float]:
    """Return ``(cosine, relative_error_percent)`` after un-rotation."""

    target_device = (
        torch.device(device) if device is not None else torch.device(reference.device)
    )
    q = weight_int8.to(target_device)
    scale = weight_scale.to(device=target_device, dtype=torch.float32)
    dequant_rotated = q.float() * scale.reshape(-1, 1)
    h = _build_hadamard(group_size, device=target_device, dtype=torch.float32)
    reconstructed = _rotate_weight(dequant_rotated, h, group_size)
    expected = reference.to(device=target_device, dtype=torch.float32)
    cosine = F.cosine_similarity(reconstructed.flatten(), expected.flatten(), dim=0).item()
    denominator = expected.norm().clamp_min(1.0e-30)
    relative = ((reconstructed - expected).norm() / denominator).item() * 100.0
    return float(cosine), float(relative)


@torch.no_grad()
def quantize_best_convrot(
    weight: torch.Tensor,
    *,
    group_sizes: Sequence[int] = DEFAULT_GROUP_SIZES,
    mse_clip: bool = True,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, float]]:
    """Try every divisible group size and retain the lowest-error result."""

    if weight.ndim != 2:
        raise ValueError(f"Expected a 2-D weight, got shape {tuple(weight.shape)}")
    target_device = torch.device(device) if device is not None else weight.device
    original = weight.to(device=target_device, dtype=torch.float32)
    valid_groups = [
        int(size)
        for size in group_sizes
        if _is_power_of_four(int(size)) and original.shape[1] % int(size) == 0
    ]
    if not valid_groups:
        raise ValueError(
            f"No regular-Hadamard group in {tuple(group_sizes)} divides K={original.shape[1]}"
        )

    energy = original.square().sum().clamp_min(1.0e-30)
    best: tuple[float, int, torch.Tensor, torch.Tensor] | None = None
    for group_size in valid_groups:
        h = _build_hadamard(group_size, device=target_device, dtype=torch.float32)
        rotated = _rotate_weight(original, h, group_size)
        if mse_clip:
            quantized, scale = _search_scales(rotated)
        else:
            scale = (rotated.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1.0e-30)
            quantized = (rotated / scale).round().clamp(-127, 127).to(torch.int8)
        squared_error = (quantized.float() * scale - rotated).square().sum()
        relative = float((squared_error / energy).sqrt().item() * 100.0)
        if best is None or relative < best[0]:
            best = (relative, group_size, quantized, scale)
        del rotated

    assert best is not None
    _, group_size, quantized, scale = best
    cosine, relative = reconstruction_metrics(
        quantized, scale, original, group_size, device=target_device
    )
    if cosine <= 0.99:
        raise AssertionError(
            f"Broken ConvRot reconstruction: cosine={cosine:.6f}, error={relative:.4f}%"
        )
    metrics = {"cosine_similarity": cosine, "relative_weight_error_pct": relative}
    return quantized.cpu(), scale.cpu(), group_size, metrics


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _cuda_device_is_turing(device: torch.device) -> bool:
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    cached = _CUDA_TURING_CACHE.get(index)
    if cached is None:
        cached = torch.cuda.get_device_capability(index) == (7, 5)
        _CUDA_TURING_CACHE[index] = cached
    return cached


def _int8_gemm_supported(device: str | torch.device) -> bool:
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return False
    index = resolved.index
    if index is None:
        index = torch.cuda.current_device()
    cached = _INT8_GEMM_CACHE.get(index)
    if cached is not None:
        return cached
    # A duplicate probe from two first-use threads is harmless and avoids putting
    # a lock in every autoregressive forward.
    try:
        lhs = torch.zeros((32, 64), dtype=torch.int8, device=resolved)
        rhs = torch.zeros((64, 32), dtype=torch.int8, device=resolved)
        torch._int_mm(lhs, rhs)
        supported = True
    except (RuntimeError, NotImplementedError):
        supported = False
    _INT8_GEMM_CACHE[index] = supported
    return supported


def _int8_device_name(device: torch.device) -> str:
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    cached = _INT8_DEVICE_NAMES.get(index)
    if cached is None:
        cached = torch.cuda.get_device_name(index)
        _INT8_DEVICE_NAMES[index] = cached
    return cached


def _m_bucket(rows: int) -> str:
    if rows <= 16:
        return "1-16"
    if rows <= 64:
        return "17-64"
    if rows <= 256:
        return "65-256"
    return "257+"


def _load_int8_kernel_choices() -> None:
    global _INT8_KERNEL_CACHE_LOADED
    if _INT8_KERNEL_CACHE_LOADED:
        return
    _INT8_KERNEL_CACHE_LOADED = True
    try:
        payload = json.loads(_INT8_KERNEL_CACHE_PATH.read_text(encoding="utf-8"))
        if payload.get("version") != _INT8_KERNEL_CACHE_VERSION:
            return
        for encoded, choice in payload.get("choices", {}).items():
            if choice not in ("w8a16", "w8a8"):
                continue
            device_name, k, n, bucket = encoded.split("\t")
            if bucket not in ("1-16", "17-64", "65-256", "257+"):
                continue
            _INT8_KERNEL_CHOICES[(device_name, int(k), int(n), bucket)] = choice
    except (AttributeError, OSError, TypeError, ValueError, json.JSONDecodeError):
        pass


def _save_int8_kernel_choices() -> None:
    choices = {
        "\t".join((device_name, str(k), str(n), bucket)): choice
        for (device_name, k, n, bucket), choice in sorted(
            _INT8_KERNEL_CHOICES.items()
        )
    }
    payload = {"version": _INT8_KERNEL_CACHE_VERSION, "choices": choices}
    partial = Path(str(_INT8_KERNEL_CACHE_PATH) + f".{os.getpid()}.partial")
    try:
        _INT8_KERNEL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(partial, _INT8_KERNEL_CACHE_PATH)
    except OSError:
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            pass


def _quantize_activation_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    value = x.float()
    scale = (value.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1.0e-30)
    quantized = (value / scale).round().clamp(-127, 127).to(torch.int8)
    return quantized, scale


def _prepare_int8_rhs(weight: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Transpose/pad [N,K] codes into the [K,N] layout used by _int_mm."""

    original_n, original_k = weight.shape
    padded_k = _round_up(original_k, 8)
    n_alignment = (
        32
        if weight.device.type == "cuda" and _cuda_device_is_turing(weight.device)
        else 8
    )
    padded_n = _round_up(original_n, n_alignment)
    transposed = weight.T.contiguous()
    if transposed.shape != (padded_k, padded_n):
        padded = torch.zeros(
            (padded_k, padded_n), device=weight.device, dtype=torch.int8
        )
        padded[:original_k, :original_n].copy_(transposed)
        transposed = padded
    return transposed, original_k, original_n


def _int8_mm_with_prepared_rhs(
    activation: torch.Tensor,
    rhs: torch.Tensor,
    original_k: int,
    original_n: int,
) -> torch.Tensor:
    original_m = activation.shape[0]
    if (
        activation.device.type == "cuda"
        and (
            _cuda_device_is_turing(activation.device)
            or rhs.shape[0] < 128
        )
    ):
        # cuBLASLt has narrower shape coverage for tiny K, and Turing has
        # stricter tensor-core alignment. M32 is accepted by both.
        padded_m = _round_up(max(original_m, 32), 32)
    else:
        # Current CUDA kernels accept arbitrary M once it is greater than 16.
        padded_m = max(original_m, 17)
    if activation.shape != (padded_m, rhs.shape[0]):
        padded = torch.zeros(
            (padded_m, rhs.shape[0]), device=activation.device, dtype=torch.int8
        )
        padded[:original_m, :original_k].copy_(activation[:, :original_k])
        activation = padded
    result = torch._int_mm(activation, rhs)
    return result[:original_m, :original_n]


def _fast_chunk_rows(in_features: int, out_features: int) -> int:
    # Account conservatively for fp32 activation work, int8 codes, int32 output,
    # fp32 epilogue, and the final output. The persistent result is not counted.
    budget = 256 * 1024 * 1024
    bytes_per_row = max(1, in_features * 5 + out_features * 10)
    return max(1, budget // bytes_per_row)


def _w8a8_linear_rotated_once(
    x_2d: torch.Tensor,
    rhs: torch.Tensor,
    original_k: int,
    original_n: int,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    quantized, scale_a = _quantize_activation_rows(x_2d)
    accumulated = _int8_mm_with_prepared_rhs(
        quantized, rhs, original_k, original_n
    )
    scale_w = weight_scale.reshape(1, -1)
    bias_fp32 = (
        bias.to(device=x_2d.device, dtype=torch.float32).reshape(1, -1)
        if bias is not None
        else None
    )
    # Reuse the fp32 accumulator allocation for the broadcast epilogue. This
    # avoids the old fp32 output plus dtype-sized staging allocation and copy_.
    output = accumulated.float()
    output.mul_(scale_a).mul_(scale_w)
    if bias_fp32 is not None:
        output.add_(bias_fp32)
    return output.to(x_2d.dtype)


def _int8_linear_fast_rotated(
    x_2d: torch.Tensor,
    rhs: torch.Tensor,
    original_k: int,
    original_n: int,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """W8A8 Linear for an activation already in the rotated basis."""

    if x_2d.shape[0] <= 8192:
        return _w8a8_linear_rotated_once(
            x_2d, rhs, original_k, original_n, weight_scale, bias
        )

    chunk_rows = max(8192, _fast_chunk_rows(original_k, original_n))
    output = torch.empty(
        (x_2d.shape[0], original_n), device=x_2d.device, dtype=x_2d.dtype
    )
    for start in range(0, x_2d.shape[0], chunk_rows):
        end = min(start + chunk_rows, x_2d.shape[0])
        output[start:end].copy_(
            _w8a8_linear_rotated_once(
                x_2d[start:end],
                rhs,
                original_k,
                original_n,
                weight_scale,
                bias,
            )
        )
    return output


def _w8a16_linear_rotated(
    x_2d: torch.Tensor,
    weight_int8: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Dequantize once into a call-local W8A16 weight and dispatch F.linear."""

    weight = torch.empty(
        weight_int8.shape, device=x_2d.device, dtype=x_2d.dtype
    )
    # ``out=`` lets TensorIterator fuse int8 conversion, scale multiplication,
    # and the final dtype conversion into a single memory-bound CUDA kernel.
    torch.mul(weight_int8, weight_scale, out=weight)
    current_bias = (
        bias
        if bias is None or (bias.device == x_2d.device and bias.dtype == x_2d.dtype)
        else bias.to(device=x_2d.device, dtype=x_2d.dtype)
    )
    return F.linear(x_2d, weight, current_bias)


def _fallback_linear_rotated(
    x_2d: torch.Tensor,
    weight_int8: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    weight = (weight_int8.float() * weight_scale.float().reshape(-1, 1)).to(
        device=x_2d.device, dtype=x_2d.dtype
    )
    current_bias = (
        bias.to(device=x_2d.device, dtype=x_2d.dtype) if bias is not None else None
    )
    return F.linear(x_2d, weight, current_bias)


def _time_cuda_kernel(call: Callable[[], torch.Tensor], repeats: int) -> float:
    for _ in range(2):
        result = call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        result = call()
    end.record()
    end.synchronize()
    elapsed_us = float(start.elapsed_time(end) * 1000.0 / repeats)
    del result
    return elapsed_us


def _int8_kernel_key(
    device: torch.device, original_k: int, original_n: int, rows: int
) -> tuple[str, int, int, str]:
    return (
        _int8_device_name(device),
        original_k,
        original_n,
        _m_bucket(rows),
    )


def _cached_int8_kernel(
    device: torch.device, original_k: int, original_n: int, rows: int
) -> str | None:
    _load_int8_kernel_choices()
    return _INT8_KERNEL_CHOICES.get(
        _int8_kernel_key(device, original_k, original_n, rows)
    )


def _choose_int8_kernel(
    x_2d: torch.Tensor,
    weight_int8: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    rhs: torch.Tensor,
    original_k: int,
    original_n: int,
) -> str:
    """Benchmark a shape bucket once, then persist its faster CUDA path."""

    key = _int8_kernel_key(
        x_2d.device, original_k, original_n, x_2d.shape[0]
    )
    cached = _INT8_KERNEL_CHOICES.get(key)
    if cached is not None:
        return cached

    repeats = 8 if x_2d.shape[0] <= 64 else 5
    try:
        with torch.no_grad():
            w8a16_us = _time_cuda_kernel(
                lambda: _w8a16_linear_rotated(
                    x_2d, weight_int8, weight_scale, bias
                ),
                repeats,
            )
            w8a8_us = _time_cuda_kernel(
                lambda: _int8_linear_fast_rotated(
                    x_2d,
                    rhs,
                    original_k,
                    original_n,
                    weight_scale,
                    bias,
                ),
                repeats,
            )
            choice = "w8a8" if w8a8_us < w8a16_us else "w8a16"
    except (RuntimeError, NotImplementedError):
        choice = "w8a16"
    _INT8_KERNEL_CHOICES[key] = choice
    _save_int8_kernel_choices()
    return choice


class _Int8LinearSTE(torch.autograd.Function):
    """Frozen-weight straight-through gradient for the rotated Linear."""

    @staticmethod
    def forward(
        ctx: Any,
        x_2d: torch.Tensor,
        weight_int8: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        use_w8a16: bool,
    ) -> torch.Tensor:
        ctx.save_for_backward(weight_int8, weight_scale)
        if use_w8a16:
            return _w8a16_linear_rotated(
                x_2d, weight_int8, weight_scale, bias
            )
        return _fallback_linear_rotated(x_2d, weight_int8, weight_scale, bias)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        weight_int8, weight_scale = ctx.saved_tensors
        weight = weight_int8.to(grad_output.dtype) * weight_scale.to(
            grad_output.dtype
        ).reshape(-1, 1)
        grad_input = grad_output.reshape(-1, grad_output.shape[-1]) @ weight
        return grad_input, None, None, None, None


def _torch_is_compiling() -> bool:
    try:
        return bool(torch.compiler.is_compiling())
    except (AttributeError, RuntimeError):
        return False


class ConvRotInt8Linear(nn.Module):
    """Linear-layout ComfyUI INT8 ConvRot layer.

    ``weight_int8`` deliberately remains the ComfyUI/on-disk ``[N, K]`` tensor
    so state dicts and block-swap byte packing stay compatible. CUDA W8A8 uses
    ``weight_int8_rhs``, a non-persistent ``[K_pad, N_pad]`` cache. A data-pointer
    and device check invalidates that cache when a block streamer rebinds the
    source buffer. ``training_ste`` enables an analytic input gradient while
    retaining a deployed W8A16 forward.
    """

    _PROTECTED_BUFFERS = ("weight_int8", "weight_scale", "weight_int8_rhs")

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 256,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        training_ste: bool = False,
        groupsize: int | None = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = int(groupsize if groupsize is not None else group_size)
        self.convrot_groupsize = self.group_size
        self.nf = self.out_features
        self.training_ste = bool(training_ste)
        self.force_fallback = False
        self.kernel_mode = "auto"
        self._hf_conv1d_compatible = False
        self._rhs_source_ptr: int | None = None
        self._rhs_source_device: torch.device | None = None
        self._hadamards: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
        self._local_kernel_choices: dict[
            tuple[torch.device, str], str
        ] = {}
        parameter_dtype = dtype or torch.get_default_dtype()
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features, device=device, dtype=parameter_dtype)
            )
        else:
            self.register_parameter("bias", None)
        self.register_buffer(
            "weight_int8",
            torch.empty(
                (self.out_features, self.in_features), device=device, dtype=torch.int8
            ),
        )
        self.register_buffer(
            "weight_scale",
            torch.empty((self.out_features, 1), device=device, dtype=torch.float32),
        )
        self.register_buffer(
            "weight_int8_rhs",
            torch.empty((0, 0), device=device, dtype=torch.int8),
            persistent=False,
        )

    def _rhs_is_current(self) -> bool:
        source = self.weight_int8
        return (
            self.weight_int8_rhs.numel() > 0
            and self._rhs_source_ptr == source.data_ptr()
            and self._rhs_source_device == source.device
        )

    def _remember_rhs_source(self) -> None:
        self._rhs_source_ptr = self.weight_int8.data_ptr()
        self._rhs_source_device = self.weight_int8.device

    def _invalidate_rhs(self, *, drop: bool) -> None:
        self._rhs_source_ptr = None
        self._rhs_source_device = None
        if drop and self._buffers.get("weight_int8_rhs") is not None:
            self._buffers["weight_int8_rhs"] = torch.empty(
                (0, 0), device=self.weight_int8.device, dtype=torch.int8
            )

    def _get_weight_int8_rhs(self) -> torch.Tensor:
        """Return the current padded RHS, rebuilding after source rebinding."""

        if self._rhs_is_current():
            return self.weight_int8_rhs
        rhs, original_k, original_n = _prepare_int8_rhs(self.weight_int8)
        if original_k != self.in_features or original_n != self.out_features:
            raise ValueError(
                "weight_int8 shape changed from the configured "
                f"[{self.out_features}, {self.in_features}]"
            )
        self._buffers["weight_int8_rhs"] = rhs
        self._remember_rhs_source()
        return rhs

    def _get_hadamard(
        self, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        key = (device, dtype)
        cached = self._hadamards.get(key)
        if cached is None:
            cached = _build_hadamard(
                self.group_size, device=device, dtype=dtype
            )
            self._hadamards[key] = cached
        return cached

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True):
        """Move protected buffers between devices without ever dtype-casting them."""

        rhs_was_current = self._rhs_is_current()
        protected = {
            name: self._buffers.get(name)
            for name in self._PROTECTED_BUFFERS
            if self._buffers.get(name) is not None
        }
        for name in protected:
            self._buffers[name] = None
        try:
            super()._apply(fn, recurse=recurse)
        finally:
            for name, value in protected.items():
                probe = torch.empty(0, device=value.device, dtype=value.dtype)
                destination = fn(probe).device
                self._buffers[name] = value.to(device=destination)
        self._hadamards.clear()
        if rhs_was_current:
            self._remember_rhs_source()
        else:
            self._invalidate_rhs(drop=False)
        return self

    def _load_from_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._invalidate_rhs(drop=True)

    def _rotated_weight(self, dtype: torch.dtype) -> torch.Tensor:
        return (self.weight_int8.float() * self.weight_scale.float()).to(dtype=dtype)

    @torch.no_grad()
    def dequantize_weight(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Return the original-space Linear weight with shape ``[N, K]``."""

        rotated = self.weight_int8.float() * self.weight_scale.float()
        h = self._get_hadamard(rotated.device, torch.float32)
        return _rotate_weight(rotated, h, self.group_size).to(dtype=dtype)

    @property
    def weight(self) -> torch.Tensor:
        """Compatibility view for diagnostics and Transformers tooling.

        A patched HF ``Conv1D`` exposes its historical ``[in, out]`` layout;
        ordinary users of this Linear receive ``[out, in]``.
        """

        value = self.dequantize_weight()
        return value.T if self._hf_conv1d_compatible else value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected input width {self.in_features}, got {x.shape[-1]}"
            )
        h = self._get_hadamard(x.device, x.dtype)
        rotated = _rotate_activation(x, h, self.group_size)
        original_shape = rotated.shape
        x_2d = rotated.reshape(-1, self.in_features)
        cuda_w8a16 = (
            not _torch_is_compiling()
            and x.device.type == "cuda"
            and self.weight_int8.device == x.device
            and self.weight_scale.device == x.device
        )
        use_ste = self.training_ste and (x.requires_grad or torch.is_grad_enabled())
        if use_ste:
            output = _Int8LinearSTE.apply(
                x_2d,
                self.weight_int8,
                self.weight_scale,
                self.bias,
                cuda_w8a16 and not self.force_fallback,
            )
        elif self.force_fallback or not cuda_w8a16:
            output = _fallback_linear_rotated(
                x_2d, self.weight_int8, self.weight_scale, self.bias
            )
        else:
            mode = self.kernel_mode
            if mode not in ("auto", "w8a16", "w8a8"):
                raise ValueError(
                    "kernel_mode must be 'auto', 'w8a16', or 'w8a8', "
                    f"got {mode!r}"
                )
            if mode == "auto":
                local_key = (x.device, _m_bucket(x_2d.shape[0]))
                mode = self._local_kernel_choices.get(local_key)
                if mode is None:
                    mode = _cached_int8_kernel(
                        x.device,
                        self.in_features,
                        self.out_features,
                        x_2d.shape[0],
                    )
                    if mode is None:
                        if _int8_gemm_supported(x.device):
                            rhs = self._get_weight_int8_rhs()
                            mode = _choose_int8_kernel(
                                x_2d,
                                self.weight_int8,
                                self.weight_scale,
                                self.bias,
                                rhs,
                                self.in_features,
                                self.out_features,
                            )
                        else:
                            mode = "w8a16"
                    self._local_kernel_choices[local_key] = mode
            if mode == "w8a8" and _int8_gemm_supported(x.device):
                output = _int8_linear_fast_rotated(
                    x_2d,
                    self._get_weight_int8_rhs(),
                    self.in_features,
                    self.out_features,
                    self.weight_scale,
                    self.bias,
                )
            else:
                output = _w8a16_linear_rotated(
                    x_2d, self.weight_int8, self.weight_scale, self.bias
                )
        return output.reshape(*original_shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, group_size={self.group_size}, int8=True"
        )


@dataclass
class LoadReport:
    quantized_layers: int
    bf16_tensors: int
    bytes_loaded: int
    seconds: float
    missing_keys: list[str]
    unexpected_keys: list[str]


def _find_tensor_state_dict(value: Any) -> Mapping[str, torch.Tensor] | None:
    if not isinstance(value, Mapping):
        return None
    tensors = OrderedDict(
        (str(key), tensor)
        for key, tensor in value.items()
        if isinstance(tensor, torch.Tensor)
    )
    if tensors:
        return tensors
    for key in ("model", "state_dict", "model_state_dict", "module", "net", "ema", "params"):
        found = _find_tensor_state_dict(value.get(key))
        if found is not None:
            return found
    return None


def _torch_load_state_dict(path: str | os.PathLike[str]) -> Mapping[str, torch.Tensor]:
    kwargs = {"map_location": "cpu", "weights_only": True}
    try:
        checkpoint = torch.load(path, mmap=True, **kwargs)
    except (TypeError, RuntimeError, ValueError):
        checkpoint = torch.load(path, **kwargs)
    # Match indextts.utils.checkpoint exactly: an explicit ``model`` payload wins
    # even when the outer checkpoint also contains tensor-valued bookkeeping.
    if isinstance(checkpoint, Mapping) and isinstance(checkpoint.get("model"), Mapping):
        state = _find_tensor_state_dict(checkpoint["model"])
    else:
        state = _find_tensor_state_dict(checkpoint)
    if state is None:
        raise ValueError(f"No tensor state dict found in {path}")
    return state


def _parse_comfy_marker(value: torch.Tensor) -> dict[str, Any] | None:
    try:
        payload = bytes(value.detach().cpu().contiguous().view(torch.uint8).tolist())
        return json.loads(payload.decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _parse_quantization_metadata(metadata: Mapping[str, str] | None) -> dict[str, dict[str, Any]]:
    if not metadata:
        return {}
    encoded = metadata.get("_quantization_metadata")
    if not encoded:
        return {}
    try:
        root = json.loads(encoded)
    except (TypeError, json.JSONDecodeError):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for base, item in root.get("layers", {}).items():
        if not isinstance(item, Mapping):
            continue
        if item.get("format") != COMFY_FORMAT or not item.get("convrot", True):
            continue
        result[str(base)] = {
            "format": COMFY_FORMAT,
            "convrot": True,
            "group_size": int(item.get("convrot_groupsize", 256)),
        }
    return result


def _detect_safetensors_plan(handle: Any) -> dict[str, dict[str, Any]]:
    plan = _parse_quantization_metadata(handle.metadata() or {})
    keys = set(handle.keys())
    for key in keys:
        if not key.endswith(".comfy_quant"):
            continue
        base = key[: -len(".comfy_quant")]
        config = _parse_comfy_marker(handle.get_tensor(key))
        if not config or config.get("format") != COMFY_FORMAT:
            continue
        if not config.get("convrot", True):
            continue
        plan[base] = {
            "format": COMFY_FORMAT,
            "convrot": True,
            "group_size": int(config.get("convrot_groupsize", 256)),
        }
    for base, item in plan.items():
        weight_key = f"{base}.weight"
        scale_key = f"{base}.weight_scale"
        if weight_key in keys:
            item["weight_shape"] = tuple(handle.get_slice(weight_key).get_shape())
        if scale_key in keys:
            item["scale_shape"] = tuple(handle.get_slice(scale_key).get_shape())
    return plan


def detect_convrot_layers(
    checkpoint: str | os.PathLike[str] | Mapping[str, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    """Detect supported ConvRot markers in a file or materialized state dict."""

    if isinstance(checkpoint, (str, os.PathLike)):
        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            return _detect_safetensors_plan(handle)
    plan: dict[str, dict[str, Any]] = {}
    for key, value in checkpoint.items():
        if not key.endswith(".comfy_quant") or not isinstance(value, torch.Tensor):
            continue
        config = _parse_comfy_marker(value)
        if not config or config.get("format") != COMFY_FORMAT:
            continue
        base = key[: -len(".comfy_quant")]
        item = {
            "format": COMFY_FORMAT,
            "convrot": bool(config.get("convrot", True)),
            "group_size": int(config.get("convrot_groupsize", 256)),
        }
        weight = checkpoint.get(f"{base}.weight")
        if isinstance(weight, torch.Tensor):
            item["weight_shape"] = tuple(weight.shape)
        plan[base] = item
    return plan


def _get_parent_and_leaf(model: nn.Module, path: str) -> tuple[nn.Module | None, str | None]:
    parent_name, separator, leaf = path.rpartition(".")
    if not separator:
        return model, path
    try:
        return model.get_submodule(parent_name), leaf
    except (AttributeError, KeyError):
        return None, None


def _looks_like_hf_conv1d(module: nn.Module) -> bool:
    return (
        module.__class__.__name__ == "Conv1D"
        and hasattr(module, "nf")
        and isinstance(getattr(module, "weight", None), torch.Tensor)
    )


def patch_model_with_convrot(
    model: nn.Module, plan: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    """Replace planned ``nn.Linear`` or HF ``Conv1D`` children in-place."""

    replaced: list[str] = []
    for base, item in plan.items():
        parent, leaf = _get_parent_and_leaf(model, base)
        if parent is None or leaf is None:
            continue
        original = getattr(parent, leaf, None)
        if isinstance(original, ConvRotInt8Linear):
            replaced.append(base)
            continue
        is_hf = isinstance(original, nn.Module) and _looks_like_hf_conv1d(original)
        if not isinstance(original, nn.Linear) and not is_hf:
            continue
        shape = item.get("weight_shape")
        if shape is not None:
            out_features, in_features = int(shape[0]), int(shape[1])
        elif is_hf:
            in_features, out_features = map(int, original.weight.shape)
        else:
            in_features = int(original.in_features)
            out_features = int(original.out_features)
        old_bias = getattr(original, "bias", None)
        old_weight = getattr(original, "weight", None)
        old_device = (
            old_bias.device
            if isinstance(old_bias, torch.Tensor)
            else old_weight.device
            if isinstance(old_weight, torch.Tensor)
            else torch.device("cpu")
        )
        old_dtype = (
            old_bias.dtype
            if isinstance(old_bias, torch.Tensor) and old_bias.is_floating_point()
            else old_weight.dtype
            if isinstance(old_weight, torch.Tensor) and old_weight.is_floating_point()
            else torch.get_default_dtype()
        )
        replacement = ConvRotInt8Linear(
            in_features,
            out_features,
            bias=old_bias is not None,
            group_size=int(
                item.get(
                    "group_size",
                    item.get("groupsize", item.get("convrot_groupsize", 256)),
                )
            ),
            device=old_device,
            dtype=old_dtype,
        )
        replacement._hf_conv1d_compatible = bool(is_hf)
        if old_bias is not None:
            with torch.no_grad():
                replacement.bias.copy_(old_bias)
            replacement.bias.requires_grad_(old_bias.requires_grad)
        setattr(parent, leaf, replacement)
        replaced.append(base)
    return replaced


def remap_state_dict_keys(
    state_dict: dict[str, torch.Tensor],
    plan: Mapping[str, Mapping[str, Any]],
    *,
    drop_markers: bool = True,
) -> dict[str, torch.Tensor]:
    """Remap Comfy ``.weight`` keys to runtime ``.weight_int8`` buffers."""

    for base in plan:
        source = f"{base}.weight"
        if source in state_dict:
            state_dict[f"{base}.weight_int8"] = state_dict.pop(source)
        if drop_markers:
            state_dict.pop(f"{base}.comfy_quant", None)
    return state_dict


def _assign_model_tensor(model: nn.Module, key: str, value: torch.Tensor) -> bool:
    parent, leaf = _get_parent_and_leaf(model, key)
    if parent is None or leaf is None:
        return False
    if leaf in parent._parameters:
        old = parent._parameters[leaf]
        requires_grad = bool(old.requires_grad) if old is not None else False
        parent._parameters[leaf] = nn.Parameter(value, requires_grad=requires_grad)
        return True
    if leaf in parent._buffers:
        parent._buffers[leaf] = value
        return True
    return False


def _tensor_bytes_from_slice(value: Any) -> int:
    return math.prod(value.get_shape()) * _DTYPE_BYTES[value.get_dtype()]


def _load_safetensors_checkpoint(
    model: nn.Module,
    path: str,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    strict: bool,
) -> LoadReport:
    started = time.perf_counter()
    target_device = torch.device(device)
    with safe_open(path, framework="pt", device="cpu") as handle:
        names = list(handle.keys())
        plan = _detect_safetensors_plan(handle)
        quantized = bool(plan)
        replaced = patch_model_with_convrot(model, plan) if quantized else []
        active = set(replaced)
        expected = set(model.state_dict().keys())
        consumed: set[str] = set()
        unexpected: list[str] = []
        bf16_tensors = 0
        bytes_loaded = sum(_tensor_bytes_from_slice(handle.get_slice(name)) for name in names)

        quant_keys: set[str] = set()
        for base in plan:
            quant_keys.update(
                {
                    f"{base}.weight",
                    f"{base}.weight_scale",
                    f"{base}.comfy_quant",
                }
            )

        for name in names:
            if name.endswith(".comfy_quant") and name[: -len(".comfy_quant")] in plan:
                consumed.add(name)
                continue
            target_name = name
            base = name[: -len(".weight")] if name.endswith(".weight") else None
            if base in plan:
                if base not in active:
                    unexpected.append(name)
                    continue
                target_name = f"{base}.weight_int8"
            value = handle.get_tensor(name)
            if target_name.endswith(".weight_int8"):
                if value.dtype != torch.int8:
                    raise TypeError(f"{name} must be int8, got {value.dtype}")
                moved = value.to(device=target_device).contiguous()
            elif target_name.endswith(".weight_scale") and base is None:
                moved = value.to(device=target_device, dtype=torch.float32).contiguous()
            elif value.is_floating_point():
                moved = value.to(device=target_device, dtype=dtype)
                bf16_tensors += 1
            else:
                moved = value.to(device=target_device)
            if not _assign_model_tensor(model, target_name, moved):
                unexpected.append(name)
                continue
            consumed.add(target_name)

        missing_quant: list[str] = []
        name_set = set(names)
        for base in plan:
            for required in (f"{base}.weight", f"{base}.weight_scale"):
                if required not in name_set:
                    missing_quant.append(required)
            if base not in active:
                missing_quant.append(base)
        if missing_quant:
            raise RuntimeError(
                "Quantized checkpoint keys were not consumed: " + ", ".join(missing_quant[:8])
            )
        unexpected_quant = sorted(set(unexpected) & quant_keys)
        if unexpected_quant:
            raise RuntimeError(
                "Unexpected quantized checkpoint keys: " + ", ".join(unexpected_quant[:8])
            )

    model.to(device=target_device, dtype=dtype)
    missing = sorted(expected - consumed)
    unexpected = sorted(set(unexpected))
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Checkpoint load was not strict: {len(missing)} missing, "
            f"{len(unexpected)} unexpected"
        )
    return LoadReport(
        quantized_layers=len(active),
        bf16_tensors=bf16_tensors,
        bytes_loaded=bytes_loaded,
        seconds=time.perf_counter() - started,
        missing_keys=missing,
        unexpected_keys=unexpected,
    )


def load_gpt_checkpoint(
    model: nn.Module,
    path: str,
    *,
    device: str | torch.device,
    dtype: torch.dtype = torch.bfloat16,
    strict: bool = False,
) -> LoadReport:
    """Load an official or INT8 ConvRot GPT checkpoint into ``model``."""

    checkpoint_path = str(Path(path).expanduser().resolve())
    if checkpoint_path.lower().endswith(".safetensors"):
        return _load_safetensors_checkpoint(
            model,
            checkpoint_path,
            device=device,
            dtype=dtype,
            strict=strict,
        )

    started = time.perf_counter()
    state = _torch_load_state_dict(checkpoint_path)
    bytes_loaded = sum(value.numel() * value.element_size() for value in state.values())
    bf16_tensors = sum(1 for value in state.values() if value.is_floating_point())
    model.to(device=device, dtype=dtype)
    result = model.load_state_dict(state, strict=strict)
    return LoadReport(
        quantized_layers=0,
        bf16_tensors=bf16_tensors,
        bytes_loaded=bytes_loaded,
        seconds=time.perf_counter() - started,
        missing_keys=list(result.missing_keys),
        unexpected_keys=list(result.unexpected_keys),
    )


def is_int8_convrot_checkpoint(path: str | os.PathLike[str]) -> bool:
    """Cheaply inspect a safetensors header for this ConvRot format."""

    value = Path(path)
    if value.suffix.lower() != ".safetensors" or not value.is_file():
        return False
    try:
        with safe_open(str(value), framework="pt", device="cpu") as handle:
            metadata_plan = _parse_quantization_metadata(handle.metadata() or {})
            if metadata_plan:
                return True
            for marker in handle.keys():
                if not marker.endswith(".comfy_quant"):
                    continue
                config = _parse_comfy_marker(handle.get_tensor(marker))
                if (
                    config
                    and config.get("format") == COMFY_FORMAT
                    and config.get("convrot", True)
                ):
                    return True
            return False
    except (OSError, ValueError, RuntimeError):
        return False


def describe_checkpoint(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Describe checkpoint type, header metadata, and stored tensor bytes."""

    value = Path(path).expanduser().resolve()
    result: dict[str, Any] = {
        "path": str(value),
        "exists": value.is_file(),
        "source_bytes": value.stat().st_size if value.is_file() else 0,
        "kind": value.suffix.lower().lstrip(".") or "unknown",
        "int8_convrot": False,
        "quantized_layers": 0,
    }
    if not value.is_file() or value.suffix.lower() != ".safetensors":
        return result
    with safe_open(str(value), framework="pt", device="cpu") as handle:
        plan = _detect_safetensors_plan(handle)
        names = list(handle.keys())
        result.update(
            {
                "metadata": dict(handle.metadata() or {}),
                "tensor_count": len(names),
                "tensor_bytes": sum(
                    _tensor_bytes_from_slice(handle.get_slice(name)) for name in names
                ),
                "int8_convrot": bool(plan),
                "quantized_layers": len(plan),
                "group_sizes": sorted(
                    {int(item["group_size"]) for item in plan.values()}
                ),
            }
        )
    return result


def _select_conversion_targets(
    state: Mapping[str, torch.Tensor], *, quantize_emo_encoder: bool
) -> OrderedDict[str, dict[str, Any]]:
    targets: OrderedDict[str, dict[str, Any]] = OrderedDict()
    expected = [
        f"gpt.h.{index}.{projection}.weight"
        for index in range(24)
        for projection in _GPT_PROJECTIONS
    ]
    missing = [key for key in expected if key not in state]
    if missing:
        raise KeyError(
            f"The source is missing {len(missing)} required GPT projection(s): {missing[:4]}"
        )
    for key in expected:
        tensor = state[key]
        if tensor.ndim != 2:
            raise ValueError(f"Expected a 2-D HF Conv1D weight at {key}")
        base = key[: -len(".weight")]
        targets[base] = {"source_key": key, "transpose": True}

    if quantize_emo_encoder:
        for key in sorted(state):
            tensor = state[key]
            if (
                key.startswith("emo_conditioning_encoder.")
                and key.endswith(".weight")
                and tensor.ndim == 2
                and tensor.shape[1] % 16 == 0
            ):
                base = key[: -len(".weight")]
                targets.setdefault(base, {"source_key": key, "transpose": False})
    return targets


def _json_metadata(
    groups: Mapping[str, int], source_name: str
) -> dict[str, str]:
    layer_metadata = {
        base: {
            "format": COMFY_FORMAT,
            "convrot": True,
            "convrot_groupsize": int(group_size),
        }
        for base, group_size in groups.items()
    }
    quantization_metadata = {
        "format_version": "1.0",
        "layers": layer_metadata,
    }
    return {
        "_quantization_metadata": json.dumps(
            quantization_metadata, separators=(",", ":")
        ),
        "indextts_variant": "int8_convrot",
        "indextts_model": "IndexTeam/IndexTTS-2.5",
        "indextts_source": source_name,
        "indextts_weight_layout": "linear_out_in",
        "indextts_quantized_layers": str(len(groups)),
        "indextts_converter_version": "1",
    }


def _write_raw_tensor(handle: Any, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous().view(torch.uint8)
    handle.write(value.numpy().tobytes())


def _write_bf16_tensor(handle: Any, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu()
    if not value.is_contiguous():
        value = value.contiguous()
    flat = value.view(-1)
    chunk_elements = 16 * 1024 * 1024
    for start in range(0, flat.numel(), chunk_elements):
        chunk = flat[start : start + chunk_elements].to(torch.bfloat16).contiguous()
        handle.write(chunk.view(torch.uint8).numpy().tobytes())


def _make_output_plan(
    state: Mapping[str, torch.Tensor],
    targets: Mapping[str, Mapping[str, Any]],
    quantized: Mapping[str, tuple[torch.Tensor, torch.Tensor, int]],
) -> dict[str, tuple[str, Any]]:
    target_by_key = {item["source_key"]: base for base, item in targets.items()}
    plan: dict[str, tuple[str, Any]] = {}
    for key, value in state.items():
        base = target_by_key.get(key)
        if base is None:
            plan[key] = ("bf16", key)
            continue
        q, scale, group_size = quantized[base]
        plan[key] = ("tensor", q)
        plan[f"{base}.weight_scale"] = ("tensor", scale)
        plan[f"{base}.comfy_quant"] = ("tensor", comfy_quant_tensor(group_size))
    return plan


def _plan_dtype_shape(
    item: tuple[str, Any], state: Mapping[str, torch.Tensor]
) -> tuple[str, list[int]]:
    kind, payload = item
    if kind == "bf16":
        return "BF16", list(state[payload].shape)
    tensor = payload
    dtype = {
        torch.int8: "I8",
        torch.uint8: "U8",
        torch.float32: "F32",
        torch.bfloat16: "BF16",
    }.get(tensor.dtype)
    if dtype is None:
        raise TypeError(f"Unsupported output dtype: {tensor.dtype}")
    return dtype, list(tensor.shape)


def _write_streaming_safetensors(
    destination: Path,
    state: Mapping[str, torch.Tensor],
    plan: Mapping[str, tuple[str, Any]],
    metadata: Mapping[str, str],
) -> None:
    header: dict[str, Any] = {"__metadata__": dict(metadata)}
    offset = 0
    for key in sorted(plan):
        dtype, shape = _plan_dtype_shape(plan[key], state)
        nbytes = math.prod(shape) * _DTYPE_BYTES[dtype]
        header[key] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes

    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    padded_length = (len(encoded) + 7) & ~7
    partial = Path(str(destination) + ".partial")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("wb") as handle:
        handle.write(struct.pack("<Q", padded_length))
        handle.write(encoded)
        handle.write(b" " * (padded_length - len(encoded)))
        for key in sorted(plan):
            kind, payload = plan[key]
            if kind == "bf16":
                _write_bf16_tensor(handle, state[payload])
            else:
                _write_raw_tensor(handle, payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, destination)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def _emit(progress: Callable[[str], Any] | None, message: str) -> None:
    if progress is not None:
        progress(message)


def convert_gpt_checkpoint(
    src_pth: str,
    dst_safetensors: str,
    *,
    group_sizes: Sequence[int] = DEFAULT_GROUP_SIZES,
    mse_clip: bool = True,
    device: str | torch.device = "cuda",
    report_path: str | None = None,
    progress: Callable[[str], Any] | None = print,
    quantize_emo_encoder: bool = False,
) -> dict[str, Any]:
    """Convert the IndexTTS 2.5 GPT checkpoint to mixed BF16/INT8 ConvRot.

    HF GPT-2 ``Conv1D`` weights are transposed from ``[in, out]`` to the
    ComfyUI/``nn.Linear`` ``[out, in]`` layout before rotation. The destination
    and JSON report are both installed atomically.
    """

    started = time.perf_counter()
    source = Path(src_pth).expanduser().resolve()
    destination = Path(dst_safetensors).expanduser().resolve()
    report_destination = (
        Path(report_path).expanduser().resolve()
        if report_path is not None
        else destination.with_suffix(".report.json")
    )
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.suffix.lower() != ".safetensors":
        raise ValueError("Destination must end with .safetensors")
    resolved_groups = tuple(int(size) for size in group_sizes)
    if not resolved_groups or any(not _is_power_of_four(size) for size in resolved_groups):
        raise ValueError(f"All group sizes must be powers of four: {resolved_groups}")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA conversion was requested, but CUDA is unavailable")

    _emit(progress, f"Loading source checkpoint with mmap: {source}")
    state = _torch_load_state_dict(source)
    targets = _select_conversion_targets(
        state, quantize_emo_encoder=quantize_emo_encoder
    )
    _emit(
        progress,
        f"Quantizing {len(targets)} layers with groups {resolved_groups} "
        f"({'HQ MSE clip' if mse_clip else 'absmax'})",
    )

    quantized: dict[str, tuple[torch.Tensor, torch.Tensor, int]] = {}
    layer_reports: list[dict[str, Any]] = []
    group_counts: dict[int, int] = {}
    total = len(targets)
    for index, (base, item) in enumerate(targets.items(), 1):
        source_weight = state[item["source_key"]]
        linear_weight = source_weight.T if item["transpose"] else source_weight
        before = time.perf_counter()
        q, scale, group_size, metrics = quantize_best_convrot(
            linear_weight,
            group_sizes=resolved_groups,
            mse_clip=mse_clip,
            device=target_device,
        )
        quantized[base] = (q, scale, group_size)
        group_counts[group_size] = group_counts.get(group_size, 0) + 1
        entry = {
            "layer": base,
            "source_shape": list(source_weight.shape),
            "stored_shape": list(q.shape),
            "group_size": group_size,
            "cosine_similarity": metrics["cosine_similarity"],
            "relative_weight_error_pct": metrics["relative_weight_error_pct"],
            "seconds": time.perf_counter() - before,
        }
        layer_reports.append(entry)
        _emit(
            progress,
            f"[{index:3d}/{total}] gs{group_size:<3d} "
            f"err={entry['relative_weight_error_pct']:.4f}% "
            f"cos={entry['cosine_similarity']:.6f} {base}",
        )

    groups = {base: value[2] for base, value in quantized.items()}
    metadata = _json_metadata(groups, source.name)
    output_plan = _make_output_plan(state, targets, quantized)
    _emit(progress, f"Writing {len(output_plan)} tensors atomically to {destination}")
    _write_streaming_safetensors(destination, state, output_plan, metadata)

    errors = [item["relative_weight_error_pct"] for item in layer_reports]
    source_bytes = source.stat().st_size
    output_bytes = destination.stat().st_size
    elapsed = time.perf_counter() - started
    report: dict[str, Any] = {
        "format": COMFY_FORMAT,
        "format_version": "1.0",
        "source": str(source),
        "output": str(destination),
        "report": str(report_destination),
        "method": (
            "per-row three-stage MSE clipping and per-layer group-size search"
            if mse_clip
            else "per-row absmax and per-layer group-size search"
        ),
        "mse_clip": bool(mse_clip),
        "quantize_emo_encoder": bool(quantize_emo_encoder),
        "quantized_layers": len(layer_reports),
        "bf16_tensors": sum(1 for item in output_plan.values() if item[0] == "bf16"),
        "output_tensors": len(output_plan),
        "group_sizes": {str(key): value for key, value in sorted(group_counts.items())},
        "mean_relative_weight_error_pct": sum(errors) / len(errors),
        "max_relative_weight_error_pct": max(errors),
        "min_cosine_similarity": min(
            item["cosine_similarity"] for item in layer_reports
        ),
        "source_bytes": source_bytes,
        "output_bytes": output_bytes,
        "compression_ratio": source_bytes / output_bytes,
        "conversion_seconds": elapsed,
        "metadata": metadata,
        "layers": layer_reports,
    }
    _write_json_atomic(report_destination, report)
    _emit(
        progress,
        f"Done: {len(layer_reports)} layers, mean/max error "
        f"{report['mean_relative_weight_error_pct']:.4f}%/"
        f"{report['max_relative_weight_error_pct']:.4f}%, "
        f"{source_bytes / 1024**3:.2f} GiB -> {output_bytes / 1024**3:.2f} GiB, "
        f"{elapsed:.1f}s",
    )
    return report


__all__ = [
    "COMFY_FORMAT",
    "ConvRotInt8Linear",
    "DEFAULT_GROUP_SIZES",
    "LoadReport",
    "_build_hadamard",
    "_rotate_activation",
    "_rotate_weight",
    "clear_hadamard_cache",
    "comfy_quant_tensor",
    "convert_gpt_checkpoint",
    "describe_checkpoint",
    "detect_convrot_layers",
    "is_int8_convrot_checkpoint",
    "load_gpt_checkpoint",
    "patch_model_with_convrot",
    "quantize_best_convrot",
    "quantize_convrot",
    "reconstruction_metrics",
    "remap_state_dict_keys",
]
