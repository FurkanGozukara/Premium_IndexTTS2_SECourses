"""Micro-benchmark the legacy and cached LoRA/DoRA layer implementations.

Production keeps the rank-32 factorization. For one to three CUDA rows, an
optional Triton kernel fuses it with the floating base projection; larger and
portable paths use two small BF16 projections. ``--compare-merged-delta`` also
times cached ``B @ A``. On the RTX 5090 that full delta loses to the fused token
path at M=1..3. It can win isolated larger-M calls, but consumes another 12.5
MiB for this single K=1280, N=5120 layer and prefill does not amortize building
it, so the decoding-oriented production path deliberately keeps factorization.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import nn
from torch.nn import functional as F
from transformers.pytorch_utils import Conv1D

from indextts.lora.layers import LoRAAdapter


K = 1280
N = 5120
RANK = 32
DEFAULT_M = (1, 3, 32, 512)


class _LegacyLoRAAdapter(LoRAAdapter):
    """The pre-cache implementation retained for before/after measurements."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_result = self.base(x)
        if not self.enabled or self.strength == 0.0:
            return base_result

        strength = float(self.strength)
        lora_result = self._adapter_forward_training(x)
        if not self.use_dora:
            return base_result + (strength * lora_result).to(
                device=base_result.device, dtype=base_result.dtype
            )

        delta = self.delta_weight()
        weight = self.base_weight_linear().to(device=delta.device, dtype=delta.dtype)
        weight_norm = torch.linalg.vector_norm(weight + delta.detach(), dim=1).detach()
        magnitude_scale = self.lora_magnitude / weight_norm
        effective_scale = 1.0 + strength * (magnitude_scale - 1.0)
        base_projection = self._base_projection(x, weight)
        correction = (effective_scale - 1.0) * base_projection
        correction = correction + effective_scale * (strength * lora_result)
        return base_result + correction.to(
            device=base_result.device, dtype=base_result.dtype
        )


class _MergedDeltaLoRA(nn.Module):
    def __init__(self, adapter: LoRAAdapter, dtype: torch.dtype) -> None:
        super().__init__()
        self.base = adapter.base
        delta = float(adapter.strength) * adapter.delta_weight()
        self.register_buffer("delta", delta.to(device=adapter.lora_A.weight.device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(x, self.delta)


def _make_base(kind: str, device: torch.device) -> nn.Module:
    if kind == "bf16":
        return Conv1D(N, K).to(device=device, dtype=torch.bfloat16).eval()

    from indextts.quant import ConvRotInt8Linear

    base = ConvRotInt8Linear(
        K,
        N,
        group_size=256,
        device=device,
        dtype=torch.bfloat16,
    ).eval()
    with torch.no_grad():
        base.weight_int8.random_(-32, 33)
        base.weight_scale.fill_(0.002)
        if base.bias is not None:
            base.bias.zero_()
    return base


@torch.no_grad()
def _make_adapters(
    base: nn.Module, use_dora: bool
) -> tuple[LoRAAdapter, _LegacyLoRAAdapter]:
    optimized = LoRAAdapter(
        base, rank=RANK, alpha=RANK, use_dora=use_dora
    ).eval()
    legacy = _LegacyLoRAAdapter(
        base, rank=RANK, alpha=RANK, use_dora=use_dora
    ).eval()
    optimized.lora_A.weight.normal_(std=0.02)
    optimized.lora_B.weight.normal_(std=0.02)
    legacy.lora_A.weight.copy_(optimized.lora_A.weight)
    legacy.lora_B.weight.copy_(optimized.lora_B.weight)
    if use_dora:
        optimized.lora_magnitude.mul_(0.95)
        legacy.lora_magnitude.copy_(optimized.lora_magnitude)
    optimized.strength = 0.8
    legacy.strength = 0.8
    return optimized, legacy


def _time_cuda(
    function: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    warmup: int,
    iterations: int,
    repeats: int,
) -> float:
    samples: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup):
            function(x)
        torch.cuda.synchronize(x.device)
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                function(x)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000.0 / iterations)
    return statistics.median(samples)


def _time_cpu(
    function: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    warmup: int,
    iterations: int,
    repeats: int,
) -> float:
    samples: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup):
            function(x)
        for _ in range(repeats):
            started = time.perf_counter()
            for _ in range(iterations):
                function(x)
            samples.append(
                (time.perf_counter() - started) * 1_000_000.0 / iterations
            )
    return statistics.median(samples)


def _parse_m(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("M must be a comma-separated list of positive integers")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base", choices=("all", "bf16", "int8"), default="all")
    parser.add_argument("--adapter", choices=("both", "lora", "dora"), default="both")
    parser.add_argument(
        "--implementation",
        choices=("both", "legacy", "optimized"),
        default="both",
        help="Keep the old implementation reachable for regression measurements.",
    )
    parser.add_argument("--m", type=_parse_m, default=DEFAULT_M)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--no-fused-kernel",
        action="store_true",
        help="Measure the portable eager optimized path without optional Triton fusion.",
    )
    parser.add_argument("--compare-merged-delta", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0 or args.repeats <= 0:
        parser.error("warmup must be non-negative; iterations and repeats must be positive")
    if args.no_fused_kernel:
        os.environ["INDEXTTS_LORA_TRITON"] = "0"

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    timer = _time_cuda if device.type == "cuda" else _time_cpu
    base_kinds = ("bf16", "int8") if args.base == "all" else (args.base,)
    adapter_kinds = ("lora", "dora") if args.adapter == "both" else (args.adapter,)

    device_label = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    )
    print(f"device={device_label} K={K} N={N} rank={RANK} dtype=bf16")
    header = (
        f"{'base':<6} {'mode':<5} {'M':>4} {'bare_us':>10} "
        f"{'before_us':>11} {'before/base':>12} {'after_us':>10} {'after/base':>11}"
    )
    if args.compare_merged_delta:
        header += f" {'merged_us':>11} {'merged/base':>12}"
    print(header)
    print("-" * len(header))

    for base_kind in base_kinds:
        try:
            base = _make_base(base_kind, device)
        except ImportError as error:
            print(f"{base_kind:<6} skipped: {error}")
            continue
        for adapter_kind in adapter_kinds:
            use_dora = adapter_kind == "dora"
            optimized, legacy = _make_adapters(base, use_dora)
            merged = (
                _MergedDeltaLoRA(optimized, torch.bfloat16).eval()
                if args.compare_merged_delta and not use_dora
                else None
            )
            for m in args.m:
                x = torch.randn(m, K, device=device, dtype=torch.bfloat16)
                # Time the legacy path first: its extra full FP32 projection can
                # materially change GPU clocks. The adjacent bare and optimized
                # measurements then see the same steady-state clock/power regime.
                before_us = (
                    timer(legacy, x, args.warmup, args.iterations, args.repeats)
                    if args.implementation in ("both", "legacy")
                    else float("nan")
                )
                bare_us = timer(base, x, args.warmup, args.iterations, args.repeats)
                after_us = (
                    timer(optimized, x, args.warmup, args.iterations, args.repeats)
                    if args.implementation in ("both", "optimized")
                    else float("nan")
                )
                row = (
                    f"{base_kind:<6} {adapter_kind:<5} {m:>4} {bare_us:>10.2f} "
                    f"{before_us:>11.2f} {before_us / bare_us:>12.2f} "
                    f"{after_us:>10.2f} {after_us / bare_us:>11.2f}"
                )
                if args.compare_merged_delta:
                    if merged is None:
                        row += f" {'-':>11} {'-':>12}"
                    else:
                        merged_us = timer(
                            merged, x, args.warmup, args.iterations, args.repeats
                        )
                        row += f" {merged_us:>11.2f} {merged_us / bare_us:>12.2f}"
                print(row)


if __name__ == "__main__":
    main()
