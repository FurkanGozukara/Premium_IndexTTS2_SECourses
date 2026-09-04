"""Benchmark ConvRotInt8Linear decoding and prefill kernels on CUDA."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from indextts.quant.convrot_int8 import (
    ConvRotInt8Linear,
    _int8_gemm_supported,
)


SHAPES = ((1280, 3840), (1280, 5120), (5120, 1280))
ROWS = (1, 512)


def _benchmark(
    call: Callable[[], torch.Tensor], *, warmup: int, repeats: int
) -> float:
    for _ in range(warmup):
        result = call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        result = call()
    end.record()
    end.synchronize()
    del result
    return float(start.elapsed_time(end) * 1000.0 / repeats)


def _format(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f} us"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--decode-repeats", type=int, default=1000)
    parser.add_argument("--prefill-repeats", type=int, default=300)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("A CUDA device is required")
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    has_w8a8 = _int8_gemm_supported(device)
    print(f"{torch.cuda.get_device_name(device)} | torch {torch.__version__} | bf16")

    headings = ["shape"]
    for rows in ROWS:
        headings.extend(
            [
                f"M={rows} W8A16",
                f"M={rows} W8A8",
                f"M={rows} auto",
                f"M={rows} bf16",
            ]
        )
    widths = [18] + [14] * (len(headings) - 1)
    print("| " + " | ".join(h.ljust(w) for h, w in zip(headings, widths)) + " |")
    print("|-" + "-|-".join("-" * w for w in widths) + "-|")

    with torch.inference_mode():
        for k, n in SHAPES:
            layer = ConvRotInt8Linear(
                k,
                n,
                bias=False,
                group_size=256,
                device=device,
                dtype=dtype,
            )
            layer.weight_int8.random_(-127, 128)
            layer.weight_scale.fill_(0.0005)
            bf16_weight = torch.randn((n, k), device=device, dtype=dtype)
            values: list[str] = [f"K={k}, N={n}"]
            for rows in ROWS:
                x = torch.randn((rows, k), device=device, dtype=dtype)
                repeats = (
                    args.decode_repeats if rows == 1 else args.prefill_repeats
                )
                timings: list[float | None] = []
                for mode in ("w8a16", "w8a8", "auto"):
                    if mode == "w8a8" and not has_w8a8:
                        timings.append(None)
                        continue
                    layer.kernel_mode = mode
                    timings.append(
                        _benchmark(
                            lambda: layer(x),
                            warmup=args.warmup,
                            repeats=repeats,
                        )
                    )
                timings.append(
                    _benchmark(
                        lambda: F.linear(x, bf16_weight),
                        warmup=args.warmup,
                        repeats=repeats,
                    )
                )
                values.extend(_format(value) for value in timings)
            print(
                "| "
                + " | ".join(
                    value.ljust(width) for value, width in zip(values, widths)
                )
                + " |"
            )
            del layer, bf16_weight


if __name__ == "__main__":
    main()
