"""Small, dependency-light GPU inventory and memory helpers."""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
from dataclasses import dataclass
from io import StringIO
from typing import Any

import torch


_GIB = float(1024**3)


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    total_gb: float
    free_gb: float
    is_default: bool = False


def _torch_gpus() -> list[GpuInfo]:
    if not torch.cuda.is_available():
        return []
    result: list[GpuInfo] = []
    try:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            except (RuntimeError, TypeError):
                total_bytes = int(props.total_memory)
                free_bytes = max(0, total_bytes - torch.cuda.memory_reserved(index))
            result.append(
                GpuInfo(
                    index=index,
                    name=str(props.name),
                    total_gb=float(total_bytes) / _GIB,
                    free_gb=float(free_bytes) / _GIB,
                    is_default=index == torch.cuda.current_device(),
                )
            )
    except (RuntimeError, AssertionError):
        return []
    return result


def _smi_gpus() -> list[GpuInfo]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode != 0:
        return []

    default_index = 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",", 1)[0].strip()
    if visible.isdigit():
        default_index = int(visible)
    try:
        rows = csv.reader(StringIO(completed.stdout))
        return [
            GpuInfo(
                index=int(row[0].strip()),
                name=row[1].strip(),
                total_gb=float(row[2].strip()) / 1024.0,
                free_gb=float(row[3].strip()) / 1024.0,
                is_default=int(row[0].strip()) == default_index,
            )
            for row in rows
            if len(row) >= 4
        ]
    except (TypeError, ValueError, IndexError):
        return []


def list_gpus() -> list[GpuInfo]:
    """Return CUDA-visible GPUs, falling back to physical ``nvidia-smi`` data."""

    return _torch_gpus() or _smi_gpus()


def _device_index(device: int | str | torch.device | None) -> int:
    if isinstance(device, int):
        return device
    resolved = torch.device("cuda:0" if device is None else device)
    if resolved.type != "cuda":
        raise ValueError(f"Expected a CUDA device, got {resolved}")
    return torch.cuda.current_device() if resolved.index is None else resolved.index


def gpu_total_gb(index: int = 0) -> float:
    try:
        return float(torch.cuda.get_device_properties(int(index)).total_memory) / _GIB
    except (RuntimeError, AssertionError, ValueError):
        match = next((gpu for gpu in list_gpus() if gpu.index == int(index)), None)
        return float(match.total_gb) if match else 0.0


def gpu_free_gb(index: int = 0) -> float:
    try:
        free_bytes, _ = torch.cuda.mem_get_info(int(index))
        return float(free_bytes) / _GIB
    except (RuntimeError, AssertionError, ValueError):
        match = next((gpu for gpu in list_gpus() if gpu.index == int(index)), None)
        return float(match.free_gb) if match else 0.0


def memory_stats(device: int | str | torch.device = "cuda:0") -> dict[str, float]:
    """Return allocator statistics in GiB; non-CUDA devices report zeroes."""

    try:
        resolved = torch.device(device)
    except (TypeError, RuntimeError):
        resolved = torch.device("cpu")
    keys = ("allocated_gb", "reserved_gb", "peak_allocated_gb", "peak_reserved_gb")
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return {key: 0.0 for key in keys}
    index = _device_index(resolved)
    return {
        "allocated_gb": torch.cuda.memory_allocated(index) / _GIB,
        "reserved_gb": torch.cuda.memory_reserved(index) / _GIB,
        "peak_allocated_gb": torch.cuda.max_memory_allocated(index) / _GIB,
        "peak_reserved_gb": torch.cuda.max_memory_reserved(index) / _GIB,
    }


def apply_vram_cap(device: int | str | torch.device, cap_gb: float) -> float:
    """Apply a process allocator cap and return the fraction passed to Torch."""

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required to apply a VRAM cap")
    index = _device_index(device)
    total_gb = gpu_total_gb(index)
    if total_gb <= 0:
        raise RuntimeError(f"Unable to determine total memory for cuda:{index}")
    requested = float(cap_gb)
    if requested <= 0:
        raise ValueError("cap_gb must be greater than zero")
    fraction = min(1.0, requested / total_gb)
    torch.cuda.set_per_process_memory_fraction(fraction, index)
    return fraction


def format_gb(value: Any) -> str:
    try:
        return f"{float(value):.2f} GB"
    except (TypeError, ValueError):
        return "0.00 GB"


def device_from_string(value: str | torch.device | None) -> torch.device:
    """Resolve ``auto``, ``cpu`` or a CUDA device string to ``torch.device``."""

    if isinstance(value, torch.device):
        return value
    text = str(value or "auto").strip().lower()
    if text == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    try:
        resolved = torch.device(text)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid runtime device {value!r}") from exc
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {resolved} was requested, but CUDA is unavailable")
    return resolved


__all__ = [
    "GpuInfo",
    "apply_vram_cap",
    "device_from_string",
    "format_gb",
    "gpu_free_gb",
    "gpu_total_gb",
    "list_gpus",
    "memory_stats",
]
