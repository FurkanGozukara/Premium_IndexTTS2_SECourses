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
# Set INDEXTTS_VRAM_EMULATE_GB to a tier size (6, 8, 10, 12, 16, 24 or 32) to make every
# process behave like a card of that size: the allocator is capped at the tier budget,
# the card reports the tier size, and free memory is derived from this process's own
# use. Calibration and tests use it; the app never sets it.
EMULATE_ENV = "INDEXTTS_VRAM_EMULATE_GB"
# A display-attached card keeps roughly this much memory for the driver and desktop.
_EMULATED_DRIVER_RESERVE_GB = 0.45
# The application process holds a CUDA context beside every worker it starts.
_EMULATED_APP_CONTEXT_GB = 0.5
_emulation_applied: set[int] = set()


def emulated_gpu_gb() -> float | None:
    """The emulated card size from the environment, or ``None`` when not emulating."""

    raw = os.environ.get(EMULATE_ENV, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def emulated_cap_gb() -> float | None:
    """Allocator cap for the emulated card: its tier budget."""

    size = emulated_gpu_gb()
    if size is None:
        return None
    from indextts.runtime.vram_presets import auto_tier, tier_budget_gb

    return float(tier_budget_gb(auto_tier(size)))


def apply_emulated_vram_cap(index: int = 0) -> float | None:
    """Cap the allocator of this process at the emulated card budget (once per device)."""

    cap = emulated_cap_gb()
    if cap is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        resolved = int(index)
        if resolved in _emulation_applied:
            return cap
        total_gb = float(torch.cuda.get_device_properties(resolved).total_memory) / _GIB
        fraction = min(1.0, cap / total_gb) if total_gb > 0 else 1.0
        torch.cuda.set_per_process_memory_fraction(fraction, resolved)
        _emulation_applied.add(resolved)
        print(
            f">> VRAM emulation: behaving like a {emulated_gpu_gb():g} GB card; allocator capped at "
            f"{cap:.2f} GB ({fraction:.3f} of cuda:{resolved})",
            flush=True,
        )
    except (RuntimeError, AssertionError, ValueError) as exc:
        print(f">> VRAM emulation could not cap cuda:{index}: {exc}", flush=True)
        return None
    return cap


def _emulated_free_gb(index: int) -> float | None:
    size = emulated_gpu_gb()
    if size is None:
        return None
    try:
        reserved = float(torch.cuda.memory_reserved(int(index))) / _GIB if torch.cuda.is_available() else 0.0
    except (RuntimeError, AssertionError, ValueError):
        reserved = 0.0
    return max(0.0, size - _EMULATED_DRIVER_RESERVE_GB - _EMULATED_APP_CONTEXT_GB - reserved)


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
    emulated = emulated_gpu_gb()
    try:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            except (RuntimeError, TypeError):
                total_bytes = int(props.total_memory)
                free_bytes = max(0, total_bytes - torch.cuda.memory_reserved(index))
            total_gb = float(total_bytes) / _GIB
            free_gb = float(free_bytes) / _GIB
            if emulated is not None:
                total_gb = min(total_gb, emulated)
                free_gb = min(free_gb, _emulated_free_gb(index) or 0.0)
            result.append(
                GpuInfo(
                    index=index,
                    name=str(props.name) + (f" (emulated {emulated:g} GB)" if emulated is not None else ""),
                    total_gb=total_gb,
                    free_gb=free_gb,
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
    emulated = emulated_gpu_gb()
    try:
        total = float(torch.cuda.get_device_properties(int(index)).total_memory) / _GIB
    except (RuntimeError, AssertionError, ValueError):
        match = next((gpu for gpu in list_gpus() if gpu.index == int(index)), None)
        total = float(match.total_gb) if match else 0.0
    return min(total, emulated) if emulated is not None and total > 0 else total


def gpu_free_gb(index: int = 0) -> float:
    try:
        free_bytes, _ = torch.cuda.mem_get_info(int(index))
        free = float(free_bytes) / _GIB
    except (RuntimeError, AssertionError, ValueError):
        match = next((gpu for gpu in list_gpus() if gpu.index == int(index)), None)
        free = float(match.free_gb) if match else 0.0
    emulated_free = _emulated_free_gb(int(index))
    return min(free, emulated_free) if emulated_free is not None else free


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
    "EMULATE_ENV",
    "GpuInfo",
    "apply_emulated_vram_cap",
    "apply_vram_cap",
    "emulated_cap_gb",
    "emulated_gpu_gb",
    "device_from_string",
    "format_gb",
    "gpu_free_gb",
    "gpu_total_gb",
    "list_gpus",
    "memory_stats",
]
