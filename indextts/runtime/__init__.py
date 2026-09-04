"""Runtime configuration and memory-management helpers for IndexTTS."""

from .block_swap import BlockSwapConfig, BlockSwapController, enable_block_swap
from .gpu import (
    GpuInfo,
    apply_vram_cap,
    device_from_string,
    gpu_free_gb,
    gpu_total_gb,
    list_gpus,
    memory_stats,
)
from .progress import ProgressReporter, read_progress_file
from .residency import ResidencyManager
from .vram_presets import RuntimeConfig, resolve_preset

__all__ = [
    "BlockSwapConfig",
    "BlockSwapController",
    "GpuInfo",
    "ProgressReporter",
    "ResidencyManager",
    "RuntimeConfig",
    "apply_vram_cap",
    "device_from_string",
    "enable_block_swap",
    "gpu_free_gb",
    "gpu_total_gb",
    "list_gpus",
    "memory_stats",
    "read_progress_file",
    "resolve_preset",
]
