"""Runtime configuration and memory-management helpers for IndexTTS."""

from .block_swap import BlockSwapConfig, BlockSwapController, enable_block_swap
from .gpu import (
    GpuInfo,
    apply_emulated_vram_cap,
    apply_vram_cap,
    device_from_string,
    emulated_gpu_gb,
    gpu_free_gb,
    gpu_total_gb,
    list_gpus,
    memory_stats,
)

# Every model process imports this package, so an emulated card size in the
# environment caps each of them before any model is loaded.
if emulated_gpu_gb() is not None:
    apply_emulated_vram_cap()
from .progress import ProgressReporter, read_progress_file
from .residency import ResidencyManager
from .vram_presets import RuntimeConfig, resolve_preset

__all__ = [
    "BlockSwapConfig",
    "BlockSwapController",
    "GpuInfo",
    "ProgressReporter",
    "apply_emulated_vram_cap",
    "emulated_gpu_gb",
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
