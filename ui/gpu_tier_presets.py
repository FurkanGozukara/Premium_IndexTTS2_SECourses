"""GPU VRAM tier presets: the read-only system presets, one per supported card size.

Each preset stores every registered control of every tab. It starts from the
registry defaults and applies the tier's inference runtime, its generation
decoding settings and its training settings, so one selection fits generation,
dataset preparation and LoRA / DoRA training on that card. The tables themselves
live in ``indextts.runtime.vram_presets``; this module maps them onto the flat
preset keys and knows which tier the machine's GPU belongs to.
"""

from __future__ import annotations

import re
from typing import Any

from indextts.runtime.vram_presets import (
    VRAM_TIERS,
    auto_tier,
    generation_preset,
    resolve_preset,
    resolve_training_preset,
    tier_budget_gb,
)


TIER_PRESET_SUFFIX = " GB GPU"
# System presets that older builds shipped; they are replaced by the tier presets.
LEGACY_SYSTEM_PRESETS = ("default", "quality", "fast", "low_vram_8gb")
# Runtime values that describe a voice, a device choice or an optional loader,
# not the card's memory; a tier preset leaves them at the registry defaults.
_RUNTIME_KEYS_OUTSIDE_TIERS = frozenset(
    {
        "device",
        "lora_path",
        "lora_strength",
        "lora_merge_into_base",
        "decoder_adapter",
        "decoder_adapter_strength",
        "use_qwen_emo",
        "use_deepspeed",
        "attention_backend",
        "use_accel",
        "torch_compile_s2mel",
        "use_cuda_kernel_bigvgan",
        "gpt_dtype",
    }
)
_TIER_NAME_RE = re.compile(r"^\s*(\d+)\s*gb\s*gpu\s*$", re.IGNORECASE)


def tier_preset_name(tier: int | str | float) -> str:
    return f"{int(float(tier))}{TIER_PRESET_SUFFIX}"


TIER_PRESET_NAMES: dict[int, str] = {tier: tier_preset_name(tier) for tier in VRAM_TIERS}


def tier_from_preset_name(name: str | None) -> int | None:
    """Return the tier of a tier preset name, or ``None`` for any other preset."""

    match = _TIER_NAME_RE.match(str(name or "").lstrip("★").strip())
    if match is None:
        return None
    tier = int(match.group(1))
    return tier if tier in VRAM_TIERS else None


def detected_gpu_total_gb() -> float:
    """Advertised memory of the default CUDA GPU in GiB, or 0 without a GPU."""

    try:
        from indextts.runtime.gpu import list_gpus

        gpus = list_gpus()
    except Exception:
        return 0.0
    if not gpus:
        return 0.0
    default = next((gpu for gpu in gpus if gpu.is_default), gpus[0])
    return float(default.total_gb)


def detect_gpu_tier(total_gb: float | None = None) -> int:
    """Tier of the machine's GPU; the smallest tier when no CUDA GPU is present."""

    total = detected_gpu_total_gb() if total_gb is None else float(total_gb)
    return auto_tier(total) if total > 0 else VRAM_TIERS[0]


def tier_registry_overrides(tier: int | str | float) -> dict[str, Any]:
    """Flat preset values a tier changes on top of the registry defaults."""

    resolved = int(resolve_preset(tier, float(tier)).vram_tier)
    runtime = resolve_preset(resolved, float(resolved)).to_dict()
    generation = generation_preset(resolved)
    training = resolve_training_preset(resolved)
    values: dict[str, Any] = {}
    for key, value in runtime.items():
        if key == "aux_residency":
            for name, policy in value.items():
                values[f"runtime.aux_residency.{name}"] = policy
        elif key not in _RUNTIME_KEYS_OUTSIDE_TIERS:
            values[f"runtime.{key}"] = value
    values.update(
        {
            "generation.num_beams": generation["num_beams"],
            "generation.section_batch_size": generation["section_batch_size"],
            "generation.max_text_tokens_per_segment": generation["max_text_tokens_per_segment"],
            "generation.cfm_cache_length": generation["cfm_cache_length"],
            "generation.low_memory_mode": generation["low_memory_mode"],
            "generation.diffusion_steps": generation["diffusion_steps"],
            "generation.cfm_temperature": generation["cfm_temperature"],
            "grid.num_beams": generation["num_beams"],
            "grid.diffusion_steps": generation["diffusion_steps"],
            "grid.max_text_tokens_per_segment": generation["max_text_tokens_per_segment"],
        }
    )
    # Training samples, the speech benchmark and the decoding sweep keep their
    # measured defaults (3 beams, 25 steps) on every tier, so comparisons between
    # runs and releases stay comparable.
    for key, value in training.items():
        values[f"training.{key}"] = value
    return values


def tier_preset_summary(tier: int | str | float) -> str:
    resolved = int(resolve_preset(tier, float(tier)).vram_tier)
    return f"{tier_preset_name(resolved)} preset: peak GPU use stays within {tier_budget_gb(resolved):.0f} GB."


__all__ = [
    "LEGACY_SYSTEM_PRESETS",
    "TIER_PRESET_NAMES",
    "TIER_PRESET_SUFFIX",
    "detect_gpu_tier",
    "detected_gpu_total_gb",
    "tier_from_preset_name",
    "tier_preset_name",
    "tier_preset_summary",
    "tier_registry_overrides",
]
