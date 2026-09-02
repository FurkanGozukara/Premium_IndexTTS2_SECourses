"""VRAM presets and the JSON contract shared by the UI and inference engine."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
import re
from typing import Any, Mapping


VRAM_TIERS = [6, 8, 10, 12, 16, 24, 32]


def _default_aux_residency() -> dict[str, str]:
    return {
        "semantic_model": "gpu",
        "qwen_emo": "on_demand",
        "campplus": "gpu",
        "semantic_codec": "gpu",
        "s2mel": "gpu",
        "bigvgan": "gpu",
    }


@dataclass
class RuntimeConfig:
    device: str = "cuda:0"
    model_variant: str = "bf16"
    gpt_dtype: str = "bf16"
    blocks_to_swap: int = 0
    swap_ring_size: int = 2
    pin_swap_memory: bool = True
    aux_residency: dict = field(default_factory=_default_aux_residency)
    attention_backend: str = "sdpa"
    use_accel: bool = False
    torch_compile_s2mel: bool = False
    use_cuda_kernel_bigvgan: bool = False
    s2mel_estimator_autocast: bool = False
    cfm_cache_length: int = 8192
    vram_reserve_gb: float = 2.0
    vram_tier: str = "auto"
    lora_path: str = ""
    lora_strength: float = 1.0
    lora_merge_into_base: bool = False
    max_section_batch_size_hint: int = 8

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return deepcopy(asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | "RuntimeConfig" | None) -> "RuntimeConfig":
        """Build a config while tolerating partial, future, and legacy dictionaries."""

        if isinstance(value, cls):
            return cls(**deepcopy(asdict(value))).validate()
        if not isinstance(value, Mapping):
            return cls().validate()
        raw = dict(value)
        nested = raw.get("runtime")
        if isinstance(nested, Mapping):
            raw = {**raw, **dict(nested)}

        aliases = {
            "block_swap_ring_size": "swap_ring_size",
            "use_pinned_memory_for_block_swap": "pin_swap_memory",
            "pin_memory": "pin_swap_memory",
            "use_torch_compile": "torch_compile_s2mel",
            "use_cuda_kernel": "use_cuda_kernel_bigvgan",
            "variant": "model_variant",
            "dtype": "gpt_dtype",
            "tier": "vram_tier",
            "gpu_device": "device",
        }
        for old, new in aliases.items():
            if new not in raw and old in raw:
                raw[new] = raw[old]
        if "gpt_dtype" not in raw:
            if raw.get("use_bf16") or raw.get("use_fp16"):
                raw["gpt_dtype"] = "bf16"
            elif "use_bf16" in raw:
                raw["gpt_dtype"] = "fp32"

        allowed = {item.name for item in fields(cls)}
        kwargs = {key: deepcopy(item) for key, item in raw.items() if key in allowed}
        try:
            config = cls(**kwargs)
        except TypeError:
            config = cls()
            for key, item in kwargs.items():
                setattr(config, key, item)
        return config.validate()

    def validate(self) -> "RuntimeConfig":
        device = str(self.device or "auto").strip().lower()
        valid_device = bool(re.fullmatch(r"cuda(?::\d+)?|cpu|auto|mps|xpu(?::\d+)?", device))
        if not valid_device:
            device = "auto"
        self.device = device

        variant = str(self.model_variant or "bf16").strip().lower()
        self.model_variant = variant if variant in {"bf16", "int8_convrot"} else "bf16"
        dtype = str(self.gpt_dtype or "bf16").strip().lower()
        dtype_aliases = {"bfloat16": "bf16", "float16": "fp16", "half": "fp16", "float32": "fp32"}
        dtype = dtype_aliases.get(dtype, dtype)
        self.gpt_dtype = dtype if dtype in {"bf16", "fp16", "fp32"} else "bf16"

        self.blocks_to_swap = _clamp_int(self.blocks_to_swap, -1, 24, 0)
        self.swap_ring_size = _clamp_int(self.swap_ring_size, 1, 4, 2)
        self.pin_swap_memory = _as_bool(self.pin_swap_memory, True)

        policies = _default_aux_residency()
        if isinstance(self.aux_residency, Mapping):
            for name in policies:
                policy = str(self.aux_residency.get(name, policies[name])).strip().lower()
                allowed = {"gpu", "on_demand"}
                if name in {"semantic_model", "campplus", "qwen_emo"}:
                    allowed.add("cpu")
                policies[name] = policy if policy in allowed else policies[name]
        self.aux_residency = policies

        backend = str(self.attention_backend or "sdpa").strip().lower()
        self.attention_backend = backend if backend in {"sdpa", "flash_attention_2", "eager"} else "sdpa"
        self.use_accel = _as_bool(self.use_accel, False)
        self.torch_compile_s2mel = _as_bool(self.torch_compile_s2mel, False)
        self.use_cuda_kernel_bigvgan = _as_bool(self.use_cuda_kernel_bigvgan, False)
        self.s2mel_estimator_autocast = _as_bool(self.s2mel_estimator_autocast, False)
        self.cfm_cache_length = _clamp_int(self.cfm_cache_length, 1024, 32768, 8192)
        self.vram_reserve_gb = _clamp_float(self.vram_reserve_gb, 0.0, 32.0, 2.0)

        tier = str(self.vram_tier or "auto").strip().lower()
        valid_tiers = {str(item) for item in VRAM_TIERS} | {"auto", "custom"}
        self.vram_tier = tier if tier in valid_tiers else "auto"
        self.lora_path = str(self.lora_path or "")
        self.lora_strength = _clamp_float(self.lora_strength, 0.0, 4.0, 1.0)
        self.lora_merge_into_base = _as_bool(self.lora_merge_into_base, False)
        self.max_section_batch_size_hint = _clamp_int(self.max_section_batch_size_hint, 1, 64, 8)
        return self


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
        return default
    return bool(value) if value is not None else default


def _clamp_int(value: Any, minimum: int, maximum: int, default: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        result = default
    return min(maximum, max(minimum, result))


def _clamp_float(value: Any, minimum: float, maximum: float, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        result = default
    if result != result:
        result = default
    return min(maximum, max(minimum, result))


def auto_tier(total_gb: float) -> int:
    """Pick the largest supported tier no greater than the advertised capacity."""

    try:
        capacity = max(0.0, float(total_gb))
    except (TypeError, ValueError):
        capacity = 0.0
    # CUDA reports a nominal 32 GB card a little below 32 GiB.
    eligible = [tier for tier in VRAM_TIERS if tier <= capacity + 0.5]
    return eligible[-1] if eligible else VRAM_TIERS[0]


# Calibrated on the RTX 5090 with ``tools/vram_benchmark.py --all --emulate``.
# The measured per-tier load/peak table lives in ARCHITECTURE_NOTES.md.
_PRESETS: dict[int, dict[str, Any]] = {
    32: {"model_variant": "bf16", "blocks_to_swap": 0, "ring": 2, "semantic_model": "gpu", "campplus": "gpu", "qwen_emo": "gpu", "cfm": 8192, "s2mel_bf16": False},
    24: {"model_variant": "bf16", "blocks_to_swap": 0, "ring": 2, "semantic_model": "gpu", "campplus": "gpu", "qwen_emo": "gpu", "cfm": 8192, "s2mel_bf16": False},
    16: {"model_variant": "bf16", "blocks_to_swap": 0, "ring": 2, "semantic_model": "gpu", "campplus": "gpu", "qwen_emo": "on_demand", "cfm": 8192, "s2mel_bf16": False},
    12: {"model_variant": "bf16", "blocks_to_swap": 0, "ring": 2, "semantic_model": "on_demand", "campplus": "gpu", "qwen_emo": "on_demand", "cfm": 8192, "s2mel_bf16": False},
    10: {"model_variant": "bf16", "blocks_to_swap": 8, "ring": 2, "semantic_model": "on_demand", "campplus": "gpu", "qwen_emo": "on_demand", "cfm": 6144, "s2mel_bf16": False},
    8: {"model_variant": "int8_convrot", "blocks_to_swap": 8, "ring": 2, "semantic_model": "on_demand", "campplus": "gpu", "qwen_emo": "on_demand", "cfm": 4096, "s2mel_bf16": False},
    6: {"model_variant": "int8_convrot", "blocks_to_swap": 22, "ring": 1, "semantic_model": "cpu", "campplus": "cpu", "qwen_emo": "cpu", "cfm": 2048, "s2mel_bf16": True},
}


_HINTS: dict[int, dict[str, int]] = {
    6: {"num_beams_max": 1, "section_batch_size_max": 1, "max_text_tokens_per_segment": 40, "cfm_cache_length": 2048},
    8: {"num_beams_max": 2, "section_batch_size_max": 2, "max_text_tokens_per_segment": 80, "cfm_cache_length": 4096},
    10: {"num_beams_max": 3, "section_batch_size_max": 2, "max_text_tokens_per_segment": 100, "cfm_cache_length": 6144},
    12: {"num_beams_max": 3, "section_batch_size_max": 4, "max_text_tokens_per_segment": 120, "cfm_cache_length": 8192},
    16: {"num_beams_max": 4, "section_batch_size_max": 4, "max_text_tokens_per_segment": 120, "cfm_cache_length": 8192},
    24: {"num_beams_max": 6, "section_batch_size_max": 8, "max_text_tokens_per_segment": 160, "cfm_cache_length": 8192},
    32: {"num_beams_max": 8, "section_batch_size_max": 8, "max_text_tokens_per_segment": 200, "cfm_cache_length": 8192},
}


def _normalize_tier(tier: str | int | float, gpu_total_gb: float | None = None) -> int:
    if str(tier).strip().lower() == "auto":
        return auto_tier(0.0 if gpu_total_gb is None else gpu_total_gb)
    try:
        requested = float(tier)
    except (TypeError, ValueError):
        return auto_tier(0.0 if gpu_total_gb is None else gpu_total_gb)
    eligible = [item for item in VRAM_TIERS if item <= requested + 0.01]
    return eligible[-1] if eligible else VRAM_TIERS[0]


def resolve_preset(
    tier: str | int | float,
    gpu_total_gb: float,
    gpu_free_gb: float | None = None,
) -> RuntimeConfig:
    """Resolve the binding preset table for a nominal VRAM tier."""

    del gpu_free_gb  # Free memory is displayed by the UI; it does not alter a named preset.
    resolved = _normalize_tier(tier, gpu_total_gb)
    row = _PRESETS[resolved]
    config = RuntimeConfig(
        model_variant=row["model_variant"],
        gpt_dtype="bf16",
        blocks_to_swap=row["blocks_to_swap"],
        swap_ring_size=row["ring"],
        cfm_cache_length=row["cfm"],
        s2mel_estimator_autocast=row["s2mel_bf16"],
        vram_tier=str(resolved),
        max_section_batch_size_hint=_HINTS[resolved]["section_batch_size_max"],
    )
    config.aux_residency["semantic_model"] = row["semantic_model"]
    config.aux_residency["campplus"] = row["campplus"]
    config.aux_residency["qwen_emo"] = row["qwen_emo"]
    return config.validate()


def generation_hints(tier: str | int | float) -> dict[str, int]:
    resolved = _normalize_tier(tier, 32.0)
    return dict(_HINTS[resolved])


def preset_notes(tier: str | int | float) -> str:
    resolved = _normalize_tier(tier, 32.0)
    config = resolve_preset(resolved, float(resolved))
    if resolved >= 24:
        detail = "All core and auxiliary models remain resident for fastest repeated generation."
    elif resolved >= 16:
        detail = "GPT stays resident; rarely used emotion analysis moves on demand."
    elif resolved == 12:
        detail = "GPT stays resident while the two large reference-only models move on demand."
    elif resolved == 6:
        detail = (
            "Uses INT8 GPT, streams 22/24 GPT blocks through one ring slot, keeps the reference "
            "encoders and Qwen emotion model on CPU, uses a 2048-frame CFM cache, and runs only "
            "the s2mel DiT estimator under BF16 autocast. Beams and section batch are limited to 1. "
            "The strict 4 GB emulated budget measured 2.63 GB peak reserved, so the 2 GB reserve is retained."
        )
    else:
        detail = f"Streams {config.blocks_to_swap} of 24 GPT blocks and moves large auxiliary models on demand."
    return f"{resolved} GB preset: {detail} Keeps about {config.vram_reserve_gb:.0f} GB free for generation peaks."


def describe(config: RuntimeConfig) -> str:
    cfg = RuntimeConfig.from_dict(config.to_dict())
    swap = "auto swap" if cfg.blocks_to_swap == -1 else f"swap {cfg.blocks_to_swap}/24"
    on_demand = [name for name, policy in cfg.aux_residency.items() if policy == "on_demand"]
    cpu = [name for name, policy in cfg.aux_residency.items() if policy == "cpu"]
    aux = ",".join(on_demand) if on_demand else "none"
    cpu_aux = ",".join(cpu) if cpu else "none"
    return (
        f"{cfg.device} | {cfg.model_variant}/{cfg.gpt_dtype} | {cfg.attention_backend} | "
        f"{swap}, ring {cfg.swap_ring_size} | on-demand: {aux} | CPU: {cpu_aux} | "
        f"CFM cache {cfg.cfm_cache_length} | s2mel DiT BF16: {cfg.s2mel_estimator_autocast} | "
        f"LoRA merge: {cfg.lora_merge_into_base}"
    )


def estimate_vram_gb(config: RuntimeConfig, gpu_total_gb: float) -> dict[str, Any]:
    """Return a deliberately rough steady-state and generation-peak VRAM estimate.

    Numbers are model-size estimates, not allocator guarantees. Swapping is modeled only
    for the transformer-block share of GPT, and on-demand models contribute to peak but
    not steady residency.
    """

    cfg = RuntimeConfig.from_dict(config.to_dict())
    gpt_full = 0.92 if cfg.model_variant == "int8_convrot" else {"bf16": 1.58, "fp16": 1.58, "fp32": 3.16}[cfg.gpt_dtype]
    block_share = gpt_full * 0.84
    swapped = max(0, cfg.blocks_to_swap)
    if cfg.blocks_to_swap == -1:
        swapped = 8
    resident_fraction = (24 - min(24, swapped)) / 24.0
    ring_fraction = min(cfg.swap_ring_size, max(0, swapped)) / 24.0
    gpt_resident = gpt_full - block_share + block_share * (resident_fraction + ring_fraction)

    weights = {
        "gpt": gpt_resident,
        "semantic_model": 2.16,
        "qwen_emo": 1.18,
        "campplus": 0.03,
        "semantic_codec": 0.12,
        "s2mel": 0.42,
        "bigvgan": 0.45,
        "other": 0.35,
    }
    resident_aux = sum(
        weights[name]
        for name in cfg.aux_residency
        if cfg.aux_residency[name] == "gpu"
    )
    on_demand_peak = max(
        [weights[name] for name in cfg.aux_residency if cfg.aux_residency[name] == "on_demand"] or [0.0]
    )
    kv_cache = 0.18
    activations = 0.75 + 0.00008 * cfg.cfm_cache_length
    if cfg.s2mel_estimator_autocast:
        activations *= 0.68
    allocator_slack = 0.45
    resident_weights = weights["gpt"] + weights["other"] + resident_aux
    generation_peak = resident_weights + kv_cache + activations + allocator_slack
    reference_peak = resident_weights + on_demand_peak + 0.25 + allocator_slack
    estimated_peak = max(generation_peak, reference_peak)
    return {
        "components_gb": {key: round(value, 3) for key, value in weights.items()},
        "resident_weights_gb": round(resident_weights, 3),
        "on_demand_peak_gb": round(on_demand_peak, 3),
        "kv_cache_gb": round(kv_cache, 3),
        "activations_gb": round(activations, 3),
        "allocator_slack_gb": round(allocator_slack, 3),
        "generation_peak_gb": round(generation_peak, 3),
        "reference_peak_gb": round(reference_peak, 3),
        "estimated_peak_gb": round(estimated_peak, 3),
        "reserve_gb": round(cfg.vram_reserve_gb, 3),
        "headroom_gb": round(float(gpu_total_gb) - estimated_peak, 3),
        "fits": estimated_peak + cfg.vram_reserve_gb <= float(gpu_total_gb),
    }


__all__ = [
    "RuntimeConfig",
    "VRAM_TIERS",
    "auto_tier",
    "describe",
    "estimate_vram_gb",
    "generation_hints",
    "preset_notes",
    "resolve_preset",
]
