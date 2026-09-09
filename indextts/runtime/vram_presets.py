"""VRAM presets and the JSON contract shared by the UI and inference engine."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
import re
from typing import Any, Mapping


VRAM_TIERS = [6, 8, 10, 12, 16, 24, 32]

# A card counts as a tier from 500 MB below the tier's nominal size, because
# drivers advertise slightly less than the marketing figure (a 32 GB card
# reports 31.8 GB, a 10 GB card 9.9 GB).
AUTO_TIER_TOLERANCE_GB = 0.5

# Whole-GPU peak (GiB as reported by nvidia-smi for every process on the GPU,
# CUDA contexts included) that a tier's presets may reach. Cards up to 16 GB
# keep 1 GB free; 24 GB and 32 GB cards keep 2 GB free.
TIER_BUDGET_GB: dict[int, float] = {6: 5.0, 8: 7.0, 10: 9.0, 12: 11.0, 16: 15.0, 24: 22.0, 32: 30.0}


def tier_budget_gb(tier: str | int | float) -> float:
    """Whole-GPU memory a tier's presets may use at their peak."""

    return TIER_BUDGET_GB[_normalize_tier(tier, 32.0)]


def tier_reserve_gb(tier: str | int | float) -> float:
    """VRAM a tier deliberately leaves free (1 GB up to 16 GB, 2 GB above)."""

    resolved = _normalize_tier(tier, 32.0)
    return float(resolved) - TIER_BUDGET_GB[resolved]


def fit_tier_to_free_vram(tier: str | int | float, free_gb: float) -> int:
    """Shrink a requested tier until its budget fits into the memory that is free now.

    Used when a second model process must share the GPU with a running job, such
    as the epoch sample rendered while training holds its model. The smallest tier
    is returned when nothing fits; callers gate on their own free-memory threshold.
    """

    requested = _normalize_tier(tier, 32.0)
    try:
        available = float(free_gb)
    except (TypeError, ValueError):
        available = 0.0
    eligible = [item for item in VRAM_TIERS if item <= requested and TIER_BUDGET_GB[item] <= available]
    return eligible[-1] if eligible else VRAM_TIERS[0]


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
    use_qwen_emo: bool = True
    use_deepspeed: bool = False
    torch_compile_s2mel: bool = False
    use_cuda_kernel_bigvgan: bool = False
    s2mel_estimator_autocast: bool = False
    cfm_cache_length: int = 8192
    vram_reserve_gb: float = 2.0
    vram_tier: str = "auto"
    lora_path: str = ""
    lora_strength: float = 1.0
    lora_merge_into_base: bool = False
    # Voice decoder adapter choice: "auto" (the file saved with the selected LoRA / DoRA), "none", or an
    # explicit decoder adapter file.
    decoder_adapter: str = "auto"
    # Strength of the voice decoder adapter, independent of the GPT adapter's strength.
    decoder_adapter_strength: float = 1.0
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
        if "decoder_adapter" not in raw:
            # Earlier builds stored a boolean switch and, briefly, a separate explicit path.
            if raw.get("decoder_adapter_path"):
                raw["decoder_adapter"] = str(raw["decoder_adapter_path"])
            elif "use_decoder_adapter" in raw:
                raw["decoder_adapter"] = "auto" if _as_bool(raw["use_decoder_adapter"], True) else "none"

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
        self.use_qwen_emo = _as_bool(self.use_qwen_emo, True)
        self.use_deepspeed = _as_bool(self.use_deepspeed, False)
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
        choice = str(self.decoder_adapter or "auto").strip()
        self.decoder_adapter = choice.lower() if choice.lower() in {"auto", "none", ""} else choice
        self.decoder_adapter = self.decoder_adapter or "auto"
        self.decoder_adapter_strength = _clamp_float(self.decoder_adapter_strength, 0.0, 4.0, 1.0)
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
    eligible = [tier for tier in VRAM_TIERS if tier <= capacity + AUTO_TIER_TOLERANCE_GB]
    return eligible[-1] if eligible else VRAM_TIERS[0]


# Inference runtime per tier. Quality is kept as long as possible: every tier keeps
# the BF16 GPT and pays for a smaller card with speed first (auxiliary models on
# demand or on CPU, a shorter CFM cache, block streaming) and only then with
# decoding settings. Calibrated with ``tools/gpu_tier_calibration.py`` on GPU 0
# (whole-GPU peaks, every process and CUDA context included): 12 GB and above
# keep everything resident (8.7 GB peak with a LoRA and its decoder adapter);
# 10 GB moves the two large reference models on demand (6.9 GB); 8 GB runs the
# semantic encoder on CPU, which frees the 2.2 GB it needs on the GPU during
# reference encoding, so the GPT no longer streams blocks (5.6 GB, twice as fast
# as streaming twelve); 6 GB streams 22 of 24 blocks through one ring slot with
# every reference encoder on CPU (4.3 GB). The measured table is in
# ARCHITECTURE_NOTES.md.
_PRESETS: dict[int, dict[str, Any]] = {
    32: {"model_variant": "bf16", "blocks_to_swap": 0, "ring": 2, "semantic_model": "gpu", "campplus": "gpu", "qwen_emo": "gpu", "cfm": 8192, "s2mel_bf16": False},
    24: {"model_variant": "bf16", "blocks_to_swap": 0, "ring": 2, "semantic_model": "gpu", "campplus": "gpu", "qwen_emo": "gpu", "cfm": 8192, "s2mel_bf16": False},
    16: {"model_variant": "bf16", "blocks_to_swap": 0, "ring": 2, "semantic_model": "gpu", "campplus": "gpu", "qwen_emo": "gpu", "cfm": 8192, "s2mel_bf16": False},
    12: {"model_variant": "bf16", "blocks_to_swap": 0, "ring": 2, "semantic_model": "gpu", "campplus": "gpu", "qwen_emo": "gpu", "cfm": 8192, "s2mel_bf16": False},
    10: {"model_variant": "bf16", "blocks_to_swap": 0, "ring": 2, "semantic_model": "on_demand", "campplus": "gpu", "qwen_emo": "on_demand", "cfm": 6144, "s2mel_bf16": False},
    8: {"model_variant": "bf16", "blocks_to_swap": 0, "ring": 2, "semantic_model": "cpu", "campplus": "gpu", "qwen_emo": "on_demand", "cfm": 4096, "s2mel_bf16": False},
    6: {"model_variant": "bf16", "blocks_to_swap": 22, "ring": 1, "semantic_model": "cpu", "campplus": "cpu", "qwen_emo": "cpu", "cfm": 2048, "s2mel_bf16": True},
}


# Upper limits used by the stress benchmark and the section-batch hint.
_HINTS: dict[int, dict[str, int]] = {
    6: {"num_beams_max": 2, "section_batch_size_max": 1, "max_text_tokens_per_segment": 60, "cfm_cache_length": 2048},
    8: {"num_beams_max": 4, "section_batch_size_max": 2, "max_text_tokens_per_segment": 80, "cfm_cache_length": 4096},
    10: {"num_beams_max": 4, "section_batch_size_max": 2, "max_text_tokens_per_segment": 100, "cfm_cache_length": 6144},
    12: {"num_beams_max": 4, "section_batch_size_max": 4, "max_text_tokens_per_segment": 120, "cfm_cache_length": 8192},
    16: {"num_beams_max": 4, "section_batch_size_max": 4, "max_text_tokens_per_segment": 120, "cfm_cache_length": 8192},
    24: {"num_beams_max": 6, "section_batch_size_max": 8, "max_text_tokens_per_segment": 160, "cfm_cache_length": 8192},
    32: {"num_beams_max": 8, "section_batch_size_max": 8, "max_text_tokens_per_segment": 200, "cfm_cache_length": 8192},
}


# Generation settings a tier preset selects. Every tier keeps sampling, CFM
# temperature 0.9 and at least the 40 diffusion steps of the former quality
# preset; 24 GB and 32 GB cards refine with 50 steps, which costs time, not
# memory. Four beams are the measured optimum: the decoding sweeps of real
# training runs scored five beams worse than three on the speech benchmark, so
# no tier goes higher, and only the 6 GB tier drops to two because the
# key/value cache grows with the beam count.
_GENERATION: dict[int, dict[str, Any]] = {
    32: {"num_beams": 4, "low_memory_mode": False, "diffusion_steps": 50},
    24: {"num_beams": 4, "low_memory_mode": False, "diffusion_steps": 50},
    16: {"num_beams": 4, "low_memory_mode": False, "diffusion_steps": 40},
    12: {"num_beams": 4, "low_memory_mode": False, "diffusion_steps": 40},
    10: {"num_beams": 4, "low_memory_mode": False, "diffusion_steps": 40},
    8: {"num_beams": 4, "low_memory_mode": True, "diffusion_steps": 40},
    6: {"num_beams": 2, "low_memory_mode": True, "diffusion_steps": 40},
}
QUALITY_DIFFUSION_STEPS = 40
BEST_DIFFUSION_STEPS = 50
QUALITY_CFM_TEMPERATURE = 0.9


# LoRA / DoRA training per tier. Training keeps the BF16 base, rank 128, batch
# size 1 and gradient checkpointing everywhere (checkpointing changed neither the
# 2.8 GB allocated peak nor the step speed on this model, so it stays on). The
# resident training run peaks near 4 GB of whole-GPU use, which fits every tier
# from 8 GB up; only the 6 GB tier streams frozen GPT blocks from CPU (3.2 GB).
_TRAINING: dict[int, dict[str, Any]] = {
    32: {"base_variant": "bf16", "blocks_to_swap": 0, "ring": 2, "gradient_checkpointing": True},
    24: {"base_variant": "bf16", "blocks_to_swap": 0, "ring": 2, "gradient_checkpointing": True},
    16: {"base_variant": "bf16", "blocks_to_swap": 0, "ring": 2, "gradient_checkpointing": True},
    12: {"base_variant": "bf16", "blocks_to_swap": 0, "ring": 2, "gradient_checkpointing": True},
    10: {"base_variant": "bf16", "blocks_to_swap": 0, "ring": 2, "gradient_checkpointing": True},
    8: {"base_variant": "bf16", "blocks_to_swap": 0, "ring": 2, "gradient_checkpointing": True},
    6: {"base_variant": "bf16", "blocks_to_swap": 22, "ring": 1, "gradient_checkpointing": True},
}
# Free VRAM a second model process (the per-epoch sample) needs before it is
# started beside a running training job; the sample shrinks to the tier that
# fits into the free memory, and the smallest tier needs about 3.5 GB with its
# CUDA context. Measured beside the resident training run, an 8 GB card has
# about 4 GB free and skips the sample; a 10 GB card keeps it.
TRAINING_SAMPLE_MIN_FREE_GB = 4.5


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
        vram_reserve_gb=tier_reserve_gb(resolved),
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


def generation_preset(tier: str | int | float) -> dict[str, Any]:
    """Generation settings selected by a tier preset (beams, batch, cache, quality decoding)."""

    resolved = _normalize_tier(tier, 32.0)
    row = _GENERATION[resolved]
    return {
        "num_beams": int(row["num_beams"]),
        "section_batch_size": 1,
        "max_text_tokens_per_segment": 60,
        "cfm_cache_length": int(_PRESETS[resolved]["cfm"]),
        "low_memory_mode": bool(row["low_memory_mode"]),
        "diffusion_steps": int(row["diffusion_steps"]),
        "cfm_temperature": QUALITY_CFM_TEMPERATURE,
    }


def resolve_training_preset(tier: str | int | float, gpu_total_gb: float | None = None) -> dict[str, Any]:
    """Training settings selected by a tier preset, keyed like ``TrainConfig`` fields."""

    resolved = _normalize_tier(tier, 32.0 if gpu_total_gb is None else gpu_total_gb)
    row = _TRAINING[resolved]
    return {
        "vram_tier": str(resolved),
        "base_variant": str(row["base_variant"]),
        "base_dtype": "bf16",
        "mixed_precision": "bf16",
        "gradient_checkpointing": bool(row["gradient_checkpointing"]),
        "blocks_to_swap": int(row["blocks_to_swap"]),
        "swap_ring_size": int(row["ring"]),
        "pin_swap_memory": True,
        "sample_runtime_tier": "auto",
        "sample_min_free_vram_gb": TRAINING_SAMPLE_MIN_FREE_GB,
    }


def preset_notes(tier: str | int | float) -> str:
    resolved = _normalize_tier(tier, 32.0)
    config = resolve_preset(resolved, float(resolved))
    generation = generation_preset(resolved)
    training = resolve_training_preset(resolved)
    if resolved >= 12:
        detail = "All core and auxiliary models remain resident for fastest repeated generation."
    elif resolved == 10:
        detail = "GPT stays resident while the two large reference-only models move on demand."
    elif resolved == 8:
        detail = (
            "GPT stays resident; the semantic reference encoder runs on CPU (a few seconds per new reference) "
            "and the emotion-text model moves on demand, with a 4096-frame CFM cache."
        )
    else:
        detail = (
            f"Keeps the BF16 GPT but streams {config.blocks_to_swap}/24 GPT blocks through one ring slot, "
            "keeps the reference encoders and the emotion-text model on CPU, uses a 2048-frame CFM cache, "
            "and runs only the s2mel DiT estimator under BF16 autocast."
        )
    training_detail = (
        f"Training keeps the BF16 base with rank 128 and streams {training['blocks_to_swap']} of 24 frozen blocks."
        if training["blocks_to_swap"]
        else "Training keeps the BF16 base with rank 128 fully resident"
        + ("." if training["gradient_checkpointing"] else " without gradient checkpointing.")
    )
    return (
        f"{resolved} GB preset: {detail} Generation uses {generation['num_beams']} beams, section batch 1, "
        f"{generation['diffusion_steps']} diffusion steps. {training_detail} "
        f"Peak use stays within {tier_budget_gb(resolved):.0f} GB, leaving about {config.vram_reserve_gb:.0f} GB free."
    )


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
        f"LoRA / DoRA merge: {cfg.lora_merge_into_base}"
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
    "AUTO_TIER_TOLERANCE_GB",
    "BEST_DIFFUSION_STEPS",
    "QUALITY_CFM_TEMPERATURE",
    "QUALITY_DIFFUSION_STEPS",
    "RuntimeConfig",
    "TIER_BUDGET_GB",
    "TRAINING_SAMPLE_MIN_FREE_GB",
    "VRAM_TIERS",
    "auto_tier",
    "describe",
    "estimate_vram_gb",
    "fit_tier_to_free_vram",
    "generation_hints",
    "generation_preset",
    "preset_notes",
    "resolve_preset",
    "resolve_training_preset",
    "tier_budget_gb",
    "tier_reserve_gb",
]
