"""Memory budgets for AuK: a 1.5B flow transformer plus the 3B Qwen2.5-Omni Thinker encoder.

Weights (GB): Thinker BF16 7.5 (INT8 ConvRot about 4.5), transformer BF16 3.1 (INT8 about 1.6),
VAE 0.6 (float32). Measured on an RTX A6000 with BF16 everywhere: 10.5 GB resident,
11.0 GB peak for a 30 s context. The smaller tiers trade speed for memory: INT8 weights,
then the encoder waiting in pinned CPU memory between encodes ("on_demand"), which also
parks the transformer while the encoder runs.
"""

from .vram_presets import RuntimeConfig, auto_tier

LEARNING_RATE = {"full": 1e-5, "adapter": 1e-4}
FULL_MIN_TIER = 32

# tier: (transformer variant, text encoder variant, text encoder residency, section batch hint)
_INFERENCE = {
    32: ("bf16", "bf16", "gpu", 8),
    24: ("bf16", "bf16", "gpu", 4),
    16: ("bf16", "bf16", "gpu", 2),
    12: ("bf16", "int8_convrot", "gpu", 1),
    10: ("int8_convrot", "int8_convrot", "gpu", 1),
    8: ("int8_convrot", "int8_convrot", "on_demand", 1),
    6: ("int8_convrot", "int8_convrot", "on_demand", 1),
}


def _tier(tier, total_gb=32):
    return auto_tier(total_gb or 6) if str(tier) in {"auto", "custom"} else int(tier)


def resolve_preset(tier, total_gb=32, free_gb=None):
    selected = _tier(tier, total_gb)
    selected = max(key for key in _INFERENCE if key <= max(6, selected))
    dit, text, residency, batch = _INFERENCE[selected]
    return RuntimeConfig(vram_tier=str(selected), model_variant=dit, gpt_dtype="bf16", blocks_to_swap=0,
                         use_qwen_emo=False, decoder_adapter="none", vram_reserve_gb=1 if selected <= 16 else 2,
                         max_section_batch_size_hint=batch, auk_text_encoder_variant=text,
                         auk_text_encoder_residency=residency)


def default_training_method(tier):
    """Full fine-tuning of the 1.5B transformer needs FP32 weights, gradients and optimizer state."""
    return "full" if int(tier) >= FULL_MIN_TIER else "dora"


def resolve_training_preset(tier, method="dora"):
    selected = int(tier)
    full = method == "full"
    return {"base_variant": "bf16" if full or selected >= 16 else "int8_convrot", "base_dtype": "bf16",
            "mixed_precision": "bf16", "blocks_to_swap": 0, "gradient_checkpointing": full or selected < 32,
            "swap_ring_size": 2, "pin_swap_memory": False, "batch_size": 1,
            "learning_rate": LEARNING_RATE["full" if full else "adapter"], "keep_last_n": 2 if full else 0,
            "train_mel_embed_head": False, "sample_runtime_tier": str(selected), "sample_min_free_vram_gb": 6.0}


def preset_notes(tier):
    cfg = resolve_preset(tier)
    parts = {
        "gpu": "stays on the GPU",
        "on_demand": "waits in CPU memory and is lent to the GPU for each encode (the transformer steps aside meanwhile)",
    }
    return (f"AuK {cfg.vram_tier} GB preset: {cfg.model_variant.replace('_convrot', ' ConvRot').upper()} transformer; "
            f"{cfg.auk_text_encoder_variant.replace('_convrot', ' ConvRot').upper()} Qwen2.5-Omni encoder that "
            f"{parts[cfg.auk_text_encoder_residency]}; float32 VAE. Section batch hint "
            f"{cfg.max_section_batch_size_hint}. Long references and long sections use more memory; run the VRAM "
            "benchmark to measure your workload.")
