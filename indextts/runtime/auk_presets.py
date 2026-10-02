"""Memory budgets for AuK: a 1.5B flow transformer plus the 3B Qwen2.5-Omni Thinker encoder.

Measured whole-process peaks (GiB, CUDA context included; RTX A6000, 34 s clone request,
docs/AUK.md): BF16 transformer and encoder resident 12.0 at section batch 8; BF16
transformer with the INT8 encoder (token table in CPU memory) 8.7 at batch 4; both INT8 7.3.
On demand, the encoder and the transformer with the VAE take turns on the GPU, one loan
each per request: 5.0 (BF16 transformer) or 4.6 (INT8). Audio is bit-identical across
residencies. INT8 saves memory, not time (13-28 % slower), so the BF16 transformer stays
wherever it fits.
"""

from .vram_presets import RuntimeConfig, auto_tier

LEARNING_RATE = {"full": 1e-5, "adapter": 1e-4}
FULL_MIN_TIER = 32

# tier: (transformer variant, text encoder variant, text encoder residency, section batch hint)
_INFERENCE = {
    32: ("bf16", "bf16", "gpu", 8),
    24: ("bf16", "bf16", "gpu", 8),
    16: ("bf16", "bf16", "gpu", 8),
    12: ("bf16", "int8_convrot", "gpu", 4),
    10: ("int8_convrot", "int8_convrot", "gpu", 4),
    8: ("bf16", "int8_convrot", "on_demand", 2),
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


# Latent frames per micro-batch by training tier (50 frames per second of target audio).
_TRAINING_FRAMES = {6: 600, 8: 800, 10: 1000, 12: 1200, 16: 1600, 24: 2400, 32: 2700}
UPDATE_FRAMES = 5400  # two upstream micro-batches per optimizer update (about 108 s of audio)


def resolve_training_preset(tier, method="dora"):
    """Memory and learning settings of one GPU tier and training method (provisional; see docs/AUK.md)."""
    selected = max(key for key in _TRAINING_FRAMES if key <= max(6, int(tier)))
    full = method == "full"
    frames = _TRAINING_FRAMES[selected]
    return {"base_variant": "bf16" if full or selected >= 16 else "int8_convrot", "base_dtype": "bf16",
            # Full fine-tuning peaks in the optimizer step, so checkpointing would only cost speed (65 %).
            "mixed_precision": "bf16", "blocks_to_swap": 0, "gradient_checkpointing": not full,
            "swap_ring_size": 2, "pin_swap_memory": False, "batch_size": 4, "auk_batch_frames": frames,
            "grad_accumulation": max(1, round(UPDATE_FRAMES / frames)),
            "learning_rate": LEARNING_RATE["full" if full else "adapter"], "keep_last_n": 2 if full else 0,
            "train_mel_embed_head": not full, "sample_runtime_tier": str(selected), "sample_min_free_vram_gb": 6.0}


def training_note(tier, method="dora"):
    values = resolve_training_preset(tier, method)
    precision = "ConvRot INT8" if values["base_variant"] == "int8_convrot" else "BF16"
    frames = int(values["auk_batch_frames"])
    return (f"**AuK {str(method).upper()}: {tier} GB tier profile.** {precision} transformer base, {frames:,} latent "
            f"frames per micro-batch x accumulation {values['grad_accumulation']} (about "
            f"{frames * values['grad_accumulation'] / 50:.0f} s of audio per update), gradient checkpointing "
            f"{'on' if values['gradient_checkpointing'] else 'off'}, learning rate {values['learning_rate']:g}. "
            "Without reference prompts the Qwen2.5-Omni encoder is not loaded "
            "(its conditioning is cached); with them it is resident in BF16.")


def preset_notes(tier):
    cfg = resolve_preset(tier)
    parts = {
        "gpu": "stays on the GPU",
        "on_demand": ("waits in CPU memory and takes turns with the transformer: the text encoder is lent to the GPU "
                      "while a request is encoded, then the transformer while it is sampled"),
    }
    return (f"AuK {cfg.vram_tier} GB preset: {cfg.model_variant.replace('_convrot', ' ConvRot').upper()} transformer; "
            f"{cfg.auk_text_encoder_variant.replace('_convrot', ' ConvRot').upper()} Qwen2.5-Omni encoder that "
            f"{parts[cfg.auk_text_encoder_residency]}; float32 VAE. Section batch hint "
            f"{cfg.max_section_batch_size_hint}. Long references and long sections use more memory; run the VRAM "
            "benchmark to measure your workload.")
