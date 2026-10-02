"""Conservative memory budgets for OmniVoice's much smaller transformer."""

from .vram_presets import RuntimeConfig, auto_tier


FULL_MIN_TIER = 16
BATCH_TOKENS = 4096
UPDATE_TOKENS = 8192  # the upstream recipe's tokens per optimizer update
LEARNING_RATE = {"full": 2e-5, "adapter": 1e-4}


def default_training_method(tier):
    """Full fine-tuning where it fits: it measured better and three times faster per update than DoRA."""
    return "full" if int(tier) >= FULL_MIN_TIER else "dora"


def resolve_training_preset(tier, method="dora"):
    """Memory and learning settings of one GPU tier and training method.

    Micro-batches hold a token budget and gradient accumulation completes the
    upstream 8192 tokens per optimizer update. Peak PyTorch memory at 4096 tokens
    (30-step runs, clips up to 16 s, RTX 5090): full fine-tuning 19.5 GB, or
    11.5 GB with gradient checkpointing; rank-32 DoRA 17.1 GB, or 2.7 GB, so the
    BF16 base fits every tier. Full fine-tuning also trains the audio embeddings
    and heads and keeps the last three of its 1.2 GB checkpoints.
    """

    selected = int(tier)
    full = method == "full"
    batch = (1 if selected <= 12 else 2 if selected <= 16 else 4) if full else (
        1 if selected <= 8 else 2 if selected <= 12 else 4 if selected <= 16 else 8)
    return {"base_variant": "bf16", "base_dtype": "bf16", "mixed_precision": "bf16", "blocks_to_swap": 0,
            "gradient_checkpointing": selected < 24, "swap_ring_size": 2, "pin_swap_memory": False,
            "batch_size": batch, "omni_batch_tokens": BATCH_TOKENS, "grad_accumulation": UPDATE_TOKENS // BATCH_TOKENS,
            "train_mel_embed_head": full, "learning_rate": LEARNING_RATE["full" if full else "adapter"],
            "keep_last_n": 3 if full else 0,
            "sample_runtime_tier": str(selected), "sample_min_free_vram_gb": 2.5}


def resolve_preset(tier, total_gb=32, free_gb=None):
    selected = auto_tier(total_gb or 6) if str(tier) in {"auto", "custom"} else int(tier)
    batch = 1 if selected <= 6 else 2 if selected <= 10 else 4 if selected <= 16 else 8
    return RuntimeConfig(vram_tier=str(selected), gpt_dtype="bf16", blocks_to_swap=0,
        use_qwen_emo=False, decoder_adapter="none", vram_reserve_gb=1 if selected <= 16 else 2,
        max_section_batch_size_hint=batch)


def preset_notes(tier):
    cfg = resolve_preset(tier)
    return (f"OmniVoice {cfg.vram_tier} GB preset: BF16 transformer and audio tokenizer remain resident; "
            f"no IndexTTS auxiliary models are loaded. Start with section batch 1; the conservative "
            f"batch hint is {cfg.max_section_batch_size_hint}. Long references and long text use more memory. "
            "The reference prompt is cached between requests. Run the VRAM benchmark to measure your workload.")
