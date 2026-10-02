"""Model-specific values inside a universal preset; legacy presets stay IndexTTS.

Controls fall into three groups. Model-only controls (``INDEX_ONLY``,
``TRAINING_INDEX_ONLY``, ``AUK_ONLY`` and every ``omnivoice.*`` / ``auk.*`` key) exist for one model,
are hidden for the other and keep one value. Shared inputs such as the text,
captions, references and output options keep one value too, so switching the
model never discards what the user typed. Shared controls whose good values
depend on the model (adapters, runtime and training settings, a few
generation limits) are profiled: each model keeps its own value and a switch
restores it.
"""

from copy import deepcopy

from indextts.backends import normalize_model


INDEX_ONLY = frozenset({
    "runtime.decoder_adapter", "runtime.decoder_adapter_strength", "runtime.use_accel",
    "runtime.use_qwen_emo", "runtime.use_deepspeed", "runtime.torch_compile_s2mel",
    "runtime.use_cuda_kernel_bigvgan", "runtime.s2mel_estimator_autocast",
    "runtime.blocks_to_swap", "runtime.swap_ring_size", "runtime.pin_swap_memory",
    "runtime.cfm_cache_length", "runtime.aux_residency.semantic_model", "runtime.aux_residency.qwen_emo",
    "runtime.aux_residency.campplus", "runtime.aux_residency.semantic_codec",
    "runtime.aux_residency.s2mel", "runtime.aux_residency.bigvgan",
    "generation.language", "generation.auto_lora_emotion_reference", "generation.latent_multiplier",
    "generation.segment_budget_scale_non_cjk", "generation.prevent_vram_accumulation",
    "generation.max_consecutive_silence", "generation.auto_retry_incomplete_speech",
    "generation.max_speech_retries", "generation.max_speech_split_depth",
    "generation.apply_pronunciation_dictionary",
    "generation.do_sample", "generation.temperature", "generation.top_p", "generation.top_k",
    "generation.num_beams", "generation.repetition_penalty", "generation.repetition_window",
    "generation.length_penalty", "generation.max_mel_tokens",
    "grid.eval_reference_mode", "grid.num_beams", "grid.temperature", "grid.top_p",
    "grid.top_k", "grid.repetition_penalty", "grid.length_penalty", "grid.max_mel_tokens",
    "grid.diffusion_steps", "grid.inference_cfg_rate", "grid.cfm_temperature", "grid.reuse_spk_cond_for_emo",
})

TRAINING_INDEX_ONLY = frozenset({
    "training.train_spk_proj", "training.train_emo_layers", "training.text_loss_weight",
    "training.mel_loss_weight", "training.label_smoothing", "training.speaker_ref_mode", "training.emo_ref_mode",
    "training.blocks_to_swap", "training.swap_ring_size", "training.pin_swap_memory",
    "training.val_reference_mode", "training.sample_temperature", "training.sample_top_p", "training.sample_top_k",
    "training.sample_repetition_penalty", "training.sample_num_beams", "training.sample_emo_alpha",
    "training.sample_diffusion_steps", "training.sample_inference_cfg_rate", "training.sample_length_penalty",
    "training.sample_max_mel_tokens", "training.probe_enabled", "training.probe_every_epochs",
    "training.probe_prompts", "training.probe_seeds", "training.probe_patience", "training.probe_min_delta",
    "training.probe_wer_tolerance", "training.probe_timeout_s", "training.probe_stop_enabled", "training.probe_device",
    "training.decoder_adapter_enabled", "training.decoder_adapter_rank", "training.decoder_adapter_alpha",
    "training.decoder_adapter_epochs", "training.decoder_adapter_learning_rate", "training.decoder_adapter_timeout_s",
    "training.decoder_adapter_code_source", "training.decoder_adapter_always_gate",
    "training.num_workers",
})

# Shared generation controls with model-specific good values.
GENERATION_PROFILED = frozenset({
    "generation.max_text_tokens_per_segment", "generation.auto_lora_max_tokens",
    "generation.auto_lora_reference", "generation.auto_lora_speaking_rate",
    "generation.speaking_rate", "generation.section_batch_size",
})
# Hardware and the prepared dataset belong to the machine, not to a model.
SHARED_ACROSS_MODELS = frozenset({"runtime.device", "training.dataset_dir", "training.device"})


# The token budget is a memory setting of the GPU tier like the batch size and
# accumulation it replaces, so each model keeps its own value.
OMNIVOICE_PROFILED = frozenset({"training.omni_batch_tokens"})


# AuK's own runtime settings (its Qwen2.5-Omni encoder); one value, shown only for AuK.
AUK_ONLY = frozenset({"runtime.auk_text_encoder_variant", "runtime.auk_text_encoder_residency"})


def is_omnivoice_only(key):
    return (key.startswith("omnivoice.") or key.startswith("training.omni_")) and key not in OMNIVOICE_PROFILED


def is_auk_only(key):
    return key.startswith(("auk.", "auk_edit.", "training.auk_")) or key in AUK_ONLY


def is_model_only(key):
    return key in INDEX_ONLY or key in TRAINING_INDEX_ONLY or is_omnivoice_only(key) or is_auk_only(key)


def migrate_model_values(values):
    """Fill values that earlier OmniVoice presets kept in shared controls.

    The first OmniVoice presets stored the OmniVoice language in the shared
    ``generation.language`` control, which now belongs to IndexTTS alone.
    Missing keys of any other kind take the registry defaults on coercion.
    """
    result = dict(values)
    if "omnivoice.language" not in result:
        profiles = result.get("app.profiles") or {}
        source = result if result.get("app.model") == "omnivoice" else profiles.get("omnivoice") or {}
        language = source.get("generation.language")
        if language:
            result["omnivoice.language"] = str(language).upper()
    return result


def profiled_keys(keys):
    """The shared controls that keep a separate value for each model, in registry order."""
    result = []
    for key in keys:
        if key.startswith("app.") or key in SHARED_ACROSS_MODELS:
            continue
        if is_model_only(key):
            continue
        if key.startswith(("runtime.", "training.", "grid.")) or key in GENERATION_PROFILED:
            result.append(key)
    return result


def capture_profile(values):
    """Record the active model's profiled values; older presets are IndexTTS."""
    result = deepcopy(dict(values))
    model = normalize_model(result.get("app.model"))
    profiles = dict(result.get("app.profiles") or {})
    keys = profiled_keys(result)
    profiles[model] = {key: result[key] for key in keys}
    profiles["_active"] = model
    result["app.model"], result["app.profiles"] = model, profiles
    return result


def model_defaults(registry, model, tier="auto"):
    result = registry.defaults()
    if model == "omnivoice":
        from indextts.runtime.omnivoice_presets import default_training_method, resolve_preset, resolve_training_preset
        cfg = resolve_preset(tier)
        method = default_training_method(cfg.vram_tier)
        result.update({"runtime." + key: value for key, value in cfg.to_dict().items() if key != "aux_residency"})
        result.update({
            "runtime.lora_path": "", "runtime.decoder_adapter": "none",
            "runtime.blocks_to_swap": 0, "runtime.use_qwen_emo": False,
            "generation.max_text_tokens_per_segment": 120,
            "generation.latent_multiplier": 1.72,
            "generation.apply_pronunciation_dictionary": False,
            "generation.auto_lora_emotion_reference": False,
            "generation.auto_lora_speaking_rate": False,
            "generation.auto_lora_max_tokens": False,
            "generation.section_batch_size": 1,
            "training.tts_model": "omnivoice", "training.name": "omnivoice_voice",
            # Round-2 recipe: epochs 0 sizes the run from the training audio (25 epochs
            # for 14 hours, more for less); samples every fifth epoch keep long runs short.
            "training.adapter_type": method, "training.epochs": 0, "training.max_steps": 0,
            "training.sample_every_epochs": 5,
            # The measured adapter: rank 32, alpha 64 (2.7 GB with checkpointing).
            "training.rank": 32, "training.alpha": 64.0,
            "training.blocks_to_swap": 0,
            "training.num_workers": 0,
            "training.train_spk_proj": False, "training.train_emo_layers": False,
            "training.sample_min_free_vram_gb": 2.5, "training.base_variant": "bf16",
            "training.decoder_adapter_enabled": False, "training.decoding_sweep_enabled": False,
            "training.sample_max_text_tokens": 120,
        })
        result.update({"training." + key: value for key, value in resolve_training_preset(cfg.vram_tier, method).items()})
        result["training.vram_tier"] = cfg.vram_tier
    elif model == "auk":
        result.update(auk_defaults(tier))
    result["app.model"] = model
    return result


def auk_defaults(tier="auto"):
    """AuK's GPU-tier runtime and its generation and training starting points."""
    from indextts.runtime.auk_presets import default_training_method, resolve_preset, resolve_training_preset

    cfg = resolve_preset(tier)
    method = default_training_method(cfg.vram_tier)
    values = {"runtime." + key: value for key, value in cfg.to_dict().items() if key != "aux_residency"}
    values.update({
        "runtime.lora_path": "", "runtime.decoder_adapter": "none", "runtime.blocks_to_swap": 0,
        "runtime.use_qwen_emo": False,
        # About 20 seconds of speech per section (AuK's training clips are at most 30 s).
        "generation.max_text_tokens_per_segment": 80,
        "generation.latent_multiplier": 1.72,
        "generation.auto_lora_emotion_reference": False,
        "generation.auto_lora_speaking_rate": False,
        "generation.auto_lora_max_tokens": False,
        "generation.section_batch_size": 1,
        "training.tts_model": "auk", "training.name": "auk_voice",
        "training.adapter_type": method, "training.max_steps": 0,
        "training.rank": 32, "training.alpha": 64.0,
        "training.blocks_to_swap": 0, "training.num_workers": 0,
        "training.train_spk_proj": False, "training.train_emo_layers": False,
        "training.decoder_adapter_enabled": False, "training.decoding_sweep_enabled": False,
        "training.sample_max_text_tokens": 80,
    })
    values.update({"training." + key: value for key, value in resolve_training_preset(cfg.vram_tier, method).items()})
    values["training.vram_tier"] = cfg.vram_tier
    return values


def switch_profile(registry, target, values):
    """Values after switching to ``target``: its profiled values, everything else unchanged."""
    target = normalize_model(target)
    previous = dict(values.get("app.profiles") or {}).get("_active", "indextts")
    captured = capture_profile({**values, "app.model": previous})
    profiles = captured["app.profiles"]
    saved = profiles.get(target) or {}
    restored = dict(captured)
    # A first visit, or a key added after the preset was saved, takes the
    # target model's default for the current GPU tier. Older presets stored
    # every key per model; only profiled keys switch.
    defaults = model_defaults(registry, target, values.get("runtime.vram_tier", "auto"))
    restored.update({key: saved.get(key, defaults[key]) for key in profiled_keys(defaults)})
    restored["app.model"] = target
    profiles["_active"] = target
    restored["app.profiles"] = profiles
    return registry.coerce(restored)
