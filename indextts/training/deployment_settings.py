"""Generation settings the app applies by default for a checkpoint: what a user actually hears.

The Voice Generation tab derives several settings from the selected adapter: the GPU tier's beams and
diffusion steps, Smart sentences packed to the adapter's typical clip length, the token limit and the
sentence/maximum pauses measured on its training clips, its expressive clip as the emotion prompt, and
its calibrated speaking rate. Training-time comparisons that render with other settings measure a
different system than the one deployed; this module gives every benchmark the deployed settings.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from indextts.utils.text_segmentation import default_segment_tokens

from indextts.runtime.vram_presets import generation_preset

from .sampling import SAMPLE_FIXED_INFER_KWARGS

DEPLOYMENT_SEGMENTATION_MODE = "smart"
DEPLOYMENT_BUDGET_SCALE = 0.72
DEFAULT_MAX_TEXT_TOKENS = 60
_FALLBACK_DECODING = {"num_beams": 4, "diffusion_steps": 50, "cfm_temperature": 0.9}


def tier_decoding(tier: str | int | float | None) -> dict[str, Any]:
    """Beams, diffusion steps and CFM temperature of the GPU tier the run deploys on."""

    value = tier
    if value in (None, "", "auto"):
        value = 32
    try:
        preset = generation_preset(value)
        return {"num_beams": int(preset["num_beams"]), "diffusion_steps": int(preset["diffusion_steps"]),
                "cfm_temperature": float(preset.get("cfm_temperature", _FALLBACK_DECODING["cfm_temperature"]))}
    except Exception:
        return dict(_FALLBACK_DECODING)


def adapter_defaults(checkpoint: str | Path | None, *, language: str = "EN",
                     speaking_rate: float | None = None) -> dict[str, Any]:
    """Token target and limit, pauses, expressive clip and speaking rate the tab derives for an adapter.

    Base (an empty checkpoint) keeps the language default token limit, no target, no pauses, no
    expressive prompt and the model's natural pace.
    """

    result: dict[str, Any] = {"max_text_tokens_per_segment": default_segment_tokens(language), "segment_target_tokens": None,
                              "sentence_pause_ms": 0, "max_pause_ms": 0, "emo_audio_prompt": None,
                              "speaking_rate": float(speaking_rate) if speaking_rate else 1.0}
    if not checkpoint:
        return result
    path = str(Path(checkpoint).expanduser())
    from .dataset_profile import (ensure_dataset_profile, expressive_reference_path, load_dataset_profile,
                                  recommended_max_tokens, recommended_pauses, smart_target_tokens)
    try:
        profile = load_dataset_profile(path) or ensure_dataset_profile(path, write=False)
    except Exception:
        profile = None
    if profile:
        try:
            tokens = recommended_max_tokens(profile, language=language, budget_scale=DEPLOYMENT_BUDGET_SCALE)
            if tokens:
                result["max_text_tokens_per_segment"] = int(tokens)
        except Exception:
            pass
        try:
            result["segment_target_tokens"] = smart_target_tokens(profile)
        except Exception:
            pass
        try:
            pauses = recommended_pauses(profile)
            if pauses:
                result["sentence_pause_ms"], result["max_pause_ms"] = int(pauses[0]), int(pauses[1])
        except Exception:
            pass
    try:
        result["emo_audio_prompt"] = expressive_reference_path(path)
    except Exception:
        result["emo_audio_prompt"] = None
    if not speaking_rate:
        try:
            from .speaking_rate import load_speaking_rate
            report = load_speaking_rate(path)
            if report is not None and report.recommended_speaking_rate:
                result["speaking_rate"] = float(report.recommended_speaking_rate)
        except Exception:
            pass
    return result


def deployment_infer_kwargs(config: Any, checkpoint: str | Path | None, *, language: str = "EN",
                            tier: str | int | float | None = None, speaking_rate: float | None = None) -> dict[str, Any]:
    """Inference settings Voice Generation would use for ``checkpoint`` (Base when empty).

    Sampling values come from the run's sample settings (the same defaults as the tab), decoding from the
    GPU tier, and the text budget, pauses, emotion prompt and pace from the adapter's own profile.
    """

    infer = dict(SAMPLE_FIXED_INFER_KWARGS)
    infer.update(top_p=config.sample_top_p, top_k=config.sample_top_k or None, temperature=config.sample_temperature,
                 length_penalty=config.sample_length_penalty, repetition_penalty=config.sample_repetition_penalty,
                 repetition_window=0, max_mel_tokens=config.sample_max_mel_tokens, emo_alpha=config.sample_emo_alpha,
                 inference_cfg_rate=config.sample_inference_cfg_rate)
    infer.update(tier_decoding(tier if tier is not None else getattr(config, "vram_tier", "auto")))
    defaults = adapter_defaults(checkpoint, language=language, speaking_rate=speaking_rate)
    infer.update(max_text_tokens_per_segment=defaults["max_text_tokens_per_segment"],
                 emo_audio_prompt=defaults["emo_audio_prompt"], segment_target_tokens=defaults["segment_target_tokens"],
                 sentence_pause_ms=defaults["sentence_pause_ms"], max_pause_ms=defaults["max_pause_ms"],
                 segmentation_mode=DEPLOYMENT_SEGMENTATION_MODE, segment_budget_scale_non_cjk=DEPLOYMENT_BUDGET_SCALE)
    infer["latent_multiplier"] = round(float(SAMPLE_FIXED_INFER_KWARGS["latent_multiplier"]) / float(defaults["speaking_rate"]), 4)
    return infer


__all__ = ["DEFAULT_MAX_TEXT_TOKENS", "DEPLOYMENT_BUDGET_SCALE", "DEPLOYMENT_SEGMENTATION_MODE",
           "adapter_defaults", "deployment_infer_kwargs", "tier_decoding"]
