"""Voice generation tab and the UI-to-runner request contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import html
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Any, Mapping, Sequence

import gradio as gr

from indextts.lora.io import inspect_lora, scan_lora_files
from indextts.runtime.progress import read_progress_file
from indextts.training.media import SUPPORTED_MEDIA_EXTENSIONS, probe_media
from indextts.utils.subtitle_utils import SUBTITLE_FORMAT_SUMMARY, SUPPORTED_SUBTITLE_EXTENSIONS
from indextts.lora.decoder import decoder_adapter_choices, find_decoder_adapter, recommended_decoder_strength
from indextts.training.dataset_profile import (
    budget_scale_for,
    dataset_dir_for_adapter,
    ensure_dataset_profile,
    expressive_profile_entry,
    expressive_reference_path,
    load_dataset_vocabulary,
    profile_path,
    recommended_max_tokens,
    recommended_pauses,
    save_expressive_reference,
    smart_target_tokens,
    seconds_for_words,
    update_profile_expressive_reference,
    words_for_max_tokens,
)
from indextts.training.decoding_sweep import load_decoding_settings
from indextts.training.speaking_rate import (
    ensure_calibration_fields,
    load_speaking_rate,
    original_calibration,
    save_manual_speaking_rate,
    speaking_rate_method_label,
)
from indextts.utils.pronunciation import (
    DictionaryEntry,
    apply_dictionary,
    builtin_entries,
    check_text,
    cmu_available,
    default_dictionary_path,
    dictionary_rows,
    entries_from_rows,
    load_dictionary,
    merge_entries,
    normalize_entry,
    save_dictionary,
)
from indextts.utils.pause_tags import PauseChunk, TextChunk, describe_pauses, split_text_with_pauses
from indextts.utils.subtitle_utils import (
    build_subtitle_render_units,
    format_srt_timestamp,
    get_subtitle_extension,
    get_subtitle_format_label,
    parse_subtitle_file,
    subtitle_cues_to_text,
)
from indextts.utils.task_output_utils import (
    create_task_output_layout,
    normalize_file_extension,
    write_metadata_file,
)
from indextts.utils.text_segmentation import (
    DEFAULT_SEGMENTATION_MODE,
    SEGMENTATION_MODES,
    SpeechRecoveryConfig,
    default_segment_tokens,
    normalize_segmentation_mode,
    normalize_sentence_whitespace,
    split_text_by_tokens,
)
from webui_generation_runner import current_timestamp, format_elapsed_duration, run_generation_request

from .common import (
    GenerationCanceled,
    LAZY_ENGINE,
    PROCESS_MANAGER,
    ROOT,
    adopt_output_task,
    btn,
    extract_reference_audio,
    is_cancellation,
    open_folder,
    output_task_is_active,
    progress_panel_html,
    read_json,
    resolve_path_value,
    runtime_config_from_values,
    tail_text,
    write_json_atomic,
)
from .presets_store import PresetRegistry


LANGUAGES = ("ZH", "EN", "JA", "AR", "ES")
_ACTIVE_INPROCESS_TASK = ""
EMOTION_MODES = (
    "Same as speaker voice",
    "Emotion reference audio",
    "Emotion vector",
    "Emotion text",
)
EMOTION_NAMES = ("joy", "anger", "sad", "fear", "disgust", "depression", "surprise", "calm")
EMOTION_LABELS = ("Joy", "Anger", "Sadness", "Fear", "Disgust", "Depression", "Surprise", "Calm")
EMOTION_BIAS_DEFAULTS = (0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625)
REFERENCE_AUDIO_DIR = ROOT / "reference_audios"
REFERENCE_AUDIO_EXTENSIONS = frozenset(
    {
        ".aac",
        ".ac3",
        ".aif",
        ".aiff",
        ".alac",
        ".amr",
        ".ape",
        ".au",
        ".caf",
        ".dts",
        ".flac",
        ".m4a",
        ".mka",
        ".mp2",
        ".mp3",
        ".oga",
        ".ogg",
        ".opus",
        ".ra",
        ".tta",
        ".voc",
        ".wav",
        ".weba",
        ".wma",
        ".wv",
    }
)
_AUTO_REFERENCE_SOURCES = frozenset({"library_auto", "lora_auto"})


@dataclass(frozen=True)
class ReferenceSelection:
    prompt: str | None
    library_value: str | None
    source: str
    message: str
    choices: list[tuple[str, str]]


@dataclass(frozen=True)
class PreparedReference:
    prompt: str
    media: str
    video: str | None
    library_value: str | None
    source: str
    message: str
    choices: list[tuple[str, str]]


GENERATION_DEFAULTS: dict[str, Any] = {
    "generation.language": "EN",
    "generation.max_text_tokens_per_segment": 60,
    "generation.auto_lora_max_tokens": True,
    "generation.segmentation_mode": DEFAULT_SEGMENTATION_MODE,
    "generation.sentence_pause_ms": 0,
    "generation.auto_lora_pauses": True,
    "generation.use_caption_timing": False,
    "generation.auto_lora_reference": True,
    "generation.auto_lora_speaking_rate": True,
    "generation.auto_lora_emotion_reference": True,
    "generation.emotion_mode": EMOTION_MODES[0],
    "generation.emotion_weight": 0.65,
    "generation.emotion_random": False,
    "generation.emotion_text": "",
    "generation.apply_emotion_bias": True,
    "generation.max_emotion_sum": 0.8,
    "generation.do_sample": True,
    "generation.temperature": 0.8,
    "generation.top_p": 0.8,
    "generation.top_k": 30,
    "generation.num_beams": 3,
    "generation.repetition_penalty": 10.0,
    "generation.repetition_window": 0,
    "generation.length_penalty": 0.0,
    "generation.max_mel_tokens": 1500,
    "generation.seed": -1,
    "generation.num_candidates": 1,
    "generation.diffusion_steps": 25,
    "generation.inference_cfg_rate": 0.7,
    "generation.cfm_temperature": 1.0,
    "generation.cfm_cache_length": 8192,
    "generation.segment_budget_scale_non_cjk": 0.72,
    "generation.interval_silence": 200,
    "generation.max_consecutive_silence": 0,
    "generation.max_pause_ms": 0,
    "generation.latent_multiplier": 1.72,
    "generation.speaking_rate": 1.0,
    "generation.target_duration_s": None,
    "generation.target_duration_mode": "off",
    "generation.enable_pause_tags": True,
    "generation.text_normalization": True,
    "generation.apply_pronunciation_dictionary": True,
    "generation.auto_retry_incomplete_speech": SpeechRecoveryConfig.enabled,
    "generation.max_speech_retries": SpeechRecoveryConfig.max_attempts,
    "generation.max_speech_split_depth": SpeechRecoveryConfig.max_split_depth,
    "generation.max_speaker_audio_length": 15.0,
    "generation.max_emotion_audio_length": 15.0,
    "generation.semantic_layer": 17,
    "generation.reuse_spk_cond_for_emo": False,
    "generation.save_used_audio": False,
    "generation.output_filename": "",
    "generation.save_as_mp3": False,
    "generation.mp3_bitrate": "256k",
    "generation.audio_tuning_preset": "bypass",
    "generation.tuning_low_cut_hz": None,
    "generation.tuning_high_cut_hz": None,
    "generation.tuning_gain_db": None,
    "generation.tuning_loudnorm_i": None,
    "generation.tuning_deess": None,
    "generation.trim_silence_ms_threshold": 0,
    "generation.use_subprocess": False,
    "generation.section_batch_size": 1,
    "generation.low_memory_mode": False,
    "generation.prevent_vram_accumulation": False,
    "generation.verbose": False,
}
for _name in EMOTION_NAMES:
    GENERATION_DEFAULTS[f"generation.emotion_{_name}"] = 0.0
for _name, _default in zip(EMOTION_NAMES, EMOTION_BIAS_DEFAULTS):
    GENERATION_DEFAULTS[f"generation.emotion_bias_{_name}"] = _default


INFER_KWARG_KEYS = frozenset(
    {
        "do_sample",
        "top_p",
        "top_k",
        "temperature",
        "length_penalty",
        "num_beams",
        "repetition_penalty",
        "repetition_window",
        "max_mel_tokens",
        "emo_audio_prompt",
        "emo_alpha",
        "emo_vector",
        "use_emo_text",
        "emo_text",
        "use_random",
        "verbose",
        "max_text_tokens_per_segment",
        "interval_silence",
        "diffusion_steps",
        "inference_cfg_rate",
        "max_speaker_audio_length",
        "max_emotion_audio_length",
        "section_batch_size",
        "max_emotion_sum",
        "latent_multiplier",
        "max_consecutive_silence",
        "semantic_layer",
        "cfm_cache_length",
        "reset_beam_cache_per_segment",
        "text_normalization",
        "auto_retry_incomplete_speech",
        "max_speech_retries",
        "max_speech_split_depth",
    }
)

RUNNER_REQUEST_KEYS = frozenset(
    {
        "prompt",
        "text",
        "subtitle_mode",
        "subtitle_file",
        "language",
        "save_used_audio",
        "save_as_mp3",
        "mp3_bitrate",
        "image_path",
        "infer_kwargs",
        "runtime",
        "low_memory_mode",
        "task_layout",
        "metadata_path",
        "max_text_tokens",
        "progress_file",
        "lora_path",
        "lora_strength",
        "lora_merge_into_base",
        "decoder_adapter",
        "decoder_adapter_strength",
        "num_candidates",
        "audio_tuning_preset",
        "audio_tuning_overrides",
        "segment_budget_scale_non_cjk",
        "cfm_temperature",
        "seed",
        "reuse_spk_cond_for_emo",
        "enable_pause_tags",
        "trim_silence_ms_threshold",
        "max_pause_ms",
        "segmentation_mode",
        "segment_target_tokens",
        "sentence_pause_ms",
        "target_duration_s",
        "target_duration_mode",
    }
)


def _value(values: Mapping[str, Any], key: str) -> Any:
    return values.get(key, GENERATION_DEFAULTS.get(key))


def _normalize_emotion_vector(values: Mapping[str, Any]) -> list[float] | None:
    if _value(values, "generation.emotion_mode") != EMOTION_MODES[2]:
        return None
    vector = [float(_value(values, f"generation.emotion_{name}") or 0.0) for name in EMOTION_NAMES]
    if bool(_value(values, "generation.apply_emotion_bias")):
        biases = [float(_value(values, f"generation.emotion_bias_{name}")) for name in EMOTION_NAMES]
        vector = [item * bias for item, bias in zip(vector, biases)]
    limit = max(0.0, float(_value(values, "generation.max_emotion_sum") or 0.0))
    total = sum(vector)
    if total > limit and total > 0:
        vector = [item * limit / total for item in vector]
    return vector


def build_generation_request(
    values: Mapping[str, Any] | None = None,
    *,
    prompt: str = "",
    text: str = "",
    subtitle_file: str | None = None,
    image_path: str | None = None,
    emotion_audio: str | None = None,
    runtime: Mapping[str, Any] | None = None,
    task_layout: Mapping[str, Any] | None = None,
    metadata_path: str = "",
    progress_file: str | None = None,
    model_dir: str = "models",
) -> dict[str, Any]:
    """Build the exact runner request contract entirely from UI values.

    This function is intentionally side-effect free so request-coverage tests can
    call it with registry defaults.
    """

    merged = dict(GENERATION_DEFAULTS)
    if values:
        merged.update(values)
    mode = str(_value(merged, "generation.emotion_mode"))
    mode_index = EMOTION_MODES.index(mode) if mode in EMOTION_MODES else 0
    emotion_vector = _normalize_emotion_vector(merged)
    if mode_index != 1:
        emotion_audio = None
    if mode_index == 0 and bool(_value(merged, "generation.auto_lora_emotion_reference")):
        # The adapter's expressive training clip drives the delivery while the speaker prompt keeps the identity.
        expressive = expressive_reference_path(str(merged.get("runtime.lora_path") or ""))
        if expressive:
            emotion_audio = expressive
    emotion_text = str(_value(merged, "generation.emotion_text") or "") or None
    top_k_value = int(_value(merged, "generation.top_k") or 0)
    target_duration = _value(merged, "generation.target_duration_s")
    if target_duration in (None, "", 0, 0.0):
        target_duration = None
    else:
        target_duration = float(target_duration)

    infer_kwargs = {
        "do_sample": bool(_value(merged, "generation.do_sample")),
        "top_p": float(_value(merged, "generation.top_p")),
        "top_k": top_k_value if top_k_value > 0 else None,
        "temperature": float(_value(merged, "generation.temperature")),
        "length_penalty": float(_value(merged, "generation.length_penalty")),
        "num_beams": int(_value(merged, "generation.num_beams")),
        "repetition_penalty": float(_value(merged, "generation.repetition_penalty")),
        "repetition_window": int(_value(merged, "generation.repetition_window") or 0),
        "max_mel_tokens": int(_value(merged, "generation.max_mel_tokens")),
        "emo_audio_prompt": emotion_audio,
        "emo_alpha": float(_value(merged, "generation.emotion_weight")),
        "emo_vector": emotion_vector,
        "use_emo_text": mode_index == 3,
        "emo_text": emotion_text,
        "use_random": bool(_value(merged, "generation.emotion_random")),
        "verbose": bool(_value(merged, "generation.verbose")),
        "max_text_tokens_per_segment": int(_value(merged, "generation.max_text_tokens_per_segment")),
        "interval_silence": int(_value(merged, "generation.interval_silence")),
        "diffusion_steps": int(_value(merged, "generation.diffusion_steps")),
        "inference_cfg_rate": float(_value(merged, "generation.inference_cfg_rate")),
        "max_speaker_audio_length": float(_value(merged, "generation.max_speaker_audio_length")),
        "max_emotion_audio_length": float(_value(merged, "generation.max_emotion_audio_length")),
        "section_batch_size": int(_value(merged, "generation.section_batch_size")),
        "max_emotion_sum": float(_value(merged, "generation.max_emotion_sum")),
        "latent_multiplier": round(
            float(_value(merged, "generation.latent_multiplier"))
            / min(
                1.5,
                max(0.5, float(_value(merged, "generation.speaking_rate"))),
            ),
            4,
        ),
        "max_consecutive_silence": int(_value(merged, "generation.max_consecutive_silence")),
        "semantic_layer": int(_value(merged, "generation.semantic_layer")),
        "cfm_cache_length": int(_value(merged, "generation.cfm_cache_length")),
        "reset_beam_cache_per_segment": bool(_value(merged, "generation.prevent_vram_accumulation")),
        "text_normalization": bool(_value(merged, "generation.text_normalization")),
        "auto_retry_incomplete_speech": bool(_value(merged, "generation.auto_retry_incomplete_speech")),
        "max_speech_retries": int(_value(merged, "generation.max_speech_retries")),
        "max_speech_split_depth": int(_value(merged, "generation.max_speech_split_depth")),
    }
    runtime_value = dict(runtime or runtime_config_from_values(merged, model_dir=model_dir))
    lora_path = str(merged.get("runtime.lora_path", runtime_value.get("lora_path", "")) or "")
    raw_lora_strength = merged.get(
        "runtime.lora_strength", runtime_value.get("lora_strength", 1.0)
    )
    lora_strength = float(
        1.0 if raw_lora_strength in (None, "") else raw_lora_strength
    )
    lora_merge_into_base = bool(
        merged.get(
            "runtime.lora_merge_into_base",
            runtime_value.get("lora_merge_into_base", False),
        )
    )
    decoder_adapter = str(merged.get("runtime.decoder_adapter", runtime_value.get("decoder_adapter", "auto")) or "auto")
    runtime_value["lora_path"] = lora_path
    runtime_value["lora_strength"] = lora_strength
    runtime_value["lora_merge_into_base"] = lora_merge_into_base
    runtime_value["decoder_adapter"] = decoder_adapter
    raw_decoder_strength = merged.get("runtime.decoder_adapter_strength", runtime_value.get("decoder_adapter_strength", 1.0))
    decoder_adapter_strength = float(1.0 if raw_decoder_strength in (None, "") else raw_decoder_strength)
    runtime_value["decoder_adapter_strength"] = decoder_adapter_strength

    overrides = {}
    for ui_key, backend_key in (
        ("generation.tuning_low_cut_hz", "low_cut_hz"),
        ("generation.tuning_high_cut_hz", "high_cut_hz"),
        ("generation.tuning_gain_db", "gain_db"),
        ("generation.tuning_loudnorm_i", "loudnorm_i"),
        ("generation.tuning_deess", "deess"),
    ):
        item = _value(merged, ui_key)
        if item not in (None, ""):
            overrides[backend_key] = float(item)

    segmentation_mode = normalize_segmentation_mode(_value(merged, "generation.segmentation_mode"))
    segment_target_tokens = (
        smart_segment_target(str(merged.get("runtime.lora_path") or "")) if segmentation_mode == "smart" else None
    )
    request = {
        "prompt": str(prompt or ""),
        "text": str(text or ""),
        "subtitle_mode": bool(_value(merged, "generation.use_caption_timing")),
        "subtitle_file": subtitle_file,
        "language": str(_value(merged, "generation.language") or "EN").upper(),
        "save_used_audio": bool(_value(merged, "generation.save_used_audio")),
        "save_as_mp3": bool(_value(merged, "generation.save_as_mp3")),
        "mp3_bitrate": str(_value(merged, "generation.mp3_bitrate")),
        "image_path": image_path,
        "infer_kwargs": infer_kwargs,
        "runtime": runtime_value,
        "low_memory_mode": bool(_value(merged, "generation.low_memory_mode")),
        "task_layout": dict(task_layout or {}),
        "metadata_path": str(metadata_path or ""),
        "max_text_tokens": int(_value(merged, "generation.max_text_tokens_per_segment")),
        "progress_file": str(progress_file) if progress_file else None,
        "lora_path": lora_path,
        "lora_strength": lora_strength,
        "lora_merge_into_base": lora_merge_into_base,
        "decoder_adapter": decoder_adapter,
        "decoder_adapter_strength": decoder_adapter_strength,
        "num_candidates": int(_value(merged, "generation.num_candidates")),
        "audio_tuning_preset": str(_value(merged, "generation.audio_tuning_preset") or "bypass"),
        "audio_tuning_overrides": overrides,
        "segment_budget_scale_non_cjk": float(_value(merged, "generation.segment_budget_scale_non_cjk")),
        "cfm_temperature": float(_value(merged, "generation.cfm_temperature")),
        "seed": int(_value(merged, "generation.seed")),
        "reuse_spk_cond_for_emo": bool(_value(merged, "generation.reuse_spk_cond_for_emo")),
        "enable_pause_tags": bool(_value(merged, "generation.enable_pause_tags")),
        "trim_silence_ms_threshold": int(_value(merged, "generation.trim_silence_ms_threshold")),
        "max_pause_ms": int(_value(merged, "generation.max_pause_ms") or 0),
        "segmentation_mode": segmentation_mode,
        "segment_target_tokens": segment_target_tokens,
        "sentence_pause_ms": int(_value(merged, "generation.sentence_pause_ms") or 0),
        "target_duration_s": target_duration,
        "target_duration_mode": str(_value(merged, "generation.target_duration_mode") or "off"),
    }
    assert set(request) == RUNNER_REQUEST_KEYS
    assert set(infer_kwargs) == INFER_KWARG_KEYS
    return request


def build_default_generation_request(
    registry: PresetRegistry | None = None,
    *,
    model_dir: str = "models",
) -> dict[str, Any]:
    values = registry.defaults() if registry is not None else dict(GENERATION_DEFAULTS)
    return build_generation_request(values, model_dir=model_dir)


request_from_registry_defaults = build_default_generation_request


def validate_request_coverage(request: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    missing = set(RUNNER_REQUEST_KEYS) - set(request)
    unknown = set(request) - set(RUNNER_REQUEST_KEYS)
    infer = request.get("infer_kwargs", {})
    missing.update(f"infer_kwargs.{key}" for key in INFER_KWARG_KEYS - set(infer))
    unknown.update(f"infer_kwargs.{key}" for key in set(infer) - INFER_KWARG_KEYS)
    return missing, unknown


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def prepare_generation_request(
    values: Mapping[str, Any],
    *,
    prompt: str,
    text: str,
    subtitle_file: str | None,
    image_path: str | None,
    emotion_audio: str | None,
    model_dir: str,
    output_root: str | os.PathLike[str] = "outputs",
) -> dict[str, Any]:
    prompt_path = resolve_path_value(prompt)
    if not prompt_path or not Path(prompt_path).is_file():
        raise ValueError("Reference Voice audio is required before generation")
    subtitle_path = resolve_path_value(subtitle_file)
    image_source = resolve_path_value(image_path)
    emotion_path = resolve_path_value(emotion_audio)
    subtitle_mode = bool(_value(values, "generation.use_caption_timing"))
    if subtitle_mode and not subtitle_path:
        raise ValueError("Caption cue timing is enabled, but no caption file is selected")
    if not str(text or "").strip() and not subtitle_path:
        raise ValueError("Enter text or load a caption file")
    if image_source and not Path(image_source).is_file():
        raise ValueError(f"Image file not found: {image_source}")
    if bool(_value(values, "generation.apply_pronunciation_dictionary")):
        text = apply_pronunciation_dictionary(text, str(values.get("runtime.lora_path") or ""))
    # Validate captions before allocating an output task.
    cues = parse_subtitle_file(subtitle_path) if subtitle_mode else []
    units = build_subtitle_render_units(cues) if cues else []

    extension = Path(image_source).suffix if image_source else None
    layout = create_task_output_layout(
        output_root=str(Path(output_root).expanduser()),
        filename=str(_value(values, "generation.output_filename") or ""),
        subtitle_mode=subtitle_mode,
        subtitle_extension=get_subtitle_extension(subtitle_path) if subtitle_path else None,
        image_extension=normalize_file_extension(extension) if extension else None,
    )
    task_folder = Path(str(layout["task_folder"])).resolve()
    if image_source and layout.get("source_image_copy_path"):
        shutil.copy2(image_source, layout["source_image_copy_path"])
        image_source = str(layout["source_image_copy_path"])
    if subtitle_mode and subtitle_path and layout.get("subtitle_copy_path"):
        shutil.copy2(subtitle_path, layout["subtitle_copy_path"])

    progress_file = task_folder / "progress.json"
    metadata_path = str(layout["metadata_path"])
    request = build_generation_request(
        values,
        prompt=prompt_path,
        text=text,
        subtitle_file=subtitle_path,
        image_path=image_source,
        emotion_audio=emotion_path,
        task_layout=layout,
        metadata_path=metadata_path,
        progress_file=str(progress_file),
        model_dir=model_dir,
    )

    started = _now()
    metadata = {
        "status": "in_progress",
        "created_at": started,
        "updated_at": started,
        "task": {
            "id": layout["task_id"],
            "folder": str(task_folder),
            "mode": "subtitle" if subtitle_mode else "text",
            "requested_output_filename": str(_value(values, "generation.output_filename") or ""),
            "resolved_output_basename": layout["final_basename"],
        },
        "inputs": {
            "text": text,
            "language": request["language"],
            "speaker_reference_audio": str(Path(prompt_path).resolve()),
            "emotion_reference_audio": str(Path(emotion_path).resolve()) if emotion_path else None,
            "subtitle_file": str(Path(subtitle_path).resolve()) if subtitle_path else None,
            "source_image": str(Path(image_path).resolve()) if image_path else None,
        },
        "settings": {
            "execution_mode": "subprocess" if _value(values, "generation.use_subprocess") else "in_process",
            "resolved_generation_kwargs": request["infer_kwargs"],
            "runtime": request["runtime"],
            "request_values": {key: value for key, value in values.items() if key.startswith(("generation.", "runtime."))},
        },
        "outputs": {
            "final_audio_path": None,
            "final_video_path": None,
            "final_wav_path": str(Path(str(layout["final_wav_path"])).resolve()),
            "final_mp3_path": str(Path(str(layout["final_mp3_path"])).resolve()),
            "final_mp4_path": str(Path(str(layout["final_mp4_path"])).resolve()),
            "metadata_path": str(Path(metadata_path).resolve()),
            "segments_dir": str(Path(str(layout["segments_dir"])).resolve()) if layout.get("segments_dir") else None,
            "speaker_reference_copy_path": None,
            "source_image_copy_path": str(Path(image_source).resolve()) if image_source else None,
            "subtitle_copy_path": str(Path(str(layout["subtitle_copy_path"])).resolve()) if layout.get("subtitle_copy_path") else None,
        },
        "processing": {
            "started_at": started,
            "ended_at": None,
            "elapsed_ms": None,
            "elapsed_seconds": None,
            "elapsed_human": None,
        },
        "subtitle": None,
        "error": None,
    }
    if subtitle_mode:
        metadata["subtitle"] = {
            "format": get_subtitle_format_label(subtitle_path),
            "cue_count": len(cues),
            "render_unit_count": len(units),
            "timeline_end_ms": cues[-1].end_ms if cues else 0,
            "cues": [
                {
                    "index": cue.index,
                    "start_ms": cue.start_ms,
                    "end_ms": cue.end_ms,
                    "duration_ms": cue.duration_ms,
                    "text": cue.text,
                    "segment_file": None,
                    "generated_duration_ms": None,
                }
                for cue in cues
            ],
            "render_units": [
                {
                    "index": unit.index,
                    "start_ms": unit.start_ms,
                    "end_ms": unit.end_ms,
                    "duration_ms": unit.duration_ms,
                    "text": unit.text,
                    "source_cue_indices": list(unit.cue_indices),
                    "segment_file": None,
                }
                for unit in units
            ],
            "timing_issues": [],
        }
    write_metadata_file(metadata_path, metadata)
    write_json_atomic(task_folder / "request.json", request)
    write_json_atomic(progress_file, {
        "stage": "initializing", "desc": "Loading model / preparing generation",
        "completed": 0, "total": 0, "fraction": 0.0, "updated_at": time.time(),
    })
    return request


@lru_cache(maxsize=1)
def _preview_tokenizer(model_dir: str):
    from indextts.utils.tokenizer import get_tokenizer

    return get_tokenizer(multilingual=True, model_dir=model_dir)


@lru_cache(maxsize=8)
def _preview_capacity(model_dir: str) -> int:
    try:
        from omegaconf import OmegaConf

        return int(OmegaConf.load(str(Path(model_dir) / "config.yaml")).gpt.max_text_tokens)
    except Exception:
        return 602


@lru_cache(maxsize=8)
def model_version(model_dir: str) -> str:
    """The ``version`` field of the model folder's config.yaml ("2.5" for IndexTTS 2.5), or ""."""

    try:
        from omegaconf import OmegaConf

        return str(OmegaConf.load(str(Path(model_dir) / "config.yaml")).get("version", "") or "")
    except Exception:
        return ""


def codec_has_silence_runs(model_dir: str) -> bool:
    """Whether **Max consecutive silence tokens** can do anything: the 2.5 codec never repeats a code."""

    version = model_version(str(Path(model_dir).resolve()))
    return not version.startswith("2.5")


def smart_segment_target(lora_path: str | None) -> int | None:
    """Text tokens the smart splitter aims for with the selected voice (None for the base model)."""

    if not lora_path:
        return None
    try:
        return smart_target_tokens(adapter_dataset_profile(str(lora_path)))
    except Exception:
        return None


SEGMENTATION_CHOICES: tuple[tuple[str, str], ...] = (
    ("Smart sentences", "smart"),
    ("Every sentence", "sentence"),
    ("Token budget", "budget"),
)
_SEGMENTATION_LABELS = {value: label for label, value in SEGMENTATION_CHOICES}


def preview_segments(
    text: str,
    language: str,
    max_tokens: int,
    caption_timing: bool = False,
    subtitle_file: str | None = None,
    enable_pause_tags: bool = True,
    segment_scale: float = 0.72,
    model_dir: str = "models",
    segmentation_mode: str = "budget",
    target_tokens: int | None = None,
    words_per_second: float = 0.0,
) -> tuple[list[list[Any]], str]:
    subtitle_path = resolve_path_value(subtitle_file)
    if caption_timing and subtitle_path:
        try:
            cues = parse_subtitle_file(subtitle_path)
            rows = [
                [row_index, "Caption cue", cue.text, f"{format_srt_timestamp(cue.start_ms)} -> {format_srt_timestamp(cue.end_ms)}"]
                for row_index, cue in enumerate(cues, start=1)
            ]
            return rows, f"{len(rows)} caption cue(s), {len(build_subtitle_render_units(cues))} timing unit(s)"
        except Exception as exc:
            return [[0, "Caption error", str(exc), ""]], f"Caption error: {exc}"
    if not str(text or "").strip():
        return [], "0 sections"
    try:
        tokenizer = _preview_tokenizer(str(Path(model_dir).resolve()))
        raw_token_len = lambda value: len(tokenizer.encode(value, allowed_special="all"))
    except Exception:
        raw_token_len = lambda value: max(1, len(str(value).split()) * 2)
    prefix = f"<|{str(language or 'EN').lower()}|> "
    mode = normalize_segmentation_mode(segmentation_mode) if str(segmentation_mode or "budget") != "budget" else "budget"

    def token_len(value: str) -> int:
        return raw_token_len(normalize_sentence_whitespace(value) if mode != "budget" else value)

    rows: list[list[Any]] = []
    section_index = 0
    row_index = 0
    chunks = split_text_with_pauses(text) if enable_pause_tags else [TextChunk(str(text))]
    for chunk in chunks:
        if isinstance(chunk, PauseChunk):
            row_index += 1
            rows.append([row_index, "Pause", f"{chunk.duration_ms} ms", "Inserted silence (kept exactly)"])
            continue
        for segment in split_text_by_tokens(
            chunk.text,
            int(max_tokens),
            capacity=_preview_capacity(str(Path(model_dir).resolve())),
            token_len=token_len,
            lang_prefix=prefix,
            segment_budget_scale_non_cjk=float(segment_scale),
            mode=mode,
            target_tokens=target_tokens,
        ):
            if not segment.strip():
                continue
            section_index += 1
            row_index += 1
            word_count = len(segment.split())
            details = f"{token_len(prefix + segment)} tokens · {word_count} words"
            if words_per_second and words_per_second > 0.0:
                details += f" · about {word_count / float(words_per_second):.1f} s"
            rows.append([row_index, "Text segment", segment, details])
    pause_note = describe_pauses(text) if enable_pause_tags else "Pause tags disabled"
    mode_note = _SEGMENTATION_LABELS.get(mode, mode)
    if mode == "smart" and target_tokens:
        mode_note += f", aiming at {int(target_tokens)} tokens per section"
    return rows, f"{section_index} speech section(s) | {mode_note} | {pause_note}"


def reference_audio_choices(
    root: str | os.PathLike[str] = REFERENCE_AUDIO_DIR,
) -> list[tuple[str, str]]:
    """Create and scan the user-managed reference audio library."""

    root_path = Path(root).expanduser().resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        (
            path.resolve()
            for path in root_path.rglob("*")
            if path.is_file() and path.suffix.lower() in REFERENCE_AUDIO_EXTENSIONS
        ),
        key=lambda path: path.relative_to(root_path).as_posix().casefold(),
    )
    return [
        (path.relative_to(root_path).as_posix(), str(path))
        for path in paths
    ]


def _latest_reference_choice(choices: Sequence[tuple[str, str]]) -> str | None:
    candidates: list[tuple[int, str, str]] = []
    for label, value in choices:
        try:
            modified = Path(value).stat().st_mtime_ns
        except OSError:
            continue
        candidates.append((modified, label.casefold(), value))
    return max(candidates, default=(0, "", None))[2]


def latest_reference_audio(
    root: str | os.PathLike[str] = REFERENCE_AUDIO_DIR,
) -> str | None:
    """Return the most recently modified audio in the reference library."""

    return _latest_reference_choice(reference_audio_choices(root))


def _path_key(value: Any) -> str:
    path = str(value) if isinstance(value, os.PathLike) else resolve_path_value(value)
    if not path:
        return ""
    return os.path.normcase(str(Path(path).expanduser().resolve()))


def _existing_path(value: Any) -> str | None:
    path = str(value) if isinstance(value, os.PathLike) else resolve_path_value(value)
    if not path:
        return None
    resolved = Path(path).expanduser().resolve()
    return str(resolved) if resolved.is_file() else None


def _browser_safe_media_path(
    source: str | os.PathLike[str],
    *,
    allowed_roots: Sequence[str | os.PathLike[str]] | None = None,
    cache_root: str | os.PathLike[str] | None = None,
) -> str:
    """Return a media path that Gradio is permitted to serve to the browser.

    Uploaded files already live in Gradio's temporary directory, while paths
    entered by the user may be anywhere on disk. External media is staged in
    the app-owned UI state directory. A hard link avoids copying large videos
    when the source and app are on the same volume; copying is the portable
    fallback.
    """

    resolved = Path(source).expanduser().resolve()
    roots = allowed_roots
    if roots is None:
        roots = (
            Path(tempfile.gettempdir()),
            ROOT / "outputs",
            ROOT / "datasets",
            ROOT / "loras",
            ROOT / "reference_audios",
            ROOT / ".ui_state",
        )
    temp_root = Path(tempfile.gettempdir()).resolve()
    for root in roots:
        allowed_root = Path(root).expanduser().resolve()
        try:
            resolved.relative_to(allowed_root)
            if allowed_root != temp_root:
                gr.set_static_paths(resolved)
            return str(resolved)
        except ValueError:
            continue

    stat = resolved.stat()
    identity = f"{os.path.normcase(str(resolved))}\0{stat.st_size}\0{stat.st_mtime_ns}"
    digest = hashlib.sha256(identity.encode("utf-8", errors="surrogatepass")).hexdigest()[:20]
    destination_root = Path(cache_root or (ROOT / ".ui_state" / "reference_media"))
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / f"{digest}_{resolved.name}"
    if destination.is_file() and destination.stat().st_size == stat.st_size:
        gr.set_static_paths(destination)
        return str(destination.resolve())

    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.unlink(missing_ok=True)
    try:
        try:
            os.link(resolved, temporary)
        except OSError:
            shutil.copy2(resolved, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    # Large videos must be served in place. Otherwise Gradio copies the staged
    # file into its component cache once per output, defeating the hard link.
    gr.set_static_paths(destination)
    return str(destination.resolve())


def _browser_safe_video_path(source: str | os.PathLike[str]) -> str:
    """Stage a directly-served, browser-playable preview for a video source."""

    resolved = Path(source).expanduser().resolve()
    from gradio.processing_utils import video_is_playable

    if video_is_playable(str(resolved)):
        return _browser_safe_media_path(resolved)

    stat = resolved.stat()
    identity = hashlib.sha256(
        f"preview-v2\0{stat.st_size}\0{stat.st_mtime_ns}".encode("ascii")
    )
    with resolved.open("rb") as source_file:
        identity.update(source_file.read(65536))
        if stat.st_size > 65536:
            source_file.seek(max(0, stat.st_size - 65536))
            identity.update(source_file.read(65536))
    digest = identity.hexdigest()[:20]
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(resolved),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    codec = probe.stdout.strip().lower() if probe.returncode == 0 else ""
    if codec in {"vp8", "vp9", "av1"}:
        suffix = ".webm"
        video_args = ["-c:v", "copy"]
    elif codec == "h264":
        suffix = ".mp4"
        video_args = ["-c:v", "copy", "-movflags", "+faststart"]
    elif codec == "theora":
        suffix = ".ogg"
        video_args = ["-c:v", "copy"]
    else:
        suffix = ".mp4"
        video_args = [
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "28",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]

    destination_root = ROOT / ".ui_state" / "reference_video_previews"
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / f"{digest}_preview{suffix}"
    if destination.is_file() and destination.stat().st_size > 0:
        gr.set_static_paths(destination)
        return str(destination.resolve())

    temporary = destination.with_name(
        f".{destination.stem}.{os.getpid()}.{threading.get_ident()}.tmp{suffix}"
    )
    temporary.unlink(missing_ok=True)
    try:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(resolved),
                "-map",
                "0:v:0",
                "-an",
                *video_args,
                str(temporary),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
            detail = (completed.stderr or completed.stdout or "FFmpeg produced no preview").strip()
            raise OSError(f"video preview conversion failed: {detail[-800:]}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    gr.set_static_paths(destination)
    return str(destination.resolve())


def _same_reference_file(current: str | None, expected: str | None) -> bool:
    """Match originals with Gradio's same-named cached file copies."""

    if not current or not expected:
        return False
    return (
        _path_key(current) == _path_key(expected)
        or Path(current).name.casefold() == Path(expected).name.casefold()
    )


def load_reference_media(
    media_path: Any,
    time_ranges: str = "",
    *,
    require_ranges: bool = False,
) -> tuple[str | None, str | None, str]:
    """Validate any FFmpeg-readable media, extract its audio, and flag video preview."""

    source = _existing_path(media_path)
    if not source:
        return None, None, "Choose an existing audio or video file."
    try:
        media = probe_media(source)
    except Exception as exc:
        return None, None, f"Could not read {Path(source).name}: {exc}"
    if not media.has_audio:
        return None, source if media.has_video else None, (
            f"{Path(source).name} has no audio stream to use as a Reference Voice."
        )
    audio, message = extract_reference_audio(
        source,
        time_ranges,
        require_ranges=require_ranges,
    )
    video = None
    preview_warning = ""
    if media.has_video:
        try:
            video = _browser_safe_video_path(source)
        except OSError as exc:
            preview_warning = f" Video preview could not be staged: {exc}"
    if audio:
        media_type = "video" if media.has_video else "audio"
        message = (
            f"{message[:-1]} from {media_type}; its audio is shown below and ready to use."
            f"{preview_warning}"
        )
    return audio, video, message


def _resolve_lora_reference_path(
    adapter_path: str | os.PathLike[str],
    configured_reference: str | os.PathLike[str] | None,
) -> str | None:
    """Resolve adapter metadata and the conventional run-level reference copy."""

    source = Path(adapter_path).expanduser().resolve()
    adapter_dir = source.parent
    run_dir = adapter_dir.parent if adapter_dir.name.casefold() == "best" else adapter_dir
    candidates: list[Path] = []
    if configured_reference:
        configured = Path(configured_reference).expanduser()
        candidates.append(configured)
        if not configured.is_absolute():
            candidates.extend((ROOT / configured, adapter_dir / configured))
        candidates.extend((adapter_dir / configured.name, run_dir / configured.name))
    candidates.extend(
        (
            source.with_name(f"{source.stem}_reference.wav"),
            run_dir / f"{run_dir.name}_reference.wav",
        )
    )
    candidates.extend(
        sorted(
            (
                path
                for path in run_dir.glob("*_reference.*")
                if path.suffix.lower() in REFERENCE_AUDIO_EXTENSIONS
                and "_expressive_reference" not in path.name.lower()  # the emotion clip is not the speaker reference
            ),
            key=lambda path: path.name.casefold(),
        )
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and resolved.suffix.lower() in REFERENCE_AUDIO_EXTENSIONS:
            return str(resolved)
    return None


def _recommended_lora_reference(path: str | None) -> str | None:
    if not path:
        return None
    try:
        info = inspect_lora(path)
    except Exception:
        return None
    return _resolve_lora_reference_path(path, info.get("recommended_reference"))


def resolve_reference_selection(
    current_reference: Any,
    selected_library: Any,
    reference_source: str | None,
    lora_path: str | None,
    auto_lora_reference: bool,
    *,
    reference_root: str | os.PathLike[str] = REFERENCE_AUDIO_DIR,
    lora_reference: str | None = None,
) -> ReferenceSelection:
    """Choose manual, LoRA, then newest library audio in that priority order."""

    choices = reference_audio_choices(reference_root)
    choice_values = {_path_key(value): value for _, value in choices}
    selected = choice_values.get(_path_key(selected_library))
    newest = _latest_reference_choice(choices)

    source = str(reference_source or "empty")
    current = _existing_path(current_reference)
    if current and source not in _AUTO_REFERENCE_SOURCES:
        return ReferenceSelection(
            prompt=current,
            library_value=selected,
            source=source if source != "empty" else "manual",
            message="",
            choices=choices,
        )

    lora_name = ""
    missing_lora_reference = False
    if lora_path and auto_lora_reference:
        recommended = _existing_path(lora_reference) or _recommended_lora_reference(lora_path)
        adapter_dir = Path(lora_path).expanduser().resolve().parent
        if adapter_dir.name.casefold() == "best":
            adapter_dir = adapter_dir.parent
        lora_name = adapter_dir.name
        if recommended:
            unchanged = source == "lora_auto" and _same_reference_file(current, recommended)
            message = "" if unchanged else (
                "No Reference Voice was provided. Automatically loaded and will use "
                f"{Path(recommended).name} from LoRA / DoRA {adapter_dir.name}."
            )
            return ReferenceSelection(
                prompt=recommended,
                library_value=selected,
                source="lora_auto",
                message=message,
                choices=choices,
            )
        missing_lora_reference = True

    if newest:
        unchanged = source == "library_auto" and _same_reference_file(current, newest)
        if unchanged:
            message = ""
        elif missing_lora_reference:
            message = (
                f"The selected LoRA / DoRA {lora_name} has no saved reference audio. "
                f"Automatically loaded and will use {Path(newest).name}, the latest modified "
                "audio in reference_audios."
            )
        elif lora_path and not auto_lora_reference:
            message = (
                "Automatic LoRA / DoRA reference loading is disabled. Automatically loaded and "
                f"will use {Path(newest).name}, the latest modified audio in reference_audios."
            )
        else:
            message = (
                "No Reference Voice was provided. Automatically loaded and will use "
                f"{Path(newest).name}, the latest modified audio in reference_audios."
            )
        return ReferenceSelection(
            prompt=newest,
            library_value=newest,
            source="library_auto",
            message=message,
            choices=choices,
        )

    return ReferenceSelection(
        prompt=None,
        library_value=None,
        source="empty",
        message=(
            (
                f"The selected LoRA / DoRA {lora_name} has no saved reference audio, and "
                "reference_audios is empty. "
            )
            if missing_lora_reference
            else "No Reference Voice is available. "
        )
        + (
            "Add audio to reference_audios, choose audio or video, load a media path, or "
            "select a LoRA / DoRA with a saved reference."
        ),
        choices=choices,
    )


def reference_selection_updates(
    current_reference: Any,
    selected_library: Any,
    reference_source: str | None,
    lora_path: str | None,
    auto_lora_reference: bool,
) -> tuple[Any, Any, str, Any]:
    """Refresh automatic references and surface the decision in the Gradio UI."""

    selection = resolve_reference_selection(
        current_reference,
        selected_library,
        reference_source,
        lora_path,
        auto_lora_reference,
    )
    if selection.message:
        if selection.prompt:
            gr.Info(selection.message, title="Reference Voice")
        else:
            gr.Warning(selection.message, title="Reference Voice")
    current_key = _path_key(_existing_path(current_reference))
    prompt_update = (
        gr.skip()
        if selection.prompt and current_key == _path_key(selection.prompt)
        else gr.update(value=selection.prompt)
    )
    return (
        prompt_update,
        gr.update(choices=selection.choices, value=selection.library_value),
        selection.source,
        selection.message or gr.skip(),
    )


def prepare_reference_for_generation(
    current_reference: Any,
    reference_media: Any,
    selected_library: Any,
    reference_source: str | None,
    time_ranges: str,
    lora_path: str | None,
    auto_lora_reference: bool,
    *,
    reference_root: str | os.PathLike[str] = REFERENCE_AUDIO_DIR,
    lora_reference: str | None = None,
) -> PreparedReference:
    """Resolve and, when requested, trim the visible Gradio Reference Voice."""

    current_source = str(reference_source or "empty")
    original_media = _existing_path(reference_media)
    resolver_current = (
        original_media
        if current_source in _AUTO_REFERENCE_SOURCES and original_media
        else current_reference
    )
    selection = resolve_reference_selection(
        resolver_current,
        selected_library,
        reference_source,
        lora_path,
        auto_lora_reference,
        reference_root=reference_root,
        lora_reference=lora_reference,
    )
    if not selection.prompt:
        raise ValueError(selection.message)

    selected_prompt = selection.prompt
    ranges_value = str(time_ranges or "").strip()
    if selection.source in _AUTO_REFERENCE_SOURCES:
        media_source = selected_prompt
    else:
        media_source = original_media or selected_prompt

    video: str | None = None
    if ranges_value:
        trimmed_audio, video, extraction_message = load_reference_media(
            media_source,
            ranges_value,
            require_ranges=True,
        )
        if not trimmed_audio:
            raise ValueError(extraction_message)
        selected_prompt = trimmed_audio
        message = " ".join(
            part
            for part in (
                selection.message,
                extraction_message,
                "The joined range audio is shown in Reference Voice and will be used.",
            )
            if part
        )
    else:
        if original_media and selection.source not in _AUTO_REFERENCE_SOURCES:
            try:
                video = (
                    _browser_safe_video_path(original_media)
                    if probe_media(original_media).has_video
                    else None
                )
            except Exception:
                video = None
        message = selection.message or (
            f"Using the selected Reference Voice: {Path(selected_prompt).name}."
        )

    return PreparedReference(
        prompt=selected_prompt,
        media=_browser_safe_media_path(media_source),
        video=video,
        library_value=selection.library_value,
        source=selection.source,
        message=message,
        choices=selection.choices,
    )


def _lora_choices() -> list[tuple[str, str]]:
    entries = scan_lora_files([str(ROOT / "loras")])
    choices: list[tuple[str, str]] = [("None", "")]
    for entry in entries:
        source = Path(entry.path).resolve()
        try:
            info = inspect_lora(source)
        except Exception:
            continue
        parent = source.parent.parent.name if source.parent.name.lower() == "best" else source.parent.name
        adapter_type = "DoRA" if str(info.get("adapter_type", "")).lower() == "dora" else "LoRA"
        label = (
            f"{parent}/{source.stem}  ·  {adapter_type} r{int(info.get('rank', 0) or 0)}"
            f"  ·  {int(info.get('steps', 0) or 0)} steps"
        )
        if source.parent.name.lower() == "best":
            label += "  [best - lowest validation loss]"
        choices.append((label, str(source)))
    return choices


_MODEL_DIR = str(ROOT / "models")
_PROFILE_CACHE: dict[str, tuple[float, dict[str, Any] | None]] = {}
_DICTIONARY_CACHE: dict[str, tuple[float, tuple[DictionaryEntry, ...]]] = {}


def set_model_dir(model_dir: str | os.PathLike[str]) -> None:
    """Remember the model folder so adapter panels can count text tokens like the engine."""

    global _MODEL_DIR
    _MODEL_DIR = str(model_dir)


def _token_len_for_profiles():
    try:
        tokenizer = _preview_tokenizer(str(Path(_MODEL_DIR).resolve()))
    except Exception:
        return None
    return lambda value: len(tokenizer.encode(value, allowed_special="all"))


def adapter_dataset_profile(path: str | None) -> dict[str, Any] | None:
    """The adapter's dataset profile, measured on first use while its dataset still exists."""

    if not path:
        return None
    key = os.path.normcase(str(Path(path).expanduser().resolve()))
    try:
        stamp = profile_path(path).stat().st_mtime
    except OSError:
        stamp = 0.0
    cached = _PROFILE_CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    profile = ensure_dataset_profile(path, token_len=_token_len_for_profiles(), datasets_root=ROOT / "datasets")
    try:
        stamp = profile_path(path).stat().st_mtime
    except OSError:
        stamp = 0.0
    _PROFILE_CACHE[key] = (stamp, profile)
    return profile


def adapter_vocabulary(path: str | None) -> frozenset[str]:
    """Every word the adapter's training transcripts contain (empty for the base model)."""

    if not path:
        return frozenset()
    adapter_dataset_profile(path)
    return load_dataset_vocabulary(path)


def pronunciation_dictionary_path() -> Path:
    return default_dictionary_path(ROOT)


def pronunciation_entries(path: str | os.PathLike[str] | None = None) -> list[DictionaryEntry]:
    """The user's pronunciation dictionary, created with the built-in technical terms on first use."""

    target = Path(path) if path else pronunciation_dictionary_path()
    if not target.is_file():
        try:
            save_dictionary(target, builtin_entries())
        except OSError:
            return builtin_entries()
    key = os.path.normcase(str(target.resolve()))
    try:
        stamp = target.stat().st_mtime
    except OSError:
        stamp = 0.0
    cached = _DICTIONARY_CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return list(cached[1])
    entries = load_dictionary(target)
    _DICTIONARY_CACHE[key] = (stamp, tuple(entries))
    return entries


def apply_pronunciation_dictionary(text: str, lora_path: str | None) -> str:
    """Rewrite words with dictionary readings, leaving words the selected voice was trained on alone."""

    entries = pronunciation_entries()
    if not entries:
        return str(text or "")
    return apply_dictionary(str(text or ""), entries, known_words=adapter_vocabulary(lora_path))


def check_unknown_words(text: str, lora_path: str | None, table_rows: Any) -> tuple[list[list[Any]], str]:
    """Words in the text that neither the dictionary, the voice's training transcripts nor the CMU dictionary cover."""

    entries = entries_from_rows(table_rows) if table_rows else pronunciation_entries()
    if not str(text or "").strip():
        return [], "Enter text first; the check lists the words that need a reading."
    known = adapter_vocabulary(lora_path)
    rows = check_text(str(text), known_words=known, entries=entries, token_len=_token_len_for_profiles())
    table = [
        [row["word"], row["suggestion"], row["kind"], row["method"], row["confidence"], int(row["fragments"] or 0)]
        for row in rows
    ]
    voice = (
        f"the selected voice's {len(known)} training words" if known
        else ("the base model" if not lora_path else "this voice (no training vocabulary saved)")
    )
    cmu_note = "" if cmu_available() else " The CMU dictionary package is not installed, so only letter rules propose readings."
    if not table:
        return [], f"Every word is covered by the pronunciation dictionary, {voice}, or the CMU dictionary.{cmu_note}"
    return table, (
        f"{len(table)} word(s) have no reading from the dictionary, {voice}, or the CMU dictionary. "
        "Review the suggested ARPAbet (stress digits, dots between syllables). Dictionary-backed readings (medium or high confidence) "
        "are added by the button; low-confidence letter-rule guesses are listed for you to correct by hand."
        + cmu_note
    )


def add_suggestions_to_dictionary(unknown_rows: Any, table_rows: Any) -> tuple[list[list[Any]], str]:
    """Add the dictionary-backed suggestions; letter-rule guesses (low confidence) are left for hand editing.

    A listening check found that annotations built from dictionary parts fix words the voice mangles,
    while guessed readings can make a word worse than its plain spelling.
    """

    current = entries_from_rows(table_rows)
    additions: list[DictionaryEntry] = []
    skipped: list[str] = []
    for row in unknown_rows or []:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        confidence = str(row[4] or "").strip().lower() if len(row) > 4 else ""
        if confidence == "low":
            skipped.append(str(row[0] or ""))
            continue
        entry = normalize_entry(str(row[0] or ""), str(row[1] or ""), source="suggested")
        if entry is not None:
            additions.append(entry)
    note = (
        f" Left out {len(skipped)} letter-rule guess(es) ({', '.join(skipped[:6])}{'…' if len(skipped) > 6 else ''}): "
        "type their readings by hand, a wrong guess sounds worse than the plain spelling."
        if skipped else ""
    )
    if not additions:
        return dictionary_rows(current), ("No suggestions to add; run the check first." if not skipped else "No dictionary-backed suggestions to add." + note)
    merged = merge_entries(current, additions)
    return dictionary_rows(merged), f"Added {len(additions)} reading(s) to the dictionary." + note


def add_suggestions_and_save(unknown_rows: Any, table_rows: Any) -> tuple[list[list[Any]], str]:
    """Merge the suggested readings into the dictionary and store it at once."""

    rows, message = add_suggestions_to_dictionary(unknown_rows, table_rows)
    if "No suggestions" in message:
        return rows, message
    saved_rows, saved_message = save_dictionary_rows(rows)
    return saved_rows, saved_message


def save_dictionary_rows(table_rows: Any) -> tuple[list[list[Any]], str]:
    entries = entries_from_rows(table_rows)
    try:
        path = save_dictionary(pronunciation_dictionary_path(), entries)
    except OSError as exc:
        return dictionary_rows(entries), f"The dictionary was not saved: {exc}"
    _DICTIONARY_CACHE.clear()
    return dictionary_rows(entries), f"Saved {len(entries)} entrie(s) to `{path}`."


def reload_dictionary_rows() -> tuple[list[list[Any]], str]:
    _DICTIONARY_CACHE.clear()
    entries = pronunciation_entries()
    return dictionary_rows(entries), f"Loaded {len(entries)} entrie(s) from `{pronunciation_dictionary_path()}`."


def _lora_reference(path: str | None) -> str | None:
    if not path:
        return None
    try:
        info = inspect_lora(path)
    except Exception:
        return None
    return _resolve_lora_reference_path(path, info.get("recommended_reference"))


def _seconds_cell(words: float | None, words_per_second: float) -> str:
    if words is None or words_per_second <= 0.0:
        return "–"
    return f"{seconds_for_words(float(words), words_per_second):.1f}"


def _range_cells(words: Sequence[int] | None, words_per_second: float) -> tuple[str, str]:
    if not words or len(words) < 2:
        return "–", "–"
    low, high = int(words[0]), int(words[1])
    if low == high:
        return str(low), _seconds_cell(low, words_per_second)
    return f"{low} to {high}", (
        f"{seconds_for_words(low, words_per_second):.1f} to {seconds_for_words(high, words_per_second):.1f}"
        if words_per_second > 0.0 else "–"
    )


def _histogram_notes(profile: Mapping[str, Any], words_per_second: float, recommendation: Mapping[str, Any]) -> list[str]:
    notes: list[str] = []
    histogram = profile.get("duration_histogram") or {}
    clips = int(profile.get("clips") or 0)
    duration = profile.get("duration_s") or {}
    words = profile.get("words") or {}
    if histogram and clips:
        peak_label, peak_count = max(histogram.items(), key=lambda item: item[1])
        under_six = int(histogram.get("<3s", 0)) + int(histogram.get("3-6s", 0))
        over_fifteen = int(histogram.get(">15s", 0))
        notes.append(
            f"The training clips peak at {html.escape(str(peak_label))} ({peak_count / clips:.0%} of {clips} clips); "
            f"the median clip is {float(duration.get('p50', 0.0)):.1f} s and {float(words.get('p50', 0.0)):.0f} words. "
            f"{under_six / clips:.0%} are under 6 s and {over_fifteen / clips:.0%} over 15 s (longest {float(duration.get('max', 0.0)):.1f} s)."
        )
    budget_words = recommendation.get("sentence_max_words")
    minimum = recommendation.get("sentence_min_alone_words")
    if budget_words:
        notes.append(
            f"Sentences are merged into one line up to the token budget, so lines land between about "
            f"{max(1, int(budget_words) - int(round(float((profile.get('words_per_sentence') or {}).get('p50', 0) or 0))))} and {int(budget_words)} words; "
            f"a single sentence longer than {int(budget_words)} words is cut at a comma or a word boundary."
        )
    if minimum:
        notes.append(
            f"A sentence under {int(minimum)} words standing alone is shorter than 95 percent of the training clips, "
            "where the voice has the least practice; join it with its neighbours."
        )
    if words_per_second > 0.0:
        notes.append(
            f"Seconds are computed at {words_per_second:.2f} words/s, the pace of this voice at the current speaking rate; "
            "move the slider and they update."
        )
    return notes


def adapter_panel_html(
    info: Mapping[str, Any],
    path: str,
    *,
    rate_report: Any,
    calibrated_rate: float | None,
    calibration_method: str,
    profile: Mapping[str, Any] | None,
    decoder_path: str,
    decoding: Mapping[str, Any] | None,
    speaking_rate: float | None,
    max_tokens: int | None,
    budget_scale: float | None,
    language: str | None,
    auto_tokens: bool | None,
    expressive_path: str | None = None,
    auto_pauses: bool | None = None,
    segmentation_mode: str | None = None,
) -> str:
    """The Voice LoRA / DoRA panel: identity, speaking rate, line-length rules, pauses, decoder and decoding."""

    esc = html.escape
    chips = [
        f"rank {esc(str(info.get('rank', '?')))} · alpha {esc(str(info.get('alpha', '?')))}",
        f"{int(info.get('steps', 0) or 0)} steps",
        f"dataset {esc(str(info.get('dataset') or 'not recorded'))}",
        f"{esc(str(info.get('date') or 'date not recorded')[:19])}",
        f"{len(info.get('targets') or [])} targets · {float(info.get('size_mb', 0.0) or 0.0):.2f} MB",
    ]
    head = (
        f'<div class="adapter-head"><b>{esc(str(info.get("adapter_type", "adapter")).upper())}</b>'
        + "".join(f'<span class="adapter-chip">{chip}</span>' for chip in chips)
        + "</div>"
    )

    # ---- speaking rate ------------------------------------------------------------------
    current_rate = float(speaking_rate) if isinstance(speaking_rate, (int, float)) else None
    dataset_wps = 0.0
    generated_wps = 0.0
    if rate_report is not None:
        dataset_wps = float(rate_report.dataset_words_per_second or 0.0)
        generated_wps = float(rate_report.generated_words_per_second or 0.0)
    if dataset_wps <= 0.0 and profile:
        dataset_wps = float(profile.get("words_per_second") or 0.0)
    if current_rate is None:
        current_rate = float(rate_report.recommended_speaking_rate) if rate_report is not None else 1.0
    effective_wps = generated_wps * current_rate if generated_wps > 0.0 else dataset_wps
    def rate_row(label: str, note: str, rate: str, pace: str, css: str = "") -> str:
        # The provenance note wraps under the label, so the card never clips a long source.
        sub = f'<span class="sub">{note}</span>' if note else ""
        return f'<tr class="{css}"><td>{esc(label)}{sub}</td><td>{rate}</td><td>{pace}</td></tr>'

    rate_rows: list[str] = []
    if rate_report is None:
        rate_rows.append(
            "<tr><td colspan=\"3\">No calibrated speaking rate yet: train with epoch samples or use the "
            "Checkpoint Grid calibration button. The slider value is used as is.</td></tr>"
        )
    else:
        if calibrated_rate is not None:
            calibrated_wps = generated_wps * float(calibrated_rate)
            rate_rows.append(
                rate_row(
                    "Calibrated (original)",
                    esc(speaking_rate_method_label(calibration_method)),
                    f"{float(calibrated_rate):.3f}",
                    f"{calibrated_wps:.2f} words/s",
                )
            )
        saved_note = (
            f"set manually {esc(str(rate_report.generated_at)[:10])}" if rate_report.is_manual
            else "the calibration is in use"
        )
        rate_rows.append(
            rate_row(
                "Saved for this adapter",
                saved_note,
                f"{float(rate_report.recommended_speaking_rate):.3f}",
                f"{generated_wps * float(rate_report.recommended_speaking_rate):.2f} words/s",
            )
        )
        difference = ""
        if dataset_wps > 0.0 and effective_wps > 0.0:
            ratio = effective_wps / dataset_wps - 1.0
            difference = (
                "at the narrator's pace" if abs(ratio) < 0.005
                else f"{abs(ratio) * 100:.0f} % {'faster' if ratio > 0 else 'slower'} than the recordings"
            )
        rate_rows.append(
            rate_row("Speaking rate slider now", esc(difference), f"{current_rate:.3f}", f"{effective_wps:.2f} words/s", "target")
        )
    if dataset_wps > 0.0:
        clips = f"{int(profile.get('clips', 0))} clips" if profile else f"{int(getattr(rate_report, 'clips_used', 0) or 0)} matched clips"
        rate_rows.append(rate_row("Recordings", esc(clips), "–", f"{dataset_wps:.2f} words/s"))
    rate_card = (
        '<div class="adapter-card rate"><h4>Speaking rate</h4><table>'
        "<tr><th>Source</th><th>Rate</th><th>Pace</th></tr>" + "".join(rate_rows) + "</table></div>"
    )

    # ---- words per line -----------------------------------------------------------------
    scale = float(budget_scale) if isinstance(budget_scale, (int, float)) and budget_scale else 0.72
    lang = str(language or (profile or {}).get("language") or "EN")
    if profile:
        recommendation = profile.get("recommendation") or {}
        target = _range_cells(recommendation.get("target_words"), effective_wps)
        acceptable = _range_cells(recommendation.get("acceptable_words"), effective_wps)
        rows = [
            ("Target", *target, "target"),
            ("Acceptable range", *acceptable, ""),
            ("Hard minimum", str(recommendation.get("hard_min_words", "–")), _seconds_cell(recommendation.get("hard_min_words"), effective_wps), ""),
            ("Hard maximum", str(recommendation.get("hard_max_words", "–")), _seconds_cell(recommendation.get("hard_max_words"), effective_wps), ""),
            ("Never exceed", str(recommendation.get("never_exceed_words", "–")), _seconds_cell(recommendation.get("never_exceed_words"), effective_wps), ""),
        ]
        line_table = (
            f"<table><tr><th>Rule</th><th>Words</th><th>Seconds at {effective_wps:.2f} words/s</th></tr>"
            + "".join(
                f'<tr class="{css}"><td>{esc(label)}</td><td>{esc(words)}</td><td>{esc(seconds)}</td></tr>'
                for label, words, seconds, css in rows
            )
            + "</table>"
        )
        recommended = recommended_max_tokens(profile, language=lang, budget_scale=scale)
        current_tokens = int(max_tokens) if isinstance(max_tokens, (int, float)) else None
        token_lines: list[str] = []
        if recommended is not None:
            rec_words = words_for_max_tokens(profile, recommended, language=lang, budget_scale=scale)
            token_lines.append(
                f"<b>Max tokens per segment {recommended}</b> fits this voice: one line holds up to about {rec_words:.0f} words "
                f"({_seconds_cell(rec_words, effective_wps)} s)"
                + (" and is applied automatically." if auto_tokens else "; enable <b>Auto from LoRA / DoRA dataset</b> to apply it.")
            )
        if current_tokens is not None and (recommended is None or current_tokens != recommended):
            cur_words = words_for_max_tokens(profile, current_tokens, language=lang, budget_scale=scale)
            token_lines.append(
                f"The current setting {current_tokens} cuts lines at about {cur_words:.0f} words ({_seconds_cell(cur_words, effective_wps)} s)."
            )
        target_tokens = smart_target_tokens(profile)
        if target_tokens:
            tokens_per_word = float(profile.get("tokens_per_word") or 0.0) or 1.3
            target_words = target_tokens / tokens_per_word
            selected = " (selected)" if str(segmentation_mode or "") == "smart" else ""
            token_lines.append(
                f"<b>Smart sentences</b>{selected} packs whole sentences to about {target_words:.0f} words "
                f"({_seconds_cell(target_words, effective_wps)} s) per line, the median training clip, and only cuts inside a "
                "sentence that is longer than the token limit."
            )
        if budget_scale_for(lang, scale) != 1.0:
            token_lines.append(f"Budget scale {scale:.2f} is included in these counts.")
        lines_card = (
            '<div class="adapter-card lines"><h4>Words per generated line'
            '<span class="adapter-hint">one speech segment</span></h4>'
            + line_table + "<div>" + " ".join(token_lines) + "</div></div>"
        )
        sentence_rows = [
            ("Minimum standing alone", recommendation.get("sentence_min_alone_words")),
            ("Maximum", recommendation.get("sentence_max_words")),
        ]
        sentence_card = (
            '<div class="adapter-card sentences"><h4>Per sentence inside a line</h4><table>'
            "<tr><th>Rule</th><th>Words</th><th>Seconds</th></tr>"
            + "".join(
                f"<tr><td>{esc(label)}</td><td>{esc(str(value if value is not None else '–'))}</td>"
                f"<td>{_seconds_cell(value, effective_wps)}</td></tr>"
                for label, value in sentence_rows
            )
            + "</table><ul>"
            + "".join(f"<li>{note}</li>" for note in _histogram_notes(profile, effective_wps, recommendation))
            + "</ul></div>"
        )
    else:
        lines_card = (
            '<div class="adapter-card lines"><h4>Words per generated line</h4>'
            "<div>No dataset statistics: the training dataset of this adapter is not available, so the line-length "
            "rules and the automatic token budget cannot be derived. Keep the dataset folder next to the app or "
            "retrain with the current version, which saves the profile with the adapter.</div></div>"
        )
        sentence_card = ""

    pause_card = _pause_card_html(profile, auto_pauses)

    # ---- decoder, decoding, provenance ----------------------------------------------------
    decoder_line = (
        f"Voice decoder adapter <b>found</b> (<code>{esc(Path(decoder_path).name)}</code>, selected automatically; "
        "choose <b>None</b> under Voice decoder adapter to hear the GPT adapter alone)."
        if decoder_path else
        "Voice decoder adapter <b>none</b> (train with <b>Adapt the voice decoder after training</b> enabled to add one)."
    )
    decoding_line = (
        f"Decoding settings from the sweep: temperature <b>{decoding['temperature']:g}</b>, guidance "
        f"<b>{decoding['inference_cfg_rate']:g}</b>, beams <b>{decoding['num_beams']}</b> (applied with the calibrated speaking rate)."
        if decoding else "Decoding settings: <b>defaults</b> (no accepted sweep override for this training)."
    )
    provenance = ""
    if profile:
        provenance = (
            f"Dataset profile: {int(profile.get('clips', 0))} training clips of <code>{esc(str(profile.get('dataset_name') or ''))}</code>, "
            f"measured {esc(str(profile.get('generated_at') or '')[:10])}."
        )
    expressive_entry = (profile or {}).get("expressive_reference") or {}
    if expressive_path:
        detail = ""
        if expressive_entry.get("pitch_std_st"):
            detail = (
                f" (pitch std {float(expressive_entry['pitch_std_st']):.2f} st against {float(expressive_entry.get('pool_pitch_std_st', 0.0)):.2f} "
                f"for the pool, loudness std {float(expressive_entry.get('energy_std_db', 0.0)):.2f} dB)"
            )
        expressive_line = (
            f"Expressive emotion clip <b>found</b> (<code>{esc(Path(expressive_path).name)}</code>{esc(detail)}): used as the emotion prompt "
            "while Emotion source is <b>Same as speaker voice</b> and <b>Use the LoRA / DoRA expressive clip</b> is on; the speaker "
            "reference keeps the identity."
        )
    else:
        expressive_line = (
            "Expressive emotion clip <b>none</b>: press <b>Pick expressive clip</b> to choose the liveliest clean training clip "
            "(new training runs save one automatically)."
        )
    notes_card = (
        '<div class="adapter-card notes"><h4>Decoder, decoding and files</h4>'
        f"<div>{decoder_line}</div><div>{decoding_line}</div><div>{expressive_line}</div>"
        + (f"<div>{provenance}</div>" if provenance else "")
        + f'<div class="adapter-path">{esc(str(Path(path).expanduser().resolve()))}</div></div>'
    )
    return (
        '<div class="adapter-panel">' + head
        + '<div class="adapter-grid">' + rate_card + lines_card + sentence_card + pause_card + notes_card + "</div></div>"
    )


def _pause_card_html(profile: Mapping[str, Any] | None, auto_pauses: bool | None) -> str:
    """The panel card with the speaker's measured pauses and the settings they recommend."""

    esc = html.escape
    pauses = (profile or {}).get("pauses") or {}
    recommendation = pauses.get("recommendation") or {}
    if not recommendation:
        return (
            '<div class="adapter-card pauses"><h4>Pauses</h4>'
            "<div>No pause statistics yet: they are measured from the training clips when the dataset folder is present "
            "(select the adapter again) or saved by a new training run.</div></div>"
        )
    sentence = pauses.get("sentence_pauses_ms") or {}
    within = pauses.get("within_sentence_pauses_ms") or {}
    fraction = pauses.get("pause_time_fraction") or {}
    rows = [
        (
            "Sentence pause", f"{int(recommendation.get('sentence_pause_ms', 0))} ms",
            f"median of {int(sentence.get('count', 0) or 0)} pauses between sentences "
            f"({float(sentence.get('p25', 0.0)):.0f} to {float(sentence.get('p75', 0.0)):.0f} ms for half of them)"
            if int(sentence.get("count", 0) or 0) else f"estimated from {esc(str(recommendation.get('sentence_pause_source') or 'the recordings'))}",
            "target",
        ),
        (
            "Maximum pause", f"{int(recommendation.get('max_pause_ms', 0))} ms",
            "only one in ten of the speaker's sentence pauses is longer; generated pauses above it are shortened", "",
        ),
        (
            "Inside a sentence", f"{float(within.get('p50', 0.0)):.0f} ms",
            f"median pause at commas and breaths; nine in ten are under {float(within.get('p90', 0.0)):.0f} ms"
            if int(within.get("count", 0) or 0) else "not measured", "",
        ),
        (
            "Pause share", f"{float(fraction.get('mean', 0.0) or 0.0) * 100:.0f} %",
            f"of clip time is silence between words, over {int(pauses.get('clips', 0))} clips", "",
        ),
    ]
    note = (
        "Applied to <b>Sentence pause</b> and <b>Maximum pause</b> automatically." if auto_pauses
        else "Enable <b>Auto pauses from LoRA / DoRA dataset</b> to apply them to Sentence pause and Maximum pause."
    )
    return (
        '<div class="adapter-card pauses"><h4>Pauses of this speaker</h4><table>'
        "<tr><th>Setting</th><th>Value</th></tr>"
        + "".join(
            # The measurement wraps under the setting name instead of sitting in a clipped third column.
            f'<tr class="{css}"><td>{esc(label)}<span class="sub">{detail}</span></td><td>{esc(value)}</td></tr>'
            for label, value, detail, css in rows
        )
        + f"</table><div>{note}</div></div>"
    )


def _lora_info(
    path: str | None,
    *,
    speaking_rate: float | None = None,
    max_tokens: int | None = None,
    budget_scale: float | None = None,
    language: str | None = None,
    auto_tokens: bool | None = None,
    auto_pauses: bool | None = None,
    segmentation_mode: str | None = None,
) -> tuple[str, str | None]:
    """HTML for the Voice LoRA / DoRA panel and the adapter's recommended reference path."""

    if not path:
        return (
            '<div class="adapter-panel"><div class="adapter-head">No LoRA / DoRA selected. Base model (no LoRA / DoRA) '
            "will clone from the reference only.</div></div>",
            None,
        )
    try:
        info = inspect_lora(path)
        rate_report = ensure_calibration_fields(path, load_speaking_rate(path))
        calibrated_rate, calibration_method = original_calibration(path, rate_report) if rate_report is not None else (None, "")
        markdown = adapter_panel_html(
            info,
            str(path),
            rate_report=rate_report,
            calibrated_rate=calibrated_rate,
            calibration_method=calibration_method,
            profile=adapter_dataset_profile(str(path)),
            decoder_path=find_decoder_adapter(path),
            decoding=load_decoding_settings(path),
            speaking_rate=speaking_rate,
            max_tokens=max_tokens,
            budget_scale=budget_scale,
            language=language,
            auto_tokens=auto_tokens,
            expressive_path=expressive_reference_path(str(path)),
            auto_pauses=auto_pauses,
            segmentation_mode=segmentation_mode,
        )
        reference = _resolve_lora_reference_path(path, info.get("recommended_reference"))
        return markdown, reference
    except Exception as exc:
        return f'<div class="adapter-panel">LoRA / DoRA inspection failed: {html.escape(str(exc))}</div>', None


def pick_expressive_clip(path: str | None) -> str:
    """Choose the liveliest clean training clip of the adapter's dataset and save it beside the adapter."""

    if not path or not Path(path).expanduser().is_file():
        return "Select a LoRA / DoRA file first."
    dataset_dir = dataset_dir_for_adapter(path, datasets_root=ROOT / "datasets")
    if dataset_dir is None:
        return "The adapter's training dataset was not found, so no clip can be measured. Keep the dataset folder next to the app."
    from indextts.training.dataset_manifest import load_manifest
    from indextts.training.voice_profile import choose_expressive_reference, describe_expressive_choice

    rows = load_manifest(dataset_dir)
    has_split = any(row.get("split") for row in rows)
    records = [row for row in rows if not has_split or str(row.get("split") or "train") == "train"]
    try:
        choice = choose_expressive_reference(dataset_dir, records)
    except Exception as exc:  # measurement failures must not break the tab
        return f"Expressive clip selection failed: {exc}"
    if choice is None:
        return "No clean clip of prompt length could be measured in the training split."
    adapter_dir = Path(path).expanduser().resolve().parent
    if adapter_dir.name.lower() == "best":
        adapter_dir = adapter_dir.parent
    try:
        saved = save_expressive_reference(adapter_dir, adapter_dir.name, dataset_dir, choice)
        adapter_dataset_profile(str(path))
        update_profile_expressive_reference(path, expressive_profile_entry(choice, saved))
    except OSError as exc:
        return f"The expressive clip could not be saved: {exc}"
    _PROFILE_CACHE.clear()
    return f"Saved {saved.name}: {describe_expressive_choice(choice)}."


def preview_words_per_second(lora_path: str | None, speaking_rate: Any = None) -> float:
    """Words per second the live preview uses for its seconds estimate: the voice's pace at the slider's rate."""

    if not lora_path:
        return 0.0
    try:
        report = load_speaking_rate(lora_path)
        rate = float(speaking_rate) if isinstance(speaking_rate, (int, float)) and speaking_rate else None
        if report is not None and float(report.generated_words_per_second or 0.0) > 0.0:
            return float(report.generated_words_per_second) * (rate if rate else float(report.recommended_speaking_rate or 1.0))
        profile = adapter_dataset_profile(str(lora_path))
        return float((profile or {}).get("words_per_second") or 0.0)
    except Exception:
        return 0.0


def auto_pause_updates(path: str | None, enabled: bool) -> tuple[Any, Any]:
    """**Sentence pause** and **Maximum pause** values measured from the adapter's recordings, or no change."""

    if not enabled or not path:
        return gr.skip(), gr.skip()
    values = recommended_pauses(adapter_dataset_profile(str(path)))
    if not values:
        return gr.skip(), gr.skip()
    return gr.update(value=int(values[0])), gr.update(value=int(values[1]))


def auto_max_tokens_update(path: str | None, enabled: bool, budget_scale: float | None, language: str | None) -> Any:
    """The **Max tokens per segment** value the selected adapter's dataset calls for, or no change."""

    if not enabled or not path:
        return gr.skip()
    profile = adapter_dataset_profile(str(path))
    if not profile:
        return gr.skip()
    scale = float(budget_scale) if isinstance(budget_scale, (int, float)) and budget_scale else 0.72
    value = recommended_max_tokens(profile, language=str(language or profile.get("language") or "EN"), budget_scale=scale)
    return gr.skip() if value is None else gr.update(value=int(value))


def lora_selection_updates(
    path: str,
    current_reference: str | None,
    auto_reference: bool,
    auto_speaking_rate: bool,
    reference_source: str | None = "empty",
    *,
    panel: Mapping[str, Any] | None = None,
) -> tuple[Any, Any, str, Any, Any]:
    """Apply adapter metadata to the reference and speaking-rate controls.

    ``panel`` carries the current slider values (speaking_rate, max_tokens,
    budget_scale, language, auto_tokens) so the adapter panel reflects them.
    """

    recommended_reference = _lora_reference(path)
    reference_update: Any = gr.skip()
    source_update: Any = gr.skip()
    messages: list[str] = []
    current = _existing_path(current_reference)
    source = str(reference_source or "empty")
    can_replace = not current or source in _AUTO_REFERENCE_SOURCES
    if auto_reference and path and recommended_reference and can_replace:
        reference_update = gr.update(value=recommended_reference)
        source_update = "lora_auto"
        reference_message = (
            "No manual Reference Voice was selected. Automatically loaded and will use "
            f"{Path(recommended_reference).name} from the selected LoRA / DoRA."
        )
        messages.append(reference_message)
        if source != "lora_auto" or _path_key(current) != _path_key(recommended_reference):
            gr.Info(reference_message, title="LoRA / DoRA Reference Voice")
    elif auto_reference and path and recommended_reference and current:
        messages.append("Kept the manually selected Reference Voice; the LoRA / DoRA default was not applied.")
    elif auto_reference and path and not recommended_reference and can_replace:
        if source == "lora_auto":
            reference_update = gr.update(value=None)
            source_update = "empty"
        messages.append("The selected LoRA / DoRA has no usable saved reference audio.")
    elif source == "lora_auto" and (not path or not auto_reference):
        reference_update = gr.update(value=None)
        source_update = "empty"
        messages.append("Cleared the automatically loaded LoRA / DoRA reference.")

    rate_update: Any = gr.skip()
    if auto_speaking_rate:
        if not path:
            rate_update = 1.0
            messages.append("Reset speaking rate to the model's natural pace (1.0).")
        else:
            report = load_speaking_rate(path)
            if report is not None:
                rate_update = report.recommended_speaking_rate
                messages.append(
                    f"Applied calibrated speaking rate {report.recommended_speaking_rate:.3f}."
                )
            else:
                rate_update = 1.0
                messages.append("No calibration for this adapter; reset speaking rate to the model's natural pace (1.0).")

    if not messages:
        messages.append(
            "LoRA / DoRA selected."
            if path
            else "Base model (no LoRA / DoRA) selected."
        )
    panel_kwargs = dict(panel or {})
    if isinstance(rate_update, (int, float)):
        panel_kwargs["speaking_rate"] = float(rate_update)
    info, _reference = _lora_info(path, **panel_kwargs)
    return info, reference_update, " ".join(messages), rate_update, source_update


def decoding_updates(path: str | None, auto_apply: bool) -> tuple[Any, Any, Any]:
    """Temperature, guidance rate, and beams for the selected adapter's sweep result, or the defaults."""
    if not auto_apply:
        return gr.skip(), gr.skip(), gr.skip()
    settings = load_decoding_settings(path) if path else None
    if settings is None:
        return (gr.update(value=GENERATION_DEFAULTS["generation.temperature"]),
                gr.update(value=GENERATION_DEFAULTS["generation.inference_cfg_rate"]),
                gr.update(value=GENERATION_DEFAULTS["generation.num_beams"]))
    return (gr.update(value=settings["temperature"]), gr.update(value=settings["inference_cfg_rate"]),
            gr.update(value=settings["num_beams"]))


def saved_lora_speaking_rate(path: str | None) -> float | None:
    """Return the adapter's stored speaking rate for the editable field, or None."""

    if not path:
        return None
    report = load_speaking_rate(path)
    return float(report.recommended_speaking_rate) if report is not None else None


def save_lora_speaking_rate(
    path: str | None,
    value: float | None,
    auto_speaking_rate: bool,
    *,
    panel: Mapping[str, Any] | None = None,
) -> tuple[str, str, Any, Any]:
    """Store a manual speaking rate for the selected adapter and refresh its summary."""

    panel_kwargs = dict(panel or {})
    if not path or not Path(path).expanduser().is_file():
        return (
            _lora_info(path, **panel_kwargs)[0],
            "Select a LoRA / DoRA file before saving a speaking rate.",
            gr.skip(),
            gr.skip(),
        )
    try:
        if value is None:
            raise ValueError("enter a speaking rate between 0.5 and 1.5")
        report = save_manual_speaking_rate(path, float(value))
    except (TypeError, ValueError, OSError) as exc:
        return _lora_info(path, **panel_kwargs)[0], f"Speaking rate was not saved: {exc}", gr.skip(), gr.skip()
    message = f"Saved speaking rate {report.recommended_speaking_rate:.3f} for {Path(path).name}."
    if report.calibrated_speaking_rate is not None:
        message += f" The automatic calibration {report.calibrated_speaking_rate:.3f} stays on record."
    rate_update: Any = gr.skip()
    if auto_speaking_rate:
        rate_update = report.recommended_speaking_rate
        panel_kwargs["speaking_rate"] = float(report.recommended_speaking_rate)
        message += " Applied it to the Speaking rate slider."
    return _lora_info(path, **panel_kwargs)[0], message, rate_update, report.recommended_speaking_rate


def recent_outputs(root: str | os.PathLike[str] = ROOT / "outputs", limit: int = 10) -> list[list[Any]]:
    rows: list[tuple[float, list[Any]]] = []
    root_path = Path(root).expanduser().resolve()
    for metadata_path in root_path.rglob("metadata.json") if root_path.is_dir() else []:
        try:
            parts = metadata_path.parent.resolve().relative_to(root_path).parts
        except ValueError:
            continue
        lowered = [part.lower() for part in parts]
        if any(part.startswith("_") for part in parts):
            continue
        if any(part in {"grids", "worker_runtime_e2e", ".sample_jobs"} for part in lowered):
            continue
        first = lowered[0] if lowered else ""
        if first.startswith("ui_") and any(
            token in part for part in lowered for token in ("batch", "smoke")
        ):
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        output = metadata.get("outputs", {}).get("final_audio_path")
        if not output or not Path(output).is_file():
            continue
        task = metadata.get("task", {})
        created = metadata.get("created_at", "")
        row = [task.get("id", metadata_path.parent.name), created, metadata.get("status", ""), output, str(metadata_path.parent)]
        rows.append((metadata_path.stat().st_mtime, row))
    return [row for _, row in sorted(rows, key=lambda item: item[0], reverse=True)[:limit]]


def _summary_html(result: Mapping[str, Any]) -> str:
    return (
        '<div class="summary-strip">'
        f"Seed <b>{html.escape(str(result.get('seed', '--')))}</b> | "
        f"Segments <b>{html.escape(str(result.get('segments_count', '--')))}</b> | "
        f"Audio <b>{float(result.get('audio_seconds', 0.0) or 0.0):.2f}s</b> | "
        f"RTF <b>{float(result.get('rtf', 0.0) or 0.0):.3f}</b> | "
        f"GPT {float(result.get('gpt_time', 0.0) or 0.0):.2f}s, "
        f"s2mel {float(result.get('s2mel_time', 0.0) or 0.0):.2f}s, "
        f"vocoder {float(result.get('vocoder_time', 0.0) or 0.0):.2f}s | "
        f"Peak VRAM {float(result.get('peak_vram_gb', 0.0) or 0.0):.2f} GB"
        "</div>"
    )


def _candidate_state_value(paths: Sequence[Any] | None) -> str:
    """Serialize candidate paths for the hidden render trigger component."""

    return json.dumps([str(path) for path in paths or [] if path], ensure_ascii=True)


def _candidate_paths_from_state(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            decoded = []
    else:
        decoded = value
    if not isinstance(decoded, Sequence) or isinstance(decoded, (str, bytes)):
        return []
    return [str(path) for path in decoded if path]


def _result_updates(result: Mapping[str, Any], request: Mapping[str, Any]) -> tuple[Any, ...]:
    output = result.get("output_path")
    video = result.get("video_path")
    caption = result.get("subtitle_status") or ""
    runtime_warning = str(result.get("runtime_warning") or "")
    status = "Generation complete."
    if runtime_warning:
        status += f" {runtime_warning}"
    final_progress = read_progress_file(request.get("progress_file")) or {}
    final_progress.update(
        {
            "fraction": 1.0,
            "eta_s": 0,
            "desc": "Complete" if not runtime_warning else f"Complete | {runtime_warning}",
        }
    )
    if not final_progress.get("total"):
        final_progress.update({"completed": result.get("segments_count", 1), "total": result.get("segments_count", 1)})
    if not final_progress.get("vram_used_gb") and result.get("peak_vram_gb"):
        final_progress["vram_used_gb"] = result["peak_vram_gb"]
    return (
        progress_panel_html(final_progress, title="Generation complete"),
        status,
        tail_text(Path(str(request["task_layout"]["task_folder"])) / "generation.log", 60),
        gr.update(value=output, visible=bool(output)),
        gr.update(value=video, visible=bool(video)),
        _candidate_state_value(
            result.get("candidate_paths") or ([output] if output else [])
        ),
        gr.update(value=caption, visible=bool(caption)),
        _summary_html(result),
        recent_outputs(),
    )


def _terminal_generation_updates(
    request: Mapping[str, Any] | None,
    *,
    title: str,
    message: str,
) -> tuple[Any, ...]:
    request_value = dict(request or {})
    payload = read_progress_file(request_value.get("progress_file")) or {}
    payload.update({"eta_s": 0, "desc": message})
    task_layout = request_value.get("task_layout") or {}
    task_folder = task_layout.get("task_folder")
    log_value = tail_text(Path(str(task_folder)) / "generation.log", 60) if task_folder else ""
    return (
        progress_panel_html(payload, title=title),
        message,
        log_value,
        gr.update(value=None, visible=False),
        gr.update(value=None, visible=False),
        [],
        gr.update(value="", visible=False),
        "",
        recent_outputs(),
    )


def _generation_result_from_disk(task_folder: Path, metadata: Mapping[str, Any]) -> dict[str, Any]:
    result = read_json(task_folder / "result.json", {}) or {}
    if result.get("status") == "ok":
        return dict(result)
    outputs = dict(metadata.get("outputs") or {})
    generation = dict(metadata.get("generation") or {})
    candidates = list(outputs.get("candidate_wav_paths") or [])
    output = outputs.get("final_audio_path")
    return {
        **generation,
        "output_path": output,
        "video_path": outputs.get("final_video_path"),
        "candidate_paths": candidates or ([output] if output else []),
        "runtime_warning": metadata.get("runtime_warning", ""),
    }


_GENERATION_CARD_OWNERS: set[str] = set()
_GENERATION_CARD_LOCK = threading.Lock()


def _claim_generation_card(gr_request: gr.Request | None) -> None:
    """The connected click/stream owns this page until its next reload."""

    session = str(getattr(gr_request, "session_hash", "") or "")
    if session:
        with _GENERATION_CARD_LOCK:
            _GENERATION_CARD_OWNERS.add(session)


def _generation_card_is_owned(gr_request: gr.Request | None) -> bool:
    session = str(getattr(gr_request, "session_hash", "") or "")
    with _GENERATION_CARD_LOCK:
        return bool(session and session in _GENERATION_CARD_OWNERS)


def _guard_generation_poll(gr_request: gr.Request | None, updates: tuple[Any, ...]) -> tuple[Any, ...]:
    # A timer may have started before a Generate click and finished its disk
    # reads after validation failed. Check ownership again at the return boundary
    # rather than trusting that the timer's captured task state is still current.
    if _generation_card_is_owned(gr_request):
        return (*[gr.skip()] * 10, gr.Timer(5.0, active=False))
    return updates


def generation_task_updates(
    state_value: str,
    gr_request: gr.Request = None,
    *,
    output_root: str | os.PathLike[str] = ROOT / "outputs",
    page_load: bool = False,
) -> tuple[Any, ...]:
    """Discover and render the per-session generation task card."""

    if _generation_card_is_owned(gr_request):
        return _guard_generation_poll(gr_request, ())
    task_value, running = adopt_output_task(
        state_value,
        root=output_root,
        scope="generation",
        page_load=page_load,
    )
    if not task_value:
        return _guard_generation_poll(gr_request, (
            "",
            progress_panel_html({}, title="Ready"),
            "",
            "",
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            recent_outputs(output_root) if page_load else [],
            gr.Timer(5.0, active=False),
        ))

    task_folder = Path(task_value)
    metadata = read_json(task_folder / "metadata.json", {}) or {}
    request = dict(read_json(task_folder / "request.json", {}) or {})
    request.setdefault("progress_file", str(task_folder / "progress.json"))
    request.setdefault("task_layout", {"task_folder": str(task_folder)})
    task = dict(metadata.get("task") or {})
    task_name = str(task.get("id") or task_folder.name)
    payload = read_progress_file(task_folder / "progress.json") or {}
    log_value = tail_text(task_folder / "generation.log", 60)
    if running:
        description = str(payload.get("desc") or payload.get("stage") or "Model is working...")
        return _guard_generation_poll(gr_request, (
            task_value,
            progress_panel_html(payload, title="Generating voice"),
            f"Attached to running run {task_name} | {description}",
            log_value,
            gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
            recent_outputs(output_root),
            gr.Timer(1.0, active=True),
        ))

    metadata_status = str(metadata.get("status") or "").strip().lower()
    if metadata_status in {"complete", "completed"}:
        card = list(_result_updates(_generation_result_from_disk(task_folder, metadata), request))
        card[1] = f"Last task {task_name} | {card[1]}"
    else:
        title = "Canceled" if metadata_status in {"cancelled", "canceled"} else "Failed"
        message = str(metadata.get("error") or f"Generation {metadata_status or 'ended'}.")
        card = list(_terminal_generation_updates(request, title=title, message=message))
        card[1] = f"Last task {task_name} | {card[1]}"
    card[2] = log_value
    card[8] = recent_outputs(output_root)
    return _guard_generation_poll(gr_request, (task_value, *card, gr.Timer(5.0, active=False)))


class _Tee:
    """Forward all console writes, but capture only the creating worker's log.

    stdout/stderr redirection is process-wide. Foreign threads (for example,
    child-process log pumps) must still reach the console without being copied
    into this task's file. A retained tee stays a console-only forwarder after
    close, including while a late foreign write races worker shutdown.
    """

    def __init__(self, stream: Any, path: Path) -> None:
        self.stream = stream
        self.handle = path.open("a", encoding="utf-8", newline="\n")
        self.lock = threading.RLock()
        # Thread objects cannot be confused when the OS reuses a thread ID.
        self.owner_thread = threading.current_thread()

    def write(self, value: str) -> int:
        with self.lock:
            self.stream.write(value)
            self.stream.flush()
            if threading.current_thread() is self.owner_thread and not self.handle.closed:
                self.handle.write(value)
                self.handle.flush()
        return len(value)

    def flush(self) -> None:
        with self.lock:
            self.stream.flush()
            if threading.current_thread() is self.owner_thread and not self.handle.closed:
                self.handle.flush()

    def close(self) -> None:
        with self.lock:
            if not self.handle.closed:
                self.handle.close()


def stream_generation_request(
    request: Mapping[str, Any],
    *,
    use_subprocess: bool,
    gr_progress: Any = None,
    process_kind: str = "generation",
):
    """Execute one prepared request and yield a shared nine-output dashboard tuple."""

    global _ACTIVE_INPROCESS_TASK
    task_identity = str(Path(str(request["task_layout"]["task_folder"])).resolve())
    if not use_subprocess:
        _ACTIVE_INPROCESS_TASK = task_identity
    started = time.perf_counter()
    try:
        yield from _stream_generation_request(
            request,
            use_subprocess=use_subprocess,
            gr_progress=gr_progress,
            process_kind=process_kind,
        )
    except Exception as exc:
        # Model loading and worker startup happen before the runner's failure
        # handler. Persist those failures too, or polling resurrects the task as
        # "in progress" after the UI has already displayed its error.
        try:
            _record_generation_failure(request, str(exc), time.perf_counter() - started)
        except OSError as metadata_error:
            print(f">> Could not save generation failure: {metadata_error}", file=sys.stderr, flush=True)
        raise
    except GeneratorExit:
        # An in-process worker is joined by the inner generator before closing.
        # If it never started or failed during loading, do not leave stale live metadata.
        if not use_subprocess:
            _record_generation_failure(request, "Generation canceled after its stream closed.", time.perf_counter() - started)
        raise
    finally:
        if not use_subprocess and _ACTIVE_INPROCESS_TASK == task_identity:
            _ACTIVE_INPROCESS_TASK = ""


def _record_generation_failure(request: Mapping[str, Any], error: str, elapsed: float) -> None:
    metadata_path = request.get("metadata_path")
    if not metadata_path:
        return
    metadata = read_json(metadata_path, {}) or {}
    if not metadata or metadata.get("status") in {"completed", "complete", "failed", "error", "canceled", "cancelled"}:
        return
    ended_at = current_timestamp()
    metadata.update(status="canceled" if "cancel" in error.lower() else "failed", error=error, updated_at=ended_at)
    metadata.setdefault("processing", {}).update(
        ended_at=ended_at,
        elapsed_ms=round(elapsed * 1000),
        elapsed_seconds=round(elapsed, 3),
        elapsed_human=format_elapsed_duration(elapsed),
    )
    write_metadata_file(str(metadata_path), metadata)


def _stream_generation_request(
    request: Mapping[str, Any],
    *,
    use_subprocess: bool,
    gr_progress: Any = None,
    process_kind: str = "generation",
):
    started = time.perf_counter()
    task_folder = Path(str(request["task_layout"]["task_folder"]))
    if not use_subprocess:
        LAZY_ENGINE.reset_cancel(task_id=str(task_folder.resolve()))
    log_path = task_folder / "generation.log"
    progress_file = request.get("progress_file")
    initial = (
        progress_panel_html({}, title="Starting generation"),
        "Starting generation...",
        "",
        gr.skip(),
        gr.skip(),
        [],
        gr.skip(),
        "",
        recent_outputs(),
    )
    yield initial
    if use_subprocess:
        result_path = task_folder / "result.json"
        command = [
            sys.executable,
            str(ROOT / "webui_subprocess_worker.py"),
            "--request-file",
            str(task_folder / "request.json"),
            "--result-file",
            str(result_path),
        ]
        job = PROCESS_MANAGER.start(
            process_kind,
            command,
            state_dir=task_folder,
            log_path=log_path,
            cwd=ROOT,
            metadata={
                "metadata_path": request["metadata_path"],
                "result_path": str(result_path),
                "progress_file": str(progress_file or ""),
            },
        )
        while job.running:
            payload = read_progress_file(progress_file) or {}
            yield (
                progress_panel_html(payload, title="Generating voice"),
                str(payload.get("desc") or payload.get("stage") or "Model is working..."),
                tail_text(log_path, 60),
                gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
            )
            time.sleep(0.5)
        payload = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
        if job.canceled:
            _record_generation_failure(request, "Generation canceled by user.", time.monotonic() - job.started_at)
            yield _terminal_generation_updates(
                request,
                title="Canceled",
                message="Generation canceled by user.",
            )
            return
        if job.process.returncode != 0 or payload.get("status") != "ok":
            raise RuntimeError(payload.get("error") or f"Generation worker exited with code {job.process.returncode}")
        yield _result_updates(payload, request)
        return

    result_box: dict[str, Any] = {}

    def run_in_process() -> None:
        tee = _Tee(sys.stdout, log_path)
        try:
            # Redirect only this worker's writes while keeping the real console live.
            import contextlib

            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee), LAZY_ENGINE.in_use():
                engine = LAZY_ENGINE.get(
                    request["runtime"],
                    progress_file=progress_file,
                    progress_callback=gr_progress,
                )
                result_box["result"] = run_generation_request(
                    dict(request), engine, progress_callback=gr_progress,
                    cancellation_check=LAZY_ENGINE.raise_if_canceled,
                )
        except BaseException as exc:
            result_box["error"] = exc
            result_box["traceback"] = traceback.format_exc()
            if is_cancellation(exc):
                print(">> Generation canceled by user; the in-process worker stopped.", flush=True)
            else:
                print(result_box["traceback"], file=sys.stderr, flush=True)
        finally:
            tee.close()
            result_box["done"] = True

    thread = threading.Thread(target=run_in_process, daemon=True, name="indextts-inprocess-generation")
    thread.start()
    try:
        while not result_box.get("done"):
            payload = read_progress_file(progress_file) or {}
            yield (
                progress_panel_html(payload, title="Generating voice in process"),
                str(payload.get("desc") or payload.get("stage") or "Model is working..."),
                tail_text(log_path, 60),
                gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
            )
            time.sleep(0.5)
    finally:
        # Keep the shared generation queue occupied if the stream disconnects;
        # reloading the browser must not race a second request against this one.
        thread.join()
        if "error" in result_box:
            _record_generation_failure(request, str(result_box["error"]), time.perf_counter() - started)
    if "error" in result_box:
        error = result_box["error"]
        if is_cancellation(error):
            raise GenerationCanceled(str(error))
        raise RuntimeError(str(error))
    yield _result_updates(result_box["result"], request)


def _register(
    registry: PresetRegistry,
    key: str,
    component: Any,
    *,
    kind: str = "auto",
    choices: Sequence[Any] | None = None,
    minimum: float | int | None = None,
    maximum: float | int | None = None,
    nullable: bool = False,
) -> Any:
    return registry.register(
        key,
        component,
        GENERATION_DEFAULTS[key],
        kind=kind,
        choices=choices,
        minimum=minimum,
        maximum=maximum,
        nullable=nullable,
    )


@dataclass
class GenerationTab:
    controls: dict[str, Any] = field(default_factory=dict)
    prompt_audio: Any = None
    reference_media: Any = None
    reference_recording: Any = None
    reference_video: Any = None
    reference_ranges: Any = None
    reference_audio_dropdown: Any = None
    reference_source: Any = None
    reference_status: Any = None
    text: Any = None
    subtitle_file: Any = None
    image: Any = None
    emotion_audio: Any = None
    generate_button: Any = None
    cancel_button: Any = None
    cancel_panel: Any = None
    cancel_yes: Any = None
    cancel_no: Any = None
    cancel_target: Any = None
    progress_html: Any = None
    status: Any = None
    log_tail: Any = None
    output_audio: Any = None
    output_video: Any = None
    candidate_state: Any = None
    caption_status: Any = None
    final_summary: Any = None
    recent_table: Any = None
    task_state: Any = None
    task_timer: Any = None
    request_keys: list[str] = field(default_factory=list)
    request_components: list[Any] = field(default_factory=list)


def build_generation_tab(
    args: Any,
    registry: PresetRegistry,
    *,
    load_hook: Any | None = None,
) -> GenerationTab:
    model_dir = str(getattr(args, "model_dir", ROOT / "models"))
    initial_reference_choices = reference_audio_choices()
    tab = GenerationTab()
    c = tab.controls

    with gr.Tab("Voice Generation", id="voice-generation"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### Reference Voice")
                with gr.Group(elem_classes=["reference-voice-field"]):
                    tab.reference_media = gr.File(
                        label="Reference Voice (audio or video)",
                        file_types=["audio", "video", *SUPPORTED_MEDIA_EXTENSIONS],
                        type="filepath",
                        height=130,
                    )
                    tab.prompt_audio = gr.Audio(
                        label="Reference Voice audio preview",
                        type="filepath",
                        format="wav",
                        interactive=False,
                        visible=False,
                        show_label=False,
                        container=False,
                        buttons=["download"],
                    )
                    tab.reference_video = gr.Video(
                        label="Reference Voice video preview",
                        include_audio=True,
                        interactive=False,
                        visible=False,
                        show_label=False,
                        container=False,
                        buttons=["download"],
                    )
                with gr.Accordion("Record from microphone", open=False):
                    tab.reference_recording = gr.Audio(
                        label="Microphone reference",
                        sources=["microphone"],
                        type="filepath",
                        format="wav",
                    )
                gr.Markdown(
                    "The selected or automatically resolved file always appears in Reference Voice. "
                    "Audio and video are decoded by FFmpeg, and the audio preview is exactly what generation uses.",
                    elem_classes=["section-note"],
                )
                tab.reference_ranges = gr.Textbox(
                    label="Time ranges",
                    placeholder="1:4; 7.5:12 or 01:02-01:08",
                    info="Optional ranges are joined in order and applied automatically before generation.",
                )
                ranges = tab.reference_ranges
                with gr.Row():
                    extract_button = gr.Button("✂️  Extract ranges", elem_classes=btn("teal"))
                    clear_reference = gr.Button("⌫  Clear", elem_classes=btn("orange"))
                path_input = gr.Textbox(
                    label="Reference media path",
                    info="Load any local audio or video path without uploading it.",
                )
                with gr.Row():
                    tab.reference_audio_dropdown = gr.Dropdown(
                        choices=initial_reference_choices,
                        value=None,
                        label="Reference audio library",
                        info="Manual selection wins. Otherwise a LoRA / DoRA reference is preferred, then the latest modified audio in reference_audios.",
                        scale=8,
                    )
                    refresh_reference_audios = gr.Button(
                        "🔄  Refresh",
                        elem_classes=btn("green"),
                        scale=1,
                    )
                load_path = gr.Button("📂  Load path", elem_classes=btn("sky"))
                tab.reference_source = gr.State("empty")
                library_note = (
                    f"Found {len(initial_reference_choices)} file(s) in `reference_audios`. "
                    "Nothing is loaded at startup; Generate voice selects the latest modified fallback when needed."
                    if initial_reference_choices
                    else "`reference_audios` is ready but empty. Add audio there, then refresh the library."
                )
                tab.reference_status = gr.Markdown(library_note, elem_classes=["section-note"])
                reference_status = tab.reference_status
                with gr.Accordion("Reference audio tips", open=False):
                    gr.Markdown(
                        "Choose a quiet 3-15 second clip with one speaker, natural pacing, no music, and little room echo. "
                        "A representative emotional tone is more useful than an unusually dramatic take."
                    )

            with gr.Column(scale=2, min_width=420):
                gr.Markdown("### Text & Timing")
                tab.text = gr.Textbox(
                    label="Text",
                    lines=10,
                    max_lines=24,
                    placeholder="Write the speech here. Add [pause:500ms] where a precise pause is needed.",
                    buttons=["copy"],
                    info="Text to synthesize; long text is split by the shared language-aware segmenter.",
                )
                with gr.Group(elem_classes=["timing-panel"]):
                    with gr.Row(equal_height=False):
                        language = gr.Dropdown(
                            choices=list(LANGUAGES), value="EN", label="Language",
                            info="Normalization and pronunciation.", scale=1, min_width=100,
                        )
                        max_tokens = gr.Slider(
                            20, 300, value=60, step=1, label="Max tokens per segment",
                            info="Hard limit per speech segment; longer segments need more VRAM.",
                            scale=2, min_width=180,
                        )
                        with gr.Column(scale=1, min_width=150):
                            auto_lora_tokens = gr.Checkbox(
                                value=GENERATION_DEFAULTS["generation.auto_lora_max_tokens"],
                                label="Auto from LoRA / DoRA dataset",
                                info="Limit from the voice's training clips.",
                            )
                            auto_tokens = gr.Button("✨  Language default", elem_classes=btn("lime"), size="sm")
                    segmentation_mode = gr.Radio(
                        choices=list(SEGMENTATION_CHOICES),
                        value=GENERATION_DEFAULTS["generation.segmentation_mode"],
                        label="Text splitting",
                        info="Smart sentences packs whole sentences to the voice's typical clip length; Every sentence renders each separately. Both ignore line wrapping and subtitle cue breaks unless cue timing is enabled. Token budget cuts at any punctuation up to the limit.",
                    )
                    with gr.Row(equal_height=False):
                        sentence_pause = gr.Slider(
                            0, 2000, value=GENERATION_DEFAULTS["generation.sentence_pause_ms"], step=10, label="Sentence pause (ms)",
                            info="Pause between two sentences the splitter separated, measured word to word; 0 uses Section silence.",
                            scale=1, min_width=150,
                        )
                        max_pause = gr.Slider(
                            0, 2000, value=GENERATION_DEFAULTS["generation.max_pause_ms"], step=10, label="Maximum pause (ms)",
                            info="Longer pauses in the finished audio are shortened to this; 0 keeps the model's pauses. Tagged pauses and caption timing stay as written.",
                            scale=1, min_width=150,
                        )
                        auto_lora_pauses = gr.Checkbox(
                            value=GENERATION_DEFAULTS["generation.auto_lora_pauses"],
                            label="Auto pauses from LoRA / DoRA dataset",
                            info="Both values from the speaker's own recordings.",
                            scale=1, min_width=150,
                        )
                    gr.Markdown(
                        "Pause syntax: `[pause:500ms]`, `[pause:0.8s]`, or `<pause=0.5>`. Tagged pauses are inserted exactly as written.",
                        elem_classes=["section-note"],
                    )
                _register(registry, "generation.language", language, kind="choice", choices=LANGUAGES)
                _register(registry, "generation.max_text_tokens_per_segment", max_tokens, kind="int", minimum=20, maximum=300)
                _register(registry, "generation.auto_lora_max_tokens", auto_lora_tokens, kind="bool")
                _register(registry, "generation.segmentation_mode", segmentation_mode, kind="choice", choices=list(SEGMENTATION_MODES))
                _register(registry, "generation.sentence_pause_ms", sentence_pause, kind="int", minimum=0, maximum=2000)
                _register(registry, "generation.max_pause_ms", max_pause, kind="int", minimum=0, maximum=2000)
                _register(registry, "generation.auto_lora_pauses", auto_lora_pauses, kind="bool")

                with gr.Row():
                    tab.subtitle_file = gr.File(
                        label=f"Captions ({SUBTITLE_FORMAT_SUMMARY})",
                        file_types=list(SUPPORTED_SUBTITLE_EXTENSIONS),
                        type="filepath",
                    )
                    caption_timing = gr.Checkbox(
                        value=False,
                        label="Use caption cue timing",
                        info="Retimes each caption unit to its cue slot and preserves cue start times.",
                    )
                _register(registry, "generation.use_caption_timing", caption_timing, kind="bool")
                caption_load_status = gr.Markdown("")
                tab.image = gr.Image(
                    label="Still image for MP4",
                    type="filepath",
                    sources=["upload", "clipboard"],
                    height=180,
                    buttons=["fullscreen"],
                )
                gr.Markdown("Add a still image only when an MP4 output is needed.", elem_classes=["section-note"])
                preview_count = gr.Markdown("0 sections", elem_classes=["section-note"])
                segment_preview = gr.Dataframe(
                    headers=["#", "Type", "Text / pause", "Details"],
                    datatype=["number", "str", "str", "str"],
                    value=[],
                    type="array",
                    interactive=False,
                    wrap=True,
                    max_height=300,
                    label="Live section preview",
                    buttons=["fullscreen"],
                )
                with gr.Accordion("🔤 Pronunciation check & dictionary", open=False):
                    gr.Markdown(
                        "The engine reads `<word|PHONES>` annotations natively: ARPAbet phones with stress digits and dots between "
                        "syllables, for example `<Qwen|K W EH1 N>`. **Check unknown words** lists the words in the text that the "
                        "selected voice never spoke in training and the base model has no dictionary reading for, with a proposed "
                        "reading. **Add suggestions and save** stores the ones listed; edits in the dictionary table are saved as you "
                        "make them. Entries with scope `unseen` never override a word the voice learned from its recordings; `always` "
                        "applies everywhere. A plain respelling (`Comfy U I`) is also accepted and replaces the word as text.",
                        elem_classes=["section-note"],
                    )
                    with gr.Row():
                        apply_pronunciation = gr.Checkbox(
                            value=GENERATION_DEFAULTS["generation.apply_pronunciation_dictionary"],
                            label="Apply the pronunciation dictionary when generating",
                            info="Rewrites dictionary words before synthesis and in the live preview; also used by Batch Generation.",
                            scale=3,
                        )
                        check_words = gr.Button("🔍  Check unknown words", elem_classes=btn("olive"), scale=1)
                    unknown_words = gr.Dataframe(
                        headers=["Word", "Suggested reading", "Kind", "How it was derived", "Confidence", "Tokenizer pieces"],
                        datatype=["str", "str", "str", "str", "str", "number"],
                        value=[],
                        type="array",
                        interactive=False,
                        wrap=True,
                        max_height=220,
                        label="Words without a known reading",
                        buttons=["fullscreen"],
                    )
                    pronunciation_status = gr.Markdown("", elem_classes=["section-note"])
                    add_suggestions = gr.Button("➕  Add suggestions and save", elem_classes=btn("coral"))
                    dictionary_table = gr.Dataframe(
                        headers=["Word", "Reading (ARPAbet phones or respelling)", "Kind", "Scope (unseen / always)", "Source"],
                        datatype=["str", "str", "str", "str", "str"],
                        value=dictionary_rows(pronunciation_entries()),
                        type="array",
                        interactive=True,
                        wrap=True,
                        max_height=280,
                        label="Pronunciation dictionary (pronunciations/dictionary.json)",
                        buttons=["fullscreen"],
                    )
                _register(registry, "generation.apply_pronunciation_dictionary", apply_pronunciation, kind="bool")

            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### Run")
                with gr.Row():
                    tab.generate_button = gr.Button(
                        "🎙️  Generate voice", variant="primary", elem_classes=btn("emerald"), scale=3,
                    )
                    tab.cancel_button = gr.Button(
                        "⛔  Cancel", variant="stop", elem_classes=btn("red"), scale=1,
                    )
                with gr.Group(visible=False) as tab.cancel_panel:
                    tab.cancel_target = gr.State("")
                    gr.Markdown("Cancel the running generation? Completed files are kept.")
                    with gr.Row():
                        tab.cancel_yes = gr.Button("🛑  Yes, cancel generation", variant="stop", elem_classes=btn("crimson"))
                        tab.cancel_no = gr.Button("▶️  Keep generating", elem_classes=btn("pink"))
                open_outputs = gr.Button("📁  Open outputs folder", elem_classes=btn("indigo"))
                tab.output_audio = gr.Audio(
                    label="Generated audio",
                    type="filepath",
                    visible=False,
                    buttons=["download"],
                )
                tab.progress_html = gr.HTML(progress_panel_html({}, title="Ready"))
                tab.status = gr.Markdown("")
                tab.log_tail = gr.Textbox(
                    label="Live log (last 60 lines)", lines=10, max_lines=16,
                    interactive=False, buttons=["copy"], elem_classes=["log-tail"],
                )
                tab.final_summary = gr.HTML("")
                tab.task_state = gr.State("")
                tab.task_timer = gr.Timer(5.0, active=True)

        gr.Markdown("### Voice LoRA / DoRA")
        # Three rows of like with like: the two adapter pickers (one Refresh reloads
        # both), the two strengths side by side, then the four automation switches at
        # equal width so the long notes wrap in wide columns instead of tall slivers.
        with gr.Row():
            lora = gr.Dropdown(
                choices=_lora_choices(),
                value="",
                label="LoRA / DoRA",
                info="Select a trained LoRA / DoRA, or None for Base model (no LoRA / DoRA), which clones from the reference only.",
                scale=6,
            )
            use_decoder = gr.Dropdown(
                choices=decoder_adapter_choices("", ROOT / "loras"),
                value="auto",
                label="Voice decoder adapter",
                info="Selecting a LoRA / DoRA picks the decoder adapter saved with it. Choose None to hear the GPT adapter alone, or any other decoder adapter file.",
                scale=6,
            )
            refresh_lora = gr.Button("↻  Refresh", elem_classes=btn("violet"), scale=1)
        with gr.Row(equal_height=True):
            strength = gr.Slider(
                0.0, 2.0, value=1.0, step=0.05,
                label="LoRA / DoRA strength",
                info="1.0 is the trained strength; lower is subtler and higher is stronger.",
            )
            decoder_strength = gr.Slider(
                0.0, 2.0, value=1.0, step=0.05,
                label="Voice decoder adapter strength",
                info="1.0 is the trained strength of the decoder adapter; it is independent of the LoRA / DoRA strength.",
            )
        with gr.Row(equal_height=True):
            auto_ref = gr.Checkbox(
                value=GENERATION_DEFAULTS["generation.auto_lora_reference"],
                label="Auto-load the LoRA / DoRA recommended reference audio",
                info="Loads the LoRA / DoRA's saved reference whenever no manual Reference Voice is selected.",
            )
            auto_rate = gr.Checkbox(
                value=GENERATION_DEFAULTS["generation.auto_lora_speaking_rate"],
                label="Auto-apply the LoRA / DoRA calibrated speaking rate and decoding settings",
                info="Uses the selected voice's measured pace and the temperature, guidance, and beams its decoding sweep adopted; selecting None restores the defaults.",
            )
            auto_emotion = gr.Checkbox(
                value=GENERATION_DEFAULTS["generation.auto_lora_emotion_reference"],
                label="Use the LoRA / DoRA expressive clip as the emotion prompt",
                info="While Emotion source is Same as speaker voice, the liveliest clean training clip saved with the adapter drives the delivery (Emotion weight applies) and the speaker reference keeps the identity. Measured: fewer word errors and more pitch movement at unchanged identity; blind listening rounds were split (a hand-picked clip was preferred, the automatically chosen one slightly not), so listen and uncheck if you prefer the plain prompt.",
            )
            merge_lora = gr.Checkbox(
                value=False,
                label="Merge LoRA / DoRA into base weights for speed (BF16 only)",
                info="Temporarily folds the selected LoRA / DoRA into floating GPT weights and restores them before switching.",
            )
        lora_info = gr.HTML(_lora_info("")[0])
        with gr.Row(equal_height=True):
            lora_saved_rate = gr.Number(
                value=None, step=0.01, precision=3,
                label="Saved speaking rate for this LoRA / DoRA",
                info="The pace multiplier stored with the selected adapter (0.5 to 1.5). Edit it and press Save to override the automatic estimate; auto-apply then uses your value.",
                scale=4,
            )
            save_lora_rate = gr.Button("⏱️  Save speaking rate", elem_classes=btn("purple"), scale=1)
            pick_expressive = gr.Button("🎭  Pick expressive clip", elem_classes=btn("mint"), scale=1)
        registry.register("runtime.lora_path", lora, "", kind="str")
        registry.register("runtime.lora_strength", strength, 1.0, kind="float", minimum=0.0, maximum=2.0)
        registry.register("runtime.lora_merge_into_base", merge_lora, False, kind="bool")
        registry.register("runtime.decoder_adapter", use_decoder, "auto", kind="str")
        registry.register("runtime.decoder_adapter_strength", decoder_strength, 1.0, kind="float", minimum=0.0, maximum=2.0)
        _register(registry, "generation.auto_lora_reference", auto_ref, kind="bool")
        _register(
            registry,
            "generation.auto_lora_speaking_rate",
            auto_rate,
            kind="bool",
        )
        _register(registry, "generation.auto_lora_emotion_reference", auto_emotion, kind="bool")

        with gr.Accordion("Emotion Control", open=False):
            emotion_mode = gr.Radio(
                choices=list(EMOTION_MODES), value=EMOTION_MODES[0], label="Emotion source",
                info="Use the speaker tone, another reference, eight manual vectors, or emotion text analysis.",
            )
            _register(registry, "generation.emotion_mode", emotion_mode, kind="choice", choices=EMOTION_MODES)
            with gr.Row():
                tab.emotion_audio = gr.Audio(
                    label="Emotion reference audio", sources=["upload", "microphone"], type="filepath",
                )
                emotion_weight = gr.Slider(
                    0.0, 1.0, value=0.65, step=0.05, label="Emotion weight",
                    info="0 keeps more speaker emotion; 1 follows the selected emotion source fully.",
                )
                emotion_random = gr.Checkbox(
                    value=False, label="Random emotion exemplar",
                    info="Randomizes the internal exemplar used with manual emotion vectors.",
                )
            gr.Markdown("Emotion reference mode transfers delivery from a clean clip while keeping the speaker identity separate.", elem_classes=["section-note"])
            _register(registry, "generation.emotion_weight", emotion_weight, kind="float", minimum=0, maximum=1)
            _register(registry, "generation.emotion_random", emotion_random, kind="bool")
            emotion_text = gr.Textbox(
                label="Emotion description",
                placeholder="Warm, quietly confident, and reassuring",
                info="Used only in Emotion text mode; blank analyzes the speech text itself.",
            )
            _register(registry, "generation.emotion_text", emotion_text, kind="str")
            with gr.Row():
                for name, label in zip(EMOTION_NAMES[:4], EMOTION_LABELS[:4]):
                    component = gr.Slider(0, 1, value=0, step=0.05, label=label, info=f"Manual {label.lower()} strength.")
                    _register(registry, f"generation.emotion_{name}", component, kind="float", minimum=0, maximum=1)
            with gr.Row():
                for name, label in zip(EMOTION_NAMES[4:], EMOTION_LABELS[4:]):
                    component = gr.Slider(0, 1, value=0, step=0.05, label=label, info=f"Manual {label.lower()} strength.")
                    _register(registry, f"generation.emotion_{name}", component, kind="float", minimum=0, maximum=1)
            with gr.Accordion("Emotion vector limits and biases", open=False):
                with gr.Row():
                    apply_bias = gr.Checkbox(
                        value=True, label="Apply tuned emotion biases",
                        info="Recommended balancing prevents several emotion channels from dominating.",
                    )
                    max_sum = gr.Slider(
                        0.1, 2.0, value=0.8, step=0.05, label="Maximum vector sum",
                        info="0.8 is the model-tuned recommendation; larger values can sound exaggerated.",
                    )
                _register(registry, "generation.apply_emotion_bias", apply_bias, kind="bool")
                _register(registry, "generation.max_emotion_sum", max_sum, kind="float", minimum=0.1, maximum=2)
                with gr.Row():
                    for name, label, default in zip(EMOTION_NAMES[:4], EMOTION_LABELS[:4], EMOTION_BIAS_DEFAULTS[:4]):
                        component = gr.Slider(0.5, 1.5, value=default, step=0.0625, label=f"{label} bias", info="Multiplier applied before the vector sum limit.")
                        _register(registry, f"generation.emotion_bias_{name}", component, kind="float", minimum=0.5, maximum=1.5)
                with gr.Row():
                    for name, label, default in zip(EMOTION_NAMES[4:], EMOTION_LABELS[4:], EMOTION_BIAS_DEFAULTS[4:]):
                        component = gr.Slider(0.5, 1.5, value=default, step=0.0625, label=f"{label} bias", info="Multiplier applied before the vector sum limit.")
                        _register(registry, f"generation.emotion_bias_{name}", component, kind="float", minimum=0.5, maximum=1.5)

        with gr.Accordion("Sampling", open=False):
            with gr.Row():
                do_sample = gr.Checkbox(value=True, label="Sample", info="Recommended for natural variation; disable for greedy/beam decoding.")
                temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature", info="0.8 balances expressiveness and stability.")
                top_p = gr.Slider(0.0, 1.0, value=0.8, step=0.01, label="Top-p", info="Nucleus sampling threshold; 0.8 is recommended.")
                top_k = gr.Slider(0, 100, value=30, step=1, label="Top-k", info="Candidate token cutoff; 0 disables top-k filtering.")
            with gr.Row():
                beams = gr.Slider(1, 10, value=3, step=1, label="Beams", info="More beams can improve stability but increase time and VRAM.")
                repetition = gr.Slider(1, 20, value=10.0, step=0.1, label="Repetition penalty", info="10 is the established model default. Above about 1.3 it works as a ban on every code the segment has already used.")
                repetition_window = gr.Slider(
                    0, 256, value=GENERATION_DEFAULTS["generation.repetition_window"], step=1, label="Repetition window (codes)",
                    info="0 applies the penalty to the whole segment (the model default). Otherwise only the last N generated codes are penalized, so a stuck loop is still stopped while sounds from earlier in the segment may return; 25 codes are about one second.",
                )
                length = gr.Slider(-2, 2, value=0, step=0.05, label="Length penalty", info="Only affects beam search; 0 is neutral.")
                max_mel = gr.Slider(50, 1815, value=1500, step=5, label="Max mel tokens", info="Upper limit on generated semantic tokens per section.")
            with gr.Row():
                seed = gr.Number(value=-1, precision=0, label="Seed", info="-1 chooses a fresh random seed; reuse a shown seed for repeatability.")
                candidates = gr.Slider(1, 8, value=1, step=1, label="Candidates", info="Generates consecutive seeded alternatives; each adds generation time.")
            for key, component, kind, minimum, maximum in (
                ("generation.do_sample", do_sample, "bool", None, None),
                ("generation.temperature", temperature, "float", 0.1, 2),
                ("generation.top_p", top_p, "float", 0, 1),
                ("generation.top_k", top_k, "int", 0, 100),
                ("generation.num_beams", beams, "int", 1, 10),
                ("generation.repetition_penalty", repetition, "float", 1, 20),
                ("generation.repetition_window", repetition_window, "int", 0, 256),
                ("generation.length_penalty", length, "float", -2, 2),
                ("generation.max_mel_tokens", max_mel, "int", 50, 1815),
                ("generation.seed", seed, "int", -1, 4294967295),
                ("generation.num_candidates", candidates, "int", 1, 8),
            ):
                _register(registry, key, component, kind=kind, minimum=minimum, maximum=maximum)

        with gr.Accordion("Diffusion / CFM", open=False):
            with gr.Row():
                steps = gr.Slider(2, 100, value=25, step=1, label="Diffusion steps", info="25 is the quality default; 12-16 is faster and 35-50 can refine difficult audio.")
                cfg = gr.Slider(0, 2, value=0.7, step=0.05, label="CFG rate", info="0.7 is recommended; high values follow conditioning more aggressively.")
                cfm_temp = gr.Slider(0, 2, value=1.0, step=0.05, label="CFM temperature", info="1.0 is the best-quality default; lower values reduce diffusion variation.")
                cfm_cache = gr.Slider(1024, 32768, value=8192, step=256, label="CFM cache length", info="8192 fits typical sections; lower values reduce reserved VRAM.")
            _register(registry, "generation.diffusion_steps", steps, kind="int", minimum=2, maximum=100)
            _register(registry, "generation.inference_cfg_rate", cfg, kind="float", minimum=0, maximum=2)
            _register(registry, "generation.cfm_temperature", cfm_temp, kind="float", minimum=0, maximum=2)
            _register(registry, "generation.cfm_cache_length", cfm_cache, kind="int", minimum=1024, maximum=32768)

        with gr.Accordion("Segmentation & Timing", open=False):
            with gr.Row():
                budget_scale = gr.Slider(0.3, 1.0, value=0.72, step=0.01, label="Non-CJK token budget scale", info="0.72 leaves room for subword expansion in English, Arabic, and Spanish.")
                interval = gr.Slider(0, 2000, value=200, step=10, label="Section silence (ms)", info="Silence inserted between generated text sections; cue timing overrides this to zero.")
                max_silence = gr.Slider(
                    0, 200, value=0, step=1, label="Max consecutive silence tokens",
                    info="Trims runs of the codec's silence token (IndexTTS 2.0 codec only; the 2.5 codec never repeats a code, so the control is hidden there). Use Maximum pause instead.",
                    visible=codec_has_silence_runs(model_dir),
                )
                latent = gr.Slider(0.5, 3.0, value=1.72, step=0.01, label="Latent multiplier", info="1.72 is natural duration; the runner converts this to the engine duration factor.")
                speaking_rate = gr.Slider(
                    0.5,
                    1.5,
                    value=GENERATION_DEFAULTS["generation.speaking_rate"],
                    step=0.01,
                    label="Speaking rate",
                    info="1.0 is the model's natural pace; below 1.0 speaks slower, above 1.0 faster. A trained LoRA / DoRA can carry a calibrated value that matches the speaker's real pace.",
                )
            with gr.Row():
                target_duration = gr.Number(value=None, minimum=0.1, maximum=3600, step=0.1, label="Target duration (seconds)", info="Leave blank unless a whole-output duration target is needed.")
                target_mode = gr.Dropdown(choices=["off", "natural", "pad", "trim"], value="off", label="Target duration mode", info="Natural regenerates timing; pad/trim only adjust the assembled result.")
                pause_tags = gr.Checkbox(value=True, label="Enable pause tags", info="Parses inline pause tags before tokenization.")
                normalization = gr.Checkbox(value=True, label="Text normalization", info="Recommended: expands and normalizes text before phonetic processing.")
            with gr.Row():
                speech_recovery = gr.Checkbox(
                    value=GENERATION_DEFAULTS["generation.auto_retry_incomplete_speech"],
                    label="Automatically retry incomplete speech",
                    info="Retries unfinished sections before saving audio. When off, incomplete speech fails immediately.",
                )
                speech_retries = gr.Slider(
                    0, 64, value=GENERATION_DEFAULTS["generation.max_speech_retries"], step=1,
                    label="Maximum speech retries",
                    info="Total additional attempts per original section. 0 disables retries; the selected limit is used exactly.",
                )
                speech_split_depth = gr.Slider(
                    0, 8, value=GENERATION_DEFAULTS["generation.max_speech_split_depth"], step=1,
                    label="Maximum recovery split depth",
                    info="How many times an unfinished section may be split at word or clause boundaries. 0 retries without splitting.",
                )
            _register(registry, "generation.segment_budget_scale_non_cjk", budget_scale, kind="float", minimum=0.3, maximum=1)
            _register(registry, "generation.interval_silence", interval, kind="int", minimum=0, maximum=2000)
            _register(registry, "generation.max_consecutive_silence", max_silence, kind="int", minimum=0, maximum=200)
            _register(registry, "generation.latent_multiplier", latent, kind="float", minimum=0.5, maximum=3)
            _register(
                registry,
                "generation.speaking_rate",
                speaking_rate,
                kind="float",
                minimum=0.5,
                maximum=1.5,
            )
            _register(registry, "generation.target_duration_s", target_duration, kind="float", minimum=0.1, maximum=3600, nullable=True)
            _register(registry, "generation.target_duration_mode", target_mode, kind="choice", choices=["off", "natural", "pad", "trim"])
            _register(registry, "generation.enable_pause_tags", pause_tags, kind="bool")
            _register(registry, "generation.text_normalization", normalization, kind="bool")
            _register(registry, "generation.auto_retry_incomplete_speech", speech_recovery, kind="bool")
            _register(registry, "generation.max_speech_retries", speech_retries, kind="int", minimum=0, maximum=64)
            _register(registry, "generation.max_speech_split_depth", speech_split_depth, kind="int", minimum=0, maximum=8)

        with gr.Accordion("Reference Processing", open=False):
            with gr.Row():
                max_spk = gr.Slider(3, 90, value=15, step=1, label="Maximum speaker audio length (s)", info="15 seconds preserves enough identity without wasting reference compute.")
                max_emo = gr.Slider(3, 90, value=15, step=1, label="Maximum emotion audio length (s)", info="15 seconds is recommended for an emotion reference.")
                semantic = gr.Slider(1, 24, value=17, step=1, label="Semantic layer", info="Layer 17 is trained and recommended; changing it alters reference embeddings.")
                reuse_spk = gr.Checkbox(value=False, label="Reuse speaker conditioning for emotion", info="Faster default-emotion path; enable when no separate emotion source is used.")
            _register(registry, "generation.max_speaker_audio_length", max_spk, kind="float", minimum=3, maximum=90)
            _register(registry, "generation.max_emotion_audio_length", max_emo, kind="float", minimum=3, maximum=90)
            _register(registry, "generation.semantic_layer", semantic, kind="int", minimum=1, maximum=24)
            _register(registry, "generation.reuse_spk_cond_for_emo", reuse_spk, kind="bool")

        with gr.Accordion("Output", open=False):
            with gr.Row():
                filename = gr.Textbox(label="Output filename", info="Optional safe basename; task numbering is used when blank.")
                save_ref = gr.Checkbox(value=False, label="Save used reference", info="Copies the active Reference Voice into the task folder for reproducibility.")
                save_mp3 = gr.Checkbox(value=False, label="Save MP3", info="Converts the final output to MP3; WAV candidates remain available.")
                bitrate = gr.Dropdown(choices=["128k", "192k", "256k", "320k"], value="256k", label="MP3 bitrate", info="256k is a strong quality/size balance for voice.")
            with gr.Row():
                tuning = gr.Dropdown(choices=["bypass", "voice_clarity", "clear_narration", "deharsh", "warm", "normalize"], value="bypass", label="Audio tuning preset", info="Bypass preserves model audio exactly; other presets use FFmpeg post-processing.")
                trim_ms = gr.Slider(0, 3000, value=0, step=10, label="Trim edge silence threshold (ms)", info="0 disables trimming; only edge silence at least this long is removed.")
            with gr.Accordion("Audio tuning overrides", open=False):
                with gr.Row():
                    low_cut = gr.Number(value=None, minimum=20, maximum=500, label="Low cut (Hz)", info="Optional high-pass cutoff; leave blank to use the preset.")
                    high_cut = gr.Number(value=None, minimum=1000, maximum=24000, label="High cut (Hz)", info="Optional low-pass cutoff; leave blank to use the preset.")
                    gain = gr.Number(value=None, minimum=-24, maximum=24, label="Gain (dB)", info="Optional final gain before limiting.")
                    loudness = gr.Number(value=None, minimum=-30, maximum=-5, label="Loudness target (LUFS)", info="Optional integrated loudness normalization target.")
                    deess = gr.Number(value=None, minimum=0, maximum=12, label="De-ess amount", info="Optional attenuation around sibilance frequencies.")
            for key, component, kind, choices, minimum, maximum, nullable in (
                ("generation.output_filename", filename, "str", None, None, None, False),
                ("generation.save_used_audio", save_ref, "bool", None, None, None, False),
                ("generation.save_as_mp3", save_mp3, "bool", None, None, None, False),
                ("generation.mp3_bitrate", bitrate, "choice", ["128k", "192k", "256k", "320k"], None, None, False),
                ("generation.audio_tuning_preset", tuning, "choice", ["bypass", "voice_clarity", "clear_narration", "deharsh", "warm", "normalize"], None, None, False),
                ("generation.trim_silence_ms_threshold", trim_ms, "int", None, 0, 3000, False),
                ("generation.tuning_low_cut_hz", low_cut, "float", None, 20, 500, True),
                ("generation.tuning_high_cut_hz", high_cut, "float", None, 1000, 24000, True),
                ("generation.tuning_gain_db", gain, "float", None, -24, 24, True),
                ("generation.tuning_loudnorm_i", loudness, "float", None, -30, -5, True),
                ("generation.tuning_deess", deess, "float", None, 0, 12, True),
            ):
                _register(registry, key, component, kind=kind, choices=choices, minimum=minimum, maximum=maximum, nullable=nullable)

        with gr.Accordion("Execution", open=False):
            with gr.Row():
                use_subprocess = gr.Checkbox(value=False, label="Use isolated subprocess", info="Recommended: cancellation can terminate the complete model process and release VRAM.")
                batch_size = gr.Slider(1, 16, value=1, step=1, label="Section batch size", info="1 is safest; use the active VRAM tier hint before increasing this.")
                low_memory = gr.Checkbox(value=False, label="Low memory mode", info="Uses sequential paths and aggressive memory behavior for constrained GPUs.")
                prevent = gr.Checkbox(value=False, label="Prevent VRAM accumulation", info="Clears autoregressive caches between segments; slower but useful for long jobs.")
                verbose = gr.Checkbox(value=bool(getattr(args, "verbose", False)), label="Verbose logging", info="Prints detailed model inputs and timing diagnostics to console and the live log.")
            _register(registry, "generation.use_subprocess", use_subprocess, kind="bool")
            _register(registry, "generation.section_batch_size", batch_size, kind="int", minimum=1, maximum=16)
            _register(registry, "generation.low_memory_mode", low_memory, kind="bool")
            _register(registry, "generation.prevent_vram_accumulation", prevent, kind="bool")
            registry.register("generation.verbose", verbose, bool(getattr(args, "verbose", False)), kind="bool")

        gr.Markdown("### Outputs")
        with gr.Row(equal_height=False):
            tab.output_video = gr.Video(label="Generated MP4", visible=False, buttons=["download"])
        # A queued generation or header restore updates this textbox. Unlike
        # gr.State, its change event reliably retriggers gr.render after reload.
        tab.candidate_state = gr.Textbox(
            value="", visible=False, label="Candidate audio paths"
        )
        with gr.Column(elem_classes=["candidate-list"]):
            @gr.render(inputs=tab.candidate_state, triggers=[tab.candidate_state.change])
            def render_candidates(paths: str | None):
                candidates_value = _candidate_paths_from_state(paths)
                if len(candidates_value) <= 1:
                    return
                gr.Markdown(f"#### Candidates ({len(candidates_value)})")
                for index, path in enumerate(candidates_value, start=1):
                    gr.Audio(value=path, label=f"Candidate {index}", type="filepath", buttons=["download"], key=f"candidate-{index}-{path}")
        tab.caption_status = gr.Markdown("", visible=False)
        tab.recent_table = gr.Dataframe(
            headers=["Task", "Created", "Status", "Audio", "Folder"],
            datatype=["str", "str", "str", "str", "str"],
            value=[], type="array", interactive=False, wrap=True,
            label="Recent outputs (last 10)", max_height=300, buttons=["fullscreen"],
        )
        recent_audio = gr.State("")
        load_recent_reference = gr.Button("🎯  Load selected output into reference", elem_classes=btn("fuchsia"))

        def select_recent(evt: gr.SelectData):
            row = list(evt.row_value or [])
            audio_path = str(row[3]) if len(row) > 3 else ""
            if not audio_path or not Path(audio_path).is_file():
                return gr.skip(), "", "The selected row has no playable audio."
            return gr.update(value=audio_path, visible=True), audio_path, f"Previewing recent output: {Path(audio_path).name}"

        tab.recent_table.select(
            select_recent,
            outputs=[tab.output_audio, recent_audio, tab.status],
            queue=False,
            show_progress="hidden",
        )
        def use_recent_reference(path: str | None):
            if path and Path(path).is_file():
                return (
                    gr.update(value=path, visible=True),
                    gr.update(value=path),
                    gr.update(value=None, visible=False),
                    f"Loaded recent output as a manual reference: {Path(path).name}",
                    "manual",
                )
            return (
                gr.skip(),
                gr.skip(),
                gr.skip(),
                "Select a recent output first.",
                gr.skip(),
            )

        load_recent_reference.click(
            use_recent_reference,
            recent_audio,
            [
                tab.prompt_audio,
                tab.reference_media,
                tab.reference_video,
                reference_status,
                tab.reference_source,
            ],
            queue=False,
        )

    # Capture the complete generation/runtime request surface after Models adds runtime controls.
    c.update({spec.key: spec.component for spec in registry.specs if spec.component is not None and spec.key.startswith("generation.")})
    c["runtime.lora_path"] = lora
    c["runtime.lora_strength"] = strength
    c["runtime.lora_merge_into_base"] = merge_lora
    c["runtime.decoder_adapter"] = use_decoder
    c["runtime.decoder_adapter_strength"] = decoder_strength

    def media_updates(path: Any, value_ranges: str, *, require_ranges: bool = False):
        output, video, message = load_reference_media(
            path,
            value_ranges,
            require_ranges=require_ranges,
        )
        if not output:
            gr.Warning(message, title="Reference Voice")
            return gr.skip(), gr.skip(), message, gr.skip()
        return (
            gr.update(value=output, visible=True),
            gr.update(value=video, visible=bool(video)),
            message,
            "manual_media",
        )

    def on_extract(
        active_audio: str | None,
        media_path: str | None,
        selected_library: str | None,
        current_source: str,
        value_ranges: str,
        lora_path: str,
        auto_lora_reference: bool,
    ):
        if not str(value_ranges or "").strip():
            message = "Enter one or more time ranges before extracting."
            gr.Warning(message, title="Reference Voice")
            return (gr.skip(),) * 4 + (message, gr.skip())
        try:
            prepared = prepare_reference_for_generation(
                active_audio,
                media_path,
                selected_library,
                current_source,
                value_ranges,
                lora_path,
                auto_lora_reference,
            )
        except ValueError as exc:
            message = str(exc)
            gr.Warning(message, title="Reference Voice")
            return (gr.skip(),) * 4 + (message, gr.skip())
        gr.Info(prepared.message, title="Reference Voice")
        return (
            gr.update(value=prepared.prompt, visible=True),
            gr.update(value=prepared.media),
            gr.update(value=prepared.video, visible=bool(prepared.video)),
            gr.update(choices=prepared.choices, value=prepared.library_value),
            prepared.message,
            prepared.source,
        )

    def on_load_path(path: str, value_ranges: str):
        source = _existing_path(path)
        output, video, message = load_reference_media(source, value_ranges)
        if not output:
            gr.Warning(message, title="Reference Voice")
            return gr.skip(), gr.skip(), gr.skip(), message, gr.skip()
        try:
            browser_source = _browser_safe_media_path(source)
        except OSError as exc:
            message = f"Reference media could not be staged for the browser: {exc}"
            gr.Warning(message, title="Reference Voice")
            return gr.skip(), gr.skip(), gr.skip(), message, gr.skip()
        gr.Info(message, title="Reference Voice")
        return (
            gr.update(value=browser_source),
            gr.update(value=output, visible=True),
            gr.update(value=video, visible=bool(video)),
            message,
            "manual_media",
        )

    def on_recording(path: str | None, value_ranges: str):
        source = _existing_path(path)
        output, video, message = load_reference_media(source, value_ranges)
        if not output:
            gr.Warning(message, title="Reference Voice")
            return (gr.skip(),) * 3 + (message, gr.skip())
        try:
            browser_source = _browser_safe_media_path(source)
        except OSError as exc:
            message = f"Microphone recording could not be staged for the browser: {exc}"
            gr.Warning(message, title="Reference Voice")
            return (gr.skip(),) * 3 + (message, gr.skip())
        gr.Info(message, title="Reference Voice")
        return (
            gr.update(value=browser_source),
            gr.update(value=output, visible=True),
            gr.update(value=video, visible=bool(video)),
            message,
            "manual_media",
        )

    def on_library_select(
        path: str | None,
        current_source: str,
        value_ranges: str,
    ):
        choices = reference_audio_choices()
        available = {_path_key(value): value for _, value in choices}
        selected = available.get(_path_key(path))
        if selected:
            output, _, extraction_message = load_reference_media(selected, value_ranges)
            if not output:
                gr.Warning(extraction_message, title="Reference Voice")
                return gr.skip(), gr.skip(), gr.skip(), extraction_message, gr.skip()
            message = f"Loaded {Path(selected).name} as a manually selected library Reference Voice."
            if value_ranges.strip():
                message += " The entered ranges were extracted and joined."
            return (
                gr.update(value=output, visible=True),
                gr.update(value=selected),
                gr.update(value=None, visible=False),
                message,
                "library_manual",
            )
        if str(current_source or "").startswith("library"):
            return (
                gr.update(value=None, visible=False),
                gr.update(value=None),
                gr.update(value=None, visible=False),
                "Reference library selection cleared.",
                "empty",
            )
        return (
            gr.skip(),
            gr.skip(),
            gr.skip(),
            "Reference library selection cleared; the active manual reference was kept.",
            gr.skip(),
        )

    def refresh_reference_library(selected_library: str | None, current_source: str):
        choices = reference_audio_choices()
        available = {_path_key(value): value for _, value in choices}
        selected = available.get(_path_key(selected_library))
        source = str(current_source or "empty")
        dropdown_update = gr.update(choices=choices, value=selected)
        if not choices and source.startswith("library"):
            message = "reference_audios is empty; the previous library Reference Voice was cleared."
            return (
                dropdown_update,
                gr.update(value=None, visible=False),
                gr.update(value=None),
                gr.update(value=None, visible=False),
                "empty",
                message,
            )
        if choices:
            newest = _latest_reference_choice(choices)
            message = (
                f"Found {len(choices)} audio file(s) in reference_audios. "
                f"Latest modified fallback: {Path(newest).name if newest else 'none'}."
            )
        else:
            message = "reference_audios is ready but empty."
        return dropdown_update, gr.skip(), gr.skip(), gr.skip(), gr.skip(), message

    def on_direct_reference_input(path: str | None):
        current = _existing_path(path)
        if current:
            return (
                gr.update(value=current),
                gr.update(value=None, visible=False),
                "manual",
                f"Loaded manual Reference Voice audio: {Path(current).name}",
            )
        return (
            gr.update(value=None),
            gr.update(value=None, visible=False),
            "empty",
            "Reference Voice cleared; an automatic reference will be chosen on generation.",
        )

    tab.reference_media.upload(
        media_updates,
        [tab.reference_media, ranges],
        [tab.prompt_audio, tab.reference_video, reference_status, tab.reference_source],
        queue=False,
        show_progress="minimal",
    )
    tab.reference_media.clear(
        lambda: (
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(value=None),
            "empty",
            "Reference Voice source cleared; an automatic reference will be chosen on generation.",
        ),
        outputs=[
            tab.prompt_audio,
            tab.reference_video,
            tab.reference_recording,
            tab.reference_source,
            reference_status,
        ],
        queue=False,
    )
    tab.reference_recording.stop_recording(
        on_recording,
        [tab.reference_recording, ranges],
        [
            tab.reference_media,
            tab.prompt_audio,
            tab.reference_video,
            reference_status,
            tab.reference_source,
        ],
        queue=False,
        show_progress="minimal",
    )
    extract_button.click(
        on_extract,
        [
            tab.prompt_audio,
            tab.reference_media,
            tab.reference_audio_dropdown,
            tab.reference_source,
            ranges,
            lora,
            auto_ref,
        ],
        [
            tab.prompt_audio,
            tab.reference_media,
            tab.reference_video,
            tab.reference_audio_dropdown,
            reference_status,
            tab.reference_source,
        ],
        queue=False,
    )
    load_path.click(
        on_load_path,
        [path_input, ranges],
        [
            tab.reference_media,
            tab.prompt_audio,
            tab.reference_video,
            reference_status,
            tab.reference_source,
        ],
        queue=False,
    )
    tab.reference_audio_dropdown.input(
        on_library_select,
        [tab.reference_audio_dropdown, tab.reference_source, ranges],
        [
            tab.prompt_audio,
            tab.reference_media,
            tab.reference_video,
            reference_status,
            tab.reference_source,
        ],
        queue=False,
    )
    refresh_reference_audios.click(
        refresh_reference_library,
        [tab.reference_audio_dropdown, tab.reference_source],
        [
            tab.reference_audio_dropdown,
            tab.prompt_audio,
            tab.reference_media,
            tab.reference_video,
            tab.reference_source,
            reference_status,
        ],
        queue=False,
    )
    tab.prompt_audio.input(
        on_direct_reference_input,
        tab.prompt_audio,
        [tab.reference_media, tab.reference_video, tab.reference_source, reference_status],
        queue=False,
        show_progress="hidden",
    )
    clear_reference.click(
        lambda: (
            None,
            None,
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            "",
            "",
            None,
            "Reference Voice cleared; automatic LoRA / DoRA or latest-library fallback is ready.",
            "empty",
        ),
        outputs=[
            tab.reference_media,
            tab.reference_recording,
            tab.prompt_audio,
            tab.reference_video,
            path_input,
            ranges,
            tab.reference_audio_dropdown,
            reference_status,
            tab.reference_source,
        ],
        queue=False,
    )

    refresh_lora.click(lambda: gr.update(choices=_lora_choices()), outputs=lora, queue=False)
    refresh_lora.click(lambda path: gr.update(choices=decoder_adapter_choices(str(path or ""), ROOT / "loras")),
                       inputs=lora, outputs=use_decoder, queue=False)

    lora_selection_inputs = [
        lora, tab.prompt_audio, auto_ref, auto_rate, tab.reference_source,
        auto_lora_tokens, budget_scale, language, speaking_rate, max_tokens,
        auto_lora_pauses, segmentation_mode,
    ]

    def on_lora_selection(*items: Any):
        path = str(items[0] or "")
        auto_tokens_on, scale, lang, current_rate, current_tokens = items[5:10]
        auto_pauses_on, split_mode = items[10:12]
        tokens_update = auto_max_tokens_update(path, bool(auto_tokens_on), scale, lang)
        panel_tokens = tokens_update["value"] if isinstance(tokens_update, dict) and "value" in tokens_update else current_tokens
        sentence_pause_update, max_pause_update = auto_pause_updates(path, bool(auto_pauses_on))
        panel = {
            "speaking_rate": current_rate,
            "max_tokens": panel_tokens,
            "budget_scale": scale,
            "language": lang,
            "auto_tokens": bool(auto_tokens_on),
            "auto_pauses": bool(auto_pauses_on),
            "segmentation_mode": str(split_mode or ""),
        }
        info, audio_update, message, rate_update, source_update = lora_selection_updates(*items[:5], panel=panel)
        media_update: Any = gr.skip()
        video_update: Any = gr.skip()
        if source_update == "lora_auto":
            recommended = _recommended_lora_reference(str(items[0] or ""))
            if recommended:
                audio_update = gr.update(value=recommended, visible=True)
            media_update = gr.update(value=recommended) if recommended else gr.skip()
            video_update = gr.update(value=None, visible=False)
        elif source_update == "empty":
            audio_update = gr.update(value=None, visible=False)
            media_update = gr.update(value=None)
            video_update = gr.update(value=None, visible=False)
        return (
            info,
            audio_update,
            media_update,
            video_update,
            message,
            rate_update,
            source_update,
            gr.update(value=saved_lora_speaking_rate(str(items[0] or ""))),
            # Loading a LoRA / DoRA selects its own decoder adapter again; None stays one click away.
            gr.update(choices=decoder_adapter_choices(str(items[0] or ""), ROOT / "loras"), value="auto"),
            # The strength its full-pipeline test chose, or the trained strength when nothing was measured.
            gr.update(value=recommended_decoder_strength(str(items[0] or "")) or 1.0),
            *decoding_updates(str(items[0] or ""), bool(items[3])),
            tokens_update,
            sentence_pause_update,
            max_pause_update,
        )

    lora_selection_outputs = [
        lora_info,
        tab.prompt_audio,
        tab.reference_media,
        tab.reference_video,
        reference_status,
        speaking_rate,
        tab.reference_source,
        lora_saved_rate,
        use_decoder,
        decoder_strength,
        temperature,
        cfg,
        beams,
        max_tokens,
        sentence_pause,
        max_pause,
    ]
    panel_inputs = [lora, speaking_rate, max_tokens, budget_scale, language, auto_lora_tokens, auto_lora_pauses, segmentation_mode]

    def _panel_values(rate: Any, tokens: Any, scale: Any, lang: Any, auto_tokens_on: Any, auto_pauses_on: Any, split_mode: Any) -> dict[str, Any]:
        return {
            "speaking_rate": rate, "max_tokens": tokens, "budget_scale": scale, "language": lang,
            "auto_tokens": bool(auto_tokens_on), "auto_pauses": bool(auto_pauses_on), "segmentation_mode": str(split_mode or ""),
        }

    def on_save_lora_rate(path: Any, value: Any, auto: Any, *panel_values: Any):
        return save_lora_speaking_rate(str(path or ""), value, bool(auto), panel=_panel_values(*panel_values))

    save_lora_rate.click(
        on_save_lora_rate,
        [lora, lora_saved_rate, auto_rate, *panel_inputs[1:]],
        [lora_info, reference_status, speaking_rate, lora_saved_rate],
        queue=False,
    )

    def on_pick_expressive(path: Any, *panel_values: Any):
        message = pick_expressive_clip(str(path or ""))
        return _lora_info(str(path or ""), **_panel_values(*panel_values))[0], message

    pick_expressive.click(on_pick_expressive, panel_inputs, [lora_info, reference_status], show_progress="full")

    def refresh_adapter_panel(path: Any, *panel_values: Any) -> str:
        return _lora_info(str(path or ""), **_panel_values(*panel_values))[0]

    # The panel's seconds follow the speaking-rate slider and its token and pause lines follow the timing controls.
    for component in (speaking_rate, max_tokens, budget_scale, language, auto_lora_tokens, auto_lora_pauses, segmentation_mode):
        component.change(
            refresh_adapter_panel, panel_inputs, lora_info,
            queue=False, show_progress="hidden", trigger_mode="always_last",
        )

    def on_auto_tokens(path: Any, enabled: Any, scale: Any, lang: Any):
        return auto_max_tokens_update(str(path or ""), bool(enabled), scale, lang)

    for component in (auto_lora_tokens, budget_scale, language):
        component.change(on_auto_tokens, [lora, auto_lora_tokens, budget_scale, language], max_tokens, queue=False, show_progress="hidden")

    def on_auto_pauses(path: Any, enabled: Any):
        return auto_pause_updates(str(path or ""), bool(enabled))

    auto_lora_pauses.change(on_auto_pauses, [lora, auto_lora_pauses], [sentence_pause, max_pause], queue=False, show_progress="hidden")

    check_words.click(check_unknown_words, [tab.text, lora, dictionary_table], [unknown_words, pronunciation_status], queue=False)
    add_suggestions.click(add_suggestions_and_save, [unknown_words, dictionary_table], [dictionary_table, pronunciation_status], queue=False)
    # Every edit in the table is saved; only the status is returned so the table does not re-trigger itself.
    dictionary_table.change(lambda rows: save_dictionary_rows(rows)[1], dictionary_table, pronunciation_status, queue=False, show_progress="hidden")
    lora.change(
        on_lora_selection,
        lora_selection_inputs,
        lora_selection_outputs,
        queue=False,
    )
    auto_rate.change(
        on_lora_selection,
        lora_selection_inputs,
        lora_selection_outputs,
        queue=False,
    )
    auto_ref.change(
        on_lora_selection,
        lora_selection_inputs,
        lora_selection_outputs,
        queue=False,
    )
    auto_tokens.click(lambda lang: default_segment_tokens(lang), language, max_tokens, queue=False)

    preview_inputs = [
        tab.text, language, max_tokens, caption_timing, tab.subtitle_file, pause_tags, budget_scale, apply_pronunciation, lora,
        segmentation_mode, speaking_rate,
    ]

    def update_preview(*items: Any):
        text, lang, tokens, timing, subtitle, pauses, scale, apply_pron, lora_path, split_mode, rate = items
        if apply_pron:
            try:
                text = apply_pronunciation_dictionary(str(text or ""), str(lora_path or ""))
            except Exception:
                pass
        mode = normalize_segmentation_mode(split_mode)
        target = smart_segment_target(str(lora_path or "")) if mode == "smart" else None
        return preview_segments(
            text, lang, tokens, timing, subtitle, pauses, scale, model_dir=model_dir,
            segmentation_mode=mode, target_tokens=target, words_per_second=preview_words_per_second(str(lora_path or ""), rate),
        )

    for component in (tab.text, language, max_tokens, caption_timing, pause_tags, budget_scale, apply_pronunciation, lora, segmentation_mode, speaking_rate):
        component.change(update_preview, preview_inputs, [segment_preview, preview_count], queue=False, show_progress="hidden", trigger_mode="always_last")
    for component in (tab.text, max_tokens, budget_scale):
        component.input(update_preview, preview_inputs, [segment_preview, preview_count], queue=False, show_progress="hidden", trigger_mode="always_last")

    def load_caption(*items: Any):
        path = resolve_path_value(items[4])
        if not path:
            rows, count = update_preview(*items)
            return gr.skip(), "", rows, count
        try:
            cues = parse_subtitle_file(path)
            text_value = subtitle_cues_to_text(cues)
            rows, count = update_preview(text_value, *items[1:])
            status = f"Loaded {len(cues)} {get_subtitle_format_label(path)} cue(s); timeline ends at {format_srt_timestamp(cues[-1].end_ms) if cues else '00:00:00.000'}."
            return text_value, status, rows, count
        except Exception as exc:
            gr.Warning(f"Caption load failed: {exc}")
            return gr.skip(), f"Caption load failed: {exc}", [[0, "Caption error", str(exc), ""]], "Caption error"

    tab.subtitle_file.change(
        load_caption,
        preview_inputs,
        [tab.text, caption_load_status, segment_preview, preview_count],
        queue=False,
    )
    open_outputs.click(lambda: open_folder(ROOT / "outputs"), outputs=tab.status, queue=False)
    task_outputs = [
        tab.task_state,
        tab.progress_html,
        tab.status,
        tab.log_tail,
        tab.output_audio,
        tab.output_video,
        tab.candidate_state,
        tab.caption_status,
        tab.final_summary,
        tab.recent_table,
        tab.task_timer,
    ]
    tab.task_timer.tick(
        generation_task_updates,
        tab.task_state,
        task_outputs,
        queue=False,
        show_progress="hidden",
    )
    if load_hook is not None:
        def attach_generation(state: str, gr_request: gr.Request = None):
            return generation_task_updates(state, gr_request, page_load=True)

        load_hook(
            attach_generation,
            tab.task_state,
            task_outputs,
            queue=False,
            show_progress="hidden",
            api_name="attach_generation",
        )
    return tab


def bind_generation_events(
    tab: GenerationTab,
    args: Any,
    registry: PresetRegistry,
) -> None:
    request_specs = [
        spec
        for spec in registry.specs
        if spec.component is not None and spec.key.startswith(("generation.", "runtime."))
    ]
    tab.request_keys = [spec.key for spec in request_specs]
    tab.request_components = [spec.component for spec in request_specs]
    model_dir = str(getattr(args, "model_dir", ROOT / "models"))
    set_model_dir(model_dir)

    def generate(
        prompt: str,
        text: str,
        subtitle_file: str | None,
        image_path: str | None,
        emotion_audio: str | None,
        gr_request: gr.Request = None,
        *component_values: Any,
        progress=gr.Progress(track_tqdm=False),
    ):
        _claim_generation_card(gr_request)
        values = dict(zip(tab.request_keys, component_values))
        started = time.perf_counter()
        request: dict[str, Any] | None = None
        try:
            request = prepare_generation_request(
                values,
                prompt=prompt,
                text=text,
                subtitle_file=subtitle_file,
                image_path=image_path,
                emotion_audio=emotion_audio,
                model_dir=model_dir,
            )
            print(f">> Generation task {request['task_layout']['task_id']} started", flush=True)
            task_folder = str(request["task_layout"]["task_folder"])
            for updates in stream_generation_request(
                request,
                use_subprocess=bool(values.get("generation.use_subprocess", False)),
                gr_progress=progress,
            ):
                running = output_task_is_active(task_folder)
                yield (
                    task_folder,
                    *updates,
                    gr.Timer(1.0 if running else 5.0, active=False),
                )
            print(f">> Generation finished in {time.perf_counter() - started:.2f}s", flush=True)
        except Exception as exc:
            canceled = is_cancellation(exc)
            if canceled:
                print(f">> Generation canceled by user after {time.perf_counter() - started:.2f}s", flush=True)
            else:
                traceback.print_exc()
            message = "Generation canceled by user." if canceled else f"Generation failed: {exc}"
            terminal = _terminal_generation_updates(
                request,
                title="Canceled" if canceled else "Failed",
                message=message,
            )
            task_folder = (
                str((request.get("task_layout") or {}).get("task_folder") or "")
                if request
                else ""
            )
            # Keep validation failures visible instead of reattaching an old
            # completed task on the next timer tick.
            yield task_folder, *terminal, gr.Timer(5.0, active=False)

    generation_outputs = [
        tab.task_state,
        tab.progress_html,
        tab.status,
        tab.log_tail,
        tab.output_audio,
        tab.output_video,
        tab.candidate_state,
        tab.caption_status,
        tab.final_summary,
        tab.recent_table,
        tab.task_timer,
    ]

    def prepare_visible_reference(
        prompt: str | None,
        media_path: str | None,
        selected_library: str | None,
        reference_source: str,
        time_ranges: str,
        lora_path: str,
        auto_lora_reference: bool,
        gr_request: gr.Request = None,
    ):
        _claim_generation_card(gr_request)
        try:
            prepared = prepare_reference_for_generation(
                prompt,
                media_path,
                selected_library,
                reference_source,
                time_ranges,
                lora_path,
                auto_lora_reference,
            )
        except ValueError as exc:
            gr.Warning(str(exc), title="Reference Voice")
            raise gr.Error(str(exc)) from exc
        gr.Info(prepared.message, title="Reference Voice")
        return (
            gr.update(value=prepared.prompt, visible=True),
            gr.update(value=prepared.media),
            gr.update(value=prepared.video, visible=bool(prepared.video)),
            gr.update(choices=prepared.choices, value=prepared.library_value),
            prepared.source,
            prepared.message,
        )

    reference_event = tab.generate_button.click(
        prepare_visible_reference,
        inputs=[
            tab.prompt_audio,
            tab.reference_media,
            tab.reference_audio_dropdown,
            tab.reference_source,
            tab.reference_ranges,
            tab.controls["runtime.lora_path"],
            tab.controls["generation.auto_lora_reference"],
        ],
        outputs=[
            tab.prompt_audio,
            tab.reference_media,
            tab.reference_video,
            tab.reference_audio_dropdown,
            tab.reference_source,
            tab.reference_status,
        ],
        queue=False,
        show_progress="minimal",
        api_name="prepare_reference_voice",
    )
    generation_event = reference_event.success(
        generate,
        inputs=[
            tab.prompt_audio,
            tab.text,
            tab.subtitle_file,
            tab.image,
            tab.emotion_audio,
            *tab.request_components,
        ],
        outputs=generation_outputs,
        api_name="generate_voice",
        concurrency_limit=1,
        concurrency_id="generation",
        show_progress="hidden",
        stream_every=0.5,
    )

    def cancel(confirmation_target: str, state_value: str, subprocess_mode: bool):
        if not confirmation_target or _path_key(confirmation_target) != _path_key(state_value):
            return gr.skip(), "The displayed run changed; click Cancel again to review the current run."
        if not state_value or not output_task_is_active(state_value):
            return gr.skip(), "No active run."
        displayed = Path(state_value).resolve()
        metadata = read_json(displayed / "metadata.json", {}) or {}
        execution_mode = str((metadata.get("settings") or {}).get("execution_mode") or "")
        if execution_mode == "subprocess" or (not execution_mode and subprocess_mode):
            job = PROCESS_MANAGER.get("generation")
            if job is None or not job.running or job.state_dir.resolve() != displayed:
                return gr.skip(), "The active displayed run is not managed by this app process."
            if not PROCESS_MANAGER.terminate("generation", expected_job=job):
                return gr.skip(), "The displayed worker has already ended or could not be stopped; check its live log."
            payload = read_progress_file(displayed / "progress.json") or {}
            payload.update({"eta_s": 0, "desc": "Waiting for the canceled subprocess to exit"})
            write_json_atomic(displayed / "progress.json", payload)
            return (
                progress_panel_html(payload, title="Cancel requested"),
                "Cancellation requested; waiting for the generation subprocess tree to stop.",
            )
        if _path_key(_ACTIVE_INPROCESS_TASK) != _path_key(displayed):
            return gr.skip(), "The displayed in-process run is not active in this app process."
        if LAZY_ENGINE.request_cancel(expected_task=str(displayed)):
            return (
                progress_panel_html({"desc": "Stopping at the next safe boundary"}, title="Cancel requested"),
                "In-process cancellation requested; synthesis will stop at the next progress boundary.",
            )
        return gr.skip(), "No in-process generation is running."

    use_subprocess_component = tab.controls["generation.use_subprocess"]
    tab.cancel_button.click(
        lambda state: (gr.update(visible=True), state), inputs=tab.task_state,
        outputs=[tab.cancel_panel, tab.cancel_target], queue=False,
    )
    tab.cancel_no.click(lambda: (gr.update(visible=False), ""), outputs=[tab.cancel_panel, tab.cancel_target], queue=False)
    tab.cancel_yes.click(
        cancel,
        inputs=[tab.cancel_target, tab.task_state, use_subprocess_component],
        outputs=[tab.progress_html, tab.status],
        api_name="confirm_cancel_generation",
        queue=False,
        show_progress="hidden",
    ).then(lambda: (gr.update(visible=False), ""), outputs=[tab.cancel_panel, tab.cancel_target], queue=False)


__all__ = [
    "EMOTION_MODES",
    "GENERATION_DEFAULTS",
    "GenerationTab",
    "INFER_KWARG_KEYS",
    "LANGUAGES",
    "PreparedReference",
    "REFERENCE_AUDIO_DIR",
    "REFERENCE_AUDIO_EXTENSIONS",
    "RUNNER_REQUEST_KEYS",
    "ReferenceSelection",
    "bind_generation_events",
    "build_default_generation_request",
    "build_generation_request",
    "build_generation_tab",
    "generation_task_updates",
    "latest_reference_audio",
    "load_reference_media",
    "lora_selection_updates",
    "prepare_generation_request",
    "prepare_reference_for_generation",
    "preview_segments",
    "reference_audio_choices",
    "reference_selection_updates",
    "resolve_reference_selection",
    "request_from_registry_defaults",
    "save_lora_speaking_rate",
    "saved_lora_speaking_rate",
    "stream_generation_request",
    "validate_request_coverage",
]
