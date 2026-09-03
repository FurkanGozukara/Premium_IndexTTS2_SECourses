"""Voice generation tab and the UI-to-runner request contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import html
import json
import os
from pathlib import Path
import secrets
import shutil
import sys
import threading
import time
import traceback
from typing import Any, Mapping, Sequence

import gradio as gr

from indextts.lora.io import inspect_lora, scan_lora_files
from indextts.runtime.progress import read_progress_file
from indextts.training.speaking_rate import (
    load_speaking_rate,
    speaking_rate_method_label,
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
from indextts.utils.text_segmentation import default_segment_tokens, split_text_by_tokens
from webui_generation_runner import run_generation_request

from .common import (
    LAZY_ENGINE,
    PROCESS_MANAGER,
    ROOT,
    adopt_output_task,
    btn,
    extract_reference_audio,
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
EMOTION_MODES = (
    "Same as speaker voice",
    "Emotion reference audio",
    "Emotion vector",
    "Emotion text",
)
EMOTION_NAMES = ("joy", "anger", "sad", "fear", "disgust", "depression", "surprise", "calm")
EMOTION_LABELS = ("Joy", "Anger", "Sadness", "Fear", "Disgust", "Depression", "Surprise", "Calm")
EMOTION_BIAS_DEFAULTS = (0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625)


GENERATION_DEFAULTS: dict[str, Any] = {
    "generation.language": "EN",
    "generation.max_text_tokens_per_segment": 60,
    "generation.use_caption_timing": False,
    "generation.auto_lora_reference": True,
    "generation.auto_lora_speaking_rate": True,
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
    "generation.latent_multiplier": 1.72,
    "generation.speaking_rate": 1.0,
    "generation.target_duration_s": None,
    "generation.target_duration_mode": "off",
    "generation.enable_pause_tags": True,
    "generation.text_normalization": True,
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
    "generation.use_subprocess": True,
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
        "num_candidates",
        "audio_tuning_preset",
        "audio_tuning_overrides",
        "segment_budget_scale_non_cjk",
        "cfm_temperature",
        "seed",
        "reuse_spk_cond_for_emo",
        "enable_pause_tags",
        "trim_silence_ms_threshold",
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
    runtime_value["lora_path"] = lora_path
    runtime_value["lora_strength"] = lora_strength
    runtime_value["lora_merge_into_base"] = lora_merge_into_base

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
        "num_candidates": int(_value(merged, "generation.num_candidates")),
        "audio_tuning_preset": str(_value(merged, "generation.audio_tuning_preset") or "bypass"),
        "audio_tuning_overrides": overrides,
        "segment_budget_scale_non_cjk": float(_value(merged, "generation.segment_budget_scale_non_cjk")),
        "cfm_temperature": float(_value(merged, "generation.cfm_temperature")),
        "seed": int(_value(merged, "generation.seed")),
        "reuse_spk_cond_for_emo": bool(_value(merged, "generation.reuse_spk_cond_for_emo")),
        "enable_pause_tags": bool(_value(merged, "generation.enable_pause_tags")),
        "trim_silence_ms_threshold": int(_value(merged, "generation.trim_silence_ms_threshold")),
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
        raise ValueError("Speaker reference audio is required before generation")
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

    cues = parse_subtitle_file(subtitle_path) if subtitle_mode else []
    units = build_subtitle_render_units(cues) if cues else []
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


def preview_segments(
    text: str,
    language: str,
    max_tokens: int,
    caption_timing: bool = False,
    subtitle_file: str | None = None,
    enable_pause_tags: bool = True,
    segment_scale: float = 0.72,
    model_dir: str = "models",
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
        token_len = lambda value: len(tokenizer.encode(value, allowed_special="all"))
    except Exception:
        token_len = lambda value: max(1, len(str(value).split()) * 2)
    prefix = f"<|{str(language or 'EN').lower()}|> "
    rows: list[list[Any]] = []
    section_index = 0
    row_index = 0
    chunks = split_text_with_pauses(text) if enable_pause_tags else [TextChunk(str(text))]
    for chunk in chunks:
        if isinstance(chunk, PauseChunk):
            row_index += 1
            rows.append([row_index, "Pause", f"{chunk.duration_ms} ms", "Inserted silence"])
            continue
        for segment in split_text_by_tokens(
            chunk.text,
            int(max_tokens),
            capacity=_preview_capacity(str(Path(model_dir).resolve())),
            token_len=token_len,
            lang_prefix=prefix,
            segment_budget_scale_non_cjk=float(segment_scale),
        ):
            if not segment.strip():
                continue
            section_index += 1
            row_index += 1
            rows.append([row_index, "Text segment", segment, f"{token_len(prefix + segment)} tokens"])
    pause_note = describe_pauses(text) if enable_pause_tags else "Pause tags disabled"
    return rows, f"{section_index} speech section(s) | {pause_note}"


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
            label += "  [best]"
        choices.append((label, str(source)))
    return choices


def _lora_info(path: str | None) -> tuple[str, str | None]:
    if not path:
        return "No LoRA / DoRA selected. Base model (no LoRA / DoRA) will clone from the reference only.", None
    try:
        info = inspect_lora(path)
        targets = info.get("targets") or []
        speaking_rate = load_speaking_rate(path)
        if speaking_rate is None:
            rate_line = (
                "Speaking rate: **no calibrated speaking rate yet**; train with epoch "
                "samples or use the Checkpoint Grid calibration button."
            )
        else:
            rate_line = (
                f"Speaking rate: **{speaking_rate.recommended_speaking_rate:.3f}** "
                f"(recordings {speaking_rate.dataset_words_per_second:.2f} words/s, "
                f"generated {speaking_rate.generated_words_per_second:.2f} words/s; "
                f"calibration: {speaking_rate_method_label(speaking_rate.method)})."
            )
        markdown = (
            f"**{str(info['adapter_type']).upper()}** | rank **{info['rank']}** | alpha **{info['alpha']}**  \n"
            f"Steps: **{info.get('steps', 0)}** | Dataset: **{info.get('dataset') or 'not recorded'}** | "
            f"Date: **{info.get('date') or 'not recorded'}**  \n"
            f"Targets: {len(targets)} | Size: **{info.get('size_mb', 0):.2f} MB**  \n"
            f"{rate_line}  \n"
            f"Full path: `{Path(path).expanduser().resolve()}`"
        )
        reference = info.get("recommended_reference")
        return markdown, str(reference) if reference and Path(reference).is_file() else None
    except Exception as exc:
        return f"LoRA / DoRA inspection failed: {exc}", None


def lora_selection_updates(
    path: str,
    current_reference: str | None,
    auto_reference: bool,
    auto_speaking_rate: bool,
) -> tuple[Any, Any, str, Any]:
    """Apply adapter metadata to the reference and speaking-rate controls."""

    info, recommended_reference = _lora_info(path)
    reference_update: Any = gr.skip()
    messages: list[str] = []
    if auto_reference and not current_reference and recommended_reference:
        reference_update = gr.update(value=recommended_reference)
        messages.append(
            f"Loaded recommended reference: {Path(recommended_reference).name}"
        )

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

    if not messages:
        messages.append(
            "LoRA / DoRA selected."
            if path
            else "Base model (no LoRA / DoRA) selected."
        )
    return info, reference_update, " ".join(messages), rate_update


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
        list(result.get("candidate_paths") or ([output] if output else [])),
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
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
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


def generation_task_updates(
    state_value: str,
    *,
    output_root: str | os.PathLike[str] = ROOT / "outputs",
    page_load: bool = False,
) -> tuple[Any, ...]:
    """Discover and render the per-session generation task card."""

    task_value, running = adopt_output_task(
        state_value,
        root=output_root,
        scope="generation",
        page_load=page_load,
    )
    if not task_value:
        return (
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
            gr.Timer(5.0, active=True),
        )

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
        return (
            task_value,
            progress_panel_html(payload, title="Generating voice"),
            f"Attached to running run {task_name} | {description}",
            log_value,
            gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
            recent_outputs(output_root),
            gr.Timer(1.0, active=True),
        )

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
    return task_value, *card, gr.Timer(5.0, active=True)


class _Tee:
    def __init__(self, stream: Any, path: Path) -> None:
        self.stream = stream
        self.handle = path.open("a", encoding="utf-8", newline="\n")
        self.lock = threading.Lock()

    def write(self, value: str) -> int:
        with self.lock:
            self.stream.write(value)
            self.stream.flush()
            self.handle.write(value)
            self.handle.flush()
        return len(value)

    def flush(self) -> None:
        with self.lock:
            self.stream.flush()
            self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def stream_generation_request(
    request: Mapping[str, Any],
    *,
    use_subprocess: bool,
    gr_progress: Any = None,
    process_kind: str = "generation",
):
    """Execute one prepared request and yield a shared nine-output dashboard tuple."""

    task_folder = Path(str(request["task_layout"]["task_folder"]))
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

            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                engine = LAZY_ENGINE.get(
                    request["runtime"],
                    progress_file=progress_file,
                    progress_callback=gr_progress,
                )
                result_box["result"] = run_generation_request(dict(request), engine, progress_callback=gr_progress)
        except BaseException as exc:
            result_box["error"] = exc
            result_box["traceback"] = traceback.format_exc()
            print(result_box["traceback"], file=sys.stderr, flush=True)
        finally:
            tee.close()
            result_box["done"] = True

    thread = threading.Thread(target=run_in_process, daemon=True, name="indextts-inprocess-generation")
    thread.start()
    while not result_box.get("done"):
        payload = read_progress_file(progress_file) or {}
        yield (
            progress_panel_html(payload, title="Generating voice in process"),
            str(payload.get("desc") or payload.get("stage") or "Model is working..."),
            tail_text(log_path, 60),
            gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
        )
        time.sleep(0.5)
    thread.join()
    if "error" in result_box:
        raise RuntimeError(str(result_box["error"]))
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
    text: Any = None
    subtitle_file: Any = None
    image: Any = None
    emotion_audio: Any = None
    generate_button: Any = None
    cancel_button: Any = None
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
    tab = GenerationTab()
    c = tab.controls

    with gr.Tab("Voice Generation", id="voice-generation"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### Reference Voice")
                media_upload = gr.File(
                    label="Audio or video",
                    file_types=["audio", "video"],
                    type="filepath",
                )
                gr.Markdown("Any FFmpeg-readable audio or video is accepted; video audio is extracted automatically.", elem_classes=["section-note"])
                ranges = gr.Textbox(
                    label="Time ranges",
                    placeholder="1:4; 7.5:12",
                    info="Optional start:end ranges to merge from the uploaded media.",
                )
                with gr.Row():
                    extract_button = gr.Button("✂️  Extract ranges", elem_classes=btn("teal"))
                    clear_reference = gr.Button("⌫  Clear", elem_classes=btn("orange"))
                path_input = gr.Textbox(
                    label="Reference media path",
                    info="Load any local audio or video path without uploading it.",
                )
                load_path = gr.Button("📂  Load path", elem_classes=btn("sky"))
                tab.prompt_audio = gr.Audio(
                    label="Speaker reference",
                    sources=["upload", "microphone"],
                    type="filepath",
                    format="wav",
                    buttons=["download"],
                )
                reference_status = gr.Markdown("Use 3-15 seconds of clean, single-speaker audio.", elem_classes=["section-note"])
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
                with gr.Row():
                    language = gr.Dropdown(
                        choices=list(LANGUAGES), value="EN", label="Language",
                        info="Language code used by text normalization and pronunciation.",
                    )
                    max_tokens = gr.Slider(
                        20, 300, value=60, step=1, label="Max tokens per segment",
                        info="Per-language defaults are recommended; shorter segments use less VRAM.",
                    )
                    auto_tokens = gr.Button("✨  Auto", elem_classes=btn("lime"))
                _register(registry, "generation.language", language, kind="choice", choices=LANGUAGES)
                _register(registry, "generation.max_text_tokens_per_segment", max_tokens, kind="int", minimum=20, maximum=300)
                gr.Markdown("Pause syntax: `[pause:500ms]`, `[pause:0.8s]`, or `<pause=0.5>`.", elem_classes=["section-note"])

                with gr.Row():
                    tab.subtitle_file = gr.File(
                        label="Captions (SRT / VTT / SBV)",
                        file_types=[".srt", ".vtt", ".sbv"],
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

            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### Run")
                with gr.Row():
                    tab.generate_button = gr.Button(
                        "🎙️  Generate voice", variant="primary", elem_classes=btn("emerald"), scale=3,
                    )
                    tab.cancel_button = gr.Button(
                        "⛔  Cancel", variant="stop", elem_classes=btn("red"), scale=1,
                    )
                open_outputs = gr.Button("📁  Open outputs folder", elem_classes=btn("indigo"))
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
        with gr.Row():
            lora = gr.Dropdown(
                choices=_lora_choices(),
                value="",
                label="LoRA / DoRA",
                info="Select a trained LoRA / DoRA, or None for Base model (no LoRA / DoRA), which clones from the reference only.",
                scale=12,
            )
            refresh_lora = gr.Button("↻  Refresh", elem_classes=btn("violet"), scale=1)
        with gr.Row(equal_height=False):
            strength = gr.Slider(
                0.0, 2.0, value=1.0, step=0.05,
                label="LoRA / DoRA strength",
                info="1.0 is the trained strength; lower is subtler and higher is stronger.",
                scale=2,
            )
            auto_ref = gr.Checkbox(
                value=GENERATION_DEFAULTS["generation.auto_lora_reference"],
                label="Auto-load the LoRA / DoRA recommended reference audio",
                info="Loads the LoRA / DoRA's saved reference only when the speaker reference is empty.",
                scale=2,
            )
            auto_rate = gr.Checkbox(
                value=GENERATION_DEFAULTS["generation.auto_lora_speaking_rate"],
                label="Auto-apply the LoRA / DoRA calibrated speaking rate",
                info="Uses the selected voice's measured pace; selecting None resets speaking rate to 1.0.",
                scale=2,
            )
            merge_lora = gr.Checkbox(
                value=False,
                label="Merge LoRA / DoRA into base weights for speed (BF16 only)",
                info="Temporarily folds the selected LoRA / DoRA into floating GPT weights and restores them before switching.",
                scale=3,
            )
        lora_info = gr.Markdown("No LoRA / DoRA selected.", elem_classes=["section-note"])
        registry.register("runtime.lora_path", lora, "", kind="str")
        registry.register("runtime.lora_strength", strength, 1.0, kind="float", minimum=0.0, maximum=2.0)
        registry.register("runtime.lora_merge_into_base", merge_lora, False, kind="bool")
        _register(registry, "generation.auto_lora_reference", auto_ref, kind="bool")
        _register(
            registry,
            "generation.auto_lora_speaking_rate",
            auto_rate,
            kind="bool",
        )

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
                repetition = gr.Slider(1, 20, value=10.0, step=0.1, label="Repetition penalty", info="10 is the established model default.")
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
                max_silence = gr.Slider(0, 200, value=0, step=1, label="Max consecutive silence tokens", info="0 disables token trimming; use only to suppress unusually long model silences.")
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
                save_ref = gr.Checkbox(value=False, label="Save used reference", info="Copies the speaker reference into the task folder for reproducibility.")
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
                use_subprocess = gr.Checkbox(value=True, label="Use isolated subprocess", info="Recommended: cancellation can terminate the complete model process and release VRAM.")
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
            tab.output_audio = gr.Audio(label="Generated audio", type="filepath", visible=False, buttons=["download"])
            tab.output_video = gr.Video(label="Generated MP4", visible=False, buttons=["download"])
        tab.candidate_state = gr.State([])
        with gr.Column(elem_classes=["candidate-list"]):
            @gr.render(inputs=tab.candidate_state, triggers=[tab.candidate_state.change])
            def render_candidates(paths: list[str] | None):
                candidates_value = list(paths or [])
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
        load_recent_reference.click(
            lambda path: (gr.update(value=path), f"Loaded recent output as reference: {Path(path).name}") if path and Path(path).is_file() else (gr.skip(), "Select a recent output first."),
            recent_audio,
            [tab.prompt_audio, reference_status],
            queue=False,
        )

    # Capture the complete generation/runtime request surface after Models adds runtime controls.
    c.update({spec.key: spec.component for spec in registry.specs if spec.component is not None and spec.key.startswith("generation.")})
    c["runtime.lora_path"] = lora
    c["runtime.lora_strength"] = strength
    c["runtime.lora_merge_into_base"] = merge_lora

    def on_media(path: str | None, value_ranges: str):
        if not path:
            return gr.skip(), "Use 3-15 seconds of clean, single-speaker audio."
        output, message = extract_reference_audio(path, value_ranges)
        return gr.update(value=output) if output else gr.skip(), message

    def on_extract(path: str | None, value_ranges: str):
        output, message = extract_reference_audio(path or "", value_ranges, require_ranges=True)
        return gr.update(value=output) if output else gr.skip(), message

    def on_load_path(path: str, value_ranges: str):
        output, message = extract_reference_audio(path, value_ranges)
        if not output:
            gr.Warning(message)
            return gr.skip(), message
        return output, message

    media_upload.upload(on_media, [media_upload, ranges], [tab.prompt_audio, reference_status], queue=False, show_progress="hidden")
    extract_button.click(on_extract, [media_upload, ranges], [tab.prompt_audio, reference_status], queue=False)
    load_path.click(on_load_path, [path_input, ranges], [tab.prompt_audio, reference_status], queue=False)
    clear_reference.click(lambda: (None, None, "", "Reference cleared."), outputs=[media_upload, tab.prompt_audio, path_input, reference_status], queue=False)

    refresh_lora.click(lambda: gr.update(choices=_lora_choices()), outputs=lora, queue=False)

    lora_selection_inputs = [lora, tab.prompt_audio, auto_ref, auto_rate]
    lora_selection_outputs = [
        lora_info,
        tab.prompt_audio,
        reference_status,
        speaking_rate,
    ]
    lora.change(
        lora_selection_updates,
        lora_selection_inputs,
        lora_selection_outputs,
        queue=False,
    )
    auto_rate.change(
        lora_selection_updates,
        lora_selection_inputs,
        lora_selection_outputs,
        queue=False,
    )
    auto_tokens.click(lambda lang: default_segment_tokens(lang), language, max_tokens, queue=False)

    preview_inputs = [tab.text, language, max_tokens, caption_timing, tab.subtitle_file, pause_tags, budget_scale]

    def update_preview(*items: Any):
        return preview_segments(*items, model_dir=model_dir)

    for component in (tab.text, language, max_tokens, caption_timing, pause_tags, budget_scale):
        component.change(update_preview, preview_inputs, [segment_preview, preview_count], queue=False, show_progress="hidden", trigger_mode="always_last")
    for component in (tab.text, max_tokens, budget_scale):
        component.input(update_preview, preview_inputs, [segment_preview, preview_count], queue=False, show_progress="hidden", trigger_mode="always_last")

    def load_caption(path: str | None, current_text: str, use_timing: bool, lang: str, token_limit: int, pauses: bool, scale: float):
        if not path:
            rows, count = preview_segments(current_text, lang, token_limit, False, None, pauses, scale, model_dir)
            return gr.skip(), "", rows, count
        try:
            cues = parse_subtitle_file(path)
            text_value = subtitle_cues_to_text(cues)
            rows, count = preview_segments(text_value, lang, token_limit, use_timing, path, pauses, scale, model_dir)
            status = f"Loaded {len(cues)} {get_subtitle_format_label(path)} cue(s); timeline ends at {format_srt_timestamp(cues[-1].end_ms) if cues else '00:00:00.000'}."
            return text_value, status, rows, count
        except Exception as exc:
            gr.Warning(f"Caption load failed: {exc}")
            return gr.skip(), f"Caption load failed: {exc}", [[0, "Caption error", str(exc), ""]], "Caption error"

    tab.subtitle_file.change(
        load_caption,
        [tab.subtitle_file, tab.text, caption_timing, language, max_tokens, pause_tags, budget_scale],
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
        load_hook(
            lambda state: generation_task_updates(state, page_load=True),
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

    def generate(
        prompt: str,
        text: str,
        subtitle_file: str | None,
        image_path: str | None,
        emotion_audio: str | None,
        *component_values: Any,
        progress=gr.Progress(track_tqdm=False),
    ):
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
                use_subprocess=bool(values.get("generation.use_subprocess", True)),
                gr_progress=progress,
            ):
                running = output_task_is_active(task_folder)
                yield task_folder, *updates, gr.Timer(1.0 if running else 5.0, active=True)
            print(f">> Generation finished in {time.perf_counter() - started:.2f}s", flush=True)
        except Exception as exc:
            traceback.print_exc()
            canceled = "cancel" in str(exc).lower()
            message = "Generation canceled by user." if canceled else f"Generation failed: {exc}"
            terminal = _terminal_generation_updates(
                request,
                title="Canceled" if canceled else "Failed",
                message=message,
            )
            task_folder = (
                str((request.get("task_layout") or {}).get("task_folder") or "")
                if request
                else gr.skip()
            )
            yield task_folder, *terminal, gr.Timer(5.0, active=True)

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
    generation_event = tab.generate_button.click(
        generate,
        inputs=[tab.prompt_audio, tab.text, tab.subtitle_file, tab.image, tab.emotion_audio, *tab.request_components],
        outputs=generation_outputs,
        api_name="generate_voice",
        concurrency_limit=1,
        concurrency_id="generation",
        show_progress="minimal",
        stream_every=0.5,
    )

    # State inputs are resolved server-side and cannot be replaced by an
    # event's JavaScript return value.  A hidden non-stateful component lets
    # confirm() pass its boolean to the handler exactly once.
    confirm_state = gr.Checkbox(value=False, visible=False, label="Generation cancel confirmation")

    def cancel(confirmed: bool, state_value: str, subprocess_mode: bool):
        if not confirmed:
            return gr.skip(), "Cancellation dismissed."
        if not state_value or not output_task_is_active(state_value):
            return gr.skip(), "No active run."
        displayed = Path(state_value).resolve()
        metadata = read_json(displayed / "metadata.json", {}) or {}
        execution_mode = str((metadata.get("settings") or {}).get("execution_mode") or "")
        if execution_mode == "subprocess" or (not execution_mode and subprocess_mode):
            job = PROCESS_MANAGER.get("generation")
            if job is None or not job.running or job.state_dir.resolve() != displayed:
                return gr.skip(), "The active displayed run is not managed by this app process."
            PROCESS_MANAGER.terminate("generation")
            metadata.update(status="canceled", updated_at=_now(), error="Generation canceled by user")
            write_metadata_file(str(displayed / "metadata.json"), metadata)
            payload = read_progress_file(displayed / "progress.json") or {}
            payload.update({"eta_s": 0, "desc": "Canceled"})
            write_json_atomic(displayed / "progress.json", payload)
            return (
                progress_panel_html(payload, title="Canceled"),
                "Generation canceled and its subprocess tree was stopped.",
            )
        if LAZY_ENGINE.request_cancel():
            return (
                progress_panel_html({"desc": "Canceled"}, title="Canceled"),
                "In-process cancellation requested; synthesis will stop at the next progress boundary.",
            )
        return gr.skip(), "No in-process generation is running."

    use_subprocess_component = tab.controls["generation.use_subprocess"]
    tab.cancel_button.click(
        cancel,
        inputs=[confirm_state, tab.task_state, use_subprocess_component],
        outputs=[tab.progress_html, tab.status],
        js="(confirmed, state, mode) => [window.confirm('Cancel the running generation?'), state, mode]",
        queue=False,
        show_progress="hidden",
    )


__all__ = [
    "EMOTION_MODES",
    "GENERATION_DEFAULTS",
    "GenerationTab",
    "INFER_KWARG_KEYS",
    "LANGUAGES",
    "RUNNER_REQUEST_KEYS",
    "bind_generation_events",
    "build_default_generation_request",
    "build_generation_request",
    "build_generation_tab",
    "generation_task_updates",
    "lora_selection_updates",
    "prepare_generation_request",
    "preview_segments",
    "request_from_registry_defaults",
    "stream_generation_request",
    "validate_request_coverage",
]
