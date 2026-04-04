from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

import numpy as np

from indextts.utils.subtitle_utils import (
    assemble_subtitle_audio,
    build_subtitle_render_units,
    ensure_audio_matrix,
    fit_audio_to_duration,
    format_srt_timestamp,
    get_subtitle_format_label,
    parse_subtitle_file,
    read_pcm16_wav,
    retime_audio_file_with_ffmpeg,
    write_pcm16_wav,
)
from indextts.utils.task_output_utils import build_segment_output_path, write_metadata_file

try:
    from pydub import AudioSegment

    MP3_AVAILABLE = True
except ImportError:
    AudioSegment = None
    MP3_AVAILABLE = False


SUBTITLE_TIMING_INTERVAL_SILENCE_MS = 0


def check_ffmpeg() -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


FFMPEG_AVAILABLE = check_ffmpeg()


def create_tts(runtime_options: Dict[str, Any]):
    from indextts.infer_v2 import IndexTTS2

    return IndexTTS2(
        model_dir=runtime_options["model_dir"],
        cfg_path=runtime_options["cfg_path"],
        use_fp16=bool(runtime_options.get("use_fp16")),
        use_deepspeed=bool(runtime_options.get("use_deepspeed")),
        use_cuda_kernel=bool(runtime_options.get("use_cuda_kernel")),
    )


def convert_wav_to_mp3(wav_path: str, mp3_path: str, bitrate: str = "256k") -> str:
    if not MP3_AVAILABLE or AudioSegment is None:
        return wav_path

    try:
        audio = AudioSegment.from_wav(wav_path)
        audio.export(mp3_path, format="mp3", bitrate=bitrate)
        os.remove(wav_path)
        return mp3_path
    except Exception as exc:
        print(f"Error converting to MP3: {exc}")
        return wav_path


def save_pcm16_wav(audio_matrix: np.ndarray, sampling_rate: int, output_path: str) -> str:
    audio_matrix = ensure_audio_matrix(audio_matrix)

    if os.path.isfile(output_path):
        os.remove(output_path)
        print(">> remove old wav file:", output_path)
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    write_pcm16_wav(audio_matrix, sampling_rate, output_path)
    print(">> wav file saved to:", output_path)
    return output_path


def current_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def format_elapsed_duration(elapsed_seconds: float) -> str:
    elapsed_ms = max(0, int(round(float(elapsed_seconds) * 1000.0)))
    hours, remainder_ms = divmod(elapsed_ms, 3600000)
    minutes, remainder_ms = divmod(remainder_ms, 60000)
    seconds, milliseconds = divmod(remainder_ms, 1000)

    if hours:
        return f"{hours}h {minutes}m {seconds}.{milliseconds:03d}s"
    if minutes:
        return f"{minutes}m {seconds}.{milliseconds:03d}s"
    return f"{seconds}.{milliseconds:03d}s"


def print_console_progress(
    label: str,
    completed: int,
    total: int,
    started_at: float,
    processed_audio_seconds: Optional[float] = None,
    item_label: str = "item",
) -> None:
    total = max(1, int(total))
    completed = min(total, max(0, int(completed)))
    elapsed_seconds = max(0.0, time.perf_counter() - started_at)
    eta_seconds = ((elapsed_seconds / completed) * (total - completed)) if completed else None
    percent = 100.0 * completed / total
    parts = [
        f">> {label} {completed}/{total} {item_label}{'' if completed == 1 else 's'} ({percent:.1f}%)",
        f"elapsed {format_elapsed_duration(elapsed_seconds)}",
    ]
    if eta_seconds is not None:
        parts.append(f"eta {format_elapsed_duration(eta_seconds)}")
    if processed_audio_seconds is not None and processed_audio_seconds > 0 and elapsed_seconds > 0:
        parts.append(f"speed {processed_audio_seconds / elapsed_seconds:.2f}x RT")
        parts.append(f"audio {processed_audio_seconds:.2f}s")
    print(" | ".join(parts))


def audio_duration_ms(audio: np.ndarray, sampling_rate: int) -> int:
    matrix = np.asarray(audio)
    if matrix.ndim == 0:
        return 0
    return int(round(matrix.shape[0] * 1000.0 / sampling_rate))


def finalize_subtitle_segment_audio(
    unit,
    unit_audio: np.ndarray,
    sampling_rate: int,
    segment_path: str,
    temp_dir: Optional[str] = None,
) -> Dict[str, Any]:
    natural_audio = ensure_audio_matrix(unit_audio)
    natural_duration_ms = audio_duration_ms(natural_audio, sampling_rate)

    fit_info = {
        "method": "copy",
        "source_duration_ms": int(natural_duration_ms),
        "target_duration_ms": int(unit.duration_ms),
        "delta_ms_before_fit": int(natural_duration_ms - unit.duration_ms),
        "stretch_rate": 1.0,
        "output_duration_ms": int(natural_duration_ms),
    }

    if natural_audio.shape[0] == 0 and unit.duration_ms <= 0:
        final_audio = natural_audio
        save_pcm16_wav(final_audio, sampling_rate, segment_path)
    elif natural_duration_ms == unit.duration_ms:
        final_audio = natural_audio
        save_pcm16_wav(final_audio, sampling_rate, segment_path)
    elif FFMPEG_AVAILABLE:
        if temp_dir:
            os.makedirs(temp_dir, exist_ok=True)
            raw_path = os.path.join(temp_dir, f"{unit.index:04d}_raw.wav")
        else:
            raw_path = f"{segment_path}.raw.wav"
        try:
            save_pcm16_wav(natural_audio, sampling_rate, raw_path)
            fit_info = retime_audio_file_with_ffmpeg(
                input_path=raw_path,
                output_path=segment_path,
                target_duration_ms=unit.duration_ms,
            )
            fitted_sampling_rate, final_audio = read_pcm16_wav(segment_path)
            if fitted_sampling_rate != sampling_rate:
                raise ValueError(
                    f"Subtitle unit {unit.index} retimed at {fitted_sampling_rate} Hz instead of {sampling_rate} Hz"
                )
        finally:
            if os.path.exists(raw_path):
                os.remove(raw_path)
    else:
        final_audio, fit_info = fit_audio_to_duration(
            natural_audio,
            sampling_rate=sampling_rate,
            target_duration_ms=unit.duration_ms,
            return_info=True,
        )
        fit_info = dict(fit_info)
        fit_info["method"] = f"python_{fit_info['method']}"
        fit_info["output_duration_ms"] = audio_duration_ms(final_audio, sampling_rate)
        save_pcm16_wav(final_audio, sampling_rate, segment_path)

    print(
        f">> Subtitle unit saved | unit {unit.index} | "
        f"target {unit.duration_ms / 1000.0:.2f}s | natural {natural_duration_ms / 1000.0:.2f}s | "
        f"final {fit_info['output_duration_ms'] / 1000.0:.2f}s | method {fit_info['method']} | "
        f"stretch {fit_info['stretch_rate']:.4f}"
    )
    return {
        "audio": final_audio,
        "segment_path": segment_path,
        "natural_duration_ms": int(natural_duration_ms),
        "fit_info": fit_info,
    }


def abs_path_or_none(path: Optional[str]) -> Optional[str]:
    return os.path.abspath(path) if path else None


def build_subtitle_status_message(
    cues,
    issues=None,
    sample_count: Optional[int] = None,
    sampling_rate: Optional[int] = None,
    task_folder: Optional[str] = None,
    segments_dir: Optional[str] = None,
    subtitle_file: Optional[str] = None,
) -> str:
    if not cues:
        return "No subtitle cues loaded."

    format_label = get_subtitle_format_label(subtitle_file)
    render_units = build_subtitle_render_units(cues)
    message_parts = [
        f"Loaded {len(cues)} {format_label} cue(s).",
        f"Timeline end: {format_srt_timestamp(cues[-1].end_ms)}.",
        f"Synthesis units: {len(render_units)}.",
        (
            "Subtitle timing uses cue start times, merges overlapping cues into larger render units when needed, "
            "avoids extra section-gap silence inside each unit, and retimes each finished unit to its target slot."
            if FFMPEG_AVAILABLE
            else "Subtitle timing uses cue start times, merges overlapping cues into larger render units when needed, "
            "avoids extra section-gap silence inside each unit, and falls back to in-process duration fitting because FFmpeg is unavailable."
        ),
    ]

    if len(render_units) < len(cues):
        message_parts.append(
            f"Detected {len(cues) - len(render_units)} overlapping cue transition(s); overlapping cues will be synthesized as merged units."
        )

    if sample_count is not None and sampling_rate:
        message_parts.append(f"Generated output length: {sample_count / float(sampling_rate):.2f}s.")

    if issues is not None:
        late_starts = [issue["delta_ms"] for issue in issues if issue["type"] == "late_start"]
        overruns = [issue["delta_ms"] for issue in issues if issue["type"] == "slot_overrun"]
        if late_starts:
            message_parts.append(
                f"{len(late_starts)} cue(s) started late because earlier speech ran long. Max late start: {max(late_starts)}ms."
            )
        if overruns:
            message_parts.append(
                f"{len(overruns)} cue(s) exceeded their subtitle duration. Max overrun: {max(overruns)}ms."
            )
        if not late_starts and not overruns:
            message_parts.append("All subtitle cue starts were preserved without timing overruns.")

    if task_folder:
        message_parts.append(f"Task folder: {os.path.abspath(task_folder)}.")
    if segments_dir:
        message_parts.append(f"Separate cue WAVs: {os.path.abspath(segments_dir)}.")

    return " ".join(message_parts)


def _load_metadata(metadata_path: str) -> Dict[str, Any]:
    with open(metadata_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _emit_progress(progress_callback, value: float, desc: str) -> None:
    if progress_callback is None:
        return
    progress_callback(value, desc=desc)


def run_generation_request(
    request: Dict[str, Any],
    tts,
    progress_callback: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    prompt = request["prompt"]
    text = request["text"]
    subtitle_mode = bool(request["subtitle_mode"])
    subtitle_file = request.get("subtitle_file")
    save_used_audio = bool(request["save_used_audio"])
    save_as_mp3 = bool(request["save_as_mp3"])
    mp3_bitrate = request["mp3_bitrate"]
    infer_kwargs = dict(request["infer_kwargs"])
    low_memory_mode = bool(request["low_memory_mode"])
    task_layout = dict(request["task_layout"])
    metadata_path = request["metadata_path"]
    output_path = task_layout["final_wav_path"]
    task_folder = task_layout["task_folder"]

    processing_started_perf = time.perf_counter()
    metadata = _load_metadata(metadata_path)
    subtitle_status_message = ""
    output = None

    tts.gr_progress = progress_callback
    tts.hybrid_model_device = low_memory_mode

    try:
        subtitle_cues = parse_subtitle_file(subtitle_file) if subtitle_mode else []
        subtitle_render_units = build_subtitle_render_units(subtitle_cues) if subtitle_mode else []

        if subtitle_mode:
            if not subtitle_cues:
                raise ValueError("No caption cues were found in the selected file.")

            rendered_units = []
            original_progress = tts.gr_progress
            sampling_rate = 22050
            subtitle_console_started_at = time.perf_counter()
            assembled_audio_seconds = 0.0
            timing_executor = None
            subtitle_temp_dir = os.path.join(task_folder, "_subtitle_timing_tmp")

            try:
                tts.gr_progress = None
                non_empty_units = [unit for unit in subtitle_render_units if unit.text.strip()]
                unit_render_metadata = {}
                subtitle_infer_kwargs = dict(infer_kwargs)
                subtitle_infer_kwargs["interval_silence"] = SUBTITLE_TIMING_INTERVAL_SILENCE_MS
                subtitle_infer_kwargs["console_progress_enabled"] = True
                subtitle_infer_kwargs["console_progress_label"] = "Subtitle synthesis"
                subtitle_infer_kwargs["console_progress_item_label"] = "section"
                base_latent_multiplier = float(subtitle_infer_kwargs["latent_multiplier"])
                processing_sections = sum(
                    len(tts.tokenizer.split_segments(tts.tokenizer.tokenize(unit.text), request["max_text_tokens"]))
                    for unit in non_empty_units
                )
                unit_render_futures = {}
                print(
                    f">> Subtitle timing started | cues {len(subtitle_cues)} | "
                    f"timing units {len(subtitle_render_units)} | sections {processing_sections} | "
                    f"batch size {subtitle_infer_kwargs['section_batch_size']}"
                )
                if non_empty_units:
                    worker_count = max(1, min(4, os.cpu_count() or 1))
                    timing_executor = ThreadPoolExecutor(
                        max_workers=worker_count,
                        thread_name_prefix="subtitle_timing",
                    )
                    print(
                        f">> Subtitle timing workers started | workers {worker_count} | "
                        f"mode {'ffmpeg' if FFMPEG_AVAILABLE else 'python_fallback'}"
                    )

                    def schedule_subtitle_unit_timing(result_idx, result):
                        unit = non_empty_units[result_idx]
                        unit_sampling_rate, unit_audio = result
                        segment_path = build_segment_output_path(task_layout["segments_dir"], unit.index)
                        unit_render_futures[unit.index] = timing_executor.submit(
                            finalize_subtitle_segment_audio,
                            unit=unit,
                            unit_audio=unit_audio,
                            sampling_rate=unit_sampling_rate,
                            segment_path=segment_path,
                            temp_dir=subtitle_temp_dir,
                        )

                    tts.infer_texts(
                        spk_audio_prompt=prompt,
                        texts=[unit.text for unit in non_empty_units],
                        on_text_complete=schedule_subtitle_unit_timing,
                        **subtitle_infer_kwargs,
                    )

                for unit_idx, unit in enumerate(subtitle_render_units):
                    _emit_progress(
                        progress_callback,
                        0.05 + 0.8 * unit_idx / max(len(subtitle_render_units), 1),
                        f"subtitle unit {unit_idx + 1}/{len(subtitle_render_units)}...",
                    )
                    if unit.text.strip():
                        unit_result = unit_render_futures[unit.index].result()
                        unit_audio = unit_result["audio"]
                        segment_path = unit_result["segment_path"]
                        fit_info = unit_result["fit_info"]
                        natural_duration_ms = unit_result["natural_duration_ms"]
                    else:
                        unit_audio = np.zeros((0, 1), dtype=np.int16)
                        segment_path = build_segment_output_path(task_layout["segments_dir"], unit_idx + 1)
                        save_pcm16_wav(unit_audio, sampling_rate, segment_path)
                        fit_info = {
                            "method": "target_silence",
                            "stretch_rate": 1.0,
                            "output_duration_ms": 0,
                        }
                        natural_duration_ms = 0

                    unit_render_metadata[unit.index] = {
                        "natural_duration_ms": int(natural_duration_ms),
                        "duration_delta_before_fit_ms": int(natural_duration_ms - unit.duration_ms),
                        "fit_method": fit_info["method"],
                        "fit_stretch_rate": float(fit_info["stretch_rate"]),
                        "selected_latent_multiplier": float(base_latent_multiplier),
                        "retry_attempted": False,
                        "retry_selected": False,
                        "retry_latent_multiplier": None,
                    }

                    metadata["subtitle"]["render_units"][unit_idx]["segment_file"] = abs_path_or_none(segment_path)
                    metadata["subtitle"]["render_units"][unit_idx]["natural_duration_ms"] = (
                        unit_render_metadata.get(unit.index, {}).get("natural_duration_ms")
                    )
                    metadata["subtitle"]["render_units"][unit_idx]["generated_duration_ms"] = int(
                        round(unit_audio.shape[0] * 1000.0 / sampling_rate)
                    )
                    metadata["subtitle"]["render_units"][unit_idx]["duration_delta_before_fit_ms"] = (
                        unit_render_metadata.get(unit.index, {}).get("duration_delta_before_fit_ms")
                    )
                    metadata["subtitle"]["render_units"][unit_idx]["fit_method"] = (
                        unit_render_metadata.get(unit.index, {}).get("fit_method")
                    )
                    metadata["subtitle"]["render_units"][unit_idx]["fit_stretch_rate"] = (
                        unit_render_metadata.get(unit.index, {}).get("fit_stretch_rate")
                    )
                    metadata["subtitle"]["render_units"][unit_idx]["selected_latent_multiplier"] = (
                        unit_render_metadata.get(unit.index, {}).get("selected_latent_multiplier")
                        or float(base_latent_multiplier)
                    )
                    metadata["subtitle"]["render_units"][unit_idx]["retry_attempted"] = (
                        unit_render_metadata.get(unit.index, {}).get("retry_attempted", False)
                    )
                    metadata["subtitle"]["render_units"][unit_idx]["retry_selected"] = (
                        unit_render_metadata.get(unit.index, {}).get("retry_selected", False)
                    )
                    metadata["subtitle"]["render_units"][unit_idx]["retry_latent_multiplier"] = (
                        unit_render_metadata.get(unit.index, {}).get("retry_latent_multiplier")
                    )
                    metadata["updated_at"] = current_timestamp()
                    write_metadata_file(metadata_path, metadata)

                    rendered_units.append((unit, unit_audio))
                    assembled_audio_seconds += unit_audio.shape[0] / float(sampling_rate) if sampling_rate else 0.0
                    print_console_progress(
                        "Subtitle timeline",
                        unit_idx + 1,
                        len(subtitle_render_units),
                        subtitle_console_started_at,
                        processed_audio_seconds=assembled_audio_seconds,
                        item_label="unit",
                    )
            finally:
                if timing_executor is not None:
                    timing_executor.shutdown(wait=True)
                if os.path.isdir(subtitle_temp_dir):
                    shutil.rmtree(subtitle_temp_dir, ignore_errors=True)
                tts.gr_progress = original_progress

            _emit_progress(progress_callback, 0.92, "assembling subtitle timeline...")
            print(">> Subtitle timeline assembly started")
            combined_audio, subtitle_issues = assemble_subtitle_audio(rendered_units, sampling_rate=sampling_rate)
            if combined_audio.shape[0] == 0:
                raise ValueError("The caption file does not contain any spoken text to synthesize.")

            print(
                f">> Subtitle timeline complete | duration {combined_audio.shape[0] / float(sampling_rate):.2f}s | "
                f"timing issues {len(subtitle_issues)}"
            )
            output = save_pcm16_wav(combined_audio, sampling_rate, output_path)
            metadata["subtitle"]["timing_issues"] = subtitle_issues
            subtitle_status_message = build_subtitle_status_message(
                subtitle_cues,
                issues=subtitle_issues,
                sample_count=combined_audio.shape[0],
                sampling_rate=sampling_rate,
                task_folder=task_folder,
                segments_dir=task_layout["segments_dir"],
                subtitle_file=subtitle_file,
            )
        else:
            output = tts.infer(
                spk_audio_prompt=prompt,
                text=text,
                output_path=output_path,
                **infer_kwargs,
            )

        if save_used_audio and prompt:
            try:
                shutil.copy2(prompt, task_layout["speaker_reference_copy_path"])
                metadata["outputs"]["speaker_reference_copy_path"] = abs_path_or_none(
                    task_layout["speaker_reference_copy_path"]
                )
                print(f"Saved used reference audio to: {task_layout['speaker_reference_copy_path']}")
            except Exception as exc:
                print(f"Error saving used audio: {exc}")

        if save_as_mp3 and MP3_AVAILABLE:
            output = convert_wav_to_mp3(output, task_layout["final_mp3_path"], bitrate=mp3_bitrate)

        processing_elapsed_seconds = time.perf_counter() - processing_started_perf
        metadata["status"] = "completed"
        metadata["updated_at"] = current_timestamp()
        metadata["error"] = None
        metadata["outputs"]["final_audio_path"] = abs_path_or_none(output)
        metadata["outputs"]["final_wav_exists"] = bool(
            task_layout["final_wav_path"] and os.path.exists(task_layout["final_wav_path"])
        )
        metadata["outputs"]["final_mp3_exists"] = bool(
            task_layout["final_mp3_path"] and os.path.exists(task_layout["final_mp3_path"])
        )
        metadata["processing"]["ended_at"] = metadata["updated_at"]
        metadata["processing"]["elapsed_ms"] = int(round(processing_elapsed_seconds * 1000.0))
        metadata["processing"]["elapsed_seconds"] = round(processing_elapsed_seconds, 3)
        metadata["processing"]["elapsed_human"] = format_elapsed_duration(processing_elapsed_seconds)
        write_metadata_file(metadata_path, metadata)

        return {
            "output_path": output,
            "subtitle_status": subtitle_status_message,
        }
    except Exception as exc:
        processing_elapsed_seconds = time.perf_counter() - processing_started_perf
        metadata["status"] = "failed"
        metadata["updated_at"] = current_timestamp()
        metadata["error"] = str(exc)
        metadata["outputs"]["final_audio_path"] = abs_path_or_none(output)
        metadata["processing"]["ended_at"] = metadata["updated_at"]
        metadata["processing"]["elapsed_ms"] = int(round(processing_elapsed_seconds * 1000.0))
        metadata["processing"]["elapsed_seconds"] = round(processing_elapsed_seconds, 3)
        metadata["processing"]["elapsed_human"] = format_elapsed_duration(processing_elapsed_seconds)
        write_metadata_file(metadata_path, metadata)
        raise
