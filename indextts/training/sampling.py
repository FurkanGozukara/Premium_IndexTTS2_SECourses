"""Out-of-process sample generation used by the adapter trainer."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from indextts.runtime import gpu_free_gb, gpu_total_gb, resolve_preset
from indextts.utils.atomic_json import write_json_atomic

from .train_config import TrainConfig


@dataclass
class SampleResult:
    generated: bool
    path: str = ""
    message: str = ""
    elapsed_s: float = 0.0


def _device_index(device: str) -> int:
    try:
        return int(str(device).split(":", 1)[1]) if ":" in str(device) else 0
    except (TypeError, ValueError):
        return 0


def generate_training_sample(
    config: TrainConfig,
    *,
    adapter_path: str | Path,
    reference_path: str | Path,
    output_path: str | Path,
    epoch: int,
    log: Callable[[str], None] | None = None,
) -> SampleResult:
    """Run the normal generation worker while the training process stays alive."""

    emit = log or (lambda message: print(message, flush=True))
    started = time.perf_counter()
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    reference = Path(reference_path).expanduser().resolve()
    adapter = Path(adapter_path).expanduser().resolve()
    if not reference.is_file():
        return SampleResult(False, message=f"sample reference is missing: {reference}")
    if not adapter.is_file():
        return SampleResult(False, message=f"sample adapter is missing: {adapter}")

    index = _device_index(config.device)
    free_gb = gpu_free_gb(index)
    if free_gb < config.sample_min_free_vram_gb:
        message = (
            f"sample skipped: {free_gb:.1f} GB free VRAM is below "
            f"the {config.sample_min_free_vram_gb:.1f} GB threshold"
        )
        emit(message)
        return SampleResult(False, message=message, elapsed_s=time.perf_counter() - started)

    runtime = resolve_preset(
        config.sample_runtime_tier,
        gpu_total_gb(index),
        free_gb,
    )
    runtime.device = config.device
    runtime.lora_path = str(adapter)
    runtime.lora_strength = 1.0

    job_dir = output.parent / ".sample_jobs" / f"epoch_{int(epoch):03d}"
    job_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = job_dir / "metadata.json"
    request_path = job_dir / "request.json"
    result_path = job_dir / "result.json"
    progress_path = job_dir / "progress.json"
    task_layout = {
        "task_id": f"training_epoch_{int(epoch):03d}",
        "task_folder": str(job_dir),
        "final_basename": output.stem,
        "final_wav_path": str(output),
        "final_mp3_path": str(output.with_suffix(".mp3")),
        "final_mp4_path": str(output.with_suffix(".mp4")),
        "segments_dir": str(job_dir / "segments"),
        "speaker_reference_copy_path": str(job_dir / "reference.wav"),
    }
    now = datetime.now(timezone.utc).isoformat()
    metadata = {
        "status": "in_progress",
        "created_at": now,
        "updated_at": now,
        "outputs": {},
        "processing": {
            "started_at": now,
            "ended_at": None,
            "elapsed_ms": None,
            "elapsed_seconds": None,
            "elapsed_human": None,
        },
        "error": None,
    }
    write_json_atomic(metadata_path, metadata, indent=2, ensure_ascii=False)
    request = {
        "runtime": {
            "runtime": runtime.to_dict(),
            "model_dir": config.model_dir,
            "cfg_path": config.model_config,
            "use_qwen_emo": False,
        },
        "progress_file": str(progress_path),
        "lora_path": str(adapter),
        "lora_strength": 1.0,
        "task_layout": task_layout,
        "metadata_path": str(metadata_path),
        "prompt": str(reference),
        "text": config.sample_text,
        "language": "EN",
        "subtitle_mode": False,
        "subtitle_file": None,
        "save_used_audio": False,
        "save_as_mp3": False,
        "mp3_bitrate": "256k",
        "image_path": None,
        "low_memory_mode": runtime.blocks_to_swap > 0,
        "max_text_tokens": 120,
        "infer_kwargs": {
            "do_sample": True,
            "top_p": 0.8,
            "top_k": 30,
            "temperature": 0.8,
            "length_penalty": 0.0,
            "num_beams": 1,
            "repetition_penalty": 10.0,
            "max_mel_tokens": 1500,
            "emo_audio_prompt": None,
            "emo_alpha": 1.0,
            "emo_vector": None,
            "use_emo_text": False,
            "emo_text": None,
            "use_random": False,
            "verbose": False,
            "max_text_tokens_per_segment": 120,
            "interval_silence": 200,
            "diffusion_steps": 25,
            "inference_cfg_rate": 0.7,
            "max_speaker_audio_length": 15.0,
            "max_emotion_audio_length": 15.0,
            "section_batch_size": 1,
            "latent_multiplier": 1.72,
            "max_consecutive_silence": 0,
            "semantic_layer": 17,
            "cfm_cache_length": runtime.cfm_cache_length,
            "reset_beam_cache_per_segment": True,
        },
    }
    write_json_atomic(request_path, request, indent=2, ensure_ascii=False)
    worker = Path(__file__).resolve().parents[2] / "webui_subprocess_worker.py"
    command = [
        sys.executable,
        str(worker),
        "--request-file",
        str(request_path),
        "--result-file",
        str(result_path),
    ]
    emit(f">> generating epoch {epoch} sample in a subprocess")
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        completed = subprocess.run(
            command,
            cwd=str(worker.parent),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=config.sample_timeout_s,
            check=False,
            creationflags=creationflags,
        )
    except subprocess.TimeoutExpired:
        message = f"sample generation timed out after {config.sample_timeout_s:.0f}s"
        emit(message)
        return SampleResult(False, message=message, elapsed_s=time.perf_counter() - started)

    if completed.stdout.strip():
        emit(completed.stdout.strip())
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        message = "sample generation failed"
        if detail:
            message += f": {detail[-1]}"
        emit(message)
        return SampleResult(False, message=message, elapsed_s=time.perf_counter() - started)
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        message = f"sample worker returned no readable result: {exc}"
        emit(message)
        return SampleResult(False, message=message, elapsed_s=time.perf_counter() - started)
    generated_path = Path(str(result.get("output_path") or output))
    if result.get("status") != "ok" or not generated_path.is_file():
        message = str(result.get("error") or "sample worker did not create audio")
        emit(message)
        return SampleResult(False, message=message, elapsed_s=time.perf_counter() - started)
    message = f"sample saved: {generated_path}"
    emit(message)
    return SampleResult(
        True,
        path=str(generated_path.resolve()),
        message=message,
        elapsed_s=time.perf_counter() - started,
    )


__all__ = ["SampleResult", "generate_training_sample"]
