from __future__ import annotations

import os
import re
from typing import Dict

from .atomic_json import write_json_atomic


TASK_ID_RE = re.compile(r"^(\d{4})")
INVALID_FILENAME_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def sanitize_output_basename(filename: str | None, fallback: str = "final") -> str:
    if not filename:
        return fallback

    name = os.path.basename(str(filename).strip())
    if not name:
        return fallback

    stem, _ = os.path.splitext(name)
    stem = INVALID_FILENAME_CHARS_RE.sub("_", stem).strip(" ._")
    return stem or fallback


def normalize_file_extension(extension: str | None, fallback: str = ".png") -> str:
    normalized = (extension or fallback).strip().lower()
    if not normalized:
        normalized = fallback
    if not normalized.startswith("."):
        normalized = f".{normalized}"
    normalized = re.sub(r"[^a-z0-9.]+", "", normalized)
    return normalized if normalized.strip(".") else fallback


def get_next_output_index(output_root: str = "outputs") -> int:
    os.makedirs(output_root, exist_ok=True)

    numbers = []
    for entry in os.listdir(output_root):
        stem = os.path.splitext(entry)[0]
        match = TASK_ID_RE.match(stem)
        if match:
            numbers.append(int(match.group(1)))

    return max(numbers, default=0) + 1


def create_task_output_layout(
    output_root: str = "outputs",
    filename: str | None = None,
    subtitle_mode: bool = False,
    subtitle_extension: str | None = None,
    image_extension: str | None = None,
) -> Dict[str, str | None]:
    while True:
        task_id = f"{get_next_output_index(output_root):04d}"
        task_folder = os.path.join(output_root, task_id)
        if not os.path.exists(task_folder):
            break

    os.makedirs(task_folder, exist_ok=False)

    final_basename = sanitize_output_basename(filename, fallback=task_id)
    normalized_subtitle_extension = (subtitle_extension or ".srt").strip().lower()
    if not normalized_subtitle_extension.startswith("."):
        normalized_subtitle_extension = f".{normalized_subtitle_extension}"
    normalized_image_extension = normalize_file_extension(image_extension) if image_extension else None
    layout: Dict[str, str | None] = {
        "task_id": task_id,
        "task_folder": task_folder,
        "final_basename": final_basename,
        "final_wav_path": os.path.join(task_folder, f"{final_basename}.wav"),
        "final_mp3_path": os.path.join(task_folder, f"{final_basename}.mp3"),
        "final_mp4_path": os.path.join(task_folder, f"{final_basename}.mp4"),
        "metadata_path": os.path.join(task_folder, "metadata.json"),
        "subtitle_copy_path": (
            os.path.join(task_folder, f"source_subtitles{normalized_subtitle_extension}")
            if subtitle_mode
            else None
        ),
        "source_image_copy_path": (
            os.path.join(task_folder, f"source_image{normalized_image_extension}")
            if normalized_image_extension
            else None
        ),
        "speaker_reference_copy_path": os.path.join(task_folder, "speaker_reference.wav"),
        "segments_dir": None,
    }

    if subtitle_mode:
        segments_dir = os.path.join(task_folder, "segments")
        os.makedirs(segments_dir, exist_ok=True)
        layout["segments_dir"] = segments_dir

    return layout


def build_segment_output_path(segments_dir: str, cue_order: int) -> str:
    return os.path.join(segments_dir, f"{cue_order:04d}.wav")


def write_metadata_file(path: str, data: dict) -> None:
    write_json_atomic(path, data, indent=2, ensure_ascii=False)
