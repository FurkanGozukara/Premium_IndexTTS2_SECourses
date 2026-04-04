from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import List, Sequence, Tuple

import numpy as np


TIMECODE_RE = re.compile(
    r"^(?:(?P<hours>\d+):)?(?P<minutes>\d{1,2}):(?P<seconds>\d{1,2})[,.](?P<milliseconds>\d{1,3})$"
)
SUPPORTED_SUBTITLE_EXTENSIONS = (".srt", ".vtt", ".sbv")
SUBTITLE_FORMAT_LABELS = {
    ".srt": "SRT",
    ".vtt": "WebVTT",
    ".sbv": "SBV",
}


@dataclass(frozen=True)
class SubtitleCue:
    index: int
    start_ms: int
    end_ms: int
    text: str

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


def parse_srt_timestamp(value: str) -> int:
    value = value.strip()
    match = TIMECODE_RE.match(value)
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {value}")

    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    milliseconds = int(match.group("milliseconds").ljust(3, "0"))
    return (((hours * 60) + minutes) * 60 + seconds) * 1000 + milliseconds


def format_srt_timestamp(value_ms: int) -> str:
    total_ms = max(0, int(value_ms))
    hours, remainder = divmod(total_ms, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def read_subtitle_file(path: str) -> str:
    last_error = None
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            with open(path, "r", encoding=encoding) as handle:
                return handle.read()
        except UnicodeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error

    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def read_srt_file(path: str) -> str:
    return read_subtitle_file(path)


def get_subtitle_extension(path: str | None) -> str:
    if not path:
        return ".srt"
    raw_value = str(path).strip().lower()
    extension = os.path.splitext(raw_value)[1]
    if not extension and raw_value.startswith("."):
        extension = raw_value
    return extension or ".srt"


def get_subtitle_format_label(path: str | None) -> str:
    extension = get_subtitle_extension(path)
    return SUBTITLE_FORMAT_LABELS.get(extension, extension.lstrip(".").upper() or "caption")


def parse_srt(content: str) -> List[SubtitleCue]:
    normalized = content.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    blocks = re.split(r"\n\s*\n", normalized)
    cues: List[SubtitleCue] = []

    for block in blocks:
        lines = block.split("\n")
        if not any(line.strip() for line in lines):
            continue

        timeline_index = 0
        cue_index = len(cues) + 1

        if lines[0].strip().isdigit():
            cue_index = int(lines[0].strip())
            timeline_index = 1

        if timeline_index >= len(lines) or "-->" not in lines[timeline_index]:
            raise ValueError(f"Invalid SRT block: {block}")

        start_raw, end_raw = lines[timeline_index].split("-->", 1)
        start_ms = parse_srt_timestamp(start_raw.strip().split()[0])
        end_ms = parse_srt_timestamp(end_raw.strip().split()[0])
        if end_ms < start_ms:
            raise ValueError(f"SRT cue end time is before start time: {lines[timeline_index]}")

        text = "\n".join(lines[timeline_index + 1 :]).strip("\n")
        cues.append(SubtitleCue(index=cue_index, start_ms=start_ms, end_ms=end_ms, text=text))

    return cues


def parse_vtt(content: str) -> List[SubtitleCue]:
    normalized = content.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    blocks = re.split(r"\n\s*\n", normalized)
    cues: List[SubtitleCue] = []

    for block in blocks:
        lines = [line.rstrip() for line in block.split("\n") if line.strip()]
        if not lines:
            continue

        first_line = lines[0].strip()
        if first_line.startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue

        timeline_index = 0
        if "-->" not in lines[timeline_index]:
            timeline_index = 1

        if timeline_index >= len(lines) or "-->" not in lines[timeline_index]:
            continue

        start_raw, end_raw = lines[timeline_index].split("-->", 1)
        start_ms = parse_srt_timestamp(start_raw.strip().split()[0])
        end_ms = parse_srt_timestamp(end_raw.strip().split()[0])
        if end_ms < start_ms:
            raise ValueError(f"WebVTT cue end time is before start time: {lines[timeline_index]}")

        text = "\n".join(lines[timeline_index + 1 :]).strip("\n")
        cues.append(
            SubtitleCue(index=len(cues) + 1, start_ms=start_ms, end_ms=end_ms, text=text)
        )

    return cues


def parse_sbv(content: str) -> List[SubtitleCue]:
    normalized = content.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    blocks = re.split(r"\n\s*\n", normalized)
    cues: List[SubtitleCue] = []

    for block in blocks:
        lines = [line.rstrip() for line in block.split("\n")]
        if not any(line.strip() for line in lines):
            continue

        timeline = lines[0].strip()
        if "," not in timeline:
            raise ValueError(f"Invalid SBV block: {block}")

        start_raw, end_raw = timeline.split(",", 1)
        start_ms = parse_srt_timestamp(start_raw.strip())
        end_ms = parse_srt_timestamp(end_raw.strip())
        if end_ms < start_ms:
            raise ValueError(f"SBV cue end time is before start time: {timeline}")

        text = "\n".join(lines[1:]).strip("\n")
        cues.append(
            SubtitleCue(index=len(cues) + 1, start_ms=start_ms, end_ms=end_ms, text=text)
        )

    return cues


def parse_subtitle(content: str, extension: str | None = None) -> List[SubtitleCue]:
    normalized_extension = get_subtitle_extension(extension)
    if normalized_extension == ".srt":
        return parse_srt(content)
    if normalized_extension == ".vtt":
        return parse_vtt(content)
    if normalized_extension == ".sbv":
        return parse_sbv(content)
    raise ValueError(
        f"Unsupported caption format '{normalized_extension}'. Supported formats: "
        + ", ".join(SUPPORTED_SUBTITLE_EXTENSIONS)
    )


def parse_subtitle_file(path: str | None) -> List[SubtitleCue]:
    if not path:
        return []
    return parse_subtitle(read_subtitle_file(path), get_subtitle_extension(path))


def subtitle_cues_to_text(cues: Sequence[SubtitleCue]) -> str:
    return "\n\n".join(cue.text for cue in cues)


def ensure_audio_matrix(audio: np.ndarray) -> np.ndarray:
    matrix = np.asarray(audio)
    if matrix.ndim == 1:
        matrix = matrix[:, np.newaxis]
    elif matrix.ndim != 2:
        raise ValueError(f"Expected 1D or 2D audio array, got shape {matrix.shape}")

    if matrix.dtype != np.int16:
        matrix = matrix.astype(np.int16)

    return matrix


def samples_to_ms(sample_count: int, sampling_rate: int) -> int:
    return int(round(sample_count * 1000.0 / sampling_rate))


def ms_to_samples(value_ms: int, sampling_rate: int) -> int:
    return int(round(value_ms * sampling_rate / 1000.0))


def assemble_subtitle_audio(
    rendered_cues: Sequence[Tuple[SubtitleCue, np.ndarray]],
    sampling_rate: int,
) -> Tuple[np.ndarray, List[dict]]:
    pieces: List[np.ndarray] = []
    issues: List[dict] = []
    cursor_samples = 0
    channel_count = None

    for cue, raw_audio in rendered_cues:
        audio = ensure_audio_matrix(raw_audio)
        if channel_count is None:
            channel_count = audio.shape[1]
        elif audio.shape[1] != channel_count:
            raise ValueError("All rendered subtitle cues must use the same channel count")

        cue_start_samples = ms_to_samples(cue.start_ms, sampling_rate)
        if cue_start_samples > cursor_samples:
            silence = np.zeros((cue_start_samples - cursor_samples, channel_count), dtype=np.int16)
            pieces.append(silence)
            cursor_samples = cue_start_samples
        elif cue_start_samples < cursor_samples:
            issues.append(
                {
                    "cue_index": cue.index,
                    "type": "late_start",
                    "delta_ms": samples_to_ms(cursor_samples - cue_start_samples, sampling_rate),
                }
            )

        if audio.shape[0] > 0:
            pieces.append(audio)
            cursor_samples += audio.shape[0]

        audio_duration_ms = samples_to_ms(audio.shape[0], sampling_rate)
        slot_overrun_ms = audio_duration_ms - cue.duration_ms
        if slot_overrun_ms > 0:
            issues.append(
                {
                    "cue_index": cue.index,
                    "type": "slot_overrun",
                    "delta_ms": slot_overrun_ms,
                }
            )

    if not pieces:
        channel_count = 1 if channel_count is None else channel_count
        return np.zeros((0, channel_count), dtype=np.int16), issues

    return np.concatenate(pieces, axis=0), issues
