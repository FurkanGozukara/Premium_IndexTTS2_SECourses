from __future__ import annotations

from dataclasses import dataclass
import os
import re
import shutil
import subprocess
from typing import List, Sequence, Tuple
import wave

import librosa
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


@dataclass(frozen=True)
class SubtitleRenderUnit:
    index: int
    start_ms: int
    end_ms: int
    text: str
    cue_indices: Tuple[int, ...]

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


def normalize_subtitle_text(text: str) -> str:
    return " ".join(part.strip() for part in text.splitlines() if part.strip())


def build_subtitle_render_units(cues: Sequence[SubtitleCue]) -> List[SubtitleRenderUnit]:
    if not cues:
        return []

    units: List[SubtitleRenderUnit] = []
    current_group: List[SubtitleCue] = []
    current_end_ms = -1

    def flush_group() -> None:
        if not current_group:
            return

        normalized_parts = [normalize_subtitle_text(cue.text) for cue in current_group]
        text = " ".join(part for part in normalized_parts if part).strip()
        units.append(
            SubtitleRenderUnit(
                index=len(units) + 1,
                start_ms=current_group[0].start_ms,
                end_ms=max(cue.end_ms for cue in current_group),
                text=text,
                cue_indices=tuple(cue.index for cue in current_group),
            )
        )

    for cue in cues:
        if not current_group:
            current_group = [cue]
            current_end_ms = cue.end_ms
            continue

        if cue.start_ms < current_end_ms:
            current_group.append(cue)
            current_end_ms = max(current_end_ms, cue.end_ms)
            continue

        flush_group()
        current_group = [cue]
        current_end_ms = cue.end_ms

    flush_group()
    return units


def ensure_audio_matrix(audio: np.ndarray) -> np.ndarray:
    matrix = np.asarray(audio)
    if matrix.ndim == 1:
        matrix = matrix[:, np.newaxis]
    elif matrix.ndim != 2:
        raise ValueError(f"Expected 1D or 2D audio array, got shape {matrix.shape}")

    if matrix.dtype != np.int16:
        matrix = matrix.astype(np.int16)

    return matrix


def write_pcm16_wav(audio: np.ndarray, sampling_rate: int, output_path: str) -> str:
    matrix = ensure_audio_matrix(audio)
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(matrix.shape[1])
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sampling_rate))
        wav_file.writeframes(matrix.tobytes())

    return output_path


def read_pcm16_wav(path: str) -> Tuple[int, np.ndarray]:
    with wave.open(path, "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sampling_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        frames = wav_file.readframes(frame_count)

    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, got sample width {sample_width} bytes: {path}")

    audio = np.frombuffer(frames, dtype=np.int16)
    if audio.size == 0:
        return sampling_rate, np.zeros((0, channels), dtype=np.int16)

    if audio.size % channels != 0:
        raise ValueError(f"PCM frame data is not divisible by channel count for {path}")

    return sampling_rate, audio.reshape(-1, channels).copy()


def pad_or_trim_audio_to_samples(audio: np.ndarray, target_samples: int) -> np.ndarray:
    matrix = ensure_audio_matrix(audio)
    target_samples = max(0, int(target_samples))
    current_samples = matrix.shape[0]

    if current_samples == target_samples:
        return matrix

    if target_samples == 0:
        return np.zeros((0, matrix.shape[1]), dtype=np.int16)

    if current_samples == 0:
        return np.zeros((target_samples, matrix.shape[1]), dtype=np.int16)

    if current_samples > target_samples:
        return matrix[:target_samples]

    padding = np.zeros((target_samples - current_samples, matrix.shape[1]), dtype=np.int16)
    return np.concatenate([matrix, padding], axis=0)


def samples_to_ms(sample_count: int, sampling_rate: int) -> int:
    return int(round(sample_count * 1000.0 / sampling_rate))


def ms_to_samples(value_ms: int, sampling_rate: int) -> int:
    return int(round(value_ms * sampling_rate / 1000.0))


def build_ffmpeg_atempo_chain(playback_rate: float) -> List[float]:
    rate = float(playback_rate)
    if rate <= 0:
        raise ValueError(f"Playback rate must be positive, got {playback_rate}")

    chain: List[float] = []
    while rate < 0.5:
        chain.append(0.5)
        rate /= 0.5
    while rate > 2.0:
        chain.append(2.0)
        rate /= 2.0

    if not chain or abs(rate - 1.0) > 1e-9:
        chain.append(rate)

    return chain or [1.0]


def retime_audio_file_with_ffmpeg(
    input_path: str,
    output_path: str,
    target_duration_ms: int,
) -> dict:
    sampling_rate, source_audio = read_pcm16_wav(input_path)
    source_audio = ensure_audio_matrix(source_audio)
    target_duration_ms = max(0, int(target_duration_ms))
    target_samples = ms_to_samples(target_duration_ms, sampling_rate)
    source_duration_ms = samples_to_ms(source_audio.shape[0], sampling_rate)

    info = {
        "method": "copy",
        "source_duration_ms": source_duration_ms,
        "target_duration_ms": target_duration_ms,
        "delta_ms_before_fit": int(source_duration_ms - target_duration_ms),
        "stretch_rate": 1.0,
        "output_duration_ms": target_duration_ms,
    }

    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    if source_audio.shape[0] == target_samples:
        shutil.copyfile(input_path, output_path)
        return info

    if target_samples == 0:
        info["method"] = "target_silence"
        write_pcm16_wav(np.zeros((0, source_audio.shape[1]), dtype=np.int16), sampling_rate, output_path)
        info["output_duration_ms"] = 0
        return info

    if source_audio.shape[0] == 0:
        info["method"] = "source_silence"
        write_pcm16_wav(
            np.zeros((target_samples, source_audio.shape[1]), dtype=np.int16),
            sampling_rate,
            output_path,
        )
        return info

    stretch_rate = source_audio.shape[0] / float(target_samples)
    atempo_chain = build_ffmpeg_atempo_chain(stretch_rate)
    target_seconds = target_samples / float(sampling_rate)
    filters = [f"atempo={factor:.10f}" for factor in atempo_chain]
    filters.append(f"apad=whole_dur={target_seconds:.10f}")
    filters.append(f"atrim=duration={target_seconds:.10f}")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-filter:a",
        ",".join(filters),
        "-ar",
        str(sampling_rate),
        "-ac",
        str(source_audio.shape[1]),
        "-acodec",
        "pcm_s16le",
        "-f",
        "wav",
        output_path,
        "-loglevel",
        "error",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg retime failed for {os.path.basename(output_path)}: {result.stderr.strip()}")

    output_sampling_rate, output_audio = read_pcm16_wav(output_path)
    exact_audio = pad_or_trim_audio_to_samples(output_audio, target_samples)
    if exact_audio.shape[0] != output_audio.shape[0]:
        write_pcm16_wav(exact_audio, output_sampling_rate, output_path)
    else:
        exact_audio = ensure_audio_matrix(output_audio)

    info["method"] = "ffmpeg_atempo"
    info["stretch_rate"] = float(stretch_rate)
    info["output_duration_ms"] = samples_to_ms(exact_audio.shape[0], output_sampling_rate)
    return info


def fit_audio_to_duration(
    audio: np.ndarray,
    sampling_rate: int,
    target_duration_ms: int,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    matrix = ensure_audio_matrix(audio)
    source_duration_ms = samples_to_ms(matrix.shape[0], sampling_rate)
    target_samples = max(0, ms_to_samples(target_duration_ms, sampling_rate))
    info = {
        "method": "none",
        "source_duration_ms": source_duration_ms,
        "target_duration_ms": int(target_duration_ms),
        "delta_ms_before_fit": int(source_duration_ms - target_duration_ms),
        "stretch_rate": 1.0,
    }

    if matrix.shape[0] == target_samples:
        return (matrix, info) if return_info else matrix

    if target_samples == 0:
        info["method"] = "target_silence"
        fitted = np.zeros((0, matrix.shape[1]), dtype=np.int16)
        return (fitted, info) if return_info else fitted

    if matrix.shape[0] == 0:
        info["method"] = "source_silence"
        fitted = np.zeros((target_samples, matrix.shape[1]), dtype=np.int16)
        return (fitted, info) if return_info else fitted

    tolerance_ms = min(250, max(80, int(round(target_duration_ms * 0.01))))
    tolerance_samples = ms_to_samples(tolerance_ms, sampling_rate)
    delta_samples = target_samples - matrix.shape[0]

    if abs(delta_samples) <= tolerance_samples:
        if delta_samples > 0:
            info["method"] = "pad_silence"
            padding = np.zeros((delta_samples, matrix.shape[1]), dtype=np.int16)
            fitted = np.concatenate([matrix, padding], axis=0)
        else:
            info["method"] = "trim_tail"
            fitted = matrix[:target_samples]
        return (fitted, info) if return_info else fitted

    stretch_rate = matrix.shape[0] / float(target_samples)
    info["method"] = "time_stretch"
    info["stretch_rate"] = float(stretch_rate)
    stretched_channels: List[np.ndarray] = []

    for channel_idx in range(matrix.shape[1]):
        samples = matrix[:, channel_idx].astype(np.float32) / 32768.0
        stretched = librosa.effects.time_stretch(samples, rate=stretch_rate)
        stretched_channels.append(stretched)

    max_len = max(channel.shape[0] for channel in stretched_channels)
    stretched_matrix = np.zeros((max_len, len(stretched_channels)), dtype=np.float32)
    for channel_idx, channel in enumerate(stretched_channels):
        stretched_matrix[: channel.shape[0], channel_idx] = channel

    if stretched_matrix.shape[0] > target_samples:
        stretched_matrix = stretched_matrix[:target_samples]
    elif stretched_matrix.shape[0] < target_samples:
        padding = np.zeros((target_samples - stretched_matrix.shape[0], stretched_matrix.shape[1]), dtype=np.float32)
        stretched_matrix = np.concatenate([stretched_matrix, padding], axis=0)

    stretched_matrix = np.clip(stretched_matrix, -1.0, 1.0)
    fitted = (stretched_matrix * 32767.0).astype(np.int16)
    return (fitted, info) if return_info else fitted


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
