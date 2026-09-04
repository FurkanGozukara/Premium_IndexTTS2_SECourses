from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import subprocess
from typing import Iterable, Sequence

import numpy as np
import soundfile as sf


SUPPORTED_MEDIA_EXTENSIONS = (
    ".mp4",
    ".mkv",
    ".webm",
    ".mov",
    ".avi",
    ".flv",
    ".wmv",
    ".m4v",
    ".ts",
    ".mts",
    ".m2ts",
    ".mpg",
    ".mpeg",
    ".3gp",
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".oga",
    ".opus",
    ".m4a",
    ".aac",
    ".wma",
    ".aiff",
    ".aif",
    ".ape",
    ".alac",
    ".caf",
)
SUPPORTED_SUBTITLE_EXTENSIONS = (".srt", ".vtt", ".sbv")


@dataclass(frozen=True)
class MediaInfo:
    duration_s: float
    has_audio: bool
    has_video: bool
    sample_rate: int | None
    channels: int | None
    codec: str | None


@dataclass(frozen=True)
class AudioQuality:
    peak_dbfs: float
    clipping_ratio: float
    silence_ratio: float


def _run(args: Sequence[str], operation: str) -> subprocess.CompletedProcess[str]:
    command = [str(argument) for argument in args]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError as exc:
        executable = command[0] if command else operation
        raise RuntimeError(f"{executable} was not found on PATH") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"{operation} failed: {detail}")
    return result


def probe_media(path: str | Path) -> MediaInfo:
    """Inspect the first audio stream and basic container information with ffprobe."""

    source = str(Path(path))
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,codec_name,sample_rate,channels,duration",
            "-of",
            "json",
            source,
        ],
        f"ffprobe for {source}",
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe returned invalid JSON for {source}") from exc

    streams = payload.get("streams") or []
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    duration_raw = (payload.get("format") or {}).get("duration")
    if duration_raw in (None, "N/A"):
        duration_raw = next(
            (s.get("duration") for s in streams if s.get("duration") not in (None, "N/A")),
            0.0,
        )
    try:
        duration_s = max(0.0, float(duration_raw or 0.0))
    except (TypeError, ValueError):
        duration_s = 0.0

    def optional_int(value: object) -> int | None:
        try:
            return int(value) if value not in (None, "N/A", "") else None
        except (TypeError, ValueError):
            return None

    codec_stream = audio_stream or video_stream or {}
    return MediaInfo(
        duration_s=duration_s,
        has_audio=audio_stream is not None,
        has_video=video_stream is not None,
        sample_rate=optional_int((audio_stream or {}).get("sample_rate")),
        channels=optional_int((audio_stream or {}).get("channels")),
        codec=codec_stream.get("codec_name"),
    )


def extract_audio(
    path: str | Path,
    out_wav: str | Path,
    sample_rate: int = 24000,
    mono: bool = True,
    start_s: float | None = None,
    end_s: float | None = None,
    loudness_normalize: bool = False,
) -> str:
    """Decode media to PCM WAV with one progress-free ffmpeg invocation."""

    source = str(Path(path))
    output = Path(out_wav)
    output.parent.mkdir(parents=True, exist_ok=True)
    if start_s is not None and start_s < 0:
        raise ValueError("start_s must be non-negative")
    if end_s is not None and start_s is not None and end_s <= start_s:
        raise ValueError("end_s must be greater than start_s")

    args = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostats"]
    if start_s is not None:
        args.extend(["-ss", f"{float(start_s):.6f}"])
    args.extend(["-i", source])
    if end_s is not None:
        duration = float(end_s) - float(start_s or 0.0)
        args.extend(["-t", f"{duration:.6f}"])
    args.append("-vn")
    if loudness_normalize:
        args.extend(["-af", "loudnorm=I=-20:TP=-1.5:LRA=11"])
    if mono:
        args.extend(["-ac", "1"])
    args.extend(["-ar", str(int(sample_rate)), "-c:a", "pcm_s16le", str(output)])
    _run(args, f"audio extraction for {source}")
    if not output.is_file() or output.stat().st_size < 44:
        raise RuntimeError(f"ffmpeg produced no usable audio for {source}")
    _read_audio_cached.cache_clear()
    return str(output)


@lru_cache(maxsize=1)
def _read_audio_cached(path: str, mtime_ns: int, size: int) -> tuple[np.ndarray, int]:
    del mtime_ns, size
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32), int(sample_rate)


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return _read_audio_cached(str(resolved), stat.st_mtime_ns, stat.st_size)


def slice_audio(
    audio: np.ndarray,
    sample_rate: int,
    start_s: float,
    end_s: float,
    pad_ms: int = 0,
    fade_ms: int = 0,
) -> tuple[np.ndarray, float, float]:
    """Slice a decoded mono array and return samples plus its clamped time range."""

    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 2:
        samples = np.mean(samples, axis=1, dtype=np.float32)
    if samples.ndim != 1:
        raise ValueError(f"Expected mono or channel-last audio, got shape {samples.shape}")
    if end_s <= start_s:
        raise ValueError("end_s must be greater than start_s")
    duration_s = samples.shape[0] / float(sample_rate)
    actual_start = max(0.0, float(start_s) - max(0, pad_ms) / 1000.0)
    actual_end = min(duration_s, float(end_s) + max(0, pad_ms) / 1000.0)
    start_i = max(0, min(samples.shape[0], int(round(actual_start * sample_rate))))
    end_i = max(start_i, min(samples.shape[0], int(round(actual_end * sample_rate))))
    piece = np.array(samples[start_i:end_i], dtype=np.float32, copy=True)
    fade_samples = min(int(round(max(0, fade_ms) * sample_rate / 1000.0)), piece.size // 2)
    if fade_samples:
        ramp = np.linspace(0.0, 1.0, fade_samples, endpoint=True, dtype=np.float32)
        piece[:fade_samples] *= ramp
        piece[-fade_samples:] *= ramp[::-1]
    return piece, start_i / float(sample_rate), end_i / float(sample_rate)


def cut_segment(
    src_wav: str | Path,
    out_wav: str | Path,
    start_s: float,
    end_s: float,
    pad_ms: int = 0,
    fade_ms: int = 0,
) -> str:
    """Cut from a cached, fully decoded WAV; ffmpeg is never spawned per segment."""

    audio, sample_rate = read_audio(src_wav)
    piece, _, _ = slice_audio(audio, sample_rate, start_s, end_s, pad_ms, fade_ms)
    output = Path(out_wav)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, piece, sample_rate, subtype="PCM_16")
    return str(output)


def _iter_media_in_folder(folder: Path, recursive: bool) -> Iterable[Path]:
    iterator = folder.rglob("*") if recursive else folder.glob("*")
    for candidate in iterator:
        if candidate.is_file() and candidate.suffix.casefold() in SUPPORTED_MEDIA_EXTENSIONS:
            yield candidate


def find_media_files(inputs: list[str], recursive: bool = True) -> list[str]:
    found: dict[str, str] = {}
    for raw_input in inputs:
        path = Path(raw_input).expanduser()
        candidates: Iterable[Path]
        if path.is_file():
            candidates = [path] if path.suffix.casefold() in SUPPORTED_MEDIA_EXTENSIONS else []
        elif path.is_dir():
            candidates = _iter_media_in_folder(path, recursive)
        else:
            continue
        for candidate in candidates:
            resolved = str(candidate.resolve())
            found.setdefault(resolved.casefold(), resolved)
    return sorted(found.values(), key=lambda value: value.casefold())


def find_sidecar_subtitles(media_path: str | Path) -> list[str]:
    media = Path(media_path)
    if not media.parent.is_dir():
        return []
    stem = media.stem.casefold()
    matches: list[Path] = []
    for candidate in media.parent.iterdir():
        if not candidate.is_file() or candidate.suffix.casefold() not in SUPPORTED_SUBTITLE_EXTENSIONS:
            continue
        candidate_stem = candidate.stem.casefold()
        if candidate_stem == stem or candidate_stem.startswith(stem + "."):
            matches.append(candidate)

    extension_order = {".srt": 0, ".vtt": 1, ".sbv": 2}
    matches.sort(
        key=lambda p: (
            0 if p.stem.casefold() == stem else 1,
            extension_order.get(p.suffix.casefold(), 99),
            p.name.casefold(),
        )
    )
    return [str(path.resolve()) for path in matches]


def find_sidecar_transcript(media_path: str | Path) -> str | None:
    media = Path(media_path)
    if not media.parent.is_dir():
        return None
    target_name = f"{media.stem}.txt".casefold()
    matches = sorted(
        (
            candidate
            for candidate in media.parent.iterdir()
            if candidate.is_file() and candidate.name.casefold() == target_name
        ),
        key=lambda path: path.name.casefold(),
    )
    return str(matches[0].resolve()) if matches else None


def measure_loudness_lufs(audio: np.ndarray, sr: int) -> float:
    samples = np.asarray(audio, dtype=np.float64)
    if samples.ndim == 2:
        samples = np.mean(samples, axis=1)
    if samples.size == 0 or not np.any(np.abs(samples) > 1e-12):
        return float("-inf")
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(int(sr))
        value = float(meter.integrated_loudness(samples))
        if math.isfinite(value):
            return value
    except (ImportError, ValueError, OverflowError, FloatingPointError):
        pass
    rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
    return 20.0 * math.log10(max(rms, 1e-12))


def normalize_loudness(
    audio: np.ndarray,
    sr: int,
    target_lufs: float = -20.0,
) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float32)
    try:
        import pyloudnorm  # noqa: F401
    except ImportError:
        peak = float(np.max(np.abs(samples), initial=0.0))
        if peak <= 1e-12:
            return np.array(samples, dtype=np.float32, copy=True)
        # A conservative peak fallback cannot promise integrated LUFS, but it
        # does provide stable, unclipped levels when pyloudnorm is unavailable.
        return np.ascontiguousarray(samples * np.float32(0.95 / peak), dtype=np.float32)
    current = measure_loudness_lufs(samples, sr)
    if not math.isfinite(current):
        return np.array(samples, dtype=np.float32, copy=True)
    gain = 10.0 ** ((float(target_lufs) - current) / 20.0)
    normalized = samples * np.float32(gain)
    peak = float(np.max(np.abs(normalized), initial=0.0))
    if peak > 0.999:
        normalized *= np.float32(0.999 / peak)
    return np.ascontiguousarray(normalized, dtype=np.float32)


def trim_silence(
    audio: np.ndarray,
    sr: int,
    top_db: float = 40.0,
    pad_ms: int = 50,
    *,
    return_indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, tuple[int, int]]:
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 2:
        samples = np.mean(samples, axis=1, dtype=np.float32)
    if samples.size == 0:
        result = np.array(samples, copy=True)
        return (result, (0, 0)) if return_indices else result
    try:
        import librosa

        _, bounds = librosa.effects.trim(samples, top_db=float(top_db))
        start, end = int(bounds[0]), int(bounds[1])
    except (ImportError, ValueError):
        threshold = float(np.max(np.abs(samples), initial=0.0)) * 10.0 ** (-float(top_db) / 20.0)
        active = np.flatnonzero(np.abs(samples) > threshold)
        if not active.size:
            start, end = 0, samples.size
        else:
            start, end = int(active[0]), int(active[-1] + 1)
    padding = max(0, int(round(pad_ms * sr / 1000.0)))
    start = max(0, start - padding)
    end = min(samples.size, end + padding)
    result = np.ascontiguousarray(samples[start:end], dtype=np.float32)
    return (result, (start, end)) if return_indices else result


def compute_energy_envelope(audio: np.ndarray, sr: int, hop_ms: int = 10) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 2:
        samples = np.mean(samples, axis=1, dtype=np.float32)
    hop = max(1, int(round(sr * hop_ms / 1000.0)))
    if samples.size == 0:
        return np.zeros(0, dtype=np.float32)
    pad = (-samples.size) % hop
    if pad:
        samples = np.pad(samples, (0, pad))
    frames = samples.reshape(-1, hop)
    rms = np.sqrt(np.mean(np.square(frames), axis=1, dtype=np.float64))
    return np.asarray(rms, dtype=np.float32)


def analyze_audio_quality(
    audio: np.ndarray,
    sr: int,
    *,
    clipping_threshold: float = 0.999,
    silence_threshold_dbfs: float = -40.0,
    frame_ms: int = 20,
) -> AudioQuality:
    """Measure peak, clipped samples, and frame-level internal silence."""

    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 2:
        samples = np.mean(samples, axis=1, dtype=np.float32)
    samples = samples.reshape(-1)
    if samples.size == 0:
        return AudioQuality(float("-inf"), 0.0, 1.0)
    peak = float(np.max(np.abs(samples), initial=0.0))
    peak_dbfs = 20.0 * math.log10(max(peak, 1e-12))
    clipping_ratio = float(np.mean(np.abs(samples) >= float(clipping_threshold)))

    frame_size = max(1, int(round(int(sr) * max(1, int(frame_ms)) / 1000.0)))
    padding = (-samples.size) % frame_size
    framed = np.pad(samples, (0, padding)).reshape(-1, frame_size) if padding else samples.reshape(-1, frame_size)
    rms = np.sqrt(np.mean(np.square(framed), axis=1, dtype=np.float64))
    silence_threshold = 10.0 ** (float(silence_threshold_dbfs) / 20.0)
    silence_ratio = float(np.mean(rms <= silence_threshold)) if rms.size else 1.0
    return AudioQuality(peak_dbfs, clipping_ratio, silence_ratio)
