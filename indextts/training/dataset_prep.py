from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import re
import shutil
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf

from indextts.utils.pause_tags import TextChunk, split_text_with_pauses

from .dataset_manifest import (
    DATASET_INFO_FILENAME,
    MANIFEST_FILENAME,
    PREVIEW_FILENAME,
    append_manifest_row,
    atomic_write_json,
    summarize_manifest,
    write_manifest,
    write_preview_csv,
)
from .media import (
    SUPPORTED_MEDIA_EXTENSIONS,
    SUPPORTED_SUBTITLE_EXTENSIONS,
    analyze_audio_quality,
    compute_energy_envelope,
    extract_audio,
    find_media_files,
    find_sidecar_subtitles,
    find_sidecar_transcript,
    measure_loudness_lufs,
    normalize_loudness,
    probe_media,
    slice_audio,
    trim_silence,
)
from .segmenter import (
    apply_padding_and_limits,
    build_sentence_aligned_segments,
    build_segments_from_words,
    filter_segments,
    is_sentence_aligned_text,
    snap_boundaries_to_silence,
    split_long_segment,
)
from .subtitles import (
    Segment,
    build_caption_transcript,
    clean_cues,
    merge_cues_into_sentences,
    parse_subtitle_file,
)


_WORD_RE = re.compile(r"\b[\w]+(?:['’-][\w]+)*\b", flags=re.UNICODE)
_BRACKET_ANNOTATION_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)|\{[^}]*\}")


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _default_segmentation_mode() -> str:
    return "sentence_aligned" if _cuda_available() else "cue_boundaries"


@dataclass
class DatasetPrepConfig:
    """Configuration for caption/Whisper segmentation and audio filtering.

    ``segmentation_mode`` accepts ``sentence_aligned`` (caption text plus
    Whisper word timing), ``cue_boundaries`` (legacy caption timing),
    ``whisper_only``, or ``auto``. Auto resolves to sentence alignment when
    CUDA is available and cue boundaries otherwise. ``align_with_whisper`` is
    retained as a compatibility alias for ``sentence_aligned``.
    """

    name: str
    inputs: list[str]
    recursive: bool = True
    language: str = "EN"
    output_root: str = "datasets"
    subtitle_policy: str = "prefer_sidecar"
    whisper_model: str = "openai/whisper-large-v3-turbo"
    whisper_device: str = "cuda:0"
    align_with_whisper: bool = False
    segmentation_mode: str = field(default_factory=_default_segmentation_mode)
    target_s: float = 8.0
    min_s: float = 4.0
    max_s: float = 12.0
    max_gap_ms: int = 700
    pad_ms: int = 60
    snap_to_silence: bool = True
    snap_window_ms: int = 200
    trim_silence: bool = True
    trim_top_db: float = 40.0
    loudness_normalize: bool = True
    target_lufs: float = -20.0
    sample_rate: int = 24000
    min_words: int = 2
    max_words: int = 80
    min_file_alignment_coverage: float = 0.60
    min_segment_alignment_coverage: float = 0.70
    min_words_per_second: float = 1.0
    max_words_per_second: float = 5.5
    min_peak_dbfs: float = -35.0
    max_clipping_ratio: float = 0.001
    clipping_threshold: float = 0.999
    max_silence_ratio: float | None = None
    silence_threshold_dbfs: float = -40.0
    silence_frame_ms: int = 20
    remove_bracket_annotations: bool = True
    dedupe_rolling_captions: bool = True
    drop_duplicate_sentences: bool = True
    export_reference_candidates: int = 5
    overwrite: bool = False
    max_segments: int = 0
    seed: int = 0
    speaker_name: str = ""
    speaker_from_folder: bool = False

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("Dataset name must not be empty")
        if Path(self.name).name != self.name or self.name in {".", ".."}:
            raise ValueError("Dataset name must be a single directory name")
        if not self.inputs:
            raise ValueError("At least one input is required")
        if self.subtitle_policy not in {"prefer_sidecar", "whisper_only", "sidecar_only"}:
            raise ValueError(f"Unsupported subtitle_policy: {self.subtitle_policy}")
        if self.segmentation_mode not in {
            "auto",
            "sentence_aligned",
            "cue_boundaries",
            "whisper_only",
        }:
            raise ValueError(f"Unsupported segmentation_mode: {self.segmentation_mode}")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not 0 < self.min_s <= self.target_s <= self.max_s:
            raise ValueError("Durations must satisfy 0 < min_s <= target_s <= max_s")
        if self.min_words < 0 or self.max_words < self.min_words:
            raise ValueError("Word limits are invalid")
        if self.max_segments < 0:
            raise ValueError("max_segments must be zero or positive")
        for name in ("min_file_alignment_coverage", "min_segment_alignment_coverage"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if not 0.0 <= self.max_clipping_ratio <= 1.0:
            raise ValueError("max_clipping_ratio must be between zero and one")
        if not 0.0 < self.clipping_threshold <= 1.0:
            raise ValueError("clipping_threshold must be in (0, 1]")
        if self.max_silence_ratio is not None and not 0.0 <= self.max_silence_ratio <= 1.0:
            raise ValueError("max_silence_ratio must be between zero and one")
        if not 0.0 < self.min_words_per_second <= self.max_words_per_second:
            raise ValueError("Words/second limits are invalid")
        if self.silence_frame_ms <= 0:
            raise ValueError("silence_frame_ms must be positive")

    def resolved_segmentation_mode(self) -> str:
        if self.align_with_whisper:
            return "sentence_aligned"
        if self.subtitle_policy == "whisper_only":
            return "whisper_only"
        if self.segmentation_mode == "auto":
            return _default_segmentation_mode()
        return self.segmentation_mode

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DatasetPrepConfig":
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown dataset preparation config fields: {', '.join(unknown)}")
        config = cls(**dict(payload))
        config.inputs = [str(value) for value in config.inputs]
        return config


@dataclass
class DatasetSummary:
    name: str
    output_dir: str
    status: str
    segment_count: int
    total_duration_s: float
    word_count: int
    sources: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reference_candidates: list[str] = field(default_factory=list)
    manifest_path: str = ""
    dataset_info_path: str = ""
    duration_histogram: dict[str, int] = field(default_factory=dict)
    subtitle_stats: dict[str, int] = field(default_factory=dict)
    alignment: dict[str, Any] = field(default_factory=dict)
    filter_drop_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _PrintingReporter:
    def __init__(self) -> None:
        self._last_line = ""

    def update(
        self,
        completed: int | float,
        total: int | float | None = None,
        desc: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        total_value = total or 0
        prefix = f"[{completed}/{total_value}]" if total_value else f"[{completed}]"
        line = f"{prefix} {desc}".strip()
        if line != self._last_line:
            print(line, flush=True)
            self._last_line = line

    def log(self, msg: str) -> None:
        print(msg, flush=True)

    def set_stage(self, name: str) -> None:
        self.log(name)

    def finish(self) -> None:
        return None


def _default_reporter(name: str) -> Any:
    try:
        from indextts.runtime.progress import ProgressReporter

        return ProgressReporter(f"Prepare dataset {name}")
    except Exception:
        return _PrintingReporter()


def _log(reporter: Any, message: str) -> None:
    if hasattr(reporter, "log"):
        reporter.log(message)
    else:
        print(message, flush=True)


def _stage(reporter: Any, name: str) -> None:
    if hasattr(reporter, "set_stage"):
        reporter.set_stage(name)
    else:
        _log(reporter, name)


def _update(
    reporter: Any,
    completed: int,
    total: int,
    desc: str,
    extra: dict[str, Any] | None = None,
) -> None:
    reporter.update(completed, total, desc, extra=extra)


def _cancelled(cancel_check: Callable[[], bool] | None) -> bool:
    return bool(cancel_check and cancel_check())


def _read_text(path: Path) -> str:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeError as exc:
            last_error = exc
    if last_error:
        raise last_error
    return path.read_text(encoding="utf-8")


@dataclass(frozen=True)
class _ImportItem:
    audio_path: Path
    text: str
    speaker: str
    transcript_source: str


def _resolve_metadata_audio(folder: Path, value: str) -> Path:
    raw = Path(value.strip().strip('"'))
    candidates = [raw] if raw.is_absolute() else [folder / raw, folder / "wavs" / raw]
    expanded: list[Path] = []
    for candidate in candidates:
        expanded.append(candidate)
        if not candidate.suffix:
            expanded.append(candidate.with_suffix(".wav"))
    return next((candidate.resolve() for candidate in expanded if candidate.is_file()), expanded[0].resolve())


def _parse_metadata_csv(path: Path, warnings: list[str]) -> list[_ImportItem]:
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle, delimiter="|"):
            if row and any(cell.strip() for cell in row):
                rows.append(row)
    if not rows:
        warnings.append(f"Empty metadata file: {path}")
        return []

    first = [cell.strip().casefold() for cell in rows[0]]
    has_header = any(value in {"wav", "audio", "audio_path", "path", "file", "filename"} for value in first)
    header = first if has_header else []
    data_rows = rows[1:] if has_header else rows
    items: list[_ImportItem] = []
    for row_number, row in enumerate(data_rows, start=2 if has_header else 1):
        if len(row) < 2:
            warnings.append(f"Skipped malformed metadata row {row_number}: {path}")
            continue
        if has_header:
            values = {header[index]: row[index].strip() for index in range(min(len(header), len(row)))}
            audio_value = next(
                (values[key] for key in ("audio_path", "wav", "audio", "path", "file", "filename") if values.get(key)),
                "",
            )
            text = values.get("text") or values.get("transcript") or values.get("normalized_text") or ""
            speaker = values.get("speaker") or values.get("speaker_name") or ""
        else:
            audio_value = row[0].strip()
            text = row[1].strip()
            speaker = row[2].strip() if len(row) >= 3 else ""
        audio_path = _resolve_metadata_audio(path.parent, audio_value)
        if not audio_path.is_file():
            warnings.append(f"Metadata audio not found, skipped: {audio_path}")
            continue
        if not text.strip():
            warnings.append(f"Empty transcript in metadata row {row_number}, skipped: {path}")
            continue
        items.append(_ImportItem(audio_path, text.strip(), speaker, "metadata_csv"))
    return items


def _discover_import_items(config: DatasetPrepConfig, warnings: list[str]) -> list[_ImportItem]:
    metadata_files: dict[str, Path] = {}
    pair_candidates: dict[str, Path] = {}
    for raw in config.inputs:
        path = Path(raw).expanduser()
        if path.is_file() and path.name.casefold() == "metadata.csv":
            metadata_files[str(path.resolve()).casefold()] = path.resolve()
            continue
        if path.is_dir():
            iterator = path.rglob("metadata.csv") if config.recursive else path.glob("metadata.csv")
            for metadata in iterator:
                metadata_files[str(metadata.resolve()).casefold()] = metadata.resolve()
        if path.is_file() and path.suffix.casefold() in SUPPORTED_MEDIA_EXTENSIONS:
            pair_candidates[str(path.resolve()).casefold()] = path.resolve()
        elif path.is_dir():
            iterator = path.rglob("*") if config.recursive else path.glob("*")
            for candidate in iterator:
                if candidate.is_file() and candidate.suffix.casefold() in {
                    ".wav",
                    ".flac",
                    ".aiff",
                    ".aif",
                    ".ogg",
                }:
                    pair_candidates[str(candidate.resolve()).casefold()] = candidate.resolve()

    items: list[_ImportItem] = []
    for metadata in sorted(metadata_files.values(), key=lambda item: str(item).casefold()):
        items.extend(_parse_metadata_csv(metadata, warnings))
    metadata_audio = {str(item.audio_path.resolve()).casefold() for item in items}
    for audio in sorted(pair_candidates.values(), key=lambda item: str(item).casefold()):
        if str(audio.resolve()).casefold() in metadata_audio:
            continue
        transcript = next(
            (
                candidate
                for candidate in audio.parent.iterdir()
                if candidate.is_file()
                and candidate.name.casefold() == f"{audio.stem}.txt".casefold()
            ),
            None,
        )
        if transcript is None:
            continue
        if find_sidecar_subtitles(audio):
            continue
        text = _read_text(transcript).strip()
        if text:
            items.append(_ImportItem(audio.resolve(), text, "", "wav_txt"))
    unique: dict[str, _ImportItem] = {}
    for item in items:
        unique.setdefault(str(item.audio_path.resolve()).casefold(), item)
    return sorted(unique.values(), key=lambda item: str(item.audio_path).casefold())


def _orphan_subtitle_warnings(
    config: DatasetPrepConfig,
    media_files: Sequence[str],
) -> list[str]:
    media_keys = {
        (str(Path(path).parent.resolve()).casefold(), Path(path).stem.casefold()) for path in media_files
    }
    orphans: dict[tuple[str, str], Path] = {}
    for raw in config.inputs:
        path = Path(raw).expanduser()
        candidates: Iterable[Path]
        if path.is_file() and path.suffix.casefold() in SUPPORTED_SUBTITLE_EXTENSIONS:
            candidates = [path]
        elif path.is_dir():
            candidates = path.rglob("*") if config.recursive else path.glob("*")
        else:
            continue
        for candidate in candidates:
            if not candidate.is_file() or candidate.suffix.casefold() not in SUPPORTED_SUBTITLE_EXTENSIONS:
                continue
            parent_key = str(candidate.parent.resolve()).casefold()
            subtitle_stem = candidate.stem.casefold()
            possible_stems = [subtitle_stem]
            if "." in subtitle_stem:
                possible_stems.append(subtitle_stem.rsplit(".", 1)[0])
            if any((parent_key, stem) in media_keys for stem in possible_stems):
                continue
            base = possible_stems[-1]
            orphans.setdefault((parent_key, base), candidate)
    return [
        f"No media found for subtitle source {path.parent / base}; skipped"
        for (_, base), path in sorted(orphans.items(), key=lambda item: str(item[1]).casefold())
    ]


def _safe_key(path: Path, used: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_") or "source"
    key = base
    suffix = 2
    while key.casefold() in used:
        key = f"{base}_{suffix}"
        suffix += 1
    used.add(key.casefold())
    return key


def _speaker_for(config: DatasetPrepConfig, path: Path, embedded: str = "") -> str:
    if config.speaker_name.strip():
        return config.speaker_name.strip()
    if embedded.strip():
        return embedded.strip()
    return path.parent.name if config.speaker_from_folder else ""


def _reserve_segmentation_max(config: DatasetPrepConfig) -> float:
    reserve_ms = config.pad_ms * 2
    if config.snap_to_silence:
        reserve_ms += config.snap_window_ms * 2
    return max(config.min_s, config.max_s - reserve_ms / 1000.0)


def _split_overlong(
    segments: Sequence[Segment],
    units: Sequence[Any],
    max_s: float,
) -> list[Segment]:
    output: list[Segment] = []
    for segment in segments:
        output.extend(split_long_segment(segment, units, max_s))
    return output


def _snap_segments(
    segments: Sequence[Segment],
    energy: np.ndarray,
    config: DatasetPrepConfig,
    *,
    protect_words: bool = False,
) -> list[Segment]:
    if not config.snap_to_silence:
        return [replace(segment) for segment in segments]
    snapped: list[Segment] = []
    for index, segment in enumerate(segments):
        first_word_start = None
        last_word_end = None
        if protect_words and segment.word_timestamps:
            first_word_start = int(round(float(segment.word_timestamps[0]["start_s"]) * 1000.0))
            last_word_end = int(round(float(segment.word_timestamps[-1]["end_s"]) * 1000.0))
        if protect_words and index and segments[index - 1].word_timestamps:
            previous_end = int(
                round(float(segments[index - 1].word_timestamps[-1]["end_s"]) * 1000.0)
            )
        else:
            previous_end = segments[index - 1].end_ms if index else 0
        if protect_words and index + 1 < len(segments) and segments[index + 1].word_timestamps:
            next_start = int(
                round(float(segments[index + 1].word_timestamps[0]["start_s"]) * 1000.0)
            )
        else:
            next_start = segments[index + 1].start_ms if index + 1 < len(segments) else None
        candidate = snap_boundaries_to_silence(
            segment,
            energy,
            hop_ms=10,
            window_ms=config.snap_window_ms,
            previous_end_ms=previous_end,
            next_start_ms=next_start,
            start_upper_ms=first_word_start,
            end_lower_ms=last_word_end,
        )
        if protect_words:
            snapped.append(candidate)
        else:
            # Cue edges are our only word-boundary evidence. Keep snapping
            # outward so a local in-word dip cannot cut the first/last word.
            snapped.append(
                replace(
                    candidate,
                    start_ms=min(segment.start_ms, candidate.start_ms),
                    end_ms=max(segment.end_ms, candidate.end_ms),
                )
            )

    if protect_words:
        for index in range(1, len(snapped)):
            previous = snapped[index - 1]
            current = snapped[index]
            if previous.end_ms > current.start_ms:
                previous_word_end = int(
                    round(float(previous.word_timestamps[-1]["end_s"]) * 1000.0)
                )
                current_word_start = int(
                    round(float(current.word_timestamps[0]["start_s"]) * 1000.0)
                )
                boundary = max(previous_word_end, min(current_word_start, (previous.end_ms + current.start_ms) // 2))
                snapped[index - 1] = replace(previous, end_ms=boundary)
                snapped[index] = replace(current, start_ms=boundary)
    return snapped


def _segments_from_plain_transcript(
    text: str,
    timed_segments: Sequence[Segment],
) -> list[Segment]:
    transcript_matches = list(_WORD_RE.finditer(text))
    if not transcript_matches or not timed_segments:
        return []
    weights = [max(1, len(_WORD_RE.findall(segment.text))) for segment in timed_segments]
    total_weight = sum(weights)
    result: list[Segment] = []
    cursor = 0
    cumulative = 0
    for index, (segment, weight) in enumerate(zip(timed_segments, weights)):
        cumulative += weight
        next_cursor = (
            len(transcript_matches)
            if index == len(timed_segments) - 1
            else max(cursor + 1, round(cumulative * len(transcript_matches) / total_weight))
        )
        next_cursor = min(len(transcript_matches), next_cursor)
        if cursor >= len(transcript_matches):
            break
        char_start = transcript_matches[cursor].start()
        char_end = transcript_matches[next_cursor - 1].end()
        while char_end < len(text) and text[char_end] in ",.;:!?。！？\"') ]":
            char_end += 1
        result.append(replace(segment, text=text[char_start:char_end].strip()))
        cursor = next_cursor
    return result


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _source_name(path: Path) -> str:
    return path.resolve().as_posix()


def _load_audio(path: Path, sample_rate: int) -> np.ndarray:
    audio, original_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    if original_rate != sample_rate:
        import librosa

        expected_samples = int(round(audio.shape[0] * sample_rate / float(original_rate)))
        audio = librosa.resample(audio, orig_sr=int(original_rate), target_sr=int(sample_rate))
        if audio.shape[0] > expected_samples:
            audio = audio[:expected_samples]
        elif audio.shape[0] < expected_samples:
            audio = np.pad(audio, (0, expected_samples - audio.shape[0]))
    return np.ascontiguousarray(audio, dtype=np.float32)


def _increment_reason(counts: dict[str, int], reason: str) -> None:
    counts[reason] = counts.get(reason, 0) + 1


def _normalize_duplicate_sentence(text: str) -> str:
    without_pauses = " ".join(
        chunk.text
        for chunk in split_text_with_pauses(str(text or ""))
        if isinstance(chunk, TextChunk)
    )
    without_annotations = (
        _BRACKET_ANNOTATION_RE.sub(" ", without_pauses).lower().replace("’", "'")
    )
    alphanumeric = "".join(
        character if character.isalnum() or character == "'" else " "
        for character in without_annotations
    )
    return " ".join(alphanumeric.split())


def _duplicate_preference(
    row: Mapping[str, Any],
    target_s: float,
    original_index: int,
) -> tuple[int, float, float, int]:
    try:
        alignment_coverage = float(row.get("alignment_coverage"))
    except (TypeError, ValueError, OverflowError):
        alignment_coverage = float("nan")
    has_alignment = math.isfinite(alignment_coverage)
    try:
        duration_s = float(row.get("duration_s", 0.0))
    except (TypeError, ValueError, OverflowError):
        duration_s = float("inf")
    duration_distance = (
        abs(duration_s - target_s) if math.isfinite(duration_s) else float("inf")
    )
    return (
        0 if has_alignment else 1,
        -alignment_coverage if has_alignment else 0.0,
        duration_distance,
        original_index,
    )


def _deduplicate_sentence_rows(
    rows: Sequence[dict[str, Any]],
    target_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    best_by_sentence: dict[str, int] = {}
    for index, row in enumerate(rows):
        normalized = _normalize_duplicate_sentence(str(row.get("text", "")))
        previous_index = best_by_sentence.get(normalized)
        if previous_index is None or _duplicate_preference(
            row, target_s, index
        ) < _duplicate_preference(rows[previous_index], target_s, previous_index):
            best_by_sentence[normalized] = index

    kept_indices = set(best_by_sentence.values())
    kept = [row for index, row in enumerate(rows) if index in kept_indices]
    dropped = [row for index, row in enumerate(rows) if index not in kept_indices]
    return kept, dropped


def _transcribe_cached(
    *,
    audio: np.ndarray,
    media_path: Path,
    cache_path: Path,
    config: DatasetPrepConfig,
    reporter: Any,
) -> tuple[Any, bool]:
    from .whisper_asr import load_word_timestamps, save_word_timestamps, transcribe

    if cache_path.is_file() and not config.overwrite:
        transcript = load_word_timestamps(cache_path)
        _log(reporter, f"Reused Whisper word timings from {cache_path.name}.")
        return transcript, True
    transcript = transcribe(
        audio,
        config.sample_rate,
        config.language,
        config.whisper_model,
        config.whisper_device,
        reporter,
    )
    save_word_timestamps(
        cache_path,
        transcript,
        {
            "source_media": _source_name(media_path),
            "model": config.whisper_model,
            "language": config.language,
            "sample_rate": config.sample_rate,
            "duration_s": round(audio.size / float(config.sample_rate), 6),
        },
    )
    return transcript, False


def _audio_filter_reason(
    audio: np.ndarray,
    duration_s: float,
    text: str,
    config: DatasetPrepConfig,
) -> tuple[str | None, Any]:
    metrics = analyze_audio_quality(
        audio,
        config.sample_rate,
        clipping_threshold=config.clipping_threshold,
        silence_threshold_dbfs=config.silence_threshold_dbfs,
        frame_ms=config.silence_frame_ms,
    )
    words_per_second = _word_count(text) / max(duration_s, 1e-9)
    if words_per_second < config.min_words_per_second:
        return "words_per_second_low", metrics
    if words_per_second > config.max_words_per_second:
        return "words_per_second_high", metrics
    if metrics.peak_dbfs < config.min_peak_dbfs:
        return "peak_too_low", metrics
    if metrics.clipping_ratio > config.max_clipping_ratio:
        return "clipping", metrics
    if config.max_silence_ratio is not None and metrics.silence_ratio > config.max_silence_ratio:
        return "silence_ratio", metrics
    return None, metrics


def _write_segment(
    output: Path,
    audio: np.ndarray,
    sample_rate: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, audio, sample_rate, subtype="PCM_16")


def _rank_reference_candidates(
    candidates: Sequence[dict[str, Any]],
    count: int,
    output_dir: Path,
    seed: int,
) -> list[str]:
    if count <= 0 or not candidates:
        return []
    finite_loudness = [float(item["row"]["lufs"]) for item in candidates if item["row"]["lufs"] is not None]
    median = float(np.median(finite_loudness)) if finite_loudness else -20.0
    rng = random.Random(seed)
    scored: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for item in candidates:
        row = item["row"]
        duration = float(row["duration_s"])
        lufs = float(row["lufs"]) if row["lufs"] is not None else -1000.0
        score = (
            0 if 6.0 <= duration <= 15.0 else 1,
            1 if item["clipped"] else 0,
            abs(lufs - median),
            0 if str(row["text"]).rstrip().endswith(".") else 1,
            -_word_count(str(row["text"])),
            rng.random(),
            str(row["id"]),
        )
        scored.append((score, item))
    scored.sort(key=lambda pair: pair[0])
    reference_dir = output_dir / "reference_candidates"
    reference_dir.mkdir(parents=True, exist_ok=True)
    exported: list[str] = []
    for _, item in scored[:count]:
        source = Path(item["path"])
        destination = reference_dir / source.name
        shutil.copy2(source, destination)
        exported.append(destination.relative_to(output_dir).as_posix())
    return exported


def _base_row(
    *,
    segment_id: str,
    relative_audio: str,
    text: str,
    duration_s: float,
    source: Path,
    source_start_s: float,
    source_end_s: float,
    config: DatasetPrepConfig,
    speaker: str,
    transcript_source: str,
    lufs: float,
    alignment_coverage: float | None = None,
    sentence_aligned: bool | None = None,
    peak_dbfs: float | None = None,
    clipping_ratio: float | None = None,
    silence_ratio: float | None = None,
) -> dict[str, Any]:
    row = {
        "id": segment_id,
        "audio": relative_audio,
        "text": text.strip(),
        "duration_s": round(float(duration_s), 6),
        "source_media": _source_name(source),
        "source_start_s": round(float(source_start_s), 6),
        "source_end_s": round(float(source_end_s), 6),
        "language": config.language,
        "speaker": speaker,
        "words": _word_count(text),
        "transcript_source": transcript_source,
        "lufs": round(float(lufs), 3),
    }
    if alignment_coverage is not None:
        row["alignment_coverage"] = round(float(alignment_coverage), 6)
    if sentence_aligned is not None:
        row["sentence_aligned"] = bool(sentence_aligned)
    if peak_dbfs is not None:
        row["peak_dbfs"] = round(float(peak_dbfs), 3)
    if clipping_ratio is not None:
        row["clipping_ratio"] = round(float(clipping_ratio), 8)
    if silence_ratio is not None:
        row["silence_ratio"] = round(float(silence_ratio), 6)
    return row


def run_dataset_prep(
    config: DatasetPrepConfig,
    reporter: Any = None,
    cancel_check: Callable[[], bool] | None = None,
) -> DatasetSummary:
    config.validate()
    requested_segmentation_mode = config.segmentation_mode
    segmentation_mode = config.resolved_segmentation_mode()
    reporter = reporter or _default_reporter(config.name)
    output_dir = Path(config.output_root).expanduser() / config.name
    manifest_path = output_dir / MANIFEST_FILENAME
    info_path = output_dir / DATASET_INFO_FILENAME
    preview_path = output_dir / PREVIEW_FILENAME
    if manifest_path.exists() and not config.overwrite:
        raise FileExistsError(
            f"Dataset already exists at {output_dir}; pass overwrite=True to rebuild its manifest"
        )
    if config.overwrite:
        for generated_name in ("segments", "reference_candidates"):
            generated_path = output_dir / generated_name
            if generated_path.is_dir():
                shutil.rmtree(generated_path)
    (output_dir / "segments").mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    filter_drop_counts: dict[str, int] = {
        reason: 0
        for reason in (
            "duration",
            "duration_after_trim",
            "word_count",
            "alignment_coverage",
            "sentence_boundary",
            "words_per_second_low",
            "words_per_second_high",
            "peak_too_low",
            "clipping",
            "silence_ratio",
            "empty_audio",
            "non_finite_loudness",
            "duplicate_sentence",
        )
    }
    alignment_files: list[dict[str, Any]] = []
    subtitle_stats = {
        "cues_total": 0,
        "cues_cleaned": 0,
        "cues_dropped": 0,
        "cues_merged": 0,
        "subtitle_segments": 0,
        "duplicate_sentences_dropped": 0,
    }
    used_keys: set[str] = set()
    started = time.monotonic()
    status = "running"

    _stage(reporter, "discover")
    _log(
        reporter,
        f"Segmentation mode: {segmentation_mode}"
        + (f" (requested {requested_segmentation_mode})" if requested_segmentation_mode != segmentation_mode else ""),
    )
    _log(reporter, "Discovering media, subtitle sidecars, and pre-segmented inputs ...")
    import_items = _discover_import_items(config, warnings)
    import_paths = {str(item.audio_path.resolve()).casefold() for item in import_items}
    media_files = [
        path
        for path in find_media_files(config.inputs, config.recursive)
        if str(Path(path).resolve()).casefold() not in import_paths
    ]
    warnings.extend(_orphan_subtitle_warnings(config, media_files))
    for raw in config.inputs:
        if not Path(raw).expanduser().exists():
            warnings.append(f"Input does not exist: {raw}")
    total_sources = len(import_items) + len(media_files)
    _log(
        reporter,
        f"Found {len(media_files)} media file(s), {len(import_items)} imported segment(s), "
        f"and {len(warnings)} warning(s).",
    )
    for warning in warnings:
        _log(reporter, f"Warning: {warning}")

    processed_sources = 0
    with manifest_path.open("w", encoding="utf-8", newline="\n") as manifest_handle:
        # Import already-segmented audio without recutting it.
        for item in import_items:
            if _cancelled(cancel_check) or (config.max_segments and len(rows) >= config.max_segments):
                status = "cancelled" if _cancelled(cancel_check) else "complete"
                break
            processed_sources += 1
            key = _safe_key(item.audio_path, used_keys)
            _update(reporter, processed_sources - 1, total_sources, f"Importing {item.audio_path.name}")
            try:
                audio = _load_audio(item.audio_path, config.sample_rate)
                duration_s = audio.size / float(config.sample_rate)
                words = _word_count(item.text)
                if not (config.min_s <= duration_s <= config.max_s):
                    _increment_reason(filter_drop_counts, "duration")
                    _log(reporter, f"Filtered {item.audio_path.name}: duration")
                    continue
                if not (config.min_words <= words <= config.max_words):
                    _increment_reason(filter_drop_counts, "word_count")
                    _log(reporter, f"Filtered {item.audio_path.name}: word_count")
                    continue
                quality_reason, quality = _audio_filter_reason(audio, duration_s, item.text, config)
                if quality_reason:
                    _increment_reason(filter_drop_counts, quality_reason)
                    _log(reporter, f"Filtered {item.audio_path.name}: {quality_reason}")
                    continue
                if config.loudness_normalize:
                    audio = normalize_loudness(audio, config.sample_rate, config.target_lufs)
                segment_id = f"{key}_0001"
                destination = output_dir / "segments" / f"{segment_id}.wav"
                _write_segment(destination, audio, config.sample_rate)
                lufs = measure_loudness_lufs(audio, config.sample_rate)
                row = _base_row(
                    segment_id=segment_id,
                    relative_audio=destination.relative_to(output_dir).as_posix(),
                    text=item.text,
                    duration_s=duration_s,
                    source=item.audio_path,
                    source_start_s=0.0,
                    source_end_s=duration_s,
                    config=config,
                    speaker=_speaker_for(config, item.audio_path, item.speaker),
                    transcript_source=item.transcript_source,
                    lufs=lufs,
                    peak_dbfs=quality.peak_dbfs,
                    clipping_ratio=quality.clipping_ratio,
                    silence_ratio=quality.silence_ratio,
                )
                append_manifest_row(manifest_handle, row)
                rows.append(row)
                candidates.append(
                    {
                        "path": destination,
                        "row": row,
                        "clipped": float(np.max(np.abs(audio), initial=0.0)) >= 0.9995,
                    }
                )
                sources.append(
                    {
                        "source_media": _source_name(item.audio_path),
                        "transcript_source": item.transcript_source,
                        "segments": 1,
                        "duration_s": round(duration_s, 6),
                        "filter_drop_counts": {"duplicate_sentence": 0},
                    }
                )
            except Exception as exc:
                warning = f"Could not import {item.audio_path}: {exc}; skipped"
                warnings.append(warning)
                _log(reporter, f"Warning: {warning}")

        segmentation_max_s = _reserve_segmentation_max(config)
        with tempfile.TemporaryDirectory(prefix="indextts_dataset_prep_") as work_dir_raw:
            work_dir = Path(work_dir_raw)
            for media_raw in media_files:
                if status == "cancelled" or _cancelled(cancel_check):
                    status = "cancelled"
                    break
                if config.max_segments and len(rows) >= config.max_segments:
                    break
                processed_sources += 1
                media_path = Path(media_raw)
                key = _safe_key(media_path, used_keys)
                _stage(reporter, "extract")
                _update(
                    reporter,
                    processed_sources - 1,
                    total_sources,
                    f"Extracting {media_path.name}",
                    {"phase": "extract", "file_i": processed_sources, "file_n": total_sources},
                )
                _log(reporter, f"[{processed_sources}/{total_sources}] Extracting {media_path}")
                source_row_count = len(rows)
                source_cues = 0
                source_cleaned_cues = 0
                source_merged = 0
                source_filter_counts: dict[str, int] = {}
                source_alignment_coverage: float | None = None
                source_effective_mode = segmentation_mode
                try:
                    media_info = probe_media(media_path)
                    if not media_info.has_audio:
                        raise RuntimeError("media has no audio stream")
                    decoded_path = work_dir / f"{key}.wav"
                    extract_audio(media_path, decoded_path, sample_rate=config.sample_rate, mono=True)
                    audio, decoded_sr = sf.read(decoded_path, dtype="float32", always_2d=False)
                    if decoded_sr != config.sample_rate:
                        raise RuntimeError(
                            f"decoded sample rate is {decoded_sr}, expected {config.sample_rate}"
                        )
                    if audio.ndim == 2:
                        audio = np.mean(audio, axis=1, dtype=np.float32)
                    audio = np.ascontiguousarray(audio, dtype=np.float32)
                    media_duration_ms = int(round(audio.size * 1000.0 / config.sample_rate))
                    if audio.size == 0:
                        raise RuntimeError("decoded audio is empty")

                    segments: list[Segment] = []
                    transcript_source = ""
                    sidecars = find_sidecar_subtitles(media_path)
                    transcript_path = find_sidecar_transcript(media_path)
                    selected_sidecar: Path | None = None
                    raw_cues: Sequence[Any] = []
                    if segmentation_mode != "whisper_only" and config.subtitle_policy != "whisper_only":
                        for candidate_raw in sidecars:
                            candidate = Path(candidate_raw)
                            try:
                                parsed = parse_subtitle_file(str(candidate))
                                if parsed:
                                    selected_sidecar = candidate
                                    raw_cues = parsed
                                    break
                            except Exception as exc:
                                warning = f"Could not parse subtitle {candidate}: {exc}"
                                warnings.append(warning)
                                _log(reporter, f"Warning: {warning}")

                    if selected_sidecar is not None:
                        _stage(reporter, "subtitles")
                        source_cues = len(raw_cues)
                        cleaned = clean_cues(
                            raw_cues,
                            remove_bracket_annotations=config.remove_bracket_annotations,
                            dedupe_rolling_captions=config.dedupe_rolling_captions,
                        )
                        source_cleaned_cues = len(cleaned)
                        cue_segments = merge_cues_into_sentences(
                            raw_cues,
                            max_gap_ms=config.max_gap_ms,
                            target_s=min(config.target_s, segmentation_max_s),
                            max_s=segmentation_max_s,
                            min_s=config.min_s,
                            remove_bracket_annotations=config.remove_bracket_annotations,
                            dedupe_rolling_captions=config.dedupe_rolling_captions,
                        )
                        cue_segments = _split_overlong(cue_segments, cleaned, segmentation_max_s)
                        source_merged = sum(
                            max(0, len(segment.source_cue_indices) - 1) for segment in cue_segments
                        )
                        sidecar_source = f"sidecar_{selected_sidecar.suffix.lstrip('.').lower()}"
                        if segmentation_mode == "sentence_aligned":
                            _stage(reporter, "whisper_alignment")
                            from .whisper_asr import align_caption_words

                            transcript, cache_reused = _transcribe_cached(
                                audio=audio,
                                media_path=media_path,
                                cache_path=output_dir / "whisper" / f"{key}.words.json",
                                config=config,
                                reporter=reporter,
                            )
                            caption = build_caption_transcript(cleaned)
                            alignment = align_caption_words(caption.words, transcript.words)
                            source_alignment_coverage = alignment.coverage
                            alignment_entry = {
                                "source_media": _source_name(media_path),
                                "caption_words": alignment.total_words,
                                "whisper_words": len(transcript.words),
                                "matched_caption_words": alignment.matched_words,
                                "coverage": round(alignment.coverage, 6),
                                "cache_reused": cache_reused,
                                "fallback_to_cue_boundaries": False,
                            }
                            alignment_files.append(alignment_entry)
                            if alignment.coverage < config.min_file_alignment_coverage:
                                source_effective_mode = "cue_boundaries"
                                alignment_entry["fallback_to_cue_boundaries"] = True
                                warning = (
                                    f"Caption/Whisper alignment for {media_path.name} covered "
                                    f"{alignment.coverage:.1%} of caption words (< "
                                    f"{config.min_file_alignment_coverage:.0%}); using cue boundaries."
                                )
                                warnings.append(warning)
                                _log(reporter, f"Warning: {warning}")
                                segments = cue_segments
                                transcript_source = sidecar_source
                            else:
                                segments = build_sentence_aligned_segments(
                                    caption,
                                    alignment.words,
                                    target_s=min(config.target_s, segmentation_max_s),
                                    max_s=segmentation_max_s,
                                    min_s=config.min_s,
                                    max_gap_ms=config.max_gap_ms,
                                )
                                transcript_source = sidecar_source + "+whisper_sentence_aligned"
                            _log(
                                reporter,
                                f"Aligned {alignment.matched_words}/{alignment.total_words} caption words "
                                f"({alignment.coverage:.1%}); produced {len(segments)} segments.",
                            )
                        else:
                            segments = cue_segments
                            transcript_source = sidecar_source
                            _log(
                                reporter,
                                f"Using {selected_sidecar.name}: {source_cues} cues -> {len(segments)} segments.",
                            )
                    elif (
                        segmentation_mode != "whisper_only"
                        and config.subtitle_policy == "sidecar_only"
                        and transcript_path is None
                    ):
                        warning = f"No sidecar subtitle found for {media_path}; skipped"
                        warnings.append(warning)
                        _log(reporter, f"Warning: {warning}")
                        continue
                    else:
                        _stage(reporter, "whisper")
                        source_effective_mode = "whisper_only"
                        transcript, _ = _transcribe_cached(
                            audio=audio,
                            media_path=media_path,
                            cache_path=output_dir / "whisper" / f"{key}.words.json",
                            config=config,
                            reporter=reporter,
                        )
                        segments = build_segments_from_words(
                            transcript.words,
                            target_s=min(config.target_s, segmentation_max_s),
                            max_s=segmentation_max_s,
                            min_s=config.min_s,
                            max_gap_ms=config.max_gap_ms,
                        )
                        if transcript_path is not None and config.subtitle_policy != "whisper_only":
                            provided_text = _read_text(Path(transcript_path)).strip()
                            segments = _segments_from_plain_transcript(provided_text, segments)
                            transcript_source = "sidecar_txt+whisper_aligned"
                        else:
                            transcript_source = "whisper"
                        _log(
                            reporter,
                            f"Whisper produced {len(transcript.words)} words and {len(segments)} segments.",
                        )

                    if not segments:
                        raise RuntimeError("transcript produced no usable timed segments")
                    segments = sorted(segments, key=lambda item: (item.start_ms, item.end_ms))
                    preliminary_count = len(segments)
                    energy = compute_energy_envelope(audio, config.sample_rate, hop_ms=10)
                    word_safe_boundaries = source_effective_mode == "sentence_aligned"
                    if word_safe_boundaries:
                        segments = apply_padding_and_limits(
                            segments,
                            config.pad_ms,
                            media_duration_ms,
                        )
                        segments = _snap_segments(
                            segments,
                            energy,
                            config,
                            protect_words=True,
                        )
                    else:
                        segments = _snap_segments(segments, energy, config)
                        segments = apply_padding_and_limits(
                            segments,
                            config.pad_ms,
                            media_duration_ms,
                        )
                    structural_drop_counts: dict[str, int] = {}
                    segments = filter_segments(
                        segments,
                        config.min_s,
                        config.max_s,
                        config.min_words,
                        config.max_words,
                        min_alignment_coverage=(
                            config.min_segment_alignment_coverage
                            if word_safe_boundaries
                            else None
                        ),
                        require_sentence_aligned=word_safe_boundaries,
                        reason_counts=structural_drop_counts,
                    )
                    for reason, count in structural_drop_counts.items():
                        source_filter_counts[reason] = source_filter_counts.get(reason, 0) + count
                        filter_drop_counts[reason] = filter_drop_counts.get(reason, 0) + count
                    filtered_count = preliminary_count - len(segments)
                    if filtered_count:
                        _log(
                            reporter,
                            f"Filtered {filtered_count} segment(s) by duration, words, or alignment coverage.",
                        )

                    _stage(reporter, "segments")
                    accepted_for_source = 0
                    for segment_index, segment in enumerate(segments, start=1):
                        if _cancelled(cancel_check):
                            status = "cancelled"
                            break
                        if config.max_segments and len(rows) >= config.max_segments:
                            break
                        piece, actual_start_s, actual_end_s = slice_audio(
                            audio,
                            config.sample_rate,
                            segment.start_ms / 1000.0,
                            segment.end_ms / 1000.0,
                        )
                        trim_start = 0
                        trim_end = piece.size
                        if config.trim_silence:
                            piece, (trim_start, trim_end) = trim_silence(
                                piece,
                                config.sample_rate,
                                config.trim_top_db,
                                pad_ms=50,
                                return_indices=True,
                            )
                        if piece.size == 0 or float(np.max(np.abs(piece), initial=0.0)) < 1e-6:
                            _increment_reason(source_filter_counts, "empty_audio")
                            _increment_reason(filter_drop_counts, "empty_audio")
                            continue
                        source_start_s = actual_start_s + trim_start / float(config.sample_rate)
                        source_end_s = actual_start_s + trim_end / float(config.sample_rate)
                        duration_s = piece.size / float(config.sample_rate)
                        if not config.min_s <= duration_s <= config.max_s + 1.0 / config.sample_rate:
                            _increment_reason(source_filter_counts, "duration_after_trim")
                            _increment_reason(filter_drop_counts, "duration_after_trim")
                            continue
                        quality_reason, quality = _audio_filter_reason(
                            piece,
                            duration_s,
                            segment.text,
                            config,
                        )
                        if quality_reason:
                            _increment_reason(source_filter_counts, quality_reason)
                            _increment_reason(filter_drop_counts, quality_reason)
                            continue
                        if config.loudness_normalize:
                            piece = normalize_loudness(piece, config.sample_rate, config.target_lufs)
                        lufs = measure_loudness_lufs(piece, config.sample_rate)
                        if not math.isfinite(lufs):
                            _increment_reason(source_filter_counts, "non_finite_loudness")
                            _increment_reason(filter_drop_counts, "non_finite_loudness")
                            continue
                        accepted_for_source += 1
                        segment_id = f"{key}_{accepted_for_source:04d}"
                        destination = output_dir / "segments" / f"{segment_id}.wav"
                        _write_segment(destination, piece, config.sample_rate)
                        row = _base_row(
                            segment_id=segment_id,
                            relative_audio=destination.relative_to(output_dir).as_posix(),
                            text=segment.text,
                            duration_s=duration_s,
                            source=media_path,
                            source_start_s=source_start_s,
                            source_end_s=source_end_s,
                            config=config,
                            speaker=_speaker_for(config, media_path),
                            transcript_source=transcript_source,
                            lufs=lufs,
                            alignment_coverage=segment.alignment_coverage,
                            sentence_aligned=segment.sentence_aligned,
                            peak_dbfs=quality.peak_dbfs,
                            clipping_ratio=quality.clipping_ratio,
                            silence_ratio=quality.silence_ratio,
                        )
                        append_manifest_row(manifest_handle, row)
                        rows.append(row)
                        candidates.append(
                            {
                                "path": destination,
                                "row": row,
                                "clipped": quality.clipping_ratio > 0.0,
                            }
                        )
                        _update(
                            reporter,
                            processed_sources - 1,
                            total_sources,
                            f"{media_path.name}: segment {segment_index}/{len(segments)}",
                            {
                                "phase": "segments",
                                "file_i": processed_sources,
                                "file_n": total_sources,
                                "segment_count": len(rows),
                                "total_audio_seconds": round(sum(r["duration_s"] for r in rows), 3),
                            },
                        )

                    source_rows = rows[source_row_count:]
                    sources.append(
                        {
                            "source_media": _source_name(media_path),
                            "media_duration_s": round(audio.size / config.sample_rate, 6),
                            "transcript_source": transcript_source,
                            "segmentation_mode": source_effective_mode,
                            "subtitle": _source_name(selected_sidecar) if selected_sidecar else None,
                            "alignment_coverage": (
                                round(source_alignment_coverage, 6)
                                if source_alignment_coverage is not None
                                else None
                            ),
                            "cues_total": source_cues,
                            "cues_cleaned": source_cleaned_cues,
                            "cues_dropped": source_cues - source_cleaned_cues,
                            "cues_merged": source_merged,
                            "segments": len(source_rows),
                            "segment_duration_s": round(sum(row["duration_s"] for row in source_rows), 6),
                            "filter_drop_counts": dict(sorted(source_filter_counts.items())),
                        }
                    )
                    subtitle_stats["cues_total"] += source_cues
                    subtitle_stats["cues_cleaned"] += source_cleaned_cues
                    subtitle_stats["cues_dropped"] += source_cues - source_cleaned_cues
                    subtitle_stats["cues_merged"] += source_merged
                    if selected_sidecar:
                        subtitle_stats["subtitle_segments"] += len(source_rows)
                    _log(
                        reporter,
                        f"Completed {media_path.name}: {len(source_rows)} segment(s), "
                        f"{sum(row['duration_s'] for row in source_rows) / 60.0:.2f} min.",
                    )
                except Exception as exc:
                    warning = f"Could not decode/process {media_path}: {exc}; skipped"
                    warnings.append(warning)
                    _log(reporter, f"Warning: {warning}")

    if config.drop_duplicate_sentences:
        rows, duplicate_rows = _deduplicate_sentence_rows(rows, config.target_s)
        duplicate_count = len(duplicate_rows)
        kept_ids = {str(row.get("id", "")) for row in rows}
        candidates = [
            candidate
            for candidate in candidates
            if str(candidate["row"].get("id", "")) in kept_ids
        ]
        duplicate_counts_by_source: dict[str, int] = {}
        for row in duplicate_rows:
            source_name = str(row.get("source_media", ""))
            duplicate_counts_by_source[source_name] = (
                duplicate_counts_by_source.get(source_name, 0) + 1
            )
            (output_dir / str(row["audio"])).unlink(missing_ok=True)
        filter_drop_counts["duplicate_sentence"] = duplicate_count
        subtitle_stats["duplicate_sentences_dropped"] = duplicate_count

        kept_rows_by_source: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            kept_rows_by_source.setdefault(str(row.get("source_media", "")), []).append(row)
        for source in sources:
            source_name = str(source.get("source_media", ""))
            source_rows = kept_rows_by_source.get(source_name, [])
            source_counts = dict(source.get("filter_drop_counts") or {})
            source_counts["duplicate_sentence"] = duplicate_counts_by_source.get(
                source_name, 0
            )
            source["filter_drop_counts"] = dict(sorted(source_counts.items()))
            source["segments"] = len(source_rows)
            if "segment_duration_s" in source:
                source["segment_duration_s"] = round(
                    sum(float(row["duration_s"]) for row in source_rows), 6
                )
        subtitle_stats["subtitle_segments"] = sum(
            int(source.get("segments", 0) or 0)
            for source in sources
            if source.get("subtitle")
        )
        if duplicate_count:
            write_manifest(manifest_path, rows)
        _log(
            reporter,
            f">> dropped {duplicate_count} duplicate sentence(s) "
            "(kept the best-aligned copy of each)",
        )
    else:
        for source in sources:
            source_counts = dict(source.get("filter_drop_counts") or {})
            source_counts.setdefault("duplicate_sentence", 0)
            source["filter_drop_counts"] = dict(sorted(source_counts.items()))

    if status == "running":
        status = "cancelled" if _cancelled(cancel_check) else "complete"
    _stage(reporter, "finalize")
    references = _rank_reference_candidates(
        candidates,
        config.export_reference_candidates,
        output_dir,
        config.seed,
    )
    stats = summarize_manifest(rows)
    alignment_caption_words = sum(int(item["caption_words"]) for item in alignment_files)
    alignment_matched_words = sum(int(item["matched_caption_words"]) for item in alignment_files)
    alignment_summary = {
        "caption_words": alignment_caption_words,
        "matched_caption_words": alignment_matched_words,
        "coverage": round(alignment_matched_words / alignment_caption_words, 6)
        if alignment_caption_words
        else None,
        "minimum_file_coverage": config.min_file_alignment_coverage,
        "minimum_segment_coverage": config.min_segment_alignment_coverage,
        "files": alignment_files,
    }
    sentence_aligned_count = sum(is_sentence_aligned_text(str(row.get("text", ""))) for row in rows)
    sentence_alignment = {
        "aligned_segments": sentence_aligned_count,
        "exception_segments": len(rows) - sentence_aligned_count,
        "aligned_fraction": round(sentence_aligned_count / len(rows), 6) if rows else 0.0,
        "aligned_percent": round(100.0 * sentence_aligned_count / len(rows), 3) if rows else 0.0,
    }
    info: dict[str, Any] = {
        "name": config.name,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_rate": config.sample_rate,
        "language": config.language,
        **stats,
        "words": stats["word_count"],
        "sources": sources,
        "source_files": [source["source_media"] for source in sources],
        "subtitle_stats": subtitle_stats,
        "segmentation": {
            "requested_mode": requested_segmentation_mode,
            "resolved_mode": segmentation_mode,
        },
        "alignment": alignment_summary,
        "sentence_alignment": sentence_alignment,
        "filter_drop_counts": dict(sorted(filter_drop_counts.items())),
        "warnings": warnings,
        "reference_candidates": references,
        "config": config.to_dict(),
        "manifest": MANIFEST_FILENAME,
        "preview": PREVIEW_FILENAME,
        "cache": {
            "directory": "cache",
            "index": "cache/index.jsonl",
            "manifest_rewrite_required": False,
        },
        "elapsed_s": round(time.monotonic() - started, 3),
    }
    atomic_write_json(info_path, info)
    write_preview_csv(preview_path, rows)
    _update(
        reporter,
        total_sources,
        total_sources,
        f"{status}: {len(rows)} segments, {stats['total_duration_minutes']:.2f} min",
        {
            "phase": status,
            "file_i": processed_sources,
            "file_n": total_sources,
            "segment_count": len(rows),
            "total_audio_seconds": stats["total_duration_s"],
        },
    )
    if hasattr(reporter, "finish"):
        reporter.finish()
    summary = DatasetSummary(
        name=config.name,
        output_dir=str(output_dir.resolve()),
        status=status,
        segment_count=len(rows),
        total_duration_s=float(stats["total_duration_s"]),
        word_count=int(stats["word_count"]),
        sources=sources,
        warnings=warnings,
        reference_candidates=references,
        manifest_path=str(manifest_path.resolve()),
        dataset_info_path=str(info_path.resolve()),
        duration_histogram=dict(stats["duration_histogram"]),
        subtitle_stats=subtitle_stats,
        alignment=alignment_summary,
        filter_drop_counts=dict(sorted(filter_drop_counts.items())),
    )
    _log(
        reporter,
        f"Dataset {config.name}: {summary.segment_count} segments, "
        f"{summary.total_duration_s / 60.0:.2f} minutes, status={summary.status}.",
    )
    return summary


__all__ = ["DatasetPrepConfig", "DatasetSummary", "run_dataset_prep"]
