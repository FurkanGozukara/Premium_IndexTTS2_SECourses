from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
from functools import partial
import hashlib
import io
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
from indextts.utils.subtitle_utils import AMBIGUOUS_SUBTITLE_EXTENSIONS, looks_like_subtitle_file
from indextts.utils.text_encoding import read_text_resilient

from .dataset_manifest import (
    DATASET_INFO_FILENAME,
    MANIFEST_FILENAME,
    PREVIEW_FILENAME,
    append_manifest_row,
    atomic_write_json,
    empty_dataset_reason,
    summarize_manifest,
    write_manifest,
    write_preview_csv,
)
from .audio_boundaries import (
    PAUSE_LOOKBACK_MS,
    PAUSE_RELATIVE_LIMIT,
    RangeLoudness,
    build_safe_sentence_segments,
    pause_phrase_spans,
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
    measure_edge_silence,
    measure_loudness_lufs,
    normalize_loudness,
    probe_media,
    slice_audio,
    trim_silence,
)
from .segmenter import (
    _unit_times_ms,
    _word_dict,
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
    SubtitleCue,
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
    ``boundary_mode`` keeps strict sentence edges by default or also accepts
    fragments whose missing sentence edges have sufficiently long word gaps.
    """

    name: str
    inputs: list[str]
    recursive: bool = True
    language: str = "EN"
    output_root: str = "datasets"
    subtitle_policy: str = "prefer_sidecar"
    whisper_model: str = "large-v3-int8-convrot"
    whisper_device: str = "cuda:0"
    align_with_whisper: bool = False
    segmentation_mode: str = field(default_factory=_default_segmentation_mode)
    # 14-second clips packed from whole sentences, never longer than 16 seconds: the
    # generation tab's automatic token budget reproduces the median clip length, and a
    # 16-second ceiling keeps every training clip inside the segment lengths it renders.
    target_s: float = 14.0
    min_s: float = 4.0
    max_s: float = 16.0
    max_gap_ms: int = 700
    boundary_mode: str = "sentence"
    min_pause_boundary_ms: int = 400
    pad_ms: int = 60
    snap_to_silence: bool = True
    snap_window_ms: int = 400
    min_edge_silence_ms: int = 30
    # Shares of packed clips aimed at one short sentence (about 6 s) and at a medium clip (about 10 s),
    # cut only between clear pauses. 0.25 and 0.25 measured better than 0 and 0 on the reference voice:
    # higher identity and style similarity at every prompt length, pauses closer to the speaker's,
    # a speaking rate that no longer needs correcting, and no change in word error.
    short_clip_fraction: float = 0.25
    medium_clip_fraction: float = 0.25
    trim_silence: bool = True
    trim_top_db: float = 40.0
    loudness_normalize: bool = True
    target_lufs: float = -20.0
    sample_rate: int = 24000
    min_words: int = 2
    max_words: int = 80
    min_file_alignment_coverage: float = 0.60
    min_segment_alignment_coverage: float = 0.70
    cue_fallback_enabled: bool = True
    cue_fallback_max_error: float = 0.15
    cue_fallback_check_boundary_words: bool = True
    cue_fallback_margin_ms: int = 40
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
        if self.boundary_mode not in {"sentence", "sentence_or_pause"}:
            raise ValueError(f"Unsupported boundary_mode: {self.boundary_mode}")
        if self.min_pause_boundary_ms < 0:
            raise ValueError("min_pause_boundary_ms must be zero or positive")
        if not 0 <= self.min_edge_silence_ms <= 500:
            raise ValueError("min_edge_silence_ms must be between zero and 500")
        if not 0.0 <= float(self.short_clip_fraction) <= 0.8:
            raise ValueError("short_clip_fraction must be between zero and 0.8")
        if not 0.0 <= float(self.medium_clip_fraction) <= 0.8:
            raise ValueError("medium_clip_fraction must be between zero and 0.8")
        if float(self.short_clip_fraction) + float(self.medium_clip_fraction) > 0.9:
            raise ValueError("short_clip_fraction and medium_clip_fraction together must not exceed 0.9")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not 0 < self.min_s <= self.target_s <= self.max_s:
            raise ValueError("Durations must satisfy 0 < min_s <= target_s <= max_s")
        if self.min_words < 0 or self.max_words < self.min_words:
            raise ValueError("Word limits are invalid")
        if self.max_segments < 0:
            raise ValueError("max_segments must be zero or positive")
        for name in ("min_file_alignment_coverage", "min_segment_alignment_coverage", "cue_fallback_max_error"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        for name in ("cue_fallback_enabled", "cue_fallback_check_boundary_words"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be true or false")
        if not 0 <= self.cue_fallback_margin_ms <= 1000:
            raise ValueError("cue_fallback_margin_ms must be between zero and 1000")
        if not 0.0 <= self.max_clipping_ratio <= 1.0:
            raise ValueError("max_clipping_ratio must be between zero and one")
        if not 0.0 < self.clipping_threshold <= 1.0:
            raise ValueError("clipping_threshold must be in (0, 1]")
        if self.max_silence_ratio is not None and float(self.max_silence_ratio) == 0.0:
            # 0 is the interface's blank field (Gradio renders blank as 0); a 0 limit would reject every clip.
            self.max_silence_ratio = None
        if self.max_silence_ratio is not None and not 0.0 <= self.max_silence_ratio <= 1.0:
            raise ValueError("max_silence_ratio must be between zero and one")
        if not 0.0 < self.min_words_per_second <= self.max_words_per_second:
            raise ValueError("Words/second limits are invalid")
        if self.silence_frame_ms <= 0:
            raise ValueError("silence_frame_ms must be positive")

    def resolved_segmentation_mode(self) -> str:
        if self.subtitle_policy == "whisper_only" or self.segmentation_mode == "whisper_only":
            return "whisper_only"
        if self.align_with_whisper:
            return "sentence_aligned"
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
    filter_keep_counts: dict[str, int] = field(default_factory=dict)
    # Why a finished preparation kept no clip (status "empty"); blank otherwise.
    empty_reason: str = ""

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


def _read_text(
    path: Path,
    warning_callback: Callable[[str], None] | None = None,
) -> str:
    return read_text_resilient(path, warning_callback=warning_callback)


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
    content = _read_text(path, warnings.append)
    for row in csv.reader(io.StringIO(content, newline=""), delimiter="|"):
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
    clip_extensions = {".wav", ".flac", ".aiff", ".aif", ".ogg"}
    for raw in config.inputs:
        path = Path(raw).expanduser()
        if path.is_file() and path.name.casefold() == "metadata.csv":
            metadata_files[str(path.resolve()).casefold()] = path.resolve()
            continue
        if path.is_dir():
            iterator = path.rglob("metadata.csv") if config.recursive else path.glob("metadata.csv")
            for metadata in iterator:
                metadata_files[str(metadata.resolve()).casefold()] = metadata.resolve()
        if path.is_file() and path.suffix.casefold() in clip_extensions:
            pair_candidates[str(path.resolve()).casefold()] = path.resolve()
        elif path.is_dir():
            iterator = path.rglob("*") if config.recursive else path.glob("*")
            for candidate in iterator:
                if candidate.is_file() and candidate.suffix.casefold() in clip_extensions:
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
        # A transcript next to a long recording is not an already-cut clip.
        # Leave it in media discovery so its text is aligned and segmented.
        try:
            if sf.info(audio).duration > config.max_s:
                continue
        except (OSError, RuntimeError):
            continue
        text = _read_text(transcript, warnings.append).strip()
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
            if candidate.suffix.casefold() in AMBIGUOUS_SUBTITLE_EXTENSIONS and not looks_like_subtitle_file(candidate):
                # Download metadata, recognizer side files and binary VobSub are not orphan captions.
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
    stem = path.stem
    replacement = re.search(r"[^A-Za-z0-9_-]", stem)
    if replacement is not None:
        base = stem[: replacement.start()].strip("_-") or "source"
        digest = hashlib.sha256(stem.encode("utf-8", errors="surrogatepass")).hexdigest()[:6]
        base = f"{base}_{digest}"
    else:
        base = stem.strip("_") or "source"
        if not re.search(r"[A-Za-z0-9]", base):
            base = "source"
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


def _shared_word_boundary(
    previous: Segment,
    current: Segment,
    energy: np.ndarray,
    config: DatasetPrepConfig,
) -> int:
    """Refine touching ASR word times within the two boundary words only."""
    previous_word = previous.word_timestamps[-1]
    current_word = current.word_timestamps[0]
    previous_end = int(round(float(previous_word["end_s"]) * 1000.0))
    current_start = int(round(float(current_word["start_s"]) * 1000.0))
    boundary = (previous_end + current_start) // 2
    if not config.snap_to_silence or config.snap_window_ms <= 0:
        return boundary

    # Whisper can assign the last word's release to the next word. Clamping
    # both clips to that shared timestamp makes silence snapping ineffective.
    # Search inside these two words, without reaching another word, and require
    # a quiet run instead of mistaking an isolated in-word dip for a pause.
    hop_ms = 10
    radius = config.snap_window_ms + config.pad_ms
    # Never move an end before its aligned final word: a plosive closure within
    # that word can be quieter than the real pause that follows it.
    low = max(boundary - radius, previous_end)
    high = min(boundary + radius, int(round(float(current_word["end_s"]) * 1000.0)) - 1)
    first = max(0, int(math.ceil(low / hop_ms)))
    stop = min(len(energy), int(math.floor(high / hop_ms)))
    window = np.asarray(energy[first:stop], dtype=np.float32)
    if window.size < 3:
        return boundary
    threshold = min(10.0 ** (config.silence_threshold_dbfs / 20.0), float(window.max()) * 0.1)
    quiet = window <= threshold
    edges = np.diff(np.pad(quiet.astype(np.int8), (1, 1)))
    runs = [
        (first + start, first + end)
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
        if end - start >= max(3, 2 * math.ceil(config.min_edge_silence_ms / hop_ms))
    ]
    if not runs:
        return boundary
    start, end = min(runs, key=lambda run: abs((run[0] + run[1]) * hop_ms / 2 - boundary))
    return int((start + end) * hop_ms // 2)


def _snap_segments(
    segments: Sequence[Segment],
    energy: np.ndarray,
    config: DatasetPrepConfig,
    *,
    protect_words: bool = False,
) -> list[Segment]:
    if not config.snap_to_silence and not protect_words:
        return [replace(segment) for segment in segments]
    snapped: list[Segment] = []
    for index, segment in enumerate(segments):
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
        if config.snap_to_silence:
            candidate = snap_boundaries_to_silence(
                segment,
                energy,
                hop_ms=10,
                window_ms=config.snap_window_ms,
                previous_end_ms=previous_end,
                next_start_ms=next_start,
                # Padding protects audio beyond imperfect word timestamps.
                start_upper_ms=segment.start_ms if protect_words else None,
                end_lower_ms=segment.end_ms if protect_words else None,
            )
        else:
            candidate = replace(segment)
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
            if (
                previous.end_ms >= current.start_ms
                and previous.word_timestamps
                and current.word_timestamps
            ):
                boundary = _shared_word_boundary(previous, current, energy, config)
                snapped[index - 1] = replace(previous, end_ms=boundary)
                snapped[index] = replace(current, start_ms=boundary)
    return snapped


def _align_plain_transcript(text: str, words: Sequence[Any], duration_ms: int) -> tuple[Any, Any]:
    """Keep supplied text while anchoring its words to actual recognized speech."""
    from .whisper_asr import align_caption_words

    caption = build_caption_transcript([SubtitleCue(1, 0, duration_ms, text)])
    return caption, align_caption_words(caption.words, words)


def _source_transcript(words: Sequence[Any]) -> Any:
    from .dataset_quality import TimedTranscript
    from .whisper_asr import _word_text, _word_times

    return TimedTranscript([
        {"text": _word_text(word), "start_s": _word_times(word)[0], "end_s": _word_times(word)[1]}
        for word in words
    ])


def _verified_cue_text(
    text: str, start_s: float, end_s: float, transcript: Any, config: DatasetPrepConfig,
) -> bool:
    """Use curation's spoken-form and edge-word checks before trusting cue timing."""
    from .speech_metrics import transcript_metrics

    try:
        margin = config.cue_fallback_margin_ms / 1000.0
        agreement = transcript_metrics(
            text, transcript.between(start_s - margin, end_s + margin), config.language)
    except ValueError:
        return False
    edges_match = agreement["start_matches"] and agreement["end_matches"]
    return (agreement["error_rate"] <= config.cue_fallback_max_error
            and (not config.cue_fallback_check_boundary_words or edges_match))


@dataclass
class _AlignedSidecar:
    path: Path
    cues: Sequence[Any]
    caption: Any
    alignment: Any
    verified_segments: list[Segment] | None = None


def _choose_aligned_sidecar(
    candidates: Sequence[tuple[Path, Sequence[Any]]], words: Sequence[Any], config: DatasetPrepConfig,
) -> tuple[_AlignedSidecar | None, list[dict[str, Any]]]:
    """Try preferred subtitles first; cue fallback retains only verified pairs."""
    from .whisper_asr import align_caption_words

    attempted: list[dict[str, Any]] = []
    low_coverage: list[_AlignedSidecar] = []
    for path, cues in candidates:
        cleaned = clean_cues(cues, remove_bracket_annotations=config.remove_bracket_annotations,
                             dedupe_rolling_captions=config.dedupe_rolling_captions)
        caption = build_caption_transcript(cleaned)
        alignment = align_caption_words(caption.words, words)
        attempted.append({"subtitle": _source_name(path), "coverage": round(alignment.coverage, 6)})
        choice = _AlignedSidecar(path, cues, caption, alignment)
        if alignment.coverage >= config.min_file_alignment_coverage:
            return choice, attempted
        low_coverage.append(choice)

    if not config.cue_fallback_enabled:
        return None, attempted
    transcript = _source_transcript(words)
    selected = None
    best_duration = 0
    for choice, attempt in zip(low_coverage, attempted):
        cleaned = clean_cues(choice.cues, remove_bracket_annotations=config.remove_bracket_annotations,
                             dedupe_rolling_captions=config.dedupe_rolling_captions)
        segments = merge_cues_into_sentences(
            choice.cues, max_gap_ms=config.max_gap_ms,
            target_s=min(config.target_s, _reserve_segmentation_max(config)),
            max_s=_reserve_segmentation_max(config), min_s=config.min_s,
            remove_bracket_annotations=config.remove_bracket_annotations,
            dedupe_rolling_captions=config.dedupe_rolling_captions,
        )
        segments = _split_overlong(segments, cleaned, _reserve_segmentation_max(config))
        verified = [segment for segment in segments if _verified_cue_text(
            segment.text, segment.start_ms / 1000, segment.end_ms / 1000, transcript, config)]
        attempt.update(verified_cue_segments=len(verified), rejected_cue_segments=len(segments) - len(verified))
        duration = sum(segment.duration_ms for segment in verified)
        if duration > best_duration:
            choice.verified_segments = verified
            selected, best_duration = choice, duration
    return selected, attempted


def _repacks_at_pauses(config: DatasetPrepConfig) -> bool:
    return bool(config.min_edge_silence_ms and config.snap_to_silence and config.boundary_mode == "sentence")


def _build_aligned_segments(
    caption: Any, words: Sequence[Any], energy: np.ndarray, config: DatasetPrepConfig,
    *, audio: np.ndarray, progress_cb: Callable[[str], None] | None = None,
    pause_relative_limit: float = PAUSE_RELATIVE_LIMIT, spans: Sequence[Any] | None = None,
    loudness: Any = None,
) -> tuple[list[Segment], bool, list[dict[str, Any]]]:
    if _repacks_at_pauses(config):
        segments, rejected = build_safe_sentence_segments(
            caption, words, energy, config, audio=audio, progress_cb=progress_cb,
            pause_relative_limit=pause_relative_limit, spans=spans, loudness=loudness)
        return segments, True, rejected
    maximum = _reserve_segmentation_max(config)
    return build_sentence_aligned_segments(
        caption, words, target_s=min(config.target_s, maximum), max_s=maximum,
        min_s=config.min_s, max_gap_ms=config.max_gap_ms, boundary_mode=config.boundary_mode,
        min_pause_boundary_ms=config.min_pause_boundary_ms,
    ), False, []


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


@dataclass
class _ClipCut:
    """One segment cut from its source and checked; ``reason`` names the failed check (None: kept)."""

    reason: str | None
    piece: np.ndarray | None = None
    source_start_s: float = 0.0
    source_end_s: float = 0.0
    duration_s: float = 0.0
    quality: Any = None
    edge_quality: dict[str, float] = field(default_factory=dict)
    lufs: float = float("nan")


def _cut_clip(
    audio: np.ndarray,
    segment: Segment,
    config: DatasetPrepConfig,
    *,
    verified_source: Any = None,
) -> _ClipCut:
    """Slice, trim, check and normalize one segment, exactly as every written clip is made."""

    piece, actual_start_s, _ = slice_audio(
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
        return _ClipCut("empty_audio")
    source_start_s = actual_start_s + trim_start / float(config.sample_rate)
    source_end_s = actual_start_s + trim_end / float(config.sample_rate)
    if verified_source is not None and not _verified_cue_text(
        segment.text, source_start_s, source_end_s, verified_source, config
    ):
        return _ClipCut("transcript_disagreement", source_start_s=source_start_s, source_end_s=source_end_s)
    duration_s = piece.size / float(config.sample_rate)
    if not config.min_s <= duration_s <= config.max_s + 1.0 / config.sample_rate:
        return _ClipCut("duration_after_trim")
    quality_reason, quality = _audio_filter_reason(piece, duration_s, segment.text, config)
    if quality_reason:
        return _ClipCut(quality_reason)
    if config.loudness_normalize:
        piece = normalize_loudness(piece, config.sample_rate, config.target_lufs)
    edge_quality = measure_edge_silence(piece, config.sample_rate, config.silence_threshold_dbfs)
    if config.min_edge_silence_ms and any(value < config.min_edge_silence_ms for value in edge_quality.values()):
        return _ClipCut(
            "unsafe_audio_boundary",
            source_start_s=source_start_s,
            source_end_s=source_end_s,
            edge_quality=edge_quality,
        )
    lufs = measure_loudness_lufs(piece, config.sample_rate)
    if not math.isfinite(lufs):
        return _ClipCut("non_finite_loudness")
    return _ClipCut(None, piece, source_start_s, source_end_s, duration_s, quality, edge_quality, lufs)


@dataclass
class _CutPlan:
    """One way to cut a source into clips, with the labels its manifest rows carry."""

    segments: list[Segment]
    transcript_source: str
    effective_mode: str
    boundary_method: str
    word_safe_boundaries: bool = False
    acoustic_repacked: bool = False
    verified_source: Any = None
    unresolved: list[dict[str, Any]] = field(default_factory=list)
    # The whole recording as one clip: its edges are final, so it skips padding, snapping and the
    # structural filter; trimming and every clip check still apply.
    precut: bool = False
    # Stretches of the transcript that cannot become a clip may also end at acoustic pauses, and such clips
    # need not start and end a sentence: "unpunctuated" (stretches without sentence punctuation) or
    # "unpunctuated_or_long" (also sentences longer than a clip); "" keeps sentence edges only.
    pause_phrases: str = ""


def _fixed_plan(plan: _CutPlan, settings: DatasetPrepConfig, pause_relative_limit: float, audio: np.ndarray) -> _CutPlan:
    """A plan whose segments do not depend on the pause settings (cue, word-group or whole-recording timing)."""

    return replace(plan, segments=list(plan.segments), unresolved=list(plan.unresolved))


def _aligned_plan(
    caption: Any,
    words: Sequence[Any],
    energy: np.ndarray,
    template: _CutPlan,
    settings: DatasetPrepConfig,
    pause_relative_limit: float,
    audio: np.ndarray,
) -> _CutPlan:
    """Sentence segments of an aligned transcript, cut at pauses found with ``settings``."""

    spans = None
    loudness = None
    if template.pause_phrases and _repacks_at_pauses(settings):
        spans = pause_phrase_spans(
            caption, words, energy, settings,
            gain_db=_envelope_gain_db(energy, settings), pause_relative_limit=pause_relative_limit,
            split_long_sentences=template.pause_phrases == "unpunctuated_or_long",
        )
        if settings.loudness_normalize:
            # Pause phrases give many more candidate clips than sentences; measure their gains in one pass.
            try:
                loudness = RangeLoudness(audio, settings.sample_rate)
            except ImportError:
                loudness = None
    segments, repacked, unresolved = _build_aligned_segments(
        caption, words, energy, settings, audio=audio, pause_relative_limit=pause_relative_limit, spans=spans,
        loudness=loudness)
    if template.pause_phrases:
        # Rows say whether a clip starts and ends a sentence or ends at a pause inside one.
        segments = [
            replace(segment, sentence_aligned=aligned, boundary="sentence" if aligned else "pause")
            for segment, aligned in ((item, is_sentence_aligned_text(item.text)) for item in segments)
        ]
    return replace(
        template,
        segments=segments,
        acoustic_repacked=repacked,
        unresolved=unresolved,
        boundary_method="acoustic_sentence_repack" if repacked else "silence_snap",
    )


def _envelope_gain_db(energy: np.ndarray, config: DatasetPrepConfig) -> float:
    """Approximate loudness-normalization gain of a recording, from its frames above -60 dBFS."""

    if not config.loudness_normalize:
        return 0.0
    frames = np.asarray(energy, dtype=np.float64).reshape(-1)
    active = frames[np.isfinite(frames) & (frames > 1e-3)]
    if not active.size:
        return 0.0
    return float(config.target_lufs) - 10.0 * math.log10(float(np.mean(np.square(active))))


_WHOLE_RECORDING_EDGE_MS = 1500


def _whole_recording_plan(
    text: str,
    words: Sequence[Any],
    media_duration_ms: int,
    config: DatasetPrepConfig,
    *,
    transcript_source: str,
    alignment_coverage: float | None = None,
) -> _CutPlan | None:
    """All of a short recording's speech as one clip, like a pre-cut clip with its transcript.

    Offered only when the timed speech fits one clip. The clip keeps the recording's own edges when at most
    1.5 seconds lie beyond the first or last recognized word (recognizers often end the last word early);
    longer music or silence there is left out beyond the silence snap window. Trimming and every clip check,
    including quiet audio at both edges, still decide whether it is kept.
    """

    text = str(text or "").strip()
    times = [_unit_times_ms(word) for word in words]
    if not text or not times:
        return None
    first_ms = min(start for start, _ in times)
    last_ms = max(end for _, end in times)
    if last_ms - first_ms > config.max_s * 1000.0:
        return None
    if not config.min_words <= _word_count(text) <= config.max_words:
        return None
    if alignment_coverage is not None and alignment_coverage < config.min_segment_alignment_coverage:
        return None
    margin = max(int(config.snap_window_ms), int(config.pad_ms))
    media_end = int(media_duration_ms)
    segment = Segment(
        start_ms=0 if first_ms <= _WHOLE_RECORDING_EDGE_MS else first_ms - margin,
        end_ms=media_end if media_end - last_ms <= _WHOLE_RECORDING_EDGE_MS else min(media_end, last_ms + margin),
        text=text,
        word_timestamps=[_word_dict(word) for word in words],
        alignment_coverage=alignment_coverage,
        sentence_aligned=is_sentence_aligned_text(text),
    )
    if segment.duration_ms <= 0:
        return None
    return _CutPlan([segment], transcript_source, "whole_recording", "whole_recording", precut=True)


def _alternative_plans(
    text_kind: str,
    *,
    transcript: Any,
    caption: Any,
    alignment: Any,
    energy: np.ndarray,
    media_duration_ms: int,
    config: DatasetPrepConfig,
) -> list[Callable[[DatasetPrepConfig, float, np.ndarray], _CutPlan]]:
    """Other strict ways to cut a source whose text is Whisper's or a plain TXT transcript.

    Whisper text: its own sentences, repacked at verified pauses exactly as a TXT transcript is (a stretch
    without punctuation or a sentence longer than a clip may also end at an acoustic pause), and the whole
    recording when all of its speech fits one clip. TXT text: the same repacking, where only a stretch without
    sentence punctuation may end at a pause (the transcript's own sentences are kept), and the whole
    recording. Subtitle sources have none.
    """

    factories: list[Callable[[DatasetPrepConfig, float, np.ndarray], _CutPlan]] = []
    whole: _CutPlan | None = None
    if text_kind == "whisper" and transcript is not None and transcript.words:
        whisper_caption, whisper_alignment = _align_plain_transcript(
            transcript.text, transcript.words, media_duration_ms)
        template = _CutPlan([], "whisper", "whisper_sentence_aligned", "acoustic_sentence_repack",
                            word_safe_boundaries=True, pause_phrases="unpunctuated_or_long")
        factories.append(partial(_aligned_plan, whisper_caption, whisper_alignment.words, energy, template))
        whole = _whole_recording_plan(
            transcript.text, transcript.words, media_duration_ms, config, transcript_source="whisper")
    elif text_kind == "txt" and caption is not None and alignment is not None:
        template = _CutPlan([], "sidecar_txt+whisper_sentence_aligned", "sentence_aligned",
                            "acoustic_sentence_repack", word_safe_boundaries=True, pause_phrases="unpunctuated")
        factories.append(partial(_aligned_plan, caption, alignment.words, energy, template))
        whole = _whole_recording_plan(
            caption.text, alignment.words, media_duration_ms, config,
            transcript_source="sidecar_txt", alignment_coverage=alignment.coverage)
    if whole is not None:
        factories.append(partial(_fixed_plan, whole))
    return factories


def _prepare_cut_segments(
    plan: _CutPlan,
    config: DatasetPrepConfig,
    energy: np.ndarray,
    media_duration_ms: int,
) -> tuple[list[Segment], dict[str, int], dict[str, int]]:
    """Sort, pad and snap a plan's segments as configured, then apply the structural filters."""

    segments = sorted(plan.segments, key=lambda item: (item.start_ms, item.end_ms))
    if plan.precut:
        return segments, {}, {}
    if plan.acoustic_repacked:
        pass  # These edges already include real quiet source audio.
    elif plan.word_safe_boundaries:
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
    drop_counts: dict[str, int] = {}
    keep_counts: dict[str, int] = {}
    segments = filter_segments(
        segments,
        config.min_s,
        config.max_s,
        config.min_words,
        config.max_words,
        min_alignment_coverage=(
            config.min_segment_alignment_coverage
            if plan.word_safe_boundaries
            else None
        ),
        require_sentence_aligned=plan.word_safe_boundaries and not plan.pause_phrases,
        boundary_mode=config.boundary_mode,
        reason_counts=drop_counts,
        keep_counts=keep_counts,
    )
    return segments, drop_counts, keep_counts


def _retained_seconds(
    audio: np.ndarray,
    segments: Sequence[Segment],
    config: DatasetPrepConfig,
    *,
    verified_source: Any = None,
    capacity: int | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> float:
    """Seconds of audio a plan would keep after every clip check, without writing anything."""

    kept = 0
    total = 0.0
    for segment in segments:
        if (capacity is not None and kept >= capacity) or _cancelled(cancel_check):
            break
        cut = _cut_clip(audio, segment, config, verified_source=verified_source)
        if cut.reason is None:
            kept += 1
            total += cut.duration_s
    return total


# A recording whose pauses carry background noise or music never reaches the fixed silence threshold, so no
# cut can be verified and the recording would keep no clip. For such a recording alone, a pause may instead
# sit up to 6 dB above its own noise floor (the quietest 5 % of its 10 ms frames between the first and last
# word), while nearby speech stays at least 6 dB louder. Loud speech (90th percentile) must be at least
# 15 dB above that floor; otherwise pauses cannot be told from soft speech and the configured threshold stays.
_NOISE_FLOOR_PERCENTILE = 5.0
_SPEECH_LEVEL_PERCENTILE = 90.0
_NOISE_FLOOR_MARGIN_DB = 6.0
_NOISE_FLOOR_MIN_SPEECH_DB = 15.0
_NOISE_FLOOR_PAUSE_RELATIVE_LIMIT = 10.0 ** (-6.0 / 20.0)


def _noise_floor_settings(
    config: DatasetPrepConfig,
    audio: np.ndarray,
    energy: np.ndarray,
    speech_ms: tuple[int, int] | None,
) -> tuple[DatasetPrepConfig, dict[str, float]] | None:
    """Settings whose silence threshold follows this recording's own noise floor.

    Returns None when the configured threshold already lies above that floor (a clean recording, where pauses
    can be verified as configured) or when speech is too close to the floor to tell pauses apart.
    """

    frames = np.asarray(energy, dtype=np.float64).reshape(-1)
    first, last = 0, frames.size
    if speech_ms is not None:
        first = max(0, min(frames.size, int(speech_ms[0]) // 10))
        last = max(first, min(frames.size, int(math.ceil(int(speech_ms[1]) / 10))))
    region = frames[first:last] if last - first >= 50 else frames
    region = region[np.isfinite(region)]
    if region.size < 50:
        return None
    levels = 20.0 * np.log10(np.maximum(region, 1e-9))
    floor_db = float(np.percentile(levels, _NOISE_FLOOR_PERCENTILE))
    speech_db = float(np.percentile(levels, _SPEECH_LEVEL_PERCENTILE))
    if speech_db - floor_db < _NOISE_FLOOR_MIN_SPEECH_DB:
        return None
    gain_db = 0.0
    if config.loudness_normalize:
        # Clip edges are checked after loudness normalization; measure the floor at that level.
        speech = audio[first * config.sample_rate // 100:last * config.sample_rate // 100]
        loudness = measure_loudness_lufs(speech if speech.size else audio, config.sample_rate)
        if math.isfinite(loudness):
            gain_db = float(config.target_lufs) - loudness
    threshold = round(floor_db + _NOISE_FLOOR_MARGIN_DB + gain_db, 1)
    if threshold <= float(config.silence_threshold_dbfs):
        return None
    return replace(config, silence_threshold_dbfs=threshold), {
        "noise_floor_dbfs": round(floor_db + gain_db, 1),
        "speech_level_dbfs": round(speech_db + gain_db, 1),
        "silence_threshold_dbfs": threshold,
    }


@dataclass
class _BoundaryFailure:
    """A source that kept no clip because none of its cuts had quiet audio at both edges."""

    media_path: Path
    key: str
    decoded_path: Path
    energy: np.ndarray
    media_duration_ms: int
    source_index: int
    factories: list[Callable[[DatasetPrepConfig, float, np.ndarray], _CutPlan]]
    branch_counts: dict[str, int]


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
    boundary: str | None = None,
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
    if boundary is not None:
        row["boundary"] = str(boundary)
    if peak_dbfs is not None:
        row["peak_dbfs"] = round(float(peak_dbfs), 3)
    if clipping_ratio is not None:
        row["clipping_ratio"] = round(float(clipping_ratio), 8)
    if silence_ratio is not None:
        row["silence_ratio"] = round(float(silence_ratio), 6)
    return row


# Everything a preparation writes into its dataset folder. A folder holding only these names, without a completed
# dataset_info.json, is what a canceled or failed preparation left behind.
_PREPARATION_ENTRIES = frozenset({
    MANIFEST_FILENAME, DATASET_INFO_FILENAME, PREVIEW_FILENAME, "segments", "reference_candidates", "whisper",
    "boundary_rejections.jsonl", "sentence_rejections.jsonl",
})


def unfinished_preparation(output_dir: str | Path) -> bool:
    """True when ``output_dir`` holds only an interrupted preparation's work files.

    Canceling a preparation left ``manifest.jsonl`` and ``segments/`` behind, and preparing the same name again
    stopped with "Dataset already exists; pass overwrite=True". Such a folder is rebuilt (its Whisper cache is
    kept); a completed dataset, or a folder with anything else in it, still needs **Overwrite dataset**.
    """

    folder = Path(output_dir)
    try:
        names = {entry.name for entry in folder.iterdir()}
    except OSError:
        return False
    if not names or not names <= _PREPARATION_ENTRIES:
        return False
    info_path = folder / DATASET_INFO_FILENAME
    try:
        info = json.loads(info_path.read_text(encoding="utf-8-sig")) if info_path.is_file() else {}
    except (OSError, ValueError):
        info = {}
    return str((info or {}).get("status", "")).strip().lower() != "complete"


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
    rebuild = bool(config.overwrite) or (manifest_path.exists() and unfinished_preparation(output_dir))
    if manifest_path.exists() and not rebuild:
        raise FileExistsError(
            f"Dataset already exists at {output_dir}; choose a new dataset name or tick Overwrite dataset "
            "to rebuild it"
        )
    if rebuild:
        for generated_name in ("segments", "reference_candidates"):
            generated_path = output_dir / generated_name
            if generated_path.is_dir():
                shutil.rmtree(generated_path)
    (output_dir / "segments").mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []

    def record_runtime_warning(message: str) -> None:
        warnings.append(message)
        _log(reporter, f"Warning: {message}")

    rows: list[dict[str, Any]] = []
    boundary_rejections: list[dict[str, Any]] = []
    sentence_rejections: list[dict[str, Any]] = []
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
            "unsafe_audio_boundary",
            "transcript_disagreement",
        )
    }
    filter_keep_counts: dict[str, int] = {"pause_boundary": 0}
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
    imported_speakers = {str(item.audio_path.resolve()).casefold(): item.speaker for item in import_items}
    discovery_inputs = list(config.inputs)
    if segmentation_mode == "whisper_only":
        # Metadata may be the only listing of the audio files. Keep that list,
        # but do not import its text when the user explicitly requested ASR text.
        discovery_inputs.extend(str(item.audio_path) for item in import_items)
        import_items = []
        _log(reporter, "Whisper-only selected: supplied subtitle, TXT, and metadata text will be ignored.")
    import_paths = {str(item.audio_path.resolve()).casefold() for item in import_items}
    media_files = [
        path
        for path in find_media_files(discovery_inputs, config.recursive)
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
    # Sources that raised an error or were skipped before cutting (named in the warnings).
    skipped_sources = 0
    # Sources re-cut by a second strict method or at pauses measured against their own noise floor.
    recovered_sources: list[dict[str, Any]] = []
    # Sources that kept no clip because no cut had quiet audio at both edges (see the last resort below).
    boundary_failures: list[_BoundaryFailure] = []

    def cut_and_write(
        plan: _CutPlan,
        settings: DatasetPrepConfig,
        audio: np.ndarray,
        media_path: Path,
        key: str,
        source_filter_counts: dict[str, int],
        recovery: str | None = None,
    ) -> int:
        """Cut a source's prepared plan, write every kept clip, and count each rejection."""

        nonlocal status
        accepted_for_source = 0
        for segment_index, segment in enumerate(plan.segments, start=1):
            if _cancelled(cancel_check):
                status = "cancelled"
                break
            if config.max_segments and len(rows) >= config.max_segments:
                break
            cut = _cut_clip(audio, segment, settings, verified_source=plan.verified_source)
            if cut.reason is not None:
                _increment_reason(source_filter_counts, cut.reason)
                _increment_reason(filter_drop_counts, cut.reason)
                if cut.reason == "transcript_disagreement":
                    _log(reporter, f"Skipped {media_path.name} cue after trimming: "
                         "transcript or boundary words no longer match the extracted range.")
                elif cut.reason == "unsafe_audio_boundary":
                    boundary_rejections.append({
                        "source_media": _source_name(media_path),
                        "source_start_s": cut.source_start_s,
                        "source_end_s": cut.source_end_s,
                        "text": segment.text,
                        "reason": "unsafe_audio_boundary",
                        **cut.edge_quality,
                    })
                continue
            accepted_for_source += 1
            segment_id = f"{key}_{accepted_for_source:04d}"
            destination = output_dir / "segments" / f"{segment_id}.wav"
            _write_segment(destination, cut.piece, settings.sample_rate)
            row = _base_row(
                segment_id=segment_id,
                relative_audio=destination.relative_to(output_dir).as_posix(),
                text=segment.text,
                duration_s=cut.duration_s,
                source=media_path,
                source_start_s=cut.source_start_s,
                source_end_s=cut.source_end_s,
                config=settings,
                speaker=_speaker_for(
                    settings, media_path, imported_speakers.get(str(media_path.resolve()).casefold(), "")),
                transcript_source=plan.transcript_source,
                lufs=cut.lufs,
                alignment_coverage=segment.alignment_coverage,
                sentence_aligned=segment.sentence_aligned,
                boundary=segment.boundary if plan.word_safe_boundaries else None,
                peak_dbfs=cut.quality.peak_dbfs,
                clipping_ratio=cut.quality.clipping_ratio,
                silence_ratio=cut.quality.silence_ratio,
            )
            row.update(cut.edge_quality)
            row["boundary_method"] = plan.boundary_method
            if plan.acoustic_repacked and getattr(segment, "length_aim", None):
                row["length_aim"] = str(segment.length_aim)
            if recovery:
                row["boundary_recovery"] = recovery
            append_manifest_row(manifest_handle, row)
            rows.append(row)
            candidates.append(
                {
                    "path": destination,
                    "row": row,
                    "clipped": cut.quality.clipping_ratio > 0.0,
                }
            )
            _update(
                reporter,
                processed_sources - 1,
                total_sources,
                f"{media_path.name}: segment {segment_index}/{len(plan.segments)}",
                {
                    "phase": "segments",
                    "file_i": processed_sources,
                    "file_n": total_sources,
                    "segment_count": len(rows),
                    "total_audio_seconds": round(sum(r["duration_s"] for r in rows), 3),
                },
            )
        return accepted_for_source

    def reset_source(
        media_path: Path,
        source_row_count: int,
        source_filter_counts: dict[str, int],
        source_filter_keep_counts: dict[str, int],
        branch_counts: Mapping[str, int],
    ) -> list[dict[str, Any]]:
        """Remove a source's clips, cut counts and rejections before it is cut again; returns its removed rows."""

        removed = rows[source_row_count:]
        removed_ids = {str(row.get("id", "")) for row in removed}
        for row in removed:
            (output_dir / str(row["audio"])).unlink(missing_ok=True)
        del rows[source_row_count:]
        candidates[:] = [item for item in candidates if str(item["row"].get("id", "")) not in removed_ids]
        for reason, count in source_filter_counts.items():
            filter_drop_counts[reason] = filter_drop_counts.get(reason, 0) - (count - int(branch_counts.get(reason, 0)))
        for reason, count in source_filter_keep_counts.items():
            filter_keep_counts[reason] = filter_keep_counts.get(reason, 0) - count
        source_filter_counts.clear()
        source_filter_counts.update(branch_counts)
        source_filter_keep_counts.clear()
        source_filter_keep_counts["pause_boundary"] = 0
        source_name = _source_name(media_path)
        boundary_rejections[:] = [item for item in boundary_rejections if item.get("source_media") != source_name]
        sentence_rejections[:] = [item for item in sentence_rejections if item.get("source_media") != source_name]
        if removed:
            manifest_handle.seek(0)
            manifest_handle.truncate()
            for row in rows:
                append_manifest_row(manifest_handle, row)
        return removed

    def apply_plan(
        plan: _CutPlan,
        settings: DatasetPrepConfig,
        structural: tuple[dict[str, int], dict[str, int]],
        audio: np.ndarray,
        media_path: Path,
        key: str,
        source_filter_counts: dict[str, int],
        source_filter_keep_counts: dict[str, int],
        recovery: str | None,
    ) -> int:
        """Count a prepared plan's structural drops and unresolved sentences, then cut and write it."""

        drop_counts, keep_counts = structural
        for reason, count in drop_counts.items():
            source_filter_counts[reason] = source_filter_counts.get(reason, 0) + count
            filter_drop_counts[reason] = filter_drop_counts.get(reason, 0) + count
        for reason, count in keep_counts.items():
            source_filter_keep_counts[reason] = source_filter_keep_counts.get(reason, 0) + count
            filter_keep_counts[reason] = filter_keep_counts.get(reason, 0) + count
        sentence_rejections.extend({"source_media": _source_name(media_path), **item} for item in plan.unresolved)
        return cut_and_write(plan, settings, audio, media_path, key, source_filter_counts, recovery=recovery)

    def best_plan(
        factories: Sequence[Callable[[DatasetPrepConfig, float, np.ndarray], _CutPlan]],
        settings: DatasetPrepConfig,
        pause_relative_limit: float,
        audio: np.ndarray,
        energy: np.ndarray,
        media_duration_ms: int,
        minimum_s: float,
        capacity: int | None,
    ) -> tuple[float, _CutPlan, tuple[dict[str, int], dict[str, int]]] | None:
        """The plan keeping the most audio after every clip check, if it keeps more than ``minimum_s``."""

        best: tuple[float, _CutPlan, tuple[dict[str, int], dict[str, int]]] | None = None
        for factory in factories:
            if _cancelled(cancel_check):
                return None
            plan = factory(settings, pause_relative_limit, audio)
            prepared, drop_counts, keep_counts = _prepare_cut_segments(plan, settings, energy, media_duration_ms)
            retained = _retained_seconds(
                audio, prepared, settings, verified_source=plan.verified_source,
                capacity=capacity, cancel_check=cancel_check,
            )
            if retained > (best[0] if best else minimum_s) + 1e-6:
                best = (retained, replace(plan, segments=prepared), (drop_counts, keep_counts))
        return best

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
                        "filter_keep_counts": {"pause_boundary": 0},
                    }
                )
            except Exception as exc:
                skipped_sources += 1
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
                source_filter_keep_counts: dict[str, int] = {"pause_boundary": 0}
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
                    energy = compute_energy_envelope(audio, config.sample_rate, hop_ms=10)

                    segments: list[Segment] = []
                    acoustic_repacked = False
                    transcript_source = ""
                    # For cutting this source again: the aligned transcript whose pauses were searched, and whose
                    # text it is ("whisper" or "txt" have other strict ways to cut; subtitle text has none).
                    aligned_inputs: tuple[Any, Sequence[Any]] | None = None
                    text_kind = ""
                    transcript: Any = None
                    caption: Any = None
                    alignment: Any = None
                    sidecars = find_sidecar_subtitles(media_path, language=config.language)
                    transcript_path = find_sidecar_transcript(media_path)
                    selected_sidecar: Path | None = None
                    raw_cues: Sequence[Any] = []
                    aligned_sidecar: _AlignedSidecar | None = None
                    verified_source = None
                    sidecar_attempts: list[dict[str, Any]] = []
                    parsed_sidecars: list[tuple[Path, Sequence[Any]]] = []
                    if segmentation_mode != "whisper_only" and config.subtitle_policy != "whisper_only":
                        for candidate_raw in sidecars:
                            candidate = Path(candidate_raw)
                            try:
                                parsed = parse_subtitle_file(
                                    str(candidate),
                                    warning_callback=record_runtime_warning,
                                )
                                if parsed:
                                    parsed_sidecars.append((candidate, parsed))
                                    if segmentation_mode != "sentence_aligned":
                                        break
                            except Exception as exc:
                                warning = f"Could not parse subtitle {candidate}: {exc}"
                                warnings.append(warning)
                                _log(reporter, f"Warning: {warning}")

                    if parsed_sidecars:
                        if segmentation_mode == "sentence_aligned":
                            _stage(reporter, "whisper_alignment")
                            transcript, cache_reused = _transcribe_cached(
                                audio=audio, media_path=media_path,
                                cache_path=output_dir / "whisper" / f"{key}.words.json",
                                config=config, reporter=reporter,
                            )
                            aligned_sidecar, sidecar_attempts = _choose_aligned_sidecar(
                                parsed_sidecars, transcript.words, config)
                            for attempt in sidecar_attempts:
                                _log(reporter, f"Subtitle {Path(attempt['subtitle']).name}: "
                                     f"{attempt['coverage']:.1%} word alignment")
                            if aligned_sidecar is None:
                                detail = ("No cue passed the configured transcript verification checks. "
                                          if config.cue_fallback_enabled else "Verified cue fallback is disabled. ")
                                raise ValueError(
                                    "No subtitle matched the recognized speech. " + detail +
                                    "Check the recording language/subtitle files "
                                    "or use Whisper-only transcription. Original subtitles were preserved."
                                )
                            selected_sidecar, raw_cues = aligned_sidecar.path, aligned_sidecar.cues
                        else:
                            selected_sidecar, raw_cues = parsed_sidecars[0]

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
                            assert aligned_sidecar is not None
                            caption, alignment = aligned_sidecar.caption, aligned_sidecar.alignment
                            source_alignment_coverage = alignment.coverage
                            alignment_entry = {
                                "source_media": _source_name(media_path),
                                "caption_words": alignment.total_words,
                                "whisper_words": len(transcript.words),
                                "matched_caption_words": alignment.matched_words,
                                "coverage": round(alignment.coverage, 6),
                                "cache_reused": cache_reused,
                                "fallback_to_cue_boundaries": False,
                                "subtitle_candidates": sidecar_attempts,
                            }
                            alignment_files.append(alignment_entry)
                            if aligned_sidecar.verified_segments is not None:
                                source_effective_mode = "verified_cue_boundaries"
                                alignment_entry["fallback_to_cue_boundaries"] = True
                                alignment_entry["cue_transcripts_verified"] = True
                                alignment_entry["cue_verification"] = {
                                    "max_error": config.cue_fallback_max_error,
                                    "check_boundary_words": config.cue_fallback_check_boundary_words,
                                    "margin_ms": config.cue_fallback_margin_ms,
                                }
                                verified_source = _source_transcript(transcript.words)
                                segments = aligned_sidecar.verified_segments
                                rejected_count = len(cue_segments) - len(segments)
                                source_filter_counts["transcript_disagreement"] = rejected_count
                                filter_drop_counts["transcript_disagreement"] += rejected_count
                                warning = (
                                    f"Caption/Whisper alignment for {media_path.name} covered "
                                    f"{alignment.coverage:.1%} of caption words; retained {len(segments)} "
                                    f"of {len(cue_segments)} cue segments after the configured transcript "
                                    "checks. Unverified pairs were skipped; original subtitles were preserved."
                                )
                                warnings.append(warning)
                                _log(reporter, f"Warning: {warning}")
                                transcript_source = sidecar_source + "+whisper_verified_cues"
                            else:
                                _stage(reporter, "audio_boundaries")
                                segments, acoustic_repacked, rejected_sentences = _build_aligned_segments(
                                    caption, alignment.words, energy, config, audio=audio,
                                    progress_cb=lambda message: _update(
                                        reporter, processed_sources - 1, total_sources, message,
                                        {"file_i": processed_sources, "file_n": total_sources},
                                    ),
                                )
                                aligned_inputs = (caption, alignment.words)
                                sentence_rejections.extend(
                                    {"source_media": _source_name(media_path), **item}
                                    for item in rejected_sentences
                                )
                                if acoustic_repacked:
                                    _log(reporter, f"Repacked complete sentences at verified pauses: {len(segments)} clips; "
                                         f"{len(rejected_sentences)} sentences could not form a safe group within the limits.")
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
                        skipped_sources += 1
                        warning = f"No sidecar subtitle found for {media_path}; skipped"
                        warnings.append(warning)
                        _log(reporter, f"Warning: {warning}")
                        continue
                    else:
                        if (segmentation_mode == "cue_boundaries" and requested_segmentation_mode != "auto"
                                and transcript_path is not None):
                            raise ValueError(
                                "Cue-boundary segmentation requires timed subtitles for a long recording. "
                                "Choose sentence_aligned or auto to align its TXT transcript, or whisper_only "
                                "to use recognized text. The selected cue-boundary mode was preserved."
                            )
                        _stage(reporter, "whisper")
                        source_effective_mode = "whisper_only"
                        transcript, _ = _transcribe_cached(
                            audio=audio,
                            media_path=media_path,
                            cache_path=output_dir / "whisper" / f"{key}.words.json",
                            config=config,
                            reporter=reporter,
                        )
                        if transcript_path is not None and segmentation_mode != "whisper_only":
                            provided_text = _read_text(
                                Path(transcript_path),
                                record_runtime_warning,
                            ).strip()
                            _log(reporter, f"Aligning supplied TXT words from {Path(transcript_path).name} "
                                 "to recognized speech before sentence segmentation.")
                            caption, alignment = _align_plain_transcript(
                                provided_text, transcript.words, media_duration_ms)
                            if alignment.coverage < config.min_file_alignment_coverage:
                                raise ValueError(
                                    f"TXT/Whisper word alignment covered only {alignment.coverage:.1%} "
                                    f"of {Path(transcript_path).name}; no proportional text assignment was used. "
                                    "Check that the transcript matches this recording and language, or use "
                                    "Whisper-only transcription. Original text was preserved."
                                )
                            source_effective_mode = "sentence_aligned"
                            source_alignment_coverage = alignment.coverage
                            alignment_files.append({
                                "source_media": _source_name(media_path),
                                "caption_words": alignment.total_words,
                                "whisper_words": len(transcript.words),
                                "matched_caption_words": alignment.matched_words,
                                "coverage": round(alignment.coverage, 6),
                                "fallback_to_cue_boundaries": False,
                                "transcript": _source_name(Path(transcript_path)),
                            })
                            segments, acoustic_repacked, rejected_sentences = _build_aligned_segments(
                                caption, alignment.words, energy, config, audio=audio,
                                progress_cb=lambda message: _update(
                                    reporter, processed_sources - 1, total_sources, message,
                                    {"file_i": processed_sources, "file_n": total_sources},
                                ),
                            )
                            aligned_inputs = (caption, alignment.words)
                            text_kind = "txt"
                            sentence_rejections.extend(
                                {"source_media": _source_name(media_path), **item}
                                for item in rejected_sentences
                            )
                            transcript_source = "sidecar_txt+whisper_sentence_aligned"
                        else:
                            segments = build_segments_from_words(
                                transcript.words, target_s=min(config.target_s, segmentation_max_s),
                                max_s=segmentation_max_s, min_s=config.min_s, max_gap_ms=config.max_gap_ms,
                            )
                            text_kind = "whisper"
                            transcript_source = "whisper"
                        _log(
                            reporter,
                            f"Whisper produced {len(transcript.words)} words and {len(segments)} segments.",
                        )

                    if not segments and not text_kind and aligned_inputs is None:
                        raise RuntimeError("transcript produced no usable timed segments")
                    primary = _CutPlan(
                        segments=list(segments),
                        transcript_source=transcript_source,
                        effective_mode=source_effective_mode,
                        boundary_method="acoustic_sentence_repack" if acoustic_repacked else "silence_snap",
                        word_safe_boundaries=source_effective_mode == "sentence_aligned",
                        acoustic_repacked=acoustic_repacked,
                        verified_source=verified_source,
                    )
                    # Counts the branch itself recorded (cue transcript checks) also hold for a re-cut.
                    branch_counts = dict(source_filter_counts)
                    preliminary_count = len(segments)
                    segments, structural_drop_counts, structural_keep_counts = _prepare_cut_segments(
                        primary, config, energy, media_duration_ms)
                    for reason, count in structural_drop_counts.items():
                        source_filter_counts[reason] = source_filter_counts.get(reason, 0) + count
                        filter_drop_counts[reason] = filter_drop_counts.get(reason, 0) + count
                    for reason, count in structural_keep_counts.items():
                        source_filter_keep_counts[reason] = (
                            source_filter_keep_counts.get(reason, 0) + count
                        )
                        filter_keep_counts[reason] = filter_keep_counts.get(reason, 0) + count
                    filtered_count = preliminary_count - len(segments)
                    if filtered_count:
                        _log(
                            reporter,
                            f"Filtered {filtered_count} segment(s) by duration, words, or alignment coverage.",
                        )

                    _stage(reporter, "segments")
                    cut_and_write(
                        replace(primary, segments=segments), config, audio, media_path, key, source_filter_counts)

                    # A source whose cuts lacked quiet audio at an edge can keep more with another strict cut:
                    # Whisper's own sentences repacked at verified pauses (as a TXT transcript is; a stretch that
                    # cannot become a clip also ends at acoustic pauses), or the whole recording when all of its
                    # speech fits one clip. A source that still keeps nothing is cut at pauses measured against
                    # its own noise floor. The first result stays unless another one keeps more audio after the
                    # same checks, so sources that already work are unchanged.
                    source_name = _source_name(media_path)
                    boundary_failed = bool(source_filter_counts.get("unsafe_audio_boundary")) or any(
                        item.get("source_media") == source_name for item in sentence_rejections)
                    kept_s = sum(float(row["duration_s"]) for row in rows[source_row_count:])
                    capacity = config.max_segments - source_row_count if config.max_segments else None
                    factories: list[Callable[[DatasetPrepConfig, float, np.ndarray], _CutPlan]] = []
                    if text_kind and (boundary_failed or not kept_s) and status != "cancelled":
                        factories = _alternative_plans(
                            text_kind, transcript=transcript, caption=caption, alignment=alignment,
                            energy=energy, media_duration_ms=media_duration_ms, config=config,
                        )
                    primary_factory = (
                        partial(_aligned_plan, aligned_inputs[0], aligned_inputs[1], energy,
                                replace(primary, segments=[]))
                        if aligned_inputs is not None
                        else partial(_fixed_plan, replace(primary))
                    )
                    choice = None
                    chosen_settings = config
                    noise_floor: dict[str, float] | None = None
                    if status != "cancelled" and (capacity is None or capacity > 0):
                        if factories:
                            choice = best_plan(factories, config, PAUSE_RELATIVE_LIMIT, audio, energy,
                                               media_duration_ms, kept_s, capacity)
                        if choice is None and not kept_s and boundary_failed:
                            timed = list(primary.segments) or [
                                Segment(*_unit_times_ms(word), "") for word in (
                                    alignment.words if alignment is not None
                                    else transcript.words if transcript is not None else ())]
                            adapted = _noise_floor_settings(config, audio, energy, (
                                min(item.start_ms for item in timed), max(item.end_ms for item in timed),
                            ) if timed else None)
                            if adapted is not None:
                                chosen_settings, noise_floor = adapted
                                choice = best_plan((primary_factory, *factories), chosen_settings,
                                                   _NOISE_FLOOR_PAUSE_RELATIVE_LIMIT, audio, energy,
                                                   media_duration_ms, 0.0, capacity)
                    source_recovery: dict[str, Any] | None = None
                    if choice is not None and status != "cancelled" and not _cancelled(cancel_check):
                        _, chosen_plan, structural = choice
                        removed = reset_source(media_path, source_row_count, source_filter_counts,
                                               source_filter_keep_counts, branch_counts)
                        apply_plan(chosen_plan, chosen_settings, structural, audio, media_path, key,
                                   source_filter_counts, source_filter_keep_counts,
                                   "noise_floor_pauses" if noise_floor else None)
                        transcript_source = chosen_plan.transcript_source
                        source_effective_mode = chosen_plan.effective_mode
                        recut_rows = rows[source_row_count:]
                        recut_s = sum(float(row["duration_s"]) for row in recut_rows)
                        source_recovery = {
                            "method": chosen_plan.boundary_method,
                            "pauses": "noise_floor" if noise_floor else "configured_threshold",
                            "first_pass_segments": len(removed),
                            "first_pass_duration_s": round(kept_s, 3),
                            "segments": len(recut_rows),
                            "duration_s": round(recut_s, 3),
                            **(noise_floor or {}),
                        }
                        recovered_sources.append({"source_media": source_name, **source_recovery})
                        if noise_floor:
                            warning = (
                                f"{media_path.name}: background noise or music kept every pause above the silence "
                                f"threshold ({config.silence_threshold_dbfs:g} dBFS), so its cuts were placed in "
                                f"pauses measured against its own noise floor ({noise_floor['noise_floor_dbfs']:g} "
                                f"dBFS; threshold {noise_floor['silence_threshold_dbfs']:g} dBFS): "
                                f"{len(recut_rows)} clip(s), {recut_s:.1f} s. Its clips include that background "
                                "sound; listen to a few before training."
                            )
                            warnings.append(warning)
                            _log(reporter, f"Warning: {warning}")
                        else:
                            method = (
                                "one clip of the whole recording" if chosen_plan.precut
                                else "complete sentences at verified pauses"
                                if all(row.get("boundary") != "pause" for row in recut_rows)
                                else "sentences and pause-delimited phrases at verified pauses"
                            )
                            _log(
                                reporter,
                                f"Re-cut {media_path.name} as {method}: {len(recut_rows)} clip(s), {recut_s:.1f} s "
                                f"(the first cut kept {len(removed)} clip(s), {kept_s:.1f} s).",
                            )
                    elif not primary.segments and len(rows) == source_row_count and not boundary_failed:
                        # Nothing to cut, and not for want of a pause (too little speech): skipped as before.
                        raise RuntimeError(
                            "transcript produced no usable timed segments (its speech is shorter than the "
                            f"minimum clip length of {config.min_s:g} s, or has too few words)")
                    elif boundary_failed and len(rows) == source_row_count and status != "cancelled":
                        boundary_failures.append(_BoundaryFailure(
                            media_path=media_path,
                            key=key,
                            decoded_path=decoded_path,
                            energy=energy,
                            media_duration_ms=media_duration_ms,
                            source_index=len(sources),
                            factories=[primary_factory, *factories],
                            branch_counts=branch_counts,
                        ))

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
                            "filter_keep_counts": dict(sorted(source_filter_keep_counts.items())),
                            **({"boundary_recovery": source_recovery} if source_recovery else {}),
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
                    skipped_sources += 1
                    warning = f"Could not decode/process {media_path}: {exc}; skipped"
                    warnings.append(warning)
                    _log(reporter, f"Warning: {warning}")

            if not rows and boundary_failures and status != "cancelled" and not _cancelled(cancel_check):
                # Last resort, only when the preparation would otherwise keep nothing: no recording had a cut
                # with quiet audio at both edges, even against its own noise floor (continuous speech, or noise
                # as loud as soft speech). Cut those recordings once more without that check, as "Minimum quiet
                # audio at cut edges" = 0 would; every other check still applies and the clips are labeled.
                relaxed = replace(config, min_edge_silence_ms=0)
                _stage(reporter, "segments")
                _log(
                    reporter,
                    f"No clip had quiet audio at both cut edges; cutting {len(boundary_failures)} recording(s) "
                    "again without that check so the dataset is not empty.",
                )
                relaxed_sources = 0
                for failure in boundary_failures:
                    if _cancelled(cancel_check):
                        status = "cancelled"
                        break
                    capacity = config.max_segments - len(rows) if config.max_segments else None
                    if capacity is not None and capacity <= 0:
                        break
                    audio, _ = sf.read(failure.decoded_path, dtype="float32", always_2d=False)
                    if audio.ndim == 2:
                        audio = np.mean(audio, axis=1, dtype=np.float32)
                    audio = np.ascontiguousarray(audio, dtype=np.float32)
                    choice = best_plan(failure.factories, relaxed, PAUSE_RELATIVE_LIMIT, audio, failure.energy,
                                       failure.media_duration_ms, 0.0, capacity)
                    if choice is None or _cancelled(cancel_check):
                        continue
                    _, chosen_plan, structural = choice
                    summary_entry = sources[failure.source_index]
                    first_pass_counts = dict(summary_entry.get("filter_drop_counts") or {})
                    source_filter_counts = dict(first_pass_counts)
                    source_filter_keep_counts = dict(summary_entry.get("filter_keep_counts") or {})
                    row_count = len(rows)
                    reset_source(failure.media_path, row_count, source_filter_counts, source_filter_keep_counts,
                                 failure.branch_counts)
                    apply_plan(chosen_plan, relaxed, structural, audio, failure.media_path, failure.key,
                               source_filter_counts, source_filter_keep_counts, "unverified_edges")
                    recut_rows = rows[row_count:]
                    recut_s = sum(float(row["duration_s"]) for row in recut_rows)
                    summary_entry.update(
                        segments=len(recut_rows),
                        segment_duration_s=round(recut_s, 6),
                        filter_drop_counts=dict(sorted(source_filter_counts.items())),
                        filter_keep_counts=dict(sorted(source_filter_keep_counts.items())),
                        first_pass_filter_drop_counts=dict(sorted(first_pass_counts.items())),
                    )
                    if not recut_rows:
                        continue
                    relaxed_sources += 1
                    source_recovery = {
                        "method": chosen_plan.boundary_method,
                        "pauses": "unverified",
                        "first_pass_segments": 0,
                        "first_pass_duration_s": 0.0,
                        "segments": len(recut_rows),
                        "duration_s": round(recut_s, 3),
                    }
                    summary_entry.update(
                        transcript_source=chosen_plan.transcript_source,
                        segmentation_mode=chosen_plan.effective_mode,
                        boundary_recovery=source_recovery,
                    )
                    if summary_entry.get("subtitle"):
                        subtitle_stats["subtitle_segments"] += len(recut_rows)
                    recovered_sources.append({"source_media": _source_name(failure.media_path), **source_recovery})
                    _log(reporter, f"Cut {failure.media_path.name} without the quiet-edge check: "
                                   f"{len(recut_rows)} clip(s), {recut_s:.1f} s.")
                if relaxed_sources:
                    warning = (
                        f"No clip had quiet audio at both cut edges, so {len(rows)} clip(s) from {relaxed_sources} "
                        "recording(s) were cut at sentence and word boundaries without that check (as with Minimum "
                        "quiet audio at cut edges = 0). Their edges may clip a breath or the end of a word: listen "
                        "to a few before training, or use recordings with clearer pauses."
                    )
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
        # A finished preparation that kept no clip is "empty", not complete: there is nothing to cache or
        # train on, and preparing the same name again rebuilds it without Overwrite dataset.
        status = "cancelled" if _cancelled(cancel_check) else ("complete" if rows else "empty")
    empty_reason = ""
    if status == "empty":
        empty_reason = empty_dataset_reason(
            filter_drop_counts,
            total_sources=total_sources,
            skipped_sources=skipped_sources,
            unresolved_sentences=len(sentence_rejections),
        )
        warning = f"No usable clips: {empty_reason}."
        warnings.append(warning)
        _log(reporter, f"Warning: {warning}")
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
    sentence_aligned_count = sum(
        bool(row["sentence_aligned"])
        if "sentence_aligned" in row
        else is_sentence_aligned_text(str(row.get("text", "")))
        for row in rows
    )
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
        "filter_keep_counts": dict(sorted(filter_keep_counts.items())),
        "audio_boundaries": {
            "algorithm": "shared_pause_sentence_repack_v2",
            "applies_to": "automatically_cut_segments",
            "minimum_quiet_ms": config.min_edge_silence_ms,
            "pause_lookback_ms": PAUSE_LOOKBACK_MS,
            "short_clip_fraction": float(config.short_clip_fraction),
            "medium_clip_fraction": float(config.medium_clip_fraction),
            # How many packed clips were actually cut at each aim; a short or medium aim that found no
            # clear pause on both edges was packed at the target length instead.
            "length_aims": {aim: sum(1 for row in rows if row.get("length_aim") == aim) for aim in ("short", "medium", "target")},
            "threshold_dbfs": config.silence_threshold_dbfs,
            "rejected_segments": len(boundary_rejections),
            "rejections": "boundary_rejections.jsonl",
            "unresolved_sentences": len(sentence_rejections),
            "sentence_rejections": "sentence_rejections.jsonl",
        },
        # Recordings cut a second way because their first cuts lacked quiet audio at an edge.
        "boundary_recovery": {
            "sources": recovered_sources,
            "noise_floor_segments": sum(1 for row in rows if row.get("boundary_recovery") == "noise_floor_pauses"),
            "unverified_edge_segments": sum(1 for row in rows if row.get("boundary_recovery") == "unverified_edges"),
        },
        **({"empty_reason": empty_reason} if empty_reason else {}),
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
    write_manifest(output_dir / "boundary_rejections.jsonl", boundary_rejections)
    write_manifest(output_dir / "sentence_rejections.jsonl", sentence_rejections)
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
        filter_keep_counts=dict(sorted(filter_keep_counts.items())),
        empty_reason=empty_reason,
    )
    _log(
        reporter,
        f"Dataset {config.name}: {summary.segment_count} segments, "
        f"{summary.total_duration_s / 60.0:.2f} minutes, status={summary.status}.",
    )
    return summary


__all__ = ["DatasetPrepConfig", "DatasetSummary", "run_dataset_prep"]
