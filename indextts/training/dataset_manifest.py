from __future__ import annotations

import csv
import json
import math
import os
import threading
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from indextts.utils.atomic_json import replace_with_retry, write_json_atomic


MANIFEST_FILENAME = "manifest.jsonl"
DATASET_INFO_FILENAME = "dataset_info.json"
PREVIEW_FILENAME = "preview.csv"
CACHE_INDEX_RELATIVE_PATH = Path("cache") / "index.jsonl"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomic JSON write that survives a concurrent reader (Windows sharing violations are retried)."""

    write_json_atomic(path, payload, indent=2, default=_json_default, fsync=True)


def write_manifest(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Each writer has its own temporary file: two trainings caching one dataset at once
    # must not replace each other's half-written index.
    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False, default=_json_default) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        replace_with_retry(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def append_manifest_row(handle: Any, row: Mapping[str, Any]) -> None:
    handle.write(json.dumps(dict(row), ensure_ascii=False, default=_json_default) + "\n")
    handle.flush()


def load_manifest(path_or_dataset: str | Path) -> list[dict[str, Any]]:
    path = Path(path_or_dataset)
    if path.is_dir():
        path = path / MANIFEST_FILENAME
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = line.strip()
            if not value:
                continue
            try:
                row = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Manifest row {line_number} is not an object: {path}")
            rows.append(row)
    return rows


def write_cache_index(dataset_dir: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Write feature locations separately so the immutable audio manifest stays stable."""

    path = Path(dataset_dir) / CACHE_INDEX_RELATIVE_PATH
    write_manifest(path, rows)
    return path


def load_cache_index(dataset_dir: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(dataset_dir) / CACHE_INDEX_RELATIVE_PATH
    return {str(row["id"]): row for row in load_manifest(path) if row.get("id") is not None}


def iter_dataset_records(
    dataset_dir: str | Path,
    *,
    include_cache: bool = True,
) -> Iterator[dict[str, Any]]:
    cache = load_cache_index(dataset_dir) if include_cache else {}
    for manifest_row in load_manifest(dataset_dir):
        row = dict(manifest_row)
        cached = cache.get(str(row.get("id")))
        if cached:
            row["cache"] = {key: value for key, value in cached.items() if key != "id"}
        yield row


DURATION_BUCKETS = ("<3s", "3-6s", "6-9s", "9-12s", "12-15s", ">15s")


def duration_histogram(durations_s: Sequence[float]) -> dict[str, int]:
    counts = dict.fromkeys(DURATION_BUCKETS, 0)
    for duration in durations_s:
        value = float(duration)
        if value < 3:
            counts["<3s"] += 1
        elif value < 6:
            counts["3-6s"] += 1
        elif value < 9:
            counts["6-9s"] += 1
        elif value < 12:
            counts["9-12s"] += 1
        elif value <= 15:
            counts["12-15s"] += 1
        else:
            counts[">15s"] += 1
    return counts


def summarize_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    durations = [float(row.get("duration_s", 0.0) or 0.0) for row in rows]
    total = float(sum(durations))
    word_count = sum(int(row.get("words", 0) or 0) for row in rows)
    return {
        "segment_count": len(rows),
        "total_duration_s": round(total, 6),
        "total_duration_minutes": round(total / 60.0, 6),
        "mean_duration_s": round(total / len(rows), 6) if rows else 0.0,
        "min_duration_s": round(min(durations), 6) if durations else 0.0,
        "max_duration_s": round(max(durations), 6) if durations else 0.0,
        "word_count": word_count,
        "duration_histogram": duration_histogram(durations),
    }


def write_preview_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "audio",
        "duration_s",
        "words",
        "speaker",
        "language",
        "transcript_source",
        "text",
        "source_media",
        "source_start_s",
        "source_end_s",
        "lufs",
    ]
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


# Plain-language names for dataset_info.json filter_drop_counts, used when a dataset has no clip.
DROP_REASON_TEXT = {
    "alignment_coverage": "transcript words not found in the speech",
    "clipping": "clipped (distorted) audio",
    "duplicate_sentence": "duplicate sentences",
    "duration": "outside the minimum and maximum clip length",
    "duration_after_trim": "outside the clip length range after trimming silence",
    "empty_audio": "no audio",
    "non_finite_loudness": "silent audio",
    "peak_too_low": "too quiet (minimum peak)",
    "sentence_boundary": "not complete sentences",
    "silence_ratio": "mostly silence",
    "transcript_disagreement": "subtitle text that disagreed with the speech",
    "unsafe_audio_boundary": "no quiet audio at a cut edge",
    "word_count": "outside the minimum and maximum word count",
    "words_per_second_high": "too many words for their length (wrong language or transcript?)",
    "words_per_second_low": "too few words for their length (wrong language or transcript?)",
}


def empty_dataset_reason(
    filter_drop_counts: Mapping[str, Any],
    *,
    total_sources: int,
    skipped_sources: int = 0,
    unresolved_sentences: int = 0,
) -> str:
    """Why a preparation kept no clip, from its drop counts and skipped sources."""

    if not total_sources:
        return "no media files or pre-segmented clips were found in the inputs"
    reasons = sorted(
        ((int(count or 0), str(reason)) for reason, count in filter_drop_counts.items() if count),
        key=lambda item: (-item[0], item[1]),
    )
    if skipped_sources >= total_sources and not reasons and not unresolved_sentences:
        return f"{skipped_sources} source(s) could not be processed (see the warnings)"
    parts = [f"no clip passed the checks in {total_sources} source(s)"]
    if reasons:
        parts.append("dropped clips: " + ", ".join(
            f"{count} {DROP_REASON_TEXT.get(reason, reason.replace('_', ' '))}" for count, reason in reasons[:5]))
    if unresolved_sentences:
        parts.append(f"{unresolved_sentences} sentence(s) had no clear pause to cut at")
    if skipped_sources:
        parts.append(f"{skipped_sources} source(s) could not be processed (see the warnings)")
    return "; ".join(parts)


def empty_dataset_message(dataset_dir: str | Path) -> str:
    """The error for a dataset without clips: what its preparation dropped and what to do next."""

    root = Path(dataset_dir)
    info: Any = {}
    try:
        info = json.loads((root / DATASET_INFO_FILENAME).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        info = {}
    if not isinstance(info, dict):
        info = {}
    reason = str(info.get("empty_reason") or "").strip()
    if not reason and isinstance(info.get("sources"), list):
        # Datasets prepared before "empty_reason" existed still record their drop counts.
        boundaries = info.get("audio_boundaries") if isinstance(info.get("audio_boundaries"), dict) else {}
        reason = empty_dataset_reason(
            info.get("filter_drop_counts") if isinstance(info.get("filter_drop_counts"), dict) else {},
            total_sources=len(info["sources"]),
            unresolved_sentences=int(boundaries.get("unresolved_sentences") or 0),
        )
    message = f"Dataset '{root.name}' has no clips to cache or train on ({MANIFEST_FILENAME} is empty or missing)."
    if reason:
        message += f" Its preparation kept no clip: {reason}."
    return message + (
        " Prepare it again in LoRA Dataset Preparation (its warnings name each problem), "
        "or select another dataset."
    )


def validate_manifest_row(row: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "id",
        "audio",
        "text",
        "duration_s",
        "source_media",
        "source_start_s",
        "source_end_s",
        "language",
        "speaker",
        "words",
        "transcript_source",
        "lufs",
    )
    for key in required:
        if key not in row:
            errors.append(f"missing field: {key}")
    try:
        duration = float(row.get("duration_s", 0.0))
        if not math.isfinite(duration) or duration <= 0:
            errors.append("duration_s must be finite and positive")
    except (TypeError, ValueError):
        errors.append("duration_s must be numeric")
    if not str(row.get("text", "")).strip():
        errors.append("text must not be empty")
    return errors


__all__ = [
    "CACHE_INDEX_RELATIVE_PATH",
    "DATASET_INFO_FILENAME",
    "DROP_REASON_TEXT",
    "MANIFEST_FILENAME",
    "PREVIEW_FILENAME",
    "append_manifest_row",
    "atomic_write_json",
    "duration_histogram",
    "empty_dataset_message",
    "empty_dataset_reason",
    "iter_dataset_records",
    "load_cache_index",
    "load_manifest",
    "summarize_manifest",
    "validate_manifest_row",
    "write_cache_index",
    "write_manifest",
    "write_preview_csv",
]
