"""Pause statistics of a dataset's recordings and the pause settings they imply.

A trained voice copies how its speaker pauses, but the app also inserts silence at
the joins between generated segments and can cap over-long pauses. Both settings
are best taken from the speaker's own recordings: this module measures every
internal pause of the training clips (the same gate as the speech metrics: a run of
at least 120 ms below -40 dBFS between the first and last loud frame), splits them
into pauses at sentence boundaries and pauses inside sentences, and recommends

* **Sentence pause**: the median pause the speaker leaves between two sentences,
  used for the join between two generated segments that end a sentence;
* **Maximum pause**: the length that only the speaker's rarest long pauses exceed,
  used as the ceiling for pauses inside the finished audio.

Which of a clip's pauses sit at sentence boundaries is not known without an
aligner, so the longest ``sentences - 1`` pauses of a clip with several sentences
are taken as its sentence pauses. Measurements are cached beside the dataset
(``analysis/pause_cache.json``) by file size and modification time. CPU only.
"""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from indextts.utils.atomic_json import read_json_retry, write_json_atomic
from indextts.utils.pause_cap import find_internal_pauses


PAUSE_PROFILE_VERSION = 1
PAUSE_CACHE_FILENAME = "pause_cache.json"
PERCENTILES = (10, 25, 50, 75, 90, 95)
# The join between two generated sentences gets the speaker's median sentence pause.
SENTENCE_PAUSE_PERCENTILE = 50
# Pauses longer than the speaker's own 90th percentile are the ones a trained voice adds.
MAX_PAUSE_PERCENTILE = 90
MIN_SENTENCE_PAUSES = 20  # fewer measured sentence pauses than this fall back to the longest pauses
MIN_RECOMMENDED_MAX_PAUSE_MS = 200
MAX_RECOMMENDED_MAX_PAUSE_MS = 2000
_SENTENCE_END_RE = re.compile(r"(?<=[.!?…。！？])\s+")
_WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


def _audio_path(dataset_dir: str | Path, record: Mapping[str, Any]) -> Path:
    audio = Path(str(record.get("audio") or ""))
    return audio if audio.is_absolute() else Path(dataset_dir) / audio


def sentence_count(text: str) -> int:
    """Sentences in a transcript (ends at . ! ? and their CJK forms), at least one."""

    from indextts.training.dataset_profile import spoken_text

    parts = [part for part in _SENTENCE_END_RE.split(spoken_text(text).strip()) if _WORD_RE.search(part)]
    return max(1, len(parts))


def _round10(value: float) -> int:
    return int(round(float(value) / 10.0)) * 10


def _stats(values: Sequence[float], digits: int = 1) -> dict[str, float | int]:
    clean = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if clean.size == 0:
        return {"count": 0}
    result: dict[str, float | int] = {
        "count": int(clean.size),
        "mean": round(float(clean.mean()), digits),
        "min": round(float(clean.min()), digits),
        "max": round(float(clean.max()), digits),
    }
    for percent in PERCENTILES:
        result[f"p{percent:02d}"] = round(float(np.percentile(clean, percent)), digits)
    return result


def measure_clip_pauses(path: str | Path) -> dict[str, Any] | None:
    """Internal pauses of one recording in milliseconds (longest first) plus its speech span."""

    import soundfile as sf

    try:
        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except (OSError, RuntimeError, ValueError):
        return None
    if audio.size == 0:
        return None
    mono = audio.mean(axis=1)
    pauses = find_internal_pauses(mono, int(sample_rate))
    durations = sorted((int(round((end - start) * 1000.0 / sample_rate)) for start, end in pauses), reverse=True)
    if pauses:
        span_ms = (pauses[-1][1] - pauses[0][0]) * 1000.0 / sample_rate
    else:
        span_ms = 0.0
    return {
        "duration_ms": int(round(len(mono) * 1000.0 / sample_rate)),
        "pauses_ms": durations,
        "pause_ms": int(sum(durations)),
        "speech_span_hint_ms": int(round(span_ms)),
    }


class PauseCache:
    """Per-clip pause measurements stored beside the dataset, keyed by file size and modification time."""

    def __init__(self, dataset_dir: str | Path) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.path = self.dataset_dir / "analysis" / PAUSE_CACHE_FILENAME
        loaded = read_json_retry(self.path, None)
        self._entries: dict[str, dict[str, Any]] = dict(loaded) if isinstance(loaded, dict) else {}
        self._dirty = False

    def metrics_of(self, record: Mapping[str, Any]) -> dict[str, Any] | None:
        path = _audio_path(self.dataset_dir, record)
        try:
            stat = path.stat()
        except OSError:
            return None
        key = str(record.get("id") or path.name)
        entry = self._entries.get(key)
        if isinstance(entry, dict) and entry.get("size") == stat.st_size and entry.get("mtime") == int(stat.st_mtime):
            metrics = entry.get("metrics")
            return dict(metrics) if isinstance(metrics, dict) else None
        metrics = measure_clip_pauses(path)
        self._entries[key] = {"metrics": metrics, "size": stat.st_size, "mtime": int(stat.st_mtime)}
        self._dirty = True
        return metrics

    def save(self) -> None:
        if not self._dirty:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(self.path, self._entries, indent=0, ensure_ascii=False)
            self._dirty = False
        except OSError:
            pass


def recommend_pauses(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Sentence pause and maximum pause (ms) from a pause profile; empty when nothing was measured."""

    sentence = profile.get("sentence_pauses_ms") or {}
    every = profile.get("all_pauses_ms") or {}
    if int(every.get("count", 0) or 0) == 0:
        return {}
    key_sentence = f"p{SENTENCE_PAUSE_PERCENTILE:02d}"
    key_max = f"p{MAX_PAUSE_PERCENTILE:02d}"
    if int(sentence.get("count", 0) or 0) >= MIN_SENTENCE_PAUSES:
        sentence_pause = _round10(sentence[key_sentence])
        max_pause = _round10(sentence[key_max])
        source = "sentence boundaries"
    else:
        # Single-sentence datasets have no measurable sentence boundary: the speaker's longer pauses stand in.
        sentence_pause = _round10(every.get("p75", every[key_sentence]))
        max_pause = _round10(every[key_max])
        source = "the longest pauses inside sentences"
    max_pause = max(MIN_RECOMMENDED_MAX_PAUSE_MS, min(MAX_RECOMMENDED_MAX_PAUSE_MS, max_pause))
    sentence_pause = max(0, min(sentence_pause, max_pause))
    return {
        "sentence_pause_ms": int(sentence_pause),
        "max_pause_ms": int(max_pause),
        "sentence_pause_source": source,
        "sentence_pause_percentile": SENTENCE_PAUSE_PERCENTILE,
        "max_pause_percentile": MAX_PAUSE_PERCENTILE,
    }


def build_pause_profile(
    dataset_dir: str | Path,
    records: Sequence[Mapping[str, Any]],
    *,
    cache: PauseCache | None = None,
    max_clips: int | None = None,
) -> dict[str, Any] | None:
    """Measure the pauses of the given manifest rows; None when no recording could be read."""

    root = Path(dataset_dir)
    cache = cache or PauseCache(root)
    rows = [row for row in records if row.get("audio")]
    if max_clips is not None and max_clips > 0:
        rows = rows[: int(max_clips)]
    all_pauses: list[int] = []
    sentence_pauses: list[int] = []
    within_pauses: list[int] = []
    longest: list[int] = []
    fractions: list[float] = []
    per_clip_counts: list[int] = []
    clips = 0
    multi_sentence_clips = 0
    for row in rows:
        metrics = cache.metrics_of(row)
        if not metrics:
            continue
        clips += 1
        pauses = [int(value) for value in metrics.get("pauses_ms") or []]
        duration_ms = int(metrics.get("duration_ms") or 0)
        sentences = sentence_count(str(row.get("text") or ""))
        boundaries = max(0, sentences - 1)
        if boundaries:
            multi_sentence_clips += 1
        sentence_pauses.extend(pauses[:boundaries])
        within_pauses.extend(pauses[boundaries:])
        all_pauses.extend(pauses)
        per_clip_counts.append(len(pauses))
        if pauses:
            longest.append(pauses[0])
        if duration_ms > 0:
            fractions.append(sum(pauses) / duration_ms)
    cache.save()
    if clips == 0:
        return None
    profile: dict[str, Any] = {
        "version": PAUSE_PROFILE_VERSION,
        "clips": clips,
        "clips_with_multiple_sentences": multi_sentence_clips,
        "pauses_per_clip": _stats(per_clip_counts, digits=2),
        "pause_time_fraction": _stats(fractions, digits=4),
        "all_pauses_ms": _stats(all_pauses, digits=0),
        "sentence_pauses_ms": _stats(sentence_pauses, digits=0),
        "within_sentence_pauses_ms": _stats(within_pauses, digits=0),
        "longest_pause_per_clip_ms": _stats(longest, digits=0),
    }
    profile["recommendation"] = recommend_pauses(profile)
    return profile


def describe_pause_profile(profile: Mapping[str, Any]) -> str:
    """One line for logs and the adapter panel."""

    recommendation = profile.get("recommendation") or {}
    sentence = profile.get("sentence_pauses_ms") or {}
    every = profile.get("all_pauses_ms") or {}
    fraction = profile.get("pause_time_fraction") or {}
    parts = [f"{int(profile.get('clips', 0))} clips measured"]
    if int(every.get("count", 0) or 0):
        parts.append(f"{int(every['count'])} pauses, median {float(every.get('p50', 0.0)):.0f} ms")
    if int(sentence.get("count", 0) or 0):
        parts.append(f"sentence pauses median {float(sentence.get('p50', 0.0)):.0f} ms (p90 {float(sentence.get('p90', 0.0)):.0f} ms)")
    if fraction.get("mean") is not None:
        parts.append(f"{float(fraction['mean']) * 100:.1f} percent of clip time is pause")
    if recommendation:
        parts.append(
            f"recommended sentence pause {int(recommendation['sentence_pause_ms'])} ms and maximum pause "
            f"{int(recommendation['max_pause_ms'])} ms from {recommendation.get('sentence_pause_source', 'the recordings')}"
        )
    return "; ".join(parts)


__all__ = [
    "MAX_PAUSE_PERCENTILE",
    "PAUSE_CACHE_FILENAME",
    "PAUSE_PROFILE_VERSION",
    "SENTENCE_PAUSE_PERCENTILE",
    "PauseCache",
    "build_pause_profile",
    "describe_pause_profile",
    "measure_clip_pauses",
    "recommend_pauses",
    "sentence_count",
]
