"""Median pitch and pace of a dataset's speaker, and reference clips chosen to match them.

A reference clip sets the pitch and pace baseline of every generation, so an atypical clip (a low,
slow intro sentence, say) pulls the cloned voice away from how the person usually sounds. The
quality ranking in :mod:`reference_selection` stays first; among its best candidates this module
prefers the clip whose median pitch and words per second are nearest the speaker's medians.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from indextts.utils.atomic_json import read_json_retry, write_json_atomic

from .reference_selection import training_reference_priority

PITCH_SAMPLE_RATE = 16000
PITCH_FMIN_HZ = 60.0
PITCH_FMAX_HZ = 400.0
PROFILE_SAMPLE_CLIPS = 48  # clips measured for the speaker's median pitch
REFERENCE_SHORTLIST = 24  # best-quality candidates measured before choosing a reference
PITCH_TOLERANCE = 0.10  # one unit of distance: ten percent of pitch
PACE_TOLERANCE = 0.15  # one unit of distance: fifteen percent of words per second
DURATION_TOLERANCE_S = 5.0  # one unit of distance: five seconds away from the reference target


def _audio_path(dataset_dir: str | Path, record: Mapping[str, Any]) -> Path:
    value = Path(str(record.get("audio") or ""))
    return value if value.is_absolute() else Path(dataset_dir) / value


def _stable_order(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()


def clip_pace(record: Mapping[str, Any]) -> float | None:
    """Words per second of a clip from its transcript and duration."""
    try:
        duration = float(record.get("duration_s") or 0.0)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(duration) or duration <= 0:
        return None
    try:
        words = int(record.get("words") or 0)
    except (TypeError, ValueError):
        words = 0
    if words <= 0:
        words = len(str(record.get("text") or "").split())
    return words / duration if words > 0 else None


def measure_clip_pitch(path: str | Path) -> float | None:
    """Median voiced fundamental frequency of a recording in Hz (pyin), or None when nothing is voiced."""
    import librosa
    import numpy as np
    import soundfile as sf

    try:
        array, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except (OSError, RuntimeError, ValueError):
        return None
    wave = array.mean(axis=1)
    if sample_rate != PITCH_SAMPLE_RATE:
        wave = librosa.resample(wave, orig_sr=sample_rate, target_sr=PITCH_SAMPLE_RATE)
    if wave.size < PITCH_SAMPLE_RATE // 4:
        return None
    f0, voiced, _ = librosa.pyin(wave, sr=PITCH_SAMPLE_RATE, fmin=PITCH_FMIN_HZ, fmax=PITCH_FMAX_HZ,
                                 frame_length=1024, hop_length=256)
    values = f0[voiced & np.isfinite(f0)]
    return float(np.median(values)) if len(values) else None


class PitchCache:
    """Per-clip median pitch stored beside the dataset, keyed by file size and modification time."""

    def __init__(self, dataset_dir: str | Path) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.path = self.dataset_dir / "analysis" / "pitch_cache.json"
        loaded = read_json_retry(self.path, None)
        self._entries: dict[str, dict[str, Any]] = dict(loaded) if isinstance(loaded, dict) else {}
        self._dirty = False

    def pitch_of(self, record: Mapping[str, Any]) -> float | None:
        path = _audio_path(self.dataset_dir, record)
        try:
            stat = path.stat()
        except OSError:
            return None
        key = str(record.get("id") or path.name)
        entry = self._entries.get(key)
        if isinstance(entry, dict) and entry.get("size") == stat.st_size and entry.get("mtime") == int(stat.st_mtime):
            value = entry.get("f0_median_hz")
            return float(value) if isinstance(value, (int, float)) else None
        value = measure_clip_pitch(path)
        self._entries[key] = {"f0_median_hz": value, "size": stat.st_size, "mtime": int(stat.st_mtime)}
        self._dirty = True
        return value

    def save(self) -> None:
        if not self._dirty:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(self.path, self._entries, indent=1, ensure_ascii=False)
            self._dirty = False
        except OSError:
            pass


def speaker_profile(dataset_dir: str | Path, records: Sequence[Mapping[str, Any]], *,
                    sample_clips: int = PROFILE_SAMPLE_CLIPS, cache: PitchCache | None = None) -> dict[str, Any]:
    """Median pitch (over a deterministic sample of clips) and median pace (over every clip) of one speaker."""
    cache = cache or PitchCache(dataset_dir)
    paces = sorted(value for value in (clip_pace(row) for row in records) if value is not None)
    ordered = sorted(records, key=lambda row: _stable_order(str(row.get("id", ""))))
    pitches: list[float] = []
    measured = 0
    for row in ordered:
        if measured >= sample_clips:
            break
        value = cache.pitch_of(row)
        measured += 1
        if value is not None:
            pitches.append(value)
    cache.save()
    pitches.sort()
    return {
        "pitch_median_hz": pitches[len(pitches) // 2] if pitches else None,
        "pace_median_wps": paces[len(paces) // 2] if paces else None,
        "pitch_clips": len(pitches),
        "pace_clips": len(paces),
    }


def typicality_distance(pitch: float | None, pace: float | None, profile: Mapping[str, Any],
                        duration: float | None = None) -> float:
    """How far a clip sits from the speaker's medians (and the duration target), in tolerance units."""
    total = 0.0
    median_pitch = profile.get("pitch_median_hz")
    if pitch and median_pitch:
        total += (math.log(pitch / float(median_pitch)) / PITCH_TOLERANCE) ** 2
    median_pace = profile.get("pace_median_wps")
    if pace and median_pace:
        total += ((pace - float(median_pace)) / (PACE_TOLERANCE * float(median_pace))) ** 2
    if duration is not None:
        from .reference_selection import AUTO_REFERENCE_TARGET_SECONDS
        if math.isfinite(duration) and duration > 0:
            total += ((duration - AUTO_REFERENCE_TARGET_SECONDS) / DURATION_TOLERANCE_S) ** 2
        else:
            total += 3.0 ** 2  # unknown duration: keep the known ones ahead
    return round(math.sqrt(total), 6)


def choose_typical_reference(dataset_dir: str | Path, records: Sequence[Mapping[str, Any]], *,
                             shortlist: int = REFERENCE_SHORTLIST, profile: Mapping[str, Any] | None = None,
                             cache: PitchCache | None = None) -> dict[str, Any] | None:
    """Among the best-quality candidates, the clip nearest the speaker's median pitch and pace.

    Returns ``{"record", "pitch_hz", "pace_wps", "distance", "profile", "candidates"}`` or None when
    no candidate has audio. Candidates are ranked by :func:`training_reference_priority` (transcript
    quality, then duration near the target) and only the first ``shortlist`` are measured.
    """
    existing = [row for row in records if row.get("audio") and _audio_path(dataset_dir, row).is_file()]
    if not existing:
        return None
    ranked = sorted(existing, key=lambda row: (*training_reference_priority(row), str(row["id"])))
    # Quality first: only clips of the best transcript and boundary class compete, ordered by duration
    # nearness; typicality then decides among them.
    best_class = training_reference_priority(ranked[0])[:2]
    pool = [row for row in ranked if training_reference_priority(row)[:2] == best_class][: max(1, int(shortlist))]
    cache = cache or PitchCache(dataset_dir)
    profile = dict(profile) if profile is not None else speaker_profile(dataset_dir, existing, cache=cache)
    candidates = []
    for row in pool:
        pitch = cache.pitch_of(row)
        pace = clip_pace(row)
        try:
            duration = float(row.get("duration_s") or 0.0)
        except (TypeError, ValueError):
            duration = 0.0
        candidates.append({"id": str(row["id"]), "pitch_hz": pitch, "pace_wps": pace, "duration_s": duration,
                           "distance": typicality_distance(pitch, pace, profile, duration), "record": row})
    cache.save()
    best = min(candidates, key=lambda item: (item["distance"], candidates.index(item)))
    return {"record": best["record"], "pitch_hz": best["pitch_hz"], "pace_wps": best["pace_wps"],
            "distance": best["distance"], "profile": profile,
            "candidates": [{key: value for key, value in item.items() if key != "record"} for item in candidates]}


EXPRESSIVE_SHORTLIST = 120  # clean clips measured before choosing the liveliest one
EXPRESSIVE_MIN_SECONDS = 8.0
EXPRESSIVE_MAX_SECONDS = 16.0
EXPRESSIVE_SAMPLE_RATE = 16000


def measure_clip_expressiveness(path: str | Path) -> dict[str, float] | None:
    """Pitch variability (semitones) and loudness variability (dB) of a recording, or None without voiced audio."""
    import librosa
    import numpy as np
    import soundfile as sf

    try:
        array, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except (OSError, RuntimeError, ValueError):
        return None
    wave = array.mean(axis=1)
    if sample_rate != EXPRESSIVE_SAMPLE_RATE:
        wave = librosa.resample(wave, orig_sr=sample_rate, target_sr=EXPRESSIVE_SAMPLE_RATE)
    if wave.size < EXPRESSIVE_SAMPLE_RATE // 2:
        return None
    frame, hop = 1024, 160
    f0 = librosa.yin(wave, fmin=70, fmax=350, sr=EXPRESSIVE_SAMPLE_RATE, frame_length=frame, hop_length=hop)
    rms = librosa.feature.rms(y=wave, frame_length=frame, hop_length=hop)[0]
    count = min(len(f0), len(rms))
    f0, rms = f0[:count], rms[:count]
    level = 20.0 * np.log10(np.maximum(rms, 1e-6))
    if not count or level.max() < -60.0:
        return None
    voiced = (level > level.max() - 30.0) & (f0 > 75.0) & (f0 < 340.0)
    speech = level > level.max() - 35.0
    if voiced.sum() < 10:
        return None
    semitones = 12.0 * np.log2(f0[voiced] / 100.0)
    return {
        "pitch_std_st": round(float(np.std(semitones)), 3),
        "pitch_range90_st": round(float(np.percentile(semitones, 95) - np.percentile(semitones, 5)), 3),
        "energy_std_db": round(float(np.std(level[speech])), 3) if speech.any() else 0.0,
        "voiced_fraction": round(float(voiced.mean()), 3),
    }


class ExpressivenessCache:
    """Per-clip expressiveness measurements stored beside the dataset, keyed by file size and modification time."""

    def __init__(self, dataset_dir: str | Path) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.path = self.dataset_dir / "analysis" / "expressiveness_cache.json"
        loaded = read_json_retry(self.path, None)
        self._entries: dict[str, dict[str, Any]] = dict(loaded) if isinstance(loaded, dict) else {}
        self._dirty = False

    def metrics_of(self, record: Mapping[str, Any]) -> dict[str, float] | None:
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
        metrics = measure_clip_expressiveness(path)
        self._entries[key] = {"metrics": metrics, "size": stat.st_size, "mtime": int(stat.st_mtime)}
        self._dirty = True
        return metrics

    def save(self) -> None:
        if not self._dirty:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(self.path, self._entries, indent=1, ensure_ascii=False)
            self._dirty = False
        except OSError:
            pass


def choose_expressive_reference(dataset_dir: str | Path, records: Sequence[Mapping[str, Any]], *,
                                shortlist: int = EXPRESSIVE_SHORTLIST, min_seconds: float = EXPRESSIVE_MIN_SECONDS,
                                max_seconds: float = EXPRESSIVE_MAX_SECONDS,
                                cache: ExpressivenessCache | None = None) -> dict[str, Any] | None:
    """Among the cleanest clips of prompt length, the one whose pitch and loudness move the most.

    The typical reference (median pitch and pace) is the safest identity prompt; this clip is the
    liveliest delivery of the same speaker and serves as the emotion prompt, so the voice speaks
    with the person's own energy instead of an average one. Candidates keep the best transcript
    and boundary class (as the typical reference does) and a duration between ``min_seconds`` and
    ``max_seconds``; up to ``shortlist`` of them are measured. Returns
    ``{"record", "metrics", "score", "candidates"}`` or None when nothing can be measured.
    """
    existing = [row for row in records if row.get("audio") and _audio_path(dataset_dir, row).is_file()]
    if not existing:
        return None
    ranked = sorted(existing, key=lambda row: (*training_reference_priority(row), str(row["id"])))
    best_class = training_reference_priority(ranked[0])[:2]
    pool = [row for row in ranked if training_reference_priority(row)[:2] == best_class]

    def duration_of(row: Mapping[str, Any]) -> float:
        try:
            return float(row.get("duration_s") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    windowed = [row for row in pool if min_seconds <= duration_of(row) <= max_seconds]
    if len(windowed) >= 8:
        pool = windowed
    pool = sorted(pool, key=lambda row: _stable_order(str(row.get("id", ""))))[: max(1, int(shortlist))]
    cache = cache or ExpressivenessCache(dataset_dir)
    measured = []
    for row in pool:
        metrics = cache.metrics_of(row)
        if metrics and metrics.get("pitch_std_st", 0.0) > 0.0:
            measured.append((row, metrics))
    cache.save()
    if not measured:
        return None
    import statistics

    pitch_values = [item[1]["pitch_std_st"] for item in measured]
    energy_values = [item[1]["energy_std_db"] for item in measured]
    pitch_mean, energy_mean = statistics.fmean(pitch_values), statistics.fmean(energy_values)
    pitch_sd = statistics.pstdev(pitch_values) or 1.0
    energy_sd = statistics.pstdev(energy_values) or 1.0
    candidates = []
    for row, metrics in measured:
        score = (metrics["pitch_std_st"] - pitch_mean) / pitch_sd + (metrics["energy_std_db"] - energy_mean) / energy_sd
        candidates.append({"id": str(row["id"]), "duration_s": duration_of(row), "score": round(float(score), 3),
                           "metrics": metrics, "record": row})
    best = max(candidates, key=lambda item: (item["score"], -candidates.index(item)))
    return {"record": best["record"], "metrics": best["metrics"], "score": best["score"],
            "pool_pitch_std_st": round(pitch_mean, 3), "pool_energy_std_db": round(energy_mean, 3),
            "candidates": [{key: value for key, value in item.items() if key != "record"} for item in candidates]}


def describe_expressive_choice(choice: Mapping[str, Any]) -> str:
    metrics = choice.get("metrics") or {}
    return (
        f"expressive clip {choice['record'].get('id', '')}: pitch std {float(metrics.get('pitch_std_st', 0.0)):.2f} st "
        f"(pool {float(choice.get('pool_pitch_std_st', 0.0)):.2f}), loudness std {float(metrics.get('energy_std_db', 0.0)):.2f} dB "
        f"(pool {float(choice.get('pool_energy_std_db', 0.0)):.2f}), {len(choice.get('candidates') or [])} clips measured"
    )


def describe_reference_choice(choice: Mapping[str, Any]) -> str:
    profile = choice.get("profile") or {}
    pitch = choice.get("pitch_hz")
    pace = choice.get("pace_wps")
    median_pitch = profile.get("pitch_median_hz")
    median_pace = profile.get("pace_median_wps")
    parts = [f"reference {choice['record'].get('id', '')}"]
    if pitch and median_pitch:
        parts.append(f"pitch {pitch:.0f} Hz (speaker median {median_pitch:.0f} Hz)")
    if pace and median_pace:
        parts.append(f"pace {pace:.2f} words/s (median {median_pace:.2f})")
    parts.append(f"distance {float(choice.get('distance', 0.0)):.2f} over {len(choice.get('candidates') or [])} candidates")
    return " | ".join(parts)


__all__ = [
    "PitchCache",
    "choose_typical_reference",
    "clip_pace",
    "describe_reference_choice",
    "measure_clip_pitch",
    "speaker_profile",
    "typicality_distance",
]
