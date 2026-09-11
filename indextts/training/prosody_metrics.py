"""Prosody statistics of a clip and paired comparisons against the real recording of the same text.

The speech metrics module measures intelligibility, identity and pausing; this
module measures how the speech moves: pitch variability in semitones, melodic
movement between frames, loudness dynamics, articulation rate and pauses. The
numbers describe liveliness, not correctness, and they are read against the
person's own recording of the same sentence. CPU only (librosa pyin).
"""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .speech_metrics import internal_pause_metrics


PROSODY_SAMPLE_RATE = 16000
PROSODY_HOP = 160  # 10 ms
PROSODY_KEYS = (
    "words_per_s",
    "articulation_wps",
    "voiced_fraction",
    "f0_median_hz",
    "f0_std_st",
    "f0_iqr_st",
    "f0_range90_st",
    "f0_movement_st",
    "energy_std_db",
    "energy_range90_db",
    "pause_count",
    "pause_fraction",
    "longest_pause_ms",
)
# Higher is livelier for these; lower means flatter than the recording.
LIVELINESS_KEYS = ("f0_std_st", "f0_iqr_st", "f0_range90_st", "f0_movement_st", "energy_std_db", "energy_range90_db")
_WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


def _load_mono(path: str | Path, sample_rate: int) -> np.ndarray:
    import librosa
    import soundfile as sf

    audio, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
    signal = audio.mean(axis=1)
    if source_rate != sample_rate:
        signal = librosa.resample(signal, orig_sr=source_rate, target_sr=sample_rate)
    return signal


def _trim_edges(signal: np.ndarray, floor: float = 0.01) -> np.ndarray:
    peak = float(np.max(np.abs(signal), initial=0.0))
    if peak <= 0.0:
        return signal
    audible = np.flatnonzero(np.abs(signal) >= peak * floor)
    return signal[audible[0] : audible[-1] + 1] if audible.size else signal


def measure_prosody(path: str | Path, text: str, *, sample_rate: int = PROSODY_SAMPLE_RATE) -> dict[str, Any]:
    """Pitch, loudness, rate and pause statistics of one clip (edge silence removed)."""

    import librosa

    signal = _trim_edges(_load_mono(path, sample_rate))
    duration = len(signal) / sample_rate
    if duration <= 0.0:
        return {"duration_s": 0.0, "words": 0, **{key: 0.0 for key in PROSODY_KEYS}}
    f0, voiced, _probability = librosa.pyin(
        signal, fmin=60, fmax=400, sr=sample_rate, frame_length=1024, hop_length=PROSODY_HOP
    )
    voiced_mask = np.isfinite(f0) & np.asarray(voiced, dtype=bool)
    voiced_f0 = f0[voiced_mask]
    semitones = 12.0 * np.log2(voiced_f0 / 100.0) if voiced_f0.size else np.zeros(0)
    rms = librosa.feature.rms(y=signal, frame_length=400, hop_length=PROSODY_HOP)[0]
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-6))
    speech_db = rms_db[rms_db > rms_db.max() - 35.0] if rms_db.size else np.zeros(0)
    pauses = internal_pause_metrics(signal, sample_rate)
    words = len(_WORD_RE.findall(str(text or "")))
    pause_s = float(pauses.get("pause_s", 0.0) or 0.0)

    def percentile_range(values: np.ndarray) -> float:
        return float(np.percentile(values, 95) - np.percentile(values, 5)) if values.size else 0.0

    return {
        "duration_s": round(duration, 3),
        "words": words,
        "words_per_s": round(words / duration, 3),
        "articulation_wps": round(words / max(1e-6, duration - pause_s), 3),
        "voiced_fraction": round(float(voiced_mask.mean()), 3) if voiced_mask.size else 0.0,
        "f0_median_hz": round(float(np.median(voiced_f0)), 1) if voiced_f0.size else 0.0,
        "f0_std_st": round(float(np.std(semitones)), 2) if semitones.size else 0.0,
        "f0_iqr_st": round(float(np.percentile(semitones, 75) - np.percentile(semitones, 25)), 2) if semitones.size else 0.0,
        "f0_range90_st": round(percentile_range(semitones), 2),
        "f0_movement_st": round(float(np.mean(np.abs(np.diff(semitones)))), 3) if semitones.size > 1 else 0.0,
        "energy_std_db": round(float(np.std(speech_db)), 2) if speech_db.size else 0.0,
        "energy_range90_db": round(percentile_range(speech_db), 2),
        "pause_count": int(pauses.get("pause_count", 0) or 0),
        "pause_fraction": round(float(pauses.get("pause_time_fraction", 0.0) or 0.0), 3),
        "longest_pause_ms": int(pauses.get("longest_pause_ms", 0) or 0),
    }


def compare_prosody(pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> dict[str, Any]:
    """Paired summary of generated against real metrics: means, mean difference, share below the real value."""

    summary: dict[str, Any] = {"clips": len(pairs)}
    if not pairs:
        return summary
    for key in PROSODY_KEYS:
        generated = np.array([float(item[0].get(key, 0.0) or 0.0) for item in pairs], dtype=float)
        real = np.array([float(item[1].get(key, 0.0) or 0.0) for item in pairs], dtype=float)
        difference = generated - real
        summary[key] = {
            "generated_mean": round(float(generated.mean()), 3),
            "real_mean": round(float(real.mean()), 3),
            "paired_diff_mean": round(float(difference.mean()), 3),
            "generated_lower_share": round(float(np.mean(difference < 0.0)), 3),
        }
    liveliness = []
    for key in LIVELINESS_KEYS:
        real_mean = summary[key]["real_mean"]
        if real_mean:
            liveliness.append(summary[key]["generated_mean"] / real_mean)
    # 1.0 means the generated speech moves as much as the recordings; below 1.0 is flatter.
    summary["liveliness_ratio"] = round(float(np.mean(liveliness)), 3) if liveliness else None
    pause = summary["pause_fraction"]
    summary["pause_fraction_ratio"] = (
        round(pause["generated_mean"] / pause["real_mean"], 3) if pause["real_mean"] else None
    )
    return summary


def prosody_table(summaries: Mapping[str, Mapping[str, Any]], keys: Sequence[str] = PROSODY_KEYS) -> str:
    """Markdown table: one column per condition, one row per metric, generated vs real."""

    names = list(summaries)
    lines = ["| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    lines.append("| clips | " + " | ".join(str(summaries[name].get("clips", 0)) for name in names) + " |")
    for key in keys:
        cells = []
        for name in names:
            entry = summaries[name].get(key)
            if not isinstance(entry, Mapping):
                cells.append("–")
                continue
            cells.append(
                f"{entry['generated_mean']:.2f} vs {entry['real_mean']:.2f} ({entry['generated_lower_share'] * 100:.0f}% below)"
            )
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    for extra in ("liveliness_ratio", "pause_fraction_ratio"):
        values = []
        for name in names:
            value = summaries[name].get(extra)
            values.append("–" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.3f}")
        lines.append(f"| {extra} | " + " | ".join(values) + " |")
    return "\n".join(lines)


__all__ = [
    "LIVELINESS_KEYS",
    "PROSODY_HOP",
    "PROSODY_KEYS",
    "PROSODY_SAMPLE_RATE",
    "compare_prosody",
    "measure_prosody",
    "prosody_table",
]
