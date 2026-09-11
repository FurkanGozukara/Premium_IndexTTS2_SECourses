"""Shorten over-long pauses inside generated speech without touching the words.

A fine-tuned voice often pauses more than the person it imitates: it reproduces
the average pause density of its multi-sentence training clips on every
sentence. This module finds the quiet stretches between words and sentences
(the same rule the speech metrics use to measure pauses) and shortens every
pause longer than a chosen ceiling to that ceiling, removing its middle with a
short crossfade. Leading and trailing silence, and pauses below the ceiling,
are left exactly as generated. Nothing here depends on a particular voice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from indextts.training.speech_metrics import (
    MIN_PAUSE_MS,
    PAUSE_HOP_MS,
    PAUSE_NO_SIGNAL_DBFS,
    PAUSE_RELATIVE_DB,
    PAUSE_THRESHOLD_DBFS,
)

CROSSFADE_MS = 8
MAX_PAUSE_MS_LIMIT = 5000


def find_internal_pauses(samples: np.ndarray, sample_rate: int, *, min_pause_ms: int = MIN_PAUSE_MS) -> list[tuple[int, int]]:
    """Quiet stretches between the first and last loud frame, as (start, end) sample indices."""

    source = np.asarray(samples)
    array = source.astype(np.float32)
    if source.dtype.kind in "iu":
        # PCM samples arrive as integers; the dBFS thresholds below assume a full scale of 1.0.
        array /= float(np.iinfo(source.dtype).max)
    mono = array.mean(axis=1) if array.ndim == 2 else array.reshape(-1)
    hop = max(1, int(round(sample_rate * PAUSE_HOP_MS / 1000)))
    frames = len(mono) // hop
    if frames == 0:
        return []
    rms = np.sqrt(np.mean(np.square(mono[: frames * hop]).reshape(frames, hop), axis=1) + 1e-12)
    level = 20.0 * np.log10(rms)
    loudest = float(level.max())
    if loudest < PAUSE_NO_SIGNAL_DBFS:
        return []
    quiet = level < min(PAUSE_THRESHOLD_DBFS, loudest - PAUSE_RELATIVE_DB)
    loud = np.flatnonzero(~quiet)
    if len(loud) == 0:
        return []
    first, last = int(loud[0]), int(loud[-1])
    inner = quiet[first : last + 1]
    edges = np.diff(np.pad(inner.astype(np.int8), (1, 1)))
    pauses: list[tuple[int, int]] = []
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if (end - start) * PAUSE_HOP_MS >= min_pause_ms:
            pauses.append(((first + int(start)) * hop, (first + int(end)) * hop))
    return pauses


def shorten_long_pauses(
    samples: np.ndarray,
    sample_rate: int,
    max_pause_ms: int,
    *,
    crossfade_ms: int = CROSSFADE_MS,
    protected_s: Sequence[tuple[float, float]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the audio with every internal pause longer than ``max_pause_ms`` cut down to it, plus a report.

    ``protected_s`` lists time ranges (seconds) that must keep their length, such as the
    silences written for explicit pause tags; a pause overlapping one is left as it is.
    """

    array = np.asarray(samples)
    ceiling = int(max_pause_ms)
    report: dict[str, Any] = {"max_pause_ms": ceiling, "pauses": 0, "shortened": 0, "removed_ms": 0.0, "protected": 0}
    if ceiling <= 0 or array.size == 0:
        return array, report
    pauses = find_internal_pauses(array, sample_rate)
    report["pauses"] = len(pauses)
    if protected_s:
        guarded = [(int(round(float(start) * sample_rate)), int(round(float(end) * sample_rate))) for start, end in protected_s]
        kept = [
            (start, end) for start, end in pauses
            if not any(start < guard_end and end > guard_start for guard_start, guard_end in guarded)
        ]
        report["protected"] = len(pauses) - len(kept)
        pauses = kept
    keep = int(round(sample_rate * ceiling / 1000))
    fade = max(1, int(round(sample_rate * crossfade_ms / 1000)))
    pieces: list[np.ndarray] = []
    cursor = 0
    removed = 0
    working = array.astype(np.float32) if array.dtype.kind in "iu" else array
    for start, end in pauses:
        length = end - start
        if length <= keep:
            continue
        head_end = start + keep // 2
        tail_start = end - (keep - keep // 2)
        if tail_start - head_end < fade:
            continue
        pieces.append(working[cursor:head_end])
        # Crossfade the end of the kept head into the start of the kept tail.
        head_fade = working[head_end - fade : head_end] if head_end - fade >= cursor else None
        tail_fade = working[tail_start : tail_start + fade]
        if head_fade is not None and len(head_fade) == len(tail_fade) == fade:
            ramp = np.linspace(1.0, 0.0, fade, dtype=np.float32)
            shape = (fade, 1) if working.ndim == 2 else (fade,)
            blended = head_fade * ramp.reshape(shape) + tail_fade * (1.0 - ramp).reshape(shape)
            pieces[-1] = pieces[-1][: len(pieces[-1]) - fade]
            pieces.append(blended)
            cursor = tail_start + fade
        else:
            cursor = tail_start
        removed += tail_start - head_end
        report["shortened"] += 1
    pieces.append(working[cursor:])
    result = np.concatenate(pieces, axis=0) if len(pieces) > 1 else working
    if array.dtype.kind in "iu":
        info = np.iinfo(array.dtype)
        result = np.clip(np.round(result), info.min, info.max).astype(array.dtype)
    report["removed_ms"] = round(removed * 1000.0 / sample_rate, 1)
    return result, report


def shorten_long_pauses_file(
    source: str | Path,
    destination: str | Path,
    max_pause_ms: int,
    *,
    protected_s: Sequence[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Apply :func:`shorten_long_pauses` to a WAV file, keeping its sample rate, channels and PCM format."""

    import soundfile as sf

    with sf.SoundFile(str(source)) as handle:
        subtype = handle.subtype
        sample_rate = handle.samplerate
        dtype = "int16" if subtype == "PCM_16" else "float32"
        audio = handle.read(dtype=dtype, always_2d=False)
    result, report = shorten_long_pauses(audio, sample_rate, max_pause_ms, protected_s=protected_s)
    if report["shortened"]:
        sf.write(str(destination), result, sample_rate, subtype=subtype)
    elif str(Path(source).resolve()) != str(Path(destination).resolve()):
        sf.write(str(destination), audio, sample_rate, subtype=subtype)
    report["duration_before_s"] = round(len(audio) / sample_rate, 3)
    report["duration_after_s"] = round(len(result) / sample_rate, 3)
    return report


__all__ = ["CROSSFADE_MS", "MAX_PAUSE_MS_LIMIT", "find_internal_pauses", "shorten_long_pauses", "shorten_long_pauses_file"]
