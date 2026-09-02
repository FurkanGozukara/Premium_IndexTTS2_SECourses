"""CPU-only speaking-rate measurement and per-voice calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import soundfile as sf

from indextts.utils.atomic_json import read_json_retry, write_json_atomic
from indextts.utils.pause_tags import TextChunk, split_text_with_pauses

from .dataset_manifest import load_manifest


_WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
_STRENGTH_SUFFIX_RE = re.compile(
    r"\s+@(?P<strength>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$"
)
_MIN_CLIP_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class SpeakingRateReport:
    recommended_speaking_rate: float
    dataset_words_per_second: float
    generated_words_per_second: float
    clips_used: int
    method: str
    generated_at: str
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any] | "SpeakingRateReport"
    ) -> "SpeakingRateReport":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("speaking-rate report must be a mapping")
        allowed = {item.name for item in fields(cls)}
        payload = {key: item for key, item in value.items() if key in allowed}
        report = cls(**payload)
        if report.method not in {"training_samples", "grid"}:
            raise ValueError("speaking-rate method must be training_samples or grid")
        for name in (
            "recommended_speaking_rate",
            "dataset_words_per_second",
            "generated_words_per_second",
        ):
            number = float(getattr(report, name))
            if not math.isfinite(number) or number < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0.5 <= float(report.recommended_speaking_rate) <= 1.5:
            raise ValueError("recommended_speaking_rate must be in [0.5, 1.5]")
        if int(report.clips_used) < 1:
            raise ValueError("clips_used must be at least 1")
        return cls(
            recommended_speaking_rate=float(report.recommended_speaking_rate),
            dataset_words_per_second=float(report.dataset_words_per_second),
            generated_words_per_second=float(report.generated_words_per_second),
            clips_used=int(report.clips_used),
            method=str(report.method),
            generated_at=str(report.generated_at),
            summary=str(report.summary),
        )


def _text_word_count(text: str) -> int:
    without_pauses = " ".join(
        chunk.text
        for chunk in split_text_with_pauses(str(text or ""))
        if isinstance(chunk, TextChunk)
    )
    return len(_WORD_RE.findall(without_pauses))


def _trimmed_duration(audio_path: str | Path) -> float:
    try:
        audio, sample_rate = sf.read(
            str(Path(audio_path).expanduser()), dtype="float32", always_2d=True
        )
    except (OSError, RuntimeError, ValueError):
        return 0.0
    if sample_rate <= 0 or audio.size == 0:
        return 0.0
    envelope = np.max(np.abs(audio), axis=1)
    peak = float(np.max(envelope, initial=0.0))
    if not math.isfinite(peak) or peak <= 0.0:
        return 0.0
    audible = np.flatnonzero(envelope >= peak * 0.01)
    if audible.size == 0:
        return 0.0
    frames = int(audible[-1]) - int(audible[0]) + 1
    duration = frames / float(sample_rate)
    return duration if duration >= _MIN_CLIP_SECONDS else 0.0


def words_per_second(text: str, audio_path: str | Path) -> float:
    """Return letter/digit word tokens divided by 40 dB edge-trimmed duration."""

    words = _text_word_count(text)
    duration = _trimmed_duration(audio_path)
    return words / duration if words > 0 and duration > 0.0 else 0.0


def dataset_words_per_second(dataset_dir: str | Path) -> float:
    """Return aggregate manifest words divided by aggregate recorded duration."""

    words = 0
    duration = 0.0
    for row in load_manifest(Path(dataset_dir).expanduser()):
        try:
            row_words = int(row.get("words", 0) or 0)
            row_duration = float(row.get("duration_s", 0.0) or 0.0)
        except (TypeError, ValueError, OverflowError):
            continue
        if row_words < 0 or not math.isfinite(row_duration) or row_duration <= 0.0:
            continue
        words += row_words
        duration += row_duration
    return words / duration if words > 0 and duration > 0.0 else 0.0


def _report(
    *,
    dataset_wps: float,
    generated_words: int,
    generated_duration: float,
    clips_used: int,
    method: str,
) -> SpeakingRateReport | None:
    if (
        dataset_wps <= 0.0
        or generated_words <= 0
        or generated_duration <= 0.0
        or clips_used <= 0
    ):
        return None
    generated_wps = generated_words / generated_duration
    rate = round(min(1.5, max(0.5, dataset_wps / generated_wps)), 3)
    summary = (
        f"Your recordings average {dataset_wps:.2f} words/s; this LoRA / DoRA "
        f"generated {generated_wps:.2f} words/s at speaking rate 1.0, so "
        f"{rate:.2f} matches your real pace."
    )
    return SpeakingRateReport(
        recommended_speaking_rate=rate,
        dataset_words_per_second=float(dataset_wps),
        generated_words_per_second=float(generated_wps),
        clips_used=int(clips_used),
        method=method,
        generated_at=datetime.now(timezone.utc).isoformat(),
        summary=summary,
    )


def calibrate_from_samples(
    adapter_dir: str | Path,
    dataset_dir: str | Path,
    sample_text: str,
) -> SpeakingRateReport | None:
    """Calibrate from every usable ``samples/epoch_*.wav`` training sample."""

    words_per_clip = _text_word_count(sample_text)
    if words_per_clip <= 0:
        return None
    total_words = 0
    total_duration = 0.0
    clips_used = 0
    samples = Path(adapter_dir).expanduser().resolve() / "samples"
    for path in sorted(samples.glob("epoch_*.wav")) if samples.is_dir() else []:
        duration = _trimmed_duration(path)
        if duration <= 0.0:
            continue
        total_words += words_per_clip
        total_duration += duration
        clips_used += 1
    return _report(
        dataset_wps=dataset_words_per_second(dataset_dir),
        generated_words=total_words,
        generated_duration=total_duration,
        clips_used=clips_used,
        method="training_samples",
    )


def _same_path(left: str | Path, right: str | Path) -> bool:
    try:
        return os.path.normcase(str(Path(left).expanduser().resolve())) == os.path.normcase(
            str(Path(right).expanduser().resolve())
        )
    except (OSError, RuntimeError, ValueError):
        return False


def calibrate_from_grid(
    grid_dir: str | Path,
    checkpoint_label: str,
    dataset_dir: str | Path,
) -> SpeakingRateReport | None:
    """Calibrate from all usable cells belonging to one grid checkpoint."""

    root = Path(grid_dir).expanduser().resolve()
    payload = read_json_retry(root / "grid.json", {}) or {}
    target = str(checkpoint_label or "").strip()
    strength_match = _STRENGTH_SUFFIX_RE.search(target)
    target_strength = (
        float(strength_match.group("strength")) if strength_match else None
    )
    target_core = target[: strength_match.start()].rstrip() if strength_match else target
    total_words = 0
    total_duration = 0.0
    clips_used = 0
    for cell in payload.get("cells", []):
        if not isinstance(cell, Mapping) or not cell.get("checkpoint_path"):
            continue
        cell_label = str(cell.get("checkpoint_label") or "").strip()
        cell_path = str(cell.get("checkpoint_path") or "").strip()
        if target_core != cell_label and not _same_path(target_core, cell_path):
            continue
        if target_strength is not None:
            try:
                if abs(float(cell.get("strength", 1.0)) - target_strength) >= 1e-9:
                    continue
            except (TypeError, ValueError):
                continue
        words = _text_word_count(str(cell.get("text") or ""))
        audio_path = Path(str(cell.get("audio_path") or cell.get("filename") or ""))
        if not audio_path.is_absolute():
            audio_path = root / audio_path
        duration = _trimmed_duration(audio_path)
        if words <= 0 or duration <= 0.0:
            continue
        total_words += words
        total_duration += duration
        clips_used += 1
    return _report(
        dataset_wps=dataset_words_per_second(dataset_dir),
        generated_words=total_words,
        generated_duration=total_duration,
        clips_used=clips_used,
        method="grid",
    )


def write_speaking_rate(
    adapter_dir: str | Path, report: SpeakingRateReport
) -> Path:
    """Atomically persist the adapter-wide calibrated speaking rate."""

    validated = SpeakingRateReport.from_dict(report)
    destination = (
        Path(adapter_dir).expanduser().resolve() / "analysis" / "speaking_rate.json"
    )
    return write_json_atomic(
        destination,
        validated.to_dict(),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )


def load_speaking_rate(
    adapter_or_checkpoint_path: str | Path,
) -> SpeakingRateReport | None:
    """Load calibration from an adapter folder, checkpoint, or ``best/`` file."""

    source = Path(adapter_or_checkpoint_path).expanduser().resolve()
    if source.is_dir():
        adapter_dir = source.parent if source.name.lower() == "best" else source
    else:
        adapter_dir = source.parent.parent if source.parent.name.lower() == "best" else source.parent
    value = read_json_retry(adapter_dir / "analysis" / "speaking_rate.json", None)
    if not isinstance(value, Mapping):
        return None
    try:
        return SpeakingRateReport.from_dict(value)
    except (KeyError, TypeError, ValueError):
        return None


__all__ = [
    "SpeakingRateReport",
    "calibrate_from_grid",
    "calibrate_from_samples",
    "dataset_words_per_second",
    "load_speaking_rate",
    "words_per_second",
    "write_speaking_rate",
]
