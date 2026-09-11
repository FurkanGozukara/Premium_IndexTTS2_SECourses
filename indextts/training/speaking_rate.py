"""CPU-only speaking-rate measurement and per-voice calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping
import unicodedata

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
_METHOD_LABELS = {
    "training_samples": "training samples",
    "grid": "grid",
    "grid_matched": "grid (matched sentences)",
    "speech_matched": "speech comparison (matched sentences)",
    "manual": "set manually",
}
# Matched-sentence calibration needs this many generated clips with real recordings.
MIN_MATCHED_CLIPS = 4


@dataclass(frozen=True, slots=True)
class SpeakingRateReport:
    recommended_speaking_rate: float
    dataset_words_per_second: float
    generated_words_per_second: float
    clips_used: int
    method: str
    generated_at: str
    summary: str
    # The automatic estimate and how it was measured; kept when a manual value replaces it,
    # so the original calibration stays visible beside the value in use.
    calibrated_speaking_rate: float | None = None
    calibration_method: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def is_manual(self) -> bool:
        return self.method == "manual"

    def original_rate(self) -> float | None:
        """The automatic calibration: the report's own value unless a manual one replaced it."""

        if self.calibrated_speaking_rate is not None:
            return float(self.calibrated_speaking_rate)
        return None if self.is_manual else float(self.recommended_speaking_rate)

    def original_method(self) -> str:
        if self.calibration_method:
            return str(self.calibration_method)
        return "" if self.is_manual else str(self.method)

    def effective_words_per_second(self, speaking_rate: float | None = None) -> float:
        """Generated words per second at a speaking-rate slider value (the rate scales the pace)."""

        rate = float(self.recommended_speaking_rate if speaking_rate is None else speaking_rate)
        return float(self.generated_words_per_second) * max(0.0, rate)

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
        if report.method not in _METHOD_LABELS:
            allowed_methods = ", ".join(_METHOD_LABELS)
            raise ValueError(f"speaking-rate method must be one of: {allowed_methods}")
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
        calibrated = report.calibrated_speaking_rate
        if calibrated is not None:
            calibrated = float(calibrated)
            if not math.isfinite(calibrated) or not 0.5 <= calibrated <= 1.5:
                raise ValueError("calibrated_speaking_rate must be in [0.5, 1.5]")
        calibration_method = str(report.calibration_method or "")
        if calibration_method and calibration_method not in _METHOD_LABELS:
            raise ValueError("calibration_method must be a known speaking-rate method")
        return cls(
            recommended_speaking_rate=float(report.recommended_speaking_rate),
            dataset_words_per_second=float(report.dataset_words_per_second),
            generated_words_per_second=float(report.generated_words_per_second),
            clips_used=int(report.clips_used),
            method=str(report.method),
            generated_at=str(report.generated_at),
            summary=str(report.summary),
            calibrated_speaking_rate=calibrated,
            calibration_method=calibration_method,
        )


def _text_word_count(text: str) -> int:
    without_pauses = " ".join(
        chunk.text
        for chunk in split_text_with_pauses(str(text or ""))
        if isinstance(chunk, TextChunk)
    )
    return len(_WORD_RE.findall(without_pauses))


def _normalise_matching_text(text: str) -> str:
    value = " ".join(str(text or "").casefold().split())
    while value and unicodedata.category(value[0]).startswith("P"):
        value = value[1:].lstrip()
    while value and unicodedata.category(value[-1]).startswith("P"):
        value = value[:-1].rstrip()
    return value


def speaking_rate_method_label(method: str) -> str:
    """Return the human-readable source label used by speaking-rate UIs."""

    return _METHOD_LABELS.get(str(method), str(method).replace("_", " "))


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
        calibrated_speaking_rate=rate,
        calibration_method=method,
    )


def _matched_report(
    *,
    matched_words: int,
    generated_duration: float,
    real_duration: float,
    clips_used: int,
    method: str = "grid_matched",
) -> SpeakingRateReport | None:
    if (
        matched_words <= 0
        or generated_duration <= 0.0
        or real_duration <= 0.0
        or clips_used <= 0
    ):
        return None
    real_wps = matched_words / real_duration
    generated_wps = matched_words / generated_duration
    rate = round(min(1.5, max(0.5, generated_duration / real_duration)), 3)
    relative_speed = generated_wps / real_wps - 1.0
    if abs(relative_speed) < 0.005:
        comparison = "at the same pace as you"
    else:
        direction = "faster" if relative_speed > 0.0 else "slower"
        comparison = f"{abs(relative_speed) * 100.0:.0f} % {direction} than you"
    summary = (
        f"Across {clips_used} sentences that exist in your recordings, this LoRA / "
        f"DoRA spoke {comparison} at speaking rate 1.0, so {rate:.2f} matches your "
        "real pace."
    )
    return SpeakingRateReport(
        recommended_speaking_rate=rate,
        dataset_words_per_second=float(real_wps),
        generated_words_per_second=float(generated_wps),
        clips_used=int(clips_used),
        method=method,
        generated_at=datetime.now(timezone.utc).isoformat(),
        summary=summary,
        calibrated_speaking_rate=rate,
        calibration_method=method,
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
    dataset_root = Path(dataset_dir).expanduser().resolve()
    manifest_by_text: dict[str, list[Mapping[str, Any]]] = {}
    for row in load_manifest(dataset_root):
        normalised = _normalise_matching_text(str(row.get("text") or ""))
        if normalised:
            manifest_by_text.setdefault(normalised, []).append(row)

    total_words = 0
    total_duration = 0.0
    clips_used = 0
    matched_words = 0
    matched_generated_duration = 0.0
    matched_real_duration = 0.0
    matched_clips = 0
    recording_duration_cache: dict[Path, float] = {}
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

        normalised = _normalise_matching_text(str(cell.get("text") or ""))
        for row in manifest_by_text.get(normalised, []):
            audio_value = str(row.get("audio") or "").strip()
            if not audio_value:
                continue
            recording_path = Path(audio_value).expanduser()
            if not recording_path.is_absolute():
                recording_path = dataset_root / recording_path
            recording_duration = recording_duration_cache.get(recording_path)
            if recording_duration is None:
                recording_duration = _trimmed_duration(recording_path)
                recording_duration_cache[recording_path] = recording_duration
            if recording_duration <= 0.0:
                continue
            matched_words += words
            matched_generated_duration += duration
            matched_real_duration += recording_duration
            matched_clips += 1
            break

    if matched_clips >= MIN_MATCHED_CLIPS:
        return _matched_report(
            matched_words=matched_words,
            generated_duration=matched_generated_duration,
            real_duration=matched_real_duration,
            clips_used=matched_clips,
        )
    return _report(
        dataset_wps=dataset_words_per_second(dataset_root),
        generated_words=total_words,
        generated_duration=total_duration,
        clips_used=clips_used,
        method="grid",
    )


def calibrate_from_speech_report(
    report: Mapping[str, Any],
    checkpoint_path: str | Path | None = None,
) -> SpeakingRateReport | None:
    """Calibrate from the automatic speech comparison's matched held-out sentences.

    The short epoch sample compares one ten-word sentence against long
    multi-sentence recordings and therefore overstates how slow the voice is.
    Matched sentences compare the same text spoken by the person and by the
    adapter, so their duration ratio is the pace correction that reproduces the
    real recordings.
    """

    candidates = [item for item in report.get("candidates") or [] if isinstance(item, Mapping)]
    label: str | None = None
    if checkpoint_path:
        for item in candidates:
            if item.get("path") and _same_path(str(item["path"]), checkpoint_path):
                label = str(item.get("label") or "")
                break
    if label is None:
        recommended = str(report.get("recommended_label") or "")
        if any(item.get("path") and str(item.get("label") or "") == recommended for item in candidates):
            label = recommended
    if not label:
        return None
    matched_words = 0
    generated_duration = 0.0
    real_duration = 0.0
    clips_used = 0
    real_durations: dict[str, float] = {}
    for cell in report.get("cells") or []:
        if not isinstance(cell, Mapping) or str(cell.get("checkpoint") or "") != label:
            continue
        if cell.get("kind") != "matched" or not cell.get("real_audio") or cell.get("invalid_audio"):
            continue
        words = _text_word_count(str(cell.get("text") or ""))
        generated = _trimmed_duration(str(cell.get("audio") or ""))
        real_path = str(cell["real_audio"])
        if real_path not in real_durations:
            real_durations[real_path] = _trimmed_duration(real_path)
        real = real_durations[real_path]
        if words <= 0 or generated <= 0.0 or real <= 0.0:
            continue
        matched_words += words
        generated_duration += generated
        real_duration += real
        clips_used += 1
    if clips_used < MIN_MATCHED_CLIPS:
        return None
    return _matched_report(
        matched_words=matched_words,
        generated_duration=generated_duration,
        real_duration=real_duration,
        clips_used=clips_used,
        method="speech_matched",
    )


def _adapter_dir(adapter_or_checkpoint_path: str | Path) -> Path:
    source = Path(adapter_or_checkpoint_path).expanduser().resolve()
    if source.is_dir():
        return source.parent if source.name.lower() == "best" else source
    return source.parent.parent if source.parent.name.lower() == "best" else source.parent


def save_manual_speaking_rate(
    adapter_or_checkpoint_path: str | Path,
    rate: float,
) -> SpeakingRateReport:
    """Store a user-chosen speaking rate for an adapter, keeping the measurement fields."""

    value = float(rate)
    if not math.isfinite(value) or not 0.5 <= value <= 1.5:
        raise ValueError("Speaking rate must be between 0.5 and 1.5")
    previous = load_speaking_rate(adapter_or_checkpoint_path)
    now = datetime.now(timezone.utc)
    if previous is None:
        note = "No automatic estimate existed."
    elif previous.method == "manual":
        note = f"Replaced the earlier manual value {previous.recommended_speaking_rate:.3f}."
    else:
        note = (
            f"Replaced the automatic estimate {previous.recommended_speaking_rate:.3f} "
            f"({speaking_rate_method_label(previous.method)})."
        )
    calibrated, calibration_method = original_calibration(adapter_or_checkpoint_path, previous)
    if calibrated is not None:
        note += (
            f" The automatic estimate {calibrated:.3f} "
            f"({speaking_rate_method_label(calibration_method)}) is kept for reference."
        )
    report = SpeakingRateReport(
        recommended_speaking_rate=round(value, 3),
        dataset_words_per_second=previous.dataset_words_per_second if previous else 0.0,
        generated_words_per_second=previous.generated_words_per_second if previous else 0.0,
        clips_used=previous.clips_used if previous else 1,
        method="manual",
        generated_at=now.isoformat(),
        summary=f"Speaking rate {value:.3f} was set manually on {now:%Y-%m-%d}. {note}",
        calibrated_speaking_rate=calibrated,
        calibration_method=calibration_method if calibrated is not None else "",
    )
    write_speaking_rate(_adapter_dir(adapter_or_checkpoint_path), report)
    return report


_SUMMARY_ESTIMATE_RE = re.compile(r"automatic estimate (?P<rate>\d+(?:\.\d+)?) \((?P<label>[^)]+)\)")
_LABEL_TO_METHOD = {label: method for method, label in _METHOD_LABELS.items()}


def original_calibration(
    adapter_or_checkpoint_path: str | Path,
    report: SpeakingRateReport | None = None,
) -> tuple[float | None, str]:
    """The automatic speaking-rate estimate of an adapter and its method, even after manual edits.

    Order: the report's own calibration fields; an automatic report; the estimate
    named in a manual report's summary; the matched-sentence speech comparison
    saved by training; the training-sample estimate kept beside it.
    """

    if report is None:
        report = load_speaking_rate(adapter_or_checkpoint_path)
    if report is not None:
        rate = report.original_rate()
        if rate is not None:
            return rate, report.original_method() or report.method
        match = _SUMMARY_ESTIMATE_RE.search(report.summary or "")
        if match:
            method = _LABEL_TO_METHOD.get(match.group("label").strip(), "")
            if method:
                try:
                    return float(match.group("rate")), method
                except ValueError:
                    pass
    adapter_dir = _adapter_dir(adapter_or_checkpoint_path)
    speech_report = read_json_retry(adapter_dir / "analysis" / "speech_evaluation" / "report.json", None)
    if isinstance(speech_report, Mapping):
        try:
            matched = calibrate_from_speech_report(speech_report, None)
        except Exception:
            matched = None
        if matched is not None:
            return matched.recommended_speaking_rate, matched.method
    sidecar = read_json_retry(adapter_dir / "analysis" / "speaking_rate_training_samples.json", None)
    if isinstance(sidecar, Mapping):
        try:
            earlier = SpeakingRateReport.from_dict(sidecar)
        except (KeyError, TypeError, ValueError):
            earlier = None
        if earlier is not None and not earlier.is_manual:
            return earlier.recommended_speaking_rate, earlier.method
    return None, ""


def ensure_calibration_fields(
    adapter_or_checkpoint_path: str | Path,
    report: SpeakingRateReport | None = None,
) -> SpeakingRateReport | None:
    """Return the adapter's report with the original calibration filled in, persisting it once.

    Manual reports written before v6.13 carry no ``calibrated_speaking_rate``;
    recovering it means reading the saved speech comparison, so the recovered
    value is stored back into ``speaking_rate.json`` the first time it is needed.
    """

    if report is None:
        report = load_speaking_rate(adapter_or_checkpoint_path)
    if report is None or report.original_rate() is not None:
        return report
    calibrated, method = original_calibration(adapter_or_checkpoint_path, report)
    if calibrated is None:
        return report
    updated = SpeakingRateReport.from_dict({**report.to_dict(), "calibrated_speaking_rate": calibrated, "calibration_method": method})
    try:
        write_speaking_rate(_adapter_dir(adapter_or_checkpoint_path), updated)
    except OSError:
        pass
    return updated


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

    adapter_dir = _adapter_dir(adapter_or_checkpoint_path)
    value = read_json_retry(adapter_dir / "analysis" / "speaking_rate.json", None)
    if not isinstance(value, Mapping):
        return None
    try:
        return SpeakingRateReport.from_dict(value)
    except (KeyError, TypeError, ValueError):
        return None


__all__ = [
    "MIN_MATCHED_CLIPS",
    "SpeakingRateReport",
    "calibrate_from_grid",
    "calibrate_from_samples",
    "calibrate_from_speech_report",
    "dataset_words_per_second",
    "ensure_calibration_fields",
    "load_speaking_rate",
    "original_calibration",
    "save_manual_speaking_rate",
    "speaking_rate_method_label",
    "words_per_second",
    "write_speaking_rate",
]
