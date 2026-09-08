"""FFmpeg-based, voice-oriented final WAV post-processing."""

from __future__ import annotations

import math
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable

import soundfile as sf


TUNING_PRESETS: dict[str, dict[str, Any]] = {
    "bypass": {},
    "voice_clarity": {
        "low_cut_hz": 62.0,
        "high_cut_hz": 10500.0,
        "eq": ((330, -3.0, 0.9), (750, -1.5, 0.9), (2800, 1.0, 1.0)),
        "deess": 0.0,
    },
    "clear_narration": {
        "low_cut_hz": 58.0,
        "high_cut_hz": 9800.0,
        "eq": ((145, -1.0, 0.8), (330, -3.2, 0.9), (750, -1.8, 0.9), (2800, 1.2, 1.0)),
        "deess": 0.8,
        "loudnorm_i": -16.0,
    },
    "deharsh": {
        "low_cut_hz": 55.0,
        "high_cut_hz": 9200.0,
        "eq": ((330, -2.0, 0.9), (750, -1.0, 0.9)),
        "deess": 2.5,
    },
    "warm": {
        "low_cut_hz": 45.0,
        "high_cut_hz": 10500.0,
        "eq": ((145, 1.0, 0.8), (330, -2.2, 0.9), (750, -1.0, 0.9), (2800, 0.5, 1.0)),
        "deess": 1.0,
    },
    "normalize": {
        "low_cut_hz": 20.0,
        "high_cut_hz": 20000.0,
        "loudnorm_i": -16.0,
    },
}

_EXPLICIT_FIELDS = {"low_cut_hz", "high_cut_hz", "gain_db", "loudnorm_i", "deess"}


def _number(name: str, value: Any, minimum: float, maximum: float) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
    return result


def apply_audio_tuning(
    in_wav: str | Path,
    out_wav: str | Path,
    preset: str = "bypass",
    *,
    warning_callback: Callable[[str], Any] | None = None,
    **overrides: Any,
) -> str:
    """Apply a voice preset, reporting sample-rate corrections to the console and callback."""

    source = Path(in_wav)
    destination = Path(out_wav)
    if not source.is_file():
        raise FileNotFoundError(f"Audio tuning input not found: {source}")
    preset_id = str(preset or "bypass").strip().lower()
    if preset_id not in TUNING_PRESETS:
        choices = ", ".join(TUNING_PRESETS)
        raise ValueError(f"Unknown audio tuning preset {preset!r}; expected one of: {choices}")
    unknown = set(overrides) - _EXPLICIT_FIELDS
    if unknown:
        raise ValueError(f"Unknown audio tuning override(s): {', '.join(sorted(unknown))}")
    if source.resolve() == destination.resolve():
        raise ValueError("Audio tuning input and output paths must differ")

    parameters = dict(TUNING_PRESETS[preset_id])
    parameters.update({key: value for key, value in overrides.items() if value is not None})
    low_cut = _number("low_cut_hz", parameters.get("low_cut_hz"), 20.0, 500.0)
    high_cut = _number("high_cut_hz", parameters.get("high_cut_hz"), 1000.0, 24000.0)
    gain = _number("gain_db", parameters.get("gain_db"), -24.0, 24.0)
    loudness = _number("loudnorm_i", parameters.get("loudnorm_i"), -30.0, -5.0)
    deess = _number("deess", parameters.get("deess"), 0.0, 12.0)
    if low_cut is not None and high_cut is not None and low_cut >= high_cut:
        raise ValueError("low_cut_hz must be lower than high_cut_hz")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if preset_id == "bypass" and not any(value is not None for value in overrides.values()):
        shutil.copy2(source, destination)
        return str(destination)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for audio tuning")
    sample_rate = int(sf.info(str(source)).samplerate)
    if sample_rate <= 0:
        raise ValueError(f"Audio tuning input has an invalid sample rate: {source}")

    nyquist = sample_rate / 2.0

    def warn(message: str) -> None:
        message = f"Audio tuning warning: {message}"
        print(f">> {message}", flush=True)
        if warning_callback is not None:
            warning_callback(message)

    def valid_cutoff(name: str, frequency: float | None) -> float | None:
        if frequency is None or frequency < nyquist:
            return frequency
        # Leave a small margin below Nyquist; FFmpeg otherwise logs an invalid
        # filter but exits successfully, silently leaving the audio unfiltered.
        effective = round(nyquist * 0.99, 2)
        warn(
            f"{name}={frequency:g} Hz is at or above the {nyquist:g} Hz Nyquist limit "
            f"for {sample_rate} Hz audio; using {effective:.2f} Hz."
        )
        return effective

    low_cut = valid_cutoff("low_cut_hz", low_cut)
    high_cut = valid_cutoff("high_cut_hz", high_cut)
    if low_cut is not None and high_cut is not None and low_cut >= high_cut:
        raise ValueError(f"low_cut_hz must be lower than high_cut_hz at the input's {sample_rate} Hz sample rate")

    filters: list[str] = []
    if low_cut is not None and low_cut > 20.5:
        filters.append(f"highpass=f={low_cut:.2f}:p=2")
    for frequency, eq_gain, width in parameters.get("eq", ()):
        if frequency < nyquist:
            filters.append(f"equalizer=f={frequency}:t=q:w={width}:g={eq_gain}")
        else:
            warn(f"skipping the {frequency:g} Hz equalizer band at the input's {sample_rate} Hz sample rate.")
    if deess is not None and deess >= 0.05:
        if 6500 < nyquist:
            filters.append(f"equalizer=f=6500:t=q:w=1.2:g={-abs(deess):.2f}")
        else:
            warn(f"skipping the 6500 Hz de-esser at the input's {sample_rate} Hz sample rate.")
    if high_cut is not None:
        filters.append(f"lowpass=f={high_cut:.2f}:p=2")
    if gain is not None and abs(gain) >= 0.05:
        filters.append(f"volume={gain:.2f}dB")
    if loudness is not None:
        filters.append(f"loudnorm=I={loudness:.2f}:TP=-1.5:LRA=11")
    filters.append("alimiter=limit=0.95")

    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-af",
        ",".join(filters),
        "-c:a",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        str(destination),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg audio tuning failed: {(completed.stderr or '').strip()[-2000:]}")
    return str(destination)


__all__ = ["TUNING_PRESETS", "apply_audio_tuning"]
