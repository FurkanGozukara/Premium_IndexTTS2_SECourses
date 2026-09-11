from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from indextts.training.speech_metrics import internal_pause_metrics
from indextts.utils.pause_cap import find_internal_pauses, shorten_long_pauses, shorten_long_pauses_file
from ui.generation_tab import GENERATION_DEFAULTS, RUNNER_REQUEST_KEYS, build_generation_request

RATE = 16000


def _speech(seconds: float, hz: float = 140.0) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (0.4 * np.sin(2 * np.pi * hz * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4.0 * t))).astype(np.float32)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * RATE), dtype=np.float32)


def test_long_pauses_are_shortened_and_edges_kept() -> None:
    audio = np.concatenate([_silence(0.3), _speech(1.0), _silence(0.8), _speech(0.8), _silence(0.2), _speech(0.6), _silence(0.5)])
    before = internal_pause_metrics(audio, RATE)
    assert before["pause_count"] == 2 and before["longest_pause_ms"] >= 780
    pauses = find_internal_pauses(audio, RATE)
    assert len(pauses) == 2
    result, report = shorten_long_pauses(audio, RATE, 300)
    assert report == {"max_pause_ms": 300, "pauses": 2, "shortened": 1, "removed_ms": report["removed_ms"], "protected": 0}
    assert 480 <= report["removed_ms"] <= 520
    after = internal_pause_metrics(result, RATE)
    assert after["pause_count"] == 2 and after["longest_pause_ms"] <= 320
    # Leading and trailing silence stay untouched, and the speech itself is not shortened.
    assert np.all(result[: int(0.29 * RATE)] == 0.0) and np.all(result[-int(0.49 * RATE):] == 0.0)
    assert abs((len(audio) - len(result)) / RATE - 0.5) < 0.03
    assert np.max(np.abs(np.diff(result))) < 0.5  # no click at the join


def test_short_pauses_and_disabled_cap_leave_audio_unchanged() -> None:
    audio = np.concatenate([_speech(0.5), _silence(0.25), _speech(0.5)])
    unchanged, report = shorten_long_pauses(audio, RATE, 300)
    assert np.array_equal(unchanged, audio) and report["shortened"] == 0 and report["pauses"] == 1
    disabled, report = shorten_long_pauses(audio, RATE, 0)
    assert disabled is audio and report["pauses"] == 0
    silent, report = shorten_long_pauses(np.zeros(RATE, dtype=np.float32), RATE, 200)
    assert len(silent) == RATE and report["pauses"] == 0


def test_file_round_trip_keeps_pcm16(tmp_path: Path) -> None:
    audio = np.concatenate([_speech(0.6), _silence(1.0), _speech(0.6)])
    source = tmp_path / "in.wav"
    sf.write(source, (audio * 32767).astype(np.int16), RATE, subtype="PCM_16")
    report = shorten_long_pauses_file(source, source, 400)
    assert report["shortened"] == 1 and report["duration_after_s"] < report["duration_before_s"]
    with sf.SoundFile(str(source)) as handle:
        assert handle.subtype == "PCM_16" and handle.samplerate == RATE
    stereo = tmp_path / "stereo.wav"
    sf.write(stereo, np.stack([audio, audio], axis=1), RATE)
    result, report = shorten_long_pauses(sf.read(str(stereo), dtype="float32")[0], RATE, 400)
    assert result.ndim == 2 and report["shortened"] == 1


def test_request_contract_carries_the_cap() -> None:
    assert "max_pause_ms" in RUNNER_REQUEST_KEYS and GENERATION_DEFAULTS["generation.max_pause_ms"] == 0
    values = dict(GENERATION_DEFAULTS)
    values["generation.max_pause_ms"] = 350
    request = build_generation_request(values, prompt="ref.wav", text="Hello there.")
    assert request["max_pause_ms"] == 350


def test_pcm16_pauses_with_a_noise_floor_are_detected(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    floor = (rng.standard_normal(int(0.9 * RATE)) * 0.0015).astype(np.float32)  # about -56 dBFS, not digital silence
    audio = np.concatenate([_speech(0.7), floor, _speech(0.7)])
    pcm = (audio * 32767).astype(np.int16)
    assert len(find_internal_pauses(pcm, RATE)) == 1
    result, report = shorten_long_pauses(pcm, RATE, 300)
    assert result.dtype == np.int16 and report["shortened"] == 1 and 560 <= report["removed_ms"] <= 640
    source = tmp_path / "noisy.wav"
    sf.write(source, pcm, RATE, subtype="PCM_16")
    file_report = shorten_long_pauses_file(source, source, 300)
    assert file_report["shortened"] == 1
