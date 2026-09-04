from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from indextts.training.media import (
    analyze_audio_quality,
    extract_audio,
    find_sidecar_subtitles,
    measure_loudness_lufs,
    normalize_loudness,
    probe_media,
)


FIXTURE = Path(r"G:\Index_TTS_v4\Lora_Training_Dataset\source2\video2.mp4")


@pytest.mark.skipif(not FIXTURE.is_file(), reason="SECourses media fixture is not installed")
def test_probe_extract_and_sidecars(tmp_path: Path) -> None:
    info = probe_media(FIXTURE)
    assert info.has_audio
    assert info.has_video
    assert info.duration_s > 800
    assert info.sample_rate == 48000
    assert info.channels == 2
    assert info.codec == "opus"

    output = tmp_path / "slice.wav"
    extract_audio(FIXTURE, output, sample_rate=24000, mono=True, start_s=1.0, end_s=4.0)
    audio, sample_rate = sf.read(output, dtype="float32", always_2d=False)
    assert sample_rate == 24000
    assert audio.ndim == 1
    assert audio.shape[0] == pytest.approx(3 * sample_rate, abs=sample_rate // 20)

    sidecars = [Path(path).suffix.casefold() for path in find_sidecar_subtitles(FIXTURE)]
    assert sidecars == [".srt", ".vtt", ".sbv"]


def test_loudness_normalization_on_synthetic_audio() -> None:
    sample_rate = 24000
    time = np.arange(sample_rate * 4, dtype=np.float32) / sample_rate
    audio = 0.04 * np.sin(2 * np.pi * 220 * time)
    before = measure_loudness_lufs(audio, sample_rate)
    normalized = normalize_loudness(audio, sample_rate, target_lufs=-20.0)
    after = measure_loudness_lufs(normalized, sample_rate)
    assert before < -20.0
    assert after == pytest.approx(-20.0, abs=0.35)
    assert np.max(np.abs(normalized)) < 1.0


def test_audio_quality_metrics_detect_silence_and_clipping() -> None:
    sample_rate = 1000
    audio = np.concatenate(
        [
            np.zeros(sample_rate, dtype=np.float32),
            np.full(sample_rate, 0.2, dtype=np.float32),
            np.ones(10, dtype=np.float32),
        ]
    )
    quality = analyze_audio_quality(
        audio,
        sample_rate,
        clipping_threshold=0.999,
        silence_threshold_dbfs=-40.0,
        frame_ms=20,
    )
    assert quality.peak_dbfs == pytest.approx(0.0, abs=1e-6)
    assert quality.clipping_ratio == pytest.approx(10 / audio.size)
    assert 0.45 <= quality.silence_ratio <= 0.55
