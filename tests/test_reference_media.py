from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from ui.common import extract_reference_audio, parse_reference_time_ranges
from ui.generation_tab import load_reference_media, prepare_reference_for_generation


def _tone(path: Path, duration_s: float = 4.0, sample_rate: int = 24000) -> Path:
    samples = np.arange(round(duration_s * sample_rate), dtype=np.float32)
    audio = 0.15 * np.sin(2.0 * np.pi * 220.0 * samples / sample_rate)
    sf.write(path, audio, sample_rate, subtype="PCM_16")
    return path


def test_reference_range_parser_supports_seconds_and_timestamps() -> None:
    assert parse_reference_time_ranges("1:4; 7.5:12") == [(1.0, 4.0), (7.5, 12.0)]
    assert parse_reference_time_ranges("01:02-01:08.5\n1:02:03->1:02:04.25") == [
        (62.0, 68.5),
        (3723.0, 3724.25),
    ]
    with pytest.raises(ValueError, match="must end after"):
        parse_reference_time_ranges("4:1")
    with pytest.raises(ValueError, match="Invalid range"):
        parse_reference_time_ranges("not a range")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_reference_audio_ranges_are_trimmed_and_joined(tmp_path: Path) -> None:
    source = _tone(tmp_path / "source.wav")
    output, message = extract_reference_audio(source, "0.25:1.25; 2:3.5")
    try:
        assert output is not None
        assert sf.info(output).duration == pytest.approx(2.5, abs=0.03)
        assert "2 selected ranges (2.50s total)" in message
    finally:
        if output:
            Path(output).unlink(missing_ok=True)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_extract_ranges_resolves_manual_lora_and_latest_library_sources(
    tmp_path: Path,
) -> None:
    library = tmp_path / "reference_audios"
    library.mkdir()
    older = _tone(library / "older.wav")
    newest = _tone(library / "newest.wav")
    manual = _tone(tmp_path / "manual.wav")
    lora_reference = _tone(tmp_path / "voice_reference.wav")
    adapter = tmp_path / "voice" / "best" / "voice.safetensors"
    adapter.parent.mkdir(parents=True)
    adapter.touch()
    os.utime(older, ns=(1_700_000_000_000_000_000,) * 2)
    os.utime(newest, ns=(1_800_000_000_000_000_000,) * 2)

    cases = [
        {
            "current_reference": manual,
            "reference_media": manual,
            "reference_source": "manual_media",
            "lora_path": str(adapter),
            "lora_reference": str(lora_reference),
            "expected_media": manual,
            "expected_source": "manual_media",
        },
        {
            "current_reference": None,
            "reference_media": None,
            "reference_source": "empty",
            "lora_path": str(adapter),
            "lora_reference": str(lora_reference),
            "expected_media": lora_reference,
            "expected_source": "lora_auto",
        },
        {
            "current_reference": None,
            "reference_media": None,
            "reference_source": "empty",
            "lora_path": "",
            "lora_reference": None,
            "expected_media": newest,
            "expected_source": "library_auto",
        },
        {
            "current_reference": None,
            "reference_media": None,
            "reference_source": "empty",
            "lora_path": str(tmp_path / "missing" / "missing.safetensors"),
            "lora_reference": None,
            "expected_media": newest,
            "expected_source": "library_auto",
        },
    ]

    outputs: list[str] = []
    try:
        for case in cases:
            prepared = prepare_reference_for_generation(
                case["current_reference"],
                case["reference_media"],
                None,
                case["reference_source"],
                "0.25:1.25; 2:2.5",
                case["lora_path"],
                True,
                reference_root=library,
                lora_reference=case["lora_reference"],
            )
            outputs.append(prepared.prompt)
            assert prepared.media == str(case["expected_media"].resolve())
            assert prepared.source == case["expected_source"]
            assert sf.info(prepared.prompt).duration == pytest.approx(1.5, abs=0.04)
            assert "2 selected ranges (1.50s total)" in prepared.message
            assert "will be used" in prepared.message
    finally:
        for output in outputs:
            Path(output).unlink(missing_ok=True)


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg and FFprobe are required",
)
def test_video_reference_has_video_preview_and_extracted_audio(tmp_path: Path) -> None:
    source_audio = _tone(tmp_path / "source.wav", duration_s=2.0)
    video = tmp_path / "reference.mkv"
    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x180:r=24:d=2",
            "-i",
            str(source_audio),
            "-shortest",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    output, preview, message = load_reference_media(video, "0.25:1.25")
    try:
        assert output is not None
        assert preview == str(video.resolve())
        assert sf.info(output).duration == pytest.approx(1.0, abs=0.04)
        assert "from video" in message
    finally:
        if output:
            Path(output).unlink(missing_ok=True)
