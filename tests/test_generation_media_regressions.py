"""Synthetic, CPU-only FFmpeg checks for export and Nyquist safety."""

import json
import shutil
import subprocess
import wave

import numpy as np
import pytest

from indextts.utils.audio_tuning import apply_audio_tuning


def _wav(path, duration=0.25, rate=22050):
    frames = round(duration * rate)
    samples = (5000 * np.sin(2 * np.pi * 220 * np.arange(frames) / rate)).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(samples.tobytes())
    return frames / rate


@pytest.fixture
def ffmpeg_tools():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe are required")


def _probe(path):
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(completed.stdout)


@pytest.mark.parametrize("bitrate", ["128k", "192k", "256k", "320k"])
def test_mp3_exports_the_selected_bitrate_from_native_22050_hz(tmp_path, ffmpeg_tools, bitrate):
    from webui_generation_runner import convert_wav_to_mp3

    source, destination = tmp_path / "source.wav", tmp_path / "export.mp3"
    _wav(source, duration=1.0)
    assert convert_wav_to_mp3(str(source), str(destination), bitrate=bitrate, remove_source=False) == str(destination)
    audio = next(stream for stream in _probe(destination)["streams"] if stream["codec_type"] == "audio")
    assert audio["codec_name"] == "mp3"
    assert int(audio["bit_rate"]) == int(bitrate[:-1]) * 1000
    assert int(audio["sample_rate"]) == (22050 if bitrate == "128k" else 44100)
    assert source.is_file()


def test_unsupported_mp3_bitrate_does_not_silently_quantize_or_remove_source(tmp_path, ffmpeg_tools, capsys):
    from webui_generation_runner import convert_wav_to_mp3

    source, destination = tmp_path / "source.wav", tmp_path / "export.mp3"
    _wav(source)
    assert convert_wav_to_mp3(str(source), str(destination), bitrate="200k") == str(source)
    assert source.is_file() and not destination.exists()
    assert "Unsupported MP3 bitrate: 200k" in capsys.readouterr().out


@pytest.mark.parametrize("duration", [0.013, 1.013, 11.0])
def test_mp4_container_and_streams_end_at_audio_boundary(tmp_path, ffmpeg_tools, duration):
    from PIL import Image
    from webui_generation_runner import create_mp4_from_image_audio

    source, image, destination = tmp_path / "source.wav", tmp_path / "still.png", tmp_path / "export.mp4"
    actual_duration = _wav(source, duration=duration)
    # A panoramic source also exercises the existing even-dimension scaling.
    Image.new("RGB", (96, 32), (20, 40, 60)).save(image)
    assert create_mp4_from_image_audio(str(image), str(source), str(destination)) == str(destination)
    probe = _probe(destination)
    video = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
    audio = next(stream for stream in probe["streams"] if stream["codec_type"] == "audio")
    assert video["codec_name"] == "h264" and audio["codec_name"] == "aac"
    assert video["width"] % 2 == video["height"] % 2 == 0
    assert abs(float(video["duration"]) - actual_duration) <= 1 / 30 + 0.001
    assert float(video["duration"]) + 0.000001 >= actual_duration
    assert abs(float(probe["format"]["duration"]) - actual_duration) <= 1 / 30 + 0.001
    assert abs(float(audio["duration"]) - actual_duration) <= 1 / 22050 + 0.001


@pytest.mark.parametrize("preset,cutoff", [("bypass", 12000), ("bypass", 11025), ("normalize", None)])
def test_tuning_clamps_invalid_nyquist_cutoffs_with_visible_warning(
    tmp_path, ffmpeg_tools, monkeypatch, capsys, preset, cutoff,
):
    import indextts.utils.audio_tuning as tuning

    source, destination = tmp_path / "source.wav", tmp_path / "tuned.wav"
    _wav(source)
    warnings, commands = [], []
    real_run = subprocess.run

    def capture_run(command, **kwargs):
        completed = real_run(command, **kwargs)
        commands.append(command)
        assert "Invalid frequency" not in completed.stderr
        return completed

    monkeypatch.setattr(tuning.subprocess, "run", capture_run)
    overrides = {} if cutoff is None else {"high_cut_hz": cutoff}
    apply_audio_tuning(source, destination, preset, warning_callback=warnings.append, **overrides)
    audio_filter = commands[0][commands[0].index("-af") + 1]
    assert "lowpass=f=10914.75:p=2" in audio_filter
    assert len(warnings) == 1 and "Nyquist" in warnings[0]
    assert warnings[0] in capsys.readouterr().out
    with wave.open(str(destination), "rb") as output:
        assert output.getframerate() == 22050
        assert abs(output.getnframes() - round(0.25 * 22050)) <= 2


def test_tuning_skips_equalizer_bands_above_nyquist_for_low_rate_audio(tmp_path, ffmpeg_tools):
    source, destination = tmp_path / "source.wav", tmp_path / "tuned.wav"
    _wav(source, rate=8000)
    warnings = []
    apply_audio_tuning(source, destination, "clear_narration", warning_callback=warnings.append)
    assert len(warnings) == 2
    assert "using 3960.00 Hz" in warnings[0]
    assert "skipping the 6500 Hz de-esser" in warnings[1]
    assert destination.is_file()


def test_bypass_keeps_original_bytes_and_does_not_warn(tmp_path):
    source, destination = tmp_path / "source.wav", tmp_path / "tuned.wav"
    _wav(source)
    warnings = []
    apply_audio_tuning(source, destination, warning_callback=warnings.append)
    assert destination.read_bytes() == source.read_bytes()
    assert warnings == []
