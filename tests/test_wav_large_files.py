"""RIFF/RF64 export boundaries and lossless PCM16 round trips."""

import wave

import numpy as np
import pytest
import soundfile as sf

from indextts.utils import subtitle_utils as wav_io


@pytest.mark.parametrize("channels,frames,container", [
    (1, 9, b"RIFF"), (1, 10, b"RF64"),
    (2, 4, b"RIFF"), (2, 5, b"RF64"),
])
def test_wav_size_boundary_counts_all_channels(tmp_path, monkeypatch, channels, frames, container):
    # Like the real limit, 19 is odd: the last complete PCM16 frame must fit.
    monkeypatch.setattr(wav_io, "_WAV_MAX_DATA_BYTES", 19)
    monkeypatch.setattr(wav_io, "_WAV_WRITE_CHUNK_FRAMES", 3)
    audio = np.resize(np.array([-32768, -1, 0, 1, 32767], dtype=np.int16), (frames, channels))
    path = tmp_path / "audio.wav"
    assert wav_io.write_pcm16_wav(audio, 22050, str(path)) == str(path)
    assert path.read_bytes()[:4] == container
    rate, decoded = wav_io.read_pcm16_wav(str(path))
    assert rate == 22050
    assert decoded.dtype == np.int16
    np.testing.assert_array_equal(decoded, audio)
    if container == b"RIFF":
        with wave.open(str(path), "rb") as handle:
            assert handle.getnframes() == frames
            assert handle.getnchannels() == channels
            assert handle.readframes(frames) == audio.astype("<i2").tobytes()


@pytest.mark.parametrize("audio", [
    np.array([], dtype=np.int16),
    np.zeros((0, 2), dtype=np.int16),
    np.array([-32768, 0, 32767], dtype=np.int16),
    np.arange(60, dtype=np.int16).reshape(3, 20).T[::2],
    np.broadcast_to(np.array([123, -456], dtype=np.int16), (13, 2)),
])
def test_wav_empty_mono_and_strided_audio(tmp_path, monkeypatch, audio):
    monkeypatch.setattr(wav_io, "_WAV_WRITE_CHUNK_FRAMES", 3)
    path = tmp_path / "nested" / "audio.wav"
    wav_io.write_pcm16_wav(audio, 44100, str(path))
    rate, decoded = wav_io.read_pcm16_wav(str(path))
    assert rate == 44100
    assert path.read_bytes()[:4] == b"RIFF"
    np.testing.assert_array_equal(decoded, audio[:, None] if audio.ndim == 1 else audio)


@pytest.mark.parametrize("container", ["WAV", "RF64"])
def test_reader_still_rejects_non_pcm16(tmp_path, container):
    path = tmp_path / "pcm24.wav"
    sf.write(path, np.zeros(10), 22050, format=container, subtype="PCM_24")
    with pytest.raises(ValueError, match="Expected 16-bit PCM WAV"):
        wav_io.read_pcm16_wav(str(path))


def test_tensor_writer_uses_rf64_and_preserves_pcm_scale(tmp_path, monkeypatch):
    import torch
    from indextts.utils.common import save_pcm_wav

    monkeypatch.setattr(wav_io, "_WAV_MAX_DATA_BYTES", 0)
    audio = torch.tensor([[-40000.0, -0.7, 0.7, 40000.0], [100.0, 200.0, 300.0, 400.0]])
    path = tmp_path / "tensor.wav"
    save_pcm_wav(path, audio, 22050)
    assert path.read_bytes()[:4] == b"RF64"
    rate, decoded = wav_io.read_pcm16_wav(str(path))
    assert rate == 22050
    np.testing.assert_array_equal(decoded, [[-32767, 100], [-1, 200], [1, 300], [32767, 400]])


@pytest.mark.parametrize("shorten", [False, True])
def test_pause_rewrite_preserves_rf64(tmp_path, shorten):
    from indextts.utils.pause_cap import shorten_long_pauses_file

    audio = np.concatenate([
        np.full(16000, 10000, dtype=np.int16),
        np.zeros(16000, dtype=np.int16),
        np.full(16000, -10000, dtype=np.int16),
    ])
    source, target = tmp_path / "source.wav", tmp_path / "target.wav"
    sf.write(source, audio, 16000, subtype="PCM_16", format="RF64")
    report = shorten_long_pauses_file(source, target, 300 if shorten else 0)
    assert report["shortened"] == int(shorten)
    with sf.SoundFile(target) as result:
        assert result.format == "RF64" and result.subtype == "PCM_16"
        assert (result.frames < len(audio)) == shorten
