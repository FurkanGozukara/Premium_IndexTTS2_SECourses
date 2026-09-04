from pathlib import Path

import pytest

from tools.vram_benchmark import _parser, _resolve_reference_audio


def test_benchmark_reference_falls_back_to_reference_library(tmp_path: Path) -> None:
    fallback = tmp_path / "reference_audios" / "demo_voice.mp3"
    fallback.parent.mkdir(parents=True)
    fallback.write_bytes(b"audio")

    assert _resolve_reference_audio(None, root=tmp_path) == fallback.resolve()


def test_benchmark_reference_prefers_explicit_path(tmp_path: Path) -> None:
    explicit = tmp_path / "custom.wav"
    explicit.write_bytes(b"audio")

    parsed = _parser().parse_args(["--reference", str(explicit)])

    assert _resolve_reference_audio(parsed.reference, root=tmp_path) == explicit.resolve()


def test_benchmark_reference_error_lists_override(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="--reference"):
        _resolve_reference_audio(None, root=tmp_path)
