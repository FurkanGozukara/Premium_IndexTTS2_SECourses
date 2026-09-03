import os
from pathlib import Path

from ui.generation_tab import (
    _resolve_lora_reference_path,
    latest_reference_audio,
    reference_audio_choices,
    resolve_reference_selection,
)


def _audio(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"audio placeholder")
    return path.resolve()


def test_reference_audio_choices_creates_scans_and_sorts_library(tmp_path: Path) -> None:
    library = tmp_path / "reference_audios"

    assert reference_audio_choices(library) == []
    assert library.is_dir()

    first = _audio(library / "a.wav")
    second = _audio(library / "nested" / "B.MP3")
    _audio(library / "ignore.mp4")
    (library / "notes.txt").write_text("not audio", encoding="utf-8")

    assert reference_audio_choices(library) == [
        ("a.wav", str(first)),
        ("nested/B.MP3", str(second)),
    ]


def test_reference_priority_is_manual_then_lora_then_library(tmp_path: Path) -> None:
    library = tmp_path / "reference_audios"
    first = _audio(library / "a.wav")
    second = _audio(library / "b.wav")
    os.utime(first, ns=(1_700_000_000_000_000_000,) * 2)
    os.utime(second, ns=(1_800_000_000_000_000_000,) * 2)
    manual = _audio(tmp_path / "manual.wav")
    lora_reference = _audio(tmp_path / "voice_reference.wav")
    adapter = tmp_path / "voice.safetensors"

    selected = resolve_reference_selection(
        manual,
        second,
        "empty",
        str(adapter),
        True,
        reference_root=library,
        lora_reference=str(lora_reference),
    )
    assert selected.prompt == str(manual)
    assert selected.source == "manual"

    selected = resolve_reference_selection(
        None,
        second,
        "empty",
        str(adapter),
        True,
        reference_root=library,
        lora_reference=str(lora_reference),
    )
    assert selected.prompt == str(lora_reference)
    assert selected.source == "lora_auto"
    assert "LoRA / DoRA" in selected.message

    selected = resolve_reference_selection(
        first,
        second,
        "library_auto",
        str(adapter),
        True,
        reference_root=library,
        lora_reference=str(lora_reference),
    )
    assert selected.prompt == str(lora_reference)
    assert selected.source == "lora_auto"

    selected = resolve_reference_selection(
        None,
        second,
        "empty",
        None,
        True,
        reference_root=library,
    )
    assert selected.prompt == str(second)
    assert selected.source == "library_auto"

    selected = resolve_reference_selection(
        None,
        None,
        "empty",
        None,
        True,
        reference_root=library,
    )
    assert selected.prompt == str(second)
    assert selected.library_value == str(second)
    assert "latest modified" in selected.message
    assert latest_reference_audio(library) == str(second)


def test_missing_lora_reference_falls_back_to_latest_library_audio(tmp_path: Path) -> None:
    library = tmp_path / "reference_audios"
    older = _audio(library / "older.wav")
    newest = _audio(library / "newest.mp3")
    os.utime(older, ns=(1_700_000_000_000_000_000,) * 2)
    os.utime(newest, ns=(1_800_000_000_000_000_000,) * 2)
    adapter = _audio(tmp_path / "voice" / "voice.safetensors")

    selected = resolve_reference_selection(
        None,
        older,
        "empty",
        str(adapter),
        True,
        reference_root=library,
    )

    assert selected.prompt == str(newest)
    assert selected.library_value == str(newest)
    assert selected.source == "library_auto"
    assert "has no saved reference audio" in selected.message
    assert "latest modified" in selected.message


def test_best_checkpoint_finds_run_level_lora_reference(tmp_path: Path) -> None:
    run = tmp_path / "SECourses_Furkan_EN_DoRA_r128_v3"
    adapter = _audio(run / "best" / "SECourses_Furkan_EN_DoRA_r128_v3.safetensors")
    expected = _audio(run / "SECourses_Furkan_EN_DoRA_r128_v3_reference.wav")
    missing_best_copy = run / "best" / expected.name

    resolved = _resolve_lora_reference_path(adapter, missing_best_copy)

    assert resolved == str(expected)
