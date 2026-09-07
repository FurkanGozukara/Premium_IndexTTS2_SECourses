from __future__ import annotations

import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from indextts.training import dataset_prep
from indextts.training.dataset_manifest import load_manifest
from indextts.training.dataset_prep import DatasetPrepConfig, run_dataset_prep
from indextts.training.media import find_sidecar_subtitles
from indextts.training.whisper_asr import Transcript, Word
from indextts.utils.subtitle_utils import format_srt_timestamp


def _write_srt(path: Path, cues: list[tuple[str, float, float]]) -> None:
    path.write_text("\n\n".join(
        f"{i}\n{format_srt_timestamp(round(start * 1000))} --> "
        f"{format_srt_timestamp(round(end * 1000))}\n{text}"
        for i, (text, start, end) in enumerate(cues, 1)
    ) + "\n", encoding="utf-8")


@pytest.fixture
def recording(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Exercise actual preparation/cutting with deterministic CPU-only ASR output."""
    source = tmp_path / "input"
    source.mkdir()
    calls = []

    def create(sentences: list[tuple[str, float, float]], extension: str = ".mp4"):
        rate = 24000
        audio = np.zeros(round((max(end for _, _, end in sentences) + 1) * rate), dtype=np.float32)
        words = []
        for text, start, end in sentences:
            first, last = round(start * rate), round(end * rate)
            audio[first:last] = .08 * np.sin(2 * np.pi * 190 * np.arange(last - first) / rate)
            tokens = text.split()
            for index, token in enumerate(tokens):
                words.append(Word(token, start + (end - start) * index / len(tokens),
                                  start + (end - start) * (index + 1) / len(tokens)))
        media = source / ("talk" + extension)
        # The decoder boundary is mocked; actual filtering, alignment, trimming,
        # safe sentence cuts, normalization, and manifest writes run unchanged.
        sf.write(media, audio, rate, format="WAV")
        monkeypatch.setattr(dataset_prep, "probe_media", lambda _: SimpleNamespace(has_audio=True))
        monkeypatch.setattr(dataset_prep, "extract_audio", lambda src, dst, **_: shutil.copyfile(src, dst))

        def transcribe(**kwargs):
            calls.append(kwargs["media_path"])
            return Transcript(words, [], " ".join(text for text, _, _ in sentences)), False

        monkeypatch.setattr(dataset_prep, "_transcribe_cached", transcribe)
        return media, calls

    return create


def _config(tmp_path: Path, media: Path, **kwargs) -> DatasetPrepConfig:
    return DatasetPrepConfig(name="prepared", inputs=[str(media.parent)], output_root=str(tmp_path),
                             segmentation_mode="sentence_aligned", whisper_device="cpu", **kwargs)


def test_subtitle_order_prefers_requested_language_and_region(tmp_path: Path) -> None:
    media = tmp_path / "talk.mp4"
    names = ["talk.en.srt", "talk.es-ES.srt", "talk.srt", "talk.en.vtt"]
    for name in names:
        (tmp_path / name).write_text("", encoding="utf-8")
    assert Path(find_sidecar_subtitles(media, "ES")[0]).name == "talk.es-ES.srt"
    assert Path(find_sidecar_subtitles(media, "EN")[0]).name == "talk.en.srt"
    assert Path(find_sidecar_subtitles(media)[0]).name == "talk.srt"


def test_spanish_recording_uses_spanish_sidecar_not_english_translation(tmp_path, recording) -> None:
    spoken = "Vamos a guardar el archivo antes de cerrar la ventana."
    media, calls = recording([(spoken, 1, 5.2)])
    _write_srt(media.with_suffix(".en.srt"), [("We will save the file before closing the window.", .9, 5.3)])
    spanish = media.with_suffix(".es.srt")
    _write_srt(spanish, [(spoken, .9, 5.3)])
    summary = run_dataset_prep(_config(tmp_path, media, language="ES"))
    rows = load_manifest(summary.output_dir)
    assert [row["text"] for row in rows] == [spoken]
    assert rows[0]["language"] == "ES"
    assert summary.sources[0]["subtitle"] == spanish.resolve().as_posix()
    assert len(calls) == 1


def test_bad_preferred_sidecar_tries_an_alternative_with_the_same_asr(tmp_path, recording) -> None:
    spoken = "Vamos a guardar el archivo antes de cerrar la ventana."
    media, calls = recording([(spoken, 1, 5.2)])
    _write_srt(media.with_suffix(".es.srt"), [("Yesterday another person visited a completely different place.", .9, 5.3)])
    correct = media.with_suffix(".alt.srt")
    _write_srt(correct, [(spoken, .9, 5.3)])
    summary = run_dataset_prep(_config(tmp_path, media, language="ES"))
    assert [row["text"] for row in load_manifest(summary.output_dir)] == [spoken]
    assert summary.sources[0]["subtitle"] == correct.resolve().as_posix()
    attempts = summary.alignment["files"][0]["subtitle_candidates"]
    assert len(attempts) == 2
    assert attempts[0]["coverage"] < .6 < attempts[1]["coverage"]
    assert len(calls) == 1


def test_unrelated_caption_is_not_accepted_by_cue_timing_fallback(tmp_path, recording) -> None:
    spoken = "Vamos a guardar el archivo antes de cerrar la ventana."
    media, calls = recording([(spoken, 1, 5.2)])
    subtitle = media.with_suffix(".srt")
    _write_srt(subtitle, [("Yesterday another person visited a completely different place.", .9, 5.3)])
    original = subtitle.read_bytes()
    summary = run_dataset_prep(_config(tmp_path, media, language="ES"))
    assert load_manifest(summary.output_dir) == []
    assert any("No subtitle matched" in warning for warning in summary.warnings)
    assert subtitle.read_bytes() == original
    assert len(calls) == 1


def test_weak_lexical_alignment_keeps_only_cues_verified_by_spoken_form(tmp_path, recording) -> None:
    spoken = "six gigabytes five megabytes four terabytes three gigahertz two megahertz"
    caption = "6 GB, 5 MB, 4 TB, 3 GHz, 2 MHz."
    media, calls = recording([(spoken, 1, 5.2), ("Then close the window after saving your work.", 7, 11.2)])
    _write_srt(media.with_suffix(".srt"), [
        (caption, .9, 5.3),
        ("Yesterday another person visited a completely different place.", 6.9, 11.3),
    ])
    summary = run_dataset_prep(_config(tmp_path, media, target_s=5, max_s=7))
    rows = load_manifest(summary.output_dir)
    assert [row["text"] for row in rows] == [caption]
    assert rows[0]["transcript_source"] == "sidecar_srt+whisper_verified_cues"
    assert summary.alignment["files"][0]["coverage"] < .6
    assert summary.alignment["files"][0]["cue_transcripts_verified"]
    assert summary.filter_drop_counts["transcript_disagreement"] == 1
    assert len(calls) == 1  # No additional model or full-recording transcription.


def test_verified_cue_is_rechecked_when_silence_trimming_removes_a_quiet_word(tmp_path, recording) -> None:
    spoken = "six gigabytes five megabytes four terabytes three gigahertz two megahertz"
    media, _ = recording([(spoken, 1, 7)])
    waveform, rate = sf.read(media, dtype="float32")
    # The first word exists in the source ASR but is quieter than the trim
    # threshold. The remaining five seconds still pass duration/density checks.
    waveform[rate:round(1.6 * rate)] *= .0001
    sf.write(media, waveform, rate, format="WAV")
    _write_srt(media.with_suffix(".srt"), [("6 GB, 5 MB, 4 TB, 3 GHz, 2 MHz.", .9, 7.1)])
    summary = run_dataset_prep(_config(tmp_path, media))
    assert load_manifest(summary.output_dir) == []
    assert summary.alignment["files"][0]["cue_transcripts_verified"]
    assert summary.filter_drop_counts["transcript_disagreement"] == 1


@pytest.mark.parametrize(("extension", "direct"), [(".mp4", False), (".mp4", True), (".wav", True)])
def test_txt_numbers_keep_words_with_the_correct_audio_sentence(tmp_path, recording, extension, direct) -> None:
    media, calls = recording([
        ("Set the values to one hundred twenty three.", 1, 5.2),
        ("Then click save and close the window.", 7, 11.2),
    ], extension)
    original = "Set the values to 123. Then click save and close the window."
    transcript = media.with_suffix(".txt")
    transcript.write_text(original, encoding="utf-8")
    config = _config(tmp_path, media, target_s=5, max_s=7)
    if direct:
        config.inputs = [str(media)]
    summary = run_dataset_prep(config)
    rows = load_manifest(summary.output_dir)
    assert [row["text"] for row in rows] == [
        "Set the values to 123.", "Then click save and close the window."]
    assert rows[0]["source_end_s"] < 7
    assert rows[1]["source_start_s"] > 5.2
    assert all(row["sentence_aligned"] and row["alignment_coverage"] == 1 for row in rows)
    assert all(row["transcript_source"] == "sidecar_txt+whisper_sentence_aligned" for row in rows)
    assert transcript.read_text(encoding="utf-8") == original
    assert len(calls) == 1


def test_unrelated_txt_is_rejected_instead_of_spread_across_audio(tmp_path, recording) -> None:
    media, _ = recording([("Set the values before closing the window.", 1, 5.2)])
    media.with_suffix(".txt").write_text("Yesterday another person visited a completely different place.", encoding="utf-8")
    summary = run_dataset_prep(_config(tmp_path, media))
    assert load_manifest(summary.output_dir) == []
    assert any("TXT/Whisper word alignment" in warning for warning in summary.warnings)
    info = json.loads((Path(summary.output_dir) / "dataset_info.json").read_text(encoding="utf-8"))
    assert info["segment_count"] == 0


@pytest.mark.parametrize(("maximum", "kept"), [(.09, 0), (.10, 1)])
def test_configured_cue_error_threshold_changes_acceptance(tmp_path, recording, maximum, kept) -> None:
    # One internal unit differs: 10% error, with both transcript edges correct.
    media, _ = recording([("six gigabytes five megabytes four megabytes three gigahertz two megahertz", 1, 5.2)])
    _write_srt(media.with_suffix(".srt"), [("6 GB, 5 MB, 4 TB, 3 GHz, 2 MHz.", .9, 5.3)])
    summary = run_dataset_prep(_config(tmp_path, media, cue_fallback_max_error=maximum))
    assert len(load_manifest(summary.output_dir)) == kept


@pytest.mark.parametrize(("check_edges", "kept"), [(True, 0), (False, 1)])
def test_configured_cue_boundary_check_is_not_forced_on(tmp_path, recording, check_edges, kept) -> None:
    media, _ = recording([("gigabytes five megabytes four terabytes three gigahertz two megahertz", 1, 5.2)])
    _write_srt(media.with_suffix(".srt"), [("6 GB, 5 MB, 4 TB, 3 GHz, 2 MHz.", .9, 5.3)])
    summary = run_dataset_prep(_config(tmp_path, media, cue_fallback_check_boundary_words=check_edges))
    assert len(load_manifest(summary.output_dir)) == kept


def test_disabled_cue_recovery_does_not_run_verification(tmp_path, recording, monkeypatch) -> None:
    media, _ = recording([("six gigabytes five megabytes four terabytes three gigahertz two megahertz", 1, 5.2)])
    _write_srt(media.with_suffix(".srt"), [("6 GB, 5 MB, 4 TB, 3 GHz, 2 MHz.", .9, 5.3)])
    monkeypatch.setattr(dataset_prep, "_verified_cue_text", lambda *args: pytest.fail("Disabled recovery ran"))
    summary = run_dataset_prep(_config(tmp_path, media, cue_fallback_enabled=False))
    assert load_manifest(summary.output_dir) == []
    assert any("fallback is disabled" in warning for warning in summary.warnings)


def test_cue_verification_uses_the_configured_timing_margin() -> None:
    transcript = dataset_prep._source_transcript([
        Word("Hello", .90, .96), Word("world", 1.10, 1.40),
    ])
    config = DatasetPrepConfig(name="margin", inputs=["recording.wav"], cue_fallback_max_error=0)
    config.cue_fallback_margin_ms = 0
    assert not dataset_prep._verified_cue_text("Hello world.", 1, 1.5, transcript, config)
    config.cue_fallback_margin_ms = 100
    assert dataset_prep._verified_cue_text("Hello world.", 1, 1.5, transcript, config)


@pytest.mark.parametrize("selection", ["mode", "policy_with_alignment_alias", "metadata"])
def test_whisper_only_selection_ignores_supplied_transcript_text(tmp_path, recording, selection) -> None:
    spoken = "Set the values before closing the window."
    media, calls = recording([(spoken, 1, 5.2)], ".wav")
    media.with_suffix(".txt").write_text("This supplied transcript must not be used.", encoding="utf-8")
    config = _config(tmp_path, media)
    if selection == "mode":
        config.segmentation_mode = "whisper_only"
    else:
        config.subtitle_policy = "whisper_only"
        config.align_with_whisper = True
    if selection == "metadata":
        metadata = media.parent / "metadata.csv"
        metadata.write_text("talk.wav|This metadata text must not be used.|Speaker A\n", encoding="utf-8")
        config.inputs = [str(metadata)]
    summary = run_dataset_prep(config)
    rows = load_manifest(summary.output_dir)
    assert [row["text"] for row in rows] == [spoken]
    assert rows[0]["transcript_source"] == "whisper"
    if selection == "metadata":
        assert rows[0]["speaker"] == "Speaker A"
    assert len(calls) == 1


def test_explicit_cue_boundaries_does_not_silently_align_untimed_txt(tmp_path, recording) -> None:
    media, calls = recording([("Set the values before closing the window.", 1, 5.2)])
    media.with_suffix(".txt").write_text("Set the values before closing the window.", encoding="utf-8")
    config = _config(tmp_path, media)
    config.segmentation_mode = "cue_boundaries"
    summary = run_dataset_prep(config)
    assert load_manifest(summary.output_dir) == []
    assert any("Cue-boundary segmentation requires timed subtitles" in warning for warning in summary.warnings)
    assert calls == []
