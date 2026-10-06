"""Recordings without subtitles, noisy recordings and recordings without pauses still give clips.

The decoder and Whisper are mocked; segmentation, pause search, trimming, every clip check, loudness
normalization and the manifest and dataset_info writes run unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from indextts.training import dataset_prep, whisper_asr
from indextts.training.dataset_manifest import empty_dataset_message, load_manifest
from indextts.training.dataset_prep import DatasetPrepConfig, run_dataset_prep
from indextts.training.media import measure_edge_silence
from indextts.training.whisper_asr import Word

RATE = 24000
WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet".split()


def _sentence(index: int) -> str:
    words = [f"{word}{index}" for word in WORDS]
    return " ".join([words[0].capitalize(), *words[1:]]) + "."


def _speech(seconds: float, words: int) -> np.ndarray:
    """A tone with a 30 ms dip between words, as spoken words have short quiet closures."""
    audio = (0.1 * np.sin(2 * np.pi * 190 * np.arange(round(seconds * RATE)) / RATE)).astype(np.float32)
    for index in range(1, words):
        start = round(seconds * index / words * RATE)
        audio[start:start + round(0.03 * RATE)] = 0
    return audio


def _recording(gaps: list[float], *, seconds: float = 4.5, lead: float = 0.5, tail: float = 0.5,
               noise_dbfs: float | None = None, first_word_at_zero: bool = False):
    """Sentences of ten words, separated by ``gaps`` seconds of silence; returns audio, words and text."""
    pieces = [np.zeros(round(lead * RATE), dtype=np.float32)]
    words: list[Word] = []
    cursor = lead
    sentences = [_sentence(index) for index in range(len(gaps) + 1)]
    for index, sentence in enumerate(sentences):
        tokens = sentence.split()
        for position, token in enumerate(tokens):
            words.append(Word(token, round(cursor + seconds * position / len(tokens), 3),
                              round(cursor + seconds * (position + 1) / len(tokens), 3)))
        pieces.append(_speech(seconds, len(tokens)))
        cursor += seconds
        if index < len(gaps):
            pieces.append(np.zeros(round(gaps[index] * RATE), dtype=np.float32))
            cursor += gaps[index]
    pieces.append(np.zeros(round(tail * RATE), dtype=np.float32))
    audio = np.concatenate(pieces)
    if noise_dbfs is not None:
        audio = audio + np.random.default_rng(7).standard_normal(audio.size).astype(np.float32) * 10 ** (noise_dbfs / 20)
    if first_word_at_zero:
        # Recognizers often place the first word at 0.0 s although speech starts later.
        words[0] = Word(words[0].text, 0.0, words[0].end_s)
    return audio.astype(np.float32), words, " ".join(sentences)


@pytest.fixture
def prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Write recordings, mock decoding and Whisper, and run the real preparation on them."""
    source = tmp_path / "input"
    source.mkdir()
    transcripts: dict[str, whisper_asr.Transcript] = {}

    def transcribe(**kwargs):
        return transcripts[Path(kwargs["media_path"]).name], False

    monkeypatch.setattr(dataset_prep, "probe_media", lambda _: SimpleNamespace(has_audio=True))
    monkeypatch.setattr(dataset_prep, "extract_audio", lambda src, dst, **_: shutil.copyfile(src, dst))
    monkeypatch.setattr(dataset_prep, "_transcribe_cached", transcribe)

    def run(recordings: dict[str, tuple[np.ndarray, list[Word]]], name: str = "prepared", **kwargs):
        for filename, (audio, words) in recordings.items():
            sf.write(source / filename, audio, RATE, subtype="PCM_24")
            transcripts[filename] = whisper_asr._transcript_from_words(words)
        config = DatasetPrepConfig(name=name, inputs=[str(source)], output_root=str(tmp_path),
                                   segmentation_mode="sentence_aligned", whisper_device="cpu",
                                   export_reference_candidates=0, **kwargs)
        summary = run_dataset_prep(config)
        return summary, load_manifest(summary.output_dir), Path(summary.output_dir)

    return run


def _edges(dataset_dir: Path, row: dict) -> dict[str, float]:
    audio, rate = sf.read(dataset_dir / row["audio"], dtype="float32")
    return measure_edge_silence(audio, rate)


def test_whisper_only_recording_is_recut_at_verified_pauses_when_first_cuts_lack_quiet_edges(prepare) -> None:
    # 50 ms gaps after the 2nd and 4th sentences: the first cut groups two sentences and pads its edges past
    # those short gaps into speech, so every clip failed. Real 300 ms pauses follow sentences 1, 3 and 5.
    audio, words, text = _recording([0.3, 0.05, 0.3, 0.05, 0.3])
    summary, rows, dataset_dir = prepare({"talk.wav": (audio, words)})

    assert summary.status == "complete"
    assert rows and " ".join(row["text"] for row in rows) == text  # every word once, in order
    assert all(row["transcript_source"] == "whisper" for row in rows)
    assert all(row["boundary_method"] == "acoustic_sentence_repack" for row in rows)
    assert all("boundary_recovery" not in row for row in rows)
    assert all(min(_edges(dataset_dir, row).values()) >= 30 for row in rows)
    source = summary.sources[0]
    assert source["segmentation_mode"] == "whisper_sentence_aligned"
    assert source["boundary_recovery"]["first_pass_segments"] == 0
    assert source["boundary_recovery"]["pauses"] == "configured_threshold"
    # The replaced first cut's rejections no longer count against the dataset.
    assert summary.filter_drop_counts["unsafe_audio_boundary"] == 0
    assert load_manifest(dataset_dir / "boundary_rejections.jsonl") == []
    info = json.loads((dataset_dir / "dataset_info.json").read_text(encoding="utf-8"))
    assert info["boundary_recovery"]["sources"][0]["segments"] == len(rows)


def test_whisper_only_recording_with_clear_pauses_keeps_its_first_cut(prepare) -> None:
    audio, words, text = _recording([0.4, 0.4, 0.4, 0.4, 0.4])
    summary, rows, _ = prepare({"talk.wav": (audio, words)})

    assert " ".join(row["text"] for row in rows) == text
    assert all(row["boundary_method"] == "silence_snap" for row in rows)
    assert summary.sources[0]["segmentation_mode"] == "whisper_only"
    assert "boundary_recovery" not in summary.sources[0]
    assert summary.filter_drop_counts["unsafe_audio_boundary"] == 0


def test_short_recording_without_inner_pauses_is_kept_as_one_whole_clip(prepare) -> None:
    # A pre-cut clip: three sentences with 40 ms gaps and quiet edges, the first word timed at 0.0 s.
    audio, words, text = _recording([0.04, 0.04], lead=0.1, tail=0.3, first_word_at_zero=True)
    summary, rows, dataset_dir = prepare({"clip.wav": (audio, words)})

    assert [row["text"] for row in rows] == [text]
    assert rows[0]["boundary_method"] == "whole_recording"
    assert summary.sources[0]["segmentation_mode"] == "whole_recording"
    assert 3 * 4.5 + 2 * 0.04 <= rows[0]["duration_s"] <= audio.size / RATE  # all of the speech
    assert min(_edges(dataset_dir, rows[0]).values()) >= 30


def test_noisy_recording_is_cut_at_pauses_measured_against_its_own_noise_floor(prepare) -> None:
    # Constant noise keeps every pause above -40 dBFS after loudness normalization, so no cut could be
    # verified at the fixed threshold; speech stays 18 dB above the noise.
    audio, words, text = _recording([0.3, 0.05, 0.3, 0.05, 0.3], noise_dbfs=-41)
    summary, rows, dataset_dir = prepare({"noisy.wav": (audio, words)})

    assert rows and " ".join(row["text"] for row in rows) == text
    assert all(row["boundary_recovery"] == "noise_floor_pauses" for row in rows)
    recovery = summary.sources[0]["boundary_recovery"]
    assert recovery["pauses"] == "noise_floor"
    assert -40 < recovery["silence_threshold_dbfs"] < -20
    threshold = recovery["silence_threshold_dbfs"]
    for row in rows:
        clip, rate = sf.read(dataset_dir / row["audio"], dtype="float32")
        assert min(measure_edge_silence(clip, rate, threshold).values()) >= 30
    assert any("noise floor" in warning for warning in summary.warnings)


def test_noisy_subtitled_recording_whose_sentences_found_no_pause_is_recovered(prepare, tmp_path: Path) -> None:
    from indextts.utils.subtitle_utils import format_srt_timestamp

    audio, words, text = _recording([0.3, 0.05, 0.3, 0.05, 0.3], noise_dbfs=-41)
    (tmp_path / "input").mkdir(exist_ok=True)
    cues, first = [], 0
    for index, word in enumerate(words):
        if word.text.endswith("."):
            cue_words = words[first:index + 1]
            cues.append((" ".join(item.text for item in cue_words), cue_words[0].start_s, cue_words[-1].end_s))
            first = index + 1
    (tmp_path / "input" / "noisy.srt").write_text("\n\n".join(
        f"{number}\n{format_srt_timestamp(round(start * 1000))} --> {format_srt_timestamp(round(end * 1000))}\n{cue}"
        for number, (cue, start, end) in enumerate(cues, 1)) + "\n", encoding="utf-8")
    summary, rows, _ = prepare({"noisy.wav": (audio, words)})

    # The sentence repack found no pause at the fixed threshold; it was skipped before, it is now recovered.
    assert " ".join(row["text"] for row in rows) == text
    assert all(row["transcript_source"] == "sidecar_srt+whisper_sentence_aligned" for row in rows)
    assert all(row["boundary_recovery"] == "noise_floor_pauses" for row in rows)


def test_recording_without_any_pause_is_cut_unverified_only_when_the_dataset_would_be_empty(prepare) -> None:
    continuous, words, text = _recording([0.0, 0.0, 0.0, 0.0, 0.0], lead=0.0, tail=0.0)
    summary, rows, _ = prepare({"continuous.wav": (continuous, words)}, name="alone")

    assert summary.status == "complete"
    assert rows and all(row["boundary_recovery"] == "unverified_edges" for row in rows)
    assert summary.sources[0]["boundary_recovery"]["pauses"] == "unverified"
    assert summary.sources[0]["first_pass_filter_drop_counts"]
    assert any("without that check" in warning for warning in summary.warnings)

    clean, clean_words, clean_text = _recording([0.4, 0.4])
    summary, rows, _ = prepare({"continuous.wav": (continuous, words), "clean.wav": (clean, clean_words)},
                               name="mixed")
    # With usable clips elsewhere, the recording without pauses is left out as before.
    assert " ".join(row["text"] for row in rows) == clean_text
    assert all("boundary_recovery" not in row for row in rows)


def _unpunctuated(words: list[Word]) -> list[Word]:
    """Whisper's lowercase output in noise: no punctuation, and each word runs on to the next word's start."""
    plain = [Word(word.text.rstrip(".").lower(), word.start_s, word.end_s) for word in words]
    return [Word(word.text, word.start_s, plain[index + 1].start_s if index + 1 < len(plain) else word.end_s)
            for index, word in enumerate(plain)]


def test_unpunctuated_whisper_text_is_cut_at_acoustic_pauses(prepare) -> None:
    # One 28-second stretch without punctuation cannot be packed as sentences, and its word times hide the
    # 300 ms pauses; clips still end in those pauses, and every word is kept once.
    audio, words, _ = _recording([0.3, 0.3, 0.3, 0.3, 0.3])
    plain = _unpunctuated(words)
    summary, rows, dataset_dir = prepare({"plain.wav": (audio, plain)})

    assert " ".join(row["text"] for row in rows) == " ".join(word.text for word in plain)
    assert all(row["boundary_method"] == "acoustic_sentence_repack" for row in rows)
    assert all(row["boundary"] == "pause" and row["sentence_aligned"] is False for row in rows)
    assert all(min(_edges(dataset_dir, row).values()) >= 30 for row in rows)
    assert summary.sources[0]["segmentation_mode"] == "whisper_sentence_aligned"


def test_unpunctuated_txt_transcript_is_cut_at_acoustic_pauses(prepare, tmp_path: Path) -> None:
    audio, words, _ = _recording([0.3, 0.3, 0.3, 0.3, 0.3])
    plain = _unpunctuated(words)
    text = " ".join(word.text for word in plain)
    (tmp_path / "input").mkdir(exist_ok=True)
    (tmp_path / "input" / "plain.txt").write_text(text, encoding="utf-8")
    summary, rows, dataset_dir = prepare({"plain.wav": (audio, plain)})

    # The supplied text had no sentence end, so the sentence cut found nothing; the source is not skipped.
    assert " ".join(row["text"] for row in rows) == text
    assert all(row["transcript_source"] == "sidecar_txt+whisper_sentence_aligned" for row in rows)
    assert all(min(_edges(dataset_dir, row).values()) >= 30 for row in rows)
    assert not any("Could not decode/process" in warning for warning in summary.warnings)


def test_recording_with_too_little_speech_is_still_skipped_with_its_reason(prepare) -> None:
    audio, words, _ = _recording([], seconds=2.0)  # one sentence of two seconds: shorter than any clip
    summary, rows, _ = prepare({"short.wav": (audio, words)})

    assert rows == [] and summary.status == "empty"
    assert any("transcript produced no usable timed segments" in warning and "minimum clip length of 4 s" in warning
               for warning in summary.warnings)
    assert summary.empty_reason == "1 source(s) could not be processed (see the warnings)"


def test_range_loudness_follows_pyloudnorm_for_every_cut() -> None:
    from indextts.training.audio_boundaries import RangeLoudness
    from indextts.training.media import measure_loudness_lufs

    audio, _, _ = _recording([0.3, 0.05, 0.3], noise_dbfs=-50)
    meter = RangeLoudness(audio, RATE)
    rng = np.random.default_rng(3)
    for _ in range(40):
        length = int(rng.uniform(0.2, 12.0) * RATE)
        first = int(rng.integers(0, audio.size - length))
        piece = audio[first:first + length]
        assert meter.loudness(first, first + length) == pytest.approx(measure_loudness_lufs(piece, RATE), abs=0.25)
        assert meter.peak(first, first + length) == float(np.max(np.abs(piece)))


def test_preparation_without_any_usable_clip_is_empty_and_names_the_reason(prepare, tmp_path: Path) -> None:
    from indextts.training.features import cache_dataset_features

    audio, words, _ = _recording([0.4])
    one_word = [Word("Hello.", words[0].start_s, words[-1].end_s)]  # one word over ten seconds
    summary, rows, dataset_dir = prepare({"sparse.wav": (audio, one_word)}, name="empty", min_words=1)

    assert rows == []
    assert summary.status == "empty"
    assert "too few words" in summary.empty_reason
    info = json.loads((dataset_dir / "dataset_info.json").read_text(encoding="utf-8"))
    assert info["status"] == "empty" and info["empty_reason"] == summary.empty_reason
    assert any(warning.startswith("No usable clips:") for warning in info["warnings"])
    message = empty_dataset_message(dataset_dir)
    assert "'empty' has no clips" in message and summary.empty_reason in message
    with pytest.raises(FileNotFoundError, match="has no clips"):
        cache_dataset_features({"dataset_dir": str(dataset_dir)})
    # An empty result is not a finished dataset: the same name can be prepared again without Overwrite.
    assert dataset_prep.unfinished_preparation(dataset_dir)


def test_empty_message_explains_datasets_prepared_before_empty_reasons(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "older"
    dataset_dir.mkdir()
    (dataset_dir / "manifest.jsonl").write_text("", encoding="utf-8")
    (dataset_dir / "dataset_info.json").write_text(json.dumps({
        "status": "complete", "sources": [{}, {}],
        "filter_drop_counts": {"unsafe_audio_boundary": 10, "duration": 0},
    }), encoding="utf-8")
    message = empty_dataset_message(dataset_dir)
    assert "2 source(s)" in message and "10 no quiet audio at a cut edge" in message


def test_prep_worker_reports_an_empty_preparation_as_failed(tmp_path: Path, monkeypatch) -> None:
    from indextts.training import prep_worker

    summary = dataset_prep.DatasetSummary(name="x", output_dir=str(tmp_path), status="empty", segment_count=0,
                                          total_duration_s=0.0, word_count=0, empty_reason="dropped clips: 3 no audio")
    monkeypatch.setattr(prep_worker, "run_dataset_prep", lambda *args, **kwargs: summary)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"name": "x", "inputs": [str(tmp_path)]}), encoding="utf-8")
    assert prep_worker.main(["--config", str(config), "--state-dir", str(tmp_path / "state")]) == 0
    status = json.loads((tmp_path / "state" / "status.json").read_text(encoding="utf-8"))
    assert status["phase"] == "failed"
    assert "kept no clip: dropped clips: 3 no audio" in status["message"]
