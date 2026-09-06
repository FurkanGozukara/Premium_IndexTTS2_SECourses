import json

import numpy as np
import pytest
import soundfile as sf

from indextts.training.audio_boundaries import build_safe_sentence_segments
from indextts.training.dataset_prep import DatasetPrepConfig, run_dataset_prep
from indextts.training.media import compute_energy_envelope, measure_edge_silence, normalize_loudness
from indextts.training.subtitles import SubtitleCue, build_caption_transcript, clean_cues
from indextts.training.whisper_asr import Word, align_caption_words


def fixture():
    text = "Alpha ends. Bravo continues. Charlie finishes."
    caption = build_caption_transcript(clean_cues([SubtitleCue(1, 0, 7200, text)]))
    words = [Word(w, a, b) for w, a, b in [
        ("Alpha", .3, 1.4), ("ends", 1.4, 2.0),
        ("Bravo", 2.0, 3.2), ("continues", 3.2, 4.2),
        ("Charlie", 4.2, 5.5), ("finishes", 5.5, 6.8),
    ]]
    aligned = align_caption_words(caption.words, words)
    energy = np.ones(720, dtype=np.float32) * .2
    energy[:30] = energy[440:460] = energy[690:] = 0
    # "continues" has a release after its ASR end, then a real pause.
    # There is no pause between the first and second sentences.
    return caption, words, aligned, energy


def config(**kwargs):
    return DatasetPrepConfig(name="safe", inputs=["source"], min_s=1, target_s=3,
                             max_s=5, min_words_per_second=.1, export_reference_candidates=0,
                             **kwargs)


def test_repacks_across_bad_cut_without_losing_or_repeating_words():
    caption, _, aligned, energy = fixture()
    segments, rejected = build_safe_sentence_segments(caption, aligned.words, energy, config())
    assert not rejected
    assert len(segments) == 2
    assert " ".join(s.text for s in segments) == caption.text
    assert segments[0].text == "Alpha ends. Bravo continues."
    assert segments[0].end_ms >= 4430
    assert segments[0].end_ms <= segments[1].start_ms
    assert all(1000 <= s.duration_ms <= 5000 for s in segments)
    assert all(len(s.word_timestamps) in (2, 4) for s in segments)


def test_continuous_audio_is_not_padded_or_cut_to_force_acceptance():
    caption, _, aligned, energy = fixture()
    energy[440:460] = .2
    segments, rejected = build_safe_sentence_segments(caption, aligned.words, energy, config())
    assert segments == []  # Whole continuous passage is over the five-second limit.
    assert len(rejected) == 3
    assert " ".join(row["text"] for row in rejected) == caption.text


def test_edge_measurement_anchors_frames_to_both_actual_edges():
    audio = np.ones(2457, dtype=np.float32) * .1
    audio[:720] = audio[-720:] = .001
    assert measure_edge_silence(audio, 24000) == {
        "leading_silence_ms": 30.0, "trailing_silence_ms": 30.0,
    }
    audio[-5:] = .8
    assert measure_edge_silence(audio, 24000)["trailing_silence_ms"] == 0
    audio[:240] = np.nan
    assert measure_edge_silence(audio, 24000)["leading_silence_ms"] == 0


@pytest.mark.parametrize("sr", [16000, 22050, 24000, 44100, 48000])
def test_energy_frame_times_do_not_drift_for_supported_sample_rates(sr):
    audio = np.full(sr * 100, .2, dtype=np.float32)
    audio[sr * 90:round(sr * 90.1)] = 0
    energy = compute_energy_envelope(audio, sr, hop_ms=10)
    assert len(energy) == 10000
    np.testing.assert_array_equal(energy[9000:9010], 0)
    assert energy[8999] > .1 and energy[9010] > .1


def test_pause_search_accounts_for_gain_on_quiet_recordings():
    caption, _, aligned, _ = fixture()
    sr = 24000
    levels = np.full(720, .02, dtype=np.float32)
    levels[:30] = levels[690:] = .00001
    levels[420:446] = .0019  # Quiet before normalization, too loud afterward.
    levels[446:466] = .00001
    audio = np.repeat(levels, sr // 100) * np.sin(2 * np.pi * 200 * np.arange(720 * sr // 100) / sr)
    energy = compute_energy_envelope(audio, sr)
    segments, rejected = build_safe_sentence_segments(caption, aligned.words, energy, config(), audio=audio)
    assert not rejected
    assert " ".join(s.text for s in segments) == caption.text
    for segment in segments:
        piece = audio[round(segment.start_ms * sr / 1000):round(segment.end_ms * sr / 1000)]
        normalized = normalize_loudness(piece, sr, -20)
        assert min(measure_edge_silence(normalized, sr).values()) >= 30
    assert segments[0].end_ms >= 4490


def test_neighbors_share_one_pause_even_when_their_normalization_gains_differ():
    caption, _, aligned, _ = fixture()
    sr = 24000
    levels = np.full(720, .02, dtype=np.float32)
    levels[240:420] = .2
    levels[460:680] = .08
    levels[:30] = levels[440:460] = levels[690:] = .00001
    levels[210:220] = .004  # Acceptable for the loud neighbor, not the quiet one.
    levels[220:230] = .05   # A late release must not appear in both clips.
    levels[230:240] = .00001
    audio = np.repeat(levels, sr // 100) * np.sin(2 * np.pi * 200 * np.arange(720 * sr // 100) / sr)
    settings = config()
    settings.max_s = 3
    settings.target_s = 2
    segments, rejected = build_safe_sentence_segments(
        caption, aligned.words, compute_energy_envelope(audio, sr), settings, audio=audio,
    )
    assert not rejected
    assert len(segments) == 3
    assert " ".join(s.text for s in segments) == caption.text
    assert all(a.end_ms <= b.start_ms for a, b in zip(segments, segments[1:]))
    assert segments[0].end_ms >= 2330


def test_full_preparation_recovers_source_release_and_preserves_transcript(tmp_path, monkeypatch):
    from indextts.training import dataset_prep, whisper_asr
    caption, words, _, energy = fixture()
    sr = 24000
    waveform = np.repeat(energy, sr // 100) * np.sin(2 * np.pi * 200 * np.arange(720 * sr // 100) / sr)
    source = tmp_path / "narration.wav"
    sf.write(source, waveform, sr, subtype="PCM_24")
    source.with_suffix(".srt").write_text("1\n00:00:00,000 --> 00:00:07,200\n" + caption.text + "\n")
    transcript = whisper_asr._transcript_from_words(words)
    monkeypatch.setattr(dataset_prep, "_transcribe_cached", lambda **kwargs: (transcript, True))
    prep = config(segmentation_mode="sentence_aligned", output_root=str(tmp_path),
                  loudness_normalize=False, trim_silence=False)
    prep.inputs = [str(source)]
    summary = run_dataset_prep(prep)
    rows = [json.loads(line) for line in (tmp_path / "safe/manifest.jsonl").read_text().splitlines()]
    assert summary.segment_count == len(rows) == 2
    assert " ".join(row["text"] for row in rows) == caption.text
    first = rows[0]
    assert first["source_end_s"] > 4.43
    for row in rows:
        audio, actual_sr = sf.read(tmp_path / "safe" / row["audio"], dtype="float32")
        assert actual_sr == sr
        assert min(measure_edge_silence(audio, sr).values()) >= 30
        start = round(row["source_start_s"] * sr)
        np.testing.assert_allclose(audio, waveform[start:start + len(audio)], atol=4e-5)
        assert row["boundary_method"] == "acoustic_sentence_repack"


@pytest.mark.parametrize("value", [-1, 501])
def test_invalid_minimum_context_is_rejected(value):
    with pytest.raises(ValueError, match="min_edge_silence_ms"):
        config(min_edge_silence_ms=value).validate()
