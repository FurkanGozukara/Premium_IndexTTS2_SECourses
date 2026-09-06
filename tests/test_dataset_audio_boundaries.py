"""Sentence alignment must retain audio beyond imperfect word timestamps."""

import numpy as np
import pytest

from indextts.training.dataset_prep import DatasetPrepConfig, _snap_segments
from indextts.training.segmenter import apply_padding_and_limits
from indextts.training.subtitles import Segment


def sentence(start, end):
    return Segment(start, end, "A complete sentence.", word_timestamps=[
        {"start_s": start / 1000, "end_s": end / 1000, "text": "sentence."},
    ])


@pytest.mark.parametrize("silence", [False, True])
def test_alignment_snapping_cannot_remove_requested_padding(silence):
    energy = np.ones(500, dtype=np.float32)
    # Word timestamps land in a low-energy phoneme, with speech on both sides.
    energy[100] = energy[300] = 0
    if silence:
        energy[80:87] = energy[313:320] = 0.001
    config = DatasetPrepConfig(name="test", inputs=[], pad_ms=60)
    padded = apply_padding_and_limits([sentence(1000, 3000)], 60, 5000)
    result = _snap_segments(padded, energy, config, protect_words=True)[0]
    assert result.start_ms <= 940
    assert result.end_ms >= 3060
    if silence:
        assert result.start_ms == 860
        assert result.end_ms == 3140


@pytest.mark.parametrize("snap_to_silence", [False, True])
def test_adjacent_sentences_share_a_tight_gap_without_cutting_word_times(snap_to_silence):
    energy = np.ones(600, dtype=np.float32)
    energy[299] = energy[306] = 0
    config = DatasetPrepConfig(name="test", inputs=[], pad_ms=60, snap_to_silence=snap_to_silence)
    padded = apply_padding_and_limits([sentence(1000, 3000), sentence(3080, 5000)], 60, 6000)
    result = _snap_segments(padded, energy, config, protect_words=True)
    assert result[0].end_ms == result[1].start_ms == 3040
    assert result[0].start_ms <= 940
    assert result[1].end_ms >= 5060


def test_padding_remains_bounded_by_media_edges():
    energy = np.ones(200, dtype=np.float32)
    config = DatasetPrepConfig(name="test", inputs=[], pad_ms=60)
    padded = apply_padding_and_limits([sentence(10, 1950)], 60, 2000)
    result = _snap_segments(padded, energy, config, protect_words=True)[0]
    assert result.start_ms == 0
    assert result.end_ms == 2000


def test_touching_asr_times_are_refined_to_sustained_acoustic_silence():
    energy = np.ones(600, dtype=np.float32)
    energy[300] = 0  # An in-word dip at the erroneous shared timestamp.
    energy[292:299] = 0.0001  # A quieter plosive closure is still inside the word.
    energy[313:320] = 0.001  # The actual pause follows the final phoneme.
    first = sentence(1000, 3000)
    second = sentence(3000, 5000)
    second.word_timestamps = [
        {"text": "Next", "start_s": 3.0, "end_s": 3.5},
        {"text": "sentence.", "start_s": 3.5, "end_s": 5.0},
    ]
    config = DatasetPrepConfig(name="test", inputs=[], pad_ms=60)
    padded = apply_padding_and_limits([first, second], 60, 6000)
    result = _snap_segments(padded, energy, config, protect_words=True)
    assert result[0].end_ms == result[1].start_ms == 3165
    assert type(result[0].end_ms) is int
    assert " ".join(s.text for s in result) == " ".join(s.text for s in padded)


def test_shared_boundary_does_not_search_past_the_next_word():
    energy = np.ones(600, dtype=np.float32)
    energy[320:327] = 0.001
    first, second = sentence(1000, 3000), sentence(3000, 5000)
    second.word_timestamps = [
        {"text": "Next", "start_s": 3.0, "end_s": 3.1},
        {"text": "sentence.", "start_s": 3.1, "end_s": 5.0},
    ]
    config = DatasetPrepConfig(name="test", inputs=[], pad_ms=60)
    padded = apply_padding_and_limits([first, second], 60, 6000)
    result = _snap_segments(padded, energy, config, protect_words=True)
    assert result[0].end_ms == result[1].start_ms == 3000
