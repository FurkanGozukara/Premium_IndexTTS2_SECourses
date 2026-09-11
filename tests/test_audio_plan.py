from __future__ import annotations

import numpy as np
import torch

from indextts.infer_v2_5 import PLAN_SILENCE_KINDS, _edge_quiet_samples, _stream_silence_samples, assemble_audio_plan
from indextts.utils.pause_cap import shorten_long_pauses

RATE = 22050


def _segment(seconds: float, lead: float = 0.1, tail: float = 0.25) -> torch.Tensor:
    count = int(seconds * RATE)
    t = torch.arange(count) / RATE
    wav = (0.4 * torch.sin(2 * torch.pi * 150 * t) * 32767).to(torch.int16)
    wav[: int(lead * RATE)] = 0
    wav[count - int(tail * RATE):] = 0
    return wav.unsqueeze(0)


def test_edges_are_measured_in_frames() -> None:
    leading, trailing = _edge_quiet_samples(_segment(1.0), RATE)
    assert abs(leading - int(0.1 * RATE)) <= RATE * 0.011
    assert abs(trailing - int(0.25 * RATE)) <= RATE * 0.011
    assert _edge_quiet_samples(torch.zeros(1, 0), RATE) == (0, 0)
    silent = _edge_quiet_samples(torch.zeros(1, RATE, dtype=torch.int16), RATE)
    assert silent == (RATE, 0)


def test_sentence_gap_is_measured_from_word_to_word_and_pause_tags_are_reported() -> None:
    first, second = _segment(1.0), _segment(0.8)
    plan = [("segment", 0), ("sentence_gap", int(0.5 * RATE)), ("segment", 1), ("pause", 4410), ("segment", 0)]
    wav, protected = assemble_audio_plan([first, second], plan, RATE)
    # The 0.25 s tail and 0.1 s head already give 0.35 s, so 0.15 s of silence is inserted.
    expected = first.shape[1] + int(0.15 * RATE) + second.shape[1] + 4410 + first.shape[1]
    assert abs(wav.shape[1] - expected) <= RATE * 0.011
    assert len(protected) == 1
    start, end = protected[0]
    assert end - start == 4410 and abs(start - (first.shape[1] + int(0.15 * RATE) + second.shape[1])) <= RATE * 0.011
    # A gap shorter than the generated edges trims the previous tail instead of inserting silence.
    trimmed, _ = assemble_audio_plan([first, second], [("segment", 0), ("sentence_gap", int(0.2 * RATE)), ("segment", 1)], RATE)
    assert abs(trimmed.shape[1] - (first.shape[1] + second.shape[1] - int(0.15 * RATE))) <= RATE * 0.011
    # Plain section silence and the old two-item plan behave exactly as before.
    plain, none_protected = assemble_audio_plan([first, second], [("segment", 0), ("silence", 4410), ("segment", 1)], RATE)
    assert plain.shape[1] == first.shape[1] + 4410 + second.shape[1] and none_protected == []
    empty, _ = assemble_audio_plan([], [], RATE)
    assert empty.shape == (1, 0)
    assert PLAN_SILENCE_KINDS == {"silence", "pause", "sentence_gap"}


def test_streaming_gap_only_knows_the_previous_tail() -> None:
    first = _segment(1.0)
    plan = [("segment", 0), ("sentence_gap", int(0.5 * RATE)), ("segment", 1), ("silence", 100)]
    gap = _stream_silence_samples(plan, 1, [first, None], RATE)
    assert abs(gap - (int(0.5 * RATE) - int(0.25 * RATE))) <= RATE * 0.011
    assert _stream_silence_samples(plan, 3, [first, None], RATE) == 100
    assert _stream_silence_samples([("sentence_gap", 500)], 0, [], RATE) == 500


def test_pause_cap_leaves_protected_ranges_alone() -> None:
    t = np.arange(int(0.6 * 16000)) / 16000
    speech = (0.4 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    silence = np.zeros(int(0.8 * 16000), dtype=np.float32)
    audio = np.concatenate([speech, silence, speech, silence, speech])
    result, report = shorten_long_pauses(audio, 16000, 300)
    assert report["shortened"] == 2 and report["protected"] == 0
    # The first pause is an explicit tag: it keeps its full length.
    first_pause = (0.6, 1.4)
    result, report = shorten_long_pauses(audio, 16000, 300, protected_s=[first_pause])
    assert report["shortened"] == 1 and report["protected"] == 1
    assert abs((len(audio) - len(result)) / 16000 - 0.5) < 0.03
