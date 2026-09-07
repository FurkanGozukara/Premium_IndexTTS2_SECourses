"""Median-matched reference selection: pitch, pace, and the typical clip among the best candidates."""
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from indextts.training.evaluation_plan import choose_training_reference, describe_training_reference
from indextts.training.voice_profile import (
    PitchCache,
    choose_typical_reference,
    clip_pace,
    measure_clip_pitch,
    speaker_profile,
    typicality_distance,
)


def _tone(path: Path, hz: float, seconds: float = 2.0, sr: int = 16000) -> None:
    t = np.arange(int(seconds * sr)) / sr
    # A harmonic-rich tone with slow vibrato so pyin sees voiced frames.
    phase = 2 * np.pi * hz * t + 0.5 * np.sin(2 * np.pi * 5 * t)
    wave = 0.4 * np.sin(phase) + 0.2 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wave.astype(np.float32), sr)


def _records(dataset: Path, spec):
    rows = []
    for index, (hz, words, seconds) in enumerate(spec):
        row_id = f"clip{index:02d}"
        _tone(dataset / "segments" / f"{row_id}.wav", hz, seconds)
        rows.append({"id": row_id, "audio": f"segments/{row_id}.wav", "duration_s": str(seconds), "words": words,
                     "text": " ".join(["word"] * words), "speaker": "A", "asr_wer": 0.0, "boundary_words_match": True})
    return rows


def test_pace_and_distance():
    assert clip_pace({"duration_s": "10", "words": 25}) == 2.5
    assert clip_pace({"duration_s": "4", "text": "one two three four"}) == 1.0
    assert clip_pace({"duration_s": "0", "words": 3}) is None and clip_pace({"duration_s": "x"}) is None
    profile = {"pitch_median_hz": 140.0, "pace_median_wps": 2.5}
    assert typicality_distance(140.0, 2.5, profile) == 0.0
    assert abs(typicality_distance(140.0 * 1.1052, 2.5, profile) - 1.0) < 0.01  # ten percent of pitch is one unit
    assert abs(typicality_distance(140.0, 2.5 * 1.15, profile) - 1.0) < 1e-6  # fifteen percent of pace is one unit
    assert typicality_distance(None, None, profile) == 0.0 and typicality_distance(120.0, 2.0, {}) == 0.0
    assert typicality_distance(140.0, 2.5, profile, duration=10.0) == 1.0  # five seconds off the target is one unit
    assert typicality_distance(140.0, 2.5, profile, duration=None) == 0.0 and typicality_distance(140.0, 2.5, profile, duration=0.0) == 3.0


def test_measured_pitch_and_cache(tmp_path):
    clip = tmp_path / "a.wav"
    _tone(clip, 150.0)
    pitch = measure_clip_pitch(clip)
    assert pitch is not None and 135 < pitch < 165
    dataset = tmp_path / "ds"
    rows = _records(dataset, [(150.0, 5, 2.0)])
    cache = PitchCache(dataset)
    first = cache.pitch_of(rows[0])
    cache.save()
    stored = json.loads((dataset / "analysis" / "pitch_cache.json").read_text(encoding="utf-8"))
    assert stored["clip00"]["f0_median_hz"] == first
    assert PitchCache(dataset).pitch_of(rows[0]) == first  # served from the cache
    assert PitchCache(dataset).pitch_of({"id": "missing", "audio": "segments/none.wav"}) is None


def test_typical_reference_prefers_the_median_voice_among_clean_candidates(tmp_path):
    dataset = tmp_path / "ds"
    # Median pitch ~150 Hz and pace 2.5 words/s; clip00 is the low, slow outlier that duration alone would pick.
    spec = [(105.0, 6, 15.0)] + [(150.0, 31, 12.4)] * 5 + [(148.0, 30, 12.1), (200.0, 32, 12.8), (152.0, 33, 13.0)]
    rows = _records(dataset, [(hz, words, 2.0) for hz, words, _ in spec])
    for row, (_, _, seconds) in zip(rows, spec):
        row["duration_s"] = str(seconds)  # clip00 sits nearest the 15-second target on paper but is low and slow
    profile = speaker_profile(dataset, rows)
    assert 140 < profile["pitch_median_hz"] < 160 and abs(profile["pace_median_wps"] - 2.5) < 0.2
    choice = choose_typical_reference(dataset, rows, profile=profile)
    assert choice["record"]["id"] != "clip00" and choice["distance"] < 1.5
    assert choice["candidates"][0]["id"] == "clip00" and choice["candidates"][0]["distance"] > 3
    # The plan-level helper follows the flag; the legacy rule still returns the nearest duration.
    assert choose_training_reference(rows, dataset, typical=False)["id"] == "clip00"
    assert choose_training_reference(rows, dataset)["id"] == choice["record"]["id"]
    assert choice["record"]["id"] in describe_training_reference(rows, dataset)
    assert choose_typical_reference(dataset, []) is None
