from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from indextts.training import pause_profile
from indextts.training.dataset_manifest import write_manifest
from indextts.training.dataset_profile import build_dataset_profile, recommended_pauses, smart_target_tokens
from indextts.training.pause_profile import (
    PauseCache,
    build_pause_profile,
    describe_pause_profile,
    measure_clip_pauses,
    recommend_pauses,
    sentence_count,
)

RATE = 16000


def _speech(seconds: float, hz: float = 140.0) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (0.4 * np.sin(2 * np.pi * hz * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4.0 * t))).astype(np.float32)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * RATE), dtype=np.float32)


def _clip(path: Path, pauses_s: list[float]) -> None:
    parts = [_silence(0.2), _speech(0.8)]
    for pause in pauses_s:
        parts.extend([_silence(pause), _speech(0.8)])
    parts.append(_silence(0.2))
    sf.write(path, np.concatenate(parts), RATE, subtype="PCM_16")


def _dataset(tmp_path: Path) -> tuple[Path, list[dict]]:
    root = tmp_path / "dataset"
    (root / "segments").mkdir(parents=True)
    rows = []
    plans = {
        # id: (pauses inside the clip, transcript with the matching number of sentences)
        "a": ([0.5, 0.15, 0.4], "First sentence here. Second one, with a comma. Third sentence!"),
        "b": ([0.6, 0.2], "One sentence, one comma. Second sentence."),
        "c": ([0.15, 0.2], "Only one sentence, with two commas, in it."),
    }
    for clip_id, (pauses, text) in plans.items():
        path = root / "segments" / f"{clip_id}.wav"
        _clip(path, pauses)
        rows.append({"id": clip_id, "audio": f"segments/{clip_id}.wav", "text": text, "split": "train",
                     "duration_s": round(sf.info(str(path)).duration, 3), "words": len(text.split()), "language": "EN"})
    write_manifest(root / "manifest.jsonl", rows)
    return root, rows


def test_sentence_count_and_clip_measurement(tmp_path: Path) -> None:
    assert sentence_count("One. Two! Three?") == 3 and sentence_count("no end") == 1 and sentence_count("") == 1
    path = tmp_path / "clip.wav"
    _clip(path, [0.5, 0.15])
    metrics = measure_clip_pauses(path)
    assert metrics is not None
    assert len(metrics["pauses_ms"]) == 2 and metrics["pauses_ms"][0] >= metrics["pauses_ms"][1]
    assert 470 <= metrics["pauses_ms"][0] <= 530 and 130 <= metrics["pauses_ms"][1] <= 180
    assert measure_clip_pauses(tmp_path / "missing.wav") is None


def test_pause_profile_splits_sentence_and_within_pauses_and_recommends(tmp_path: Path, monkeypatch) -> None:
    root, rows = _dataset(tmp_path)
    profile = build_pause_profile(root, rows)
    assert profile is not None
    assert profile["clips"] == 3 and profile["clips_with_multiple_sentences"] == 2
    # a: 3 sentences -> its two longest pauses (500, 400) are sentence pauses; b: one (600); c: none.
    assert profile["sentence_pauses_ms"]["count"] == 3
    assert profile["within_sentence_pauses_ms"]["count"] == 4
    assert 380 <= profile["sentence_pauses_ms"]["min"] <= 420 and 580 <= profile["sentence_pauses_ms"]["max"] <= 620
    # Too few sentence pauses for the boundary rule: the longer pauses of the whole set stand in.
    recommendation = profile["recommendation"]
    assert recommendation["sentence_pause_source"] == "the longest pauses inside sentences"
    assert 200 <= recommendation["max_pause_ms"] <= 2000 and recommendation["sentence_pause_ms"] <= recommendation["max_pause_ms"]
    assert "3 clips measured" in describe_pause_profile(profile)
    # With enough sentence pauses the medians of the sentence pauses are used.
    monkeypatch.setattr(pause_profile, "MIN_SENTENCE_PAUSES", 2)
    rules = recommend_pauses(profile)
    assert rules["sentence_pause_source"] == "sentence boundaries"
    assert rules["sentence_pause_ms"] == round(profile["sentence_pauses_ms"]["p50"] / 10) * 10
    assert rules["max_pause_ms"] == max(200, round(profile["sentence_pauses_ms"]["p90"] / 10) * 10)
    # The cache stores every clip and is reused without re-reading the audio.
    cache_path = root / "analysis" / "pause_cache.json"
    assert cache_path.is_file() and len(json.loads(cache_path.read_text(encoding="utf-8"))) == 3
    monkeypatch.setattr(pause_profile, "measure_clip_pauses", lambda _path: (_ for _ in ()).throw(AssertionError("re-measured")))
    again = build_pause_profile(root, rows, cache=PauseCache(root))
    assert again is not None and again["all_pauses_ms"] == profile["all_pauses_ms"]


def test_dataset_profile_carries_pauses_and_the_smart_target(tmp_path: Path, monkeypatch) -> None:
    root, _rows = _dataset(tmp_path)
    monkeypatch.setattr(pause_profile, "MIN_SENTENCE_PAUSES", 2)
    profile = build_dataset_profile(root, token_len=lambda text: len(text.split()) + 2)
    assert profile is not None and profile["version"] == 2
    assert profile["pauses"]["clips"] == 3
    rules = profile["recommendation"]
    assert rules["sentence_pause_ms"] == profile["pauses"]["recommendation"]["sentence_pause_ms"]
    assert rules["max_pause_ms"] == profile["pauses"]["recommendation"]["max_pause_ms"]
    assert rules["pause_source"] == "sentence boundaries"
    assert recommended_pauses(profile) == (rules["sentence_pause_ms"], rules["max_pause_ms"])
    # The smart target is the median clip in text tokens.
    assert smart_target_tokens(profile) == int(np.ceil(profile["text_tokens"]["p50"]))
    assert recommended_pauses(None) is None and smart_target_tokens(None) is None
    without = build_dataset_profile(root, measure_pauses=False)
    assert without is not None and "pauses" not in without and "sentence_pause_ms" not in without["recommendation"]
