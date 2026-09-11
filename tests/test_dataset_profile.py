from __future__ import annotations

import json
from pathlib import Path

from indextts.training.dataset_manifest import write_manifest
from indextts.training.dataset_profile import (
    PREFIX_TOKENS,
    build_dataset_profile,
    dataset_dir_for_adapter,
    ensure_dataset_profile,
    load_dataset_profile,
    load_dataset_vocabulary,
    recommend_line_rules,
    recommended_max_tokens,
    sentence_word_counts,
    text_for_tokens,
    word_count,
    words_for_max_tokens,
    write_dataset_profile,
)
from indextts.utils.text_segmentation import segment_token_budget


def _token_len(text: str) -> int:
    # Every word costs one token plus one for each punctuation mark, like a BPE over clean English.
    return len(text.split()) + sum(text.count(mark) for mark in ".,!?")


def _rows(count: int = 40) -> list[dict]:
    rows = []
    for index in range(count):
        words = 20 + index  # 20 .. 59 words
        seconds = words / 2.5
        first = " ".join(["alpha"] * (words // 2)) + "."
        second = " ".join(["beta"] * (words - words // 2)) + "."
        rows.append(
            {
                "id": f"clip_{index:03d}",
                "audio": f"segments/clip_{index:03d}.wav",
                "text": f"{first} {second}",
                "duration_s": round(seconds, 3),
                "words": words,
                "language": "EN",
                "split": "val" if index % 10 == 0 else "train",
                "length_aim": "short" if words < 24 else "target",
            }
        )
    return rows


def test_profile_measures_training_split_and_recommends_line_rules(tmp_path: Path) -> None:
    dataset = tmp_path / "voice_dataset"
    dataset.mkdir()
    write_manifest(dataset / "manifest.jsonl", _rows())

    profile = build_dataset_profile(dataset, token_len=_token_len)

    assert profile is not None
    assert profile["split"] == "train" and profile["clips"] == 36
    assert profile["language"] == "EN"
    assert profile["words_per_second"] == 2.5
    assert profile["tokens_per_word"] > 1.0
    assert profile["length_aims"] == {"short": 3, "target": 33}
    assert profile["vocabulary_size"] == 2 and sorted(profile["_vocabulary"]) == ["alpha", "beta"]
    words = profile["words"]
    assert words["min"] == 21.0 and words["max"] == 59.0
    assert words["p10"] <= words["p50"] <= words["p90"]
    rules = profile["recommendation"]
    assert rules["target_words"][0] <= rules["target_words"][1]
    assert rules["hard_min_words"] <= rules["acceptable_words"][0] <= rules["target_words"][0]
    assert rules["target_words"][1] <= rules["acceptable_words"][1] <= rules["hard_max_words"] <= rules["never_exceed_words"]
    assert rules["sentence_min_alone_words"] == rules["hard_min_words"]
    assert rules["budget_tokens"] >= rules["budget_words"]


def test_recommended_max_tokens_inverts_the_engine_budget() -> None:
    profile = {
        "language": "EN",
        "tokens_per_word": 1.25,
        "words": {"count": 10, "p05": 15, "p10": 20, "p40": 30, "p50": 32, "p60": 34, "p90": 44, "p95": 48, "max": 60},
        "words_per_sentence": {"count": 30, "p50": 10},
        "duration_s": {"p50": 12.0},
    }
    rules = recommend_line_rules(profile)
    assert rules["budget_words"] == 37.0 and rules["budget_tokens"] == 47
    max_tokens = recommended_max_tokens(profile, language="EN", budget_scale=0.72)
    assert max_tokens is not None and max_tokens >= 20
    budget = segment_token_budget(max_tokens, 602, "<|en|> ", lambda text: PREFIX_TOKENS, 0.72)
    # The usable budget the engine derives from the recommended value covers the line budget.
    assert budget >= rules["budget_tokens"]
    assert budget - rules["budget_tokens"] <= 2
    assert abs(words_for_max_tokens(profile, max_tokens, language="EN", budget_scale=0.72) - 37) <= 1.0
    # CJK languages skip the non-CJK scale, so the same budget needs fewer tokens.
    assert recommended_max_tokens(profile, language="ZH", budget_scale=0.72) < max_tokens
    assert recommended_max_tokens({"words": {"count": 0}}) is None


def test_profile_round_trips_through_the_adapter_folder(tmp_path: Path) -> None:
    dataset = tmp_path / "datasets" / "voice_dataset"
    dataset.mkdir(parents=True)
    write_manifest(dataset / "manifest.jsonl", _rows(12))
    adapter = tmp_path / "loras" / "voice"
    adapter.mkdir(parents=True)
    checkpoint = adapter / "voice.safetensors"
    checkpoint.write_bytes(b"x")
    (adapter / "train_config.json").write_text(json.dumps({"dataset_dir": str(dataset)}), encoding="utf-8")

    assert load_dataset_profile(checkpoint) is None
    assert dataset_dir_for_adapter(checkpoint) == dataset.resolve()
    profile = ensure_dataset_profile(checkpoint, token_len=_token_len)
    assert profile is not None
    saved = load_dataset_profile(adapter / "best" / "voice_best.safetensors")
    assert saved is not None and saved["clips"] == profile["clips"]
    assert "_vocabulary" not in saved
    assert load_dataset_vocabulary(checkpoint) == frozenset({"alpha", "beta"})
    # A second call reads the saved file instead of measuring again.
    assert ensure_dataset_profile(checkpoint, token_len=None)["generated_at"] == saved["generated_at"]


def test_profile_without_dataset_or_rows_is_none(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert build_dataset_profile(empty) is None
    write_manifest(empty / "manifest.jsonl", [{"id": "a", "audio": "a.wav", "text": "", "duration_s": 0.0, "words": 0}])
    assert build_dataset_profile(empty) is None
    orphan = tmp_path / "orphan.safetensors"
    orphan.write_bytes(b"x")
    assert ensure_dataset_profile(orphan) is None
    assert write_dataset_profile(tmp_path / "adapter", {"clips": 1, "duration_s": {"p50": 1.0}, "_vocabulary": ["a"]}).is_file()


def test_text_helpers_follow_the_engine() -> None:
    assert word_count("Hello there, [pause:500ms] world.") == 3
    assert sentence_word_counts("First one here. Second! Third one? ") == [3, 1, 2]
    assert text_for_tokens("Hello World [pause:1s]", "EN") == "hello world"
    assert text_for_tokens("Hola Mundo", "ES") == "Hola Mundo"
