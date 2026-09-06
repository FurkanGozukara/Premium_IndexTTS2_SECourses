from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from indextts.training.dataset import LoraTrainDataset, collate


def _cached_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    cache_dir = root / "cache"
    cache_dir.mkdir(parents=True)
    rows = []
    for index, (sample_id, speaker) in enumerate(
        (("a0", "speaker-a"), ("a1", "speaker-a"), ("a2", "speaker-a"), ("b0", "speaker-b"))
    ):
        value = float(index + 1)
        torch.save(
            {
                "text_tokens": torch.tensor([2, 3], dtype=torch.int32),
                "codes": torch.tensor([4, 5, 6], dtype=torch.int16),
                "campplus": torch.full((3,), value),
                "emo_raw": torch.full((4,), value + 10),
                "emo_vec": torch.full((5,), value + 20),
                "speaker": speaker,
                "language": "EN",
                "lang_id": 0,
            },
            cache_dir / f"{sample_id}.pt",
        )
        rows.append(
            {
                "id": sample_id,
                "speaker": speaker,
                "language": "EN",
                "n_codes": 3,
                "n_text_tokens": 2,
            }
        )
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return root


def _index(dataset: LoraTrainDataset, sample_id: str) -> int:
    return next(
        index for index, record in enumerate(dataset.records) if record["id"] == sample_id
    )


def _reference_metadata(root: Path, durations: dict[str, float | None], **updates) -> None:
    manifest = root / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["duration_s"] = durations.get(row["id"])
        row.update(updates.get(row["id"], {}))
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.mark.parametrize("emotion_mode", ["other", "follow_speaker"])
def test_other_references_target_fifteen_seconds_and_exclude_self(tmp_path, emotion_mode):
    root = _cached_dataset(tmp_path)
    _reference_metadata(root, {"a0": 12, "a1": 15, "a2": 20, "b0": 15})
    dataset = LoraTrainDataset(root, val_fraction=0, speaker_ref_mode="other", emo_ref_mode=emotion_mode)
    for epoch in (0, 1, 7):
        dataset.set_epoch(epoch)
        for target in ("a0", "a2"):
            item = dataset[_index(dataset, target)]
            assert item["reference_id"] == item["emo_reference_id"] == "a1"
        # The only exact 15-second clip cannot condition itself in other mode.
        item = dataset[_index(dataset, "a1")]
        assert item["reference_id"] == item["emo_reference_id"] == "a0"


def test_nearest_reference_ties_can_vary_deterministically_between_epochs(tmp_path):
    root = _cached_dataset(tmp_path)
    _reference_metadata(root, {"a0": 14, "a1": 16, "a2": 20, "b0": 15})
    dataset = LoraTrainDataset(root, val_fraction=0, seed=19, speaker_ref_mode="other", emo_ref_mode="other")
    observed = set()
    index = _index(dataset, "a2")
    for epoch in range(20):
        dataset.set_epoch(epoch)
        first = dataset[index]
        assert dataset[index]["reference_id"] == first["reference_id"]
        assert first["reference_id"] in {"a0", "a1"}
        assert first["emo_reference_id"] in {"a0", "a1"}
        observed.add(first["reference_id"])
    assert observed == {"a0", "a1"}


def test_nearest_validation_reference_comes_only_from_same_speaker_training_split(tmp_path):
    root = _cached_dataset(tmp_path)
    _reference_metadata(root, {"a0": 15, "a1": 12, "a2": 16, "b0": 15},
                        a0={"split": "val"}, a1={"split": "train"},
                        a2={"split": "train"}, b0={"split": "train"})
    dataset = LoraTrainDataset(root, split="val", speaker_ref_mode="other", emo_ref_mode="follow_speaker")
    item = dataset[_index(dataset, "a0")]
    assert item["reference_id"] == item["emo_reference_id"] == "a2"


def test_reference_duration_selection_preserves_quality_and_known_duration_priority(tmp_path):
    root = _cached_dataset(tmp_path)
    _reference_metadata(root, {"a0": 20, "a1": 15, "a2": 14, "b0": 15}, a1={"asr_wer": .1})
    dataset = LoraTrainDataset(root, val_fraction=0, speaker_ref_mode="other", emo_ref_mode="other")
    assert dataset[_index(dataset, "a0")]["reference_id"] == "a2"
    _reference_metadata(root, {"a0": 20, "a1": 15, "a2": None, "b0": 15}, a1={"asr_wer": 0})
    recreated = LoraTrainDataset(root, val_fraction=0, speaker_ref_mode="other", emo_ref_mode="other")
    assert recreated[_index(recreated, "a0")]["reference_id"] == "a1"
    assert recreated.fingerprint != dataset.fingerprint


def test_other_emotion_reference_is_different_and_deterministic(tmp_path: Path) -> None:
    root = _cached_dataset(tmp_path)
    dataset = LoraTrainDataset(
        root,
        val_fraction=0,
        seed=19,
        speaker_ref_mode="self",
        emo_ref_mode="other",
    )
    index = _index(dataset, "a0")
    dataset.set_epoch(7)

    first = dataset[index]
    second = dataset[index]
    recreated = LoraTrainDataset(
        root,
        val_fraction=0,
        seed=19,
        speaker_ref_mode="self",
        emo_ref_mode="other",
    )
    recreated.set_epoch(7)

    assert first["emo_reference_id"] != first["id"]
    assert second["emo_reference_id"] == first["emo_reference_id"]
    assert recreated[index]["emo_reference_id"] == first["emo_reference_id"]


@pytest.mark.parametrize("speaker_ref_mode", ["self", "other"])
def test_follow_speaker_uses_the_exact_speaker_reference(
    tmp_path: Path, speaker_ref_mode: str
) -> None:
    dataset = LoraTrainDataset(
        _cached_dataset(tmp_path),
        val_fraction=0,
        seed=3,
        speaker_ref_mode=speaker_ref_mode,
        emo_ref_mode="follow_speaker",
    )
    item = dataset[_index(dataset, "a0")]

    assert item["emo_reference_id"] == item["reference_id"]
    if speaker_ref_mode == "self":
        assert item["reference_id"] == item["id"]
    else:
        assert item["reference_id"] != item["id"]


def test_mixed_emotion_references_cover_self_and_other(tmp_path: Path) -> None:
    dataset = LoraTrainDataset(
        _cached_dataset(tmp_path),
        val_fraction=0,
        seed=41,
        speaker_ref_mode="self",
        emo_ref_mode="mixed",
    )
    outcomes: set[bool] = set()
    speaker_a_indices = [
        index
        for index, record in enumerate(dataset.records)
        if record["speaker"] == "speaker-a"
    ]
    for epoch in range(20):
        dataset.set_epoch(epoch)
        for index in speaker_a_indices:
            item = dataset[index]
            outcomes.add(item["emo_reference_id"] == item["id"])

    assert outcomes == {False, True}


def test_single_record_speaker_falls_back_to_self_and_collate_keeps_ids(
    tmp_path: Path,
) -> None:
    dataset = LoraTrainDataset(
        _cached_dataset(tmp_path),
        val_fraction=0,
        speaker_ref_mode="other",
        emo_ref_mode="other",
    )
    single = dataset[_index(dataset, "b0")]
    paired = dataset[_index(dataset, "a0")]
    batch = collate([single, paired])

    assert single["reference_id"] == single["id"]
    assert single["emo_reference_id"] == single["id"]
    assert batch["emo_reference_ids"] == [
        single["emo_reference_id"],
        paired["emo_reference_id"],
    ]


def test_invalid_emotion_reference_mode_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="emo_ref_mode"):
        LoraTrainDataset(_cached_dataset(tmp_path), val_fraction=0, emo_ref_mode="bad")
