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
