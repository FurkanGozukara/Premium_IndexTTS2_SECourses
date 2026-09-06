from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from indextts.training.dataset import LoraTrainDataset
from indextts.training.features import CACHE_FORMAT, CACHE_VERSION, _cache_valid, _source_fingerprint, _FeatureModels
from indextts.training.model_forward import TokenMetrics
from indextts.training.plan import validation_record_ids
from indextts.training.early_stopping import EarlyStopping
from indextts.training.dataset_quality import TimedTranscript, word_error_counts


def test_source_split_is_order_independent_and_never_splits_a_recording() -> None:
    rows = [{"id": f"{group}-{i}", "source_media": group} for group in "abcde" for i in range(7)]
    val = validation_record_ids(rows, 0.2, 42, "source")
    assert val == validation_record_ids(list(reversed(rows)), 0.2, 42, "source")
    assert 0 < len(val) < len(rows)
    for group in "abcde":
        ids = {row["id"] for row in rows if row["source_media"] == group}
        assert not (ids & val) or ids <= val


def test_explicit_holdout_takes_precedence_and_rejects_partial_labels() -> None:
    rows = [{"id": "a", "split": "train"}, {"id": "b", "split": "val"}]
    assert validation_record_ids(rows, 0, 42, "source") == {"b"}
    with pytest.raises(ValueError, match="every manifest row"):
        validation_record_ids(rows + [{"id": "c"}], 0.05, 42)


def test_validation_features_never_condition_training(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    rows = []
    for i in range(6):
        row = {"id": str(i), "speaker": "voice", "n_codes": 3, "n_text_tokens": 2,
               "split": "train" if i < 3 else "val"}
        rows.append(row)
        torch.save({"codes": torch.tensor([1, 2, 3]), "text_tokens": torch.tensor([4, 5]),
                    "campplus": torch.full((192,), float(i)), "emo_raw": torch.zeros(1024),
                    "emo_vec": torch.full((1280,), float(i)), "language": "EN"}, cache / f"{i}.pt")
    (tmp_path / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    for split in ("train", "val"):
        dataset = LoraTrainDataset(tmp_path, split=split, speaker_ref_mode="other", emo_ref_mode="follow_speaker")
        for epoch in range(4):
            dataset.set_epoch(epoch)
            for item in dataset:
                assert item["reference_id"] in {"0", "1", "2"}
                assert item["reference_id"] != item["id"]
                assert item["emo_reference_id"] == item["reference_id"]


@pytest.mark.parametrize("speaker_mode,emotion_mode", [
    ("other", "follow_speaker"), ("self", "other"), ("mixed", "self"), ("self", "mixed"),
])
def test_validation_cannot_silently_use_target_when_training_speaker_is_missing(
    tmp_path: Path, speaker_mode: str, emotion_mode: str,
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    rows = [{"id": "training", "speaker": "known", "split": "train", "n_codes": 3, "n_text_tokens": 2},
            {"id": "holdout", "speaker": "unseen", "split": "val", "n_codes": 3, "n_text_tokens": 2}]
    for row in rows:
        torch.save({"codes": torch.tensor([1, 2, 3]), "text_tokens": torch.tensor([4, 5]),
                    "campplus": torch.ones(192), "emo_raw": torch.ones(1024),
                    "emo_vec": torch.ones(1280), "language": "EN"}, cache / f"{row['id']}.pt")
    (tmp_path / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="no training reference for: unseen"):
        LoraTrainDataset(tmp_path, split="val", speaker_ref_mode=speaker_mode, emo_ref_mode=emotion_mode)
    explicit_self = LoraTrainDataset(tmp_path, split="val", speaker_ref_mode="self", emo_ref_mode="self")
    assert explicit_self[0]["reference_id"] == explicit_self[0]["emo_reference_id"] == "holdout"


def test_token_weighted_validation_is_invariant_to_batch_partition() -> None:
    split = TokenMetrics()
    split.update(dict(mel_loss=2., text_loss=4., mel_accuracy=0.5, mel_tokens=2, text_tokens=3))
    split.update(dict(mel_loss=5., text_loss=1., mel_accuracy=0.25, mel_tokens=6, text_tokens=1))
    combined = TokenMetrics()
    combined.update(dict(mel_loss=4.25, text_loss=3.25, mel_accuracy=0.3125, mel_tokens=8, text_tokens=4))
    assert split.result() == pytest.approx(combined.result())
    assert split.result()["loss"] == pytest.approx(4.575)


def test_resume_fingerprint_tracks_validation_and_cached_audio(tmp_path: Path) -> None:
    dataset = object.__new__(LoraTrainDataset)
    dataset.dataset_dir = tmp_path
    dataset.split, dataset.val_split_mode = "train", "source"
    dataset.val_fraction, dataset.seed = 0.05, 42
    dataset.speaker_ref_mode, dataset.emo_ref_mode = "other", "follow_speaker"
    dataset._all_records = [{"id": "train", "split": "train", "text": "Training text"},
                            {"id": "validation", "split": "val", "text": "Validation text"}]
    original = dataset.fingerprint
    dataset._all_records[1]["text"] = "Corrected validation text"
    corrected = dataset.fingerprint
    assert corrected != original
    (tmp_path / "cache_index.json").write_text(json.dumps({"records": [{"id": "train", "source_fingerprint": "changed_audio"}]}), encoding="utf-8")
    assert dataset.fingerprint != corrected


def test_cache_rejects_old_precision_stale_inputs_and_nonfinite_features(tmp_path: Path) -> None:
    path = tmp_path / "cache.pt"
    payload = {"format": CACHE_FORMAT, "version": CACHE_VERSION, "semantic_layer": 17,
               "source_fingerprint": "audio-v2", "extraction_fingerprint": "fp32",
               "codes": torch.tensor([1, 2]), "text_tokens": torch.tensor([3, 4]),
               "campplus": torch.zeros(192), "emo_raw": torch.zeros(1024), "emo_vec": torch.zeros(1280)}
    torch.save(payload, path)
    assert _cache_valid(path, 17, source_fingerprint="audio-v2", extraction_fingerprint="fp32")
    assert not _cache_valid(path, 17, source_fingerprint="audio-v1")
    assert not _cache_valid(path, 17, extraction_fingerprint="bf16")
    payload["version"] = 1
    torch.save(payload, path)
    assert not _cache_valid(path, 17)
    payload["version"] = CACHE_VERSION
    payload["campplus"][0] = float("nan")
    torch.save(payload, path)
    assert not _cache_valid(path, 17)


def test_source_hash_tracks_audio_and_caption_changes(tmp_path: Path) -> None:
    (tmp_path / "clip.wav").write_bytes(b"first audio")
    row = {"audio": "clip.wav", "text": "First caption", "language": "EN", "speaker": "voice"}
    original = _source_fingerprint(tmp_path, row)
    assert original != _source_fingerprint(tmp_path, {**row, "text": "Corrected caption"})
    (tmp_path / "clip.wav").write_bytes(b"other audio")
    assert original != _source_fingerprint(tmp_path, row)


def test_encoder_remains_fp32_under_outer_autocast() -> None:
    seen = []

    class Encoder:
        def __call__(self, input_features, **kwargs):
            seen.append((input_features.dtype, torch.is_autocast_enabled("cpu")))
            return SimpleNamespace(hidden_states=[input_features])

        def _get_feature_vector_attention_mask(self, width, mask):
            return mask

    features = object.__new__(_FeatureModels)
    features.device = torch.device("cpu")
    features.compute_dtype = torch.bfloat16
    features.config = SimpleNamespace(semantic_layer=17)
    features.processor = lambda *args, **kwargs: {"input_features": torch.ones(1, 3, 2), "attention_mask": torch.ones(1, 3)}
    features.semantic_model = Encoder()
    features.semantic_mean, features.semantic_std = torch.tensor(0.), torch.tensor(1.)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        values = features.w2v_features([torch.ones(1, 16000)])
    assert seen == [(torch.float32, False)]
    assert values[0].dtype == torch.float32


def test_early_stop_grace_noise_resume_and_disabled_control() -> None:
    options = dict(enabled=True, patience=2, min_delta=0.01, min_steps=100, min_epochs=2)
    tracker = EarlyStopping()
    assert tracker.observe(5., step=10, epoch=0.2, **options) == (True, False)
    assert tracker.observe(5.2, step=20, epoch=0.4, **options) == (False, False)
    assert tracker.bad_checks == 0
    assert tracker.observe(4.995, step=100, epoch=2, **options) == (True, False)
    # The absolute-best checkpoint advances, but insignificant gains consume patience.
    assert tracker.bad_checks == 1
    assert tracker.observe(4.995, step=100, epoch=2, **options) == (False, False)
    resumed = EarlyStopping.from_state(tracker.to_dict())
    assert resumed.observe(4.993, step=150, epoch=3, **options) == (True, True)
    assert resumed.best_step == 150
    assert "early stopping" in resumed.reason
    assert resumed.observe(5.2, step=200, epoch=4, **{**options, "enabled": False}) == (False, False)
    assert resumed.bad_checks == 0
    with pytest.raises(FloatingPointError):
        tracker.observe(float("nan"), step=200, epoch=4, **options)


def test_transcript_audit_catches_unlabelled_speech_and_handles_written_numbers() -> None:
    assert word_error_counts("Version 24 works.", "version twenty four works") == (0, 4)
    assert word_error_counts("Hello world.", "Hello extra unlabelled words world.") == (3, 2)
    words = [{"text": "outside", "start_s": 0., "end_s": 1.},
             {"text": "inside", "start_s": 1., "end_s": 2.}]
    assert TimedTranscript(words).between(1., 2.) == "inside"
