from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from indextts.training.dataset_manifest import load_manifest
from indextts.training.features import FeatureCacheConfig, cache_dataset_features


@pytest.mark.gpu
def test_cache_three_real_segments() -> None:
    dataset_dir = Path(
        os.environ.get("INDEXTTS_TEST_TRAINING_DATASET", "datasets/secourses_demo")
    )
    if not (dataset_dir / "manifest.jsonl").is_file():
        pytest.skip(
            "Training dataset is unavailable; set INDEXTTS_TEST_TRAINING_DATASET"
        )
    rows = load_manifest(dataset_dir)[:3]
    if not rows:
        pytest.skip("Training dataset manifest is empty")
    summary = cache_dataset_features(
        FeatureCacheConfig(
            dataset_dir=str(dataset_dir),
            batch_size=len(rows),
            max_items=len(rows),
            skip_existing=True,
            verify_count=0,
            device="cuda:0",
        )
    )
    assert summary.total == len(rows)
    for sample_id in (row["id"] for row in rows):
        value = torch.load(
            dataset_dir / "cache" / f"{sample_id}.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert value["text_tokens"].dtype == torch.int32
        assert value["codes"].dtype == torch.int16
        assert value["campplus"].dtype == torch.float32
        assert value["emo_raw"].dtype == torch.float32
        assert value["emo_vec"].dtype == torch.float32
        assert value["text_tokens"].ndim == value["codes"].ndim == 1
        assert value["campplus"].shape == (192,)
        assert value["emo_raw"].shape == (1024,)
        assert value["emo_vec"].shape == (1280,)
        assert int(value["codes"].min()) >= 0
        assert int(value["codes"].max()) < 8192
        assert abs(value["codes"].numel() / 25.0 - value["duration_s"]) < 0.15
        assert all(torch.isfinite(value[key]).all() for key in ("campplus", "emo_raw", "emo_vec"))
