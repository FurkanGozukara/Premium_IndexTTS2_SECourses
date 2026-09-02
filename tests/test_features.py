from __future__ import annotations

from pathlib import Path

import pytest
import torch

from indextts.training.features import FeatureCacheConfig, cache_dataset_features


@pytest.mark.gpu
def test_cache_three_real_segments() -> None:
    dataset_dir = Path("datasets/secourses_demo")
    if not (dataset_dir / "manifest.jsonl").is_file():
        pytest.skip("secourses_demo dataset is unavailable")
    summary = cache_dataset_features(
        FeatureCacheConfig(
            dataset_dir=str(dataset_dir),
            batch_size=3,
            max_items=3,
            skip_existing=True,
            verify_count=0,
            device="cuda:0",
        )
    )
    assert summary.total == 3
    for sample_id in ("video1_0001", "video1_0002", "video1_0003"):
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
