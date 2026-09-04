from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from indextts.training.train_config import TrainConfig
from indextts.training.trainer import run_training


def _cached_training_dataset() -> Path:
    dataset = Path(
        os.environ.get("INDEXTTS_TEST_TRAINING_DATASET", "datasets/secourses_demo")
    )
    if not (dataset / "manifest.jsonl").is_file() or not (
        dataset / "cache_index.json"
    ).is_file():
        pytest.skip(
            "Cached training dataset is unavailable; set "
            "INDEXTTS_TEST_TRAINING_DATASET"
        )
    return dataset


def _losses(metrics_path: Path) -> list[float]:
    values = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if "loss" in row:
            values.append(float(row["loss"]))
    return values


@pytest.mark.gpu
@pytest.mark.parametrize("blocks_to_swap", [0, 12])
def test_real_dora_training_resume_and_files(tmp_path: Path, blocks_to_swap: int) -> None:
    dataset = _cached_training_dataset()
    use_int8 = blocks_to_swap > 0 and Path("models/gpt_int8_convrot.safetensors").is_file()
    name = f"smoke_swap_{blocks_to_swap}"
    config = TrainConfig(
        dataset_dir=str(dataset),
        output_dir=str(tmp_path),
        name=name,
        adapter_type="dora",
        rank=8,
        alpha=8,
        base_variant="int8_convrot" if use_int8 else "bf16",
        blocks_to_swap=blocks_to_swap,
        epochs=5,
        max_steps=30,
        batch_size=2,
        grad_accumulation=1,
        val_fraction=0,
        sample_enabled=False,
        save_every_epochs=0,
        num_workers=0,
    )
    result = run_training(config)
    adapter = Path(result.output_path)
    train_state = adapter.with_name(adapter.stem + ".train_state.pt")
    metrics = adapter.parent / "metrics.jsonl"
    assert result.status == "complete"
    assert result.step == 30
    assert adapter.is_file() and train_state.is_file() and metrics.is_file()
    losses = _losses(metrics)
    assert len(losses) == 30
    assert min(losses[-10:]) < sum(losses[:5]) / 5
    log_text = (adapter.parent / "log.txt").read_text(encoding="utf-8")
    assert "automatic checkpoint evaluation skipped" in log_text
    assert not (adapter.parent / "analysis" / "eval_job" / "status.json").exists()

    resumed = TrainConfig.from_dict(config.to_dict())
    resumed.name = name + "_resumed"
    resumed.max_steps = 35
    resumed.resume_from = str(adapter)
    resumed.resume_mode = "continue"
    resumed_result = run_training(resumed)
    assert resumed_result.step == 35
    assert Path(resumed_result.output_path).is_file()
    resumed_status = json.loads(
        (Path(resumed_result.output_path).parent / "status.json").read_text(encoding="utf-8")
    )
    assert resumed_status["phase"] == "complete"


@pytest.mark.gpu
def test_stop_flag_saves_interrupted_adapter(tmp_path: Path) -> None:
    dataset = _cached_training_dataset()
    state_dir = tmp_path / "stop_state"
    state_dir.mkdir(parents=True)
    (state_dir / "stop.flag").touch()
    config = TrainConfig(
        dataset_dir=str(dataset),
        output_dir=str(tmp_path),
        name="stop_smoke",
        rank=8,
        alpha=8,
        epochs=2,
        max_steps=10,
        batch_size=2,
        grad_accumulation=1,
        val_fraction=0,
        sample_enabled=False,
        save_every_epochs=0,
        num_workers=0,
    )
    result = run_training(config, state_dir=state_dir)
    assert result.status == "stopped"
    assert result.step == 1
    interrupted = Path(result.output_path)
    assert interrupted.name.endswith("_interrupted.safetensors")
    assert interrupted.is_file()
    assert interrupted.with_name(interrupted.stem + ".train_state.pt").is_file()
    status = json.loads((state_dir / "status.json").read_text(encoding="utf-8"))
    assert status["phase"] == "stopped"
    assert "automatic checkpoint evaluation skipped" in (
        state_dir / "log.txt"
    ).read_text(encoding="utf-8")
