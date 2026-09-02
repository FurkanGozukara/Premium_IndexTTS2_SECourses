from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import indextts.training.trainer as trainer_module
from indextts.training.charts import load_metrics, loss_frame
from indextts.training.train_config import TrainConfig
from indextts.training.trainer import BuiltTrainingModel, run_training


class _SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, *_args, split="train", **_kwargs):
        self.split = split
        self.lengths = [1, 1, 1, 1]
        self.fingerprint = "synthetic-cpu-v1"

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor(float(index + 1))

    def set_epoch(self, _epoch: int) -> None:
        return None


class _TinyTrainingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))


@pytest.fixture
def synthetic_cpu_trainer(monkeypatch):
    build_calls: list[tuple[str, str]] = []

    def fake_build(config, *, log=None):
        model = _TinyTrainingModel()
        adapter = SimpleNamespace(
            rank=config.rank,
            alpha=config.alpha,
            use_dora=config.adapter_type == "dora",
        )
        build_calls.append((config.resume_from, config.resume_mode))
        if log is not None:
            log(">> synthetic CPU model ready")
        return BuiltTrainingModel(
            model=model,
            adapters={"synthetic": adapter},
            full_modules={},
            parameters=[model.weight],
            block_swap=None,
        )

    def fake_loss(model, batch, **_kwargs):
        target = batch["value"].float().mean() / 4.0
        loss = (model.weight - target).square() + 0.01
        accuracy = torch.exp(-loss.detach())
        return loss, {"mel_accuracy": accuracy}

    def fake_save_lora(path, *_args, **_kwargs):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"synthetic-adapter")

    monkeypatch.setattr(trainer_module, "LoraTrainDataset", _SyntheticDataset)
    monkeypatch.setattr(
        trainer_module,
        "collate",
        lambda values: {"value": torch.stack(values)},
    )
    monkeypatch.setattr(trainer_module, "build_training_model", fake_build)
    monkeypatch.setattr(trainer_module, "gpt_train_step_loss", fake_loss)
    monkeypatch.setattr(trainer_module, "save_lora", fake_save_lora)
    monkeypatch.setattr(trainer_module, "set_training_mode", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        trainer_module,
        "memory_stats",
        lambda _device=None: {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "peak_allocated_gb": 0.0,
            "peak_reserved_gb": 0.0,
        },
    )
    monkeypatch.setattr(trainer_module, "gpu_total_gb", lambda _index=0: 0.0)
    return build_calls


def _config(tmp_path: Path, name: str, *, max_steps: int) -> TrainConfig:
    return TrainConfig(
        dataset_dir=str(tmp_path / "dataset"),
        name=name,
        output_dir=str(tmp_path / "loras"),
        device="cpu",
        base_dtype="fp32",
        mixed_precision="fp32",
        save_dtype="fp32",
        learning_rate=0.01,
        warmup_steps=0,
        epochs=1,
        max_steps=max_steps,
        batch_size=1,
        grad_accumulation=1,
        val_fraction=0,
        save_every_epochs=0,
        save_every_steps=0,
        save_best=False,
        sample_enabled=False,
        num_workers=0,
    ).validate()


def _training_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and "loss" in json.loads(line)
    ]


def test_weights_only_resume_starts_a_fresh_two_step_run(
    tmp_path: Path, synthetic_cpu_trainer
) -> None:
    source = run_training(_config(tmp_path, "source_weights", max_steps=3))
    resumed = _config(tmp_path, "weights_only", max_steps=2)
    resumed.resume_from = source.output_path

    result = run_training(resumed)

    assert result.status == "complete"
    assert result.step == result.total_steps == 2
    assert result.initial_loss is not None and result.final_loss is not None
    assert len(_training_rows(Path(result.output_path).parent / "metrics.jsonl")) == 2
    assert synthetic_cpu_trainer[-1] == (source.output_path, "weights_only")
    log = (Path(result.output_path).parent / "log.txt").read_text(encoding="utf-8")
    assert "fresh optimizer/scheduler at step 0" in log


def test_continue_resume_extends_an_exhausted_budget(
    tmp_path: Path, synthetic_cpu_trainer, monkeypatch
) -> None:
    scaler_step = torch.amp.GradScaler.step
    step_lrs: list[float] = []

    def tracking_scaler_step(self, optimizer, *args, **kwargs):
        step_lrs.append(float(optimizer.param_groups[0]["lr"]))
        return scaler_step(self, optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.amp.GradScaler, "step", tracking_scaler_step)
    source = run_training(_config(tmp_path, "source_continue", max_steps=3))
    step_lrs.clear()
    resumed = _config(tmp_path, "continued", max_steps=2)
    resumed.resume_from = source.output_path
    resumed.resume_mode = "continue"

    result = run_training(resumed)

    assert result.status == "complete"
    assert result.step == result.total_steps == 5
    assert result.initial_loss is not None and result.final_loss is not None
    rows = _training_rows(Path(result.output_path).parent / "metrics.jsonl")
    assert [row["step"] for row in rows] == [4, 5]
    assert rows[0]["lr"] > 0.0
    assert len(step_lrs) == 2
    assert step_lrs[0] > 0.0
    assert synthetic_cpu_trainer[-1] == (source.output_path, "continue")
    log = (Path(result.output_path).parent / "log.txt").read_text(encoding="utf-8")
    assert "resumed at step 3; training 2 more steps" in log


def test_final_step_validation_is_not_duplicated(
    tmp_path: Path, synthetic_cpu_trainer
) -> None:
    config = _config(tmp_path, "validation_once", max_steps=2)
    config.val_fraction = 0.25
    config.val_every_steps = 2

    result = run_training(config)
    rows = [
        json.loads(line)
        for line in (Path(result.output_path).parent / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    validation = [row for row in rows if row.get("event") == "validation"]
    assert len(validation) == 1
    assert validation[0]["step"] == 2


def test_zero_step_run_raises_nothing_to_train(
    tmp_path: Path, synthetic_cpu_trainer, monkeypatch
) -> None:
    class EmptyBatchSampler:
        def __init__(self, *_args, **_kwargs):
            pass

        def set_epoch(self, _epoch: int) -> None:
            pass

        def __iter__(self):
            return iter(())

        def __len__(self) -> int:
            return 0

    monkeypatch.setattr(trainer_module, "LengthBucketBatchSampler", EmptyBatchSampler)
    config = _config(tmp_path, "empty_batches", max_steps=1)

    with pytest.raises(RuntimeError, match=r"nothing to train:"):
        run_training(config)

    status = json.loads(
        (Path(config.output_dir) / config.name / "status.json").read_text(
            encoding="utf-8"
        )
    )
    assert status["phase"] == "failed"


def test_metric_charts_collapse_duplicate_event_rows(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    rows = [
        {"step": 2, "loss": 1.2, "lr": 1e-4},
        {"event": "validation", "step": 2, "val_loss": 1.1},
        {"event": "validation", "step": 2, "val_loss": 0.9},
    ]
    (state / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    metrics = load_metrics(state)
    chart = loss_frame(metrics, smoothing=0)

    assert len(metrics) == 2
    validation = chart[chart["series"] == "validation"]
    assert validation[["step", "value"]].values.tolist() == [[2, 0.9]]
