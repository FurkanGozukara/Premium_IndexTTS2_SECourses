from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
import wave

import pytest
import torch

import indextts.training.trainer as trainer_module
from indextts.training.charts import load_metrics, loss_frame
from indextts.training.dataset_manifest import write_manifest
from indextts.training.train_config import TrainConfig
from indextts.training.trainer import BuiltTrainingModel, LoraTrainer, run_training


class _SyntheticDataset(torch.utils.data.Dataset):
    calls: list[dict] = []

    def __init__(self, *_args, split="train", **_kwargs):
        self.split = split
        self.calls.append({"split": split, **_kwargs})
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
    _SyntheticDataset.calls.clear()

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
        return loss, {"mel_accuracy": accuracy, "mel_loss": loss,
                      "text_loss": torch.tensor(0.0), "mel_tokens": 1, "text_tokens": 1}

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
        auto_analyze=False,
        auto_evaluate_checkpoints=False,
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
    saved_config = json.loads((Path(result.output_path).parent / "train_config.json").read_text(encoding="utf-8"))
    assert saved_config["name"] == "weights_only"
    assert saved_config["val_split_mode"] == resumed.val_split_mode
    assert saved_config["early_stop_enabled"] == resumed.early_stop_enabled
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


def test_missing_validation_reference_fails_before_model_loading(
    tmp_path: Path, monkeypatch
) -> None:
    config = _config(tmp_path, "invalid_holdout", max_steps=2)
    dataset = Path(config.dataset_dir)
    dataset.mkdir()
    cache = dataset / "cache"
    cache.mkdir()
    for name in ("train", "holdout"):
        torch.save(
            {"codes": torch.tensor([1, 2, 3]), "text_tokens": torch.tensor([4, 5, 6]),
             "campplus": torch.ones(192), "emo_raw": torch.ones(1024),
             "emo_vec": torch.ones(1280), "language": "EN"},
            cache / f"{name}.pt",
        )
    write_manifest(
        dataset / "manifest.jsonl",
        [
            {"id": "train", "speaker": "training_voice", "split": "train", "n_codes": 3, "n_text_tokens": 3},
            {"id": "holdout", "speaker": "held_out_voice", "split": "val", "n_codes": 3, "n_text_tokens": 3},
        ],
    )
    monkeypatch.setattr(
        trainer_module, "build_training_model",
        lambda *_args, **_kwargs: pytest.fail("Invalid validation data must fail before loading a model"),
    )

    with pytest.raises(ValueError, match="no training reference for: held_out_voice"):
        run_training(config)

    status = json.loads((Path(config.output_dir) / config.name / "status.json").read_text(encoding="utf-8"))
    assert status["phase"] == "failed"
    assert "no training reference" in status["message"]


def test_empty_validation_split_still_allows_training(
    tmp_path: Path, synthetic_cpu_trainer, monkeypatch
) -> None:
    class NoValidationDataset(_SyntheticDataset):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.split == "val":
                self.lengths = []

    monkeypatch.setattr(trainer_module, "LoraTrainDataset", NoValidationDataset)
    result = run_training(_config(tmp_path, "no_validation", max_steps=2))

    assert result.status == "complete"
    assert result.step == 2
    rows = [json.loads(line) for line in (Path(result.output_path).parent / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert not any(row.get("event") == "validation" for row in rows)


def test_fp16_scaler_recovers_overflow_without_counting_an_update(
    tmp_path: Path, synthetic_cpu_trainer, monkeypatch
) -> None:
    original_scaler = torch.amp.GradScaler
    monkeypatch.setattr(torch.amp, "GradScaler", lambda *_args, **_kwargs: original_scaler("cpu", enabled=True))
    original_loss = trainer_module.gpt_train_step_loss
    training_calls = 0

    def overflow_once(*args, **kwargs):
        nonlocal training_calls
        loss, values = original_loss(*args, **kwargs)
        if torch.is_grad_enabled():
            training_calls += 1
            if training_calls == 1:
                loss.register_hook(lambda grad: torch.full_like(grad, float("inf")))
        return loss, values

    monkeypatch.setattr(trainer_module, "gpt_train_step_loss", overflow_once)
    result = run_training(_config(tmp_path, "overflow_recovery", max_steps=3))
    assert result.step == 3
    assert training_calls == 4
    rows = _training_rows(Path(result.output_path).parent / "metrics.jsonl")
    assert len(rows) == 3
    assert all(row["grad_norm"] < float("inf") for row in rows)
    log = (Path(result.output_path).parent / "log.txt").read_text(encoding="utf-8")
    assert "skipped overflowing FP16 update" in log


def test_step_checkpoint_contains_same_step_validation_state(
    tmp_path: Path, synthetic_cpu_trainer
) -> None:
    config = _config(tmp_path, "step_state", max_steps=3)
    config.val_fraction = 0.25
    config.val_every_steps = 2
    config.save_every_steps = 2
    config.epoch_train_state = True
    config.save_best = True
    result = run_training(config)
    state_path = Path(result.output_path).parent / "step_state_step_000002.train_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    assert state["early_stopping"]["last_step"] == 2
    assert state["early_stopping"]["checks"] == 1
    assert state["best_val_loss"] == state["early_stopping"]["best_loss"]


def test_conditioning_modes_and_resolved_sample_seed_are_reported(
    tmp_path: Path, synthetic_cpu_trainer, monkeypatch
) -> None:
    sample_calls: list[dict] = []

    def fake_sample(_config, **kwargs):
        sample_calls.append(kwargs)
        return SimpleNamespace(generated=True, path=str(tmp_path / "sample.wav"))

    monkeypatch.setattr(trainer_module, "generate_training_sample", fake_sample)
    monkeypatch.setattr(trainer_module.secrets, "randbelow", lambda _upper: 24680)
    config = _config(tmp_path, "reference_modes", max_steps=5)
    config.val_fraction = 0.25
    config.speaker_ref_mode = "mixed"
    config.emo_ref_mode = "follow_speaker"
    config.val_reference_mode = "other"
    config.sample_enabled = True
    config.sample_seed = -1

    result = run_training(config)
    adapter_dir = Path(result.output_path).parent
    log = (adapter_dir / "log.txt").read_text(encoding="utf-8")
    status = json.loads((adapter_dir / "status.json").read_text(encoding="utf-8"))
    metrics = [
        json.loads(line)
        for line in (adapter_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    validation = [row for row in metrics if row.get("event") == "validation"]

    assert (
        ">> conditioning | train speaker ref: mixed | train emotion ref: "
        "follow_speaker | validation ref: other"
    ) in log
    assert ">> training samples use seed 24680" in log
    assert validation and all(row["reference_mode"] == "other" for row in validation)
    assert status["sample_seed"] == 24680
    assert status["val_reference_mode"] == "other"
    assert len(sample_calls) == 2
    assert all(call["seed"] == 24680 for call in sample_calls)
    train_call = next(call for call in _SyntheticDataset.calls if call["split"] == "train")
    val_call = next(call for call in _SyntheticDataset.calls if call["split"] == "val")
    assert train_call["speaker_ref_mode"] == "mixed"
    assert train_call["emo_ref_mode"] == "follow_speaker"
    assert val_call["speaker_ref_mode"] == "other"
    assert val_call["emo_ref_mode"] == "follow_speaker"


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


def test_early_stopping_writes_the_normal_final_adapter(
    tmp_path: Path, synthetic_cpu_trainer, monkeypatch
) -> None:
    values = iter([(1.0, 0.1), (1.2, 0.1)])
    monkeypatch.setattr(
        trainer_module.LoraTrainer,
        "validate",
        lambda self, built, loader, device: next(values),
    )
    config = _config(tmp_path, "early_stop", max_steps=4)
    config.val_fraction = 0.25
    config.val_every_steps = 1
    config.early_stop_patience = 1
    config.early_stop_min_steps = 0
    config.early_stop_min_epochs = 0

    result = run_training(config)

    assert result.status == "stopped"
    assert result.step == 2
    assert Path(result.output_path).name == "early_stop.safetensors"
    status = json.loads((Path(result.output_path).parent / "status.json").read_text(encoding="utf-8"))
    assert "early stopping" in status["message"]


@pytest.mark.parametrize(
    ("keep_last_n", "expected_epochs"),
    [(0, {1, 2, 3}), (1, {3})],
)
def test_epoch_checkpoint_retention_honors_zero_as_keep_everything(
    tmp_path: Path,
    synthetic_cpu_trainer,
    keep_last_n: int,
    expected_epochs: set[int],
) -> None:
    config = _config(tmp_path, f"keep_{keep_last_n}", max_steps=0)
    config.epochs = 3
    config.save_every_epochs = 1
    config.keep_last_n = keep_last_n

    result = run_training(config)

    saved_epochs = {
        int(path.stem.rsplit("_", 1)[-1])
        for path in Path(result.output_path).parent.glob("*_epoch_*.safetensors")
    }
    assert saved_epochs == expected_epochs


def test_periodic_checkpoints_skip_train_state_but_best_and_final_keep_it(
    tmp_path: Path,
    synthetic_cpu_trainer,
) -> None:
    config = _config(tmp_path, "compact_epochs", max_steps=0)
    config.epochs = 2
    config.val_fraction = 0.25
    config.val_every_steps = 0
    config.save_every_epochs = 1
    config.save_every_steps = 1
    config.save_best = True
    config.save_train_state = True
    config.epoch_train_state = False

    result = run_training(config)
    adapter_dir = Path(result.output_path).parent

    periodic = [
        *adapter_dir.glob("*_epoch_*.safetensors"),
        *adapter_dir.glob("*_step_*.safetensors"),
    ]
    assert periodic
    assert all(
        not path.with_name(f"{path.stem}.train_state.pt").exists()
        for path in periodic
    )
    final_state = Path(result.output_path).with_name(
        f"{Path(result.output_path).stem}.train_state.pt"
    )
    best_path = Path(result.best_path)
    best_state = best_path.with_name(f"{best_path.stem}.train_state.pt")
    assert final_state.is_file()
    assert best_path.is_file()
    assert best_state.is_file()


def test_trainer_persists_sample_calibration_in_status(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, "paced_voice", max_steps=1)
    config.sample_text = "one two three four"
    dataset = Path(config.dataset_dir)
    dataset.mkdir()
    write_manifest(
        dataset / "manifest.jsonl",
        [{"id": "one", "words": 4, "duration_s": 2.0}],
    )
    trainer = LoraTrainer(config)
    sample = trainer.adapter_dir / "samples" / "epoch_001.wav"
    sample.parent.mkdir()
    with wave.open(str(sample), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\x00\x40" * 16000)

    rate = trainer._write_speaking_rate_calibration()

    assert rate == 1.0
    status = json.loads(trainer.status_path.read_text(encoding="utf-8"))
    assert status["recommended_speaking_rate"] == 1.0
    assert (
        trainer.adapter_dir / "analysis" / "speaking_rate.json"
    ).is_file()
    assert "matches your real pace" in trainer.log_path.read_text(encoding="utf-8")


def test_auto_analysis_writes_report(
    tmp_path: Path, synthetic_cpu_trainer
) -> None:
    config = _config(tmp_path, "auto_analysis", max_steps=2)
    config.val_fraction = 0.25
    config.auto_analyze = True
    config.auto_evaluate_checkpoints = False

    result = run_training(config)

    report = Path(result.output_path).parent / "analysis" / "training_analysis.json"
    assert report.is_file()
    assert json.loads(report.read_text(encoding="utf-8"))["best_epoch"] == 1


def test_automatic_evaluation_uses_frontend_configured_settings(
    tmp_path: Path, monkeypatch
) -> None:
    class FinishedProcess:
        returncode = 1
        stdout = io.StringIO("")

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -1

    monkeypatch.setattr(trainer_module.subprocess, "Popen", lambda *_args, **_kwargs: FinishedProcess())
    config = _config(tmp_path, "eval_settings", max_steps=1)
    config.auto_evaluate_checkpoints = True
    config.eval_include_base = False
    config.eval_train_subset = 7
    config.eval_strengths = "0.5, 1.25"
    config.val_reference_mode = "other"
    trainer = LoraTrainer(config)

    trainer._run_automatic_evaluation(
        terminal_phase="complete",
        terminal_message="done",
        recommended_checkpoint="",
    )

    payload = json.loads(
        (trainer.adapter_dir / "analysis" / "eval_job" / "eval_config.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["include_base"] is False
    assert payload["train_subset"] == 7
    assert payload["strengths"] == [0.5, 1.25]
    assert payload["reference_mode"] == "other"
    assert "automatic checkpoint evaluation settings" in trainer.log_path.read_text(
        encoding="utf-8"
    )
