from __future__ import annotations

import json
from pathlib import Path

import pytest

import indextts.training.analysis as analysis_module
from indextts.training.analysis import (
    ANALYSIS_SERIES,
    analysis_epoch_frame,
    analyze_training_run,
    checkpoint_descriptor,
    checkpoint_display_label,
    classify_epoch_phases,
    display_legacy_report_text,
    load_training_analysis,
    write_training_analysis,
)


def _metrics(path: Path, train: list[float], validation: list[float] | None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rows = []
    for epoch, loss in enumerate(train, start=1):
        rows.append(
            {
                "step": epoch,
                "epoch": epoch,
                "loss": loss,
                "mel_accuracy": 0.1 + epoch / 100,
                "lr": 1e-4,
            }
        )
        if validation is not None:
            rows.append(
                {
                    "event": "validation",
                    "step": epoch,
                    "epoch": epoch,
                    "val_loss": validation[epoch - 1],
                    "val_mel_accuracy": 0.05 + epoch / 100,
                }
            )
    (path / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _checkpoints(adapter: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (adapter / "best").mkdir(parents=True, exist_ok=True)
    for path in (
        adapter / "voice.safetensors",
        adapter / "voice_epoch_002.safetensors",
        adapter / "best" / "voice.safetensors",
    ):
        path.write_bytes(b"test")

    def inspect(path):
        source = Path(path)
        if source.parent.name == "best" or "epoch_002" in source.stem:
            epoch = 2
        else:
            epoch = 4
        return {
            "adapter_type": "dora",
            "rank": 2,
            "steps": epoch,
            "epochs": epoch,
            "train_config": {},
        }

    monkeypatch.setattr(analysis_module, "inspect_lora", inspect)


def test_phase_classification_is_pure_and_uses_earliest_best() -> None:
    values = {1: 4.0, 2: 3.0, 3: 3.0, 4: 3.2}
    original = dict(values)

    phases, best_epoch, overfit_start = classify_epoch_phases(values, tolerance=0.01)

    assert values == original
    assert best_epoch == 2
    assert overfit_start == 4
    assert phases == {1: "improving", 2: "best", 3: "plateau", 4: "overfitting"}


def test_legacy_report_wording_is_upgraded_only_for_display() -> None:
    stored = "Base model (no adapter): the adapter beats other adapters."

    displayed = display_legacy_report_text(stored)

    assert stored == "Base model (no adapter): the adapter beats other adapters."
    assert displayed == (
        "Base model (no LoRA / DoRA): the LoRA / DoRA beats other LoRA / DoRA."
    )
    assert display_legacy_report_text("base model (no adapter)") == (
        "Base model (no LoRA / DoRA)"
    )


@pytest.mark.parametrize(
    ("relative_path", "adapter_type", "epoch", "steps", "label", "file_label"),
    [
        ("best/voice.safetensors", "dora", 10, 100, "best (epoch 10 DoRA Checkpoint)", "best_ep10"),
        ("voice_epoch_030.safetensors", "dora", 30, 300, "epoch 30 (DoRA Checkpoint)", "epoch_030"),
        ("voice.safetensors", "dora", 40, 400, "final (epoch 40 DoRA Checkpoint)", "final_ep40"),
        ("voice_interrupted.safetensors", "dora", 7, 70, "interrupted (epoch 7 DoRA Checkpoint)", "interrupted_ep07"),
        ("voice_step_000500.safetensors", "dora", 0, 500, "step 500 (DoRA Checkpoint)", "step_000500"),
        ("voice_epoch_002.safetensors", "unknown", 2, 20, "epoch 2 (LoRA / DoRA Checkpoint)", "epoch_002"),
    ],
)
def test_checkpoint_descriptor_includes_saved_lora_or_dora_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
    adapter_type: str,
    epoch: int,
    steps: int,
    label: str,
    file_label: str,
) -> None:
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        analysis_module,
        "inspect_lora",
        lambda _path: {
            "adapter_type": adapter_type,
            "epochs": epoch,
            "steps": steps,
            "train_config": {},
        },
    )

    descriptor = checkpoint_descriptor(path)

    assert descriptor["label"] == label
    assert descriptor["file_label"] == file_label


def test_legacy_checkpoint_display_label_upgrades_once_and_keeps_strength(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "voice_epoch_010.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    calls = 0

    def inspect(_path):
        nonlocal calls
        calls += 1
        return {
            "adapter_type": "dora",
            "epochs": 10,
            "steps": 100,
            "train_config": {},
        }

    monkeypatch.setattr(analysis_module, "inspect_lora", inspect)
    cache: dict[str, str] = {}

    assert checkpoint_display_label(
        "epoch 10 @0.5", path=checkpoint, kind="epoch", cache=cache
    ) == "epoch 10 (DoRA Checkpoint) @0.5"
    assert checkpoint_display_label(
        "epoch 10", path=checkpoint, kind="epoch", cache=cache
    ) == "epoch 10 (DoRA Checkpoint)"
    assert calls == 1


def test_middle_best_detects_sustained_overfit_and_round_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = tmp_path / "voice"
    state = tmp_path / "state"
    _metrics(state, [4.0, 3.0, 2.0, 1.0], [5.0, 4.0, 4.2, 4.3])
    _checkpoints(adapter, monkeypatch)

    result = analyze_training_run(adapter, state, tolerance=0.01)

    assert result.status == "best_found"
    assert result.best_epoch == 2
    assert result.overfit_start_epoch == 3
    assert Path(result.recommended_checkpoint).parent.name == "best"
    assert "epoch 2" in result.summary_markdown
    assert "epoch 3" in result.summary_markdown
    frame = analysis_epoch_frame(result)
    assert set(frame["series"]) == set(ANALYSIS_SERIES)
    path = write_training_analysis(result)
    assert path.is_file()
    assert path.with_suffix(".md").is_file()
    assert load_training_analysis(adapter).best_epoch == 2


@pytest.mark.parametrize(
    ("validation", "expected"),
    [
        ([5.0, 4.0, 3.0], "still_improving"),
        ([4.0, 3.0, 3.02, 3.01], "plateau"),
    ],
)
def test_monotonic_and_plateau_statuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    validation: list[float],
    expected: str,
) -> None:
    adapter = tmp_path / expected
    state = tmp_path / f"{expected}_state"
    _metrics(state, [4.0 - index * 0.2 for index in range(len(validation))], validation)
    _checkpoints(adapter, monkeypatch)
    result = analyze_training_run(adapter, state)
    assert result.status == expected


def test_no_validation_recommends_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = tmp_path / "no_validation"
    state = tmp_path / "no_validation_state"
    _metrics(state, [3.0, 2.0, 1.0, 0.5], None)
    _checkpoints(adapter, monkeypatch)
    result = analyze_training_run(adapter, state)
    assert result.status == "no_validation"
    assert result.best_epoch is None
    assert Path(result.recommended_checkpoint).parent == adapter.resolve()
    assert all(item.phase == "unknown" for item in result.epochs)


def test_reference_run_generalization_when_available() -> None:
    root = Path(__file__).resolve().parents[1]
    adapter = root / "loras" / "SECourses_Furkan_EN_DoRA_r32"
    state = root / "outputs" / "training_runs" / "furkan_dora_r32"
    if not adapter.is_dir() or not (state / "metrics.jsonl").is_file():
        pytest.skip("reference training run is not present")
    result = analyze_training_run(adapter, state, tolerance=0.01)
    assert result.best_epoch == 10
    assert result.overfit_start_epoch is not None
    assert abs(result.overfit_start_epoch - 12) <= 2
