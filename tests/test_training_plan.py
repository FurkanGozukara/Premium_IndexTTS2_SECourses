from __future__ import annotations

from indextts.training.plan import (
    suggested_epochs,
    training_plan,
    training_plan_advisory,
)


def test_training_plan_with_no_validation() -> None:
    plan = training_plan(
        manifest_count=10,
        batch_size=3,
        grad_accumulation=2,
        epochs=4,
        max_steps=0,
        val_fraction=0,
    )

    assert plan == {
        "manifest_count": 10,
        "training_clips": 10,
        "validation_clips": 0,
        "micro_batches_per_epoch": 4,
        "optimizer_updates_per_epoch": 2,
        "total_optimizer_updates": 8,
    }


def test_training_plan_keeps_one_validation_clip_for_small_datasets() -> None:
    plan = training_plan(3, 1, 1, 15, 0, 0.05)

    assert plan["training_clips"] == 2
    assert plan["validation_clips"] == 1
    assert plan["micro_batches_per_epoch"] == 2
    assert plan["optimizer_updates_per_epoch"] == 2
    assert plan["total_optimizer_updates"] == 30


def test_training_plan_max_steps_caps_optimizer_updates() -> None:
    plan = training_plan(100, 8, 2, 10, 7, 0.1)

    assert plan["training_clips"] == 90
    assert plan["validation_clips"] == 10
    assert plan["micro_batches_per_epoch"] == 12
    assert plan["optimizer_updates_per_epoch"] == 6
    assert plan["total_optimizer_updates"] == 7


def test_suggested_epochs_uses_many_epochs_for_a_small_dataset() -> None:
    assert suggested_epochs(10, 1, 1) == 200


def test_suggested_epochs_respects_minimum_for_a_large_dataset() -> None:
    assert suggested_epochs(10_000, 1, 1) == 3


def test_suggested_epochs_returns_zero_without_training_clips() -> None:
    assert suggested_epochs(0, 1, 1) == 0


def test_training_plan_advisory_suggests_epochs_below_measured_range() -> None:
    plan = training_plan(1_000, 1, 1, 4, 0, 0)

    text = training_plan_advisory(plan, 1, 1)
    assert "Maximum budget: 4,000" in text
    assert "depends on this dataset" in text
    assert "sweet spot" not in text


def test_training_plan_advisory_identifies_measured_range() -> None:
    plan = training_plan(1_000, 1, 1, 10, 0, 0)

    text = training_plan_advisory(plan, 1, 1)
    assert "Maximum budget: 10,000" in text
    assert "held-out validation controls" in text


def test_training_plan_advisory_suggests_reduction_above_measured_range() -> None:
    plan = training_plan(1_000, 1, 1, 21, 0, 0)

    text = training_plan_advisory(plan, 1, 1)
    assert "Maximum budget: 21,000" in text
    assert "saved checkpoints and Base" in text
