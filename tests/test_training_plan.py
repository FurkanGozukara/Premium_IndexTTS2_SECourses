from __future__ import annotations

from indextts.training.plan import training_plan


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
