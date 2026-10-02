from __future__ import annotations

from indextts.training.dataset import TokenBudgetBatchSampler
from indextts.training.plan import (
    automatic_epochs,
    suggested_epochs,
    token_budget_micro_batches,
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


def test_token_budget_plan_counts_the_trainers_own_batches() -> None:
    # 2 s clips with 40-character transcripts: 50 frames + 10 text tokens + 20 special tokens.
    rows = [{"duration_s": 2.0, "text": "x" * 40} for _ in range(100)]
    assert token_budget_micro_batches(rows, 800, seed=7) == len(TokenBudgetBatchSampler([80] * 100, 800, seed=7))
    plan = training_plan(100, 64, 2, 3, 0, 0, micro_batches_per_epoch=token_budget_micro_batches(rows, 800, seed=7))
    assert plan["micro_batches_per_epoch"] == len(TokenBudgetBatchSampler([80] * 100, 800, seed=7))
    assert plan["total_optimizer_updates"] == 3 * -(-plan["micro_batches_per_epoch"] // 2)


def test_automatic_epochs_train_smaller_voices_longer() -> None:
    # 14 hours trained best after about 25 epochs; a 2-hour subset still improved until about epoch 55.
    assert automatic_epochs(14 * 3600) == 25
    assert 55 <= automatic_epochs(2 * 3600) <= 75
    assert automatic_epochs(60) == 100 and automatic_epochs(500 * 3600) == 10


def test_epochs_zero_means_automatic_only_for_omnivoice() -> None:
    from indextts.training.train_config import TrainConfig

    base = {"dataset_dir": "data", "name": "test"}
    assert TrainConfig.from_dict({**base, "tts_model": "omnivoice", "epochs": 0}).epochs == 0
    assert TrainConfig.from_dict({**base, "tts_model": "indextts", "epochs": 0}).epochs == 1
