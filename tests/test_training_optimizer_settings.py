from __future__ import annotations

import pytest
import torch

from indextts.training.train_config import TrainConfig
from indextts.training.trainer import _optimizer, _scheduler


def test_prodigy_uses_selected_learning_rate_and_optimizer_settings() -> None:
    pytest.importorskip("prodigyopt")
    config = TrainConfig(
        dataset_dir="dataset",
        name="adapter",
        optimizer="prodigy",
        learning_rate=0.25,
        betas=(0.8, 0.95),
        eps=1e-7,
        weight_decay=0.03,
    ).validate()
    parameter = torch.nn.Parameter(torch.tensor([1.0]))

    optimizer = _optimizer(config, [parameter])
    group = optimizer.param_groups[0]

    assert group["lr"] == config.learning_rate
    assert group["betas"] == config.betas
    assert group["eps"] == config.eps
    assert group["weight_decay"] == config.weight_decay
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert parameter.item() < 1.0


@pytest.mark.parametrize("schedule", ["cosine", "linear", "constant_with_warmup"])
def test_short_run_remains_in_requested_long_warmup(schedule: str) -> None:
    config = TrainConfig(
        dataset_dir="dataset",
        name="adapter",
        learning_rate=0.2,
        lr_scheduler=schedule,
        warmup_steps=20,
    ).validate()
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = _optimizer(config, [parameter])
    scheduler = _scheduler(config, optimizer, total_steps=5)

    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(5):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])

    assert rates == pytest.approx([config.learning_rate * step / 20 for step in range(6)])
    assert rates[-1] == pytest.approx(config.learning_rate / 4)
