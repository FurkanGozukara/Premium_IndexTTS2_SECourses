"""Pure dataset-split and optimizer-step planning helpers."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Sequence


def _stable_unit_interval(*parts: Any) -> float:
    value = "\x1f".join(str(part) for part in parts).encode("utf-8", "surrogatepass")
    integer = int.from_bytes(hashlib.sha256(value).digest()[:8], "big")
    return integer / float(2**64)


def validation_split_ids(
    record_ids: Sequence[str],
    val_fraction: float,
    seed: int,
) -> set[str]:
    """Return the deterministic validation IDs used by ``LoraTrainDataset``."""

    ids = [str(record_id) for record_id in record_ids]
    fraction = min(0.5, max(0.0, float(val_fraction)))
    if fraction <= 0.0 or len(ids) < 2:
        return set()
    selected = {
        record_id
        for record_id in ids
        if _stable_unit_interval(seed, record_id, "split") < fraction
    }
    if not selected:
        selected.add(min(ids, key=lambda record_id: _stable_unit_interval(seed, record_id, "split")))
    if len(selected) == len(ids):
        selected.remove(
            max(ids, key=lambda record_id: _stable_unit_interval(seed, record_id, "split"))
        )
    return selected


def _validation_count(manifest_count: int, val_fraction: float) -> int:
    fraction = min(0.5, max(0.0, float(val_fraction)))
    if fraction <= 0.0 or manifest_count < 2:
        return 0
    return min(manifest_count - 1, max(1, int(manifest_count * fraction)))


def training_plan(
    manifest_count: int,
    batch_size: int,
    grad_accumulation: int,
    epochs: int,
    max_steps: int,
    val_fraction: float,
    *,
    record_ids: Sequence[str] | None = None,
    seed: int = 42,
    validation_count: int | None = None,
) -> dict[str, int]:
    """Calculate split sizes and optimizer updates without loading model assets.

    Supplying ``record_ids`` applies the trainer's exact deterministic split.
    ``validation_count`` is useful when the caller already constructed the split.
    """

    clips = max(0, int(manifest_count))
    batch = max(1, int(batch_size))
    accumulation = max(1, int(grad_accumulation))
    epoch_count = max(1, int(epochs))
    step_limit = max(0, int(max_steps))

    if validation_count is not None:
        validation_clips = min(clips, max(0, int(validation_count)))
    elif record_ids is not None:
        ids = [str(record_id) for record_id in record_ids]
        if len(ids) != clips:
            raise ValueError("record_ids length must match manifest_count")
        validation_ids = validation_split_ids(ids, val_fraction, int(seed))
        validation_clips = sum(record_id in validation_ids for record_id in ids)
    else:
        validation_clips = _validation_count(clips, val_fraction)

    training_clips = clips - validation_clips
    if training_clips:
        micro_batches = max(1, math.ceil(training_clips / batch))
        optimizer_updates = max(1, math.ceil(micro_batches / accumulation))
        total_updates = step_limit if step_limit else epoch_count * optimizer_updates
    else:
        micro_batches = 0
        optimizer_updates = 0
        total_updates = 0

    return {
        "manifest_count": clips,
        "training_clips": training_clips,
        "validation_clips": validation_clips,
        "micro_batches_per_epoch": micro_batches,
        "optimizer_updates_per_epoch": optimizer_updates,
        "total_optimizer_updates": total_updates,
    }


def training_plan_line(plan: dict[str, int]) -> str:
    """Format the shared one-line plan used by the UI and training log."""

    return (
        f"{plan['manifest_count']:,} clips in dataset: {plan['training_clips']:,} training, "
        f"{plan['validation_clips']:,} validation; "
        f"{plan['micro_batches_per_epoch']:,} micro-batches per epoch; "
        f"{plan['optimizer_updates_per_epoch']:,} optimizer updates per epoch; "
        f"{plan['total_optimizer_updates']:,} total optimizer updates."
    )


__all__ = ["training_plan", "training_plan_line", "validation_split_ids"]
