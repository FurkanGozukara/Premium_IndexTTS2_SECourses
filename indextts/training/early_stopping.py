"""Resumable progress tracking for checkpoint selection and early stopping."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


@dataclass
class EarlyStopping:
    best_loss: float | None = None
    best_step: int = 0
    best_epoch: int = 0
    meaningful_best: float | None = None
    bad_checks: int = 0
    checks: int = 0
    last_step: int = -1
    reason: str = ""

    @classmethod
    def from_state(cls, state: Mapping[str, Any] | None) -> "EarlyStopping":
        values = dict(state or {})
        return cls(**{key: value for key, value in values.items() if key in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def observe(
        self, loss: float, *, step: int, epoch: float, enabled: bool,
        patience: int, min_delta: float, min_steps: int, min_epochs: float,
    ) -> tuple[bool, bool]:
        """Return (new absolute best, stop). Tiny gains do not reset patience.

        The grace period never consumes patience. A step evaluated twice (for
        example an epoch boundary) cannot accidentally count as two checks.
        """
        if not math.isfinite(loss):
            raise FloatingPointError("validation loss is non-finite; refusing checkpoint selection")
        if step == self.last_step:
            return False, bool(self.reason)
        self.last_step = step
        self.checks += 1
        is_best = self.best_loss is None or loss < self.best_loss
        if is_best:
            self.best_loss, self.best_step = loss, step
            self.best_epoch = max(1, math.ceil(epoch))
        meaningful = self.meaningful_best is None or loss < self.meaningful_best - min_delta
        if meaningful:
            self.meaningful_best = loss
        eligible = enabled and patience > 0 and step >= min_steps and epoch >= min_epochs
        self.bad_checks = 0 if meaningful or not eligible else self.bad_checks + 1
        self.reason = ""
        if eligible and self.bad_checks >= patience:
            self.reason = (
                f"early stopping at step {step}: {self.bad_checks} validation checks without "
                f"an improvement greater than {min_delta:g}; best loss {self.best_loss:.4f} "
                f"at step {self.best_step} (epoch {self.best_epoch})"
            )
        return is_best, bool(self.reason)
