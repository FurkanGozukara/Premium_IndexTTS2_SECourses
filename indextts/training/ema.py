"""Exponential moving average of the trainable adapter weights.

A fine-tuned adapter's last update is one noisy point on its trajectory; the
running average of the recent updates is smoother and, in published TTS
fine-tunes, was rated more natural than the raw weights at the same loss. The
trainer keeps one FP32 shadow per trainable tensor, updates it after every
optimizer step, and saves the shadow beside each raw checkpoint as
``<name>_ema.safetensors`` (final) or ``<name>_ema_epoch_NNN.safetensors``
(epoch files). The saved EMA file is an ordinary adapter: the speech comparison
judges the final one next to the raw checkpoints and the app loads it like any
other. Nothing here depends on a particular voice.
"""

from __future__ import annotations

from contextlib import contextmanager
import re
from typing import Iterator, Sequence

import torch


EMA_STEM_RE = re.compile(r"_ema(?:_epoch_(\d+))?$", re.IGNORECASE)
# The first updates are averaged with a shorter horizon so the shadow does not
# stay glued to the initial weights: decay_t = min(decay, (1 + t) / (10 + t)).
EMA_WARMUP_OFFSET = 10


def ema_checkpoint_name(name: str, epoch: int | None = None) -> str:
    """File stem of the EMA sibling of a checkpoint: ``<name>_ema`` or ``<name>_ema_epoch_NNN``."""

    return f"{name}_ema" if epoch is None else f"{name}_ema_epoch_{int(epoch):03d}"


def ema_epoch_of(stem: str) -> tuple[bool, int | None]:
    """``(is_ema, epoch)`` for a checkpoint file stem."""

    match = EMA_STEM_RE.search(str(stem))
    if not match:
        return False, None
    return True, (int(match.group(1)) if match.group(1) else None)


def effective_decay(decay: float, updates: int) -> float:
    """The decay used for update number ``updates`` (1-based) with the warm-up rule."""

    return min(float(decay), (1.0 + updates) / (EMA_WARMUP_OFFSET + updates))


class AdapterEMA:
    """FP32 shadows of the trainable parameters, updated after every optimizer step."""

    def __init__(self, parameters: Sequence[torch.nn.Parameter], decay: float) -> None:
        if not 0.0 < float(decay) < 1.0:
            raise ValueError("EMA decay must be between 0 and 1 (exclusive)")
        self.decay = float(decay)
        self.parameters = [parameter for parameter in parameters if parameter is not None]
        self.shadows = [parameter.detach().to(torch.float32).clone() for parameter in self.parameters]
        self.updates = 0

    @property
    def current_decay(self) -> float:
        return effective_decay(self.decay, max(1, self.updates))

    @torch.no_grad()
    def update(self) -> float:
        """Blend the current weights into the shadows; returns the decay that was used."""

        self.updates += 1
        decay = effective_decay(self.decay, self.updates)
        for shadow, parameter in zip(self.shadows, self.parameters):
            value = parameter.detach()
            if value.device != shadow.device or value.dtype != shadow.dtype:
                value = value.to(device=shadow.device, dtype=shadow.dtype)
            shadow.mul_(decay).add_(value, alpha=1.0 - decay)
        return decay

    @contextmanager
    def averaged_weights(self) -> Iterator[None]:
        """Temporarily replace the live weights with the shadows (for saving or evaluation)."""

        backups = [parameter.data for parameter in self.parameters]
        try:
            with torch.no_grad():
                for parameter, shadow in zip(self.parameters, self.shadows):
                    parameter.data = shadow.to(device=parameter.device, dtype=parameter.dtype)
            yield
        finally:
            with torch.no_grad():
                for parameter, backup in zip(self.parameters, backups):
                    parameter.data = backup

    def state_dict(self) -> dict[str, object]:
        return {"decay": self.decay, "updates": self.updates, "shadows": [shadow.cpu() for shadow in self.shadows]}

    def load_state_dict(self, state: dict[str, object]) -> bool:
        """Restore shadows saved by :meth:`state_dict`; False when the shapes no longer match."""

        shadows = state.get("shadows") if isinstance(state, dict) else None
        if not isinstance(shadows, (list, tuple)) or len(shadows) != len(self.shadows):
            return False
        if any(tuple(saved.shape) != tuple(current.shape) for saved, current in zip(shadows, self.shadows)):
            return False
        for current, saved in zip(self.shadows, shadows):
            current.copy_(saved.to(device=current.device, dtype=current.dtype))
        self.updates = int(state.get("updates", 0) or 0)
        return True


__all__ = ["AdapterEMA", "EMA_STEM_RE", "EMA_WARMUP_OFFSET", "effective_decay", "ema_checkpoint_name", "ema_epoch_of"]
