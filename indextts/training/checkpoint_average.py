"""Average the saved checkpoints of one training into a single LoRA / DoRA file.

The last epochs of a run differ by hundredths in validation loss and swap places on the speech
benchmark from run to run. Averaging their adapter parameters (low-rank factors, DoRA magnitudes and
any fully trained modules) in parameter space is the classic remedy for that end-of-run wobble: the
files come from one optimization trajectory, so their mean sits in the same basin and usually
generalizes at least as well as any member. The averaged file is one more candidate for the speech
comparison; it never replaces a member on its own.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Sequence

import torch

from indextts.lora.io import LoraMetadata, load_lora

AVERAGED_SUFFIX_RE = re.compile(r"_avg_ep(\d+)_(\d+)$", re.IGNORECASE)


def averaged_checkpoint_name(name: str, first_epoch: int, last_epoch: int) -> str:
    """File stem of the averaged checkpoint: ``<name>_avg_ep<first>_<last>``."""
    return f"{name}_avg_ep{int(first_epoch)}_{int(last_epoch)}"


def averaged_epoch_span(path: str | os.PathLike[str]) -> tuple[int, int] | None:
    """``(first_epoch, last_epoch)`` when the file name marks an averaged checkpoint, else ``None``."""
    match = AVERAGED_SUFFIX_RE.search(Path(path).stem)
    return (int(match.group(1)), int(match.group(2))) if match else None


def average_lora_files(paths: Sequence[str | os.PathLike[str]], output: str | os.PathLike[str], *,
                       dtype: torch.dtype = torch.bfloat16) -> Path:
    """Write the parameter-space mean of several checkpoints of one training to ``output``.

    Every file must carry the same tensors with the same shapes (the same adapter layout); the
    metadata of the newest member is kept, annotated with the members and their training steps.
    """
    from safetensors.torch import save_file

    members = [Path(item).expanduser().resolve() for item in paths]
    if len(members) < 2:
        raise ValueError("averaging needs at least two checkpoints")
    if len({str(item) for item in members}) != len(members):
        raise ValueError("averaging needs distinct checkpoint files")
    if dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("averaged checkpoint dtype must be torch.bfloat16 or torch.float32")
    loaded = [load_lora(item) for item in members]
    keys = list(loaded[0].tensors)
    for file, item in zip(loaded, members):
        if list(file.tensors) != keys:
            raise ValueError(f"{item.name} does not have the same tensors as {members[0].name}; averaging needs one adapter layout")
        for key in keys:
            if tuple(file.tensors[key].shape) != tuple(loaded[0].tensors[key].shape):
                raise ValueError(f"{item.name}: tensor {key} has a different shape from {members[0].name}")
    scale = 1.0 / len(loaded)
    averaged: dict[str, torch.Tensor] = {}
    for key in keys:
        total = None
        for file in loaded:
            value = file.tensors[key].to(torch.float32)
            total = value.clone() if total is None else total.add_(value)
        averaged[key] = (total * scale).to(dtype).contiguous()
    newest = max(loaded, key=lambda file: (int(file.metadata.trained_steps), int(file.metadata.epochs)))
    metadata = LoraMetadata(**{**newest.metadata.__dict__})
    metadata.created_at = datetime.now(timezone.utc).isoformat()
    train_config = dict(metadata.train_config or {})
    train_config["averaged_from"] = [item.name for item in members]
    train_config["averaged_steps"] = [int(file.metadata.trained_steps) for file in loaded]
    train_config["averaged_epochs"] = [int(file.metadata.epochs) for file in loaded]
    metadata.train_config = train_config
    metadata.trained_steps = int(newest.metadata.trained_steps)
    metadata.epochs = int(newest.metadata.epochs)
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    save_file(averaged, str(temporary), metadata=metadata.to_header())
    os.replace(temporary, destination)
    return destination


def choose_average_members(checkpoints: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """The last ``count`` distinct training updates among epoch and final files, oldest first.

    ``checkpoints`` are descriptors from ``discover_checkpoints``. Best-loss copies of an update that
    is already present are skipped, as are earlier averaged files and interrupted saves.
    """
    usable = [item for item in checkpoints if item.get("kind") in {"epoch", "final"} and int(item.get("steps") or 0) > 0]
    by_steps: dict[int, dict[str, Any]] = {}
    for item in sorted(usable, key=lambda item: (int(item["steps"]), str(item["path"]))):
        by_steps.setdefault(int(item["steps"]), item)
    ordered = [by_steps[steps] for steps in sorted(by_steps)]
    return ordered[-max(0, int(count)):] if count and count >= 2 else []


def write_averaged_checkpoint(adapter_dir: str | os.PathLike[str], name: str, count: int) -> tuple[Path | None, str]:
    """Average the last ``count`` saved updates of a training folder into ``<name>_avg_ep<a>_<b>.safetensors``.

    Returns the written path (or ``None``) and a one-line explanation for the training log.
    """
    from .analysis import discover_checkpoints

    root = Path(adapter_dir).expanduser().resolve()
    members = choose_average_members(discover_checkpoints(root), count)
    if len(members) < 2:
        return None, f"checkpoint averaging skipped: fewer than two distinct saved updates ({len(members)} found)"
    epochs = [int(item.get("epoch") or 0) for item in members]
    destination = root / f"{averaged_checkpoint_name(name, min(epochs), max(epochs))}.safetensors"
    average_lora_files([item["path"] for item in members], destination)
    steps = ", ".join(f"{int(item['steps']):,}" for item in members)
    return destination, f"averaged the last {len(members)} saved updates ({steps}) into {destination.name}"


def averaged_members_of(path: str | os.PathLike[str]) -> list[str]:
    """The member file names recorded in an averaged checkpoint's metadata (empty for other files)."""
    try:
        from indextts.lora.io import inspect_lora
        info = inspect_lora(path)
    except Exception:
        return []
    train_config = info.get("train_config")
    if isinstance(train_config, str):
        try:
            train_config = json.loads(train_config)
        except ValueError:
            train_config = {}
    members = (train_config or {}).get("averaged_from") if isinstance(train_config, dict) else None
    return [str(item) for item in members] if isinstance(members, list) else []


__all__ = [
    "AVERAGED_SUFFIX_RE",
    "average_lora_files",
    "averaged_checkpoint_name",
    "averaged_epoch_span",
    "averaged_members_of",
    "choose_average_members",
    "write_averaged_checkpoint",
]
