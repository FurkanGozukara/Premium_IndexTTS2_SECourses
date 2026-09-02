"""On-demand device residency for inference-only auxiliary models."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Literal

import torch
from torch import nn


ResidencyPolicy = Literal["gpu", "on_demand", "cpu"]


@dataclass
class _Entry:
    module: Any
    policy: ResidencyPolicy
    uses: int = 0


def _managed_module(value: Any) -> Any:
    if isinstance(value, nn.Module):
        return value
    for attribute in ("llm", "model"):
        candidate = getattr(value, attribute, None)
        if isinstance(candidate, nn.Module) or callable(getattr(candidate, "to", None)):
            return candidate
    return value


def _move(value: Any, device: torch.device) -> None:
    module = _managed_module(value)
    move = getattr(module, "to", None)
    if not callable(move):
        return
    try:
        move(device=device, non_blocking=device.type != "cpu")
    except TypeError:
        try:
            move(device, non_blocking=device.type != "cpu")
        except TypeError:
            move(device)


def _module_bytes(value: Any, device: torch.device | None = None) -> int:
    module = _managed_module(value)
    if not isinstance(module, nn.Module):
        return 0
    seen: set[tuple[str, int]] = set()
    total = 0
    for tensor in (*module.parameters(recurse=True), *module.buffers(recurse=True)):
        pointer = tensor.untyped_storage().data_ptr() if tensor.numel() else id(tensor)
        identity = (str(tensor.device), pointer)
        if identity in seen:
            continue
        seen.add(identity)
        if device is None or tensor.device == device:
            total += tensor.numel() * tensor.element_size()
    return total


class ResidencyManager:
    def __init__(self, device: str | torch.device):
        resolved = torch.device(device)
        if resolved.type == "cuda" and resolved.index is None and torch.cuda.is_available():
            resolved = torch.device("cuda", torch.cuda.current_device())
        self.device = resolved
        self._entries: dict[str, _Entry] = {}

    def register(self, name: str, module: Any, policy: ResidencyPolicy) -> Any:
        normalized = str(policy).strip().lower()
        if normalized not in {"gpu", "on_demand", "cpu"}:
            raise ValueError(f"Invalid residency policy for {name}: {policy!r}")
        entry = _Entry(module=module, policy=normalized)  # type: ignore[arg-type]
        self._entries[str(name)] = entry
        if self.device.type != "cpu":
            _move(module, self.device if normalized == "gpu" else torch.device("cpu"))
        return module

    def policy(self, name: str) -> ResidencyPolicy:
        return self._entries[name].policy

    def current_device(self, name: str) -> torch.device:
        managed = _managed_module(self._entries[name].module)
        if isinstance(managed, nn.Module):
            parameter = next(managed.parameters(), None)
            if parameter is not None:
                return parameter.device
            buffer = next(managed.buffers(), None)
            if buffer is not None:
                return buffer.device
        return torch.device("cpu")

    @contextmanager
    def use(self, name: str) -> Iterator[Any]:
        try:
            entry = self._entries[name]
        except KeyError as exc:
            raise KeyError(f"No auxiliary model named {name!r} is registered") from exc

        move_on_demand = self.device.type != "cpu" and entry.policy == "on_demand" and entry.uses == 0
        if move_on_demand:
            try:
                _move(entry.module, self.device)
            except Exception:
                _move(entry.module, torch.device("cpu"))
                if self.device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise
        entry.uses += 1
        try:
            yield entry.module
        finally:
            entry.uses = max(0, entry.uses - 1)
            if move_on_demand and entry.uses == 0:
                _move(entry.module, torch.device("cpu"))
                if self.device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def resident_bytes(self) -> int:
        if self.device.type == "cpu":
            return sum(_module_bytes(entry.module, self.device) for entry in self._entries.values())
        return sum(_module_bytes(entry.module, self.device) for entry in self._entries.values())

    def summary(self) -> dict[str, Any]:
        items: dict[str, Any] = {}
        for name, entry in self._entries.items():
            managed = _managed_module(entry.module)
            first = next(managed.parameters(), None) if isinstance(managed, nn.Module) else None
            current = str(first.device) if first is not None else "unknown"
            items[name] = {
                "policy": entry.policy,
                "device": current,
                "resident_bytes": _module_bytes(entry.module, self.device),
                "total_bytes": _module_bytes(entry.module),
            }
        return {
            "device": str(self.device),
            "resident_bytes": self.resident_bytes(),
            "models": items,
        }

    def to_cpu_all(self) -> None:
        for entry in self._entries.values():
            _move(entry.module, torch.device("cpu"))
            entry.uses = 0
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def report(self) -> dict[str, Any]:
        summary = self.summary()
        print(
            f">> Auxiliary residency: {summary['resident_bytes'] / 1024**3:.2f} GB on {self.device} | "
            + ", ".join(f"{name}={item['policy']}" for name, item in summary["models"].items())
        )
        return summary


__all__ = ["ResidencyManager", "ResidencyPolicy"]
