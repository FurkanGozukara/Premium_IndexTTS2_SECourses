"""Adapter discovery, injection, hot swapping, and removal."""

from __future__ import annotations

import os
import weakref
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import torch
from torch import nn

from .io import LoraFile, load_lora
from .layers import LoRAAdapter


_HANDLE_ATTRIBUTE = "_indextts_lora_handle"
_BACKUP_ATTRIBUTE = "_indextts_lora_full_backups"


@dataclass
class LoraHandle:
    path: str
    strength: float
    adapter_type: str
    rank: int
    alpha: float
    targets: list[str]
    _model_ref: weakref.ReferenceType[nn.Module] | None = field(
        default=None, repr=False, compare=False
    )
    _adapters: dict[str, LoRAAdapter] = field(default_factory=dict, repr=False, compare=False)


@dataclass
class _FullTensorBackup:
    module_path: str
    state_name: str
    tensor: torch.Tensor


def _get_submodule(model: nn.Module, path: str) -> nn.Module:
    if not path:
        return model
    try:
        return model.get_submodule(path)
    except (AttributeError, KeyError):
        current: nn.Module = model
        for component in path.split("."):
            if component in current._modules:
                current = current._modules[component]
            else:
                current = getattr(current, component)
        return current


def _replace_submodule(model: nn.Module, path: str, replacement: nn.Module) -> None:
    if not path:
        raise ValueError("cannot replace the root model with an adapter")
    parent_path, _, name = path.rpartition(".")
    parent = _get_submodule(model, parent_path)
    if name in parent._modules:
        parent._modules[name] = replacement
    else:
        setattr(parent, name, replacement)


def _find_adapters(model: nn.Module) -> dict[str, LoRAAdapter]:
    return {
        name: module
        for name, module in model.named_modules()
        if name and isinstance(module, LoRAAdapter)
    }


def list_target_modules(
    model: nn.Module, attention: bool = True, mlp: bool = True
) -> list[str]:
    """List supported GPT block projection paths on a UnifiedVoice-like model."""

    if not attention and not mlp:
        return []
    try:
        blocks = _get_submodule(model, "gpt.h")
    except (AttributeError, KeyError):
        return []

    paths: list[str] = []
    for index, block in enumerate(blocks):
        candidates: list[str] = []
        if attention:
            candidates.extend(("attn.c_attn", "attn.c_proj"))
        if mlp:
            candidates.extend(("mlp.c_fc", "mlp.c_proj"))
        for suffix in candidates:
            path = f"gpt.h.{index}.{suffix}"
            try:
                _get_submodule(model, path)
            except (AttributeError, KeyError):
                continue
            paths.append(path)
    return paths


def inject_adapters(
    model: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    use_dora: bool,
    target_modules: list[str],
) -> dict[str, LoRAAdapter]:
    """Replace the selected projection modules in place with adapters."""

    if len(set(target_modules)) != len(target_modules):
        raise ValueError("target_modules contains duplicate paths")
    adapters: dict[str, LoRAAdapter] = {}
    for path in target_modules:
        base = _get_submodule(model, path)
        if isinstance(base, LoRAAdapter):
            if (
                base.rank != int(rank)
                or base.use_dora != bool(use_dora)
                or base.in_features <= 0
                or base.out_features <= 0
            ):
                raise ValueError(f"target {path!r} already has an incompatible adapter")
            base.alpha = float(alpha)
            base.scaling = base.alpha / base.rank
            base.lora_dropout.p = float(dropout)
            base.invalidate_cache()
            adapters[path] = base
            continue

        adapter = LoRAAdapter(
            base=base,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            use_dora=use_dora,
            dtype=torch.float32,
        )
        adapter.train(base.training)
        _replace_submodule(model, path, adapter)
        adapters[path] = adapter
    return adapters


def _adapter_shapes_match(adapter: LoRAAdapter, loaded: LoraFile, path: str) -> bool:
    a_tensor = loaded.tensors.get(f"{path}.lora_A.weight")
    b_tensor = loaded.tensors.get(f"{path}.lora_B.weight")
    magnitude = loaded.tensors.get(f"{path}.lora_magnitude")
    return (
        a_tensor is not None
        and b_tensor is not None
        and tuple(a_tensor.shape) == tuple(adapter.lora_A.weight.shape)
        and tuple(b_tensor.shape) == tuple(adapter.lora_B.weight.shape)
        and (magnitude is not None) == adapter.use_dora
        and (
            magnitude is None
            or tuple(magnitude.shape) == tuple(adapter.lora_magnitude.shape)
        )
    )


@torch.no_grad()
def _copy_adapter_tensors(adapters: Mapping[str, LoRAAdapter], loaded: LoraFile) -> None:
    mismatches = [
        path
        for path, adapter in adapters.items()
        if not _adapter_shapes_match(adapter, loaded, path)
    ]
    if mismatches:
        raise ValueError(
            "saved tensors do not match adapter target(s): " + ", ".join(mismatches)
        )
    for path, adapter in adapters.items():
        if adapter._merged:
            adapter.unmerge_from_base()
        a_tensor = loaded.tensors[f"{path}.lora_A.weight"]
        b_tensor = loaded.tensors[f"{path}.lora_B.weight"]
        adapter.lora_A.weight.copy_(
            a_tensor.to(device=adapter.lora_A.weight.device, dtype=adapter.lora_A.weight.dtype)
        )
        adapter.lora_B.weight.copy_(
            b_tensor.to(device=adapter.lora_B.weight.device, dtype=adapter.lora_B.weight.dtype)
        )
        if adapter.use_dora:
            magnitude = loaded.tensors[f"{path}.lora_magnitude"]
            adapter.lora_magnitude.copy_(
                magnitude.to(
                    device=adapter.lora_magnitude.device,
                    dtype=adapter.lora_magnitude.dtype,
                )
            )
        adapter.alpha = float(loaded.alpha)
        adapter.scaling = adapter.alpha / adapter.rank
        adapter.lora_dropout.p = float(loaded.metadata.dropout)
        adapter.enabled = True
        adapter._merged = False
        adapter.invalidate_cache()


def _module_state_tensors(module: nn.Module) -> dict[str, torch.Tensor]:
    tensors = dict(module.named_parameters(recurse=True))
    tensors.update(dict(module.named_buffers(recurse=True)))
    return tensors


def _resolve_full_tensor(
    model: nn.Module, serialized_name: str
) -> tuple[str, nn.Module, str, torch.Tensor]:
    """Split ``module.path.state.name`` using the longest valid module prefix."""

    components = serialized_name.split(".")
    for split_at in range(len(components) - 1, 0, -1):
        module_path = ".".join(components[:split_at])
        state_name = ".".join(components[split_at:])
        try:
            module = _get_submodule(model, module_path)
        except (AttributeError, KeyError):
            continue
        state_tensors = _module_state_tensors(module)
        if state_name in state_tensors:
            return module_path, module, state_name, state_tensors[state_name]
    raise KeyError(f"cannot resolve full-module tensor {serialized_name!r} in model")


def _full_backups(model: nn.Module) -> dict[str, _FullTensorBackup]:
    backups = getattr(model, _BACKUP_ATTRIBUTE, None)
    if backups is None:
        backups = {}
        setattr(model, _BACKUP_ATTRIBUTE, backups)
    return backups


@torch.no_grad()
def _restore_full_tensors(model: nn.Module, *, clear: bool) -> None:
    backups: dict[str, _FullTensorBackup] | None = getattr(
        model, _BACKUP_ATTRIBUTE, None
    )
    if not backups:
        if clear and hasattr(model, _BACKUP_ATTRIBUTE):
            delattr(model, _BACKUP_ATTRIBUTE)
        return
    for backup in backups.values():
        module = _get_submodule(model, backup.module_path)
        destination = _module_state_tensors(module).get(backup.state_name)
        if destination is None:
            raise KeyError(
                f"cannot restore {backup.module_path}.{backup.state_name}; tensor no longer exists"
            )
        if tuple(destination.shape) != tuple(backup.tensor.shape):
            raise ValueError(
                f"cannot restore {backup.module_path}.{backup.state_name}; shape changed"
            )
        destination.copy_(
            backup.tensor.to(device=destination.device, dtype=destination.dtype)
        )
    if clear:
        delattr(model, _BACKUP_ATTRIBUTE)


@torch.no_grad()
def _apply_full_tensors(model: nn.Module, loaded: LoraFile) -> None:
    # A hot swap first removes the preceding file's full-module contribution.
    _restore_full_tensors(model, clear=False)
    full_tensors = {
        key[len("full.") :]: tensor
        for key, tensor in loaded.tensors.items()
        if key.startswith("full.")
    }
    if not full_tensors:
        return

    backups = _full_backups(model)
    for serialized_name, source in full_tensors.items():
        module_path, _, state_name, destination = _resolve_full_tensor(
            model, serialized_name
        )
        backup_key = f"{module_path}.{state_name}"
        if backup_key not in backups:
            backups[backup_key] = _FullTensorBackup(
                module_path=module_path,
                state_name=state_name,
                tensor=destination.detach().cpu().clone(),
            )
        if tuple(destination.shape) != tuple(source.shape):
            raise ValueError(
                f"full-module tensor {serialized_name!r} has shape {tuple(source.shape)}, "
                f"expected {tuple(destination.shape)}"
            )
        destination.copy_(source.to(device=destination.device, dtype=destination.dtype))


def _can_reuse(
    existing: Mapping[str, LoRAAdapter], loaded: LoraFile, use_dora: bool
) -> bool:
    if set(existing) != set(loaded.module_paths):
        return False
    return all(
        adapter.rank == loaded.rank
        and adapter.use_dora == use_dora
        and _adapter_shapes_match(adapter, loaded, path)
        for path, adapter in existing.items()
    )


def apply_lora(
    model: nn.Module, path: str, strength: float = 1.0
) -> LoraHandle:
    """Load an adapter file and apply it without reloading any base weights."""

    loaded = load_lora(path)
    use_dora = loaded.adapter_type == "dora"
    existing = _find_adapters(model)
    if not _can_reuse(existing, loaded, use_dora):
        if existing or get_lora_handle(model) is not None:
            remove_lora(model)
        existing = inject_adapters(
            model,
            rank=loaded.rank,
            alpha=loaded.alpha,
            dropout=loaded.metadata.dropout,
            use_dora=use_dora,
            target_modules=loaded.module_paths,
        )

    _copy_adapter_tensors(existing, loaded)
    _apply_full_tensors(model, loaded)
    for adapter in existing.values():
        adapter.strength = float(strength)

    old_handle = get_lora_handle(model)
    if old_handle is not None:
        old_handle._adapters = {}
    handle = LoraHandle(
        path=os.fspath(path),
        strength=float(strength),
        adapter_type=loaded.adapter_type,
        rank=loaded.rank,
        alpha=loaded.alpha,
        targets=list(loaded.module_paths),
        _model_ref=weakref.ref(model),
        _adapters=dict(existing),
    )
    setattr(model, _HANDLE_ATTRIBUTE, handle)
    return handle


def set_lora_strength(model_or_handle: nn.Module | LoraHandle, strength: float) -> None:
    value = float(strength)
    if isinstance(model_or_handle, LoraHandle):
        handle = model_or_handle
        adapters = handle._adapters
        model = handle._model_ref() if handle._model_ref is not None else None
    else:
        model = model_or_handle
        adapters = _find_adapters(model)
        handle = get_lora_handle(model)
    for adapter in adapters.values():
        if adapter._merged:
            adapter.unmerge_from_base()
        adapter.strength = value
    if handle is not None:
        handle.strength = value
    if model is not None:
        attached = get_lora_handle(model)
        if attached is not None:
            attached.strength = value


def move_adapters_to_device(
    model: nn.Module, device: str | torch.device
) -> dict[str, LoRAAdapter]:
    """Move only active adapter tensors, leaving frozen base residency unchanged."""

    target = torch.device(device)
    adapters = _find_adapters(model)
    for adapter in adapters.values():
        adapter.lora_A.to(target)
        adapter.lora_B.to(target)
        if adapter.lora_magnitude is not None:
            adapter.lora_magnitude.data = adapter.lora_magnitude.data.to(target)
        adapter.invalidate_cache()
    return adapters


def remove_lora(model: nn.Module) -> None:
    """Unwrap every adapter and restore all backed-up full-module tensors."""

    handle = get_lora_handle(model)
    adapters = _find_adapters(model)
    for path in sorted(adapters, key=lambda value: value.count("."), reverse=True):
        adapter = adapters[path]
        if adapter._merged:
            adapter.unmerge_from_base()
        adapter.restore_base_requires_grad()
        _replace_submodule(model, path, adapter.base)
    _restore_full_tensors(model, clear=True)
    if handle is not None:
        handle._adapters = {}
    if hasattr(model, _HANDLE_ATTRIBUTE):
        delattr(model, _HANDLE_ATTRIBUTE)


def get_lora_handle(model: nn.Module) -> LoraHandle | None:
    handle = getattr(model, _HANDLE_ATTRIBUTE, None)
    return handle if isinstance(handle, LoraHandle) else None


def _iter_full_modules(
    model: nn.Module, full_modules: Mapping[str, nn.Module] | Iterable[nn.Module | str]
) -> Iterable[nn.Module]:
    values: Iterable[nn.Module | str]
    if isinstance(full_modules, Mapping):
        values = full_modules.values()
    else:
        values = full_modules
    for value in values:
        if isinstance(value, str):
            yield _get_submodule(model, value)
        elif isinstance(value, nn.Module):
            yield value
        else:
            raise TypeError("full_modules values must be modules or module paths")


def trainable_parameters(
    model: nn.Module,
    adapters: Mapping[str, LoRAAdapter],
    full_modules: Mapping[str, nn.Module] | Iterable[nn.Module | str],
) -> list[nn.Parameter]:
    """Freeze the model, enable selected trainables, and return them once each."""

    model.requires_grad_(False)
    parameters: list[nn.Parameter] = []
    seen: set[int] = set()

    def add(parameter: nn.Parameter) -> None:
        parameter.requires_grad_(True)
        if id(parameter) not in seen:
            seen.add(id(parameter))
            parameters.append(parameter)

    for adapter in adapters.values():
        add(adapter.lora_A.weight)
        add(adapter.lora_B.weight)
        if adapter.lora_magnitude is not None:
            add(adapter.lora_magnitude)
    for module in _iter_full_modules(model, full_modules):
        for parameter in module.parameters():
            add(parameter)
    return parameters


def set_training_mode(model: nn.Module, training: bool) -> None:
    """Toggle adapter dropout and quantized-base STE behavior."""

    enabled = bool(training)
    for adapter in _find_adapters(model).values():
        adapter.train(enabled)
        if hasattr(adapter.base, "training_ste"):
            adapter.base.training_ste = enabled


def merge_lora_for_inference(model: nn.Module) -> None:
    """Merge active adapters while keeping their wrappers available to unmerge."""

    adapters = _find_adapters(model)
    int8_targets = [
        path
        for path, adapter in adapters.items()
        if callable(getattr(adapter.base, "dequantize_weight", None))
    ]
    if int8_targets:
        raise TypeError(
            "cannot merge LoRA into int8 bases: " + ", ".join(sorted(int8_targets))
        )
    for adapter in adapters.values():
        adapter.merge_into_base()


def unmerge_lora_from_model(model: nn.Module) -> None:
    """Restore the exact floating base weights for every temporarily merged adapter."""

    for adapter in _find_adapters(model).values():
        adapter.unmerge_from_base()


def merge_lora_into_model(model: nn.Module) -> None:
    """Merge adapters into floating bases, unwrap them, and retain full tensors."""

    adapters = _find_adapters(model)
    merge_lora_for_inference(model)
    for path in sorted(adapters, key=lambda value: value.count("."), reverse=True):
        adapter = adapters[path]
        adapter.restore_base_requires_grad()
        _replace_submodule(model, path, adapter.base)

    handle = get_lora_handle(model)
    if handle is not None:
        handle._adapters = {}
    if hasattr(model, _HANDLE_ATTRIBUTE):
        delattr(model, _HANDLE_ATTRIBUTE)
    # A merge makes the current full-module values part of the exported model.
    if hasattr(model, _BACKUP_ATTRIBUTE):
        delattr(model, _BACKUP_ATTRIBUTE)


__all__ = [
    "LoraHandle",
    "apply_lora",
    "get_lora_handle",
    "inject_adapters",
    "list_target_modules",
    "merge_lora_for_inference",
    "merge_lora_into_model",
    "move_adapters_to_device",
    "remove_lora",
    "set_lora_strength",
    "set_training_mode",
    "trainable_parameters",
    "unmerge_lora_from_model",
]
