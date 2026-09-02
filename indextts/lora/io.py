"""Safetensors persistence and discovery for IndexTTS LoRA adapters."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

from .layers import LoRAAdapter


LORA_FORMAT = "indextts2_premium_lora"
LORA_VERSION = "1"


def _json_load(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


def _as_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return fallback


def _as_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return fallback


@dataclass
class LoraMetadata:
    format: str = LORA_FORMAT
    version: str = LORA_VERSION
    adapter_type: str = "lora"
    rank: int = 0
    alpha: float = 0.0
    dropout: float = 0.0
    target_modules: list[str] = field(default_factory=list)
    base_model: str = "IndexTeam/IndexTTS-2.5"
    base_variant: str = "bf16"
    trained_steps: int = 0
    epochs: int = 0
    dataset_name: str = ""
    created_at: str = ""
    app_version: str = ""
    train_config: dict[str, Any] = field(default_factory=dict)
    recommended_reference: str = ""
    sample_rate: int = 24000

    def to_header(self) -> dict[str, str]:
        """Convert metadata to the string-only safetensors header contract."""

        return {
            "format": str(self.format),
            "version": str(self.version),
            "adapter_type": str(self.adapter_type).lower(),
            "rank": str(int(self.rank)),
            "alpha": str(float(self.alpha)),
            "dropout": str(float(self.dropout)),
            "target_modules": json.dumps(list(self.target_modules), ensure_ascii=True),
            "base_model": str(self.base_model),
            "base_variant": str(self.base_variant),
            "trained_steps": str(int(self.trained_steps)),
            "epochs": str(int(self.epochs)),
            "dataset_name": str(self.dataset_name),
            "created_at": str(self.created_at),
            "app_version": str(self.app_version),
            "train_config": json.dumps(self.train_config, ensure_ascii=True, sort_keys=True),
            "recommended_reference": str(self.recommended_reference),
            "sample_rate": str(int(self.sample_rate)),
        }

    @classmethod
    def from_header(cls, header: Mapping[str, str] | None) -> LoraMetadata:
        values = dict(header or {})
        targets = _json_load(values.get("target_modules"), [])
        if isinstance(targets, str):
            targets = [targets]
        if not isinstance(targets, list):
            targets = []
        config = _json_load(values.get("train_config"), {})
        if not isinstance(config, dict):
            config = {}
        return cls(
            format=values.get("format", LORA_FORMAT),
            version=values.get("version", LORA_VERSION),
            adapter_type=values.get("adapter_type", "lora").lower(),
            rank=_as_int(values.get("rank"), 0),
            alpha=_as_float(values.get("alpha"), 0.0),
            dropout=_as_float(values.get("dropout"), 0.0),
            target_modules=[str(target) for target in targets],
            base_model=values.get("base_model", "IndexTeam/IndexTTS-2.5"),
            base_variant=values.get("base_variant", "bf16"),
            trained_steps=_as_int(values.get("trained_steps"), 0),
            epochs=_as_int(values.get("epochs"), 0),
            dataset_name=values.get("dataset_name", ""),
            created_at=values.get("created_at", ""),
            app_version=values.get("app_version", ""),
            train_config=config,
            recommended_reference=values.get("recommended_reference", ""),
            sample_rate=_as_int(values.get("sample_rate"), 24000),
        )


@dataclass
class LoraFile:
    tensors: dict[str, torch.Tensor]
    metadata: LoraMetadata
    adapter_type: str
    rank: int
    alpha: float
    target_modules: list[str]
    module_paths: list[str]
    has_full: bool


@dataclass
class LoraEntry:
    name: str
    path: str
    relative_label: str
    metadata_summary: str


@dataclass
class _LoraStructure:
    metadata: LoraMetadata
    adapter_type: str
    rank: int
    alpha: float
    target_modules: list[str]
    module_paths: list[str]
    has_full: bool


def _coerce_metadata(metadata: LoraMetadata | Mapping[str, Any]) -> LoraMetadata:
    if isinstance(metadata, LoraMetadata):
        return metadata
    if isinstance(metadata, Mapping):
        string_header = {str(key): str(value) for key, value in metadata.items()}
        # Preserve structured values supplied by Python callers.
        if isinstance(metadata.get("target_modules"), (list, tuple)):
            string_header["target_modules"] = json.dumps(metadata["target_modules"])
        if isinstance(metadata.get("train_config"), Mapping):
            string_header["train_config"] = json.dumps(metadata["train_config"])
        return LoraMetadata.from_header(string_header)
    raise TypeError("metadata must be LoraMetadata or a mapping")


def _cast_for_save(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    saved = tensor.detach().to(device="cpu")
    if saved.is_floating_point():
        saved = saved.to(dtype=dtype)
    return saved.contiguous().clone()


def save_lora(
    path: str | os.PathLike[str],
    adapters: dict[str, LoRAAdapter],
    full_modules: dict[str, nn.Module],
    metadata: LoraMetadata | Mapping[str, Any],
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Atomically save adapters and optional fully-trained small modules."""

    if dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("LoRA safetensors dtype must be torch.bfloat16 or torch.float32")
    if not adapters:
        raise ValueError("at least one adapter is required")

    tensors: dict[str, torch.Tensor] = {}
    for module_path, adapter in adapters.items():
        if not module_path:
            raise ValueError("adapter module paths cannot be empty")
        tensors[f"{module_path}.lora_A.weight"] = _cast_for_save(
            adapter.lora_A.weight, dtype
        )
        tensors[f"{module_path}.lora_B.weight"] = _cast_for_save(
            adapter.lora_B.weight, dtype
        )
        if adapter.use_dora:
            tensors[f"{module_path}.lora_magnitude"] = _cast_for_save(
                adapter.lora_magnitude, dtype
            )

    for module_path, module in full_modules.items():
        if not module_path:
            raise ValueError("full-module paths cannot be empty")
        for state_name, tensor in module.state_dict().items():
            tensors[f"full.{module_path}.{state_name}"] = _cast_for_save(tensor, dtype)

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_value = _coerce_metadata(metadata)
    ranks = {adapter.rank for adapter in adapters.values()}
    alphas = {adapter.alpha for adapter in adapters.values()}
    adapter_types = {"dora" if adapter.use_dora else "lora" for adapter in adapters.values()}
    if len(ranks) != 1 or len(alphas) != 1 or len(adapter_types) != 1:
        raise ValueError("one LoRA file cannot contain mixed ranks, alphas, or adapter types")
    actual_rank = next(iter(ranks))
    actual_alpha = next(iter(alphas))
    actual_type = next(iter(adapter_types))
    if metadata_value.rank not in (0, actual_rank):
        raise ValueError("metadata rank disagrees with the adapters being saved")
    if metadata_value.alpha > 0.0 and metadata_value.alpha != actual_alpha:
        raise ValueError("metadata alpha disagrees with the adapters being saved")
    metadata_value = replace(
        metadata_value,
        rank=actual_rank,
        alpha=actual_alpha,
        adapter_type=actual_type,
        target_modules=list(metadata_value.target_modules) or list(adapters),
    )
    metadata_header = metadata_value.to_header()
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(destination.parent),
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    try:
        save_file(tensors, temporary_name, metadata=metadata_header)
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


_PEFT_PREFIX = re.compile(r"^(?:base_model\.model\.)+")
_LORA_A_SUFFIX = re.compile(r"\.lora_A\.(?:default\.)?weight$")
_LORA_B_SUFFIX = re.compile(r"\.lora_B\.(?:default\.)?weight$")
_DORA_SUFFIX = re.compile(r"\.lora_magnitude_vector\.(?:default\.)?weight$")


def _normalize_tensor_key(key: str) -> str:
    if key.startswith("full."):
        return key
    normalized = _PEFT_PREFIX.sub("", key)
    normalized = _LORA_A_SUFFIX.sub(".lora_A.weight", normalized)
    normalized = _LORA_B_SUFFIX.sub(".lora_B.weight", normalized)
    normalized = _DORA_SUFFIX.sub(".lora_magnitude", normalized)
    return normalized


def _natural_key(value: str):
    return [
        int(piece) if piece.isdigit() else piece.lower()
        for piece in re.split(r"(\d+)", value)
    ]


def _analyze_lora_structure(
    source: Path,
    header: Mapping[str, str],
    shapes: Mapping[str, tuple[int, ...]],
) -> _LoraStructure:
    module_paths = {
        key[: -len(".lora_A.weight")]
        for key in shapes
        if key.endswith(".lora_A.weight")
    }
    if not module_paths:
        raise ValueError(f"{source} does not contain LoRA adapter tensors")

    inferred_ranks: set[int] = set()
    magnitude_paths: set[str] = set()
    for module_path in module_paths:
        a_key = f"{module_path}.lora_A.weight"
        b_key = f"{module_path}.lora_B.weight"
        if b_key not in shapes:
            raise ValueError(f"adapter {module_path!r} is missing lora_B.weight")
        a_shape = shapes[a_key]
        b_shape = shapes[b_key]
        if len(a_shape) != 2 or len(b_shape) != 2:
            raise ValueError(f"adapter {module_path!r} A/B tensors must be two-dimensional")
        if a_shape[0] != b_shape[1]:
            raise ValueError(f"adapter {module_path!r} has incompatible A/B shapes")
        inferred_ranks.add(int(a_shape[0]))
        magnitude_key = f"{module_path}.lora_magnitude"
        if magnitude_key in shapes:
            if shapes[magnitude_key] != (b_shape[0],):
                raise ValueError(
                    f"adapter {module_path!r} has an incompatible DoRA magnitude shape"
                )
            magnitude_paths.add(module_path)
    if len(inferred_ranks) != 1:
        raise ValueError(f"adapter file contains mixed ranks: {sorted(inferred_ranks)}")
    inferred_rank = next(iter(inferred_ranks))

    metadata = LoraMetadata.from_header(header)
    rank = metadata.rank if metadata.rank > 0 else inferred_rank
    if rank != inferred_rank:
        raise ValueError(
            f"metadata rank {rank} disagrees with adapter tensor rank {inferred_rank}"
        )
    adapter_type = (
        metadata.adapter_type if metadata.adapter_type in {"lora", "dora"} else ""
    )
    if magnitude_paths:
        adapter_type = "dora"
    elif not adapter_type:
        adapter_type = "lora"
    if adapter_type == "dora" and magnitude_paths != module_paths:
        missing = sorted(module_paths - magnitude_paths, key=_natural_key)
        raise ValueError(f"DoRA magnitude tensors are missing for: {', '.join(missing)}")
    try:
        alpha = float(header["alpha"]) if "alpha" in header else float(rank)
    except (TypeError, ValueError, OverflowError):
        alpha = float(rank)
    if not math.isfinite(alpha):
        alpha = float(rank)

    ordered_paths = sorted(module_paths, key=_natural_key)
    targets = list(metadata.target_modules) or ordered_paths
    metadata = replace(
        metadata,
        adapter_type=adapter_type,
        rank=rank,
        alpha=alpha,
        target_modules=list(targets),
    )
    return _LoraStructure(
        metadata=metadata,
        adapter_type=adapter_type,
        rank=rank,
        alpha=alpha,
        target_modules=targets,
        module_paths=ordered_paths,
        has_full=any(key.startswith("full.") for key in shapes),
    )


def _read_lora_structure(source: Path) -> _LoraStructure:
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        header = handle.metadata() or {}
        shapes: dict[str, tuple[int, ...]] = {}
        for raw_key in handle.keys():
            key = _normalize_tensor_key(raw_key)
            if key in shapes:
                raise ValueError(f"duplicate tensor key after PEFT normalization: {key}")
            shapes[key] = tuple(handle.get_slice(raw_key).get_shape())
    return _analyze_lora_structure(source, header, shapes)


def load_lora(path: str | os.PathLike[str]) -> LoraFile:
    """Load and normalize an IndexTTS or PEFT-style adapter file on CPU."""

    source = Path(path)
    structure = _read_lora_structure(source)
    raw_tensors = load_file(str(source), device="cpu")
    tensors: dict[str, torch.Tensor] = {}
    for raw_key, tensor in raw_tensors.items():
        key = _normalize_tensor_key(raw_key)
        if key in tensors:
            raise ValueError(f"duplicate tensor key after PEFT normalization: {key}")
        tensors[key] = tensor

    return LoraFile(
        tensors=tensors,
        metadata=structure.metadata,
        adapter_type=structure.adapter_type,
        rank=structure.rank,
        alpha=structure.alpha,
        target_modules=structure.target_modules,
        module_paths=structure.module_paths,
        has_full=structure.has_full,
    )


def inspect_lora(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return the compact adapter information used by the model manager UI."""

    source = Path(path)
    structure = _read_lora_structure(source)
    metadata = structure.metadata

    reference = ""
    if metadata.recommended_reference:
        configured = Path(metadata.recommended_reference)
        if not configured.is_absolute():
            configured = source.parent / configured
        reference = str(configured)
    else:
        sibling = source.with_name(f"{source.stem}_reference.wav")
        if sibling.is_file():
            reference = str(sibling)

    created = metadata.created_at
    if not created:
        created = datetime.fromtimestamp(source.stat().st_mtime).astimezone().isoformat()
    return {
        "adapter_type": structure.adapter_type,
        "rank": structure.rank,
        "alpha": structure.alpha,
        "targets": list(structure.target_modules),
        "steps": metadata.trained_steps,
        "dataset": metadata.dataset_name,
        "date": created,
        "size_mb": round(source.stat().st_size / (1024 * 1024), 3),
        "recommended_reference": reference,
    }


def _metadata_summary(info: Mapping[str, Any]) -> str:
    parts = [str(info["adapter_type"]).upper(), f"r{info['rank']}"]
    if info.get("steps"):
        parts.append(f"{info['steps']} steps")
    if info.get("dataset"):
        parts.append(str(info["dataset"]))
    return " | ".join(parts)


def scan_lora_files(root_dirs: list[str]) -> list[LoraEntry]:
    """Recursively find valid LoRA safetensors under one or more roots."""

    entries: list[LoraEntry] = []
    seen: set[str] = set()
    for root_value in root_dirs:
        root = Path(root_value)
        if not root.is_dir():
            continue
        for candidate in root.rglob("*"):
            if not candidate.is_file() or candidate.suffix.lower() != ".safetensors":
                continue
            canonical = os.path.normcase(str(candidate.resolve()))
            if canonical in seen:
                continue
            try:
                info = inspect_lora(candidate)
            except Exception:
                continue
            seen.add(canonical)
            relative = candidate.relative_to(root).as_posix()
            label = f"{root.name}/{relative}" if root.name else relative
            entries.append(
                LoraEntry(
                    name=candidate.stem,
                    path=str(candidate),
                    relative_label=label,
                    metadata_summary=_metadata_summary(info),
                )
            )
    entries.sort(key=lambda entry: (entry.relative_label.lower(), entry.path.lower()))
    return entries


def resume_state_path_for(lora_path: str | os.PathLike[str]) -> str:
    source = Path(lora_path)
    return str(source.with_name(f"{source.stem}.train_state.pt"))


def save_train_state(path: str | os.PathLike[str], state_dict: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(destination.parent),
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    try:
        torch.save(dict(state_dict), temporary_name)
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_train_state(path: str | os.PathLike[str]) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise TypeError("training state must contain a dictionary")
    return state


__all__ = [
    "LORA_FORMAT",
    "LORA_VERSION",
    "LoraEntry",
    "LoraFile",
    "LoraMetadata",
    "inspect_lora",
    "load_lora",
    "load_train_state",
    "resume_state_path_for",
    "save_lora",
    "save_train_state",
    "scan_lora_files",
]
