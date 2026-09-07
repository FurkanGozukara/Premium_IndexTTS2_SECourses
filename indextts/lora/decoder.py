"""Voice decoder (s2mel) adapters: file discovery, component tags, and LoRA / DoRA target modules.

A voice decoder adapter is a LoRA / DoRA on the semantic-to-mel flow-matching DiT. It is saved next to
the GPT adapter of the same training as ``<name>.s2mel.safetensors`` and is loaded automatically whenever
that GPT adapter is selected. The safetensors metadata marks it with ``train_config.component = "s2mel"``
so the two kinds of file are never applied to the wrong model.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from torch import nn

DECODER_ADAPTER_SUFFIX = ".s2mel.safetensors"
DECODER_COMPONENT = "s2mel"
GPT_COMPONENT = "gpt"


def is_decoder_adapter_path(path: str | os.PathLike[str] | None) -> bool:
    return str(path or "").lower().endswith(DECODER_ADAPTER_SUFFIX)


def adapter_root(gpt_adapter_path: str | os.PathLike[str]) -> Path:
    """The training folder of a GPT adapter file: its parent, or the grandparent of a ``best/`` file."""
    source = Path(gpt_adapter_path).expanduser().resolve()
    parent = source.parent
    return parent.parent if parent.name.lower() == "best" else parent


def decoder_adapter_candidates(gpt_adapter_path: str | os.PathLike[str]) -> list[Path]:
    """Where a decoder adapter for this GPT adapter file may live, most specific first."""
    source = Path(gpt_adapter_path).expanduser().resolve()
    root = adapter_root(source)
    candidates = [
        source.with_name(source.stem + DECODER_ADAPTER_SUFFIX),
        root / (root.name + DECODER_ADAPTER_SUFFIX),
    ]
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def find_decoder_adapter(gpt_adapter_path: str | os.PathLike[str] | None) -> str:
    """Return the decoder adapter file that belongs to a GPT adapter file, or an empty string.

    Every checkpoint of one training (epoch files, the final file, and ``best/``) shares the decoder
    adapter saved in the training folder, because the decoder is trained from the dataset rather than from
    a particular GPT checkpoint.
    """
    if not gpt_adapter_path or is_decoder_adapter_path(gpt_adapter_path):
        return ""
    source = Path(gpt_adapter_path).expanduser()
    if not source.is_file():
        return ""
    for candidate in decoder_adapter_candidates(source):
        if candidate.is_file():
            return str(candidate)
    root = adapter_root(source)
    singles = sorted(path for path in root.glob("*" + DECODER_ADAPTER_SUFFIX) if path.is_file())
    if len(singles) == 1:
        return str(singles[0])
    return ""


def component_of_metadata(train_config: Any, path: str | os.PathLike[str] | None = None) -> str:
    """``"s2mel"`` for decoder adapters, ``"gpt"`` otherwise (file suffix decides for files without a tag)."""
    if isinstance(train_config, str):
        try:
            train_config = json.loads(train_config)
        except (TypeError, ValueError):
            train_config = {}
    if isinstance(train_config, dict):
        component = str(train_config.get("component") or "").strip().lower()
        if component in {DECODER_COMPONENT, GPT_COMPONENT}:
            return component
    return DECODER_COMPONENT if is_decoder_adapter_path(path) else GPT_COMPONENT


def lora_component(path: str | os.PathLike[str]) -> str:
    """Read the component tag of a LoRA / DoRA file without loading its tensors."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        header = handle.metadata() or {}
    return component_of_metadata(header.get("train_config"), path)


def unwrap_estimator(estimator: nn.Module) -> nn.Module:
    """The DiT module itself, even when ``torch.compile`` wrapped it."""
    return getattr(estimator, "_orig_mod", estimator)


def decoder_target_modules(estimator: nn.Module, attention: bool = True, mlp: bool = True) -> list[str]:
    """Projection paths inside the DiT transformer blocks that accept a LoRA / DoRA."""
    estimator = unwrap_estimator(estimator)
    transformer = getattr(estimator, "transformer", None)
    layers = getattr(transformer, "layers", None)
    if layers is None:
        return []
    names: list[str] = []
    if attention:
        names.extend(("attention.wqkv", "attention.wq", "attention.wkv", "attention.wo"))
    if mlp:
        names.extend(("feed_forward.w1", "feed_forward.w2", "feed_forward.w3"))
    paths: list[str] = []
    for index, block in enumerate(layers):
        for name in names:
            module: Any = block
            for part in name.split("."):
                module = getattr(module, part, None)
                if module is None:
                    break
            if isinstance(module, nn.Linear):
                paths.append(f"transformer.layers.{index}.{name}")
    return paths


__all__ = [
    "DECODER_ADAPTER_SUFFIX",
    "DECODER_COMPONENT",
    "GPT_COMPONENT",
    "adapter_root",
    "component_of_metadata",
    "decoder_adapter_candidates",
    "decoder_target_modules",
    "find_decoder_adapter",
    "is_decoder_adapter_path",
    "lora_component",
    "unwrap_estimator",
]


DECODER_CHOICE_AUTO = "auto"
DECODER_CHOICE_NONE = "none"


def decoder_adapter_selection(choice: object) -> tuple[bool, str]:
    """Map the generation tab's "Voice decoder adapter" choice to (use_decoder_adapter, explicit path).

    ``auto`` (or empty) applies the file saved with the selected LoRA / DoRA, ``none`` applies no decoder
    adapter, and anything else is an explicit decoder adapter file.
    """
    value = str(choice or "").strip()
    if value.lower() == DECODER_CHOICE_NONE:
        return False, ""
    if not value or value.lower() == DECODER_CHOICE_AUTO:
        return True, ""
    return True, str(Path(value).expanduser().resolve())


def decoder_adapter_choices(lora_path: str | os.PathLike[str] | None, loras_root: str | os.PathLike[str] | None = None) -> list[tuple[str, str]]:
    """Dropdown choices: the selected LoRA / DoRA's own adapter (automatic), none, then every other decoder file."""
    found = find_decoder_adapter(lora_path) if lora_path else ""
    if found:
        auto_label = f"Automatic: {Path(found).name} (saved with this LoRA / DoRA)"
    elif lora_path:
        auto_label = "Automatic: none saved with this LoRA / DoRA"
    else:
        auto_label = "Automatic: the selected LoRA / DoRA's own adapter"
    choices: list[tuple[str, str]] = [(auto_label, DECODER_CHOICE_AUTO), ("None (GPT adapter only)", DECODER_CHOICE_NONE)]
    root = Path(loras_root) if loras_root else None
    if root is not None and root.is_dir():
        found_key = str(Path(found).resolve()) if found else ""
        for file in sorted(root.rglob(f"*{DECODER_ADAPTER_SUFFIX}"), key=lambda item: str(item).lower()):
            resolved = str(file.resolve())
            if resolved == found_key:
                continue
            try:
                label = str(file.resolve().relative_to(root.resolve()))
            except ValueError:
                label = file.name
            choices.append((label, resolved))
    return choices


def recommended_decoder_strength(gpt_adapter_path: str | os.PathLike[str] | None) -> float | None:
    """The decoder strength chosen by the training's full-pipeline test, or None when it was never measured."""
    if not gpt_adapter_path or not find_decoder_adapter(gpt_adapter_path):
        return None
    report_path = adapter_root(gpt_adapter_path) / "analysis" / "speech_evaluation" / "decoder_test" / "report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict) or not report.get("accepted"):
            return None
        value = float(report.get("strength", 1.0))
    except (OSError, ValueError, TypeError):
        return None
    return value if 0.0 < value <= 4.0 else None
