"""INT8 ConvRot versions of full fine-tuned speech models.

A full fine-tuning checkpoint (``adapter_type`` "full") stores every trained module as ``full.<module>.<tensor>``.
``export_int8_finetune`` merges those tensors into the base weights and converts the result with the converter of
the public INT8 base models, writing ``<checkpoint>.int8_convrot.safetensors`` (and its ``.report.json``) beside
the checkpoint:

* IndexTTS: a complete GPT checkpoint in the same format as ``models/gpt_int8_convrot.safetensors``.
* OmniVoice: the transformer in the same format as the public OmniVoice INT8 file, plus the fine-tuned audio
  embeddings and heads in BF16 under the ``omnivoice.`` prefix.

The header keeps the checkpoint's training metadata (base model, steps, dataset, reference) with
``base_variant`` ``int8_convrot`` and an ``indextts_int8_finetune`` record, so adapter lists show the file as the
INT8 version of that fine-tuned model. Selecting it loads the fine-tuned model itself in INT8; it is a model, not
an adapter, so changing to or from it reloads the speech model.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

FINETUNE_KEY = "indextts_int8_finetune"
OMNIVOICE_EXTRA_PREFIX = "omnivoice."
SUFFIX = ".int8_convrot.safetensors"
OMNIVOICE_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def int8_finetune_path(checkpoint: str | os.PathLike[str]) -> Path:
    source = Path(checkpoint)
    return source.with_name(source.name[: -len(".safetensors")] + SUFFIX if source.name.endswith(".safetensors")
                            else source.name + SUFFIX)


@lru_cache(maxsize=1024)
def _header_record(path: str, modified_ns: int, size: int) -> dict[str, Any] | None:
    del modified_ns, size  # cache key only
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
    except Exception:
        return None
    raw = metadata.get(FINETUNE_KEY)
    if not raw:
        return None
    try:
        record = json.loads(raw)
    except ValueError:
        return None
    return {**record, "base_model": metadata.get("base_model", ""), "adapter_type": metadata.get("adapter_type", "full")}


def int8_finetune_info(path: str | os.PathLike[str] | None) -> dict[str, Any] | None:
    """The export record of an INT8 fine-tuned model file, or ``None`` for any other file."""

    if not path or not str(path).lower().endswith(".safetensors"):
        return None
    source = Path(path)
    try:
        stat = source.stat()
    except OSError:
        return None
    return _header_record(str(source.resolve()), stat.st_mtime_ns, stat.st_size)


def is_int8_finetune(path: str | os.PathLike[str] | None) -> bool:
    return int8_finetune_info(path) is not None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tuned_tensors(checkpoint: Path) -> tuple[dict[str, str], OrderedDict]:
    from safetensors import safe_open

    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        header = dict(handle.metadata() or {})
        tensors = OrderedDict((key[len("full."):], handle.get_tensor(key)) for key in handle.keys() if key.startswith("full."))
    if str(header.get("adapter_type", "")).lower() != "full" or not tensors:
        raise ValueError(f"{checkpoint.name} is not a full fine-tuning checkpoint; INT8 export applies to full fine-tunes.")
    return header, tensors


def export_int8_finetune(checkpoint: str | os.PathLike[str], *, model_dir: str | os.PathLike[str] = "models",
                         model_config: str | os.PathLike[str] | None = None, device: str = "cuda:0",
                         output: str | os.PathLike[str] | None = None,
                         progress: Callable[[str], Any] | None = print) -> dict[str, Any]:
    """Write the INT8 ConvRot version of a full fine-tuning checkpoint; returns the converter's report."""

    import torch

    from indextts.quant.convrot_int8 import _torch_load_state_dict, convert_state_dict

    source = Path(checkpoint).expanduser().resolve()
    destination = Path(output).expanduser().resolve() if output else int8_finetune_path(source)
    header, tuned = _tuned_tensors(source)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    record = {"version": 1, "source": source.name, "source_sha256": _sha256(source),
              "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    metadata = {**header, "base_variant": "int8_convrot", FINETUNE_KEY: json.dumps(record)}
    base_model = str(header.get("base_model", ""))
    if "omnivoice" in base_model.lower():
        from safetensors.torch import load_file

        from indextts.backends.omnivoice import MODEL_REPO, ensure_model

        folder, _ = ensure_model(model_dir)
        base = load_file(str(folder / "model.safetensors"), device="cpu")
        llm = OrderedDict((key[len("llm."):], value) for key, value in base.items() if key.startswith("llm."))
        extras = OrderedDict()
        for key, value in tuned.items():
            if key.startswith("llm."):
                llm[key[len("llm."):]] = value
            else:
                extras[OMNIVOICE_EXTRA_PREFIX + key] = value
        targets = [key[: -len(".weight")] for key in llm if key.startswith("layers.") and key.endswith(".weight")
                   and key[: -len(".weight")].rsplit(".", 1)[-1] in OMNIVOICE_PROJECTIONS]
        state = OrderedDict([*llm.items(), *extras.items()])
        model_id = MODEL_REPO
    else:
        from omegaconf import OmegaConf

        root = Path(model_dir)
        config = OmegaConf.load(str(model_config or root / "config.yaml"))
        state = OrderedDict(_torch_load_state_dict(root / str(config.gpt_checkpoint)))
        added = [key for key in tuned if key not in state]
        if added and progress:
            progress(f">> {len(added)} fine-tuned tensors are new to the base checkpoint (kept as trained)")
        state.update(tuned)
        targets = None  # the IndexTTS GPT layers the public INT8 base quantizes
        model_id = "IndexTeam/IndexTTS-2.5"
    if progress:
        progress(f">> Writing the INT8 ConvRot version of {source.name}")
    report = convert_state_dict(state, str(destination), device=device, linear_targets=targets, model_id=model_id,
                                source_name=source.name, source_path=str(source), source_bytes=source.stat().st_size,
                                extra_metadata=metadata, progress=progress)
    _header_record.cache_clear()
    return report


__all__ = ["FINETUNE_KEY", "OMNIVOICE_EXTRA_PREFIX", "export_int8_finetune", "int8_finetune_info",
           "int8_finetune_path", "is_int8_finetune"]
