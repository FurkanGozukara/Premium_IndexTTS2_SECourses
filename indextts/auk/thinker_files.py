"""The Qwen2.5-Omni files AuK's text and audio encoder loads.

AuK conditions on the Qwen2.5-Omni-3B Thinker: its text model and audio tower.
The public snapshot (``models/qwen2_5_omni_3b``, 12 GB) also carries the vision
tower, the Thinker's vocabulary head and the speech Talker and Token2Wav, which
AuK never runs. The slim folder (``models/quantized/AuK/qwen2_5_omni_thinker``,
7.5 GB) holds only the Thinker weights AuK runs, a Thinker config and the
tokenizer, processor and chat-template files; ``Qwen2_5OmniThinkerForConditionalGeneration``
and ``Qwen2_5OmniProcessor`` load it with ``from_pretrained`` exactly like the
snapshot and give bitwise-identical hidden states.

The slim config declares the vision tower with no transformer blocks (its small
patch embedding and merger, 76 MB, are kept) and ties the vocabulary head to the
token embeddings, so loading reports no missing or unexpected weights and never
randomly initialises the 1.9 GB AuK discards (AuK deletes the vision tower and
replaces the head with a hidden-states stub right after loading).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import struct
import time
from pathlib import Path
from typing import Any, Callable

FULL_FOLDER = "qwen2_5_omni_3b"
SLIM_FOLDER = Path("quantized") / "AuK" / "qwen2_5_omni_thinker"
# What Qwen2_5OmniProcessor reads: tokenizer, chat template, audio feature extractor
# and the (unused) image processor settings that share preprocessor_config.json.
PROCESSOR_FILES = ("added_tokens.json", "chat_template.json", "merges.txt", "preprocessor_config.json",
                   "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json")
NOTICE_FILES = ("LICENSE",)
# Thinker weights AuK runs, as stored in the public snapshot, plus the vision tower's
# block-free remainder, which the Thinker class always builds.
KEPT_PREFIXES = ("thinker.model.", "thinker.audio_tower.", "thinker.visual.patch_embed.", "thinker.visual.merger.")
WEIGHTS_INDEX = "model.safetensors.index.json"
SLIM_MARKER = "auk_thinker_slim"
SLIM_VERSION = 1


def full_folder(model_dir: str | os.PathLike) -> Path:
    return Path(model_dir) / FULL_FOLDER


def slim_folder(model_dir: str | os.PathLike) -> Path:
    return Path(model_dir) / SLIM_FOLDER


def _safetensors_complete(path: Path) -> bool:
    """A cheap truncation check: the file is exactly its header plus the data it declares."""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            if length <= 0 or 8 + length > size:
                return False
            header = json.loads(handle.read(length))
    except (OSError, ValueError, struct.error):
        return False
    end = max((int(item["data_offsets"][1]) for key, item in header.items() if key != "__metadata__"), default=0)
    return size == 8 + length + end


def weight_files(folder: str | os.PathLike) -> list[Path]:
    """The safetensors files a Hugging Face folder's index names, or its single ``model.safetensors``."""
    folder = Path(folder)
    index = folder / WEIGHTS_INDEX
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
        return [folder / name for name in sorted(set(weight_map.values()))]
    single = folder / "model.safetensors"
    return [single] if single.is_file() else []


def is_complete(folder: str | os.PathLike) -> bool:
    """True when the folder has its config, every processor file and untruncated weights."""
    folder = Path(folder)
    if not (folder / "config.json").is_file() or not all((folder / name).is_file() for name in PROCESSOR_FILES):
        return False
    try:
        files = weight_files(folder)
    except (OSError, ValueError):
        return False
    return bool(files) and all(path.is_file() and _safetensors_complete(path) for path in files)


def thinker_folder(model_dir: str | os.PathLike) -> Path:
    """The folder AuK loads its Qwen2.5-Omni Thinker from: the slim one when complete, else the snapshot."""
    slim = slim_folder(model_dir)
    return slim if is_complete(slim) else full_folder(model_dir)


def load_int8_text_model(text_model, path: str | os.PathLike, *, device, strict: bool = True):
    """Load the ConvRot INT8 file into the Thinker text model (``thinker.model``).

    ``load_gpt_checkpoint`` ends with ``model.to(dtype)``, which would also round the
    text model's non-persistent float32 rotary frequencies (``inv_freq``,
    ``original_inv_freq``) to BF16; the BF16 Thinker keeps them float32, so they are
    restored here. Loading on the CPU and moving the module afterwards keeps the
    GPU from ever holding the BF16 text weights.
    """
    import torch

    from indextts.quant.convrot_int8 import load_gpt_checkpoint

    persistent = set(text_model.state_dict().keys())
    kept = {name: buffer.detach().clone() for name, buffer in text_model.named_buffers()
            if name not in persistent and buffer.dtype == torch.float32}
    report = load_gpt_checkpoint(text_model, str(path), device=device, dtype=torch.bfloat16, strict=strict)
    for name, value in kept.items():
        owner_path, _, leaf = name.rpartition(".")
        owner = text_model.get_submodule(owner_path) if owner_path else text_model
        owner._buffers[leaf] = value.to(device)
    return report


def slim_config(full_config: dict) -> dict:
    """The Thinker section of the Omni config, loadable as ``Qwen2_5OmniThinkerConfig``.

    The vision tower keeps its settings but declares no transformer blocks, and the
    head is tied to the embeddings: neither has weights in the slim folder, and
    hidden states never pass through either.
    """
    thinker = json.loads(json.dumps(full_config.get("thinker_config", full_config)))
    thinker["model_type"] = "qwen2_5_omni_thinker"
    thinker["architectures"] = ["Qwen2_5OmniThinkerForConditionalGeneration"]
    thinker["_name_or_path"] = "Qwen/Qwen2.5-Omni-3B (Thinker, AuK slim)"
    thinker.pop("_attn_implementation_autoset", None)
    thinker["tie_word_embeddings"] = True
    thinker.setdefault("text_config", {})["tie_word_embeddings"] = True
    vision = thinker.setdefault("vision_config", {})
    vision["depth"] = 0
    vision["fullatt_block_indexes"] = []
    thinker["torch_dtype"] = thinker.get("torch_dtype") or "bfloat16"
    thinker["transformers_version"] = full_config.get("transformers_version", thinker.get("transformers_version"))
    thinker[SLIM_MARKER] = {"version": SLIM_VERSION, "source": "Qwen/Qwen2.5-Omni-3B",
                            "kept": list(KEPT_PREFIXES),
                            "dropped": ["thinker.visual.blocks", "thinker.lm_head", "talker", "token2wav"]}
    return thinker


def _write_json(path: Path, value: Any) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def build_slim_folder(source: str | os.PathLike, destination: str | os.PathLike, *,
                      max_shard_bytes: int = 4 * 1024**3,
                      progress: Callable[[str], Any] | None = print) -> dict:
    """Write the slim Thinker folder from a full Qwen2.5-Omni snapshot.

    Weights keep their dtype and their names relative to the Thinker (``thinker.``
    removed, as ``save_pretrained`` of the Thinker names them) and are copied
    tensor by tensor into shards of at most ``max_shard_bytes``. Each file is
    installed atomically; the index is written last, so an interrupted build is
    never reported complete.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    started = time.perf_counter()
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / WEIGHTS_INDEX).unlink(missing_ok=True)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    entries = []  # (new name, file, original name, nbytes)
    for path in weight_files(source):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(KEPT_PREFIXES):
                    view = handle.get_slice(key)
                    nbytes = math.prod(view.get_shape()) * {"BF16": 2, "F16": 2, "F32": 4}[view.get_dtype()]
                    entries.append((key[len("thinker."):], path, key, nbytes))
    if not entries:
        raise ValueError(f"No Thinker weights found in {source}")
    entries.sort(key=lambda item: item[0])
    shards, current, size = [], [], 0
    for entry in entries:
        if current and size + entry[3] > max_shard_bytes:
            shards.append(current)
            current, size = [], 0
        current.append(entry)
        size += entry[3]
    shards.append(current)
    names = [f"model-{index:05d}-of-{len(shards):05d}.safetensors" for index in range(1, len(shards) + 1)]
    for stale in destination.glob("model-*.safetensors"):
        if stale.name not in names:
            stale.unlink()
    handles: dict[Path, Any] = {}
    weight_map, total = {}, 0
    try:
        for name, shard in zip(names, shards):
            tensors = {}
            for new_name, path, key, nbytes in shard:
                handle = handles.get(path)
                if handle is None:
                    handle = handles[path] = safe_open(str(path), framework="pt", device="cpu")
                tensors[new_name] = handle.get_tensor(key)
                weight_map[new_name] = name
                total += nbytes
            partial = destination / (name + ".partial")
            save_file(tensors, str(partial), metadata={"format": "pt"})
            shutil.copymode(source / "config.json", partial)  # safetensors creates owner-only files
            os.replace(partial, destination / name)
            if progress:
                progress(f">> {name}: {len(tensors)} tensors, {sum(item[3] for item in shard) / 1e9:.2f} GB")
            del tensors
    finally:
        handles.clear()
    for name in PROCESSOR_FILES + NOTICE_FILES:
        if (source / name).is_file():
            shutil.copy2(source / name, destination / name)
    _write_json(destination / "config.json", slim_config(config))
    readme = destination / "README.md"
    readme.write_text(
        "# Qwen2.5-Omni-3B Thinker (AuK slim)\n\n"
        "The Thinker text model and audio tower of [Qwen/Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B), "
        "unchanged BF16 weights, as Tencent AuK's text and audio encoder uses them. The vision tower's transformer "
        "blocks, the Thinker's vocabulary head and the Talker/Token2Wav speech decoder are not included (the config "
        "declares no vision blocks and ties the head to the embeddings), so the folder is for hidden-state "
        "conditioning only, not for text, image or speech generation.\n\n"
        "Load with `Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(folder)` and "
        "`Qwen2_5OmniProcessor.from_pretrained(folder)`. Licensed under the Qwen Research License (LICENSE).\n",
        encoding="utf-8")
    _write_json(destination / WEIGHTS_INDEX, {"metadata": {"total_size": total}, "weight_map": weight_map})
    report = {"source": str(source), "destination": str(destination), "tensors": len(entries),
              "weight_bytes": total, "shards": names, "seconds": time.perf_counter() - started,
              "folder_bytes": sum(path.stat().st_size for path in destination.iterdir() if path.is_file())}
    if progress:
        progress(f">> Slim Thinker folder: {len(entries)} tensors, {total / 1e9:.2f} GB in {report['seconds']:.0f}s")
    return report
