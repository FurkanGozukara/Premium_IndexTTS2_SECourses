"""AuK training data over the application's reviewed manifest and split.

Two caches live beside the dataset in ``cache/auk``:

* the VAE posterior of every clip (mean and log standard deviation of its 50 Hz,
  64-channel latents); training samples latents from it on every step, as upstream
  encodes audio on every step;
* the fused Qwen2.5-Omni conditioning of each clip's no-reference instruction. The
  Thinker and the layer fusion stay frozen, so this conditioning is fixed and the
  encoder need not be loaded while training without reference prompts.

Training samples with a reference prompt (``auk_prompt_fraction``) pair a clip with
another clip of the same dataset, cropped like upstream's cross-utterance training,
and are encoded live.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset_manifest import atomic_write_json, load_manifest, write_manifest
from .plan import validation_record_ids

CACHE_DIR = Path("cache") / "auk"
LATENT_VERSION = "auk-vae-posterior-v1"
CONDITION_VERSION = "auk-fused-condition-v1"
TEXT_VERSION = "auk-normalized-text-v1"
VOICE_FILE = "auk_voice.json"
LATENT_RATE = 50
HOP = 480
MAX_TARGET_SECONDS = 30.0
MIN_TARGET_SECONDS = 0.3


def _identity(path: Path) -> list:
    stat = path.stat()
    return [str(path.resolve()), stat.st_size, stat.st_mtime_ns]


def _hash_once(path: Path, store: Path) -> str:
    """SHA-256 of a large file, recomputed only when its size or time changes."""
    from .features import _sha256

    try:
        stored = json.loads(store.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        stored = {}
    identity = _identity(path)
    if stored.get("identity") != identity:
        stored = {"identity": identity, "sha256": _sha256(path)}
        atomic_write_json(store, stored)
    return stored["sha256"]


def speech_span_seconds(audio: np.ndarray, rate: int, gate_db: float = -40.0) -> float:
    frame = max(1, int(rate * 0.01))
    count = len(audio) // frame
    if not count:
        return len(audio) / rate
    rms = np.sqrt(np.mean(audio[: count * frame].reshape(count, frame) ** 2, axis=1) + 1e-12)
    loud = np.nonzero(rms >= rms.max() * 10 ** (gate_db / 20.0))[0]
    return (int(loud[-1]) + 1 - int(loud[0])) * frame / rate if len(loud) else len(audio) / rate


def cache_auk_features(config, reporter=None, cancel_callback=None):
    """VAE posterior statistics of every manifest clip (``cache/auk/*.pt`` and ``index.jsonl``)."""
    import torchaudio

    from indextts.auk.loader import build_vae, read_config
    from indextts.backends.auk import ensure_model
    from indextts.runtime.progress import ProgressReporter
    from .features import FeatureCacheSummary, _atomic_torch_save, _audio_path, _cancelled, _read_audio, _sha256

    started = time.perf_counter()
    root = Path(config.dataset_dir).resolve()
    cache = root / CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(root)
    manifest_ids = {str(row["id"]) for row in rows}
    if getattr(config, "max_items", 0):
        rows = rows[: config.max_items]
    if not rows:
        raise ValueError("The dataset manifest is empty.")
    progress = reporter or ProgressReporter("AuK feature cache", total=len(rows))
    folder, _, _ = ensure_model(config.model_dir)
    signature = LATENT_VERSION + _hash_once(folder / "vae.safetensors", cache / "vae_hash.json")
    vae = None
    records, cached, skipped, cancelled = [], 0, 0, False
    try:
        for index, row in enumerate(rows):
            if _cancelled(cancel_callback, index):
                cancelled = True
                break
            audio_path = _audio_path(root, row)
            content = hashlib.sha256((signature + _sha256(audio_path)).encode()).hexdigest()
            path = cache / (hashlib.sha256(str(row["id"]).encode()).hexdigest()[:24] + ".pt")
            value = None
            if getattr(config, "skip_existing", True) and path.is_file():
                try:
                    candidate = torch.load(path, map_location="cpu", weights_only=True)
                    if candidate.get("fingerprint") == content:
                        value = candidate
                except (OSError, RuntimeError, ValueError, EOFError):
                    pass
            if value is None:
                if vae is None:
                    progress.log(">> Loading the AuK VAE for feature caching")
                    vae = build_vae(read_config(folder), folder / "vae.safetensors", device=config.device)
                audio, rate = _read_audio(audio_path)
                span = speech_span_seconds(audio.reshape(-1).numpy(), rate)
                if rate != 24000:
                    audio = torchaudio.functional.resample(audio, rate, 24000)
                audio = audio.reshape(1, 1, -1).to(config.device)
                with torch.inference_mode():
                    stats = vae.audio_encoder(audio.float())
                frames = min(int(audio.shape[-1]) // HOP, int(stats.shape[-1]))
                mean, log_std = stats[0, :, :frames].chunk(2, 0)
                value = {"fingerprint": content, "mean": mean.T.contiguous().to(torch.float16).cpu(),
                         "log_std": log_std.T.contiguous().to(torch.float16).cpu(), "speech_seconds": float(span)}
                _atomic_torch_save(path, value)
                cached += 1
            else:
                skipped += 1
            records.append({"id": row["id"], "path": path.relative_to(root).as_posix(),
                            "frames": int(value["mean"].shape[0]), "speech_seconds": float(value["speech_seconds"]),
                            "fingerprint": content})
            progress.update(index + 1, total=len(rows), desc=f"AuK latent cache {index + 1}/{len(rows)}")
        if not cancelled:
            previous = {str(item["id"]): item for item in load_manifest(cache / "index.jsonl")}
            previous.update({str(item["id"]): item for item in records})
            write_manifest(cache / "index.jsonl", (item for key, item in previous.items() if key in manifest_ids))
    finally:
        del vae
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    frames = [item["frames"] for item in records]
    summary = FeatureCacheSummary(str(root), len(rows), cached, skipped, cancelled, max(frames, default=0),
                                  sum(frames) / max(1, len(frames)), 0, 0.0,
                                  sum(value > MAX_TARGET_SECONDS * LATENT_RATE for value in frames),
                                  time.perf_counter() - started, [])
    atomic_write_json(cache / "summary.json", summary.to_dict())
    return summary


def normalized_transcripts(root, rows, log=print):
    """Transcripts normalized as generation normalizes them, cached by text."""
    from indextts.auk.text import normalize_auk_text

    path = Path(root) / CACHE_DIR / "normalized_text.json"
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        stored = {}
    texts = stored.get("texts", {}) if stored.get("version") == TEXT_VERSION else {}

    def key(row):
        return hashlib.sha256(f"{str(row.get('language') or 'en').lower()}\0{row['text']}".encode("utf-8")).hexdigest()

    def save():
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, {"version": TEXT_VERSION, "texts": texts})

    missing = [row for row in rows if key(row) not in texts]
    if missing:
        log(f">> Normalizing {len(missing)} transcripts as generation does (cached for later runs)")
    for index, row in enumerate(missing, start=1):
        text = str(row["text"])
        try:
            normalized = normalize_auk_text(text, str(row.get("language") or "en").lower())
        except Exception:
            normalized = text
        texts[key(row)] = normalized if normalized.strip() else text
        if index % 500 == 0:
            save()
            log(f">> Normalized {index}/{len(missing)} transcripts")
    if missing:
        save()
    return {str(row["id"]): texts[key(row)] for row in rows}


def training_instruction(text: str, description: str, language: str = "en") -> str:
    """The no-reference instruction a trained voice learns (and Auto voice later sends)."""
    from indextts.auk.text import build_instruction

    return build_instruction(text, "auto", description, language)


def clone_instruction(text: str) -> str:
    from indextts.auk.text import build_instruction

    return build_instruction(text, "clone")


def fusion_state(folder: Path) -> dict:
    """The base checkpoint's layer-fusion tensors (frozen during fine-tuning)."""
    from safetensors import safe_open

    with safe_open(str(folder / "auk_base.safetensors"), framework="pt", device="cpu") as handle:
        return {name: handle.get_tensor(name).float() for name in ("layer_weights", "layer_scale")}


def fuse(hidden_states, fusion) -> torch.Tensor:
    import torch.nn.functional as F

    d_llm = hidden_states[0].shape[-1]
    stacked = torch.stack([F.layer_norm(h, [d_llm]) for h in hidden_states[1:]], dim=0)
    weights = F.softmax(fusion["layer_weights"].to(stacked.device), dim=0)
    return (stacked * weights[:, None, None, None]).sum(dim=0) * fusion["layer_scale"].to(stacked.device)


def cache_auk_conditions(config, rows, instructions, reporter=None, cancel_callback=None, batch_size=16):
    """Fused Qwen conditioning of each instruction (no reference audio), cached by content."""
    from indextts.auk import text_encoder_folder
    from indextts.auk.conditioning import QwenConditioner
    from indextts.backends.auk import ensure_model
    from .features import _atomic_torch_save, _sha256

    root = Path(config.dataset_dir).resolve()
    cache = root / CACHE_DIR / "conditions"
    cache.mkdir(parents=True, exist_ok=True)
    folder, _, _ = ensure_model(config.model_dir)
    encoder_dir = text_encoder_folder(config.model_dir)
    weights = sorted(encoder_dir.glob("*.safetensors"))
    encoder_hash = hashlib.sha256("".join(_hash_once(path, root / CACHE_DIR / f"encoder_{path.stem}.json")
                                          for path in weights).encode()).hexdigest()
    fusion = fusion_state(folder)
    fusion_hash = hashlib.sha256(b"".join(value.numpy().tobytes() for value in fusion.values())).hexdigest()
    signature = CONDITION_VERSION + encoder_hash + fusion_hash
    paths, todo = {}, []
    for row in rows:
        key = hashlib.sha256((signature + instructions[str(row["id"])]).encode("utf-8")).hexdigest()
        path = cache / f"{key[:32]}.pt"
        paths[str(row["id"])] = path
        if not path.is_file():
            todo.append(row)
    if todo:
        log = reporter.log if reporter is not None and hasattr(reporter, "log") else print
        log(f">> Encoding {len(todo)} AuK training instructions with the frozen Qwen2.5-Omni encoder (cached)")
        encoder = QwenConditioner(encoder_dir, device=config.device)
        try:
            for start in range(0, len(todo), batch_size):
                if cancel_callback and cancel_callback():
                    break
                batch = todo[start:start + batch_size]
                texts = [instructions[str(row["id"])] for row in batch]
                with torch.inference_mode():
                    hidden, mask = encoder.hidden_states(encoder.inputs(texts, [None] * len(batch)))
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(config.device).startswith("cuda")):
                        fused = fuse(hidden, fusion)
                for offset, row in enumerate(batch):
                    valid = mask[offset]
                    _atomic_torch_save(paths[str(row["id"])], {"text": fused[offset][valid].to(torch.bfloat16).cpu()})
                if reporter is not None:
                    reporter.update(min(len(todo), start + batch_size), total=len(todo),
                                    desc=f"AuK conditioning {min(len(todo), start + batch_size)}/{len(todo)}")
        finally:
            encoder.unload()
            del encoder
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return {key: value.relative_to(root).as_posix() for key, value in paths.items()}


def voice_record(records, description) -> dict | None:
    """The trained voice's pace relative to upstream's byte model, and its training description."""
    from indextts.auk.text import f5_seconds

    speech = sum(float(row["cache"]["speech_seconds"]) for row in records)
    reference = sum(f5_seconds(row["train_text"], str(row.get("language") or "en").lower()) for row in records)
    if speech <= 0 or reference <= 0:
        return None
    return {"pace": speech / reference, "description": description, "clips": len(records),
            "speech_hours": round(speech / 3600, 3)}


class AukDataset(Dataset):
    """Cached clips of one split; items carry their objective (no reference, or a reference prompt)."""

    def __init__(self, config, split, log=print):
        self.config, self.split, self.epoch = config, split, 0
        self.root = Path(config.dataset_dir).resolve()
        records = load_manifest(self.root)
        validation = validation_record_ids(records, config.val_fraction, config.seed, config.val_split_mode)
        cached = {str(row["id"]): row for row in load_manifest(self.root / CACHE_DIR / "index.jsonl")}
        selected = [row for row in records if (str(row["id"]) in validation) == (split == "val")]
        texts = normalized_transcripts(self.root, selected, log=log) if config.auk_normalize_text else \
            {str(row["id"]): str(row["text"]) for row in selected}
        self.records = []
        limit = min(MAX_TARGET_SECONDS, float(config.max_codes) / LATENT_RATE) * LATENT_RATE
        for row in selected:
            data = cached.get(str(row["id"]))
            if not data:
                raise ValueError(f"Missing AuK feature cache for {row['id']}")
            if not MIN_TARGET_SECONDS * LATENT_RATE <= data["frames"] <= limit:
                continue
            self.records.append({**row, "train_text": texts[str(row["id"])], "cache": data})
        if split == "val":
            self.records.sort(key=lambda r: hashlib.sha256(f"{config.seed}:{r['id']}".encode()).hexdigest())
        if not self.records and split == "train":
            raise ValueError("No AuK training clips remain after length filtering.")
        description = str(config.auk_voice_description or "").strip()
        if not description:
            from indextts.auk.text import TRAINED_VOICE_DESCRIPTION

            description = TRAINED_VOICE_DESCRIPTION
        self.description = description
        self.instructions = {str(row["id"]): training_instruction(row["train_text"], description,
                                                                  str(row.get("language") or "en").lower())
                             for row in self.records}
        self.clone_instructions = {str(row["id"]): clone_instruction(row["train_text"]) for row in self.records}
        self.conditions: dict[str, str] = {}
        self.lengths = [row["cache"]["frames"] + 2 * len(row["train_text"].split()) + 40 for row in self.records]
        self.fingerprint = hashlib.sha256(json.dumps([(row["id"], row["cache"]["fingerprint"], row["train_text"])
                                                      for row in self.records] + [description]).encode()).hexdigest()

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _rng(self, row, purpose):
        epoch = self.epoch if self.split == "train" else -1
        digest = hashlib.sha256(f"{self.config.seed}:{epoch}:{purpose}:{row['id']}".encode()).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def objective(self, index) -> str:
        """"clone" (with a reference prompt) for ``auk_prompt_fraction`` of the clips, else "auto"."""
        fraction = float(self.config.auk_prompt_fraction)
        if fraction <= 0 or len(self.records) < 2:
            return "auto"
        return "clone" if self._rng(self.records[index], "objective").random() < fraction else "auto"

    def _stats(self, row):
        data = torch.load(self.root / row["cache"]["path"], map_location="cpu", weights_only=True)
        return data["mean"].float(), data["log_std"].float()

    def __getitem__(self, index):
        row = self.records[index]
        mean, log_std = self._stats(row)
        item = {"id": row["id"], "mean": mean, "log_std": log_std, "objective": self.objective(index)}
        if item["objective"] == "auto":
            item["instruction"] = self.instructions[str(row["id"])]
            cond = self.conditions.get(str(row["id"]))
            if cond:
                item["text"] = torch.load(self.root / cond, map_location="cpu", weights_only=True)["text"]
            return item
        rng = self._rng(row, "reference")
        partner = self.records[rng.randrange(len(self.records) - 1)]
        if partner["id"] == row["id"]:
            partner = self.records[-1]
        ref_mean, ref_log_std = self._stats(partner)
        frames = ref_mean.shape[0]
        # Upstream crops half of the references to a random length of at least 3 seconds.
        length = frames
        limit = int(float(self.config.auk_reference_seconds) * LATENT_RATE)
        if rng.random() < 0.5 and frames > 3 * LATENT_RATE:
            length = rng.randint(3 * LATENT_RATE, frames)
        length = min(length, limit) if limit > 0 else length
        start = rng.randint(0, frames - length)
        item.update(ref_mean=ref_mean[start:start + length], ref_log_std=ref_log_std[start:start + length],
                    ref_audio=_reference_audio(self.root, partner, start, length),
                    instruction=self.clone_instructions[str(row["id"])])
        return item


def _reference_audio(root, row, start_frame, frames):
    """16 kHz audio of a latent-frame window of a clip (the Qwen encoder's input)."""
    from .evaluation_plan import audio_path
    from .features import _read_audio
    from indextts.auk.conditioning import to_encoder_rate

    audio, rate = _read_audio(audio_path(root, row))
    audio = audio.reshape(-1).numpy()
    begin = int(start_frame / LATENT_RATE * rate)
    end = int((start_frame + frames) / LATENT_RATE * rate)
    return to_encoder_rate(audio[begin:end], rate)


def collate(items):
    """Pad one micro-batch: targets, optional references and the cached conditioning."""
    def pad(tensors, width):
        length = max((tensor.shape[0] for tensor in tensors), default=0)
        out = torch.zeros(len(tensors), length, width)
        for index, tensor in enumerate(tensors):
            out[index, : tensor.shape[0]] = tensor
        return out

    batch = {"ids": [item["id"] for item in items], "objective": [item["objective"] for item in items],
             "instructions": [item["instruction"] for item in items],
             "target_mean": pad([item["mean"] for item in items], 64),
             "target_log_std": pad([item["log_std"] for item in items], 64),
             "target_lens": torch.tensor([item["mean"].shape[0] for item in items])}
    if all("text" in item for item in items):
        texts = [item["text"] for item in items]
        batch["text"] = pad(texts, texts[0].shape[-1]).to(torch.bfloat16)
        batch["text_lens"] = torch.tensor([text.shape[0] for text in texts])
    references = [item.get("ref_mean") for item in items]
    if any(reference is not None for reference in references):
        empty = torch.zeros(0, 64)
        batch["ref_mean"] = pad([item.get("ref_mean", empty) for item in items], 64)
        batch["ref_log_std"] = pad([item.get("ref_log_std", empty) for item in items], 64)
        batch["ref_lens"] = torch.tensor([item["ref_mean"].shape[0] if "ref_mean" in item else 0 for item in items])
        batch["ref_audio"] = [item.get("ref_audio") for item in items]
    return batch


def sample_latents(mean, log_std, global_mean, global_var, generator=None):
    """Normalised latents drawn from the cached posterior (upstream's encoding_and_normalization)."""
    noise = torch.randn(mean.shape, device=mean.device, generator=generator)
    latent = mean + noise * torch.exp(log_std)
    return (latent - global_mean) / torch.sqrt(global_var)


def normalised_means(mean, global_mean, global_var):
    return (mean - global_mean) / torch.sqrt(global_var)


VALIDATION_TIMES = tuple(round(0.1 * index, 1) for index in range(10))


def flow_validation_loss(model, loader, device, encode, vae_stats, *, max_batches=0, cancel_callback=None):
    """Upstream's per-t validation: flow-matching loss at t = 0.0..0.9 with each clip's own fixed noise.

    Posterior means (no sampling) and per-clip seeded noise make the number identical for
    every checkpoint. Returns the mean over the grid and the per-t losses.
    """
    global_mean, global_var = vae_stats
    model.eval()
    sums = torch.zeros(len(VALIDATION_TIMES), dtype=torch.float64)
    count = 0
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if (cancel_callback and cancel_callback()) or (max_batches and index >= max_batches):
                break
            text, context, ref, ref_lens = encode(batch)
            target = normalised_means(batch["target_mean"].to(device), global_mean, global_var)
            lens = batch["target_lens"].to(device)
            noise = torch.stack([torch.randn(target.shape[1:], generator=torch.Generator().manual_seed(
                int(hashlib.sha256(str(item).encode()).hexdigest()[:8], 16))) for item in batch["ids"]]).to(device)
            for position, value in enumerate(VALIDATION_TIMES):
                time_value = torch.full((target.shape[0],), float(value), device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    loss = model(target, text, context, ref_latent=ref, ref_lens=ref_lens, target_lens=lens,
                                 time=time_value, x0=noise, apply_cond_drop=False)
                sums[position] += float(loss) * target.shape[0]
            count += target.shape[0]
    model.train()
    if not count:
        return None, []
    per_t = (sums / count).tolist()
    return float(sum(per_t) / len(per_t)), per_t
