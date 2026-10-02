"""OmniVoice codec cache over the application's reviewed manifest and split."""
from __future__ import annotations

import gc
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .dataset_manifest import atomic_write_json, load_manifest, write_manifest
from .plan import validation_record_ids


def cache_omnivoice_features(config, reporter=None, cancel_callback=None):
    from transformers import AutoFeatureExtractor, AutoTokenizer, HiggsAudioV2TokenizerModel
    from indextts.backends.omnivoice import ensure_model
    from indextts.runtime.progress import ProgressReporter
    from indextts.utils.torch_compat import install_native_enum_pytree_compatibility
    from .features import (FeatureCacheSummary, _atomic_torch_save, _audio_path,
                           _cancelled, _read_audio, _sha256)
    import torchaudio

    install_native_enum_pytree_compatibility()
    started = time.perf_counter()
    root = Path(config.dataset_dir).resolve()
    cache = root / "cache" / "omnivoice"
    cache.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(root)
    manifest_ids = {str(row["id"]) for row in rows}
    if config.max_items:
        rows = rows[:config.max_items]
    if not rows:
        raise ValueError("The dataset manifest is empty.")
    progress = reporter or ProgressReporter("OmniVoice feature cache", total=len(rows))
    folder, _ = ensure_model(config.model_dir)
    # Hash the codec once per unchanged file identity, then use content hashes in
    # every clip key. Updating either audio or tokenizer invalidates the cache.
    codec_file = folder / "audio_tokenizer/model.safetensors"
    identity = [str(codec_file.resolve()), codec_file.stat().st_size, codec_file.stat().st_mtime_ns]
    model_hash_path = cache / "model_hash.json"
    try:
        model_hash = json.loads(model_hash_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        model_hash = {}
    if model_hash.get("identity") != identity:
        model_hash = {"identity": identity, "sha256": _sha256(codec_file)}
        atomic_write_json(model_hash_path, model_hash)
    tokenizer_hash = _sha256(folder / "tokenizer.json")
    signature = model_hash["sha256"] + tokenizer_hash + "higgs-feature-extractor-v1"
    tokenizer = AutoTokenizer.from_pretrained(str(folder))
    extractor = AutoFeatureExtractor.from_pretrained(str(folder / "audio_tokenizer"))
    codec = None
    records, cached, skipped, cancelled = [], 0, 0, False
    try:
        for index, row in enumerate(rows):
            if _cancelled(cancel_callback, index):
                cancelled = True
                break
            audio_path = _audio_path(root, row)
            content = hashlib.sha256((signature + _sha256(audio_path) + str(row["text"])).encode()).hexdigest()
            path = cache / (hashlib.sha256(str(row["id"]).encode()).hexdigest()[:24] + ".pt")
            value = None
            if config.skip_existing and path.is_file():
                try:
                    candidate = torch.load(path, map_location="cpu", weights_only=True)
                    if candidate.get("fingerprint") == content:
                        value = candidate
                except (OSError, RuntimeError, ValueError, EOFError):
                    pass
            if value is None:
                if codec is None:
                    progress.log(">> Loading OmniVoice audio tokenizer for feature caching")
                    codec = HiggsAudioV2TokenizerModel.from_pretrained(str(folder / "audio_tokenizer"), device_map=config.device).eval()
                audio, rate = _read_audio(audio_path)
                if rate != extractor.sampling_rate:
                    audio = torchaudio.functional.resample(audio, rate, extractor.sampling_rate)
                inputs = extractor(raw_audio=audio.reshape(-1).numpy(), sampling_rate=extractor.sampling_rate, return_tensors="pt")
                with torch.inference_mode():
                    codes = codec.encode(inputs["input_values"].to(config.device)).audio_codes[0].cpu().to(torch.int16)
                if codes.ndim != 2 or codes.shape[0] != 8 or not codes.shape[1]:
                    raise ValueError(f"Invalid OmniVoice audio tokens for {row['id']}")
                value = {"fingerprint": content, "audio_tokens": codes,
                         "n_text_tokens": len(tokenizer.encode(row["text"], add_special_tokens=False))}
                _atomic_torch_save(path, value)
                cached += 1
            else:
                skipped += 1
            records.append({"id": row["id"], "path": path.relative_to(root).as_posix(),
                            "n_codes": int(value["audio_tokens"].shape[1]),
                            "n_text_tokens": int(value["n_text_tokens"]), "fingerprint": content})
            progress.update(index + 1, total=len(rows), desc=f"OmniVoice codec cache {index + 1}/{len(rows)}")
        if not cancelled:
            # Readers (training and grids) must keep seeing the complete index
            # while an existing cache is checked. Publishing each partial pass
            # temporarily made valid later clips look uncached.
            previous = {str(row["id"]): row for row in load_manifest(cache / "index.jsonl")}
            previous.update({str(row["id"]): row for row in records})
            write_manifest(cache / "index.jsonl", (row for key, row in previous.items() if key in manifest_ids))
    finally:
        del codec
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    codes = [r["n_codes"] for r in records]
    texts = [r["n_text_tokens"] for r in records]
    summary = FeatureCacheSummary(str(root), len(rows), cached, skipped, cancelled,
        max(codes, default=0), sum(codes) / max(1, len(codes)), max(texts, default=0),
        sum(texts) / max(1, len(texts)), sum(c > config.max_codes or t > config.max_text_tokens for c,t in zip(codes,texts)),
        time.perf_counter() - started, [])
    atomic_write_json(cache / "summary.json", summary.to_dict())
    return summary


NORMALIZED_TEXT_VERSION = "omnivoice-normalized-text-v1"


def normalized_transcripts(root, rows, log=print):
    """Transcripts normalized exactly as generation normalizes text, cached by text.

    The cache lives beside the codec cache but separately, so normalizing never
    invalidates the audio tokens.
    """
    from indextts.backends.omnivoice import normalize_omnivoice_text

    path = Path(root) / "cache" / "omnivoice" / "normalized_text.json"
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        stored = {}
    cache = stored.get("texts", {}) if stored.get("version") == NORMALIZED_TEXT_VERSION else {}

    def key(row):
        language = str(row.get("language") or "en").lower()
        return hashlib.sha256(f"{language}\0{row['text']}".encode("utf-8")).hexdigest()

    def save():
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, {"version": NORMALIZED_TEXT_VERSION, "texts": cache})

    missing = [row for row in rows if key(row) not in cache]
    if missing:
        log(f">> Normalizing {len(missing)} transcripts as generation does (cached for later runs)")
    for index, row in enumerate(missing, start=1):
        text = str(row["text"])
        try:
            normalized = normalize_omnivoice_text(text, str(row.get("language") or "en"))
        except Exception:
            normalized = text
        cache[key(row)] = normalized if normalized.strip() else text
        if index % 500 == 0:
            save()
            log(f">> Normalized {index}/{len(missing)} transcripts")
    if missing:
        save()
    return {str(row["id"]): cache[key(row)] for row in rows}


VOICE_CALIBRATION_FILE = "omnivoice_voice.json"
# OmniVoice sizes speech without a reference from this phrase and token count.
_FALLBACK_TEXT, _FALLBACK_TOKENS = "Nice to meet you.", 25


def speaking_rate_calibration(records):
    """The trained voice's pace in OmniVoice's own duration units.

    Without a reference, OmniVoice estimates every sentence's length from a
    fixed phrase. A fine-tuned voice speaks at its speaker's pace instead, so
    the trainer stores the speed that converts that estimate to this voice.
    """
    from omnivoice.utils.duration import RuleDurationEstimator

    estimator = RuleDurationEstimator()
    weight = sum(estimator.calculate_total_weight(row.get("train_text") or row["text"]) for row in records)
    tokens = sum(int(row["cache"]["n_codes"]) for row in records)
    if weight <= 0 or tokens <= 0:
        return None
    voice = tokens / weight
    fallback = _FALLBACK_TOKENS / estimator.calculate_total_weight(_FALLBACK_TEXT)
    return {"tokens_per_weight": voice, "fallback_tokens_per_weight": fallback,
            "speed_without_reference": fallback / voice, "clips": len(records)}


def masked_audio_metrics(model, loader, device, *, dtype=torch.bfloat16, max_batches=0, cancel_callback=None):
    """The same fixed-mask, token-weighted metric for training and checkpoint grids."""
    if loader is None or not len(loader):
        return {"loss": None, "mel_loss": None, "text_loss": None, "accuracy": None}
    model.eval()
    sums, counts = torch.zeros(8, device=device), torch.zeros(8, device=device)
    correct = torch.zeros((), device=device)
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if (cancel_callback and cancel_callback()) or (max_batches and index >= max_batches):
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
                logits = model(**batch).logits
                losses = torch.nn.functional.cross_entropy(logits.permute(0,3,1,2), labels, reduction="none", ignore_index=-100)
            valid = labels != -100
            sums += (losses * valid).sum((0,2))
            counts += valid.sum((0,2))
            correct += ((logits.argmax(-1) == labels) & valid).sum()
    if not counts.sum():
        return {"loss": None, "mel_loss": None, "text_loss": None, "accuracy": None}
    weights = torch.tensor(model.normalized_audio_codebook_weights, device=device)
    loss = ((sums / counts.clamp_min(1)) * weights).sum().item()
    return {"loss": loss, "mel_loss": loss, "text_loss": None, "accuracy": (correct / counts.sum()).item()}


class OmniVoiceDataset(Dataset):
    def __init__(self, config, tokenizer, split, log=print):
        from omnivoice.data.processor import OmniVoiceSampleProcessor
        self.config, self.split, self.epoch = config, split, 0
        self.root = Path(config.dataset_dir).resolve()
        records = load_manifest(self.root)
        validation = validation_record_ids(records, config.val_fraction, config.seed, config.val_split_mode)
        cached = {str(r["id"]): r for r in load_manifest(self.root / "cache/omnivoice/index.jsonl")}
        selected = [row for row in records if (str(row["id"]) in validation) == (split == "val")]
        texts = (normalized_transcripts(self.root, selected, log=log) if getattr(config, "omni_normalize_text", False)
                 else {str(row["id"]): str(row["text"]) for row in selected})
        self.records = []
        for row in selected:
            data = cached.get(str(row["id"]))
            if not data:
                raise ValueError(f"Missing OmniVoice feature cache for {row['id']}")
            text = texts[str(row["id"])]
            # The prompt template adds about 20 tokens to the transcript.
            n_text = len(tokenizer.encode(text, add_special_tokens=False)) if text != row["text"] else data["n_text_tokens"]
            if data["n_codes"] > config.max_codes or n_text > config.max_text_tokens:
                continue
            self.records.append({**row, "train_text": text, "cache": {**data, "n_text_tokens": n_text}})
        if split == "val":
            self.records.sort(key=lambda r: hashlib.sha256(f"{config.seed}:{r['id']}".encode()).hexdigest())
        if not self.records and split == "train":
            raise ValueError("No OmniVoice training clips remain after length filtering.")
        self.lengths = [r["cache"]["n_codes"] + r["cache"]["n_text_tokens"] + 20 for r in self.records]
        self.processor = OmniVoiceSampleProcessor(tokenizer, 8, 1024, (0.0, config.omni_prompt_ratio),
            (0.0, 1.0), config.omni_drop_condition, config.omni_language_ratio, 0.0, 0.0, 0.0)
        # Rewritten transcripts are part of the data a continued run must match.
        self.fingerprint = hashlib.sha256(json.dumps([
            (r["id"], r["cache"]["fingerprint"]) if r["train_text"] == r["text"] else (r["id"], r["cache"]["fingerprint"], r["train_text"])
            for r in self.records]).encode()).hexdigest()

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        row = self.records[index]
        codes = torch.load(self.root / row["cache"]["path"], map_location="cpu", weights_only=True)["audio_tokens"]
        # Validation uses identical masks at every checkpoint; training changes
        # them by epoch. Per-record RNG makes continued runs reproducible.
        seed = int.from_bytes(hashlib.sha256(f"{self.config.seed}:{self.epoch if self.split == 'train' else -1}:{row['id']}".encode()).digest()[:8], "big") % (2**32)
        previous = random.getstate()
        try:
            with torch.random.fork_rng(devices=[]):
                random.seed(seed)
                torch.random.default_generator.manual_seed(seed)
                return self.processor({"audio_tokens": codes, "label": {
                    "text": row.get("train_text") or row["text"], "language_id": str(row.get("language") or "en").lower()}})
        finally:
            random.setstate(previous)
