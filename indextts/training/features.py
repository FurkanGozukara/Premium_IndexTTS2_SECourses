"""Feature caching for inference-aligned IndexTTS 2.5 LoRA / DoRA training."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from omegaconf import OmegaConf

from indextts.runtime import ProgressReporter
from indextts.utils.front import TextNormalizer
from indextts.utils.ja_g2p import JapaneseG2PProcessor
from indextts.utils.nemo_tn import normalize_text as nemo_text_normalize
from indextts.utils.tokenizer import get_tokenizer, lang_to_token

from .dataset_manifest import atomic_write_json, load_manifest, write_cache_index


CACHE_FORMAT = "indextts2_training_features"
CACHE_VERSION = 2


@dataclass
class FeatureCacheConfig:
    dataset_dir: str
    model_dir: str = "models"
    model_config: str = "models/config.yaml"
    device: str = "cuda:0"
    semantic_layer: int = 17
    batch_size: int = 4
    skip_existing: bool = True
    max_items: int = 0
    max_codes: int = 1500
    max_text_tokens: int = 600
    verify_count: int = 0
    verify_output_dir: str = "outputs/_feature_check"

    def validate(self) -> "FeatureCacheConfig":
        self.dataset_dir = str(self.dataset_dir)
        self.model_dir = str(self.model_dir or "models")
        self.model_config = str(self.model_config or Path(self.model_dir) / "config.yaml")
        self.device = str(self.device or "cuda:0")
        self.semantic_layer = max(1, int(self.semantic_layer))
        self.batch_size = max(1, int(self.batch_size))
        self.max_items = max(0, int(self.max_items))
        self.max_codes = max(1, int(self.max_codes))
        self.max_text_tokens = max(1, int(self.max_text_tokens))
        self.verify_count = max(0, int(self.verify_count))
        self.verify_output_dir = str(self.verify_output_dir or "outputs/_feature_check")
        return self

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | "FeatureCacheConfig") -> "FeatureCacheConfig":
        if isinstance(value, cls):
            return cls(**asdict(value)).validate()
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: item for key, item in value.items() if key in allowed}).validate()


@dataclass
class FeatureCacheSummary:
    dataset_dir: str
    total: int
    cached: int
    skipped: int
    cancelled: bool
    max_codes: int
    mean_codes: float
    max_text_tokens: int
    mean_text_tokens: float
    over_length_count: int
    elapsed_s: float
    verification_files: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TextFeaturePipeline:
    """The text-only subset of IndexTTS2's inference preprocessing."""

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = str(model_dir)
        self.tokenizer = get_tokenizer(multilingual=True, model_dir=self.model_dir)
        self.normalizer = TextNormalizer(enable_glossary=True)
        self.normalizer.load()
        glossary = Path(self.model_dir) / "glossary.yaml"
        if glossary.is_file():
            self.normalizer.load_glossary_from_yaml(str(glossary))
        self.japanese = JapaneseG2PProcessor(g2p_ratio=0)

    def process(self, text: str, language: str) -> tuple[str, torch.Tensor, int]:
        # Importing here avoids constructing any of the inference engine models.
        from indextts.infer_v2_5 import apply_pronunciation_annotations

        lang = str(language or "EN").lower()
        processed = self.normalizer.clean_pattern.sub(
            lambda match: self.normalizer.char_rep_map[match.group()], str(text)
        )
        if lang in {"zh", "zhen", "en"}:
            processed = self.normalizer.normalize(processed)
        elif lang in {"ja", "es"}:
            processed = nemo_text_normalize(processed, lang)
        if lang in {"ja", "zh", "zhen", "en"}:
            processed = processed.lower()
        elif lang == "es":
            processed = processed.upper()
        processed = apply_pronunciation_annotations(processed)
        if lang == "ja":
            processed = self.japanese.process_ja_text(processed)
        processed = re.sub(
            r"<\|([^|]+)\|>", lambda match: f"<|{match.group(1).upper()}|>", processed
        )
        prefix = f"<|{lang}|> "
        tokens = self.tokenizer.encode(prefix + processed, allowed_special="all")
        return processed, torch.tensor(tokens, dtype=torch.int32), int(lang_to_token(lang))


class _EmotionFeatureModel(torch.nn.Module):
    """Only the four GPT modules required to cache emotion conditioning."""

    def __init__(self, gpt_config: Mapping[str, Any]) -> None:
        super().__init__()
        from indextts.gpt.conformer_encoder import ConformerEncoder
        from indextts.gpt.perceiver import PerceiverResampler

        emo = gpt_config["emo_condition_module"]
        model_dim = int(gpt_config["model_dim"])
        self.emo_conditioning_encoder = ConformerEncoder(
            input_size=1024,
            output_size=int(emo["output_size"]),
            linear_units=int(emo["linear_units"]),
            attention_heads=int(emo["attention_heads"]),
            num_blocks=int(emo["num_blocks"]),
            input_layer=str(emo["input_layer"]),
        )
        self.emo_perceiver_encoder = PerceiverResampler(
            1024,
            dim_context=int(emo["output_size"]),
            ff_mult=int(emo["perceiver_mult"]),
            heads=int(emo["attention_heads"]),
            num_latents=1,
        )
        self.emovec_layer = torch.nn.Linear(1024, model_dim)
        self.emo_layer = torch.nn.Linear(model_dim, model_dim)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, mask = self.emo_conditioning_encoder(features, lengths)
        cond_mask = F.pad(mask.squeeze(1), (1, 0), value=True)
        raw = self.emo_perceiver_encoder(encoded, cond_mask).squeeze(1)
        vector = self.emo_layer(self.emovec_layer(raw))
        return raw, vector


def _load_prefixed_state(
    module: torch.nn.Module, state: Mapping[str, torch.Tensor], prefix: str
) -> None:
    marker = prefix + "."
    selected = {key[len(marker) :]: value for key, value in state.items() if key.startswith(marker)}
    result = module.load_state_dict(selected, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"failed to load GPT feature module {prefix}")


def _load_emotion_model(
    gpt_config: Mapping[str, Any], checkpoint: Path, device: torch.device, dtype: torch.dtype
) -> _EmotionFeatureModel:
    model = _EmotionFeatureModel(gpt_config)
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    except (TypeError, RuntimeError, ValueError):
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, Mapping) and isinstance(state.get("model"), Mapping):
        state = state["model"]
    if not isinstance(state, Mapping):
        raise TypeError(f"GPT checkpoint does not contain a state dictionary: {checkpoint}")
    for name in (
        "emo_conditioning_encoder",
        "emo_perceiver_encoder",
        "emovec_layer",
        "emo_layer",
    ):
        _load_prefixed_state(getattr(model, name), state, name)
    del state
    model.to(device=device, dtype=dtype).eval()
    return model


class _FeatureModels:
    def __init__(self, config: FeatureCacheConfig, reporter: ProgressReporter) -> None:
        from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel

        from indextts.codec.models import EnhancedCodec
        from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus

        self.config = config
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was requested for feature caching, but is unavailable: {self.device}")
        self.compute_dtype = (
            torch.bfloat16
            if self.device.type == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        cfg = OmegaConf.load(config.model_config)
        self.model_cfg = cfg
        model_dir = Path(config.model_dir)

        reporter.set_stage("load w2v-bert")
        w2v_dir = model_dir / "hf_cache" / "w2v-bert-2.0"
        self.processor = SeamlessM4TFeatureExtractor.from_pretrained(
            str(w2v_dir), local_files_only=True
        )
        self.semantic_model = Wav2Vec2BertModel.from_pretrained(
            str(w2v_dir), local_files_only=True
        ).to(device=self.device, dtype=torch.float32).eval()
        stats = torch.load(model_dir / str(cfg.w2v_stat), map_location="cpu", weights_only=True)
        self.semantic_mean = stats["mean"].float().to(self.device)
        self.semantic_std = torch.sqrt(stats["var"].float()).to(self.device)

        reporter.set_stage("load semantic codec")
        self.codec = EnhancedCodec(**cfg.semantic_codec, cfg=cfg.semantic_codec)
        self.codec.load_checkpoint(str(model_dir / "codec.pth"))
        self.codec.to(device=self.device, dtype=torch.float32).eval()

        reporter.set_stage("load CAMPPlus")
        self.campplus = CAMPPlus(feat_dim=80, embedding_size=192)
        camp_path = model_dir / "hf_cache" / "campplus_cn_common.bin"
        self.campplus.load_state_dict(
            torch.load(camp_path, map_location="cpu", weights_only=True), strict=True
        )
        self.campplus.to(device=self.device, dtype=torch.float32).eval()

        reporter.set_stage("load GPT emotion modules")
        self.emotion = _load_emotion_model(
            OmegaConf.to_container(cfg.gpt, resolve=True),
            model_dir / str(cfg.gpt_checkpoint),
            self.device,
            self.compute_dtype,
        )

    @torch.no_grad()
    def w2v_features(self, waveforms: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        arrays = [waveform.squeeze(0).cpu().numpy() for waveform in waveforms]
        values = self.processor(
            arrays, sampling_rate=16000, return_tensors="pt", padding=True
        )
        input_features = values["input_features"].to(self.device)
        attention_mask = values.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones(
                input_features.shape[:2], dtype=torch.long, device=self.device
            )
        else:
            attention_mask = attention_mask.to(self.device)
        # The inference encoder and codec run in FP32. BF16 perturbations cross
        # quantizer boundaries and silently change the supervision code IDs.
        # Emotion projection below can still use the GPT's compute precision.
        with torch.autocast(device_type=self.device.type, enabled=False):
            output = self.semantic_model(
                input_features=input_features.float(),
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        layer = min(self.config.semantic_layer, len(output.hidden_states) - 1)
        hidden = output.hidden_states[layer]
        output_mask = self.semantic_model._get_feature_vector_attention_mask(
            hidden.shape[1], attention_mask
        )
        result: list[torch.Tensor] = []
        for index in range(hidden.shape[0]):
            length = max(1, int(output_mask[index].sum().item()))
            feature = hidden[index : index + 1, :length].float()
            result.append((feature - self.semantic_mean) / self.semantic_std)
        return result

    @torch.no_grad()
    def item_features(
        self, waveform: torch.Tensor, normalized_feature: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        codes, _ = self.codec.quantize(normalized_feature.to(dtype=torch.float32))
        if codes.ndim == 1:
            code_values = codes
        elif codes.ndim == 2:
            code_values = codes[0]
        else:
            code_values = codes.reshape(-1, codes.shape[-1])[0]

        fbank = torchaudio.compliance.kaldi.fbank(
            waveform.cpu(), num_mel_bins=80, dither=0, sample_frequency=16000
        )
        fbank = fbank - fbank.mean(dim=0, keepdim=True)
        campplus = self.campplus(fbank.unsqueeze(0).to(self.device)).squeeze(0)

        lengths = torch.tensor(
            [normalized_feature.shape[1]], dtype=torch.long, device=self.device
        )
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.compute_dtype,
            enabled=self.device.type == "cuda" and self.compute_dtype != torch.float32,
        ):
            emo_raw, emo_vec = self.emotion(
                normalized_feature.to(dtype=self.compute_dtype), lengths
            )
        return (
            code_values.to(dtype=torch.int16).cpu(),
            campplus.float().cpu(),
            emo_raw.squeeze(0).float().cpu(),
            emo_vec.squeeze(0).float().cpu(),
        )


def _audio_path(dataset_dir: Path, row: Mapping[str, Any]) -> Path:
    value = Path(str(row.get("audio") or ""))
    return value if value.is_absolute() else dataset_dir / value


def _read_audio(path: Path) -> tuple[torch.Tensor, int]:
    # Torchaudio 2.13 delegates file I/O to the optional torchcodec package.
    # SoundFile is already an IndexTTS dependency and keeps cache generation
    # independent of that optional decoder.
    array, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(array.T.copy())
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, int(sample_rate)


def _load_audio_16k(path: Path) -> tuple[torch.Tensor, float]:
    waveform, sample_rate = _read_audio(path)
    if int(sample_rate) != 16000:
        waveform = torchaudio.functional.resample(waveform, int(sample_rate), 16000)
    waveform = waveform.float().clamp_(-1.0, 1.0).contiguous()
    return waveform, waveform.shape[-1] / 16000.0


def _cache_valid(
    path: Path,
    semantic_layer: int | None = None,
    *,
    source_fingerprint: str | None = None,
    extraction_fingerprint: str | None = None,
) -> bool:
    if not path.is_file():
        return False
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError, EOFError):
        return False
    required = {"text_tokens", "codes", "campplus", "emo_raw", "emo_vec"}
    if not isinstance(value, Mapping) or not required.issubset(value):
        return False
    if semantic_layer is not None and int(value.get("semantic_layer", -1)) != int(semantic_layer):
        return False
    if value.get("format") != CACHE_FORMAT or value.get("version") != CACHE_VERSION:
        return False
    for key, expected in (("source_fingerprint", source_fingerprint), ("extraction_fingerprint", extraction_fingerprint)):
        if expected is not None and value.get(key) != expected:
            return False
    for key in required:
        tensor = value[key]
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 1 or not tensor.numel():
            return False
        if not torch.isfinite(tensor).all():
            return False
    return bool((value["codes"] >= 0).all() and (value["codes"] < 8192).all())


def _source_fingerprint(dataset_dir: Path, row: Mapping[str, Any]) -> str:
    """Bind cached labels/features to audio bytes and the current transcript."""
    payload = {
        "audio_sha256": _sha256(_audio_path(dataset_dir, row)),
        "text": str(row.get("text", "")),
        "language": str(row.get("language", "EN")),
        "speaker": str(row.get("speaker", "")),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _extraction_fingerprint(config: FeatureCacheConfig, model_hashes: Mapping[str, str]) -> str:
    root = Path(config.model_dir)
    assets = sorted(set(root.glob("*.tiktoken")) | set(root.glob("*.model")) | set(root.glob("*glossary*.yaml")))
    w2v = root / "hf_cache" / "w2v-bert-2.0"
    assets += sorted(w2v.glob("*.json"))
    source_root = Path(__file__).resolve().parents[1]
    payload = {
        "version": CACHE_VERSION, "semantic_dtype": "fp32", "semantic_layer": config.semantic_layer,
        "model_hashes": dict(model_hashes), "model_config": _sha256(Path(config.model_config)),
        "text_assets": {str(path.relative_to(root)): _sha256(path) for path in assets},
        "code": {str(path.relative_to(source_root)): _sha256(path) for path in (
            Path(__file__), source_root / "utils" / "tokenizer.py", source_root / "utils" / "front.py",
        )},
        "torch": torch.__version__, "transformers": _package_version("transformers"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _atomic_torch_save(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(value), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _cancelled(callback: Callable[..., bool] | None, completed: int) -> bool:
    if callback is None:
        return False
    try:
        return bool(callback(completed))
    except TypeError:
        return bool(callback())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return "unknown"


def _model_hashes(model_dir: Path) -> dict[str, str]:
    w2v_files = sorted((model_dir / "hf_cache" / "w2v-bert-2.0").glob("*.safetensors"))
    paths = {
        "gpt": model_dir / "gpt.pth",
        "codec": model_dir / "codec.pth",
        "campplus": model_dir / "hf_cache" / "campplus_cn_common.bin",
        "wav2vec2bert_stats": model_dir / "wav2vec2bert_stats.pt",
    }
    if w2v_files:
        paths["wav2vec2bert"] = w2v_files[0]
    return {name: _sha256(path) for name, path in paths.items() if path.is_file()}


def _record_from_cache(row: Mapping[str, Any], path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "id": str(row["id"]),
        "path": path.relative_to(path.parent.parent).as_posix(),
        "n_codes": int(torch.as_tensor(value["codes"]).numel()),
        "n_text_tokens": int(torch.as_tensor(value["text_tokens"]).numel()),
        "duration_s": float(value.get("duration_s", row.get("duration_s", 0.0))),
        "speaker": str(value.get("speaker", row.get("speaker", ""))),
        "source_fingerprint": str(value.get("source_fingerprint", "")),
    }


def _update_dataset_info(
    dataset_dir: Path, records: Sequence[Mapping[str, Any]], config: FeatureCacheConfig
) -> dict[str, Any]:
    path = dataset_dir / "dataset_info.json"
    try:
        info = json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}
    except (OSError, UnicodeError, json.JSONDecodeError):
        info = {}
    code_counts = [int(item["n_codes"]) for item in records]
    text_counts = [int(item["n_text_tokens"]) for item in records]
    over = sum(
        codes > config.max_codes or text > config.max_text_tokens
        for codes, text in zip(code_counts, text_counts)
    )
    statistics = {
        "cached_segments": len(records),
        "max_codes": max(code_counts, default=0),
        "mean_codes": sum(code_counts) / len(code_counts) if code_counts else 0.0,
        "max_text_tokens": max(text_counts, default=0),
        "mean_text_tokens": sum(text_counts) / len(text_counts) if text_counts else 0.0,
        "over_length_count": int(over),
        "limits": {"max_codes": config.max_codes, "max_text_tokens": config.max_text_tokens},
    }
    info["token_statistics"] = statistics
    info.setdefault("cache", {})
    info["cache"].update(
        {
            "directory": "cache",
            "index": "cache/index.jsonl",
            "metadata": "cache_index.json",
            "semantic_layer": config.semantic_layer,
        }
    )
    atomic_write_json(path, info)
    return statistics


def cache_dataset_features(
    config: FeatureCacheConfig | Mapping[str, Any] | str | Path,
    reporter: ProgressReporter | None = None,
    cancel_callback: Callable[..., bool] | None = None,
) -> FeatureCacheSummary:
    """Cache all selected manifest rows without loading the full TTS engine."""

    if isinstance(config, (str, Path)):
        resolved = FeatureCacheConfig(dataset_dir=str(config)).validate()
    else:
        resolved = FeatureCacheConfig.from_dict(config)
    dataset_dir = Path(resolved.dataset_dir).expanduser().resolve()
    rows = load_manifest(dataset_dir)
    if resolved.max_items:
        rows = rows[: resolved.max_items]
    if not rows:
        raise FileNotFoundError(f"manifest.jsonl is empty or missing in {dataset_dir}")

    active_reporter = reporter or ProgressReporter("segments", total=len(rows))
    active_reporter.set_stage("text preprocessing")
    started = time.perf_counter()
    text_pipeline = TextFeaturePipeline(resolved.model_dir)
    cache_dir = dataset_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_hashes = _model_hashes(Path(resolved.model_dir))
    extraction_fingerprint = _extraction_fingerprint(resolved, model_hashes)
    source_fingerprints = {
        str(row["id"]): _source_fingerprint(dataset_dir, row) for row in rows
    }

    existing: dict[str, dict[str, Any]] = {}
    pending: list[Mapping[str, Any]] = []
    for row in rows:
        destination = cache_dir / f"{row['id']}.pt"
        if resolved.skip_existing and _cache_valid(
            destination, resolved.semantic_layer,
            source_fingerprint=source_fingerprints[str(row["id"])],
            extraction_fingerprint=extraction_fingerprint,
        ):
            existing[str(row["id"])] = _record_from_cache(row, destination)
        else:
            pending.append(row)
    pending.sort(key=lambda item: float(item.get("duration_s", 0.0)))

    models: _FeatureModels | None = None
    completed = len(existing)
    written = 0
    cancelled = _cancelled(cancel_callback, completed)
    if pending and not cancelled:
        models = _FeatureModels(resolved, active_reporter)
        active_reporter.set_stage("cache features")
        active_reporter.update(completed, desc=f"{completed} reused, {len(pending)} pending")
        for batch_start in range(0, len(pending), resolved.batch_size):
            if _cancelled(cancel_callback, completed):
                cancelled = True
                break
            batch_rows = pending[batch_start : batch_start + resolved.batch_size]
            waveforms: list[torch.Tensor] = []
            durations: list[float] = []
            for row in batch_rows:
                waveform, duration = _load_audio_16k(_audio_path(dataset_dir, row))
                waveforms.append(waveform)
                durations.append(duration)
            normalized_features = models.w2v_features(waveforms)
            for row, waveform, duration, normalized_feature in zip(
                batch_rows, waveforms, durations, normalized_features
            ):
                processed, tokens, language_id = text_pipeline.process(
                    str(row.get("text", "")), str(row.get("language", "EN"))
                )
                codes, campplus, emo_raw, emo_vec = models.item_features(
                    waveform, normalized_feature
                )
                destination = cache_dir / f"{row['id']}.pt"
                payload = {
                    "format": CACHE_FORMAT,
                    "version": CACHE_VERSION,
                    "id": str(row["id"]),
                    "text_tokens": tokens.cpu(),
                    "text_normalized": processed,
                    "codes": codes.cpu(),
                    "campplus": campplus.reshape(192).cpu(),
                    "emo_raw": emo_raw.reshape(1024).cpu(),
                    "emo_vec": emo_vec.reshape(-1).cpu(),
                    "duration_s": float(duration),
                    "n_codes": int(codes.numel()),
                    "n_text_tokens": int(tokens.numel()),
                    "speaker": str(row.get("speaker", "")),
                    "language": str(row.get("language", "EN")),
                    "lang_id": int(language_id),
                    "semantic_layer": resolved.semantic_layer,
                    "semantic_dtype": "fp32",
                    "source_fingerprint": source_fingerprints[str(row["id"])],
                    "extraction_fingerprint": extraction_fingerprint,
                }
                _atomic_torch_save(destination, payload)
                existing[str(row["id"])] = _record_from_cache(row, destination)
                written += 1
                completed += 1
                active_reporter.update(
                    completed,
                    total=len(rows),
                    desc=f"{row['id']} | {codes.numel()} codes | {tokens.numel()} text tokens",
                )
                if _cancelled(cancel_callback, completed):
                    cancelled = True
                    break
            if cancelled:
                break

    del models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ordered_records = [existing[str(row["id"])] for row in rows if str(row["id"]) in existing]
    write_cache_index(dataset_dir, ordered_records)
    statistics = _update_dataset_info(dataset_dir, ordered_records, resolved)
    cache_metadata = {
        "format": CACHE_FORMAT,
        "version": CACHE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_dir.name,
        "ids": [item["id"] for item in ordered_records],
        "records": ordered_records,
        "dtypes": {
            "text_tokens": "int32",
            "codes": "int16",
            "campplus": "float32",
            "emo_raw": "float32",
            "emo_vec": "float32",
        },
        "versions": {
            "cache": CACHE_VERSION,
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "torchaudio": _package_version("torchaudio"),
        },
        "semantic_layer": resolved.semantic_layer,
        "model_hashes": model_hashes,
        "semantic_dtype": "fp32",
        "extraction_fingerprint": extraction_fingerprint,
        "token_statistics": statistics,
        "cancelled": cancelled,
    }
    atomic_write_json(dataset_dir / "cache_index.json", cache_metadata)

    verification_files: list[str] = []
    if resolved.verify_count and not cancelled and ordered_records:
        active_reporter.log(">> feature cache complete; starting cached-code reconstruction checks")
        verification_files = verify_cached_codes(
            dataset_dir,
            count=min(resolved.verify_count, len(ordered_records)),
            model_dir=resolved.model_dir,
            model_config=resolved.model_config,
            device=resolved.device,
            output_dir=resolved.verify_output_dir,
        )
    if not cancelled:
        active_reporter.finish()
    elapsed = time.perf_counter() - started
    return FeatureCacheSummary(
        dataset_dir=str(dataset_dir),
        total=len(rows),
        cached=written,
        skipped=len(ordered_records) - written,
        cancelled=cancelled,
        max_codes=int(statistics["max_codes"]),
        mean_codes=float(statistics["mean_codes"]),
        max_text_tokens=int(statistics["max_text_tokens"]),
        mean_text_tokens=float(statistics["mean_text_tokens"]),
        over_length_count=int(statistics["over_length_count"]),
        elapsed_s=elapsed,
        verification_files=verification_files,
    )


@torch.no_grad()
def verify_cached_codes(
    dataset_dir: str | Path,
    *,
    count: int = 5,
    model_dir: str = "models",
    model_config: str = "models/config.yaml",
    device: str = "cuda:0",
    output_dir: str | Path = "outputs/_feature_check",
) -> list[str]:
    """Decode cached semantic codes through the deployed codec/s2mel/vocoder path."""

    from indextts.infer_v2_5 import IndexTTS2
    from indextts.runtime import RuntimeConfig
    from indextts.utils.common import save_pcm_wav

    root = Path(dataset_dir).expanduser().resolve()
    rows = load_manifest(root)[: max(0, int(count))]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    runtime = RuntimeConfig(device=device, model_variant="bf16", gpt_dtype="bf16")
    engine = IndexTTS2(
        cfg_path=model_config,
        model_dir=model_dir,
        runtime=runtime,
        use_qwen_emo=False,
    )
    outputs: list[str] = []
    report_rows: list[dict[str, Any]] = []
    for row in rows:
        cached = torch.load(root / "cache" / f"{row['id']}.pt", map_location="cpu", weights_only=False)
        source = _audio_path(root, row)
        audio, sample_rate = _read_audio(source)
        audio_16k = torchaudio.functional.resample(audio, sample_rate, 16000)
        audio_22k = torchaudio.functional.resample(audio, sample_rate, 22050)
        processed = engine.extract_features(
            audio_16k.squeeze(0), sampling_rate=16000, return_tensors="pt"
        )
        spk_feature = engine.get_emb(
            processed["input_features"].to(engine.device),
            processed["attention_mask"].to(engine.device),
            int(cached.get("semantic_layer", 17)),
        )
        ref_mel = engine.mel_fn(audio_22k.to(engine.device).float())
        ref_lengths = torch.tensor([ref_mel.shape[-1]], device=engine.device, dtype=torch.long)
        style = torch.as_tensor(cached["campplus"], device=engine.device).float().unsqueeze(0)
        with engine._use_s2mel():
            prompt_condition = engine.s2mel.models["length_regulator"](
                spk_feature, ylens=ref_lengths, n_quantizers=3, f0=None
            )[0]
        codes = torch.as_tensor(cached["codes"], device=engine.device).long().unsqueeze(0)
        with engine.residency.use("semantic_codec"):
            semantic = engine.semantic_codec.decode(codes)
        target_lengths = torch.tensor(
            [int(semantic.shape[1] * 1.72)], device=engine.device, dtype=torch.long
        )
        with engine._use_s2mel():
            condition = engine.s2mel.models["length_regulator"](
                semantic, ylens=target_lengths, n_quantizers=3, f0=None
            )[0]
            combined = torch.cat((prompt_condition, condition), dim=1)
            engine._setup_s2mel_caches(1, max(engine.runtime.cfm_cache_length, combined.shape[1]))
            mel = engine.s2mel.models["cfm"].inference(
                combined,
                torch.tensor([combined.shape[1]], device=engine.device, dtype=torch.long),
                ref_mel,
                style,
                None,
                25,
                inference_cfg_rate=0.7,
            )[:, :, ref_mel.shape[-1] :]
        with engine.residency.use("bigvgan"):
            waveform = engine.bigvgan(mel.float()).squeeze().unsqueeze(0)
        waveform = torch.clamp(waveform * 32767.0, -32767.0, 32767.0)
        output_path = destination / f"{row['id']}_cached_codes.wav"
        save_pcm_wav(output_path, waveform, 22050)
        decoded_duration = waveform.shape[-1] / 22050.0
        outputs.append(str(output_path.resolve()))
        report_rows.append(
            {
                "id": row["id"],
                "source_duration_s": float(cached.get("duration_s", row.get("duration_s", 0.0))),
                "decoded_duration_s": decoded_duration,
                "duration_ratio": decoded_duration
                / max(float(cached.get("duration_s", row.get("duration_s", 0.0))), 1e-6),
                "wav": output_path.name,
            }
        )
    atomic_write_json(destination / "report.json", {"items": report_rows})
    return outputs


__all__ = [
    "CACHE_FORMAT",
    "CACHE_VERSION",
    "FeatureCacheConfig",
    "FeatureCacheSummary",
    "TextFeaturePipeline",
    "cache_dataset_features",
    "verify_cached_codes",
]
