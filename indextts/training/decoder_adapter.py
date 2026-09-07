"""Adapt the semantic-to-mel decoder (the s2mel flow-matching DiT) to one voice with a LoRA / DoRA.

The GPT adapter decides what is said and when. The decoder turns semantic codes into the mel
spectrogram and therefore owns timbre and spectral detail, which no GPT adapter can change. In this
build the decoder is conditioned on the target's semantic codes, on an in-context prompt clip of the
same speaker (its mel and semantic features), and on the prompt's CAMPPlus style vector; the GPT's
latents are not an input. The decoder adapter is therefore trained from the cached dataset features
alone, with the pretrained flow-matching objective, using a different clip of the speaker as the prompt
for every target clip so it learns the voice instead of one reference recording.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import torch
import torchaudio

from indextts.lora.apply import inject_adapters
from indextts.lora.decoder import DECODER_ADAPTER_SUFFIX, DECODER_COMPONENT, decoder_target_modules, unwrap_estimator
from indextts.lora.io import LoraMetadata, save_lora
from indextts.runtime.progress import ProgressReporter
from indextts.utils.atomic_json import read_json_retry, write_json_atomic
from indextts.version import APP_VERSION

from .dataset import LoraTrainDataset, _stable_unit_interval
from .dataset_manifest import atomic_write_json
from .features import FeatureCacheConfig, _FeatureModels, _audio_path, _read_audio

MEL_SAMPLE_RATE = 22050
SEMANTIC_SAMPLE_RATE = 16000


@dataclass
class DecoderAdapterConfig:
    dataset_dir: str
    output_path: str
    name: str = ""
    model_dir: str = "models"
    model_config: str = "models/config.yaml"
    device: str = "cuda:0"
    adapter_type: str = "dora"
    rank: int = 128
    alpha: float = 128.0
    dropout: float = 0.0
    target_attention: bool = True
    target_mlp: bool = True
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    epochs: int = 10
    max_steps: int = 0
    max_grad_norm: float = 1.0
    val_fraction: float = 0.05
    val_split_mode: str = "source"
    # The flow-matching loss is dominated by the noise it regresses against, so adaptation shows up as
    # small, steady decreases: patience counts checks without a 0.0005 gain, after 1,000 updates and two
    # dataset passes, and a plateau first earns one lower-learning-rate trial, as in the GPT trainer.
    val_every_steps: int = 500
    val_max_items: int = 64
    early_stop_patience: int = 6
    early_stop_min_delta: float = 0.0005
    early_stop_min_steps: int = 1000
    early_stop_min_epochs: float = 2.0
    plateau_lr_enabled: bool = True
    plateau_lr_factor: float = 0.5
    plateau_lr_grace_steps: int = 1000
    # Selection follows speaker identity, not the flow loss: a few held-out clips are re-rendered from their
    # own codes at every check and compared with the real recordings by CAMPPlus similarity. An adapter
    # that does not beat the pretrained decoder by ``identity_min_gain`` is not installed.
    select_by: str = "identity"
    identity_clips: int = 8
    identity_steps: int = 16
    identity_cfg_rate: float = 0.7
    identity_min_gain: float = 0.005
    identity_min_delta: float = 0.002
    max_frames: int = 3200
    # Every target is paired with a random other clip of its speaker as the in-context prompt. One fixed
    # prompt would let the adapter learn a prompt-independent offset (the dataset's average pitch and
    # spectral tilt) that overrides whatever reference a user later supplies.
    min_prompt_seconds: float = 3.0
    max_prompt_seconds: float = 15.0
    mixed_precision: str = "bf16"
    seed: int = 42
    log_every_steps: int = 10
    cache_gb: float = 4.0
    max_codes: int = 1500
    max_text_tokens: int = 600

    def validate(self) -> "DecoderAdapterConfig":
        self.dataset_dir = str(self.dataset_dir or "")
        if not self.dataset_dir:
            raise ValueError("dataset_dir is required")
        self.output_path = str(self.output_path or "")
        if not self.output_path.lower().endswith(DECODER_ADAPTER_SUFFIX):
            raise ValueError(f"output_path must end with {DECODER_ADAPTER_SUFFIX}")
        self.name = str(self.name or Path(self.output_path).name[: -len(DECODER_ADAPTER_SUFFIX)])
        self.model_dir = str(self.model_dir or "models")
        self.model_config = str(self.model_config or Path(self.model_dir) / "config.yaml")
        self.device = str(self.device or "cuda:0")
        self.adapter_type = str(self.adapter_type).lower()
        if self.adapter_type not in {"lora", "dora"}:
            raise ValueError("adapter_type must be 'lora' or 'dora'")
        self.rank = max(1, int(self.rank))
        self.alpha = float(self.alpha)
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be a positive number")
        self.dropout = float(self.dropout)
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not (self.target_attention or self.target_mlp):
            raise ValueError("at least one decoder target group must be enabled")
        self.learning_rate = float(self.learning_rate)
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.weight_decay = max(0.0, float(self.weight_decay))
        self.warmup_steps = max(0, int(self.warmup_steps))
        self.epochs = max(1, int(self.epochs))
        self.max_steps = max(0, int(self.max_steps))
        self.max_grad_norm = max(0.0, float(self.max_grad_norm))
        self.val_fraction = min(0.5, max(0.0, float(self.val_fraction)))
        self.val_split_mode = str(self.val_split_mode).lower()
        if self.val_split_mode not in {"record", "source"}:
            raise ValueError("val_split_mode must be record or source")
        self.val_every_steps = max(0, int(self.val_every_steps))
        self.val_max_items = max(1, int(self.val_max_items))
        self.early_stop_patience = max(0, int(self.early_stop_patience))
        self.early_stop_min_delta = max(0.0, float(self.early_stop_min_delta))
        self.early_stop_min_steps = max(0, int(self.early_stop_min_steps))
        self.early_stop_min_epochs = max(0.0, float(self.early_stop_min_epochs))
        self.plateau_lr_enabled = bool(self.plateau_lr_enabled)
        self.plateau_lr_factor = float(self.plateau_lr_factor)
        if not 0.0 < self.plateau_lr_factor < 1.0:
            raise ValueError("plateau_lr_factor must be between 0 and 1")
        self.plateau_lr_grace_steps = max(0, int(self.plateau_lr_grace_steps))
        self.select_by = str(self.select_by).lower()
        if self.select_by not in {"identity", "loss"}:
            raise ValueError("select_by must be identity or loss")
        self.identity_clips = max(0, int(self.identity_clips))
        self.identity_steps = max(2, int(self.identity_steps))
        self.identity_cfg_rate = max(0.0, float(self.identity_cfg_rate))
        self.identity_min_gain = max(0.0, float(self.identity_min_gain))
        self.identity_min_delta = max(0.0, float(self.identity_min_delta))
        self.max_frames = max(400, int(self.max_frames))
        self.min_prompt_seconds = max(0.5, float(self.min_prompt_seconds))
        self.max_prompt_seconds = max(self.min_prompt_seconds, float(self.max_prompt_seconds))
        self.mixed_precision = str(self.mixed_precision).lower()
        if self.mixed_precision not in {"bf16", "fp32", "no"}:
            raise ValueError("mixed_precision must be bf16, fp32, or no")
        self.seed = int(self.seed)
        self.log_every_steps = max(1, int(self.log_every_steps))
        self.cache_gb = max(0.0, float(self.cache_gb))
        self.max_codes = max(1, int(self.max_codes))
        self.max_text_tokens = max(1, int(self.max_text_tokens))
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | "DecoderAdapterConfig") -> "DecoderAdapterConfig":
        if isinstance(value, cls):
            return cls(**asdict(value)).validate()
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: item for key, item in value.items() if key in allowed}).validate()

    @classmethod
    def from_json(cls, path: str | Path) -> "DecoderAdapterConfig":
        with Path(path).open("r", encoding="utf-8-sig") as handle:
            return cls.from_dict(json.load(handle))


@dataclass
class DecoderAdapterResult:
    status: str
    output_path: str
    steps: int
    total_steps: int
    epochs_completed: float
    best_step: int
    best_val_loss: float | None
    first_loss: float | None
    final_loss: float | None
    elapsed_s: float
    pairs_skipped: int
    target_modules: int
    message: str = ""
    initial_val_loss: float | None = None
    initial_identity: float | None = None
    best_identity: float | None = None
    accepted: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decoder_report_path(output_path: str | Path) -> Path:
    return Path(output_path).expanduser().resolve().parent / "analysis" / "decoder_adapter.json"


def load_decoder_report(adapter_dir: str | Path) -> dict[str, Any] | None:
    value = read_json_retry(Path(adapter_dir) / "analysis" / "decoder_adapter.json", None)
    return value if isinstance(value, dict) else None


def reject_decoder_adapter(output_path: str | Path) -> Path:
    """Move a decoder adapter out of the paths generation searches, keeping the file for inspection."""
    source = Path(output_path).expanduser().resolve()
    destination = source.parent / "analysis" / f"{source.name[: -len(DECODER_ADAPTER_SUFFIX)]}.s2mel.rejected"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    source.replace(destination)
    return destination


class _StatusWriter:
    def __init__(self, state_dir: Path, log_fn: Callable[[str], None] | None) -> None:
        self.state_dir = state_dir
        self.status_path = state_dir / "status.json"
        self.log_path = state_dir / "log.txt"
        self._log_fn = log_fn
        self.started = time.perf_counter()
        self._current: dict[str, Any] = {}

    def write(self, **updates: Any) -> None:
        self._current.update(updates)
        self._current.update({"elapsed_s": time.perf_counter() - self.started, "updated_at": time.time()})
        write_json_atomic(self.status_path, self._current, indent=2, ensure_ascii=False)

    def log(self, message: str) -> None:
        line = str(message)
        with self.log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
        if self._log_fn is not None:
            self._log_fn(line)
        else:
            print(line, flush=True)


class DecoderAdapterTrainer:
    """Train a LoRA / DoRA on the s2mel DiT from a cached training dataset."""

    def __init__(self, config: DecoderAdapterConfig, state_dir: str | Path, *,
                 log_fn: Callable[[str], None] | None = None,
                 cancel_callback: Callable[[], bool] | None = None) -> None:
        self.config = DecoderAdapterConfig.from_dict(config)
        self.state_dir = Path(state_dir).expanduser().resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.status = _StatusWriter(self.state_dir, log_fn)
        self.cancel_callback = cancel_callback or (lambda: False)
        self.device = torch.device(self.config.device)
        self._prompt_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._target_cache: dict[str, torch.Tensor] = {}
        self._cache_bytes = 0
        self._cache_limit = int(self.config.cache_gb * 1024 ** 3)

    # ------------------------------------------------------------------ models
    def _load_models(self) -> None:
        from omegaconf import OmegaConf
        from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
        from indextts.s2mel.modules.audio import mel_spectrogram

        config = self.config
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was requested for decoder adaptation, but is unavailable: {config.device}")
        self.status.write(phase="initializing", message="loading semantic encoder and codec")
        self.status.log(">> decoder adaptation: loading the semantic encoder, codec, and CAMPPlus")
        feature_config = FeatureCacheConfig(dataset_dir=config.dataset_dir, model_dir=config.model_dir,
                                            model_config=config.model_config, device=config.device)
        self.features = _FeatureModels(feature_config, ProgressReporter("decoder features"))
        cfg = OmegaConf.load(config.model_config)
        self.status.write(message="loading the voice decoder")
        s2mel = MyModel(cfg.s2mel)
        s2mel, _, _, _ = load_checkpoint2(s2mel, None, str(Path(config.model_dir) / str(cfg.s2mel_checkpoint)),
                                          load_only_params=True, ignore_modules=[], is_distributed=False)
        self.s2mel = s2mel.float().to(self.device)
        self.s2mel.eval()
        for parameter in self.s2mel.parameters():
            parameter.requires_grad_(False)
        self.cfm = self.s2mel.models["cfm"]
        self.regulator = self.s2mel.models["length_regulator"]
        self.estimator = unwrap_estimator(self.cfm.estimator)
        spect = cfg.s2mel["preprocess_params"]["spect_params"]
        fmax = spect.get("fmax", "None")
        mel_args = {"n_fft": int(spect["n_fft"]), "win_size": int(spect["win_length"]), "hop_size": int(spect["hop_length"]),
                    "num_mels": int(spect["n_mels"]), "sampling_rate": int(cfg.s2mel["preprocess_params"]["sr"]),
                    "fmin": spect.get("fmin", 0), "fmax": None if fmax in (None, "None") else 8000, "center": False}
        self.mel_sample_rate = int(mel_args["sampling_rate"])
        self.hop_size = int(mel_args["hop_size"])
        self.mel_fn = lambda wave: mel_spectrogram(wave, **mel_args)

        self.vocoder = None
        if config.select_by == "identity" and config.identity_clips > 0:
            from indextts.s2mel.modules.bigvgan import bigvgan as bigvgan_module

            vocoder_dir = Path(config.model_dir) / "hf_cache" / "bigvgan"
            if not vocoder_dir.is_dir():
                from indextts.utils.model_download import ensure_models_available

                vocoder_dir = Path(ensure_models_available(config.model_dir)["bigvgan"])
            self.status.write(message="loading the vocoder for identity checks")
            self.vocoder = bigvgan_module.BigVGAN.from_pretrained(str(vocoder_dir), use_cuda_kernel=False).to(self.device)
            self.vocoder.remove_weight_norm()
            self.vocoder.eval()
        targets = decoder_target_modules(self.estimator, attention=config.target_attention, mlp=config.target_mlp)
        if not targets:
            raise RuntimeError("the voice decoder has no projection layers that accept a LoRA / DoRA")
        self.adapters = inject_adapters(self.estimator, rank=config.rank, alpha=config.alpha, dropout=config.dropout,
                                        use_dora=config.adapter_type == "dora", target_modules=targets)
        self.estimator.to(self.device)
        self.trainable: list[torch.nn.Parameter] = []
        for adapter in self.adapters.values():
            for name, parameter in adapter.named_parameters():
                if name.startswith("lora_"):
                    parameter.requires_grad_(True)
                    self.trainable.append(parameter)
        self.estimator.setup_caches(2 if config.identity_cfg_rate > 0 else 1, config.max_frames + 16)
        self.estimator.train()
        self.regulator.eval()
        count = sum(parameter.numel() for parameter in self.trainable)
        self.status.log(f">> decoder adapter: {config.adapter_type.upper()} rank {config.rank} alpha {config.alpha:g} on "
                        f"{len(targets)} projections ({count / 1e6:.2f}M trainable parameters)")

    # ------------------------------------------------------------------- data
    def _datasets(self) -> tuple[LoraTrainDataset, LoraTrainDataset | None]:
        config = self.config
        common = dict(val_fraction=config.val_fraction, seed=config.seed, max_codes=config.max_codes,
                      max_text_tokens=config.max_text_tokens, speaker_ref_mode="other", emo_ref_mode="follow_speaker",
                      val_split_mode=config.val_split_mode)
        train = LoraTrainDataset(config.dataset_dir, split="train", **common)
        try:
            val: LoraTrainDataset | None = LoraTrainDataset(config.dataset_dir, split="val", **common)
            if len(val) == 0:
                val = None
        except ValueError:
            val = None
        if val is None and config.val_split_mode == "source" and config.val_fraction > 0:
            # A single-recording dataset cannot hold out a whole recording; hold out clips instead,
            # as the GPT trainer does, so the run still has a held-out loss to select and stop on.
            fallback = {**common, "val_split_mode": "record"}
            try:
                train = LoraTrainDataset(config.dataset_dir, split="train", **fallback)
                val = LoraTrainDataset(config.dataset_dir, split="val", **fallback)
                if len(val) == 0:
                    val = None
                else:
                    self.status.log(">> decoder validation falls back to held-out clips: the dataset has too few recordings for a source split")
            except ValueError:
                val = None
        return train, val

    def _build_prompt_pool(self, train: LoraTrainDataset) -> None:
        """Training clips per speaker that can serve as in-context prompts (never validation audio)."""
        config = self.config
        self._pool: dict[str, list[Mapping[str, Any]]] = {}
        for record in train.records:
            try:
                duration = float(record.get("duration_s") or 0.0)
            except (TypeError, ValueError):
                duration = 0.0
            if duration < config.min_prompt_seconds or duration > config.max_prompt_seconds:
                continue
            self._pool.setdefault(str(record["speaker"]), []).append(record)
        # Ordered by id so the seeded choice below is repeatable across runs.
        for speaker in self._pool:
            self._pool[speaker].sort(key=lambda item: str(item["id"]))

    def _sample_prompt(self, record: Mapping[str, Any], *parts: Any) -> Mapping[str, Any] | None:
        """A different clip of the record's speaker, chosen at random but repeatably for the given parts."""
        choices = self._pool.get(str(record["speaker"]), [])
        if len(choices) > 1:
            choices = [item for item in choices if item["id"] != record["id"]]
        elif choices and choices[0]["id"] == record["id"]:
            choices = []
        if not choices:
            return None
        unit = _stable_unit_interval(self.config.seed, "decoder_prompt", *parts)
        return choices[min(len(choices) - 1, int(unit * len(choices)))]

    def _remember(self, size: int) -> bool:
        if self._cache_bytes + size > self._cache_limit:
            return False
        self._cache_bytes += size
        return True

    def _load_clip(self, record: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """(22.05 kHz waveform, 16 kHz waveform) as (1, N) float tensors on the training device."""
        waveform, sample_rate = _read_audio(_audio_path(Path(self.config.dataset_dir), record))
        waveform = waveform.float().clamp_(-1.0, 1.0)
        mel_wave = waveform if sample_rate == self.mel_sample_rate else torchaudio.functional.resample(waveform, sample_rate, self.mel_sample_rate)
        semantic_wave = waveform if sample_rate == SEMANTIC_SAMPLE_RATE else torchaudio.functional.resample(waveform, sample_rate, SEMANTIC_SAMPLE_RATE)
        return mel_wave.to(self.device), semantic_wave.contiguous()

    @torch.no_grad()
    def _target(self, record: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """(mel (1, 80, T), semantic content (1, Ts, D)) of a target clip."""
        record_id = str(record["id"])
        cached = torch.load(record["cache_path"], map_location="cpu", weights_only=False)
        codes = torch.as_tensor(cached["codes"], dtype=torch.long).flatten().unsqueeze(0).to(self.device)
        if record_id in self._target_cache:
            mel = self._target_cache[record_id].to(self.device, dtype=torch.float32)
        else:
            mel_wave, _ = self._load_clip(record)
            mel = self.mel_fn(mel_wave)
            stored = mel.to(torch.float16).cpu()
            if self._remember(stored.numel() * 2):
                self._target_cache[record_id] = stored
        semantic = self.features.codec.decode(codes)
        return mel, semantic

    @torch.no_grad()
    def _prompt(self, record: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(mel (1, 80, Tp), semantic features (1, Tw, 1024), style (1, 192)) of a prompt clip."""
        record_id = str(record["id"])
        if record_id in self._prompt_cache:
            mel, feature, style = self._prompt_cache[record_id]
            return (mel.to(self.device, dtype=torch.float32), feature.to(self.device, dtype=torch.float32),
                    style.to(self.device, dtype=torch.float32))
        mel_wave, semantic_wave = self._load_clip(record)
        mel = self.mel_fn(mel_wave)
        feature = self.features.w2v_features([semantic_wave])[0].to(dtype=torch.float32)
        while feature.dim() > 2:
            feature = feature.squeeze(0)
        feature = feature.unsqueeze(0)  # (1, frames, 1024), the layout the length regulator expects
        cached = torch.load(record["cache_path"], map_location="cpu", weights_only=False)
        style = torch.as_tensor(cached["campplus"], dtype=torch.float32).flatten().unsqueeze(0).to(self.device)
        stored = (mel.to(torch.float16).cpu(), feature.to(torch.float16).cpu(), style.cpu())
        if self._remember(sum(item.numel() * 2 for item in stored)):
            self._prompt_cache[record_id] = stored
        return mel, feature, style

    def _example(self, target: Mapping[str, Any], prompt: Mapping[str, Any]) -> dict[str, torch.Tensor] | None:
        """One in-context training example: prompt mel and content followed by the target's."""
        target_mel, target_semantic = self._target(target)
        prompt_mel, prompt_feature, style = self._prompt(prompt)
        target_frames, prompt_frames = int(target_mel.shape[-1]), int(prompt_mel.shape[-1])
        if prompt_frames < int(self.config.min_prompt_seconds * self.mel_sample_rate / self.hop_size):
            return None
        if target_frames + prompt_frames > self.config.max_frames:
            return None
        with torch.no_grad():
            target_cond = self.regulator(target_semantic, ylens=torch.tensor([target_frames], device=self.device),
                                         n_quantizers=3, f0=None)[0]
            prompt_cond = self.regulator(prompt_feature, ylens=torch.tensor([prompt_frames], device=self.device),
                                         n_quantizers=3, f0=None)[0]
        return {
            "x1": torch.cat([prompt_mel, target_mel], dim=2).float(),
            "mu": torch.cat([prompt_cond, target_cond], dim=1).float(),
            "x_lens": torch.tensor([prompt_frames + target_frames], dtype=torch.long, device=self.device),
            "prompt_lens": torch.tensor([prompt_frames], dtype=torch.long, device=self.device),
            "style": style,
        }

    def _loss(self, example: dict[str, torch.Tensor], generator: torch.Generator | None = None) -> torch.Tensor:
        """Conditional flow-matching loss on the target region, as the decoder was pretrained.

        Mirrors ``CFM.forward`` but calls the DiT with its documented arguments, so validation in eval
        mode is a real conditional loss, and accepts a generator so validation noise is repeatable.
        """
        x1, mu, style = example["x1"], example["mu"], example["style"]
        x_lens, prompt_lens = example["x_lens"], example["prompt_lens"]
        prompt_frames, total_frames = int(prompt_lens[0]), int(x_lens[0])
        sigma_min = float(getattr(self.cfm, "sigma_min", 1e-6))
        t = torch.rand([1, 1, 1], device=x1.device, dtype=x1.dtype, generator=generator)
        z = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=generator)
        y = (1 - (1 - sigma_min) * t) * z + t * x1
        u = x1 - (1 - sigma_min) * z
        prompt = torch.zeros_like(x1)
        prompt[:, :, :prompt_frames] = x1[:, :, :prompt_frames]
        y[:, :, :prompt_frames] = 0
        use_autocast = self.device.type == "cuda" and self.config.mixed_precision == "bf16"
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=use_autocast):
            estimated = self.estimator(y, prompt, x_lens, t.squeeze(1).squeeze(1), style, mu)
        criterion = getattr(self.cfm, "criterion", None) or torch.nn.L1Loss()
        return criterion(estimated[:, :, prompt_frames:total_frames].float(), u[:, :, prompt_frames:total_frames])

    # --------------------------------------------------------------- identity
    def _set_adapter_strength(self, strength: float) -> None:
        for adapter in self.adapters.values():
            adapter.strength = float(strength)

    @torch.no_grad()
    def _render_identity(self, example: dict[str, torch.Tensor], *, adapted: bool) -> torch.Tensor:
        """Render the target region of an example from its codes, as inference would, and embed the audio."""
        config = self.config
        prompt_frames = int(example["prompt_lens"][0])
        prompt_mel = example["x1"][:, :, :prompt_frames]
        # Same guidance as the engine: the conditioned branch through the adapter, the unconditional branch
        # through the pretrained decoder.
        use_toggle = adapted and config.identity_cfg_rate > 0
        self.cfm.guidance_adapter_toggle = (lambda enabled: self._set_adapter_strength(1.0 if enabled else 0.0)) if use_toggle else None
        try:
            mel = self.cfm.inference(example["mu"].clone(), example["x_lens"], prompt_mel, example["style"], None,
                                     config.identity_steps, temperature=1.0, inference_cfg_rate=config.identity_cfg_rate)
        finally:
            self.cfm.guidance_adapter_toggle = None
            self._set_adapter_strength(1.0 if adapted else 0.0)
        mel = mel[:, :, prompt_frames:].float()
        wave = self.vocoder(mel).squeeze(1).clamp(-1.0, 1.0)
        wave16 = torchaudio.functional.resample(wave, self.mel_sample_rate, SEMANTIC_SAMPLE_RATE)
        fbank = torchaudio.compliance.kaldi.fbank(wave16.cpu(), num_mel_bins=80, dither=0, sample_frequency=SEMANTIC_SAMPLE_RATE)
        fbank = fbank - fbank.mean(dim=0, keepdim=True)
        embedding = self.features.campplus(fbank.unsqueeze(0).to(self.device)).squeeze(0).float()
        return torch.nn.functional.normalize(embedding, dim=0)

    def _identity(self, pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]], *, adapted: bool = True) -> float | None:
        """Mean CAMPPlus similarity between re-rendered held-out clips and the real recordings."""
        if self.vocoder is None or not pairs:
            return None
        self.estimator.eval()
        self._set_adapter_strength(1.0 if adapted else 0.0)
        similarities: list[float] = []
        try:
            for target, prompt in pairs:
                example = self._example(target, prompt)
                if example is None:
                    continue
                # The solver draws its own noise; the same seed per clip makes checks comparable.
                torch.manual_seed(self.config.seed + 1)
                if self.device.type == "cuda":
                    torch.cuda.manual_seed_all(self.config.seed + 1)
                cached = torch.load(target["cache_path"], map_location="cpu", weights_only=False)
                real = torch.nn.functional.normalize(torch.as_tensor(cached["campplus"], dtype=torch.float32).flatten().to(self.device), dim=0)
                similarities.append(float(torch.dot(self._render_identity(example, adapted=adapted), real)))
        finally:
            self._set_adapter_strength(1.0)
            self.estimator.train()
        return sum(similarities) / len(similarities) if similarities else None

    # --------------------------------------------------------------- training
    def _validate(self, pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> float | None:
        if not pairs:
            return None
        self.estimator.eval()
        losses: list[float] = []
        generator = torch.Generator(device=self.device).manual_seed(self.config.seed)
        with torch.no_grad():
            for target, prompt in pairs:
                example = self._example(target, prompt)
                if example is None:
                    continue
                losses.append(float(self._loss(example, generator)))
        self.estimator.train()
        return sum(losses) / len(losses) if losses else None

    def _save(self, step: int, epochs_completed: float, best_val_loss: float | None, best_identity: float | None = None) -> Path:
        config = self.config
        metadata = LoraMetadata(
            adapter_type=config.adapter_type, rank=config.rank, alpha=config.alpha, dropout=config.dropout,
            target_modules=list(self.adapters), base_model="IndexTeam/IndexTTS-2.5", base_variant="fp32",
            trained_steps=int(step), epochs=int(math.ceil(epochs_completed)), dataset_name=Path(config.dataset_dir).name,
            created_at=datetime.now(timezone.utc).isoformat(), app_version=APP_VERSION,
            train_config={"component": DECODER_COMPONENT, "best_val_loss": best_val_loss, "best_identity": best_identity,
                          **config.to_dict()},
            recommended_reference="", sample_rate=self.mel_sample_rate)
        destination = Path(config.output_path).expanduser().resolve()
        save_lora(destination, self.adapters, {}, metadata, dtype=torch.bfloat16)
        return destination

    def run(self) -> DecoderAdapterResult:
        config = self.config
        started = time.perf_counter()
        self._load_models()
        train, val = self._datasets()
        total_steps = len(train) * config.epochs
        if config.max_steps:
            total_steps = min(total_steps, config.max_steps)
        self._build_prompt_pool(train)
        if not any(self._pool.values()):
            raise RuntimeError(f"no training clip between {config.min_prompt_seconds:g} and {config.max_prompt_seconds:g} seconds "
                               "can serve as a decoder prompt")
        val_pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        if val is not None:
            for record in val.records[: config.val_max_items]:
                prompt = self._sample_prompt(record, "val", record["id"])  # fixed per target across checks
                if prompt is not None:
                    val_pairs.append((record, prompt))
        prompt_count = sum(len(items) for items in self._pool.values())
        self.status.log(f">> decoder adaptation plan | {len(train)} training clips, {prompt_count} prompt clips, "
                        f"{len(val_pairs)} validation pairs | {total_steps} updates over {config.epochs} epochs | "
                        f"learning rate {config.learning_rate:g}")
        optimizer = torch.optim.AdamW(self.trainable, lr=config.learning_rate, betas=(0.9, 0.99), eps=1e-8,
                                      weight_decay=config.weight_decay)
        warmup = min(config.warmup_steps, max(1, total_steps // 10))

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, total_steps - warmup)
            return max(0.02, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

        refinement = {"factor": 1.0}  # the plateau trial scales the remaining schedule

        def refined_lr_lambda(step: int) -> float:
            return lr_lambda(step) * refinement["factor"]

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, refined_lr_lambda)
        step = 0
        skipped = 0
        first_loss: float | None = None
        last_loss: float | None = None
        ema: float | None = None
        best_val: float | None = None  # held-out flow loss at the selected step
        best_identity: float | None = None  # speaker similarity at the selected step
        best_score: float | None = None  # the selection score (identity, or the negated loss)
        best_step = 0
        bad_checks = 0
        lr_reductions = 0
        cooldown_until = 0
        stopped = False
        status_message = "training complete"
        val_history: list[dict[str, float]] = []
        min_epoch_step = int(math.ceil(config.early_stop_min_epochs * max(1, len(train))))
        output_path = Path(config.output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.unlink(missing_ok=True)  # only a checkpoint that beats the pretrained decoder may remain
        generator = torch.Generator().manual_seed(config.seed)
        epochs_completed = 0.0
        identity_pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        identity_source = ""
        if self.vocoder is not None:
            if val_pairs:
                identity_pairs, identity_source = val_pairs[: config.identity_clips], "held-out clips"
            else:
                # A dataset too small for a validation split still gets an identity check, on training
                # clips: whether the rendered voice moves toward the recordings is measurable either way.
                for record in train.records:
                    prompt = self._sample_prompt(record, "identity", record["id"])
                    if prompt is not None:
                        identity_pairs.append((record, prompt))
                    if len(identity_pairs) >= config.identity_clips:
                        break
                identity_source = "training clips"
        select_identity = config.select_by == "identity" and bool(identity_pairs)
        initial_val: float | None = None
        initial_identity: float | None = None
        if val_pairs:
            # The untrained adapter is the identity, so this is the pretrained decoder's loss on this voice.
            self.status.write(phase="validating", message="measuring the pretrained decoder on held-out clips")
            initial_val = self._validate(val_pairs)
            if initial_val is not None:
                self.status.log(f">> decoder validation | step 0 | loss {initial_val:.4f} (pretrained decoder, no adaptation)")
        if identity_pairs:
            self.status.write(phase="validating", message="rendering clips with the pretrained decoder for the identity baseline")
            initial_identity = self._identity(identity_pairs, adapted=False)
            if initial_identity is not None:
                self.status.log(f">> decoder identity | step 0 | speaker similarity to the real recordings {initial_identity:.4f} "
                                f"(pretrained decoder, {len(identity_pairs)} {identity_source})")
        if initial_val is not None or initial_identity is not None:
            val_history.append({"step": 0, "val_loss": initial_val, "identity": initial_identity})
        self.status.write(phase="training", step=0, total_steps=total_steps, epoch=0, total_epochs=config.epochs,
                          message="adapting the voice decoder", output_path=str(output_path), initial_val_loss=initial_val,
                          initial_identity=initial_identity, selection="identity" if select_identity else "loss")

        def check_validation(force: bool = False) -> bool:
            """Measure the held-out loss and speaker identity, keep the best file, and decide whether to stop.

            Selection follows speaker similarity of re-rendered clips to the real recordings when it is
            available, and the flow loss otherwise. Mirrors the GPT trainer: a check counts against patience
            only after the minimum updates and dataset passes; the first exhausted patience halves the
            learning rate once and grants a grace period; the second stops the run.
            """
            nonlocal best_val, best_identity, best_score, best_step, bad_checks, lr_reductions, cooldown_until
            if not val_pairs and not identity_pairs:
                if force:
                    self._save(step, epochs_completed, None, None)
                return False
            loss: float | None = None
            identity: float | None = None
            if val_pairs:
                self.status.write(phase="validating", message="measuring held-out flow-matching loss")
                loss = self._validate(val_pairs)
            if identity_pairs:
                self.status.write(phase="validating", message="rendering clips to measure speaker identity")
                identity = self._identity(identity_pairs, adapted=True)
            self.status.write(phase="training")
            if loss is None and identity is None:
                return False
            val_history.append({"step": step, "val_loss": loss, "identity": identity, "lr": float(scheduler.get_last_lr()[0])})
            if select_identity and identity is not None:
                score, min_delta = identity, config.identity_min_delta
                # Higher similarity with a flow loss above the pretrained decoder's means the adapter is
                # leaving the decoder's own objective (an over-fitted texture, not the voice): not kept.
                sane = loss is None or initial_val is None or loss <= initial_val
            else:
                score, min_delta = -(loss if loss is not None else math.inf), config.early_stop_min_delta
                sane = True
            improved = sane and (best_score is None or score > best_score + min_delta)
            if sane and (best_score is None or score > best_score):
                best_score, best_step, best_val, best_identity = score, step, loss, identity
                self._save(step, epochs_completed, best_val, best_identity)
            counted = step >= config.early_stop_min_steps and step >= min_epoch_step and step >= cooldown_until
            if improved:
                bad_checks = 0
            elif counted:
                bad_checks += 1
            self.status.log(f">> decoder validation | step {step}"
                            + (f" | loss {loss:.4f}" if loss is not None else "")
                            + (f" | identity {identity:.4f}" if identity is not None else "")
                            + (" (not kept: the flow loss is above the pretrained decoder's)" if not sane else "")
                            + f" | best at step {best_step}"
                            + (f" (identity {best_identity:.4f})" if select_identity and best_identity is not None
                               else (f" (loss {best_val:.4f})" if best_val is not None else ""))
                            + f" | stalled checks {bad_checks}/{config.early_stop_patience}"
                            + ("" if counted else " (not counted yet)"))
            self.status.write(val_loss=loss, identity=identity, best_step=best_step, best_val_loss=best_val,
                              best_identity=best_identity, lr_reductions=lr_reductions)
            if not config.early_stop_patience or bad_checks < config.early_stop_patience:
                return False
            remaining = total_steps - step
            if config.plateau_lr_enabled and lr_reductions == 0 and remaining > config.plateau_lr_grace_steps:
                refinement["factor"] *= config.plateau_lr_factor
                for group in optimizer.param_groups:  # the scheduler applies the new factor from the next update on
                    group["lr"] *= config.plateau_lr_factor
                lr_reductions += 1
                bad_checks = 0
                cooldown_until = step + config.plateau_lr_grace_steps
                self.status.log(f">> decoder validation plateau: learning rate scaled by {config.plateau_lr_factor:g}; "
                                f"patience resumes at step {cooldown_until}")
                return False
            return True

        for epoch in range(config.epochs):
            train.set_epoch(epoch)
            order = torch.randperm(len(train), generator=generator).tolist()
            for position, index in enumerate(order):
                if step >= total_steps:
                    break
                if self.cancel_callback():
                    stopped = True
                    status_message = "stopped by request; the best decoder adapter so far is kept"
                    break
                record = train.records[index]
                prompt = self._sample_prompt(record, "train", epoch, record["id"])  # a new clip every epoch
                example = self._example(record, prompt) if prompt is not None else None
                if example is None:
                    skipped += 1
                    continue
                step_started = time.perf_counter()
                try:
                    loss = self._loss(example)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite decoder loss at step {step}")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                except torch.OutOfMemoryError:
                    # Smaller GPUs: drop this pair, free the cache, and shorten the frame limit so
                    # later pairs fit; the run continues instead of failing.
                    optimizer.zero_grad(set_to_none=True)
                    del example
                    gc.collect()
                    torch.cuda.empty_cache()
                    skipped += 1
                    config.max_frames = max(400, int(config.max_frames * 0.8))
                    self.status.log(f">> decoder step {step}: out of GPU memory; frame limit lowered to {config.max_frames}")
                    continue
                if config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.trainable, config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                step += 1
                epochs_completed = epoch + (position + 1) / max(1, len(order))
                value = float(loss.detach())
                first_loss = value if first_loss is None else first_loss
                last_loss = value
                ema = value if ema is None else 0.98 * ema + 0.02 * value
                if step % config.log_every_steps == 0 or step == 1:
                    elapsed = time.perf_counter() - started
                    speed = 1.0 / max(1e-6, time.perf_counter() - step_started)
                    remaining = (total_steps - step) / max(1e-6, speed)
                    self.status.write(phase="training", step=step, total_steps=total_steps, epoch=epoch + 1,
                                      total_epochs=config.epochs, loss=value, avg_loss=ema, lr=float(scheduler.get_last_lr()[0]),
                                      it_s=speed, eta_s=remaining, message=f"adapting the voice decoder | step {step}/{total_steps}")
                    if step % (config.log_every_steps * 10) == 0 or step == 1:
                        self.status.log(f">> decoder step {step}/{total_steps} | loss {value:.4f} | avg {ema:.4f} | "
                                        f"lr {scheduler.get_last_lr()[0]:.2e} | {speed:.2f} it/s | elapsed {elapsed:.0f}s")
                epoch_end = position == len(order) - 1
                if (config.val_every_steps and step % config.val_every_steps == 0) or (epoch_end and step < total_steps):
                    # Every N updates and at each epoch boundary, as the GPT trainer does, so small
                    # datasets still get a check per pass.
                    if check_validation():
                        stopped = True
                        status_message = (f"early stopping at step {step}: {config.early_stop_patience} validation checks "
                                          f"without an improvement greater than {config.early_stop_min_delta:g}")
                        break
            if stopped or step >= total_steps:
                break
        if step == 0:
            raise RuntimeError("no decoder training example fit the limits; nothing was trained")
        if not stopped or "early stopping" in status_message:
            if not val_history or val_history[-1]["step"] != step:
                check_validation(force=True)
        measured = bool(val_pairs or identity_pairs)
        if not measured and not output_path.is_file():
            self._save(step, epochs_completed, None, None)
        # Acceptance: the adapter is installed only when it beats the pretrained decoder on the selection
        # measure, so a run that did not help leaves the pretrained decoder in place.
        accepted = True
        rejection = ""
        if measured and not output_path.is_file():
            accepted, rejection = False, "no checkpoint improved on the pretrained decoder"
        elif select_identity and initial_identity is not None and best_identity is not None:
            accepted = best_identity >= initial_identity + config.identity_min_gain
            if not accepted:
                rejection = (f"speaker similarity to the real recordings {best_identity:.4f} did not beat the pretrained "
                             f"decoder's {initial_identity:.4f} by {config.identity_min_gain:g}")
        elif initial_val is not None and best_val is not None:
            accepted = best_val < initial_val
            if not accepted:
                rejection = f"the held-out flow loss {best_val:.4f} did not improve on the pretrained decoder's {initial_val:.4f}"
        if not accepted:
            output_path.unlink(missing_ok=True)
            status_message = f"not installed: {rejection}; the pretrained decoder is kept"
        elapsed = time.perf_counter() - started
        if not accepted:
            status = "rejected"
        else:
            status = "stopped" if stopped and "early stopping" not in status_message else "complete"
        result = DecoderAdapterResult(status=status, output_path=str(output_path) if accepted else "", steps=step,
                                      total_steps=total_steps, epochs_completed=round(epochs_completed, 3), best_step=best_step,
                                      best_val_loss=best_val, first_loss=first_loss, final_loss=last_loss, elapsed_s=elapsed,
                                      pairs_skipped=skipped, target_modules=len(self.adapters), message=status_message,
                                      initial_val_loss=initial_val, initial_identity=initial_identity, best_identity=best_identity,
                                      accepted=accepted)
        report_path = decoder_report_path(output_path)
        atomic_write_json(report_path, {
            **result.to_dict(), "config": config.to_dict(), "validation": val_history, "lr_reductions": lr_reductions,
            "selection": "identity" if select_identity else "loss", "identity_clips": len(identity_pairs),
            "identity_source": identity_source, "target_modules": list(self.adapters),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "note": ("identity: CAMPPlus similarity between clips re-rendered from their own codes (a different clip of the "
                     "speaker as the prompt) and the real recordings, higher is better; val_loss: flow-matching loss on "
                     "held-out clips, lower is better, a spectrogram regression rather than a listening score."),
        })
        lines = [f"**Voice decoder adapter: {config.adapter_type.upper()} rank {config.rank}, alpha {config.alpha:g}, "
                 f"{len(self.adapters)} projections; {status_message}.**", ""]
        if initial_identity is not None and best_identity is not None:
            lines.append(f"Speaker similarity of re-rendered {identity_source} to the real recordings: {initial_identity:.4f} with the "
                         f"pretrained decoder, {best_identity:.4f} with the adapter at update {best_step:,} "
                         f"({len(identity_pairs)} clips, {config.identity_steps} solver steps, guidance {config.identity_cfg_rate:g}).")
        if initial_val is not None and best_val is not None:
            gain = 100.0 * (initial_val - best_val) / initial_val
            lines.append(f"Held-out flow-matching loss {initial_val:.4f} with the pretrained decoder, {best_val:.4f} at the selected "
                         f"update ({gain:.1f} percent lower).")
        if accepted:
            lines.append(f"The selected file is `{output_path.name}`; every checkpoint of this training uses it automatically.")
        else:
            lines.append("No adapter file was kept; generation uses the pretrained decoder for this training.")
        if lr_reductions:
            lines.append(f"The learning rate was reduced once on a plateau (factor {config.plateau_lr_factor:g}).")
        lines.extend(["", "| Update | Speaker similarity | Held-out flow loss |", "|---:|---:|---:|"])
        for item in val_history:
            identity_text = f"{item['identity']:.4f}" if item.get("identity") is not None else "-"
            loss_text = f"{item['val_loss']:.4f}" if item.get("val_loss") is not None else "-"
            lines.append(f"| {int(item['step']):,} | {identity_text} | {loss_text} |")
        lines.extend(["", "Similarity is higher-is-better and decides the selection; the loss is lower-is-better and guards "
                          "against adapters that leave the decoder's objective. Listening still decides the final quality."])
        report_path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.status.write(phase=status, step=step, total_steps=total_steps, message=status_message, best_step=best_step,
                          best_val_loss=best_val, best_identity=best_identity, accepted=accepted,
                          output_path=str(output_path) if accepted else "", eta_s=0.0)
        if accepted:
            self.status.log(f">> voice decoder adapter saved to {output_path} | {status_message} | best step {best_step}"
                            + (f" | identity {best_identity:.4f}" if best_identity is not None else "")
                            + (f" | loss {best_val:.4f}" if best_val is not None else ""))
        else:
            self.status.log(f">> voice decoder adapter {status_message}")
        self._prompt_cache.clear()
        self._target_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result


def train_decoder_adapter(config: DecoderAdapterConfig | Mapping[str, Any], state_dir: str | Path, *,
                          log_fn: Callable[[str], None] | None = None,
                          cancel_callback: Callable[[], bool] | None = None) -> DecoderAdapterResult:
    return DecoderAdapterTrainer(config, state_dir, log_fn=log_fn, cancel_callback=cancel_callback).run()


__all__ = [
    "DecoderAdapterConfig",
    "DecoderAdapterResult",
    "DecoderAdapterTrainer",
    "decoder_report_path",
    "load_decoder_report",
    "reject_decoder_adapter",
    "train_decoder_adapter",
]
