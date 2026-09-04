"""Configuration contract for IndexTTS 2.5 LoRA/DoRA training."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping


@dataclass
class TrainConfig:
    dataset_dir: str
    name: str
    output_dir: str = "loras"

    adapter_type: str = "dora"
    rank: int = 128
    alpha: float = 129.0
    dropout: float = 0.05
    target_attention: bool = True
    target_mlp: bool = True
    train_spk_proj: bool = True
    train_emo_layers: bool = False
    train_mel_embed_head: bool = False

    base_variant: str = "bf16"
    base_dtype: str = "bf16"
    learning_rate: float = 4e-5
    lr_scheduler: str = "cosine"
    warmup_steps: int = 200
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.99)
    eps: float = 1e-8
    optimizer: str = "adamw"

    epochs: int = 10
    max_steps: int = 0
    batch_size: int = 1
    grad_accumulation: int = 1
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    blocks_to_swap: int = 0
    swap_ring_size: int = 2
    pin_swap_memory: bool = True

    mel_loss_weight: float = 1.0
    text_loss_weight: float = 0.1
    label_smoothing: float = 0.0
    speaker_ref_mode: str = "other"
    emo_ref_mode: str = "follow_speaker"
    max_codes: int = 1500
    max_text_tokens: int = 600

    val_fraction: float = 0.05
    val_reference_mode: str = "other"
    val_every_steps: int = 50
    val_max_batches: int = 20
    early_stop_patience: int = 0
    early_stop_min_delta: float = 0.0
    save_every_epochs: int = 1
    save_every_steps: int = 0
    keep_last_n: int = 0
    save_best: bool = True
    save_dtype: str = "bf16"
    save_train_state: bool = True
    epoch_train_state: bool = False
    resume_from: str = ""
    resume_mode: str = "weights_only"

    sample_every_epochs: int = 1
    sample_text: str = "This is a training progress sample for the adapted voice."
    sample_reference: str = ""
    sample_enabled: bool = True
    sample_runtime_tier: str = "auto"
    sample_min_free_vram_gb: float = 6.0
    sample_timeout_s: float = 300.0
    sample_language: str = "auto"
    sample_seed: int = -1
    sample_temperature: float = 0.8
    sample_top_p: float = 0.8
    sample_top_k: int = 30
    sample_repetition_penalty: float = 10.0
    sample_num_beams: int = 3
    sample_emo_alpha: float = 0.65
    sample_diffusion_steps: int = 25
    sample_inference_cfg_rate: float = 0.7
    sample_max_text_tokens: int = 60
    sample_length_penalty: float = 0.0
    sample_max_mel_tokens: int = 1500
    sample_speaking_rate: float = 1.0
    auto_analyze: bool = True
    auto_evaluate_checkpoints: bool = True
    eval_train_subset: int = 48
    eval_strengths: str = "1.0"
    eval_include_base: bool = True
    eval_timeout_s: float = 900.0

    seed: int = 42
    num_workers: int = 2
    log_every_steps: int = 1
    device: str = "cuda:0"

    # Paths and attention are advanced settings, but keeping them in the JSON
    # contract makes CLI runs independent of the current working directory.
    model_dir: str = "models"
    model_config: str = "models/config.yaml"
    attention_backend: str = "sdpa"

    def validate(self) -> "TrainConfig":
        self.dataset_dir = str(self.dataset_dir or "")
        self.name = _safe_name(self.name)
        self.output_dir = str(self.output_dir or "loras")
        self.adapter_type = str(self.adapter_type).lower()
        if self.adapter_type not in {"lora", "dora"}:
            raise ValueError("LoRA / DoRA type must be 'lora' or 'dora'")
        self.rank = max(1, int(self.rank))
        self.alpha = float(self.alpha)
        self.dropout = float(self.dropout)
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not (self.target_attention or self.target_mlp):
            raise ValueError("at least one LoRA / DoRA target group must be enabled")

        self.base_variant = str(self.base_variant).lower()
        if self.base_variant not in {"bf16", "int8_convrot"}:
            raise ValueError("base_variant must be 'bf16' or 'int8_convrot'")
        self.base_dtype = _dtype_name(self.base_dtype)
        self.mixed_precision = _dtype_name(self.mixed_precision)
        self.save_dtype = str(self.save_dtype).lower()
        if self.save_dtype not in {"bf16", "fp32"}:
            raise ValueError("save_dtype must be 'bf16' or 'fp32'")

        self.learning_rate = float(self.learning_rate)
        self.lr_scheduler = str(self.lr_scheduler).lower()
        if self.lr_scheduler not in {"cosine", "linear", "constant", "constant_with_warmup"}:
            raise ValueError("unsupported lr_scheduler")
        self.warmup_steps = max(0, int(self.warmup_steps))
        self.weight_decay = max(0.0, float(self.weight_decay))
        if len(self.betas) != 2:
            raise ValueError("betas must contain two values")
        self.betas = (float(self.betas[0]), float(self.betas[1]))
        self.eps = float(self.eps)
        self.optimizer = str(self.optimizer).lower()
        if self.optimizer not in {"adamw", "adamw_fused", "prodigy"}:
            raise ValueError("unsupported optimizer")

        self.epochs = max(1, int(self.epochs))
        self.max_steps = max(0, int(self.max_steps))
        self.batch_size = max(1, int(self.batch_size))
        self.grad_accumulation = max(1, int(self.grad_accumulation))
        self.max_grad_norm = max(0.0, float(self.max_grad_norm))
        self.blocks_to_swap = min(24, max(0, int(self.blocks_to_swap)))
        self.swap_ring_size = min(4, max(1, int(self.swap_ring_size)))

        self.mel_loss_weight = max(0.0, float(self.mel_loss_weight))
        self.text_loss_weight = max(0.0, float(self.text_loss_weight))
        self.label_smoothing = min(1.0, max(0.0, float(self.label_smoothing)))
        self.speaker_ref_mode = str(self.speaker_ref_mode).lower()
        if self.speaker_ref_mode not in {"self", "other", "mixed"}:
            raise ValueError("speaker_ref_mode must be self, other, or mixed")
        self.emo_ref_mode = str(self.emo_ref_mode).strip().lower()
        if self.emo_ref_mode not in {"self", "other", "mixed", "follow_speaker"}:
            raise ValueError(
                "emo_ref_mode must be self, other, mixed, or follow_speaker"
            )
        self.max_codes = max(1, int(self.max_codes))
        self.max_text_tokens = max(1, int(self.max_text_tokens))
        self.val_fraction = min(0.5, max(0.0, float(self.val_fraction)))
        self.val_reference_mode = str(self.val_reference_mode).strip().lower()
        if self.val_reference_mode not in {"self", "other"}:
            raise ValueError("val_reference_mode must be self or other")
        self.val_every_steps = max(0, int(self.val_every_steps))
        self.val_max_batches = max(1, int(self.val_max_batches))
        self.early_stop_patience = max(0, int(self.early_stop_patience))
        self.early_stop_min_delta = max(0.0, float(self.early_stop_min_delta))
        self.save_every_epochs = max(0, int(self.save_every_epochs))
        self.save_every_steps = max(0, int(self.save_every_steps))
        self.keep_last_n = max(0, int(self.keep_last_n))
        self.resume_mode = str(self.resume_mode or "weights_only").lower()
        if self.resume_mode not in {"weights_only", "continue"}:
            raise ValueError("resume_mode must be 'weights_only' or 'continue'")
        self.sample_every_epochs = max(1, int(self.sample_every_epochs))
        self.sample_min_free_vram_gb = max(0.0, float(self.sample_min_free_vram_gb))
        self.sample_timeout_s = max(1.0, float(self.sample_timeout_s))
        sample_language = str(self.sample_language or "auto").strip()
        self.sample_language = (
            "auto" if sample_language.lower() == "auto" else sample_language.upper()
        )
        if self.sample_language not in {"auto", "ZH", "EN", "JA", "AR", "ES"}:
            raise ValueError("sample_language must be auto, ZH, EN, JA, AR, or ES")
        self.sample_seed = int(self.sample_seed)
        if self.sample_seed < -1:
            raise ValueError("sample_seed must be -1 or greater")
        self.sample_temperature = _finite_float(
            self.sample_temperature, "sample_temperature"
        )
        if self.sample_temperature <= 0.0:
            raise ValueError("sample_temperature must be greater than 0")
        self.sample_top_p = _finite_float(self.sample_top_p, "sample_top_p")
        if not 0.0 <= self.sample_top_p <= 1.0:
            raise ValueError("sample_top_p must be in [0, 1]")
        self.sample_top_k = int(self.sample_top_k)
        if self.sample_top_k < 0:
            raise ValueError("sample_top_k must be 0 or greater")
        self.sample_repetition_penalty = _finite_float(
            self.sample_repetition_penalty, "sample_repetition_penalty"
        )
        if self.sample_repetition_penalty <= 0.0:
            raise ValueError("sample_repetition_penalty must be greater than 0")
        self.sample_num_beams = int(self.sample_num_beams)
        if self.sample_num_beams < 1:
            raise ValueError("sample_num_beams must be at least 1")
        self.sample_emo_alpha = _finite_float(
            self.sample_emo_alpha, "sample_emo_alpha"
        )
        if not 0.0 <= self.sample_emo_alpha <= 1.0:
            raise ValueError("sample_emo_alpha must be in [0, 1]")
        self.sample_diffusion_steps = int(self.sample_diffusion_steps)
        if self.sample_diffusion_steps < 2:
            raise ValueError("sample_diffusion_steps must be at least 2")
        self.sample_inference_cfg_rate = _finite_float(
            self.sample_inference_cfg_rate, "sample_inference_cfg_rate"
        )
        if self.sample_inference_cfg_rate < 0.0:
            raise ValueError("sample_inference_cfg_rate must be 0 or greater")
        self.sample_max_text_tokens = int(self.sample_max_text_tokens)
        if self.sample_max_text_tokens < 20:
            raise ValueError("sample_max_text_tokens must be at least 20")
        self.sample_length_penalty = _finite_float(
            self.sample_length_penalty, "sample_length_penalty"
        )
        self.sample_max_mel_tokens = int(self.sample_max_mel_tokens)
        if self.sample_max_mel_tokens < 1:
            raise ValueError("sample_max_mel_tokens must be at least 1")
        self.sample_speaking_rate = _finite_float(
            self.sample_speaking_rate, "sample_speaking_rate"
        )
        if not 0.5 <= self.sample_speaking_rate <= 1.5:
            raise ValueError("sample_speaking_rate must be in [0.5, 1.5]")
        self.eval_train_subset = int(self.eval_train_subset)
        if self.eval_train_subset < 0:
            raise ValueError("eval_train_subset must be 0 or greater")
        self.eval_strengths = str(self.eval_strengths or "")
        from .checkpoint_eval import parse_strengths

        parse_strengths(self.eval_strengths)
        self.eval_include_base = bool(self.eval_include_base)
        self.eval_timeout_s = max(1.0, float(self.eval_timeout_s))
        self.num_workers = max(0, int(self.num_workers))
        self.log_every_steps = max(1, int(self.log_every_steps))
        self.attention_backend = str(self.attention_backend).lower()
        if self.attention_backend not in {"sdpa", "eager", "flash_attention_2"}:
            raise ValueError("unsupported attention_backend")
        return self

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["betas"] = list(self.betas)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | "TrainConfig") -> "TrainConfig":
        if isinstance(value, cls):
            return cls(**value.to_dict()).validate()
        if not isinstance(value, Mapping):
            raise TypeError("training config must be a mapping")
        allowed = {item.name for item in fields(cls)}
        kwargs = {key: item for key, item in value.items() if key in allowed}
        if "dataset_dir" not in kwargs or "name" not in kwargs:
            raise ValueError("dataset_dir and name are required")
        if "betas" in kwargs:
            kwargs["betas"] = tuple(kwargs["betas"])
        return cls(**kwargs).validate()

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        with Path(path).open("r", encoding="utf-8-sig") as handle:
            value = json.load(handle)
        return cls.from_dict(value)


def _dtype_name(value: Any) -> str:
    name = str(value or "bf16").lower()
    aliases = {"bfloat16": "bf16", "float16": "fp16", "half": "fp16", "float32": "fp32"}
    name = aliases.get(name, name)
    if name not in {"bf16", "fp16", "fp32"}:
        raise ValueError(f"unsupported dtype {value!r}")
    return name


def _finite_float(value: Any, field_name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _safe_name(value: Any) -> str:
    name = str(value or "").strip()
    if not name:
        raise ValueError("name must not be empty")
    if name in {".", ".."} or any(char in name for char in '<>:"/\\|?*\x00'):
        raise ValueError(f"invalid LoRA / DoRA name {name!r}")
    return name


__all__ = ["TrainConfig"]
