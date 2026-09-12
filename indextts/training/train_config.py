"""Configuration contract for IndexTTS 2.5 LoRA/DoRA training."""

from __future__ import annotations

import json
import math
import re
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
    alpha: float = 128.0
    dropout: float = 0.05
    target_attention: bool = True
    target_mlp: bool = True
    train_spk_proj: bool = True
    train_emo_layers: bool = False
    train_mel_embed_head: bool = False
    train_full_modules_fp32: bool = True

    # GPU VRAM tier the training settings were chosen for: "auto" (the detected
    # card) or a nominal size such as "8". It records the choice and resolves the
    # sample tier; the memory-relevant fields below carry the actual values.
    vram_tier: str = "auto"
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
    val_split_mode: str = "source"
    val_reference_mode: str = "other"
    val_every_steps: int = 250
    val_max_batches: int = 0
    early_stop_enabled: bool = True
    early_stop_patience: int = 6
    early_stop_min_delta: float = 0.005
    early_stop_min_steps: int = 1000
    early_stop_min_epochs: float = 2.0
    early_stop_check_steps: int = 0
    plateau_lr_enabled: bool = True
    plateau_lr_factor: float = 0.5
    plateau_lr_grace_steps: int = 1000
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
    speech_eval_enabled: bool = True
    # Automatic references (the saved recommended reference, training conditioning, the speech benchmark)
    # prefer, among the best-quality clips, the one nearest the speaker's median pitch and pace.
    reference_typical: bool = True
    # 0 = automatic: six prompts per held-out recording, at least 12 and at most 24, spread evenly over
    # the held-out recordings so one recording cannot decide the comparison on its own.
    speech_eval_prompts: int = 0
    speech_eval_seeds: int = 3
    speech_eval_candidates: int = 3
    speech_eval_timeout_s: float = 7200.0
    speech_eval_max_wer_increase: float = 0.02
    speech_eval_max_speaker_drop: float = 0.03
    # Render the benchmark with the settings Voice Generation applies by default for each candidate (the
    # GPU tier's beams and diffusion steps, Smart sentences with the adapter's token target and dataset
    # pauses, its expressive clip and speaking rate) instead of the fixed short-sample settings, so the
    # comparison measures what a user hears.
    speech_eval_deployment_settings: bool = True
    # "interval": a candidate is rejected for a word-error increase or an identity drop only when the paired
    # difference is positive beyond its prompt-bootstrap 95% interval, or holds on a majority of the held-out
    # recordings. "mean": the earlier hard threshold on the mean.
    speech_eval_guard_mode: str = "interval"
    # Weight of a paired word-error increase in the deployment score (one point of word error costs
    # weight/100 of speaker similarity): raise it to favor accuracy, lower it to favor likeness.
    speech_eval_score_wer_weight: float = 4.0
    # Judge the voice decoder adapter through the full pipeline against the best adapter checkpoint even
    # when the GPT comparison preferred Base, and recommend adapter + decoder when that deployment wins.
    decoder_adapter_always_gate: bool = True
    # Epoch probe: after an epoch the checkpoint renders a few held-out sentences with the deployment settings
    # in its own process (the same free-VRAM gate and tier fitting as the epoch sample, or another GPU when one
    # is free) and is measured against the real recordings; Base is rendered once for the comparison.
    probe_enabled: bool = True
    # 0 = automatic: every epoch while a probe costs less than a third of an epoch, otherwise spaced so the
    # probes stay under that share; N probes every N epochs.
    probe_every_epochs: int = 0
    probe_prompts: int = 0  # 0 = automatic: three per held-out recording, at least 6 and at most 12
    # Two seeds per sentence halve the swing a single misread word or an unlucky render gives a probe.
    probe_seeds: int = 2
    # Consecutive probe checks without a deployment-score improvement greater than probe_min_delta before the
    # probe counts as stalled; with probe_every_epochs above 1 each check spans that many epochs.
    probe_patience: int = 2
    probe_min_delta: float = 0.002
    # 0 = automatic: the recognizer's own error on the real probe recordings, at least one point. A probe word
    # error above the best check's by more than this for probe_patience checks, while the score also stalls,
    # stops training as overfitting.
    probe_wer_tolerance: float = 0.0
    probe_timeout_s: float = 900.0
    # With the probe enabled, a validation-loss stall alone does not stop training while the probe's
    # deployment score is still improving; both signals must stall.
    probe_stop_enabled: bool = True
    # "auto": another CUDA device with enough free memory when the machine has one, else the training GPU
    # behind the sample free-VRAM gate; "same": always the training GPU; or an explicit device such as cuda:1.
    probe_device: str = "auto"
    # After training, the last N saved updates (epoch files and the final file) can be averaged in parameter
    # space into one more candidate for the speech comparison. Off by default; an averaged checkpoint
    # must pass the same speech comparison as the saved checkpoints before it can be recommended.
    average_last_checkpoints: int = 0
    # Exponential moving average of the trainable weights, updated after every optimizer step and saved
    # beside each epoch and final file as ``<name>_ema*.safetensors``. 0 disables it; 0.999 averages over
    # roughly the last thousand updates. The final EMA file joins the speech comparison as a candidate.
    ema_decay: float = 0.0
    final_test_dataset: str = ""
    decoder_adapter_enabled: bool = True
    decoder_adapter_rank: int = 128
    decoder_adapter_alpha: float = 128.0
    decoder_adapter_epochs: int = 10
    decoder_adapter_learning_rate: float = 2e-4
    decoder_adapter_timeout_s: float = 7200.0
    # Which semantic codes the decoder adapter learns to render: "real" (quantized from the recordings, as
    # the decoder was pretrained), "gpt" (the selected checkpoint's own teacher-forced predictions for the
    # same clips, what generation actually feeds the decoder), or "mixed" (half of each).
    decoder_adapter_code_source: str = "real"
    # After the decoder adapter is judged, sweep temperature, guidance rate, and beams on the speech
    # benchmark for the selected checkpoint and save the winner for Voice Generation.
    decoding_sweep_enabled: bool = True
    decoding_sweep_timeout_s: float = 5400.0

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
        self.train_full_modules_fp32 = bool(self.train_full_modules_fp32)

        from indextts.runtime.vram_presets import VRAM_TIERS

        tier = str(self.vram_tier or "auto").strip().lower()
        if tier != "auto" and tier not in {str(item) for item in VRAM_TIERS}:
            raise ValueError("vram_tier must be 'auto' or one of " + ", ".join(str(item) for item in VRAM_TIERS))
        self.vram_tier = tier
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
        self.val_split_mode = str(self.val_split_mode).strip().lower()
        if self.val_split_mode not in {"record", "source"}:
            raise ValueError("val_split_mode must be record or source")
        self.val_reference_mode = str(self.val_reference_mode).strip().lower()
        if self.val_reference_mode not in {"self", "other"}:
            raise ValueError("val_reference_mode must be self or other")
        self.val_every_steps = max(0, int(self.val_every_steps))
        self.val_max_batches = max(0, int(self.val_max_batches))
        self.early_stop_patience = max(0, int(self.early_stop_patience))
        self.early_stop_min_delta = max(0.0, float(self.early_stop_min_delta))
        self.early_stop_enabled = bool(self.early_stop_enabled)
        self.early_stop_min_steps = max(0, int(self.early_stop_min_steps))
        self.early_stop_min_epochs = max(0.0, _finite_float(self.early_stop_min_epochs, "early_stop_min_epochs"))
        self.early_stop_check_steps = max(0, int(self.early_stop_check_steps))
        self.plateau_lr_enabled = bool(self.plateau_lr_enabled)
        self.plateau_lr_factor = _finite_float(self.plateau_lr_factor, "plateau_lr_factor")
        if not 0.0 < self.plateau_lr_factor < 1.0:
            raise ValueError("plateau_lr_factor must be between 0 and 1")
        self.plateau_lr_grace_steps = max(0, int(self.plateau_lr_grace_steps))
        self.save_every_epochs = max(0, int(self.save_every_epochs))
        self.save_every_steps = max(0, int(self.save_every_steps))
        self.keep_last_n = max(0, int(self.keep_last_n))
        self.resume_mode = str(self.resume_mode or "weights_only").lower()
        if self.resume_mode not in {"weights_only", "continue"}:
            raise ValueError("resume_mode must be 'weights_only' or 'continue'")
        self.average_last_checkpoints = max(0, int(self.average_last_checkpoints))
        self.ema_decay = _finite_float(self.ema_decay, "ema_decay")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be 0 (off) or between 0 and 1")
        self.decoder_adapter_enabled = bool(self.decoder_adapter_enabled)
        self.decoder_adapter_rank = max(1, int(self.decoder_adapter_rank))
        self.decoder_adapter_alpha = _finite_float(self.decoder_adapter_alpha, "decoder_adapter_alpha")
        if self.decoder_adapter_alpha <= 0:
            raise ValueError("decoder_adapter_alpha must be positive")
        self.decoder_adapter_epochs = max(1, int(self.decoder_adapter_epochs))
        self.decoder_adapter_learning_rate = _finite_float(self.decoder_adapter_learning_rate, "decoder_adapter_learning_rate")
        if self.decoder_adapter_learning_rate <= 0:
            raise ValueError("decoder_adapter_learning_rate must be positive")
        self.decoder_adapter_timeout_s = max(60.0, float(self.decoder_adapter_timeout_s))
        self.decoder_adapter_code_source = str(self.decoder_adapter_code_source or "real").strip().lower()
        if self.decoder_adapter_code_source not in {"real", "gpt", "mixed"}:
            raise ValueError("decoder_adapter_code_source must be 'real', 'gpt', or 'mixed'")
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
        self.speech_eval_enabled = bool(self.speech_eval_enabled)
        self.reference_typical = bool(self.reference_typical)
        self.decoding_sweep_enabled = bool(self.decoding_sweep_enabled)
        self.decoding_sweep_timeout_s = max(60.0, _finite_float(self.decoding_sweep_timeout_s, "decoding_sweep_timeout_s"))
        self.final_test_dataset = str(self.final_test_dataset or "").strip()
        for key, lower, upper in (("speech_eval_prompts", 0, 100),
                                  ("speech_eval_seeds", 1, 10),
                                  ("speech_eval_candidates", 1, 10),
                                  ("probe_every_epochs", 0, 10000),
                                  ("probe_prompts", 0, 50),
                                  ("probe_seeds", 1, 5),
                                  ("probe_patience", 1, 100)):
            value = int(getattr(self, key))
            if not lower <= value <= upper:
                raise ValueError(f"{key} must be between {lower} and {upper}")
            setattr(self, key, value)
        self.speech_eval_deployment_settings = bool(self.speech_eval_deployment_settings)
        self.speech_eval_guard_mode = str(self.speech_eval_guard_mode or "interval").strip().lower()
        if self.speech_eval_guard_mode not in {"interval", "mean"}:
            raise ValueError("speech_eval_guard_mode must be 'interval' or 'mean'")
        self.speech_eval_score_wer_weight = _finite_float(self.speech_eval_score_wer_weight, "speech_eval_score_wer_weight")
        if self.speech_eval_score_wer_weight < 0.0:
            raise ValueError("speech_eval_score_wer_weight must be 0 or greater")
        self.decoder_adapter_always_gate = bool(self.decoder_adapter_always_gate)
        self.probe_enabled = bool(self.probe_enabled)
        self.probe_stop_enabled = bool(self.probe_stop_enabled)
        self.probe_min_delta = max(0.0, _finite_float(self.probe_min_delta, "probe_min_delta"))
        self.probe_wer_tolerance = _finite_float(self.probe_wer_tolerance, "probe_wer_tolerance")
        if not 0.0 <= self.probe_wer_tolerance <= 1.0:
            raise ValueError("probe_wer_tolerance must be between 0 (automatic) and 1")
        self.probe_device = str(self.probe_device or "auto").strip().lower()
        if self.probe_device not in {"auto", "same"} and not re.fullmatch(r"cuda:\d+", self.probe_device):
            raise ValueError("probe_device must be 'auto', 'same', or a CUDA device such as cuda:1")
        self.probe_timeout_s = max(60.0, _finite_float(self.probe_timeout_s, "probe_timeout_s"))
        self.speech_eval_timeout_s = max(1.0, _finite_float(self.speech_eval_timeout_s, "speech_eval_timeout_s"))
        for key in ("speech_eval_max_wer_increase", "speech_eval_max_speaker_drop"):
            value = _finite_float(getattr(self, key), key)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{key} must be between 0 and 1")
            setattr(self, key, value)
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
