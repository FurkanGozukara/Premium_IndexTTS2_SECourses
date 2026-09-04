"""Resumable LoRA/DoRA training loop for the IndexTTS 2.5 GPT."""

from __future__ import annotations

import gc
import json
import math
import os
import random
import secrets
import shutil
import subprocess
import sys
import threading
import time
import warnings
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.lora import (
    LoraMetadata,
    apply_lora,
    inject_adapters,
    list_target_modules,
    load_lora,
    load_train_state,
    resume_state_path_for,
    save_lora,
    save_train_state,
    set_training_mode,
    trainable_parameters,
)
from indextts.lora.layers import LoRAAdapter
from indextts.quant.convrot_int8 import load_gpt_checkpoint
from indextts.runtime import (
    BlockSwapConfig,
    ProgressReporter,
    enable_block_swap,
    gpu_free_gb,
    gpu_total_gb,
    memory_stats,
)
from indextts.utils import model_downloads
from indextts.utils.atomic_json import read_json_retry

from .dataset import LengthBucketBatchSampler, LoraTrainDataset, collate
from .dataset_manifest import atomic_write_json
from .model_forward import enable_gradient_checkpointing, gpt_train_step_loss
from .plan import suggested_epochs, training_plan, training_plan_line
from .sampling import generate_training_sample
from .speaking_rate import calibrate_from_samples, write_speaking_rate
from .train_config import TrainConfig


@dataclass
class TrainingResult:
    status: str
    step: int
    total_steps: int
    epoch: int
    output_path: str
    best_path: str
    best_val_loss: float | None
    initial_loss: float | None
    final_loss: float | None
    avg_it_s: float
    peak_vram_gb: float
    elapsed_s: float


@dataclass
class BuiltTrainingModel:
    model: UnifiedVoice
    adapters: dict[str, LoRAAdapter]
    full_modules: dict[str, torch.nn.Module]
    parameters: list[torch.nn.Parameter]
    block_swap: Any = None


def _dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _move_non_block_modules(model: UnifiedVoice, device: torch.device) -> None:
    for name, child in model.named_children():
        if name != "gpt":
            child.to(device)
            continue
        for transformer_name, transformer_child in child.named_children():
            if transformer_name != "h":
                transformer_child.to(device)


def _adapter_mapping(model: torch.nn.Module) -> dict[str, LoRAAdapter]:
    return {
        name: module
        for name, module in model.named_modules()
        if name and isinstance(module, LoRAAdapter)
    }


def build_training_model(
    config: TrainConfig,
    *,
    log: Callable[[str], None] | None = None,
) -> BuiltTrainingModel:
    """Load the frozen base, attach LoRA / DoRA modules, and establish device residency."""

    cfg = TrainConfig.from_dict(config)
    emit = log or (lambda message: print(message, flush=True))
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA training device is unavailable: {device}")
    base_dtype = _dtype(cfg.base_dtype)
    if device.type == "cpu" and base_dtype != torch.float32:
        base_dtype = torch.float32

    model_cfg = OmegaConf.load(cfg.model_config)
    model = UnifiedVoice(
        **model_cfg.gpt,
        spk_cond_mode="campplus",
        attention_backend=cfg.attention_backend,
    )
    model_dir = Path(cfg.model_dir)
    checkpoint = (
        model_dir / "gpt_int8_convrot.safetensors"
        if cfg.base_variant == "int8_convrot"
        else model_dir / str(model_cfg.gpt_checkpoint)
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"GPT checkpoint not found: {checkpoint}")
    emit(f">> loading {cfg.base_variant} GPT from {checkpoint}")
    report = load_gpt_checkpoint(
        model,
        str(checkpoint),
        device="cpu",
        dtype=base_dtype,
        strict=False,
    )
    emit(
        f">> GPT loaded in {report.seconds:.1f}s | quantized layers {report.quantized_layers} | "
        f"missing {len(report.missing_keys)} | unexpected {len(report.unexpected_keys)}"
    )
    model.requires_grad_(False)

    resume_file = load_lora(cfg.resume_from) if cfg.resume_from else None
    if resume_file is not None:
        changes = []
        if cfg.rank != resume_file.rank:
            changes.append(f"rank {cfg.rank} -> {resume_file.rank}")
            cfg.rank = resume_file.rank
        if cfg.alpha != resume_file.alpha:
            changes.append(f"alpha {cfg.alpha:g} -> {resume_file.alpha:g}")
            cfg.alpha = resume_file.alpha
        saved_type = resume_file.adapter_type
        if cfg.adapter_type != saved_type:
            changes.append(f"LoRA / DoRA type {cfg.adapter_type} -> {saved_type}")
            cfg.adapter_type = saved_type
        if changes:
            emit(">> resume metadata overrides config: " + ", ".join(changes))
        targets = list(resume_file.module_paths)
    else:
        targets = list_target_modules(
            model, attention=cfg.target_attention, mlp=cfg.target_mlp
        )
    if not targets:
        raise ValueError("no GPT projection modules were selected for LoRA / DoRA")

    adapters = inject_adapters(
        model,
        rank=cfg.rank,
        alpha=cfg.alpha,
        dropout=cfg.dropout,
        use_dora=cfg.adapter_type == "dora",
        target_modules=targets,
    )
    if cfg.resume_from:
        apply_lora(model, cfg.resume_from, strength=1.0)
        adapters = _adapter_mapping(model)

    full_modules: dict[str, torch.nn.Module] = {}
    if cfg.train_spk_proj:
        full_modules["spk_emb_proj"] = model.spk_emb_proj
    if cfg.train_emo_layers:
        full_modules["emovec_layer"] = model.emovec_layer
        full_modules["emo_layer"] = model.emo_layer
    if cfg.train_mel_embed_head:
        full_modules["mel_embedding"] = model.mel_embedding
        full_modules["mel_head"] = model.mel_head
    parameters = trainable_parameters(model, adapters, full_modules)
    if not parameters:
        raise ValueError("the training configuration produced no trainable parameters")

    if cfg.gradient_checkpointing:
        enable_gradient_checkpointing(model, True)
    elif cfg.blocks_to_swap > 0:
        raise ValueError("block-swap training requires gradient_checkpointing=True")

    block_swap = None
    if cfg.blocks_to_swap > 0 and device.type == "cuda":
        block_swap = enable_block_swap(
            list(model.gpt.h),
            cfg.blocks_to_swap,
            BlockSwapConfig(
                device=device,
                supports_backward=True,
                use_pinned_memory=cfg.pin_swap_memory,
                ring_size=cfg.swap_ring_size,
                gradient_checkpointing=cfg.gradient_checkpointing,
            ),
        )
        _move_non_block_modules(model, device)
        emit(f">> GPT block swap: {block_swap.summary()}")
    else:
        model.to(device)

    model.train()
    set_training_mode(model, True)
    trainable = sum(parameter.numel() for parameter in parameters)
    total = sum(parameter.numel() for parameter in model.parameters())
    emit(f">> trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")
    return BuiltTrainingModel(model, adapters, full_modules, parameters, block_swap)


def _batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: Mapping[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.random.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _optimizer(config: TrainConfig, parameters: list[torch.nn.Parameter]) -> torch.optim.Optimizer:
    kwargs = {
        "lr": 1.0 if config.optimizer == "prodigy" else config.learning_rate,
        "betas": config.betas,
        "eps": config.eps,
        "weight_decay": config.weight_decay,
    }
    if config.optimizer == "prodigy":
        try:
            from prodigyopt import Prodigy
        except ImportError as exc:
            raise ImportError("optimizer='prodigy' requires the unpinned prodigyopt package") from exc
        return Prodigy(parameters, **kwargs)
    if config.optimizer == "adamw_fused":
        try:
            return torch.optim.AdamW(parameters, fused=True, **kwargs)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(parameters, **kwargs)


def _scheduler(
    config: TrainConfig, optimizer: torch.optim.Optimizer, total_steps: int
) -> Any:
    from transformers import get_scheduler

    return get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=min(config.warmup_steps, total_steps),
        num_training_steps=max(1, total_steps),
    )


class LoraTrainer:
    def __init__(
        self,
        config: TrainConfig | Mapping[str, Any],
        *,
        state_dir: str | Path | None = None,
        reporter: ProgressReporter | None = None,
    ) -> None:
        self.config = TrainConfig.from_dict(config)
        self.dataset_dir = Path(self.config.dataset_dir).expanduser().resolve()
        self.adapter_dir = (
            Path(self.config.output_dir).expanduser().resolve() / self.config.name
        )
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir = (
            Path(state_dir).expanduser().resolve() if state_dir else self.adapter_dir
        )
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.state_dir / "status.json"
        self.metrics_path = self.state_dir / "metrics.jsonl"
        self.log_path = self.state_dir / "log.txt"
        self.stop_path = self.state_dir / "stop.flag"
        self.reporter = reporter
        self.started_perf = time.perf_counter()
        self.reference_copy = self.adapter_dir / f"{self.config.name}_reference.wav"
        self.best_path = self.adapter_dir / "best" / f"{self.config.name}.safetensors"
        self.last_sample = ""
        self.last_checkpoint = ""
        self.resolved_sample_seed: int | None = None
        self._checkpoint_history: list[Path] = []

    def log(self, message: str) -> None:
        line = str(message)
        with self.log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
        if self.reporter is not None:
            self.reporter.log(line)
        else:
            print(line, flush=True)

    def write_status(self, **updates: Any) -> dict[str, Any]:
        try:
            current = json.loads(self.status_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            current = {}
        current.update(updates)
        current.setdefault("phase", "initializing")
        current.setdefault("step", 0)
        current.setdefault("total_steps", 0)
        current.setdefault("epoch", 0)
        current.setdefault("total_epochs", self.config.epochs)
        current.setdefault("loss", None)
        current.setdefault("avg_loss", None)
        current.setdefault("val_loss", None)
        current.setdefault("lr", self.config.learning_rate)
        current.setdefault("grad_norm", None)
        current.setdefault("it_s", 0.0)
        current.setdefault("eta_s", None)
        current.setdefault("elapsed_s", time.perf_counter() - self.started_perf)
        current.setdefault("vram_used_gb", 0.0)
        current.setdefault("message", "")
        current.setdefault("last_checkpoint", self.last_checkpoint)
        current.setdefault("last_sample", self.last_sample)
        current.setdefault("sample_seed", self.resolved_sample_seed)
        current.setdefault("val_reference_mode", self.config.val_reference_mode)
        current.setdefault("recommended_speaking_rate", None)
        current["updated_at"] = time.time()
        atomic_write_json(self.status_path, current)
        return current

    def metric(self, payload: Mapping[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(dict(payload), ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()

    def _prepare_base_variant(self) -> str:
        config = self.config
        if config.base_variant != "int8_convrot":
            return ""
        checkpoint = model_downloads.int8_gpt_path(config.model_dir)
        if checkpoint.is_file():
            return ""

        self.log(
            f">> INT8 ConvRot GPT is missing at {checkpoint}; starting automatic download"
        )
        download_reporter = ProgressReporter(
            "INT8 ConvRot GPT",
            total=1000,
            progress_file=self.state_dir / "progress.json",
        )
        download_reporter.set_stage("model download")

        def download_progress(*values: Any, **kwargs: Any) -> None:
            fraction = kwargs.get("fraction")
            if fraction is None and values and isinstance(values[0], (int, float)):
                fraction = values[0]
            try:
                normalized = max(0.0, min(1.0, float(fraction or 0.0)))
            except (TypeError, ValueError):
                normalized = 0.0
            description = str(
                kwargs.get("desc")
                or kwargs.get("message")
                or (values[-1] if values and isinstance(values[-1], str) else "")
            )
            message = model_downloads.int8_download_progress_message(
                normalized,
                description,
                models_dir=config.model_dir,
            )
            download_reporter.update(round(normalized * 1000), desc=message)
            self.write_status(
                phase="initializing",
                step=0,
                message=message,
            )

        try:
            model_downloads.ensure_int8_gpt(config.model_dir, download_progress)
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"download returned without creating {checkpoint}"
                )
            ready = model_downloads.int8_download_progress_message(
                1.0, models_dir=config.model_dir
            )
            download_reporter.update(1000, desc=ready)
            self.write_status(phase="initializing", step=0, message=ready)
            self.log(f">> INT8 ConvRot GPT ready at {checkpoint}")
            return ""
        except Exception as exc:
            warning = model_downloads.int8_fallback_warning(config.model_dir, exc)
            config.base_variant = "bf16"
            warning_reporter = ProgressReporter(
                "model load",
                total=1,
                progress_file=self.state_dir / "progress.json",
            )
            warning_reporter.update(0, desc=warning)
            self.write_status(
                phase="initializing",
                step=0,
                message=warning,
                base_variant="bf16",
                runtime_warning=warning,
            )
            self.log(">> " + warning)
            return warning

    def _reference_candidate(self) -> Path | None:
        info_path = self.dataset_dir / "dataset_info.json"
        try:
            info = json.loads(info_path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            info = {}
        for value in info.get("reference_candidates", []):
            candidate = Path(str(value))
            if not candidate.is_absolute():
                candidate = self.dataset_dir / candidate
            if candidate.is_file():
                return candidate
        return None

    def _prepare_reference(self) -> Path | None:
        candidate = self._reference_candidate()
        if candidate is not None and candidate.resolve() != self.reference_copy.resolve():
            shutil.copy2(candidate, self.reference_copy)
        return self.reference_copy if self.reference_copy.is_file() else candidate

    def _write_automatic_analysis(self) -> tuple[str, str]:
        """Analyze saved metrics without allowing a reporting error to fail training."""

        if not self.config.auto_analyze:
            self.log(">> automatic generalization analysis is disabled")
            return "", ""
        try:
            from .analysis import analyze_training_run, write_training_analysis

            analysis = analyze_training_run(self.adapter_dir, self.state_dir)
            path = write_training_analysis(analysis)
            for line in analysis.summary_markdown.splitlines():
                if line.strip():
                    self.log(">> " + line.replace("**", ""))
            if analysis.best_epoch is not None and analysis.best_val_loss is not None:
                detail = (
                    f">> Best validation loss {analysis.best_val_loss:.4f} at epoch "
                    f"{analysis.best_epoch}"
                )
                if (
                    analysis.status == "best_found"
                    and analysis.final_epoch is not None
                    and analysis.final_val_loss is not None
                ):
                    detail += (
                        f"; the final epoch {analysis.final_epoch} LoRA / DoRA is overfitted "
                        f"(validation {analysis.final_val_loss:.4f})"
                    )
                if analysis.recommended_checkpoint:
                    try:
                        recommended = Path(analysis.recommended_checkpoint).relative_to(
                            self.adapter_dir
                        )
                    except ValueError:
                        recommended = Path(analysis.recommended_checkpoint)
                    detail += f". Recommended: {recommended.as_posix()}"
                self.log(detail)
            self.write_status(
                analysis_path=str(path.resolve()),
                recommended_checkpoint=analysis.recommended_checkpoint,
            )
            return str(path.resolve()), analysis.recommended_checkpoint
        except Exception as exc:
            self.log(f">> automatic generalization analysis failed but training is safe: {exc}")
            return "", ""

    def _run_automatic_evaluation(
        self,
        *,
        terminal_phase: str,
        terminal_message: str,
        recommended_checkpoint: str,
    ) -> str:
        """Run checkpoint evaluation in a bounded child process after model cleanup."""

        config = self.config
        if not config.auto_evaluate_checkpoints:
            self.log(">> automatic checkpoint evaluation is disabled")
            return recommended_checkpoint
        device = torch.device(config.device)
        if device.type == "cuda":
            free_gb = gpu_free_gb(device.index or 0)
            if free_gb < config.sample_min_free_vram_gb:
                self.log(
                    f">> checkpoint evaluation skipped: {free_gb:.1f} GB free VRAM is below "
                    f"the {config.sample_min_free_vram_gb:.1f} GB threshold"
                )
                return recommended_checkpoint

        from .checkpoint_eval import (
            CheckpointEvalConfig,
            load_checkpoint_eval,
            parse_strengths,
        )

        job_dir = self.adapter_dir / "analysis" / "eval_job"
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "stop.flag").unlink(missing_ok=True)
        strengths = parse_strengths(config.eval_strengths)
        eval_config = CheckpointEvalConfig(
            adapter_dir=str(self.adapter_dir),
            dataset_dir=str(self.dataset_dir),
            include_base=config.eval_include_base,
            strengths=strengths,
            train_subset=config.eval_train_subset,
            device=config.device,
            base_variant=config.base_variant,
            base_dtype=config.base_dtype,
            model_dir=config.model_dir,
            model_config=config.model_config,
            attention_backend=config.attention_backend,
            val_fraction=config.val_fraction,
            seed=config.seed,
            reference_mode=config.val_reference_mode,
        ).validate()
        config_path = job_dir / "eval_config.json"
        atomic_write_json(config_path, eval_config.to_dict())
        command = [
            sys.executable,
            "-m",
            "indextts.training.eval_worker",
            "--config",
            str(config_path),
            "--state-dir",
            str(job_dir),
        ]
        self.log(
            ">> automatic checkpoint evaluation settings | "
            f"train subset: {eval_config.train_subset} | strengths: "
            f"{','.join(f'{value:g}' for value in eval_config.strengths)} | "
            f"include base: {eval_config.include_base} | reference: {eval_config.reference_mode}"
        )
        self.log(">> starting automatic checkpoint evaluation")
        self.write_status(phase="evaluating", message="loading model for checkpoint evaluation")
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parents[2]),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )

        def pump() -> None:
            if process.stdout is None:
                return
            for line in iter(process.stdout.readline, ""):
                if line:
                    self.log(line.rstrip())

        pump_thread = threading.Thread(target=pump, daemon=True, name="checkpoint-eval-log")
        pump_thread.start()
        started = time.perf_counter()
        timed_out = False
        while process.poll() is None:
            elapsed = time.perf_counter() - started
            child = read_json_retry(job_dir / "status.json", {}) or {}
            self.write_status(
                phase="evaluating",
                message=str(child.get("message") or "evaluating checkpoints"),
                eval_completed=int(child.get("completed", 0) or 0),
                eval_total=int(child.get("total", 0) or 0),
                eval_elapsed_s=float(child.get("elapsed_s", elapsed) or elapsed),
            )
            if elapsed >= config.eval_timeout_s:
                timed_out = True
                process.kill()
                break
            time.sleep(0.5)
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        pump_thread.join(timeout=2.0)
        if timed_out:
            self.log(
                f">> checkpoint evaluation timed out after {config.eval_timeout_s:.0f}s; "
                "training output remains valid"
            )
        elif process.returncode != 0:
            self.log(
                f">> checkpoint evaluation failed with code {process.returncode}; "
                "training output remains valid"
            )
        else:
            report = load_checkpoint_eval(self.adapter_dir)
            if report is not None and report.recommended_checkpoint:
                recommended_checkpoint = report.recommended_checkpoint
            self.log(">> automatic checkpoint evaluation complete")
        self.write_status(
            phase=terminal_phase,
            message=terminal_message,
            recommended_checkpoint=recommended_checkpoint,
            evaluation_path=str((self.adapter_dir / "analysis" / "checkpoint_eval.json").resolve())
            if (self.adapter_dir / "analysis" / "checkpoint_eval.json").is_file()
            else "",
        )
        return recommended_checkpoint

    def _write_speaking_rate_calibration(self) -> float | None:
        """Measure completed epoch samples without risking the training result."""

        samples_dir = self.adapter_dir / "samples"
        if not samples_dir.is_dir() or not any(samples_dir.glob("epoch_*.wav")):
            return None
        try:
            report = calibrate_from_samples(
                self.adapter_dir,
                self.dataset_dir,
                self.config.sample_text,
            )
            if report is None:
                self.log(
                    ">> speaking-rate calibration skipped: no usable one-second training samples"
                )
                return None
            write_speaking_rate(self.adapter_dir, report)
            self.log(">> " + report.summary)
            self.write_status(
                recommended_speaking_rate=report.recommended_speaking_rate,
            )
            return report.recommended_speaking_rate
        except Exception as exc:
            self.log(
                f">> speaking-rate calibration failed but training is safe: {exc}"
            )
            return None

    def _metadata(self, step: int, epochs: int, targets: list[str]) -> LoraMetadata:
        return LoraMetadata(
            adapter_type=self.config.adapter_type,
            rank=self.config.rank,
            alpha=self.config.alpha,
            dropout=self.config.dropout,
            target_modules=targets,
            base_variant=self.config.base_variant,
            trained_steps=step,
            epochs=epochs,
            dataset_name=self.dataset_dir.name,
            created_at=datetime.now(timezone.utc).isoformat(),
            app_version="5",
            train_config=self.config.to_dict(),
            recommended_reference=self.reference_copy.name if self.reference_copy.is_file() else "",
            sample_rate=24000,
        )

    def _train_state(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: Any,
        step: int,
        next_epoch: int,
        next_batch: int,
        dataset_fingerprint: str,
        best_val_loss: float | None,
        ema_loss: float | None,
        moving_losses: deque[float],
    ) -> dict[str, Any]:
        return {
            "version": 1,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "epoch": next_epoch,
            "batch_in_epoch": next_batch,
            "dataset_fingerprint": dataset_fingerprint,
            "best_val_loss": best_val_loss,
            "ema_loss": ema_loss,
            "moving_losses": list(moving_losses),
            "rng": _rng_state(),
            "config": self.config.to_dict(),
        }

    def save_checkpoint(
        self,
        destination: Path,
        built: BuiltTrainingModel,
        *,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: Any,
        step: int,
        epochs_completed: int,
        next_epoch: int,
        next_batch: int,
        dataset_fingerprint: str,
        best_val_loss: float | None,
        ema_loss: float | None,
        moving_losses: deque[float],
        keep: bool = False,
    ) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        periodic_checkpoint = any(
            destination.stem.startswith(f"{self.config.name}_{kind}_")
            and destination.stem.removeprefix(
                f"{self.config.name}_{kind}_"
            ).isdigit()
            for kind in ("epoch", "step")
        )
        save_lora(
            destination,
            built.adapters,
            built.full_modules,
            self._metadata(step, epochs_completed, list(built.adapters)),
            dtype=_dtype(self.config.save_dtype),
        )
        if self.config.save_train_state and (
            self.config.epoch_train_state or not periodic_checkpoint
        ):
            state = self._train_state(
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                step=step,
                next_epoch=next_epoch,
                next_batch=next_batch,
                dataset_fingerprint=dataset_fingerprint,
                best_val_loss=best_val_loss,
                ema_loss=ema_loss,
                moving_losses=moving_losses,
            )
            save_train_state(resume_state_path_for(destination), state)
        self.last_checkpoint = str(destination.resolve())
        if not keep and ("_step_" in destination.stem or "_epoch_" in destination.stem):
            self._checkpoint_history.append(destination)
            self._prune_checkpoints()
        return destination

    def _prune_checkpoints(self) -> None:
        keep = self.config.keep_last_n
        if keep <= 0:
            return
        victims = self._checkpoint_history[:-keep]
        for path in victims:
            path.unlink(missing_ok=True)
            Path(resume_state_path_for(path)).unlink(missing_ok=True)
            if path in self._checkpoint_history:
                self._checkpoint_history.remove(path)

    @torch.no_grad()
    def validate(
        self,
        built: BuiltTrainingModel,
        loader: DataLoader,
        device: torch.device,
    ) -> tuple[float, float]:
        model = built.model
        model.eval()
        set_training_mode(model, False)
        losses: list[float] = []
        accuracies: list[float] = []
        amp_dtype = _dtype(self.config.mixed_precision)
        amp_enabled = device.type == "cuda" and amp_dtype != torch.float32
        for index, batch in enumerate(loader):
            if index >= self.config.val_max_batches:
                break
            batch = _batch_to_device(batch, device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
                loss, values = gpt_train_step_loss(
                    model,
                    batch,
                    mel_loss_weight=self.config.mel_loss_weight,
                    text_loss_weight=self.config.text_loss_weight,
                    label_smoothing=self.config.label_smoothing,
                )
            losses.append(float(loss))
            accuracies.append(float(values["mel_accuracy"]))
        model.train()
        set_training_mode(model, True)
        if not losses:
            return math.nan, math.nan
        return sum(losses) / len(losses), sum(accuracies) / len(accuracies)

    def _read_resume_state(self, dataset_fingerprint: str) -> dict[str, Any]:
        if not self.config.resume_from:
            return {}
        if self.config.resume_mode == "weights_only":
            self.log(
                ">> resumed LoRA / DoRA weights only; using a fresh optimizer/scheduler at step 0"
            )
            return {}
        path = Path(resume_state_path_for(self.config.resume_from))
        if not path.is_file():
            self.log(
                f">> continue mode requested, but train state was not found at {path}; "
                "using LoRA / DoRA weights with a fresh optimizer/scheduler at step 0"
            )
            return {}
        state = load_train_state(path)
        changed = state.get("dataset_fingerprint") not in {None, dataset_fingerprint}
        if changed:
            self.log(
                ">> warning: dataset fingerprint changed since the checkpoint; "
                "continuing optimizer/scheduler state from the start of the saved epoch"
            )
            state["batch_in_epoch"] = 0
        return state

    def _restore_resume_state(
        self,
        state: Mapping[str, Any],
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: Any,
        *,
        rebuild_scheduler_horizon: bool = False,
    ) -> None:
        if not state:
            return
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if rebuild_scheduler_horizon:
            # Loading the old optimizer state also restores its terminal LR. Reposition
            # the newly constructed schedule so the first additional step uses the LR
            # for the extended horizon rather than the old run's final (often zero) LR.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                scheduler.step(max(0, int(state.get("step", 0))))
        if state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        if isinstance(state.get("rng"), Mapping):
            _restore_rng(state["rng"])
        self.log(
            f">> resumed optimizer/scheduler at step {int(state.get('step', 0))}, "
            f"epoch {int(state.get('epoch', 0)) + 1}"
        )

    def run(self) -> TrainingResult:
        config = self.config
        _seed_everything(config.seed)
        self.resolved_sample_seed = (
            secrets.randbelow(2**32) if config.sample_seed == -1 else config.sample_seed
        )
        self.log(
            f">> conditioning | train speaker ref: {config.speaker_ref_mode} | "
            f"train emotion ref: {config.emo_ref_mode} | "
            f"validation ref: {config.val_reference_mode}"
        )
        self.log(f">> training samples use seed {self.resolved_sample_seed}")
        self.write_status(
            phase="initializing",
            message="loading cached dataset",
            sample_seed=self.resolved_sample_seed,
            val_reference_mode=config.val_reference_mode,
        )
        train_dataset = LoraTrainDataset(
            self.dataset_dir,
            split="train",
            val_fraction=config.val_fraction,
            seed=config.seed,
            max_codes=config.max_codes,
            max_text_tokens=config.max_text_tokens,
            speaker_ref_mode=config.speaker_ref_mode,
            emo_ref_mode=config.emo_ref_mode,
        )
        val_speaker_ref_mode = "other" if config.val_reference_mode == "other" else "self"
        val_emo_ref_mode = (
            "follow_speaker" if config.val_reference_mode == "other" else "self"
        )
        try:
            val_dataset = LoraTrainDataset(
                self.dataset_dir,
                split="val",
                val_fraction=config.val_fraction,
                seed=config.seed,
                max_codes=config.max_codes,
                max_text_tokens=config.max_text_tokens,
                speaker_ref_mode=val_speaker_ref_mode,
                emo_ref_mode=val_emo_ref_mode,
            )
        except (ValueError, FileNotFoundError):
            val_dataset = None
        if val_dataset is not None and len(val_dataset) == 0:
            val_dataset = None

        val_count = len(val_dataset) if val_dataset is not None else 0
        plan = training_plan(
            len(train_dataset) + val_count,
            config.batch_size,
            config.grad_accumulation,
            config.epochs,
            config.max_steps,
            config.val_fraction,
            validation_count=val_count,
        )
        epoch_suggestion = suggested_epochs(
            plan["training_clips"], config.batch_size, config.grad_accumulation
        )
        self.log(
            f">> training plan | {training_plan_line(plan)} "
            f"suggested epochs for ~10,000 updates: {epoch_suggestion:,}"
        )
        micro_batches_per_epoch = plan["micro_batches_per_epoch"]
        steps_per_epoch = plan["optimizer_updates_per_epoch"]
        planned_steps = plan["total_optimizer_updates"]
        total_steps = planned_steps
        self.reporter = self.reporter or ProgressReporter(
            "training steps", total=total_steps, progress_file=self.state_dir / "progress.json"
        )
        self.reporter.set_stage("model download")
        self._prepare_base_variant()
        self.reporter.set_stage("load model")
        built = build_training_model(config, log=self.log)
        # Resume metadata may have overridden these values in build_training_model.
        first_adapter = next(iter(built.adapters.values()))
        config.rank = first_adapter.rank
        config.alpha = first_adapter.alpha
        config.adapter_type = "dora" if first_adapter.use_dora else "lora"
        self._prepare_reference()
        self.reporter.set_stage("training")

        device = torch.device(config.device)
        optimizer = _optimizer(config, built.parameters)
        try:
            scaler = torch.amp.GradScaler(
                device.type, enabled=device.type == "cuda" and config.mixed_precision == "fp16"
            )
        except TypeError:
            scaler = torch.cuda.amp.GradScaler(
                enabled=device.type == "cuda" and config.mixed_precision == "fp16"
            )
        state = self._read_resume_state(train_dataset.fingerprint)
        global_step = max(0, int(state.get("step", 0)))
        start_epoch = max(0, int(state.get("epoch", 0)))
        resume_batch = max(0, int(state.get("batch_in_epoch", 0)))
        extended_horizon = bool(state and global_step >= planned_steps)
        if extended_horizon:
            total_steps = global_step + planned_steps
        remaining_steps = max(0, total_steps - global_step)
        remaining_in_start_epoch = max(
            0,
            math.ceil(
                max(0, micro_batches_per_epoch - resume_batch)
                / config.grad_accumulation
            ),
        )
        if remaining_steps:
            extra_steps = max(0, remaining_steps - remaining_in_start_epoch)
            resume_epochs_needed = 1 + math.ceil(extra_steps / steps_per_epoch)
        else:
            resume_epochs_needed = 0
        effective_epochs = max(
            config.epochs,
            math.ceil(total_steps / steps_per_epoch),
            start_epoch + resume_epochs_needed,
        )
        scheduler = _scheduler(config, optimizer, total_steps)
        self._restore_resume_state(
            state,
            optimizer,
            scheduler,
            scaler,
            rebuild_scheduler_horizon=extended_horizon,
        )
        if state:
            self.log(
                f">> resumed at step {global_step}; training {remaining_steps} more steps"
            )
        self.reporter.total = total_steps
        self.reporter.completed = global_step
        epoch_index = max(0, start_epoch)
        batch_index = max(-1, resume_batch - 1)
        resume_next_epoch = start_epoch
        resume_next_batch = resume_batch
        best_val_loss = state.get("best_val_loss")
        best_val_loss = float(best_val_loss) if best_val_loss is not None else None
        early_stop_best = best_val_loss
        validations_without_improvement = 0
        ema_loss = state.get("ema_loss")
        ema_loss = float(ema_loss) if ema_loss is not None else None
        moving_losses: deque[float] = deque(
            (float(value) for value in state.get("moving_losses", [])), maxlen=50
        )
        initial_loss: float | None = None
        final_loss: float | None = None
        speed_ema: float | None = None
        speed_values: list[float] = []
        last_step_time = time.perf_counter()
        last_val_loss: float | None = None
        last_validation_step: int | None = None
        stopped = False
        early_stopped = False
        early_stop_reason = ""
        starting_step = global_step
        amp_dtype = _dtype(config.mixed_precision)
        amp_enabled = device.type == "cuda" and amp_dtype != torch.float32
        val_loader = (
            DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=device.type == "cuda",
                collate_fn=collate,
            )
            if val_dataset is not None
            else None
        )

        def update_early_stopping(validation_loss: float) -> bool:
            nonlocal early_stop_best, validations_without_improvement, early_stop_reason
            if config.early_stop_patience <= 0 or not math.isfinite(validation_loss):
                return False
            if (
                early_stop_best is None
                or validation_loss < early_stop_best - config.early_stop_min_delta
            ):
                early_stop_best = validation_loss
                validations_without_improvement = 0
                return False
            validations_without_improvement += 1
            if validations_without_improvement < config.early_stop_patience:
                return False
            early_stop_reason = (
                f"early stopping after {validations_without_improvement} consecutive validation "
                f"check(s) without an improvement of at least {config.early_stop_min_delta:.4g}; "
                f"best validation loss {early_stop_best:.4f}"
            )
            self.log(">> " + early_stop_reason)
            return True

        self.write_status(
            phase="training",
            step=global_step,
            total_steps=total_steps,
            epoch=start_epoch + 1,
            total_epochs=effective_epochs,
            message="training",
        )
        optimizer.zero_grad(set_to_none=True)
        try:
            for epoch_index in range(start_epoch, effective_epochs):
                if global_step >= total_steps:
                    break
                train_dataset.set_epoch(epoch_index)
                sampler = LengthBucketBatchSampler(
                    train_dataset.lengths,
                    config.batch_size,
                    shuffle=True,
                    drop_last=False,
                    seed=config.seed,
                )
                sampler.set_epoch(epoch_index)
                train_loader = DataLoader(
                    train_dataset,
                    batch_sampler=sampler,
                    num_workers=config.num_workers,
                    pin_memory=device.type == "cuda",
                    collate_fn=collate,
                )
                micro_count = 0
                group_loss = 0.0
                group_accuracy = 0.0
                batch_count = len(train_loader)
                for batch_index, batch in enumerate(train_loader):
                    if epoch_index == start_epoch and batch_index < resume_batch:
                        continue
                    batch = _batch_to_device(batch, device)
                    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
                        loss, values = gpt_train_step_loss(
                            built.model,
                            batch,
                            mel_loss_weight=config.mel_loss_weight,
                            text_loss_weight=config.text_loss_weight,
                            label_smoothing=config.label_smoothing,
                        )
                        scaled_loss = loss / config.grad_accumulation
                    scaler.scale(scaled_loss).backward()
                    micro_count += 1
                    group_loss += float(loss.detach())
                    group_accuracy += float(values["mel_accuracy"].detach())
                    boundary = micro_count >= config.grad_accumulation or batch_index + 1 == batch_count
                    if not boundary:
                        continue

                    scaler.unscale_(optimizer)
                    if micro_count < config.grad_accumulation:
                        correction = config.grad_accumulation / micro_count
                        for parameter in built.parameters:
                            if parameter.grad is not None:
                                parameter.grad.mul_(correction)
                    if config.max_grad_norm > 0:
                        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                            built.parameters, config.max_grad_norm
                        )
                    else:
                        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                            built.parameters, float("inf")
                        )
                    grad_norm = float(grad_norm_tensor)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1

                    step_loss = group_loss / micro_count
                    step_accuracy = group_accuracy / micro_count
                    group_loss = 0.0
                    group_accuracy = 0.0
                    micro_count = 0
                    initial_loss = step_loss if initial_loss is None else initial_loss
                    final_loss = step_loss
                    ema_loss = step_loss if ema_loss is None else 0.9 * ema_loss + 0.1 * step_loss
                    moving_losses.append(step_loss)
                    moving_average = sum(moving_losses) / len(moving_losses)
                    now = time.perf_counter()
                    instantaneous_speed = 1.0 / max(now - last_step_time, 1e-9)
                    last_step_time = now
                    speed_ema = (
                        instantaneous_speed
                        if speed_ema is None
                        else 0.9 * speed_ema + 0.1 * instantaneous_speed
                    )
                    speed_values.append(speed_ema)
                    eta = (total_steps - global_step) / max(speed_ema, 1e-9)
                    memory = memory_stats(device)
                    lr = float(optimizer.param_groups[0]["lr"])
                    elapsed = now - self.started_perf
                    description = (
                        f"step {global_step}/{total_steps} | ep {epoch_index + 1}/{effective_epochs} | "
                        f"loss {step_loss:.3f} (avg {moving_average:.3f}) | acc {step_accuracy:.2f} | "
                        f"lr {lr:.1e} | {speed_ema:.2f} it/s | ETA {_format_eta(eta)} | "
                        f"VRAM {memory['allocated_gb']:.1f}/{gpu_total_gb(device.index or 0):.0f} GB"
                    )
                    payload = {
                        "step": global_step,
                        "epoch": epoch_index + 1,
                        "loss": step_loss,
                        "avg_loss": ema_loss,
                        "moving_avg_loss": moving_average,
                        "mel_accuracy": step_accuracy,
                        "lr": lr,
                        "grad_norm": grad_norm,
                        "it_s": speed_ema,
                        "elapsed_s": elapsed,
                        "eta_s": eta,
                        "vram_used_gb": memory["allocated_gb"],
                        "vram_peak_gb": memory["peak_allocated_gb"],
                    }
                    if global_step % config.log_every_steps == 0:
                        self.metric(payload)
                        with self.log_path.open("a", encoding="utf-8", newline="\n") as handle:
                            handle.write(description + "\n")
                        self.reporter.update(
                            global_step,
                            total=total_steps,
                            desc=description,
                            extra={"speed": speed_ema, "speed_unit": "it/s"},
                        )
                    self.write_status(
                        phase="training",
                        step=global_step,
                        total_steps=total_steps,
                        epoch=epoch_index + 1,
                        total_epochs=effective_epochs,
                        loss=step_loss,
                        avg_loss=ema_loss,
                        val_loss=last_val_loss,
                        lr=lr,
                        grad_norm=grad_norm,
                        it_s=speed_ema,
                        eta_s=eta,
                        elapsed_s=elapsed,
                        vram_used_gb=memory["allocated_gb"],
                        message="training",
                        last_checkpoint=self.last_checkpoint,
                        last_sample=self.last_sample,
                    )

                    next_epoch = epoch_index
                    next_batch = batch_index + 1
                    if next_batch >= batch_count:
                        next_epoch, next_batch = epoch_index + 1, 0
                    resume_next_epoch, resume_next_batch = next_epoch, next_batch
                    if config.save_every_steps and global_step % config.save_every_steps == 0:
                        path = self.adapter_dir / f"{config.name}_step_{global_step:06d}.safetensors"
                        self.save_checkpoint(
                            path,
                            built,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            step=global_step,
                            epochs_completed=epoch_index + 1,
                            next_epoch=next_epoch,
                            next_batch=next_batch,
                            dataset_fingerprint=train_dataset.fingerprint,
                            best_val_loss=best_val_loss,
                            ema_loss=ema_loss,
                            moving_losses=moving_losses,
                        )

                    if (
                        val_loader is not None
                        and config.val_every_steps
                        and global_step % config.val_every_steps == 0
                    ):
                        last_val_loss, val_accuracy = self.validate(built, val_loader, device)
                        last_validation_step = global_step
                        self.metric(
                            {
                                "event": "validation",
                                "step": global_step,
                                "epoch": epoch_index + 1,
                                "val_loss": last_val_loss,
                                "val_mel_accuracy": val_accuracy,
                                "reference_mode": config.val_reference_mode,
                            }
                        )
                        self.log(
                            f">> validation step {global_step}: loss {last_val_loss:.4f}, "
                            f"mel accuracy {val_accuracy:.3f}"
                        )
                        if config.save_best and (
                            best_val_loss is None or last_val_loss < best_val_loss
                        ):
                            best_val_loss = last_val_loss
                            self.save_checkpoint(
                                self.best_path,
                                built,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                scaler=scaler,
                                step=global_step,
                                epochs_completed=epoch_index + 1,
                                next_epoch=next_epoch,
                                next_batch=next_batch,
                                dataset_fingerprint=train_dataset.fingerprint,
                                best_val_loss=best_val_loss,
                                ema_loss=ema_loss,
                                moving_losses=moving_losses,
                                keep=True,
                            )

                        if update_early_stopping(last_val_loss):
                            early_stopped = True
                            break

                    if self.stop_path.is_file():
                        stopped = True
                        break
                    if global_step >= total_steps:
                        break

                resume_batch = 0
                if stopped or early_stopped:
                    break

                if val_loader is not None and last_validation_step != global_step:
                    last_val_loss, val_accuracy = self.validate(built, val_loader, device)
                    last_validation_step = global_step
                    self.metric(
                        {
                            "event": "validation",
                            "step": global_step,
                            "epoch": epoch_index + 1,
                            "val_loss": last_val_loss,
                            "val_mel_accuracy": val_accuracy,
                            "reference_mode": config.val_reference_mode,
                        }
                    )
                    self.log(
                        f">> epoch {epoch_index + 1} validation: loss {last_val_loss:.4f}, "
                        f"mel accuracy {val_accuracy:.3f}"
                    )
                    if config.save_best and (
                        best_val_loss is None or last_val_loss < best_val_loss
                    ):
                        best_val_loss = last_val_loss
                        self.save_checkpoint(
                            self.best_path,
                            built,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            step=global_step,
                            epochs_completed=epoch_index + 1,
                            next_epoch=epoch_index + 1,
                            next_batch=0,
                            dataset_fingerprint=train_dataset.fingerprint,
                            best_val_loss=best_val_loss,
                            ema_loss=ema_loss,
                            moving_losses=moving_losses,
                            keep=True,
                        )

                    if update_early_stopping(last_val_loss):
                        early_stopped = True

                if early_stopped:
                    break

                if config.save_every_epochs and (epoch_index + 1) % config.save_every_epochs == 0:
                    epoch_path = self.adapter_dir / f"{config.name}_epoch_{epoch_index + 1:03d}.safetensors"
                    self.save_checkpoint(
                        epoch_path,
                        built,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        step=global_step,
                        epochs_completed=epoch_index + 1,
                        next_epoch=epoch_index + 1,
                        next_batch=0,
                        dataset_fingerprint=train_dataset.fingerprint,
                        best_val_loss=best_val_loss,
                        ema_loss=ema_loss,
                        moving_losses=moving_losses,
                    )

                if (
                    config.sample_enabled
                    and (epoch_index + 1) % config.sample_every_epochs == 0
                ):
                    temp_adapter = self.adapter_dir / f".{config.name}_sample_epoch_{epoch_index + 1:03d}.safetensors"
                    save_lora(
                        temp_adapter,
                        built.adapters,
                        built.full_modules,
                        self._metadata(global_step, epoch_index + 1, list(built.adapters)),
                        dtype=_dtype(config.save_dtype),
                    )
                    configured_reference = Path(config.sample_reference).expanduser() if config.sample_reference else None
                    sample_reference = (
                        configured_reference
                        if configured_reference is not None and configured_reference.is_file()
                        else self.reference_copy
                    )
                    sample_path = self.adapter_dir / "samples" / f"epoch_{epoch_index + 1:03d}.wav"
                    result = generate_training_sample(
                        config,
                        adapter_path=temp_adapter,
                        reference_path=sample_reference,
                        output_path=sample_path,
                        epoch=epoch_index + 1,
                        seed=self.resolved_sample_seed,
                        log=self.log,
                    )
                    temp_adapter.unlink(missing_ok=True)
                    if result.generated:
                        self.last_sample = result.path
                        self.write_status(last_sample=self.last_sample)

            trained_steps = global_step - starting_step
            if trained_steps <= 0:
                raise RuntimeError(
                    "nothing to train: "
                    f"resumed at step {starting_step}, planned total {total_steps}, "
                    f"and exhausted epochs {start_epoch + 1}-{effective_epochs}"
                )

            if stopped:
                interrupted = self.adapter_dir / f"{config.name}_interrupted.safetensors"
                self.save_checkpoint(
                    interrupted,
                    built,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    step=global_step,
                    epochs_completed=min(effective_epochs, epoch_index + 1),
                    next_epoch=resume_next_epoch,
                    next_batch=resume_next_batch,
                    dataset_fingerprint=train_dataset.fingerprint,
                    best_val_loss=best_val_loss,
                    ema_loss=ema_loss,
                    moving_losses=moving_losses,
                    keep=True,
                )
                self.write_status(
                    phase="stopped",
                    step=global_step,
                    total_steps=total_steps,
                    epoch=epoch_index + 1,
                    message="stopped gracefully after the current optimizer step",
                    last_checkpoint=str(interrupted.resolve()),
                )
                self.log(f">> stop.flag honored; interrupted checkpoint saved to {interrupted}")
                result_status = "stopped"
                final_path = interrupted
            else:
                final_path = self.adapter_dir / f"{config.name}.safetensors"
                self.save_checkpoint(
                    final_path,
                    built,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    step=global_step,
                    epochs_completed=min(effective_epochs, epoch_index + 1),
                    next_epoch=resume_next_epoch,
                    next_batch=resume_next_batch,
                    dataset_fingerprint=train_dataset.fingerprint,
                    best_val_loss=best_val_loss,
                    ema_loss=ema_loss,
                    moving_losses=moving_losses,
                    keep=True,
                )
                if not early_stopped:
                    self.reporter.finish()
                terminal_phase = "stopped" if early_stopped else "complete"
                terminal_message = early_stop_reason if early_stopped else "training complete"
                self.write_status(
                    phase=terminal_phase,
                    step=global_step,
                    total_steps=total_steps,
                    epoch=min(effective_epochs, epoch_index + 1),
                    loss=final_loss,
                    avg_loss=ema_loss,
                    val_loss=last_val_loss,
                    eta_s=0.0,
                    message=terminal_message,
                    last_checkpoint=str(final_path.resolve()),
                    last_sample=self.last_sample,
                )
                self.log(f">> final LoRA / DoRA saved to {final_path}")
                result_status = terminal_phase
        except BaseException as exc:
            self.write_status(phase="failed", message=str(exc), elapsed_s=time.perf_counter() - self.started_perf)
            self.log(f">> training failed: {exc}")
            raise
        finally:
            if built.block_swap is not None:
                built.block_swap.remove(to_cpu=True)

        stats = memory_stats(device)
        _analysis_path, recommended_checkpoint = self._write_automatic_analysis()
        self._write_speaking_rate_calibration()
        terminal_status = read_json_retry(self.status_path, {}) or {}
        terminal_message = str(
            terminal_status.get("message")
            or ("training complete" if result_status == "complete" else "training stopped")
        )
        if config.auto_evaluate_checkpoints and val_count > 0:
            del built
            del optimizer, scheduler, scaler
            del train_loader, val_loader, batch, loss, values
            del first_adapter, scaled_loss, grad_norm_tensor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                recommended_checkpoint = self._run_automatic_evaluation(
                    terminal_phase=result_status,
                    terminal_message=terminal_message,
                    recommended_checkpoint=recommended_checkpoint,
                )
            except Exception as exc:
                self.log(
                    f">> automatic checkpoint evaluation failed but training is safe: {exc}"
                )
                self.write_status(
                    phase=result_status,
                    message=terminal_message,
                    recommended_checkpoint=recommended_checkpoint,
                )
        elif config.auto_evaluate_checkpoints:
            self.log(
                ">> automatic checkpoint evaluation skipped: the validation split "
                "contains no items"
            )
            if recommended_checkpoint:
                self.write_status(recommended_checkpoint=recommended_checkpoint)
        elif recommended_checkpoint:
            self.write_status(recommended_checkpoint=recommended_checkpoint)
        return TrainingResult(
            status=result_status,
            step=global_step,
            total_steps=total_steps,
            epoch=min(effective_epochs, epoch_index + 1),
            output_path=str(final_path.resolve()),
            best_path=str(self.best_path.resolve()) if self.best_path.is_file() else "",
            best_val_loss=best_val_loss,
            initial_loss=initial_loss,
            final_loss=final_loss,
            avg_it_s=sum(speed_values) / len(speed_values) if speed_values else 0.0,
            peak_vram_gb=stats["peak_allocated_gb"],
            elapsed_s=time.perf_counter() - self.started_perf,
        )


def _format_eta(seconds: float) -> str:
    value = max(0, int(seconds))
    minutes, secs = divmod(value, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def run_training(
    config: TrainConfig | Mapping[str, Any],
    *,
    state_dir: str | Path | None = None,
    reporter: ProgressReporter | None = None,
) -> TrainingResult:
    return LoraTrainer(config, state_dir=state_dir, reporter=reporter).run()


__all__ = [
    "BuiltTrainingModel",
    "LoraTrainer",
    "TrainingResult",
    "build_training_model",
    "run_training",
]
