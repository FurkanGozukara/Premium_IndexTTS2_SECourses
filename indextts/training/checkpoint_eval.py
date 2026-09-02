"""Teacher-forced comparison of saved adapters on the original data split."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.lora.apply import apply_lora, move_adapters_to_device, remove_lora
from indextts.quant.convrot_int8 import load_gpt_checkpoint
from indextts.runtime.progress import ProgressReporter
from indextts.utils.atomic_json import write_json_atomic

from .analysis import (
    GENERALIZATION_LEGEND,
    _write_text_atomic,
    checkpoint_descriptor,
    classify_epoch_phases,
    discover_checkpoints,
    load_training_analysis,
)
from .dataset import LoraTrainDataset, collate
from .model_forward import gpt_train_step_loss


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


@dataclass
class CheckpointEvalConfig:
    adapter_dir: str
    dataset_dir: str = ""
    checkpoints: list[str] = field(default_factory=list)
    include_base: bool = True
    strengths: list[float] = field(default_factory=lambda: [1.0])
    train_subset: int = 48
    batch_size: int = 4
    max_batches: int = 0
    device: str = "cuda:0"
    base_variant: str = ""
    base_dtype: str = ""
    model_dir: str = ""
    model_config: str = ""
    attention_backend: str = ""
    val_fraction: float | None = None
    seed: int | None = None

    def validate(self) -> "CheckpointEvalConfig":
        self.adapter_dir = str(Path(self.adapter_dir).expanduser().resolve())
        self.dataset_dir = str(self.dataset_dir or "")
        self.checkpoints = [str(item) for item in self.checkpoints if str(item).strip()]
        normalized_strengths: list[float] = []
        for raw in self.strengths or [1.0]:
            value = float(raw)
            if not math.isfinite(value) or not 0.0 <= value <= 4.0:
                raise ValueError("checkpoint strengths must be finite values from 0 to 4")
            if value not in normalized_strengths:
                normalized_strengths.append(value)
        self.strengths = normalized_strengths or [1.0]
        self.train_subset = max(0, int(self.train_subset))
        self.batch_size = max(1, int(self.batch_size))
        self.max_batches = max(0, int(self.max_batches))
        self.device = str(self.device or "cuda:0")
        if self.val_fraction is not None:
            self.val_fraction = min(0.5, max(0.0, float(self.val_fraction)))
        if self.seed is not None:
            self.seed = int(self.seed)
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any] | "CheckpointEvalConfig"
    ) -> "CheckpointEvalConfig":
        if isinstance(value, cls):
            return cls(**value.to_dict()).validate()
        if not isinstance(value, Mapping):
            raise TypeError("checkpoint evaluation config must be a mapping")
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: item for key, item in value.items() if key in allowed}).validate()

    @classmethod
    def from_json(cls, path: str | Path) -> "CheckpointEvalConfig":
        with Path(path).open("r", encoding="utf-8-sig") as handle:
            return cls.from_dict(json.load(handle))


@dataclass
class CheckpointEvalRow:
    label: str
    path: str
    kind: str
    epoch: int | None
    steps: int
    strength: float
    val_loss: float | None
    val_mel_loss: float | None
    val_text_loss: float | None
    val_accuracy: float | None
    train_loss: float | None
    train_accuracy: float | None
    gap: float | None
    phase: str
    elapsed_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CheckpointEvalRow":
        allowed = {item.name for item in fields(cls)}
        payload = {key: item for key, item in value.items() if key in allowed}
        payload.setdefault("phase", "unknown")
        return cls(**payload)


@dataclass
class CheckpointEvalReport:
    adapter_dir: str
    dataset_dir: str
    val_items: int
    train_subset_items: int
    rows: list[CheckpointEvalRow]
    best_label: str
    best_path: str
    recommended_checkpoint: str
    summary_markdown: str
    device: str
    generated_at: str
    elapsed_s: float

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["rows"] = [item.to_dict() for item in self.rows]
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CheckpointEvalReport":
        payload = dict(value)
        payload["rows"] = [
            item if isinstance(item, CheckpointEvalRow) else CheckpointEvalRow.from_dict(item)
            for item in payload.get("rows", [])
            if isinstance(item, (CheckpointEvalRow, Mapping))
        ]
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: item for key, item in payload.items() if key in allowed})


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _training_defaults(adapter_dir: Path, checkpoints: list[str]) -> dict[str, Any]:
    config = _read_json(adapter_dir / "train_config.json")
    if config:
        return config
    for raw in checkpoints:
        try:
            descriptor = checkpoint_descriptor(raw)
            metadata = descriptor.get("metadata") or {}
            value = metadata.get("train_config") or {}
            if isinstance(value, dict) and value:
                return dict(value)
        except (OSError, RuntimeError, ValueError):
            continue
    return {}


def _resolved_config(config: CheckpointEvalConfig) -> tuple[CheckpointEvalConfig, dict[str, Any]]:
    result = CheckpointEvalConfig.from_dict(config)
    adapter_dir = Path(result.adapter_dir)
    if not result.checkpoints:
        result.checkpoints = [item["path"] for item in discover_checkpoints(adapter_dir)]
    resolved_checkpoints: list[str] = []
    for raw in result.checkpoints:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = adapter_dir / path
        path = path.resolve()
        if path.is_file() and not path.name.startswith("."):
            resolved_checkpoints.append(str(path))
    result.checkpoints = resolved_checkpoints
    defaults = _training_defaults(adapter_dir, result.checkpoints)
    result.dataset_dir = str(result.dataset_dir or defaults.get("dataset_dir") or "")
    if not result.dataset_dir:
        raise ValueError("dataset_dir is missing from the evaluation config and adapter metadata")
    dataset = Path(result.dataset_dir).expanduser()
    if not dataset.is_absolute():
        dataset = Path.cwd() / dataset
    result.dataset_dir = str(dataset.resolve())
    result.base_variant = str(result.base_variant or defaults.get("base_variant") or "bf16")
    result.base_dtype = str(result.base_dtype or defaults.get("base_dtype") or "bf16")
    result.model_dir = str(result.model_dir or defaults.get("model_dir") or "models")
    model_dir = Path(result.model_dir).expanduser()
    if not model_dir.is_absolute():
        model_dir = Path.cwd() / model_dir
    result.model_dir = str(model_dir.resolve())
    result.model_config = str(
        result.model_config
        or defaults.get("model_config")
        or (model_dir / "config.yaml")
    )
    model_config = Path(result.model_config).expanduser()
    if not model_config.is_absolute():
        model_config = Path.cwd() / model_config
    result.model_config = str(model_config.resolve())
    result.attention_backend = str(
        result.attention_backend or defaults.get("attention_backend") or "sdpa"
    )
    if result.val_fraction is None:
        result.val_fraction = float(defaults.get("val_fraction", 0.05))
    if result.seed is None:
        result.seed = int(defaults.get("seed", 42))
    if result.device.startswith("cuda") and not torch.cuda.is_available():
        result.device = "cpu"
    if result.device == "cpu" and result.attention_backend == "flash_attention_2":
        result.attention_backend = "sdpa"
    return result, defaults


def build_evaluation_model(config: CheckpointEvalConfig) -> UnifiedVoice:
    """Load a fresh base GPT with no adapter attached."""

    device = torch.device(config.device)
    dtype_name = config.base_dtype
    if device.type == "cpu":
        dtype_name = "fp32"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]
    model_cfg = OmegaConf.load(config.model_config)
    model = UnifiedVoice(
        **model_cfg.gpt,
        spk_cond_mode="campplus",
        attention_backend=config.attention_backend,
    )
    checkpoint = (
        Path(config.model_dir) / "gpt_int8_convrot.safetensors"
        if config.base_variant == "int8_convrot"
        else Path(config.model_dir) / str(model_cfg.gpt_checkpoint)
    )
    load_gpt_checkpoint(model, str(checkpoint), device="cpu", dtype=dtype, strict=False)
    model.requires_grad_(False)
    model.to(device)
    model.eval()
    return model


def _loader_metrics(
    model: torch.nn.Module,
    loader: DataLoader | None,
    device: torch.device,
    *,
    max_batches: int,
    loss_options: Mapping[str, Any],
) -> dict[str, float | None]:
    if loader is None:
        return {"loss": None, "mel_loss": None, "text_loss": None, "accuracy": None}
    values: dict[str, list[float]] = {
        "loss": [],
        "mel_loss": [],
        "text_loss": [],
        "accuracy": [],
    }
    amp_enabled = device.type == "cuda"
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if max_batches and index >= max_batches:
                break
            moved = {
                key: item.to(device, non_blocking=True) if isinstance(item, torch.Tensor) else item
                for key, item in batch.items()
            }
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                total, metrics = gpt_train_step_loss(
                    model,
                    moved,
                    mel_loss_weight=float(loss_options.get("mel_loss_weight", 1.0)),
                    text_loss_weight=float(loss_options.get("text_loss_weight", 0.1)),
                    label_smoothing=float(loss_options.get("label_smoothing", 0.0)),
                )
            for key, tensor in (
                ("loss", total),
                ("mel_loss", metrics["mel_loss"]),
                ("text_loss", metrics["text_loss"]),
                ("accuracy", metrics["mel_accuracy"]),
            ):
                values[key].append(float(tensor.detach().float().cpu()))
    return {
        key: (sum(items) / len(items) if items else None)
        for key, items in values.items()
    }


def _make_loader(dataset: Any, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate,
    )


def _row_label(descriptor: Mapping[str, Any], strength: float) -> str:
    label = str(descriptor["label"])
    return label if abs(strength - 1.0) < 1e-9 else f"{label} @{strength:g}"


def _summary_markdown(
    rows: list[CheckpointEvalRow],
    best: CheckpointEvalRow | None,
    analysis: Any = None,
) -> str:
    base = next((row for row in rows if row.kind == "base"), None)
    lines: list[str] = []
    if base is not None and base.val_loss is not None:
        lines.append(
            f"Without any adapter the base model scores {base.val_loss:.2f}; every checkpoint below "
            "that number learned something from your data."
        )
    if best is None or best.val_loss is None:
        lines.append("No adapter checkpoint produced a measurable validation score.")
        return "  \n".join(lines) + "\n\n" + GENERALIZATION_LEGEND
    accuracy = (
        f", {best.val_accuracy * 100:.1f}% next-token accuracy"
        if best.val_accuracy is not None
        else ""
    )
    lines.append(
        f"**Best measured generalization: {best.label}** (validation loss {best.val_loss:.2f}{accuracy} "
        "on unseen sentences)."
    )
    overfit = sorted(
        (
            row
            for row in rows
            if row.phase == "overfitting"
            and row.epoch is not None
            and abs(row.strength - 1.0) < 1e-9
        ),
        key=lambda row: int(row.epoch or 0),
    )
    if overfit:
        onset = getattr(analysis, "overfit_start_epoch", None)
        if onset is not None and int(onset) != int(overfit[0].epoch or 0):
            lines.append(
                f"**Overfitting starts at epoch {int(onset)}** according to the training log; among the "
                f"saved checkpoints, epoch {overfit[0].epoch} is the earliest one that is already overfitted. "
                "Later checkpoints score worse on unseen sentences even while fitting the training clips "
                "more closely."
            )
        else:
            lines.append(
                f"**Overfitting starts at epoch {overfit[0].epoch}.** Later checkpoints score worse on "
                "unseen sentences even while fitting the training clips more closely."
            )
        final = max(
            (
                row
                for row in rows
                if row.kind != "base"
                and row.epoch is not None
                and row.val_loss is not None
                and abs(row.strength - 1.0) < 1e-9
            ),
            key=lambda row: int(row.epoch or 0),
            default=None,
        )
        if final is not None and final.epoch != best.epoch:
            detail = (
                f"The final measured epoch {final.epoch} scores {final.val_loss:.2f}"
            )
            if float(best.val_loss) != 0:
                increase = 100.0 * (float(final.val_loss) / float(best.val_loss) - 1.0)
                detail += f" ({increase:+.1f}% vs best)"
            if best.train_accuracy is not None and final.train_accuracy is not None:
                detail += (
                    f" while training-text accuracy climbed from {best.train_accuracy * 100:.1f}% "
                    f"to {final.train_accuracy * 100:.1f}%"
                )
            lines.append(detail + ".")
    else:
        measured_epochs = [
            row.epoch
            for row in rows
            if row.kind != "base"
            and row.epoch is not None
            and row.val_loss is not None
            and abs(row.strength - 1.0) < 1e-9
        ]
        if best.epoch is not None and measured_epochs and best.epoch == max(measured_epochs):
            lines.append(
                "The latest measured epoch is also the best, so training was still improving."
            )
        else:
            lines.append(
                "Later measured epochs stayed near the best score, so the run reached a plateau "
                "without a sustained overfitting rise."
            )
    lines.append(f"**Recommended checkpoint:** `{best.path}`.")
    return "  \n".join(lines) + "\n\n" + GENERALIZATION_LEGEND


def evaluate_checkpoints(
    config: CheckpointEvalConfig | Mapping[str, Any],
    reporter: ProgressReporter | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> CheckpointEvalReport:
    started = time.perf_counter()
    cfg, train_defaults = _resolved_config(CheckpointEvalConfig.from_dict(config))
    device = torch.device(cfg.device)
    descriptors = [checkpoint_descriptor(path) for path in cfg.checkpoints]
    total = (1 if cfg.include_base else 0) + len(descriptors) * len(cfg.strengths)
    progress = reporter or ProgressReporter("checkpoints", total=total)
    progress.total = total
    progress.set_stage("load model")
    model = build_evaluation_model(cfg)

    dataset_options = {
        "val_fraction": float(cfg.val_fraction or 0.0),
        "seed": int(cfg.seed or 0),
        "max_codes": int(train_defaults.get("max_codes", 1500)),
        "max_text_tokens": int(train_defaults.get("max_text_tokens", 600)),
    }
    val_dataset = LoraTrainDataset(
        cfg.dataset_dir, split="val", speaker_ref_mode="self", **dataset_options
    )
    if len(val_dataset) == 0:
        raise ValueError("the configured validation split contains no items")
    val_loader = _make_loader(val_dataset, cfg.batch_size)
    train_loader: DataLoader | None = None
    train_items = 0
    if cfg.train_subset > 0:
        train_dataset = LoraTrainDataset(
            cfg.dataset_dir, split="train", speaker_ref_mode="self", **dataset_options
        )
        indices = list(range(min(cfg.train_subset, len(train_dataset))))
        train_items = len(indices)
        if indices:
            train_loader = _make_loader(Subset(train_dataset, indices), cfg.batch_size)

    rows: list[CheckpointEvalRow] = []
    completed = 0

    def evaluate_row(
        *, label: str, path: str, kind: str, epoch: int | None, steps: int, strength: float
    ) -> None:
        nonlocal completed
        if cancel_callback is not None and cancel_callback():
            raise RuntimeError("Checkpoint evaluation canceled")
        row_started = time.perf_counter()
        val = _loader_metrics(
            model,
            val_loader,
            device,
            max_batches=cfg.max_batches,
            loss_options=train_defaults,
        )
        train = _loader_metrics(
            model,
            train_loader,
            device,
            max_batches=0,
            loss_options=train_defaults,
        )
        elapsed = time.perf_counter() - row_started
        val_loss = _float(val["loss"])
        train_loss = _float(train["loss"])
        row = CheckpointEvalRow(
            label=label,
            path=path,
            kind=kind,
            epoch=epoch,
            steps=steps,
            strength=strength,
            val_loss=val_loss,
            val_mel_loss=_float(val["mel_loss"]),
            val_text_loss=_float(val["text_loss"]),
            val_accuracy=_float(val["accuracy"]),
            train_loss=train_loss,
            train_accuracy=_float(train["accuracy"]),
            gap=(train_loss - val_loss) if train_loss is not None and val_loss is not None else None,
            phase="unknown",
            elapsed_s=elapsed,
        )
        rows.append(row)
        completed += 1
        progress.update(completed, total=total, desc=label)
        val_text = f"{val_loss:.3f}" if val_loss is not None else "n/a"
        val_acc = f"{(row.val_accuracy or 0.0) * 100:.1f}%"
        train_text = f"{train_loss:.3f}" if train_loss is not None else "n/a"
        train_acc = f"{(row.train_accuracy or 0.0) * 100:.1f}%"
        progress.log(
            f">> {label} [{kind}]: val {val_text} acc {val_acc} | train {train_text} "
            f"acc {train_acc} ({elapsed:.1f}s)"
        )

    progress.set_stage("evaluating")
    if cfg.include_base:
        evaluate_row(
            label="base model", path="", kind="base", epoch=None, steps=0, strength=0.0
        )
    try:
        for descriptor in descriptors:
            for strength in cfg.strengths:
                remove_lora(model)
                apply_lora(model, str(descriptor["path"]), strength)
                move_adapters_to_device(model, device)
                model.eval()
                evaluate_row(
                    label=_row_label(descriptor, strength),
                    path=str(descriptor["path"]),
                    kind=str(descriptor["kind"]),
                    epoch=(int(descriptor["epoch"]) if descriptor.get("epoch") else None),
                    steps=int(descriptor.get("steps") or 0),
                    strength=float(strength),
                )
    finally:
        remove_lora(model)

    primary = [
        row
        for row in rows
        if row.kind != "base" and abs(row.strength - 1.0) < 1e-9 and row.val_loss is not None
    ]
    if not primary:
        seen_paths: set[str] = set()
        primary = []
        for row in rows:
            if row.kind != "base" and row.path not in seen_paths and row.val_loss is not None:
                seen_paths.add(row.path)
                primary.append(row)
    epoch_losses: dict[int, float] = {}
    for row in primary:
        if row.epoch is not None and row.val_loss is not None:
            epoch_losses[row.epoch] = min(epoch_losses.get(row.epoch, row.val_loss), row.val_loss)
    phase_map, _best_epoch, _overfit = classify_epoch_phases(epoch_losses, 0.01)
    for row in rows:
        if row.kind == "base":
            row.phase = "base"
        elif abs(row.strength - 1.0) >= 1e-9:
            row.phase = "variant"
        else:
            row.phase = phase_map.get(int(row.epoch or 0), "unknown")
    candidates = primary or [
        row for row in rows if row.kind != "base" and row.val_loss is not None
    ]
    best = min(candidates, key=lambda row: float(row.val_loss)) if candidates else None
    summary = _summary_markdown(rows, best, load_training_analysis(cfg.adapter_dir))
    elapsed = time.perf_counter() - started
    progress.finish()
    return CheckpointEvalReport(
        adapter_dir=cfg.adapter_dir,
        dataset_dir=cfg.dataset_dir,
        val_items=len(val_dataset),
        train_subset_items=train_items,
        rows=rows,
        best_label=best.label if best else "",
        best_path=best.path if best else "",
        recommended_checkpoint=best.path if best else "",
        summary_markdown=summary,
        device=str(device),
        generated_at=_utc_now(),
        elapsed_s=elapsed,
    )


def write_checkpoint_eval(
    report: CheckpointEvalReport, adapter_dir: str | Path
) -> Path:
    root = Path(adapter_dir).expanduser().resolve() / "analysis"
    path = root / "checkpoint_eval.json"
    write_json_atomic(path, report.to_dict(), indent=2, ensure_ascii=False, allow_nan=False)
    _write_text_atomic(root / "checkpoint_eval.md", report.summary_markdown)
    return path


def load_checkpoint_eval(
    adapter_dir: str | Path,
) -> CheckpointEvalReport | None:
    path = Path(adapter_dir).expanduser().resolve() / "analysis" / "checkpoint_eval.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return CheckpointEvalReport.from_dict(value) if isinstance(value, dict) else None
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return None


__all__ = [
    "CheckpointEvalConfig",
    "CheckpointEvalReport",
    "CheckpointEvalRow",
    "build_evaluation_model",
    "evaluate_checkpoints",
    "load_checkpoint_eval",
    "write_checkpoint_eval",
]
