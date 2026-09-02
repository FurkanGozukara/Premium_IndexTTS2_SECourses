"""Training-log generalization analysis without loading torch or model weights."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

import pandas as pd
from safetensors import safe_open

from indextts.utils.atomic_json import replace_with_retry, write_json_atomic

from .charts import empty_series_frame, load_metrics


ANALYSIS_SERIES = (
    "train loss",
    "validation (improving)",
    "validation (overfitting)",
)
GENERALIZATION_LEGEND = (
    "Validation loss measures how well the adapter predicts sentences it never saw during "
    "training (lower is better). Training loss measures the clips it trains on. When training "
    "loss keeps falling but validation loss rises, the adapter is memorizing (overfitting)."
)

_EPOCH_RE = re.compile(r"_epoch_(\d+)$", re.IGNORECASE)
_STEP_RE = re.compile(r"_step_(\d+)$", re.IGNORECASE)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default


def inspect_lora(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read only safetensors metadata, keeping analysis independent from torch."""

    source = Path(path)
    with safe_open(str(source), framework="numpy", device="cpu") as handle:
        header = dict(handle.metadata() or {})
    train_config: dict[str, Any] = {}
    try:
        decoded = json.loads(header.get("train_config", "{}"))
        if isinstance(decoded, dict):
            train_config = decoded
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    return {
        "adapter_type": str(header.get("adapter_type", "lora")).lower(),
        "rank": _integer(header.get("rank")),
        "alpha": _finite_float(header.get("alpha")) or 0.0,
        "steps": _integer(header.get("trained_steps")),
        "epochs": _integer(header.get("epochs")),
        "dataset": str(header.get("dataset_name", "")),
        "train_config": train_config,
        "recommended_reference": str(header.get("recommended_reference", "")),
    }


@dataclass
class EpochSummary:
    epoch: int
    steps: int
    train_loss: float | None
    train_accuracy: float | None
    val_loss: float | None
    val_accuracy: float | None
    gap: float | None
    lr: float | None
    phase: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EpochSummary":
        allowed = {item.name for item in fields(cls)}
        payload = {key: item for key, item in value.items() if key in allowed}
        payload.setdefault("phase", "unknown")
        return cls(**payload)


@dataclass
class TrainingAnalysis:
    adapter_dir: str
    state_dir: str
    status: str
    epochs: list[EpochSummary]
    best_epoch: int | None
    best_step: int | None
    best_val_loss: float | None
    final_epoch: int | None
    final_val_loss: float | None
    overfit_start_epoch: int | None
    tolerance: float
    recommended_checkpoint: str
    recommended_label: str
    checkpoints: list[dict[str, Any]]
    summary_markdown: str
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["epochs"] = [item.to_dict() for item in self.epochs]
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingAnalysis":
        payload = dict(value)
        payload["epochs"] = [
            item if isinstance(item, EpochSummary) else EpochSummary.from_dict(item)
            for item in payload.get("epochs", [])
            if isinstance(item, (EpochSummary, Mapping))
        ]
        allowed = {item.name for item in fields(cls)}
        payload = {key: item for key, item in payload.items() if key in allowed}
        payload.setdefault("checkpoints", [])
        payload.setdefault("summary_markdown", "")
        payload.setdefault("generated_at", "")
        return cls(**payload)


def classify_epoch_phases(
    validation_losses: Mapping[int, float | None] | Sequence[tuple[int, float | None]],
    tolerance: float = 0.01,
) -> tuple[dict[int, str], int | None, int | None]:
    """Return phases, the earliest best epoch, and the sustained-rise epoch."""

    source = (
        validation_losses.items()
        if isinstance(validation_losses, Mapping)
        else validation_losses
    )
    values = sorted(
        (
            (int(epoch), loss)
            for epoch, raw_loss in source
            if (loss := _finite_float(raw_loss)) is not None
        ),
        key=lambda item: item[0],
    )
    if not values:
        return {}, None, None
    tolerance_value = max(0.0, float(tolerance))
    best_loss = min(loss for _, loss in values)
    best_epoch = next(epoch for epoch, loss in values if loss == best_loss)
    threshold = best_loss * (1.0 + tolerance_value)
    after_best = [(epoch, loss) for epoch, loss in values if epoch > best_epoch]
    overfit_start: int | None = None
    for index, (epoch, loss) in enumerate(after_best):
        if loss >= threshold and all(later >= threshold for _, later in after_best[index:]):
            overfit_start = epoch
            break

    phases: dict[int, str] = {}
    for epoch, _loss in values:
        if epoch < best_epoch:
            phase = "improving"
        elif epoch == best_epoch:
            phase = "best"
        elif overfit_start is not None and epoch >= overfit_start:
            phase = "overfitting"
        else:
            phase = "plateau"
        phases[epoch] = phase
    return phases, best_epoch, overfit_start


def checkpoint_descriptor(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Describe an adapter file using its location, name, and saved metadata."""

    source = Path(path).expanduser().resolve()
    info = inspect_lora(source)
    stem = source.stem
    epoch_match = _EPOCH_RE.search(stem)
    step_match = _STEP_RE.search(stem)
    if source.parent.name.lower() == "best":
        kind = "best"
    elif epoch_match:
        kind = "epoch"
    elif step_match:
        kind = "step"
    elif stem.lower().endswith("_interrupted"):
        kind = "interrupted"
    else:
        kind = "final"
    epoch = _integer(info.get("epochs")) or (
        int(epoch_match.group(1)) if epoch_match else 0
    )
    steps = _integer(info.get("steps")) or (
        int(step_match.group(1)) if step_match else 0
    )
    if kind == "best":
        label = f"best (epoch {epoch})" if epoch else "best"
        file_label = f"best_ep{epoch}" if epoch else "best"
    elif kind == "epoch":
        label = f"epoch {epoch}"
        file_label = f"epoch_{epoch:03d}"
    elif kind == "step":
        label = f"step {steps}"
        file_label = f"step_{steps:06d}"
    elif kind == "interrupted":
        label = f"interrupted (epoch {epoch})" if epoch else "interrupted"
        file_label = f"interrupted_ep{epoch:02d}" if epoch else "interrupted"
    else:
        label = f"final (epoch {epoch})" if epoch else "final"
        file_label = f"final_ep{epoch}" if epoch else "final"
    return {
        "path": str(source),
        "label": label,
        "file_label": file_label,
        "kind": kind,
        "epoch": epoch or None,
        "steps": steps,
        "metadata": info,
    }


def discover_checkpoints(adapter_dir: str | os.PathLike[str]) -> list[dict[str, Any]]:
    root = Path(adapter_dir).expanduser().resolve()
    candidates = list(root.glob("*.safetensors"))
    best_dir = root / "best"
    if best_dir.is_dir():
        candidates.extend(best_dir.glob("*.safetensors"))
    descriptors: list[dict[str, Any]] = []
    for path in candidates:
        if path.name.startswith("."):
            continue
        try:
            descriptors.append(checkpoint_descriptor(path))
        except Exception:
            continue
    order = {"best": 0, "epoch": 1, "step": 2, "interrupted": 3, "final": 4}
    descriptors.sort(
        key=lambda item: (
            order.get(str(item["kind"]), 9),
            int(item.get("epoch") or 0),
            int(item.get("steps") or 0),
            str(item["path"]).lower(),
        )
    )
    return descriptors


def _mean(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _last(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[-1])


def _recommended_checkpoint(
    adapter_dir: Path,
    checkpoints: list[dict[str, Any]],
    best_epoch: int | None,
    final_epoch: int | None,
) -> tuple[str, str]:
    best_files = [item for item in checkpoints if item["kind"] == "best"]
    epoch_files = [item for item in checkpoints if item["kind"] == "epoch"]
    final_files = [item for item in checkpoints if item["kind"] == "final"]
    interrupted = [item for item in checkpoints if item["kind"] == "interrupted"]
    chosen: dict[str, Any] | None = None
    if best_epoch is not None and best_files:
        exact = next(
            (item for item in best_files if int(item.get("epoch") or 0) == best_epoch),
            None,
        )
        if exact is not None or (final_epoch is not None and best_epoch < final_epoch):
            chosen = exact or best_files[0]
    if chosen is None and best_epoch is not None:
        chosen = next(
            (item for item in epoch_files if int(item.get("epoch") or 0) == best_epoch),
            None,
        )
    if chosen is None:
        chosen = final_files[0] if final_files else (interrupted[-1] if interrupted else None)
    if chosen is None and checkpoints:
        chosen = max(
            checkpoints,
            key=lambda item: (int(item.get("epoch") or 0), int(item.get("steps") or 0)),
        )
    if chosen is None:
        return "", "No checkpoint found"
    kind = str(chosen["kind"])
    epoch = int(chosen.get("epoch") or 0)
    if kind == "best":
        label = f"best/ (epoch {epoch})" if epoch else "best/"
    else:
        label = str(chosen["label"])
    return str(Path(chosen["path"]).resolve()), label


def _display_checkpoint(path: str, adapter_dir: Path) -> str:
    if not path:
        return "not available"
    source = Path(path)
    try:
        return source.relative_to(adapter_dir).as_posix()
    except ValueError:
        return str(source)


def _summary(
    *,
    status: str,
    epochs: list[EpochSummary],
    best_epoch: int | None,
    best_val_loss: float | None,
    overfit_start: int | None,
    final_epoch: int | None,
    final_val_loss: float | None,
    recommended_path: str,
    adapter_dir: Path,
    tolerance: float,
) -> str:
    by_epoch = {item.epoch: item for item in epochs}
    checkpoint_text = _display_checkpoint(recommended_path, adapter_dir)
    if status == "empty":
        verdict = "No training metrics were found, so the app cannot judge generalization yet."
    elif status == "no_validation":
        verdict = (
            "Validation was disabled for this run, so the app cannot judge overfitting. "
            f"The final saved checkpoint is selected by default: `{checkpoint_text}`."
        )
    elif best_epoch is None or best_val_loss is None:
        verdict = "The validation log is incomplete, so no best epoch could be selected."
    else:
        best = by_epoch.get(best_epoch)
        accuracy = best.val_accuracy if best is not None else None
        accuracy_text = f", {accuracy * 100:.1f}% next-token accuracy" if accuracy is not None else ""
        lines = [
            f"**Best generalization: epoch {best_epoch}** (validation loss {best_val_loss:.2f}{accuracy_text} on unseen sentences)."
        ]
        if status == "still_improving":
            lines.append(
                "The best result is the last epoch, so training was still improving and could run longer; "
                "the final file is the best one available."
            )
        elif status == "best_found" and overfit_start is not None:
            lines.append(
                f"**Overfitting starts at epoch {overfit_start}.** From there validation loss rises while "
                "training loss keeps falling, which means the adapter memorizes the training clips instead "
                "of learning the voice."
            )
            final = by_epoch.get(final_epoch or -1)
            if final_val_loss is not None and final_epoch is not None:
                detail = (
                    f"The final file (epoch {final_epoch}) is overfitted: validation loss "
                    f"{final_val_loss:.2f}"
                )
                if best_val_loss != 0:
                    increase = 100.0 * (final_val_loss / best_val_loss - 1.0)
                    detail += f" ({increase:+.1f}% vs best)"
                if (
                    best is not None
                    and final is not None
                    and best.train_accuracy is not None
                    and final.train_accuracy is not None
                ):
                    detail += (
                        f" while training-set accuracy climbed from {best.train_accuracy * 100:.1f}% "
                        f"to {final.train_accuracy * 100:.1f}%"
                    )
                lines.append(detail + ".")
        else:
            lines.append(
                f"Later epochs stayed within {tolerance * 100:.1f}% of the best validation loss, so the run reached a plateau "
                "without a sustained overfitting rise."
            )
        lines.append(f"**Recommended checkpoint:** `{checkpoint_text}` (epoch {best_epoch}).")
        if status == "best_found" and overfit_start is not None:
            lines.append(
                f"Tip: stop training around epoch {best_epoch}-{overfit_start} next time, or keep every epoch "
                "checkpoint (Keep last N = 0) and compare them in the Checkpoint Grid tab."
            )
        verdict = "  \n".join(lines)
    return verdict + "\n\n" + GENERALIZATION_LEGEND


def analyze_training_run(
    adapter_dir: str | os.PathLike[str],
    state_dir: str | os.PathLike[str] | None = None,
    *,
    tolerance: float = 0.01,
) -> TrainingAnalysis:
    adapter_root = Path(adapter_dir).expanduser().resolve()
    state_root = Path(state_dir).expanduser().resolve() if state_dir else adapter_root
    tolerance_value = max(0.0, float(tolerance))
    metrics = load_metrics(state_root)
    if metrics.empty:
        training_rows = metrics
        validation_rows = metrics
    else:
        if "event" in metrics:
            events = metrics["event"].astype("string").fillna("")
            validation_rows = metrics[events == "validation"]
            training_rows = metrics[events != "validation"]
        else:
            validation_rows = metrics.iloc[0:0]
            training_rows = metrics

    epoch_numbers: set[int] = set()
    for frame in (training_rows, validation_rows):
        if "epoch" in frame:
            epoch_numbers.update(
                int(value)
                for value in pd.to_numeric(frame["epoch"], errors="coerce").dropna()
                if int(value) > 0
            )

    validation_by_epoch: dict[int, dict[str, Any]] = {}
    if not validation_rows.empty and "epoch" in validation_rows:
        for _, row in validation_rows.iterrows():
            epoch = _integer(row.get("epoch"))
            val_loss = _finite_float(row.get("val_loss"))
            if epoch > 0 and val_loss is not None:
                validation_by_epoch[epoch] = {
                    "loss": val_loss,
                    "accuracy": _finite_float(row.get("val_mel_accuracy")),
                    "step": _integer(row.get("step")),
                }

    phases, best_epoch, overfit_start = classify_epoch_phases(
        {epoch: item["loss"] for epoch, item in validation_by_epoch.items()},
        tolerance_value,
    )
    summaries: list[EpochSummary] = []
    for epoch in sorted(epoch_numbers):
        if "epoch" in training_rows:
            mask = pd.to_numeric(training_rows["epoch"], errors="coerce") == epoch
            epoch_train = training_rows[mask]
        else:
            epoch_train = training_rows.iloc[0:0]
        validation = validation_by_epoch.get(epoch, {})
        train_loss = _mean(epoch_train, "loss")
        val_loss = _finite_float(validation.get("loss"))
        summaries.append(
            EpochSummary(
                epoch=epoch,
                steps=len(epoch_train),
                train_loss=train_loss,
                train_accuracy=_mean(epoch_train, "mel_accuracy"),
                val_loss=val_loss,
                val_accuracy=_finite_float(validation.get("accuracy")),
                gap=(train_loss - val_loss) if train_loss is not None and val_loss is not None else None,
                lr=_last(epoch_train, "lr"),
                phase=phases.get(epoch, "unknown"),
            )
        )

    final_epoch = max(epoch_numbers) if epoch_numbers else None
    final_validation = validation_by_epoch.get(final_epoch or -1, {})
    final_val_loss = _finite_float(final_validation.get("loss"))
    best_validation = validation_by_epoch.get(best_epoch or -1, {})
    best_val_loss = _finite_float(best_validation.get("loss"))
    best_step = _integer(best_validation.get("step")) or None
    if metrics.empty:
        status = "empty"
    elif not validation_by_epoch:
        status = "no_validation"
    elif best_epoch == max(validation_by_epoch):
        status = "still_improving"
    elif overfit_start is not None:
        status = "best_found"
    else:
        status = "plateau"

    checkpoints = discover_checkpoints(adapter_root)
    for item in checkpoints:
        epoch = int(item.get("epoch") or 0)
        if item["kind"] == "best" and best_epoch is not None and epoch == 0:
            epoch = best_epoch
            item["epoch"] = epoch
        if item["kind"] in {"final", "interrupted"} and final_epoch is not None and epoch == 0:
            epoch = final_epoch
            item["epoch"] = epoch
        validation = validation_by_epoch.get(epoch, {})
        item["val_loss"] = _finite_float(validation.get("loss"))
        item["phase"] = phases.get(epoch, "unknown")
        item.pop("metadata", None)
        item.pop("file_label", None)
    recommended_path, recommended_label = _recommended_checkpoint(
        adapter_root, checkpoints, best_epoch, final_epoch
    )
    summary = _summary(
        status=status,
        epochs=summaries,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        overfit_start=overfit_start,
        final_epoch=final_epoch,
        final_val_loss=final_val_loss,
        recommended_path=recommended_path,
        adapter_dir=adapter_root,
        tolerance=tolerance_value,
    )
    return TrainingAnalysis(
        adapter_dir=str(adapter_root),
        state_dir=str(state_root),
        status=status,
        epochs=summaries,
        best_epoch=best_epoch,
        best_step=best_step,
        best_val_loss=best_val_loss,
        final_epoch=final_epoch,
        final_val_loss=final_val_loss,
        overfit_start_epoch=overfit_start,
        tolerance=tolerance_value,
        recommended_checkpoint=recommended_path,
        recommended_label=recommended_label,
        checkpoints=checkpoints,
        summary_markdown=summary,
        generated_at=_utc_now(),
    )


def _write_text_atomic(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text.rstrip() + "\n")
        replace_with_retry(temporary_name, path)
    finally:
        try:
            Path(temporary_name).unlink()
        except OSError:
            pass
    return path


def write_training_analysis(analysis: TrainingAnalysis) -> Path:
    root = Path(analysis.adapter_dir).expanduser().resolve() / "analysis"
    path = root / "training_analysis.json"
    write_json_atomic(path, analysis.to_dict(), indent=2, ensure_ascii=False, allow_nan=False)
    _write_text_atomic(root / "training_analysis.md", analysis.summary_markdown)
    return path


def load_training_analysis(
    adapter_dir: str | os.PathLike[str],
) -> TrainingAnalysis | None:
    path = Path(adapter_dir).expanduser().resolve() / "analysis" / "training_analysis.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return TrainingAnalysis.from_dict(value) if isinstance(value, dict) else None
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return None


def analysis_epoch_frame(analysis: TrainingAnalysis | None) -> pd.DataFrame:
    if analysis is None or not analysis.epochs:
        return empty_series_frame(ANALYSIS_SERIES)
    rows: list[dict[str, Any]] = []
    for epoch in analysis.epochs:
        if epoch.train_loss is not None:
            rows.append({"step": epoch.epoch, "value": epoch.train_loss, "series": "train loss"})
        if epoch.val_loss is not None:
            series = (
                "validation (overfitting)"
                if epoch.phase == "overfitting"
                else "validation (improving)"
            )
            rows.append({"step": epoch.epoch, "value": epoch.val_loss, "series": series})
    if not rows:
        return empty_series_frame(ANALYSIS_SERIES)
    present = {str(item["series"]) for item in rows}
    rows.extend(
        {"step": 0, "value": float("nan"), "series": series}
        for series in ANALYSIS_SERIES
        if series not in present
    )
    frame = pd.DataFrame(rows)
    frame["step"] = pd.to_numeric(frame["step"], errors="coerce").astype("int64")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce").astype("float64")
    return frame.sort_values(["series", "step"], kind="stable").reset_index(drop=True)


__all__ = [
    "ANALYSIS_SERIES",
    "GENERALIZATION_LEGEND",
    "EpochSummary",
    "TrainingAnalysis",
    "analysis_epoch_frame",
    "analyze_training_run",
    "checkpoint_descriptor",
    "classify_epoch_phases",
    "discover_checkpoints",
    "inspect_lora",
    "load_training_analysis",
    "write_training_analysis",
]
