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
    "validation",
    "validation (regression)",
)
BASE_CHECKPOINT_LABEL = "Base model (no LoRA / DoRA)"
BASE_CHECKPOINT_CHOICE_LABEL = (
    BASE_CHECKPOINT_LABEL
    + " - plain voice clone from the reference audio only"
)
BASE_PHASE_LABEL = "Reference-only baseline (no LoRA / DoRA)"
BASE_GRID_HEADER_DETAIL = "Plain voice clone: only the reference audio shapes the voice"
_LEGACY_BASE_LABELS = frozenset({"base model", "base model (no adapter)"})
GENERALIZATION_LEGEND = (
    "Validation loss measures how well the LoRA / DoRA predicts sentences it never saw during "
    "training (lower is better). Training loss measures the clips it trains on. When training "
    "loss keeps falling but validation loss rises, that can indicate overfitting. Loss alone does not "
    "establish memorization or generated-speech quality; audio-token accuracy is not word accuracy."
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
        "adapter_type": str(header.get("adapter_type", "")).lower(),
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
    """Describe a LoRA / DoRA file using its location, name, and saved metadata."""

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
    saved_type = str(info.get("adapter_type") or "").strip().lower()
    checkpoint_type = {
        "lora": "LoRA Checkpoint",
        "dora": "DoRA Checkpoint",
    }.get(saved_type, "LoRA / DoRA Checkpoint")
    if kind == "best":
        label = (
            f"best (epoch {epoch} {checkpoint_type})"
            if epoch
            else f"best ({checkpoint_type})"
        )
        file_label = f"best_ep{epoch}" if epoch else "best"
    elif kind == "epoch":
        label = f"epoch {epoch} ({checkpoint_type})"
        file_label = f"epoch_{epoch:03d}"
    elif kind == "step":
        label = f"step {steps} ({checkpoint_type})"
        file_label = f"step_{steps:06d}"
    elif kind == "interrupted":
        label = (
            f"interrupted (epoch {epoch} {checkpoint_type})"
            if epoch
            else f"interrupted ({checkpoint_type})"
        )
        file_label = f"interrupted_ep{epoch:02d}" if epoch else "interrupted"
    else:
        label = (
            f"final (epoch {epoch} {checkpoint_type})"
            if epoch
            else f"final ({checkpoint_type})"
        )
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


def checkpoint_display_label(
    label: str | None,
    *,
    path: str | os.PathLike[str] | None = None,
    kind: str | None = None,
    cache: dict[str, str] | None = None,
) -> str:
    """Upgrade legacy checkpoint labels at display time without rewriting reports."""

    value = str(label or "").strip()
    if not path or str(kind or "").strip().lower() == "base":
        if not value or value.lower() in _LEGACY_BASE_LABELS:
            return BASE_CHECKPOINT_LABEL
        return value

    strength_match = re.search(
        r"(?P<suffix>\s+@[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$", value
    )
    suffix = strength_match.group("suffix") if strength_match else ""
    core = value[: strength_match.start()].rstrip() if strength_match else value
    if core.endswith("Checkpoint)"):
        return value

    source = Path(path).expanduser()
    if not source.is_file():
        return value
    cache_key = os.path.normcase(str(source.resolve()))
    derived = cache.get(cache_key) if cache is not None else None
    if derived is None:
        try:
            derived = str(checkpoint_descriptor(source)["label"])
        except Exception:
            return value
        if cache is not None:
            cache[cache_key] = derived
    return derived + suffix


def display_legacy_base_labels(text: str | None) -> str:
    """Upgrade legacy base-row wording for display without changing saved data."""

    value = re.sub(
        r"\bbase model\s*\(no adapter\)",
        BASE_CHECKPOINT_LABEL,
        str(text or ""),
        flags=re.IGNORECASE,
    )
    return re.sub(
        r"\bbase model\b(?!\s*\(no LoRA / DoRA\))",
        BASE_CHECKPOINT_LABEL,
        value,
        flags=re.IGNORECASE,
    )


def display_legacy_report_text(text: str | None) -> str:
    """Modernize legacy report prose only while it is being displayed."""

    value = display_legacy_base_labels(text)
    return re.sub(r"\badapters?\b", "LoRA / DoRA", value, flags=re.IGNORECASE)


def phase_display_label(phase: str | None) -> str:
    return {
        "best": "Lowest validation loss",
        "improving": "Improving",
        "plateau": "Plateau",
        "overfitting": "Validation regression (possible overfitting)",
        "base": BASE_PHASE_LABEL,
        "variant": "Strength variant",
        "unknown": "Not measured",
    }.get(str(phase or "").strip().lower(), "Not measured")


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
            f"**Lowest logged validation loss: epoch {best_epoch}** (loss {best_val_loss:.2f}{accuracy_text} on held-out clips)."
        ]
        if status == "still_improving":
            lines.append(
                "The last epoch contains the lowest logged validation loss; "
                "the final file is the best one available."
            )
        elif status == "best_found" and overfit_start is not None:
            lines.append(
                f"**Validation regression from epoch {overfit_start}.** The epoch minimum exceeds the tolerance "
                "above the best value. This is an overfitting warning, not a statistical test or proof of memorization."
            )
            final = by_epoch.get(final_epoch or -1)
            if final_val_loss is not None and final_epoch is not None:
                detail = (
                    f"The final epoch {final_epoch} has logged validation loss "
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
                        f"; training audio-token accuracy changed from {best.train_accuracy * 100:.1f}% "
                        f"to {final.train_accuracy * 100:.1f}%"
                    )
                lines.append(detail + ".")
        else:
            lines.append(
                f"Later epochs stayed within {tolerance * 100:.1f}% of the best validation loss, so the run reached a plateau "
                "within this report's tolerance."
            )
        lines.append(f"**Recommended checkpoint:** `{checkpoint_text}` (epoch {best_epoch}).")
        if status == "best_found" and overfit_start is not None:
            lines.append(
                "Keep the lowest-loss checkpoint and compare generated speech. A different dataset may need a different training length."
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
    validation_by_step: dict[int, dict[str, Any]] = {}
    minimum_by_epoch: dict[int, float] = {}
    if not validation_rows.empty and "epoch" in validation_rows:
        for _, row in validation_rows.iterrows():
            epoch = _integer(row.get("epoch"))
            val_loss = _finite_float(row.get("val_loss"))
            if epoch > 0 and val_loss is not None:
                validation_by_epoch[epoch] = {
                    "loss": val_loss,
                    "accuracy": _finite_float(row.get("val_mel_accuracy")),
                    "step": _integer(row.get("step")),
                    "epoch": epoch,
                }
                validation_by_step[_integer(row.get("step"))] = validation_by_epoch[epoch]
                minimum_by_epoch[epoch] = min(minimum_by_epoch.get(epoch, val_loss), val_loss)

    phases, best_epoch, overfit_start = classify_epoch_phases(
        minimum_by_epoch,
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
    best_validation = min(validation_by_step.values(), key=lambda item: (item["loss"], item["step"]), default={})
    best_epoch = _integer(best_validation.get("epoch")) or None
    best_val_loss = _finite_float(best_validation.get("loss"))
    best_step = _integer(best_validation.get("step")) or None
    if metrics.empty:
        status = "empty"
    elif not validation_by_epoch:
        status = "no_validation"
    elif best_step == _integer(final_validation.get("step")):
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
        validation = validation_by_step.get(int(item.get("steps") or 0), validation_by_epoch.get(epoch, {}))
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
    if best_step is not None:
        summary += f"\n\nCheckpoint selection uses every validation check; the lowest loss occurred at optimizer step {best_step:,}."
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
                "validation (regression)"
                if epoch.phase == "overfitting"
                else "validation"
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
    "BASE_CHECKPOINT_CHOICE_LABEL",
    "BASE_CHECKPOINT_LABEL",
    "BASE_GRID_HEADER_DETAIL",
    "BASE_PHASE_LABEL",
    "GENERALIZATION_LEGEND",
    "EpochSummary",
    "TrainingAnalysis",
    "analysis_epoch_frame",
    "analyze_training_run",
    "checkpoint_descriptor",
    "checkpoint_display_label",
    "display_legacy_base_labels",
    "display_legacy_report_text",
    "classify_epoch_phases",
    "discover_checkpoints",
    "inspect_lora",
    "load_training_analysis",
    "phase_display_label",
    "write_training_analysis",
]
