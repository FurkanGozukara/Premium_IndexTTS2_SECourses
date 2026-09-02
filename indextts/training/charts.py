"""Pandas frames consumed by the Gradio training dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, Any, Mapping

import pandas as pd



def empty_series_frame(series: Sequence[str] = ()) -> pd.DataFrame:
    """Placeholder chart frame with numeric dtypes and one invisible row per known series.

    The Gradio LinePlot frontend builds its Vega spec (axis types and the colour-scale domain) from the first value it
    receives and afterwards only swaps the data.  An all-object empty frame therefore turns both axes "nominal" and an
    empty colour domain leaves every later line without a stroke colour, so the placeholder carries the dtypes and the
    series names (with a null value, which Vega filters out) up front.
    """

    names = list(series)
    return pd.DataFrame(
        {
            "step": pd.Series([0] * len(names), dtype="int64"),
            "value": pd.Series([float("nan")] * len(names), dtype="float64"),
            "series": pd.Series(names, dtype="object"),
        }
    )


LOSS_SERIES = ("train raw", "train smoothed", "train EMA", "validation")
LR_SERIES = ("learning rate",)
GRAD_SERIES = ("grad norm",)
SPEED_SERIES = ("VRAM GB", "steps/s")

def _canonical_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    work = frame.copy()
    if "step" not in work:
        work["step"] = pd.Series(dtype="int64")
    numeric_steps = pd.to_numeric(work["step"], errors="coerce")
    step_keys = numeric_steps.astype("string").fillna(work["step"].astype("string"))
    if "event" in work:
        event_keys = work["event"].astype("string").fillna("training")
    else:
        event_keys = pd.Series("training", index=work.index, dtype="string")
    duplicate_keys = pd.DataFrame({"step": step_keys, "event": event_keys})
    return work.loc[~duplicate_keys.duplicated(keep="last")].reset_index(drop=True)


def _plot_rows(frame: pd.DataFrame, series: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return empty_series_frame(series)
    plotted = (
        frame.dropna(subset=["value"])
        .drop_duplicates(subset=["step", "series"], keep="last")
        .sort_values(["series", "step"], kind="stable")
        .reset_index(drop=True)
    )
    return plotted if not plotted.empty else empty_series_frame(series)


def load_metrics(state_dir: str | Path) -> pd.DataFrame:
    path = Path(state_dir) / "metrics.jsonl"
    rows: list[dict[str, Any]] = []
    if path.is_file():
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    frame = pd.DataFrame(rows)
    if "step" not in frame:
        frame["step"] = pd.Series(dtype="int64")
    return _canonical_metrics(frame)


def loss_frame(df: pd.DataFrame, smoothing: float = 0.9) -> pd.DataFrame:
    if df.empty:
        return empty_series_frame(LOSS_SERIES)
    df = _canonical_metrics(df)
    alpha = 1.0 - min(0.9999, max(0.0, float(smoothing)))
    pieces = []
    if "loss" in df:
        raw = pd.to_numeric(df["loss"], errors="coerce")
        smooth = raw.ewm(alpha=alpha, adjust=False).mean() if smoothing > 0 else raw
        pieces.extend(
            [
                pd.DataFrame({"step": df["step"], "value": raw, "series": "train raw"}),
                pd.DataFrame({"step": df["step"], "value": smooth, "series": "train smoothed"}),
            ]
        )
    for column, label in (("avg_loss", "train EMA"), ("val_loss", "validation")):
        if column in df:
            pieces.append(
                pd.DataFrame(
                    {
                        "step": df["step"],
                        "value": pd.to_numeric(df[column], errors="coerce"),
                        "series": label,
                    }
                )
            )
    return _plot_rows(pd.concat(pieces, ignore_index=True), LOSS_SERIES) if pieces else empty_series_frame(LOSS_SERIES)


def lr_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "lr" not in df:
        return empty_series_frame(LR_SERIES)
    df = _canonical_metrics(df)
    return _plot_rows(pd.DataFrame(
        {"step": df["step"], "value": pd.to_numeric(df["lr"], errors="coerce"), "series": "learning rate"}
    ), LR_SERIES)


def speed_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return empty_series_frame(SPEED_SERIES)
    df = _canonical_metrics(df)
    pieces = []
    for column, label in (("it_s", "steps/s"), ("vram_used_gb", "VRAM GB")):
        if column in df:
            pieces.append(
                pd.DataFrame(
                    {"step": df["step"], "value": pd.to_numeric(df[column], errors="coerce"), "series": label}
                )
            )
    return _plot_rows(pd.concat(pieces, ignore_index=True), SPEED_SERIES) if pieces else empty_series_frame(SPEED_SERIES)


def summary_text(status: Mapping[str, Any] | None) -> str:
    value = dict(status or {})
    phase = str(value.get("phase") or "idle").replace("_", " ").title()
    step = int(value.get("step") or 0)
    total = int(value.get("total_steps") or 0)
    epoch = int(value.get("epoch") or 0)
    epochs = int(value.get("total_epochs") or 0)
    loss = value.get("loss")
    val = value.get("val_loss")
    parts = [phase, f"step {step}/{total}", f"epoch {epoch}/{epochs}"]
    if loss is not None:
        parts.append(f"loss {float(loss):.4f}")
    if val is not None:
        parts.append(f"val {float(val):.4f}")
    if value.get("message"):
        parts.append(str(value["message"]))
    return " | ".join(parts)


__all__ = [
    "GRAD_SERIES",
    "LOSS_SERIES",
    "LR_SERIES",
    "SPEED_SERIES",
    "empty_series_frame",
    "load_metrics",
    "loss_frame",
    "lr_frame",
    "speed_frame",
    "summary_text",
]
