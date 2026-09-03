"""LoRA / DoRA training dashboard and file manager."""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import gradio as gr
import pandas as pd

from indextts.lora.io import inspect_lora, scan_lora_files
from indextts.runtime.gpu import gpu_total_gb
from indextts.runtime.vram_presets import auto_tier, resolve_preset
from indextts.training.charts import GRAD_SERIES, LOSS_SERIES, LR_SERIES, SPEED_SERIES, downsample_series, empty_series_frame, load_metrics, lr_frame, speed_frame
from indextts.training.analysis import (
    ANALYSIS_SERIES,
    analysis_epoch_frame,
    checkpoint_descriptor,
    display_legacy_report_text,
    load_training_analysis,
    phase_display_label,
)
from indextts.training.checkpoint_eval import load_checkpoint_eval
from indextts.training.dataset_manifest import load_manifest
from indextts.training.plan import training_plan, training_plan_advisory, training_plan_line
from indextts.training.train_config import TrainConfig

from .common import (
    PROCESS_MANAGER,
    ROOT,
    btn,
    dedupe_updates,
    open_folder,
    progress_panel_html,
    read_json,
    tail_text,
    write_json_atomic,
)
from .dataset_tab import scan_datasets
from .generation_tab import GenerationTab
from .models_tab import ModelsTab
from .presets_store import PresetRegistry


TRAIN_DEFAULTS = TrainConfig(dataset_dir="datasets/secourses_demo", name="voice_adapter").to_dict()
TRAIN_BETAS_TEXT = ", ".join(str(value) for value in TRAIN_DEFAULTS["betas"])
_LAST_TRAINING_FOLDER = ROOT / "loras"
TRAINING_TERMINAL_PHASES = frozenset(
    {"complete", "stopped", "failed", "error", "cancelled", "canceled"}
)


_NON_TRAINING_STATE_FOLDERS = frozenset({"analysis", "eval_jobs", "eval_job", ".sample_jobs", "samples"})


def _training_states(root: str | Path = ROOT / "loras") -> list[Path]:
    values: list[tuple[float, Path]] = []
    base = Path(root).expanduser()
    for status_path in base.rglob("status.json") if base.is_dir() else []:
        try:
            relative_parts = status_path.relative_to(base).parts
        except ValueError:
            relative_parts = status_path.parts
        # Checkpoint evaluation jobs and training samples keep their own status.json
        # below an adapter folder; only the adapter folder itself is a training run.
        if any(part.lower() in _NON_TRAINING_STATE_FOLDERS for part in relative_parts[:-1]):
            continue
        try:
            values.append((status_path.stat().st_mtime, status_path.parent.resolve()))
        except OSError:
            continue
    return [path for _, path in sorted(values, key=lambda item: item[0], reverse=True)]


def latest_training_state(root: str | Path = ROOT / "loras") -> str:
    states = _training_states(root)
    return str(states[0]) if states else ""


def _state_running(path: str | Path | None) -> bool:
    status = read_json(Path(path) / "status.json", {}) if path else {}
    phase = str((status or {}).get("phase", "")).strip().lower()
    return bool(phase and phase not in TRAINING_TERMINAL_PHASES)


def adopt_training_state(
    displayed: str | Path | None,
    *,
    root: str | Path = ROOT / "loras",
    page_load: bool = False,
) -> tuple[str, bool]:
    """Resolve the per-session dashboard state without clinging to a stale run."""

    current = str(Path(displayed).expanduser().resolve()) if displayed else ""
    if current and _state_running(current):
        return current, True
    newest = latest_training_state(root)
    if page_load and newest:
        return newest, _state_running(newest)
    if newest and _state_running(newest):
        return newest, True
    return current, False


def _adapter_entries() -> list[Any]:
    return scan_lora_files([str(ROOT / "loras")])


def adapter_rows() -> tuple[list[list[Any]], list[str]]:
    rows = []
    paths = []
    for entry in _adapter_entries():
        try:
            info = inspect_lora(entry.path)
        except Exception:
            continue
        rows.append([
            entry.name,
            str(info.get("adapter_type", "")).upper(),
            info.get("rank", 0),
            info.get("alpha", 0),
            info.get("steps", 0),
            info.get("dataset", ""),
            info.get("date", ""),
            info.get("size_mb", 0),
            str(Path(entry.path).resolve()),
        ])
        paths.append(str(Path(entry.path).resolve()))
    return rows, paths


def _resume_choices() -> list[tuple[str, str]]:
    return [("Start fresh", "")] + [(entry.relative_label, str(Path(entry.path).resolve())) for entry in _adapter_entries()]


def _dataset_choices() -> list[tuple[str, str]]:
    return scan_datasets(ROOT / "datasets")


def _dataset_summary(path: str | None) -> str:
    if not path:
        return "Select a prepared dataset."
    root = Path(path)
    info = read_json(root / "dataset_info.json", {}) or {}
    cache_index = root / "cache" / "index.jsonl"
    return (
        f"**{root.name}** | {info.get('segment_count', 0)} segments | "
        f"{float(info.get('total_duration_minutes', 0.0) or 0.0):.2f} minutes | "
        f"features **{'cached' if cache_index.is_file() else 'not cached'}**"
    )


def _training_plan_markdown(
    dataset_path: str | None,
    batch_size: int,
    grad_accumulation: int,
    epochs: int,
    max_steps: int,
    val_fraction: float,
    seed: int = 42,
) -> str:
    try:
        if not dataset_path:
            return "Select a dataset to see the training plan."
        root = Path(dataset_path).expanduser()
        if not root.is_absolute():
            root = ROOT / root
        manifest = root / "manifest.jsonl"
        if not manifest.is_file():
            return "Select a dataset to see the training plan."
        rows = load_manifest(manifest)
        record_ids = [str(row["id"]) for row in rows if row.get("id")]
        if not record_ids:
            return "Training plan unavailable: the manifest is empty."
        plan = training_plan(
            len(record_ids),
            batch_size,
            grad_accumulation,
            epochs,
            max_steps,
            val_fraction,
            record_ids=record_ids,
            seed=seed,
        )
        advisory = training_plan_advisory(plan, batch_size, grad_accumulation)
        return f"### Training plan\n\n{training_plan_line(plan)}\n\n{advisory}"
    except Exception as exc:
        message = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        return f"Training plan unavailable: {message[:160]}"


def _loss_plot_frame(frame: pd.DataFrame, smoothing: float) -> pd.DataFrame:
    if frame.empty:
        return empty_series_frame(LOSS_SERIES)
    pieces = []
    if "loss" in frame:
        raw = pd.to_numeric(frame["loss"], errors="coerce")
        pieces.append(pd.DataFrame({"step": frame["step"], "value": raw, "series": "train raw"}).dropna(subset=["value"]))
        alpha = 1.0 - min(0.9999, max(0.0, float(smoothing)))
        smooth = raw.ewm(alpha=alpha, adjust=False).mean() if smoothing > 0 else raw
        pieces.append(pd.DataFrame({"step": frame["step"], "value": smooth, "series": "train smoothed"}).dropna(subset=["value"]))
    for column, label in (("avg_loss", "train EMA"), ("val_loss", "validation")):
        if column in frame:
            pieces.append(pd.DataFrame({"step": frame["step"], "value": pd.to_numeric(frame[column], errors="coerce"), "series": label}).dropna(subset=["value"]))
    if not pieces:
        return empty_series_frame(LOSS_SERIES)
    plotted = downsample_series(
        pd.concat(pieces, ignore_index=True)
        .drop_duplicates(subset=["step", "series"], keep="last")
        .sort_values(["series", "step"], kind="stable")
        .reset_index(drop=True)
    )
    return plotted if not plotted.empty else empty_series_frame(LOSS_SERIES)


def _grad_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "grad_norm" not in frame:
        return empty_series_frame(GRAD_SERIES)
    plotted = downsample_series(
        pd.DataFrame({"step": frame["step"], "value": pd.to_numeric(frame["grad_norm"], errors="coerce"), "series": "grad norm"})
        .dropna(subset=["value"])
        .drop_duplicates(subset=["step", "series"], keep="last")
        .sort_values("step", kind="stable")
        .reset_index(drop=True)
    )
    return plotted if not plotted.empty else empty_series_frame(GRAD_SERIES)


def _checkpoint_rows(state_dir: str | Path | None) -> list[list[Any]]:
    if not state_dir:
        return []
    root = Path(state_dir)
    analysis = load_training_analysis(root)
    measured = load_checkpoint_eval(root)
    by_path: dict[str, tuple[float | None, str]] = {}
    if analysis is not None:
        for item in analysis.checkpoints:
            path_value = str(Path(str(item.get("path") or "")).resolve())
            by_path[path_value] = (item.get("val_loss"), str(item.get("phase") or "unknown"))
    if measured is not None:
        for item in measured.rows:
            if item.path and abs(item.strength - 1.0) < 1e-9:
                by_path[str(Path(item.path).resolve())] = (item.val_loss, item.phase)
    rows = []
    for path in sorted(root.rglob("*.safetensors"), key=lambda item: item.stat().st_mtime, reverse=True):
        if path.name.startswith("."):
            continue
        try:
            info = inspect_lora(path)
            val_loss, phase = by_path.get(str(path.resolve()), (None, "unknown"))
            rows.append([
                checkpoint_descriptor(path)["label"],
                info.get("adapter_type", ""),
                info.get("rank", 0),
                info.get("steps", 0),
                info.get("epochs", 0),
                round(float(val_loss), 4) if val_loss is not None else None,
                _verdict_text(phase),
                round(path.stat().st_size / 1024**2, 2),
                str(path),
            ])
        except Exception:
            rows.append([path.name, "", 0, 0, 0, None, "Not measured", round(path.stat().st_size / 1024**2, 2), str(path)])
    return rows


def _verdict_text(phase: str) -> str:
    return phase_display_label(phase)


def _training_generalization(state_dir: str | Path | None) -> tuple[str, pd.DataFrame]:
    if not state_dir:
        return "", analysis_epoch_frame(None)
    root = Path(state_dir)
    analysis = load_training_analysis(root)
    measured = load_checkpoint_eval(root)
    if measured is not None:
        summary = display_legacy_report_text(measured.summary_markdown)
    elif analysis is not None:
        summary = display_legacy_report_text(analysis.summary_markdown)
    else:
        summary = "Run training with validation enabled to identify the checkpoint that generalizes best."
    return "### Generalization summary\n\n" + summary, analysis_epoch_frame(analysis)


def _training_status_text(status: Mapping[str, Any], metrics: pd.DataFrame) -> str:
    value = dict(status)
    step = int(value.get("step", 0) or 0)
    total = int(value.get("total_steps", 0) or 0)
    epoch = int(value.get("epoch", 0) or 0)
    epochs = int(value.get("total_epochs", 0) or 0)
    parts = [str(value.get("phase", "idle")).replace("_", " ").title(), f"step {step}/{total}", f"epoch {epoch}/{epochs}"]
    for key, label, fmt in (
        ("loss", "loss", ".4f"), ("avg_loss", "avg", ".4f"), ("val_loss", "val", ".4f"),
        ("lr", "lr", ".2e"), ("it_s", "speed", ".2f"), ("vram_used_gb", "VRAM", ".2f"),
    ):
        item = value.get(key)
        if item is not None:
            suffix = " it/s" if key == "it_s" else (" GB" if key == "vram_used_gb" else "")
            parts.append(f"{label} {format(float(item), fmt)}{suffix}")
    if not metrics.empty and "mel_accuracy" in metrics:
        accuracy = pd.to_numeric(metrics["mel_accuracy"], errors="coerce").dropna()
        if not accuracy.empty:
            parts.append(f"acc {accuracy.iloc[-1]:.3f}")
    eta = value.get("eta_s")
    elapsed = value.get("elapsed_s")
    if elapsed is not None:
        parts.append(f"elapsed {float(elapsed):.1f}s")
    if eta is not None:
        parts.append(f"ETA {float(eta):.1f}s")
    if value.get("message"):
        parts.append(str(value["message"]))
    runtime_warning = str(value.get("runtime_warning") or "")
    if runtime_warning and runtime_warning != str(value.get("message") or ""):
        parts.append(runtime_warning)
    return " | ".join(parts)


def _training_vram_total(root: Path) -> float:
    config = read_json(root / "train_config.json", {}) or {}
    device = str(config.get("device") or "")
    if not device.startswith("cuda"):
        return 0.0
    try:
        index = int(device.split(":", 1)[1]) if ":" in device else 0
        return gpu_total_gb(index)
    except (RuntimeError, TypeError, ValueError):
        return 0.0


def training_status_updates(state_value: str, smoothing_value: float) -> tuple[Any, ...]:
    """Return the complete training dashboard update for polling and server push."""

    if not state_value:
        return (
            progress_panel_html({}, title="Ready"),
            "",
            empty_series_frame(LOSS_SERIES),
            empty_series_frame(LR_SERIES),
            empty_series_frame(GRAD_SERIES),
            empty_series_frame(SPEED_SERIES),
            "",
            None,
            "",
            [],
            "",
            analysis_epoch_frame(None),
            gr.Timer(5.0, active=True),
        )
    root = Path(state_value)
    status = read_json(root / "status.json", {}) or {}
    metrics = load_metrics(root)
    step = int(status.get("step", 0) or 0)
    total = int(status.get("total_steps", 0) or 0)
    fraction = step / total if total else 0.0
    phase = str(status.get("phase", "initializing")).strip().lower()
    terminal = phase in TRAINING_TERMINAL_PHASES
    payload = {
        "fraction": 1.0 if phase == "complete" else fraction,
        "completed": step,
        "total": total,
        "elapsed_s": status.get("elapsed_s", 0),
        "eta_s": status.get("eta_s"),
        "speed": status.get("it_s"),
        "speed_unit": "it/s",
        "vram_used_gb": status.get("vram_used_gb", 0),
        "vram_total_gb": _training_vram_total(root),
        "desc": f"epoch {status.get('epoch', 0)}/{status.get('total_epochs', 0)}",
    }
    if phase == "evaluating":
        evaluation_progress = read_json(root / "analysis" / "eval_job" / "progress.json", {}) or {}
        if evaluation_progress:
            payload = dict(evaluation_progress)
        payload["desc"] = str(status.get("message") or payload.get("desc") or "evaluating checkpoints")
    titles = {
        "evaluating": "Evaluating checkpoints",
        "complete": "Training complete",
        "stopped": "Training stopped",
        "cancelled": "Training canceled",
        "canceled": "Training canceled",
        "failed": "Training failed",
        "error": "Training failed",
    }
    sample = status.get("last_sample") or None
    if sample and not Path(sample).is_file():
        sample = None
    sample_text_value = (
        f"Latest sample: {Path(sample).name} | epoch {status.get('epoch', 0)}"
        if sample
        else "No sample yet."
    )
    generalization_summary, generalization_chart = _training_generalization(root)
    return (
        progress_panel_html(payload, title=titles.get(phase, "Training in progress")),
        _training_status_text(status, metrics),
        _loss_plot_frame(metrics, smoothing_value),
        lr_frame(metrics),
        _grad_frame(metrics),
        speed_frame(metrics),
        tail_text(root / "log.txt", 60) or tail_text(root / "worker_console.log", 60),
        sample,
        sample_text_value,
        _checkpoint_rows(root),
        generalization_summary,
        generalization_chart,
        gr.Timer(5.0 if terminal else 1.0, active=True),
    )


def training_poll_updates(
    state_value: str,
    smoothing_value: float,
    *,
    state_root: str | Path = ROOT / "loras",
    page_load: bool = False,
) -> tuple[Any, ...]:
    """Adopt a live run when idle, then return state plus the full dashboard."""

    adopted_state, running = adopt_training_state(
        state_value,
        root=state_root,
        page_load=page_load,
    )
    updates = list(training_status_updates(adopted_state, smoothing_value))
    if running and adopted_state:
        updates[1] = f"Attached to running run {Path(adopted_state).name} | {updates[1]}"
        updates[-1] = gr.Timer(1.0, active=True)
    else:
        updates[-1] = gr.Timer(5.0, active=True)
    return adopted_state, *updates


@dataclass
class TrainingTab:
    controls: dict[str, Any]
    apply_tier_button: Any
    base_variant: Any
    mixed_precision: Any
    blocks_to_swap: Any
    swap_ring_size: Any
    pin_swap_memory: Any
    state_dir: Any
    use_in_generation: Any
    compare_grid: Any
    dataset: Any
    dataset_info: Any
    training_plan: Any
    start_event: Any = None


def _reg(
    registry: PresetRegistry,
    controls: dict[str, Any],
    field_name: str,
    component: Any,
    *,
    kind: str = "auto",
    choices: Sequence[Any] | None = None,
    minimum: float | int | None = None,
    maximum: float | int | None = None,
) -> None:
    key = f"training.{field_name}"
    registry.register(key, component, TRAIN_DEFAULTS[field_name], kind=kind, choices=choices, minimum=minimum, maximum=maximum)
    controls[key] = component


def build_training_tab(
    args: Any,
    registry: PresetRegistry,
    *,
    load_hook: Any | None = None,
) -> TrainingTab:
    controls: dict[str, Any] = {}
    datasets = _dataset_choices()
    initial_dataset = datasets[0][1] if datasets else str(ROOT / TRAIN_DEFAULTS["dataset_dir"])
    # State is deliberately session-local and empty at build time. The header load
    # control discovers the previous run for each browser session on demand.
    current_state = ""
    model_dir_default = str(
        Path(getattr(args, "model_dir", ROOT / TRAIN_DEFAULTS["model_dir"])).resolve()
    )
    device_default = str(
        getattr(args, "device", TRAIN_DEFAULTS["device"])
        or TRAIN_DEFAULTS["device"]
    )
    if device_default == "auto":
        device_default = TRAIN_DEFAULTS["device"]

    with gr.Tab("LoRA / DoRA Training", id="lora-training") as tab_block:
        with gr.Row():
            dataset = gr.Dropdown(
                choices=datasets,
                value=initial_dataset if Path(initial_dataset).is_dir() else None,
                allow_custom_value=True,
                label="Dataset",
                info="Prepared manifest dataset used for cached-feature training.",
                scale=5,
            )
            refresh_dataset = gr.Button("↻  Refresh", elem_classes=btn("violet"), scale=1)
        dataset_info = gr.Markdown(_dataset_summary(initial_dataset if Path(initial_dataset).is_dir() else None))
        _reg(registry, controls, "dataset_dir", dataset, kind="str")

        with gr.Accordion("LoRA / DoRA", open=True):
            with gr.Row():
                name = gr.Textbox(value=TRAIN_DEFAULTS["name"], label="LoRA / DoRA name", info="Safe output folder and final safetensors basename.")
                adapter_type = gr.Dropdown(choices=["lora", "dora"], value=TRAIN_DEFAULTS["adapter_type"], label="LoRA / DoRA type", info="DoRA is the quality default; LoRA uses slightly less compute.")
                rank = gr.Slider(1, 256, value=TRAIN_DEFAULTS["rank"], step=1, label="Rank", info="128 with alpha 129 learned the voice fastest in measured runs; rank 32 reached the same floor more slowly.")
                alpha = gr.Number(value=TRAIN_DEFAULTS["alpha"], minimum=1, maximum=1024, label="Alpha", info="129 is the measured companion scale for the recommended rank 128.")
                dropout = gr.Slider(0, 0.5, value=TRAIN_DEFAULTS["dropout"], step=0.01, label="Dropout", info="0.05 remains the quality default; stronger measured regularization gave no benefit.")
            with gr.Row():
                target_attention = gr.Checkbox(value=TRAIN_DEFAULTS["target_attention"], label="Target attention", info="Adapts GPT attention projections; recommended.")
                target_mlp = gr.Checkbox(value=TRAIN_DEFAULTS["target_mlp"], label="Target MLP", info="Adapts GPT feed-forward projections; recommended for voice fidelity.")
                train_spk = gr.Checkbox(value=TRAIN_DEFAULTS["train_spk_proj"], label="Train speaker projection", info="Fully trains the small speaker projection module.")
                train_emo = gr.Checkbox(value=TRAIN_DEFAULTS["train_emo_layers"], label="Train emotion layers", info="Advanced: trains small emotion modules in addition to LoRA / DoRA layers.")
                train_mel = gr.Checkbox(value=TRAIN_DEFAULTS["train_mel_embed_head"], label="Train mel embedding head", info="Advanced: trains the mel token embedding/head modules.")
            for field_name, component, kind, choices, minimum, maximum in (
                ("name", name, "str", None, None, None),
                ("adapter_type", adapter_type, "choice", ["lora", "dora"], None, None),
                ("rank", rank, "int", None, 1, 256), ("alpha", alpha, "float", None, 1, 1024),
                ("dropout", dropout, "float", None, 0, 0.5),
                ("target_attention", target_attention, "bool", None, None, None),
                ("target_mlp", target_mlp, "bool", None, None, None),
                ("train_spk_proj", train_spk, "bool", None, None, None),
                ("train_emo_layers", train_emo, "bool", None, None, None),
                ("train_mel_embed_head", train_mel, "bool", None, None, None),
            ):
                _reg(registry, controls, field_name, component, kind=kind, choices=choices, minimum=minimum, maximum=maximum)

        with gr.Accordion("Optimization", open=False):
            with gr.Row():
                learning_rate = gr.Number(value=TRAIN_DEFAULTS["learning_rate"], minimum=1e-8, maximum=1, label="Learning rate", info="4e-5 is the robust batch-1 default: it reached 5.061 held-out loss, while 8e-5 overfit within two epochs.")
                optimizer = gr.Dropdown(choices=["adamw", "adamw_fused", "prodigy"], value=TRAIN_DEFAULTS["optimizer"], label="Optimizer", info="AdamW is portable; fused AdamW is faster on supported CUDA builds.")
                scheduler = gr.Dropdown(choices=["cosine", "linear", "constant", "constant_with_warmup"], value=TRAIN_DEFAULTS["lr_scheduler"], label="Scheduler", info="Cosine decay is recommended for multi-epoch voice adaptation.")
                warmup = gr.Number(value=TRAIN_DEFAULTS["warmup_steps"], minimum=0, precision=0, label="Warmup steps", info="200 steps is the measured batch-1 default, easing training into the 4e-5 learning rate.")
                weight_decay = gr.Number(value=TRAIN_DEFAULTS["weight_decay"], minimum=0, maximum=1, label="Weight decay", info="0.01 is a mild regularizer; 0.05 showed no measured benefit.")
            with gr.Row():
                betas = gr.Textbox(value=TRAIN_BETAS_TEXT, label="Adam betas", info="Two comma-separated momentum coefficients; 0.9, 0.99 is recommended.")
                eps = gr.Number(value=TRAIN_DEFAULTS["eps"], minimum=1e-12, maximum=0.1, label="Adam epsilon", info="Numerical stability term for Adam-family optimizers.")
                epochs = gr.Number(value=TRAIN_DEFAULTS["epochs"], minimum=1, maximum=10000, precision=0, label="Epochs", info="10 epochs is the measured batch-1 default; 20 epochs overfit after the held-out optimum around epoch 6.")
                max_steps = gr.Number(value=TRAIN_DEFAULTS["max_steps"], minimum=0, precision=0, label="Maximum steps", info="0 derives steps from epochs; set 5 for a quick smoke run.")
                batch_size = gr.Number(value=TRAIN_DEFAULTS["batch_size"], minimum=1, maximum=128, precision=0, label="Batch size", info="1 is the measured quality default: every training clip becomes an optimizer update each epoch.")
                accumulation = gr.Number(value=TRAIN_DEFAULTS["grad_accumulation"], minimum=1, maximum=128, precision=0, label="Gradient accumulation", info="1 is the measured default; accumulation 2 or 4 removed updates and performed worse at the same learning rate.")
            training_plan_readout = gr.Markdown(
                _training_plan_markdown(
                    initial_dataset if Path(initial_dataset).is_dir() else None,
                    TRAIN_DEFAULTS["batch_size"],
                    TRAIN_DEFAULTS["grad_accumulation"],
                    TRAIN_DEFAULTS["epochs"],
                    TRAIN_DEFAULTS["max_steps"],
                    TRAIN_DEFAULTS["val_fraction"],
                    TRAIN_DEFAULTS["seed"],
                )
            )
            with gr.Row():
                grad_clip = gr.Number(value=TRAIN_DEFAULTS["max_grad_norm"], minimum=0, label="Gradient clip", info="1.0 limits unstable gradient spikes; 0 disables clipping.")
                smoothing = gr.Slider(0, 0.5, value=TRAIN_DEFAULTS["label_smoothing"], step=0.01, label="Label smoothing", info="0 is recommended; increase only for overconfident large datasets.")
                mel_weight = gr.Number(value=TRAIN_DEFAULTS["mel_loss_weight"], minimum=0, label="Mel loss weight", info="Primary autoregressive acoustic-token loss weight.")
                text_weight = gr.Number(value=TRAIN_DEFAULTS["text_loss_weight"], minimum=0, label="Text loss weight", info="Auxiliary text modeling loss weight.")
                speaker_mode = gr.Dropdown(choices=["self", "other", "mixed"], value=TRAIN_DEFAULTS["speaker_ref_mode"], label="Speaker reference mode", info="self uses the target clip, other uses a deterministic different same-speaker clip, and mixed alternates between them; other is the measured quality default.")
                emo_ref_mode = gr.Dropdown(
                    choices=["self", "other", "mixed", "follow_speaker"],
                    value=TRAIN_DEFAULTS["emo_ref_mode"],
                    label="Emotion reference mode",
                    info="self uses the target emotion, other uses another same-speaker clip, mixed alternates, and follow_speaker reuses the speaker-reference clip; follow_speaker is the measured inference-like default.",
                )
            with gr.Row():
                max_codes = gr.Number(value=TRAIN_DEFAULTS["max_codes"], minimum=1, precision=0, label="Maximum codes", info="Cached samples longer than this semantic-code limit are rejected.")
                max_text = gr.Number(value=TRAIN_DEFAULTS["max_text_tokens"], minimum=1, precision=0, label="Maximum text tokens", info="Cached text length safety limit.")
                val_fraction = gr.Slider(0, 0.5, value=TRAIN_DEFAULTS["val_fraction"], step=0.01, label="Validation fraction", info="5% provides useful validation without sacrificing much training data.")
                val_steps = gr.Number(value=TRAIN_DEFAULTS["val_every_steps"], minimum=0, precision=0, label="Validate every steps", info="0 disables step validation; epoch validation still runs when a split exists.")
                val_batches = gr.Number(value=TRAIN_DEFAULTS["val_max_batches"], minimum=1, precision=0, label="Maximum validation batches", info="Caps validation time on large datasets.")
                val_reference_mode = gr.Dropdown(
                    choices=["self", "other"],
                    value=TRAIN_DEFAULTS["val_reference_mode"],
                    label="Validation reference",
                    info="self validates each target with itself, while other uses a different same-speaker clip for both vectors; other is inference-like and measured more accurately.",
                )
            with gr.Row():
                early_patience = gr.Number(
                    value=TRAIN_DEFAULTS["early_stop_patience"],
                    minimum=0,
                    precision=0,
                    label="Early-stop patience",
                    info="0 disables early stopping; otherwise stop after this many validations without a meaningful improvement.",
                )
                early_delta = gr.Number(
                    value=TRAIN_DEFAULTS["early_stop_min_delta"],
                    minimum=0,
                    label="Early-stop minimum improvement",
                    info="Validation loss must fall by at least this amount to reset patience.",
                )
            optimization_fields = (
                ("learning_rate", learning_rate, "float", None, 1e-8, 1), ("optimizer", optimizer, "choice", ["adamw", "adamw_fused", "prodigy"], None, None),
                ("lr_scheduler", scheduler, "choice", ["cosine", "linear", "constant", "constant_with_warmup"], None, None),
                ("warmup_steps", warmup, "int", None, 0, 1000000), ("weight_decay", weight_decay, "float", None, 0, 1),
                ("betas", betas, "str", None, None, None), ("eps", eps, "float", None, 1e-12, 0.1),
                ("epochs", epochs, "int", None, 1, 10000), ("max_steps", max_steps, "int", None, 0, 100000000),
                ("batch_size", batch_size, "int", None, 1, 128), ("grad_accumulation", accumulation, "int", None, 1, 128),
                ("max_grad_norm", grad_clip, "float", None, 0, 100), ("label_smoothing", smoothing, "float", None, 0, 0.5),
                ("mel_loss_weight", mel_weight, "float", None, 0, 100), ("text_loss_weight", text_weight, "float", None, 0, 100),
                ("speaker_ref_mode", speaker_mode, "choice", ["self", "other", "mixed"], None, None),
                ("emo_ref_mode", emo_ref_mode, "choice", ["self", "other", "mixed", "follow_speaker"], None, None),
                ("max_codes", max_codes, "int", None, 1, 100000), ("max_text_tokens", max_text, "int", None, 1, 100000),
                ("val_fraction", val_fraction, "float", None, 0, 0.5), ("val_every_steps", val_steps, "int", None, 0, 1000000),
                ("val_max_batches", val_batches, "int", None, 1, 1000000),
                ("val_reference_mode", val_reference_mode, "choice", ["self", "other"], None, None),
                ("early_stop_patience", early_patience, "int", None, 0, 1000000),
                ("early_stop_min_delta", early_delta, "float", None, 0, 1000000),
            )
            for field_name, component, kind, choices, minimum, maximum in optimization_fields:
                if field_name == "betas":
                    registry.register("training.betas", component, TRAIN_BETAS_TEXT, kind="str")
                    controls["training.betas"] = component
                else:
                    _reg(registry, controls, field_name, component, kind=kind, choices=choices, minimum=minimum, maximum=maximum)

        with gr.Accordion("VRAM & Precision", open=False):
            with gr.Row():
                base_variant = gr.Dropdown(choices=["bf16", "int8_convrot"], value=TRAIN_DEFAULTS["base_variant"], label="Base variant", info="BF16 is the quality default; INT8 ConvRot reduces frozen base weight memory.")
                base_dtype = gr.Dropdown(choices=["bf16", "fp16", "fp32"], value=TRAIN_DEFAULTS["base_dtype"], label="Base dtype", info="Compute/storage dtype for the BF16 base variant.")
                precision = gr.Dropdown(choices=["bf16", "fp16", "fp32"], value=TRAIN_DEFAULTS["mixed_precision"], label="Mixed precision", info="BF16 is recommended on modern GPUs and avoids FP16 overflow.")
                checkpointing = gr.Checkbox(value=TRAIN_DEFAULTS["gradient_checkpointing"], label="Gradient checkpointing", info="Recommended; recomputes activations to save substantial VRAM.")
            with gr.Row():
                blocks = gr.Slider(0, 24, value=TRAIN_DEFAULTS["blocks_to_swap"], step=1, label="Blocks to swap", info="Streams this many frozen GPT blocks from CPU; requires gradient checkpointing.")
                ring = gr.Slider(1, 4, value=TRAIN_DEFAULTS["swap_ring_size"], step=1, label="Swap ring size", info="2 balances overlap and VRAM; 1 uses the least memory.")
                pinned = gr.Checkbox(value=TRAIN_DEFAULTS["pin_swap_memory"], label="Pinned swap memory", info="Recommended for faster CPU-to-GPU transfers.")
                apply_tier = gr.Button("🎚️  Apply VRAM tier defaults", elem_classes=btn("orange"))
            _reg(registry, controls, "base_variant", base_variant, kind="choice", choices=["bf16", "int8_convrot"])
            _reg(registry, controls, "base_dtype", base_dtype, kind="choice", choices=["bf16", "fp16", "fp32"])
            _reg(registry, controls, "mixed_precision", precision, kind="choice", choices=["bf16", "fp16", "fp32"])
            _reg(registry, controls, "gradient_checkpointing", checkpointing, kind="bool")
            _reg(registry, controls, "blocks_to_swap", blocks, kind="int", minimum=0, maximum=24)
            _reg(registry, controls, "swap_ring_size", ring, kind="int", minimum=1, maximum=4)
            _reg(registry, controls, "pin_swap_memory", pinned, kind="bool")

        with gr.Accordion("Saving & Resume", open=False):
            with gr.Row():
                output_dir = gr.Textbox(value=TRAIN_DEFAULTS["output_dir"], label="Output root", info="LoRA / DoRA parent folder; relative paths resolve from the app directory.")
                save_epochs = gr.Number(value=TRAIN_DEFAULTS["save_every_epochs"], minimum=0, precision=0, label="Save every epochs", info="1 keeps an epoch checkpoint; 0 disables epoch checkpoints.")
                save_steps = gr.Number(value=TRAIN_DEFAULTS["save_every_steps"], minimum=0, precision=0, label="Save every steps", info="0 disables step checkpoints.")
                keep = gr.Number(value=TRAIN_DEFAULTS["keep_last_n"], minimum=0, precision=0, label="Keep last N", info="0 keeps every epoch checkpoint so measured checkpoint comparison can choose the best voice.")
            with gr.Row():
                save_best = gr.Checkbox(value=TRAIN_DEFAULTS["save_best"], label="Save best", info="Keeps the checkpoint with the lowest validation loss.")
                save_dtype = gr.Dropdown(choices=["bf16", "fp32"], value=TRAIN_DEFAULTS["save_dtype"], label="LoRA / DoRA save dtype", info="BF16 halves LoRA / DoRA file size; FP32 preserves full update precision.")
                save_state = gr.Checkbox(value=TRAIN_DEFAULTS["save_train_state"], label="Save train state", info="Saves optimizer, scheduler, scaler, RNG, and data position for exact resume from best, final, and interrupted checkpoints.")
                epoch_train_state = gr.Checkbox(
                    value=TRAIN_DEFAULTS["epoch_train_state"],
                    label="Save train state with every epoch checkpoint",
                    info="Only needed to Continue run from a specific epoch; costs ~4x disk per checkpoint.",
                )
                resume = gr.Dropdown(choices=_resume_choices(), value=TRAIN_DEFAULTS["resume_from"], label="Resume from", info="Select a LoRA / DoRA checkpoint; rank, alpha, and type are inspected before launch.")
                resume_mode = gr.Radio(
                    choices=[("Weights only", "weights_only"), ("Continue run", "continue")],
                    value=TRAIN_DEFAULTS["resume_mode"],
                    label="Resume mode",
                    info="Weights only starts a fresh schedule at step 0; Continue run restores train state when available.",
                )
                refresh_resume = gr.Button("🔄  Refresh resume list", elem_classes=btn("sky"))
            with gr.Row():
                auto_analyze = gr.Checkbox(
                    value=TRAIN_DEFAULTS["auto_analyze"],
                    label="Analyze generalization automatically",
                    info="Reads the CPU-only training log after complete or stopped runs and recommends a checkpoint.",
                )
                auto_evaluate = gr.Checkbox(
                    value=TRAIN_DEFAULTS["auto_evaluate_checkpoints"],
                    label="Evaluate checkpoints automatically",
                    info="After training releases its model, measures saved checkpoints on validation and a small training subset.",
                )
                eval_timeout = gr.Number(
                    value=TRAIN_DEFAULTS["eval_timeout_s"],
                    minimum=1,
                    label="Evaluation timeout (s)",
                    info="Stops automatic checkpoint evaluation after this many seconds without failing the completed training run.",
                )
                eval_include_base = gr.Checkbox(
                    value=TRAIN_DEFAULTS["eval_include_base"],
                    label="Evaluate Base model (no LoRA / DoRA)",
                    info="Measures the reference-only baseline before checkpoints for an automatic comparison.",
                )
                eval_train_subset = gr.Number(
                    value=TRAIN_DEFAULTS["eval_train_subset"],
                    minimum=0,
                    maximum=100000,
                    precision=0,
                    label="Evaluation training subset",
                    info="Deterministic training items measured during automatic evaluation; 0 disables the training subset.",
                )
                eval_strengths = gr.Textbox(
                    value=TRAIN_DEFAULTS["eval_strengths"],
                    label="Evaluation strengths",
                    info="Comma-separated LoRA / DoRA strengths from 0 to 4 for automatic checkpoint evaluation.",
                )
            resume_info = gr.Markdown("Start fresh.")
            for field_name, component, kind, choices, minimum, maximum in (
                ("output_dir", output_dir, "str", None, None, None), ("save_every_epochs", save_epochs, "int", None, 0, 100000),
                ("save_every_steps", save_steps, "int", None, 0, 10000000), ("keep_last_n", keep, "int", None, 0, 10000),
                ("save_best", save_best, "bool", None, None, None), ("save_dtype", save_dtype, "choice", ["bf16", "fp32"], None, None),
                ("save_train_state", save_state, "bool", None, None, None),
                ("epoch_train_state", epoch_train_state, "bool", None, None, None),
                ("resume_from", resume, "str", None, None, None),
                ("resume_mode", resume_mode, "choice", ["weights_only", "continue"], None, None),
                ("auto_analyze", auto_analyze, "bool", None, None, None),
                ("auto_evaluate_checkpoints", auto_evaluate, "bool", None, None, None),
                ("eval_include_base", eval_include_base, "bool", None, None, None),
                ("eval_train_subset", eval_train_subset, "int", None, 0, 100000),
                ("eval_strengths", eval_strengths, "str", None, None, None),
                ("eval_timeout_s", eval_timeout, "float", None, 1, 100000),
            ):
                _reg(registry, controls, field_name, component, kind=kind, choices=choices, minimum=minimum, maximum=maximum)

        with gr.Accordion("Sampling", open=False):
            with gr.Row():
                sample_enabled = gr.Checkbox(value=TRAIN_DEFAULTS["sample_enabled"], label="Generate training samples", info="Renders a short sample at the configured epoch interval.")
                sample_epochs = gr.Number(value=TRAIN_DEFAULTS["sample_every_epochs"], minimum=1, precision=0, label="Sample every epochs", info="1 provides a sample after each completed epoch.")
                sample_tier = gr.Dropdown(choices=["auto", "6", "8", "10", "12", "16", "24", "32"], value=TRAIN_DEFAULTS["sample_runtime_tier"], label="Sample runtime tier", info="Memory tier for the isolated sampling process.")
                min_free = gr.Number(value=TRAIN_DEFAULTS["sample_min_free_vram_gb"], minimum=0, label="Minimum free VRAM (GB)", info="Skips sampling rather than risking training OOM below this free-memory threshold.")
                timeout = gr.Number(value=TRAIN_DEFAULTS["sample_timeout_s"], minimum=1, label="Sample timeout (s)", info="Kills a stuck sampling subprocess after this time.")
            sample_text = gr.Textbox(value=TRAIN_DEFAULTS["sample_text"], label="Sample text", lines=3, info="Short representative phrase used to compare epochs.")
            sample_reference = gr.Textbox(value=TRAIN_DEFAULTS["sample_reference"], label="Custom sample reference", info="Optional audio path; blank uses the dataset's best reference candidate automatically.")
            sample_speaking_rate = gr.Slider(
                0.5,
                1.5,
                value=TRAIN_DEFAULTS["sample_speaking_rate"],
                step=0.01,
                label="Sample speaking rate",
                info="1.0 is the model's natural pace; below 1.0 speaks slower and above 1.0 faster.",
            )
            with gr.Row():
                sample_language = gr.Dropdown(
                    choices=["auto", "ZH", "EN", "JA", "AR", "ES"],
                    value=TRAIN_DEFAULTS["sample_language"],
                    label="Sample language",
                    info="Mirrors Voice Generation for per-epoch samples; auto uses the prepared dataset language.",
                )
                sample_seed = gr.Number(
                    value=TRAIN_DEFAULTS["sample_seed"],
                    minimum=-1,
                    maximum=4294967295,
                    precision=0,
                    label="Sample seed",
                    info="Mirrors Voice Generation for per-epoch samples; -1 chooses one seed at training start and reuses it across epochs.",
                )
                sample_num_beams = gr.Number(
                    value=TRAIN_DEFAULTS["sample_num_beams"],
                    minimum=1,
                    maximum=10,
                    precision=0,
                    label="Sample beams",
                    info="Mirrors Voice Generation beam search for every per-epoch sample.",
                )
                sample_temperature = gr.Number(
                    value=TRAIN_DEFAULTS["sample_temperature"],
                    minimum=0.1,
                    maximum=2,
                    label="Sample temperature",
                    info="Mirrors Voice Generation temperature for every per-epoch sample.",
                )
            with gr.Row():
                sample_top_p = gr.Number(
                    value=TRAIN_DEFAULTS["sample_top_p"],
                    minimum=0,
                    maximum=1,
                    label="Sample top-p",
                    info="Mirrors Voice Generation nucleus sampling for every per-epoch sample.",
                )
                sample_top_k = gr.Number(
                    value=TRAIN_DEFAULTS["sample_top_k"],
                    minimum=0,
                    maximum=100,
                    precision=0,
                    label="Sample top-k",
                    info="Mirrors Voice Generation token filtering for per-epoch samples; 0 disables it.",
                )
                sample_repetition_penalty = gr.Number(
                    value=TRAIN_DEFAULTS["sample_repetition_penalty"],
                    minimum=1,
                    maximum=20,
                    label="Sample repetition penalty",
                    info="Mirrors Voice Generation repetition control for every per-epoch sample.",
                )
                sample_emo_alpha = gr.Number(
                    value=TRAIN_DEFAULTS["sample_emo_alpha"],
                    minimum=0,
                    maximum=1,
                    label="Sample emotion weight",
                    info="Mirrors Voice Generation emotion weight for every per-epoch sample.",
                )
            with gr.Row():
                sample_diffusion_steps = gr.Number(
                    value=TRAIN_DEFAULTS["sample_diffusion_steps"],
                    minimum=2,
                    maximum=100,
                    precision=0,
                    label="Sample diffusion steps",
                    info="Mirrors Voice Generation diffusion quality for every per-epoch sample.",
                )
                sample_inference_cfg_rate = gr.Number(
                    value=TRAIN_DEFAULTS["sample_inference_cfg_rate"],
                    minimum=0,
                    maximum=2,
                    label="Sample CFG rate",
                    info="Mirrors Voice Generation conditioning strength for every per-epoch sample.",
                )
                sample_max_text_tokens = gr.Number(
                    value=TRAIN_DEFAULTS["sample_max_text_tokens"],
                    minimum=20,
                    maximum=300,
                    precision=0,
                    label="Sample maximum text tokens",
                    info="Mirrors Voice Generation segment length for every per-epoch sample.",
                )
                sample_length_penalty = gr.Number(
                    value=TRAIN_DEFAULTS["sample_length_penalty"],
                    minimum=-2,
                    maximum=2,
                    label="Sample length penalty",
                    info="Mirrors Voice Generation beam length control for every per-epoch sample.",
                )
                sample_max_mel_tokens = gr.Number(
                    value=TRAIN_DEFAULTS["sample_max_mel_tokens"],
                    minimum=1,
                    maximum=1815,
                    precision=0,
                    label="Sample maximum mel tokens",
                    info="Mirrors Voice Generation output limit for every per-epoch sample.",
                )
            for field_name, component, kind, choices, minimum, maximum in (
                ("sample_enabled", sample_enabled, "bool", None, None, None), ("sample_every_epochs", sample_epochs, "int", None, 1, 10000),
                ("sample_runtime_tier", sample_tier, "choice", ["auto", "6", "8", "10", "12", "16", "24", "32"], None, None),
                ("sample_min_free_vram_gb", min_free, "float", None, 0, 128), ("sample_timeout_s", timeout, "float", None, 1, 100000),
                ("sample_text", sample_text, "str", None, None, None), ("sample_reference", sample_reference, "str", None, None, None),
                ("sample_language", sample_language, "choice", ["auto", "ZH", "EN", "JA", "AR", "ES"], None, None),
                ("sample_seed", sample_seed, "int", None, -1, 4294967295),
                ("sample_num_beams", sample_num_beams, "int", None, 1, 10),
                ("sample_temperature", sample_temperature, "float", None, 0.1, 2),
                ("sample_top_p", sample_top_p, "float", None, 0, 1),
                ("sample_top_k", sample_top_k, "int", None, 0, 100),
                ("sample_repetition_penalty", sample_repetition_penalty, "float", None, 1, 20),
                ("sample_emo_alpha", sample_emo_alpha, "float", None, 0, 1),
                ("sample_diffusion_steps", sample_diffusion_steps, "int", None, 2, 100),
                ("sample_inference_cfg_rate", sample_inference_cfg_rate, "float", None, 0, 2),
                ("sample_max_text_tokens", sample_max_text_tokens, "int", None, 20, 300),
                ("sample_length_penalty", sample_length_penalty, "float", None, -2, 2),
                ("sample_max_mel_tokens", sample_max_mel_tokens, "int", None, 1, 1815),
                ("sample_speaking_rate", sample_speaking_rate, "float", None, 0.5, 1.5),
            ):
                _reg(registry, controls, field_name, component, kind=kind, choices=choices, minimum=minimum, maximum=maximum)

        with gr.Accordion("Miscellaneous", open=False):
            with gr.Row():
                seed = gr.Number(value=TRAIN_DEFAULTS["seed"], precision=0, label="Seed", info="Controls split, sampler, initialization, and training randomness.")
                workers = gr.Number(value=TRAIN_DEFAULTS["num_workers"], minimum=0, maximum=64, precision=0, label="Data workers", info="2 is a safe Windows/Linux default; use 0 to debug worker issues.")
                log_steps = gr.Number(value=TRAIN_DEFAULTS["log_every_steps"], minimum=1, precision=0, label="Log every steps", info="1 gives fully live charts; increase slightly for very fast runs.")
                device = gr.Textbox(value=device_default, label="Training device", info="CUDA device used by the training worker.")
                attention = gr.Dropdown(choices=["sdpa", "eager", "flash_attention_2"], value=TRAIN_DEFAULTS["attention_backend"], label="Attention backend", info="SDPA is the compatible default.")
            with gr.Row():
                model_dir = gr.Textbox(value=model_dir_default, label="Model directory", info="Base IndexTTS 2.5 model directory.")
                model_config = gr.Textbox(value=str(Path(model_dir_default) / "config.yaml"), label="Model config", info="IndexTTS 2.5 YAML configuration path.")
            for field_name, component, kind, choices, minimum, maximum in (
                ("seed", seed, "int", None, -2147483648, 4294967295), ("num_workers", workers, "int", None, 0, 64),
                ("log_every_steps", log_steps, "int", None, 1, 100000), ("device", device, "str", None, None, None),
                ("attention_backend", attention, "choice", ["sdpa", "eager", "flash_attention_2"], None, None),
                ("model_dir", model_dir, "str", None, None, None), ("model_config", model_config, "str", None, None, None),
            ):
                _reg(registry, controls, field_name, component, kind=kind, choices=choices, minimum=minimum, maximum=maximum)

        with gr.Row():
            start = gr.Button("🚀  Start training", variant="primary", elem_classes=btn("emerald"))
            stop = gr.Button("⏹️  Stop", variant="stop", elem_classes=btn("red"))
            force = gr.Button("⛔  Force stop", variant="stop", elem_classes=btn("crimson"))
            open_output = gr.Button("📁  Open output folder", elem_classes=btn("indigo"))
            compare_grid = gr.Button("📊  Compare in grid", elem_classes=btn("fuchsia"))
            use_generation = gr.Button("⭐  Use best checkpoint", elem_classes=btn("purple"))

        state_dir = gr.State(current_state)
        timer = gr.Timer(5.0, active=True)
        dashboard_progress = gr.HTML(progress_panel_html({}, title="Ready"))
        status_text = gr.Markdown("")
        smoothing_slider = gr.Slider(0, 0.99, value=0.9, step=0.01, label="Loss chart smoothing", info="Exponential smoothing for the displayed train-loss line only; raw loss remains visible.")
        with gr.Row():
            loss_plot = gr.LinePlot(empty_series_frame(LOSS_SERIES), x="step", y="value", color="series", title="Training / validation loss", height=300, buttons=["fullscreen", "export"], x_title="step", x_axis_format="d", y_title="loss", colors_in_legend=["train raw", "train smoothed", "train EMA", "validation"], color_map={"train raw": "#6b7280", "train smoothed": "#e11d48", "train EMA": "#f59e0b", "validation": "#22d3ee"})
            lr_plot_component = gr.LinePlot(empty_series_frame(LR_SERIES), x="step", y="value", color="series", title="Learning rate", height=300, buttons=["fullscreen", "export"], x_title="step", x_axis_format="d", y_title="lr", colors_in_legend=["learning rate"], color_map={"learning rate": "#a78bfa"}, y_axis_format=".0e")
        with gr.Row():
            grad_plot = gr.LinePlot(empty_series_frame(GRAD_SERIES), x="step", y="value", color="series", title="Gradient norm", height=280, buttons=["fullscreen", "export"], x_title="step", x_axis_format="d", y_title="grad norm", colors_in_legend=["grad norm"], color_map={"grad norm": "#f97316"})
            speed_plot_component = gr.LinePlot(empty_series_frame(SPEED_SERIES), x="step", y="value", color="series", title="Speed / VRAM", height=280, buttons=["fullscreen", "export"], x_title="step", x_axis_format="d", y_title="value", colors_in_legend=['VRAM GB', 'steps/s'], color_map={'VRAM GB': '#34d399', 'steps/s': '#60a5fa'})
        log = gr.Textbox(label="Training log (last 60 lines)", lines=12, max_lines=18, interactive=False, buttons=["copy"], elem_classes=["log-tail"])
        with gr.Row():
            latest_sample = gr.Audio(label="Latest training sample", type="filepath", buttons=["download"])
            sample_label = gr.Markdown("")
        checkpoints = gr.Dataframe(
            headers=["Checkpoint", "Type", "Rank", "Steps", "Epoch", "Validation loss", "Verdict", "Size MB", "Path"],
            datatype=["str", "str", "number", "number", "number", "number", "str", "number", "str"],
            value=[], type="array", interactive=False, wrap=True,
            label="Checkpoints", max_height=320, buttons=["fullscreen", "copy"],
        )
        initial_generalization, initial_generalization_frame = _training_generalization(current_state)
        generalization_summary = gr.Markdown(initial_generalization)
        generalization_plot = gr.LinePlot(
            initial_generalization_frame,
            x="step",
            y="value",
            color="series",
            title="Generalization by epoch",
            height=300,
            buttons=["fullscreen", "export"],
            x_title="epoch",
            x_axis_format="d",
            y_title="loss",
            colors_in_legend=list(ANALYSIS_SERIES),
            color_map={
                "train loss": "#6b7280",
                "validation (improving)": "#1ca881",
                "validation (overfitting)": "#df345b",
            },
        )

        with gr.Accordion("LoRA Manager", open=False):
            manager_rows, manager_paths = adapter_rows()
            manager_paths_state = gr.State(manager_paths)
            selected_adapter = gr.State("")
            manager_table = gr.Dataframe(
                headers=["Name", "Type", "Rank", "Alpha", "Steps", "Dataset", "Date", "Size MB", "Path"],
                datatype=["str", "str", "number", "number", "number", "str", "str", "number", "str"],
                value=manager_rows, type="array", interactive=False, wrap=True,
                label="LoRA / DoRA files", max_height=380, buttons=["fullscreen", "copy"], elem_classes=["manager-table"],
            )
            manager_details = gr.Markdown("Select a LoRA / DoRA row for details.")
            with gr.Row():
                manager_refresh = gr.Button("🔃  Refresh", elem_classes=btn("green"))
                manager_delete = gr.Button("✖️  Delete", variant="stop", elem_classes=btn("pink"))
                manager_open = gr.Button("🗂️  Open folder", elem_classes=btn("teal"))

    config_specs = [spec for spec in registry.specs if spec.component is not None and spec.key.startswith("training.")]
    config_keys = [spec.key for spec in config_specs]
    config_components = [spec.component for spec in config_specs]

    def build_config(*items: Any) -> TrainConfig:
        values = dict(zip(config_keys, items))
        payload = {key.removeprefix("training."): value for key, value in values.items()}
        beta_value = payload.get("betas", TRAIN_BETAS_TEXT)
        if isinstance(beta_value, str):
            pieces = [piece.strip() for piece in beta_value.split(",")]
            if len(pieces) != 2:
                raise ValueError("Adam betas must contain exactly two comma-separated values")
            payload["betas"] = [float(piece) for piece in pieces]
        for path_field in ("dataset_dir", "output_dir", "model_dir", "model_config", "resume_from", "sample_reference"):
            value = str(payload.get(path_field) or "")
            if value and path_field in {"dataset_dir", "output_dir", "model_dir", "model_config"}:
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = ROOT / path
                payload[path_field] = str(path.resolve())
        return TrainConfig.from_dict(payload)

    def start_training(*items: Any):
        global _LAST_TRAINING_FOLDER
        try:
            smoothing_value = float(items[-1])
            config = build_config(*items[:-1])
            dataset_root = Path(config.dataset_dir)
            if not (dataset_root / "manifest.jsonl").is_file():
                raise ValueError(f"Dataset manifest not found: {dataset_root}")
            if not (dataset_root / "cache" / "index.jsonl").is_file():
                raise ValueError("Dataset features are not cached. Use 'Cache features now' in Dataset Preparation first.")
            adapter_dir = Path(config.output_dir) / config.name
            adapter_dir.mkdir(parents=True, exist_ok=True)
            _LAST_TRAINING_FOLDER = adapter_dir
            stop_flag = adapter_dir / "stop.flag"
            stop_flag.unlink(missing_ok=True)
            config_path = write_json_atomic(adapter_dir / "train_config.json", config.to_dict())
            write_json_atomic(
                adapter_dir / "status.json",
                {
                    "phase": "initializing",
                    "step": 0,
                    "total_steps": int(config.max_steps or 0),
                    "epoch": 0,
                    "total_epochs": int(config.epochs),
                    "elapsed_s": 0.0,
                    "eta_s": None,
                    "message": "Starting training worker",
                    "updated_at": time.time(),
                },
            )
            job = PROCESS_MANAGER.start(
                "training",
                [sys.executable, "-m", "indextts.training.train_worker", "--config", str(config_path), "--state-dir", str(adapter_dir)],
                state_dir=adapter_dir,
                log_path=adapter_dir / "worker_console.log",
                cwd=ROOT,
                metadata={"adapter_dir": str(adapter_dir)},
            )
            message = f"Training {config.name} started with {config.adapter_type.upper()} rank {config.rank}."
            print(">> " + message, flush=True)
            updates = list(training_poll_updates(str(adapter_dir), smoothing_value))
            updates[-1] = gr.Timer(1.0, active=True)
            emitted, fingerprints = dedupe_updates(updates)
            yield emitted
            while job.running:
                time.sleep(1.0)
                updates = list(training_poll_updates(str(adapter_dir), smoothing_value))
                updates[0] = gr.skip()
                updates[-1] = gr.skip()
                emitted, fingerprints = dedupe_updates(updates, fingerprints)
                yield emitted
            current = read_json(adapter_dir / "status.json", {}) or {}
            phase = str(current.get("phase") or "")
            if phase not in TRAINING_TERMINAL_PHASES:
                current.update(
                    phase="cancelled" if job.canceled else "failed",
                    message=(
                        "Training canceled by user"
                        if job.canceled
                        else f"Training worker exited with code {job.process.returncode}"
                    ),
                    updated_at=time.time(),
                )
                write_json_atomic(adapter_dir / "status.json", current)
            updates = list(training_poll_updates(str(adapter_dir), smoothing_value))
            updates[0] = gr.skip()
            updates[-1] = gr.Timer(5.0, active=True)
            emitted, _ = dedupe_updates(updates, fingerprints)
            yield emitted
        except Exception as exc:
            traceback.print_exc()
            raise gr.Error(str(exc)) from exc

    start_event = start.click(
        start_training,
        [*config_components, smoothing_slider],
        [
            state_dir,
            dashboard_progress,
            status_text,
            loss_plot,
            lr_plot_component,
            grad_plot,
            speed_plot_component,
            log,
            latest_sample,
            sample_label,
            checkpoints,
            generalization_summary,
            generalization_plot,
            timer,
        ],
        api_name="start_training",
        concurrency_limit=1,
        concurrency_id="training",
        stream_every=0.5,
    )

    poll_outputs = [state_dir, dashboard_progress, status_text, loss_plot, lr_plot_component, grad_plot, speed_plot_component, log, latest_sample, sample_label, checkpoints, generalization_summary, generalization_plot, timer]
    timer.tick(training_poll_updates, [state_dir, smoothing_slider], poll_outputs, queue=False, show_progress="hidden")
    smoothing_slider.change(training_poll_updates, [state_dir, smoothing_slider], poll_outputs, queue=False, show_progress="hidden", trigger_mode="always_last")
    if load_hook is not None:
        load_hook(
            lambda state, smooth: training_poll_updates(state, smooth, page_load=True),
            [state_dir, smoothing_slider],
            poll_outputs,
            queue=False,
            show_progress="hidden",
            api_name="attach_training",
        )

    with tab_block:
        stop_confirm = gr.Checkbox(value=False, visible=False, label="Training stop confirmation")
        force_confirm = gr.Checkbox(value=False, visible=False, label="Training force-stop confirmation")

    def graceful_stop(confirmed: bool, state_value: str):
        if not confirmed:
            return "Stop dismissed."
        if not state_value or not _state_running(state_value):
            return "No active run."
        (Path(state_value) / "stop.flag").touch()
        return "Graceful stop requested. Training will finish the current optimizer step and save an interrupted checkpoint."

    def force_stop(confirmed: bool, state_value: str):
        if not confirmed:
            return "Force stop dismissed."
        if not state_value or not _state_running(state_value):
            return "No active run."
        job = PROCESS_MANAGER.get("training")
        displayed = Path(state_value).resolve()
        if job is not None and job.running and job.state_dir.resolve() == displayed and PROCESS_MANAGER.terminate("training"):
            current = read_json(displayed / "status.json", {}) or {}
            current.update(
                phase="cancelled",
                message="Training subprocess tree was force-stopped",
                updated_at=time.time(),
            )
            write_json_atomic(displayed / "status.json", current)
            return "Training subprocess tree was force-stopped. The last completed checkpoint remains available."
        return "The active displayed run is not managed by this app process."

    stop.click(graceful_stop, [stop_confirm, state_dir], status_text, js="(value, state) => [window.confirm('Stop gracefully after the current step and save?'), state]", queue=False)
    force.click(force_stop, [force_confirm, state_dir], status_text, js="(value, state) => [window.confirm('Force stop training immediately? Unsaved work will be lost.'), state]", queue=False)
    open_output.click(lambda state: open_folder(state or _LAST_TRAINING_FOLDER), state_dir, status_text, queue=False)

    refresh_dataset_event = refresh_dataset.click(
        lambda: gr.update(choices=_dataset_choices()), outputs=dataset, queue=False
    )
    dataset.change(_dataset_summary, dataset, dataset_info, queue=False)
    plan_inputs = [dataset, batch_size, accumulation, epochs, max_steps, val_fraction, seed]
    for plan_input in plan_inputs:
        plan_input.change(
            _training_plan_markdown,
            plan_inputs,
            training_plan_readout,
            queue=False,
            show_progress="hidden",
            trigger_mode="always_last",
        )
    refresh_dataset_event.then(
        _training_plan_markdown,
        plan_inputs,
        training_plan_readout,
        queue=False,
        show_progress="hidden",
    )
    refresh_resume.click(lambda: gr.update(choices=_resume_choices()), outputs=resume, queue=False)

    def inspect_resume(path: str, current_type: str, current_rank: int, current_alpha: float):
        if not path:
            return "Start fresh."
        try:
            info = inspect_lora(path)
            warnings = []
            if info["adapter_type"] != current_type:
                warnings.append(f"type mismatch: checkpoint {info['adapter_type']} vs UI {current_type}")
            if int(info["rank"]) != int(current_rank):
                warnings.append(f"rank mismatch: checkpoint {info['rank']} vs UI {current_rank}")
            if abs(float(info["alpha"]) - float(current_alpha)) > 1e-6:
                warnings.append(f"alpha mismatch: checkpoint {info['alpha']} vs UI {current_alpha}")
            note = " | **Warning:** " + "; ".join(warnings) if warnings else " | Settings match."
            return f"{str(info['adapter_type']).upper()} rank {info['rank']} alpha {info['alpha']} | {info.get('steps', 0)} steps{note}"
        except Exception as exc:
            return f"Resume inspection failed: {exc}"

    resume.change(inspect_resume, [resume, adapter_type, rank, alpha], resume_info, queue=False)

    def manager_select(paths: list[str], evt: gr.SelectData):
        index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        try:
            path = paths[int(index)]
            info = inspect_lora(path)
            detail = (
                f"**{Path(path).stem}** | {str(info['adapter_type']).upper()} rank {info['rank']} alpha {info['alpha']} | "
                f"{info.get('steps', 0)} steps | dataset {info.get('dataset') or 'unknown'} | {info.get('size_mb', 0):.2f} MB  \n"
                f"`{path}`"
            )
            return path, detail
        except Exception as exc:
            return "", f"Selection failed: {exc}"

    manager_table.select(manager_select, manager_paths_state, [selected_adapter, manager_details], queue=False)

    def refresh_manager():
        rows, paths = adapter_rows()
        return rows, paths, "LoRA / DoRA list refreshed."

    manager_refresh.click(refresh_manager, outputs=[manager_table, manager_paths_state, manager_details], queue=False)

    with tab_block:
        delete_confirm = gr.Checkbox(value=False, visible=False, label="LoRA / DoRA delete confirmation")

    def delete_adapter(confirmed: bool, path: str):
        if not confirmed:
            return gr.skip(), gr.skip(), "Deletion dismissed.", ""
        if not path:
            return gr.skip(), gr.skip(), "Select a LoRA / DoRA first.", ""
        source = Path(path).resolve()
        lora_root = (ROOT / "loras").resolve()
        if lora_root not in source.parents:
            raise gr.Error("Refusing to delete a LoRA / DoRA outside the loras directory")
        folder = source.parent
        if folder.parent == lora_root or (folder / "status.json").is_file() or folder.name == source.stem:
            shutil.rmtree(folder)
        else:
            source.unlink(missing_ok=True)
            source.with_name(f"{source.stem}.train_state.pt").unlink(missing_ok=True)
        rows, paths = adapter_rows()
        return rows, paths, f"Deleted {source.name}.", ""

    manager_delete.click(
        delete_adapter,
        [delete_confirm, selected_adapter],
        [manager_table, manager_paths_state, manager_details, selected_adapter],
        js="(value, path) => [window.confirm('Delete the selected LoRA / DoRA and its training folder?'), path]",
        queue=False,
    )
    manager_open.click(lambda path: open_folder(Path(path).parent if path else ROOT / "loras"), selected_adapter, manager_details, queue=False)

    expected = {f"training.{item.name}" for item in fields(TrainConfig)}
    actual = set(controls)
    if expected != actual:
        raise RuntimeError(f"Training UI field mismatch: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    return TrainingTab(
        controls=controls,
        apply_tier_button=apply_tier,
        base_variant=base_variant,
        mixed_precision=precision,
        blocks_to_swap=blocks,
        swap_ring_size=ring,
        pin_swap_memory=pinned,
        state_dir=state_dir,
        use_in_generation=use_generation,
        compare_grid=compare_grid,
        dataset=dataset,
        dataset_info=dataset_info,
        training_plan=training_plan_readout,
        start_event=start_event,
    )


def bind_training_events(
    tab: TrainingTab,
    models: ModelsTab,
    generation: GenerationTab,
    main_tabs: Any,
) -> None:
    def apply_tier(tier_value: str):
        total = 32.0
        if tier_value in {"auto", "custom"}:
            try:
                from indextts.runtime.gpu import list_gpus

                gpus = list_gpus()
                total = gpus[0].total_gb if gpus else 6.0
            except Exception:
                total = 6.0
            tier_value = str(auto_tier(total))
        cfg = resolve_preset(tier_value, total)
        return cfg.model_variant, cfg.gpt_dtype, max(0, cfg.blocks_to_swap), cfg.swap_ring_size, cfg.pin_swap_memory

    tab.apply_tier_button.click(
        apply_tier,
        models.tier,
        [tab.base_variant, tab.mixed_precision, tab.blocks_to_swap, tab.swap_ring_size, tab.pin_swap_memory],
        queue=False,
    )

    lora_component = generation.controls.get("runtime.lora_path")
    if lora_component is not None:
        def refresh_generation_adapters():
            from .generation_tab import _lora_choices

            return gr.update(choices=_lora_choices())

        if tab.start_event is not None:
            tab.start_event.then(
                refresh_generation_adapters,
                outputs=lora_component,
                queue=False,
            )

        def use_adapter(state_value: str):
            if not state_value:
                raise gr.Error("No training state is attached")
            root = Path(state_value)
            measured = load_checkpoint_eval(root)
            analysis = load_training_analysis(root)
            status = read_json(root / "status.json", {}) or {}
            path = (
                measured.recommended_checkpoint
                if measured is not None
                else (
                    analysis.recommended_checkpoint
                    if analysis is not None
                    else status.get("recommended_checkpoint")
                )
            )
            if not path or not Path(path).is_file():
                path = status.get("last_checkpoint")
            if not path or not Path(path).is_file():
                candidates = sorted(root.glob("*.safetensors"), key=lambda item: item.stat().st_mtime, reverse=True)
                path = str(candidates[0]) if candidates else ""
            if not path:
                raise gr.Error("No completed checkpoint is available yet")
            from .generation_tab import _lora_choices

            return (
                gr.update(choices=_lora_choices(), value=str(Path(path).resolve())),
                gr.Tabs(selected="voice-generation"),
            )

        tab.use_in_generation.click(
            use_adapter,
            tab.state_dir,
            [lora_component, main_tabs],
            queue=False,
        )


__all__ = [
    "TRAIN_DEFAULTS",
    "TRAINING_TERMINAL_PHASES",
    "TrainingTab",
    "adopt_training_state",
    "adapter_rows",
    "bind_training_events",
    "build_training_tab",
    "latest_training_state",
    "training_poll_updates",
    "training_status_updates",
]
