"""LoRA / DoRA training dashboard and adapter manager."""

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
from indextts.training.charts import GRAD_SERIES, LOSS_SERIES, LR_SERIES, SPEED_SERIES, empty_series_frame, load_metrics, lr_frame, speed_frame
from indextts.training.train_config import TrainConfig

from .common import (
    PROCESS_MANAGER,
    ROOT,
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
_LAST_TRAINING_FOLDER = ROOT / "loras"
TRAINING_TERMINAL_PHASES = frozenset(
    {"complete", "stopped", "failed", "error", "cancelled", "canceled"}
)


def _training_states(root: str | Path = ROOT / "loras") -> list[Path]:
    values: list[tuple[float, Path]] = []
    base = Path(root).expanduser()
    for status_path in base.rglob("status.json") if base.is_dir() else []:
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
    plotted = (
        pd.concat(pieces, ignore_index=True)
        .drop_duplicates(subset=["step", "series"], keep="last")
        .sort_values(["series", "step"], kind="stable")
        .reset_index(drop=True)
    )
    return plotted if not plotted.empty else empty_series_frame(LOSS_SERIES)


def _grad_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "grad_norm" not in frame:
        return empty_series_frame(GRAD_SERIES)
    plotted = (
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
    rows = []
    for path in sorted(root.rglob("*.safetensors"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            info = inspect_lora(path)
            rows.append([path.name, info.get("adapter_type", ""), info.get("rank", 0), info.get("steps", 0), round(path.stat().st_size / 1024**2, 2), str(path)])
        except Exception:
            rows.append([path.name, "", 0, 0, round(path.stat().st_size / 1024**2, 2), str(path)])
    return rows


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
            progress_panel_html({}, title="Training ready"),
            "Ready.",
            empty_series_frame(LOSS_SERIES),
            empty_series_frame(LR_SERIES),
            empty_series_frame(GRAD_SERIES),
            empty_series_frame(SPEED_SERIES),
            "",
            None,
            "No sample yet.",
            [],
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
    titles = {
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
    dataset: Any
    dataset_info: Any
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
    initial_dataset = datasets[0][1] if datasets else str(ROOT / "datasets" / "secourses_demo")
    # State is deliberately session-local and empty at build time.  demo.load
    # discovers the current run for each new browser session.
    current_state = ""
    model_dir_default = str(Path(getattr(args, "model_dir", ROOT / "models")).resolve())
    device_default = str(getattr(args, "device", "cuda:0") or "cuda:0")
    if device_default == "auto":
        device_default = "cuda:0"

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
            refresh_dataset = gr.Button("Refresh", elem_classes=["compact-button"], scale=1)
        dataset_info = gr.Markdown(_dataset_summary(initial_dataset if Path(initial_dataset).is_dir() else None))
        _reg(registry, controls, "dataset_dir", dataset, kind="str")

        with gr.Accordion("Adapter", open=True):
            with gr.Row():
                name = gr.Textbox(value="voice_adapter", label="Adapter name", info="Safe output folder and final safetensors basename.")
                adapter_type = gr.Dropdown(choices=["lora", "dora"], value="dora", label="Adapter type", info="DoRA is the quality default; LoRA uses slightly less compute.")
                rank = gr.Slider(1, 256, value=32, step=1, label="Rank", info="32 is the recommended balance of voice capacity, size, and VRAM.")
                alpha = gr.Number(value=32.0, minimum=1, maximum=1024, label="Alpha", info="Usually equal to rank for a neutral adapter scale.")
                dropout = gr.Slider(0, 0.5, value=0.05, step=0.01, label="Dropout", info="0.05 helps regularize small voice datasets.")
            with gr.Row():
                target_attention = gr.Checkbox(value=True, label="Target attention", info="Adapts GPT attention projections; recommended.")
                target_mlp = gr.Checkbox(value=True, label="Target MLP", info="Adapts GPT feed-forward projections; recommended for voice fidelity.")
                train_spk = gr.Checkbox(value=True, label="Train speaker projection", info="Fully trains the small speaker projection module.")
                train_emo = gr.Checkbox(value=False, label="Train emotion layers", info="Advanced: trains small emotion modules in addition to adapters.")
                train_mel = gr.Checkbox(value=False, label="Train mel embedding head", info="Advanced: trains the mel token embedding/head modules.")
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
                learning_rate = gr.Number(value=1e-4, minimum=1e-8, maximum=1, label="Learning rate", info="1e-4 is the established adapter training default.")
                optimizer = gr.Dropdown(choices=["adamw", "adamw_fused", "prodigy"], value="adamw", label="Optimizer", info="AdamW is portable; fused AdamW is faster on supported CUDA builds.")
                scheduler = gr.Dropdown(choices=["cosine", "linear", "constant", "constant_with_warmup"], value="cosine", label="Scheduler", info="Cosine decay is recommended for multi-epoch voice adaptation.")
                warmup = gr.Number(value=50, minimum=0, precision=0, label="Warmup steps", info="Ramps the learning rate to avoid unstable early updates.")
                weight_decay = gr.Number(value=0.01, minimum=0, maximum=1, label="Weight decay", info="0.01 is a mild regularizer.")
            with gr.Row():
                betas = gr.Textbox(value="0.9, 0.99", label="Adam betas", info="Two comma-separated momentum coefficients; 0.9, 0.99 is recommended.")
                eps = gr.Number(value=1e-8, minimum=1e-12, maximum=0.1, label="Adam epsilon", info="Numerical stability term for Adam-family optimizers.")
                epochs = gr.Number(value=15, minimum=1, maximum=10000, precision=0, label="Epochs", info="15 is a sensible starting point; validation helps select the best checkpoint.")
                max_steps = gr.Number(value=0, minimum=0, precision=0, label="Maximum steps", info="0 derives steps from epochs; set 5 for a quick smoke run.")
                batch_size = gr.Number(value=4, minimum=1, maximum=128, precision=0, label="Batch size", info="4 is the quality default; lower it first when VRAM is tight.")
                accumulation = gr.Number(value=2, minimum=1, maximum=128, precision=0, label="Gradient accumulation", info="Effective batch is batch size times accumulation.")
            with gr.Row():
                grad_clip = gr.Number(value=1.0, minimum=0, label="Gradient clip", info="1.0 limits unstable gradient spikes; 0 disables clipping.")
                smoothing = gr.Slider(0, 0.5, value=0, step=0.01, label="Label smoothing", info="0 is recommended; increase only for overconfident large datasets.")
                mel_weight = gr.Number(value=1.0, minimum=0, label="Mel loss weight", info="Primary autoregressive acoustic-token loss weight.")
                text_weight = gr.Number(value=0.1, minimum=0, label="Text loss weight", info="Auxiliary text modeling loss weight.")
                speaker_mode = gr.Dropdown(choices=["self", "other", "mixed"], value="mixed", label="Speaker reference mode", info="Mixed alternates self/other segments and is recommended for robust identity.")
            with gr.Row():
                max_codes = gr.Number(value=1500, minimum=1, precision=0, label="Maximum codes", info="Cached samples longer than this semantic-code limit are rejected.")
                max_text = gr.Number(value=600, minimum=1, precision=0, label="Maximum text tokens", info="Cached text length safety limit.")
                val_fraction = gr.Slider(0, 0.5, value=0.05, step=0.01, label="Validation fraction", info="5% provides useful validation without sacrificing much training data.")
                val_steps = gr.Number(value=50, minimum=0, precision=0, label="Validate every steps", info="0 disables step validation; epoch validation still runs when a split exists.")
                val_batches = gr.Number(value=20, minimum=1, precision=0, label="Maximum validation batches", info="Caps validation time on large datasets.")
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
                ("max_codes", max_codes, "int", None, 1, 100000), ("max_text_tokens", max_text, "int", None, 1, 100000),
                ("val_fraction", val_fraction, "float", None, 0, 0.5), ("val_every_steps", val_steps, "int", None, 0, 1000000),
                ("val_max_batches", val_batches, "int", None, 1, 1000000),
            )
            for field_name, component, kind, choices, minimum, maximum in optimization_fields:
                default = TRAIN_DEFAULTS[field_name]
                if field_name == "betas":
                    registry.register("training.betas", component, "0.9, 0.99", kind="str")
                    controls["training.betas"] = component
                else:
                    _reg(registry, controls, field_name, component, kind=kind, choices=choices, minimum=minimum, maximum=maximum)

        with gr.Accordion("VRAM & Precision", open=False):
            with gr.Row():
                base_variant = gr.Dropdown(choices=["bf16", "int8_convrot"], value="bf16", label="Base variant", info="BF16 is the quality default; INT8 ConvRot reduces frozen base weight memory.")
                base_dtype = gr.Dropdown(choices=["bf16", "fp16", "fp32"], value="bf16", label="Base dtype", info="Compute/storage dtype for the BF16 base variant.")
                precision = gr.Dropdown(choices=["bf16", "fp16", "fp32"], value="bf16", label="Mixed precision", info="BF16 is recommended on modern GPUs and avoids FP16 overflow.")
                checkpointing = gr.Checkbox(value=True, label="Gradient checkpointing", info="Recommended; recomputes activations to save substantial VRAM.")
            with gr.Row():
                blocks = gr.Slider(0, 24, value=0, step=1, label="Blocks to swap", info="Streams this many frozen GPT blocks from CPU; requires gradient checkpointing.")
                ring = gr.Slider(1, 4, value=2, step=1, label="Swap ring size", info="2 balances overlap and VRAM; 1 uses the least memory.")
                pinned = gr.Checkbox(value=True, label="Pinned swap memory", info="Recommended for faster CPU-to-GPU transfers.")
                apply_tier = gr.Button("Apply VRAM tier defaults")
            _reg(registry, controls, "base_variant", base_variant, kind="choice", choices=["bf16", "int8_convrot"])
            _reg(registry, controls, "base_dtype", base_dtype, kind="choice", choices=["bf16", "fp16", "fp32"])
            _reg(registry, controls, "mixed_precision", precision, kind="choice", choices=["bf16", "fp16", "fp32"])
            _reg(registry, controls, "gradient_checkpointing", checkpointing, kind="bool")
            _reg(registry, controls, "blocks_to_swap", blocks, kind="int", minimum=0, maximum=24)
            _reg(registry, controls, "swap_ring_size", ring, kind="int", minimum=1, maximum=4)
            _reg(registry, controls, "pin_swap_memory", pinned, kind="bool")

        with gr.Accordion("Saving & Resume", open=False):
            with gr.Row():
                output_dir = gr.Textbox(value="loras", label="Output root", info="Adapter parent folder; relative paths resolve from the app directory.")
                save_epochs = gr.Number(value=1, minimum=0, precision=0, label="Save every epochs", info="1 keeps an epoch checkpoint; 0 disables epoch checkpoints.")
                save_steps = gr.Number(value=0, minimum=0, precision=0, label="Save every steps", info="0 disables step checkpoints.")
                keep = gr.Number(value=3, minimum=0, precision=0, label="Keep last N", info="Prunes periodic checkpoints; best/final checkpoints are retained.")
            with gr.Row():
                save_best = gr.Checkbox(value=True, label="Save best", info="Keeps the checkpoint with the lowest validation loss.")
                save_dtype = gr.Dropdown(choices=["bf16", "fp32"], value="bf16", label="Adapter save dtype", info="BF16 halves adapter file size; FP32 preserves full update precision.")
                save_state = gr.Checkbox(value=True, label="Save train state", info="Saves optimizer, scheduler, scaler, RNG, and data position for exact resume.")
                resume = gr.Dropdown(choices=_resume_choices(), value="", label="Resume from", info="Select an adapter checkpoint; rank, alpha, and type are inspected before launch.")
                resume_mode = gr.Radio(
                    choices=[("Weights only", "weights_only"), ("Continue run", "continue")],
                    value="weights_only",
                    label="Resume mode",
                    info="Weights only starts a fresh schedule at step 0; Continue run restores train state when available.",
                )
                refresh_resume = gr.Button("Refresh resume list", elem_classes=["compact-button"])
            resume_info = gr.Markdown("Start fresh.")
            for field_name, component, kind, choices, minimum, maximum in (
                ("output_dir", output_dir, "str", None, None, None), ("save_every_epochs", save_epochs, "int", None, 0, 100000),
                ("save_every_steps", save_steps, "int", None, 0, 10000000), ("keep_last_n", keep, "int", None, 0, 10000),
                ("save_best", save_best, "bool", None, None, None), ("save_dtype", save_dtype, "choice", ["bf16", "fp32"], None, None),
                ("save_train_state", save_state, "bool", None, None, None), ("resume_from", resume, "str", None, None, None),
                ("resume_mode", resume_mode, "choice", ["weights_only", "continue"], None, None),
            ):
                _reg(registry, controls, field_name, component, kind=kind, choices=choices, minimum=minimum, maximum=maximum)

        with gr.Accordion("Sampling", open=False):
            with gr.Row():
                sample_enabled = gr.Checkbox(value=True, label="Generate training samples", info="Renders a short sample at the configured epoch interval.")
                sample_epochs = gr.Number(value=1, minimum=1, precision=0, label="Sample every epochs", info="1 provides a sample after each completed epoch.")
                sample_tier = gr.Dropdown(choices=["auto", "6", "8", "10", "12", "16", "24", "32"], value="auto", label="Sample runtime tier", info="Memory tier for the isolated sampling process.")
                min_free = gr.Number(value=6.0, minimum=0, label="Minimum free VRAM (GB)", info="Skips sampling rather than risking training OOM below this free-memory threshold.")
                timeout = gr.Number(value=300, minimum=1, label="Sample timeout (s)", info="Kills a stuck sampling subprocess after this time.")
            sample_text = gr.Textbox(value=TRAIN_DEFAULTS["sample_text"], label="Sample text", lines=3, info="Short representative phrase used to compare epochs.")
            sample_reference = gr.Textbox(value="", label="Custom sample reference", info="Optional audio path; blank uses the dataset's best reference candidate automatically.")
            for field_name, component, kind, choices, minimum, maximum in (
                ("sample_enabled", sample_enabled, "bool", None, None, None), ("sample_every_epochs", sample_epochs, "int", None, 1, 10000),
                ("sample_runtime_tier", sample_tier, "choice", ["auto", "6", "8", "10", "12", "16", "24", "32"], None, None),
                ("sample_min_free_vram_gb", min_free, "float", None, 0, 128), ("sample_timeout_s", timeout, "float", None, 1, 100000),
                ("sample_text", sample_text, "str", None, None, None), ("sample_reference", sample_reference, "str", None, None, None),
            ):
                _reg(registry, controls, field_name, component, kind=kind, choices=choices, minimum=minimum, maximum=maximum)

        with gr.Accordion("Miscellaneous", open=False):
            with gr.Row():
                seed = gr.Number(value=42, precision=0, label="Seed", info="Controls split, sampler, initialization, and training randomness.")
                workers = gr.Number(value=2, minimum=0, maximum=64, precision=0, label="Data workers", info="2 is a safe Windows/Linux default; use 0 to debug worker issues.")
                log_steps = gr.Number(value=1, minimum=1, precision=0, label="Log every steps", info="1 gives fully live charts; increase slightly for very fast runs.")
                device = gr.Textbox(value=device_default, label="Training device", info="CUDA device used by the training worker.")
                attention = gr.Dropdown(choices=["sdpa", "eager", "flash_attention_2"], value="sdpa", label="Attention backend", info="SDPA is the compatible default.")
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
            start = gr.Button("Start training", variant="primary", elem_classes=["premium-primary"])
            stop = gr.Button("Stop", variant="stop", elem_classes=["danger-button"])
            force = gr.Button("Force stop", variant="stop", elem_classes=["danger-button"])
            open_output = gr.Button("Open output folder")
            use_generation = gr.Button("Use this LoRA in Voice Generation")

        state_dir = gr.State(current_state)
        timer = gr.Timer(5.0, active=True)
        dashboard_progress = gr.HTML(progress_panel_html({}, title="Training ready"))
        status_text = gr.Markdown("Ready.")
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
            sample_label = gr.Markdown("No sample yet.")
        checkpoints = gr.Dataframe(
            headers=["Checkpoint", "Type", "Rank", "Steps", "Size MB", "Path"],
            datatype=["str", "str", "number", "number", "number", "str"],
            value=_checkpoint_rows(current_state), type="array", interactive=False, wrap=True,
            label="Checkpoints", max_height=320, buttons=["fullscreen", "copy"],
        )

        with gr.Accordion("LoRA Manager", open=False):
            manager_rows, manager_paths = adapter_rows()
            manager_paths_state = gr.State(manager_paths)
            selected_adapter = gr.State("")
            manager_table = gr.Dataframe(
                headers=["Name", "Type", "Rank", "Alpha", "Steps", "Dataset", "Date", "Size MB", "Path"],
                datatype=["str", "str", "number", "number", "number", "str", "str", "number", "str"],
                value=manager_rows, type="array", interactive=False, wrap=True,
                label="Adapters", max_height=380, buttons=["fullscreen", "copy"], elem_classes=["manager-table"],
            )
            manager_details = gr.Markdown("Select an adapter row for details.")
            with gr.Row():
                manager_refresh = gr.Button("Refresh")
                manager_delete = gr.Button("Delete", variant="stop")
                manager_open = gr.Button("Open folder")

    config_specs = [spec for spec in registry.specs if spec.component is not None and spec.key.startswith("training.")]
    config_keys = [spec.key for spec in config_specs]
    config_components = [spec.component for spec in config_specs]

    def build_config(*items: Any) -> TrainConfig:
        values = dict(zip(config_keys, items))
        payload = {key.removeprefix("training."): value for key, value in values.items()}
        beta_value = payload.get("betas", "0.9,0.99")
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
            timer,
        ],
        api_name="start_training",
        concurrency_limit=1,
        concurrency_id="training",
        stream_every=0.5,
    )

    poll_outputs = [state_dir, dashboard_progress, status_text, loss_plot, lr_plot_component, grad_plot, speed_plot_component, log, latest_sample, sample_label, checkpoints, timer]
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

    refresh_dataset.click(lambda: gr.update(choices=_dataset_choices()), outputs=dataset, queue=False)
    dataset.change(_dataset_summary, dataset, dataset_info, queue=False)
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
        return rows, paths, "Adapter list refreshed."

    manager_refresh.click(refresh_manager, outputs=[manager_table, manager_paths_state, manager_details], queue=False)

    with tab_block:
        delete_confirm = gr.Checkbox(value=False, visible=False, label="Adapter delete confirmation")

    def delete_adapter(confirmed: bool, path: str):
        if not confirmed:
            return gr.skip(), gr.skip(), "Deletion dismissed.", ""
        if not path:
            return gr.skip(), gr.skip(), "Select an adapter first.", ""
        source = Path(path).resolve()
        lora_root = (ROOT / "loras").resolve()
        if lora_root not in source.parents:
            raise gr.Error("Refusing to delete an adapter outside the loras directory")
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
        js="(value, path) => [window.confirm('Delete the selected adapter and its managed training folder?'), path]",
        queue=False,
    )
    manager_open.click(lambda path: open_folder(Path(path).parent if path else ROOT / "loras"), selected_adapter, manager_details, queue=False)

    expected = {f"training.{item.name}" for item in fields(TrainConfig)}
    actual = set(controls)
    if expected != actual:
        raise RuntimeError(f"Training UI field mismatch: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    return TrainingTab(
        controls,
        apply_tier,
        base_variant,
        precision,
        blocks,
        ring,
        pinned,
        state_dir,
        use_generation,
        dataset,
        dataset_info,
        start_event,
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
            status = read_json(Path(state_value) / "status.json", {}) or {}
            path = status.get("last_checkpoint")
            if not path or not Path(path).is_file():
                candidates = sorted(Path(state_value).glob("*.safetensors"), key=lambda item: item.stat().st_mtime, reverse=True)
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
