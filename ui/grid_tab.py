"""Checkpoint generalization dashboard and listening-grid controls."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import gradio as gr

from indextts.runtime.gpu import list_gpus
from indextts.training.analysis import (
    ANALYSIS_SERIES,
    BASE_CHECKPOINT_CHOICE_LABEL,
    BASE_CHECKPOINT_LABEL,
    BASE_GRID_HEADER_DETAIL,
    GENERALIZATION_LEGEND,
    analysis_epoch_frame,
    analyze_training_run,
    checkpoint_descriptor,
    checkpoint_display_label,
    display_legacy_report_text,
    discover_checkpoints,
    load_training_analysis,
    phase_display_label,
    write_training_analysis,
)
from indextts.training.checkpoint_eval import (
    CheckpointEvalConfig,
    load_checkpoint_eval,
)
from indextts.training.grid import (
    GridCheckpoint,
    GridConfig,
    list_grids,
    load_grid,
)
from indextts.training.speaking_rate import (
    calibrate_from_grid,
    speaking_rate_method_label,
    write_speaking_rate,
)

from .common import (
    PROCESS_MANAGER,
    ROOT,
    btn,
    open_folder,
    parse_multiline_paths,
    progress_panel_html,
    read_json,
    runtime_config_from_values,
    tail_text,
    write_json_atomic,
)
from .generation_tab import (
    GENERATION_DEFAULTS,
    LANGUAGES,
    GenerationTab,
    _lora_choices,
    build_generation_request,
)
from .models_tab import ModelsTab
from .presets_store import PresetRegistry


GRID_TERMINAL_PHASES = frozenset(
    {"complete", "failed", "error", "cancelled", "canceled"}
)
GRID_DEFAULTS: dict[str, Any] = {
    "grid.adapter_dir": "",
    "grid.checkpoints": [],
    "grid.strengths": "1.0",
    "grid.texts": (
        "This is a training progress sample for the adapted voice.\n"
        "A calm, natural voice should remain clear on a sentence it never heard during training."
    ),
    "grid.references": "",
    "grid.seed": -1,
    "grid.language": "EN",
    "grid.eval_reference_mode": "",
    "grid.eval_train_subset": 48,
    "grid.eval_include_base": True,
    "grid.temperature": GENERATION_DEFAULTS["generation.temperature"],
    "grid.top_p": GENERATION_DEFAULTS["generation.top_p"],
    "grid.top_k": GENERATION_DEFAULTS["generation.top_k"],
    "grid.num_beams": GENERATION_DEFAULTS["generation.num_beams"],
    "grid.repetition_penalty": GENERATION_DEFAULTS["generation.repetition_penalty"],
    "grid.diffusion_steps": GENERATION_DEFAULTS["generation.diffusion_steps"],
    "grid.inference_cfg_rate": GENERATION_DEFAULTS["generation.inference_cfg_rate"],
    "grid.max_text_tokens_per_segment": GENERATION_DEFAULTS[
        "generation.max_text_tokens_per_segment"
    ],
    "grid.emotion_weight": GENERATION_DEFAULTS["generation.emotion_weight"],
    "grid.speaking_rate": 1.0,
}
GRID_EMPTY_HINT = (
    "Select a LoRA / DoRA folder, or use Load last values in the header."
)
EVAL_REFERENCE_CHOICES = (
    ("Same as training validation", ""),
    ("self", "self"),
    ("other (inference-like: a different clip of the same speaker)", "other"),
)
_GRID_RUNNER_EXTRAS = (
    "segment_budget_scale_non_cjk",
    "cfm_temperature",
    "reuse_spk_cond_for_emo",
    "enable_pause_tags",
    "trim_silence_ms_threshold",
    "target_duration_s",
    "target_duration_mode",
)


def _preprocess_dynamic_checkpoint_ids(
    payload: list[str | int | float] | None,
) -> list[str | int | float]:
    """Accept IDs supplied by the selected adapter's client-side choice update."""

    return list(payload or [])


def _adapter_folders(root: str | Path = ROOT / "loras") -> list[tuple[str, str]]:
    base = Path(root).expanduser().resolve()
    choices: list[tuple[str, str]] = []
    for folder in sorted((item for item in base.iterdir() if item.is_dir()), key=lambda item: item.name.lower()) if base.is_dir() else []:
        descriptors = discover_checkpoints(folder)
        if not descriptors:
            continue
        preferred = next((item for item in descriptors if item["kind"] == "final"), descriptors[0])
        try:
            full = checkpoint_descriptor(preferred["path"])
            metadata = full.get("metadata") or {}
            saved_type = str(metadata.get("adapter_type") or "").strip().lower()
            adapter_type = {
                "lora": "LoRA",
                "dora": "DoRA",
            }.get(saved_type, "LoRA / DoRA")
            rank = int(metadata.get("rank") or 0)
            steps = int(metadata.get("steps") or 0)
            label = f"{folder.name}  |  {adapter_type} r{rank}  |  {steps} steps"
        except Exception:
            label = folder.name
        choices.append((label, str(folder)))
    return choices


def _phase_label(phase: str) -> str:
    return phase_display_label(phase)


def _evaluation_reference_line(reference_mode: str | None) -> str:
    mode = str(reference_mode or "self").strip().lower()
    if mode == "other":
        label = "other (inference-like: a different clip of the same speaker)"
    else:
        label = "self"
    return f"Evaluation references used by this report: **{label}**."


def _analysis_payload(adapter_dir: str | Path | None) -> dict[str, Any]:
    if not adapter_dir:
        return {
            "summary": GRID_EMPTY_HINT,
            "reference_mode_line": "",
            "chart": analysis_epoch_frame(None),
            "rows": [],
            "choices": [],
            "selected": [],
            "mapping": {},
            "recommended": "",
        }
    root = Path(adapter_dir).expanduser().resolve()
    analysis = load_training_analysis(root)
    measured = load_checkpoint_eval(root)
    descriptors = discover_checkpoints(root)
    recommended = ""
    rows: list[list[Any]] = []
    phases_by_path: dict[str, str] = {}
    label_cache: dict[str, str] = {}
    if measured is not None:
        summary = display_legacy_report_text(measured.summary_markdown)
        reference_mode_line = _evaluation_reference_line(measured.reference_mode)
        if GENERALIZATION_LEGEND not in summary:
            summary += "\n\n" + GENERALIZATION_LEGEND
        recommended = measured.recommended_checkpoint
        for row in measured.rows:
            rows.append(
                [
                    checkpoint_display_label(
                        row.label,
                        path=row.path,
                        kind=row.kind,
                        cache=label_cache,
                    ),
                    row.epoch if row.epoch is not None else "-",
                    f"{row.val_loss:.4f}" if row.val_loss is not None else "-",
                    f"{row.val_accuracy * 100:.1f}%" if row.val_accuracy is not None else "-",
                    f"{row.train_accuracy * 100:.1f}%" if row.train_accuracy is not None else "-",
                    _phase_label(row.phase),
                    row.path,
                ]
            )
            if row.path and abs(row.strength - 1.0) < 1e-9:
                phases_by_path[str(Path(row.path).resolve())] = row.phase
    elif analysis is not None:
        summary = display_legacy_report_text(analysis.summary_markdown)
        reference_mode_line = ""
        recommended = analysis.recommended_checkpoint
        for item in analysis.checkpoints:
            path = str(Path(str(item.get("path") or "")).resolve())
            phase = str(item.get("phase") or "unknown")
            phases_by_path[path] = phase
            rows.append(
                [
                    checkpoint_display_label(
                        str(item.get("label") or Path(path).name),
                        path=path,
                        kind=str(item.get("kind") or ""),
                        cache=label_cache,
                    ),
                    item.get("epoch") or "-",
                    f"{float(item['val_loss']):.4f}" if item.get("val_loss") is not None else "-",
                    "-",
                    "-",
                    _phase_label(phase),
                    path,
                ]
            )
    else:
        reference_mode_line = ""
        summary = (
            "Run **Analyze training log** or train with validation enabled.\n\n"
            + GENERALIZATION_LEGEND
        )
    chart = analysis_epoch_frame(analysis)
    mapping: dict[str, dict[str, str]] = {
        "base": {"label": BASE_CHECKPOINT_LABEL, "path": ""}
    }
    choices: list[tuple[str, str]] = [(BASE_CHECKPOINT_CHOICE_LABEL, "base")]
    final_ids: list[str] = []
    epoch_ids: list[str] = []
    recommended_id = ""
    for index, item in enumerate(descriptors):
        path = str(Path(item["path"]).resolve())
        identifier = f"checkpoint-{index}"
        phase = phases_by_path.get(path, "unknown")
        display = f"{item['label']} - {_phase_label(phase)}"
        mapping[identifier] = {"label": str(item["label"]), "path": path}
        choices.append((display, identifier))
        if recommended and path == str(Path(recommended).resolve()):
            recommended_id = identifier
        if item["kind"] == "final":
            final_ids.append(identifier)
        if item["kind"] == "epoch":
            epoch_ids.append(identifier)
    selected = ["base"]
    if recommended_id:
        selected.append(recommended_id)
    selected.extend(identifier for identifier in final_ids if identifier not in selected)
    if len(descriptors) <= 6:
        selected.extend(identifier for identifier in epoch_ids if identifier not in selected)
    return {
        "summary": summary,
        "reference_mode_line": reference_mode_line,
        "chart": chart,
        "rows": rows,
        "choices": choices,
        "selected": selected,
        "mapping": mapping,
        "recommended": recommended,
    }


def _adapter_context(adapter_dir: str | Path | None) -> dict[str, Any]:
    if not adapter_dir:
        return {
            "info": GRID_EMPTY_HINT,
            "reference": "",
            "texts": GRID_DEFAULTS["grid.texts"],
        }
    root = Path(adapter_dir).expanduser().resolve()
    descriptors = discover_checkpoints(root)
    preferred = next((item for item in descriptors if item["kind"] == "final"), descriptors[0] if descriptors else None)
    metadata: dict[str, Any] = {}
    if preferred is not None:
        try:
            metadata = checkpoint_descriptor(preferred["path"]).get("metadata") or {}
        except Exception:
            metadata = {}
    config = read_json(root / "train_config.json", {}) or metadata.get("train_config") or {}
    reference = str(metadata.get("recommended_reference") or "")
    if reference:
        candidate = Path(reference)
        if not candidate.is_absolute():
            root_candidate = root / candidate
            if root_candidate.is_file():
                candidate = root_candidate
            elif preferred:
                candidate = Path(preferred["path"]).parent / candidate
            else:
                candidate = root_candidate
        reference = str(candidate.resolve()) if candidate.is_file() else ""
    if not reference:
        candidates = sorted(root.glob("*_reference.wav"))
        reference = str(candidates[0].resolve()) if candidates else ""
    dataset_dir = str(config.get("dataset_dir") or "")
    dataset_name = Path(dataset_dir).name if dataset_dir else str(metadata.get("dataset") or "unknown")
    analysis_available = (root / "analysis" / "training_analysis.json").is_file()
    measured_available = (root / "analysis" / "checkpoint_eval.json").is_file()
    epochs = max((int(item.get("epoch") or 0) for item in descriptors), default=0)
    steps = max((int(item.get("steps") or 0) for item in descriptors), default=0)
    info = (
        f"**{root.name}** | dataset **{dataset_name}** | {epochs} epochs | {steps} steps | "
        f"reference **{Path(reference).name if reference else 'not recorded'}** | "
        f"log analysis **{'ready' if analysis_available else 'not saved'}** | "
        f"measured comparison **{'ready' if measured_available else 'not run'}**"
    )
    sample_text = str(config.get("sample_text") or "").strip()
    neutral = "A calm, natural voice should remain clear on a sentence it never heard during training."
    texts = "\n".join(item for item in (sample_text, neutral) if item)
    return {"info": info, "reference": reference, "texts": texts}


def _same_folder(left: str | Path, right: str | Path) -> bool:
    try:
        return os.path.normcase(str(Path(left).expanduser().resolve())) == os.path.normcase(
            str(Path(right).expanduser().resolve())
        )
    except (OSError, RuntimeError, ValueError):
        return False


def _adapter_dataset_dir(adapter_dir: str | Path) -> Path | None:
    root = Path(adapter_dir).expanduser().resolve()
    config = read_json(root / "train_config.json", {}) or {}
    if not config:
        descriptors = discover_checkpoints(root)
        if descriptors:
            try:
                config = (
                    checkpoint_descriptor(descriptors[0]["path"])
                    .get("metadata", {})
                    .get("train_config", {})
                )
            except Exception:
                config = {}
    value = str(config.get("dataset_dir") or "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def calibrate_grid_speaking_rates(
    grid_dir: str | Path | None,
    adapter_dir: str | Path | None,
    recommended_checkpoint: str | Path | None = None,
) -> tuple[str, str]:
    """Measure each non-base checkpoint row in the displayed listening grid."""

    result = load_grid(grid_dir) if grid_dir else None
    if result is None:
        message = "Open a completed grid before calibrating speaking rate."
        print(">> " + message, flush=True)
        return "", message
    if not adapter_dir:
        message = "Select the LoRA / DoRA folder that produced this grid."
        print(">> " + message, flush=True)
        return "", message
    configured_adapter = str((result.config or {}).get("adapter_dir") or "")
    if not configured_adapter or not _same_folder(configured_adapter, adapter_dir):
        message = (
            "This grid does not belong to the selected LoRA / DoRA folder; "
            "select its original folder before calibrating."
        )
        print(">> " + message, flush=True)
        return "", message

    dataset_dir = _adapter_dataset_dir(adapter_dir)
    if dataset_dir is None or not (dataset_dir / "manifest.jsonl").is_file():
        message = "The selected LoRA / DoRA has no readable training dataset manifest."
        print(">> " + message, flush=True)
        return "", message

    rows: list[tuple[Any, Any]] = []
    seen: set[str] = set()
    label_cache: dict[str, str] = {}
    for cell in result.cells:
        if not cell.checkpoint_path:
            continue
        path_key = os.path.normcase(str(Path(cell.checkpoint_path).expanduser().resolve()))
        row_key = f"{path_key}@{cell.strength:g}"
        if row_key in seen:
            continue
        seen.add(row_key)
        calibration_label = cell.checkpoint_label
        if abs(float(cell.strength) - 1.0) >= 1e-9:
            calibration_label += f" @{cell.strength:g}"
        report = calibrate_from_grid(
            grid_dir,
            calibration_label,
            dataset_dir,
        )
        if report is None:
            continue
        label = checkpoint_display_label(
            cell.checkpoint_label,
            path=cell.checkpoint_path,
            kind=cell.checkpoint_kind,
            cache=label_cache,
        )
        if abs(float(cell.strength) - 1.0) >= 1e-9:
            label += f" @{cell.strength:g}"
        rows.append((cell, (label, report)))
        print(">> " + report.summary, flush=True)

    if not rows:
        message = "No usable one-second non-base checkpoint cells were found in this grid."
        print(">> " + message, flush=True)
        return "", message

    selected = rows[0]
    if recommended_checkpoint:
        matching = [
            item
            for item in rows
            if _same_folder(item[0].checkpoint_path, recommended_checkpoint)
        ]
        selected = next(
            (
                item
                for item in matching
                if abs(float(item[0].strength) - 1.0) < 1e-9
            ),
            matching[0] if matching else selected,
        )
    selected_label, selected_report = selected[1]
    saved_path = write_speaking_rate(adapter_dir, selected_report)
    lines = [
        "| Checkpoint | Generated words/s | Your recordings words/s | Recommended speaking rate |",
        "|---|---:|---:|---:|",
    ]
    for _cell, (label, report) in rows:
        escaped_label = str(label).replace("|", "\\|")
        lines.append(
            f"| {escaped_label} | "
            f"{report.generated_words_per_second:.2f} | "
            f"{report.dataset_words_per_second:.2f} | "
            f"{report.recommended_speaking_rate:.3f} |"
        )
    message = (
        f"Saved speaking rate {selected_report.recommended_speaking_rate:.3f} from "
        f"{selected_label} to {saved_path} using "
        f"{speaking_rate_method_label(selected_report.method)}."
    )
    print(">> " + message, flush=True)
    return "\n".join(lines), message


def adapter_selection_updates(
    adapter_dir: str | None, current_references: str = "", current_texts: str = "",
) -> tuple[Any, ...]:
    payload = _analysis_payload(adapter_dir)
    context = _adapter_context(adapter_dir)
    # Keep comparison inputs stable across adapters. Fill an empty form from
    # the run, while the explicit stored-reference button can replace a voice.
    references = gr.skip() if str(current_references or "").strip() else context["reference"]
    text_value = str(current_texts or "").strip()
    texts = gr.skip() if text_value and text_value != GRID_DEFAULTS["grid.texts"].strip() else context["texts"]
    return (
        context["info"],
        payload["chart"],
        payload["summary"],
        payload["reference_mode_line"],
        payload["rows"],
        gr.update(choices=payload["choices"], value=payload["selected"]),
        payload["mapping"],
        references,
        texts,
        payload["recommended"],
    )


def latest_grid_state(root: str | Path = ROOT / "outputs" / "grids") -> str:
    base = Path(root).expanduser().resolve()
    values: list[tuple[float, Path]] = []
    for status_path in base.glob("*/status.json") if base.is_dir() else []:
        try:
            values.append((status_path.stat().st_mtime, status_path.parent.resolve()))
        except OSError:
            continue
    values.sort(key=lambda item: item[0], reverse=True)
    return str(values[0][1]) if values else ""


def latest_lora_folder(root: str | Path = ROOT / "loras") -> str:
    """Return the newest LoRA / DoRA training folder by status/checkpoint mtime."""

    base = Path(root).expanduser().resolve()
    values: list[tuple[float, Path]] = []
    folders = (
        (item for item in base.iterdir() if item.is_dir())
        if base.is_dir()
        else []
    )
    for folder in folders:
        candidates = [folder / "status.json", *folder.rglob("*.safetensors")]
        modified = 0.0
        found = False
        for candidate in candidates:
            try:
                if candidate.is_file():
                    found = True
                    modified = max(modified, candidate.stat().st_mtime)
            except OSError:
                continue
        if found:
            values.append((modified, folder.resolve()))
    values.sort(key=lambda item: (item[0], str(item[1]).lower()), reverse=True)
    return str(values[0][1]) if values else ""


def latest_checkpoint_eval_state(adapter_dir: str | Path | None) -> str:
    if not adapter_dir:
        return ""
    analysis_dir = Path(adapter_dir).expanduser().resolve() / "analysis"
    candidates = list(analysis_dir.glob("eval_jobs/*/status.json"))
    legacy = analysis_dir / "eval_job" / "status.json"
    if legacy.is_file():
        candidates.append(legacy)
    values: list[tuple[float, Path]] = []
    for status_path in candidates:
        try:
            values.append((status_path.stat().st_mtime, status_path.parent.resolve()))
        except OSError:
            continue
    values.sort(key=lambda item: (item[0], str(item[1]).lower()), reverse=True)
    return str(values[0][1]) if values else ""


def _grid_running(path: str | Path | None) -> bool:
    status = read_json(Path(path) / "status.json", {}) if path else {}
    phase = str((status or {}).get("phase") or "").lower()
    return bool(phase and phase not in GRID_TERMINAL_PHASES)


def adopt_grid_state(
    displayed: str | Path | None,
    *,
    root: str | Path = ROOT / "outputs" / "grids",
    page_load: bool = False,
) -> tuple[str, bool]:
    current = str(Path(displayed).expanduser().resolve()) if displayed else ""
    if current and _grid_running(current):
        return current, True
    newest = latest_grid_state(root)
    if page_load and newest:
        return newest, _grid_running(newest)
    if newest and _grid_running(newest):
        return newest, True
    return current, False


def _grid_rows(grid_dir: str | Path | None) -> list[list[Any]]:
    result = load_grid(grid_dir) if grid_dir else None
    if result is None:
        return []
    label_cache: dict[str, str] = {}
    return [
        [
            cell.index,
            checkpoint_display_label(
                cell.checkpoint_label,
                path=cell.checkpoint_path,
                kind=cell.checkpoint_kind,
                cache=label_cache,
            ),
            None if not cell.checkpoint_path else cell.strength,
            cell.reference_index,
            cell.text,
            round(cell.audio_seconds, 2),
            _phase_label(cell.verdict),
            cell.audio_path,
        ]
        for cell in result.cells
    ]


def _renderable_grid_cells(result: Any) -> list[Any]:
    """Return only cells whose audio is ready for a Gradio player."""
    if result is None:
        return []
    return [
        cell
        for cell in result.cells
        if cell.audio_path and Path(cell.audio_path).is_file()
    ]


def _grid_result_heading(
    cell: Any, label_cache: dict[str, str] | None = None
) -> str:
    display_label = checkpoint_display_label(
        cell.checkpoint_label,
        path=cell.checkpoint_path,
        kind=cell.checkpoint_kind,
        cache=label_cache,
    )
    if not cell.checkpoint_path:
        return f"#### {BASE_CHECKPOINT_LABEL} | {BASE_GRID_HEADER_DETAIL}"
    loss = f" | validation loss {cell.val_loss:.4f}" if cell.val_loss is not None else ""
    return (
        f"#### {display_label} @ {cell.strength:g} | "
        f"{_phase_label(cell.verdict)}{loss}"
    )


def _saved_grid_choices(root: str | Path = ROOT / "outputs" / "grids") -> list[tuple[str, str]]:
    return [
        (
            f"{item.grid_name} | {item.cells} cells | {item.status} | seed {item.seed}",
            item.grid_dir,
        )
        for item in list_grids(root)
    ]


def grid_status_updates(
    state_value: str,
    *,
    output_root: str | Path = ROOT / "outputs" / "grids",
    page_load: bool = False,
) -> tuple[Any, ...]:
    state, running = adopt_grid_state(state_value, root=output_root, page_load=page_load)
    if not state:
        return (
            "",
            progress_panel_html({}, title="Ready"),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.update(choices=_saved_grid_choices(output_root)),
            gr.Timer(5.0, active=True),
        )
    root = Path(state)
    status = read_json(root / "status.json", {}) or {}
    progress = read_json(root / "progress.json", {}) or {}
    if not progress:
        progress = dict(status)
        completed = int(progress.get("completed", 0) or 0)
        total = int(progress.get("total", 0) or 0)
        progress["fraction"] = completed / total if total else 0.0
    phase = str(status.get("phase") or "initializing").lower()
    title = {
        "complete": "Listening grid complete",
        "failed": "Listening grid failed",
        "cancelled": "Listening grid canceled",
        "canceled": "Listening grid canceled",
    }.get(phase, "Generating listening grid")
    message = str(status.get("message") or phase.replace("_", " ").title())
    if running:
        message = f"Attached to running grid {root.name} | {message}"
    result_state = (
        state
        if phase in GRID_TERMINAL_PHASES and (root / "grid.json").is_file()
        else ""
    )
    return (
        state,
        progress_panel_html(progress or status, title=title),
        message,
        tail_text(root / "log.txt", 60) or tail_text(root / "worker_console.log", 60),
        result_state,
        _grid_rows(state),
        gr.update(choices=_saved_grid_choices(output_root), value=result_state or None),
        gr.Timer(1.0 if running else 5.0, active=True),
    )


def checkpoint_eval_status_updates(
    state_value: str,
    adapter_dir: str,
    selected_checkpoints: Sequence[str] | None = None,
) -> tuple[Any, ...]:
    if not state_value:
        return (
            progress_panel_html({}, title="Ready"),
            "",
            "",
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.Timer(5.0, active=True),
        )
    root = Path(state_value)
    status = read_json(root / "status.json", {}) or {}
    progress = read_json(root / "progress.json", {}) or {}
    phase = str(status.get("phase") or "initializing").lower()
    terminal = phase in GRID_TERMINAL_PHASES
    payload = _analysis_payload(adapter_dir) if terminal else None
    checkpoint_update: Any = gr.skip()
    if payload is not None:
        valid_ids = {identifier for _label, identifier in payload["choices"]}
        preserved = [
            identifier
            for identifier in (selected_checkpoints or [])
            if identifier in valid_ids
        ]
        checkpoint_update = gr.update(
            choices=payload["choices"],
            value=preserved or payload["selected"],
        )
    return (
        progress_panel_html(progress or status, title="Checkpoint evaluation"),
        str(status.get("message") or phase.replace("_", " ").title()),
        tail_text(root / "log.txt", 60) or tail_text(root / "worker_console.log", 60),
        _adapter_context(adapter_dir)["info"] if payload else gr.skip(),
        payload["summary"] if payload else gr.skip(),
        payload["reference_mode_line"] if payload else gr.skip(),
        payload["rows"] if payload else gr.skip(),
        checkpoint_update,
        payload["mapping"] if payload else gr.skip(),
        payload["recommended"] if payload else gr.skip(),
        gr.Timer(5.0 if terminal else 1.0, active=True),
    )


def _parse_strengths(value: str) -> list[float]:
    pieces = [piece.strip() for piece in str(value or "").replace(";", ",").split(",")]
    strengths: list[float] = []
    for piece in pieces:
        if not piece:
            continue
        item = float(piece)
        if not 0.0 <= item <= 4.0:
            raise ValueError("Strengths must be comma-separated values from 0 to 4")
        if item not in strengths:
            strengths.append(item)
    if not strengths:
        raise ValueError("Enter at least one LoRA / DoRA strength")
    return strengths


@dataclass
class GridTab:
    controls: dict[str, Any] = field(default_factory=dict)
    model_dir: str = ""
    adapter: Any = None
    adapter_info: Any = None
    checkpoint_group: Any = None
    checkpoint_map: Any = None
    recommended: Any = None
    summary: Any = None
    eval_reference_summary: Any = None
    chart: Any = None
    checkpoint_table: Any = None
    analyze_button: Any = None
    evaluate_button: Any = None
    use_generation: Any = None
    eval_state: Any = None
    eval_progress: Any = None
    eval_status: Any = None
    eval_log: Any = None
    eval_timer: Any = None
    generate_button: Any = None
    cancel_button: Any = None
    open_button: Any = None
    calibrate_button: Any = None
    calibration_result: Any = None
    runtime_summary: Any = None
    state: Any = None
    progress: Any = None
    status: Any = None
    log: Any = None
    timer: Any = None
    result_state: Any = None
    result_table: Any = None
    saved_grids: Any = None
    selection_outputs: list[Any] = field(default_factory=list)


def _register(
    registry: PresetRegistry,
    controls: dict[str, Any],
    key: str,
    component: Any,
    *,
    kind: str = "auto",
    choices: Sequence[Any] | None = None,
    minimum: float | int | None = None,
    maximum: float | int | None = None,
) -> None:
    registry.register(
        key,
        component,
        GRID_DEFAULTS[key],
        kind=kind,
        choices=choices,
        minimum=minimum,
        maximum=maximum,
    )
    controls[key] = component


def build_grid_tab(
    options: Any,
    registry: PresetRegistry,
    *,
    load_hook: Any | None = None,
) -> GridTab:
    tab = GridTab()
    tab.model_dir = str(Path(getattr(options, "model_dir", ROOT / "models")).expanduser().resolve())
    controls = tab.controls
    adapter_choices = _adapter_folders()
    initial_adapter = ""
    initial_payload = _analysis_payload(initial_adapter)
    initial_context = _adapter_context(initial_adapter)

    with gr.Tab("Checkpoint Grid", id="checkpoint-grid"):
        with gr.Row():
            tab.adapter = gr.Dropdown(
                choices=adapter_choices,
                value=None,
                allow_custom_value=True,
                label="LoRA / DoRA folder",
                info="Choose one training run to analyze and compare.",
                scale=10,
            )
            refresh_adapter = gr.Button("↻  Refresh", elem_classes=btn("violet"), scale=1)
        tab.adapter_info = gr.Markdown(initial_context["info"])
        # The adapter selection is session state, not a preset value: a preset load must not
        # blank the dropdown or point it at a folder from another machine.
        registry.register(
            "grid.adapter_dir", tab.adapter, GRID_DEFAULTS["grid.adapter_dir"], kind="str", preset=False
        )
        controls["grid.adapter_dir"] = tab.adapter

        with gr.Accordion("Which checkpoint generalizes best?", open=True):
            tab.chart = gr.LinePlot(
                initial_payload["chart"],
                x="step",
                y="value",
                color="series",
                title="Training and unseen-sentence loss by epoch",
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
            tab.summary = gr.Markdown(initial_payload["summary"])
            tab.eval_reference_summary = gr.Markdown(
                initial_payload["reference_mode_line"],
                elem_classes=["section-note"],
            )
            tab.checkpoint_table = gr.Dataframe(
                headers=[
                    "Checkpoint",
                    "Epoch",
                    "Validation loss (lower = better)",
                    "Unseen-text accuracy",
                    "Training-text accuracy",
                    "Verdict",
                    "Path",
                ],
                datatype=["str"] * 7,
                value=initial_payload["rows"],
                type="array",
                interactive=False,
                wrap=True,
                label="Checkpoints",
                max_height=360,
                buttons=["fullscreen", "copy"],
            )
            with gr.Row():
                tab.analyze_button = gr.Button("🔬  Analyze training log", elem_classes=btn("orange"))
                tab.evaluate_button = gr.Button("📈  Evaluate checkpoints now", variant="primary", elem_classes=btn("purple"))
                tab.use_generation = gr.Button("⭐  Use best checkpoint", elem_classes=btn("green"))
            with gr.Row():
                eval_reference_mode = gr.Dropdown(
                    choices=list(EVAL_REFERENCE_CHOICES),
                    value=GRID_DEFAULTS["grid.eval_reference_mode"],
                    label="Evaluation references",
                    info="Same as training validation inherits that run's validation mode; other uses a different same-speaker clip like inference.",
                )
                eval_train_subset = gr.Number(
                    value=GRID_DEFAULTS["grid.eval_train_subset"],
                    minimum=0,
                    maximum=100000,
                    precision=0,
                    label="Training subset",
                    info="Deterministic training items measured beside validation; 0 disables the training subset.",
                )
                eval_include_base = gr.Checkbox(
                    value=GRID_DEFAULTS["grid.eval_include_base"],
                    label="Include base model",
                    info="Adds Base model (no LoRA / DoRA), the reference-only comparison baseline.",
                )
            for key, component, kind, choices, minimum, maximum in (
                ("grid.eval_reference_mode", eval_reference_mode, "choice", ["", "self", "other"], None, None),
                ("grid.eval_train_subset", eval_train_subset, "int", None, 0, 100000),
                ("grid.eval_include_base", eval_include_base, "bool", None, None, None),
            ):
                _register(
                    registry,
                    controls,
                    key,
                    component,
                    kind=kind,
                    choices=choices,
                    minimum=minimum,
                    maximum=maximum,
                )
            tab.eval_state = gr.State("")
            tab.eval_progress = gr.HTML(progress_panel_html({}, title="Ready"))
            tab.eval_status = gr.Markdown("")
            tab.eval_log = gr.Textbox(
                label="Evaluation log (last 60 lines)",
                lines=8,
                max_lines=14,
                interactive=False,
                buttons=["copy"],
                elem_classes=["log-tail"],
            )
            tab.eval_timer = gr.Timer(5.0, active=True)

        gr.Markdown("### Grid setup")
        tab.checkpoint_group = gr.CheckboxGroup(
            choices=initial_payload["choices"],
            value=initial_payload["selected"],
            label="Checkpoints",
            info="Base model (no LoRA / DoRA), the recommended checkpoint, and the final checkpoint are selected first.",
        )
        # Gradio preprocesses against the Python component's startup choices, but this
        # list is intentionally filled later when an adapter is selected.  The mapping
        # used to build a grid remains the authority and ignores unknown identifiers.
        tab.checkpoint_group.preprocess = _preprocess_dynamic_checkpoint_ids  # type: ignore[method-assign]
        tab.checkpoint_map = gr.State(initial_payload["mapping"])
        tab.recommended = gr.State(initial_payload["recommended"])
        # Checkpoint identifiers are rebuilt for every adapter, so they stay out of presets too.
        registry.register(
            "grid.checkpoints", tab.checkpoint_group, GRID_DEFAULTS["grid.checkpoints"], kind="list", preset=False
        )
        controls["grid.checkpoints"] = tab.checkpoint_group
        with gr.Row():
            strengths = gr.Textbox(
                value="1.0",
                label="Strengths",
                info="Comma-separated LoRA / DoRA strengths from 0 to 4; Base model (no LoRA / DoRA) is generated once without a strength.",
            )
            seed = gr.Number(
                value=-1,
                precision=0,
                label="Seed",
                info="-1 draws one random seed at grid start and then fixes it across cells.",
            )
            language = gr.Dropdown(
                choices=list(LANGUAGES),
                value="EN",
                label="Language",
                info="Use the language of every sentence in this grid.",
            )
        texts = gr.Textbox(
            value=initial_context["texts"],
            label="Texts (one per line)",
            lines=5,
            max_lines=12,
            info="Each non-empty line is generated for every selected row and reference.",
        )
        references = gr.Textbox(
            value=initial_context["reference"],
            label="Reference audio paths (one per line)",
            lines=3,
            max_lines=10,
            info="Every checkpoint uses these same references in the same order.",
        )
        with gr.Row():
            use_reference = gr.Button("🎤  Use LoRA / DoRA reference", elem_classes=btn("fuchsia"))
            add_candidates = gr.Button("➕  Add dataset reference candidates", elem_classes=btn("lime"))
            reference_upload = gr.Audio(
                label="Add a reference audio file",
                type="filepath",
                sources=["upload"],
            )
        for key, component, kind, choices, minimum, maximum in (
            ("grid.strengths", strengths, "str", None, None, None),
            ("grid.texts", texts, "str", None, None, None),
            ("grid.references", references, "str", None, None, None),
            ("grid.seed", seed, "int", None, -1, 4294967295),
            ("grid.language", language, "choice", list(LANGUAGES), None, None),
        ):
            _register(
                registry,
                controls,
                key,
                component,
                kind=kind,
                choices=choices,
                minimum=minimum,
                maximum=maximum,
            )

        with gr.Accordion("Sampling", open=False):
            with gr.Row():
                temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature", info="0.8 matches Voice Generation defaults.")
                top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top-p", info="Nucleus sampling threshold shared by every cell.")
                top_k = gr.Slider(0, 100, value=30, step=1, label="Top-k", info="Token candidate cutoff; 0 disables it.")
                num_beams = gr.Slider(1, 10, value=GRID_DEFAULTS["grid.num_beams"], step=1, label="Beams", info="Beam count strongly affects quality as well as generation time and VRAM.")
                repetition = gr.Slider(1, 20, value=10.0, step=0.1, label="Repetition penalty", info="Keeps semantic-token loops under control.")
            with gr.Row():
                diffusion = gr.Slider(2, 100, value=25, step=1, label="Diffusion steps", info="25 is the quality default.")
                cfg_rate = gr.Slider(0, 2, value=0.7, step=0.05, label="CFG rate", info="Diffusion conditioning strength.")
                max_tokens = gr.Slider(20, 300, value=60, step=1, label="Max text tokens per segment", info="Use the same segment size for a fair comparison.")
                emotion_weight = gr.Slider(0, 1, value=0.65, step=0.05, label="Emotion weight (alpha)", info="Shared emotion-conditioning blend for every cell.")
                speaking_rate = gr.Slider(
                    0.5,
                    1.5,
                    value=GRID_DEFAULTS["grid.speaking_rate"],
                    step=0.01,
                    label="Speaking rate",
                    info="1.0 is the model's natural pace; below 1.0 speaks slower, above 1.0 faster. A trained LoRA / DoRA can carry a calibrated value that matches the speaker's real pace.",
                )
            for key, component, kind, minimum, maximum in (
                ("grid.temperature", temperature, "float", 0.1, 2),
                ("grid.top_p", top_p, "float", 0, 1),
                ("grid.top_k", top_k, "int", 0, 100),
                ("grid.num_beams", num_beams, "int", 1, 10),
                ("grid.repetition_penalty", repetition, "float", 1, 20),
                ("grid.diffusion_steps", diffusion, "int", 2, 100),
                ("grid.inference_cfg_rate", cfg_rate, "float", 0, 2),
                ("grid.max_text_tokens_per_segment", max_tokens, "int", 20, 300),
                ("grid.emotion_weight", emotion_weight, "float", 0, 1),
                ("grid.speaking_rate", speaking_rate, "float", 0.5, 1.5),
            ):
                _register(registry, controls, key, component, kind=kind, minimum=minimum, maximum=maximum)

        tab.runtime_summary = gr.Markdown("Runtime is resolved from Models & Performance when the grid starts.")
        with gr.Row():
            tab.generate_button = gr.Button(
                "🧩  Generate grid", variant="primary", elem_classes=btn("emerald")
            )
            tab.cancel_button = gr.Button("⛔  Cancel", variant="stop", elem_classes=btn("red"))
            tab.open_button = gr.Button("📁  Open grid folder", elem_classes=btn("indigo"))
        tab.calibrate_button = gr.Button(
            "🐢  Calibrate speaking rate from this grid",
            elem_classes=btn("teal"),
        )
        tab.state = gr.State("")
        tab.progress = gr.HTML(progress_panel_html({}, title="Ready"))
        tab.status = gr.Markdown("")
        tab.calibration_result = gr.Markdown("")
        tab.log = gr.Textbox(
            label="Grid log (last 60 lines)",
            lines=10,
            max_lines=16,
            interactive=False,
            buttons=["copy"],
            elem_classes=["log-tail"],
        )
        tab.timer = gr.Timer(5.0, active=True)

        gr.Markdown("### Results")
        tab.saved_grids = gr.Dropdown(
            choices=_saved_grid_choices(),
            label="Saved grids",
            info="Open any earlier listening grid without regenerating it.",
        )
        # A hidden textbox drives the dynamic audio-player render.  Source events are
        # queued below so Gradio dispatches this component's change trigger reliably.
        tab.result_state = gr.Textbox(value="", visible=False, label="Grid result folder")
        with gr.Column(elem_classes=["grid-results"]):
            @gr.render(inputs=tab.result_state, triggers=[tab.result_state.change])
            def render_grid(result_dir: str | None):
                result = load_grid(result_dir) if result_dir else None
                cells = _renderable_grid_cells(result)
                if not cells:
                    return
                label_cache: dict[str, str] = {}
                groups: list[tuple[tuple[str, float], list[Any]]] = []
                for cell in cells:
                    key = (cell.checkpoint_label, cell.strength)
                    match = next((item for item in groups if item[0] == key), None)
                    if match is None:
                        match = (key, [])
                        groups.append(match)
                    match[1].append(cell)
                for group_index, ((_label, _strength), cells_value) in enumerate(groups):
                    first = cells_value[0]
                    gr.Markdown(_grid_result_heading(first, label_cache))
                    for start in range(0, len(cells_value), 4):
                        with gr.Row():
                            for cell in cells_value[start : start + 4]:
                                with gr.Column(min_width=220):
                                    gr.Markdown(cell.text)
                                    gr.Audio(
                                        value=cell.audio_path,
                                        label=f"ref {cell.reference_index} | text {cell.text_index}",
                                        type="filepath",
                                        buttons=["download"],
                                        key=f"grid-{group_index}-{cell.index}-{cell.audio_path}",
                                    )
        tab.result_table = gr.Dataframe(
            headers=["Row", "Checkpoint", "Strength", "Reference", "Text", "Seconds", "Verdict", "File"],
            datatype=["number", "str", "number", "number", "str", "number", "str", "str"],
            value=[],
            type="array",
            interactive=False,
            wrap=True,
            label="Grid cells",
            max_height=420,
            buttons=["fullscreen", "copy"],
        )

    tab.selection_outputs = [
        tab.adapter_info,
        tab.chart,
        tab.summary,
        tab.eval_reference_summary,
        tab.checkpoint_table,
        tab.checkpoint_group,
        tab.checkpoint_map,
        references,
        texts,
        tab.recommended,
    ]
    refresh_adapter.click(
        lambda: gr.update(choices=_adapter_folders()), outputs=tab.adapter, queue=False
    )
    tab.adapter.change(
        adapter_selection_updates,
        [tab.adapter, references, texts],
        tab.selection_outputs,
        queue=False,
    )

    def analyze_now(adapter_dir: str, current_references: str, current_texts: str):
        if not adapter_dir:
            raise gr.Error("Select a LoRA / DoRA folder first")
        try:
            analysis = analyze_training_run(adapter_dir)
            write_training_analysis(analysis)
            return adapter_selection_updates(adapter_dir, current_references, current_texts)
        except Exception as exc:
            raise gr.Error(f"Training-log analysis failed: {exc}") from exc

    tab.analyze_button.click(
        analyze_now,
        [tab.adapter, references, texts],
        tab.selection_outputs,
        queue=False,
        api_name="analyze_training_log",
    )
    use_reference.click(
        lambda adapter_dir: _adapter_context(adapter_dir)["reference"],
        tab.adapter,
        references,
        queue=False,
    )

    def append_dataset_candidates(adapter_dir: str, current: str) -> str:
        if not adapter_dir:
            return current
        root = Path(adapter_dir)
        descriptors = discover_checkpoints(root)
        config = read_json(root / "train_config.json", {}) or {}
        if not config and descriptors:
            try:
                config = checkpoint_descriptor(descriptors[0]["path"])["metadata"].get("train_config") or {}
            except Exception:
                config = {}
        dataset_dir = Path(str(config.get("dataset_dir") or ""))
        info = read_json(dataset_dir / "dataset_info.json", {}) or {}
        values = parse_multiline_paths(current)
        for raw in info.get("reference_candidates", []):
            path = Path(str(raw))
            if not path.is_absolute():
                path = dataset_dir / path
            if path.is_file() and str(path.resolve()) not in values:
                values.append(str(path.resolve()))
        return "\n".join(values)

    add_candidates.click(
        append_dataset_candidates, [tab.adapter, references], references, queue=False
    )
    reference_upload.change(
        lambda upload, current: "\n".join(
            parse_multiline_paths(current)
            + ([str(Path(upload).resolve())] if upload and str(Path(upload).resolve()) not in parse_multiline_paths(current) else [])
        ),
        [reference_upload, references],
        references,
        queue=False,
    )
    # Completion polling also updates this dropdown.  Listen only to direct user
    # input so that a completed grid cannot trigger two overlapping renders.
    tab.saved_grids.input(
        lambda path: (path or "", _grid_rows(path), f"Opened saved grid {Path(path).name}." if path else "Select a saved grid."),
        tab.saved_grids,
        [tab.result_state, tab.result_table, tab.status],
        queue=True,
    )
    tab.open_button.click(
        lambda result, state: open_folder(result or state or ROOT / "outputs" / "grids"),
        [tab.result_state, tab.state],
        tab.status,
        queue=False,
    )
    tab.calibrate_button.click(
        calibrate_grid_speaking_rates,
        [tab.result_state, tab.adapter, tab.recommended],
        [tab.calibration_result, tab.status],
        queue=False,
        api_name="calibrate_grid_speaking_rate",
    )
    grid_poll_outputs = [
        tab.state,
        tab.progress,
        tab.status,
        tab.log,
        tab.result_state,
        tab.result_table,
        tab.saved_grids,
        tab.timer,
    ]
    tab.timer.tick(
        grid_status_updates,
        tab.state,
        grid_poll_outputs,
        queue=True,
        show_progress="hidden",
    )
    tab.eval_timer.tick(
        checkpoint_eval_status_updates,
        [tab.eval_state, tab.adapter, tab.checkpoint_group],
        [
            tab.eval_progress,
            tab.eval_status,
            tab.eval_log,
            tab.adapter_info,
            tab.summary,
            tab.eval_reference_summary,
            tab.checkpoint_table,
            tab.checkpoint_group,
            tab.checkpoint_map,
            tab.recommended,
            tab.eval_timer,
        ],
        queue=False,
        show_progress="hidden",
    )
    if load_hook is not None:
        load_hook(
            lambda state: grid_status_updates(state, page_load=True),
            tab.state,
            grid_poll_outputs,
            queue=False,
            show_progress="hidden",
            api_name="attach_checkpoint_grid",
        )

        def load_last_checkpoint_values(current_references: str, current_texts: str):
            adapter_dir = latest_lora_folder()
            selection = adapter_selection_updates(adapter_dir or None, current_references, current_texts)
            eval_state = latest_checkpoint_eval_state(adapter_dir)
            eval_updates = checkpoint_eval_status_updates(
                eval_state,
                adapter_dir,
                _analysis_payload(adapter_dir)["selected"] if adapter_dir else [],
            )
            return (
                gr.update(
                    choices=_adapter_folders(),
                    value=adapter_dir or None,
                ),
                *selection,
                eval_state,
                eval_updates[0],
                eval_updates[1],
                eval_updates[2],
                eval_updates[-1],
            )

        load_hook(
            load_last_checkpoint_values,
            inputs=[references, texts],
            outputs=[
                tab.adapter,
                *tab.selection_outputs,
                tab.eval_state,
                tab.eval_progress,
                tab.eval_status,
                tab.eval_log,
                tab.eval_timer,
            ],
            queue=False,
            show_progress="hidden",
            api_name="load_last_checkpoint_analysis",
        )
    return tab


def build_grid_config_from_ui(
    mapping: Mapping[str, Any],
    grid_values: Mapping[str, Any],
    generation_values: Mapping[str, Any],
    *,
    model_dir: str,
    output_root: str | Path,
    grid_name: str,
) -> GridConfig:
    """Build a worker config from the live front-end generation and grid controls."""

    adapter_dir = str(grid_values.get("grid.adapter_dir") or "")
    checkpoints: list[GridCheckpoint] = []
    for identifier in list(grid_values.get("grid.checkpoints") or []):
        item = dict((mapping or {}).get(identifier) or {})
        if item:
            checkpoints.append(
                GridCheckpoint(item.get("label", identifier), item.get("path", ""))
            )

    generation_request = build_generation_request(
        generation_values,
        model_dir=model_dir,
    )
    runtime = dict(generation_request["runtime"])
    runtime["lora_path"] = ""
    runtime["lora_strength"] = 1.0
    runtime["lora_merge_into_base"] = False

    infer_kwargs = dict(generation_request["infer_kwargs"])
    grid_speaking_rate = min(
        1.5,
        max(0.5, float(grid_values.get("grid.speaking_rate", 1.0))),
    )
    infer_kwargs["latent_multiplier"] = round(
        float(
            generation_values.get(
                "generation.latent_multiplier",
                GENERATION_DEFAULTS["generation.latent_multiplier"],
            )
        )
        / grid_speaking_rate,
        4,
    )
    for key in _GRID_RUNNER_EXTRAS:
        infer_kwargs[key] = generation_request[key]
    top_k = int(grid_values["grid.top_k"])
    infer_kwargs.update(
        {
            "temperature": float(grid_values["grid.temperature"]),
            "top_p": float(grid_values["grid.top_p"]),
            "top_k": top_k if top_k > 0 else None,
            "num_beams": int(grid_values["grid.num_beams"]),
            "repetition_penalty": float(grid_values["grid.repetition_penalty"]),
            "diffusion_steps": int(grid_values["grid.diffusion_steps"]),
            "inference_cfg_rate": float(grid_values["grid.inference_cfg_rate"]),
            "max_text_tokens_per_segment": int(
                grid_values["grid.max_text_tokens_per_segment"]
            ),
            "emo_alpha": float(grid_values["grid.emotion_weight"]),
        }
    )
    return GridConfig(
        adapter_dir=adapter_dir,
        checkpoints=checkpoints,
        strengths=_parse_strengths(str(grid_values["grid.strengths"])),
        references=parse_multiline_paths(str(grid_values["grid.references"])),
        texts=[
            line.strip()
            for line in str(grid_values["grid.texts"]).splitlines()
            if line.strip()
        ],
        language=str(grid_values["grid.language"]),
        seed=int(grid_values["grid.seed"]),
        same_seed_for_all_cells=True,
        output_root=str(Path(output_root).expanduser().resolve()),
        grid_name=grid_name,
        runtime=runtime,
        infer_kwargs=infer_kwargs,
        include_verdicts=True,
    ).validate()


def bind_grid_events(
    tab: GridTab,
    training: Any,
    generation: GenerationTab,
    models: ModelsTab,
    main_tabs: Any,
) -> None:
    runtime_specs = [
        spec
        for spec in getattr(models, "_registry_specs", [])
        if getattr(spec, "component", None) is not None
    ]
    if not runtime_specs:
        runtime_components = list(models.controls.values())
        runtime_keys = list(models.controls)
    else:
        runtime_components = [spec.component for spec in runtime_specs]
        runtime_keys = [spec.key for spec in runtime_specs]

    def runtime_values(*items: Any) -> dict[str, Any]:
        return dict(zip(runtime_keys, items))

    def runtime_line(
        selected: list[str], mapping: Mapping[str, Any], strengths_text: str,
        references_text: str, texts_text: str, *items: Any
    ) -> str:
        try:
            strengths = _parse_strengths(strengths_text)
        except Exception:
            strengths = [1.0]
        references_count = len(parse_multiline_paths(references_text))
        texts_count = len([line for line in str(texts_text or "").splitlines() if line.strip()])
        adapter_rows = sum(bool((mapping.get(item) or {}).get("path")) for item in selected or [])
        base_rows = sum(not bool((mapping.get(item) or {}).get("path")) for item in selected or [])
        cells = (adapter_rows * len(strengths) + base_rows) * references_count * texts_count
        values = runtime_values(*items)
        runtime = runtime_config_from_values(values, model_dir=tab.model_dir)
        seconds = cells * 9
        return (
            f"Resolved runtime: **{runtime.get('device', 'auto')}**, tier **{runtime.get('vram_tier', 'auto')}** | "
            f"**{cells} cells** | rough estimate on a 32 GB tier: **{seconds // 60}m {seconds % 60}s**"
        )

    runtime_line_inputs = [
        tab.checkpoint_group,
        tab.checkpoint_map,
        tab.controls["grid.strengths"],
        tab.controls["grid.references"],
        tab.controls["grid.texts"],
        *runtime_components,
    ]
    summary_triggers = [
        tab.checkpoint_group,
        tab.controls["grid.strengths"],
        tab.controls["grid.references"],
        tab.controls["grid.texts"],
        *runtime_components,
    ]
    seen_triggers: set[int] = set()
    for component in summary_triggers:
        if id(component) in seen_triggers:
            continue
        seen_triggers.add(id(component))
        component.change(
            runtime_line,
            runtime_line_inputs,
            tab.runtime_summary,
            queue=False,
            show_progress="hidden",
        )

    def start_evaluation(
        adapter_dir: str,
        reference_mode: str,
        train_subset: int,
        include_base: bool,
        strengths_text: str,
        *items: Any,
    ):
        if not adapter_dir:
            raise gr.Error("Select a LoRA / DoRA folder first")
        values = runtime_values(*items)
        runtime = runtime_config_from_values(values)
        device = str(runtime.get("device") or "auto")
        if device == "auto":
            device = "cuda:0" if list_gpus() else "cpu"
        job_dir = Path(adapter_dir) / "analysis" / "eval_jobs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        job_dir.mkdir(parents=True, exist_ok=False)
        config = CheckpointEvalConfig(
            adapter_dir=adapter_dir,
            device=device,
            reference_mode=str(reference_mode or ""),
            train_subset=int(train_subset),
            include_base=bool(include_base),
            strengths=_parse_strengths(strengths_text),
        )
        config_path = write_json_atomic(job_dir / "config.json", config.to_dict())
        write_json_atomic(
            job_dir / "status.json",
            {"phase": "initializing", "message": "Starting checkpoint evaluation", "completed": 0, "total": 0, "updated_at": time.time()},
        )
        PROCESS_MANAGER.start(
            "checkpoint_eval",
            [sys.executable, "-m", "indextts.training.eval_worker", "--config", str(config_path), "--state-dir", str(job_dir)],
            state_dir=job_dir,
            log_path=job_dir / "worker_console.log",
            cwd=ROOT,
            metadata={"adapter_dir": adapter_dir},
        )
        return (
            str(job_dir),
            progress_panel_html({}, title="Starting checkpoint evaluation"),
            "Checkpoint evaluation started.",
            "",
            gr.Timer(1.0, active=True),
        )

    tab.evaluate_button.click(
        start_evaluation,
        [
            tab.adapter,
            tab.controls["grid.eval_reference_mode"],
            tab.controls["grid.eval_train_subset"],
            tab.controls["grid.eval_include_base"],
            tab.controls["grid.strengths"],
            *runtime_components,
        ],
        [tab.eval_state, tab.eval_progress, tab.eval_status, tab.eval_log, tab.eval_timer],
        concurrency_limit=1,
        concurrency_id="checkpoint_eval",
    )

    grid_keys = list(tab.controls)
    grid_components = [tab.controls[key] for key in grid_keys]
    generation_keys = generation.request_keys
    generation_components = generation.request_components

    def start_grid(
        mapping: Mapping[str, Any], *items: Any
    ):
        grid_values = dict(zip(grid_keys, items[: len(grid_keys)]))
        generation_values = dict(
            zip(generation_keys, items[len(grid_keys) :])
        )
        adapter_dir = str(grid_values.get("grid.adapter_dir") or "")
        output_root = ROOT / "outputs" / "grids"
        output_root.mkdir(parents=True, exist_ok=True)
        name = f"{Path(adapter_dir).name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        state_dir = output_root / name
        config = build_grid_config_from_ui(
            mapping,
            grid_values,
            generation_values,
            model_dir=tab.model_dir,
            output_root=output_root,
            grid_name=name,
        )
        state_dir.mkdir(parents=True, exist_ok=False)
        config_path = write_json_atomic(state_dir / "config.json", config.to_dict())
        write_json_atomic(
            state_dir / "status.json",
            {"phase": "initializing", "message": "Starting listening grid worker", "completed": 0, "total": 0, "updated_at": time.time()},
        )
        PROCESS_MANAGER.start(
            "grid_generation",
            [sys.executable, "-m", "indextts.training.grid_worker", "--config", str(config_path), "--state-dir", str(state_dir)],
            state_dir=state_dir,
            log_path=state_dir / "worker_console.log",
            cwd=ROOT,
            metadata={"adapter_dir": adapter_dir},
        )
        return (
            str(state_dir),
            progress_panel_html({}, title="Starting listening grid"),
            "Listening grid started.",
            "",
            gr.Timer(1.0, active=True),
        )

    tab.generate_button.click(
        start_grid,
        [tab.checkpoint_map, *grid_components, *generation_components],
        [tab.state, tab.progress, tab.status, tab.log, tab.timer],
        concurrency_limit=1,
        concurrency_id="grid_generation",
        api_name="generate_checkpoint_grid",
    )
    with gr.Group(visible=False):
        cancel_confirm = gr.Checkbox(value=False, visible=False, label="Grid cancellation confirmation")

    def cancel_grid(confirmed: bool, state_value: str):
        if not confirmed:
            return "Cancellation dismissed."
        if not state_value or not _grid_running(state_value):
            return "No active grid is displayed."
        (Path(state_value) / "stop.flag").touch()
        return "Cancellation requested. The current cell will finish, then the grid will stop."

    tab.cancel_button.click(
        cancel_grid,
        [cancel_confirm, tab.state],
        tab.status,
        js="(confirmed, state) => [window.confirm('Cancel after the current grid cell finishes?'), state]",
        queue=False,
    )

    lora_component = generation.controls.get("runtime.lora_path")
    if lora_component is not None:
        def use_recommended(path: str):
            if not path or not Path(path).is_file():
                raise gr.Error("No recommended checkpoint is available")
            return (
                gr.update(choices=_lora_choices(), value=str(Path(path).resolve())),
                gr.Tabs(selected="voice-generation"),
            )

        tab.use_generation.click(
            use_recommended,
            tab.recommended,
            [lora_component, main_tabs],
            queue=False,
        )

    compare_button = getattr(training, "compare_grid", None)
    if compare_button is not None:
        def compare_training_run(state_value: str, current_references: str, current_texts: str):
            if not state_value:
                raise gr.Error("No training run is attached")
            adapter_dir = str(Path(state_value).resolve())
            updates = adapter_selection_updates(adapter_dir, current_references, current_texts)
            return (
                gr.update(choices=_adapter_folders(), value=adapter_dir),
                gr.Tabs(selected="checkpoint-grid"),
                *updates,
            )

        compare_button.click(
            compare_training_run,
            [training.state_dir, tab.controls["grid.references"], tab.controls["grid.texts"]],
            [tab.adapter, main_tabs, *tab.selection_outputs],
            queue=False,
        )
        if getattr(training, "start_event", None) is not None:
            training.start_event.then(
                compare_training_run,
                [training.state_dir, tab.controls["grid.references"], tab.controls["grid.texts"]],
                [tab.adapter, main_tabs, *tab.selection_outputs],
                queue=False,
            )


__all__ = [
    "GRID_DEFAULTS",
    "GRID_TERMINAL_PHASES",
    "GridTab",
    "adapter_selection_updates",
    "adopt_grid_state",
    "bind_grid_events",
    "build_grid_config_from_ui",
    "build_grid_tab",
    "calibrate_grid_speaking_rates",
    "checkpoint_eval_status_updates",
    "grid_status_updates",
    "latest_checkpoint_eval_state",
    "latest_grid_state",
    "latest_lora_folder",
]
