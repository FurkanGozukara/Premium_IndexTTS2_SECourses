"""Sequential batch generation using the complete Voice Generation settings."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import traceback
from typing import Any

import gradio as gr

from indextts.runtime.progress import read_progress_file
from indextts.utils.subtitle_utils import parse_subtitle_file, subtitle_cues_to_text
from indextts.utils.task_output_utils import write_metadata_file
from webui_generation_runner import run_generation_request

from .common import (
    LAZY_ENGINE,
    PROCESS_MANAGER,
    ROOT,
    adopt_output_task,
    btn,
    open_folder,
    output_task_is_active,
    progress_panel_html,
    read_json,
    tail_text,
    write_json_atomic,
)
from .generation_tab import GenerationTab, prepare_generation_request
from .presets_store import PresetRegistry


_BATCH_CANCEL = threading.Event()
_LAST_BATCH_FOLDER = ROOT / "outputs"


@dataclass
class BatchTab:
    start_button: Any
    cancel_button: Any
    status: Any
    progress: Any
    log: Any
    results: Any
    files: Any
    text: Any
    folder: Any
    controls: dict[str, Any]
    task_state: Any = None
    task_timer: Any = None


def _safe_subfolder(value: str) -> str:
    parts = [re.sub(r'[^A-Za-z0-9_. -]+', "_", part).strip(" ._") for part in Path(str(value or "batch")).parts]
    parts = [part for part in parts if part and part not in {".", ".."}]
    return str(Path(*parts)) if parts else "batch"


def _batch_items(files: list[str] | None, paragraphs: str, folder: str) -> list[dict[str, Any]]:
    paths: list[Path] = []
    for value in files or []:
        path = Path(str(value))
        if path.is_file() and path.suffix.lower() in {".txt", ".srt", ".vtt", ".sbv"}:
            paths.append(path.resolve())
    folder_text = str(folder or "").strip()
    folder_path = Path(folder_text).expanduser() if folder_text else None
    if folder_path is not None and folder_path.is_dir():
        paths.extend(
            path.resolve()
            for path in folder_path.rglob("*")
            if path.is_file() and path.suffix.lower() in {".txt", ".srt", ".vtt", ".sbv"}
        )
    unique = sorted(dict.fromkeys(paths), key=lambda path: str(path).lower())
    items: list[dict[str, Any]] = []
    for path in unique:
        if path.suffix.lower() in {".srt", ".vtt", ".sbv"}:
            cues = parse_subtitle_file(str(path))
            text = subtitle_cues_to_text(cues)
            subtitle = str(path)
        else:
            text = path.read_text(encoding="utf-8-sig", errors="replace").strip()
            subtitle = None
        items.append({"name": path.stem, "path": str(path), "text": text, "subtitle": subtitle})
    for index, paragraph in enumerate(re.split(r"\n\s*\n", str(paragraphs or "")), start=1):
        text = paragraph.strip()
        if text:
            items.append({"name": f"pasted_{index:03d}", "path": "", "text": text, "subtitle": None})
    return items


def _per_file_reference(item: dict[str, Any]) -> str | None:
    if not item.get("path"):
        return None
    path = Path(item["path"])
    for extension in (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"):
        candidate = path.with_suffix(extension)
        if candidate.is_file():
            return str(candidate)
    return None


def _duration(path: str | None) -> float:
    if not path:
        return 0.0
    try:
        import soundfile as sf

        return float(sf.info(path).duration)
    except Exception:
        return 0.0


def batch_task_updates(
    state_value: str,
    *,
    output_root: str | os.PathLike[str] = ROOT / "outputs",
    page_load: bool = False,
) -> tuple[Any, ...]:
    """Discover and render the most recent batch item's progress card."""

    task_value, running = adopt_output_task(
        state_value,
        root=output_root,
        scope="batch",
        page_load=page_load,
    )
    if not task_value:
        return (
            "",
            progress_panel_html({}, title="Ready"),
            "",
            gr.skip(),
            "",
            gr.Timer(5.0, active=True),
        )
    task_folder = Path(task_value)
    metadata = read_json(task_folder / "metadata.json", {}) or {}
    payload = read_progress_file(task_folder / "progress.json") or {}
    task_name = str((metadata.get("task") or {}).get("id") or task_folder.name)
    log_value = tail_text(task_folder / "generation.log", 60)
    if running:
        description = str(payload.get("desc") or payload.get("stage") or "Model is working...")
        return (
            task_value,
            progress_panel_html(payload, title="Batch generation"),
            f"Attached to running run {task_name} | {description}",
            gr.skip(),
            log_value,
            gr.Timer(1.0, active=True),
        )
    status = str(metadata.get("status") or "").strip().lower()
    if status in {"complete", "completed"}:
        title = "Batch item complete"
        message = f"Last batch task {task_name} | Generation complete."
        payload.update({"fraction": 1.0, "eta_s": 0, "desc": "Complete"})
    elif status in {"cancelled", "canceled"}:
        title = "Batch item canceled"
        message = f"Last batch task {task_name} | Generation canceled."
        payload.update({"eta_s": 0, "desc": "Canceled"})
    else:
        title = "Batch item failed"
        message = f"Last batch task {task_name} | {metadata.get('error') or 'Generation failed.'}"
        payload.update({"eta_s": 0, "desc": "Failed"})
    return (
        task_value,
        progress_panel_html(payload, title=title),
        message,
        gr.skip(),
        log_value,
        gr.Timer(5.0, active=True),
    )
def _poll_batch_item(request: dict[str, Any], subprocess_mode: bool, reuse_model: bool):
    task_folder = Path(request["task_layout"]["task_folder"])
    result_path = task_folder / "result.json"
    log_path = task_folder / "generation.log"
    if subprocess_mode:
        job = PROCESS_MANAGER.start(
            "batch_generation",
            [
                sys.executable,
                str(ROOT / "webui_subprocess_worker.py"),
                "--request-file",
                str(task_folder / "request.json"),
                "--result-file",
                str(result_path),
            ],
            state_dir=task_folder,
            log_path=log_path,
            cwd=ROOT,
            metadata={"metadata_path": request["metadata_path"]},
        )
        while job.running:
            yield read_progress_file(request["progress_file"]) or {}, tail_text(log_path, 40)
            if _BATCH_CANCEL.is_set():
                PROCESS_MANAGER.terminate("batch_generation")
                raise RuntimeError("Batch canceled by user")
            time.sleep(0.5)
        payload = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
        if job.process.returncode != 0 or payload.get("status") != "ok":
            raise RuntimeError(payload.get("error") or f"Generation worker exited with code {job.process.returncode}")
        return payload

    box: dict[str, Any] = {}

    def worker() -> None:
        try:
            engine = LAZY_ENGINE.get(request["runtime"])
            box["result"] = run_generation_request(request, engine)
        except BaseException as exc:
            box["error"] = exc
            traceback.print_exc()
        finally:
            box["done"] = True

    thread = threading.Thread(target=worker, daemon=True, name="batch-inprocess-item")
    thread.start()
    while not box.get("done"):
        yield read_progress_file(request["progress_file"]) or {}, tail_text(log_path, 40)
        if _BATCH_CANCEL.is_set():
            LAZY_ENGINE.request_cancel()
            raise RuntimeError("Batch canceled by user")
        time.sleep(0.5)
    thread.join()
    if not reuse_model:
        LAZY_ENGINE.unload()
    if "error" in box:
        raise RuntimeError(str(box["error"])) from box["error"]
    return box["result"]


def build_batch_tab(
    args: Any,
    registry: PresetRegistry,
    *,
    load_hook: Any | None = None,
) -> BatchTab:
    controls: dict[str, Any] = {}
    with gr.Tab("Batch Generation", id="batch-generation"):
        gr.Markdown("### Batch Inputs")
        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                files = gr.File(
                    label="Text or caption files",
                    file_count="multiple",
                    file_types=[".txt", ".srt", ".vtt", ".sbv"],
                    type="filepath",
                )
                gr.Markdown("Selected text and caption files are processed sequentially.", elem_classes=["section-note"])
                text = gr.Textbox(
                    label="Pasted items",
                    lines=8,
                    placeholder="One item per paragraph.\n\nA blank line starts the next item.",
                    info="Each paragraph becomes one named batch item; blank lines separate items.",
                )
                folder = gr.Textbox(
                    label="Input folder path",
                    info="Scans recursively for TXT, SRT, VTT, and SBV files.",
                )
            with gr.Column(scale=1):
                naming = gr.Textbox(
                    value="{index:03d}_{name}", label="Naming pattern",
                    info="Supports {index}, {name}, and {stem}; the extension is selected by Output settings.",
                )
                subfolder = gr.Textbox(
                    value="batch", label="Output subfolder",
                    info="A safe subfolder below outputs; task folders are created inside it.",
                )
                reference_mode = gr.Dropdown(
                    choices=["One reference for all", "Per-file reference"],
                    value="One reference for all", label="Reference mode",
                    info="Per-file mode looks for a same-stem audio file beside each text/caption file.",
                )
                execution = gr.Dropdown(
                    choices=["Subprocess per item", "Reuse loaded model between items", "Reload in-process model per item"],
                    value="Subprocess per item", label="Batch execution",
                    info="Subprocess is easiest to cancel; reuse is fastest after the first in-process load.",
                )
                continue_errors = gr.Checkbox(
                    value=True, label="Continue after item errors",
                    info="Records a failed row and proceeds to the next item instead of ending the batch.",
                )
                registry.register("batch.naming_pattern", naming, "{index:03d}_{name}", kind="str")
                registry.register("batch.output_subfolder", subfolder, "batch", kind="str")
                registry.register("batch.reference_mode", reference_mode, "One reference for all", kind="choice", choices=["One reference for all", "Per-file reference"])
                registry.register("batch.execution", execution, "Subprocess per item", kind="choice", choices=["Subprocess per item", "Reuse loaded model between items", "Reload in-process model per item"])
                registry.register("batch.continue_errors", continue_errors, True, kind="bool")
                controls.update({
                    "batch.naming_pattern": naming,
                    "batch.output_subfolder": subfolder,
                    "batch.reference_mode": reference_mode,
                    "batch.execution": execution,
                    "batch.continue_errors": continue_errors,
                })

        with gr.Row():
            start = gr.Button("🎬  Generate batch", variant="primary", elem_classes=btn("emerald"))
            cancel = gr.Button("⛔  Cancel batch", variant="stop", elem_classes=btn("red"))
            open_button = gr.Button("📁  Open batch folder", elem_classes=btn("indigo"))
        progress = gr.HTML(progress_panel_html({}, title="Ready"))
        status = gr.Markdown("")
        results = gr.Dataframe(
            headers=["Item", "Status", "Audio seconds", "Output path", "Time seconds"],
            datatype=["str", "str", "number", "str", "number"],
            value=[], type="array", interactive=False, wrap=True,
            label="Batch results", max_height=420, buttons=["fullscreen", "copy"],
        )
        log = gr.Textbox(label="Current item log", lines=10, max_lines=16, interactive=False, buttons=["copy"], elem_classes=["log-tail"])
        task_state = gr.State("")
        task_timer = gr.Timer(5.0, active=True)

        open_button.click(lambda: open_folder(_LAST_BATCH_FOLDER), outputs=status, queue=False)

    task_outputs = [task_state, progress, status, results, log, task_timer]
    task_timer.tick(
        batch_task_updates,
        task_state,
        task_outputs,
        queue=False,
        show_progress="hidden",
    )
    if load_hook is not None:
        load_hook(
            lambda state: batch_task_updates(state, page_load=True),
            task_state,
            task_outputs,
            queue=False,
            show_progress="hidden",
            api_name="attach_batch",
        )
    return BatchTab(
        start,
        cancel,
        status,
        progress,
        log,
        results,
        files,
        text,
        folder,
        controls,
        task_state,
        task_timer,
    )


def bind_batch_events(tab: BatchTab, generation: GenerationTab, args: Any, registry: PresetRegistry) -> None:
    model_dir = str(getattr(args, "model_dir", ROOT / "models"))
    batch_keys = list(tab.controls)
    batch_components = [tab.controls[key] for key in batch_keys]
    generation_keys = generation.request_keys
    generation_components = generation.request_components

    def run_batch(
        files: list[str] | None,
        paragraphs: str,
        folder: str,
        common_reference: str | None,
        *values: Any,
    ):
        global _LAST_BATCH_FOLDER
        _BATCH_CANCEL.clear()
        batch_values = dict(zip(batch_keys, values[:len(batch_keys)]))
        generation_values = dict(zip(generation_keys, values[len(batch_keys):]))
        items: list[dict[str, Any]] = []
        rows: list[list[Any]] = []
        started = time.perf_counter()
        output_root = ROOT / "outputs"
        last_item_progress: dict[str, Any] = {}
        current_task = ""

        def emit(panel: Any, message: str, result_rows: list[list[Any]], log_value: str, *, running: bool) -> tuple[Any, ...]:
            return (
                current_task or gr.skip(),
                panel,
                message,
                result_rows,
                log_value,
                gr.Timer(1.0 if running else 5.0, active=True),
            )

        try:
            items = _batch_items(files, paragraphs, folder)
            if not items:
                raise ValueError("No batch items were found")
            output_root = ROOT / "outputs" / _safe_subfolder(batch_values["batch.output_subfolder"])
            output_root.mkdir(parents=True, exist_ok=True)
            _LAST_BATCH_FOLDER = output_root
            execution = batch_values["batch.execution"]
            subprocess_mode = execution == "Subprocess per item"
            reuse_model = execution == "Reuse loaded model between items"
            print(f">> Batch started | {len(items)} items | {execution}", flush=True)
            yield emit(
                progress_panel_html({"fraction": 0, "completed": 0, "total": len(items), "desc": "Starting"}, title="Batch generation"),
                f"Starting {len(items)} items...",
                rows,
                "",
                running=True,
            )

            for index, item in enumerate(items, start=1):
                if _BATCH_CANCEL.is_set():
                    break
                item_started = time.perf_counter()
                reference = common_reference
                if batch_values["batch.reference_mode"] == "Per-file reference":
                    reference = _per_file_reference(item)
                    if not reference:
                        rows.append([item["name"], "Missing same-stem reference", 0.0, "", round(time.perf_counter() - item_started, 2)])
                        if not batch_values["batch.continue_errors"]:
                            break
                        continue
                try:
                    pattern = str(batch_values["batch.naming_pattern"] or "{index:03d}_{name}")
                    filename = pattern.format(index=index, name=item["name"], stem=item["name"])
                    item_values = dict(generation_values)
                    item_values["generation.output_filename"] = filename
                    request = prepare_generation_request(
                        item_values,
                        prompt=str(reference or ""),
                        text=item["text"],
                        subtitle_file=item["subtitle"],
                        image_path=None,
                        emotion_audio=None,
                        model_dir=model_dir,
                        output_root=output_root,
                    )
                    current_task = str(request["task_layout"]["task_folder"])
                    poller = _poll_batch_item(request, subprocess_mode, reuse_model)
                    while True:
                        try:
                            item_progress, item_log = next(poller)
                        except StopIteration as completed:
                            result = completed.value
                            break
                        last_item_progress = dict(item_progress)
                        fraction = ((index - 1) + float(item_progress.get("fraction", 0.0) or 0.0)) / len(items)
                        elapsed = time.perf_counter() - started
                        eta = elapsed / max(fraction, 1e-9) * (1 - fraction) if fraction > 0 else None
                        payload = {
                            "fraction": fraction,
                            "completed": index - 1,
                            "total": len(items),
                            "elapsed_s": elapsed,
                            "eta_s": eta,
                            "desc": f"{item['name']} ({index}/{len(items)})",
                            "speed": item_progress.get("speed"),
                            "speed_unit": item_progress.get("speed_unit", "x RT"),
                            "vram_used_gb": item_progress.get("vram_used_gb", 0),
                            "vram_total_gb": item_progress.get("vram_total_gb", 0),
                        }
                        yield emit(
                            progress_panel_html(payload, title="Batch generation"),
                            f"Generating {item['name']} ({index}/{len(items)})",
                            rows,
                            item_log,
                            running=True,
                        )
                    duration = float(result.get("audio_seconds") or _duration(result.get("output_path")))
                    final_item_progress = read_progress_file(request["progress_file"]) or {}
                    if final_item_progress:
                        last_item_progress = dict(final_item_progress)
                    rows.append([item["name"], "Complete", round(duration, 3), result.get("output_path", ""), round(time.perf_counter() - item_started, 2)])
                except Exception as exc:
                    traceback.print_exc()
                    rows.append([item["name"], f"Failed: {exc}", 0.0, "", round(time.perf_counter() - item_started, 2)])
                    if not batch_values["batch.continue_errors"]:
                        break
                completed_count = len(rows)
                elapsed = time.perf_counter() - started
                eta = elapsed / completed_count * (len(items) - completed_count) if completed_count else None
                panel = progress_panel_html({
                    "fraction": completed_count / len(items),
                    "completed": completed_count,
                    "total": len(items),
                    "elapsed_s": elapsed,
                    "eta_s": eta,
                    "desc": item["name"],
                    "speed": last_item_progress.get("speed"),
                    "speed_unit": last_item_progress.get("speed_unit", "x RT"),
                    "vram_used_gb": last_item_progress.get("vram_used_gb", 0),
                    "vram_total_gb": last_item_progress.get("vram_total_gb", 0),
                }, title="Batch generation")
                yield emit(panel, f"Completed {completed_count}/{len(items)}", rows, "", running=True)

            elapsed = time.perf_counter() - started
            complete = sum(row[1] == "Complete" for row in rows)
            audio_seconds = sum(float(row[2] or 0.0) for row in rows if row[1] == "Complete")
            rtf = elapsed / audio_seconds if audio_seconds > 0 else 0.0
            if _BATCH_CANCEL.is_set():
                message = f"Batch canceled after {len(rows)}/{len(items)} items."
                title = "Batch canceled"
            else:
                message = f"Batch finished: {complete}/{len(items)} complete in {elapsed:.1f}s."
                title = "Batch complete" if complete == len(items) else "Failed"
            print(
                f">> Batch summary | items={complete}/{len(items)} | audio={audio_seconds:.3f}s | "
                f"elapsed={elapsed:.3f}s | RTF={rtf:.4f} | output={output_root}",
                flush=True,
            )
            panel = progress_panel_html({
                "fraction": len(rows) / len(items),
                "completed": len(rows),
                "total": len(items),
                "elapsed_s": elapsed,
                "eta_s": 0,
                "desc": message,
                "speed": last_item_progress.get("speed"),
                "speed_unit": last_item_progress.get("speed_unit", "x RT"),
                "vram_used_gb": last_item_progress.get("vram_used_gb", 0),
                "vram_total_gb": last_item_progress.get("vram_total_gb", 0),
            }, title=title)
            yield emit(panel, message, rows, "", running=False)
        except Exception as exc:
            traceback.print_exc()
            elapsed = time.perf_counter() - started
            audio_seconds = sum(float(row[2] or 0.0) for row in rows if row[1] == "Complete")
            rtf = elapsed / audio_seconds if audio_seconds > 0 else 0.0
            print(
                f">> Batch summary | items={len(rows)}/{len(items)} | audio={audio_seconds:.3f}s | "
                f"elapsed={elapsed:.3f}s | RTF={rtf:.4f} | output={output_root} | failed={exc}",
                flush=True,
            )
            panel = progress_panel_html(
                    {
                        "fraction": len(rows) / len(items) if items else 0.0,
                        "completed": len(rows),
                        "total": len(items),
                        "elapsed_s": elapsed,
                        "eta_s": 0,
                        "desc": str(exc),
                        "vram_used_gb": last_item_progress.get("vram_used_gb", 0),
                        "vram_total_gb": last_item_progress.get("vram_total_gb", 0),
                    },
                    title="Failed",
                )
            yield emit(panel, f"Batch failed: {exc}", rows, "", running=False)

    event = tab.start_button.click(
        run_batch,
        inputs=[tab.files, tab.text, tab.folder, generation.prompt_audio, *batch_components, *generation_components],
        outputs=[tab.task_state, tab.progress, tab.status, tab.results, tab.log, tab.task_timer],
        api_name="generate_batch",
        concurrency_limit=1,
        concurrency_id="generation",
        stream_every=0.5,
    )
    confirmation = gr.Checkbox(value=False, visible=False, label="Batch cancel confirmation")

    def cancel_batch(confirmed: bool, state_value: str):
        if not confirmed:
            return gr.skip(), "Batch cancellation dismissed."
        if not state_value or not output_task_is_active(state_value):
            return gr.skip(), "No active run."
        displayed = Path(state_value).resolve()
        _BATCH_CANCEL.set()
        job = PROCESS_MANAGER.get("batch_generation")
        if job is not None and job.running and job.state_dir.resolve() == displayed:
            PROCESS_MANAGER.terminate("batch_generation")
        else:
            LAZY_ENGINE.request_cancel()
        metadata = read_json(displayed / "metadata.json", {}) or {}
        metadata.update(
            status="canceled",
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            error="Batch generation canceled by user",
        )
        write_metadata_file(str(displayed / "metadata.json"), metadata)
        payload = read_progress_file(displayed / "progress.json") or {}
        payload.update({"desc": "Canceled", "eta_s": 0})
        write_json_atomic(displayed / "progress.json", payload)
        return (
            progress_panel_html(payload, title="Batch canceled"),
            "Batch cancellation requested; the active item was stopped.",
        )

    tab.cancel_button.click(
        cancel_batch,
        [confirmation, tab.task_state],
        [tab.progress, tab.status],
        js="(value, state) => [window.confirm('Cancel the batch and stop the active item?'), state]",
        queue=False,
    )


__all__ = ["BatchTab", "batch_task_updates", "bind_batch_events", "build_batch_tab"]
