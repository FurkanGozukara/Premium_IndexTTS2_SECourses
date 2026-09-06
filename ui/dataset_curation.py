"""Run the optional voice/transcript dataset audit from the browser."""
from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Any

import gradio as gr

from indextts.training.dataset_manifest import load_manifest
from .common import PROCESS_MANAGER, ROOT, STATE_ROOT, btn, parse_multiline_paths, progress_panel_html, read_json, tail_text, write_json_atomic

CURATION_STATE = STATE_ROOT / "curation"
CURATION_DEFAULTS = {
    "name": "voice_curated", "references": "", "validation_sources": "", "test_sources": "",
    "max_wer": .15, "min_speaker_similarity": .70, "min_window_similarity": .60,
    "transcribe_all": True, "check_boundary_words": True, "min_edge_silence_ms": 30,
    "device": "cuda:0",
}


def curation_command(dataset: str, values: dict[str, Any], *, model_dir: str = "models") -> tuple[list[str], Path]:
    for key in ("max_wer", "min_speaker_similarity", "min_window_similarity"):
        if not 0 <= float(values[key]) <= 1:
            raise ValueError(f"{key} must be between 0 and 1")
    if not 0 <= float(values["min_edge_silence_ms"]) <= 500:
        raise ValueError("Minimum quiet edge must be between 0 and 500 ms")
    source = Path(dataset).expanduser().resolve()
    info = read_json(source / "dataset_info.json", {}) or {}
    if info.get("status") != "complete":
        raise ValueError("Select a completed prepared dataset before auditing it")
    rows = load_manifest(source)
    if not rows:
        raise ValueError("The selected dataset contains no clips")
    name = str(values["name"]).strip()
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("The audited dataset name must be a single directory name")
    output = source.parent / name
    if output.exists():
        raise FileExistsError(f"Choose a new audited dataset name; this folder already exists: {output}")
    refs = [Path(value).expanduser().resolve() for value in parse_multiline_paths(str(values["references"]))]
    if not refs or any(not path.is_file() for path in refs):
        raise ValueError("Provide at least one existing, clean reference recording of the intended speaker")
    validation = [Path(value).stem for value in parse_multiline_paths(str(values["validation_sources"]))]
    test = [Path(value).stem for value in parse_multiline_paths(str(values["test_sources"]))]
    available = {Path(row["source_media"]).stem for row in rows}
    if not validation:
        raise ValueError("Reserve at least one source recording for validation")
    if set(validation) & set(test):
        raise ValueError("Validation and final-test sources must differ")
    unknown = (set(validation) | set(test)) - available
    if unknown:
        raise ValueError(f"Unknown source recordings: {', '.join(sorted(unknown))}. Available: {', '.join(sorted(available))}")
    if not available - set(validation + test):
        raise ValueError("Leave at least one source recording for training")
    if test and output.with_name(output.name + "_test").exists():
        raise FileExistsError(output.with_name(output.name + "_test"))
    command = [sys.executable, "-u", str(ROOT / "tools/curate_voice_dataset.py"), str(source), str(output),
               "--model-dir", model_dir, "--whisper", str((info.get("config") or {}).get("whisper_model", "openai/whisper-large-v3-turbo"))]
    for flag, items in (("--reference", refs), ("--validation-source", validation), ("--test-source", test)):
        for item in items:
            command.extend([flag, str(item)])
    for field, flag in (("max_wer", "--max-wer"), ("min_speaker_similarity", "--min-speaker-similarity"),
                        ("min_window_similarity", "--min-window-similarity"), ("device", "--device"),
                        ("min_edge_silence_ms", "--min-edge-silence-ms")):
        command.extend([flag, str(int(values[field])) if field == "min_edge_silence_ms" else str(values[field])])
    for field, flag in (("transcribe_all", "--transcribe-all"), ("check_boundary_words", "--check-boundary-words")):
        if values[field]:
            command.append(flag)
    return command, output


def build_curation_controls(registry: Any, existing: Any, dataset_path: Any, *, load_hook: Any = None, model_dir: str = "models") -> Any:
    controls: dict[str, Any] = {}

    def register(field: str, component: Any, kind: str, **kwargs: Any) -> Any:
        registry.register(f"curation.{field}", component, CURATION_DEFAULTS[field], kind=kind, **kwargs)
        controls[field] = component
        return component

    with gr.Accordion("Voice and transcript audit", open=False):
        gr.Markdown("After preparation, create a separate training dataset with speaker consistency and transcript checks. "
                    "Original clips are preserved. Similarity and speech recognition are screening tools; review the audit report for rejected clips.")
        register("name", gr.Textbox(value="voice_curated", label="Audited dataset name"), "str")
        register("references", gr.Textbox(lines=2, label="Verified speaker reference paths",
                 info="One clean audio file path per line, containing only the intended speaker."), "str")
        with gr.Row():
            register("validation_sources", gr.Textbox(lines=2, label="Validation source recordings",
                     info="One source filename stem per line. Complete recordings are held out from training."), "str")
            register("test_sources", gr.Textbox(lines=2, label="Final-test source recordings",
                     info="Optional. Kept in a separate dataset, outside training and checkpoint selection."), "str")
        with gr.Row():
            register("max_wer", gr.Slider(0, 1, value=.15, step=.01, label="Maximum transcript error",
                     info="Word errors for EN/ES/AR; character errors for Chinese/Japanese."), "float", minimum=0, maximum=1)
            register("min_speaker_similarity", gr.Slider(0, 1, value=.70, step=.01, label="Minimum voice similarity"), "float", minimum=0, maximum=1)
            register("min_window_similarity", gr.Slider(0, 1, value=.60, step=.01, label="Minimum voice-window similarity"), "float", minimum=0, maximum=1)
        with gr.Row():
            register("transcribe_all", gr.Checkbox(value=True, label="Transcribe every voice-matched clip",
                     info="Checks the extracted audio itself; takes longer than reusing source transcriptions."), "bool")
            register("check_boundary_words", gr.Checkbox(value=True, label="Verify first and last words",
                     info="Requires both transcript edges to match fresh clip transcription, even when overall word error is low."), "bool")
            register("min_edge_silence_ms", gr.Slider(0, 500, value=30, step=10, label="Audit minimum quiet edge (ms)"), "int", minimum=0, maximum=500)
            register("device", gr.Textbox(value="cuda:0", label="Audit device"), "str")
        with gr.Row():
            start = gr.Button("🧪  Audit and create training dataset", variant="primary", elem_classes=btn("teal"))
            stop = gr.Button("🛑  Stop audit", variant="stop", elem_classes=btn("crimson"))
        state = gr.State("")
        applied = gr.State("")
        completed_path = gr.Textbox(value="", visible=False, label="Completed audited dataset")
        timer = gr.Timer(1.0, active=False)
        panel = gr.HTML(progress_panel_html({}, title="Dataset audit ready"))
        status = gr.Markdown("")
        log = gr.Textbox(label="Dataset audit log", lines=8, interactive=False, buttons=["copy"], elem_classes=["log-tail"])

    def poll(state_value: str, applied_value: str):
        if not state_value:
            return (gr.skip(),) * 7
        directory = Path(state_value)
        config = read_json(directory / "config.json", {}) or {}
        value = read_json(directory / "status.json", {}) or {}
        phase = str(value.get("phase", "starting"))
        job = PROCESS_MANAGER.get("dataset_curation")
        if phase not in {"complete", "failed", "cancelled"} and job is not None and job.state_dir == directory and not job.running:
            phase = "cancelled" if job.canceled else "failed"
            value.update(phase=phase, message=f"Audit worker stopped with code {job.process.returncode}; see the log.")
            write_json_atomic(directory / "status.json", value)
        total = int(value.get("total", 0) or 0)
        done = int(value.get("completed", 0) or 0)
        terminal = phase in {"complete", "failed", "cancelled"}
        payload = {**value, "fraction": done / total if total else 0,
                   "desc": value.get("message", "Starting audit"), "speed_unit": "clips/s",
                   "speed": done / max(.001, float(value.get("elapsed_s", 0) or 0))}
        message = str(payload["desc"])
        if "kept" in value:
            message += f" | retained {value['kept']} + {value.get('test', 0)} test | rejected {value.get('rejected', 0)}"
        output = str(config.get("output", ""))
        apply = phase == "complete" and applied_value != state_value
        return (state_value if apply else gr.skip(),
                progress_panel_html(payload, title="Dataset audit " + phase), message,
                tail_text(directory / "audit.log", 60), output if apply else gr.skip(),
                gr.Timer(1.0, active=not terminal), gr.skip())

    def begin(selected: str, current: str, *values: Any):
        directory = None
        try:
            active = PROCESS_MANAGER.get("dataset_curation")
            if active is not None and active.running:
                raise ValueError("A dataset audit is already running")
            settings = dict(zip(controls, values))
            command, output = curation_command(selected or current, settings, model_dir=model_dir)
            directory = CURATION_STATE / f"{output.name}_{time.time_ns()}"
            directory.mkdir(parents=True, exist_ok=False)
            write_json_atomic(directory / "config.json", {"dataset": selected or current, "output": str(output), "settings": settings})
            write_json_atomic(directory / "status.json", {"phase": "starting", "message": "Starting voice and transcript audit"})
            PROCESS_MANAGER.start("dataset_curation", command + ["--state-dir", str(directory)],
                                  state_dir=directory, log_path=directory / "audit.log", cwd=ROOT)
            return "", progress_panel_html({}, title="Starting dataset audit"), "Starting voice and transcript audit", "", "", gr.Timer(1.0, active=True), str(directory)
        except Exception as exc:
            if directory is not None:
                write_json_atomic(directory / "status.json", {"phase": "failed", "message": str(exc)})
            message = f"Audit not started: {exc}"
            return (gr.skip(), progress_panel_html({"desc": message}, title="Dataset audit not started"),
                    message, gr.skip(), gr.skip(), gr.skip(), gr.skip())

    outputs = [applied, panel, status, log, completed_path, timer, state]
    start.click(begin, [existing, dataset_path, *controls.values()], outputs, queue=False,
                api_name="curate_dataset", show_progress="hidden")
    timer.tick(poll, [state, applied], outputs, queue=False, show_progress="hidden")

    def cancel(directory: str):
        if not directory:
            return "No audit is active."
        value = read_json(Path(directory) / "status.json", {}) or {}
        if value.get("phase") in {"complete", "failed", "cancelled"}:
            return "No audit is active."
        (Path(directory) / "stop.flag").touch()
        return "Stop requested. The audit will finish its current clip and preserve the source dataset."

    stop.click(cancel, state, status, queue=False, api_name="stop_dataset_audit")
    if load_hook is not None:
        def attach():
            candidates = list(CURATION_STATE.glob("*/config.json"))
            if not candidates:
                return (gr.skip(),) * 7
            directory = max(candidates, key=lambda item: item.stat().st_mtime).parent
            values = list(poll(str(directory), ""))
            values[-1] = str(directory)
            return tuple(values)
        load_hook(attach, outputs=outputs, queue=False, show_progress="hidden", api_name="attach_dataset_audit")
    return completed_path
