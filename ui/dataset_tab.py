"""LoRA dataset discovery, preparation, feature caching, and inspection."""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path
import sys
import threading
import time
import traceback
from typing import Any, Mapping, Sequence

import gradio as gr
import pandas as pd

from indextts.runtime.progress import format_duration
from indextts.training.dataset_manifest import load_manifest, summarize_manifest
from indextts.training.dataset_prep import DatasetPrepConfig
from indextts.training.media import (
    find_media_files,
    find_sidecar_subtitles,
    find_sidecar_transcript,
    probe_media,
)

from .common import (
    PROCESS_MANAGER,
    ROOT,
    STATE_ROOT,
    btn,
    dedupe_updates,
    open_folder,
    parse_multiline_paths,
    progress_panel_html,
    read_json,
    stats_html,
    tail_text,
    write_json_atomic,
)
from .presets_store import PresetRegistry


DATASET_DEFAULTS = DatasetPrepConfig(name="voice_dataset", inputs=["input_media"]).to_dict()
# The dataclass chooses a hardware-dependent fallback. Presets must remain
# deterministic, and sentence-aligned preparation is the quality-first default.
DATASET_DEFAULTS["segmentation_mode"] = "sentence_aligned"
DATASET_STATE = STATE_ROOT / "dataset"
DATASET_STATE.mkdir(parents=True, exist_ok=True)
_LAST_DATASET_FOLDER = ROOT / "datasets"
DATASET_TERMINAL_PHASES = frozenset(
    {"complete", "cancelled", "canceled", "error", "failed"}
)


def scan_datasets(root: str | Path = ROOT / "datasets") -> list[tuple[str, str]]:
    base = Path(root).expanduser()
    entries: list[tuple[str, str]] = []
    if not base.is_dir():
        return entries
    for info_path in base.glob("*/dataset_info.json"):
        info = read_json(info_path, {}) or {}
        label = (
            f"{info_path.parent.name} | {int(info.get('segment_count', 0) or 0)} segments | "
            f"{float(info.get('total_duration_minutes', 0.0) or 0.0):.1f} min"
        )
        entries.append((label, str(info_path.parent.resolve())))
    return sorted(entries, key=lambda item: item[0].lower())


def dataset_summary_line(path: str | Path) -> str:
    root = Path(path).expanduser()
    info = read_json(root / "dataset_info.json", {}) or {}
    return (
        f"**{root.name}** | {info.get('segment_count', 0)} segments | "
        f"{float(info.get('total_duration_minutes', 0.0) or 0.0):.2f} minutes | "
        f"features {'cached' if (root / 'cache' / 'index.jsonl').is_file() else 'not cached'}"
    )


def latest_dataset_state(
    root: str | Path = DATASET_STATE,
) -> tuple[str, str, bool]:
    base = Path(root).expanduser()
    candidates: list[tuple[float, Path]] = []
    for status_path in base.glob("*/status.json") if base.is_dir() else []:
        try:
            candidates.append((status_path.stat().st_mtime, status_path))
        except OSError:
            continue
    statuses = [path for _, path in sorted(candidates, key=lambda item: item[0], reverse=True)]
    if not statuses:
        return "", "", False
    status_path = statuses[0]
    status = read_json(status_path, {}) or {}
    config = read_json(status_path.parent / "config.json", {}) or {}
    output_root = Path(str(config.get("output_root") or ROOT / "datasets"))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    dataset_dir = output_root / str(config.get("name") or "")
    phase = str(status.get("phase") or "").strip().lower()
    running = bool(phase and phase not in DATASET_TERMINAL_PHASES)
    return str(status_path.parent), str(dataset_dir.resolve()), running


def _dataset_state_running(path: str | Path | None) -> bool:
    status = read_json(Path(path) / "status.json", {}) if path else {}
    phase = str((status or {}).get("phase") or "").strip().lower()
    return bool(phase and phase not in DATASET_TERMINAL_PHASES)


def _dataset_path_for_state(path: str | Path | None) -> str:
    if not path:
        return ""
    config = read_json(Path(path) / "config.json", {}) or {}
    output_root = Path(str(config.get("output_root") or ROOT / "datasets")).expanduser()
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    name = str(config.get("name") or "").strip()
    return str((output_root / name).resolve()) if name else ""


def adopt_dataset_state(
    displayed_state: str | Path | None,
    displayed_dataset: str | Path | None,
    *,
    root: str | Path = DATASET_STATE,
    page_load: bool = False,
) -> tuple[str, str, bool]:
    """Resolve a session's preparation state, adopting a newer live worker when idle."""

    current = str(Path(displayed_state).expanduser().resolve()) if displayed_state else ""
    dataset = str(Path(displayed_dataset).expanduser().resolve()) if displayed_dataset else ""
    if current and _dataset_state_running(current):
        return current, _dataset_path_for_state(current) or dataset, True
    newest_state, newest_dataset, newest_running = latest_dataset_state(root)
    if page_load and newest_state:
        return newest_state, newest_dataset, newest_running
    if newest_state and newest_running:
        return newest_state, newest_dataset, True
    return current, dataset, False


def _resolved_inputs(path_text: str, uploads: list[str] | None) -> list[str]:
    values = parse_multiline_paths(path_text)
    for upload in uploads or []:
        path = str(Path(upload).resolve())
        if path not in values:
            values.append(path)
    return values


def scan_input_rows(path_text: str, uploads: list[str] | None, recursive: bool) -> list[list[Any]]:
    inputs = _resolved_inputs(path_text, uploads)
    media = find_media_files(inputs, bool(recursive))
    rows = []
    for value in media:
        path = Path(value)
        try:
            info = probe_media(path)
            duration = round(info.duration_s, 2)
        except Exception:
            duration = 0.0
        subtitles = find_sidecar_subtitles(path)
        transcript = find_sidecar_transcript(path)
        sidecars = ", ".join(Path(item).name for item in subtitles)
        if transcript:
            sidecars = ", ".join(filter(None, [sidecars, Path(transcript).name]))
        rows.append([path.name, duration, path.suffix.lower(), sidecars or "None", str(path)])
    return rows


def _dataset_result(dataset_dir: str | Path) -> tuple[str, pd.DataFrame, list[list[Any]], list[str], str, list[str]]:
    root = Path(dataset_dir).expanduser().resolve()
    info = read_json(root / "dataset_info.json", {}) or {}
    rows = load_manifest(root)
    summary = summarize_manifest(rows)
    summary.update(info)
    histogram = summary.get("duration_histogram") or {}
    hist_frame = pd.DataFrame([{"bucket": key, "count": value} for key, value in histogram.items()])
    table = []
    paths = []
    for row in rows:
        audio = Path(str(row.get("audio", "")))
        if not audio.is_absolute():
            audio = root / audio
        table.append([row.get("id", ""), float(row.get("duration_s", 0.0) or 0.0), row.get("text", ""), str(audio)])
        paths.append(str(audio))
    references = []
    for value in info.get("reference_candidates", []):
        path = Path(str(value))
        if not path.is_absolute():
            path = root / path
        if path.is_file():
            references.append(str(path))
    warnings = info.get("warnings") or []
    warning_text = "\n".join(f"- {item}" for item in warnings) if warnings else "No dataset warnings."
    return stats_html(summary), hist_frame, table, references, warning_text, paths


def dataset_status_to_panel(value: Mapping[str, Any]) -> tuple[str, str]:
    """Map the preparation status contract to the shared card and status line."""

    status = dict(value or {})
    file_i = int(status.get("file_i", 0) or 0)
    file_n = int(status.get("file_n", 0) or 0)
    phase = str(status.get("phase") or "starting").strip().lower()
    fraction = status.get("fraction")
    if fraction is None:
        fraction = file_i / file_n if file_n else (1.0 if phase == "complete" else 0.0)
    elapsed = status.get("elapsed_s", status.get("elapsed", 0.0))
    eta = status.get("eta_s", status.get("eta"))
    segments = int(status.get("segment_count", 0) or 0)
    audio_seconds = float(status.get("total_audio_seconds", 0.0) or 0.0)
    payload = {
        **status,
        "fraction": float(fraction or 0.0),
        "completed": file_i,
        "total": file_n,
        "elapsed_s": elapsed,
        "eta_s": eta,
        "desc": status.get("message") or phase,
    }
    terminal_titles = {
        "complete": "Dataset complete",
        "cancelled": "Dataset canceled",
        "canceled": "Dataset canceled",
        "error": "Dataset failed",
        "failed": "Dataset failed",
    }
    title = terminal_titles.get(phase, "Dataset preparation")
    if phase == "complete":
        payload["fraction"] = 1.0
    parts = [
        f"**{phase.replace('_', ' ').title()}**",
        str(status.get("message") or "Waiting for worker status"),
        f"file {file_i}/{file_n}" if file_n else f"file {file_i}",
        f"{segments} segments",
        f"{audio_seconds / 60.0:.2f} minutes",
        f"elapsed {format_duration(elapsed)}",
        f"ETA {format_duration(eta)}",
    ]
    return progress_panel_html(payload, title=title), " | ".join(parts)


def dataset_status_updates(state_value: str, dataset_value: str) -> tuple[Any, ...]:
    """Return the complete dataset dashboard update used by timer and server push."""

    if not state_value:
        empty_hist = pd.DataFrame(columns=["bucket", "count"])
        return (
            progress_panel_html({}, title="Ready"),
            "",
            "",
            "",
            empty_hist,
            [],
            [],
            "",
            [],
            gr.Timer(5.0, active=True),
            gr.skip(),
        )
    state = Path(state_value)
    value = read_json(state / "status.json", {}) or {}
    panel, status_line = dataset_status_to_panel(value)
    phase = str(value.get("phase") or "starting").strip().lower()
    live_log = tail_text(state / "log.txt", 60) or tail_text(state / "worker_console.log", 60)
    terminal = phase in DATASET_TERMINAL_PHASES
    if not terminal:
        return (
            panel,
            status_line,
            live_log,
            gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
            gr.Timer(1.0, active=True),
            gr.skip(),
        )
    dataset_dir = Path(dataset_value).expanduser() if dataset_value else Path("__missing_dataset__")
    completed = phase == "complete" and dataset_dir.is_dir() and (dataset_dir / "dataset_info.json").is_file()
    if completed:
        stats_value, hist, table, refs, warning_text, paths = _dataset_result(dataset_dir)
    else:
        stats_value = ""
        hist = pd.DataFrame(columns=["bucket", "count"])
        table, refs, paths = [], [], []
        warning_text = "No completed dataset result was found."
    return (
        panel,
        status_line,
        live_log,
        stats_value,
        hist,
        table,
        refs,
        warning_text,
        paths,
        gr.Timer(5.0, active=True),
        gr.update(
            choices=scan_datasets(),
            value=str(dataset_dir.resolve()) if completed else None,
        ),
    )


def dataset_poll_updates(
    state_value: str,
    dataset_value: str,
    *,
    state_root: str | Path = DATASET_STATE,
    page_load: bool = False,
) -> tuple[Any, ...]:
    """Adopt a live preparation when idle, then render state and dashboard."""

    adopted_state, adopted_dataset, running = adopt_dataset_state(
        state_value,
        dataset_value,
        root=state_root,
        page_load=page_load,
    )
    updates = list(dataset_status_updates(adopted_state, adopted_dataset))
    if running and adopted_state:
        config = read_json(Path(adopted_state) / "config.json", {}) or {}
        run_name = str(config.get("name") or Path(adopted_state).name)
        updates[1] = f"Attached to running run {run_name} | {updates[1]}"
        updates[-2] = gr.Timer(1.0, active=True)
    else:
        updates[-2] = gr.Timer(5.0, active=True)
    return adopted_state, adopted_dataset, *updates


@dataclass
class DatasetTab:
    controls: dict[str, Any]
    existing: Any
    existing_info: Any
    dataset_path_state: Any
    prep_event: Any = None


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
    nullable: bool = False,
) -> None:
    key = f"dataset.{field_name}"
    default = DATASET_DEFAULTS[field_name]
    if field_name == "inputs":
        default = ""
    registry.register(key, component, default, kind=kind, choices=choices, minimum=minimum, maximum=maximum, nullable=nullable)
    controls[key] = component


def build_dataset_tab(
    args: Any,
    registry: PresetRegistry,
    *,
    load_hook: Any | None = None,
) -> DatasetTab:
    global _LAST_DATASET_FOLDER
    controls: dict[str, Any] = {}
    device_default = str(getattr(args, "device", "cuda:0") or "cuda:0")
    if device_default == "auto":
        device_default = "cuda:0"
    # Each browser discovers its previous run only when the header load control fires;
    # a build-time snapshot would expose stale results after reloads.
    initial_state, initial_dataset_path = "", ""

    with gr.Tab("LoRA Dataset Preparation", id="dataset-preparation") as tab_block:
        with gr.Row():
            existing = gr.Dropdown(
                choices=scan_datasets(), value=None, label="Existing datasets",
                info="Prepared datasets with dataset_info.json under datasets/.", scale=5,
            )
            refresh_existing = gr.Button("↻  Refresh", elem_classes=btn("violet"), scale=1)
            open_existing = gr.Button("🗂️  Open dataset folder", elem_classes=btn("sky"), scale=1)
        existing_info = gr.Markdown("Select a dataset to inspect it.")

        with gr.Accordion("Inputs", open=True):
            with gr.Row(equal_height=False):
                with gr.Column(scale=2):
                    uploads = gr.File(
                        label="Media files",
                        file_count="multiple",
                        file_types=["audio", "video"],
                        type="filepath",
                    )
                    gr.Markdown("Upload source recordings directly; path inputs also discover transcript sidecars.", elem_classes=["section-note"])
                    input_paths = gr.Textbox(
                        label="Input files or folders",
                        lines=4,
                        placeholder="One path per line",
                        info="Accepts media files, folders, metadata.csv, or pre-segmented WAV+TXT folders.",
                    )
                with gr.Column(scale=1):
                    name = gr.Textbox(value="voice_dataset", label="Dataset name", info="A single safe directory name created below Output root.")
                    output_root = gr.Textbox(value="datasets", label="Output root", info="Dataset parent directory; relative paths resolve from the application folder.")
                    recursive = gr.Checkbox(value=True, label="Scan recursively", info="Includes supported media in nested input folders.")
                    language = gr.Dropdown(choices=["ZH", "EN", "JA", "AR", "ES"], value="EN", label="Language", info="Language stored in manifest rows and used for Whisper.")
                    speaker_name = gr.Textbox(value="", label="Speaker name", info="Optional fixed speaker label stored in each segment.")
                    speaker_folder = gr.Checkbox(value=False, label="Speaker from folder", info="Uses each source parent folder name as the speaker label.")
            _reg(registry, controls, "inputs", input_paths, kind="str")
            _reg(registry, controls, "name", name, kind="str")
            _reg(registry, controls, "output_root", output_root, kind="str")
            _reg(registry, controls, "recursive", recursive, kind="bool")
            _reg(registry, controls, "language", language, kind="choice", choices=["ZH", "EN", "JA", "AR", "ES"])
            _reg(registry, controls, "speaker_name", speaker_name, kind="str")
            _reg(registry, controls, "speaker_from_folder", speaker_folder, kind="bool")

        with gr.Accordion("Transcripts", open=False):
            with gr.Row():
                subtitle_policy = gr.Dropdown(choices=["prefer_sidecar", "whisper_only", "sidecar_only"], value="prefer_sidecar", label="Transcript policy", info="Prefer sidecars is recommended; Whisper fills missing timing/text.")
                whisper_model = gr.Textbox(value="openai/whisper-large-v3-turbo", label="Whisper model", info="Hugging Face model id or local path used when transcription/alignment is needed.")
                whisper_device = gr.Textbox(value=device_default, label="Whisper device", info="CUDA device recommended for sentence alignment; CPU works but is much slower.")
            with gr.Row():
                segmentation_mode = gr.Dropdown(choices=["auto", "sentence_aligned", "cue_boundaries", "whisper_only"], value=DATASET_DEFAULTS["segmentation_mode"], label="Segmentation mode", info="Sentence aligned uses Whisper word times with caption sentences and is recommended on CUDA.")
                align = gr.Checkbox(value=False, label="Force Whisper alignment", info="Compatibility alias that forces sentence_aligned mode.")
            with gr.Row():
                remove_annotations = gr.Checkbox(value=True, label="Remove bracket annotations", info="Removes caption notes such as [music] and [applause].")
                dedupe = gr.Checkbox(value=True, label="Deduplicate rolling captions", info="Removes repeated text from live/rolling subtitle cues.")
                drop_duplicates = gr.Checkbox(value=True, label="Drop duplicate sentences", info="Keeps one copy (best aligned) of every sentence that is spoken more than once, e.g. repeated intros or outros; recommended for voice training.")
            _reg(registry, controls, "subtitle_policy", subtitle_policy, kind="choice", choices=["prefer_sidecar", "whisper_only", "sidecar_only"])
            _reg(registry, controls, "whisper_model", whisper_model, kind="str")
            _reg(registry, controls, "whisper_device", whisper_device, kind="str")
            _reg(registry, controls, "segmentation_mode", segmentation_mode, kind="choice", choices=["auto", "sentence_aligned", "cue_boundaries", "whisper_only"])
            _reg(registry, controls, "align_with_whisper", align, kind="bool")
            _reg(registry, controls, "remove_bracket_annotations", remove_annotations, kind="bool")
            _reg(registry, controls, "dedupe_rolling_captions", dedupe, kind="bool")
            _reg(registry, controls, "drop_duplicate_sentences", drop_duplicates, kind="bool")

        with gr.Accordion("Segmentation", open=False):
            with gr.Row():
                target_s = gr.Slider(1, 20, value=8, step=0.25, label="Target seconds", info="8 seconds gives efficient, stable training segments.")
                min_s = gr.Slider(0.5, 15, value=4, step=0.25, label="Minimum seconds", info="4 seconds is recommended for enough voice context.")
                max_s = gr.Slider(2, 30, value=12, step=0.25, label="Maximum seconds", info="12 seconds limits memory while preserving sentences.")
                max_gap = gr.Slider(0, 3000, value=700, step=25, label="Maximum cue gap (ms)", info="Cues closer than this can merge into one sentence segment.")
            with gr.Row():
                pad = gr.Slider(0, 500, value=60, step=10, label="Edge padding (ms)", info="Small context padding avoids clipped consonants.")
                snap = gr.Checkbox(value=True, label="Snap to silence", info="Moves segment boundaries toward nearby low-energy points.")
                snap_window = gr.Slider(0, 1000, value=200, step=10, label="Silence snap window (ms)", info="Search radius around a proposed boundary.")
                min_words = gr.Slider(0, 30, value=2, step=1, label="Minimum words", info="Drops fragments with too little transcript context.")
                max_words = gr.Slider(10, 200, value=80, step=1, label="Maximum words", info="Drops transcript segments that are implausibly dense.")
            for field_name, component, kind, minimum, maximum in (
                ("target_s", target_s, "float", 1, 20), ("min_s", min_s, "float", 0.5, 15),
                ("max_s", max_s, "float", 2, 30), ("max_gap_ms", max_gap, "int", 0, 3000),
                ("pad_ms", pad, "int", 0, 500), ("snap_to_silence", snap, "bool", None, None),
                ("snap_window_ms", snap_window, "int", 0, 1000), ("min_words", min_words, "int", 0, 30),
                ("max_words", max_words, "int", 10, 200),
            ):
                _reg(registry, controls, field_name, component, kind=kind, minimum=minimum, maximum=maximum)

        with gr.Accordion("Cleanup & Loudness", open=False):
            with gr.Row():
                trim = gr.Checkbox(value=True, label="Trim silence", info="Trims leading/trailing low-energy audio before filtering.")
                trim_db = gr.Slider(10, 80, value=40, step=1, label="Trim top dB", info="40 dB is a conservative silence threshold.")
                loudnorm = gr.Checkbox(value=True, label="Normalize loudness", info="Recommended for consistent training gradients across sources.")
                lufs = gr.Slider(-30, -10, value=-20, step=0.5, label="Target LUFS", info="-20 LUFS leaves headroom and matches voice training defaults.")
                sample_rate = gr.Dropdown(choices=[16000, 22050, 24000, 44100, 48000], value=24000, label="Sample rate", info="24000 Hz is required by the IndexTTS training pipeline.")
            with gr.Row():
                file_cov = gr.Slider(0, 1, value=0.6, step=0.01, label="Minimum file alignment coverage", info="Below 0.60, sentence alignment falls back or rejects unreliable timing.")
                segment_cov = gr.Slider(0, 1, value=0.7, step=0.01, label="Minimum segment alignment coverage", info="Drops individual caption segments with weak word alignment.")
                min_wps = gr.Slider(0.1, 5, value=1.0, step=0.1, label="Minimum words / second", info="Drops unusually sparse transcript/audio matches.")
                max_wps = gr.Slider(1, 12, value=5.5, step=0.1, label="Maximum words / second", info="Drops implausibly dense or misaligned speech.")
            with gr.Row():
                min_peak = gr.Slider(-80, 0, value=-35, step=1, label="Minimum peak dBFS", info="Drops audio too quiet to train reliably.")
                max_clip = gr.Number(value=0.001, minimum=0, maximum=1, step=0.0001, label="Maximum clipping ratio", info="0.001 allows at most 0.1% clipped samples.")
                clip_threshold = gr.Number(value=0.999, minimum=0.5, maximum=1, step=0.001, label="Clipping threshold", info="Absolute normalized sample level counted as clipping.")
                max_silence = gr.Number(value=None, minimum=0, maximum=1, step=0.01, label="Maximum silence ratio", info="Optional; blank disables whole-segment silence-ratio filtering.")
                silence_db = gr.Slider(-80, -10, value=-40, step=1, label="Silence threshold dBFS", info="Frames below this level count as silence.")
                silence_frame = gr.Slider(5, 200, value=20, step=5, label="Silence frame (ms)", info="20 ms gives stable silence estimates for speech.")
            for field_name, component, kind, minimum, maximum, nullable in (
                ("trim_silence", trim, "bool", None, None, False), ("trim_top_db", trim_db, "float", 10, 80, False),
                ("loudness_normalize", loudnorm, "bool", None, None, False), ("target_lufs", lufs, "float", -30, -10, False),
                ("sample_rate", sample_rate, "int", 8000, 48000, False),
                ("min_file_alignment_coverage", file_cov, "float", 0, 1, False),
                ("min_segment_alignment_coverage", segment_cov, "float", 0, 1, False),
                ("min_words_per_second", min_wps, "float", 0.1, 5, False),
                ("max_words_per_second", max_wps, "float", 1, 12, False),
                ("min_peak_dbfs", min_peak, "float", -80, 0, False),
                ("max_clipping_ratio", max_clip, "float", 0, 1, False),
                ("clipping_threshold", clip_threshold, "float", 0.5, 1, False),
                ("max_silence_ratio", max_silence, "float", 0, 1, True),
                ("silence_threshold_dbfs", silence_db, "float", -80, -10, False),
                ("silence_frame_ms", silence_frame, "int", 5, 200, False),
            ):
                _reg(registry, controls, field_name, component, kind=kind, minimum=minimum, maximum=maximum, nullable=nullable)

        with gr.Accordion("Output", open=False):
            with gr.Row():
                references = gr.Slider(0, 20, value=5, step=1, label="Reference candidates", info="Exports the cleanest segments for training samples and LoRA / DoRA use.")
                overwrite = gr.Checkbox(value=False, label="Overwrite dataset", info="Replaces an existing dataset directory with the same name.")
                max_segments = gr.Number(value=0, minimum=0, precision=0, label="Maximum segments", info="0 processes all segments; use a small value for smoke tests.")
                seed = gr.Number(value=0, precision=0, label="Preparation seed", info="Controls deterministic candidate ranking and randomized operations.")
            _reg(registry, controls, "export_reference_candidates", references, kind="int", minimum=0, maximum=20)
            _reg(registry, controls, "overwrite", overwrite, kind="bool")
            _reg(registry, controls, "max_segments", max_segments, kind="int", minimum=0, maximum=10000000)
            _reg(registry, controls, "seed", seed, kind="int", minimum=0, maximum=4294967295)

        with gr.Row():
            scan_button = gr.Button("🔍  Scan inputs", elem_classes=btn("purple"))
            prepare_button = gr.Button("🛠️  Prepare dataset", variant="primary", elem_classes=btn("emerald"))
            cancel_button = gr.Button("⛔  Cancel", variant="stop", elem_classes=btn("red"))
            cache_button = gr.Button("💽  Cache features now", elem_classes=btn("lime"))
            open_button = gr.Button("📁  Open dataset folder", elem_classes=btn("indigo"))

        discovered = gr.Dataframe(
            headers=["Media", "Duration s", "Type", "Sidecars", "Path"],
            datatype=["str", "number", "str", "str", "str"], value=[], type="array",
            interactive=False, wrap=True, label="Discovered media", max_height=320, buttons=["fullscreen"],
        )
        state_dir = gr.State(initial_state)
        dataset_path_state = gr.State(initial_dataset_path)
        segment_paths_state = gr.State([])
        reference_state = gr.State([])
        timer = gr.Timer(5.0, active=True)
        progress = gr.HTML(progress_panel_html({}, title="Ready"))
        status = gr.Markdown("")
        log = gr.Textbox(label="Preparation log", lines=10, max_lines=16, interactive=False, buttons=["copy"], elem_classes=["log-tail"])
        stats = gr.HTML("")
        histogram = gr.BarPlot(
            pd.DataFrame(columns=["bucket", "count"]), x="bucket", y="count",
            title="Segment duration distribution", x_title="Duration", y_title="Segments", height=300,
            buttons=["fullscreen", "export"],
        )
        segments = gr.Dataframe(
            headers=["ID", "Duration s", "Text", "Audio"],
            datatype=["str", "number", "str", "str"], value=[], type="array",
            interactive=False, wrap=True, label="Prepared segments", max_height=420,
            buttons=["fullscreen", "copy"],
        )
        selected_audio = gr.Audio(label="Selected segment", type="filepath", buttons=["download"])
        warnings = gr.Markdown("")
        with gr.Column():
            @gr.render(inputs=reference_state, triggers=[reference_state.change])
            def render_references(paths: list[str] | None):
                values = list(paths or [])
                if not values:
                    return
                gr.Markdown("#### Reference candidates")
                for index, path in enumerate(values, start=1):
                    gr.Audio(value=path, label=f"Reference candidate {index}", type="filepath", buttons=["download"], key=f"dataset-ref-{index}-{path}")

    config_specs = [spec for spec in registry.specs if spec.component is not None and spec.key.startswith("dataset.")]
    config_keys = [spec.key for spec in config_specs]
    config_components = [spec.component for spec in config_specs]

    def build_config(upload_values: list[str] | None, *items: Any) -> DatasetPrepConfig:
        values = dict(zip(config_keys, items))
        payload = {key.removeprefix("dataset."): value for key, value in values.items()}
        payload["inputs"] = _resolved_inputs(str(payload.get("inputs") or ""), upload_values)
        output = Path(str(payload.get("output_root") or "datasets")).expanduser()
        if not output.is_absolute():
            output = ROOT / output
        payload["output_root"] = str(output.resolve())
        config = DatasetPrepConfig.from_dict(payload)
        config.validate()
        return config

    scan_button.click(lambda uploads_value, paths, rec: scan_input_rows(paths, uploads_value, rec), [uploads, input_paths, recursive], discovered, queue=False, api_name="scan_dataset_inputs")

    def start_prep(upload_values: list[str] | None, *items: Any):
        global _LAST_DATASET_FOLDER
        try:
            config = build_config(upload_values, *items)
            started = time.time()
            state = DATASET_STATE / f"{config.name}_{int(started)}"
            state.mkdir(parents=True, exist_ok=True)
            config_path = write_json_atomic(state / "config.json", config.to_dict())
            Path(state / "stop.flag").unlink(missing_ok=True)
            _LAST_DATASET_FOLDER = Path(config.output_root) / config.name
            write_json_atomic(
                state / "status.json",
                {
                    "phase": "starting",
                    "file_i": 0,
                    "file_n": 0,
                    "segment_count": 0,
                    "total_audio_seconds": 0.0,
                    "message": "Starting dataset preparation worker",
                    "updated_at": time.time(),
                },
            )
            job = PROCESS_MANAGER.start(
                "dataset_prep",
                [sys.executable, "-m", "indextts.training.prep_worker", "--config", str(config_path), "--state-dir", str(state)],
                state_dir=state,
                log_path=state / "worker_console.log",
                cwd=ROOT,
                metadata={"dataset_dir": str(_LAST_DATASET_FOLDER)},
            )
            print(f">> Dataset preparation started: {config.name}", flush=True)
            updates = list(dataset_poll_updates(str(state), str(_LAST_DATASET_FOLDER)))
            updates[-2] = gr.Timer(1.0, active=True)
            emitted, fingerprints = dedupe_updates(updates)
            yield emitted
            while job.running:
                time.sleep(1.0)
                updates = list(dataset_poll_updates(str(state), str(_LAST_DATASET_FOLDER)))
                updates[0] = gr.skip()
                updates[1] = gr.skip()
                updates[-2] = gr.skip()
                emitted, fingerprints = dedupe_updates(updates, fingerprints)
                yield emitted
            current = read_json(state / "status.json", {}) or {}
            phase = str(current.get("phase") or "")
            if phase not in DATASET_TERMINAL_PHASES:
                current.update(
                    phase="cancelled" if job.canceled else "failed",
                    message=(
                        "Dataset preparation canceled by user"
                        if job.canceled
                        else f"Dataset worker exited with code {job.process.returncode}"
                    ),
                    updated_at=time.time(),
                )
                write_json_atomic(state / "status.json", current)
            updates = list(dataset_poll_updates(str(state), str(_LAST_DATASET_FOLDER)))
            updates[0] = gr.skip()
            updates[1] = gr.skip()
            updates[-2] = gr.Timer(5.0, active=True)
            emitted, _ = dedupe_updates(updates, fingerprints)
            yield emitted
        except Exception as exc:
            traceback.print_exc()
            raise gr.Error(str(exc)) from exc

    prep_event = prepare_button.click(
        start_prep,
        [uploads, *config_components],
        [
            state_dir,
            dataset_path_state,
            progress,
            status,
            log,
            stats,
            histogram,
            segments,
            reference_state,
            warnings,
            segment_paths_state,
            timer,
            existing,
        ],
        concurrency_limit=1,
        concurrency_id="dataset-prep",
        api_name="prepare_dataset",
        stream_every=0.5,
    )

    timer.tick(
        dataset_poll_updates,
        [state_dir, dataset_path_state],
        [state_dir, dataset_path_state, progress, status, log, stats, histogram, segments, reference_state, warnings, segment_paths_state, timer, existing],
        queue=False,
        show_progress="hidden",
    )
    if load_hook is not None:
        load_hook(
            lambda state, dataset_path: dataset_poll_updates(
                state,
                dataset_path,
                page_load=True,
            ),
            [state_dir, dataset_path_state],
            [state_dir, dataset_path_state, progress, status, log, stats, histogram, segments, reference_state, warnings, segment_paths_state, timer, existing],
            queue=False,
            show_progress="hidden",
            api_name="attach_dataset",
        )

    with tab_block:
        confirm = gr.Checkbox(value=False, visible=False, label="Dataset cancel confirmation")

    def cancel(confirmed: bool, state_value: str):
        if not confirmed:
            return "Cancellation dismissed."
        if not state_value or not _dataset_state_running(state_value):
            return "No active run."
        state = Path(state_value)
        (state / "stop.flag").touch()

        def force_after_grace() -> None:
            time.sleep(2.0)
            job = PROCESS_MANAGER.get("dataset_prep")
            if job is not None and job.running and job.state_dir.resolve() == state.resolve():
                PROCESS_MANAGER.terminate("dataset_prep")
                current = read_json(state / "status.json", {}) or {}
                if str(current.get("phase") or "").lower() not in DATASET_TERMINAL_PHASES:
                    current.update(
                        phase="cancelled",
                        message="Dataset preparation canceled by user",
                        updated_at=time.time(),
                    )
                    write_json_atomic(state / "status.json", current)

        threading.Thread(target=force_after_grace, daemon=True).start()
        return "Cancel requested; the worker has two seconds to stop cleanly before its process tree is terminated."

    cancel_button.click(
        cancel,
        [confirm, state_dir],
        status,
        js="(value, state) => [window.confirm('Cancel dataset preparation and stop its worker?'), state]",
        queue=False,
    )

    def cache_features(dataset_value: str, name_value: str, output_value: str):
        dataset_dir = Path(dataset_value) if dataset_value else Path(output_value) / name_value
        if not dataset_dir.is_absolute():
            dataset_dir = ROOT / dataset_dir
        dataset_dir = dataset_dir.resolve()
        if not (dataset_dir / "manifest.jsonl").is_file():
            raise gr.Error(f"Dataset manifest not found: {dataset_dir}")
        state = DATASET_STATE / f"cache_{dataset_dir.name}_{int(time.time())}"
        job = PROCESS_MANAGER.start(
            "dataset_cache",
            [sys.executable, str(ROOT / "tools" / "cache_dataset_features.py"), "--dataset-dir", str(dataset_dir), "--model-dir", str(getattr(args, "model_dir", ROOT / "models")), "--device", device_default],
            state_dir=state,
            log_path=state / "cache.log",
            cwd=ROOT,
        )
        started = time.perf_counter()
        while job.running:
            elapsed = time.perf_counter() - started
            yield f"Caching features | elapsed {elapsed:.1f}s", tail_text(job.log_path, 60)
            time.sleep(1)
        output = tail_text(job.log_path, 80)
        if job.process.returncode != 0:
            raise gr.Error(f"Feature caching failed with code {job.process.returncode}: {output[-1200:]}")
        yield f"Feature caching complete for {dataset_dir.name} in {time.perf_counter() - started:.1f}s.", output

    cache_button.click(cache_features, [existing, name, output_root], [status, log], concurrency_limit=1, concurrency_id="dataset-cache")

    def load_existing(path: str | None):
        global _LAST_DATASET_FOLDER
        if not path:
            return "Select a dataset.", "", pd.DataFrame(columns=["bucket", "count"]), [], [], "", [], ""
        _LAST_DATASET_FOLDER = Path(path)
        stats_value, hist, table, refs, warning_text, paths = _dataset_result(path)
        summary = dataset_summary_line(path)
        return summary, stats_value, hist, table, refs, warning_text, paths, str(Path(path).resolve())

    existing.change(load_existing, existing, [existing_info, stats, histogram, segments, reference_state, warnings, segment_paths_state, dataset_path_state], queue=False)
    refresh_existing.click(lambda: gr.update(choices=scan_datasets()), outputs=existing, queue=False)
    open_existing.click(lambda path: open_folder(path or _LAST_DATASET_FOLDER), existing, existing_info, queue=False)
    open_button.click(lambda path: open_folder(path or _LAST_DATASET_FOLDER), dataset_path_state, status, queue=False)

    def select_segment(paths: list[str], evt: gr.SelectData):
        index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        try:
            return paths[int(index)]
        except (TypeError, ValueError, IndexError):
            return gr.skip()

    segments.select(select_segment, segment_paths_state, selected_audio, queue=False)

    # Assert that every backend dataclass field is represented exactly once.
    expected = {f"dataset.{item.name}" for item in fields(DatasetPrepConfig)}
    actual = {key for key in controls if key.startswith("dataset.")}
    if expected != actual:
        raise RuntimeError(f"Dataset UI field mismatch: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    return DatasetTab(controls, existing, existing_info, dataset_path_state, prep_event)


def bind_dataset_events(tab: DatasetTab, training: Any) -> None:
    """Refresh and select the completed dataset in the Training tab."""

    if tab.prep_event is None:
        return

    def refresh_training_dataset(dataset_value: str):
        choices = scan_datasets()
        path = Path(dataset_value).expanduser() if dataset_value else None
        completed = bool(path and (path / "dataset_info.json").is_file())
        if not completed:
            return gr.update(choices=choices), gr.skip()
        info = read_json(path / "dataset_info.json", {}) or {}
        summary = (
            f"**{path.name}** | {info.get('segment_count', 0)} segments | "
            f"{float(info.get('total_duration_minutes', 0.0) or 0.0):.2f} minutes | "
            f"features **{'cached' if (path / 'cache' / 'index.jsonl').is_file() else 'not cached'}**"
        )
        return gr.update(choices=choices, value=str(path.resolve())), summary

    tab.prep_event.then(
        refresh_training_dataset,
        tab.dataset_path_state,
        [training.dataset, training.dataset_info],
        queue=False,
    )


__all__ = [
    "DATASET_DEFAULTS",
    "DATASET_TERMINAL_PHASES",
    "DatasetTab",
    "adopt_dataset_state",
    "bind_dataset_events",
    "build_dataset_tab",
    "dataset_status_to_panel",
    "dataset_status_updates",
    "dataset_summary_line",
    "dataset_poll_updates",
    "latest_dataset_state",
    "scan_datasets",
    "scan_input_rows",
]
