"""Shared UI styling, process control, progress rendering, and small helpers."""

from __future__ import annotations

import atexit
from dataclasses import dataclass, field
import gc
import html
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Any, Callable, Mapping, Sequence

import gradio as gr

from indextts.runtime.progress import format_duration, format_rate, read_progress_file
from indextts.utils.atomic_json import read_json_retry
from indextts.utils.atomic_json import write_json_atomic as _write_json_atomic
from indextts.runtime.vram_presets import RuntimeConfig


ROOT = Path(__file__).resolve().parents[1]
APP_TITLE = "IndexTTS 2.5 Premium SECourses"
APP_VERSION = "5.0"
FAVICON_PATH = ROOT / "ui_assets" / "indextts_premium_favicon.svg"
STATE_ROOT = ROOT / ".ui_state"
STATE_ROOT.mkdir(parents=True, exist_ok=True)

OUTPUT_TASK_TERMINAL_STATUSES = frozenset(
    {"complete", "completed", "failed", "error", "cancelled", "canceled"}
)


class PremiumTheme(gr.themes.Soft):
    """A restrained crimson theme with deliberately designed light/dark surfaces."""

    def __init__(self) -> None:
        from gradio.themes.utils import colors, fonts, sizes

        crimson = colors.Color(
            name="premium_crimson",
            c50="#fff1f3",
            c100="#ffe3e8",
            c200="#ffcbd5",
            c300="#fda4b5",
            c400="#f66a86",
            c500="#df345b",
            c600="#bd1f47",
            c700="#a11236",
            c800="#86132f",
            c900="#73142c",
            c950="#400615",
        )
        jade = colors.Color(
            name="premium_jade",
            c50="#effcf7",
            c100="#d8f8ea",
            c200="#b4efd8",
            c300="#7cdebd",
            c400="#3ec49d",
            c500="#1ca881",
            c600="#10866a",
            c700="#106b57",
            c800="#105546",
            c900="#0d463b",
            c950="#062820",
        )
        super().__init__(
            primary_hue=crimson,
            secondary_hue=jade,
            neutral_hue=colors.slate,
            spacing_size=sizes.spacing_md,
            radius_size=sizes.radius_sm,
            text_size=sizes.text_md,
            font=(
                fonts.GoogleFont("Plus Jakarta Sans", weights=(400, 500, 600, 700, 800)),
                "Aptos",
                "Segoe UI",
                "ui-sans-serif",
                "system-ui",
                "sans-serif",
            ),
            font_mono=(
                fonts.GoogleFont("JetBrains Mono", weights=(400, 500, 600)),
                "Cascadia Code",
                "Consolas",
                "ui-monospace",
                "monospace",
            ),
        )
        self.name = "indextts_premium"
        super().set(
            body_background_fill="#f5f7fa",
            body_background_fill_dark="#0b1018",
            body_text_color="#1d2938",
            body_text_color_dark="#e4eaf1",
            background_fill_primary="#fbfcfd",
            background_fill_primary_dark="#111925",
            background_fill_secondary="#eef2f6",
            background_fill_secondary_dark="#172230",
            block_background_fill="#fbfcfd",
            block_background_fill_dark="#111925",
            block_border_color="#d7dee7",
            block_border_color_dark="#2a3848",
            block_radius="6px",
            block_label_background_fill="#eef2f6",
            block_label_background_fill_dark="#172230",
            block_label_border_color="#d7dee7",
            block_label_border_color_dark="#2a3848",
            block_label_text_color="#5a1830",
            block_label_text_color_dark="#ffcbd5",
            input_background_fill="#ffffff",
            input_background_fill_dark="#0d1520",
            input_border_color="#c8d2de",
            input_border_color_dark="#344457",
            input_placeholder_color="#68778a",
            input_placeholder_color_dark="#94a3b5",
            button_primary_background_fill="#a11236",
            button_primary_background_fill_hover="#bd1f47",
            button_primary_text_color="#fff8fa",
            button_primary_border_color="#86132f",
            button_cancel_background_fill="#b4233f",
            button_cancel_background_fill_hover="#d02c4b",
            button_cancel_text_color="#fff8fa",
            button_cancel_border_color="#861b32",
            button_secondary_background_fill="#e5ebf1",
            button_secondary_background_fill_dark="#253346",
            button_secondary_background_fill_hover="#d8e1ea",
            button_secondary_background_fill_hover_dark="#304158",
            button_secondary_text_color="#263548",
            button_secondary_text_color_dark="#e7edf4",
            button_large_radius="6px",
            button_small_radius="5px",
            checkbox_label_background_fill_selected="#fff1f3",
            checkbox_label_background_fill_selected_dark="#3a1622",
            checkbox_label_text_color_selected="#86132f",
            checkbox_label_text_color_selected_dark="#ffcbd5",
            link_text_color="#a11236",
            link_text_color_dark="#f18aa0",
            shadow_drop="0 1px 2px rgba(28, 39, 54, 0.06)",
            shadow_drop_lg="0 8px 22px rgba(28, 39, 54, 0.08)",
        )


APP_HEAD = """
<meta name="theme-color" content="#a11236">
<meta name="color-scheme" content="light dark">
"""


APP_CSS = r"""
/* Gradio 6 tab strip: at borderline widths the overflow logic can render the selected tab twice; hide the duplicate. */
.tabs > .tab-wrapper > .tab-container[role="tablist"] > button[role="tab"].selected ~ button[role="tab"].selected { display: none !important; }
.tabs > .tab-wrapper > .tab-container[role="tablist"] > button[role="tab"] { white-space: nowrap; }
/* The main strip has exactly six tabs; Gradio 6 occasionally appends a stray duplicate button after a tab switch. */
#main-tabs > .tab-wrapper > .tab-container[role="tablist"] > button[role="tab"]:nth-child(n+7) { display: none !important; }

.gradio-container { width: 100% !important; max-width: 1540px !important; margin: 0 auto !important; padding: 0 18px 40px !important; }
.app-header { margin: 0 -18px 14px; padding: 18px 22px 15px; border-bottom: 3px solid #a11236; background: #eef2f6; }
.dark .app-header { background: #111925; border-bottom-color: #df345b; }
.app-header h1 { margin: 0 !important; font-size: 1.55rem !important; line-height: 1.25 !important; letter-spacing: 0 !important; }
.app-header p { margin: 6px 0 0 !important; color: #546579; }
.dark .app-header p { color: #aab8c8; }
.preset-bar { padding: 10px 0 12px; border-bottom: 1px solid var(--block-border-color); margin-bottom: 8px; }
.preset-status { min-height: 26px; font-size: 0.9rem; }
.section-note { color: #5d6d80; font-size: .9rem; }
.dark .section-note { color: #a7b5c5; }
.premium-primary button, button.premium-primary { font-weight: 800 !important; min-height: 48px; box-shadow: 0 4px 0 #73142c, 0 9px 18px rgba(161,18,54,.18) !important; }
.premium-primary button:hover, button.premium-primary:hover { transform: translateY(-1px); }
.premium-primary button:active, button.premium-primary:active { transform: translateY(2px); box-shadow: 0 2px 0 #73142c !important; }
.compact-button button, button.compact-button { min-height: 38px !important; padding: 6px 11px !important; }
.danger-button button, button.danger-button { font-weight: 700 !important; }
.log-tail textarea, .log-tail input { font-family: "JetBrains Mono", "Cascadia Code", Consolas, monospace !important; font-size: 12px !important; line-height: 1.45 !important; }
.progress-shell { border: 1px solid var(--block-border-color); border-radius: 6px; padding: 11px 12px; background: var(--background-fill-secondary); }
.progress-track { height: 9px; border-radius: 3px; background: #cbd5df; overflow: hidden; }
.dark .progress-track { background: #2a394b; }
.progress-fill { height: 100%; background: #a11236; transition: width .25s ease; }
.dark .progress-fill { background: #df5a78; }
.progress-metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 125px), 1fr)); gap: 8px; margin-top: 9px; }
.progress-metric { min-width: 0; }
.progress-metric strong { display: block; font-size: .74rem; color: #617185; font-weight: 600; }
.dark .progress-metric strong { color: #9bacbf; }
.progress-metric span { display: block; margin-top: 2px; font-size: .9rem; font-weight: 700; color: var(--body-text-color); overflow-wrap: anywhere; }
.status-ok { color: #106b57; font-weight: 700; }
.dark .status-ok { color: #7cdebd; }
.status-warn { color: #9a5b08; font-weight: 700; }
.dark .status-warn { color: #f4c46b; }
.status-error { color: #b4233f; font-weight: 700; }
.dark .status-error { color: #ff93aa; }
.summary-strip { padding: 10px 12px; border-left: 4px solid #10866a; background: var(--background-fill-secondary); border-radius: 4px; }
.stats-grid { display: grid; grid-template-columns: repeat(5, minmax(110px, 1fr)); gap: 8px; }
.stat-box { border-top: 3px solid #10866a; padding: 8px 2px 4px; }
.stat-box b { font-size: 1.05rem; }
.manager-table { font-size: .88rem; }
@media (max-width: 900px) {
  .progress-metrics, .stats-grid { grid-template-columns: repeat(auto-fit, minmax(min(100%, 125px), 1fr)); }
  .gradio-container { padding-left: 10px !important; padding-right: 10px !important; }
  .app-header { margin-left: -10px; margin-right: -10px; }
}

/* Let Vega axis labels extend into the block padding instead of being clipped on narrow layouts. */
.gradio-plot .vega-embed, .gradio-plot .vega-embed svg, .gradio-plot .vega-embed .chart-wrapper { overflow: visible !important; }
"""


CONFIRM_CANCEL_JS = "(value) => [window.confirm('Cancel the running job?')]"
CONFIRM_STOP_JS = "(value) => [window.confirm('Stop after the current optimizer step and save a resumable checkpoint?')]"
CONFIRM_FORCE_JS = "(value) => [window.confirm('Force stop immediately? Unsaved work since the last checkpoint will be lost.')]"
CONFIRM_DELETE_JS = "(value) => [window.confirm('Delete the selected item permanently?')]"


def format_exception(prefix: str, exc: BaseException) -> str:
    message = f"{prefix}: {type(exc).__name__}: {exc}"
    print(message, file=sys.stderr, flush=True)
    traceback.print_exc()
    return message


def tail_text(path: str | os.PathLike[str] | None, lines: int = 60) -> str:
    if not path:
        return ""
    try:
        with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
            return "".join(handle.readlines()[-max(1, int(lines)):]).rstrip()
    except OSError:
        return ""


def read_json(path: str | os.PathLike[str] | None, default: Any = None) -> Any:
    """Read a worker-written JSON file, tolerating a concurrent atomic rewrite."""

    return read_json_retry(path, default)


def output_task_is_active(task_folder: str | os.PathLike[str] | None) -> bool:
    """Return whether a generation task's atomic metadata still says it is running."""

    if not task_folder:
        return False
    metadata = read_json(Path(task_folder) / "metadata.json", {}) or {}
    status = str(metadata.get("status") or "").strip().lower()
    return bool(status and status not in OUTPUT_TASK_TERMINAL_STATUSES)


def latest_output_task(
    root: str | os.PathLike[str] = ROOT / "outputs",
    *,
    scope: str = "generation",
) -> str:
    """Find the newest real generation task for the single or batch dashboard."""

    output_root = Path(root).expanduser().resolve()
    if scope not in {"generation", "batch"}:
        raise ValueError(f"Unsupported output task scope: {scope}")
    candidates: list[tuple[float, Path]] = []
    for progress_path in output_root.rglob("progress.json") if output_root.is_dir() else []:
        task_folder = progress_path.parent
        metadata_path = task_folder / "metadata.json"
        request_path = task_folder / "request.json"
        if not metadata_path.is_file() or not request_path.is_file():
            continue
        try:
            parts = task_folder.resolve().relative_to(output_root).parts
        except (OSError, ValueError):
            continue
        if scope == "generation" and len(parts) != 1:
            continue
        if scope == "batch" and len(parts) < 2:
            continue
        lowered = [part.lower() for part in parts]
        if any(part.startswith((".", "_")) for part in parts):
            continue
        if any(part in {"training_runs", "worker_runtime_e2e", ".sample_jobs"} for part in lowered):
            continue
        first = lowered[0] if lowered else ""
        if first.startswith("ui_") and any(token in first for token in ("batch", "smoke")):
            continue
        try:
            modified = max(progress_path.stat().st_mtime, metadata_path.stat().st_mtime)
        except OSError:
            continue
        candidates.append((modified, task_folder.resolve()))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return str(candidates[0][1]) if candidates else ""


def adopt_output_task(
    displayed: str | os.PathLike[str] | None,
    *,
    root: str | os.PathLike[str] = ROOT / "outputs",
    scope: str = "generation",
    page_load: bool = False,
) -> tuple[str, bool]:
    """Keep an active card, or attach an idle/reloaded session to the newest task."""

    current = str(Path(displayed).expanduser().resolve()) if displayed else ""
    if current and output_task_is_active(current):
        return current, True
    newest = latest_output_task(root, scope=scope)
    if page_load and newest:
        return newest, output_task_is_active(newest)
    if newest and output_task_is_active(newest):
        return newest, True
    return current, False


def dedupe_updates(
    values: Sequence[Any], previous: list[str | None] | None = None
) -> tuple[tuple[Any, ...], list[str | None]]:
    """Replace unchanged generator outputs with ``gr.skip()`` updates."""

    old = list(previous or [])
    if len(old) < len(values):
        old.extend([None] * (len(values) - len(old)))
    signatures: list[str | None] = []
    outputs: list[Any] = []
    for index, value in enumerate(values):
        if isinstance(value, dict) and value == {"__type__": "update"}:
            signatures.append(old[index])
            outputs.append(value)
            continue
        try:
            if hasattr(value, "to_json") and callable(value.to_json):
                signature = value.to_json(orient="split", date_format="iso")
            else:
                signature = json.dumps(value, sort_keys=True, default=repr)
        except (TypeError, ValueError):
            signature = repr(value)
        signatures.append(signature)
        outputs.append(gr.skip() if old[index] == signature else value)
    return tuple(outputs), signatures


def write_json_atomic(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> Path:
    return _write_json_atomic(path, payload, indent=2, ensure_ascii=False)


def open_folder(path: str | os.PathLike[str]) -> str:
    folder = Path(path).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    try:
        if platform.system() == "Windows":
            os.startfile(str(folder))  # type: ignore[attr-defined]
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(folder)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.Popen(["xdg-open", str(folder)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        message = f"Opened {folder}"
    except Exception as exc:
        message = f"Could not open {folder}: {exc}"
    print(message, flush=True)
    return message


def _process_flags() -> dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}


def _terminate_process_tree(process: subprocess.Popen[Any] | None) -> bool:
    """Kill a child and all descendants on Windows and POSIX."""

    if process is None or process.poll() is not None:
        return False
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception:
        try:
            process.kill()
        except Exception:
            return False
    return True


@dataclass
class ChildJob:
    kind: str
    process: subprocess.Popen[str]
    state_dir: Path
    log_path: Path
    started_at: float = field(default_factory=time.monotonic)
    metadata: dict[str, Any] = field(default_factory=dict)
    output_lines: list[str] = field(default_factory=list)
    canceled: bool = False

    @property
    def running(self) -> bool:
        return self.process.poll() is None


class ProcessManager:
    def __init__(self) -> None:
        self._jobs: dict[str, ChildJob] = {}
        self._lock = threading.RLock()

    def get(self, kind: str) -> ChildJob | None:
        with self._lock:
            return self._jobs.get(kind)

    def start(
        self,
        kind: str,
        command: Sequence[str],
        *,
        state_dir: str | os.PathLike[str],
        log_path: str | os.PathLike[str] | None = None,
        cwd: str | os.PathLike[str] = ROOT,
        env: Mapping[str, str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ChildJob:
        with self._lock:
            existing = self._jobs.get(kind)
            if existing is not None and existing.running:
                raise RuntimeError(f"A {kind.replace('_', ' ')} job is already running")
            state = Path(state_dir).expanduser().resolve()
            state.mkdir(parents=True, exist_ok=True)
            log = Path(log_path).expanduser().resolve() if log_path else state / "ui_console.log"
            child_env = dict(os.environ)
            child_env.update({"PYTHONUNBUFFERED": "1"})
            if env:
                child_env.update({str(key): str(value) for key, value in env.items()})
            print(f">> Starting {kind}: {' '.join(map(str, command))}", flush=True)
            process = subprocess.Popen(
                [str(item) for item in command],
                cwd=str(cwd),
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                **_process_flags(),
            )
            job = ChildJob(kind, process, state, log, metadata=dict(metadata or {}))
            self._jobs[kind] = job
            thread = threading.Thread(target=self._pump, args=(job,), daemon=True, name=f"{kind}-console")
            thread.start()
            return job

    def _pump(self, job: ChildJob) -> None:
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with job.log_path.open("a", encoding="utf-8", newline="\n") as handle:
                stream = job.process.stdout
                if stream is not None:
                    for line in iter(stream.readline, ""):
                        if not line:
                            break
                        print(line.rstrip(), flush=True)
                        handle.write(line)
                        handle.flush()
                        with self._lock:
                            job.output_lines.append(line.rstrip())
                            if len(job.output_lines) > 200:
                                del job.output_lines[:-200]
        finally:
            elapsed = time.monotonic() - job.started_at
            try:
                returncode = job.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                returncode = job.process.poll()
            print(f">> {job.kind} finished with code {returncode} in {elapsed:.2f}s", flush=True)

    def terminate(self, kind: str) -> bool:
        with self._lock:
            job = self._jobs.get(kind)
            if job is None:
                return False
            job.canceled = True
            return _terminate_process_tree(job.process)

    def terminate_all(self) -> None:
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            if job.running:
                job.canceled = True
                _terminate_process_tree(job.process)


PROCESS_MANAGER = ProcessManager()
atexit.register(PROCESS_MANAGER.terminate_all)


class LazyEngine:
    """Single lazily-created in-process engine, reloaded when runtime settings change."""

    def __init__(self) -> None:
        self._instance: Any = None
        self._fingerprint = ""
        self._lock = threading.RLock()

    def get(
        self,
        runtime_options: Mapping[str, Any],
        *,
        progress_file: str | os.PathLike[str] | None = None,
        progress_callback: Callable[..., Any] | None = None,
    ) -> Any:
        from webui_generation_runner import create_tts

        fingerprint_options = dict(runtime_options)
        for key in ("lora_path", "lora_strength", "lora_merge_into_base"):
            fingerprint_options.pop(key, None)
        nested_runtime = fingerprint_options.get("runtime")
        if isinstance(nested_runtime, Mapping):
            nested_runtime = dict(nested_runtime)
            for key in ("lora_path", "lora_strength", "lora_merge_into_base"):
                nested_runtime.pop(key, None)
            fingerprint_options["runtime"] = nested_runtime
        fingerprint = json.dumps(fingerprint_options, sort_keys=True, default=str)
        with self._lock:
            if self._instance is not None and fingerprint != self._fingerprint:
                self.unload()
            if self._instance is None:
                started = time.perf_counter()
                print(">> Lazy model load started", flush=True)
                self._instance = create_tts(
                    dict(runtime_options),
                    progress_file=str(progress_file) if progress_file else None,
                    progress_callback=progress_callback,
                )
                self._fingerprint = fingerprint
                print(f">> Lazy model load finished in {time.perf_counter() - started:.2f}s", flush=True)
            return self._instance

    def peek(self) -> Any:
        with self._lock:
            return self._instance

    def request_cancel(self) -> bool:
        instance = self.peek()
        reporter = getattr(instance, "progress_reporter", None) if instance is not None else None
        if reporter is None:
            return False

        def abort(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("Generation canceled by user")

        reporter.update = abort
        reporter.finish = abort
        return True

    def unload(self) -> bool:
        with self._lock:
            instance = self._instance
            self._instance = None
            self._fingerprint = ""
        if instance is None:
            return False
        try:
            unload = getattr(instance, "unload", None)
            if callable(unload):
                unload()
        except Exception:
            traceback.print_exc()
        del instance
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            pass
        print(">> In-process model unloaded and VRAM cache released", flush=True)
        return True


LAZY_ENGINE = LazyEngine()


def progress_panel_html(payload: Mapping[str, Any] | None, *, title: str = "Ready") -> str:
    value = dict(payload or {})
    fraction = max(0.0, min(1.0, float(value.get("fraction", 0.0) or 0.0)))
    completed = int(value.get("completed", value.get("step", value.get("file_i", 0))) or 0)
    total_value = value.get("total", value.get("total_steps", value.get("file_n", 0)))
    total = int(total_value or 0)
    elapsed = value.get("elapsed_s", value.get("elapsed", 0.0))
    eta = value.get("eta_s", value.get("eta"))
    speed = value.get("speed", value.get("it_s"))
    speed_unit = str(value.get("speed_unit") or ("it/s" if value.get("it_s") is not None else "x RT"))
    vram_used = float(value.get("vram_used_gb", 0.0) or 0.0)
    vram_total = float(value.get("vram_total_gb", 0.0) or 0.0)
    desc = str(value.get("desc") or value.get("message") or value.get("stage") or title)
    count = f"{completed}/{total}" if total else str(completed)
    metrics = [
        ("ITEM", count),
        ("ELAPSED", format_duration(elapsed)),
        ("ETA", format_duration(eta)),
        ("SPEED", format_rate(speed, speed_unit)),
    ]
    if vram_used > 0.0 or vram_total > 0.0:
        metrics.append(("VRAM", f"{vram_used:.2f}/{vram_total:.2f} GB"))
    metrics.append(("CURRENT", desc))
    metric_html = "".join(
        '<div class="progress-metric"><strong>'
        + html.escape(label)
        + "</strong><span>"
        + html.escape(str(metric_value))
        + "</span></div>"
        for label, metric_value in metrics
    )
    return f"""
    <div class="progress-shell" role="status" aria-live="polite">
      <div><strong>{html.escape(title)}</strong> <span style="float:right">{fraction * 100:.1f}%</span></div>
      <div class="progress-track" aria-label="Progress" aria-valuenow="{fraction * 100:.1f}"><div class="progress-fill" style="width:{fraction * 100:.2f}%"></div></div>
      <div class="progress-metrics">{metric_html}</div>
    </div>
    """


def progress_from_file(path: str | os.PathLike[str] | None, title: str) -> tuple[str, dict[str, Any]]:
    payload = read_progress_file(path) or {}
    return progress_panel_html(payload, title=title), payload


def stats_html(values: Mapping[str, Any] | None) -> str:
    data = dict(values or {})
    fields = (
        ("Segments", data.get("segment_count", data.get("segments_count", 0))),
        ("Minutes", f"{float(data.get('total_duration_s', data.get('audio_seconds', 0.0)) or 0.0) / 60.0:.2f}"),
        ("Mean", f"{float(data.get('mean_duration_s', 0.0) or 0.0):.2f}s"),
        ("Range", f"{float(data.get('min_duration_s', 0.0) or 0.0):.2f}-{float(data.get('max_duration_s', 0.0) or 0.0):.2f}s"),
        ("Words", data.get("word_count", data.get("words", 0))),
    )
    cells = "".join(
        f'<div class="stat-box"><span>{html.escape(label)}</span><br><b>{html.escape(str(value))}</b></div>'
        for label, value in fields
    )
    return f'<div class="stats-grid">{cells}</div>'


def parse_multiline_paths(value: str | None) -> list[str]:
    result: list[str] = []
    for line in str(value or "").replace(";", "\n").splitlines():
        item = line.strip().strip('"')
        if item and item not in result:
            result.append(item)
    return result


def extract_reference_audio(
    media_path: str | os.PathLike[str],
    time_ranges: str = "",
    *,
    require_ranges: bool = False,
    sample_rate: int = 24000,
) -> tuple[str | None, str]:
    source = Path(media_path).expanduser() if media_path else None
    if source is None or not source.is_file():
        return None, "Choose an existing audio or video file."
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None, "FFmpeg is required to read reference media."
    ranges: list[tuple[float, float]] = []
    for raw in str(time_ranges or "").split(";"):
        raw = raw.strip()
        if not raw:
            continue
        try:
            start_text, end_text = raw.split(":", 1)
            start, end = float(start_text), float(end_text)
        except (ValueError, TypeError):
            continue
        if start >= 0 and end > start:
            ranges.append((start, end))
    if require_ranges and not ranges:
        return None, "Enter ranges such as 1:3; 4.5:9 before extracting."
    destination = Path(tempfile.mkstemp(prefix="indextts_reference_", suffix=".wav")[1])
    try:
        if not ranges:
            command = [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
                "-ar", str(sample_rate), "-ac", "1", "-c:a", "pcm_s16le", str(destination),
            ]
        else:
            inputs: list[str] = []
            filters: list[str] = []
            for index, (start, end) in enumerate(ranges):
                inputs.extend(["-i", str(source)])
                filters.append(
                    f"[{index}:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[s{index}]"
                )
            labels = "".join(f"[s{index}]" for index in range(len(ranges)))
            filters.append(f"{labels}concat=n={len(ranges)}:v=0:a=1[out]")
            command = [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error", *inputs,
                "-filter_complex", ";".join(filters), "-map", "[out]",
                "-ar", str(sample_rate), "-ac", "1", "-c:a", "pcm_s16le", str(destination),
            ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0 or not destination.is_file():
            destination.unlink(missing_ok=True)
            return None, f"Reference extraction failed: {(completed.stderr or '').strip()[-800:]}"
        detail = f" using {len(ranges)} selected range(s)" if ranges else ""
        return str(destination), f"Loaded {source.name}{detail}."
    except Exception as exc:
        destination.unlink(missing_ok=True)
        return None, f"Reference extraction failed: {exc}"


def resolve_path_value(value: Any) -> str | None:
    if not value:
        return None
    if isinstance(value, Mapping):
        value = value.get("path") or value.get("name")
    elif hasattr(value, "path"):
        value = value.path
    elif hasattr(value, "name"):
        value = value.name
    text = str(value or "").strip()
    return text or None


def runtime_config_from_values(values: Mapping[str, Any], *, model_dir: str = "models") -> dict[str, Any]:
    aux_names = ("semantic_model", "qwen_emo", "campplus", "semantic_codec", "s2mel", "bigvgan")
    payload = {
        "device": values.get("runtime.device", "auto"),
        "model_variant": values.get("runtime.model_variant", "bf16"),
        "gpt_dtype": values.get("runtime.gpt_dtype", "bf16"),
        "blocks_to_swap": values.get("runtime.blocks_to_swap", 0),
        "swap_ring_size": values.get("runtime.swap_ring_size", 2),
        "pin_swap_memory": values.get("runtime.pin_swap_memory", True),
        "aux_residency": {
            name: values.get(f"runtime.aux_residency.{name}", RuntimeConfig().aux_residency[name])
            for name in aux_names
        },
        "attention_backend": values.get("runtime.attention_backend", "sdpa"),
        "use_accel": values.get("runtime.use_accel", False),
        "torch_compile_s2mel": values.get("runtime.torch_compile_s2mel", False),
        "use_cuda_kernel_bigvgan": values.get("runtime.use_cuda_kernel_bigvgan", False),
        "s2mel_estimator_autocast": values.get("runtime.s2mel_estimator_autocast", False),
        "cfm_cache_length": values.get("runtime.cfm_cache_length", 8192),
        "vram_reserve_gb": values.get("runtime.vram_reserve_gb", 2.0),
        "vram_tier": values.get("runtime.vram_tier", "auto"),
        "lora_path": values.get("runtime.lora_path", ""),
        "lora_strength": values.get("runtime.lora_strength", 1.0),
        "lora_merge_into_base": values.get("runtime.lora_merge_into_base", False),
        "max_section_batch_size_hint": values.get("runtime.max_section_batch_size_hint", 8),
    }
    config = RuntimeConfig.from_dict(payload).to_dict()
    resolved_model_dir = str(Path(model_dir).expanduser().resolve())
    config.update(
        {
            "model_dir": resolved_model_dir,
            "cfg_path": str(Path(resolved_model_dir) / "config.yaml"),
            "use_qwen_emo": bool(values.get("runtime.use_qwen_emo", True)),
            "use_deepspeed": bool(values.get("runtime.use_deepspeed", False)),
        }
    )
    return config


__all__ = [
    "APP_CSS",
    "APP_HEAD",
    "APP_TITLE",
    "APP_VERSION",
    "OUTPUT_TASK_TERMINAL_STATUSES",
    "adopt_output_task",
    "dedupe_updates",
    "CONFIRM_CANCEL_JS",
    "CONFIRM_DELETE_JS",
    "CONFIRM_FORCE_JS",
    "CONFIRM_STOP_JS",
    "FAVICON_PATH",
    "LAZY_ENGINE",
    "PROCESS_MANAGER",
    "PremiumTheme",
    "ROOT",
    "STATE_ROOT",
    "_terminate_process_tree",
    "extract_reference_audio",
    "format_exception",
    "latest_output_task",
    "open_folder",
    "output_task_is_active",
    "parse_multiline_paths",
    "progress_from_file",
    "progress_panel_html",
    "read_json",
    "resolve_path_value",
    "runtime_config_from_values",
    "stats_html",
    "tail_text",
    "write_json_atomic",
]
