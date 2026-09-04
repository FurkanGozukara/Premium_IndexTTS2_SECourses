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
import re
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


def app_theme() -> gr.themes.Base:
    """Return the stock Gradio 6 ``Origin`` theme, used exactly as shipped.

    Every colour, radius, shadow and font in the interface comes from this
    theme.  The stylesheet below adds no palette of its own beyond the button
    colours; it is written against the theme's own CSS variables so light and
    dark modes stay in sync automatically.
    """

    return gr.themes.Origin()


# Dark is the default the first time the app is opened.  The choice is stored in
# localStorage and mirrored into the ``__theme`` query parameter so a reload or a
# bookmark restores it before Gradio paints the page.
APP_HEAD = """
<meta name="color-scheme" content="dark light">
<script>
(function () {
  var KEY = "indextts.theme";
  function resolve() {
    var url = new URL(window.location.href);
    var param = url.searchParams.get("__theme");
    var stored = null;
    try { stored = window.localStorage.getItem(KEY); } catch (e) {}
    var mode = param || stored || "dark";
    if (mode !== "light") { mode = "dark"; }
    try { window.localStorage.setItem(KEY, mode); } catch (e) {}
    if (param !== mode) {
      url.searchParams.set("__theme", mode);
      window.history.replaceState(null, "", url.toString());
    }
    return mode;
  }
  function paint(mode) {
    if (!document.body) { return false; }
    document.body.classList.toggle("dark", mode === "dark");
    return true;
  }
  var mode = resolve();
  if (!paint(mode)) {
    document.addEventListener("DOMContentLoaded", function () { paint(mode); });
  }
})();
</script>
"""


# Switching themes only swaps the ``dark`` class Gradio itself keys off, so it is
# instant: no reload and no round trip to the server.
TOGGLE_THEME_JS = """
() => {
  const dark = !document.body.classList.contains("dark");
  document.body.classList.toggle("dark", dark);
  const mode = dark ? "dark" : "light";
  try { window.localStorage.setItem("indextts.theme", mode); } catch (e) {}
  const url = new URL(window.location.href);
  url.searchParams.set("__theme", mode);
  window.history.replaceState(null, "", url.toString());
}
"""


# Expand or collapse every accordion on the tab that is currently on screen.
TOGGLE_SECTIONS_JS = """
async () => {
  const tabs = document.querySelector("#main-tabs");
  const panel = tabs
    ? Array.from(tabs.querySelectorAll(":scope > .tabitem")).find(
        (item) => item.style.display !== "none"
      )
    : null;
  const scope = panel || document;
  const heads = () => Array.from(scope.querySelectorAll("button.label-wrap"));
  const first = heads();
  if (!first.length) { return; }
  const expand = first.some((head) => !head.classList.contains("open"));
  // A nested accordion only reaches the DOM once its parent is open, so keep
  // sweeping until nothing is left in the wrong state.
  for (let pass = 0; pass < 6; pass++) {
    const pending = heads().filter(
      (head) => head.classList.contains("open") !== expand
    );
    if (!pending.length) { break; }
    pending.forEach((head) => head.click());
    await new Promise((resolve) =>
      requestAnimationFrame(() => requestAnimationFrame(resolve))
    );
  }
}
"""


# Every action button gets its own hue as a (deep, mid, bright) triple.  The
# gradient runs deep -> mid -> bright so the face has depth, the glow is built
# from the mid tone, and white text stays readable on all three stops.
BUTTON_HUES: dict[str, tuple[str, str, str]] = {
    "emerald": ("#065f46", "#059669", "#34d399"),
    "green":   ("#166534", "#16a34a", "#4ade80"),
    "lime":    ("#3f6212", "#65a30d", "#a3e635"),
    "teal":    ("#115e59", "#0d9488", "#2dd4bf"),
    "cyan":    ("#155e75", "#0891b2", "#22d3ee"),
    "sky":     ("#075985", "#0284c7", "#38bdf8"),
    "blue":    ("#1e40af", "#2563eb", "#60a5fa"),
    "indigo":  ("#3730a3", "#4f46e5", "#818cf8"),
    "violet":  ("#5b21b6", "#7c3aed", "#a78bfa"),
    "purple":  ("#6b21a8", "#9333ea", "#c084fc"),
    "fuchsia": ("#86198f", "#c026d3", "#e879f9"),
    "pink":    ("#9d174d", "#db2777", "#f9a8d4"),
    "rose":    ("#9f1239", "#e11d48", "#fda4af"),
    "red":     ("#991b1b", "#dc2626", "#f87171"),
    "crimson": ("#4c0519", "#9f1239", "#fb7185"),
    "orange":  ("#9a3412", "#ea580c", "#fb923c"),
    "amber":   ("#92400e", "#d97706", "#fbbf24"),
    "bronze":  ("#5c3a21", "#8b5a2b", "#d4a373"),
    "slate":   ("#334155", "#475569", "#94a3b8"),
    "gray":    ("#3f3f46", "#52525b", "#a1a1aa"),
}
BUTTON_COLORS = tuple(BUTTON_HUES)


def btn(color: str, *extra: str) -> list[str]:
    """Build the ``elem_classes`` list for a coloured action button.

    ``color`` picks one of :data:`BUTTON_COLORS`.  Every button in the app is
    the same size; only the hue changes, so a page never reads as a wall of
    identical controls.
    """

    if color not in BUTTON_HUES:
        raise ValueError(f"Unknown button colour: {color}")
    return ["ax", f"ax-{color}", *extra]


def _rgb(value: str) -> tuple[int, int, int]:
    digits = value.lstrip("#")
    return tuple(int(digits[index: index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]


def _button_palette_css() -> str:
    """Generate the gradient, border and glow rules for every button hue."""

    rules: list[str] = []
    for name, (deep, mid, bright) in BUTTON_HUES.items():
        red, green, blue = _rgb(mid)
        bright_rgb = ", ".join(str(part) for part in _rgb(bright))
        rules.append(
            f"""
button.ax-{name} {{
  background: linear-gradient(135deg, {deep} 0%, {mid} 55%, {bright} 100%) !important;
  border-color: rgba({bright_rgb}, .72) !important;
  box-shadow: 0 8px 20px rgba({red}, {green}, {blue}, .30), inset 0 1px 0 rgba(255, 255, 255, .20) !important;
}}
button.ax-{name}:hover:not(:disabled) {{
  border-color: rgba({bright_rgb}, .98) !important;
  box-shadow: 0 12px 26px rgba({red}, {green}, {blue}, .44), inset 0 1px 0 rgba(255, 255, 255, .28) !important;
}}
body:not(.dark) button.ax-{name} {{
  background: linear-gradient(135deg, {deep} 0%, {mid} 66%, {bright} 100%) !important;
  border-color: {mid} !important;
  box-shadow: 0 7px 17px rgba({red}, {green}, {blue}, .26), inset 0 1px 0 rgba(255, 255, 255, .26) !important;
}}
body:not(.dark) button.ax-{name}:hover:not(:disabled) {{
  box-shadow: 0 11px 24px rgba({red}, {green}, {blue}, .38), inset 0 1px 0 rgba(255, 255, 255, .32) !important;
}}"""
        )
    return "\n".join(rules)


_BASE_CSS = r"""
/* --------------------------------------------------------------------- *
 * Accent tokens for everything that is not a button.  Only these two
 * blocks know about light and dark; the rest of the sheet reads them.
 * --------------------------------------------------------------------- */
:root {
  --ax-ok: #047857;
  --ax-warn: #b45309;
  --ax-error: #be123c;
  --ax-accent: #0f766e;
}
:root.dark, :root .dark {
  --ax-ok: #10b981;
  --ax-warn: #f59e0b;
  --ax-error: #fb7185;
  --ax-accent: #14b8a6;
}

/* --------------------------------------------------------------------- *
 * Action buttons.  Every one is the same height, weight and type size so
 * rows of controls share a baseline; only the hue changes, and the hue is
 * generated per colour further down.
 * --------------------------------------------------------------------- */
button.ax {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--size-2);
  min-height: 44px;
  padding: var(--size-2) var(--size-4) !important;
  border-width: 1px !important;
  border-style: solid !important;
  border-radius: var(--radius-lg) !important;
  color: #f8fafc !important;
  font-size: var(--text-md) !important;
  font-weight: 650 !important;
  line-height: 1.25 !important;
  text-align: center;
  text-shadow: 0 1px 2px rgba(2, 6, 23, .45) !important;
  transition: transform 140ms ease, filter 140ms ease, box-shadow 140ms ease;
}
button.ax:hover:not(:disabled) { transform: translateY(-1px); filter: brightness(1.05); }
button.ax:active:not(:disabled) { transform: translateY(1px); filter: brightness(.96); }
button.ax:focus-visible { outline: 2px solid #bae6fd; outline-offset: 2px; }
button.ax:disabled { filter: grayscale(.45) opacity(.62); transform: none; box-shadow: none !important; }
/* Inside a row a button would otherwise stretch to the tallest neighbour, which
   is the one place the uniform height breaks down. */
.row > button.ax { align-self: center; }

/* --------------------------------------------------------------------- *
 * Page furniture
 * --------------------------------------------------------------------- */
.app-header {
  align-items: center;
  flex-wrap: nowrap;
  gap: var(--size-4);
  padding-bottom: var(--size-3);
  border-bottom: 1px solid var(--border-color-primary);
}
.app-header > :first-child { flex: 1 1 auto; min-width: 0; }
.app-header h1 { margin: 0 !important; line-height: 1.2; }
.app-header p { margin: var(--size-1) 0 0 !important; color: var(--body-text-color-subdued); }
/* Gradio's own row rule is scoped, so the header strip has to out-specify it. */
.row.header-actions {
  flex: 0 0 auto !important;
  width: auto !important;
  min-width: 0 !important;
  flex-wrap: nowrap;
  justify-content: flex-end;
  gap: var(--size-2);
}
.header-actions button.ax { flex: 0 0 auto; white-space: nowrap; }
/* Line the preset buttons up with the fields they act on rather than with the
   whole labelled block. */
.preset-bar { align-items: flex-end; gap: var(--size-2); }
.row.preset-bar > button.ax { align-self: flex-end !important; margin-bottom: 12px; }
.preset-status { min-height: var(--size-6); font-size: var(--text-sm); color: var(--body-text-color-subdued); }
.section-note { color: var(--body-text-color-subdued); font-size: var(--text-sm); }
.log-tail textarea, .log-tail input { font-family: var(--font-mono) !important; font-size: var(--text-sm) !important; line-height: 1.5 !important; }
.manager-table { font-size: var(--text-sm); }
.help-prose { max-width: 1180px; }

/* --------------------------------------------------------------------- *
 * Job progress card, status words and dataset statistics
 * --------------------------------------------------------------------- */
.progress-shell {
  border: 1px solid var(--block-border-color);
  border-radius: var(--radius-lg);
  padding: var(--size-3);
  background: var(--background-fill-secondary);
}
.progress-track {
  height: 8px;
  margin-top: var(--size-2);
  border-radius: var(--radius-full);
  background: var(--border-color-primary);
  overflow: hidden;
}
.progress-fill {
  height: 100%;
  border-radius: var(--radius-full);
  background: linear-gradient(90deg, #2563eb, #14b8a6, #22c55e);
  transition: width 250ms ease;
}
.progress-metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 128px), 1fr));
  gap: var(--size-2);
  margin-top: var(--size-3);
}
.progress-metric { min-width: 0; }
.progress-metric strong {
  display: block;
  font-size: var(--text-xs);
  font-weight: 600;
  letter-spacing: .04em;
  color: var(--body-text-color-subdued);
}
.progress-metric span {
  display: block;
  margin-top: 2px;
  font-size: var(--text-md);
  font-weight: 700;
  color: var(--body-text-color);
  overflow-wrap: anywhere;
}
.status-ok { color: var(--ax-ok); font-weight: 700; }
.status-warn { color: var(--ax-warn); font-weight: 700; }
.status-error { color: var(--ax-error); font-weight: 700; }
.summary-strip {
  padding: var(--size-2) var(--size-3);
  border-left: 4px solid var(--ax-accent);
  border-radius: var(--radius-sm);
  background: var(--background-fill-secondary);
}
.summary-strip.status-error { border-left-color: var(--ax-error); }
.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 128px), 1fr)); gap: var(--size-2); }
.stat-box { padding: var(--size-2) 0 var(--size-1); border-top: 3px solid var(--ax-accent); }
.stat-box span { font-size: var(--text-xs); color: var(--body-text-color-subdued); }
.stat-box b { font-size: var(--text-lg); }

/* Let Vega axis labels extend into the block padding instead of being clipped on narrow layouts. */
.gradio-plot .vega-embed, .gradio-plot .vega-embed svg, .gradio-plot .vega-embed .chart-wrapper { overflow: visible !important; }

@media (max-width: 900px) {
  .app-header { flex-wrap: wrap; }
  .row.header-actions {
    flex: 1 1 100% !important;
    width: 100% !important;
    flex-wrap: wrap;
    justify-content: flex-start;
  }
  .header-actions button.ax {
    flex: 1 1 180px;
    width: auto !important;
    min-width: 0 !important;
    white-space: normal;
  }
}
"""


APP_CSS = _BASE_CSS + _button_palette_css() + "\n"


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
        if any(part in {"grids", "training_runs", "worker_runtime_e2e", ".sample_jobs"} for part in lowered):
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


def _parse_media_timestamp(value: str) -> float:
    text = str(value or "").strip()
    if not text:
        raise ValueError("timestamp is empty")
    parts = text.split(":")
    if len(parts) > 3:
        raise ValueError(f"invalid timestamp '{text}'")
    try:
        numbers = [float(part.strip()) for part in parts]
    except ValueError as exc:
        raise ValueError(f"invalid timestamp '{text}'") from exc
    if any(number < 0 for number in numbers):
        raise ValueError("timestamps cannot be negative")
    if len(numbers) > 1 and numbers[-1] >= 60:
        raise ValueError(f"seconds must be below 60 in timestamp '{text}'")
    if len(numbers) == 3 and numbers[-2] >= 60:
        raise ValueError(f"minutes must be below 60 in timestamp '{text}'")
    seconds = 0.0
    for number in numbers:
        seconds = seconds * 60.0 + number
    return seconds


def parse_reference_time_ranges(value: str | None) -> list[tuple[float, float]]:
    """Parse ordered reference-media ranges.

    The compact ``start:end`` form treats both values as seconds. Timestamp
    ranges use a dash or arrow, for example ``01:02-01:08.5``.
    """

    ranges: list[tuple[float, float]] = []
    for raw in re.split(r"[;\n]+", str(value or "")):
        item = raw.strip()
        if not item:
            continue
        compact = re.fullmatch(
            r"([0-9]+(?:\.[0-9]+)?)\s*:\s*([0-9]+(?:\.[0-9]+)?)",
            item,
        )
        if compact:
            start = float(compact.group(1))
            end = float(compact.group(2))
        else:
            match = re.fullmatch(r"(.+?)\s*(?:->|-)\s*(.+)", item)
            if not match:
                raise ValueError(
                    f"Invalid range '{item}'. Use start:end (seconds) or MM:SS-MM:SS."
                )
            start = _parse_media_timestamp(match.group(1))
            end = _parse_media_timestamp(match.group(2))
        if end <= start:
            raise ValueError(f"Range '{item}' must end after it starts.")
        ranges.append((start, end))
    return ranges


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
    try:
        ranges = parse_reference_time_ranges(time_ranges)
    except ValueError as exc:
        return None, str(exc)
    if require_ranges and not ranges:
        return None, "Enter ranges such as 1:3; 4.5:9 or 01:02-01:08 before extracting."
    with tempfile.NamedTemporaryFile(
        prefix="indextts_reference_", suffix=".wav", delete=False
    ) as temporary:
        destination = Path(temporary.name)
    try:
        if not ranges:
            command = [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
                "-map", "0:a:0", "-vn", "-ar", str(sample_rate), "-ac", "1",
                "-c:a", "pcm_s16le", str(destination),
            ]
        else:
            filters: list[str] = []
            if len(ranges) > 1:
                split_labels = "".join(f"[src{index}]" for index in range(len(ranges)))
                filters.append(f"[0:a:0]asplit={len(ranges)}{split_labels}")
            for index, (start, end) in enumerate(ranges):
                source_label = f"src{index}" if len(ranges) > 1 else "0:a:0"
                filters.append(
                    f"[{source_label}]atrim=start={start:.6f}:end={end:.6f},"
                    f"asetpts=PTS-STARTPTS[s{index}]"
                )
            labels = "".join(f"[s{index}]" for index in range(len(ranges)))
            filters.append(f"{labels}concat=n={len(ranges)}:v=0:a=1[out]")
            command = [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
                "-filter_complex", ";".join(filters), "-map", "[out]",
                "-ar", str(sample_rate), "-ac", "1", "-c:a", "pcm_s16le", str(destination),
            ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        has_samples = False
        if completed.returncode == 0 and destination.is_file():
            try:
                import wave

                with wave.open(str(destination), "rb") as output_wav:
                    has_samples = output_wav.getnframes() > 0
            except (OSError, wave.Error):
                has_samples = False
        if completed.returncode != 0 or not has_samples:
            destination.unlink(missing_ok=True)
            detail = (completed.stderr or "").strip()[-800:]
            if not detail:
                detail = "the selected media or ranges produced no audio"
            return None, f"Reference extraction failed: {detail}"
        detail = ""
        if ranges:
            total_seconds = sum(end - start for start, end in ranges)
            noun = "range" if len(ranges) == 1 else "ranges"
            detail = (
                f" using {len(ranges)} selected {noun} "
                f"({total_seconds:.2f}s total)"
            )
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
    "ROOT",
    "STATE_ROOT",
    "TOGGLE_SECTIONS_JS",
    "TOGGLE_THEME_JS",
    "_terminate_process_tree",
    "app_theme",
    "btn",
    "extract_reference_audio",
    "format_exception",
    "latest_output_task",
    "open_folder",
    "output_task_is_active",
    "parse_multiline_paths",
    "parse_reference_time_ranges",
    "progress_from_file",
    "progress_panel_html",
    "read_json",
    "resolve_path_value",
    "runtime_config_from_values",
    "stats_html",
    "tail_text",
    "write_json_atomic",
]
