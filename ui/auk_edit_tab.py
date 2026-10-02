"""AuK audio editing: content, acoustic and paralinguistic edits, enhancement and separation.

The tab exists only for AuK (its button and content follow the speech-model body
class). Task fields are shown by a body class too, so changing the task needs no
server round trip; the instruction preview is rendered by the shared task catalog.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gradio as gr

from indextts.auk.tasks import (EFFECTS, EMOTIONS, ENHANCE, FIELD_DEFAULTS, GAINS_DB, MUSIC, NONVERBAL_EVENTS,
                                SEMITONES, SPEEDS, TASKS, render_instruction)

from .common import LAZY_ENGINE, bind_gathered_api, btn, on_gathered, runtime_config_from_values, write_json_atomic
from .presets_store import PresetRegistry

ROOT = Path(__file__).resolve().parents[1]
TAB_ID = "auk-editing"
# The task's fields, shown by the body class "auk-task-<task>" (set in the browser).
FIELD_LABELS = {
    "orig": "Original words", "new": "New words", "text": "Words to add", "anchor": "Anchor words",
    "target": "Words to remove", "direction": "Direction", "semitones": "Semitones", "speed": "Speed",
    "gain_db": "Change (dB)", "emotion": "Emotion", "description": "Target voice description", "sound": "Sound",
    "position": "Position", "speaker": "Speaker by talking order", "spoken": "Words the speaker says",
    "enhance": "Clean up", "effect": "Effect to repair", "music": "Keep", "instruction": "Instruction",
}
TASK_CLASS_JS = ("(task) => { document.body.classList.forEach(name => { if (name.startsWith('auk-task-')) "
                 "document.body.classList.remove(name); }); document.body.classList.add('auk-task-' + task); }")


def task_fields_css() -> str:
    """Hide every task field unless the active task uses it; the tab itself only shows for AuK."""
    rules = []
    for name in FIELD_LABELS:
        users = [key for key, task in TASKS.items() if name in task.fields]
        selector = "body" + "".join(f":not(.auk-task-{key})" for key in users) + f" .auk-field-{name}"
        rules.append(selector)
    rules.append("body:not(.auk-task-replace):not(.auk-task-insert_before):not(.auk-task-insert_after)"
                 ":not(.auk-task-remove):not(.auk-task-remove_before):not(.auk-task-remove_after)"
                 ":not(.auk-task-lyrics) .auk-field-transcript")
    rules.append(f'body:not(.tts-model-auk) button[data-tab-id="{TAB_ID}"]')
    return ",\n".join(rules) + " { display: none !important; }\n"


@dataclass
class AukEditTab:
    controls: dict[str, Any] = field(default_factory=dict)
    source: Any = None
    run_button: Any = None
    cancel_button: Any = None
    output: Any = None
    status: Any = None
    preview: Any = None
    block: Any = None


def _register(registry, key, component, **kwargs):
    registry.register(key, component, component.value, **kwargs)
    return component


def build_auk_edit_tab(args: Any, registry: PresetRegistry) -> AukEditTab:
    tab = AukEditTab()
    c = tab.controls
    with gr.Tab("🎛️ AuK Audio Editing", id=TAB_ID, elem_classes=["tts-only-auk"]) as tab.block:
        gr.Markdown("### Edit, restore or separate a recording with AuK\nChoose a task and fill in its fields; "
                    "the instruction below is exactly what AuK receives. Content, lyric and nonverbal edits work on "
                    "up to 30 seconds; clean-up, effects, pitch, volume, emotion and voice tasks also run on longer "
                    "recordings, piece by piece at pauses.")
        with gr.Row():
            with gr.Column(scale=1, min_width=380):
                tab.source = gr.Audio(label="Source audio", type="filepath", sources=["upload", "microphone"])
                c["auk_edit.task"] = _register(registry, "auk_edit.task", gr.Dropdown(
                    choices=[(task.label, key) for key, task in TASKS.items()], value="enhance", label="Task"),
                    kind="choice", choices=list(TASKS))
                with gr.Group():
                    def add(name, component, kind="auto", **kwargs):
                        classes = ["auk-edit-field", f"auk-field-{name}"]
                        component.elem_classes = classes
                        c[f"auk_edit.{name}"] = _register(registry, f"auk_edit.{name}", component, kind=kind, **kwargs)

                    with gr.Row():
                        add("orig", gr.Textbox(value="", label=FIELD_LABELS["orig"], placeholder="words as spoken"), "str")
                        add("new", gr.Textbox(value="", label=FIELD_LABELS["new"]), "str")
                    with gr.Row():
                        add("text", gr.Textbox(value="", label=FIELD_LABELS["text"]), "str")
                        add("target", gr.Textbox(value="", label=FIELD_LABELS["target"]), "str")
                        add("anchor", gr.Textbox(value="", label=FIELD_LABELS["anchor"],
                                                 info="Words next to the change, exactly as spoken."), "str")
                    with gr.Row():
                        add("direction", gr.Radio([("Raise / increase", "raise"), ("Lower / decrease", "lower")],
                                                  value="raise", label=FIELD_LABELS["direction"]), "choice",
                            choices=["raise", "lower"])
                        add("semitones", gr.Dropdown(choices=list(SEMITONES), value=2, label=FIELD_LABELS["semitones"],
                                                     info="AuK was trained on 1, 2 and 3."), "int", minimum=1, maximum=3)
                        add("gain_db", gr.Dropdown(choices=list(GAINS_DB), value=5, label=FIELD_LABELS["gain_db"],
                                                   info="Trained steps: 5, 10, 15 dB."), "int", minimum=5, maximum=15)
                        add("speed", gr.Dropdown(choices=[(f"{value:g}x", value) for value in SPEEDS], value=1.25,
                                                 label=FIELD_LABELS["speed"], info="Trained factors only."), "float",
                            minimum=0.5, maximum=2.0)
                        add("emotion", gr.Dropdown(choices=list(EMOTIONS), value="happy", label=FIELD_LABELS["emotion"]),
                            "choice", choices=list(EMOTIONS))
                    add("description", gr.Textbox(value=FIELD_DEFAULTS["description"], label=FIELD_LABELS["description"],
                                                  lines=2), "str")
                    with gr.Row():
                        add("sound", gr.Dropdown(choices=list(NONVERBAL_EVENTS), value="laugh", label=FIELD_LABELS["sound"]),
                            "choice", choices=list(NONVERBAL_EVENTS))
                        add("position", gr.Dropdown(choices=[("After the anchor", "after"), ("Before the anchor", "before"),
                                                             ("At the beginning", "beginning"), ("At the end", "end")],
                                                    value="after", label=FIELD_LABELS["position"]), "choice",
                            choices=["after", "before", "beginning", "end"])
                        add("speaker", gr.Dropdown(choices=[1, 2, 3, 4, 5, 6], value=1, label=FIELD_LABELS["speaker"]),
                            "int", minimum=1, maximum=6)
                    add("spoken", gr.Textbox(value="", label=FIELD_LABELS["spoken"]), "str")
                    with gr.Row():
                        add("enhance", gr.Dropdown(choices=list(ENHANCE), value="noise and reverberation",
                                                   label=FIELD_LABELS["enhance"]), "choice", choices=list(ENHANCE))
                        add("effect", gr.Dropdown(choices=list(EFFECTS), value="telephone", label=FIELD_LABELS["effect"]),
                            "choice", choices=list(EFFECTS))
                        add("music", gr.Dropdown(choices=list(MUSIC), value="singing voice only", label=FIELD_LABELS["music"]),
                            "choice", choices=list(MUSIC))
                    add("instruction", gr.Textbox(value="", label=FIELD_LABELS["instruction"], lines=3,
                                                  placeholder="Keep only the singing voice, drop everything else."), "str")
                    add("transcript", gr.Textbox(value="", label="Source transcript (optional)", lines=2,
                                                 info="Sets the length of content edits; blank transcribes the source once."),
                        "str")
            with gr.Column(scale=1, min_width=380):
                tab.preview = gr.Textbox(label="Instruction sent to AuK", lines=3, interactive=False, buttons=["copy"])
                tab.notes = gr.Markdown("")
                with gr.Row():
                    c["auk_edit.duration_mode"] = _register(registry, "auk_edit.duration_mode", gr.Radio(
                        [("Task rule", "auto"), ("Same as source", "source"), ("Custom", "custom")], value="auto",
                        label="Output length", info="Task rule follows AuK's own prompt enhancer."),
                        kind="choice", choices=["auto", "source", "custom"])
                    c["auk_edit.seconds"] = _register(registry, "auk_edit.seconds", gr.Number(
                        value=0, label="Custom length (s)", minimum=0, maximum=60), kind="float", minimum=0, maximum=60)
                with gr.Row():
                    c["auk_edit.num_step"] = _register(registry, "auk_edit.num_step", gr.Slider(
                        4, 128, value=32, step=1, label="Flow steps"), kind="int", minimum=4, maximum=128)
                    c["auk_edit.guidance_scale"] = _register(registry, "auk_edit.guidance_scale", gr.Slider(
                        0, 6, value=2.0, step=0.1, label="Guidance (CFG)"), kind="float", minimum=0, maximum=6)
                    c["auk_edit.seed"] = _register(registry, "auk_edit.seed", gr.Number(
                        value=-1, label="Seed", info="-1 picks a new one."), kind="int", minimum=-1, maximum=2**31 - 1)
                with gr.Row():
                    tab.run_button = gr.Button("🎛️  Run AuK edit", variant="primary", elem_classes=btn("emerald"), scale=3)
                    tab.cancel_button = gr.Button("⛔  Cancel", variant="stop", elem_classes=btn("red"), scale=1)
                tab.output = gr.Audio(label="Edited audio", type="filepath", buttons=["download"])
                tab.status = gr.Markdown("")
    return tab


FIELD_NAMES = [*FIELD_LABELS, "transcript"]


def bind_auk_edit_events(tab: AukEditTab, generation: Any, args: Any, registry: PresetRegistry) -> None:
    model_dir = str(getattr(args, "model_dir", ROOT / "models"))
    task_box = tab.controls["auk_edit.task"]
    field_components = [tab.controls[f"auk_edit.{name}"] for name in FIELD_NAMES]
    task_box.change(None, task_box, None, js=TASK_CLASS_JS, queue=False, show_progress="hidden", api_name=False)

    def preview(task, *values):
        task_values = dict(zip(FIELD_NAMES, values))
        notes = TASKS[task].notes if task in TASKS else ""
        try:
            return render_instruction(task, task_values), notes
        except (KeyError, ValueError) as exc:
            return f"({exc})", notes

    on_gathered([task_box.change, *[component.change for component in field_components]], preview,
                [task_box, *field_components], [tab.preview, tab.notes], queue=False, show_progress="hidden",
                trigger_mode="always_last", api_name=False)

    generation_keys = list(generation.request_keys)
    generation_components = list(generation.request_components)
    edit_keys = ["auk_edit.duration_mode", "auk_edit.seconds", "auk_edit.num_step", "auk_edit.guidance_scale", "auk_edit.seed"]
    edit_components = [tab.controls[key] for key in edit_keys]
    active = {"task": ""}
    lock = threading.Lock()

    def run(source, task, *values):
        from indextts.runtime.progress import ProgressReporter
        import soundfile as sf

        task_values = dict(zip(FIELD_NAMES, values[:len(FIELD_NAMES)]))
        edit_values = dict(zip(edit_keys, values[len(FIELD_NAMES):len(FIELD_NAMES) + len(edit_keys)]))
        generation_values = dict(zip(generation_keys, values[len(FIELD_NAMES) + len(edit_keys):]))
        if (generation_values.get("app.model") or "indextts") != "auk":
            raise gr.Error("Select AuK in the header to edit audio with it.")
        if not source or not Path(str(source)).is_file():
            raise gr.Error("Choose a source recording first.")
        if task not in TASKS:
            raise gr.Error("Choose a task.")
        instruction = task_values.get("instruction") if task == "custom" else None
        try:
            preview_text = instruction or render_instruction(task, task_values)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        with lock:
            if LAZY_ENGINE.busy:
                raise gr.Error("Wait for the current generation to finish or cancel it first.")
            run_id = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
            active["task"] = run_id
        folder = ROOT / "outputs" / "auk_edits" / f"{run_id}_{task}"
        folder.mkdir(parents=True, exist_ok=True)
        progress_file = folder / "progress.json"
        runtime = runtime_config_from_values(generation_values, model_dir=model_dir)
        settings = {key.removeprefix("auk."): value for key, value in generation_values.items()
                    if key.startswith("auk.") and key != "auk.language"}
        settings["num_step"] = int(edit_values["auk_edit.num_step"])
        settings["guidance_scale"] = float(edit_values["auk_edit.guidance_scale"])
        language = str(generation_values.get("auk.language") or "AUTO").lower()
        seed = int(edit_values["auk_edit.seed"] or -1)
        box: dict[str, Any] = {}

        class Reporter(ProgressReporter):
            def update(self, *args, **kwargs):
                LAZY_ENGINE.raise_if_canceled()
                return super().update(*args, **kwargs)

        def worker():
            try:
                with LAZY_ENGINE.in_use():
                    LAZY_ENGINE.reset_cancel(task_id=run_id)
                    engine = LAZY_ENGINE.get(runtime, progress_file=str(progress_file))
                    if getattr(engine, "model_id", "") != "auk":
                        raise RuntimeError("The loaded model is not AuK; select AuK in the header.")
                    engine.progress_reporter = Reporter("AuK edit", progress_file=str(progress_file))
                    if hasattr(engine, "set_lora"):
                        engine.set_lora(runtime.get("lora_path") or "", float(runtime.get("lora_strength") or 1.0),
                                        merge_into_base=bool(runtime.get("lora_merge_into_base")))
                    rate, pcm = engine.edit_audio(
                        str(source), task, task_values, instruction=instruction,
                        transcript=str(task_values.get("transcript") or ""),
                        duration_mode=str(edit_values["auk_edit.duration_mode"] or "auto"),
                        seconds=float(edit_values["auk_edit.seconds"] or 0), settings=settings,
                        seed=None if seed < 0 else seed, language="zh" if language == "zh" else "en")
                    target = folder / f"{task}.wav"
                    sf.write(str(target), pcm, rate, subtype="PCM_16")
                    stats = dict(engine.last_generation_stats)
                    write_json_atomic(folder / "metadata.json", {"source": str(source), "task": task, "fields": task_values,
                                                                  "settings": settings, "stats": stats})
                    box["result"] = (str(target), stats)
            except BaseException as exc:  # reported to the page below
                box["error"] = exc
                traceback.print_exc()
            finally:
                engine_ref = LAZY_ENGINE.peek()
                if engine_ref is not None:
                    engine_ref.progress_reporter = None

        started = time.perf_counter()
        thread = threading.Thread(target=worker, name="auk-edit", daemon=True)
        thread.start()
        yield gr.skip(), f"Running: {preview_text}"
        while thread.is_alive():
            thread.join(0.5)
            try:
                progress = json.loads(progress_file.read_text(encoding="utf-8"))
                message = progress.get("desc") or ""
            except (OSError, ValueError):
                message = "Loading AuK (first use takes about 20 seconds)"
            yield gr.skip(), f"{message} · {time.perf_counter() - started:.0f}s"
        if "error" in box:
            message = str(box["error"]) or type(box["error"]).__name__
            if "cancel" in message.lower():
                yield gr.skip(), "Canceled."
                return
            raise gr.Error(message)
        path, stats = box["result"]
        yield path, (f"Done in {stats.get('generation_time_s', 0):.1f}s · {stats.get('total_duration_s', 0):.2f}s of audio"
                     + (f" in {stats['pieces']} pieces" if stats.get("pieces", 1) > 1 else "")
                     + f" · saved to `{path}`")

    # Browser-packed inputs keep the many generation values out of Gradio's per-event status refresh.
    bind_gathered_api(tab.run_button.click, run, [tab.source, task_box, *field_components, *edit_components,
                                                  *generation_components],
                      [tab.output, tab.status], api_name="auk_edit", concurrency_limit=1, concurrency_id="generation")

    def cancel():
        if active["task"]:
            LAZY_ENGINE.request_cancel(expected_task=active["task"])
            return "Cancel requested."
        return "Nothing is running."

    tab.cancel_button.click(cancel, None, tab.status, queue=False, api_name=False)
