"""Model-specific controls, browser-side visibility and per-model values."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import threading

import gradio as gr

from .common import (LAZY_ENGINE, PROCESS_MANAGER, apply_values, gather_values, payload_values,
                     then_gathered, values_payload, values_payload_component)
from .model_profiles import AUK_ONLY, INDEX_ONLY, TRAINING_INDEX_ONLY, profiled_keys, switch_profile
from indextts.backends import SETTINGS_MODELS

# What the shared "train embeddings and heads" switch trains for each model.
HEAD_LABELS = {"indextts": "Train mel embedding head", "omnivoice": "Train audio embeddings and heads",
               "auk": "Train input and output layers"}


# The stylesheet (MODEL_VISIBILITY_CSS in common.py) hides the inactive model's
# controls from a body class, so a switch, a preset load and every lazily
# mounted tab or accordion show the right layout without a server round trip.
MODEL_ONLY_CLASS = {"indextts": "tts-only-indextts", "omnivoice": "tts-only-omnivoice", "auk": "tts-only-auk"}
# IndexTTS is the default layout (no class), so the page is right before the first event.
# Leaving AuK while its editing tab is open returns to Voice Generation (the tab hides).
MODEL_CLASS_JS = ("(model) => { for (const id of ['omnivoice', 'auk']) "
                  "document.body.classList.toggle('tts-model-' + id, model === id); "
                  "if (model !== 'auk' && document.querySelector('button[data-tab-id=\"auk-editing\"].selected')) "
                  "document.querySelector('button[data-tab-id=\"voice-generation\"]')?.click(); }")
METHOD_CLASS_JS = "(method) => { document.body.classList.toggle('train-method-full', method === 'full'); }"


@lru_cache(maxsize=None)
def _omnivoice_module(name: str):
    """Load an import-free OmniVoice data module without importing the model package.

    ``import omnivoice`` loads PyTorch and Transformers model code; the tag and
    language tables are plain dictionaries, so the interface reads them directly.
    """
    spec = importlib.util.find_spec("omnivoice")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("OmniVoice is not installed; run the installer to add it.")
    path = Path(next(iter(spec.submodule_search_locations))) / "utils" / f"{name}.py"
    module_spec = importlib.util.spec_from_file_location(f"_omnivoice_ui_{name}", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def omnivoice_language_choices() -> list[tuple[str, str]]:
    names = _omnivoice_module("lang_map").LANG_NAME_TO_ID
    return [("Auto", "AUTO"), *[(name.title(), code.upper()) for name, code in sorted(names.items())]]


def add_model_class(block, model: str) -> None:
    classes = block.elem_classes or []
    classes = [classes] if isinstance(classes, str) else list(classes)
    if MODEL_ONLY_CLASS[model] not in classes:
        block.elem_classes = [*classes, MODEL_ONLY_CLASS[model]]


def omnivoice_language_control(registry):
    """OmniVoice's language list, shown where IndexTTS shows its five languages."""
    choices = omnivoice_language_choices()
    component = gr.Dropdown(
        choices=choices, value="AUTO", label="Language",
        info="Auto detects it; 600+ languages.", scale=1, min_width=100,
        elem_classes=[MODEL_ONLY_CLASS["omnivoice"]],
    )
    registry.register("omnivoice.language", component, "AUTO", kind="choice", choices=[value for _, value in choices])
    return component


def auk_language_control(registry):
    """AuK speaks English and Chinese (its training data); Auto picks from the text."""
    from indextts.auk.text import LANGUAGES

    component = gr.Dropdown(
        choices=list(LANGUAGES), value="AUTO", label="Language",
        info="AuK was trained on English and Chinese.", scale=1, min_width=100,
        elem_classes=[MODEL_ONLY_CLASS["auk"]],
    )
    registry.register("auk.language", component, "AUTO", kind="choice", choices=[value for _, value in LANGUAGES])
    return component


AUK_SAMPLING_PRESETS = {"Max speed": 16, "Balanced": 32, "Max quality": 64}
AUK_DESCRIPTION_EXAMPLES = (
    "A calm middle-aged man with a deep, warm voice, speaking clearly at a moderate pace.",
    "A young woman with a bright, friendly voice, speaking cheerfully and a little fast.",
    "An elderly man with a slightly raspy, gentle voice, telling a story slowly.",
    "A confident female news presenter with a clear, neutral accent.",
)


def build_auk_controls(registry):
    from indextts.auk.text import GENERATION_DEFAULTS as AUK

    with gr.Accordion("AuK · voice and generation", open=True, elem_classes=[MODEL_ONLY_CLASS["auk"]]) as panel:
        gr.Markdown("Clone a reference voice, describe a new voice, or let a trained AuK voice speak without a "
                    "reference (Auto voice). References work best with 8 to 20 seconds of clean speech. The reference "
                    "transcript only sets the speaking pace; leave it blank to transcribe the reference once.")

        def register(name, component, **kwargs):
            return registry.register("auk." + name, component, component.value, **kwargs)
        with gr.Row():
            register("mode", gr.Dropdown(choices=[("Voice cloning", "clone"), ("Voice design", "design"), ("Auto voice", "auto")],
                                         value=AUK["mode"], label="Voice mode",
                                         info="Auto voice speaks in a selected AuK fine-tune's own voice."),
                     kind="choice", choices=["clone", "design", "auto"])
            register("voice_description", gr.Textbox(value="", label="Voice description", lines=2,
                                                      placeholder=AUK_DESCRIPTION_EXAMPLES[0],
                                                      info="Voice design: age, gender, timbre, emotion and pace in plain words."),
                     kind="str")
        with gr.Accordion("Voice description examples", open=False):
            gr.Markdown("\n".join(f"- {item}" for item in AUK_DESCRIPTION_EXAMPLES))
        register("reference_text", gr.Textbox(value="", label="Reference transcript", lines=2,
                                              info="Words spoken in the reference; used only to match its pace. "
                                                   "Blank transcribes the reference once with Whisper."), kind="str")
        quality = gr.Radio([*AUK_SAMPLING_PRESETS, "Custom"], value="Balanced", label="Sampling preset",
                           info="16 / 32 / 64 flow steps with guidance 2. 32 is the official setting.")
        with gr.Row():
            register("num_step", gr.Slider(4, 128, value=AUK["num_step"], step=1, label="AuK flow steps",
                                           info="32 is the official default; fewer steps are faster."),
                     kind="int", minimum=4, maximum=128)
            register("guidance_scale", gr.Slider(0, 6, value=AUK["guidance_scale"], step=0.1, label="AuK guidance (CFG)",
                                                 info="2.0 is the official default."),
                     kind="float", minimum=0, maximum=6)
        sampling = [registry["auk.num_step"].component, registry["auk.guidance_scale"].component]
        presets_js = "{" + ", ".join(f"'{name}': {steps}" for name, steps in AUK_SAMPLING_PRESETS.items()) + "}"
        steps_js = "{" + ", ".join(f"{steps}: '{name}'" for name, steps in AUK_SAMPLING_PRESETS.items()) + "}"
        quality.input(None, [quality, *sampling], sampling, queue=False, api_name=False, show_progress="hidden",
                      js=f"(preset, steps, guidance) => preset === 'Custom' ? [steps, guidance] : [{presets_js}[preset], 2]")
        gr.on([control.change for control in sampling], None, sampling, quality, queue=False, api_name=False,
              show_progress="hidden",
              js=f"(steps, guidance) => Number(guidance) === 2 ? ({steps_js}[Number(steps)] || 'Custom') : 'Custom'")
        with gr.Accordion("Advanced AuK sampling", open=False):
            with gr.Row():
                register("sway_coef", gr.Slider(-1, 1, value=AUK["sway_coef"], step=0.05, label="Sway sampling",
                                                info="-1 concentrates steps early in the flow (official)."),
                         kind="float", minimum=-1, maximum=1)
                register("solver", gr.Dropdown(choices=[("Euler (official)", "euler"), ("Midpoint", "midpoint")],
                                               value=AUK["solver"], label="ODE solver",
                                               info="Midpoint costs two model passes per step."),
                         kind="choice", choices=["euler", "midpoint"])
                register("max_reference_seconds", gr.Slider(3, 30, value=AUK["max_reference_seconds"], step=0.5,
                                                            label="Max reference length (s)",
                                                            info="Longer references are cut at a pause."),
                         kind="float", minimum=3, maximum=30)
                register("edge_seconds", gr.Slider(0, 1, value=AUK["edge_seconds"], step=0.05,
                                                   label="Extra section time (s)",
                                                   info="Added to every estimated section length."),
                         kind="float", minimum=0, maximum=1)
            with gr.Row():
                register("trim_reference_silence", gr.Checkbox(value=AUK["trim_reference_silence"],
                                                               label="Trim reference silence"), kind="bool")
                register("match_loudness", gr.Checkbox(value=AUK["match_loudness"],
                                                       label="Match the reference loudness"), kind="bool")
    return panel


def build_omnivoice_controls(registry):
    with gr.Accordion("OmniVoice · voice and generation", open=True, elem_classes=[MODEL_ONLY_CLASS["omnivoice"]]) as panel:
        gr.Markdown("Clone a reference voice, design a voice with supported tags, or let OmniVoice choose. Reference audio works best at 3–10 seconds; an exact transcript avoids loading automatic transcription.")
        def register(name, component, **kwargs):
            return registry.register("omnivoice." + name, component, component.value, **kwargs)
        quality = gr.Radio(["Max speed", "Balanced", "Max quality", "Custom"], value="Balanced",
                           label="Sampling preset", info="16 / 32 / 64 diffusion steps with guidance 2. More steps do not guarantee better speech for every prompt.")
        with gr.Row():
            register("mode", gr.Dropdown(choices=[("Voice cloning", "clone"), ("Auto voice", "auto"), ("Voice design", "design")], value="clone", label="Voice mode"), kind="choice", choices=["clone", "auto", "design"])
            register("instruct", gr.Textbox(value="", label="Voice tags", placeholder="male, middle-aged, moderate pitch, british accent", info="Supported tags separated by commas; choose at most one gender, age, pitch and accent. Open the tag list below."), kind="str")
        with gr.Accordion("Supported voice tags", open=False):
            tags = _omnivoice_module("voice_design")
            gr.Markdown("**English:** " + ", ".join(sorted(tags._INSTRUCT_VALID_EN)) + "\n\n**Chinese:** " + "，".join(sorted(tags._INSTRUCT_VALID_ZH)))
        register("reference_text", gr.Textbox(value="", label="Reference transcript", lines=2, info="The exact words in the reference clip. Blank downloads and runs automatic transcription on demand."), kind="str")
        with gr.Row():
            register("num_step", gr.Slider(4, 128, value=32, step=1, label="OmniVoice diffusion steps", info="32 is the official default. Lower values trade detail for speed."), kind="int", minimum=4, maximum=128)
            register("guidance_scale", gr.Slider(0, 8, value=2.0, step=0.1, label="OmniVoice guidance", info="2.0 is the official default."), kind="float", minimum=0, maximum=8)
        sampling = [registry["omnivoice.num_step"].component, registry["omnivoice.guidance_scale"].component]
        # The preset radio and the sliders mirror each other with plain arithmetic, in the browser.
        quality.input(None, [quality, *sampling], sampling, queue=False, api_name=False, show_progress="hidden",
                      js="(preset, steps, guidance) => preset === 'Custom' ? [steps, guidance] : [{'Max speed': 16, 'Balanced': 32, 'Max quality': 64}[preset], 2]")
        gr.on([control.change for control in sampling], None, sampling, quality, queue=False, api_name=False,
              show_progress="hidden", js="(steps, guidance) => Number(guidance) === 2 ? ({16: 'Max speed', 32: 'Balanced', 64: 'Max quality'}[Number(steps)] || 'Custom') : 'Custom'")
        with gr.Accordion("Advanced OmniVoice sampling", open=False):
            with gr.Row():
                for name, label, default, low, high, step in (
                    ("t_shift", "Time shift", .1, 0, 1, .01),
                    ("layer_penalty_factor", "Codebook layer penalty", 5., 0, 20, .1),
                    ("position_temperature", "Position temperature", 5., 0, 20, .1),
                    ("class_temperature", "Class temperature", 0., 0, 5, .05),
                    ("audio_chunk_duration", "Long-text chunk (seconds)", 15., 3, 30, 1),
                    ("audio_chunk_threshold", "Long-text threshold (seconds)", 30., 5, 60, 1),
                    ("pad_duration", "Edge padding (seconds)", .1, 0, 1, .01),
                    ("fade_duration", "Edge fade (seconds)", .1, 0, 1, .01),
                ):
                    register(name, gr.Slider(low, high, value=default, step=step, label=label), kind="float", minimum=low, maximum=high)
            with gr.Row():
                for name, label in (("denoise", "Denoise output"), ("preprocess_prompt", "Clean reference silence"), ("postprocess_output", "Fade and pad output")):
                    register(name, gr.Checkbox(value=True, label=label), kind="bool")
    return panel


def bind_model_controls(registry, generation, models, training, grid):
    """Wire the header model selector; return a hook that refreshes model-filtered lists."""

    selector = registry["app.model"].component
    profiles = registry["app.profiles"].component
    for block in (*generation.index_panels, *models.index_panels, *training.index_panels):
        add_model_class(block, "indextts")
    for spec in registry.specs:
        if spec.component is not None and (spec.key in INDEX_ONLY or spec.key in TRAINING_INDEX_ONLY):
            add_model_class(spec.component, "indextts")
        elif spec.component is not None and spec.key in AUK_ONLY:
            add_model_class(spec.component, "auk")
    for block in (generation.omnivoice_panel, training.omnivoice_panel):
        add_model_class(block, "omnivoice")
    for block in (generation.auk_panel, getattr(training, "auk_panel", None), *getattr(models, "auk_panels", ())):
        if block is not None:
            add_model_class(block, "auk")

    method = registry["training.adapter_type"].component
    adapter_fields = [registry["training." + name].component for name in
                      ("rank", "alpha", "dropout", "target_attention", "target_mlp", "train_mel_embed_head", "train_full_modules_fp32")]
    for field in adapter_fields:
        classes = field.elem_classes or []
        field.elem_classes = [*([classes] if isinstance(classes, str) else classes), "train-adapter-field"]
    method.change(None, method, None, js=METHOD_CLASS_JS, queue=False, show_progress="hidden", api_name=False)

    keys = [key for key in profiled_keys(registry.keys) if registry[key].component is not None and registry[key].preset]
    state_keys = [key for key in keys if isinstance(registry[key].component, gr.State)]
    browser_keys = [key for key in keys if key not in state_keys]
    browser_components = [registry[key].component for key in browser_keys]
    state_components = [registry[key].component for key in state_keys]
    head = registry["training.train_mel_embed_head"].component
    extra_outputs = [head, *training.manager_outputs]
    list_keys = ("runtime.lora_path", "training.resume_from", "grid.adapter_dir", "training.adapter_type")
    list_components = [registry[key].component for key in list_keys]

    def listed_updates(model, values):
        """Model-filtered choice lists, keeping each value only where it still applies."""
        from .generation_tab import _lora_choices
        from .grid_tab import _adapter_folders, latest_lora_folder
        from .training_tab import _resume_choices, adapter_rows

        settings_model = model in SETTINGS_MODELS
        updates = {}
        for key, choices in (("runtime.lora_path", _lora_choices(model)), ("training.resume_from", _resume_choices(model))):
            value = values.get(key) or ""
            updates[key] = {"choices": choices, "value": value if value in {item for _, item in choices} else ""}
        folders = _adapter_folders(model=model)
        grid_value = values.get("grid.adapter_dir")
        if grid_value not in {path for _, path in folders}:
            grid_value = latest_lora_folder(model=model) or None
        updates["grid.adapter_dir"] = {"choices": folders, "value": grid_value}
        methods = ["lora", "dora", "full"] if settings_model else ["lora", "dora"]
        method_value = values.get("training.adapter_type") if values.get("training.adapter_type") in methods else "dora"
        updates["training.adapter_type"] = {"choices": methods, "value": method_value}
        rows, paths = adapter_rows(model)
        label = HEAD_LABELS.get(model, "Train mel embedding head")
        return updates, {"label": label, "rows": rows, "paths": paths}

    def digest(value):
        return hashlib.sha1(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    # What each page already shows (the lists built at startup, for IndexTTS), so a
    # refresh sends a list of hundreds of checkpoints only when it actually changed.
    built, built_extras = listed_updates("indextts", {})
    # Inspecting every checkpoint of the other models once takes a moment; do it before the first switch.
    for other in SETTINGS_MODELS:
        threading.Thread(target=listed_updates, args=(other, {}), name=f"warm-{other}-lists", daemon=True).start()
    built_sent = {**{key: digest(update["choices"]) for key, update in built.items()},
                  "label": digest("Train mel embedding head"), "rows": digest(built_extras["rows"])}
    sent_lists = gr.State(built_sent)

    def list_outputs(updates, extras, sent):
        """Updates that change only what the page does not already show, plus the new record."""
        sent = dict(sent or {})
        result = {}
        for key, update in updates.items():
            fingerprint = digest(update["choices"])
            result[key] = (gr.update(value=update["value"]) if sent.get(key) == fingerprint
                           else gr.update(choices=update["choices"], value=update["value"]))
            sent[key] = fingerprint
        label_fp, rows_fp = digest(extras["label"]), digest(extras["rows"])
        tail = [
            gr.skip() if sent.get("label") == label_fp else gr.update(label=extras["label"]),
            gr.skip() if sent.get("rows") == rows_fp else extras["rows"],
            extras["paths"],
            "",
        ]
        sent["label"], sent["rows"] = label_fp, rows_fp
        return result, tail, sent

    def unload_other_model(model):
        engine = LAZY_ENGINE.peek()
        if engine is not None and not LAZY_ENGINE.busy and getattr(engine, "model_id", "indextts") != model:
            # Releasing GPU memory can take a moment; the interface does not wait for it.
            threading.Thread(target=LAZY_ENGINE.unload, name="unload-previous-speech-model", daemon=True).start()

    switch_box = values_payload_component()
    # A mouse selection can fire the dropdown's input event twice, and both copies
    # read the same session State. The active model is therefore also recorded per
    # page session here, so the repeated switch changes nothing.
    active_by_session: dict[str, str] = {}
    active_lock = threading.Lock()

    def switch(target, gathered, profile_state, sent, request: gr.Request, *items):
        states, shown_lists = items[:len(state_keys)], items[len(state_keys):]
        values = {**payload_values(browser_keys, gathered), **dict(zip(state_keys, states)),
                  **dict(zip(list_keys, shown_lists))}
        session = str(getattr(request, "session_hash", "") or "")
        unchanged = [gr.skip(), gr.skip(), {}, gr.skip(), *[gr.skip()] * (len(state_keys) + len(list_keys) + len(extra_outputs))]
        with active_lock:
            previous = active_by_session.get(session) or (profile_state or {}).get("_active", "indextts")
            if target == previous:
                return unchanged
            if LAZY_ENGINE.busy or any(job.running for job in PROCESS_MANAGER._jobs.values()):
                gr.Warning("Wait for the current job to finish or cancel it before switching models.")
                return [previous, *unchanged[1:]]
            active_by_session[session] = target
        restored = switch_profile(registry, target, {**values, "app.model": previous, "app.profiles": profile_state or {}})
        # Not every listed control is profiled (the grid folder is chosen per session).
        restored.update({key: values.get(key) for key in list_keys if key not in keys})
        updates, extras = listed_updates(target, restored)
        lists, tail, sent = list_outputs(updates, extras, sent)
        unload_other_model(target)
        print(f">> Active speech model: {target}; restored its settings, weights load on first generation", flush=True)
        payload = values_payload(browser_keys, {key: restored[key] for key in browser_keys
                                                if key not in lists and restored[key] != values.get(key)})
        return [gr.skip(), restored["app.profiles"], payload, sent, *[restored[key] for key in state_keys],
                *(lists[key] for key in list_keys), *tail]

    # The browser gathers the profiled values and applies the returned ones
    # (common.APPLY_VALUES_JS), so the switch adds only a handful of components
    # to Gradio's per-event status refresh. The model-filtered lists are direct outputs.
    switch_event = apply_values(
        gather_values(selector.select, browser_components, switch_box, trigger_mode="always_last").then(
            switch, [selector, switch_box, profiles, sent_lists, *state_components, *list_components],
            [selector, profiles, switch_box, sent_lists, *state_components, *list_components, *extra_outputs],
            queue=False, api_name="select_speech_model", show_progress="hidden"),
        switch_box, browser_components)
    # Displays that depend on the model itself. They follow the switch rather than
    # the selector's change event: that event already starts the Models tab's
    # deferring runtime description, and Gradio 6.29 makes two deferring handlers
    # of one trigger restart each other.
    refreshes = [*generation.model_refreshes, *models.model_refreshes, *training.model_refreshes]
    refresh_inputs = list({id(item): item for _, inputs, _ in refreshes for item in inputs}.values())
    refresh_outputs = [item for _, _, outputs in refreshes for item in outputs]

    def refresh_model_displays(*values):
        """Every display that depends on the active model, in one call: each
        separate event adds its own status refresh of the page."""
        given = dict(zip((id(item) for item in refresh_inputs), values))
        results = []
        for fn, inputs, outputs in refreshes:
            result = fn(*(given[id(item)] for item in inputs))
            results.extend([result] if len(outputs) == 1 else list(result))
        return results

    def follow(event):
        then_gathered(event, refresh_model_displays, refresh_inputs, refresh_outputs, queue=False, api_name=False,
                      show_progress="hidden", trigger_mode="always_last")

    follow(switch_event)

    def refresh_lists(model, sent, request: gr.Request, *values):
        with active_lock:
            active_by_session[str(getattr(request, "session_hash", "") or "")] = model
        unload_other_model(model)
        updates, extras = listed_updates(model, dict(zip(list_keys, values)))
        lists, tail, sent = list_outputs(updates, extras, sent)
        return [sent, *(lists[key] for key in list_keys), *tail]

    def after(event):
        """Preset loads change the model and adapters together; refresh their lists after them."""
        follow(event.success(refresh_lists, [selector, sent_lists, *list_components], [sent_lists, *list_components, *extra_outputs],
                             queue=False, api_name=False, show_progress="hidden"))
        return event

    return after
