"""Top-level Gradio application composition."""

from __future__ import annotations

import ast
from argparse import Namespace
import inspect
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from indextts.utils.torch_compat import install_native_enum_pytree_compatibility


# Transformers currently imports TorchAO code that still registers Enum classes
# as pytree constants. New PyTorch releases handle those enums natively.
install_native_enum_pytree_compatibility()

import gradio as gr

from indextts.runtime.gpu import list_gpus
from indextts.runtime.vram_presets import VRAM_TIERS, RuntimeConfig, auto_tier, describe, resolve_preset

from .batch_tab import bind_batch_events, build_batch_tab
from .auk_edit_tab import TASK_CLASS_JS, bind_auk_edit_events, build_auk_edit_tab, task_fields_css
from .changelog_tab import build_changelog_tab
from .common import (
    APP_CSS,
    install_gradio_bounds_guard,
    APP_HEAD,
    APP_TITLE,
    APP_VERSION,
    ROOT,
    LAZY_ENGINE,
    PROCESS_MANAGER,
    TOGGLE_SECTIONS_JS,
    TOGGLE_THEME_JS,
    app_theme,
    apply_values,
    btn,
    front_hidden_plumbing,
    gather_values,
    payload_values,
    runtime_config_from_values,
    untrack_hidden_progress,
    values_payload,
    values_payload_component,
)
from .dataset_tab import bind_dataset_events, build_dataset_tab
from .generation_tab import (
    INFER_KWARG_KEYS,
    RUNNER_REQUEST_KEYS,
    bind_generation_events,
    build_default_generation_request,
    build_generation_tab,
    validate_request_coverage,
)
from .help_tab import build_help_tab
from .grid_tab import bind_grid_events, build_grid_tab
from .models_tab import (
    APPLIED_RUNTIME,
    build_models_tab,
    load_persisted_runtime,
    runtime_registry_values,
)
from .gpu_tier_presets import tier_preset_name
from .presets_store import PresetRegistry, PresetStore, SYSTEM_PREFIX
from .model_controls import MODEL_CLASS_JS, bind_model_controls
from .model_profiles import DEFAULT_MODEL, capture_profile, switch_profile
from indextts.backends import MODEL_CHOICES
from .request_guard import configure_request_guard
from .training_tab import LIVE_TRAINING_JS, bind_training_events, build_training_tab, existing_training_dataset


LAST_REGISTRY: PresetRegistry | None = None
LAST_STORE: PresetStore | None = None


def _args(value: Any | None) -> Any:
    defaults = {
        "port": 7860,
        "host": "0.0.0.0",
        "share": False,
        "model_dir": str(ROOT / "models"),
        "verbose": False,
        "no_browser": True,
        "device": "auto",
    }
    if value is None:
        return SimpleNamespace(**defaults)
    for key, default in defaults.items():
        if not hasattr(value, key):
            setattr(value, key, default)
    return value


def _runner_request_keys_from_source() -> set[str]:
    path = ROOT / "webui_generation_runner.py"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return set(RUNNER_REQUEST_KEYS)
    result: set[str] = set()
    target = next((node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "run_generation_request"), None)
    if target is None:
        return set(RUNNER_REQUEST_KEYS)
    for node in ast.walk(target):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "request":
            try:
                value = ast.literal_eval(node.slice)
            except Exception:
                continue
            if isinstance(value, str):
                result.add(value)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "request"
            and node.func.attr in {"get", "pop", "setdefault"}
            and node.args
        ):
            try:
                value = ast.literal_eval(node.args[0])
            except Exception:
                continue
            if isinstance(value, str):
                result.add(value)
    # These are read through a data-driven loop in the runner.
    result.update(
        {
            "segment_budget_scale_non_cjk",
            "cfm_temperature",
            "seed",
            "reuse_spk_cond_for_emo",
            "enable_pause_tags",
            "trim_silence_ms_threshold",
            "segmentation_mode",
            "segment_target_tokens",
            "sentence_pause_ms",
            "target_duration_s",
            "target_duration_mode",
        }
    )
    return result


def _engine_infer_parameters_from_source() -> set[str]:
    path = ROOT / "indextts" / "infer_v2_5.py"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "infer":
            names = [argument.arg for argument in node.args.args]
            ignored = {"self", "spk_audio_prompt", "text", "output_path", "lang", "stream_return", "more_segment_before"}
            return set(names) - ignored
    return set()


def startup_request_self_check(registry: PresetRegistry, model_dir: str | Path) -> dict[str, Any]:
    """Compare UI output with the live runner and engine source contracts."""

    request = build_default_generation_request(registry, model_dir=str(model_dir))
    missing, unknown = validate_request_coverage(request)
    consumed = _runner_request_keys_from_source()
    missing.update(consumed - set(request))
    unknown.update(set(request) - consumed)

    infer_explicit = _engine_infer_parameters_from_source()
    effective = set(request["infer_kwargs"])
    effective.difference_update({"section_batch_size", "latent_multiplier", "max_emotion_sum"})
    effective.add("duration_factor")
    effective.update(
        {
            "segment_budget_scale_non_cjk",
            "cfm_temperature",
            "seed",
            "reuse_spk_cond_for_emo",
            "enable_pause_tags",
            "trim_silence_ms_threshold",
            "segmentation_mode",
            "segment_target_tokens",
            "sentence_pause_ms",
            "target_duration_s",
            "target_duration_mode",
        }
    )
    # Sampling options are accepted by **generation_kwargs and consumed by infer_generator / inference_speech.
    sampling = {
        "do_sample", "top_p", "top_k", "temperature", "length_penalty", "num_beams", "repetition_penalty",
        "repetition_window", "max_mel_tokens",
    }
    expected_engine = infer_explicit | sampling
    missing_engine = expected_engine - effective
    # Explicit internal-only controls are intentionally fixed by the non-streaming runner.
    missing_engine.discard("quick_streaming_tokens")
    missing.update(f"engine.{key}" for key in missing_engine)
    unknown.update(f"engine.{key}" for key in effective - expected_engine)

    runtime_fields = {item.name for item in inspect.signature(RuntimeConfig).parameters.values()}
    covered_runtime = {
        key.removeprefix("runtime.").split(".", 1)[0]
        for key in registry.keys
        if key.startswith("runtime.")
    }
    missing.update(f"runtime.{key}" for key in runtime_fields - covered_runtime)
    result = {
        "ok": not missing and not unknown,
        "missing": sorted(missing),
        "unknown": sorted(unknown),
        "request_keys": sorted(request),
        "infer_kwargs": sorted(request["infer_kwargs"]),
    }
    if result["ok"]:
        print(
            f">> UI request coverage OK | {len(request)} runner keys | "
            f"{len(request['infer_kwargs'])} infer kwargs | {len(runtime_fields)} RuntimeConfig fields",
            flush=True,
        )
    else:
        print(f">> WARNING: UI request coverage mismatch | missing={result['missing']} | unknown={result['unknown']}", flush=True)
    return result


def _display_name(store: PresetStore, name: str) -> str:
    clean = name[len(SYSTEM_PREFIX):] if name.startswith(SYSTEM_PREFIX) else name
    return SYSTEM_PREFIX + clean if store.is_system(clean) else clean


def overlay_persisted_runtime(
    registry: PresetRegistry,
    values: Mapping[str, Any],
    persisted: RuntimeConfig | None,
    *,
    system_preset: bool,
    detected_tier: int | None = None,
) -> dict[str, Any]:
    """Restore the last applied runtime over a system preset's runtime values.

    The runtime saved by **Apply runtime** belongs to the tier it was applied
    under. It is restored over a GPU tier preset of the same tier (or a custom
    runtime the user chose deliberately), never over a different tier's preset,
    so selecting the 8 GB preset on a 32 GB card keeps the 8 GB runtime after a
    restart. User presets always keep their own saved runtime.
    """

    if persisted is None or not system_preset:
        return dict(values)
    preset_tier = str(values.get("runtime.vram_tier", "auto") or "auto").strip().lower()
    persisted_tier = str(persisted.vram_tier or "auto").strip().lower()
    if persisted_tier == "auto" and detected_tier is not None:
        persisted_tier = str(int(detected_tier))
    if persisted_tier != "custom" and preset_tier not in {"auto", persisted_tier}:
        return dict(values)
    return registry.coerce({**dict(values), **runtime_registry_values(persisted)})


def _runtime_summary(
    config: RuntimeConfig | None = None,
    *,
    source: str = "automatic defaults",
) -> None:
    gpus = list_gpus()
    if config is not None:
        gpu_detail = "no CUDA GPU detected"
        if gpus:
            gpu = gpus[0]
            gpu_detail = f"{gpu.name} | {gpu.free_gb:.2f}/{gpu.total_gb:.2f} GB free"
        print(
            f">> Runtime startup | restored {source} | {gpu_detail} | "
            f"{describe(config)} | models lazy",
            flush=True,
        )
        return
    if not gpus:
        config = RuntimeConfig(device="cpu", gpt_dtype="fp32", vram_tier="auto")
        print(f">> Runtime startup | no CUDA GPU detected | {describe(config)} | models lazy", flush=True)
        return
    gpu = gpus[0]
    tier = auto_tier(gpu.total_gb)
    config = resolve_preset(str(tier), gpu.total_gb, gpu.free_gb)
    config.device = f"cuda:{gpu.index}"
    print(
        f">> Runtime startup | {gpu.name} | {gpu.free_gb:.2f}/{gpu.total_gb:.2f} GB free | "
        f"auto tier {tier} | {describe(config)} | models lazy",
        flush=True,
    )


def build_app(args: Namespace | Any | None = None) -> gr.Blocks:
    """Construct the complete application without launching or loading models."""

    global LAST_REGISTRY, LAST_STORE
    options = _args(args)
    # Typed slider values must never surface as console tracebacks or error toasts.
    install_gradio_bounds_guard()
    registry = PresetRegistry()
    store = PresetStore(registry, ROOT / "presets")
    # Retire the old system presets before reading the last-used bookmark, so an
    # upgraded installation that last used one of them starts on its GPU's tier.
    retired = store.retire_legacy_system_presets()
    if retired:
        print(f">> Retired system presets replaced by GPU VRAM tiers: {', '.join(retired)}", flush=True)
    default_preset = store.default_preset_name()
    stored_last = store.stored_last_used()
    initial_last = stored_last or default_preset
    # The tier presets are written once every tab has registered its controls;
    # their names are fixed, so the dropdown can list them before that happens.
    tier_choices = [SYSTEM_PREFIX + tier_preset_name(tier) for tier in VRAM_TIERS]
    initial_preset_choices = list(
        dict.fromkeys(tier_choices + [name for name in store.list_presets() if not name.startswith(SYSTEM_PREFIX)])
    )
    initial_preset_display = _display_name(store, initial_last)
    if initial_preset_display not in initial_preset_choices:
        initial_preset_display = SYSTEM_PREFIX + default_preset
    persisted_runtime = load_persisted_runtime()
    gpus = list_gpus()
    detected_total = float(gpus[0].total_gb) if gpus else 0.0
    print(
        f">> GPU VRAM preset | detected {detected_total:.1f} GB -> {store.detected_tier} GB tier | "
        + (
            f"restoring last used preset '{stored_last}'"
            if stored_last
            else f"no saved preset; selecting '{default_preset}' for this GPU"
        ),
        flush=True,
    )

    with gr.Blocks(title=APP_TITLE) as demo:
        with gr.Row(elem_classes=["app-header"]):
            gr.Markdown(
                f"# {APP_TITLE}\n"
                f"Version {APP_VERSION} | [Premium release, tutorials, and support](https://www.patreon.com/posts/139297407)",
                container=False,
            )
            registry.register("app.model", gr.Dropdown(
                choices=MODEL_CHOICES, value=DEFAULT_MODEL, label="Speech model",
                info="Separate saved settings.", min_width=200, scale=0,
                elem_id="speech-model-selector",
            ), DEFAULT_MODEL, kind="choice", choices=[key for _, key in MODEL_CHOICES])
            # The model's panels follow a body class (ui/model_controls.py). Gradio runs
            # an event's listeners one after another, so this one is registered first.
            speech_model = registry["app.model"].component
            speech_model.change(None, speech_model, None, js=MODEL_CLASS_JS, queue=False, show_progress="hidden",
                                api_name=False)
            registry.register("app.profiles", gr.State({"_active": DEFAULT_MODEL}), {"_active": DEFAULT_MODEL}, kind="dict")
            with gr.Row(elem_classes=["header-actions"], scale=0):
                last_values_button = gr.Button(
                    "🕘  Load last values",
                    elem_classes=btn("bronze"),
                    scale=0,
                    min_width=200,
                    interactive=False,
                )
                sections_button = gr.Button(
                    "⇕  Open / close all sections",
                    elem_classes=btn("slate"),
                    scale=0,
                    min_width=230,
                )
                theme_button = gr.Button(
                    "🌗  Light / dark theme",
                    elem_classes=btn("gray"),
                    scale=0,
                    min_width=230,
                )
        # Both switches are pure client-side DOM work, so they stay instant even
        # while a generation or training job is holding the queue.
        sections_button.click(None, None, None, js=TOGGLE_SECTIONS_JS)
        theme_button.click(None, None, None, js=TOGGLE_THEME_JS)

        with gr.Row(elem_classes=["preset-bar"]):
            preset_dropdown = gr.Dropdown(
                choices=initial_preset_choices,
                value=initial_preset_display,
                allow_custom_value=True,
                label="Universal preset",
                info=(
                    "★ GPU VRAM presets (6 to 32 GB) are read-only and fit generation, dataset preparation and "
                    "training to that card; the one matching your GPU is selected on first start. "
                    "User presets store every registered setting of every tab."
                ),
                scale=4,
            )
            preset_name = gr.Textbox(
                value=initial_last,
                label="Preset name",
                info="Enter a new user preset name or select an existing user preset to overwrite it.",
                scale=3,
            )
            save_button = gr.Button("💾  Save", elem_classes=btn("blue"), scale=1)
            load_button = gr.Button("📥  Load", elem_classes=btn("cyan"), scale=1)
            delete_button = gr.Button("🗑️  Delete", variant="stop", elem_classes=btn("rose"), scale=1)
            reset_button = gr.Button("↺  Reset", elem_classes=btn("amber"), scale=1)
        preset_status = gr.Markdown("System and user presets are separate.", elem_classes=["preset-status"])
        # The preset store already persists the last-used name.  A regular State
        # avoids BrowserState attempting to parse an absent/corrupt localStorage
        # entry during first load while preserving the same event wiring.
        browser_preset = gr.State(initial_last)

        with gr.Tabs(
            selected="voice-generation",
            elem_id="main-tabs",
            elem_classes=["main-tabs"],
        ) as main_tabs:
            def build_panel(label, factory, **kwargs):
                started = time.perf_counter()
                print(f">> Building {label}...", flush=True)
                result = factory(options, registry, **kwargs)
                print(f">> {label} ready in {time.perf_counter() - started:.2f}s", flush=True)
                return result
            generation = build_panel("Voice Generation", build_generation_tab, load_hook=last_values_button.click)
            batch = build_panel("Batch Generation", build_batch_tab, load_hook=last_values_button.click)
            dataset = build_panel("Dataset Preparation", build_dataset_tab, load_hook=last_values_button.click)
            training = build_panel("Voice Training", build_training_tab, load_hook=last_values_button.click)
            grid = build_panel("Checkpoint Grid", build_grid_tab, load_hook=last_values_button.click)
            models = build_panel("Models & Performance", build_models_tab)
            # After the busy tabs: Gradio finds a component by walking the layout in order, on every
            # event, so a tab placed before Training and Models would slow every model switch.
            auk_edit = build_panel("AuK Audio Editing", build_auk_edit_tab)
            build_help_tab()
            with gr.Tab("📜 Changelog", id="changelog", render_children=False):
                build_changelog_tab()

        # Cross-tab events are wired only after every component exists.
        bind_generation_events(generation, options, registry)
        bind_batch_events(batch, generation, options, registry)
        bind_auk_edit_events(auk_edit, generation, options, registry)
        bind_dataset_events(dataset, training)
        bind_training_events(training, generation, main_tabs)
        bind_grid_events(grid, training, generation, models, main_tabs)
        # Model visibility is a body class the stylesheet reads, so lazily
        # mounted tabs and accordions need no server round trip of their own.
        after_model_values = bind_model_controls(registry, generation, models, training, grid, status=preset_status)
        demo.load(None, registry["app.model"].component, None, js=MODEL_CLASS_JS)
        demo.load(None, registry["auk_edit.task"].component, None, js=TASK_CLASS_JS)

        def loaded_last_values_notice() -> None:
            message = "Loaded the last run of every tab."
            print(">> " + message, flush=True)
            gr.Info(message)

        # Queued on purpose: gr.Info reaches the page only from a queued event (an unqueued one has no event id,
        # and the notice went to Python's warnings instead). The event has no inputs or outputs to refresh.
        last_values_button.click(
            loaded_last_values_notice,
            show_progress="hidden",
            api_name="load_last_values",
        )

        store.ensure_system_presets()

        component_specs = registry.component_specs
        preset_components = [spec.component for spec in component_specs]
        component_keys = [spec.key for spec in component_specs]
        # Browser values travel in one hidden payload (common.APPLY_VALUES_JS);
        # server-side State values are returned directly.
        state_specs = [spec for spec in component_specs if isinstance(spec.component, gr.State)]
        browser_specs = [spec for spec in component_specs if not isinstance(spec.component, gr.State)]
        browser_keys = [spec.key for spec in browser_specs]
        browser_components = [spec.component for spec in browser_specs]
        state_components = [spec.component for spec in state_specs]
        values_box = values_payload_component()

        def load_values(
            requested: str | None,
            runtime_overlay: RuntimeConfig | None = None,
            current_model: str | None = None,
            allow_busy: bool = False,
        ) -> tuple[str, dict[str, Any], str]:
            if not allow_busy and (LAZY_ENGINE.busy or any(job.running for job in PROCESS_MANAGER._jobs.values())):
                raise gr.Error("Wait for the current job to finish or cancel it before loading a preset.")
            name = requested or store.default_preset_name()
            clean = name[len(SYSTEM_PREFIX):] if name.startswith(SYSTEM_PREFIX) else name
            values = store.load(clean)
            values = overlay_persisted_runtime(
                registry,
                values,
                runtime_overlay,
                system_preset=store.is_system(clean),
                detected_tier=store.detected_tier,
            )
            values = existing_training_dataset(capture_profile(values))
            if store.is_system(clean) and current_model and current_model != values["app.model"]:
                values = switch_profile(registry, current_model, values)
            scope = "read-only GPU VRAM" if store.is_system(clean) else "user"
            return clean, values, f"Loaded {scope} preset **{clean}**. Missing keys used defaults; unknown keys were ignored."

        def ui_load(clean: str, values: Mapping[str, Any], message: str, bookmark: str | None = None):
            return (
                gr.update(choices=store.list_presets(), value=_display_name(store, clean)),
                clean,
                message,
                bookmark or clean,
                values_payload(browser_keys, values),
                *[values[spec.key] for spec in state_specs],
            )

        ui_outputs = [preset_dropdown, preset_name, preset_status, browser_preset, values_box, *state_components]

        def preset_ui_event(event):
            """Apply the loaded values in the browser, then refresh model-filtered lists."""
            return after_model_values(apply_values(event, values_box, browser_components))

        def load_selected(requested, current_model=None):
            return ui_load(*load_values(requested, current_model=current_model))

        # A dropdown pick uses select: Gradio 6.29 also dispatches input when the list loses focus.
        def current_preset_choices():
            # Presets written outside this page (the preset after training, another tab) appear when the list opens.
            return gr.update(choices=store.list_presets())

        preset_dropdown.focus(current_preset_choices, None, preset_dropdown, queue=False, show_progress="hidden",
                              api_name=False)

        for trigger in (load_button.click, preset_dropdown.select):
            preset_ui_event(trigger(load_selected, [preset_dropdown, registry["app.model"].component], ui_outputs,
                                    queue=False, show_progress="hidden", api_name=False))

        def save_values(name: str, gathered: Any, *states: Any):
            try:
                values = capture_profile({**payload_values(browser_keys, gathered),
                                          **dict(zip((spec.key for spec in state_specs), states))})
                saved = store.save(name, values)
                return gr.update(choices=store.list_presets(), value=saved), saved, f"Saved user preset **{saved}**.", saved
            except PermissionError as exc:
                gr.Warning(str(exc))
                return gr.update(choices=store.list_presets()), gr.skip(), str(exc), gr.skip()
            except Exception as exc:
                gr.Error(str(exc))
                return gr.update(choices=store.list_presets()), gr.skip(), f"Preset save failed: {exc}", gr.skip()

        gather_values(save_button.click, browser_components, values_box).then(
            save_values,
            [preset_name, values_box, *state_components],
            [preset_dropdown, preset_name, preset_status, browser_preset],
            queue=False,
            api_name=False,
        )

        delete_confirm = gr.Checkbox(value=False, visible=False, label="Preset delete confirmation")

        def delete_value(confirmed: bool, requested: str, current_model: str):
            if not confirmed:
                return (gr.skip(), gr.skip(), "Preset deletion dismissed.", gr.skip(), gr.skip(), *[gr.skip()] * len(state_specs))
            clean = (requested or "").removeprefix(SYSTEM_PREFIX)
            try:
                if not store.delete(clean):
                    gr.Warning(f"User preset '{clean}' was not found")
                fallback, values, _ = load_values(store.default_preset_name(), current_model=current_model)
                return ui_load(fallback, values,
                               f"Deleted user preset **{clean}** and loaded the **{fallback}** preset detected for this GPU.")
            except PermissionError as exc:
                gr.Warning(str(exc))
                return (gr.update(choices=store.list_presets(), value=_display_name(store, clean)), clean, str(exc), clean,
                        gr.skip(), *[gr.skip()] * len(state_specs))

        preset_ui_event(delete_button.click(
            delete_value,
            [delete_confirm, preset_dropdown, registry["app.model"].component],
            ui_outputs,
            js="(value, name, model) => [window.confirm('Delete this user preset? System presets cannot be deleted.'), name, model]",
            queue=False,
            api_name=False,
        ))

        def reset_values(current_model):
            fallback, values, _ = load_values(store.default_preset_name(), current_model=current_model)
            return ui_load(fallback, values, f"Reset every registered control to the **{fallback}** preset detected for this GPU.")

        preset_ui_event(reset_button.click(reset_values, registry["app.model"].component, ui_outputs,
                                           queue=False, api_name=False))

        def initial_load():
            # Read the durable bookmark for every new page/session. A State
            # initialized at build time keeps the server's original selection
            # and can overwrite a more recently selected, loaded or saved preset.
            requested = store.get_last_used().removeprefix(SYSTEM_PREFIX)
            if _display_name(store, requested) not in store.list_presets():
                requested = store.default_preset_name()
            overlay = persisted_runtime if store.is_system(requested) else None
            # A GPU VRAM preset opens on the speech model shown last (a user preset keeps its saved model).
            last_model = store.stored_last_model()
            if last_model not in {key for _, key in MODEL_CHOICES}:
                last_model = None
            return ui_load(*load_values(requested, overlay, current_model=last_model, allow_busy=True))

        initial_load_event = preset_ui_event(demo.load(initial_load, None, ui_outputs, queue=False, api_name="initial_load"))
        # Every model the header shows, chosen or loaded with a preset, becomes the bookmark the next page opens on.
        speech_model.change(store.set_last_model, speech_model, None, queue=False, show_progress="hidden", api_name=False)
        # Order of the browser payload and the State values in the page's preset events.
        demo.preset_payload_keys = list(browser_keys)
        demo.preset_state_keys = [spec.key for spec in state_specs]
        initial_load_event.then(
            lambda: gr.update(interactive=True),
            outputs=last_values_button,
            queue=False,
            show_progress="hidden",
            api_name=False,
        )

        # The documented API keeps its original signatures (the tutorial narration
        # client loads presets by name and reads every value from load_preset).
        # These events have a hidden trigger the page never fires, so their many
        # components never join the browser's per-event status refresh.
        api_trigger = gr.Button(visible=False)
        preset_outputs = [preset_dropdown, preset_name, *preset_components, preset_status, browser_preset]

        def api_load(clean: str, values: Mapping[str, Any], message: str, bookmark: str | None = None):
            return (gr.update(choices=store.list_presets(), value=_display_name(store, clean)), clean,
                    *[values[key] for key in component_keys], message, bookmark or clean)

        def load_selected_preset(requested):
            return api_load(*load_values(requested))

        def api_reset_values():
            fallback, values, _ = load_values(store.default_preset_name())
            return api_load(fallback, values, f"Reset every registered control to the **{fallback}** preset detected for this GPU.")

        def api_save_values(name: str, *items: Any):
            values = dict(zip(component_keys, items))
            gathered = [values.get(key) for key in browser_keys]
            return save_values(name, gathered, *[values.get(spec.key) for spec in state_specs])

        api_trigger.click(load_selected_preset, preset_dropdown, preset_outputs, queue=False, api_name="load_preset")
        api_trigger.click(load_selected_preset, preset_dropdown, preset_outputs, queue=False, api_name="select_preset")
        api_trigger.click(api_reset_values, None, preset_outputs, queue=False, api_name="reset_preset")
        api_trigger.click(api_save_values, [preset_name, *preset_components],
                          [preset_dropdown, preset_name, preset_status, browser_preset], queue=False, api_name="save_preset")
        demo.load(None, None, None, js=LIVE_TRAINING_JS)
        def refresh_preset_choices(requested: str | None):
            available = store.list_presets()
            selected_value = requested if requested in available else SYSTEM_PREFIX + store.default_preset_name()
            return gr.update(choices=available, value=selected_value)

        for refresh_component in (models.refresh_gpu, models.refresh_files):
            if refresh_component is not None:
                refresh_component.click(
                    refresh_preset_choices,
                    preset_dropdown,
                    preset_dropdown,
                    queue=False,
                )
        # Last, once every event exists: hidden-progress events track no output status,
        # and the hidden helpers move before the page, where Gradio finds them quickly.
        untrack_hidden_progress(demo)
        front_hidden_plumbing(demo)

    coverage = startup_request_self_check(registry, options.model_dir)
    startup_values = store.load(initial_last)
    if store.is_system(initial_last) and persisted_runtime is not None:
        overlaid = overlay_persisted_runtime(
            registry,
            startup_values,
            persisted_runtime,
            system_preset=True,
            detected_tier=store.detected_tier,
        )
        if overlaid != startup_values:
            startup_values = overlaid
            runtime_source = f"{initial_last} system preset + presets/user/.last_runtime.json"
        else:
            runtime_source = f"{initial_last} system preset"
    else:
        runtime_source = (
            f"{initial_last} user preset"
            if not store.is_system(initial_last)
            else f"{initial_last} system preset"
        )
    startup_options = runtime_config_from_values(
        startup_values,
        model_dir=str(options.model_dir),
    )
    startup_runtime = RuntimeConfig.from_dict(startup_options)
    APPLIED_RUNTIME.clear()
    APPLIED_RUNTIME.update(startup_options)
    _runtime_summary(startup_runtime, source=runtime_source)
    demo.preset_registry = registry
    demo.registry = registry
    demo.preset_store = store
    demo.request_coverage = coverage
    demo.launch_theme = app_theme()
    demo.launch_css = APP_CSS + task_fields_css()
    demo.launch_head = APP_HEAD
    configure_request_guard(demo)
    demo.ui_tabs = {
        "Voice Generation": generation,
        "Batch Generation": batch,
        "LoRA Dataset Preparation": dataset,
        "LoRA / DoRA Training": training,
        "Checkpoint Grid": grid,
        "Models & Performance": models,
        "Help": True,
        "📜 Changelog": True,
    }
    LAST_REGISTRY = registry
    LAST_STORE = store
    return demo


__all__ = [
    "LAST_REGISTRY",
    "LAST_STORE",
    "build_app",
    "overlay_persisted_runtime",
    "startup_request_self_check",
]
