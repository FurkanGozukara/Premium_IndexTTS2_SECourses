"""Top-level Gradio application composition."""

from __future__ import annotations

import ast
from argparse import Namespace
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import gradio as gr

from indextts.runtime.gpu import list_gpus
from indextts.runtime.vram_presets import RuntimeConfig, auto_tier, describe, resolve_preset

from .batch_tab import bind_batch_events, build_batch_tab
from .common import (
    APP_CSS,
    APP_HEAD,
    APP_TITLE,
    APP_VERSION,
    PremiumTheme,
    ROOT,
    runtime_config_from_values,
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
from .presets_store import PresetRegistry, PresetStore, SYSTEM_PREFIX
from .training_tab import bind_training_events, build_training_tab


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
            "target_duration_s",
            "target_duration_mode",
        }
    )
    # Sampling options are accepted by **generation_kwargs and consumed by infer_generator.
    sampling = {"do_sample", "top_p", "top_k", "temperature", "length_penalty", "num_beams", "repetition_penalty", "max_mel_tokens"}
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
) -> dict[str, Any]:
    if persisted is None or not system_preset:
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
    registry = PresetRegistry()
    store = PresetStore(registry, ROOT / "presets")
    initial_last = store.get_last_used() if (ROOT / "presets" / "system" / "default.json").is_file() else "default"
    initial_preset_choices = store.list_presets() or [SYSTEM_PREFIX + "default"]
    initial_preset_display = _display_name(store, initial_last)
    if initial_preset_display not in initial_preset_choices:
        initial_preset_display = SYSTEM_PREFIX + "default"
    persisted_runtime = load_persisted_runtime()

    with gr.Blocks(
        title=APP_TITLE,
        fill_width=True,
    ) as demo:
        gr.Markdown(
            f"# {APP_TITLE}\n"
            f"Version {APP_VERSION} | [Premium release, tutorials, and support](https://www.patreon.com/posts/139297407)",
            elem_classes=["app-header"],
        )

        with gr.Row(elem_classes=["preset-bar"]):
            preset_dropdown = gr.Dropdown(
                choices=initial_preset_choices,
                value=initial_preset_display,
                allow_custom_value=True,
                label="Universal preset",
                info="System presets appear first and are read-only; user presets include every registered tab setting.",
                scale=4,
            )
            preset_name = gr.Textbox(
                value="default",
                label="Preset name",
                info="Enter a new user preset name or select an existing user preset to overwrite it.",
                scale=3,
            )
            save_button = gr.Button("Save", elem_classes=["compact-button"], scale=1)
            load_button = gr.Button("Load", elem_classes=["compact-button"], scale=1)
            delete_button = gr.Button("Delete", variant="stop", elem_classes=["compact-button"], scale=1)
            reset_button = gr.Button("Reset", elem_classes=["compact-button"], scale=1)
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
            generation = build_generation_tab(options, registry, load_hook=demo.load)
            batch = build_batch_tab(options, registry, load_hook=demo.load)
            dataset = build_dataset_tab(options, registry, load_hook=demo.load)
            training = build_training_tab(options, registry, load_hook=demo.load)
            grid = build_grid_tab(options, registry, load_hook=demo.load)
            models = build_models_tab(options, registry)
            build_help_tab()

        # Cross-tab events are wired only after every component exists.
        bind_generation_events(generation, options, registry)
        bind_batch_events(batch, generation, options, registry)
        bind_dataset_events(dataset, training)
        bind_training_events(training, models, generation, main_tabs)
        bind_grid_events(grid, training, generation, models, main_tabs)

        store.ensure_system_presets()
        choices = store.list_presets()
        selected = _display_name(store, store.get_last_used())
        if selected not in choices:
            selected = SYSTEM_PREFIX + "default"

        component_specs = registry.component_specs
        preset_components = [spec.component for spec in component_specs]
        component_keys = [spec.key for spec in component_specs]

        def load_values(
            requested: str | None,
            runtime_overlay: RuntimeConfig | None = None,
        ):
            name = requested or "default"
            clean = name[len(SYSTEM_PREFIX):] if name.startswith(SYSTEM_PREFIX) else name
            values = store.load(clean)
            values = overlay_persisted_runtime(
                registry,
                values,
                runtime_overlay,
                system_preset=store.is_system(clean),
            )
            display = _display_name(store, clean)
            scope = "read-only system" if store.is_system(clean) else "user"
            return (
                gr.update(choices=store.list_presets(), value=display),
                clean,
                *[values[key] for key in component_keys],
                f"Loaded {scope} preset **{clean}**. Missing keys used defaults; unknown keys were ignored.",
                clean,
            )

        def save_values(name: str, *items: Any):
            try:
                values = dict(zip(component_keys, items))
                saved = store.save(name, values)
                return gr.update(choices=store.list_presets(), value=saved), saved, f"Saved user preset **{saved}**.", saved
            except PermissionError as exc:
                gr.Warning(str(exc))
                return gr.update(choices=store.list_presets()), gr.skip(), str(exc), gr.skip()
            except Exception as exc:
                gr.Error(str(exc))
                return gr.update(choices=store.list_presets()), gr.skip(), f"Preset save failed: {exc}", gr.skip()

        save_button.click(
            save_values,
            [preset_name, *preset_components],
            [preset_dropdown, preset_name, preset_status, browser_preset],
            queue=False,
            api_name="save_preset",
        )
        load_button.click(
            load_values,
            preset_dropdown,
            [preset_dropdown, preset_name, *preset_components, preset_status, browser_preset],
            queue=False,
            api_name="load_preset",
        )
        preset_dropdown.input(
            load_values,
            preset_dropdown,
            [preset_dropdown, preset_name, *preset_components, preset_status, browser_preset],
            queue=False,
            api_name="select_preset",
        )

        delete_confirm = gr.Checkbox(value=False, visible=False, label="Preset delete confirmation")

        def delete_value(confirmed: bool, requested: str):
            if not confirmed:
                return (gr.skip(), gr.skip(), *[gr.skip()] * len(preset_components), "Preset deletion dismissed.", gr.skip())
            clean = (requested or "").removeprefix(SYSTEM_PREFIX)
            try:
                if not store.delete(clean):
                    gr.Warning(f"User preset '{clean}' was not found")
                return load_values("default")[:-2] + (f"Deleted user preset **{clean}** and reset to defaults.", "default")
            except PermissionError as exc:
                gr.Warning(str(exc))
                return (gr.update(choices=store.list_presets(), value=_display_name(store, clean)), clean, *[gr.skip()] * len(preset_components), str(exc), clean)

        delete_button.click(
            delete_value,
            [delete_confirm, preset_dropdown],
            [preset_dropdown, preset_name, *preset_components, preset_status, browser_preset],
            js="(value, name) => [window.confirm('Delete this user preset? System presets cannot be deleted.'), name]",
            queue=False,
        )

        def reset_values():
            store.set_last_used("default")
            return load_values("default")[:-2] + ("Reset every registered control to system defaults.", "default")

        reset_button.click(
            reset_values,
            outputs=[preset_dropdown, preset_name, *preset_components, preset_status, browser_preset],
            queue=False,
            api_name="reset_preset",
        )

        def initial_load(browser_value: str | None):
            requested = (browser_value or store.get_last_used() or "default").removeprefix(SYSTEM_PREFIX)
            if _display_name(store, requested) not in store.list_presets():
                requested = "default"
            overlay = persisted_runtime if store.is_system(requested) else None
            return load_values(requested, overlay)

        demo.load(
            initial_load,
            browser_preset,
            [preset_dropdown, preset_name, *preset_components, preset_status, browser_preset],
            queue=False,
        )

        def refresh_preset_choices(requested: str | None):
            available = store.list_presets()
            selected_value = requested if requested in available else SYSTEM_PREFIX + "default"
            return gr.update(choices=available, value=selected_value)

        for refresh_component in (models.refresh_gpu, models.refresh_files):
            if refresh_component is not None:
                refresh_component.click(
                    refresh_preset_choices,
                    preset_dropdown,
                    preset_dropdown,
                    queue=False,
                )

    coverage = startup_request_self_check(registry, options.model_dir)
    startup_values = store.load(initial_last)
    if store.is_system(initial_last) and persisted_runtime is not None:
        startup_values = overlay_persisted_runtime(
            registry,
            startup_values,
            persisted_runtime,
            system_preset=True,
        )
        runtime_source = f"{initial_last} system preset + presets/user/.last_runtime.json"
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
    demo.launch_theme = PremiumTheme()
    demo.launch_css = APP_CSS
    demo.launch_head = APP_HEAD
    demo.ui_tabs = {
        "Voice Generation": generation,
        "Batch Generation": batch,
        "LoRA Dataset Preparation": dataset,
        "LoRA / DoRA Training": training,
        "Checkpoint Grid": grid,
        "Models & Performance": models,
        "Help": True,
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
