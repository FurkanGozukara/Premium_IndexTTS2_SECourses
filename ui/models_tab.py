"""GPU inventory, runtime configuration, downloads, and VRAM tools."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import gradio as gr

from indextts.quant.convrot_int8 import describe_checkpoint, is_int8_convrot_checkpoint
from indextts.runtime.gpu import list_gpus
from indextts.runtime.vram_presets import (
    RuntimeConfig,
    auto_tier,
    estimate_vram_gb,
    preset_notes,
    resolve_preset,
)
from indextts.utils.model_downloads import ensure_base_models, ensure_int8_gpt

from .common import (
    LAZY_ENGINE,
    PROCESS_MANAGER,
    ROOT,
    btn,
    open_folder,
    runtime_config_from_values,
    tail_text,
    write_json_atomic,
)
from .presets_store import PresetRegistry


AUX_NAMES = ("semantic_model", "qwen_emo", "campplus", "semantic_codec", "s2mel", "bigvgan")
RUNTIME_DEFAULTS = RuntimeConfig(device="auto").to_dict()
RUNTIME_DEFAULTS.update({"use_qwen_emo": True, "use_deepspeed": False})
APPLIED_RUNTIME: dict[str, Any] = {}
LAST_RUNTIME_PATH = ROOT / "presets" / "user" / ".last_runtime.json"


def load_persisted_runtime(
    path: str | Path | None = None,
) -> RuntimeConfig | None:
    source = Path(path) if path is not None else LAST_RUNTIME_PATH
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    return RuntimeConfig.from_dict(payload)


def persist_runtime_config(
    config: RuntimeConfig | Mapping[str, Any],
    path: str | Path | None = None,
) -> Path:
    destination = Path(path) if path is not None else LAST_RUNTIME_PATH
    runtime = RuntimeConfig.from_dict(config)
    return write_json_atomic(destination, runtime.to_dict())


def runtime_registry_values(config: RuntimeConfig | Mapping[str, Any]) -> dict[str, Any]:
    runtime = RuntimeConfig.from_dict(config).to_dict()
    return {
        **{
            f"runtime.{key}": value
            for key, value in runtime.items()
            if key != "aux_residency"
        },
        **{
            f"runtime.aux_residency.{key}": value
            for key, value in runtime["aux_residency"].items()
        },
    }


def _gpu_rows() -> list[list[Any]]:
    return [
        [f"cuda:{gpu.index}", gpu.name, round(gpu.total_gb, 2), round(gpu.free_gb, 2), "Yes" if gpu.is_default else ""]
        for gpu in list_gpus()
    ]


def _gpu_total(device: str | None) -> float:
    gpus = list_gpus()
    match = re.search(r"(\d+)$", str(device or ""))
    index = int(match.group(1)) if match else 0
    gpu = next((item for item in gpus if item.index == index), None)
    return float(gpu.total_gb) if gpu else 0.0


def _gpu_free(device: str | None) -> float:
    gpus = list_gpus()
    match = re.search(r"(\d+)$", str(device or ""))
    index = int(match.group(1)) if match else 0
    gpu = next((item for item in gpus if item.index == index), None)
    return float(gpu.free_gb) if gpu else 0.0


def _device_choices() -> list[str]:
    choices = [f"cuda:{gpu.index}" for gpu in list_gpus()]
    choices.extend(["auto", "cpu"])
    return list(dict.fromkeys(choices))


def _tier_choices() -> list[tuple[str, str]]:
    gpus = list_gpus()
    total = gpus[0].total_gb if gpus else 0.0
    detected = auto_tier(total) if total else 6
    return [(f"Auto (detected {total:.1f} GB, tier {detected})", "auto")] + [
        (f"{tier} GB", str(tier)) for tier in (6, 8, 10, 12, 16, 24, 32)
    ] + [("Custom", "custom")]


def _model_status_rows(model_dir: str | Path) -> list[list[Any]]:
    root = Path(model_dir).expanduser().resolve()
    expected = (
        "config.yaml",
        "gpt.pth",
        "gpt_int8_convrot.safetensors",
        "s2mel.pth",
        "codec.pth",
        "feat1.pt",
        "feat2.pt",
        "multilingual_zh_ja_yue_char_del.tiktoken",
    )
    rows = []
    for name in expected:
        path = root / name
        kind = "INT8 ConvRot" if name.endswith(".safetensors") and is_int8_convrot_checkpoint(path) else path.suffix.lstrip(".").upper()
        rows.append([name, "Ready" if path.is_file() else "Missing", round(path.stat().st_size / 1024**2, 2) if path.is_file() else 0.0, kind, str(path)])
    for directory in ("qwen0.6bemo4-merge", "hf_cache/w2v-bert-2.0", "hf_cache/bigvgan"):
        path = root / directory
        rows.append([directory, "Ready" if path.is_dir() else "Missing", "", "Directory", str(path)])
    return rows


def _estimate_html(config: RuntimeConfig, total_gb: float) -> str:
    if total_gb <= 0:
        total_gb = 32.0
    estimate = estimate_vram_gb(config, total_gb)
    cls = "status-ok" if estimate["fits"] else "status-error"
    verdict = "Fits selected GPU" if estimate["fits"] else "Estimated to exceed the selected GPU"
    return (
        f'<div class="summary-strip {cls}"><b>{verdict}</b> | Peak {estimate["estimated_peak_gb"]:.2f} GB | '
        f'Reserve {estimate["reserve_gb"]:.2f} GB | Headroom {estimate["headroom_gb"]:.2f} GB | '
        f'Resident weights {estimate["resident_weights_gb"]:.2f} GB | Activations {estimate["activations_gb"]:.2f} GB</div>'
    )


@dataclass
class ModelsTab:
    controls: dict[str, Any] = field(default_factory=dict)
    tier: Any = None
    device: Any = None
    notes: Any = None
    estimate: Any = None
    apply_status: Any = None
    model_status: Any = None
    refresh_gpu: Any = None
    refresh_files: Any = None


def _register(
    registry: PresetRegistry,
    key: str,
    component: Any,
    default: Any,
    *,
    kind: str = "auto",
    choices: Sequence[Any] | None = None,
    minimum: float | int | None = None,
    maximum: float | int | None = None,
) -> None:
    registry.register(key, component, default, kind=kind, choices=choices, minimum=minimum, maximum=maximum)


def build_models_tab(args: Any, registry: PresetRegistry) -> ModelsTab:
    model_dir = str(Path(getattr(args, "model_dir", ROOT / "models")).expanduser().resolve())
    initial_device = str(getattr(args, "device", "auto") or "auto")
    if initial_device not in _device_choices():
        initial_device = "auto"
    tab = ModelsTab()
    c = tab.controls

    with gr.Tab("Models & Performance", id="models-performance"):
        gr.Markdown("### Hardware & Runtime")
        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                with gr.Row():
                    refresh_gpu = gr.Button("↻  Refresh GPU inventory", elem_classes=btn("violet"))
                    tab.refresh_gpu = refresh_gpu
                    tab.device = gr.Dropdown(
                        choices=_device_choices(), value=initial_device, label="Device",
                        info="Auto selects the first available accelerator; choose CPU only for diagnostics.",
                    )
                    tab.tier = gr.Dropdown(
                        choices=_tier_choices(), value="auto", label="VRAM tier",
                        info="Auto detects physical VRAM; selecting a named tier fills conservative runtime settings.",
                    )
                gpu_table = gr.Dataframe(
                    headers=["Device", "GPU", "Total GB", "Free GB", "Default"],
                    value=_gpu_rows(), type="array", interactive=False, label="GPU inventory",
                    datatype=["str", "str", "number", "number", "str"], buttons=["fullscreen"],
                )
                tab.notes = gr.Markdown(preset_notes("32" if not list_gpus() else str(auto_tier(list_gpus()[0].total_gb))))
            with gr.Column(scale=1):
                tab.estimate = gr.HTML(_estimate_html(RuntimeConfig(device=initial_device), _gpu_total(initial_device)))
                with gr.Row():
                    apply_button = gr.Button("⚡  Apply runtime", variant="primary", elem_classes=btn("emerald"))
                    unload_button = gr.Button("🧹  Unload model / free VRAM", variant="stop", elem_classes=btn("orange"))
                tab.apply_status = gr.Markdown("Runtime settings are ready. Models remain unloaded until first generation.")

        # Keep the shipped preset portable even when this process was launched with
        # an explicit CPU/CUDA override for diagnostics.
        _register(registry, "runtime.device", tab.device, "auto", kind="choice", choices=_device_choices())
        _register(registry, "runtime.vram_tier", tab.tier, "auto", kind="choice", choices=["auto", "6", "8", "10", "12", "16", "24", "32", "custom"])

        with gr.Accordion("Model Variant & Compute", open=True):
            with gr.Row():
                variant = gr.Dropdown(choices=["bf16", "int8_convrot"], value="bf16", label="GPT model variant", info="BF16 gives the official quality path; INT8 ConvRot reduces GPT weight memory.")
                dtype = gr.Dropdown(choices=["bf16", "fp16", "fp32"], value="bf16", label="GPT dtype", info="BF16 is recommended on modern NVIDIA GPUs; FP32 is the CPU-compatible fallback.")
                attention = gr.Dropdown(choices=["sdpa", "flash_attention_2", "eager"], value="sdpa", label="Attention backend", info="SDPA is the compatible default; FlashAttention 2 requires its optional package.")
                use_accel = gr.Checkbox(value=False, label="Use acceleration engine", info="Enables the optional CUDA-graph/flash-attention path; use beams=1.")
                use_qwen = gr.Checkbox(value=True, label="Enable emotion-text model", info="Required for Emotion text mode; on-demand residency keeps startup lazy.")
            with gr.Row():
                compile_s2mel = gr.Checkbox(value=False, label="Compile s2mel", info="Uses torch.compile for repeated workloads; first generation takes longer.")
                bigvgan_kernel = gr.Checkbox(value=False, label="BigVGAN CUDA kernel", info="Uses the optional fused activation kernel when available.")
                s2mel_bf16 = gr.Checkbox(value=False, label="CFM estimator BF16 autocast", info="Reduces activation VRAM; useful at 6 GB and sometimes slightly changes output.")
                use_deepspeed = gr.Checkbox(value=False, label="Use DeepSpeed loader", info="Optional legacy loader; leave off unless DeepSpeed is installed.")
            _register(registry, "runtime.model_variant", variant, "bf16", kind="choice", choices=["bf16", "int8_convrot"])
            _register(registry, "runtime.gpt_dtype", dtype, "bf16", kind="choice", choices=["bf16", "fp16", "fp32"])
            _register(registry, "runtime.attention_backend", attention, "sdpa", kind="choice", choices=["sdpa", "flash_attention_2", "eager"])
            _register(registry, "runtime.use_accel", use_accel, False, kind="bool")
            _register(registry, "runtime.use_qwen_emo", use_qwen, True, kind="bool")
            _register(registry, "runtime.torch_compile_s2mel", compile_s2mel, False, kind="bool")
            _register(registry, "runtime.use_cuda_kernel_bigvgan", bigvgan_kernel, False, kind="bool")
            _register(registry, "runtime.s2mel_estimator_autocast", s2mel_bf16, False, kind="bool")
            _register(registry, "runtime.use_deepspeed", use_deepspeed, False, kind="bool")

        with gr.Accordion("Block Swap & Memory", open=False):
            with gr.Row():
                blocks = gr.Slider(-1, 24, value=0, step=1, label="GPT blocks to swap", info="0 keeps all blocks resident; -1 lets runtime fit automatically; up to 24 streams from CPU.")
                ring = gr.Slider(1, 4, value=2, step=1, label="Swap ring size", info="2 overlaps transfer and compute; 1 uses least VRAM.")
                pinned = gr.Checkbox(value=True, label="Pinned swap memory", info="Recommended for faster CPU-to-GPU block transfers.")
                cache = gr.Slider(1024, 32768, value=8192, step=256, label="Runtime CFM cache length", info="Upper cache reservation used when generation does not request a larger value.")
                reserve = gr.Slider(0, 12, value=2.0, step=0.25, label="VRAM reserve (GB)", info="2 GB is recommended to absorb allocator and generation peaks.")
                hint = gr.Slider(1, 64, value=8, step=1, label="Section batch hint", info="Advisory maximum shown to generation controls for this runtime.")
            _register(registry, "runtime.blocks_to_swap", blocks, 0, kind="int", minimum=-1, maximum=24)
            _register(registry, "runtime.swap_ring_size", ring, 2, kind="int", minimum=1, maximum=4)
            _register(registry, "runtime.pin_swap_memory", pinned, True, kind="bool")
            _register(registry, "runtime.cfm_cache_length", cache, 8192, kind="int", minimum=1024, maximum=32768)
            _register(registry, "runtime.vram_reserve_gb", reserve, 2.0, kind="float", minimum=0, maximum=12)
            _register(registry, "runtime.max_section_batch_size_hint", hint, 8, kind="int", minimum=1, maximum=64)

        with gr.Accordion("Auxiliary Model Residency", open=False):
            gr.Markdown("GPU is fastest, on-demand moves a model around each use, and CPU is available for reference encoders on very small GPUs.")
            aux_components = {}
            with gr.Row():
                for name in AUX_NAMES[:3]:
                    allowed = ["gpu", "on_demand", "cpu"]
                    default = RuntimeConfig().aux_residency[name]
                    component = gr.Dropdown(choices=allowed, value=default, label=name.replace("_", " ").title(), info="Residency policy for this auxiliary model.")
                    aux_components[name] = component
                    _register(registry, f"runtime.aux_residency.{name}", component, default, kind="choice", choices=allowed)
            with gr.Row():
                for name in AUX_NAMES[3:]:
                    allowed = ["gpu", "on_demand"]
                    default = RuntimeConfig().aux_residency[name]
                    component = gr.Dropdown(choices=allowed, value=default, label=name.replace("_", " ").title(), info="Residency policy for this synthesis model.")
                    aux_components[name] = component
                    _register(registry, f"runtime.aux_residency.{name}", component, default, kind="choice", choices=allowed)

        with gr.Accordion("Model Files & Downloads", open=False):
            with gr.Row():
                int8_download = gr.Button("⬇️  Download INT8 model", elem_classes=btn("sky"))
                base_download = gr.Button("⬇️  Download / verify base models", elem_classes=btn("teal"))
                refresh_files = gr.Button("↻  Refresh file status", elem_classes=btn("green"))
                tab.refresh_files = refresh_files
                open_models = gr.Button("📁  Open model folder", elem_classes=btn("indigo"))
            download_status = gr.Markdown("Downloads are idle.")
            tab.model_status = gr.Dataframe(
                headers=["File", "Status", "Size MB", "Type", "Path"],
                value=_model_status_rows(model_dir), type="array", interactive=False,
                datatype=["str", "str", "number", "str", "str"], label="Model file status",
                max_height=360, wrap=True, buttons=["fullscreen"],
            )

        with gr.Accordion("VRAM Benchmark", open=False):
            gr.Markdown("Runs the repository benchmark in an isolated process on CUDA-visible GPU 0. Keep calibration runs short while the GPU is shared.")
            with gr.Row():
                emulate = gr.Checkbox(value=False, label="Emulate tier cap", info="Caps the PyTorch allocator to tier minus reserve for a stricter fit test.")
                subtitle_bench = gr.Checkbox(value=False, label="Exercise batch/subtitle path", info="Also runs the multi-text path used by caption generation.")
                benchmark_button = gr.Button("⏱️  Run VRAM benchmark", variant="primary", elem_classes=btn("purple"))
            benchmark_output = gr.Textbox(label="Benchmark log / result", lines=12, max_lines=20, interactive=False, buttons=["copy"], elem_classes=["log-tail"])

    runtime_specs = [spec for spec in registry.specs if spec.component is not None and spec.key.startswith("runtime.")]
    runtime_keys = [spec.key for spec in runtime_specs]
    runtime_components = [spec.component for spec in runtime_specs]
    c.update({spec.key: spec.component for spec in runtime_specs})

    def values_to_config(*items: Any) -> tuple[dict[str, Any], RuntimeConfig]:
        values = dict(zip(runtime_keys, items))
        options = runtime_config_from_values(values, model_dir=model_dir)
        return options, RuntimeConfig.from_dict(options)

    def apply_tier(tier_value: str, device_value: str):
        if tier_value == "custom":
            return (*[gr.skip()] * len(tier_output_specs), "Custom runtime settings.", gr.skip())
        total = _gpu_total(device_value)
        free = _gpu_free(device_value)
        requested = tier_value if tier_value != "auto" else str(auto_tier(total) if total else 6)
        cfg = resolve_preset(requested, total or float(requested), free)
        cfg.device = device_value
        values = cfg.to_dict()
        flat = {
            **{f"runtime.{key}": value for key, value in values.items() if key != "aux_residency"},
            **{f"runtime.aux_residency.{key}": value for key, value in values["aux_residency"].items()},
            "runtime.vram_tier": tier_value,
        }
        updates = []
        # Tier and device are inputs and are intentionally not outputs here.
        output_specs = [
            spec
            for spec in runtime_specs
            if spec.key
            not in {
                "runtime.device",
                "runtime.vram_tier",
                "runtime.lora_path",
                "runtime.lora_strength",
                "runtime.lora_merge_into_base",
                "runtime.use_qwen_emo",
                "runtime.use_deepspeed",
            }
        ]
        for spec in output_specs:
            updates.append(flat.get(spec.key, gr.skip()))
        notes = preset_notes(requested)
        estimate = _estimate_html(cfg, total or float(requested))
        return (*updates, notes, estimate)

    tier_output_specs = [
        spec
        for spec in runtime_specs
        if spec.key
        not in {
            "runtime.device",
            "runtime.vram_tier",
            "runtime.lora_path",
            "runtime.lora_strength",
            "runtime.lora_merge_into_base",
            "runtime.use_qwen_emo",
            "runtime.use_deepspeed",
        }
    ]
    tab.tier.change(
        apply_tier,
        [tab.tier, tab.device],
        [*[spec.component for spec in tier_output_specs], tab.notes, tab.estimate],
        queue=False,
    )

    def estimate_runtime(*items: Any):
        try:
            _, cfg = values_to_config(*items)
            return _estimate_html(cfg, _gpu_total(cfg.device))
        except Exception as exc:
            return f'<div class="status-error">Estimate failed: {exc}</div>'

    for component in runtime_components:
        if component in {tab.tier}:
            continue
        component.change(estimate_runtime, runtime_components, tab.estimate, queue=False, show_progress="hidden", trigger_mode="always_last")

    def apply_runtime(*items: Any):
        try:
            options, cfg = values_to_config(*items)
            previous = json.dumps(APPLIED_RUNTIME, sort_keys=True, default=str)
            current = json.dumps(options, sort_keys=True, default=str)
            APPLIED_RUNTIME.clear()
            APPLIED_RUNTIME.update(options)
            saved_path = persist_runtime_config(cfg)
            if previous and previous != current and LAZY_ENGINE.peek() is not None:
                LAZY_ENGINE.unload()
                detail = " Changed settings required the in-process model to unload; it will reload lazily."
            else:
                detail = " Models remain lazy and are not loaded by Apply."
            message = (
                f"Applied {cfg.device} | {cfg.model_variant}/{cfg.gpt_dtype} | tier {cfg.vram_tier}."
                f" Saved to {saved_path}.{detail}"
            )
            print(">> " + message, flush=True)
            return message, _estimate_html(cfg, _gpu_total(cfg.device))
        except Exception as exc:
            traceback.print_exc()
            raise gr.Error(str(exc)) from exc

    apply_button.click(apply_runtime, runtime_components, [tab.apply_status, tab.estimate], api_name="apply_runtime", queue=False)

    def unload():
        unloaded = LAZY_ENGINE.unload()
        return "Model unloaded and VRAM caches released." if unloaded else "No in-process model was loaded."

    unload_button.click(unload, outputs=tab.apply_status, queue=False)
    refresh_gpu.click(lambda: (gr.update(value=_gpu_rows()), gr.update(choices=_device_choices()), gr.update(choices=_tier_choices())), outputs=[gpu_table, tab.device, tab.tier], queue=False)
    refresh_files.click(lambda: _model_status_rows(model_dir), outputs=tab.model_status, queue=False)
    open_models.click(lambda: open_folder(model_dir), outputs=download_status, queue=False)

    def download_int8(progress=gr.Progress(track_tqdm=False)):
        started = time.perf_counter()
        print(">> INT8 model download/verification started", flush=True)

        def callback(*values: Any, **kwargs: Any):
            message = str(kwargs.get("message") or (values[-1] if values else "Downloading INT8 GPT..."))
            fraction = kwargs.get("fraction")
            if fraction is None and values and isinstance(values[0], (int, float)):
                fraction = values[0]
            try:
                progress(float(fraction), desc=message) if fraction is not None else progress(0, desc=message)
            except Exception:
                pass
            print(">> " + message, flush=True)

        try:
            path = ensure_int8_gpt(model_dir, callback)
            info = describe_checkpoint(path)
            message = f"INT8 GPT ready: {path} ({info.get('quantized_layers', 0)} quantized layers) in {time.perf_counter() - started:.1f}s."
            print(">> " + message, flush=True)
            return message, _model_status_rows(model_dir)
        except Exception as exc:
            traceback.print_exc()
            raise gr.Error(str(exc)) from exc

    def download_base(progress=gr.Progress(track_tqdm=False)):
        started = time.perf_counter()
        print(">> Base model download/verification started", flush=True)

        def callback(*values: Any, **kwargs: Any):
            message = str(kwargs.get("message") or (values[-1] if values else "Downloading base models..."))
            try:
                fraction = kwargs.get("fraction", values[0] if values and isinstance(values[0], (int, float)) else 0)
                progress(float(fraction), desc=message)
            except Exception:
                pass
            print(">> " + message, flush=True)

        try:
            ensure_base_models(model_dir, callback)
            message = f"Base models verified in {time.perf_counter() - started:.1f}s."
            print(">> " + message, flush=True)
            return message, _model_status_rows(model_dir)
        except Exception as exc:
            traceback.print_exc()
            raise gr.Error(str(exc)) from exc

    int8_download.click(download_int8, outputs=[download_status, tab.model_status], concurrency_limit=1, concurrency_id="model-download")
    base_download.click(download_base, outputs=[download_status, tab.model_status], concurrency_limit=1, concurrency_id="model-download")

    def benchmark(tier_value: str, device_value: str, emulate_value: bool, subtitle_value: bool):
        resolved = tier_value
        if resolved in {"auto", "custom"}:
            total = _gpu_total(device_value)
            resolved = str(auto_tier(total) if total else 6)
        state_dir = ROOT / "outputs" / "vram_benchmark" / f"ui_{int(time.time())}"
        command = [sys.executable, str(ROOT / "tools" / "vram_benchmark.py"), "--tier", str(resolved)]
        if emulate_value:
            command.append("--emulate")
        if subtitle_value:
            command.append("--subtitle")
        device_match = re.search(r"(\d+)$", str(device_value or ""))
        selected_env = {"CUDA_VISIBLE_DEVICES": device_match.group(1)} if device_match else None
        job = PROCESS_MANAGER.start(
            "vram_benchmark",
            command,
            state_dir=state_dir,
            log_path=state_dir / "benchmark.log",
            cwd=ROOT,
            env=selected_env,
        )
        while job.running:
            yield tail_text(job.log_path, 80) or "Benchmark starting..."
            time.sleep(1)
        output = tail_text(job.log_path, 120)
        if job.process.returncode != 0:
            raise gr.Error(f"VRAM benchmark failed with exit code {job.process.returncode}\n{output[-1500:]}")
        yield output

    benchmark_button.click(benchmark, [tab.tier, tab.device, emulate, subtitle_bench], benchmark_output, concurrency_limit=1, concurrency_id="vram-benchmark")
    return tab


__all__ = [
    "APPLIED_RUNTIME",
    "AUX_NAMES",
    "LAST_RUNTIME_PATH",
    "ModelsTab",
    "RUNTIME_DEFAULTS",
    "build_models_tab",
    "load_persisted_runtime",
    "persist_runtime_config",
    "runtime_registry_values",
]
