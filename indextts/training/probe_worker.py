"""Render and measure one epoch probe in its own process (see :mod:`indextts.training.probe`).

The trainer starts this worker after an epoch with a temporary copy of the current adapter. The worker
renders the frozen probe sentences with the deployment settings, renders Base the same way once and caches
that measurement, measures every clip against the real recordings, and writes
``analysis/probe/epoch_NNN.json``. It never touches the training process: memory follows the epoch
sample's rules (free-VRAM gate, tier fitted to the free memory, or another GPU with room), the generation
engine is released before the measurement models load, and any failure is reported in the status file
instead of raised into training.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback
from typing import Any

from .dataset_manifest import atomic_write_json


def _device_index(device: str) -> int:
    try:
        return int(str(device).split(":", 1)[1]) if ":" in str(device) else 0
    except (TypeError, ValueError):
        return 0


def resolve_probe_device(config: Any) -> tuple[str, bool, str]:
    """Pick the GPU the probe renders on: ``(device, shares_training_gpu, note)``.

    ``probe_device`` "same" or an explicit device is honored. "auto" prefers another CUDA device with at least
    the sample free-VRAM threshold available (the one with the most free memory), so the probe never competes
    with the training model; otherwise it uses the training GPU behind the same gate the epoch sample uses.
    """

    from indextts.runtime import gpu_free_gb, list_gpus
    requested = str(getattr(config, "probe_device", "auto") or "auto").strip().lower()
    training = str(config.device)
    training_index = _device_index(training)
    threshold = float(getattr(config, "sample_min_free_vram_gb", 6.0) or 0.0)
    if requested == "same" or not training.startswith("cuda"):
        return training, True, ""
    if requested.startswith("cuda:"):
        return requested, _device_index(requested) == training_index, ""
    others = []
    try:
        for gpu in list_gpus():
            if int(gpu.index) == training_index:
                continue
            free = gpu_free_gb(int(gpu.index))
            if free >= threshold:
                others.append((free, int(gpu.index), str(gpu.name)))
    except Exception:
        others = []
    if others:
        free, index, name = max(others)
        return f"cuda:{index}", False, f"probe renders on cuda:{index} ({name}, {free:.1f} GB free) beside the training GPU"
    return training, True, ""


def probe_runtime(config: Any, device: str, *, share_gpu: bool) -> Any:
    """Runtime for the probe process: the training tier fitted to the free memory of the chosen GPU."""

    from indextts.runtime import gpu_free_gb, gpu_total_gb, resolve_preset
    from indextts.runtime.vram_presets import auto_tier, fit_tier_to_free_vram
    index = _device_index(device)
    total = gpu_total_gb(index)
    free = gpu_free_gb(index)
    requested = str(getattr(config, "sample_runtime_tier", "auto") or "auto").strip().lower()
    if requested == "auto":
        training_tier = str(getattr(config, "vram_tier", "auto") or "auto").strip().lower()
        requested = training_tier if training_tier != "auto" else str(auto_tier(total))
    # Another GPU can also be busy; only its actually free memory is available.
    resolved = fit_tier_to_free_vram(requested, free)
    runtime = resolve_preset(str(resolved), total, free)
    runtime.device = device
    runtime.lora_path = ""
    runtime.decoder_adapter = "none"
    return runtime


def deployment_tier(config: Any) -> str | int:
    """The tier whose beams and diffusion steps the deployed voice will use: the training GPU's."""

    tier = str(getattr(config, "vram_tier", "auto") or "auto").strip().lower()
    if tier != "auto":
        return tier
    try:
        from indextts.runtime import gpu_total_gb
        from indextts.runtime.vram_presets import auto_tier
        return auto_tier(gpu_total_gb(_device_index(config.device)))
    except Exception:
        return 32


def _settings_digest(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _verify_plan_files(plan: dict[str, Any]) -> None:
    if hashlib.sha256(Path(plan["reference"]).read_bytes()).hexdigest() != plan["reference_sha256"]:
        raise ValueError("the probe reference clip has changed")
    for prompt in plan["prompts"]:
        if hashlib.sha256(Path(prompt["audio"]).read_bytes()).hexdigest() != prompt["audio_sha256"]:
            raise ValueError("a probe recording has changed")


def _clips_from_grid(result: Any, plan: dict[str, Any], label: str) -> list[dict[str, Any]]:
    clips = []
    for cell in result.cells:
        prompt = plan["prompts"][cell.text_index - 1]
        clips.append({"audio": cell.audio_path, "reference": plan["reference"], "real_audio": prompt["audio"],
                      "text": prompt["text"], "language": plan["language"], "kind": "matched",
                      "prompt_id": prompt["id"], "source": prompt["source"], "seed": cell.seed, "checkpoint": label})
    return clips


def run_probe(config: Any, state_dir: str | Path, *, adapter_path: str, epoch: int) -> dict[str, Any]:
    import torch
    from indextts.runtime import ProgressReporter, gpu_free_gb
    from .deployment_settings import deployment_infer_kwargs
    from .grid import GridCheckpoint, GridConfig, run_grid
    from .probe import load_probe_plan, probe_root, score_probe_rows
    from .speech_eval import _lenient_terms
    from .speech_metrics import measure_clips

    run_dir = Path(config.output_dir).resolve() / config.name
    root = probe_root(run_dir)
    state = Path(state_dir)
    state.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def cancelled() -> bool:
        return (state / "stop.flag").exists() or (run_dir / "stop.flag").exists()

    def update(message: str, completed: int = 0, total: int = 0) -> None:
        value = {"phase": "probing", "message": message, "desc": message, "completed": completed, "total": total,
                 "fraction": completed / total if total else 0, "elapsed_s": time.perf_counter() - started,
                 "updated_at": time.time()}
        atomic_write_json(state / "status.json", value)
        atomic_write_json(state / "progress.json", value)
        print(f">> {message}" + (f": {completed}/{total}" if total else ""), flush=True)

    plan = load_probe_plan(run_dir)
    if not plan or not plan.get("prompts"):
        raise ValueError("no epoch probe plan is frozen for this run")
    _verify_plan_files(plan)
    adapter = Path(adapter_path).expanduser().resolve()
    if not adapter.is_file():
        raise FileNotFoundError(f"probe adapter is missing: {adapter}")
    device, shares_gpu, note = resolve_probe_device(config)
    if note:
        print(">> " + note, flush=True)
    if str(device).startswith("cuda"):
        from indextts.runtime.vram_presets import VRAM_TIERS, tier_budget_gb
        free = gpu_free_gb(_device_index(device))
        # Same rule as the trainer's gate: the smallest generation tier must fit into the free memory.
        threshold = max(float(getattr(config, "sample_min_free_vram_gb", 6.0) or 0.0), float(tier_budget_gb(min(VRAM_TIERS))))
        if free < threshold:
            message = f"probe skipped: {free:.1f} GB free VRAM on {device} is below the {threshold:.1f} GB threshold"
            atomic_write_json(state / "status.json", {"phase": "skipped", "message": message, "elapsed_s": time.perf_counter() - started})
            print(">> " + message, flush=True)
            return {"status": "skipped", "message": message}
    runtime = probe_runtime(config, device, share_gpu=shares_gpu)
    tier = deployment_tier(config)
    language = str(plan.get("language") or "EN").upper()
    infer_adapter = deployment_infer_kwargs(config, str(adapter), language=language, tier=tier)
    infer_base = deployment_infer_kwargs(config, "", language=language, tier=tier)
    grid_runtime = {"runtime": runtime.to_dict(), "model_dir": config.model_dir, "cfg_path": config.model_config,
                    "use_qwen_emo": False}
    seeds = [int(seed) for seed in plan.get("seeds") or [plan.get("seed", 0)]]
    texts = [prompt["text"] for prompt in plan["prompts"]]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    lenient = _lenient_terms(config, {"groups": [{"language": language}]})

    base_cache = root / "base" / "measured.json"
    base_digest = _settings_digest(infer_base, plan["dataset_identity"], plan["reference_sha256"], seeds, texts,
                                   runtime.gpt_dtype, runtime.base_variant if hasattr(runtime, "base_variant") else "")
    base_rows: list[dict[str, Any]] = []
    real_rows: list[dict[str, Any]] = []
    cached = None
    try:
        cached = json.loads(base_cache.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        cached = None
    if cached and cached.get("digest") == base_digest and cached.get("rows"):
        base_rows = list(cached["rows"])
        real_rows = list(cached.get("real_rows") or [])
    clips: list[dict[str, Any]] = []
    grids: dict[str, str] = {}
    total_cells = len(texts) * len(seeds)
    base_render_s = 0.0
    if not base_rows:
        update("Rendering Base once for the epoch probe", 0, total_cells)
        result = run_grid(GridConfig(adapter_dir=str(run_dir), checkpoints=[GridCheckpoint("Base", "")],
                                     references=[plan["reference"]], texts=texts, language=language, seeds=seeds, seed=seeds[0],
                                     output_root=str(root / "base"), grid_name=f"grid_{stamp}", runtime=grid_runtime,
                                     infer_kwargs=infer_base, include_verdicts=False),
                          reporter=ProgressReporter("probe clips", progress_file=state / "progress.json"), cancel_callback=cancelled)
        if result.status != "complete":
            raise InterruptedError(f"probe Base render {result.status}")
        grids["base"] = result.grid_dir
        base_render_s = time.perf_counter() - started
        clips.extend(_clips_from_grid(result, plan, "Base"))
        for prompt in plan["prompts"]:
            clips.append({"audio": prompt["audio"], "reference": plan["reference"], "real_audio": "", "text": prompt["text"],
                          "language": language, "kind": "real", "prompt_id": prompt["id"], "source": prompt["source"],
                          "seed": 0, "checkpoint": "Real recordings"})
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    update(f"Rendering the epoch {epoch} probe with the deployment settings", 0, total_cells)
    label = f"epoch {epoch}"
    result = run_grid(GridConfig(adapter_dir=str(run_dir), checkpoints=[GridCheckpoint(label, str(adapter))],
                                 references=[plan["reference"]], texts=texts, language=language, seeds=seeds, seed=seeds[0],
                                 output_root=str(root / "grids"), grid_name=f"epoch_{int(epoch):03d}_{stamp}", runtime=grid_runtime,
                                 infer_kwargs=infer_adapter, include_verdicts=False),
                      reporter=ProgressReporter("probe clips", progress_file=state / "progress.json"), cancel_callback=cancelled)
    if result.status != "complete":
        raise InterruptedError(f"probe render {result.status}")
    grids["epoch"] = result.grid_dir
    clips.extend(_clips_from_grid(result, plan, label))
    # The generation engine is only referenced inside run_grid; release its memory before the measurement models load.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    update("Measuring the probe clips against the real recordings", 0, len(clips))
    measured = measure_clips(clips, model_dir=config.model_dir, model_config=config.model_config, device=device,
                             output_dir=root / "measure", update=lambda message, done, total: update(message, done, total),
                             cancelled=cancelled, lenient_terms=lenient)
    rows = [row for row in measured if row["checkpoint"] == label]
    if not base_rows:
        base_rows = [row for row in measured if row["checkpoint"] == "Base"]
        real_rows = [row for row in measured if row["kind"] == "real"]
        atomic_write_json(base_cache, {"digest": base_digest, "rows": base_rows, "real_rows": real_rows,
                                       "infer_kwargs": infer_base, "grid": grids.get("base", ""),
                                       "generated_at": datetime.now(timezone.utc).isoformat()})
    real_error = None
    if real_rows:
        real_error = sum(float(row["errors"]) for row in real_rows) / max(1, sum(int(row["units"]) for row in real_rows))
    scored = score_probe_rows(rows, base_rows, plan.get("policy") or {})
    report = {"status": "complete", "epoch": int(epoch), "adapter": str(adapter),
              "adapter_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(), "device": device,
              "runtime_tier": runtime.vram_tier, "deployment_tier": str(tier), "infer_kwargs": infer_adapter,
              "base_infer_kwargs": infer_base, "rows": rows, "base_rows": base_rows, "real_error_rate": real_error,
              "grid": grids.get("epoch", ""), "base_grid": grids.get("base", "") or (cached or {}).get("grid", ""),
              "elapsed_s": time.perf_counter() - started,
              # The one-time Base render is not part of what a later probe costs; pacing uses the rest.
              "pacing_elapsed_s": time.perf_counter() - started - base_render_s,
              "generated_at": datetime.now(timezone.utc).isoformat(), **scored}
    destination = root / f"epoch_{int(epoch):03d}.json"
    atomic_write_json(destination, report)
    atomic_write_json(state / "result.json", {"status": "complete", "report": str(destination)})
    atomic_write_json(state / "status.json", {"phase": "complete", "message": "Epoch probe complete", "elapsed_s": report["elapsed_s"]})
    score = report["deployment_score"]["score"]
    print(f">> epoch {epoch} probe: deployment score {score:+.4f} vs Base | word error "
          f"{100 * report['summary']['mean_error_rate']:.2f}% (Base {100 * report['base']['mean_error_rate']:.2f}%) | "
          f"speaker similarity {report['summary'].get('speaker_similarity_real') or report['summary'].get('speaker_similarity'):.4f} "
          f"| {report['elapsed_s']:.0f}s", flush=True)
    return report


def main() -> int:
    from indextts.utils.console_encoding import configure_console_output
    from .train_config import TrainConfig
    configure_console_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--adapter", required=True, help="Temporary adapter file of the epoch under test")
    parser.add_argument("--epoch", required=True, type=int)
    args = parser.parse_args()
    try:
        run_probe(TrainConfig.from_json(args.config), args.state_dir, adapter_path=args.adapter, epoch=args.epoch)
        return 0
    except BaseException as exc:
        traceback.print_exc()
        atomic_write_json(Path(args.state_dir) / "status.json",
                          {"phase": "cancelled" if isinstance(exc, InterruptedError) else "failed", "message": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
