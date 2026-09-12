"""Measure the whole-GPU peak memory of every GPU VRAM tier preset.

The tool runs the same worker processes the app starts (the generation worker,
the dataset preparation worker, the voice audit, the feature cache and the
training worker with its sample, evaluation and decoder phases) with the values a
tier preset selects, while a poller samples ``nvidia-smi`` for the selected GPU
every 200 ms. The recorded number is the memory the card itself sees: every
process, every CUDA context, allocator slack included. That is what has to stay
within a tier's budget on a real card of that size.

Examples::

    python tools/gpu_tier_calibration.py --stages inference
    python tools/gpu_tier_calibration.py --tiers 6,8 --stages inference,train
    python tools/gpu_tier_calibration.py --stages prep,audit,cache,train --source a.flac --source b.flac
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from indextts.runtime.vram_presets import VRAM_TIERS, tier_budget_gb


CALIBRATION_TEXT = (
    "Every voice carries a history of tiny choices: a pause before an important word, a smile hidden inside a "
    "sentence, and a rhythm learned over many years. This calibration asks the model to preserve those details "
    "while reading a practical passage about clear speech, patient listening, and the quiet confidence that "
    "comes from explaining a difficult idea in language that anyone can understand without rushing. When the "
    "reading ends, the measured peak memory tells us whether this card can carry the full quality settings."
)
EMOTION_TEXT_MODE = "Emotion text"
_MIB = 1024.0


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


@dataclass
class Window:
    label: str
    started: float
    ended: float | None = None
    peak_mib: float = 0.0
    baseline_mib: float | None = None
    phase_peaks: dict[str, float] = field(default_factory=dict)
    samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "seconds": round((self.ended or time.monotonic()) - self.started, 1),
            "peak_gb": round(self.peak_mib / _MIB, 3),
            "baseline_gb": None if self.baseline_mib is None else round(self.baseline_mib / _MIB, 3),
            "phase_peaks_gb": {name: round(value / _MIB, 3) for name, value in sorted(self.phase_peaks.items())},
            "samples": self.samples,
        }


class GpuPoller:
    """Sample nvidia-smi memory use of one physical GPU at a fixed interval."""

    def __init__(self, gpu: int, interval_s: float = 0.2, timeline_path: Path | None = None) -> None:
        self.gpu = int(gpu)
        self.interval_s = float(interval_s)
        self.timeline_path = timeline_path
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._window: Window | None = None
        self._phase_fn: Callable[[], str] | None = None
        self.total_mib = 0.0
        self.last_mib = 0.0
        self.overall_peak_mib = 0.0
        self._thread = threading.Thread(target=self._run, name="gpu-poller", daemon=True)
        self._timeline = None

    def read(self) -> float:
        completed = subprocess.run(
            ["nvidia-smi", f"--id={self.gpu}", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10, check=False,
        )
        rows = [row for row in csv.reader(completed.stdout.splitlines()) if row]
        if completed.returncode != 0 or not rows or len(rows[0]) < 2:
            raise RuntimeError(f"nvidia-smi failed for GPU {self.gpu}: {(completed.stderr or completed.stdout).strip()[:300]}")
        used, total = float(rows[0][0].strip()), float(rows[0][1].strip())
        self.total_mib = total
        return used

    def start(self) -> None:
        self.last_mib = self.read()
        if self.timeline_path is not None:
            self.timeline_path.parent.mkdir(parents=True, exist_ok=True)
            self._timeline = self.timeline_path.open("w", encoding="utf-8", newline="\n")
            self._timeline.write("time_s,used_mib,window,phase\n")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        if self._timeline is not None:
            self._timeline.close()

    def _run(self) -> None:
        origin = time.monotonic()
        while not self._stop.is_set():
            try:
                used = self.read()
            except Exception as exc:  # keep sampling through transient driver hiccups
                print(f">> poller: {exc}", flush=True)
                time.sleep(self.interval_s)
                continue
            with self._lock:
                self.last_mib = used
                self.overall_peak_mib = max(self.overall_peak_mib, used)
                window = self._window
                phase = ""
                if window is not None:
                    window.samples += 1
                    window.peak_mib = max(window.peak_mib, used)
                    if self._phase_fn is not None:
                        try:
                            phase = str(self._phase_fn() or "")
                        except Exception:
                            phase = ""
                        if phase:
                            window.phase_peaks[phase] = max(window.phase_peaks.get(phase, 0.0), used)
                if self._timeline is not None:
                    self._timeline.write(f"{time.monotonic() - origin:.2f},{used:.0f},{window.label if window else ''},{phase}\n")
            time.sleep(self.interval_s)

    @contextmanager
    def window(self, label: str, phase_fn: Callable[[], str] | None = None) -> Iterator[Window]:
        baseline = self.read()
        item = Window(label=label, started=time.monotonic(), baseline_mib=baseline, peak_mib=baseline)
        with self._lock:
            self._window = item
            self._phase_fn = phase_fn
        try:
            yield item
        finally:
            with self._lock:
                item.ended = time.monotonic()
                self._window = None
                self._phase_fn = None


# --------------------------------------------------------------------------- app values


class TierValues:
    """Registry values of the tier presets, taken from a built app."""

    def __init__(self, model_dir: Path) -> None:
        from types import SimpleNamespace

        from ui.app import build_app

        print(">> building the app once to read the registered presets", flush=True)
        self.demo = build_app(SimpleNamespace(model_dir=str(model_dir), device="cpu", verbose=False, no_browser=True,
                                              port=7861, host="127.0.0.1", share=False))
        self.store = self.demo.preset_store
        self.registry = self.demo.preset_registry

    def values(self, tier: int, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        values = self.store.tier_preset_values(tier)
        if overrides:
            values = self.registry.coerce({**values, **overrides})
        return values


def _parse_overrides(items: list[str]) -> dict[str, Any]:
    """``key=value`` pairs applied on top of a tier's preset values (JSON values when parseable)."""

    result: dict[str, Any] = {}
    for item in items:
        key, separator, raw = str(item).partition("=")
        if not separator or not key.strip():
            raise SystemExit(f"--override expects key=value, got {item!r}")
        try:
            result[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            result[key.strip()] = raw
    return result


def _child_env(gpu: int, cap_gb: float | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if cap_gb:
        # Every worker then behaves like a card of that size: its allocator is capped
        # at the tier budget and it reports the tier size (see indextts.runtime.gpu).
        env["INDEXTTS_VRAM_EMULATE_GB"] = f"{float(cap_gb):g}"
    else:
        env.pop("INDEXTTS_VRAM_EMULATE_GB", None)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")
    env.setdefault("CUDA_MODULE_LOADING", "LAZY")
    env.setdefault("PYTHONWARNINGS", "ignore")
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    return env


def _emulated_cap(args: argparse.Namespace, tier: int) -> float | None:
    """The card size every worker emulates for a tier when ``--emulate`` is on."""

    return float(tier) if getattr(args, "emulate", False) else None


def _run(command: list[str], *, log_path: Path, env: dict[str, str], cwd: Path = ROOT) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f">> running: {' '.join(str(part) for part in command[:4])} ... (log {log_path.name})", flush=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as handle:
        handle.write(" ".join(str(part) for part in command) + "\n\n")
        handle.flush()
        completed = subprocess.run(command, cwd=str(cwd), env=env, stdout=handle, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", check=False)
    return completed.returncode


def _tail(path: Path, lines: int = 8) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


# --------------------------------------------------------------------------- stages


def inference_stage(tiers: TierValues, tier: int, variant: str, args: argparse.Namespace, poller: GpuPoller,
                    out_dir: Path) -> dict[str, Any]:
    from ui.generation_tab import build_generation_request

    values = dict(tiers.values(tier, args.overrides))
    reference = Path(args.reference).resolve()
    if variant == "lora":
        adapter = Path(args.lora).resolve()
        values["runtime.lora_path"] = str(adapter)
        values["runtime.decoder_adapter"] = "auto"
        candidate = next(iter(sorted(p for p in adapter.parent.glob("*_reference.*") if "_expressive_reference" not in p.name.lower())), None)
        if candidate is not None:
            reference = candidate.resolve()
    elif variant == "emotion_text":
        values["generation.emotion_mode"] = EMOTION_TEXT_MODE
        values["generation.emotion_text"] = "calm, warm and confident"
    job = out_dir / f"inference_{tier}gb_{variant}{('_' + args.label) if args.label else ''}"
    job.mkdir(parents=True, exist_ok=True)
    layout = {
        "task_id": job.name,
        "task_folder": str(job),
        "final_basename": "calibration",
        "final_wav_path": str(job / "calibration.wav"),
        "final_mp3_path": str(job / "calibration.mp3"),
        "final_mp4_path": str(job / "calibration.mp4"),
        "segments_dir": str(job / "segments"),
        "speaker_reference_copy_path": str(job / "reference.wav"),
    }
    now = datetime.now(timezone.utc).isoformat()
    metadata = {"status": "in_progress", "created_at": now, "updated_at": now, "outputs": {},
                "processing": {"started_at": now, "ended_at": None, "elapsed_ms": None, "elapsed_seconds": None,
                               "elapsed_human": None}, "error": None}
    (job / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    request = build_generation_request(values, prompt=str(reference), text=args.text, task_layout=layout,
                                       metadata_path=str(job / "metadata.json"), progress_file=str(job / "progress.json"),
                                       model_dir=str(args.model_dir))
    request["seed"] = 123
    (job / "request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
    command = [sys.executable, str(ROOT / "webui_subprocess_worker.py"), "--request-file", str(job / "request.json"),
               "--result-file", str(job / "result.json")]
    started = time.perf_counter()
    with poller.window(f"inference {tier} GB {variant}") as window:
        code = _run(command, log_path=job / "worker.log", env=_child_env(args.gpu, _emulated_cap(args, tier)))
    wall = time.perf_counter() - started
    result: dict[str, Any] = {}
    try:
        result = json.loads((job / "result.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    record = {
        "tier": tier, "stage": "inference", "variant": variant, "returncode": code, "wall_s": round(wall, 1),
        "overrides": dict(args.overrides), "emulated_cap_gb": _emulated_cap(args, tier),
        "audio_seconds": result.get("audio_seconds"), "rtf": result.get("rtf"),
        "engine_peak_allocated_gb": result.get("peak_vram_gb"), "status": result.get("status"),
        "error": result.get("error") or (None if code == 0 else _tail(job / "worker.log")),
        "runtime": {key: values[key] for key in sorted(values) if key.startswith("runtime.") and "aux" not in key
                    and key.split(".")[1] in {"model_variant", "blocks_to_swap", "swap_ring_size", "cfm_cache_length",
                                              "s2mel_estimator_autocast", "vram_tier"}},
        "generation": {key: values[key] for key in ("generation.num_beams", "generation.section_batch_size",
                                                   "generation.max_text_tokens_per_segment", "generation.low_memory_mode",
                                                   "generation.diffusion_steps")},
        **window.to_dict(),
    }
    return record


def prep_stage(tiers: TierValues, tier: int, args: argparse.Namespace, poller: GpuPoller, out_dir: Path,
               name: str) -> dict[str, Any]:
    from indextts.training.dataset_prep import DatasetPrepConfig

    values = tiers.values(tier, args.overrides)
    payload = {key.removeprefix("dataset."): value for key, value in values.items() if key.startswith("dataset.")}
    payload.update({"inputs": [str(Path(item).resolve()) for item in args.source], "name": name,
                    "output_root": str((out_dir / "datasets").resolve()), "overwrite": True})
    config = DatasetPrepConfig.from_dict(payload)
    config.validate()
    state = out_dir / f"prep_{name}"
    state.mkdir(parents=True, exist_ok=True)
    (state / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    command = [sys.executable, "-m", "indextts.training.prep_worker", "--config", str(state / "config.json"),
               "--state-dir", str(state)]
    started = time.perf_counter()
    with poller.window(f"prep {name}", phase_fn=lambda: _status_phase(state / "status.json")) as window:
        code = _run(command, log_path=state / "worker_console.log", env=_child_env(args.gpu, _emulated_cap(args, tier)))
    dataset_dir = Path(config.output_root) / config.name
    info = _read_json(dataset_dir / "dataset_info.json")
    return {"tier": tier, "stage": "prep", "returncode": code, "wall_s": round(time.perf_counter() - started, 1),
            "dataset_dir": str(dataset_dir), "segments": info.get("segment_count"),
            "minutes": info.get("total_duration_minutes"), "whisper_model": config.whisper_model,
            "error": None if code == 0 else _tail(state / "worker_console.log"), **window.to_dict()}


def audit_stage(tiers: TierValues, tier: int, args: argparse.Namespace, poller: GpuPoller, out_dir: Path,
                dataset_dir: Path) -> dict[str, Any]:
    from ui.dataset_curation import curation_command

    values = tiers.values(tier, args.overrides)
    settings = {key.removeprefix("curation."): value for key, value in values.items() if key.startswith("curation.")}
    references = sorted((dataset_dir / "reference_candidates").glob("*.wav"))
    if not references:
        raise FileNotFoundError(f"no reference candidates in {dataset_dir}")
    sources = sorted({Path(str(row.get("source_media", ""))).stem for row in _read_manifest(dataset_dir)})
    if len(sources) < 2:
        raise ValueError("the audit needs at least two source recordings (one is held out for validation)")
    # --final-test reserves a second recording so the training run also exercises the frozen final test,
    # which renders Base and the deployed pipeline (adapter, decoder, adopted decoding settings) once more.
    final_test = bool(getattr(args, "final_test", False))
    if final_test and len(sources) < 3:
        raise ValueError("--final-test needs at least three source recordings (training, validation and final test)")
    settings.update({"name": f"{dataset_dir.name}_audited", "references": str(references[0]),
                     "validation_sources": sources[-1], "test_sources": sources[-2] if final_test else ""})
    command, output = curation_command(str(dataset_dir), settings, model_dir=str(args.model_dir))
    state = out_dir / f"audit_{dataset_dir.name}"
    state.mkdir(parents=True, exist_ok=True)
    command = command + ["--state-dir", str(state)]
    started = time.perf_counter()
    with poller.window(f"audit {dataset_dir.name}", phase_fn=lambda: _status_phase(state / "status.json")) as window:
        code = _run(command, log_path=state / "audit.log", env=_child_env(args.gpu, _emulated_cap(args, tier)))
    info = _read_json(output / "dataset_info.json")
    test_dir = output.with_name(output.name + "_test")
    return {"tier": tier, "stage": "audit", "returncode": code, "wall_s": round(time.perf_counter() - started, 1),
            "dataset_dir": str(output), "segments": info.get("segment_count"),
            "test_dataset_dir": str(test_dir) if final_test and (test_dir / "dataset_info.json").is_file() else "",
            "second_opinion": settings.get("second_opinion"),
            "error": None if code == 0 else _tail(state / "audit.log"), **window.to_dict()}


def cache_stage(tier: int, args: argparse.Namespace, poller: GpuPoller, out_dir: Path, dataset_dir: Path) -> dict[str, Any]:
    state = out_dir / f"cache_{dataset_dir.name}"
    state.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(ROOT / "tools" / "cache_dataset_features.py"), "--dataset-dir", str(dataset_dir),
               "--model-dir", str(args.model_dir), "--device", "cuda:0", "--progress-file", str(state / "progress.json")]
    started = time.perf_counter()
    with poller.window(f"cache {dataset_dir.name}") as window:
        code = _run(command, log_path=state / "cache.log", env=_child_env(args.gpu, _emulated_cap(args, tier)))
    return {"tier": tier, "stage": "cache", "returncode": code, "wall_s": round(time.perf_counter() - started, 1),
            "dataset_dir": str(dataset_dir), "error": None if code == 0 else _tail(state / "cache.log"),
            **window.to_dict()}


def train_stage(tiers: TierValues, tier: int, args: argparse.Namespace, poller: GpuPoller, out_dir: Path,
                dataset_dir: Path) -> dict[str, Any]:
    from ui.training_tab import train_config_from_values

    values = dict(tiers.values(tier, args.overrides))
    name = f"calib_{tier}gb_{_utc_stamp()}"
    values.update({
        "training.dataset_dir": str(dataset_dir), "training.name": name,
        "training.output_dir": str((out_dir / "loras").resolve()), "training.model_dir": str(args.model_dir),
        "training.model_config": str(Path(args.model_dir) / "config.yaml"), "training.device": "cuda:0",
        "training.epochs": int(args.epochs), "training.warmup_steps": 10, "training.val_every_steps": 20,
        "training.early_stop_enabled": False, "training.sample_every_epochs": 1, "training.sample_enabled": True,
        "training.speech_eval_prompts": 4, "training.speech_eval_seeds": 1, "training.speech_eval_candidates": 2,
        "training.decoder_adapter_epochs": 1, "training.decoding_sweep_enabled": bool(args.decoding_sweep),
        "training.eval_train_subset": 8, "training.final_test_dataset": str(getattr(args, "final_test_dataset", "") or ""),
        "training.decoder_adapter_enabled": not args.skip_decoder,
    })
    # --override values win over the stage defaults above, so a peak-only run can shrink every phase
    # (for example training.speech_eval_prompts=2 training.probe_prompts=1 training.probe_seeds=1).
    values.update({key: value for key, value in args.overrides.items() if key.startswith("training.")})
    config = train_config_from_values(values)
    adapter_dir = Path(config.output_dir) / config.name
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "train_config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    command = [sys.executable, "-m", "indextts.training.train_worker", "--config", str(adapter_dir / "train_config.json"),
               "--state-dir", str(adapter_dir)]
    tracker = _TrainingPhaseTracker(adapter_dir)
    started = time.perf_counter()
    with poller.window(f"train {tier} GB", phase_fn=tracker.phase) as window:
        code = _run(command, log_path=adapter_dir / "worker_console.log", env=_child_env(args.gpu, _emulated_cap(args, tier)))
    status = _read_json(adapter_dir / "status.json")
    return {
        "tier": tier, "stage": "train", "returncode": code, "wall_s": round(time.perf_counter() - started, 1),
        "emulated_cap_gb": _emulated_cap(args, tier),
        "adapter_dir": str(adapter_dir), "phase": status.get("phase"), "message": status.get("message"),
        "steps": status.get("step"), "trainer_vram_used_gb": status.get("vram_used_gb"),
        "sample": status.get("last_sample"), "speech_evaluation_status": status.get("speech_evaluation_status"),
        "decoder_adapter_status": status.get("decoder_adapter_status"),
        "training": {key: values[key] for key in ("training.vram_tier", "training.base_variant", "training.blocks_to_swap",
                                                 "training.swap_ring_size", "training.rank", "training.batch_size",
                                                 "training.sample_runtime_tier", "training.sample_min_free_vram_gb")},
        "sample_tier_lines": [line for line in _tail(adapter_dir / "log.txt", 400).splitlines()
                              if "sample runtime tier" in line or "sample skipped" in line or "sample saved" in line],
        "error": None if code == 0 else _tail(adapter_dir / "worker_console.log"),
        **window.to_dict(),
    }


class _TrainingPhaseTracker:
    """Name the current training phase, separating the sample subprocess window."""

    def __init__(self, adapter_dir: Path) -> None:
        self.adapter_dir = adapter_dir
        self._log_size = 0
        self._sampling = False

    def phase(self) -> str:
        phase = _status_phase(self.adapter_dir / "status.json") or "starting"
        log_path = self.adapter_dir / "log.txt"
        try:
            size = log_path.stat().st_size
        except OSError:
            size = 0
        if size != self._log_size:
            self._log_size = size
            for line in _tail(log_path, 40).splitlines():
                if "generating epoch" in line and "sample in a subprocess" in line:
                    self._sampling = True
                elif "sample saved" in line or "sample skipped" in line or "sample generation failed" in line \
                        or "sample worker" in line or "timed out" in line:
                    self._sampling = False
        if phase == "training" and self._sampling:
            return "training+sample"
        return phase


def _status_phase(path: Path) -> str:
    value = _read_json(path)
    return str(value.get("phase") or "") if value else ""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_manifest(dataset_dir: Path) -> list[dict[str, Any]]:
    from indextts.training.dataset_manifest import load_manifest

    try:
        return list(load_manifest(dataset_dir))
    except Exception:
        return []


# --------------------------------------------------------------------------- report


def _markdown(records: list[dict[str, Any]], poller: GpuPoller) -> str:
    lines = [
        f"GPU {poller.gpu}: {poller.total_mib / _MIB:.2f} GiB total. Peaks are whole-GPU nvidia-smi memory.used "
        f"(every process and CUDA context). Overall peak {poller.overall_peak_mib / _MIB:.2f} GiB.",
        "",
        "| Tier | Stage | Variant | Peak GB | Budget GB | Fits | Wall s | Detail |",
        "|---:|---|---|---:|---:|:---:|---:|---|",
    ]
    for item in records:
        tier = int(item["tier"])
        budget = tier_budget_gb(tier)
        peak = float(item.get("peak_gb") or 0.0)
        fits = "yes" if item.get("returncode") == 0 and peak <= budget else "no"
        if item["stage"] == "inference":
            detail = (f"beams {item['generation']['generation.num_beams']} | swap {item['runtime']['runtime.blocks_to_swap']} | "
                      f"audio {item.get('audio_seconds') or 0:.1f}s | RTF {item.get('rtf') or 0:.2f} | "
                      f"engine peak {item.get('engine_peak_allocated_gb') or 0:.2f}")
        elif item["stage"] == "train":
            phases = ", ".join(f"{name} {value:.2f}" for name, value in item.get("phase_peaks_gb", {}).items())
            detail = f"swap {item['training']['training.blocks_to_swap']} | {item.get('phase')} | {phases}"
        else:
            phases = ", ".join(f"{name} {value:.2f}" for name, value in item.get("phase_peaks_gb", {}).items())
            detail = phases or ""
        if item.get("emulated_cap_gb"):
            detail = f"emulated {item['emulated_cap_gb']:.0f} GB card | " + detail
        if item.get("error"):
            detail = f"ERROR: {str(item['error']).splitlines()[-1][:160]}"
        lines.append(f"| {tier} | {item['stage']} | {item.get('variant', '')} | {peak:.2f} | {budget:.0f} | {fits} | "
                     f"{item.get('wall_s', 0):.0f} | {detail} |")
    return "\n".join(lines) + "\n"


def _write_report(out_dir: Path, records: list[dict[str, Any]], poller: GpuPoller, args: argparse.Namespace) -> None:
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "gpu": poller.gpu,
               "gpu_total_gb": round(poller.total_mib / _MIB, 3), "overall_peak_gb": round(poller.overall_peak_mib / _MIB, 3),
               "arguments": {key: str(value) for key, value in vars(args).items()}, "records": records}
    (out_dir / "calibration.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "calibration.md").write_text(_markdown(records, poller), encoding="utf-8")


# --------------------------------------------------------------------------- main


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", type=int, default=0, help="Physical GPU index measured and used by the workers")
    parser.add_argument("--tiers", default=",".join(str(tier) for tier in VRAM_TIERS))
    parser.add_argument("--stages", default="inference,prep,audit,cache,train",
                        help="Comma-separated subset of inference, prep, audit, cache, train")
    parser.add_argument("--variants", default="base,lora", help="Inference variants: base, lora, emotion_text")
    parser.add_argument("--emotion-tiers", default="6,8,16,32", help="Tiers that also run the emotion_text variant")
    parser.add_argument("--model-dir", default=str(ROOT / "models"))
    parser.add_argument("--reference", default=str(ROOT / "examples" / "voice_01.wav"))
    parser.add_argument("--lora", default="", help="LoRA / DoRA file for the lora variant (default: newest in loras/)")
    parser.add_argument("--text", default=CALIBRATION_TEXT)
    parser.add_argument("--source", action="append", default=[], help="Source recording for the prep stage (repeatable)")
    parser.add_argument("--train-dataset", default="", help="Cached dataset for the train stage (default: the prepared one)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--decoding-sweep", action="store_true", help="Keep the decoding sweep in the training run")
    parser.add_argument("--skip-decoder", action="store_true", help="Skip the voice decoder adaptation phase")
    parser.add_argument("--final-test", action="store_true",
                        help="Reserve a third source recording for the frozen final test so the training run renders every phase")
    parser.add_argument("--final-test-dataset", default="",
                        help="Final-test dataset for a train-only run (the audit stage sets it automatically with --final-test)")
    parser.add_argument("--emulate", action="store_true",
                        help="Cap every worker's allocator at the tier budget and report that size as the card, so "
                             "the run behaves like a card of that tier (out-of-memory errors then mean it does not fit)")
    parser.add_argument("--interval", type=float, default=0.2, help="Poll interval in seconds")
    parser.add_argument("--output", default="", help="Output folder (default outputs/gpu_tier_calibration/<stamp>)")
    parser.add_argument("--override", action="append", default=[],
                        help="key=value applied on top of every tier preset, e.g. runtime.blocks_to_swap=16 (repeatable)")
    parser.add_argument("--label", default="", help="Suffix for the inference job folders of this run")
    return parser


def _newest_lora(root: Path) -> str:
    candidates = [path for path in root.rglob("*.safetensors") if ".s2mel" not in path.name and "epoch" not in path.name
                  and "_avg_" not in path.name and not path.name.startswith(".")]
    if not candidates:
        return ""
    return str(max(candidates, key=lambda path: path.stat().st_mtime))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    tiers_requested = [int(item) for item in str(args.tiers).split(",") if item.strip()]
    stages = [item.strip() for item in str(args.stages).split(",") if item.strip()]
    variants = [item.strip() for item in str(args.variants).split(",") if item.strip()]
    emotion_tiers = {int(item) for item in str(args.emotion_tiers).split(",") if item.strip()}
    out_dir = Path(args.output) if args.output else ROOT / "outputs" / "gpu_tier_calibration" / _utc_stamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    args.overrides = _parse_overrides(args.override)
    args.final_test_dataset = str(Path(args.final_test_dataset).resolve()) if args.final_test_dataset else ""
    if args.overrides:
        print(f">> overrides: {args.overrides}", flush=True)
    if not args.lora:
        args.lora = _newest_lora(ROOT / "loras")
    print(f">> calibration output: {out_dir}", flush=True)

    tiers = TierValues(Path(args.model_dir))
    poller = GpuPoller(args.gpu, args.interval, timeline_path=out_dir / "timeline.csv")
    poller.start()
    idle = poller.read()
    print(f">> GPU {args.gpu}: {poller.total_mib / _MIB:.2f} GiB total, {idle / _MIB:.2f} GiB used before calibration", flush=True)
    records: list[dict[str, Any]] = []
    try:
        if "inference" in stages:
            for tier in tiers_requested:
                selected = list(variants)
                if tier in emotion_tiers and "emotion_text" not in selected:
                    selected.append("emotion_text")
                for variant in selected:
                    if variant == "lora" and not args.lora:
                        print(">> no LoRA / DoRA file found; skipping the lora variant", flush=True)
                        continue
                    record = inference_stage(tiers, tier, variant, args, poller, out_dir)
                    records.append(record)
                    print(f">> {tier} GB inference/{variant}: peak {record['peak_gb']:.2f} GB (budget {tier_budget_gb(tier):.0f}), "
                          f"rtf {record.get('rtf') or 0:.2f}, code {record['returncode']}", flush=True)
                    _write_report(out_dir, records, poller, args)
        dataset_dir = Path(args.train_dataset).resolve() if args.train_dataset else None
        pipeline_tier = min(tiers_requested) if args.emulate else max(tiers_requested)
        if "prep" in stages:
            if not args.source:
                raise SystemExit("--source is required for the prep stage")
            record = prep_stage(tiers, pipeline_tier, args, poller, out_dir, "calibration_voice")
            records.append(record)
            print(f">> prep: peak {record['peak_gb']:.2f} GB, {record.get('segments')} segments, code {record['returncode']}", flush=True)
            _write_report(out_dir, records, poller, args)
            if record["returncode"] == 0:
                dataset_dir = Path(record["dataset_dir"])
        if "audit" in stages and dataset_dir is not None and not args.train_dataset:
            record = audit_stage(tiers, pipeline_tier, args, poller, out_dir, dataset_dir)
            records.append(record)
            print(f">> audit: peak {record['peak_gb']:.2f} GB, {record.get('segments')} segments, code {record['returncode']}", flush=True)
            _write_report(out_dir, records, poller, args)
            if record["returncode"] == 0:
                dataset_dir = Path(record["dataset_dir"])
                args.final_test_dataset = record.get("test_dataset_dir") or ""
        if "cache" in stages and dataset_dir is not None:
            record = cache_stage(pipeline_tier, args, poller, out_dir, dataset_dir)
            records.append(record)
            print(f">> cache: peak {record['peak_gb']:.2f} GB, code {record['returncode']}", flush=True)
            _write_report(out_dir, records, poller, args)
        if "train" in stages:
            if dataset_dir is None or not (dataset_dir / "cache" / "index.jsonl").is_file():
                raise SystemExit("the train stage needs a cached dataset: run prep/cache or pass --train-dataset")
            for tier in tiers_requested:
                record = train_stage(tiers, tier, args, poller, out_dir, dataset_dir)
                records.append(record)
                print(f">> {tier} GB train: peak {record['peak_gb']:.2f} GB (budget {tier_budget_gb(tier):.0f}), "
                      f"phases {record['phase_peaks_gb']}, code {record['returncode']}", flush=True)
                _write_report(out_dir, records, poller, args)
    finally:
        poller.stop()
        _write_report(out_dir, records, poller, args)
        print((out_dir / "calibration.md").read_text(encoding="utf-8"), flush=True)
        print(f">> wrote {out_dir / 'calibration.json'}", flush=True)
    return 0 if all(item.get("returncode") == 0 for item in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
