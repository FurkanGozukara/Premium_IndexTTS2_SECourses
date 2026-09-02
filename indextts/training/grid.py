"""Listening-grid generation across checkpoints, strengths, references, and texts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import secrets
import time
from typing import Any, Callable, Mapping
import wave

from indextts.runtime.progress import ProgressReporter
from indextts.utils.atomic_json import write_json_atomic
from webui_generation_runner import create_tts, run_generation_request

from .analysis import (
    BASE_CHECKPOINT_LABEL,
    _write_text_atomic,
    checkpoint_descriptor,
    checkpoint_display_label,
    load_training_analysis,
    phase_display_label,
)
from .checkpoint_eval import load_checkpoint_eval


_SAFE_CELL_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_EPOCH_RE = re.compile(r"_epoch_(\d+)", re.IGNORECASE)
_STEP_RE = re.compile(r"_step_(\d+)", re.IGNORECASE)
_INFER_KEYS = {
    "do_sample", "top_p", "top_k", "temperature", "length_penalty", "num_beams",
    "repetition_penalty", "max_mel_tokens", "emo_audio_prompt", "emo_alpha",
    "emo_vector", "use_emo_text", "emo_text", "use_random", "verbose",
    "max_text_tokens_per_segment", "interval_silence", "diffusion_steps",
    "inference_cfg_rate", "max_speaker_audio_length", "max_emotion_audio_length",
    "section_batch_size", "max_emotion_sum", "latent_multiplier",
    "max_consecutive_silence", "semantic_layer", "cfm_cache_length",
    "reset_beam_cache_per_segment", "text_normalization",
}
_RUNNER_EXTRA_KEYS = {
    "segment_budget_scale_non_cjk", "cfm_temperature", "seed",
    "reuse_spk_cond_for_emo", "enable_pause_tags", "trim_silence_ms_threshold",
    "target_duration_s", "target_duration_mode",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str, fallback: str = "grid") -> str:
    result = _SAFE_CELL_RE.sub("_", str(value or "").strip()).strip("._")
    return (result or fallback)[:120]


@dataclass
class GridCheckpoint:
    label: str
    path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | "GridCheckpoint") -> "GridCheckpoint":
        if isinstance(value, cls):
            return cls(value.label, value.path)
        return cls(label=str(value.get("label", "")), path=str(value.get("path", "")))


@dataclass
class GridConfig:
    adapter_dir: str
    checkpoints: list[GridCheckpoint]
    strengths: list[float] = field(default_factory=lambda: [1.0])
    references: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    language: str = "EN"
    seed: int = -1
    same_seed_for_all_cells: bool = True
    output_root: str = "outputs/grids"
    grid_name: str = ""
    runtime: dict[str, Any] = field(default_factory=dict)
    infer_kwargs: dict[str, Any] = field(default_factory=dict)
    include_verdicts: bool = True

    def validate(self) -> "GridConfig":
        self.adapter_dir = str(Path(self.adapter_dir).expanduser().resolve())
        self.checkpoints = [GridCheckpoint.from_dict(item) for item in self.checkpoints]
        if not self.checkpoints:
            raise ValueError("select at least one checkpoint or Base model (no LoRA / DoRA)")
        values: list[float] = []
        for raw in self.strengths or [1.0]:
            strength = float(raw)
            if not math.isfinite(strength) or not 0.0 <= strength <= 4.0:
                raise ValueError("grid strengths must be finite values from 0 to 4")
            if strength not in values:
                values.append(strength)
        self.strengths = values or [1.0]
        self.references = [str(Path(item).expanduser().resolve()) for item in self.references if str(item).strip()]
        self.texts = [str(item).strip() for item in self.texts if str(item).strip()]
        if not self.references:
            raise ValueError("add at least one reference audio path")
        if not self.texts:
            raise ValueError("add at least one listening-grid text")
        self.language = str(self.language or "EN").upper()
        if self.language not in {"ZH", "EN", "JA", "AR", "ES"}:
            raise ValueError("language must be ZH, EN, JA, AR, or ES")
        self.seed = int(self.seed)
        if self.seed < -1:
            raise ValueError("seed must be -1 or a non-negative integer")
        self.output_root = str(Path(self.output_root).expanduser().resolve())
        self.grid_name = _safe_name(self.grid_name, "") if self.grid_name else ""
        self.runtime = dict(self.runtime or {})
        self.infer_kwargs = dict(self.infer_kwargs or {})
        return self

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["checkpoints"] = [item.to_dict() for item in self.checkpoints]
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | "GridConfig") -> "GridConfig":
        if isinstance(value, cls):
            return cls(**value.to_dict()).validate()
        if not isinstance(value, Mapping):
            raise TypeError("grid config must be a mapping")
        allowed = {item.name for item in fields(cls)}
        payload = {key: item for key, item in value.items() if key in allowed}
        payload["checkpoints"] = [
            GridCheckpoint.from_dict(item) for item in payload.get("checkpoints", [])
        ]
        return cls(**payload).validate()

    @classmethod
    def from_json(cls, path: str | Path) -> "GridConfig":
        with Path(path).open("r", encoding="utf-8-sig") as handle:
            return cls.from_dict(json.load(handle))


@dataclass
class GridCell:
    index: int
    label: str
    filename: str
    checkpoint_label: str
    checkpoint_path: str
    checkpoint_kind: str
    strength: float
    reference_index: int
    text_index: int
    reference: str
    text: str
    seed: int
    audio_path: str = ""
    task_dir: str = ""
    audio_seconds: float = 0.0
    generation_stats: dict[str, Any] = field(default_factory=dict)
    verdict: str = ""
    val_loss: float | None = None
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GridCell":
        allowed = {item.name for item in fields(cls)}
        payload = {key: item for key, item in value.items() if key in allowed}
        return cls(**payload)


@dataclass
class GridResult:
    grid_dir: str
    grid_name: str
    config: dict[str, Any]
    seed: int
    cells: list[GridCell]
    status: str
    summary_markdown: str
    generated_at: str
    elapsed_s: float

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["cells"] = [item.to_dict() for item in self.cells]
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GridResult":
        payload = dict(value)
        payload["cells"] = [
            item if isinstance(item, GridCell) else GridCell.from_dict(item)
            for item in payload.get("cells", [])
            if isinstance(item, (GridCell, Mapping))
        ]
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: item for key, item in payload.items() if key in allowed})


@dataclass
class GridSummary:
    grid_dir: str
    grid_name: str
    status: str
    cells: int
    seed: int
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _file_label(checkpoint: GridCheckpoint) -> tuple[str, str]:
    if not checkpoint.path:
        return "base", "base"
    source = Path(checkpoint.path)
    try:
        descriptor = checkpoint_descriptor(source)
        return str(descriptor["file_label"]), str(descriptor["kind"])
    except Exception:
        stem = source.stem
        epoch = _EPOCH_RE.search(stem)
        step = _STEP_RE.search(stem)
        if source.parent.name.lower() == "best":
            return (f"best_ep{int(epoch.group(1))}" if epoch else "best"), "best"
        if epoch:
            return f"epoch_{int(epoch.group(1)):03d}", "epoch"
        if step:
            return f"step_{int(step.group(1)):06d}", "step"
        if stem.lower().endswith("_interrupted"):
            return "interrupted", "interrupted"
        return _safe_name(checkpoint.label or stem, "final"), "final"


def build_grid_cells(config: GridConfig | Mapping[str, Any]) -> list[GridCell]:
    cfg = GridConfig.from_dict(config)
    cells: list[GridCell] = []
    index = 0
    for checkpoint in cfg.checkpoints:
        file_label, kind = _file_label(checkpoint)
        strengths = [1.0] if not checkpoint.path else cfg.strengths
        for strength in strengths:
            for reference_index, reference in enumerate(cfg.references, start=1):
                for text_index, text in enumerate(cfg.texts, start=1):
                    index += 1
                    filename = (
                        f"{_safe_name(file_label)}__s{strength:g}__ref{reference_index}"
                        f"__t{text_index}.wav"
                    )
                    checkpoint_label = (
                        checkpoint.label or file_label
                        if checkpoint.path
                        else BASE_CHECKPOINT_LABEL
                    )
                    row_label = checkpoint_label
                    if checkpoint.path and abs(strength - 1.0) >= 1e-9:
                        row_label += f" @{strength:g}"
                    cells.append(
                        GridCell(
                            index=index,
                            label=f"{row_label} / ref {reference_index} / text {text_index}",
                            filename=filename,
                            checkpoint_label=checkpoint_label,
                            checkpoint_path=str(Path(checkpoint.path).expanduser().resolve())
                            if checkpoint.path
                            else "",
                            checkpoint_kind=kind,
                            strength=float(strength),
                            reference_index=reference_index,
                            text_index=text_index,
                            reference=reference,
                            text=text,
                            seed=cfg.seed,
                        )
                    )
    return cells


def _metadata(path: Path) -> dict[str, Any]:
    now = _utc_now()
    return {
        "status": "in_progress",
        "created_at": now,
        "updated_at": now,
        "task": {"id": path.parent.name, "folder": str(path.parent)},
        "inputs": {},
        "settings": {},
        "outputs": {},
        "processing": {
            "started_at": now,
            "ended_at": None,
            "elapsed_ms": None,
            "elapsed_seconds": None,
            "elapsed_human": None,
        },
        "subtitle": None,
        "error": None,
    }


def _default_infer_kwargs(runtime: Mapping[str, Any]) -> dict[str, Any]:
    runtime_payload = runtime.get("runtime") if isinstance(runtime.get("runtime"), Mapping) else runtime
    return {
        "do_sample": True,
        "top_p": 0.8,
        "top_k": 30,
        "temperature": 0.8,
        "length_penalty": 0.0,
        "num_beams": 1,
        "repetition_penalty": 10.0,
        "max_mel_tokens": 1500,
        "emo_audio_prompt": None,
        "emo_alpha": 0.65,
        "emo_vector": None,
        "use_emo_text": False,
        "emo_text": None,
        "use_random": False,
        "verbose": False,
        "max_text_tokens_per_segment": 60,
        "interval_silence": 200,
        "diffusion_steps": 25,
        "inference_cfg_rate": 0.7,
        "max_speaker_audio_length": 15.0,
        "max_emotion_audio_length": 15.0,
        "section_batch_size": 1,
        "max_emotion_sum": 0.8,
        "latent_multiplier": 1.72,
        "max_consecutive_silence": 0,
        "semantic_layer": 17,
        "cfm_cache_length": int(runtime_payload.get("cfm_cache_length", 8192)),
        "reset_beam_cache_per_segment": True,
        "text_normalization": True,
    }


def _request_for_cell(
    config: GridConfig,
    cell: GridCell,
    grid_dir: Path,
) -> dict[str, Any]:
    task_dir = grid_dir / ".cells" / Path(cell.filename).stem
    task_dir.mkdir(parents=True, exist_ok=True)
    audio_path = grid_dir / cell.filename
    metadata_path = task_dir / "metadata.json"
    layout = {
        "task_id": Path(cell.filename).stem,
        "task_folder": str(task_dir),
        "final_basename": audio_path.stem,
        "final_wav_path": str(audio_path),
        "final_mp3_path": str(audio_path.with_suffix(".mp3")),
        "final_mp4_path": str(audio_path.with_suffix(".mp4")),
        "segments_dir": str(task_dir / "segments"),
        "speaker_reference_copy_path": str(task_dir / "reference.wav"),
    }
    infer_kwargs = _default_infer_kwargs(config.runtime)
    infer_kwargs.update(config.infer_kwargs)
    runner_extras = {
        key: infer_kwargs.pop(key)
        for key in list(infer_kwargs)
        if key in _RUNNER_EXTRA_KEYS
    }
    unknown = set(infer_kwargs) - _INFER_KEYS
    if unknown:
        raise ValueError(f"unsupported grid inference setting(s): {', '.join(sorted(unknown))}")
    infer_kwargs["max_text_tokens_per_segment"] = int(
        infer_kwargs.get("max_text_tokens_per_segment", 60)
    )
    runtime = dict(config.runtime)
    nested_runtime = runtime.get("runtime")
    if isinstance(nested_runtime, Mapping):
        runtime["runtime"] = dict(nested_runtime)
        for key in ("lora_path", "lora_strength", "lora_merge_into_base"):
            runtime["runtime"].pop(key, None)
    for key in ("lora_path", "lora_strength", "lora_merge_into_base"):
        runtime.pop(key, None)
    request = {
        "prompt": cell.reference,
        "text": cell.text,
        "subtitle_mode": False,
        "subtitle_file": None,
        "language": config.language,
        "save_used_audio": False,
        "save_as_mp3": False,
        "mp3_bitrate": "256k",
        "image_path": None,
        "infer_kwargs": infer_kwargs,
        "runtime": runtime,
        "low_memory_mode": int(
            (runtime.get("runtime") or runtime).get("blocks_to_swap", 0)
            if isinstance(runtime.get("runtime") or runtime, Mapping)
            else 0
        ) > 0,
        "task_layout": layout,
        "metadata_path": str(metadata_path),
        "max_text_tokens": int(infer_kwargs["max_text_tokens_per_segment"]),
        "progress_file": str(task_dir / "progress.json"),
        "lora_path": cell.checkpoint_path,
        "lora_strength": cell.strength,
        "lora_merge_into_base": False,
        "num_candidates": 1,
        "audio_tuning_preset": "bypass",
        "audio_tuning_overrides": {},
        "segment_budget_scale_non_cjk": float(runner_extras.get("segment_budget_scale_non_cjk", 0.72)),
        "cfm_temperature": float(runner_extras.get("cfm_temperature", 1.0)),
        "seed": cell.seed,
        "reuse_spk_cond_for_emo": bool(runner_extras.get("reuse_spk_cond_for_emo", False)),
        "enable_pause_tags": bool(runner_extras.get("enable_pause_tags", True)),
        "trim_silence_ms_threshold": int(runner_extras.get("trim_silence_ms_threshold", 0)),
        "target_duration_s": runner_extras.get("target_duration_s"),
        "target_duration_mode": str(runner_extras.get("target_duration_mode", "off")),
    }
    metadata = _metadata(metadata_path)
    metadata["inputs"] = {
        "text": cell.text,
        "language": config.language,
        "speaker_reference_audio": cell.reference,
    }
    metadata["settings"] = {
        "runtime": runtime,
        "resolved_generation_kwargs": infer_kwargs,
        "checkpoint": cell.checkpoint_path,
        "strength": cell.strength,
        "seed": cell.seed,
    }
    write_json_atomic(metadata_path, metadata, indent=2, ensure_ascii=False)
    write_json_atomic(task_dir / "request.json", request, indent=2, ensure_ascii=False)
    cell.audio_path = str(audio_path)
    cell.task_dir = str(task_dir)
    return request


def _wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            return handle.getnframes() / float(rate) if rate else 0.0
    except (OSError, EOFError, wave.Error):
        return 0.0


def _verdict_map(adapter_dir: Path) -> dict[tuple[str, float | None], tuple[str, float | None]]:
    result: dict[tuple[str, float | None], tuple[str, float | None]] = {
        ("", None): ("base", None)
    }
    measured = load_checkpoint_eval(adapter_dir)
    if measured is not None:
        for row in measured.rows:
            key = (str(Path(row.path).resolve()) if row.path else "", float(row.strength))
            result[key] = (row.phase, row.val_loss)
        return result
    analysis = load_training_analysis(adapter_dir)
    if analysis is not None:
        for item in analysis.checkpoints:
            path = str(Path(str(item.get("path") or "")).resolve()) if item.get("path") else ""
            result[(path, None)] = (str(item.get("phase") or "unknown"), item.get("val_loss"))
    return result


def _grid_markdown(result: GridResult) -> str:
    lines = [
        f"# Checkpoint Grid: {result.grid_name}",
        "",
        f"Fixed seed: `{result.seed}` | Cells: {len(result.cells)} | Status: **{result.status}**",
        "",
        "| Row | Checkpoint | Strength | Reference | Text | Seconds | Verdict | File |",
        "|---:|---|---:|---:|---:|---:|---|---|",
    ]
    label_cache: dict[str, str] = {}
    for cell in result.cells:
        checkpoint_label = checkpoint_display_label(
            cell.checkpoint_label,
            path=cell.checkpoint_path,
            kind=cell.checkpoint_kind,
            cache=label_cache,
        )
        reference = f"{cell.reference_index}: {Path(cell.reference).name}".replace("|", "\\|")
        text = cell.text.replace("|", "\\|").replace("\r", " ").replace("\n", " ")
        strength = "-" if not cell.checkpoint_path else f"{cell.strength:g}"
        lines.append(
            f"| {cell.index} | {checkpoint_label} | {strength} | "
            f"{reference} | {cell.text_index}: {text} | {cell.audio_seconds:.2f} | "
            f"{phase_display_label(cell.verdict)} | [{cell.filename}]({cell.filename}) |"
        )
    return "\n".join(lines)


def _write_result(result: GridResult, grid_dir: Path) -> None:
    result.summary_markdown = _grid_markdown(result)
    write_json_atomic(grid_dir / "grid.json", result.to_dict(), indent=2, ensure_ascii=False, allow_nan=False)
    _write_text_atomic(grid_dir / "grid.md", result.summary_markdown)


def run_grid(
    config: GridConfig | Mapping[str, Any],
    reporter: ProgressReporter | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> GridResult:
    started = time.perf_counter()
    cfg = GridConfig.from_dict(config)
    for reference in cfg.references:
        if not Path(reference).is_file():
            raise FileNotFoundError(f"grid reference not found: {reference}")
    for checkpoint in cfg.checkpoints:
        if checkpoint.path and not Path(checkpoint.path).is_file():
            raise FileNotFoundError(f"grid checkpoint not found: {checkpoint.path}")
    if not cfg.grid_name:
        adapter_name = Path(cfg.adapter_dir).name or "adapter"
        cfg.grid_name = _safe_name(
            f"{adapter_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    grid_dir = Path(cfg.output_root) / cfg.grid_name
    grid_dir.mkdir(parents=True, exist_ok=True)
    if (grid_dir / "grid.json").is_file():
        raise FileExistsError(f"grid already exists: {grid_dir}")
    (grid_dir / ".cells").mkdir(exist_ok=True)
    log_path = grid_dir / "log.txt"
    write_json_atomic(
        grid_dir / "status.json",
        {
            "phase": "initializing",
            "message": "Loading the voice model once for the listening grid",
            "completed": 0,
            "total": 0,
            "elapsed_s": 0.0,
            "updated_at": time.time(),
        },
    )
    cells = build_grid_cells(cfg)
    resolved_seed = secrets.randbelow(2**32) if cfg.seed == -1 else cfg.seed % (2**32)
    for index, cell in enumerate(cells):
        cell.seed = resolved_seed if cfg.same_seed_for_all_cells else (resolved_seed + index) % (2**32)
    progress = reporter or ProgressReporter(
        "cells", total=len(cells), progress_file=grid_dir / "progress.json"
    )
    progress.total = len(cells)
    progress.set_stage("load model")
    verdicts = _verdict_map(Path(cfg.adapter_dir)) if cfg.include_verdicts else {}
    result = GridResult(
        grid_dir=str(grid_dir.resolve()),
        grid_name=cfg.grid_name,
        config=cfg.to_dict(),
        seed=resolved_seed,
        cells=cells,
        status="running",
        summary_markdown="",
        generated_at=_utc_now(),
        elapsed_s=0.0,
    )

    def log(message: str) -> None:
        with log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(str(message) + "\n")
        progress.log(message)

    try:
        runtime = dict(cfg.runtime)
        nested = runtime.get("runtime")
        if isinstance(nested, Mapping):
            runtime["runtime"] = dict(nested)
            runtime["runtime"]["lora_path"] = ""
        runtime["lora_path"] = ""
        engine = create_tts(runtime)
        progress.set_stage("generating")
        for index, cell in enumerate(cells, start=1):
            if cancel_callback is not None and cancel_callback():
                result.status = "cancelled"
                break
            request = _request_for_cell(cfg, cell, grid_dir)
            cell_started = time.perf_counter()
            cell_log_path = Path(cell.task_dir) / "generation.log"
            _write_text_atomic(
                cell_log_path,
                f">> generating {cell.label} | seed {cell.seed} | strength {cell.strength:g}",
            )
            try:
                output = run_generation_request(request, engine)
            except BaseException as exc:
                with cell_log_path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(f">> failed: {exc}\n")
                raise
            audio_path = Path(str(output.get("output_path") or cell.audio_path))
            if not audio_path.is_file():
                raise RuntimeError(f"grid cell did not create audio: {cell.filename}")
            if audio_path.resolve() != Path(cell.audio_path).resolve():
                Path(cell.audio_path).write_bytes(audio_path.read_bytes())
            cell.audio_seconds = float(output.get("audio_seconds") or _wav_duration(Path(cell.audio_path)))
            cell.generation_stats = {
                key: output.get(key)
                for key in (
                    "seed",
                    "segments_count",
                    "audio_seconds",
                    "rtf",
                    "gpt_time",
                    "s2mel_time",
                    "vocoder_time",
                    "peak_vram_gb",
                )
                if output.get(key) is not None
            }
            exact_key = (cell.checkpoint_path, float(cell.strength))
            generic_key = (cell.checkpoint_path, None)
            verdict, val_loss = verdicts.get(
                exact_key, verdicts.get(generic_key, ("base" if not cell.checkpoint_path else "unknown", None))
            )
            cell.verdict = verdict
            cell.val_loss = float(val_loss) if val_loss is not None else None
            cell.status = "complete"
            elapsed = time.perf_counter() - cell_started
            with cell_log_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(
                    f">> complete | audio {cell.audio_seconds:.2f}s | elapsed {elapsed:.2f}s | "
                    f"output {cell.audio_path}\n"
                )
            progress.update(
                index,
                total=len(cells),
                desc=cell.label,
                extra={"audio_seconds": cell.audio_seconds, "cell_elapsed_s": elapsed},
            )
            log(
                f">> cell {index}/{len(cells)} {cell.filename} | audio {cell.audio_seconds:.2f}s | "
                f"elapsed {elapsed:.2f}s"
            )
            result.elapsed_s = time.perf_counter() - started
            _write_result(result, grid_dir)
            write_json_atomic(
                grid_dir / "status.json",
                {
                    "phase": "generating",
                    "message": cell.label,
                    "completed": index,
                    "total": len(cells),
                    "elapsed_s": result.elapsed_s,
                    "updated_at": time.time(),
                },
            )
        if result.status == "running":
            result.status = "complete"
            progress.finish()
        result.elapsed_s = time.perf_counter() - started
        _write_result(result, grid_dir)
        write_json_atomic(
            grid_dir / "status.json",
            {
                "phase": result.status,
                "message": (
                    "Listening grid complete"
                    if result.status == "complete"
                    else "Listening grid canceled between cells"
                ),
                "completed": sum(cell.status == "complete" for cell in cells),
                "total": len(cells),
                "elapsed_s": result.elapsed_s,
                "updated_at": time.time(),
            },
        )
        return result
    except BaseException as exc:
        result.status = "failed"
        result.elapsed_s = time.perf_counter() - started
        _write_result(result, grid_dir)
        write_json_atomic(
            grid_dir / "status.json",
            {
                "phase": "failed",
                "message": str(exc),
                "completed": sum(cell.status == "complete" for cell in cells),
                "total": len(cells),
                "elapsed_s": result.elapsed_s,
                "updated_at": time.time(),
            },
        )
        log(f">> grid failed: {exc}")
        raise


def load_grid(grid_dir: str | Path) -> GridResult | None:
    path = Path(grid_dir).expanduser().resolve() / "grid.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return GridResult.from_dict(value) if isinstance(value, dict) else None
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return None


def list_grids(output_root: str | Path = "outputs/grids") -> list[GridSummary]:
    root = Path(output_root).expanduser().resolve()
    values: list[tuple[float, GridSummary]] = []
    for path in root.glob("*/grid.json") if root.is_dir() else []:
        if path.parent.name.startswith((".", "_")):
            continue
        result = load_grid(path.parent)
        if result is None:
            continue
        try:
            modified = path.stat().st_mtime
        except OSError:
            modified = 0.0
        values.append(
            (
                modified,
                GridSummary(
                    grid_dir=str(path.parent.resolve()),
                    grid_name=result.grid_name,
                    status=result.status,
                    cells=len(result.cells),
                    seed=result.seed,
                    generated_at=result.generated_at,
                ),
            )
        )
    return [item for _, item in sorted(values, key=lambda item: item[0], reverse=True)]


__all__ = [
    "GridCell",
    "GridCheckpoint",
    "GridConfig",
    "GridResult",
    "GridSummary",
    "build_grid_cells",
    "list_grids",
    "load_grid",
    "run_grid",
]
