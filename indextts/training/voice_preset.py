"""Cloning preset after training: an OmniVoice voice's ready-to-use user preset, from built-in measurements only.

The trainer runs ``python -m indextts.training.voice_preset --config <train_config.json> --checkpoint <file>
--state-dir <dir>`` after the reference audition, with the training model released:

1. **Likeness target:** the voice's CAMPPlus centroid from up to 60 training clips (``voice_similarity``), cached
   beside the voice for "Takes per section".
2. **Pace check:** held-out sentences rendered once each in Voice cloning with the voice's automatic reference
   (the audition winner) at speaking rate 1, against the speaker's own recordings of the same sentences (speaking
   time without pauses, ``internal_pause_metrics``). The preset speaks at 1.10, the narration rate that scored best
   (on its calibrated pace a trained voice reads written narration slower than the speaker sounds, while it matches
   the speaker's spontaneous sentences within a few percent); a voice more than 10 % off the speaker is corrected by
   the measured ratio.
3. **Preset** ``<voice>_Clone_Best_of_10``: the training tier's system preset with OmniVoice's profile, Voice
   cloning with the automatic reference, that speaking rate, the voice's measured pauses, position temperature 0.5
   and "Takes per section" keeping the most similar of 10 renders without word errors (5 Whisper checks).

On a 139-line tutorial narration this configuration scored best of 14 versions of three speech models on likeness,
delivery style and word errors; speaking rates 1.05 and 1.10 ranked above 1.0 (October 2026).
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .dataset_manifest import atomic_write_json

APP_ROOT = Path(__file__).resolve().parents[2]
WORK_FOLDER = "voice_preset"
REPORT_FILE = "voice_preset.json"
# ui.presets_store's format; that module imports the interface, so the worker writes the same payload itself.
PRESET_FORMAT = "indextts2_premium_universal"
PRESET_VERSION = 3
PRESET_SUFFIX = "_Clone_Best_of_10"
TAKES, CHECKS = 10, 5
POSITION_TEMPERATURE = 0.5
NARRATION_RATE = 1.10
PACE_TOLERANCE = 0.10
RATE_RANGE = (0.8, 1.3)
GENERATED_BY = "cloning preset after training"


def speaking_time_s(path: str | Path) -> float:
    """Seconds of speech in a file: its speech span without the pauses inside it (``internal_pause_metrics``)."""
    import soundfile as sf

    from .speech_metrics import internal_pause_metrics

    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    metrics = internal_pause_metrics(data.mean(axis=1), rate)
    return max(0.0, float(metrics["speech_span_s"]) - float(metrics["pause_s"]))


def speech_span_s(samples: Any, rate: int, floor_db: float = -40.0) -> float:
    """Seconds from the first to the last 20 ms frame within ``floor_db`` of the loudest frame."""
    signal = np.asarray(samples, dtype=np.float32).reshape(-1)
    frame, hop = max(1, int(0.02 * rate)), max(1, int(0.01 * rate))
    if signal.size < frame:
        return 0.0
    count = 1 + (signal.size - frame) // hop
    rms = np.sqrt(np.array([np.mean(signal[i * hop:i * hop + frame] ** 2) for i in range(count)]) + 1e-12)
    loud = np.nonzero(rms >= rms.max() * 10 ** (floor_db / 20))[0]
    if not loud.size:
        return 0.0
    return float((loud[-1] - loud[0]) * hop + frame) / float(rate)


def file_span_s(path: str | Path) -> float:
    import soundfile as sf

    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return speech_span_s(data.mean(axis=1), rate)


def preset_rate(take_speech: Sequence[float], real_speech: Sequence[float]) -> tuple[float, float | None]:
    """``(speaking rate, ratio)``: 1.10, unless the takes' total speaking time at rate 1 is more than 10 % off the
    speaker's own (ratio = takes / speaker); then 1.10 times the ratio (a slower voice speeds up more)."""
    takes, real = float(sum(take_speech)), float(sum(real_speech))
    if takes <= 0 or real <= 0:
        return NARRATION_RATE, None
    ratio = takes / real
    if abs(ratio - 1.0) <= PACE_TOLERANCE:
        return NARRATION_RATE, round(ratio, 3)
    return round(min(RATE_RANGE[1], max(RATE_RANGE[0], NARRATION_RATE * ratio)), 2), round(ratio, 3)


def preset_name(voice: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(voice).strip()).strip("_.") + PRESET_SUFFIX


def system_preset(tier: int | None, presets_root: Path) -> Path | None:
    """The system preset of the training's memory tier, else the largest tier preset below it, else any."""
    folder = presets_root / "system"
    tiers = {}
    for path in folder.glob("* GB GPU.json"):
        try:
            tiers[int(path.name.split(" ", 1)[0])] = path
        except ValueError:
            continue
    if not tiers:
        return None
    if tier in tiers:
        return tiers[tier]
    below = [value for value in tiers if tier is None or value <= int(tier)]
    return tiers[max(below)] if below else tiers[min(tiers)]


def compose_preset(base: Mapping[str, Any], *, voice_path: str, speaking_rate: float,
                   pauses: Sequence[int] | None = None) -> dict[str, Any]:
    """Preset values: ``base`` (a universal preset's values) with OmniVoice active and the cloning settings."""
    # app.profiles: each model's own values plus "_active", the model whose values are at the top level.
    profiles = {name: dict(profile) if isinstance(profile, Mapping) else profile
                for name, profile in dict(base.get("app.profiles") or {}).items()}
    omnivoice = dict(profiles.get("omnivoice") or {})
    values = {**dict(base), **omnivoice}
    settings: dict[str, Any] = {
        "app.model": "omnivoice",
        "runtime.lora_path": str(voice_path),
        "omnivoice.mode": "clone",
        "omnivoice.reference_text": "",
        "omnivoice.instruct": "",
        "omnivoice.position_temperature": POSITION_TEMPERATURE,
        "generation.auto_lora_reference": True,
        "generation.auto_lora_speaking_rate": False,
        "generation.speaking_rate": float(speaking_rate),
        "generation.segmentation_mode": "smart",
        "generation.section_takes": TAKES,
        "generation.section_take_rule": "similar",
        "generation.section_take_checks": CHECKS,
    }
    if pauses:
        settings.update({"generation.auto_lora_pauses": True, "generation.sentence_pause_ms": int(pauses[0]),
                         "generation.max_pause_ms": int(pauses[1])})
    values.update(settings)
    # The profile restores these when someone switches models and back.
    profiles["omnivoice"] = {**omnivoice, **{key: value for key, value in settings.items() if key in omnivoice}}
    profiles["_active"] = "omnivoice"
    values["app.profiles"] = profiles
    return values


def write_preset(presets_root: Path, name: str, values: Mapping[str, Any], generated: Mapping[str, Any]) -> Path:
    """Save a user preset; a same-named preset that this step did not write is kept and the new one is numbered."""
    folder = presets_root / "user"
    folder.mkdir(parents=True, exist_ok=True)
    candidate, number = name, 1
    while True:
        path = folder / f"{candidate}.json"
        try:
            existing = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            existing = None
        if existing is None or ((existing.get("_meta") or {}).get("generated") or {}).get("by") == GENERATED_BY:
            break
        number += 1
        candidate = f"{name}_{number}"
    payload = {"_meta": {"format": PRESET_FORMAT, "version": PRESET_VERSION, "name": candidate, "scope": "user",
                         "read_only": False, "updated_at": datetime.now(timezone.utc).isoformat(),
                         "generated": {"by": GENERATED_BY, **dict(generated)}},
               "values": dict(values)}
    atomic_write_json(path, payload)
    return path


def run_voice_preset(config_path: str | Path, checkpoint: str | Path, state_dir: str | Path, *,
                     presets_root: str | Path | None = None) -> dict[str, Any] | None:
    import torch

    from indextts.runtime import ProgressReporter
    from indextts.utils.voice_similarity import SpeakerEmbedder, build_centroid

    from .audition_worker import GRID_LANGUAGES, render_settings, stage_references, training_reference
    from .dataset_manifest import load_manifest
    from .dataset_profile import recommended_pauses
    from .evaluation_plan import audio_path
    from .grid import GridCheckpoint, GridConfig, run_grid
    from .reference_audition import audition_reference, choose_sentences, run_dir_of, sentence_language, split_rows
    from .speech_eval import _benchmark_runtime
    from .train_config import TrainConfig

    started = time.perf_counter()
    state = Path(state_dir)
    state.mkdir(parents=True, exist_ok=True)
    presets = Path(presets_root) if presets_root else APP_ROOT / "presets"
    config = TrainConfig.from_dict(json.loads(Path(config_path).read_text(encoding="utf-8")))
    checkpoint = Path(checkpoint).resolve()
    run_dir = run_dir_of(checkpoint)
    dataset_dir = Path(config.dataset_dir)

    def status(message: str, **extra: Any) -> None:
        print(f">> cloning preset: {message}", flush=True)
        atomic_write_json(state / "status.json", {"phase": extra.pop("phase", "running"), "message": message,
                                                  "elapsed_s": round(time.perf_counter() - started, 1), **extra})

    def cancelled() -> bool:
        return (state / "stop.flag").exists()

    if getattr(config, "tts_model", "indextts") != "omnivoice":
        status("OmniVoice voices only", phase="skipped")
        return None
    device = str(config.device or "cuda:0").replace("auto", "cuda:0")
    status("measuring the voice's training clips (likeness target for Takes per section)")
    embedder = SpeakerEmbedder(config.model_dir, device)
    centroid = build_centroid(checkpoint, embedder, datasets_root=dataset_dir.parent)
    embedder.close()
    rows = load_manifest(dataset_dir)
    training, validation = split_rows(dataset_dir, rows, run_dir)
    reference = audition_reference(checkpoint) or training_reference(run_dir, config.name)
    sentences = choose_sentences(dataset_dir, validation, training, int(config.voice_preset_sentences))
    if reference is None or not sentences:
        status("no reference clip or no held-out sentences with recordings; no preset written", phase="skipped")
        return None
    language = Counter(sentence_language(row, "EN") for row in sentences).most_common(1)[0][0]
    grid_language = language if language in GRID_LANGUAGES else "EN"
    work = run_dir / "analysis" / WORK_FOLDER
    shutil.rmtree(work, ignore_errors=True)  # working renders of an earlier run
    staged = stage_references([{"key": "reference", "audio": str(reference),
                                "text": _reference_text(Path(reference), rows, dataset_dir)}], work / "reference")
    runtime, infer = render_settings(config, checkpoint, grid_language)
    infer["omnivoice"] = {**dict(infer.get("omnivoice") or {}), "position_temperature": POSITION_TEMPERATURE}
    status(f"rendering {len(sentences)} held-out sentences in Voice cloning with {Path(reference).name}")
    grid = GridConfig(adapter_dir=str(run_dir), checkpoints=[GridCheckpoint("voice", str(checkpoint))],
                      references=[str(staged[0]["staged"])], texts=[str(row["text"]) for row in sentences],
                      language=grid_language, seeds=[2026], seed=2026, output_root=str(work), grid_name="pace",
                      runtime=runtime, infer_kwargs=infer, include_verdicts=False)
    result = run_grid(grid, reporter=ProgressReporter("cloning preset", progress_file=state / "progress.json"),
                      cancel_callback=cancelled)
    if result.status != "complete":
        raise InterruptedError(f"pace rendering {result.status}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    measured = []
    for cell in result.cells:
        row = sentences[cell.text_index - 1]
        real_path = audio_path(dataset_dir, row)
        measured.append({"id": str(row["id"]), "text": str(row["text"]),
                         "real_span_s": round(file_span_s(real_path), 3), "take_span_s": round(file_span_s(cell.audio_path), 3),
                         "real_speech_s": round(speaking_time_s(real_path), 3),
                         "take_speech_s": round(speaking_time_s(cell.audio_path), 3)})
    rate, ratio = preset_rate([item["take_speech_s"] for item in measured], [item["real_speech_s"] for item in measured])
    profile_path = run_dir / "analysis" / "dataset_profile.json"
    try:
        pauses = recommended_pauses(json.loads(profile_path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pauses = None
    tier = getattr(_benchmark_runtime(config), "vram_tier", None)
    base_path = system_preset(tier, presets)
    if base_path is None:
        status("no system preset to start from; no preset written", phase="failed")
        return None
    base = json.loads(base_path.read_text(encoding="utf-8-sig"))["values"]
    values = compose_preset(base, voice_path=str(checkpoint), speaking_rate=rate, pauses=pauses)
    generated = {"voice": str(checkpoint), "reference": str(reference), "speaking_rate": rate,
                 "pace_ratio": ratio, "base_preset": base_path.stem, "sentences": len(measured),
                 "centroid_clips": len((centroid or {}).get("clips") or [])}
    path = write_preset(presets, preset_name(config.name), values, generated)
    report = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "preset": str(path), **generated,
              "pace": measured, "elapsed_s": round(time.perf_counter() - started, 1)}
    atomic_write_json(run_dir / "analysis" / REPORT_FILE, report)
    if ratio is None:
        check = "pace not measured"
    else:
        check = (f"the clone's speaking time at rate 1 is {abs(ratio - 1) * 100:.0f} % "
                 f"{'longer' if ratio > 1 else 'shorter'} than the speaker's on {len(measured)} held-out sentences, "
                 + ("within 10 %" if abs(ratio - 1) <= PACE_TOLERANCE else "corrected"))
    status(f"saved preset {path.stem}: Voice cloning at speaking rate {rate:.2f} ({check}); Takes per section keeps "
           f"the most similar of {TAKES} without word errors", phase="complete", preset=str(path), speaking_rate=rate)
    return report


def _reference_text(path: Path, rows: Sequence[Mapping[str, Any]], dataset_dir: Path) -> str:
    """The reference clip's transcript: beside it, else the dataset row with the same audio."""
    try:
        return path.with_suffix(".txt").read_text(encoding="utf-8-sig").strip()
    except OSError:
        pass
    from .evaluation_plan import audio_path

    size = path.stat().st_size
    for row in rows:
        source = audio_path(dataset_dir, row)
        if source.is_file() and source.stat().st_size == size and source.read_bytes() == path.read_bytes():
            return str(row.get("text") or "")
    return ""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--state-dir", required=True)
    args = parser.parse_args(argv)
    try:
        run_voice_preset(args.config, args.checkpoint, args.state_dir)
    except BaseException as exc:
        traceback.print_exc()
        atomic_write_json(Path(args.state_dir) / "status.json",
                          {"phase": "cancelled" if isinstance(exc, InterruptedError) else "failed", "message": str(exc)})
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
