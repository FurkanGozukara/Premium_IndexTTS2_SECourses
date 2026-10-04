"""Reference audition after training: the clip the finished voice clones best from (``reference_audition``).

The trainer runs ``python -m indextts.training.audition_worker --config <train_config.json> --checkpoint <file>
--state-dir <dir>`` after the voice's other automatic checks, with the training model released. Takes are rendered
by the listening grid (``run_grid``: the speech model loads once per stage) with the settings of the speech
comparison: IndexTTS with the voice's deployment settings, OmniVoice and AuK in Voice cloning mode. Each candidate
clip is staged with its transcript beside it (``<clip>.txt``), which OmniVoice and AuK read. Every candidate is
screened on a few held-out sentences; the current reference and the finalists render more sentences with the same
seeds, and the winner becomes the voice's automatic reference (``save_choice``).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .dataset_manifest import atomic_write_json

WORK_FOLDER = "reference_audition"
GRID_LANGUAGES = {"ZH", "EN", "JA", "AR", "ES"}


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _transcript(path: Path) -> str:
    try:
        return path.with_suffix(".txt").read_text(encoding="utf-8-sig").strip()
    except OSError:
        return ""


def training_reference(run_dir: Path, name: str) -> Path | None:
    """The reference training saved beside the voice (``<name>_reference.<ext>``)."""
    for path in [run_dir / f"{name}_reference{suffix}" for suffix in (".wav", ".flac", ".mp3")] + sorted(
            run_dir.glob("*_reference.*")):
        if path.is_file() and path.suffix.lower() in {".wav", ".flac", ".mp3"} and "_expressive_reference" not in path.name:
            return path
    return None


def plan_candidates(checkpoint: Path, name: str, dataset_dir: Path, training: Sequence[Mapping[str, Any]],
                    rows: Sequence[Mapping[str, Any]], count: int) -> list[dict[str, Any]]:
    """The voice's current reference and ``count`` other clean training clips near the speaker's typical voice."""
    from .evaluation_plan import audio_path
    from .reference_audition import audition_reference, choose_candidates, run_dir_of

    run_dir = run_dir_of(checkpoint)
    current = audition_reference(checkpoint)
    current_path = Path(current) if current else training_reference(run_dir, name)
    by_sha = {}
    candidates: list[dict[str, Any]] = []
    if current_path is not None:
        digest = _sha(current_path)
        text = _transcript(current_path)
        if not text:  # the training reference is a copy of a dataset clip
            for row in rows:
                source = audio_path(dataset_dir, row)
                if source.is_file() and source.stat().st_size == current_path.stat().st_size and _sha(source) == digest:
                    text = str(row.get("text") or "")
                    break
        by_sha[digest] = True
        import soundfile as sf

        candidates.append({"key": "current", "label": current_path.name, "audio": str(current_path), "text": text,
                           "duration_s": float(sf.info(str(current_path)).duration), "current": True})
    for row in choose_candidates(dataset_dir, training, max(1, int(count)) + 2):
        source = audio_path(dataset_dir, row)
        digest = _sha(source)
        if digest in by_sha:
            continue
        by_sha[digest] = True
        candidates.append({"key": str(row["id"]), "label": source.name, "audio": str(source), "text": str(row["text"]),
                           "duration_s": float(row.get("duration_s") or 0.0), "current": False})
        if len(candidates) >= int(count) + (1 if current_path is not None else 0):
            break
    return candidates


def stage_references(candidates: Sequence[Mapping[str, Any]], folder: Path) -> list[dict[str, Any]]:
    """Copies of the candidate clips with their transcripts beside them, which OmniVoice and AuK read."""
    folder.mkdir(parents=True, exist_ok=True)
    staged = []
    for candidate in candidates:
        source = Path(str(candidate["audio"]))
        target = folder / f"{candidate['key']}{source.suffix.lower() or '.wav'}"
        shutil.copy2(source, target)
        if str(candidate.get("text") or "").strip():
            target.with_suffix(".txt").write_text(str(candidate["text"]).strip(), encoding="utf-8")
        staged.append(dict(candidate, staged=str(target)))
    return staged


def render_settings(config: Any, checkpoint: Path, language: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Grid runtime and inference settings: those of the speech comparison, with every model cloning."""
    from .deployment_settings import deployment_infer_kwargs
    from .speech_eval import _benchmark_infer_kwargs, _benchmark_runtime

    runtime = _benchmark_runtime(config)
    model = getattr(config, "tts_model", "indextts")
    if model == "indextts" and bool(getattr(config, "speech_eval_deployment_settings", True)):
        infer = deployment_infer_kwargs(config, str(checkpoint), language=language, tier=runtime.vram_tier)
    else:
        infer = _benchmark_infer_kwargs(config)
    for key in ("omnivoice", "auk"):
        if isinstance(infer.get(key), Mapping):
            # Every staged clip carries its own transcript; a voice trained for Auto voice still clones here.
            settings = {name: value for name, value in infer[key].items() if name != "reference_text"}
            settings["mode"] = "clone"
            if key == "auk":
                # An AuK voice's samples may speak in Auto voice (guidance 1.5); cloning uses the clone preset's
                # guidance, as Voice Generation does.
                from indextts.auk.text import preset_guidance

                settings["guidance_scale"] = preset_guidance("clone")
            infer[key] = settings
    grid_runtime = {"tts_model": model, "runtime": runtime.to_dict(), "model_dir": config.model_dir,
                    "cfg_path": config.model_config, "use_qwen_emo": False}
    return grid_runtime, infer


def run_audition(config_path: str | Path, checkpoint: str | Path, state_dir: str | Path) -> dict[str, Any] | None:
    """Audition the voice's reference clips and save the winner as its automatic reference."""
    import torch

    from indextts.runtime import ProgressReporter

    from .dataset_manifest import load_manifest
    from .dataset_quality import transcript_vocabulary
    from .grid import GridCheckpoint, GridConfig, run_grid
    from .reference_audition import (FINALISTS, SCREEN_SENTENCES, audition_search, choose_sentences, run_dir_of,
                                     save_choice, sentence_language, split_rows)
    from .speech_metrics import lenient_units, measure_clips
    from .train_config import TrainConfig

    started = time.perf_counter()
    state = Path(state_dir)
    state.mkdir(parents=True, exist_ok=True)
    config = TrainConfig.from_dict(json.loads(Path(config_path).read_text(encoding="utf-8")))
    checkpoint = Path(checkpoint).resolve()
    run_dir = run_dir_of(checkpoint)
    dataset_dir = Path(config.dataset_dir)

    def status(message: str, **extra: Any) -> None:
        print(f">> reference audition: {message}", flush=True)
        atomic_write_json(state / "status.json", {"phase": extra.pop("phase", "running"), "message": message,
                                                  "elapsed_s": round(time.perf_counter() - started, 1), **extra})

    def cancelled() -> bool:
        return (state / "stop.flag").exists()

    def progress(message: Any, completed: Any = 0, total: Any = 0) -> None:
        atomic_write_json(state / "status.json", {"phase": "running", "message": str(message), "completed": completed,
                                                  "total": total, "elapsed_s": round(time.perf_counter() - started, 1)})

    rows = load_manifest(dataset_dir)
    training, validation = split_rows(dataset_dir, rows, run_dir)
    count = int(getattr(config, "reference_audition_candidates", 12))
    total_sentences = int(getattr(config, "reference_audition_sentences", 8))
    candidates = plan_candidates(checkpoint, config.name, dataset_dir, training, rows, count)
    screen_count = min(SCREEN_SENTENCES, total_sentences)
    sentences = choose_sentences(dataset_dir, validation, training, total_sentences,
                                 exclude=[candidate["key"] for candidate in candidates])
    if len(candidates) < 2 or not sentences:
        status("not enough clean clips or held-out sentences in the dataset; the voice keeps its reference",
               phase="skipped")
        return None
    screen, confirm = sentences[:screen_count], sentences[screen_count:]
    languages = Counter(sentence_language(row, "EN") for row in sentences)
    language = languages.most_common(1)[0][0]
    grid_language = language if language in GRID_LANGUAGES else "EN"
    work = run_dir / "analysis" / WORK_FOLDER
    shutil.rmtree(work, ignore_errors=True)
    staged = stage_references(candidates, work / "candidates")
    runtime, infer = render_settings(config, checkpoint, grid_language)
    status(f"{len(staged)} clips (the current reference first) on {len(screen)} screening sentences; the current "
           f"reference and {FINALISTS} finalists on {len(confirm)} more")

    def render_stage(stage: str, stage_candidates: Sequence[Mapping[str, Any]], stage_sentences: Sequence[Mapping[str, Any]],
                     first_seed: int):
        yield f"rendering {len(stage_candidates)} clips x {len(stage_sentences)} sentences ({stage})"
        grid = GridConfig(adapter_dir=str(run_dir), checkpoints=[GridCheckpoint("voice", str(checkpoint))],
                          references=[str(row["staged"]) for row in stage_candidates],
                          texts=[str(row["text"]) for row in stage_sentences], language=grid_language,
                          seeds=[first_seed], seed=first_seed, output_root=str(work), grid_name=stage,
                          runtime=runtime, infer_kwargs=infer, include_verdicts=False)
        result = run_grid(grid, reporter=ProgressReporter("audition takes", progress_file=state / "progress.json"),
                          cancel_callback=cancelled)
        if result.status != "complete":
            raise InterruptedError(f"audition rendering {result.status}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return [{"candidate": stage_candidates[cell.reference_index - 1]["key"],
                 "sentence": stage_sentences[cell.text_index - 1], "audio": cell.audio_path} for cell in result.cells]

    vocabulary = transcript_vocabulary(str(row.get("text") or "") for row in rows)
    lenient = set(lenient_units(vocabulary, language)) or None

    def measure(clips: list[dict[str, Any]]) -> list[dict[str, Any]]:
        stage = clips[0]["stage"] if clips else "stage"
        return measure_clips(clips, model_dir=config.model_dir, model_config=config.model_config, device=config.device,
                             output_dir=work / f"{stage}_measurements", update=progress, cancelled=cancelled,
                             lenient_terms=lenient)

    search = audition_search(staged, screen, confirm, render_stage=render_stage, measure=measure,
                             dataset_dir=dataset_dir, language=language, finalists=FINALISTS)
    try:
        while True:
            status(next(search))
    except StopIteration as stop:
        result = stop.value
    winner = result["winner"]
    if winner is None:
        status("no take could be measured; the voice keeps its reference", phase="failed")
        return None
    table = result["summary"]  # the clips' own paths: the staged copies are working files
    report = {"audited_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "voice": str(checkpoint), "automatic": True,
              "speech_model": getattr(config, "tts_model", "indextts"),
              "sentences": [{"id": str(row["id"]), "text": str(row["text"]), "stage": "screen"} for row in screen]
              + [{"id": str(row["id"]), "text": str(row["text"]), "stage": "confirm"} for row in confirm],
              "finalists": result["finalists"], "candidates": table,
              "elapsed_s": round(time.perf_counter() - started, 1)}
    # In use at once: the reason it runs after training is that the voice loads its best clip without a manual step.
    record = save_choice(checkpoint, winner, report, use=True)
    current = next((row for row in table if row.get("current")), None)
    if winner.get("current"):
        message = f"the current reference stays the best clip ({winner['label']}, similarity {winner['similarity']:.3f})"
    else:
        against = f" against {current['similarity']:.3f} for the current reference" if current and current.get("similarity") is not None else ""
        message = (f"winner {winner['label']} (similarity {winner['similarity']:.3f}{against}); "
                   "now the voice's automatic reference")
    status(message, phase="complete", winner=winner["label"], reference=record.get("reference", ""))
    return record


def main(argv: Sequence[str] | None = None) -> int:
    # transformers imports torchao, whose enum registrations print PyTorch's register_constant()
    # deprecation warning in every worker log unless this shim is installed first (the app does it at start).
    from indextts.utils.torch_compat import install_native_enum_pytree_compatibility

    install_native_enum_pytree_compatibility()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--state-dir", required=True)
    args = parser.parse_args(argv)
    try:
        run_audition(args.config, args.checkpoint, args.state_dir)
    except BaseException as exc:
        traceback.print_exc()
        atomic_write_json(Path(args.state_dir) / "status.json",
                          {"phase": "cancelled" if isinstance(exc, InterruptedError) else "failed", "message": str(exc)})
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
