"""Voice Generation's reference audition: render held-out sentences with each candidate clip, keep the best one.

Takes are rendered by the regular generation runner with the page's current settings (the voice, the speech
model's sampling, pauses and text options), so the audition hears the voice exactly as a generation would;
only extras that do not belong in a comparison are switched off (timestamps, takes, candidates, audio tuning,
MP3). Renders stay in the voice's ``analysis/reference_audition/`` folder for listening; the measurements come
from the training's speech evaluation (``measure_clips``). See ``indextts.training.reference_audition``.
"""

from __future__ import annotations

import gc
import hashlib
import shutil
import time
from pathlib import Path
from typing import Any, Iterator, Mapping

AUDITION_RENDER_FOLDER = "reference_audition"
TABLE_HEADERS = ["Rank", "Clip", "Seconds", "Transcript", "Speaker similarity", "Style similarity", "Word errors",
                 "Duration ratio", "Result"]


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _transcript_beside(path: str | Path) -> str:
    try:
        return Path(path).with_suffix(".txt").read_text(encoding="utf-8-sig").strip()
    except OSError:
        return ""


def table_rows(summary: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for rank, row in enumerate(summary, 1):
        result = "winner" if row.get("winner") else ""
        if row.get("current"):
            result = (result + ", current reference").strip(", ")
        rows.append([rank, row["label"], round(float(row.get("duration_s") or 0.0), 1), str(row.get("text") or "")[:90],
                     row.get("similarity"), row.get("style"),
                     None if row.get("word_errors") is None else f"{100 * float(row['word_errors']):.1f} %",
                     row.get("duration_ratio"), result])
    return rows


def run_reference_audition(values: Mapping[str, Any], adapter_path: str, *, candidate_count: int, sentence_count: int,
                           use_winner: bool, model_dir: str, root: str | Path) -> Iterator[tuple[str, list[list[Any]], str | None]]:
    """Yield ``(status markdown, result rows, winner reference or None)`` while the audition runs."""

    from indextts.training.dataset_manifest import load_manifest
    from indextts.training.dataset_profile import dataset_dir_for_adapter
    from indextts.training.evaluation_plan import audio_path
    from indextts.training.reference_audition import (audition_record, choose_candidates, choose_sentences, run_dir_of,
                                                      save_choice, split_rows, summarize)
    from webui_generation_runner import run_generation_request

    from .common import LAZY_ENGINE
    from .generation_tab import _recommended_lora_reference, prepare_generation_request

    if not adapter_path or not Path(adapter_path).is_file():
        yield "Select a trained voice (LoRA / DoRA or fine-tuned model) first: the audition uses its training recordings.", [], None
        return
    dataset_dir = dataset_dir_for_adapter(adapter_path, datasets_root=Path(root) / "datasets")
    if dataset_dir is None:
        yield "This voice's training dataset was not found; keep the dataset folder next to the app to audition references.", [], None
        return
    run_dir = run_dir_of(adapter_path)
    rows = load_manifest(dataset_dir)
    training, validation = split_rows(Path(dataset_dir), rows, run_dir)
    yield "Choosing candidate clips near the speaker's typical pitch and pace...", [], None
    current_path = _recommended_lora_reference(adapter_path)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    if current_path and Path(current_path).is_file():
        seen.add(_sha(current_path))
        candidates.append({"key": "current", "label": Path(current_path).name, "audio": str(current_path),
                           "text": _transcript_beside(current_path), "duration_s": _duration_of(current_path), "current": True})
    for row in choose_candidates(Path(dataset_dir), training, max(1, int(candidate_count)) + 2):
        source = audio_path(dataset_dir, row)
        digest = _sha(source)
        if digest in seen:
            continue
        seen.add(digest)
        candidates.append({"key": str(row["id"]), "label": source.name, "audio": str(source), "text": str(row["text"]),
                           "duration_s": float(row.get("duration_s") or 0.0), "current": False})
        if len(candidates) >= int(candidate_count) + (1 if current_path else 0):
            break
    sentences = choose_sentences(Path(dataset_dir), validation, training, max(1, int(sentence_count)),
                                 exclude=[candidate["key"] for candidate in candidates])
    if len(candidates) < 2 or not sentences:
        yield "Not enough clean clips in this voice's dataset for an audition (it needs candidates of 6 to 16 seconds and held-out sentences of 5 to 14 seconds).", [], None
        return
    work = run_dir / "analysis" / AUDITION_RENDER_FOLDER
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    omnivoice = values.get("app.model") == "omnivoice"
    settings = {**values, "generation.word_timestamps": False, "generation.section_takes": 1,
                "generation.num_candidates": 1, "generation.pick_best_candidate": False, "generation.save_as_mp3": False,
                "generation.audio_tuning_preset": "bypass", "generation.use_caption_timing": False,
                "generation.save_used_audio": False, "generation.output_filename": "", "runtime.lora_path": str(adapter_path)}
    for key in ("generation.tuning_low_cut_hz", "generation.tuning_high_cut_hz", "generation.tuning_gain_db",
                "generation.tuning_loudnorm_i", "generation.tuning_deess"):
        settings[key] = None
    clips = []
    total = len(candidates) * len(sentences)
    started = time.perf_counter()
    for c_index, candidate in enumerate(candidates):
        for s_index, sentence in enumerate(sentences):
            done = c_index * len(sentences) + s_index
            yield (f"Rendering {done + 1}/{total}: clip {c_index + 1}/{len(candidates)} ({candidate['label']}), "
                   f"sentence {s_index + 1}/{len(sentences)} | {time.perf_counter() - started:.0f} s"), [], None
            take = dict(settings, **{"generation.seed": 1000 + s_index})
            if omnivoice:
                take.update({"omnivoice.mode": "clone", "omnivoice.reference_text": candidate["text"]})
            request = prepare_generation_request(take, prompt=candidate["audio"], text=str(sentence["text"]), subtitle_file=None,
                                                 image_path=None, emotion_audio=None, model_dir=model_dir,
                                                 output_root=work / f"clip_{c_index + 1:02d}")
            with LAZY_ENGINE.in_use():
                engine = LAZY_ENGINE.get(request["runtime"])
                result = run_generation_request(request, engine)
            clips.append({"audio": str(result["output_path"]), "reference": candidates[0]["audio"],
                          "real_audio": str(audio_path(dataset_dir, sentence)), "text": str(sentence["text"]),
                          "language": _metric_language(sentence, request),
                          "candidate": candidate["key"]})
    yield f"Measuring {len(clips)} takes against the speaker's recordings (speaker and style similarity, word errors)...", [], None
    import torch

    if torch.cuda.is_available():
        free = torch.cuda.mem_get_info()[0] / 1024**3
        if free < 6.0:
            LAZY_ENGINE.unload()  # the measuring models need the room; the voice reloads at the next generation
    from indextts.training.speech_metrics import measure_clips

    measured = measure_clips(clips, model_dir=model_dir, model_config=str(Path(model_dir) / "config.yaml"),
                             device=str((request.get("runtime") or {}).get("device") or "cuda:0").replace("auto", "cuda:0"),
                             output_dir=work, update=lambda *args: None, cancelled=lambda: False)
    gc.collect()
    summary = summarize(measured, candidates)
    winner = next((row for row in summary if row.get("winner")), None)
    report = {"audited_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "voice": str(adapter_path), "speech_model": values.get("app.model"),
              "sentences": [{"id": str(row["id"]), "text": str(row["text"])} for row in sentences],
              "candidates": summary}
    record = save_choice(adapter_path, winner, report, use=use_winner) if winner else audition_record(adapter_path)
    rows_out = table_rows(summary)
    if winner is None:
        yield "No take could be measured; the voice keeps its reference.", rows_out, None
        return
    if winner.get("current"):
        message = (f"**The current reference stays the best clip** ({winner['label']}, speaker similarity "
                   f"{winner['similarity']:.3f}); nothing changed.")
        yield message, rows_out, None
        return
    current = next((row for row in summary if row.get("current")), None)
    gain = f" against {current['similarity']:.3f} for the current reference" if current and current.get("similarity") is not None else ""
    chosen = str(run_dir / record["reference"]) if record and record.get("reference") else None
    message = (f"**Winner: {winner['label']}** (speaker similarity {winner['similarity']:.3f}{gain}, "
               f"style {winner['style']:.3f}, word errors {100 * float(winner['word_errors'] or 0):.1f} %). ")
    message += ("It is now this voice's automatic reference; untick **Use the winner automatically** to go back "
                "to the original." if use_winner and chosen else
                "Saved beside the voice; tick **Use the winner automatically** to make it the voice's reference.")
    yield message, rows_out, chosen if use_winner else None


def audition_overview(adapter_path: str) -> tuple[str, list[list[Any]], Any]:
    """A voice's saved audition for the panel: status, result rows and its Use-the-winner setting.

    A voice without an audition shows nothing and the setting returns to its default (on)."""

    import gradio as gr

    from indextts.training.reference_audition import audition_record

    record = audition_record(adapter_path) if adapter_path else None
    if not record:
        return "", [], gr.update(value=True)
    summary = [row for row in record.get("candidates") or [] if isinstance(row, Mapping)]
    winner = next((row for row in summary if row.get("winner")), None)
    when = str(record.get("audited_at") or "").replace("T", " ")[:16]
    if winner is None:
        status = f"Last audition ({when}): no take could be measured."
    elif not record.get("reference"):
        status = f"Last audition ({when}): the current reference stayed the best clip ({winner['label']})."
    else:
        similarity = winner.get("similarity")
        measured = f", speaker similarity {float(similarity):.3f}" if similarity is not None else ""
        state = ("in use" if record.get("use") else
                 "saved but not in use; tick **Use the winner automatically** to use it")
        status = f"Last audition ({when}): winner {winner['label']}{measured}; {state}."
    return status, table_rows(summary), gr.update(value=bool(record.get("use", True)))


def _metric_language(sentence: Mapping[str, Any], request: Mapping[str, Any]) -> str:
    """The evaluation language of a held-out sentence: its dataset language, else the speech model's language."""

    from indextts.training.speech_metrics import LANGUAGES

    for value in (sentence.get("language"), request.get("language")):
        code = str(value or "").strip().upper()
        if code in LANGUAGES:
            return code
    return "EN"


def _duration_of(path: str | Path) -> float:
    try:
        import soundfile as sf

        return float(sf.info(str(path)).duration)
    except Exception:
        return 0.0


__all__ = ["TABLE_HEADERS", "audition_overview", "run_reference_audition", "table_rows"]
