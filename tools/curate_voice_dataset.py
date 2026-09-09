"""Audit narration identity and transcript agreement; create a clean, source-held-out dataset."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import soundfile as sf
from indextts.training.dataset_manifest import atomic_write_json, load_manifest, summarize_manifest, write_manifest, write_preview_csv
from indextts.training.dataset_quality import SpeakerVerifier, TimedTranscript, transcript_vocabulary
from indextts.training.speech_metrics import lenient_units, transcript_metrics
from indextts.training.media import measure_edge_silence
from indextts.training.features import _load_audio_16k, _read_audio
from indextts.training.whisper_asr import _ensure_model, whisper_device_for_free_vram

# Whisper accepts a short prompt of expected spellings, derived here from the
# speaker's own subtitles. Optional: on a 17-recording narration dataset it
# recovered about 9 more rejected clips per 100 but made 2-5 of 60 previously
# accepted clips fail, so the comparison rules below do the work by default.
PROMPT_TERMS = 40


def transcript_prompt(topic_texts: list[str], vocabulary: list[str]) -> str:
    """Comma-separated spellings for one recording, topped up from the whole dataset."""
    terms = [term for term in transcript_vocabulary(topic_texts) if not term.isdigit()]
    if len(terms) < 10:
        terms += [term for term in vocabulary if term not in terms and not term.isdigit()]
    return ", ".join(terms[:PROMPT_TERMS])


TRANSCRIPT_REASONS = frozenset({"transcript_disagreement", "transcript_boundary_mismatch"})
# The fast turbo model spells technical vocabulary poorly. Measured on one
# narration dataset, the full model recovered a third of the clips turbo had
# rejected on transcript grounds while agreeing with an independent listener
# more often, so rejected clips get a second opinion from it.
DEFAULT_SECOND_OPINION_WHISPER = "openai/whisper-large-v3"


def transcribe_clip(pipe, waveform, language: str, beams: int, prompt_ids=None) -> str:
    """Transcribe one 16 kHz clip with a transformers ASR pipeline."""
    generate_kwargs = {"language": str(language or "EN").lower(), "task": "transcribe", "do_sample": False}
    if beams > 1:
        generate_kwargs["num_beams"] = beams
    if prompt_ids is not None:
        generate_kwargs["prompt_ids"] = prompt_ids
    result = pipe({"array": waveform.squeeze().numpy(), "sampling_rate": 16000}, return_timestamps=True,
                  generate_kwargs=generate_kwargs)
    return str(result["text"]).strip()


SECOND_OPINION_MIN_FREE_GB = 4.5


def _second_opinion_device(args) -> str:
    """Device for the second-opinion model: the audit device when its VRAM fits, else the CPU."""
    requested = str(getattr(args, "second_opinion_device", "auto") or "auto").strip()
    if requested.lower() != "auto":
        return requested
    device = str(args.device)
    if not device.lower().startswith("cuda"):
        return device
    try:
        from indextts.runtime.gpu import gpu_free_gb
        free_gb = gpu_free_gb(int(device.split(":", 1)[1]) if ":" in device else 0)
    except (RuntimeError, TypeError, ValueError, ImportError):
        free_gb = None
    return whisper_device_for_free_vram(device, free_gb, required_gb=SECOND_OPINION_MIN_FREE_GB)


def whisper_prompt_ids(pipe, prompt: str):
    """Prompt ids for the pipeline's tokenizer, or None when unsupported or empty."""
    get_prompt_ids = getattr(getattr(pipe, "tokenizer", None), "get_prompt_ids", None)
    if not prompt.strip() or get_prompt_ids is None:
        return None
    prompt_ids = get_prompt_ids(prompt, return_tensors="pt")
    device = getattr(getattr(pipe, "model", None), "device", None)
    return prompt_ids.to(device) if device is not None else prompt_ids


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", action="append", required=True)
    parser.add_argument("--validation-source", action="append", required=True)
    parser.add_argument("--test-source", action="append", default=[])
    parser.add_argument("--max-wer", type=float, default=0.15)
    parser.add_argument("--min-speaker-similarity", type=float, default=0.70)
    parser.add_argument("--min-window-similarity", type=float, default=0.60)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--whisper", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--no-asr-recheck", action="store_true", help="Disable fresh clip transcription when source-chunk ASR disagrees")
    parser.add_argument("--transcribe-all", action="store_true", help="Transcribe every voice-matched extracted clip")
    parser.add_argument("--check-boundary-words", action="store_true", help="Reject clips whose first or last two transcript words are missing, extra, or different in fresh clip ASR; the subtitles' own spellings of names and terms are accepted")
    parser.add_argument("--min-edge-silence-ms", type=int, default=0)
    parser.add_argument("--asr-beams", type=int, default=3, help="Beam search width for fresh clip transcription; 1 is greedy decoding")
    parser.add_argument("--term-prompt", action="store_true",
                        help="Prompt Whisper with the spellings used in the dataset transcripts. Measured: a few more rejected clips pass, but some clean clips start failing, so it is off by default")
    parser.add_argument("--second-opinion-whisper", default=DEFAULT_SECOND_OPINION_WHISPER,
                        help="Re-transcribe clips that failed only transcript checks with this model and keep them when it agrees with the transcript; pass an empty string to disable")
    parser.add_argument("--second-opinion-device", default="auto",
                        help="Device for the second-opinion model: auto uses --device when at least 4.5 GB of VRAM are free beside the first model, otherwise the CPU")
    parser.add_argument("--state-dir", type=Path, help="Optional UI status and graceful-stop directory")
    args = parser.parse_args()
    try:
        run_curation(args)
    except Exception as exc:
        if args.state_dir:
            try:
                previous = json.loads((args.state_dir / "status.json").read_text())
            except (OSError, ValueError):
                previous = {}
            atomic_write_json(args.state_dir / "status.json", {**previous,
                "phase": "cancelled" if isinstance(exc, CurationCancelled) else "failed",
                "message": str(exc), "updated_at": time.time(),
            })
        if isinstance(exc, CurationCancelled):
            print(str(exc), flush=True)
            return
        raise


class CurationCancelled(RuntimeError):
    pass


def run_curation(args: argparse.Namespace) -> None:
    torch.set_num_threads(4)
    source, output = args.dataset.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory to preserve prior curation: {output}")
    rows = load_manifest(source)
    if not rows:
        raise ValueError("The source dataset has no clips")
    split_sources = {Path(row["source_media"]).stem for row in rows}
    unknown = set(args.validation_source + args.test_source) - split_sources
    if unknown:
        raise ValueError(f"Holdout sources missing from manifest: {sorted(unknown)}")
    if set(args.validation_source) & set(args.test_source):
        raise ValueError("Validation and test source recordings must differ")
    if not split_sources - set(args.validation_source + args.test_source):
        raise ValueError("Reserve at least one source recording for training")
    if args.test_source and output.with_name(output.name + "_test").exists():
        raise FileExistsError(output.with_name(output.name + "_test"))
    if not 0 <= args.min_edge_silence_ms <= 500:
        raise ValueError("Minimum edge silence must be between 0 and 500 ms")
    output.mkdir(parents=True)
    started = time.monotonic()

    def report(completed: int, phase: str, message: str, **extra: object) -> None:
        elapsed = time.monotonic() - started
        payload = {"completed": completed, "total": len(rows), "phase": phase,
                   "message": message, "elapsed_s": elapsed, "updated_at": time.time(),
                   "eta_s": elapsed * (len(rows) - completed) / completed if completed else None,
                   **extra}
        atomic_write_json(output / "quality_progress.json", payload)
        if args.state_dir:
            atomic_write_json(args.state_dir / "status.json", payload)

    report(0, "running", "Loading speaker verification model")
    transcripts = {}
    for path in (source / "whisper").glob("*.words.json"):
        content = json.loads(path.read_text(encoding="utf-8"))
        transcripts[Path(content["source_media"]).stem] = TimedTranscript(content["words"])
    verifier = SpeakerVerifier(args.reference, model_dir=args.model_dir, device=args.device)
    kept, test, audit = [], [], []
    asr_pipe = None
    asr_rechecked, asr_recovered = 0, 0
    rejected = Counter()
    # The user's transcripts are the authority on how names and terms are
    # written. Recognizer spellings of those words are not transcript errors.
    vocabulary = transcript_vocabulary(row["text"] for row in rows)
    texts_by_topic: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        texts_by_topic[Path(row["source_media"]).stem].append(row["text"])
    lenient_by_language: dict[str, frozenset[str]] = {}
    prompt_ids_by_topic: dict[str, object] = {}
    asr_beams = max(1, int(getattr(args, "asr_beams", 1) or 1))
    use_term_prompt = bool(getattr(args, "term_prompt", False))
    second_opinion_model = str(getattr(args, "second_opinion_whisper", "") or "").strip()
    second_pipe = None
    second_opinions, second_recovered = 0, 0
    with (output / "quality_audit.jsonl").open("w", encoding="utf-8") as audit_file:
        for index, row in enumerate(rows):
            if args.state_dir and (args.state_dir / "stop.flag").exists():
                raise CurationCancelled("Audit stopped. Source clips and partial audit decisions are preserved.")
            audio = source / row["audio"]
            topic = Path(row["source_media"]).stem
            transcript = transcripts.get(topic)
            reasons = []
            hypothesis = transcript.between(float(row["source_start_s"]) - .04, float(row["source_end_s"]) + .04) if transcript else ""
            language = str(row.get("language") or "EN").upper()
            if language not in lenient_by_language:
                lenient_by_language[language] = lenient_units(vocabulary, language)
            lenient = lenient_by_language[language]
            try:
                agreement = transcript_metrics(row["text"], hypothesis, language, lenient_terms=lenient)
            except ValueError:
                reasons.append("empty_normalized_transcript")
                agreement = {"error_rate": 1.0, "start_matches": False, "end_matches": False, "error_unit": "unknown"}
            wer = agreement["error_rate"]
            source_hypothesis, source_wer = hypothesis, wer
            rechecked = False
            if transcript is None and args.no_asr_recheck and not (args.transcribe_all or args.check_boundary_words):
                reasons.append("missing_word_alignment")
            scores = verifier.score(audio)
            if scores["speaker_similarity"] < args.min_speaker_similarity:
                reasons.append("different_speaker_or_music")
            if scores["speaker_window_min"] < args.min_window_similarity:
                reasons.append("speaker_change_or_contaminated_window")
            edge_quality = {}
            if args.min_edge_silence_ms:
                samples, sr = sf.read(audio, dtype="float32")
                edge_quality = measure_edge_silence(samples, sr)
                if min(edge_quality.values()) < args.min_edge_silence_ms:
                    reasons.append("unsafe_audio_boundary")
            waveform = None
            if not reasons and (args.transcribe_all or args.check_boundary_words or (wer > args.max_wer and not args.no_asr_recheck)):
                # Source chunk stitching can duplicate words at overlaps. Audit
                # the actual extracted clip before discarding clean narration.
                if asr_pipe is None:
                    from transformers import pipeline
                    asr_pipe = pipeline("automatic-speech-recognition", model=str(_ensure_model(args.whisper)),
                                        device=args.device, dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32)
                waveform, _ = _load_audio_16k(audio)
                prompt_ids = None
                if use_term_prompt:
                    if topic not in prompt_ids_by_topic:
                        prompt_ids_by_topic[topic] = whisper_prompt_ids(asr_pipe, transcript_prompt(texts_by_topic[topic], vocabulary))
                    prompt_ids = prompt_ids_by_topic[topic]
                hypothesis = transcribe_clip(asr_pipe, waveform, str(row.get("language", "EN")), asr_beams, prompt_ids)
                agreement = transcript_metrics(row["text"], hypothesis, language, lenient_terms=lenient)
                wer = agreement["error_rate"]
                rechecked = True
                asr_rechecked += 1
                asr_recovered += int(source_wer > args.max_wer and wer <= args.max_wer)
            if wer > args.max_wer:
                reasons.append("transcript_disagreement")
            edge_match = agreement["start_matches"] and agreement["end_matches"]
            if args.check_boundary_words and not edge_match:
                reasons.append("transcript_boundary_mismatch")
            second_text, second_wer, second_used = None, None, False
            if second_opinion_model and reasons and set(reasons) <= TRANSCRIPT_REASONS:
                # Only transcript checks failed: a stronger recognizer decides
                # whether the recording says what the transcript says.
                if second_pipe is None:
                    second_device = _second_opinion_device(args)
                    print(f">> second-opinion Whisper {second_opinion_model} on {second_device}", flush=True)
                    second_pipe = pipeline("automatic-speech-recognition", model=str(_ensure_model(second_opinion_model)),
                                           device=second_device, dtype=torch.bfloat16 if second_device.startswith("cuda") else torch.float32)
                if waveform is None:
                    waveform, _ = _load_audio_16k(audio)
                second_text = transcribe_clip(second_pipe, waveform, str(row.get("language", "EN")), asr_beams)
                second = transcript_metrics(row["text"], second_text, language, lenient_terms=lenient)
                second_wer = second["error_rate"]
                second_opinions += 1
                second_edge = second["start_matches"] and second["end_matches"]
                if second_wer <= args.max_wer and (second_edge or not args.check_boundary_words):
                    reasons = [reason for reason in reasons if reason not in TRANSCRIPT_REASONS]
                    hypothesis, agreement, wer, edge_match = second_text, second, second_wer, second_edge
                    second_used = True
                    second_recovered += 1
            item = {"id": row["id"], "source": topic, "text": row["text"], "asr_text": hypothesis,
                    "asr_wer": wer, "asr_error_unit": agreement["error_unit"], "source_asr_wer": source_wer, "source_asr_text": source_hypothesis,
                    "asr_rechecked": rechecked, "boundary_words_match": edge_match,
                    "forgiven_units": int(agreement.get("forgiven_units", 0) or 0),
                    "asr_model": "second_opinion" if second_used else "primary",
                    "second_opinion_text": second_text, "second_opinion_wer": second_wer,
                    **edge_quality, **scores, "reasons": reasons, "audio": str(audio)}
            audit.append(item)
            audit_file.write(json.dumps(item, ensure_ascii=False) + "\n")
            audit_file.flush()
            if reasons:
                rejected.update(reasons)
            else:
                copied = {**row, "asr_wer": round(wer, 6), "speaker_similarity": round(scores["speaker_similarity"], 6),
                          "speaker_window_min": round(scores["speaker_window_min"], 6),
                          "boundary_words_match": edge_match, **edge_quality,
                          "split": "val" if topic in args.validation_source else "train"}
                if topic in args.test_source:
                    test.append(copied)
                else:
                    kept.append(copied)
            if (index + 1) % 5 == 0 or index + 1 == len(rows):
                print(f"Audited {index + 1}/{len(rows)} | retained {len(kept)} + {len(test)} test | rejected {index + 1 - len(kept) - len(test)}", flush=True)
                report(index + 1, "running", f"Audited {index + 1}/{len(rows)} clips",
                       kept=len(kept), test=len(test), rejected=index + 1 - len(kept) - len(test))
    if not any(row["split"] == "train" for row in kept) or not any(row["split"] == "val" for row in kept):
        raise ValueError("Curation must retain training and validation audio")
    # A separate test directory keeps test targets entirely outside trainer inputs.
    destinations = [(output, kept)]
    if args.test_source:
        if not test:
            raise ValueError("Curation retained no final-test audio")
        test_output = output.with_name(output.name + "_test")
        if test_output.exists():
            raise FileExistsError(test_output)
        test_output.mkdir()
        reference_rows = sorted((row for row in kept if row["split"] == "train"),
                                key=lambda row: hashlib.sha256(str(row["id"]).encode()).hexdigest())[:16]
        test_rows = [{**row, "split": "val"} for row in test] + reference_rows
        destinations.append((test_output, test_rows))
    for destination, selected in destinations:
        for row in selected:
            link_or_copy(source / row["audio"], destination / row["audio"])
        reference_candidates = []
        for index, reference in enumerate(args.reference, 1):
            original = Path(reference).resolve()
            candidate = destination / "reference_candidates" / f"verified_reference_{index:02d}.wav"
            candidate.parent.mkdir(parents=True, exist_ok=True)
            if original.suffix.lower() == ".wav":
                shutil.copy2(original, candidate)
            else:
                waveform, rate = _read_audio(original)
                sf.write(str(candidate), waveform.squeeze(0).numpy(), rate, subtype="PCM_24")
            reference_candidates.append(candidate.relative_to(destination).as_posix())
        write_manifest(destination / "manifest.jsonl", selected)
        write_preview_csv(destination / "preview.csv", selected)
        atomic_write_json(destination / "dataset_info.json", {
            "name": destination.name, "status": "complete", **summarize_manifest(selected),
            "source_dataset": str(source), "quality_thresholds": {
                "max_asr_wer": args.max_wer, "min_speaker_similarity": args.min_speaker_similarity,
                "min_window_similarity": args.min_window_similarity,
                "fresh_asr_all": args.transcribe_all, "check_boundary_words": args.check_boundary_words,
                "min_edge_silence_ms": args.min_edge_silence_ms,
                "transcript_vocabulary_terms": len(vocabulary), "asr_beams": asr_beams,
                "asr_term_prompt": use_term_prompt, "second_opinion_whisper": second_opinion_model,
            }, "split_counts": dict(Counter(row["split"] for row in selected)),
            "reference_audio": args.reference,
            "reference_candidates": reference_candidates,
        })
    summary = {
        "raw_clips": len(rows), "retained_clips": len(kept), "test_clips": len(test),
        "rejected_clips": len(rows) - len(kept) - len(test), "rejection_reasons": dict(rejected),
        "training": summarize_manifest([row for row in kept if row["split"] == "train"]),
        "validation": summarize_manifest([row for row in kept if row["split"] == "val"]),
        "test": summarize_manifest(test), "validation_sources": args.validation_source, "test_sources": args.test_source,
        "clip_asr_rechecks": asr_rechecked, "clips_recovered_by_fresh_asr": asr_recovered,
        "transcript_vocabulary_terms": len(vocabulary), "transcript_vocabulary_sample": vocabulary[:PROMPT_TERMS],
        "asr_beams": asr_beams, "asr_term_prompt": use_term_prompt,
        "second_opinion_whisper": second_opinion_model, "second_opinion_checks": second_opinions,
        "clips_recovered_by_second_opinion": second_recovered,
    }
    atomic_write_json(output / "quality_summary.json", summary)
    report(len(rows), "complete", "Voice and transcript audit complete", kept=len(kept), test=len(test),
           rejected=len(rows) - len(kept) - len(test))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
