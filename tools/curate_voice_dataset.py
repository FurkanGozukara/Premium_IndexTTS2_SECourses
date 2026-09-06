"""Audit narration identity and transcript agreement; create a clean, source-held-out dataset."""
from __future__ import annotations

import argparse
from collections import Counter
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
from indextts.training.dataset_quality import SpeakerVerifier, TimedTranscript
from indextts.training.speech_metrics import transcript_metrics
from indextts.training.media import measure_edge_silence
from indextts.training.features import _load_audio_16k, _read_audio
from indextts.training.whisper_asr import _ensure_model


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
    parser.add_argument("--check-boundary-words", action="store_true", help="Require the first and last two normalized transcript words to match fresh clip ASR")
    parser.add_argument("--min-edge-silence-ms", type=int, default=0)
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
            try:
                agreement = transcript_metrics(row["text"], hypothesis, language)
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
            if not reasons and (args.transcribe_all or args.check_boundary_words or (wer > args.max_wer and not args.no_asr_recheck)):
                # Source chunk stitching can duplicate words at overlaps. Audit
                # the actual extracted clip before discarding clean narration.
                if asr_pipe is None:
                    from transformers import pipeline
                    asr_pipe = pipeline("automatic-speech-recognition", model=str(_ensure_model(args.whisper)),
                                        device=args.device, dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32)
                waveform, _ = _load_audio_16k(audio)
                result = asr_pipe({"array": waveform.squeeze().numpy(), "sampling_rate": 16000}, return_timestamps=True,
                                  generate_kwargs={"language": str(row.get("language", "EN")).lower(), "task": "transcribe", "do_sample": False})
                hypothesis = str(result["text"]).strip()
                agreement = transcript_metrics(row["text"], hypothesis, language)
                wer = agreement["error_rate"]
                rechecked = True
                asr_rechecked += 1
                asr_recovered += int(source_wer > args.max_wer and wer <= args.max_wer)
            if wer > args.max_wer:
                reasons.append("transcript_disagreement")
            edge_match = agreement["start_matches"] and agreement["end_matches"]
            if args.check_boundary_words and not edge_match:
                reasons.append("transcript_boundary_mismatch")
            item = {"id": row["id"], "source": topic, "text": row["text"], "asr_text": hypothesis,
                    "asr_wer": wer, "asr_error_unit": agreement["error_unit"], "source_asr_wer": source_wer, "source_asr_text": source_hypothesis,
                    "asr_rechecked": rechecked, "boundary_words_match": edge_match,
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
    }
    atomic_write_json(output / "quality_summary.json", summary)
    report(len(rows), "complete", "Voice and transcript audit complete", kept=len(kept), test=len(test),
           rejected=len(rows) - len(kept) - len(test))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
