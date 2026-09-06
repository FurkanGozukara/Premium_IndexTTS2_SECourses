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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import soundfile as sf
from indextts.training.dataset_manifest import atomic_write_json, load_manifest, summarize_manifest, write_manifest, write_preview_csv
from indextts.training.dataset_quality import SpeakerVerifier, TimedTranscript, word_error_counts
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
    parser.add_argument("--whisper", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--no-asr-recheck", action="store_true", help="Disable fresh clip transcription when source-chunk ASR disagrees")
    args = parser.parse_args()
    torch.set_num_threads(4)
    source, output = args.dataset.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory to preserve prior curation: {output}")
    output.mkdir(parents=True)
    rows = load_manifest(source)
    transcripts = {}
    for path in (source / "whisper").glob("*.words.json"):
        content = json.loads(path.read_text(encoding="utf-8"))
        transcripts[Path(content["source_media"]).stem] = TimedTranscript(content["words"])
    verifier = SpeakerVerifier(args.reference, device=args.device)
    kept, test, audit = [], [], []
    asr_pipe = None
    asr_rechecked, asr_recovered = 0, 0
    rejected = Counter()
    split_sources = {Path(row["source_media"]).stem for row in rows}
    unknown = set(args.validation_source + args.test_source) - split_sources
    if unknown:
        raise ValueError(f"Holdout sources missing from manifest: {sorted(unknown)}")
    if set(args.validation_source) & set(args.test_source):
        raise ValueError("Validation and test source recordings must differ")
    with (output / "quality_audit.jsonl").open("w", encoding="utf-8") as audit_file:
        for index, row in enumerate(rows):
            audio = source / row["audio"]
            topic = Path(row["source_media"]).stem
            transcript = transcripts.get(topic)
            reasons = []
            hypothesis = transcript.between(float(row["source_start_s"]) - .04, float(row["source_end_s"]) + .04) if transcript else ""
            errors, words = word_error_counts(row["text"], hypothesis)
            wer = errors / max(1, words)
            source_hypothesis, source_wer = hypothesis, wer
            rechecked = False
            if transcript is None:
                reasons.append("missing_word_alignment")
            scores = verifier.score(audio)
            if scores["speaker_similarity"] < args.min_speaker_similarity:
                reasons.append("different_speaker_or_music")
            if scores["speaker_window_min"] < args.min_window_similarity:
                reasons.append("speaker_change_or_contaminated_window")
            if not reasons and wer > args.max_wer and not args.no_asr_recheck:
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
                errors, words = word_error_counts(row["text"], hypothesis)
                wer = errors / max(1, words)
                rechecked = True
                asr_rechecked += 1
                asr_recovered += int(wer <= args.max_wer)
            if transcript is not None and wer > args.max_wer:
                reasons.append("transcript_disagreement")
            item = {"id": row["id"], "source": topic, "text": row["text"], "asr_text": hypothesis,
                    "asr_wer": wer, "source_asr_wer": source_wer, "source_asr_text": source_hypothesis,
                    "asr_rechecked": rechecked, **scores, "reasons": reasons, "audio": str(audio)}
            audit.append(item)
            audit_file.write(json.dumps(item, ensure_ascii=False) + "\n")
            audit_file.flush()
            if reasons:
                rejected.update(reasons)
            else:
                copied = {**row, "asr_wer": round(wer, 6), "speaker_similarity": round(scores["speaker_similarity"], 6),
                          "speaker_window_min": round(scores["speaker_window_min"], 6),
                          "split": "val" if topic in args.validation_source else "train"}
                if topic in args.test_source:
                    test.append(copied)
                else:
                    kept.append(copied)
            if (index + 1) % 25 == 0 or index + 1 == len(rows):
                print(f"Audited {index + 1}/{len(rows)} | retained {len(kept)} + {len(test)} test | rejected {index + 1 - len(kept) - len(test)}", flush=True)
                atomic_write_json(output / "quality_progress.json", {"completed": index + 1, "total": len(rows), "kept": len(kept), "test": len(test)})
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
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
