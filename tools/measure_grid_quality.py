"""Measure a saved listening grid with ASR, speaker/style embeddings and matched speech rate.

Optional --targets JSON is a list of {text_index, text, audio} real recordings.
These measurements are reproducible proxies; they do not replace listening.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from indextts.runtime import ProgressReporter
from indextts.training.dataset_manifest import atomic_write_json
from indextts.training.dataset_quality import normalized_words, word_error_counts
from indextts.training.features import FeatureCacheConfig, _FeatureModels, _load_audio_16k
from indextts.training.whisper_asr import _ensure_model


def summarize(rows: list[dict]) -> dict:
    result = {"clips": len(rows), "corpus_wer": sum(row["word_errors"] for row in rows) / max(1, sum(row["words"] for row in rows)),
              "mean_wer": float(np.mean([row["wer"] for row in rows]))}
    for key in ("speaker_similarity", "style_similarity_real", "style_similarity_reference", "words_per_s", "rate_ratio_vs_real", "f0_median_hz"):
        values = [row[key] for row in rows if row.get(key) is not None]
        result[key] = float(np.mean(values)) if values else None
    ratios = [row["duration_ratio_vs_real"] for row in rows if row.get("duration_ratio_vs_real") is not None]
    result["paired_duration_ratio_median"] = float(np.median(ratios)) if ratios else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grid", type=Path)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--whisper", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = args.grid.resolve()
    grid = json.loads((root / "grid.json").read_text(encoding="utf-8"))
    cells = grid["cells"]
    if not cells or any(cell["status"] != "complete" for cell in cells):
        raise ValueError("Every grid cell must have completed before measurement")
    targets = {int(row["text_index"]): row for row in json.loads(args.targets.read_text(encoding="utf-8"))} if args.targets else {}
    for cell in cells:
        if cell["text_index"] in targets:
            if normalized_words(cell["text"]) != normalized_words(targets[cell["text_index"]]["text"]):
                raise ValueError(f"Real target text does not match grid text {cell['text_index']}")
    output = args.output or root / "quality_metrics.json"
    paths = sorted({str(Path(cell["audio_path"]).resolve()) for cell in cells}
                   | {str(Path(cell["reference"]).resolve()) for cell in cells}
                   | {str(Path(row["audio"]).resolve()) for row in targets.values()})
    models = _FeatureModels(FeatureCacheConfig(str(root), device=args.device), ProgressReporter("clips", total=len(paths)))
    features = {}
    import librosa
    for index, path in enumerate(paths):
        wave, duration = _load_audio_16k(Path(path))
        # Long-form grids are compared using the first 20 seconds for global
        # embeddings; duration and transcription cover the entire utterance.
        snippet = wave[:, :20*16000]
        semantic = models.w2v_features([snippet])[0]
        _, speaker, _, style = models.item_features(snippet, semantic)
        f0, voiced, _ = librosa.pyin(wave.squeeze().numpy(), sr=16000, fmin=60, fmax=400, frame_length=1024, hop_length=256)
        pitch = f0[voiced & np.isfinite(f0)]
        features[path] = {"speaker": torch.nn.functional.normalize(speaker, dim=0),
                          "style": torch.nn.functional.normalize(style, dim=0),
                          "duration_s": duration, "f0_median_hz": float(np.median(pitch)) if len(pitch) else None}
        print(f"Measured embeddings/pitch {index+1}/{len(paths)}", flush=True)
    del models
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    from transformers import pipeline
    pipe = pipeline("automatic-speech-recognition", model=str(_ensure_model(args.whisper)), device=args.device,
                    dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32)
    transcriptions = {}
    for index, path in enumerate(paths):
        wave, _ = _load_audio_16k(Path(path))
        transcription = pipe({"array": wave.squeeze().numpy(), "sampling_rate": 16000}, return_timestamps=True,
                             generate_kwargs={"language": "english", "task": "transcribe", "do_sample": False})
        transcriptions[path] = str(transcription["text"]).strip()
        atomic_write_json(output.with_name(output.stem + "_asr.json"), transcriptions)
        print(f"Transcribed {index+1}/{len(paths)}", flush=True)
    del pipe

    def measure(path: str, text: str, reference: str, real_path: str | None = None) -> dict:
        path, reference = str(Path(path).resolve()), str(Path(reference).resolve())
        feature, ref = features[path], features[reference]
        errors, words = word_error_counts(text, transcriptions[path])
        row = {"audio": path, "text": text, "asr_text": transcriptions[path], "word_errors": errors, "words": words,
               "wer": errors / max(1, words), "duration_s": feature["duration_s"], "words_per_s": words / feature["duration_s"],
               "speaker_similarity": float(torch.dot(feature["speaker"], ref["speaker"])),
               "style_similarity_reference": float(torch.dot(feature["style"], ref["style"])), "f0_median_hz": feature["f0_median_hz"]}
        if real_path:
            real = features[str(Path(real_path).resolve())]
            row.update(style_similarity_real=float(torch.dot(feature["style"], real["style"])),
                       duration_ratio_vs_real=feature["duration_s"] / real["duration_s"],
                       rate_ratio_vs_real=real["duration_s"] / feature["duration_s"])
        return row

    measured = []
    groups = defaultdict(list)
    for cell in cells:
        target = targets.get(cell["text_index"])
        row = measure(cell["audio_path"], cell["text"], cell["reference"], target["audio"] if target else None)
        row.update(checkpoint=cell["checkpoint_label"], strength=cell["strength"], seed=cell["seed"], text_index=cell["text_index"])
        measured.append(row)
        groups[f"{cell['checkpoint_label']} @ {cell['strength']:g}"].append(row)
    real = [measure(row["audio"], row["text"], cells[0]["reference"]) for row in targets.values()]
    report = {"grid": str(root), "whisper": args.whisper, "embedding_seconds": 20, "real": summarize(real) if real else None,
              "checkpoints": {label: summarize(rows) for label, rows in groups.items()}, "cells": measured, "real_cells": real,
              "note": "ASR and embedding scores are automated proxies, not human listening ratings. Speaking-rate ratios use matched text only."}
    atomic_write_json(output, report)
    print(json.dumps(report["checkpoints"], indent=2), flush=True)


if __name__ == "__main__":
    main()
