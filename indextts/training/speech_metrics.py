"""Language-aware, paired speech measurements; no model is loaded on import."""
from __future__ import annotations

from collections import Counter, defaultdict
import gc
import math
from pathlib import Path
import unicodedata
from typing import Any, Callable

import numpy as np


LANGUAGES = {"EN": "english", "ZH": "chinese", "JA": "japanese", "AR": "arabic", "ES": "spanish"}


def transcript_units(text: str, language: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    if language == "EN":
        from .dataset_quality import normalized_words
        return normalized_words(text)
    text = "".join(" " if unicodedata.category(c)[0] in {"P", "S"} else c for c in text)
    if language in {"ZH", "JA"}:
        return [c for c in text if not c.isspace()]
    return text.split()


def transcript_metrics(reference: str, hypothesis: str, language: str) -> dict[str, Any]:
    ref, hyp = transcript_units(reference, language), transcript_units(hypothesis, language)
    if not ref:
        raise ValueError("Speech evaluation text contains no scoreable units")
    distances = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        previous, distances = distances, [i] + [0] * len(hyp)
        for j, b in enumerate(hyp, 1):
            distances[j] = min(previous[j] + 1, distances[j - 1] + 1, previous[j - 1] + (a != b))
    edge = min(2, len(ref))
    end_matches = len(hyp) >= edge and ref[-edge:] == hyp[-edge:]
    n = 6 if language in {"ZH", "JA"} else 3
    def grams(units: list[str]) -> Counter:
        return Counter(tuple(units[i:i+n]) for i in range(len(units)-n+1))
    expected, observed = grams(ref), grams(hyp)
    repetition = any(count >= max(3, expected[gram] + 2) for gram, count in observed.items())
    return {"errors": distances[-1], "units": len(ref), "error_rate": distances[-1] / len(ref),
            "error_unit": "character" if language in {"ZH", "JA"} else "word",
            "start_matches": len(hyp) >= edge and ref[:edge] == hyp[:edge], "end_matches": end_matches,
            "possible_truncation": len(hyp) < 0.6 * len(ref) and not end_matches,
            "possible_repetition": repetition}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No speech measurements to summarize")
    def mean(key: str) -> float | None:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return float(np.mean(values)) if values else None
    return {"clips": len(rows), "mean_error_rate": mean("error_rate"),
            "corpus_error_rate": sum(row["errors"] for row in rows) / sum(row["units"] for row in rows),
            "worst_error_rate": max(row["error_rate"] for row in rows),
            "speaker_similarity": mean("speaker_similarity"), "style_similarity_real": mean("style_similarity_real"),
            "duration_ratio_vs_real": mean("duration_ratio_vs_real"),
            "failure_count": sum(bool(row["invalid_audio"] or row["possible_truncation"] or row["possible_repetition"]) for row in rows),
            "edge_mismatch_count": sum(not row["start_matches"] or not row["end_matches"] for row in rows),
            "invalid_audio_count": sum(bool(row["invalid_audio"]) for row in rows)}


def paired_difference(rows: list[dict[str, Any]], baseline: list[dict[str, Any]], key: str,
                      seed: int = 42) -> dict[str, Any]:
    """Bootstrap whole prompts, keeping a prompt's generation seeds together."""
    base = {(r["prompt_id"], r["seed"]): r for r in baseline}
    if len(base) != len(baseline) or len(rows) != len(base):
        raise ValueError("Paired speech comparisons require unique, complete prompt/seed coverage")
    differences: dict[str, list[float]] = defaultdict(list)
    seen = set()
    for row in rows:
        pair = (row["prompt_id"], row["seed"])
        if pair in seen or pair not in base:
            raise ValueError("Mismatched prompt/seed coverage in speech evaluation")
        seen.add(pair)
        if row.get(key) is not None and base[pair].get(key) is not None:
            differences[row["prompt_id"]].append(float(row[key]) - float(base[pair][key]))
    values = np.asarray([np.mean(group) for group in differences.values()], dtype=float)
    if not len(values):
        return {"mean": None, "ci95": None, "prompts": 0}
    sampled = np.random.default_rng(seed).choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
    return {"mean": float(values.mean()), "ci95": [float(x) for x in np.quantile(sampled, [0.025, 0.975])],
            "prompts": len(values), "cluster": "prompt (all generation seeds together)"}


def select_recommendation(candidates: list[dict[str, Any]], rows: list[dict[str, Any]],
                          policy: dict[str, Any]) -> dict[str, Any]:
    for row in rows:
        for key in ("error_rate", "speaker_similarity"):
            if row.get(key) is not None and not math.isfinite(float(row[key])):
                raise FloatingPointError(f"Non-finite {key}; refusing speech checkpoint selection")
    grouped = {item["label"]: [row for row in rows if row["checkpoint"] == item["label"]] for item in candidates}
    base_rows = grouped.get("Base", [])
    base = summarize(base_rows)
    if base["invalid_audio_count"] == base["clips"]:
        raise ValueError("Base produced no valid audio; the speech benchmark cannot make a reliable recommendation")
    results = []
    for candidate in candidates:
        measured = grouped[candidate["label"]]
        summary = summarize(measured)
        delta = paired_difference(measured, base_rows, "error_rate")
        speaker = paired_difference(measured, base_rows, "speaker_similarity")
        reasons = []
        if candidate["path"]:
            if delta["mean"] > float(policy["max_wer_increase"]):
                reasons.append("transcript error exceeds the allowed increase over Base")
            if speaker["mean"] is None or speaker["mean"] < -float(policy["max_speaker_drop"]):
                reasons.append("speaker similarity falls below the allowed Base margin")
            if summary["failure_count"] > base["failure_count"]:
                reasons.append("more invalid, possibly truncated, or repetitive clips than Base")
        results.append({**candidate, **summary, "error_delta_vs_base": delta,
                        "speaker_delta_vs_base": speaker, "eligible": not reasons, "rejection_reasons": reasons})
    eligible = [r for r in results if r["eligible"]]
    lowest = min(eligible, key=lambda r: r["mean_error_rate"])
    comparable = []
    for item in eligible:
        delta = paired_difference(grouped[item["label"]], grouped[lowest["label"]], "error_rate")
        # Do not select on a tiny ASR difference within this prompt suite's uncertainty.
        if delta["mean"] <= 0.002 or delta["ci95"][0] <= 0 <= delta["ci95"][1]:
            comparable.append(item)
    best = min(comparable, key=lambda r: (float(r.get("val_loss") if r.get("val_loss") is not None else float("inf")),
                                          r["mean_error_rate"], r["label"]))
    return {"status": "complete", "recommended_kind": "adapter" if best["path"] else "base",
            "recommended_checkpoint": best["path"], "recommended_label": best["label"],
            "candidates": results, "listening_status": "not_rated",
            "decision": "Observed Base regression guards, then paired transcript comparison; validation loss breaks unresolved ties.",
            "scope": "Provisional automatic recommendation for this development suite; human listening is still needed to judge naturalness."}


def measure_clips(clips: list[dict[str, Any]], *, model_dir: str, model_config: str, device: str,
                  output_dir: Path, update: Callable[[str, int, int], None],
                  cancelled: Callable[[], bool]) -> list[dict[str, Any]]:
    """Measure entire transcripts and distributed 20-second embedding windows."""
    import torch
    from transformers import pipeline
    from indextts.runtime import ProgressReporter
    from .features import FeatureCacheConfig, _FeatureModels, _load_audio_16k
    from .whisper_asr import _ensure_model
    from .dataset_manifest import atomic_write_json

    requests = {}
    for clip in clips:
        for key in ("audio", "reference", "real_audio"):
            if clip.get(key):
                requests[(clip[key], clip["language"])] = None
    features = {}
    models = _FeatureModels(FeatureCacheConfig(str(output_dir), model_dir=model_dir,
                                              model_config=model_config, device=device), ProgressReporter("speech clips"))
    for index, path in enumerate(sorted({path for path, _ in requests}), 1):
        if cancelled():
            raise InterruptedError("Speech evaluation canceled")
        wave, duration = _load_audio_16k(Path(path))
        array = wave.numpy()
        invalid = not np.isfinite(array).all() or duration < 0.25 or float(np.sqrt(np.mean(np.square(array)))) < 1e-5
        entry = {"duration_s": duration, "invalid_audio": invalid, "speaker": None, "style": None}
        if not invalid:
            length = min(wave.shape[-1], 20 * 16000)
            starts = sorted(set([0, (wave.shape[-1] - length) // 2, wave.shape[-1] - length]))
            speakers, styles = [], []
            for start in starts:
                snippet = wave[:, start:start+length]
                semantic = models.w2v_features([snippet])[0]
                _, speaker, _, style = models.item_features(snippet, semantic)
                if not torch.isfinite(speaker).all() or not torch.isfinite(style).all():
                    raise FloatingPointError(f"Non-finite speech embedding for {path}")
                speakers.append(torch.nn.functional.normalize(speaker, dim=0))
                styles.append(torch.nn.functional.normalize(style, dim=0))
            entry.update(speaker=torch.nn.functional.normalize(torch.stack(speakers).mean(0), dim=0),
                         style=torch.nn.functional.normalize(torch.stack(styles).mean(0), dim=0))
        features[path] = entry
        update("Measuring speaker and style", index, len({p for p, _ in requests}))
    del models
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if any(features[clip["reference"]]["invalid_audio"] for clip in clips):
        raise ValueError("A benchmark voice reference is empty, silent, or non-finite; choose a usable training reference")
    whisper = "openai/whisper-large-v3-turbo"
    pipe = pipeline("automatic-speech-recognition", model=str(_ensure_model(whisper)), device=device,
                    dtype=torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32)
    transcripts = {}
    for index, (path, language) in enumerate(sorted(requests), 1):
        if cancelled():
            raise InterruptedError("Speech evaluation canceled")
        text = ""
        if not features[path]["invalid_audio"]:
            wave, _ = _load_audio_16k(Path(path))
            result = pipe({"array": wave.squeeze().numpy(), "sampling_rate": 16000}, return_timestamps=True,
                          generate_kwargs={"language": LANGUAGES[language], "task": "transcribe", "do_sample": False})
            text = str(result["text"]).strip()
        transcripts[(path, language)] = text
        update("Transcribing evaluation audio", index, len(requests))
        atomic_write_json(output_dir / "transcriptions.json", [
            {"audio": p, "language": lang, "text": text} for (p, lang), text in transcripts.items()])
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    measured = []
    for clip in clips:
        feature, ref = features[clip["audio"]], features[clip["reference"]]
        text = transcripts[(clip["audio"], clip["language"])]
        def similarity(left: dict, right: dict, key: str) -> float | None:
            if left[key] is None or right[key] is None:
                return None
            return float(torch.dot(left[key], right[key]))
        row = {**clip, **transcript_metrics(clip["text"], text, clip["language"]), "asr_text": text,
               "duration_s": feature["duration_s"], "invalid_audio": feature["invalid_audio"],
               "speaker_similarity": similarity(feature, ref, "speaker")}
        if clip.get("real_audio"):
            real = features[clip["real_audio"]]
            row.update(style_similarity_real=similarity(feature, real, "style"),
                       duration_ratio_vs_real=feature["duration_s"] / max(0.001, real["duration_s"]))
        measured.append(row)
    return measured
