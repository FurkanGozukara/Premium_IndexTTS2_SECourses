from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from indextts.training.naturalness_ab import (
    DEFAULT_VARIANTS,
    EXPRESSIVE_VARIANTS,
    Variant,
    build_listening_bundles,
    load_sentences,
    load_variants,
    render_report_markdown,
    summarize_variant,
    transform_text,
)
from indextts.training.prosody_metrics import PROSODY_KEYS, compare_prosody, measure_prosody, prosody_table


def _tone(path: Path, seconds: float, *, hz: float = 150.0, vibrato_hz: float = 0.0, pause_at: float | None = None) -> None:
    rate = 16000
    t = np.arange(int(seconds * rate)) / rate
    frequency = hz * (1.0 + (0.08 * np.sin(2 * np.pi * vibrato_hz * t) if vibrato_hz else np.zeros_like(t)))
    phase = 2 * np.pi * np.cumsum(frequency) / rate
    # A plain tone: the pitch tracker resolves it exactly, and vibrato adds measurable pitch movement.
    signal = 0.5 * np.sin(phase)
    if pause_at is not None:
        start = int(pause_at * rate)
        signal[start : start + int(0.3 * rate)] = 0.0
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, signal.astype(np.float32), rate)


def test_prosody_measures_pitch_variability_and_pauses(tmp_path: Path) -> None:
    flat = tmp_path / "flat.wav"
    lively = tmp_path / "lively.wav"
    _tone(flat, 2.0)
    _tone(lively, 2.0, vibrato_hz=2.0, pause_at=0.9)
    text = "one two three four five six"
    flat_metrics = measure_prosody(flat, text)
    lively_metrics = measure_prosody(lively, text)
    assert set(PROSODY_KEYS) <= set(flat_metrics)
    assert flat_metrics["words"] == 6 and flat_metrics["duration_s"] > 1.9
    assert lively_metrics["f0_std_st"] > flat_metrics["f0_std_st"]
    assert lively_metrics["pause_count"] >= 1 and flat_metrics["pause_count"] == 0
    assert 140 < flat_metrics["f0_median_hz"] < 160
    summary = compare_prosody([(flat_metrics, lively_metrics)])
    assert summary["clips"] == 1
    assert summary["f0_std_st"]["generated_lower_share"] == 1.0
    assert summary["liveliness_ratio"] is not None and summary["liveliness_ratio"] < 1.0
    table = prosody_table({"flat": summary})
    assert "| f0_std_st |" in table and "liveliness_ratio" in table
    assert compare_prosody([]) == {"clips": 0}
    empty = tmp_path / "empty.wav"
    sf.write(empty, np.zeros(10, dtype=np.float32), 16000)
    assert measure_prosody(empty, "x")["duration_s"] < 0.01


def test_variant_table_and_text_transforms() -> None:
    assert transform_text("Hello, world; again: yes.", "strip_commas") == "Hello world; again: yes."
    assert transform_text("Hello, world; again: yes.", "strip_clause_marks") == "Hello world again yes."
    assert transform_text("Keep, it.", "none") == "Keep, it."
    defaults = load_variants(None)
    assert [variant.name for variant in defaults] == [variant.name for variant in DEFAULT_VARIANTS]
    assert all(variant.reference is None for variant in defaults)
    with_expressive = load_variants("default", expressive_reference="C:/clips/lively.wav")
    names = {variant.name for variant in with_expressive}
    assert {variant.name for variant in EXPRESSIVE_VARIANTS} <= names
    lively = next(variant for variant in with_expressive if variant.name == "expressive_emotion")
    assert lively.emotion_reference == "C:/clips/lively.wav" and lively.emo_alpha == 0.65
    round_trip = Variant.from_dict(lively.to_dict())
    assert round_trip == lively


def test_variants_from_json_and_sentences_from_reports(tmp_path: Path) -> None:
    variants_path = tmp_path / "variants.json"
    variants_path.write_text(json.dumps({"variants": [{"name": "beams1", "description": "d", "infer": {"num_beams": 1}}]}), encoding="utf-8")
    loaded = load_variants(variants_path)
    assert len(loaded) == 1 and loaded[0].infer == {"num_beams": 1}

    run_dir = tmp_path / "run"
    evaluation = run_dir / "analysis" / "speech_evaluation" / "final_test"
    evaluation.mkdir(parents=True)
    real = tmp_path / "real.wav"
    _tone(real, 1.0)
    cells = [
        {"kind": "matched", "text": "Sentence one.", "real_audio": str(real), "prompt_id": "g:a", "language": "EN", "checkpoint": "Base"},
        {"kind": "matched", "text": "Sentence one.", "real_audio": str(real), "prompt_id": "g:a", "language": "EN", "checkpoint": "final"},
        {"kind": "matched", "text": "Sentence two.", "real_audio": str(tmp_path / "missing.wav"), "prompt_id": "g:b", "language": "EN"},
        {"kind": "long_form", "text": "Long text.", "real_audio": str(real), "prompt_id": "g:c", "language": "EN"},
    ]
    (evaluation / "report.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    sentences = load_sentences(run_dir, "final_test")
    assert [item["text"] for item in sentences] == ["Sentence one."]
    listed = tmp_path / "sentences.json"
    listed.write_text(json.dumps([{"id": "x", "text": "Listed.", "audio": str(real), "bucket": "short"}]), encoding="utf-8")
    assert load_sentences(run_dir, str(listed))[0]["bucket"] == "short"
    assert load_sentences(run_dir, str(listed), limit=0) and load_sentences(run_dir, str(listed), limit=1)


def test_summaries_report_and_blind_bundles(tmp_path: Path) -> None:
    real = tmp_path / "real.wav"
    _tone(real, 1.5, vibrato_hz=2.0)
    rows_by_variant: dict[str, list[dict]] = {}
    for name, vibrato in (("deployed", 0.0), ("sampling", 1.0)):
        rows = []
        for seed in (1, 2):
            clip = tmp_path / name / f"clip_{seed}.wav"
            _tone(clip, 1.5, vibrato_hz=vibrato)
            rows.append({
                "audio": str(clip), "real_audio": str(real), "text": "one two three", "prompt_id": "p1", "seed": seed,
                "checkpoint": name, "error_rate": 0.0, "errors": 0, "units": 3, "duration_s": 1.5,
                "speaker_similarity_real": 0.9, "style_similarity_real": 0.8, "pause_ratio_vs_real": 1.2,
                "prosody": measure_prosody(clip, "one two three"), "real_prosody": measure_prosody(real, "one two three"),
            })
        rows_by_variant[name] = rows
    summaries = {name: summarize_variant(rows) for name, rows in rows_by_variant.items()}
    assert summaries["deployed"]["clips"] == 2
    assert summaries["sampling"]["prosody"]["liveliness_ratio"] > summaries["deployed"]["prosody"]["liveliness_ratio"]
    assert summaries["deployed"]["speaker_similarity_real_mean"] == 0.9
    report = {"run_dir": "run", "checkpoint": "ckpt", "sentence_count": 1, "seeds": [1, 2], "variants": summaries,
              "variant_table": [variant.to_dict() for variant in DEFAULT_VARIANTS[:2]]}
    markdown = render_report_markdown(report)
    assert "| deployed |" in markdown and "| sampling |" in markdown and "**sampling**" in markdown
    keys = build_listening_bundles(rows_by_variant, tmp_path / "listening", seed=3)
    assert len(keys) == 2
    first = tmp_path / "listening" / "blind" / keys[0]["set"]
    assert (first / "Reference.wav").is_file() and (first / "A.wav").is_file() and (first / "B.wav").is_file()
    assert set(keys[0]["letters"].values()) == {"deployed", "sampling"}
    saved = json.loads((tmp_path / "listening" / "keys.json").read_text(encoding="utf-8"))
    assert saved["variants"] == ["deployed", "sampling"]
    limited = build_listening_bundles(rows_by_variant, tmp_path / "listening2", variants=["deployed"], max_sets=1)
    assert limited == []  # a single variant has nothing to compare
