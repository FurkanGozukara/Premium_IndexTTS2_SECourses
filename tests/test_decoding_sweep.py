"""Per-adapter decoding sweep: variant judgement, knob selection, saved settings, and the report."""
import json

from indextts.training.decoding_sweep import (
    decoding_markdown,
    evaluate_variant,
    load_decoding_settings,
    select_decoding,
)


def _rows(similarity: float, errors: int, count: int = 8) -> list[dict]:
    rows = []
    for index in range(count):
        rows.append({"prompt_id": f"g:p{index}", "seed": 42, "speaker_similarity_real": similarity, "speaker_similarity": similarity,
                     "errors": errors if index == 0 else 0, "units": 10, "error_rate": (errors / 10) if index == 0 else 0.0,
                     "invalid_audio": False, "possible_truncation": False, "possible_repetition": False,
                     "start_matches": True, "end_matches": True, "style_similarity_real": 0.7, "duration_ratio_vs_real": 1.0,
                     "pause_time_fraction": 0.1, "pause_ratio_vs_real": 1.0, "pause_s": 0.5, "real_pause_s": 0.5})
    return rows


def test_variant_judgement_scores_similarity_against_word_error():
    baseline = _rows(0.80, errors=2)
    policy = {"max_wer_increase": 0.02, "max_speaker_drop": 0.03}
    better = evaluate_variant(_rows(0.84, errors=2), baseline, policy=policy)
    assert better["passes"] and abs(better["similarity"] - 0.04) < 1e-9 and better["wer_increase"] == 0.0
    costly = evaluate_variant(_rows(0.84, errors=5), baseline, policy=policy)  # +3.75 points of word error
    assert not costly["passes"] and "word error rate rose" in costly["reasons"][0]
    small = evaluate_variant(_rows(0.803, errors=2), baseline, policy=policy)
    assert not small["passes"] and "below the required" in small["reasons"][0]
    cleaner = evaluate_variant(_rows(0.80, errors=0), baseline, policy=policy)  # word error falls: score rises
    assert cleaner["passes"] and cleaner["score"] > 0.09


def test_select_decoding_combines_the_best_passing_change_per_knob():
    base = {"temperature": 0.8, "inference_cfg_rate": 0.7, "num_beams": 3}
    variants = [
        {"kind": "single", "knob": "temperature", "value": 0.6, "score": 0.02, "passes": True},
        {"kind": "single", "knob": "temperature", "value": 1.0, "score": 0.03, "passes": True},
        {"kind": "single", "knob": "inference_cfg_rate", "value": 0.5, "score": 0.05, "passes": False},
        {"kind": "single", "knob": "num_beams", "value": 1, "score": 0.01, "passes": True},
        {"kind": "combined", "knob": "combined", "value": 0, "score": 0.9, "passes": True},
    ]
    settings, winners = select_decoding(base, variants)
    assert settings == {"temperature": 1.0, "inference_cfg_rate": 0.7, "num_beams": 1}
    assert {item["knob"] for item in winners} == {"temperature", "num_beams"}
    assert select_decoding(base, [])[0] == base


def test_saved_settings_and_markdown(tmp_path):
    run = tmp_path / "loras" / "voice"
    (run / "analysis").mkdir(parents=True)
    gpt = run / "voice.safetensors"
    gpt.write_bytes(b"x")
    assert load_decoding_settings(gpt) is None
    (run / "analysis" / "decoding.json").write_text(json.dumps({
        "accepted": True, "settings": {"temperature": 1.0, "inference_cfg_rate": 0.7, "num_beams": 1}, "score": 0.031}), encoding="utf-8")
    settings = load_decoding_settings(run / "best" / "voice.safetensors")
    assert settings["temperature"] == 1.0 and settings["num_beams"] == 1 and settings["score"] == 0.031
    (run / "analysis" / "decoding.json").write_text(json.dumps({"accepted": False, "settings": {"temperature": 1.0}}), encoding="utf-8")
    assert load_decoding_settings(gpt) is None
    report = {"accepted": True, "base_settings": {"temperature": 0.8, "inference_cfg_rate": 0.7, "num_beams": 3},
              "settings": {"temperature": 1.0, "inference_cfg_rate": 0.7, "num_beams": 1}, "score": 0.031, "reasons": [],
              "variants": [{"kind": "single", "knob": "temperature", "value": 1.0, "similarity": 0.02, "wer_increase": -0.003, "score": 0.032, "passes": True, "chosen": False},
                           {"kind": "combined", "knob": "combined", "value": 0, "similarity": 0.025, "wer_increase": -0.0015, "score": 0.031, "passes": True, "chosen": True}]}
    text = decoding_markdown(report)
    assert text.startswith("**Decoding sweep adopted.**") and "| combined winners (chosen) | +0.0250 | -0.15 points | +0.0310 | yes |" in text
    report.update(accepted=False, reasons=["no single or combined change beat the defaults by the required margin"])
    assert decoding_markdown(report).startswith("**Decoding sweep kept the defaults.**")
