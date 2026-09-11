"""Voice LoRA / DoRA panel, automatic token budget, and the Gradio bounds guard."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from indextts.training.dataset_manifest import write_manifest
from indextts.training.speaking_rate import (
    SpeakingRateReport,
    ensure_calibration_fields,
    load_speaking_rate,
    original_calibration,
    save_manual_speaking_rate,
    write_speaking_rate,
)
from ui import generation_tab
from ui.common import clamp_to_bounds, install_gradio_bounds_guard


def _rows(count: int = 30) -> list[dict]:
    rows = []
    for index in range(count):
        words = 24 + index
        rows.append(
            {
                "id": f"clip_{index:03d}",
                "audio": f"segments/clip_{index:03d}.wav",
                "text": " ".join(["word"] * (words - 1)) + " end.",
                "duration_s": round(words / 2.6, 3),
                "words": words,
                "language": "EN",
                "split": "train",
            }
        )
    return rows


def _adapter(tmp_path: Path) -> tuple[Path, Path]:
    dataset = tmp_path / "datasets" / "voice_dataset"
    dataset.mkdir(parents=True)
    write_manifest(dataset / "manifest.jsonl", _rows())
    adapter = tmp_path / "loras" / "voice"
    adapter.mkdir(parents=True)
    checkpoint = adapter / "voice.safetensors"
    checkpoint.write_bytes(b"x")
    (adapter / "train_config.json").write_text(json.dumps({"dataset_dir": str(dataset)}), encoding="utf-8")
    return adapter, checkpoint


def test_manual_speaking_rate_keeps_the_original_calibration(tmp_path: Path) -> None:
    adapter, checkpoint = _adapter(tmp_path)
    automatic = SpeakingRateReport(
        recommended_speaking_rate=0.978,
        dataset_words_per_second=2.76,
        generated_words_per_second=2.82,
        clips_used=36,
        method="speech_matched",
        generated_at="2026-09-09T00:00:00+00:00",
        summary="matched",
    )
    write_speaking_rate(adapter, automatic)
    assert original_calibration(checkpoint) == (0.978, "speech_matched")

    manual = save_manual_speaking_rate(checkpoint, 1.0)
    assert manual.method == "manual" and manual.calibrated_speaking_rate == 0.978
    assert manual.calibration_method == "speech_matched"
    assert "0.978" in manual.summary
    again = save_manual_speaking_rate(checkpoint, 1.1)
    assert again.calibrated_speaking_rate == 0.978 and again.original_method() == "speech_matched"
    loaded = load_speaking_rate(checkpoint)
    assert loaded is not None and loaded.original_rate() == 0.978
    assert round(loaded.effective_words_per_second(1.1), 3) == round(2.82 * 1.1, 3)
    assert round(loaded.effective_words_per_second(), 3) == round(2.82 * 1.1, 3)


def test_old_manual_reports_recover_the_estimate_from_their_summary(tmp_path: Path) -> None:
    adapter, checkpoint = _adapter(tmp_path)
    legacy = {
        "recommended_speaking_rate": 1.0,
        "dataset_words_per_second": 2.7,
        "generated_words_per_second": 2.8,
        "clips_used": 7,
        "method": "manual",
        "generated_at": "2026-09-10T00:00:00+00:00",
        "summary": "Speaking rate 1.000 was set manually on 2026-09-10. Replaced the automatic estimate 0.964 (training samples).",
    }
    (adapter / "analysis").mkdir()
    (adapter / "analysis" / "speaking_rate.json").write_text(json.dumps(legacy), encoding="utf-8")
    report = load_speaking_rate(checkpoint)
    assert report is not None and report.calibrated_speaking_rate is None and report.original_rate() is None
    assert original_calibration(checkpoint, report) == (0.964, "training_samples")
    # The recovered estimate is written back once, so later reads need no recovery.
    filled = ensure_calibration_fields(checkpoint, report)
    assert filled is not None and filled.calibrated_speaking_rate == 0.964 and filled.calibration_method == "training_samples"
    reloaded = load_speaking_rate(checkpoint)
    assert reloaded is not None and reloaded.original_rate() == 0.964 and reloaded.recommended_speaking_rate == 1.0
    assert ensure_calibration_fields(checkpoint) == reloaded


def test_adapter_panel_reports_rate_lines_and_auto_tokens(tmp_path: Path, monkeypatch) -> None:
    adapter, checkpoint = _adapter(tmp_path)
    write_speaking_rate(
        adapter,
        SpeakingRateReport(0.95, 2.6, 2.8, 12, "speech_matched", "2026-09-09T00:00:00+00:00", "matched"),
    )
    monkeypatch.setattr(
        generation_tab,
        "inspect_lora",
        lambda _path: {
            "adapter_type": "dora", "rank": 128, "alpha": 129.0, "targets": ["a", "b"], "steps": 1500,
            "dataset": "voice_dataset", "date": "2026-09-10T00:00:00", "size_mb": 12.5, "recommended_reference": "",
            "train_config": {"dataset_dir": str(tmp_path / "datasets" / "voice_dataset")},
        },
    )
    monkeypatch.setattr(generation_tab, "find_decoder_adapter", lambda _path: "")
    monkeypatch.setattr(generation_tab, "load_decoding_settings", lambda _path: None)
    monkeypatch.setattr(generation_tab, "_token_len_for_profiles", lambda: (lambda text: len(text.split()) + 1))
    generation_tab._PROFILE_CACHE.clear()

    html, reference = generation_tab._lora_info(
        str(checkpoint), speaking_rate=1.0, max_tokens=60, budget_scale=0.72, language="EN", auto_tokens=True
    )
    assert reference is None
    assert "Calibrated (original)" in html and "0.950" in html
    assert "Speaking rate slider now" in html and "1.000" in html
    assert f"{2.8:.2f} words/s" in html  # the slider pace: generated 2.8 words/s at rate 1.0
    assert "Words per generated line" in html and "Never exceed" in html
    assert "Per sentence inside a line" in html
    assert "Max tokens per segment" in html and "applied automatically" in html
    assert "No dataset statistics" not in html

    update = generation_tab.auto_max_tokens_update(str(checkpoint), True, 0.72, "EN")
    assert isinstance(update, dict) and update["value"] >= 20
    assert generation_tab.auto_max_tokens_update(str(checkpoint), False, 0.72, "EN") is not update
    assert generation_tab.auto_max_tokens_update("", True, 0.72, "EN") is not update
    # Different slider values change the seconds column but not the word counts.
    slower, _ = generation_tab._lora_info(str(checkpoint), speaking_rate=0.5, max_tokens=60, budget_scale=0.72, language="EN", auto_tokens=False)
    assert f"{1.4:.2f} words/s" in slower and "enable <b>Auto from LoRA / DoRA dataset</b>" in slower
    base, base_reference = generation_tab._lora_info("")
    assert "No LoRA / DoRA selected" in base and base_reference is None


def test_adapter_panel_without_dataset_explains_missing_statistics(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "orphan.safetensors"
    checkpoint.write_bytes(b"x")
    monkeypatch.setattr(
        generation_tab,
        "inspect_lora",
        lambda _path: {"adapter_type": "lora", "rank": 16, "alpha": 32.0, "targets": [], "steps": 10, "dataset": "", "date": "", "size_mb": 1.0, "recommended_reference": "", "train_config": {}},
    )
    monkeypatch.setattr(generation_tab, "find_decoder_adapter", lambda _path: "")
    monkeypatch.setattr(generation_tab, "load_decoding_settings", lambda _path: None)
    generation_tab._PROFILE_CACHE.clear()
    html, _ = generation_tab._lora_info(str(checkpoint), speaking_rate=1.0)
    assert "No dataset statistics" in html and "No calibrated speaking rate yet" in html
    assert generation_tab.auto_max_tokens_update(str(checkpoint), True, 0.72, "EN") is not None


def test_pronunciation_dictionary_helpers_use_the_voice_vocabulary(tmp_path: Path, monkeypatch) -> None:
    adapter, checkpoint = _adapter(tmp_path)
    monkeypatch.setattr(generation_tab, "pronunciation_dictionary_path", lambda: tmp_path / "pron" / "dictionary.json")
    monkeypatch.setattr(generation_tab, "_token_len_for_profiles", lambda: (lambda text: len(text.split()) + 1))
    generation_tab._PROFILE_CACHE.clear()
    generation_tab._DICTIONARY_CACHE.clear()
    entries = generation_tab.pronunciation_entries()
    assert (tmp_path / "pron" / "dictionary.json").is_file() and entries
    rows, status = generation_tab.check_unknown_words("Qwen word end ComfyPod Zorbulax", str(checkpoint), generation_tab.dictionary_rows(entries))
    assert [row[0] for row in rows] == ["ComfyPod", "Zorbulax"] and "2 word" in status
    table, message = generation_tab.add_suggestions_to_dictionary(rows, generation_tab.dictionary_rows(entries))
    assert any(row[0] == "ComfyPod" for row in table) and not any(row[0] == "Zorbulax" for row in table)
    assert "Added 1" in message and "Left out 1" in message
    saved_rows, saved_message = generation_tab.add_suggestions_and_save(rows, generation_tab.dictionary_rows(entries))
    assert any(row[0] == "ComfyPod" for row in saved_rows) and "Saved" in saved_message
    assert generation_tab.save_dictionary_rows(saved_rows)[0] == saved_rows
    reloaded, _ = generation_tab.reload_dictionary_rows()
    assert reloaded == saved_rows
    text = generation_tab.apply_pronunciation_dictionary("Qwen says word end", "")
    assert text.startswith("<Qwen|")
    assert generation_tab.check_unknown_words("", str(checkpoint), None)[0] == []


def test_prepare_generation_request_applies_the_dictionary(tmp_path: Path, monkeypatch) -> None:
    prompt = tmp_path / "reference.wav"
    prompt.write_bytes(b"RIFF")
    monkeypatch.setattr(generation_tab, "pronunciation_dictionary_path", lambda: tmp_path / "pron" / "dictionary.json")
    generation_tab._DICTIONARY_CACHE.clear()
    values = dict(generation_tab.GENERATION_DEFAULTS)
    values["runtime.lora_path"] = ""
    request = generation_tab.prepare_generation_request(
        values, prompt=str(prompt), text="Qwen image edit.", subtitle_file=None, image_path=None,
        emotion_audio=None, model_dir="models", output_root=str(tmp_path / "outputs"),
    )
    assert request["text"].startswith("<Qwen|K W EH1 N>")
    values["generation.apply_pronunciation_dictionary"] = False
    plain = generation_tab.prepare_generation_request(
        values, prompt=str(prompt), text="Qwen image edit.", subtitle_file=None, image_path=None,
        emotion_audio=None, model_dir="models", output_root=str(tmp_path / "outputs"),
    )
    assert plain["text"] == "Qwen image edit."


def test_bounds_guard_clamps_typed_values_instead_of_raising(capsys) -> None:
    install_gradio_bounds_guard()
    install_gradio_bounds_guard()
    slider = gr.Slider(20, 300, value=60, step=1, label="Max tokens per segment")
    number = gr.Number(value=None, minimum=20, maximum=500, label="Low cut (Hz)")
    assert slider.preprocess(4) == 20 and slider.preprocess(999) == 300 and slider.preprocess(45) == 45
    assert number.preprocess(4) == 20 and number.preprocess(None) is None and number.preprocess(100) == 100
    assert clamp_to_bounds(5, None, 3) == 3 and clamp_to_bounds(None, 0, 1) is None
    captured = capsys.readouterr().out
    assert "Max tokens per segment" in captured and "clamped" in captured
