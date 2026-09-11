from __future__ import annotations

from pathlib import Path

from ui import generation_tab
from ui.generation_tab import (
    GENERATION_DEFAULTS,
    RUNNER_REQUEST_KEYS,
    SEGMENTATION_CHOICES,
    auto_pause_updates,
    build_generation_request,
    codec_has_silence_runs,
    preview_segments,
)


def test_request_carries_the_splitting_and_pause_settings(monkeypatch) -> None:
    assert {"segmentation_mode", "segment_target_tokens", "sentence_pause_ms"} <= RUNNER_REQUEST_KEYS
    assert GENERATION_DEFAULTS["generation.segmentation_mode"] == "smart"
    assert GENERATION_DEFAULTS["generation.sentence_pause_ms"] == 0 and GENERATION_DEFAULTS["generation.auto_lora_pauses"] is True
    assert GENERATION_DEFAULTS["generation.repetition_window"] == 0
    values = dict(GENERATION_DEFAULTS)
    values["generation.sentence_pause_ms"] = 360
    values["generation.repetition_window"] = 16
    request = build_generation_request(values, prompt="ref.wav", text="One. Two.")
    assert request["segmentation_mode"] == "smart" and request["sentence_pause_ms"] == 360
    assert request["segment_target_tokens"] is None  # base model: no dataset target
    assert request["infer_kwargs"]["repetition_window"] == 16
    # A trained voice with a profile supplies the smart target from its median clip.
    monkeypatch.setattr(generation_tab, "adapter_dataset_profile", lambda _path: {"recommendation": {"target_tokens": 44}})
    values["runtime.lora_path"] = "loras/voice/voice.safetensors"
    assert build_generation_request(values, prompt="ref.wav", text="One. Two.")["segment_target_tokens"] == 44
    values["generation.segmentation_mode"] = "Token budget"
    request = build_generation_request(values, prompt="ref.wav", text="One. Two.")
    assert request["segmentation_mode"] == "budget" and request["segment_target_tokens"] is None


def test_auto_pause_updates_follow_the_profile(monkeypatch) -> None:
    profile = {"pauses": {"recommendation": {"sentence_pause_ms": 360, "max_pause_ms": 450}}}
    monkeypatch.setattr(generation_tab, "adapter_dataset_profile", lambda _path: profile)
    sentence, ceiling = auto_pause_updates("loras/voice/voice.safetensors", True)
    assert sentence["value"] == 360 and ceiling["value"] == 450
    skipped = auto_pause_updates("loras/voice/voice.safetensors", False)
    assert all(not isinstance(item, dict) or "value" not in item for item in skipped)
    monkeypatch.setattr(generation_tab, "adapter_dataset_profile", lambda _path: None)
    assert all(not isinstance(item, dict) or "value" not in item for item in auto_pause_updates("x", True))


def test_preview_reports_mode_words_and_seconds(tmp_path: Path) -> None:
    text = "The first sentence is here. The second sentence follows it. A third one ends the text."
    rows, note = preview_segments(text, "EN", 60, segmentation_mode="sentence", words_per_second=2.5, model_dir=str(tmp_path))
    assert [row[1] for row in rows] == ["Text segment"] * 3
    assert "words" in rows[0][3] and "about" in rows[0][3]
    assert "Every sentence" in note and "3 speech section(s)" in note
    rows, note = preview_segments(text, "EN", 60, segmentation_mode="smart", target_tokens=30, model_dir=str(tmp_path))
    assert "Smart sentences" in note and "aiming at 30 tokens" in note
    assert "".join(row[2] for row in rows) == text
    rows, note = preview_segments("Hello [pause:300ms] there.", "EN", 60, model_dir=str(tmp_path))
    assert rows[1][1] == "Pause" and "kept exactly" in rows[1][3] and "Token budget" in note


def test_inert_silence_control_hides_for_the_2_5_codec(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("version: 2.5\n", encoding="utf-8")
    assert codec_has_silence_runs(str(tmp_path)) is False
    other = tmp_path / "old"
    other.mkdir()
    (other / "config.yaml").write_text("version: 2.0\n", encoding="utf-8")
    assert codec_has_silence_runs(str(other)) is True
    assert codec_has_silence_runs(str(tmp_path / "missing")) is True
    assert [value for _label, value in SEGMENTATION_CHOICES] == ["smart", "sentence", "budget"]
