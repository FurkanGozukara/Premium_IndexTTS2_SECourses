"""The voice summary must not equate no adopted override with no sweep run."""

import json

import pytest

import ui.generation_tab as generation_tab


@pytest.fixture
def summary_checkpoint(tmp_path, monkeypatch):
    run = tmp_path / "voice"
    (run / "analysis").mkdir(parents=True)
    checkpoint = run / "voice.safetensors"
    checkpoint.write_bytes(b"test placeholder; metadata inspection is mocked")
    monkeypatch.setattr(generation_tab, "inspect_lora", lambda _: {
        "adapter_type": "dora", "rank": 128, "alpha": 129, "targets": [],
    })
    monkeypatch.setattr(generation_tab, "load_speaking_rate", lambda _: None)
    monkeypatch.setattr(generation_tab, "find_decoder_adapter", lambda _: None)
    monkeypatch.setattr(generation_tab, "_resolve_lora_reference_path", lambda *_: None)
    return checkpoint


@pytest.mark.parametrize("report", [
    None,
    {"accepted": False, "settings": {"temperature": 0.8, "inference_cfg_rate": 0.7, "num_beams": 3}},
    {"accepted": True, "settings": {}},
])
def test_summary_without_override_does_not_claim_no_sweep(summary_checkpoint, report):
    if report is not None:
        (summary_checkpoint.parent / "analysis" / "decoding.json").write_text(
            json.dumps(report), encoding="utf-8",
        )
    text, reference = generation_tab._lora_info(str(summary_checkpoint))
    assert "Decoding settings: **defaults** (no accepted sweep override for this training)." in text
    assert "no sweep result" not in text
    assert "inspection failed" not in text
    assert reference is None


def test_summary_still_reports_an_accepted_override(summary_checkpoint):
    (summary_checkpoint.parent / "analysis" / "decoding.json").write_text(
        json.dumps({"accepted": True, "settings": {
            "temperature": 0.6, "inference_cfg_rate": 1.0, "num_beams": 5,
        }}), encoding="utf-8",
    )
    text, _ = generation_tab._lora_info(str(summary_checkpoint))
    assert "Decoding settings from the sweep: temperature **0.6**, guidance **1**, beams **5**" in text
    assert "no accepted sweep override" not in text
