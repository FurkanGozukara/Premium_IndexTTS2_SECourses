"""Programmatic tier restoration updates explanatory text, not runtime values."""

from types import SimpleNamespace

import gradio as gr
import pytest

from ui import models_tab as models
from ui.presets_store import PresetRegistry


@pytest.mark.parametrize("tier,device,expected", [
    ("6", "cuda:0", "notes for 6"),
    ("auto", "cuda:0", "notes for 32"),
    ("auto", "cuda:1", "notes for 8"),
    ("auto", "cpu", "notes for 6"),
    ("custom", "cuda:0", "Custom runtime settings."),
])
def test_tier_notes_follow_selected_tier_and_device(monkeypatch, tier, device, expected):
    monkeypatch.setattr(models, "_gpu_total", lambda device: {"cuda:0": 32, "cuda:1": 8, "cpu": 0}[device])
    monkeypatch.setattr(models, "preset_notes", lambda value: f"notes for {value}")
    assert models._tier_notes(tier, device) == expected


def test_programmatic_tier_and_device_changes_only_refresh_notes(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "list_gpus", lambda: [])
    monkeypatch.setattr(models, "_gpu_total", lambda _device: 32)
    monkeypatch.setattr(models, "_gpu_free", lambda _device: 30)
    registry = PresetRegistry()
    with gr.Blocks() as demo:
        tab = models.build_models_tab(SimpleNamespace(model_dir=tmp_path, device="cpu"), registry)

    callbacks = [callback for callback in demo.fns.values() if callback.fn is models._tier_notes]
    assert len(callbacks) == 2
    assert {tuple(callback.targets[0]) for callback in callbacks} == {
        (tab.tier._id, "change"), (tab.device._id, "change"),
    }
    before = {key: component.value for key, component in tab.controls.items()}
    for callback in callbacks:
        assert callback.inputs == [tab.tier, tab.device]
        assert callback.outputs == [tab.notes]
        assert callback.queue is False
        assert "32 GB" in callback.fn("auto", "cuda:0")
    assert {key: component.value for key, component in tab.controls.items()} == before

    apply_tier = next(callback for callback in demo.fns.values() if getattr(callback.fn, "__name__", "") == "apply_tier")
    assert apply_tier.targets == [(tab.tier._id, "input")]
    assert tab.controls["runtime.model_variant"] in apply_tier.outputs
