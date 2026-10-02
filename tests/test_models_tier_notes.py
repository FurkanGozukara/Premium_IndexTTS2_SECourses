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

    # One deferring description for every runtime control (the browser packs the
    # values); it writes only the notes and the estimate, never a runtime value.
    callbacks = [callback for callback in demo.fns.values() if getattr(callback.fn, "__name__", "") == "describe_runtime"]
    assert len(callbacks) == 1
    (callback,) = callbacks
    # Browser-only steps (the latest-only gate, then the packing step) come first;
    # the gate carries the triggers.
    gate = demo.fns[demo.fns[callback.trigger_after].trigger_after]
    assert {(tab.tier._id, "change"), (tab.device._id, "change")} <= {tuple(target) for target in gate.targets}
    assert callback.outputs == [tab.notes, tab.estimate]
    assert callback.queue is False
    describe, inputs, outputs = tab.describe_runtime
    assert outputs == [tab.notes, tab.estimate]
    before = {key: component.value for key, component in tab.controls.items()}
    values = [component.value for component in inputs]
    values[inputs.index(tab.tier)], values[inputs.index(tab.device)] = "auto", "cuda:0"
    assert "32 GB" in describe(*values)[0]
    assert {key: component.value for key, component in tab.controls.items()} == before

    apply_tier = next(callback for callback in demo.fns.values() if getattr(callback.fn, "__name__", "") == "apply_tier")
    assert apply_tier.targets == [(tab.tier._id, "select")]
    assert tab.controls["runtime.model_variant"] in apply_tier.outputs
