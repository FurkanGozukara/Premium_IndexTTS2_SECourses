from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import pytest

from indextts.training.dataset_prep import DatasetPrepConfig
from indextts.training import prep_worker
from ui import dataset_tab
from ui.presets_store import PresetRegistry, PresetStore


def test_cue_controls_survive_presets_ui_submission_and_worker_json(tmp_path, monkeypatch) -> None:
    registry = PresetRegistry()
    monkeypatch.setattr(dataset_tab, "DATASET_STATE", tmp_path / "jobs")
    with gr.Blocks() as demo:
        dataset_tab.build_dataset_tab(SimpleNamespace(device="cpu", model_dir="models"), registry)
    defaults = DatasetPrepConfig(name="unused", inputs=["unused"]).to_dict()
    selected = {
        "dataset.cue_fallback_enabled": False,
        "dataset.cue_fallback_max_error": .37,
        "dataset.cue_fallback_check_boundary_words": False,
        "dataset.cue_fallback_margin_ms": 120,
    }
    for key in selected:
        control = registry[key].component
        assert control.visible
        assert control.value == defaults[key.removeprefix("dataset.")]
    store = PresetStore(registry, tmp_path / "presets")
    store.save("my preparation", selected)
    restored = store.load("my preparation")
    assert {key: restored[key] for key in selected} == selected

    saved = {}

    def start(kind, command, **kwargs):
        assert kind == "dataset_prep"
        path = Path(command[command.index("--config") + 1])
        saved["path"] = path
        saved["config"] = json.loads(path.read_text(encoding="utf-8"))
        return SimpleNamespace(running=False, canceled=False, process=SimpleNamespace(returncode=0))

    monkeypatch.setattr(dataset_tab.PROCESS_MANAGER, "start", start)
    monkeypatch.setattr(dataset_tab, "dataset_poll_updates", lambda *args, **kwargs: (gr.skip(),) * 13)
    specs = [spec for spec in registry.specs if spec.key.startswith("dataset.")]
    values = {spec.key: restored[spec.key] for spec in specs}
    values.update({"dataset.inputs": str(tmp_path / "input.wav"), "dataset.output_root": str(tmp_path / "datasets"),
                   "dataset.name": "submitted"})
    event = next(fn for fn in demo.fns.values() if fn.api_name == "prepare_dataset")
    list(event.fn(None, *(values[spec.key] for spec in specs)))
    assert {key: saved["config"][key.removeprefix("dataset.")] for key in selected} == selected

    received = {}

    def run(config, **kwargs):
        config.validate()
        received.update(config.to_dict())
        return SimpleNamespace(status="complete", segment_count=0, total_duration_s=0,
                               output_dir=str(tmp_path / "datasets" / "submitted"))

    monkeypatch.setattr(prep_worker, "run_dataset_prep", run)
    monkeypatch.setattr(prep_worker.WorkerReporter, "_vram", staticmethod(lambda: (0, 0)))
    assert prep_worker.main(["--config", str(saved["path"]), "--state-dir", str(tmp_path / "worker")]) == 0
    assert {key: received[key.removeprefix("dataset.")] for key in selected} == selected


@pytest.mark.parametrize(("field", "value"), [
    ("cue_fallback_max_error", -.01), ("cue_fallback_max_error", 1.01),
    ("cue_fallback_max_error", float("nan")), ("cue_fallback_margin_ms", -1),
    ("cue_fallback_margin_ms", 1001), ("cue_fallback_enabled", "false"),
    ("cue_fallback_check_boundary_words", "true"),
])
def test_invalid_cue_controls_raise_instead_of_silently_changing_values(field, value) -> None:
    config = DatasetPrepConfig(name="invalid", inputs=["audio.wav"])
    setattr(config, field, value)
    with pytest.raises(ValueError, match=field):
        config.validate()
