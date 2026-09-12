"""CPU-only dashboard routing; no app, model, GPU query, or worker is started."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ui import training_tab


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def dashboard(tmp_path, monkeypatch):
    root = tmp_path / "voice"
    root.mkdir()
    paths = SimpleNamespace(
        root=root, status=root / "status.json",
        optimizer=root / "analysis" / "decoder_adapter_job" / "status.json",
        gate=root / "analysis" / "speech_evaluation" / "decoder_test" / "test_job" / "progress.json",
        sweep=root / "analysis" / "speech_evaluation" / "decoding_sweep" / "sweep_job" / "progress.json",
        final=root / "analysis" / "speech_evaluation" / "final_test" / "eval_job" / "progress.json",
    )
    # Deliberately plausible stale optimization data must not leak into the gate.
    _write(paths.optimizer, {"phase": "complete", "step": 10500, "total_steps": 20340,
                             "elapsed_s": 1708.475, "eta_s": 0, "it_s": 9.5,
                             "message": "optimizer early stopping at step 10500"})
    monkeypatch.setattr(training_tab, "_training_vram_total", lambda _: 0.0)
    monkeypatch.setattr(training_tab, "gpu_total_gb", lambda *args: pytest.fail("No GPU queries in routing tests"))
    original_panel = training_tab.progress_panel_html
    calls = []

    def panel(payload, *, title):
        calls.append((deepcopy(payload), title))
        return original_panel(payload, title=title)

    monkeypatch.setattr(training_tab, "progress_panel_html", panel)
    paths.calls = calls
    return paths


def _status(dashboard, *, phase="adapting_decoder", gate="running", message="Generating speech at strength 1"):
    value = {"phase": phase, "decoder_adapter_status": "running", "step": 14000, "total_steps": 20340,
             "elapsed_s": 5000, "eta_s": 0, "message": message}
    if gate is not None:
        value["decoder_test_status"] = gate
    _write(dashboard.status, value)


def _poll(dashboard):
    updates = training_tab.training_status_updates(str(dashboard.root), 0)
    payload, title = dashboard.calls[-1]
    return payload, title, updates


@pytest.mark.parametrize("gate_status", ["running", " Running "])
def test_running_gate_uses_its_live_file_not_completed_optimization(dashboard, gate_status):
    _status(dashboard, gate=gate_status)
    progress = {"completed": 31, "total": 39, "fraction": 31 / 39,
                "elapsed_s": 472.699, "eta_s": 121.987, "speed": 0.9386, "speed_unit": "x RT",
                "vram_used_gb": 7.9633, "vram_total_gb": 31.8159, "desc": "current gate cell 31"}
    _write(dashboard.gate, progress)
    original = dashboard.gate.read_bytes()
    payload, title, updates = _poll(dashboard)
    assert payload == progress
    assert title == "Validating the voice decoder through the full pipeline"
    assert "31/39" in updates[0] and "10500/20340" not in updates[0]
    assert "optimizer early stopping" not in updates[0]
    assert "Generating speech at strength 1" in updates[1]
    # The Gradio timer stays at five seconds while a run is live; the browser's own one-second poller
    # (LIVE_TRAINING_JS) carries the panel, status line and log in between.
    assert updates[-1].value == 5.0 and training_tab._state_running(dashboard.root)
    assert dashboard.gate.read_bytes() == original


@pytest.mark.parametrize("gate_status", [None, "complete", "failed", "skipped"])
def test_normal_optimization_ignores_old_gate_progress(dashboard, gate_status):
    _status(dashboard, gate=gate_status)
    _write(dashboard.gate, {"completed": 39, "total": 39, "fraction": 1, "desc": "old gate"})
    _write(dashboard.optimizer, {"phase": "training", "step": 500, "total_steps": 20340,
                                "elapsed_s": 82, "eta_s": 600, "it_s": 7, "message": "optimizer step 500"})
    payload, title, updates = _poll(dashboard)
    assert payload["completed"] == 500 and payload["total"] == 20340
    assert payload["desc"] == "optimizer step 500" and payload["elapsed_s"] == 82
    assert payload["eta_s"] == 600 and payload["speed_unit"] == "it/s"
    assert title == "Adapting the voice decoder" and "old gate" not in updates[0]


@pytest.mark.parametrize("raw", [None, b'{"unfinished":', b'[]', b'null', b'"wrong shape"', b'{}'])
def test_missing_or_corrupt_gate_never_falls_back_to_optimizer(dashboard, raw):
    _status(dashboard)
    if raw is not None:
        dashboard.gate.parent.mkdir(parents=True)
        dashboard.gate.write_bytes(raw)
    payload, title, updates = _poll(dashboard)
    assert payload["fraction"] == 0 and payload["completed"] == 0 and payload["total"] is None
    assert payload["desc"] == "Generating speech at strength 1"
    assert payload.get("elapsed_s") is None and payload.get("eta_s") is None and payload.get("speed") is None
    assert "Validating" in title and "10500" not in updates[0] and "optimizer" not in updates[0]


@pytest.mark.parametrize("field,value", [
    ("fraction", "bad"), ("fraction", float("nan")), ("fraction", float("inf")), ("fraction", 2),
    ("completed", []), ("completed", 1.5), ("total", -1), ("total", True),
    ("elapsed_s", "bad"), ("eta_s", float("nan")), ("speed", {}), ("vram_used_gb", "broken"),
])
def test_invalid_gate_numeric_payload_falls_back_safely(dashboard, field, value):
    _status(dashboard)
    progress = {"fraction": 0.5, "completed": 2, "total": 4, "desc": "invalid cell"}
    progress[field] = value
    _write(dashboard.gate, progress)
    payload, title, updates = _poll(dashboard)
    assert payload == {"fraction": 0.0, "completed": 0, "total": None, "desc": "Generating speech at strength 1"}
    assert "Validating" in title and "10500" not in updates[0]


def test_gate_measurement_update_replaces_grid_metrics_without_carryover(dashboard):
    _status(dashboard)
    _write(dashboard.gate, {"completed": 39, "total": 39, "fraction": 1.0,
                            "elapsed_s": 590, "eta_s": 0, "speed": 0.94, "speed_unit": "x RT",
                            "vram_used_gb": 8, "desc": "rendered grid"})
    first, _, _ = _poll(dashboard)
    _write(dashboard.gate, {"completed": 3, "total": 39, "fraction": 3 / 39,
                            "elapsed_s": 610, "desc": "measuring generated clips"})
    second, title, _ = _poll(dashboard)
    assert first["completed"] == 39 and second["completed"] == 3
    assert second["desc"] == "measuring generated clips" and second["elapsed_s"] == 610
    assert "speed" not in second and "eta_s" not in second and "vram_used_gb" not in second
    assert "Validating" in title


@pytest.mark.parametrize("phase,path,title", [
    ("calibrating_decoding", "sweep", "Sweeping decoding settings"),
    ("evaluating_final_test", "final", "Assessing frozen deployment on final test"),
])
def test_sweep_and_final_keep_their_own_routing_despite_other_stale_files(dashboard, phase, path, title):
    _status(dashboard, phase=phase)  # stale gate-running flag must not override the actual phase.
    _write(dashboard.gate, {"fraction": 1, "completed": 39, "total": 39, "desc": "old gate"})
    _write(dashboard.sweep, {"fraction": 0.1, "completed": 1, "total": 10, "desc": "sweep progress"})
    _write(dashboard.final, {"fraction": 0.25, "completed": 2, "total": 8, "desc": "final progress"})
    expected = json.loads(getattr(dashboard, path).read_text())
    payload, actual_title, updates = _poll(dashboard)
    assert payload == expected and actual_title == title
    assert "old gate" not in updates[0] and "10500" not in updates[0]


@pytest.mark.parametrize("phase", ["calibrating_decoding", "evaluating_final_test"])
def test_missing_sweep_or_final_progress_remains_unknown(dashboard, phase):
    _status(dashboard, phase=phase, message="phase starting")
    _write(dashboard.gate, {"fraction": 1, "completed": 39, "total": 39, "desc": "old gate"})
    payload, title, updates = _poll(dashboard)
    assert payload["fraction"] == 0 and payload["completed"] == 0 and payload["total"] is None
    assert payload["desc"] == "phase starting" and "Validating" not in title
    assert "10500" not in updates[0] and "old gate" not in updates[0]


def test_post_training_transition_does_not_show_an_old_gate_or_optimizer(dashboard):
    _status(dashboard, phase="post_training", message="preparing next check")
    _write(dashboard.gate, {"fraction": 1, "completed": 39, "total": 39, "desc": "finished gate"})
    payload, title, updates = _poll(dashboard)
    assert payload == {"fraction": 0.0, "completed": 0, "total": None, "desc": "preparing next check"}
    assert title == "Automatic quality checks in progress" and "finished gate" not in updates[0]
