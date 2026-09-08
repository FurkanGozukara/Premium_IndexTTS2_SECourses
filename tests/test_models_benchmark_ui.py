"""CPU-only coverage of benchmark lifecycle and preset event wiring."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import gradio as gr
import pytest

from ui import models_tab as models
from ui.common import ChildJob
from ui.presets_store import PresetRegistry


class FakeProcess:
    pid = 12345

    def __init__(self, returncode=None):
        self.returncode = returncode

    def poll(self):
        return self.returncode


def _job(tmp_path, name="benchmark", returncode=None):
    state = tmp_path / name
    state.mkdir()
    log = state / "benchmark.log"
    log.write_text("kept diagnostic detail\n", encoding="utf-8")
    return ChildJob("vram_benchmark", FakeProcess(returncode), state, log,
                    metadata={"idle_timeout_s": 17})


@pytest.mark.parametrize("selected,visible,expected", [
    ("auto", None, None),
    ("cuda:2", None, {"CUDA_VISIBLE_DEVICES": "2"}),
    ("cuda:1", "2,3", {"CUDA_VISIBLE_DEVICES": "3"}),
    ("cuda:0", "GPU-uuid", {"CUDA_VISIBLE_DEVICES": "GPU-uuid"}),
])
def test_benchmark_environment_preserves_selected_device(monkeypatch, selected, visible, expected):
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    assert models._benchmark_environment(selected) == expected


@pytest.mark.parametrize("selected,visible", [
    ("cpu", "0"), ("auto", ""), ("auto", "-1"), ("cuda:1", "2"), ("cuda", "0"),
])
def test_benchmark_environment_rejects_unavailable_selection(monkeypatch, selected, visible):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    with pytest.raises(ValueError):
        models._benchmark_environment(selected)


def test_cancel_only_captured_benchmark_and_preserves_log(monkeypatch, tmp_path):
    job = _job(tmp_path)
    manager = SimpleNamespace(get=Mock(return_value=job))
    monkeypatch.setattr(models, "PROCESS_MANAGER", manager)

    def stop(process):
        assert process is job.process
        process.returncode = 1
        return True

    terminate = Mock(side_effect=stop)
    monkeypatch.setattr(models, "_terminate_process_tree", terminate)
    output = models._cancel_benchmark(str(job.state_dir))
    terminate.assert_called_once_with(job.process)
    manager.get.assert_called_once_with("vram_benchmark")
    assert "Benchmark canceled." in output[0]
    assert "kept diagnostic detail" in output[0]
    assert output[2]["interactive"] is True
    assert output[3]["interactive"] is False
    status = json.loads((job.state_dir / "status.json").read_text())
    assert status["status"] == "cancelled"
    assert status["log_path"] == str(job.log_path)


@pytest.mark.parametrize("state_name", ["old", ""])
def test_cancel_rejects_stale_displayed_job(monkeypatch, tmp_path, state_name):
    job = _job(tmp_path, "new")
    monkeypatch.setattr(models, "PROCESS_MANAGER", SimpleNamespace(get=Mock(return_value=job)))
    terminate = Mock()
    monkeypatch.setattr(models, "_terminate_process_tree", terminate)
    output = models._cancel_benchmark(str(tmp_path / state_name) if state_name else "")
    terminate.assert_not_called()
    assert "Review this job" in output[0]
    assert output[1] == str(job.state_dir)
    assert job.canceled is False


def test_cancel_pending_can_be_retried(monkeypatch, tmp_path):
    job = _job(tmp_path)
    monkeypatch.setattr(models, "PROCESS_MANAGER", SimpleNamespace(get=Mock(return_value=job)))
    monkeypatch.setattr(models, "_terminate_process_tree", lambda _: False)
    output = models._cancel_benchmark(str(job.state_dir))
    assert "retry Cancel benchmark" in output[0]
    assert "waiting for its process tree" in output[0]
    assert output[2]["interactive"] is False
    assert output[3]["interactive"] is True
    assert json.loads((job.state_dir / "status.json").read_text())["status"] == "cancelling"


@pytest.mark.parametrize("returncode,message,status", [
    (0, "Benchmark completed.", "complete"),
    (2, "Benchmark failed with exit code 2.", "failed"),
])
def test_finished_benchmark_is_not_marked_canceled(monkeypatch, tmp_path, returncode, message, status):
    job = _job(tmp_path, returncode=returncode)
    monkeypatch.setattr(models, "PROCESS_MANAGER", SimpleNamespace(get=Mock(return_value=job)))
    terminate = Mock()
    monkeypatch.setattr(models, "_terminate_process_tree", terminate)
    output = models._cancel_benchmark(str(job.state_dir))
    terminate.assert_not_called()
    assert message in output[0]
    assert job.canceled is False
    assert json.loads((job.state_dir / "status.json").read_text())["status"] == status


def test_reload_adopts_managed_benchmark_state(monkeypatch, tmp_path):
    job = _job(tmp_path)
    monkeypatch.setattr(models, "PROCESS_MANAGER", SimpleNamespace(get=Mock(return_value=job)))
    output = models._refresh_benchmark()
    assert output[1] == str(job.state_dir)
    assert "Maximum GPU idle wait: 17s" in output[0]
    assert output[2]["interactive"] is False
    assert output[3]["interactive"] is True


@pytest.fixture
def models_demo(monkeypatch, tmp_path):
    monkeypatch.setattr(models, "list_gpus", lambda: [])
    monkeypatch.setattr(models, "_gpu_total", lambda _: 16)
    monkeypatch.setattr(models, "_gpu_free", lambda _: 12)
    monkeypatch.setattr(models, "ROOT", tmp_path)
    with gr.Blocks() as demo:
        tab = models.build_models_tab(SimpleNamespace(model_dir=tmp_path, device="cpu"), PresetRegistry())
    return demo, tab


def test_tier_only_responds_to_user_input_and_cancel_does_not_queue(models_demo):
    demo, tab = models_demo
    tier = next(fn for fn in demo.fns.values() if getattr(fn.fn, "__name__", "") == "apply_tier")
    assert tier.targets == [(tab.tier._id, "input")]
    for name in ("_cancel_benchmark", "_refresh_benchmark"):
        callback = next(fn for fn in demo.fns.values() if getattr(fn.fn, "__name__", "") == name)
        assert callback.queue is False
        assert len(callback.outputs) == 4


def test_ui_benchmark_returns_immediately_with_idle_limit_and_selected_gpu(monkeypatch, tmp_path, models_demo):
    demo, _ = models_demo
    job = _job(tmp_path)
    manager = SimpleNamespace(get=Mock(return_value=None), start=Mock(return_value=job))
    monkeypatch.setattr(models, "PROCESS_MANAGER", manager)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    callback = next(fn for fn in demo.fns.values() if getattr(fn.fn, "__name__", "") == "benchmark")
    result = callback.fn("16", "cuda:1", True, True, 7)
    assert len(result) == len(callback.outputs) == 4
    call = manager.start.call_args
    command = call.args[1]
    assert command[command.index("--idle-timeout") + 1] == "7.0"
    assert "--emulate" in command and "--subtitle" in command
    assert call.kwargs["env"] == {"CUDA_VISIBLE_DEVICES": "3"}
    assert call.kwargs["metadata"] == {"idle_timeout_s": 7.0}


def test_ui_benchmark_does_not_start_duplicate(monkeypatch, tmp_path, models_demo):
    demo, _ = models_demo
    job = _job(tmp_path)
    manager = SimpleNamespace(get=Mock(return_value=job), start=Mock())
    monkeypatch.setattr(models, "PROCESS_MANAGER", manager)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    callback = next(fn for fn in demo.fns.values() if getattr(fn.fn, "__name__", "") == "benchmark")
    result = callback.fn("16", "auto", False, False, 7)
    manager.start.assert_not_called()
    assert "already running" in result[0]
