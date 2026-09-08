"""CPU-only lifecycle checks; no worker/model/GPU is launched by these tests."""

import ast
import inspect
import json
import textwrap
from pathlib import Path
from unittest.mock import Mock

import pytest

from indextts.training import decoder_adapter, speech_eval, trainer as trainer_module
from indextts.training.train_config import TrainConfig
from indextts.training.trainer import LoraTrainer


class FakeProcess:
    stdout = None

    def __init__(self, returncode=0):
        self.returncode = returncode

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


@pytest.fixture
def trainer(tmp_path):
    item = object.__new__(LoraTrainer)
    item.config = TrainConfig(dataset_dir=str(tmp_path / "dataset"), output_dir=str(tmp_path),
                              name="fresh", device="cpu", final_test_dataset=str(tmp_path / "test"))
    item.adapter_dir = tmp_path / "fresh"
    item.adapter_dir.mkdir()
    item.status_path = item.adapter_dir / "status.json"
    item.stop_path = item.adapter_dir / "stop.flag"
    item.speech_plan_ready = True
    item.log = Mock()
    state = {"speech_evaluation_status": "complete"}

    def status(**updates):
        state.update(updates)
        item.status_path.write_text(json.dumps(state), encoding="utf-8")
        return dict(state)

    item.write_status = status
    item.write_status()
    item.checkpoint = item.adapter_dir / "fresh.safetensors"
    item.checkpoint.write_bytes(b"GPT")
    item.decoder = item.adapter_dir / "fresh.s2mel.safetensors"
    return item


def _install_fake_worker(monkeypatch, trainer, *, returncode=0, report_status="complete"):
    process = FakeProcess(returncode)

    def launch(*_args, **_kwargs):
        trainer.decoder.write_bytes(b"decoder evidence")
        return process

    monkeypatch.setattr(trainer_module.subprocess, "Popen", launch)
    monkeypatch.setattr(decoder_adapter, "load_decoder_report", lambda _: {
        "status": report_status, "accepted": report_status == "complete", "steps": 10,
        "best_val_loss": 1.0, "best_identity": 0.9, "initial_identity": 0.8,
    })
    return process


def _adapt(trainer):
    return trainer._run_decoder_adaptation(terminal_phase="post_training", terminal_message="checks pending",
                                           recommended_checkpoint=str(trainer.checkpoint))


def _state(trainer):
    return json.loads(trainer.status_path.read_text())


def _quarantined(trainer):
    return list((trainer.adapter_dir / "analysis" / "quarantined_decoders").glob("*"))


@pytest.mark.parametrize("gate_error", [False, True])
def test_unverified_or_exceptional_gate_quarantines_decoder(monkeypatch, trainer, gate_error):
    _install_fake_worker(monkeypatch, trainer)
    trainer._run_decoder_test = Mock(side_effect=RuntimeError("gate crashed")) if gate_error else Mock(return_value=None)
    assert _adapt(trainer) == ""
    assert not trainer.decoder.exists()
    assert _quarantined(trainer)[0].read_bytes() == b"decoder evidence"
    assert _state(trainer)["decoder_adapter_status"] == "failed"
    assert _state(trainer)["decoder_adapter_path"] == ""
    assert _state(trainer)["phase"] == "post_training"


def test_disabled_speech_gate_is_skipped_and_not_installed(monkeypatch, trainer):
    _install_fake_worker(monkeypatch, trainer)
    trainer.config.speech_eval_enabled = False
    assert _adapt(trainer) == ""
    assert not trainer.decoder.exists()
    assert _state(trainer)["decoder_test_status"] == "skipped"
    assert _state(trainer)["decoder_adapter_status"] == "skipped"


def test_failed_decoder_worker_quarantines_partial_file(monkeypatch, trainer):
    _install_fake_worker(monkeypatch, trainer, returncode=1)
    trainer._run_decoder_test = Mock()
    assert _adapt(trainer) == ""
    trainer._run_decoder_test.assert_not_called()
    assert not trainer.decoder.exists()
    assert len(_quarantined(trainer)) == 1
    assert _state(trainer)["decoder_adapter_status"] == "failed"


def test_timed_out_decoder_worker_quarantines_partial_file(monkeypatch, trainer):
    process = _install_fake_worker(monkeypatch, trainer, returncode=None)
    trainer.config.decoder_adapter_timeout_s = 1
    monkeypatch.setattr(trainer_module.time, "perf_counter", Mock(side_effect=[0.0, 2.0]))
    terminate = Mock(side_effect=lambda child: setattr(child, "returncode", -9))
    monkeypatch.setattr(trainer_module, "_kill_evaluation_worker", terminate)
    trainer._run_decoder_test = Mock()
    assert _adapt(trainer) == ""
    terminate.assert_called_once_with(process)
    trainer._run_decoder_test.assert_not_called()
    assert not trainer.decoder.exists()
    assert _state(trainer)["decoder_adapter_status"] == "failed"
    assert "timeout" in _state(trainer)["decoder_adapter_message"]


def test_rejected_training_report_cannot_leave_visible_decoder(monkeypatch, trainer):
    _install_fake_worker(monkeypatch, trainer, report_status="rejected")
    assert _adapt(trainer) == ""
    assert not trainer.decoder.exists()
    assert _state(trainer)["decoder_adapter_status"] == "rejected"


@pytest.mark.parametrize("accepted", [False, True])
def test_only_completed_passing_gate_installs_decoder(monkeypatch, trainer, accepted):
    _install_fake_worker(monkeypatch, trainer)
    trainer._run_decoder_test = Mock(return_value={
        "accepted": accepted, "reasons": [] if accepted else ["no speaker gain"],
        "speaker_gain": {"mean": 0.02}, "strength": 0.6, "wer_increase": 0.0,
    })
    result = _adapt(trainer)
    assert bool(result) is accepted
    assert trainer.decoder.exists() is accepted
    assert _state(trainer)["decoder_adapter_status"] == ("complete" if accepted else "rejected")


def test_unexpected_adaptation_exception_also_quarantines_partial_file(trainer):
    def crash(**_kwargs):
        trainer.decoder.write_bytes(b"partial decoder")
        raise RuntimeError("worker launch failed")

    trainer._run_decoder_adaptation = crash
    trainer._run_guarded_decoder_adaptation(terminal_phase="post_training", terminal_message="pending",
                                           recommended_checkpoint=str(trainer.checkpoint))
    assert not trainer.decoder.exists()
    assert _quarantined(trainer)[0].read_bytes() == b"partial decoder"
    assert _state(trainer)["decoder_adapter_status"] == "failed"


def test_quarantine_preserves_previous_evidence(trainer):
    trainer.decoder.write_bytes(b"first")
    trainer._quarantine_decoder(trainer.decoder, "first gate failed")
    trainer.decoder.write_bytes(b"second")
    trainer._quarantine_decoder(trainer.decoder, "second gate failed")
    assert {path.read_bytes() for path in _quarantined(trainer)} == {b"first", b"second"}


def test_quarantine_preserves_evidence_when_timestamps_collide(monkeypatch, trainer):
    monkeypatch.setattr(trainer_module.time, "time_ns", lambda: 123456789)
    quarantine_dir = trainer.adapter_dir / "analysis" / "quarantined_decoders"
    quarantine_dir.mkdir(parents=True)
    previous = quarantine_dir / f"{trainer.decoder.name}.123456789.failed"
    previous.write_bytes(b"previous evidence")

    for payload in (b"first", b"second"):
        trainer.decoder.write_bytes(payload)
        trainer._quarantine_decoder(trainer.decoder, "gate failed")

    preserved = _quarantined(trainer)
    assert len(preserved) == 3
    assert {path.read_bytes() for path in preserved} == {b"previous evidence", b"first", b"second"}
    assert previous.read_bytes() == b"previous evidence"
    assert not trainer.decoder.exists()
    assert _state(trainer)["decoder_adapter_path"] == ""


def test_quarantine_failed_move_keeps_source_and_removes_empty_reservation(monkeypatch, trainer):
    trainer.decoder.write_bytes(b"decoder evidence")
    original_replace = Path.replace

    def fail_decoder_move(path, destination):
        if path == trainer.decoder:
            raise PermissionError("decoder is locked")
        return original_replace(path, destination)

    monkeypatch.setattr(Path, "replace", fail_decoder_move)
    with pytest.raises(PermissionError, match="decoder is locked"):
        trainer._quarantine_decoder(trainer.decoder, "gate failed")

    assert trainer.decoder.read_bytes() == b"decoder evidence"
    assert not _quarantined(trainer)


def test_full_pipeline_child_failure_is_not_a_success(monkeypatch, trainer):
    monkeypatch.setattr(trainer_module.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess(1))
    report = Mock()
    monkeypatch.setattr(speech_eval, "load_decoder_test", report)
    assert trainer._run_decoder_test(trainer.decoder, str(trainer.checkpoint)) is None
    report.assert_not_called()
    assert _state(trainer)["decoder_test_status"] == "failed"


def test_failed_current_speech_phase_does_not_reuse_old_complete_report(monkeypatch, trainer):
    trainer.write_status(speech_evaluation_status="failed")
    monkeypatch.setattr(speech_eval, "load_speech_evaluation", lambda _: {"status": "complete"})
    popen = Mock()
    monkeypatch.setattr(trainer_module.subprocess, "Popen", popen)
    assert trainer._run_decoder_test(trainer.decoder, str(trainer.checkpoint)) is None
    trainer._run_decoding_sweep(terminal_phase="post_training", terminal_message="pending",
                                recommended_checkpoint=str(trainer.checkpoint))
    trainer._run_final_test_assessment(terminal_phase="post_training", terminal_message="pending",
                                       recommended_checkpoint=str(trainer.checkpoint))
    popen.assert_not_called()
    assert _state(trainer)["decoder_test_status"] == "skipped"
    assert _state(trainer)["decoding_sweep_status"] == "skipped"
    assert _state(trainer)["final_test_status"] == "skipped"


def test_final_assessment_child_failure_retains_failure_status(monkeypatch, trainer):
    monkeypatch.setattr(speech_eval, "load_speech_evaluation", lambda _: {"status": "complete"})
    popen = Mock(return_value=FakeProcess(1))
    monkeypatch.setattr(trainer_module.subprocess, "Popen", popen)
    trainer._run_final_test_assessment(terminal_phase="post_training", terminal_message="pending",
                                       recommended_checkpoint=str(trainer.checkpoint))
    assert "--final-test" in popen.call_args.args[0]
    assert _state(trainer)["final_test_status"] == "failed"
    assert _state(trainer)["phase"] == "post_training"


def test_final_assessment_runs_after_all_selection_phases():
    tree = ast.parse(textwrap.dedent(inspect.getsource(LoraTrainer.run)))
    calls = [(node.lineno, node.func.attr, node) for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    order = {name: line for line, name, _ in calls}
    assert order["_run_automatic_speech_evaluation"] < order["_run_guarded_decoder_adaptation"]
    assert order["_run_guarded_decoder_adaptation"] < order["_run_decoding_sweep"] < order["_run_final_test_assessment"]
    for _, name, node in calls:
        if name in {"_run_automatic_speech_evaluation", "_run_guarded_decoder_adaptation",
                    "_run_decoding_sweep", "_run_final_test_assessment"}:
            phase = next(value.value for value in node.keywords if value.arg == "terminal_phase")
            assert isinstance(phase, ast.Name) and phase.id == "post_phase"


def test_final_phase_is_active_and_uses_its_own_progress(monkeypatch, trainer):
    from ui import training_tab
    trainer.write_status(phase="evaluating_final_test", step=100, total_steps=100)
    progress = trainer.adapter_dir / "analysis" / "speech_evaluation" / "final_test" / "eval_job" / "progress.json"
    progress.parent.mkdir(parents=True)
    progress.write_text(json.dumps({"fraction": 0.25, "completed": 1, "total": 4, "desc": "frozen pipeline"}))
    monkeypatch.setattr(training_tab, "_training_vram_total", lambda _: 0)
    panel = Mock(return_value="panel")
    monkeypatch.setattr(training_tab, "progress_panel_html", panel)
    training_tab.training_status_updates(str(trainer.adapter_dir), 0)
    assert training_tab._state_running(trainer.adapter_dir) is True
    assert "final test" in panel.call_args.kwargs["title"]
    assert panel.call_args.args[0]["fraction"] == 0.25
