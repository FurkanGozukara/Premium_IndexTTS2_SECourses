"""CPU-only cancellation checks with synthetic engines and child processes."""

import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

import ui.common as common
from ui.generation_tab import prepare_generation_request
import webui_generation_runner as runner


def test_cancel_during_cold_load_is_nonblocking_and_survives_engine_creation(monkeypatch):
    lazy = common.LazyEngine()
    entered, release, requested = (threading.Event() for _ in range(3))
    engine = SimpleNamespace(progress_reporter=None)
    errors = []

    def create(*_args, **_kwargs):
        entered.set()
        assert release.wait(5), "test load was not released"
        return engine

    def load():
        try:
            lazy.get({"model_dir": "unused"})
        except Exception as error:
            errors.append(error)

    def cancel():
        assert lazy.request_cancel(expected_task="cold-load")
        requested.set()

    monkeypatch.setattr(runner, "create_tts", create)
    lazy.reset_cancel(task_id="cold-load")
    worker = threading.Thread(target=load)
    worker.start()
    canceler = threading.Thread(target=cancel)
    try:
        assert entered.wait(2)
        canceler.start()
        assert requested.wait(0.5), "cancellation blocked behind model loading"
        assert worker.is_alive()
        release.set()
        worker.join(2)
        assert not worker.is_alive()
        assert len(errors) == 1 and "canceled" in str(errors[0])
        lazy.reset_cancel(task_id="next-run")
        assert lazy.get({"model_dir": "unused"}) is engine
    finally:
        release.set()
        worker.join(2)
        if canceler.ident is not None:
            canceler.join(2)


def test_process_manager_does_not_terminate_a_replaced_job(monkeypatch, tmp_path):
    manager = common.ProcessManager()
    old = common.ChildJob("generation", object(), tmp_path / "old", tmp_path / "old.log")
    new = common.ChildJob("generation", object(), tmp_path / "new", tmp_path / "new.log")
    manager._jobs["generation"] = new
    calls = []
    monkeypatch.setattr(common, "_terminate_process_tree", lambda process: calls.append(process) or True)

    assert manager.terminate("generation", expected_job=old) is False
    assert calls == [] and new.canceled is False
    assert manager.terminate("generation", expected_job=new) is True
    assert calls == [new.process] and new.canceled is True


@pytest.mark.parametrize("returncode,exited,accepted", [(1, False, False), (1, True, True), (0, False, True)])
def test_windows_taskkill_failure_is_not_reported_as_success(monkeypatch, returncode, exited, accepted):
    polls = iter([None, 0 if exited else None])
    process = SimpleNamespace(pid=123456, poll=lambda: next(polls))
    monkeypatch.setattr(common.os, "name", "nt")
    monkeypatch.setattr(common.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=returncode))
    assert common._terminate_process_tree(process) is accepted


@pytest.mark.parametrize("method", ["update", "set_stage", "finish"])
def test_runner_cancellation_hook_survives_candidate_reporter_replacement(tmp_path, method):
    reference = tmp_path / "reference.wav"
    reference.touch()
    request = prepare_generation_request(
        {"generation.num_candidates": 2, "generation.save_as_mp3": False},
        prompt=str(reference), text="Synthetic cancellation regression.", subtitle_file=None,
        image_path=None, emotion_audio=None, model_dir="unused", output_root=tmp_path / "outputs",
    )
    canceled = threading.Event()
    reporters = []
    engine = SimpleNamespace(last_generation_stats={})

    def infer(*_args, output_path, **_kwargs):
        reporters.append(engine.progress_reporter)
        if len(reporters) == 1:
            Path(output_path).write_bytes(b"synthetic first candidate")
            return output_path
        assert reporters[0] is not reporters[1]
        canceled.set()
        callback = getattr(engine.progress_reporter, method)
        callback(*({"update": (0,), "set_stage": ("next stage",), "finish": ()}[method]))
        pytest.fail("canceled reporter should not return")

    def check():
        if canceled.is_set():
            raise RuntimeError("Generation canceled by user")

    engine.infer = infer
    with pytest.raises(RuntimeError, match="canceled"):
        runner.run_generation_request(request, engine, cancellation_check=check)
    assert len(reporters) == 2
    assert json.loads(Path(request["metadata_path"]).read_text(encoding="utf-8"))["status"] == "canceled"


def test_runner_checks_preexisting_cancellation_before_reading_request():
    def check():
        raise RuntimeError("Generation canceled by user")

    with pytest.raises(RuntimeError, match="canceled"):
        runner.run_generation_request({}, object(), cancellation_check=check)
