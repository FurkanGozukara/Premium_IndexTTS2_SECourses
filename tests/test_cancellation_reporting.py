"""A user cancellation is an expected outcome: it must not be logged as a crash."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ui import generation_tab
from ui.common import GenerationCanceled, LazyEngine, is_cancellation


def test_generation_canceled_is_a_runtime_error_recognized_as_cancellation():
    assert issubclass(GenerationCanceled, RuntimeError)
    assert is_cancellation(GenerationCanceled("Generation canceled by user"))
    assert is_cancellation(RuntimeError("Generation canceled by user"))
    assert not is_cancellation(ValueError("boom"))


def test_lazy_engine_raises_the_dedicated_cancellation_type():
    engine = LazyEngine()
    engine.reset_cancel(task_id="task")
    engine.raise_if_canceled()
    assert engine.request_cancel(expected_task="task")
    with pytest.raises(GenerationCanceled, match="canceled by user"):
        engine.raise_if_canceled()


def _request(tmp_path: Path) -> dict:
    task_folder = tmp_path / "task"
    task_folder.mkdir()
    metadata_path = task_folder / "metadata.json"
    metadata_path.write_text(json.dumps({"status": "running"}), encoding="utf-8")
    return {
        "runtime": {},
        "task_layout": {"task_folder": str(task_folder), "task_id": "task"},
        "metadata_path": str(metadata_path),
        "progress_file": str(task_folder / "progress.json"),
    }


@pytest.mark.parametrize(
    ("error", "expected_type", "traceback_expected"),
    [
        (GenerationCanceled("Generation canceled by user"), GenerationCanceled, False),
        (ValueError("synthetic failure"), RuntimeError, True),
    ],
)
def test_in_process_stream_prints_a_traceback_only_for_real_failures(
    monkeypatch, tmp_path, capsys, error, expected_type, traceback_expected
):
    monkeypatch.setattr(
        generation_tab,
        "LAZY_ENGINE",
        SimpleNamespace(
            get=lambda *_args, **_kwargs: object(),
            reset_cancel=lambda **_kwargs: None,
            raise_if_canceled=lambda: None,
        ),
    )

    def failing_run(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(generation_tab, "run_generation_request", failing_run)
    monkeypatch.setattr(generation_tab, "recent_outputs", lambda: [])
    request = _request(tmp_path)

    with pytest.raises(expected_type, match=str(error)):
        list(generation_tab._stream_generation_request(request, use_subprocess=False))

    captured = capsys.readouterr()
    console = captured.out + captured.err
    assert ("Traceback (most recent call last)" in console) is traceback_expected
    if not traceback_expected:
        assert "Generation canceled by user" in console
    status = json.loads(Path(request["metadata_path"]).read_text(encoding="utf-8"))["status"]
    assert status == ("canceled" if not traceback_expected else "failed")
