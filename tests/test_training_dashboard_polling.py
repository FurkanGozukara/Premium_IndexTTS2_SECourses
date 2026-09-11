"""The training dashboard poller sends a browser tab only what changed, throttles the heavy
components while a run is active, and parses the metrics file incrementally."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import pandas as pd

from ui import training_tab


SKIP = gr.skip()
HEAVY = (3, 4, 5, 6, 8, 9, 10, 11, 12)  # charts, sample, sample label, checkpoints, generalization


def _status(phase: str, step: int, elapsed: float = 12.0) -> dict:
    return {
        "phase": phase, "step": step, "total_steps": 100, "epoch": 1, "total_epochs": 2,
        "elapsed_s": elapsed, "message": phase, "lr": 4e-5,
    }


def _metric_line(step: int) -> str:
    return json.dumps({
        "step": step, "loss": 6.0 - step * 0.01, "avg_loss": 6.0, "lr": 4e-5, "grad_norm": 1.0,
        "it_s": 4.0, "vram_used_gb": 2.0, "mel_accuracy": 0.1,
    }) + "\n"


def _write_run(root: Path, *, phase: str, step: int, rows: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "train_config.json").write_text(json.dumps({"device": "cpu", "name": root.name}), encoding="utf-8")
    (root / "status.json").write_text(json.dumps(_status(phase, step)), encoding="utf-8")
    with (root / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(1, rows + 1):
            handle.write(_metric_line(index))
    (root / "log.txt").write_text("line 1\nline 2\n", encoding="utf-8")


def _append_metrics(root: Path, start: int, count: int, *, partial: str = "") -> None:
    with (root / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        for index in range(start, start + count):
            handle.write(_metric_line(index))
        handle.write(partial)


def _request(session: str) -> SimpleNamespace:
    return SimpleNamespace(session_hash=session)


def setup_function(function) -> None:
    training_tab.reset_dashboard_caches()


def test_finished_run_is_sent_to_each_tab_once(tmp_path: Path) -> None:
    run = tmp_path / "done"
    _write_run(run, phase="complete", step=100, rows=20)

    first = training_tab.training_poll_updates(str(run), 0.9, state_root=tmp_path, request=_request("tab-a"))
    assert len(first) == 14
    assert first[0] == str(run.resolve())
    assert isinstance(first[3], pd.DataFrame) and not first[3].empty

    second = training_tab.training_poll_updates(str(run), 0.9, state_root=tmp_path, request=_request("tab-a"))
    assert second[0] == SKIP
    assert all(second[index] == SKIP for index in HEAVY), "unchanged charts and tables must not be re-sent"
    assert isinstance(second[-1], gr.Timer)

    # A second tab has not seen anything yet and receives the complete dashboard.
    other = training_tab.training_poll_updates(str(run), 0.9, state_root=tmp_path, request=_request("tab-b"))
    assert isinstance(other[3], pd.DataFrame) and not other[3].empty

    # A page reload receives everything again, even though nothing changed.
    reloaded = training_tab.training_poll_updates(
        str(run), 0.9, state_root=tmp_path, page_load=True, request=_request("tab-a"),
    )
    assert isinstance(reloaded[3], pd.DataFrame) and not reloaded[3].empty

    # Opening the training tab (page_load semantics) adopts the newest run even when it is
    # finished and re-sends the dashboard so the hidden-tab charts draw.
    assert training_tab.training_poll_updates(str(run), 0.9, state_root=tmp_path, request=_request("tab-a"))[3] == SKIP
    reopened = training_tab.training_poll_updates(
        "", 0.9, state_root=tmp_path, page_load=True, request=_request("tab-a"),
    )
    assert isinstance(reopened[3], pd.DataFrame) and not reopened[3].empty
    assert reopened[0] == str(run.resolve())


def test_active_run_throttles_heavy_components_but_keeps_status_live(tmp_path: Path, monkeypatch) -> None:
    run = tmp_path / "live"
    _write_run(run, phase="training", step=10, rows=10)
    clock = {"now": 1000.0}
    monkeypatch.setattr(training_tab.time, "monotonic", lambda: clock["now"])
    request = _request("tab-a")

    first = training_tab.training_poll_updates(str(run), 0.9, state_root=tmp_path, request=request)
    assert isinstance(first[3], pd.DataFrame)
    assert "Attached to running run" in first[2]

    # One second later the run advanced: the panel and status line follow, the charts wait.
    _append_metrics(run, 11, 5)
    (run / "status.json").write_text(json.dumps(_status("training", 15, elapsed=13.0)), encoding="utf-8")
    clock["now"] += 1.0
    second = training_tab.training_poll_updates(str(run), 0.9, state_root=tmp_path, request=request)
    assert all(second[index] == SKIP for index in HEAVY)
    assert isinstance(second[1], str) and isinstance(second[2], str)
    assert "step 15/100" in second[2]

    # Changing the smoothing must redraw the loss chart at once.
    clock["now"] += 1.0
    forced = training_tab.training_poll_updates(str(run), 0.5, state_root=tmp_path, request=request, force_heavy=True)
    assert isinstance(forced[3], pd.DataFrame) and int(forced[3]["step"].max()) == 15

    # After the refresh interval the charts follow the new metrics.
    _append_metrics(run, 16, 4)
    (run / "status.json").write_text(json.dumps(_status("training", 19, elapsed=20.0)), encoding="utf-8")
    clock["now"] += training_tab.HEAVY_REFRESH_SECONDS
    third = training_tab.training_poll_updates(str(run), 0.5, state_root=tmp_path, request=request)
    assert isinstance(third[3], pd.DataFrame) and int(third[3]["step"].max()) == 19

    # Unchanged charts are not re-sent even when the heavy path runs again.
    clock["now"] += training_tab.HEAVY_REFRESH_SECONDS
    unchanged = training_tab.training_poll_updates(str(run), 0.5, state_root=tmp_path, request=request)
    assert all(unchanged[index] == SKIP for index in HEAVY)

    # A phase change refreshes changed charts immediately, inside the throttle window.
    _append_metrics(run, 20, 2)
    (run / "status.json").write_text(json.dumps(_status("complete", 21, elapsed=22.0)), encoding="utf-8")
    clock["now"] += 1.0
    fourth = training_tab.training_poll_updates(str(run), 0.5, state_root=tmp_path, request=request)
    assert isinstance(fourth[3], pd.DataFrame) and int(fourth[3]["step"].max()) == 21
    assert isinstance(fourth[1], str) and "Training complete" in fourth[1]
    assert "Attached to running run" not in fourth[2]


def test_direct_callers_still_receive_the_full_dashboard(tmp_path: Path) -> None:
    run = tmp_path / "direct"
    _write_run(run, phase="complete", step=100, rows=5)
    for _ in range(2):
        values = training_tab.training_poll_updates(str(run), 0.9, state_root=tmp_path)
        assert len(values) == 14
        assert isinstance(values[3], pd.DataFrame) and not values[3].empty
        assert values[0] == str(run.resolve())


def test_cached_metrics_parse_only_appended_complete_lines(tmp_path: Path) -> None:
    run = tmp_path / "cache"
    _write_run(run, phase="training", step=3, rows=3)
    assert list(training_tab._cached_metrics(run)["step"]) == [1, 2, 3]

    _append_metrics(run, 4, 2, partial='{"step": 6, "loss": 5.')
    assert list(training_tab._cached_metrics(run)["step"]) == [1, 2, 3, 4, 5]
    assert list(training_tab._cached_metrics(run)["step"]) == [1, 2, 3, 4, 5]  # unchanged size: cached

    with (run / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('9}\n')
    frame = training_tab._cached_metrics(run)
    assert list(frame["step"]) == [1, 2, 3, 4, 5, 6]
    assert list(training_tab.load_metrics(run)["step"]) == list(frame["step"])

    (run / "metrics.jsonl").write_text(_metric_line(1), encoding="utf-8")
    assert list(training_tab._cached_metrics(run)["step"]) == [1]

    (run / "metrics.jsonl").unlink()
    assert training_tab._cached_metrics(run).empty
