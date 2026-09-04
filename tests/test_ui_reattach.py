import time
import json
import os
from pathlib import Path

import pandas as pd

from indextts.training.charts import (
    GRAD_SERIES,
    LOSS_SERIES,
    LR_SERIES,
    SPEED_SERIES,
    empty_series_frame,
    loss_frame,
)
from ui.batch_tab import batch_task_updates
from ui.common import adopt_output_task, latest_output_task, output_task_is_active
from ui.dataset_tab import (
    adopt_dataset_state,
    dataset_poll_updates,
    dataset_summary_line,
    scan_datasets,
)
from ui.generation_tab import generation_task_updates
from ui.grid_tab import adopt_grid_state
from ui.training_tab import adopt_training_state, training_poll_updates


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _set_mtime(path: Path, value: float) -> None:
    os.utime(path, (value, value))


def test_training_discovers_and_adopts_newest_running_state(tmp_path: Path) -> None:
    old = tmp_path / "old_run"
    live = tmp_path / "live_60_steps"
    _write_json(old / "status.json", {"phase": "complete", "step": 10, "total_steps": 10})
    _write_json(
        live / "status.json",
        {"phase": "training", "step": 7, "total_steps": 60, "epoch": 1, "total_epochs": 3},
    )
    _set_mtime(old / "status.json", 100.0)
    _set_mtime(live / "status.json", 200.0)

    adopted, running = adopt_training_state(str(old), root=tmp_path)
    assert Path(adopted) == live.resolve()
    assert running is True

    updates = training_poll_updates(str(old), 0.9, state_root=tmp_path)
    assert Path(updates[0]) == live.resolve()
    assert "Attached to running run live_60_steps" in updates[2]
    assert "step 7/60" in updates[2]
    assert updates[-1].value == 1.0

    _write_json(live / "status.json", {"phase": "stopped", "step": 8, "total_steps": 60})
    idle = training_poll_updates(str(live), 0.9, state_root=tmp_path)
    assert Path(idle[0]) == live.resolve()
    assert idle[-1].value == 5.0


def test_dataset_discovers_running_state_and_uses_matching_dataset(tmp_path: Path) -> None:
    state_root = tmp_path / "states"
    datasets = tmp_path / "datasets"
    old = state_root / "old_prep"
    live = state_root / "live_prep"
    _write_json(old / "config.json", {"output_root": str(datasets), "name": "old_dataset"})
    _write_json(old / "status.json", {"phase": "complete"})
    _write_json(live / "config.json", {"output_root": str(datasets), "name": "ui_prep_live"})
    _write_json(
        live / "status.json",
        {"phase": "segments", "file_i": 2, "file_n": 4, "segment_count": 9},
    )
    _set_mtime(old / "status.json", 100.0)
    _set_mtime(live / "status.json", 200.0)

    state, dataset, running = adopt_dataset_state(
        str(old),
        str(datasets / "old_dataset"),
        root=state_root,
    )
    assert Path(state) == live.resolve()
    assert Path(dataset) == (datasets / "ui_prep_live").resolve()
    assert running is True

    updates = dataset_poll_updates(
        str(old),
        str(datasets / "old_dataset"),
        state_root=state_root,
    )
    assert Path(updates[0]) == live.resolve()
    assert Path(updates[1]) == (datasets / "ui_prep_live").resolve()
    assert "Attached to running run ui_prep_live" in updates[3]
    assert updates[-2].value == 1.0


def _write_task(folder: Path, *, status: str, task_id: str, modified: float) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    _write_json(
        folder / "metadata.json",
        {
            "status": status,
            "task": {"id": task_id, "folder": str(folder.resolve())},
            "settings": {"execution_mode": "subprocess"},
            "outputs": {"final_audio_path": None},
        },
    )
    _write_json(
        folder / "request.json",
        {
            "progress_file": str((folder / "progress.json").resolve()),
            "task_layout": {"task_folder": str(folder.resolve())},
        },
    )
    _write_json(folder / "progress.json", {"fraction": 0.25, "completed": 1, "total": 4, "desc": "step 1"})
    for path in (folder / "metadata.json", folder / "request.json", folder / "progress.json"):
        _set_mtime(path, modified)


def test_generation_and_batch_reload_attach_to_their_newest_tasks(tmp_path: Path) -> None:
    single = tmp_path / "0042"
    batch = tmp_path / "my_batch" / "0007"
    _write_task(single, status="in_progress", task_id="0042", modified=100.0)
    _write_task(batch, status="in_progress", task_id="0007", modified=200.0)

    assert Path(latest_output_task(tmp_path, scope="generation")) == single.resolve()
    assert Path(latest_output_task(tmp_path, scope="batch")) == batch.resolve()
    assert adopt_output_task("", root=tmp_path, scope="generation", page_load=True) == (
        str(single.resolve()),
        True,
    )
    assert output_task_is_active(single)

    generation = generation_task_updates("", output_root=tmp_path, page_load=True)
    assert Path(generation[0]) == single.resolve()
    assert "Attached to running run 0042" in generation[2]
    assert generation[-1].value == 1.0

    batch_updates = batch_task_updates("", output_root=tmp_path, page_load=True)
    assert Path(batch_updates[0]) == batch.resolve()
    assert "Attached to running run 0007" in batch_updates[2]
    assert batch_updates[-1].value == 1.0


def test_idle_timers_do_not_adopt_finished_runs_without_page_load(tmp_path: Path) -> None:
    output = tmp_path / "outputs" / "0042"
    _write_task(output, status="complete", task_id="0042", modified=100.0)
    assert adopt_output_task(
        "", root=tmp_path / "outputs", scope="generation", page_load=False
    ) == ("", False)
    assert adopt_output_task(
        "", root=tmp_path / "outputs", scope="generation", page_load=True
    ) == (str(output.resolve()), False)

    batch_output = tmp_path / "outputs" / "batch" / "0001"
    _write_task(batch_output, status="complete", task_id="0001", modified=110.0)
    batch_updates = batch_task_updates(
        str(batch_output), output_root=tmp_path / "outputs", page_load=False
    )
    assert batch_updates[0] == str(batch_output.resolve())
    assert batch_updates[-1].active is False

    dataset_state = tmp_path / "dataset_states" / "finished"
    _write_json(
        dataset_state / "config.json",
        {"output_root": str(tmp_path / "datasets"), "name": "prepared"},
    )
    _write_json(dataset_state / "status.json", {"phase": "complete"})
    assert adopt_dataset_state(
        "", "", root=tmp_path / "dataset_states", page_load=False
    ) == ("", "", False)
    loaded_dataset = adopt_dataset_state(
        "", "", root=tmp_path / "dataset_states", page_load=True
    )
    assert loaded_dataset[0] == str(dataset_state.resolve())
    assert loaded_dataset[2] is False

    training = tmp_path / "loras" / "finished"
    grid = tmp_path / "grids" / "finished"
    _write_json(training / "status.json", {"phase": "complete"})
    _write_json(grid / "status.json", {"phase": "complete"})
    assert adopt_training_state("", root=tmp_path / "loras", page_load=False) == (
        "",
        False,
    )
    assert adopt_grid_state("", root=tmp_path / "grids", page_load=False) == (
        "",
        False,
    )
    assert adopt_training_state("", root=tmp_path / "loras", page_load=True) == (
        str(training.resolve()),
        False,
    )
    assert adopt_grid_state("", root=tmp_path / "grids", page_load=True) == (
        str(grid.resolve()),
        False,
    )


def test_dataset_wording_and_chart_placeholders_keep_frontend_contract(tmp_path: Path) -> None:
    dataset = tmp_path / "ui_prep_live"
    _write_json(dataset / "dataset_info.json", {"segment_count": 83, "total_duration_minutes": 12.32})
    assert dataset_summary_line(dataset).endswith("features not cached")
    assert scan_datasets(tmp_path)[0][0] == "ui_prep_live | 83 segments | 12.3 min"
    (dataset / "cache").mkdir()
    (dataset / "cache" / "index.jsonl").write_text("", encoding="utf-8")
    assert dataset_summary_line(dataset).endswith("features cached")

    for names in (LOSS_SERIES, LR_SERIES, GRAD_SERIES, SPEED_SERIES):
        placeholder = empty_series_frame(names)
        assert tuple(placeholder["series"]) == names
        assert pd.api.types.is_integer_dtype(placeholder["step"])
        assert pd.api.types.is_float_dtype(placeholder["value"])
    plotted = loss_frame(pd.DataFrame([{"step": 1, "loss": 1.0, "avg_loss": 0.9, "val_loss": 0.8}]))
    assert set(plotted["series"]).issubset(LOSS_SERIES)


def test_training_state_discovery_ignores_evaluation_job_folders(tmp_path: Path) -> None:
    from ui.training_tab import latest_training_state

    adapter = tmp_path / "voice_adapter"
    adapter.mkdir()
    (adapter / "status.json").write_text(json.dumps({"phase": "complete"}), encoding="utf-8")
    job = adapter / "analysis" / "eval_jobs" / "20260902_000000"
    job.mkdir(parents=True)
    (job / "status.json").write_text(json.dumps({"phase": "evaluating"}), encoding="utf-8")
    sample = adapter / "samples" / ".sample_jobs" / "epoch_001"
    sample.mkdir(parents=True)
    (sample / "status.json").write_text(json.dumps({"phase": "generating"}), encoding="utf-8")
    newer = time.time() + 60
    os.utime(job / "status.json", (newer, newer))
    os.utime(sample / "status.json", (newer + 1, newer + 1))

    assert latest_training_state(tmp_path) == str(adapter.resolve())
