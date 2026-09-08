"""Batch workflow regression tests: no model loads, inference, or subprocesses."""

import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

import ui.batch_tab as batch
from ui.generation_tab import EMOTION_MODES, GenerationTab


BATCH_VALUES = {
    "batch.naming_pattern": "{index:03d}_{name}",
    "batch.output_subfolder": "batch",
    "batch.reference_mode": "One reference for all",
    "batch.execution": "Subprocess per item",
    "batch.continue_errors": True,
}


class EventCapture:
    def __init__(self, events):
        self.events = events

    def click(self, fn, inputs=None, outputs=None, **kwargs):
        self.events[kwargs.get("api_name") or fn.__name__] = {
            "fn": fn, "inputs": inputs, "outputs": outputs, **kwargs,
        }
        return self

    success = click


def _bound_events(tmp_path, monkeypatch, generation_values=None):
    events = {}
    event = EventCapture(events)
    generation_values = generation_values or {}
    generation = GenerationTab(
        controls={"runtime.lora_path": "lora", "generation.auto_lora_reference": "auto"},
        prompt_audio="speaker component",
        image="image component",
        emotion_audio="emotion component",
        request_keys=list(generation_values),
        request_components=list(generation_values),
    )
    tab = batch.BatchTab(
        start_button=event, cancel_button=event, status=None, progress=None,
        log=None, results=None, files=None, text=None, folder=None,
        controls=dict(BATCH_VALUES), cancel_confirmation="confirmation component",
        cancel_confirm_button=event, cancel_dismiss_button=event,
    )
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    batch.bind_batch_events(tab, generation, SimpleNamespace(model_dir=str(tmp_path / "models")), None)
    return events


def _run_batch(events, files, reference, *, batch_values=None, generation_values=None, image=None, emotion=None):
    values = {**BATCH_VALUES, **(batch_values or {})}
    return events["generate_batch"]["fn"](
        files, "", "", str(reference), image, emotion,
        *values.values(), *(generation_values or {}).values(),
    )


def _item_request(tmp_path):
    task = tmp_path / "item"
    task.mkdir()
    metadata = task / "metadata.json"
    metadata.write_text(json.dumps({"status": "in_progress"}), encoding="utf-8")
    progress = task / "progress.json"
    progress.write_text(json.dumps({"fraction": 0.2, "desc": "Working"}), encoding="utf-8")
    return {
        "task_layout": {"task_folder": str(task)}, "metadata_path": str(metadata),
        "progress_file": str(progress), "runtime": {},
    }


@pytest.fixture(autouse=True)
def _isolated_batch_state(monkeypatch):
    monkeypatch.setattr(batch, "_BATCH_CANCEL", threading.Event())
    monkeypatch.setattr(batch, "_BATCH_ACTIVE", threading.Event())
    monkeypatch.setattr(batch, "_BATCH_CURRENT_TASK", "")
    monkeypatch.setattr(batch, "_BATCH_RUN_ID", "")


def test_collection_preserves_upload_order_and_sorts_only_folder_additions(tmp_path):
    uploads = []
    for cache, name in (("zzz_hash", "a_first.txt"), ("aaa_hash", "b_second.txt")):
        path = tmp_path / cache / name
        path.parent.mkdir()
        path.write_text(name, encoding="utf-8")
        uploads.append(str(path))
    folder = tmp_path / "folder"
    folder.mkdir()
    for name in ("z_last.txt", "c_third.txt"):
        (folder / name).write_text(name, encoding="utf-8")

    items = batch._batch_items([*uploads, uploads[0]], "pasted last", str(folder))

    assert [item["name"] for item in items] == [
        "a_first", "b_second", "c_third", "z_last", "pasted_001",
    ]
    assert items[0]["text"] is None
    assert batch._load_batch_item(items[0])["text"] == "a_first.txt"


@pytest.mark.parametrize("continue_errors,expected_names", [(True, ["a_first", "invalid", "c_last"]), (False, ["a_first", "invalid"])])
def test_malformed_caption_is_an_item_error_not_an_eager_batch_abort(tmp_path, monkeypatch, continue_errors, expected_names):
    files = []
    for cache, filename, content in (
        ("z_hash", "a_first.txt", "First item."),
        ("a_hash", "invalid.srt", "This is not a caption file."),
        ("m_hash", "c_last.txt", "Last item."),
    ):
        path = tmp_path / cache / filename
        path.parent.mkdir()
        path.write_text(content, encoding="utf-8")
        files.append(str(path))
    reference = tmp_path / "reference.wav"
    reference.touch()
    requests = []

    def poll(request, *_args):
        requests.append(request)
        yield {"fraction": 0.5}, "synthetic progress"
        return {"output_path": "synthetic.wav", "audio_seconds": 1.0}

    monkeypatch.setattr(batch, "_poll_batch_item", poll)
    events = _bound_events(tmp_path, monkeypatch)
    updates = list(_run_batch(events, files, reference, batch_values={"batch.continue_errors": continue_errors}))
    rows = updates[-1][3]

    assert [row[0] for row in rows] == expected_names
    assert rows[0][1] == "Complete"
    assert rows[1][1].startswith("Failed:")
    assert len(requests) == (2 if continue_errors else 1)
    assert not batch._BATCH_ACTIVE.is_set()


@pytest.mark.parametrize("execution,voice_subprocess,expected_mode", [
    ("Subprocess per item", False, "subprocess"),
    ("Reuse loaded model between items", True, "in_process"),
    ("Reload in-process model per item", True, "in_process"),
])
def test_batch_forwards_voice_assets_and_records_actual_execution_mode(tmp_path, monkeypatch, execution, voice_subprocess, expected_mode):
    source, reference, image, emotion = (tmp_path / name for name in ("words.txt", "reference.wav", "image.png", "emotion.wav"))
    source.write_text("A complete sentence.", encoding="utf-8")
    for path in (reference, image, emotion):
        path.touch()
    generation_values = {
        "generation.use_subprocess": voice_subprocess,
        "generation.use_caption_timing": True,
        "generation.emotion_mode": EMOTION_MODES[1],
    }
    requests = []

    def poll(request, subprocess_mode, reuse_model):
        requests.append((request, subprocess_mode, reuse_model))
        yield {}, ""
        return {"output_path": "synthetic.wav", "audio_seconds": 1.0}

    monkeypatch.setattr(batch, "_poll_batch_item", poll)
    monkeypatch.setattr(batch, "LAZY_ENGINE", SimpleNamespace(reset_cancel=lambda **_kwargs: None))
    events = _bound_events(tmp_path, monkeypatch, generation_values)
    updates = list(_run_batch(
        events, [str(source)], reference, batch_values={"batch.execution": execution},
        generation_values=generation_values, image=str(image), emotion=str(emotion),
    ))
    assert updates[-1][3][0][1] == "Complete"
    request, subprocess_mode, reuse_model = requests[0]
    metadata = json.loads(Path(request["metadata_path"]).read_text(encoding="utf-8"))

    assert metadata["inputs"]["source_image"] == str(image.resolve())
    assert metadata["inputs"]["emotion_reference_audio"] == str(emotion.resolve())
    assert request["infer_kwargs"]["emo_audio_prompt"] == str(emotion)
    assert Path(request["image_path"]).read_bytes() == image.read_bytes()
    assert metadata["settings"]["execution_mode"] == expected_mode
    assert metadata["settings"]["request_values"]["generation.use_subprocess"] is subprocess_mode
    assert request["subtitle_mode"] is False
    assert reuse_model is (execution == "Reuse loaded model between items")
    assert generation_values["generation.use_subprocess"] is voice_subprocess
    assert events["generate_batch"]["inputs"][4:6] == ["image component", "emotion component"]


def test_missing_per_file_reference_obeys_continue_and_emits_a_failed_row(tmp_path, monkeypatch):
    reference = tmp_path / "shared.wav"
    reference.touch()
    source = tmp_path / "missing.txt"
    source.write_text("Missing its own reference.", encoding="utf-8")
    events = _bound_events(tmp_path, monkeypatch)
    updates = list(_run_batch(events, [str(source)], reference, batch_values={"batch.reference_mode": "Per-file reference"}))
    assert updates[-1][3][0][1] == "Failed: Missing same-stem reference"
    assert any("Completed 1/1" in str(update[2]) for update in updates)


def test_inprocess_cancel_waits_for_worker_and_never_unloads_it_early(tmp_path, monkeypatch):
    request = _item_request(tmp_path)
    entered, release, stopped, canceled = (threading.Event() for _ in range(4))
    actions = []

    def check_cancel():
        if canceled.is_set():
            raise RuntimeError("Generation canceled by user")

    def unload():
        assert not entered.is_set() or stopped.is_set(), "unloaded while worker was live"
        actions.append("unload")

    engine = SimpleNamespace(
        get=lambda *_args, **_kwargs: object(), raise_if_canceled=check_cancel,
        request_cancel=lambda **_kwargs: canceled.set(), unload=unload,
    )

    def run(_request, _engine, *, cancellation_check):
        entered.set()
        try:
            assert release.wait(5), "test worker was not released"
            cancellation_check()
        finally:
            stopped.set()

    monkeypatch.setattr(batch, "LAZY_ENGINE", engine)
    monkeypatch.setattr(batch, "run_generation_request", run)
    poller = batch._poll_batch_item(request, False, False)
    try:
        next(poller)
        assert entered.wait(2)
        batch._BATCH_CANCEL.set()
        payload, _ = next(poller)
        assert "waiting for the worker" in payload["desc"]
        assert not stopped.is_set()
        assert json.loads(Path(request["metadata_path"]).read_text())["status"] == "in_progress"
        assert actions == ["unload"]  # Reload-before-first-item, not teardown yet.
        release.set()
        with pytest.raises(batch._BatchCanceled):
            list(poller)
        assert stopped.is_set()
        assert actions == ["unload", "unload"]
        assert json.loads(Path(request["metadata_path"]).read_text())["status"] == "canceled"
        assert json.loads((Path(request["task_layout"]["task_folder"]) / "result.json").read_text())["status"] == "canceled"
    finally:
        release.set()
        poller.close()


def test_inprocess_load_failure_has_terminal_metadata_and_result(tmp_path, monkeypatch):
    request = _item_request(tmp_path)

    def fail_load(*_args, **_kwargs):
        raise RuntimeError("synthetic model load failure")

    monkeypatch.setattr(batch, "LAZY_ENGINE", SimpleNamespace(
        get=fail_load, raise_if_canceled=lambda: None,
    ))
    with pytest.raises(RuntimeError, match="synthetic model load failure"):
        list(batch._poll_batch_item(request, False, True))
    assert json.loads(Path(request["metadata_path"]).read_text())["status"] == "failed"
    task = Path(request["task_layout"]["task_folder"])
    assert json.loads((task / "result.json").read_text())["status"] == "error"
    assert "synthetic model load failure" in (task / "generation.log").read_text()


def test_reload_policy_unloads_before_first_item_and_after_worker_finishes(tmp_path, monkeypatch):
    request = _item_request(tmp_path)
    actions = []
    monkeypatch.setattr(batch, "LAZY_ENGINE", SimpleNamespace(
        get=lambda *_args, **_kwargs: actions.append("get"),
        raise_if_canceled=lambda: None, unload=lambda: actions.append("unload"),
    ))

    def run(*_args, **_kwargs):
        actions.append("run")
        print("synthetic item log")
        return {"audio_seconds": 1.0, "output_path": "synthetic.wav"}

    monkeypatch.setattr(batch, "run_generation_request", run)
    list(batch._poll_batch_item(request, False, False))
    assert actions == ["unload", "get", "run", "unload"]
    task = Path(request["task_layout"]["task_folder"])
    assert json.loads((task / "result.json").read_text())["status"] == "ok"
    assert "synthetic item log" in (task / "generation.log").read_text()


def test_subprocess_cancel_stays_active_until_process_really_exits(tmp_path, monkeypatch):
    request = _item_request(tmp_path)
    job = SimpleNamespace(running=True, canceled=False, process=SimpleNamespace(returncode=None))
    calls = []

    def terminate(kind, *, expected_job):
        assert expected_job is job
        calls.append(kind)
        job.canceled = True
        return True  # Windows taskkill may return before the process exits.

    monkeypatch.setattr(batch, "PROCESS_MANAGER", SimpleNamespace(start=lambda *_args, **_kwargs: job, terminate=terminate))
    poller = batch._poll_batch_item(request, True, False)
    try:
        next(poller)
        batch._BATCH_CANCEL.set()
        payload, _ = next(poller)
        assert "waiting for the worker" in payload["desc"]
        assert calls == ["batch_generation"]
        assert json.loads(Path(request["metadata_path"]).read_text())["status"] == "in_progress"
        job.running = False
        job.process.returncode = -1
        with pytest.raises(batch._BatchCanceled):
            next(poller)
        assert json.loads(Path(request["metadata_path"]).read_text())["status"] == "canceled"
    finally:
        job.running = False
        poller.close()


def test_inline_cancel_request_does_not_prematurely_change_metadata(tmp_path, monkeypatch):
    request = _item_request(tmp_path)
    Path(request["metadata_path"]).write_text(json.dumps({"status": "in_progress", "batch": {"run_id": "batch-a"}}))
    events = _bound_events(tmp_path, monkeypatch)
    monkeypatch.setattr(batch, "PROCESS_MANAGER", SimpleNamespace(get=lambda _kind: None))
    calls = []
    monkeypatch.setattr(batch, "LAZY_ENGINE", SimpleNamespace(request_cancel=lambda **kwargs: calls.append(kwargs)))
    batch._BATCH_ACTIVE.set()
    monkeypatch.setattr(batch, "_BATCH_RUN_ID", "batch-a")
    monkeypatch.setattr(batch, "_BATCH_CURRENT_TASK", request["task_layout"]["task_folder"])
    opened = events["show_batch_cancel"]["fn"](request["task_layout"]["task_folder"])
    assert opened[0]["visible"] is True
    updates = events["confirm_cancel_batch"]["fn"](opened[1])

    assert batch._BATCH_CANCEL.is_set()
    assert "requested" in updates[1]
    assert "was stopped" not in updates[1]
    assert updates[2]["visible"] is False
    assert json.loads(Path(request["metadata_path"]).read_text())["status"] == "in_progress"
    assert events["confirm_cancel_batch"].get("js") is None
    assert calls == [{"expected_task": "batch-a"}]


def test_stale_batch_cancel_cannot_stop_an_unrelated_new_worker(tmp_path, monkeypatch):
    events = _bound_events(tmp_path, monkeypatch)
    calls = []
    job = SimpleNamespace(running=True, state_dir=tmp_path / "newer_task")
    monkeypatch.setattr(batch, "PROCESS_MANAGER", SimpleNamespace(
        get=lambda _kind: job, terminate=lambda *_args, **_kwargs: calls.append("terminated"),
    ))
    batch._BATCH_ACTIVE.set()
    monkeypatch.setattr(batch, "_BATCH_RUN_ID", "newer-batch")
    updates = events["confirm_cancel_batch"]["fn"]("old-batch")

    assert "no other run was stopped" in updates[1]
    assert not batch._BATCH_CANCEL.is_set()
    assert calls == []


def test_batch_confirmation_remains_scoped_when_the_active_item_advances(tmp_path, monkeypatch):
    request = _item_request(tmp_path)
    Path(request["metadata_path"]).write_text(json.dumps({"status": "completed", "batch": {"run_id": "same-batch"}}))
    events = _bound_events(tmp_path, monkeypatch)
    batch._BATCH_ACTIVE.set()
    monkeypatch.setattr(batch, "_BATCH_RUN_ID", "same-batch")
    monkeypatch.setattr(batch, "_BATCH_CURRENT_TASK", request["task_layout"]["task_folder"])
    opened = events["show_batch_cancel"]["fn"](request["task_layout"]["task_folder"])
    second = tmp_path / "next-item"
    second.mkdir()
    (second / "progress.json").write_text("{}")
    monkeypatch.setattr(batch, "_BATCH_CURRENT_TASK", str(second))
    job = SimpleNamespace(running=True, state_dir=second)
    calls = []
    monkeypatch.setattr(batch, "PROCESS_MANAGER", SimpleNamespace(
        get=lambda _kind: job, terminate=lambda _kind, **kwargs: calls.append(kwargs),
    ))
    updates = events["confirm_cancel_batch"]["fn"](opened[1])
    assert "requested" in updates[1]
    assert calls == [{"expected_job": job}]


def test_cancel_panel_rejects_a_previous_batch_card(tmp_path, monkeypatch):
    request = _item_request(tmp_path)
    Path(request["metadata_path"]).write_text(json.dumps({"status": "completed", "batch": {"run_id": "old-batch"}}))
    events = _bound_events(tmp_path, monkeypatch)
    batch._BATCH_ACTIVE.set()
    monkeypatch.setattr(batch, "_BATCH_RUN_ID", "new-batch")
    opened = events["show_batch_cancel"]["fn"](request["task_layout"]["task_folder"])
    assert opened[0]["visible"] is False
    assert opened[1] == ""
    assert not batch._BATCH_CANCEL.is_set()


def test_late_cancel_does_not_relabel_completed_inprocess_audio(tmp_path, monkeypatch):
    request = _item_request(tmp_path)
    monkeypatch.setattr(batch, "LAZY_ENGINE", SimpleNamespace(
        get=lambda *_args, **_kwargs: object(), raise_if_canceled=lambda: None,
        request_cancel=lambda **_kwargs: None,
    ))

    def run(*_args, **_kwargs):
        Path(request["metadata_path"]).write_text(json.dumps({"status": "completed"}))
        batch._BATCH_CANCEL.set()  # Worker finished immediately before the click.
        return {"output_path": "completed.wav", "audio_seconds": 1.0}

    monkeypatch.setattr(batch, "run_generation_request", run)
    poller = batch._poll_batch_item(request, False, True)
    while True:
        try:
            next(poller)
        except StopIteration as completed:
            result = completed.value
            break
    assert result["output_path"] == "completed.wav"
    assert json.loads(Path(request["metadata_path"]).read_text())["status"] == "completed"
