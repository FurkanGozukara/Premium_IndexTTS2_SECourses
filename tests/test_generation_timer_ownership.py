"""Exercise timer/click interleavings without starting a worker or loading models."""

import json
from pathlib import Path
import threading
from types import SimpleNamespace

import gradio as gr
from gradio.helpers import special_args
import pytest

from ui.app import build_app
import ui.generation_tab as generation


def _task(tmp_path, status="completed"):
    folder = tmp_path / "outputs" / "old-task"
    folder.mkdir(parents=True)
    request = {
        "task_layout": {"task_folder": str(folder), "task_id": "old-task"},
        "metadata_path": str(folder / "metadata.json"),
        "progress_file": str(folder / "progress.json"),
    }
    metadata = {
        "status": status, "task": {"id": "old-task"}, "error": None,
        "outputs": {"final_audio_path": None},
        "generation": {"audio_seconds": 1.0, "segments_count": 1},
    }
    (folder / "request.json").write_text(json.dumps(request), encoding="utf-8")
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (folder / "progress.json").write_text(json.dumps({"fraction": 1.0, "desc": "Complete"}), encoding="utf-8")
    return folder, request, metadata


@pytest.fixture
def generation_demo(monkeypatch):
    monkeypatch.setattr(generation, "_GENERATION_CARD_OWNERS", set())
    monkeypatch.setattr(generation, "recent_outputs", lambda *_args, **_kwargs: [])
    return build_app(SimpleNamespace(
        model_dir="models", device="cpu", verbose=False, no_browser=True,
        port=7861, host="127.0.0.1", share=False,
    ))


def test_inflight_completed_timer_cannot_replace_a_new_validation_error(tmp_path, monkeypatch, generation_demo):
    folder, _, _ = _task(tmp_path)
    browser = gr.Request(session_hash="validation-race-page")
    entered, release = threading.Event(), threading.Event()
    result = []
    original = generation._generation_result_from_disk

    def delayed_result(*args):
        entered.set()
        assert release.wait(5), "test timer was not released"
        return original(*args)

    monkeypatch.setattr(generation, "_generation_result_from_disk", delayed_result)
    timer = threading.Thread(target=lambda: result.append(
        generation.generation_task_updates(str(folder), browser, output_root=folder.parent)
    ))
    timer.start()
    try:
        assert entered.wait(2)
        generate = next(fn.fn for fn in generation_demo.fns.values() if fn.api_name == "generate_voice")
        reference = tmp_path / "reference.wav"
        reference.touch()
        failed = list(generate(str(reference), "", None, None, None, browser))[-1]
        assert "Enter text or load a caption file" in failed[2]
        assert failed[0] == "" and failed[-1].active is False
        release.set()
        timer.join(2)
        assert not timer.is_alive()
        assert result[0][:-1] == (gr.skip(),) * 10
        assert result[0][-1].active is False
        # Even a tick already queued with the previous task path stays inert.
        repeated = generation.generation_task_updates(str(folder), browser, output_root=folder.parent)
        assert repeated[:-1] == (gr.skip(),) * 10
        assert repeated[-1].active is False
    finally:
        release.set()
        timer.join(2)


def test_reload_session_still_attaches_to_live_task_and_stops_after_completion(tmp_path, generation_demo):
    folder, _, metadata = _task(tmp_path, "in_progress")
    old_page = gr.Request(session_hash="old-click-owned-page")
    generation._claim_generation_card(old_page)
    reloaded = gr.Request(session_hash="new-page-load")
    attached = generation.generation_task_updates("", reloaded, output_root=folder.parent, page_load=True)
    assert attached[0] == str(folder)
    assert "Attached to running run old-task" in attached[2]
    assert attached[-1].active is True
    metadata["status"] = "completed"
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    completed = generation.generation_task_updates(str(folder), reloaded, output_root=folder.parent)
    assert "Generation complete" in completed[2]
    assert completed[-1].active is False


def test_idle_page_does_not_leave_a_timer_repainting_the_card(tmp_path, generation_demo):
    updates = generation.generation_task_updates(
        "", gr.Request(session_hash="idle-page"), output_root=tmp_path, page_load=True,
    )
    assert updates[0] == ""
    assert updates[-1].active is False


def test_connected_generator_owns_card_and_disables_competing_timer(tmp_path, monkeypatch, generation_demo):
    folder, request, metadata = _task(tmp_path, "in_progress")
    browser = gr.Request(session_hash="connected-stream-page")
    monkeypatch.setattr(generation, "prepare_generation_request", lambda *_args, **_kwargs: request)

    def stream(*_args, **_kwargs):
        yield ("working panel", "Working", "log", gr.skip(), gr.skip(), [], gr.skip(), "", [])
        metadata["status"] = "completed"
        (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        yield ("complete panel", "Complete", "log", gr.skip(), gr.skip(), [], gr.skip(), "", [])

    monkeypatch.setattr(generation, "stream_generation_request", stream)
    generate = next(fn.fn for fn in generation_demo.fns.values() if fn.api_name == "generate_voice")
    updates = list(generate("unused", "synthetic", None, None, None, browser))
    assert [update[2] for update in updates] == ["Working", "Complete"]
    assert [update[-1].active for update in updates] == [False, False]
    stale = generation.generation_task_updates(str(folder), browser, output_root=folder.parent)
    assert stale[:-1] == (gr.skip(),) * 10


def test_gradio_injects_page_identity_without_changing_component_argument_order(generation_demo):
    browser = gr.Request(session_hash="injection-check")
    generate = next(fn for fn in generation_demo.fns.values() if fn.api_name == "generate_voice")
    values = [f"component-{index}" for index in range(len(generate.inputs))]
    injected, _, _, _ = special_args(generate.fn, list(values), request=browser)
    assert injected[:5] == values[:5]
    assert injected[5] is browser
    assert injected[6:] == values[5:]

    timer = next(fn for fn in generation_demo.fns.values() if fn.fn is generation.generation_task_updates)
    injected, _, _, _ = special_args(timer.fn, ["old-task"], request=browser)
    assert injected[:2] == ["old-task", browser]
    attached = next(fn for fn in generation_demo.fns.values() if fn.api_name == "attach_generation")
    injected, _, _, _ = special_args(attached.fn, ["old-task"], request=browser)
    assert injected == ["old-task", browser]


def test_reference_validation_also_claims_the_card_before_it_can_fail(monkeypatch, generation_demo):
    browser = gr.Request(session_hash="reference-validation")

    def fail(*_args, **_kwargs):
        raise ValueError("missing test reference")

    monkeypatch.setattr(generation, "prepare_reference_for_generation", fail)
    monkeypatch.setattr(gr, "Warning", lambda *_args, **_kwargs: None)
    prepare = next(fn.fn for fn in generation_demo.fns.values() if fn.api_name == "prepare_reference_voice")
    with pytest.raises(gr.Error, match="missing test reference"):
        prepare(None, None, None, "", "", "", False, browser)
    assert generation._generation_card_is_owned(browser)
