import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file

from indextts.training.analysis import checkpoint_descriptor, checkpoint_display_label, discover_checkpoints
from indextts.training.selection import recommended_generation_value


def adapter(path, component="gpt", steps=100):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"dummy": np.zeros(1, dtype=np.float32)}, str(path), metadata={
        "adapter_type": "dora", "trained_steps": str(steps), "epochs": "6",
        "train_config": json.dumps({"component": component}),
    })
    return path


def test_checkpoint_discovery_excludes_decoder_filename_and_metadata(tmp_path):
    gpt = adapter(tmp_path / "voice_epoch_006.safetensors")
    decoder = adapter(tmp_path / "voice.s2mel.safetensors", "s2mel", 11000)
    adapter(tmp_path / "renamed_decoder.safetensors", "s2mel")
    adapter(tmp_path / "best" / "also_decoder.safetensors", "s2mel")
    assert [row["path"] for row in discover_checkpoints(tmp_path)] == [str(gpt)]
    assert checkpoint_descriptor(decoder)["kind"] == "decoder"
    assert "Voice decoder" in checkpoint_descriptor(decoder)["label"]
    assert recommended_generation_value(tmp_path) == str(gpt)


@pytest.mark.parametrize("filename", ["voice.s2mel.safetensors", "renamed_decoder.safetensors"])
def test_cached_decoder_final_labels_are_corrected_even_when_already_modern(tmp_path, filename):
    decoder = adapter(tmp_path / filename, component="s2mel")
    for old_label in ("final", "final (epoch 6 DoRA Checkpoint)", "final (epoch 6 DoRA Checkpoint) @0.6"):
        corrected = checkpoint_display_label(old_label, path=decoder, kind="final")
        assert corrected == "Voice decoder adapter (s2mel)" + (" @0.6" if old_label.endswith(" @0.6") else "")


def test_valid_modern_gpt_labels_are_preserved_and_missing_decoder_is_still_identified(tmp_path):
    gpt = adapter(tmp_path / "voice.safetensors")
    modern = "Chosen shortlist (epoch 6 DoRA Checkpoint) @0.6"
    assert checkpoint_display_label(modern, path=gpt, kind="final") == modern
    assert checkpoint_display_label(
        "final (epoch 6 DoRA Checkpoint)", path=tmp_path / "missing.s2mel.safetensors", kind="final",
    ) == "Voice decoder adapter (s2mel)"


def test_grid_and_training_prefer_same_speech_checkpoint(tmp_path, monkeypatch):
    import indextts.training.speech_eval as speech
    import indextts.training.checkpoint_eval as measured
    from ui.grid_tab import _analysis_payload
    from ui.training_tab import recommended_generation_value as training_selection
    chosen = adapter(tmp_path / "voice_epoch_006.safetensors")
    loss_winner = adapter(tmp_path / "best" / "voice.safetensors")
    monkeypatch.setattr(speech, "load_speech_evaluation", lambda _: {
        "recommended_kind": "adapter", "recommended_checkpoint": str(chosen),
    })
    monkeypatch.setattr(measured, "load_checkpoint_eval", lambda _: SimpleNamespace(
        recommended_kind="adapter", recommended_checkpoint=str(loss_winner),
    ))
    assert training_selection(tmp_path) == str(chosen)
    payload = _analysis_payload(tmp_path)
    assert payload["recommended"] == str(chosen)
    assert any(payload["mapping"][identifier]["path"] == str(chosen) for identifier in payload["selected"])
    assert "speech evaluation takes precedence" in payload["summary"]


def test_base_and_missing_speech_recommendations_are_not_replaced(tmp_path, monkeypatch):
    import indextts.training.speech_eval as speech
    from ui.grid_tab import _analysis_payload
    adapter(tmp_path / "voice.safetensors")
    report = {"recommended_kind": "base", "recommended_checkpoint": ""}
    monkeypatch.setattr(speech, "load_speech_evaluation", lambda _: report)
    assert recommended_generation_value(tmp_path) == ""
    assert _analysis_payload(tmp_path)["recommended"] == ""
    report.update(recommended_kind="adapter", recommended_checkpoint="missing.safetensors")
    with pytest.raises(ValueError, match="missing"):
        recommended_generation_value(tmp_path)
    assert "recommendation unavailable" in _analysis_payload(tmp_path)["summary"]


@pytest.mark.parametrize("source", ["speech", "measured"])
def test_authoritative_recommendations_reject_renamed_decoder_metadata(tmp_path, monkeypatch, source):
    import indextts.training.speech_eval as speech
    import indextts.training.checkpoint_eval as measured

    decoder = adapter(tmp_path / "renamed.safetensors", component="s2mel")
    report = {"recommended_kind": "adapter", "recommended_checkpoint": str(decoder)}
    monkeypatch.setattr(speech, "load_speech_evaluation", lambda _: report if source == "speech" else None)
    monkeypatch.setattr(measured, "load_checkpoint_eval", lambda _: SimpleNamespace(**report))
    with pytest.raises(ValueError, match="voice decoder"):
        recommended_generation_value(tmp_path)


@pytest.mark.parametrize("value", [[], ["invalid status"], None, 3])
def test_non_object_status_json_does_not_break_checkpoint_selection(tmp_path, value):
    chosen = adapter(tmp_path / "voice.safetensors")
    (tmp_path / "status.json").write_text(json.dumps(value), encoding="utf-8")
    assert recommended_generation_value(tmp_path) == str(chosen)


def test_uncalibrated_adapter_clears_previous_rate_only_when_auto_enabled(monkeypatch):
    import gradio as gr
    import ui.generation_tab as generation
    monkeypatch.setattr(generation, "_lora_info", lambda _path, **_panel: ("info", None))
    monkeypatch.setattr(generation, "_lora_reference", lambda _path: None)
    monkeypatch.setattr(generation, "load_speaking_rate", lambda _: None)
    assert generation.lora_selection_updates("new.safetensors", None, False, True)[3] == 1.0
    assert generation.lora_selection_updates("new.safetensors", None, False, False)[3] == gr.skip()


def test_initial_progress_makes_cold_task_discoverable(tmp_path):
    from ui.common import latest_output_task
    from ui.generation_tab import prepare_generation_request
    ref = tmp_path / "reference.wav"
    ref.write_bytes(b"reference placeholder")
    output_root = tmp_path / "outputs"
    request = prepare_generation_request({}, prompt=str(ref), text="Hello.", subtitle_file=None,
        image_path=None, emotion_audio=None, model_dir="models", output_root=output_root)
    progress = json.loads(Path(request["progress_file"]).read_text())
    assert progress["stage"] == "initializing"
    assert latest_output_task(output_root) == request["task_layout"]["task_folder"]


def test_validation_error_does_not_reattach_previous_completed_task(tmp_path):
    from ui.app import build_app
    demo = build_app(SimpleNamespace(model_dir="models", device="cpu", verbose=False,
        no_browser=True, port=7861, host="127.0.0.1", share=False))
    generate = next(fn.fn for fn in demo.fns.values() if fn.api_name == "generate_voice")
    ref = tmp_path / "reference.wav"
    ref.write_bytes(b"reference placeholder")
    updates = list(generate(str(ref), "", None, None, None))[-1]
    assert updates[0] == ""
    assert "Enter text or load a caption file" in updates[2]
    assert updates[-1].get_config()["active"] is False
    assert updates[4]["value"] is None and updates[4]["visible"] is False


def test_subprocess_cancel_becomes_terminal_after_worker_exits(tmp_path, monkeypatch):
    import time
    import ui.generation_tab as generation
    task = tmp_path / "0001"
    task.mkdir()
    metadata_path = task / "metadata.json"
    metadata_path.write_text(json.dumps({"status": "in_progress", "processing": {}}))
    request = {"task_layout": {"task_folder": str(task)},
        "metadata_path": str(metadata_path), "progress_file": str(task / "progress.json")}
    job = SimpleNamespace(running=False, canceled=True, started_at=time.monotonic(),
        process=SimpleNamespace(returncode=-1))
    monkeypatch.setattr(generation.PROCESS_MANAGER, "start", lambda *args, **kwargs: job)
    monkeypatch.setattr(generation, "recent_outputs", lambda: [])
    list(generation.stream_generation_request(request, use_subprocess=True))
    assert json.loads(metadata_path.read_text())["status"] == "canceled"


def test_stale_inline_confirmation_cannot_cancel_a_new_generation_or_grid(tmp_path, monkeypatch):
    from ui.app import build_app
    import ui.generation_tab as generation
    demo = build_app(SimpleNamespace(model_dir="models", device="cpu", verbose=False,
        no_browser=True, port=7861, host="127.0.0.1", share=False))
    calls = []
    monkeypatch.setattr(generation.LAZY_ENGINE, "request_cancel", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(generation.PROCESS_MANAGER, "terminate", lambda *args, **kwargs: calls.append(args))
    cancel_generation = next(fn.fn for fn in demo.fns.values() if fn.api_name == "confirm_cancel_generation")
    cancel_grid = next(fn.fn for fn in demo.fns.values() if fn.api_name == "confirm_cancel_grid")
    old, current = str(tmp_path / "old"), str(tmp_path / "new")
    assert "changed" in cancel_generation(old, current, True)[1]
    assert "changed" in cancel_generation(old, current, False)[1]
    assert "changed" in cancel_grid(old, current)
    assert not calls
