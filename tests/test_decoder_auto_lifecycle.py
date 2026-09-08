"""CPU-only AUTO lifecycle checks; explicit validation never uses AUTO approval."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from indextts.lora import decoder


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture weights; no model loading")
    return path


@pytest.fixture
def voice(tmp_path):
    root = tmp_path / "voice"
    return SimpleNamespace(
        root=root,
        gpt=_touch(root / "voice.safetensors"),
        epoch=_touch(root / "voice_epoch_003.safetensors"),
        best=_touch(root / "best" / "voice.safetensors"),
        adapter=_touch(root / "voice.s2mel.safetensors"),
        status=root / "status.json",
        child=root / "analysis" / "decoder_adapter_job" / "status.json",
        adaptation=root / "analysis" / "decoder_adapter.json",
        gate=root / "analysis" / "speech_evaluation" / "decoder_test" / "report.json",
        gate_child=root / "analysis" / "speech_evaluation" / "decoder_test" / "test_job" / "status.json",
    )


def _accepted(voice, *, root_fields=True):
    status = {"phase": "stopped"}  # GPT early stopping is not decoder failure.
    if root_fields:
        status.update(decoder_adapter_status="complete", decoder_test_status="complete",
                      decoder_adapter_path=str(voice.adapter.resolve()))
    _write(voice.status, status)
    _write(voice.child, {"phase": "complete", "accepted": True})
    _write(voice.adaptation, {"status": "complete", "accepted": True})
    _write(voice.gate_child, {"phase": "complete"})
    _write(voice.gate, {"status": "complete", "accepted": True,
                        "adapter": str(voice.adapter.resolve()), "strength": 0.6})


@pytest.mark.parametrize("checkpoint", ["gpt", "epoch", "best"])
@pytest.mark.parametrize("status", [None, {}, {"phase": "stopped", "speech_evaluation_status": "complete"}])
def test_metadata_free_legacy_discovery_is_preserved(voice, checkpoint, status):
    if status is not None:
        _write(voice.status, status)
    assert Path(decoder.find_decoder_adapter(getattr(voice, checkpoint))) == voice.adapter.resolve()


@pytest.mark.parametrize("checkpoint", ["gpt", "epoch", "best"])
@pytest.mark.parametrize("root_fields", [False, True])
def test_old_accepted_v8_v9_shapes_need_no_new_hash_fields(voice, checkpoint, root_fields):
    _accepted(voice, root_fields=root_fields)
    assert Path(decoder.find_decoder_adapter(getattr(voice, checkpoint))) == voice.adapter.resolve()
    assert decoder.recommended_decoder_strength(getattr(voice, checkpoint)) == 0.6


def test_minimal_historical_positive_gate_is_still_accepted(voice):
    _write(voice.gate, {"accepted": True, "strength": 0.6})
    assert Path(decoder.find_decoder_adapter(voice.best)) == voice.adapter.resolve()


@pytest.mark.parametrize("checkpoint", ["gpt", "epoch", "best"])
@pytest.mark.parametrize("phase", ["adapting_decoder", "testing_decoder"])
def test_active_root_phase_blocks_even_an_old_accepted_gate(voice, checkpoint, phase):
    _accepted(voice)
    _write(voice.status, {"phase": phase, "decoder_adapter_status": "complete"})
    assert decoder.find_decoder_adapter(getattr(voice, checkpoint)) == ""
    assert Path(decoder.find_decoder_adapter(getattr(voice, checkpoint), allow_unverified=True)) == voice.adapter.resolve()


@pytest.mark.parametrize("key", ["decoder_adapter_status", "decoder_test_status"])
@pytest.mark.parametrize("value", ["running", "failed", "rejected", "skipped", "pending", "unverified",
                                   "stopped", "canceled", "cancelled", "", None, True])
def test_noncomplete_root_outcome_is_never_auto_accepted(voice, key, value):
    _accepted(voice)
    _write(voice.status, {"phase": "stopped", key: value})
    assert decoder.find_decoder_adapter(voice.best) == ""


@pytest.mark.parametrize("which", ["child", "gate_child"])
@pytest.mark.parametrize("phase", ["initializing", "training", "validating", "testing_decoder", "failed",
                                   "rejected", "skipped", "stopped", "", None])
def test_child_lifecycle_cannot_be_overruled_by_stale_parent_acceptance(voice, which, phase):
    _accepted(voice)
    _write(getattr(voice, which), {"phase": phase})
    assert decoder.find_decoder_adapter(voice.gpt) == ""


@pytest.mark.parametrize("record", ["child", "gate_child", "adaptation"])
def test_negative_child_or_teacher_acceptance_is_not_ignored(voice, record):
    _accepted(voice)
    _write(getattr(voice, record), {"phase": "complete", "status": "complete", "accepted": False})
    assert decoder.find_decoder_adapter(voice.gpt) == ""


@pytest.mark.parametrize("report", [
    {}, {"status": "complete"}, {"accepted": False}, {"accepted": "true"}, {"accepted": 1},
    {"accepted": None}, {"accepted": True, "status": "running"},
    {"accepted": True, "status": "failed"}, {"accepted": True, "status": "rejected"},
    {"accepted": True, "status": "skipped"}, {"accepted": True, "status": None},
    {"accepted": True, "status": "complete", "error": "worker failed"},
])
def test_gate_requires_a_true_positive_completed_outcome(voice, report):
    _accepted(voice)
    _write(voice.gate, report)
    assert decoder.find_decoder_adapter(voice.gpt) == ""


@pytest.mark.parametrize("evidence", ["root_status", "adaptation", "adaptation_job", "gate_directory"])
def test_missing_gate_is_unverified_when_a_decoder_lifecycle_is_known(voice, evidence):
    if evidence == "root_status":
        _write(voice.status, {"phase": "stopped", "decoder_adapter_status": "complete"})
    elif evidence == "adaptation":
        _write(voice.adaptation, {"status": "complete", "accepted": True})
    elif evidence == "adaptation_job":
        voice.child.parent.mkdir(parents=True)
    else:
        voice.gate.parent.mkdir(parents=True)
    assert decoder.find_decoder_adapter(voice.gpt) == ""


@pytest.mark.parametrize("record", ["status", "child", "adaptation", "gate", "gate_child"])
@pytest.mark.parametrize("raw", [b'{"unfinished":', b'[]', b'null', b'\xff'])
def test_malformed_existing_metadata_fails_closed(voice, record, raw):
    _accepted(voice)
    getattr(voice, record).write_bytes(raw)
    assert decoder.find_decoder_adapter(voice.best) == ""
    assert Path(decoder.find_decoder_adapter(voice.best, allow_unverified=True)) == voice.adapter.resolve()


def test_unreadable_metadata_fails_closed_without_throwing(voice, monkeypatch):
    _accepted(voice)
    original = Path.read_text

    def unreadable(path, *args, **kwargs):
        if path == voice.gate:
            raise PermissionError("fixture denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unreadable)
    assert decoder.find_decoder_adapter(voice.best) == ""


def test_status_is_reread_when_a_new_adaptation_starts(voice):
    _accepted(voice)
    assert decoder.find_decoder_adapter(voice.best)
    _write(voice.status, {"phase": "adapting_decoder", "decoder_adapter_status": "running"})
    assert decoder.find_decoder_adapter(voice.best) == ""
    _accepted(voice)
    assert decoder.find_decoder_adapter(voice.best)
    voice.gate.unlink()
    assert decoder.find_decoder_adapter(voice.best) == ""


def test_recorded_approval_does_not_cover_another_candidate(voice):
    _accepted(voice)
    specific = _touch(voice.best.with_name("voice.s2mel.safetensors"))
    assert Path(decoder.find_decoder_adapter(voice.best)) == voice.adapter.resolve()
    assert Path(decoder.find_decoder_adapter(voice.best, allow_unverified=True)) == specific.resolve()
    _write(voice.gate, {"accepted": True, "adapter": str(specific.resolve())})
    assert decoder.find_decoder_adapter(voice.best) == ""  # parent and gate disagree


@pytest.mark.parametrize("value", [None, "", 12, []])
def test_recorded_but_invalid_accepted_path_does_not_enable_auto(voice, value):
    _accepted(voice)
    _write(voice.gate, {"accepted": True, "adapter": value})
    assert decoder.find_decoder_adapter(voice.gpt) == ""


def test_legacy_specific_and_unique_fallback_priority_is_unchanged(voice):
    specific = _touch(voice.best.with_name("voice.s2mel.safetensors"))
    assert Path(decoder.find_decoder_adapter(voice.best)) == specific.resolve()
    voice.adapter.unlink()
    renamed = _touch(voice.root / "renamed.s2mel.safetensors")
    assert Path(decoder.find_decoder_adapter(voice.gpt)) == renamed.resolve()
    _touch(voice.root / "ambiguous.s2mel.safetensors")
    assert decoder.find_decoder_adapter(voice.gpt) == ""
    assert decoder.find_decoder_adapter(voice.adapter, allow_unverified=True) == ""
    assert decoder.find_decoder_adapter(voice.root / "missing.safetensors", allow_unverified=True) == ""


def test_auto_label_and_explicit_selection_remain_distinct(voice):
    _write(voice.status, {"phase": "adapting_decoder", "decoder_adapter_status": "running"})
    choices = decoder.decoder_adapter_choices(voice.gpt, voice.root.parent)
    assert "no eligible decoder" in choices[0][0] and choices[0][1] == "auto"
    assert ("None (GPT adapter only)", "none") in choices
    assert str(voice.adapter.resolve()) in [value for _, value in choices]
    assert decoder.decoder_adapter_selection("auto") == (True, "")
    assert decoder.decoder_adapter_selection("") == (True, "")
    assert decoder.decoder_adapter_selection("none") == (False, "")
    assert decoder.decoder_adapter_selection(voice.adapter) == (True, str(voice.adapter.resolve()))


def _inference_method(name):
    """Execute the real resolver body without importing/constructing an inference model."""
    source = Path(__file__).resolve().parents[1] / "indextts" / "infer_v2_5.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "IndexTTS2")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"os": __import__("os")}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("warm", [False, True], ids=["cold-resolver", "warm-resolver"])
@pytest.mark.parametrize("choice", ["auto", "none", "explicit"])
def test_real_inference_resolver_blocks_auto_but_preserves_explicit_and_none(voice, monkeypatch, warm, choice):
    import indextts.lora.apply as apply

    _write(voice.status, {"phase": "adapting_decoder", "decoder_adapter_status": "running"})
    estimator = torch.nn.Linear(1, 1, device="cpu")
    events = []
    handle = object()
    wanted = str(voice.adapter.resolve())
    state = SimpleNamespace(
        runtime=SimpleNamespace(decoder_adapter=wanted if choice == "explicit" else choice, decoder_adapter_strength=0.6),
        _decoder_handle=handle if warm else None, _decoder_path=wanted if warm else "", decoder_guidance="base",
        _decoder_estimator=lambda: estimator, _install_decoder_guidance=lambda: None,
    )

    def remove():
        state._decoder_handle, state._decoder_path = None, ""
        events.append("remove")

    def install(module, path, strength):
        assert module is estimator and path == wanted and strength == 0.6
        events.append("apply")
        return handle

    state._remove_decoder_adapter = remove
    monkeypatch.setattr(apply, "apply_lora", install)
    monkeypatch.setattr(apply, "move_adapters_to_device", lambda *args: events.append("move"))
    monkeypatch.setattr(apply, "set_lora_strength", lambda *args: events.append("strength"))
    if choice != "auto":
        monkeypatch.setattr(decoder, "find_decoder_adapter", lambda *args, **kwargs: pytest.fail("explicit/none must not call AUTO discovery"))
    result = _inference_method("_sync_decoder_adapter")(state, str(voice.gpt), 1.0)
    if choice == "explicit":
        assert result is handle and state._decoder_path == wanted
        assert ("strength" if warm else "apply") in events
    else:
        assert result is None and state._decoder_handle is None and state._decoder_path == ""
        assert events == ["remove"]


def test_cold_initialization_still_calls_the_shared_decoder_resolver():
    source = Path(__file__).resolve().parents[1] / "indextts" / "infer_v2_5.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "IndexTTS2")
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"
               and node.func.attr == "_sync_decoder_adapter" for node in ast.walk(init))


def test_warm_auto_transitions_between_pending_approved_and_rejected(voice, monkeypatch):
    import indextts.lora.apply as apply

    estimator = torch.nn.Linear(1, 1, device="cpu")
    installed = []
    state = SimpleNamespace(
        runtime=SimpleNamespace(decoder_adapter="auto", decoder_adapter_strength=0.6),
        _decoder_handle=None, _decoder_path="", decoder_guidance="base",
        _decoder_estimator=lambda: estimator, _install_decoder_guidance=lambda: None,
    )

    def remove():
        state._decoder_handle, state._decoder_path = None, ""

    def install(*args):
        handle = object()
        installed.append(handle)
        return handle

    state._remove_decoder_adapter = remove
    monkeypatch.setattr(apply, "apply_lora", install)
    monkeypatch.setattr(apply, "move_adapters_to_device", lambda *args: None)
    monkeypatch.setattr(apply, "set_lora_strength", lambda *args: None)
    sync = _inference_method("_sync_decoder_adapter")
    _write(voice.status, {"phase": "adapting_decoder", "decoder_adapter_status": "running"})
    assert sync(state, str(voice.best), 1.0) is None and installed == []
    _accepted(voice)
    assert sync(state, str(voice.best), 1.0) is installed[0]
    assert state._decoder_path == str(voice.adapter.resolve())
    _write(voice.status, {"phase": "adapting_decoder", "decoder_adapter_status": "running"})
    assert sync(state, str(voice.best), 1.0) is None and state._decoder_handle is None
    _accepted(voice)
    assert sync(state, str(voice.best), 1.0) is installed[1]
    _write(voice.gate, {"status": "complete", "accepted": False})
    assert sync(state, str(voice.best), 1.0) is None and state._decoder_path == ""
    assert voice.adapter.is_file() and len(installed) == 2


def test_explicit_validation_gate_reaches_rendering_while_auto_is_blocked(voice, monkeypatch):
    from indextts.training import speech_eval, speech_metrics

    class ReachedExplicitRender(Exception):
        pass

    _write(voice.status, {"phase": "adapting_decoder", "decoder_adapter_status": "running", "decoder_test_status": "running"})
    config = SimpleNamespace(output_dir=str(voice.root.parent), name=voice.root.name)
    monkeypatch.setattr(speech_eval, "development_baseline", lambda *args: ({"seeds": [42]}, [], "candidate", voice.root / "baseline.json", {"infer_kwargs": {"seed_marker": 42}}))
    monkeypatch.setattr(speech_eval, "development_fingerprint", lambda *args: "development-proof")
    monkeypatch.setattr(speech_eval, "_benchmark_runtime", lambda *args: SimpleNamespace(to_dict=lambda: {"decoder_adapter": "none"}))
    monkeypatch.setattr(speech_eval, "_lenient_terms", lambda *args: [])
    monkeypatch.setattr(speech_metrics, "summarize", lambda *args: {})

    def render(*args, **kwargs):
        assert kwargs["runtime"]["decoder_adapter"] == str(voice.adapter.resolve())
        assert kwargs["runtime"]["decoder_adapter_strength"] == 1.0
        assert decoder.find_decoder_adapter(voice.gpt) == ""
        assert json.loads(voice.status.read_text())["decoder_adapter_status"] == "running"
        raise ReachedExplicitRender

    monkeypatch.setattr(speech_eval, "render_benchmark_rows", render)
    with pytest.raises(ReachedExplicitRender):
        speech_eval.run_decoder_test(config, voice.gate_child.parent, checkpoint_path=str(voice.gpt),
                                     adapter_path=str(voice.adapter), strengths=(1.0,))

    # The bypass waives lifecycle approval, never checkpoint-to-file association.
    unrelated = _touch(voice.root / "other.s2mel.safetensors")
    with pytest.raises(ValueError, match="decoder associated with"):
        speech_eval.run_decoder_test(config, voice.gate_child.parent, checkpoint_path=str(voice.gpt),
                                     adapter_path=str(unrelated), strengths=(1.0,))
