"""CPU-only adversarial checks for validation selection and final deployment freeze.

The media/model calls are replaced, but report loading, provenance validation,
per-candidate runtime construction, recommendation guards, and persistence are real.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from indextts.runtime import RuntimeConfig
from indextts.training import speech_eval as speech
from indextts.training.decoding_sweep import load_decoding_settings
from indextts.training.speaking_rate import SpeakingRateReport, write_speaking_rate
from indextts.training.speech_metrics import select_recommendation
from indextts.training.train_config import TrainConfig


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _measurement(label: str, prompt: str, *, speaker: float = 0.8, error: float = 0.0) -> dict:
    return {"checkpoint": label, "prompt_id": prompt, "seed": 42, "errors": error * 10,
            "units": 10, "error_rate": error, "speaker_similarity": speaker,
            "invalid_audio": False, "possible_truncation": False, "possible_repetition": False,
            "start_matches": True, "end_matches": True}


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    # _benchmark_runtime normally queries CUDA even when config.device is CPU.
    monkeypatch.setattr(speech, "_benchmark_runtime", lambda _config: RuntimeConfig(device="cpu"))
    config = TrainConfig(dataset_dir=str(tmp_path / "dataset"), output_dir=str(tmp_path / "loras"),
                         name="fresh", device="cpu").validate()
    run = Path(config.output_dir) / config.name
    root = run / "analysis" / "speech_evaluation"
    root.mkdir(parents=True)
    checkpoint = run / "fresh.safetensors"
    checkpoint.write_bytes(b"measured GPT adapter fixture")
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"immutable reference fixture; not decoded")
    recording = tmp_path / "recording.wav"
    recording.write_bytes(b"immutable matched recording fixture; not decoded")
    prompts = [{"id": f"p{i}", "text": f"A frozen test sentence number {i}.", "audio": str(recording),
                "audio_sha256": speech._file_sha256(recording), "kind": "matched", "source": "unseen source"}
               for i in range(2)]
    plan = {"groups": [{"id": "g", "speaker": "voice", "language": "EN", "reference": str(reference),
                        "reference_sha256": speech._file_sha256(reference), "prompts": prompts}],
            "seeds": [42], "candidate_limit": 1, "dataset_identity": "development-fixture",
            "warnings": [], "policy": {"max_wer_increase": 0.02, "max_speaker_drop": 0.03}}
    _json(root / "plan.json", plan)
    _json(root / "final_test" / "plan.json", {**plan, "dataset_identity": "unseen-fixture"})
    candidates = [{"label": "Base", "path": "", "steps": 0, "val_loss": 6},
                  {"label": "fresh", "path": str(checkpoint.resolve()), "steps": 100, "val_loss": 4,
                   "sha256": speech._file_sha256(checkpoint)}]
    rows = [_measurement(label, f"g:p{i}", error=0.1 if label == "Base" else 0.0)
            for label in ("Base", "fresh") for i in range(2)]
    report = select_recommendation(candidates, rows, plan["policy"])
    report.update(cells=rows, real_cells=[], evaluation_partition="validation", dataset_identity=plan["dataset_identity"],
                  inference={"runtime": {"runtime": {"decoder_adapter": "none"}},
                             "infer_kwargs": speech._benchmark_infer_kwargs(config)},
                  warnings=[], final_test_status="pending deployment freeze")
    _json(root / "report.json", report)
    return SimpleNamespace(config=config, run=run, root=root, checkpoint=checkpoint,
                           plan=plan, report=report, reference=reference, recording=recording)


def _accepted_pipeline(case):
    decoder = case.run / "fresh.s2mel.safetensors"
    decoder.write_bytes(b"accepted decoder fixture")
    fingerprint = speech.development_fingerprint(case.run)
    gate = {"status": "complete", "accepted": True, "evaluation_partition": "validation", "strength": 0.6,
            "checkpoint": str(case.checkpoint.resolve()), "checkpoint_sha256": speech._file_sha256(case.checkpoint),
            "adapter": str(decoder.resolve()), "adapter_sha256": speech._file_sha256(decoder),
            "baseline_report": str(case.root / "report.json"), "development_fingerprint": fingerprint,
            "checkpoint_label": "fresh", "cells": [row for row in case.report["cells"] if row["checkpoint"] == "fresh"]}
    gate_path = case.root / "decoder_test" / "report.json"
    _json(gate_path, gate)
    write_speaking_rate(case.run, SpeakingRateReport(
        recommended_speaking_rate=0.8, dataset_words_per_second=3, generated_words_per_second=3.75,
        clips_used=4, method="speech_matched", generated_at="2026-09-08T00:00:00Z", summary="Fixture calibration"))
    settings = {"temperature": 1.0, "inference_cfg_rate": 0.9, "num_beams": 1}
    sweep = {"status": "complete", "accepted": True, "evaluation_partition": "validation", "settings": settings,
             "checkpoint": str(case.checkpoint.resolve()), "checkpoint_sha256": speech._file_sha256(case.checkpoint),
             "decoder_adapter": str(decoder.resolve()), "decoder_adapter_sha256": speech._file_sha256(decoder),
             "decoder_adapter_strength": 0.6, "development_fingerprint": fingerprint, "score": 0.03}
    sweep_path = case.root / "decoding_sweep" / "report.json"
    _json(sweep_path, sweep)
    settings_path = case.run / "analysis" / "decoding.json"
    _json(settings_path, {**sweep, "report": str(sweep_path), "report_sha256": speech._file_sha256(sweep_path)})
    return SimpleNamespace(decoder=decoder, gate=gate_path, sweep=sweep_path, settings=settings_path,
                           rate=case.run / "analysis" / "speaking_rate.json")


def test_development_never_falls_back_to_an_existing_final_test_report(deployment):
    _json(deployment.root / "final_test" / "report.json", {**deployment.report, "final_test": True})
    (deployment.root / "report.json").unlink()
    with pytest.raises(ValueError, match="final-test"):
        speech.development_baseline(deployment.run, str(deployment.checkpoint.resolve()))


@pytest.mark.parametrize("consumer", ["baseline", "freeze"])
@pytest.mark.parametrize("corruption", ["weights", "missing_hash", "final_flag", "final_partition", "auto_decoder"])
def test_unmeasured_or_non_validation_development_cannot_be_adopted(deployment, consumer, corruption):
    report = deepcopy(deployment.report)
    if corruption == "weights":
        deployment.checkpoint.write_bytes(b"different weights at the same file path")
    elif corruption == "missing_hash":
        report["candidates"][1].pop("sha256")
    elif corruption == "final_flag":
        report["final_test"] = True
    elif corruption == "final_partition":
        report["evaluation_partition"] = "final_test"
    elif corruption == "auto_decoder":
        report["inference"]["runtime"]["runtime"]["decoder_adapter"] = "auto"
    _json(deployment.root / "report.json", report)
    with pytest.raises(ValueError):
        if consumer == "baseline":
            speech.development_baseline(deployment.run, str(deployment.checkpoint.resolve()))
        else:
            speech.freeze_deployment_selection(deployment.config, str(deployment.checkpoint.resolve()))


def test_freeze_rejects_a_different_checkpoint_even_when_it_is_in_the_same_run(deployment):
    other = deployment.run / "other.safetensors"
    other.write_bytes(b"another checkpoint")
    with pytest.raises(ValueError, match="cannot change"):
        speech.freeze_deployment_selection(deployment.config, str(other))


def test_accepted_pipeline_freezes_decoder_rate_and_knobs_without_contaminating_base(deployment):
    accepted = _accepted_pipeline(deployment)
    defaults = speech._benchmark_infer_kwargs(deployment.config)
    frozen = speech.freeze_deployment_selection(deployment.config, str(deployment.checkpoint.resolve()))
    base, selected = frozen["candidates"]
    assert base["path"] == "" and base["runtime"]["decoder_adapter"] == "none"
    assert base["speaking_rate"] == deployment.config.sample_speaking_rate == 1.0
    assert base["infer_kwargs"] == defaults
    assert selected["runtime"]["decoder_adapter"] == str(accepted.decoder.resolve())
    assert selected["runtime"]["decoder_adapter_strength"] == 0.6
    assert selected["speaking_rate"] == 0.8
    assert selected["infer_kwargs"]["latent_multiplier"] == round(defaults["latent_multiplier"] / 0.8, 4)
    assert {key: selected["infer_kwargs"][key] for key in ("temperature", "inference_cfg_rate", "num_beams")} == {
        "temperature": 1.0, "inference_cfg_rate": 0.9, "num_beams": 1}
    assert {str(path) for path in (accepted.decoder, accepted.gate, accepted.sweep, accepted.settings, accepted.rate)} <= set(frozen["artifacts"])
    speech._validate_frozen_deployment(frozen)


@pytest.mark.parametrize("artifact", ["checkpoint", "decoder", "gate", "sweep", "settings", "rate", "development_plan", "final_plan"])
def test_every_frozen_selection_input_is_checked_for_later_changes(deployment, artifact):
    accepted = _accepted_pipeline(deployment)
    frozen = speech.freeze_deployment_selection(deployment.config, str(deployment.checkpoint.resolve()))
    paths = {"checkpoint": deployment.checkpoint, "development_plan": deployment.root / "plan.json",
             "final_plan": deployment.root / "final_test" / "plan.json", **vars(accepted)}
    paths[artifact].write_bytes(paths[artifact].read_bytes() + b" changed after freeze")
    with pytest.raises(ValueError, match="frozen deployment artifact changed"):
        speech._validate_frozen_deployment(frozen)


@pytest.mark.parametrize(("field", "value"), [
    ("status", "running"), ("accepted", False), ("evaluation_partition", "final_test"),
    ("checkpoint_sha256", "stale"), ("adapter_sha256", "stale"), ("development_fingerprint", "stale"),
    ("baseline_report", "old_validation_report.json"),
])
def test_installed_decoder_needs_a_matching_completed_validation_gate(deployment, field, value):
    from indextts.lora.decoder import find_decoder_adapter

    accepted = _accepted_pipeline(deployment)
    gate = _read(accepted.gate)
    gate[field] = value
    _json(accepted.gate, gate)
    found = find_decoder_adapter(str(deployment.checkpoint.resolve()))
    if field in {"status", "accepted"}:
        assert found == ""  # AUTO suppresses a pending/rejected decoder before provenance checks.
        message = "^Adopted decoding settings do not match the validation-selected checkpoint and decoder$"
    else:
        assert Path(found) == accepted.decoder.resolve()
        message = "validation gate"
    with pytest.raises(ValueError, match=message):
        speech.freeze_deployment_selection(deployment.config, str(deployment.checkpoint.resolve()))


@pytest.mark.parametrize(("field", "value"), [
    ("checkpoint_sha256", "stale"), ("decoder_adapter_sha256", "stale"),
    ("development_fingerprint", "stale"), ("report_sha256", "stale"),
    ("decoder_adapter_strength", 1.0), ("evaluation_partition", "final_test"),
])
def test_stale_decoding_provenance_cannot_be_adopted(deployment, field, value):
    accepted = _accepted_pipeline(deployment)
    settings = _read(accepted.settings)
    settings[field] = value
    _json(accepted.settings, settings)
    with pytest.raises(ValueError):
        speech.freeze_deployment_selection(deployment.config, str(deployment.checkpoint.resolve()))


def test_adopted_knobs_must_equal_the_authenticated_complete_sweep_report(deployment):
    accepted = _accepted_pipeline(deployment)
    settings = _read(accepted.settings)
    settings["settings"]["temperature"] = 1.2
    _json(accepted.settings, settings)
    with pytest.raises(ValueError, match="complete decoding validation report"):
        speech.freeze_deployment_selection(deployment.config, str(deployment.checkpoint.resolve()))


def test_legacy_inference_settings_still_load_but_do_not_claim_a_new_final_freeze(deployment):
    legacy = {"accepted": True, "settings": {"temperature": 1.0, "inference_cfg_rate": 0.7, "num_beams": 1}, "score": 0.02}
    _json(deployment.run / "analysis" / "decoding.json", legacy)
    assert load_decoding_settings(deployment.checkpoint)["temperature"] == 1.0
    assert load_decoding_settings(deployment.checkpoint)["num_beams"] == 1
    with pytest.raises(ValueError):
        speech.freeze_deployment_selection(deployment.config, str(deployment.checkpoint.resolve()))


def _replace_expensive_calls(monkeypatch, *, regresses: bool):
    import indextts.training.grid as grid
    import indextts.training.speech_metrics as metrics
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    calls = []

    def render(config, **_kwargs):
        calls.append(config)
        destination = Path(config.output_root) / config.grid_name
        destination.mkdir(parents=True, exist_ok=True)
        cells = []
        for checkpoint in config.checkpoints:
            for index, _text in enumerate(config.texts, 1):
                for seed in config.seeds:
                    audio = destination / f"{checkpoint.label}_{index}_{seed}.wav"
                    audio.write_bytes(b"not decoded: measurement stub below")
                    cells.append(SimpleNamespace(text_index=index, audio_path=str(audio), seed=seed,
                                                 checkpoint_label=checkpoint.label, checkpoint_path=checkpoint.path))
        return SimpleNamespace(status="complete", grid_dir=str(destination), cells=cells)

    def measure(clips, **_kwargs):
        return [{**clip, **_measurement(clip["checkpoint"], clip["prompt_id"],
                                       speaker=0.3 if regresses and clip["checkpoint"] == "fresh" else 0.8)}
                for clip in clips]

    monkeypatch.setattr(grid, "run_grid", render)
    monkeypatch.setattr(metrics, "measure_clips", measure)
    return calls


def test_final_generation_uses_frozen_candidate_settings_and_does_not_reselect_on_regression(deployment, monkeypatch):
    accepted = _accepted_pipeline(deployment)
    frozen = speech.freeze_deployment_selection(deployment.config, str(deployment.checkpoint.resolve()))
    calls = _replace_expensive_calls(monkeypatch, regresses=True)
    # Runtime defaults changing after freeze must not alter either frozen candidate.
    deployment.config.sample_speaking_rate = 1.3
    deployment.config.sample_temperature = 1.4
    report = speech.run_speech_evaluation(deployment.config, deployment.root / "final_test" / "eval_job",
                                          frozen_selection=frozen["candidates"], frozen_deployment=frozen)
    assert len(calls) == 2 and all(len(call.checkpoints) == 1 for call in calls)
    assert calls[0].runtime["runtime"]["decoder_adapter"] == "none"
    assert calls[0].infer_kwargs == frozen["candidates"][0]["infer_kwargs"]
    assert calls[1].runtime["runtime"]["decoder_adapter"] == str(accepted.decoder.resolve())
    assert calls[1].runtime["runtime"]["decoder_adapter_strength"] == 0.6
    assert calls[1].infer_kwargs == frozen["candidates"][1]["infer_kwargs"]
    assert report["recommended_checkpoint"] == str(deployment.checkpoint.resolve())
    assert report["recommended_label"] == "fresh" and report["final_test_status"] == "regression detected"
    assert not next(row for row in report["candidates"] if row["label"] == "fresh")["eligible"]
    assert _read(deployment.root / "report.json")["recommended_checkpoint"] == str(deployment.checkpoint.resolve())


def test_final_report_update_preserves_frozen_evidence_and_repeat_history(deployment, monkeypatch):
    _accepted_pipeline(deployment)
    calls = _replace_expensive_calls(monkeypatch, regresses=False)
    before = (deployment.root / "report.json").read_bytes()
    fingerprint = speech.development_fingerprint(deployment.run)
    speech.run_final_test(deployment.config, deployment.root / "final_test" / "eval_job",
                          checkpoint_path=str(deployment.checkpoint.resolve()))
    selection_path = deployment.root / "final_test" / "selection_frozen.json"
    frozen = _read(selection_path)
    assert Path(frozen["development_report"]).read_bytes() == before
    assert str(deployment.root / "report.json") not in frozen["artifacts"]
    speech._validate_frozen_deployment(frozen)
    assert speech.development_fingerprint(deployment.run) == fingerprint
    assert _read(deployment.root / "report.json")["final_test_deployment_frozen"] is True
    first_selection = selection_path.read_bytes()
    speech.run_final_test(deployment.config, deployment.root / "final_test" / "eval_job",
                          checkpoint_path=str(deployment.checkpoint.resolve()))
    history = list((deployment.root / "final_test" / "history").glob("*/selection_frozen.json"))
    assert len(history) == 1 and history[0].read_bytes() == first_selection
    assert len(calls) == 4
    speech._validate_frozen_deployment(_read(selection_path))
