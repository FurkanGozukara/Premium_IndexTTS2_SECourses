from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import wave

import pytest
import torch

from indextts.training.early_stopping import EarlyStopping
from indextts.training.evaluation_plan import build_speech_plan, build_final_test_plan, representative_records
from indextts.training.dataset_manifest import write_manifest
from indextts.training.grid import GridCheckpoint, GridConfig, build_grid_cells
from indextts.training.run_guard import ensure_run_destination
from indextts.training.speech_metrics import paired_difference, select_recommendation, transcript_metrics
from indextts.training.train_config import TrainConfig
from indextts.training.trainer import reduce_learning_rate


def _config(tmp_path: Path, **kwargs) -> TrainConfig:
    return TrainConfig(dataset_dir=str(tmp_path / "dataset"), output_dir=str(tmp_path / "loras"),
                       name="fresh", device="cpu", **kwargs).validate()


def _records(tmp_path: Path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    rows = []
    for i in range(10):
        audio = dataset / f"clip{i}.wav"
        with wave.open(str(audio), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\x00\x10" * 1600)
        rows.append({"id": str(i), "audio": audio.name, "text": f"This is sentence number {i}.",
                     "speaker": "one", "language": "EN", "duration_s": 12, "source_media": f"source{i // 2}",
                     "split": "train" if i < 4 else "val"})
    return rows[:4], rows[4:]


def test_plan_is_frozen_source_balanced_and_uses_only_own_training_audio(tmp_path):
    train, val = _records(tmp_path)
    config = _config(tmp_path, speech_eval_prompts=6)
    run = Path(config.output_dir) / config.name
    plan = build_speech_plan(config, train, val, run)
    assert plan["groups"][0]["reference_record_id"] in {r["id"] for r in train}
    assert plan["validation_sources"] == 3 and not plan["source_overlap"]
    assert len(plan["groups"][0]["prompts"]) == 7  # six matched plus long-form
    assert len(plan["seeds"]) == len(set(plan["seeds"])) == 3
    assert plan == build_speech_plan(config, train, list(reversed(val)), run)
    assert len({r["source_media"] for r in representative_records(val, 3, 42)}) == 3
    changed = [{**r, "text": "Changed transcript"} for r in val]
    with pytest.raises(ValueError, match="different dataset"):
        build_speech_plan(config, train, changed, run)


def test_final_test_rejects_source_overlap_and_ignores_reference_pool_rows(tmp_path):
    train, val = _records(tmp_path)
    config = _config(tmp_path)
    config.final_test_dataset = config.dataset_dir
    write_manifest(Path(config.dataset_dir) / "manifest.jsonl", [*train, *val])
    with pytest.raises(ValueError, match="overlap"):
        build_final_test_plan(config, train, val, Path(config.output_dir) / config.name)
    test = tmp_path / "test_dataset"
    test.mkdir()
    (test / "unseen.wav").write_bytes((Path(config.dataset_dir) / train[0]["audio"]).read_bytes())
    write_manifest(test / "manifest.jsonl", [*train, {**val[0], "id": "unseen", "audio": "unseen.wav", "source_media": "unseen.flac"}])
    config.final_test_dataset = str(test)
    with pytest.raises(ValueError, match="exact copy"):
        build_final_test_plan(config, train, val, Path(config.output_dir) / config.name)
    # Change one PCM sample to create distinct fixture audio.
    raw = (test / "unseen.wav").read_bytes()
    (test / "unseen.wav").write_bytes(raw[:-2] + b"\x01\x10")
    plan = build_final_test_plan(config, train, val, Path(config.output_dir) / config.name)
    assert plan["validation_items"] == 1
    assert plan["groups"][0]["reference_record_id"] in {r["id"] for r in train}


def test_ui_respects_base_and_does_not_silently_replace_a_missing_recommendation(tmp_path, monkeypatch):
    import indextts.training.speech_eval as speech
    from ui.training_tab import recommended_generation_value
    (tmp_path / "final.safetensors").write_bytes(b"available final checkpoint")
    report = {"recommended_kind": "base", "recommended_checkpoint": ""}
    monkeypatch.setattr(speech, "load_speech_evaluation", lambda _: report)
    assert recommended_generation_value(tmp_path) == ""
    report.update(recommended_kind="adapter", recommended_checkpoint=str(tmp_path / "missing.safetensors"))
    with pytest.raises(ValueError, match="missing"):
        recommended_generation_value(tmp_path)


@pytest.mark.parametrize(("language", "text", "hypothesis", "error", "unit"), [
    ("EN", "Version 24 works.", "Version twenty four works", 0, "word"),
    ("ZH", "你好世界", "你好世", 0.25, "character"),
    ("JA", "こんにちは。", "こんにちは", 0, "character"),
    ("ES", "Hola, mundo.", "hola mundo", 0, "word"),
    ("AR", "مرحبا بالعالم", "مرحبا بالعالم", 0, "word"),
])
def test_language_aware_transcription(language, text, hypothesis, error, unit):
    row = transcript_metrics(text, hypothesis, language)
    assert row["error_rate"] == pytest.approx(error)
    assert row["error_unit"] == unit


def _measured(label, errors=0.1, speaker=0.8, failure=False):
    return [{"checkpoint": label, "prompt_id": f"p{p}", "seed": seed, "errors": errors * 20,
             "units": 20, "error_rate": errors, "speaker_similarity": speaker, "invalid_audio": False,
             "possible_truncation": failure, "possible_repetition": False, "start_matches": True,
             "end_matches": not failure} for p in range(4) for seed in [42, 104771, 209500]]


def test_base_wins_when_adapters_regress_and_low_loss_cannot_override_speech_guards():
    candidates = [{"label": "Base", "path": "", "val_loss": 6},
                  {"label": "bad", "path": "bad.safetensors", "val_loss": 2}]
    report = select_recommendation(candidates, _measured("Base") + _measured("bad", errors=.2),
                                   {"max_wer_increase": .02, "max_speaker_drop": .03})
    assert report["recommended_kind"] == "base" and report["recommended_checkpoint"] == ""
    assert not report["candidates"][1]["eligible"]
    assert report["listening_status"] == "not_rated"
    # Similar transcription with worse identity is also rejected.
    report = select_recommendation(candidates, _measured("Base") + _measured("bad", speaker=.7),
                                   {"max_wer_increase": .02, "max_speaker_drop": .03})
    assert report["recommended_kind"] == "base"
    with pytest.raises(FloatingPointError, match="refusing"):
        select_recommendation(candidates, _measured("Base") + _measured("bad", speaker=float("nan")),
                              {"max_wer_increase": .02, "max_speaker_drop": .03})


def test_paired_selection_preserves_seeds_and_uses_loss_for_unresolved_transcript_ties():
    candidates = [{"label": "Base", "path": "", "val_loss": 6},
                  {"label": "first", "path": "first.safetensors", "val_loss": 4},
                  {"label": "second", "path": "second.safetensors", "val_loss": 4.2}]
    rows = _measured("Base") + _measured("first", .02) + _measured("second", .02)
    report = select_recommendation(candidates, rows, {"max_wer_increase": .02, "max_speaker_drop": .03})
    assert report["recommended_label"] == "first"
    delta = report["candidates"][1]["error_delta_vs_base"]
    assert delta["prompts"] == 4  # twelve outputs do not become twelve independent prompts
    assert delta["mean"] == pytest.approx(-.08)
    with pytest.raises(ValueError, match="coverage"):
        paired_difference(_measured("first")[:-1], _measured("Base"), "error_rate")


def test_transcript_flags_detect_large_truncation_and_excess_repetition():
    assert transcript_metrics("one two three four five six seven", "one two", "EN")["possible_truncation"]
    assert transcript_metrics("one two three", "one two three one two three one two three", "EN")["possible_repetition"]


def test_fresh_destination_blocks_old_metrics_but_allows_explicit_continue(tmp_path):
    config = _config(tmp_path)
    ensure_run_destination(config)
    root = Path(config.output_dir) / config.name
    root.mkdir(parents=True)
    (root / "metrics.jsonl").write_text('{"step": 1}\n')
    with pytest.raises(FileExistsError, match="new name"):
        ensure_run_destination(config)
    config.resume_from, config.resume_mode = str(root / "fresh.safetensors"), "continue"
    ensure_run_destination(config)
    config.name = "new_continue_folder"
    ensure_run_destination(config)  # an explicit continuation may branch into an empty folder


def test_nearby_epoch_checks_do_not_spend_extra_patience_and_refinement_resumes():
    tracker = EarlyStopping()
    options = dict(enabled=True, patience=2, min_delta=.005, min_steps=0, min_epochs=0, check_interval=250)
    tracker.observe(5, step=6500, epoch=4, **options)
    tracker.observe(5.1, step=6984, epoch=4, **options)
    assert tracker.bad_checks == 1
    assert tracker.observe(5.1, step=7000, epoch=4.01, **options) == (False, False)
    assert tracker.bad_checks == 1 and not tracker.counted_check
    assert tracker.observe(5.1, step=7250, epoch=4.2, **options) == (False, True)
    tracker.begin_refinement(7250, 500)
    resumed = EarlyStopping.from_state(tracker.to_dict())
    assert resumed.lr_reductions == 1 and resumed.best_step == 6500
    resumed.observe(5.1, step=7500, epoch=4.3, **options)
    assert resumed.bad_checks == 0
    resumed.observe(4.99, step=7750, epoch=4.4, **options)
    assert resumed.best_step == 7750 and resumed.bad_checks == 0


def test_lr_refinement_scales_remaining_schedule_and_survives_state_restore():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / 10)
    optimizer.step()
    scheduler.step()
    reduce_learning_rate(optimizer, scheduler, .5)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(.0045)
    state, schedule = optimizer.state_dict(), scheduler.state_dict()
    new_optimizer = torch.optim.AdamW([parameter], lr=.01)
    new_scheduler = torch.optim.lr_scheduler.LambdaLR(new_optimizer, lambda step: 1 - step / 10)
    new_optimizer.load_state_dict(state)
    new_scheduler.load_state_dict(schedule)
    new_optimizer.step()
    new_scheduler.step()
    assert new_optimizer.param_groups[0]["lr"] == pytest.approx(.004)


def test_grid_seed_matrix_does_not_collide_or_duplicate_base_strengths(tmp_path):
    config = GridConfig(adapter_dir=str(tmp_path), checkpoints=[GridCheckpoint("Base", ""),
                        GridCheckpoint("voice", str(tmp_path / "voice.safetensors"))], strengths=[.5, 1],
                        references=[str(tmp_path / "ref.wav")], texts=["hello", "world"], seeds=[1, 2, 3])
    cells = build_grid_cells(config)
    assert len(cells) == 18 and len({c.filename for c in cells}) == 18
    assert {c.seed for c in cells} == {1, 2, 3}
    assert len([c for c in cells if not c.checkpoint_path]) == 6


def test_speech_pipeline_uses_only_current_run_and_maps_base_cells(tmp_path, monkeypatch):
    import indextts.training.analysis as analysis
    import indextts.training.checkpoint_eval as loss_eval
    import indextts.training.grid as grid
    import indextts.training.speech_metrics as metrics
    from indextts.training.speech_eval import load_speech_evaluation, run_speech_evaluation
    train, val = _records(tmp_path)
    config = _config(tmp_path, speech_eval_prompts=3, speech_eval_seeds=2)
    run = Path(config.output_dir) / config.name
    build_speech_plan(config, train, val, run)
    test_dataset = tmp_path / "final_dataset"
    test_dataset.mkdir()
    test_rows = []
    for i in range(3):
        audio = test_dataset / f"unseen{i}.wav"
        raw = (tmp_path / "dataset" / "clip0.wav").read_bytes()
        audio.write_bytes(raw[:-2] + bytes([i+1, 16]))
        test_rows.append({**val[i], "id": f"unseen{i}", "source_media": f"unseen{i}.flac", "audio": audio.name})
    write_manifest(test_dataset / "manifest.jsonl", test_rows)
    config.final_test_dataset = str(test_dataset)
    build_final_test_plan(config, train, val, run)
    checkpoint = run / "fresh.safetensors"
    checkpoint.write_bytes(b"current run")
    monkeypatch.setattr(analysis, "discover_checkpoints", lambda _: [dict(path=str(checkpoint), label="fresh", steps=100)])
    monkeypatch.setattr(loss_eval, "load_checkpoint_eval", lambda _: SimpleNamespace(rows=[
        SimpleNamespace(path="", kind="base", strength=1, val_loss=6),
        SimpleNamespace(path=str(checkpoint), kind="final", strength=1, val_loss=4)]))
    monkeypatch.setattr(grid, "create_tts", lambda _: object())
    generated = []
    def generate(request, _engine):
        path = Path(request["task_layout"]["final_wav_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((tmp_path / "dataset" / "clip0.wav").read_bytes())
        generated.append(request)
        return {"output_path": str(path), "audio_seconds": .1}
    monkeypatch.setattr(grid, "run_generation_request", generate)
    def measure(clips, **_kwargs):
        return [{**clip, **transcript_metrics(clip["text"], clip["text"], clip["language"]),
                 "invalid_audio": False,
                 "speaker_similarity": .5 if "final_test" in Path(clip["audio"]).parts and clip["checkpoint"] == "fresh" else .8}
                for clip in clips]
    monkeypatch.setattr(metrics, "measure_clips", measure)
    report = run_speech_evaluation(config, run / "analysis" / "speech_evaluation" / "eval_job")
    assert len(generated) == 32  # development and final test, each 4 prompts x 2 seeds x 2 models
    assert {r["lora_path"] for r in generated} == {"", str(checkpoint.resolve())}
    assert report["recommended_label"] == "fresh"
    assert len(report["real_cells"]) == 3
    assert report["final_test_status"] == "regression detected"
    frozen = json.loads((run / "analysis" / "speech_evaluation" / "final_test" / "selection_frozen.json").read_text())
    assert frozen["recommended_checkpoint"] == str(checkpoint.resolve())
    assert load_speech_evaluation(run)["recommended_checkpoint"] == str(checkpoint.resolve())
