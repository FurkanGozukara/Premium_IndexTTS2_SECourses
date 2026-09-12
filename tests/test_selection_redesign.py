"""CPU-only checks for the v6.16 selection redesign: deployment-settings comparison, interval guards,
the epoch probe with two-signal early stopping, the probe-best checkpoint, and the joint adapter+decoder choice."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import wave

import pytest

from indextts.training import speech_eval as speech
from indextts.training.analysis import _recommended_checkpoint, checkpoint_descriptor, discover_checkpoints
from indextts.training.deployment_settings import adapter_defaults, deployment_infer_kwargs, tier_decoding
from indextts.training.evaluation_plan import automatic_speech_prompt_count, build_speech_plan
from indextts.training.probe import (ProbeTracker, automatic_probe_count, build_probe_plan, decide_stop, probe_interval,
                                     probe_wer_tolerance, score_probe_rows)
from indextts.training.speaking_rate import SpeakingRateReport, write_speaking_rate
from indextts.training.speech_metrics import regression_guards, select_recommendation, source_regression
from indextts.training.train_config import TrainConfig
from indextts.training.trainer import LoraTrainer


def _config(tmp_path: Path, **kwargs) -> TrainConfig:
    kwargs.setdefault("name", "fresh")
    return TrainConfig(dataset_dir=str(tmp_path / "dataset"), output_dir=str(tmp_path / "loras"),
                       device="cpu", **kwargs).validate()


def _records(tmp_path: Path, *, train: int = 4, val: int = 6, sources: int = 3):
    dataset = tmp_path / "dataset"
    dataset.mkdir(exist_ok=True)
    rows = []
    for i in range(train + val):
        audio = dataset / f"clip{i}.wav"
        with wave.open(str(audio), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\x00\x10" * 1600)
        source = f"train{i}" if i < train else f"source{(i - train) % sources}"
        rows.append({"id": str(i), "audio": audio.name, "text": f"This is sentence number {i}.", "speaker": "one",
                     "language": "EN", "duration_s": 12, "source_media": source, "split": "train" if i < train else "val"})
    return rows[:train], rows[train:]


def _rows(label, prompts, *, error, speaker=0.8, speaker_real=0.8, seeds=(42, 104771, 209500)):
    """One measured row per prompt and seed; ``error`` maps prompt id to its error rate."""
    rows = []
    for prompt_id, source in prompts:
        for seed in seeds:
            rate = error(prompt_id) if callable(error) else error
            rows.append({"checkpoint": label, "prompt_id": prompt_id, "source": source, "seed": seed, "errors": rate * 20,
                         "units": 20, "error_rate": rate, "speaker_similarity": speaker, "speaker_similarity_real": speaker_real,
                         "invalid_audio": False, "possible_truncation": False, "possible_repetition": False,
                         "start_matches": True, "end_matches": True})
    return rows


PROMPTS = [(f"p{i}", f"recording{i // 4}") for i in range(12)]  # three recordings, four sentences each
POLICY = {"max_wer_increase": 0.02, "max_speaker_drop": 0.03}


# --- configuration -------------------------------------------------------------------------------------------------

def test_new_fields_default_validate_and_round_trip(tmp_path):
    config = _config(tmp_path)
    assert config.speech_eval_prompts == 0 and config.speech_eval_guard_mode == "interval"
    assert config.speech_eval_deployment_settings and config.decoder_adapter_always_gate and config.probe_enabled
    assert config.probe_every_epochs == 0 and config.probe_patience == 2 and config.probe_device == "auto" and config.probe_seeds == 2
    loaded = TrainConfig.from_dict(config.to_dict())
    assert loaded.to_dict() == config.to_dict()
    old = TrainConfig.from_dict({"dataset_dir": "dataset", "name": "old"})
    assert old.probe_enabled and old.speech_eval_score_wer_weight == 4.0


@pytest.mark.parametrize(("field_name", "value"), [
    ("speech_eval_guard_mode", "strict"), ("speech_eval_score_wer_weight", -1), ("probe_patience", 0),
    ("probe_seeds", 0), ("probe_wer_tolerance", 1.5), ("probe_device", "gpu1"), ("probe_every_epochs", -1),
])
def test_invalid_new_values_raise(field_name, value):
    config = TrainConfig(dataset_dir="dataset", name="adapter")
    setattr(config, field_name, value)
    with pytest.raises((TypeError, ValueError)):
        config.validate()


# --- automatic prompt count and plan policy ----------------------------------------------------------------------------

def test_automatic_prompt_count_scales_with_recordings_within_bounds():
    assert automatic_speech_prompt_count(1) == 12
    assert automatic_speech_prompt_count(3) == 18
    assert automatic_speech_prompt_count(9) == 24


def test_speech_plan_uses_the_automatic_count_and_records_the_guard_policy(tmp_path):
    train, val = _records(tmp_path, val=30, sources=3)
    config = _config(tmp_path)
    plan = build_speech_plan(config, train, val, Path(config.output_dir) / config.name)
    matched = [p for p in plan["groups"][0]["prompts"] if p["kind"] == "matched"]
    assert plan["prompt_count"] == 18 and plan["prompt_count_mode"] == "automatic" and len(matched) == 18
    assert {p["source"] for p in matched} == {"source0", "source1", "source2"}
    assert plan["policy"]["guard_mode"] == "interval" and plan["policy"]["score_wer_weight"] == 4.0
    assert plan["deployment_settings"] is True
    configured = _config(tmp_path, name="configured", speech_eval_prompts=6)
    plan = build_speech_plan(configured, train, val, Path(configured.output_dir) / configured.name)
    assert plan["prompt_count"] == 6 and plan["prompt_count_mode"] == "configured"


# --- interval guards ------------------------------------------------------------------------------------------------

def test_interval_guard_keeps_a_candidate_whose_regression_is_one_recording_within_noise():
    base = _rows("Base", PROMPTS, error=0.05)
    # Four sentences of one recording read much worse, the other eight slightly better: the mean crosses the
    # two-point margin but the prompt-bootstrap interval includes zero and two of three recordings improved.
    hot = {"p0", "p1", "p2", "p3"}
    adapter = _rows("voice", PROMPTS, error=lambda p: 0.17 if p in hot else 0.03, speaker_real=0.83)
    candidates = [{"label": "Base", "path": "", "val_loss": 6}, {"label": "voice", "path": "voice.safetensors", "val_loss": 4}]
    interval = select_recommendation(candidates, base + adapter, {**POLICY, "guard_mode": "interval"})
    voice = interval["candidates"][1]
    assert voice["error_delta_vs_base"]["mean"] > 0.02 and voice["error_delta_vs_base"]["ci95"][0] < 0
    assert voice["eligible"] and voice["notes"] and voice["error_sources"]["regressed"] == 1
    assert voice["deployment_score"]["wer_penalty"] == pytest.approx(4 * voice["error_delta_vs_base"]["mean"])
    assert interval["score_policy"]["guard_mode"] == "interval"
    mean = select_recommendation(candidates, base + adapter, {**POLICY, "guard_mode": "mean"})
    assert not mean["candidates"][1]["eligible"] and mean["recommended_label"] == "Base"


def test_interval_guard_still_rejects_a_regression_shared_by_most_recordings():
    base = _rows("Base", PROMPTS, error=0.05)
    cold = {"p0", "p4", "p8"}
    adapter = _rows("voice", PROMPTS, error=lambda p: 0.03 if p in cold else 0.10, speaker_real=0.83)
    report = select_recommendation([{"label": "Base", "path": "", "val_loss": 6}, {"label": "voice", "path": "v.safetensors", "val_loss": 4}],
                                   base + adapter, {**POLICY, "guard_mode": "interval"})
    voice = report["candidates"][1]
    assert not voice["eligible"] and "transcript error" in voice["rejection_reasons"][0]
    assert voice["error_sources"]["majority"] and report["recommended_label"] == "Base"


def test_source_regression_counts_recordings_in_the_bad_direction():
    base = _rows("Base", PROMPTS, error=0.05)
    adapter = _rows("voice", PROMPTS, error=lambda p: 0.10 if p in {"p0", "p1", "p2", "p3", "p4"} else 0.05)
    result = source_regression(adapter, base, "error_rate", worse_when_higher=True)
    assert result["sources"] == 3 and result["regressed"] == 2 and result["majority"]
    guards = regression_guards(adapter, base, policy={**POLICY, "guard_mode": "interval"}, speaker_metric="speaker_similarity_real")
    assert guards["reasons"] and guards["guard_mode"] == "interval"


def test_score_weight_from_the_policy_changes_the_word_error_penalty():
    base = _rows("Base", PROMPTS, error=0.05)
    adapter = _rows("voice", PROMPTS, error=0.06, speaker_real=0.86)
    candidates = [{"label": "Base", "path": "", "val_loss": 6}, {"label": "voice", "path": "v.safetensors", "val_loss": 4}]
    heavy = select_recommendation(candidates, base + adapter, {**POLICY, "score_wer_weight": 10.0})
    light = select_recommendation(candidates, base + adapter, {**POLICY, "score_wer_weight": 1.0})
    assert heavy["candidates"][1]["deployment_score"]["wer_penalty"] == pytest.approx(0.1)
    assert light["candidates"][1]["deployment_score"]["wer_penalty"] == pytest.approx(0.01)
    assert heavy["recommended_label"] == "Base" and light["recommended_label"] == "voice"


# --- deployment settings --------------------------------------------------------------------------------------------

def _adapter_with_profile(tmp_path: Path) -> Path:
    adapter_dir = tmp_path / "loras" / "fresh"
    (adapter_dir / "analysis").mkdir(parents=True, exist_ok=True)
    profile = {"version": 2, "duration_s": {"median": 12.0}, "language": "EN",
               "recommendation": {"target_tokens": 40, "budget_tokens": 46},
               "pauses": {"recommendation": {"sentence_pause_ms": 310, "max_pause_ms": 440}}}
    (adapter_dir / "analysis" / "dataset_profile.json").write_text(json.dumps(profile), encoding="utf-8")
    (adapter_dir / "fresh_expressive_reference.wav").write_bytes(b"expressive fixture")
    write_speaking_rate(adapter_dir, SpeakingRateReport(recommended_speaking_rate=0.8, dataset_words_per_second=3,
                                                        generated_words_per_second=3.75, clips_used=4, method="speech_matched",
                                                        generated_at="2026-09-12T00:00:00Z", summary="fixture"))
    checkpoint = adapter_dir / "fresh.safetensors"
    checkpoint.write_bytes(b"adapter fixture")
    return checkpoint


def test_deployment_settings_follow_the_adapter_profile_and_leave_base_at_language_defaults(tmp_path):
    config = _config(tmp_path)
    checkpoint = _adapter_with_profile(tmp_path)
    adapter = deployment_infer_kwargs(config, checkpoint, language="EN", tier=32)
    base = deployment_infer_kwargs(config, "", language="EN", tier=32)
    assert adapter["num_beams"] == 4 and adapter["diffusion_steps"] == 50 and adapter["cfm_temperature"] == 0.9
    assert adapter["segmentation_mode"] == "smart" and adapter["segment_target_tokens"] == 40
    assert (adapter["sentence_pause_ms"], adapter["max_pause_ms"]) == (310, 440)
    assert adapter["emo_audio_prompt"].endswith("fresh_expressive_reference.wav")
    assert adapter["latent_multiplier"] == pytest.approx(round(1.72 / 0.8, 4))
    assert adapter["max_text_tokens_per_segment"] > 60
    assert base["emo_audio_prompt"] is None and base["segment_target_tokens"] is None
    assert (base["sentence_pause_ms"], base["max_pause_ms"]) == (0, 0) and base["max_text_tokens_per_segment"] == 60
    assert base["latent_multiplier"] == pytest.approx(1.72) and base["num_beams"] == 4
    assert adapter_defaults(None)["speaking_rate"] == 1.0
    assert tier_decoding(16)["diffusion_steps"] == 40 and tier_decoding("auto")["num_beams"] == 4


def test_speech_comparison_renders_base_and_adapters_in_their_own_deployment_batches(tmp_path, monkeypatch):
    import indextts.training.analysis as analysis
    import indextts.training.checkpoint_eval as loss_eval
    import indextts.training.grid as grid
    import indextts.training.speech_metrics as metrics
    from indextts.training.speech_metrics import transcript_metrics
    train, val = _records(tmp_path)
    config = _config(tmp_path, speech_eval_prompts=3, speech_eval_seeds=1)
    run = Path(config.output_dir) / config.name
    build_speech_plan(config, train, val, run)
    checkpoint = _adapter_with_profile(tmp_path)
    monkeypatch.setattr(speech, "_benchmark_runtime", lambda _config: __import__("indextts.runtime", fromlist=["RuntimeConfig"]).RuntimeConfig(device="cpu"))
    monkeypatch.setattr(analysis, "discover_checkpoints", lambda _: [dict(path=str(checkpoint), label="fresh", steps=100, kind="final")])
    monkeypatch.setattr(loss_eval, "load_checkpoint_eval", lambda _: SimpleNamespace(rows=[
        SimpleNamespace(path="", kind="base", strength=1, val_loss=6),
        SimpleNamespace(path=str(checkpoint), kind="final", strength=1, val_loss=4)]))
    monkeypatch.setattr(grid, "create_tts", lambda _: object())
    requests = []

    def generate(request, _engine):
        path = Path(request["task_layout"]["final_wav_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((tmp_path / "dataset" / "clip0.wav").read_bytes())
        requests.append(request)
        return {"output_path": str(path), "audio_seconds": .1}

    monkeypatch.setattr(grid, "run_generation_request", generate)
    monkeypatch.setattr(metrics, "measure_clips", lambda clips, **_: [
        {**clip, **transcript_metrics(clip["text"], clip["text"], clip["language"]), "invalid_audio": False,
         "speaker_similarity": .8, "speaker_similarity_real": .8 if clip["checkpoint"] == "Base" else .85} for clip in clips])
    report = speech.run_speech_evaluation(config, run / "analysis" / "speech_evaluation" / "eval_job")
    base_requests = [r for r in requests if not r["lora_path"]]
    adapter_requests = [r for r in requests if r["lora_path"]]
    assert base_requests and adapter_requests
    assert all(r["segmentation_mode"] == "smart" and r["segment_target_tokens"] == 40 for r in adapter_requests)
    assert all(r["sentence_pause_ms"] == 310 and r["max_pause_ms"] == 440 for r in adapter_requests)
    assert all(r["infer_kwargs"]["emo_audio_prompt"] and r["infer_kwargs"]["emo_audio_prompt"].endswith("fresh_expressive_reference.wav") for r in adapter_requests)
    assert all(r["infer_kwargs"]["emo_audio_prompt"] is None and r["segment_target_tokens"] is None for r in base_requests)
    assert all(r["infer_kwargs"]["num_beams"] == 4 and r["infer_kwargs"]["diffusion_steps"] == 50 for r in requests)
    assert report["deployment_settings"] is True and report["recommended_label"] == "fresh"
    # The report's inference block describes the adapters' render; the decoder gate re-renders with it.
    assert report["inference"]["infer_kwargs"]["segment_target_tokens"] == 40
    assert report["inference"]["runtime"]["runtime"]["decoder_adapter"] == "none"
    assert report["candidate_inference"]["Base"]["infer_kwargs"]["emo_audio_prompt"] is None
    assert report["candidate_inference"]["fresh"]["infer_kwargs"]["emo_audio_prompt"]
    speech.development_baseline(run, str(checkpoint.resolve()))


@pytest.mark.parametrize("language,expected", [("EN", 60), ("ES", 60), ("AR", 80), ("JA", 100), ("ZH", 120)])
def test_deployment_base_uses_the_selected_language_token_limit(language, expected):
    assert adapter_defaults(None, language=language)["max_text_tokens_per_segment"] == expected


def test_probe_tier_uses_actual_free_memory_on_a_separate_gpu(monkeypatch):
    import indextts.runtime as runtime
    from indextts.runtime import vram_presets
    from indextts.training.probe_worker import probe_runtime

    monkeypatch.setattr(runtime, "gpu_total_gb", lambda _index: 24.0)
    monkeypatch.setattr(runtime, "gpu_free_gb", lambda _index: 7.0)
    fitted = []
    monkeypatch.setattr(vram_presets, "fit_tier_to_free_vram", lambda tier, free: fitted.append((tier, free)) or 8)
    monkeypatch.setattr(runtime, "resolve_preset", lambda *_args: SimpleNamespace())
    result = probe_runtime(SimpleNamespace(sample_runtime_tier="24", vram_tier="24"), "cuda:1", share_gpu=False)
    assert fitted == [("24", 7.0)] and result.device == "cuda:1"


# --- probe ------------------------------------------------------------------------------------------------------------

def test_probe_count_tolerance_and_interval_helpers():
    assert automatic_probe_count(1, 20) == 6 and automatic_probe_count(3, 20) == 9 and automatic_probe_count(9, 20) == 12
    assert automatic_probe_count(3, 2) == 2
    assert probe_wer_tolerance(0.0, 0.004) == 0.01 and probe_wer_tolerance(0.0, 0.03) == 0.03 and probe_wer_tolerance(0.05, 0.03) == 0.05
    assert probe_interval(0, 60.0, 600.0) == 1  # a minute per probe against ten-minute epochs
    assert probe_interval(0, 180.0, 120.0) == 5  # three minutes per probe against two-minute epochs
    assert probe_interval(3, 180.0, 120.0) == 3 and probe_interval(0, None, 120.0) == 1


def test_probe_plan_is_frozen_balanced_and_seeded(tmp_path):
    train, val = _records(tmp_path, val=9, sources=3)
    config = _config(tmp_path, probe_seeds=2)
    run = Path(config.output_dir) / config.name
    reference = tmp_path / "dataset" / train[0]["audio"]
    plan = build_probe_plan(config, val, run, reference)
    assert len(plan["prompts"]) == 9 and {p["source"] for p in plan["prompts"]} == {"source0", "source1", "source2"}
    assert len(plan["seeds"]) == 2 and plan["reference"] == str(reference.resolve())
    assert plan == build_probe_plan(config, list(reversed(val)), run, reference)
    with pytest.raises(ValueError, match="different dataset"):
        build_probe_plan(config, [{**row, "text": "changed"} for row in val], run, reference)
    assert build_probe_plan(config, val, run / "other", tmp_path / "missing.wav") is None


def test_probe_tracker_keeps_the_best_and_flags_overfitting_only_with_a_stalled_score():
    tracker = ProbeTracker()
    kwargs = dict(tolerance=0.01, min_delta=0.002, patience=2)
    assert tracker.observe(epoch=1, score=0.010, wer=0.05, **kwargs) == (True, False)
    assert tracker.observe(epoch=2, score=0.030, wer=0.04, **kwargs) == (True, False)
    # A repeated epoch is ignored; a tiny gain does not reset the stall count.
    assert tracker.observe(epoch=2, score=0.031, wer=0.04, **kwargs) == (False, False)
    assert tracker.observe(epoch=3, score=0.031, wer=0.06, **kwargs) == (False, False)
    assert tracker.stalled_epochs == 1 and tracker.degraded_epochs == 1 and not tracker.stalled(2)
    improved, stop = tracker.observe(epoch=4, score=0.020, wer=0.07, **kwargs)
    assert not improved and stop and tracker.stalled(2) and "word error" in tracker.reason
    assert tracker.best_epoch == 2 and tracker.best_score == pytest.approx(0.030)
    restored = ProbeTracker.from_state(json.loads(json.dumps(tracker.to_dict())))
    assert restored.best_epoch == 2 and restored.history == tracker.history
    # Degrading word error while the score still climbs is not overfitting.
    climbing = ProbeTracker()
    climbing.observe(epoch=1, score=0.01, wer=0.04, **kwargs)
    climbing.observe(epoch=2, score=0.03, wer=0.06, **kwargs)
    _, stop = climbing.observe(epoch=3, score=0.05, wer=0.08, **kwargs)
    assert not stop and climbing.degraded_epochs == 2 and climbing.stalled_epochs == 0


def test_two_signal_stop_defers_a_loss_stall_while_the_probe_improves():
    tracker = ProbeTracker()
    kwargs = dict(tolerance=0.01, min_delta=0.002, patience=2)
    assert decide_stop(True, tracker, enabled=True, patience=2) == (True, "")  # no probe yet: the loss decides
    assert decide_stop(True, None, enabled=True, patience=2) == (True, "")
    tracker.observe(epoch=1, score=0.01, wer=0.05, **kwargs)
    tracker.observe(epoch=2, score=0.03, wer=0.05, **kwargs)
    stop, note = decide_stop(True, tracker, enabled=True, patience=2)
    assert not stop and "still improves" in note
    assert decide_stop(True, tracker, enabled=False, patience=2) == (True, "")
    tracker.observe(epoch=3, score=0.030, wer=0.05, **kwargs)
    tracker.observe(epoch=4, score=0.029, wer=0.05, **kwargs)
    assert decide_stop(False, tracker, enabled=True, patience=2) == (False, "")  # probe stalled alone never stops
    stop, note = decide_stop(True, tracker, enabled=True, patience=2)
    assert stop and "both stalled" in note
    overfit = ProbeTracker()
    overfit.observe(epoch=1, score=0.03, wer=0.04, **kwargs)
    overfit.observe(epoch=2, score=0.02, wer=0.06, **kwargs)
    overfit.observe(epoch=3, score=0.01, wer=0.07, **kwargs)
    stop, note = decide_stop(False, overfit, enabled=True, patience=2)
    assert stop and note.startswith("stopping")


def test_probe_scoring_pairs_with_base_and_uses_real_similarity():
    prompts = PROMPTS[:4]
    base = _rows("Base", prompts, error=0.05, speaker_real=0.80, seeds=(7,))
    rows = _rows("epoch 3", prompts, error=0.04, speaker_real=0.86, seeds=(7,))
    scored = score_probe_rows(rows, base, {"score_wer_weight": 4.0})
    assert scored["speaker_metric"] == "speaker_similarity_real"
    assert scored["deployment_score"]["score"] == pytest.approx(0.06) and scored["deployment_score"]["wer_penalty"] == 0.0


def test_probe_device_prefers_a_free_second_gpu_and_falls_back_to_the_training_gpu(monkeypatch):
    import indextts.runtime as runtime
    from indextts.training.probe_worker import resolve_probe_device
    gpus = [SimpleNamespace(index=0, name="Training", total_gb=32, free_gb=3), SimpleNamespace(index=1, name="Spare", total_gb=24, free_gb=20)]
    monkeypatch.setattr(runtime, "list_gpus", lambda: gpus)
    monkeypatch.setattr(runtime, "gpu_free_gb", lambda index: {0: 3.0, 1: 20.0}[int(index)])
    config = TrainConfig(dataset_dir="dataset", name="adapter", device="cuda:0", probe_device="auto").validate()
    device, shares, note = resolve_probe_device(config)
    assert device == "cuda:1" and not shares and "Spare" in note
    config.probe_device = "same"
    assert resolve_probe_device(config) == ("cuda:0", True, "")
    config.probe_device = "auto"
    monkeypatch.setattr(runtime, "gpu_free_gb", lambda index: {0: 3.0, 1: 2.0}[int(index)])
    assert resolve_probe_device(config) == ("cuda:0", True, "")
    config.probe_device = "cuda:1"
    assert resolve_probe_device(config)[:2] == ("cuda:1", False)


# --- probe-best checkpoint ------------------------------------------------------------------------------------------

def test_probe_best_file_is_its_own_kind_and_never_the_loss_recommendation(tmp_path, monkeypatch):
    import indextts.training.analysis as analysis
    run = tmp_path / "fresh"
    (run / "best").mkdir(parents=True)
    for name in ("fresh_best", "fresh_probe_best"):
        (run / "best" / f"{name}.safetensors").write_bytes(b"fixture")
    (run / "fresh_epoch_003.safetensors").write_bytes(b"fixture")
    (run / "fresh.safetensors").write_bytes(b"fixture")
    monkeypatch.setattr(analysis, "inspect_lora", lambda path: {"adapter_type": "dora", "epochs": 3 if "probe" in str(path) else 0, "steps": 300 if "probe" in str(path) else 0})
    probe = checkpoint_descriptor(run / "best" / "fresh_probe_best.safetensors")
    assert probe["kind"] == "probe_best" and probe["label"].startswith("probe-best (epoch 3") and probe["file_label"] == "probe_best_ep3"
    assert checkpoint_descriptor(run / "best" / "fresh_best.safetensors")["kind"] == "best"
    found = discover_checkpoints(run)
    assert [item["kind"] for item in found][:2] == ["best", "probe_best"]
    path, label = _recommended_checkpoint(run, found, best_epoch=3, final_epoch=5)
    assert path.endswith("fresh_best.safetensors") and label.startswith("best/")


def test_shortlist_always_judges_the_probe_best_update(tmp_path, monkeypatch):
    import indextts.training.analysis as analysis
    import indextts.training.checkpoint_eval as loss_eval
    run = tmp_path / "fresh"
    (run / "best").mkdir(parents=True)
    files = {}
    for name, steps in (("fresh_epoch_001", 100), ("fresh_epoch_002", 200), ("fresh_epoch_003", 300), ("fresh", 400)):
        files[name] = run / f"{name}.safetensors"
        files[name].write_bytes(b"x")
    files["probe"] = run / "best" / "fresh_probe_best.safetensors"
    files["probe"].write_bytes(b"x")
    descriptors = [dict(path=str(files["fresh_epoch_001"]), label="epoch 1", steps=100, kind="epoch"),
                   dict(path=str(files["fresh_epoch_002"]), label="epoch 2", steps=200, kind="epoch"),
                   dict(path=str(files["fresh_epoch_003"]), label="epoch 3", steps=300, kind="epoch"),
                   dict(path=str(files["fresh"]), label="final", steps=400, kind="final"),
                   dict(path=str(files["probe"]), label="probe-best (epoch 1)", steps=100, kind="probe_best")]
    monkeypatch.setattr(analysis, "discover_checkpoints", lambda _: descriptors)
    losses = {str(files["fresh_epoch_001"].resolve()): 5.0, str(files["fresh_epoch_002"].resolve()): 4.0,
              str(files["fresh_epoch_003"].resolve()): 3.9, str(files["fresh"].resolve()): 4.2, str(files["probe"].resolve()): 5.0}
    monkeypatch.setattr(loss_eval, "load_checkpoint_eval", lambda _: SimpleNamespace(rows=[
        SimpleNamespace(path="", kind="base", strength=1, val_loss=6),
        *[SimpleNamespace(path=path, kind="epoch", strength=1, val_loss=loss) for path, loss in losses.items()]]))
    shortlist = speech.shortlist_checkpoints(run, 2)
    steps = [item["steps"] for item in shortlist]
    assert steps[0] == 0 and 300 in steps and 400 in steps and 100 in steps  # best losses, the latest, and the probe-best update
    assert 200 not in steps and len(shortlist) == 4


# --- joint adapter + decoder choice -------------------------------------------------------------------------------------

def _development_report(tmp_path: Path):
    """A development report where the plain adapter loses to Base on word error."""
    run = tmp_path / "loras" / "fresh"
    root = run / "analysis" / "speech_evaluation"
    root.mkdir(parents=True)
    checkpoint = run / "fresh.safetensors"
    checkpoint.write_bytes(b"adapter fixture")
    other = run / "fresh_epoch_001.safetensors"
    other.write_bytes(b"other adapter fixture")
    base = _rows("Base", PROMPTS, error=0.05, speaker_real=0.80)
    adapter = _rows("fresh", PROMPTS, error=0.09, speaker_real=0.82)
    weaker = _rows("epoch 1", PROMPTS, error=0.10, speaker_real=0.81)
    candidates = [{"label": "Base", "path": "", "steps": 0, "val_loss": 6},
                  {"label": "fresh", "path": str(checkpoint.resolve()), "steps": 200, "val_loss": 4, "sha256": speech._file_sha256(checkpoint)},
                  {"label": "epoch 1", "path": str(other.resolve()), "steps": 100, "val_loss": 4.5, "sha256": speech._file_sha256(other)}]
    policy = {**POLICY, "guard_mode": "interval", "score_wer_weight": 4.0}
    report = select_recommendation(candidates, base + adapter + weaker, policy)
    assert report["recommended_label"] == "Base"
    report.update(cells=base + adapter + weaker, real_cells=[], evaluation_partition="validation", dataset_identity="fixture",
                  inference={"runtime": {"runtime": {"decoder_adapter": "none"}}, "infer_kwargs": {}}, warnings=[],
                  final_test_status="pending deployment freeze", candidate_inference={})
    report["summary_markdown"] = speech.report_markdown(report)
    (root / "plan.json").write_text(json.dumps({"policy": policy, "groups": [], "seeds": [42, 104771, 209500]}), encoding="utf-8")
    (root / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return run, checkpoint, report


def test_best_adapter_candidate_prefers_eligible_then_score(tmp_path):
    run, checkpoint, report = _development_report(tmp_path)
    best = speech.best_adapter_candidate(report)
    assert best["label"] == "fresh"
    assert speech.best_adapter_candidate({"candidates": [{"label": "Base", "path": ""}]}) is None


def test_joint_choice_recommends_adapter_plus_decoder_when_the_whole_deployment_beats_base(tmp_path):
    run, checkpoint, report = _development_report(tmp_path)
    decoder = run / "fresh.s2mel.safetensors"
    decoder.write_bytes(b"decoder fixture")
    with_decoder = _rows("fresh", PROMPTS, error=0.05, speaker_real=0.88)
    gate = {"status": "complete", "accepted": True, "checkpoint": str(checkpoint.resolve()), "adapter": str(decoder.resolve()),
            "strength": 1.0, "cells": with_decoder, "checkpoint_label": "fresh"}
    fingerprint = speech.development_fingerprint(run)
    joint = speech.apply_joint_recommendation(run, gate)
    assert joint["with_decoder"] and joint["changed"] and joint["recommended_checkpoint"] == str(checkpoint.resolve())
    assert joint["recommended_label"] == "fresh + voice decoder" and joint["decoder"] == str(decoder.resolve())
    assert speech.development_fingerprint(run) == fingerprint
    updated = json.loads((run / "analysis" / "speech_evaluation" / "report.json").read_text(encoding="utf-8"))
    assert updated["recommended_checkpoint"] == str(checkpoint.resolve()) and updated["recommended_kind"] == "adapter"
    assert updated["recommended_label"] == "fresh" and updated["recommended_deployment"]["decoder"] == str(decoder.resolve())
    assert updated["candidates"] == report["candidates"] and "voice decoder" in updated["summary_markdown"]
    assert speech.load_speech_evaluation(run)["recommended_checkpoint"] == str(checkpoint.resolve())
    # A decoder that improves the adapter but still loses to Base leaves the recommendation alone.
    run2, checkpoint2, _ = _development_report(tmp_path / "second")
    decoder2 = run2 / "fresh.s2mel.safetensors"
    decoder2.write_bytes(b"decoder fixture")
    weak = _rows("fresh", PROMPTS, error=0.09, speaker_real=0.83)
    joint2 = speech.apply_joint_recommendation(run2, {**gate, "checkpoint": str(checkpoint2.resolve()), "adapter": str(decoder2.resolve()), "cells": weak})
    assert not joint2["with_decoder"] and joint2["recommended_label"] == "Base" and not joint2["changed"]
    assert json.loads((run2 / "analysis" / "speech_evaluation" / "report.json").read_text(encoding="utf-8"))["recommended_checkpoint"] == ""


class _FakeProcess:
    stdout = None

    def __init__(self, returncode=0):
        self.returncode = returncode

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


@pytest.fixture
def trainer(tmp_path):
    item = object.__new__(LoraTrainer)
    item.config = TrainConfig(dataset_dir=str(tmp_path / "dataset"), output_dir=str(tmp_path), name="fresh", device="cpu").validate()
    item.adapter_dir = tmp_path / "fresh"
    item.adapter_dir.mkdir()
    item.status_path = item.adapter_dir / "status.json"
    item.stop_path = item.adapter_dir / "stop.flag"
    item.speech_plan_ready = True
    item.log = Mock()
    state = {"speech_evaluation_status": "complete"}

    def status(**updates):
        state.update(updates)
        item.status_path.write_text(json.dumps(state), encoding="utf-8")
        return dict(state)

    item.write_status = status
    item.write_status()
    item.checkpoint = item.adapter_dir / "fresh.safetensors"
    item.checkpoint.write_bytes(b"GPT")
    item.decoder = item.adapter_dir / "fresh.s2mel.safetensors"
    return item


def test_decoder_is_gated_with_the_best_adapter_when_base_led_and_the_joint_winner_becomes_the_recommendation(monkeypatch, trainer):
    from indextts.training import decoder_adapter, trainer as trainer_module

    def launch(*_args, **_kwargs):
        trainer.decoder.write_bytes(b"decoder evidence")
        return _FakeProcess(0)

    monkeypatch.setattr(trainer_module.subprocess, "Popen", launch)
    monkeypatch.setattr(decoder_adapter, "load_decoder_report", lambda _: {"status": "complete", "accepted": True, "steps": 10,
                                                                          "best_val_loss": 1.0, "best_identity": 0.9, "initial_identity": 0.8})
    report = {"status": "complete", "candidates": [{"label": "Base", "path": "", "deployment_score": {"score": 0.0}, "eligible": True},
                                                   {"label": "fresh", "path": str(trainer.checkpoint), "val_loss": 4,
                                                    "deployment_score": {"score": -0.05}, "eligible": False}]}
    monkeypatch.setattr(speech, "load_speech_evaluation", lambda _: report)
    verdict = {"accepted": True, "reasons": [], "speaker_gain": {"mean": 0.07}, "wer_increase": -0.001, "strength": 1.0,
               "checkpoint": str(trainer.checkpoint), "adapter": str(trainer.decoder)}
    trainer._run_decoder_test = Mock(return_value=verdict)
    joint = {"with_decoder": True, "changed": True, "recommended_checkpoint": str(trainer.checkpoint), "recommended_kind": "adapter",
             "recommended_label": "fresh + voice decoder", "decoder": str(trainer.decoder), "decoder_strength": 1.0,
             "decision": "adapter + decoder scored best"}
    monkeypatch.setattr(speech, "apply_joint_recommendation", lambda _run, _verdict: joint)
    trainer._write_speech_matched_speaking_rate = Mock()
    result = trainer._run_guarded_decoder_adaptation(terminal_phase="post_training", terminal_message="checks pending",
                                                     recommended_checkpoint="")
    trainer._run_decoder_test.assert_called_once()
    assert trainer._run_decoder_test.call_args.args[1] == str(trainer.checkpoint)
    assert result == str(trainer.checkpoint) and trainer.decoder.exists()
    state = json.loads(trainer.status_path.read_text())
    assert state["decoder_adapter_status"] == "complete" and state["decoder_adapter_path"] == str(trainer.decoder.resolve())
    assert state["recommended_checkpoint"] == str(trainer.checkpoint) and state["recommended_kind"] == "adapter"
    assert state["recommended_deployment"]["decoder"] == str(trainer.decoder)
    trainer._write_speech_matched_speaking_rate.assert_called_once()
    # With the option off, a Base recommendation still skips the gate and quarantines the adapter.
    trainer.config.decoder_adapter_always_gate = False
    trainer._run_decoder_test = Mock(return_value=None)
    assert trainer._run_guarded_decoder_adaptation(terminal_phase="post_training", terminal_message="checks pending",
                                                   recommended_checkpoint="") == ""
    trainer._run_decoder_test.assert_called_once_with(trainer.decoder, "")


def test_confirm_stop_uses_the_probe_tracker_of_the_trainer(trainer):
    trainer.probe_tracker = ProbeTracker()
    trainer.probe_plan_ready = True
    assert trainer._confirm_stop(True) == (True, "")
    trainer.probe_tracker.observe(epoch=1, score=0.01, wer=0.05, tolerance=0.01, min_delta=0.002, patience=2)
    trainer.probe_tracker.observe(epoch=2, score=0.03, wer=0.05, tolerance=0.01, min_delta=0.002, patience=2)
    stop, note = trainer._confirm_stop(True)
    assert not stop and "continues" in note
    trainer.config.probe_stop_enabled = False
    assert trainer._confirm_stop(True) == (True, "")
    trainer.config.probe_stop_enabled = True
    assert trainer._probe_due(3) is True
    trainer.probe_tracker.plan_next(epoch=3, configured_interval=0, probe_elapsed_s=300.0, epoch_elapsed_s=120.0)
    assert trainer.probe_tracker.interval == 8 and not trainer._probe_due(4) and trainer._probe_due(11)


def test_probe_is_skipped_before_a_worker_starts_when_the_training_gpu_has_no_room(trainer, monkeypatch):
    import indextts.training.probe_worker as worker
    from indextts.training import trainer as trainer_module
    trainer.probe_tracker = ProbeTracker()
    trainer.probe_plan_ready = True
    trainer.config.probe_every_epochs = 1
    monkeypatch.setattr(worker, "resolve_probe_device", lambda config: ("cuda:0", True, ""))
    monkeypatch.setattr(trainer_module, "gpu_free_gb", lambda index: 2.9)
    monkeypatch.setattr(trainer_module.subprocess, "Popen", Mock(side_effect=AssertionError("no worker must start")))
    trainer._run_epoch_probe(trainer.checkpoint, 1, step=10, epoch_elapsed_s=100.0)
    state = json.loads(trainer.status_path.read_text())
    assert state["probe_status"] == "skipped" and "below the 6.0 GB" in state["probe_message"]
    assert trainer.probe_tracker.skipped == 1 and trainer.probe_tracker.epochs == 0 and trainer.probe_tracker.last_probe_epoch == 1
    # Another GPU with room is never gated by the training GPU's free memory.
    monkeypatch.setattr(worker, "resolve_probe_device", lambda config: ("cuda:1", False, "second GPU"))
    started = {}
    monkeypatch.setattr(trainer_module.subprocess, "Popen", Mock(side_effect=lambda *a, **k: started.setdefault("launched", True) and (_ for _ in ()).throw(RuntimeError("stop here"))))
    with pytest.raises(RuntimeError, match="stop here"):
        trainer._run_epoch_probe(trainer.checkpoint, 2, step=20, epoch_elapsed_s=100.0)
    assert started["launched"]
