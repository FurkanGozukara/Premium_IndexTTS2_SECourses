"""CPU-only regressions for engine initialization and per-request overrides."""

import json
from pathlib import Path
from types import SimpleNamespace
import threading
import wave

import numpy as np
import pytest
import torch
from torch import nn


def _request(tmp_path, name, **overrides):
    task = tmp_path / name
    task.mkdir()
    metadata = task / "metadata.json"
    metadata.write_text(json.dumps({"status": "in_progress", "outputs": {}, "processing": {}}), encoding="utf-8")
    result = {
        "prompt": "reference.wav", "text": "hello", "subtitle_mode": False,
        "language": "EN", "save_used_audio": False, "save_as_mp3": False,
        "mp3_bitrate": "256k", "infer_kwargs": {},
        "runtime": {"device": "cpu", "gpt_dtype": "fp32"}, "low_memory_mode": False,
        "metadata_path": str(metadata),
        "task_layout": {
            "task_folder": str(task), "final_wav_path": str(task / "final.wav"),
            "final_mp3_path": str(task / "final.mp3"), "final_mp4_path": str(task / "final.mp4"),
        },
    }
    result.update(overrides)
    return result


class _RequestEngine:
    def __init__(self, automatic_low_memory=False, explicit_policy=True):
        self.low_vram = automatic_low_memory
        if explicit_policy:
            self._runtime_low_vram = automatic_low_memory
        self.observed_low_memory = []

    def infer(self, spk_audio_prompt, text, output_path, **kwargs):
        self.observed_low_memory.append(self.low_vram)
        samples = np.zeros(2205, dtype="<i2")
        with wave.open(str(output_path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(22050)
            output.writeframes(samples.tobytes())
        self.last_generation_stats = {"seed": kwargs["seed"], "segments_count": 1, "audio_seconds": 0.1}
        return str(output_path)


@pytest.mark.parametrize("next_task", ["generation/new-task", "batch/new-run"])
def test_paused_stale_cancel_cannot_cancel_new_request_during_model_load(next_task):
    from ui.common import LazyEngine

    engine = LazyEngine()
    engine.reset_cancel(task_id="generation/old-task")
    captured, resume = threading.Event(), threading.Event()
    results = []

    def stale_confirmation():
        target = "generation/old-task"
        captured.set()
        if resume.wait(5):
            results.append(engine.request_cancel(expected_task=target))

    thread = threading.Thread(target=stale_confirmation, daemon=True)
    thread.start()
    try:
        assert captured.wait(5)
        engine.reset_cancel(task_id=next_task)
        # Simulate the next task holding its loading lock while the old UI
        # handler resumes; cancellation must neither wait on it nor target it.
        with engine._lock:
            resume.set()
            thread.join(timeout=5)
            assert not thread.is_alive()
        assert results == [False]
        engine.raise_if_canceled()
        assert engine.request_cancel(expected_task=next_task) is True
        with pytest.raises(RuntimeError, match="canceled by user"):
            engine.raise_if_canceled()
    finally:
        resume.set()
        thread.join(timeout=5)


def test_starting_new_identity_restores_reporter_and_rejects_old_cancel():
    from ui.common import LazyEngine

    class Reporter:
        def update(self):
            return "progress"

        def finish(self):
            return "finished"

    engine = LazyEngine()
    reporter = Reporter()
    engine._instance = SimpleNamespace(progress_reporter=reporter)
    engine.reset_cancel(task_id="old")
    assert engine.request_cancel(expected_task="old") is True
    with pytest.raises(RuntimeError, match="canceled by user"):
        reporter.update()
    engine.reset_cancel(task_id="new")
    assert engine.request_cancel(expected_task="old") is False
    assert reporter.update() == "progress" and reporter.finish() == "finished"
    engine.raise_if_canceled()


@pytest.mark.parametrize("automatic_low_memory", [False, True])
@pytest.mark.parametrize("explicit_policy", [False, True])
def test_reused_engine_clears_requested_low_memory_but_preserves_runtime_policy(
    tmp_path, automatic_low_memory, explicit_policy,
):
    from webui_generation_runner import run_generation_request

    engine = _RequestEngine(automatic_low_memory, explicit_policy)
    run_generation_request(_request(tmp_path, "enabled", low_memory_mode=True), engine)
    run_generation_request(_request(tmp_path, "disabled", low_memory_mode=False), engine)
    assert engine.observed_low_memory == [True, automatic_low_memory]
    assert engine._runtime_low_vram is automatic_low_memory


def test_tuning_correction_is_retained_in_result_metadata_and_progress(tmp_path):
    import shutil
    from webui_generation_runner import run_generation_request

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required")
    request = _request(tmp_path, "tuning", audio_tuning_overrides={"high_cut_hz": 12000})
    progress = []
    result = run_generation_request(request, _RequestEngine(), lambda _, desc="": progress.append(desc))
    metadata = json.loads(Path(request["metadata_path"]).read_text(encoding="utf-8"))
    warning = metadata["audio_tuning_warnings"][0]
    assert "12000" in warning and "10914.75" in warning
    assert warning in result["runtime_warning"] == metadata["runtime_warning"]
    assert any(warning in description for description in progress)
    assert metadata["status"] == "completed"


class _Projection(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False)


def _save_adapter(path, component):
    from indextts.lora.apply import inject_adapters
    from indextts.lora.io import LoraMetadata, save_lora

    model = _Projection()
    adapters = inject_adapters(model, rank=1, alpha=1.0, dropout=0.0, use_dora=False, target_modules=["proj"])
    with torch.no_grad():
        adapters["proj"].lora_A.weight.fill_(0.5)
        adapters["proj"].lora_B.weight.fill_(0.25)
    save_lora(
        path, adapters, {},
        LoraMetadata(rank=1, alpha=1.0, target_modules=["proj"], train_config={"component": component}),
        dtype=torch.float32,
    )


@pytest.mark.parametrize("choice", ["auto", "explicit_base", "none"])
def test_cold_constructor_installs_decoder_before_residency_caches_and_compile(tmp_path, monkeypatch, choice):
    """Run the actual constructor with tiny CPU models, stopping after s2mel setup."""
    from omegaconf import OmegaConf
    import indextts.infer_v2_5 as inference
    from indextts.lora.layers import LoRAAdapter
    from indextts.runtime.vram_presets import RuntimeConfig

    gpt_path = tmp_path / "voice.safetensors"
    decoder_path = tmp_path / "voice.s2mel.safetensors"
    _save_adapter(gpt_path, "gpt")
    _save_adapter(decoder_path, "s2mel")
    events = []
    has_decoder = choice != "none"

    class FakeGPT(_Projection):
        def __init__(self, **kwargs):
            super().__init__()
            self.attention_backend = kwargs["attention_backend"]

        def post_init_gpt2_config(self, **kwargs):
            pass

    class FakeEstimator(_Projection):
        def setup_caches(self, **kwargs):
            assert isinstance(self.proj, LoRAAdapter) is has_decoder
            events.append("caches")

    class FakeS2Mel(nn.Module):
        def __init__(self, _config):
            super().__init__()
            cfm = nn.Module()
            cfm.estimator = FakeEstimator()
            self.models = nn.ModuleDict({"cfm": cfm})

        def enable_torch_compile(self):
            assert isinstance(self.models["cfm"].estimator.proj, LoRAAdapter) is has_decoder
            events.append("compile")

    class FakeCodec(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

        def load_checkpoint(self, _path):
            pass

    class FakeResidency:
        def __init__(self, _device):
            pass

        def register(self, name, module, _policy):
            if name == "s2mel":
                assert isinstance(module.models["cfm"].estimator.proj, LoRAAdapter) is has_decoder
                events.append("residency")

    class DecoderInitialized(Exception):
        pass

    def model_logged(_engine, name, *_args):
        if name == "s2mel":
            raise DecoderInitialized

    config = OmegaConf.create({
        "gpt": {"stop_text_token": 0, "stop_mel_token": 1}, "gpt_checkpoint": "gpt.pth",
        "w2v_stat": "stats.pth", "semantic_codec": {}, "s2mel_checkpoint": "s2mel.pth", "s2mel": {},
    })
    (tmp_path / "hf_cache" / "w2v-bert-2.0").mkdir(parents=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf_cache"))
    monkeypatch.setenv("INDEXTTS_DECODER_GUIDANCE", "base")
    monkeypatch.setattr(inference.OmegaConf, "load", lambda _path: config)
    monkeypatch.setattr(inference, "UnifiedVoice", FakeGPT)
    monkeypatch.setattr(inference, "load_checkpoint", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(inference, "ResidencyManager", FakeResidency)
    monkeypatch.setattr(inference, "SeamlessM4TFeatureExtractor", SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: object()))
    monkeypatch.setattr(inference, "Wav2Vec2BertModel", SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: nn.Linear(2, 2)))
    monkeypatch.setattr(inference.torch, "load", lambda *_args, **_kwargs: {"mean": torch.zeros(1), "var": torch.ones(1)})
    monkeypatch.setattr(inference, "EnhancedCodec", FakeCodec)
    monkeypatch.setattr(inference, "MyModel", FakeS2Mel)
    monkeypatch.setattr(inference, "load_checkpoint2", lambda model, *_args, **_kwargs: (model, None, None, None))
    monkeypatch.setattr(inference.IndexTTS2, "_log_model", model_logged)
    runtime = RuntimeConfig(
        device="cpu", gpt_dtype="fp32", use_qwen_emo=False, torch_compile_s2mel=True,
        lora_path="" if choice == "explicit_base" else str(gpt_path), lora_strength=0.7,
        decoder_adapter=str(decoder_path) if choice == "explicit_base" else choice,
        decoder_adapter_strength=0.4,
    )
    engine = inference.IndexTTS2.__new__(inference.IndexTTS2)
    with pytest.raises(DecoderInitialized):
        engine.__init__(cfg_path="unused.yaml", model_dir=str(tmp_path), runtime=runtime)
    assert events == ["residency", "caches", "compile"]
    assert engine._runtime_low_vram is False
    cfm = engine.s2mel.models["cfm"]
    if has_decoder:
        assert engine._decoder_handle is not None
        assert Path(engine._decoder_path) == decoder_path.resolve()
        assert engine._decoder_strength == engine._decoder_handle.strength == 0.4
        assert cfm.guidance_adapter_toggle is not None
        cfm.guidance_adapter_toggle(False)
        assert cfm.estimator.proj.strength == 0.0
        cfm.guidance_adapter_toggle(True)
        assert cfm.estimator.proj.strength == 0.4
        assert next(cfm.estimator.parameters()).device.type == "cpu"
    else:
        assert engine._decoder_handle is None and engine._decoder_path == ""
        assert cfm.guidance_adapter_toggle is None
