"""Shared pipeline and model isolation checks without downloading or loading weights."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ui.model_profiles import capture_profile, switch_profile
from ui.presets_store import PresetRegistry


def registry():
    result = PresetRegistry()
    for key, default in {
        "app.model": "indextts", "app.profiles": {}, "runtime.device": "cuda:0",
        "runtime.lora_path": "", "runtime.decoder_adapter": "auto",
        "generation.speaking_rate": 1.0, "generation.max_text_tokens_per_segment": 60,
        "omnivoice.num_step": 32,
    }.items():
        result.register(key, default=default)
    return result


def test_switch_roundtrip_preserves_both_models_and_adapter_isolation():
    controls = registry()
    index = capture_profile({**controls.defaults(), "runtime.lora_path": "index.safetensors",
                             "generation.speaking_rate": 1.05, "runtime.decoder_adapter": "none"})
    omni = switch_profile(controls, "omnivoice", index)
    assert omni["runtime.lora_path"] == ""
    assert omni["generation.max_text_tokens_per_segment"] == 120
    omni.update({"runtime.lora_path": "omni.safetensors", "omnivoice.num_step": 48})
    restored = switch_profile(controls, "indextts", omni)
    assert restored["runtime.lora_path"] == "index.safetensors"
    assert restored["generation.speaking_rate"] == 1.05
    # Controls that exist for one model keep their single value across switches.
    assert restored["runtime.decoder_adapter"] == "none"
    assert restored["omnivoice.num_step"] == 48
    restored = switch_profile(controls, "omnivoice", restored)
    assert restored["runtime.lora_path"] == "omni.safetensors"
    assert restored["omnivoice.num_step"] == 48


def test_legacy_preset_is_index_and_unknown_model_fails():
    values = capture_profile({"generation.speaking_rate": 1.05})
    assert values["app.model"] == "indextts"
    with pytest.raises(ValueError, match="Unknown speech model"):
        switch_profile(registry(), "typo", values)


def test_reference_free_request_records_model_and_options(tmp_path):
    from ui.generation_tab import prepare_generation_request
    from ui.common import read_json
    request = prepare_generation_request({"app.model": "omnivoice", "omnivoice.mode": "auto",
        "omnivoice.num_step": 16, "generation.apply_pronunciation_dictionary": False},
        prompt="", text="Test voice.", subtitle_file=None, image_path=None,
        emotion_audio=None, model_dir=str(tmp_path), output_root=tmp_path / "out")
    assert request["runtime"]["tts_model"] == "omnivoice"
    assert request["omnivoice"]["num_step"] == 16
    metadata = read_json(request["metadata_path"])
    assert metadata["inputs"]["speaker_reference_audio"] is None
    assert metadata["settings"]["request_values"]["app.model"] == "omnivoice"


def test_index_and_omnivoice_clone_still_require_reference(tmp_path):
    from ui.generation_tab import prepare_generation_request
    for values in ({}, {"app.model": "omnivoice", "omnivoice.mode": "clone"}):
        with pytest.raises(ValueError, match="Reference Voice"):
            prepare_generation_request(values, prompt="", text="Test.", subtitle_file=None,
                image_path=None, emotion_audio=None, model_dir=str(tmp_path), output_root=tmp_path)


def test_shared_pause_tags_are_not_spoken_and_output_is_24k(tmp_path):
    from indextts.backends.omnivoice import OmniVoiceEngine

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.text_tokenizer = SimpleNamespace(encode=lambda text, **kwargs: text.split())
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return [np.full(2400, .125, dtype=np.float32)]

    engine = OmniVoiceEngine.__new__(OmniVoiceEngine)
    engine.model, engine.device, engine.progress_reporter = Model(), "cpu", None
    rate, audio = engine.infer("", "First sentence. [pause:500ms] Last sentence.",
        omnivoice={"mode": "auto"}, text_normalization=False, seed=42)
    assert rate == 24000
    assert len(audio) == 2 * 2400 + 12000
    assert audio.dtype == np.int16
    assert not np.any(audio[2400:14400])
    assert [item["text"] for item in engine.model.calls] == ["First sentence.", "Last sentence."]
    assert all(item["num_step"] == 32 for item in engine.model.calls)


def test_long_text_threshold_never_exceeds_one_measured_pass():
    from indextts.backends.omnivoice import MAX_SINGLE_PASS_S, OmniVoiceEngine

    class Model(torch.nn.Module):
        text_tokenizer = SimpleNamespace(encode=lambda text, **kwargs: text.split())

        def __init__(self):
            super().__init__()
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return [np.zeros(2400, dtype=np.float32)]

    engine = OmniVoiceEngine.__new__(OmniVoiceEngine)
    engine.model, engine.device, engine.progress_reporter = Model(), "cpu", None
    for requested, used in ((60.0, MAX_SINGLE_PASS_S), (20.0, 20.0)):  # an older preset's 60 s; a shorter choice
        engine.infer("", "Test.", omnivoice={"mode": "auto", "audio_chunk_threshold": requested}, text_normalization=False)
        assert engine.model.calls[-1]["audio_chunk_threshold"] == used
    assert MAX_SINGLE_PASS_S == 30.0


def test_full_checkpoint_roundtrip_and_restore(tmp_path):
    from indextts.lora import LoraMetadata, apply_lora, inspect_lora, remove_lora, save_lora
    model = torch.nn.Module()
    model.voice = torch.nn.Linear(3, 4)
    original = {k:v.clone() for k,v in model.state_dict().items()}
    trained = torch.nn.Linear(3, 4)
    path = tmp_path / "voice.safetensors"
    save_lora(path, {}, {"voice":trained}, LoraMetadata(adapter_type="full",base_model="k2-fsa/OmniVoice"), dtype=torch.float32)
    info = inspect_lora(path)
    assert info["adapter_type"] == "full" and info["rank"] == 0
    apply_lora(model, str(path))
    assert torch.equal(model.voice.weight,trained.weight)
    remove_lora(model)
    for key,value in model.state_dict().items():
        assert torch.equal(value,original[key])
    with pytest.raises(ValueError,match="strength 1.0"):
        apply_lora(model,str(path),strength=.5)


def test_chunked_generation_progress_waits_for_audio_and_cancel_removes_hook():
    from indextts.backends.omnivoice import OmniVoiceEngine
    class Model(torch.nn.Module):
        text_tokenizer = SimpleNamespace(encode=lambda text, **kwargs: text.split())
        def forward(self):
            return None
        def generate(self, **kwargs):
            for _ in range(5):  # More forward calls than the requested step count.
                self()
            return [np.zeros(2400, dtype=np.float32)]
    engine = OmniVoiceEngine.__new__(OmniVoiceEngine)
    engine.model, engine.device = Model(), "cpu"
    progress = []
    engine.progress_reporter = SimpleNamespace(update=lambda n, **kwargs: progress.append((n, kwargs["total"])))
    engine.infer("", "Test.", omnivoice={"mode":"auto", "num_step":4}, text_normalization=False)
    assert progress == [(0, 1)] * 5 + [(1, 1)]
    def cancel(*args, **kwargs):
        raise RuntimeError("cancelled")
    engine.progress_reporter = SimpleNamespace(update=cancel)
    with pytest.raises(RuntimeError, match="cancelled"):
        engine.infer("", "Test.", omnivoice={"mode":"auto"}, text_normalization=False)
    assert not engine.model._forward_pre_hooks


def test_training_model_contract_and_preset_dispatch():
    from indextts.training.train_config import TrainConfig
    from ui.training_tab import train_config_from_values, training_tier_values
    config = train_config_from_values({"app.model":"omnivoice","training.dataset_dir":"data",
                                      "training.name":"voice","training.adapter_type":"full"})
    assert config.tts_model == "omnivoice" and config.adapter_type == "full"
    with pytest.raises(ValueError,match="BF16"):
        TrainConfig.from_dict({**config.to_dict(),"base_variant":"int8_convrot"})
    # Both speech models support full fine-tuning; an unknown method is rejected.
    assert TrainConfig.from_dict({**config.to_dict(),"tts_model":"indextts"}).adapter_type == "full"
    with pytest.raises(ValueError,match="Training method"):
        TrainConfig.from_dict({**config.to_dict(),"adapter_type":"qlora"})
    assert training_tier_values("6","cuda:0","omnivoice")["blocks_to_swap"] == 0


def test_omni_dataset_fixed_validation_and_cpu_rng(tmp_path):
    from indextts.training.omnivoice_data import OmniVoiceDataset
    from indextts.training.train_config import TrainConfig
    from indextts.training.dataset_manifest import write_manifest
    root=tmp_path
    (root/"cache/omnivoice").mkdir(parents=True)
    records=[{"id":"training","text":"Training sentence.","split":"train"},
             {"id":"validation","text":"Validation sentence.","split":"val"}]
    write_manifest(root/"manifest.jsonl",records)
    cached=[]
    for row in records:
        path=root/f"cache/omnivoice/{row['id']}.pt"
        torch.save({"audio_tokens":torch.arange(320).reshape(8,40)},path)
        cached.append({"id":row['id'],"path":str(path.relative_to(root)),"n_codes":40,"n_text_tokens":3,"fingerprint":row['id']})
    write_manifest(root/"cache/omnivoice/index.jsonl",cached)
    class Tokenizer:
        def __call__(self,text,**kwargs):
            return SimpleNamespace(input_ids=torch.tensor([[1,2,3]]))
    cfg=TrainConfig(dataset_dir=str(root),name="test",tts_model="omnivoice")
    dataset=OmniVoiceDataset(cfg,Tokenizer(),"val")
    state=torch.random.get_rng_state().clone()
    first=dataset[0]
    assert torch.equal(state,torch.random.get_rng_state())
    dataset.set_epoch(4)
    second=dataset[0]
    assert torch.equal(first["input_ids"],second["input_ids"])
    assert [r["id"] for r in dataset.records] == ["validation"]


def test_resume_state_restores_unrounded_optimizer_weights(tmp_path):
    from collections import deque
    from indextts.training.trainer import LoraTrainer, _rng_state
    from indextts.lora.io import save_train_state, load_train_state
    trainer = LoraTrainer.__new__(LoraTrainer)
    trainer.log = lambda _message: None
    trainer.early_stopping = trainer.probe_tracker = SimpleNamespace(to_dict=lambda: {})
    trainer.config = SimpleNamespace(to_dict=lambda: {})
    parameter = torch.nn.Parameter(torch.tensor([0.050071231, 0.123456789]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step(); scheduler.step(); optimizer.zero_grad()
    expected = parameter.detach().clone()
    state = trainer._train_state(optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        step=1, next_epoch=0, next_batch=1, dataset_fingerprint="fixed", best_val_loss=None,
        ema_loss=None, moving_losses=deque([1.0]))
    path = tmp_path / "state.pt"
    save_train_state(path, state)
    with torch.no_grad():
        parameter.copy_(parameter.bfloat16().float())
    assert not torch.equal(parameter, expected)
    trainer._restore_resume_state(load_train_state(path), optimizer, scheduler, scaler)
    assert torch.equal(parameter, expected)
    reference = torch.nn.Parameter(expected.clone())
    reference_optimizer = torch.optim.AdamW([reference], lr=1e-3)
    reference_optimizer.load_state_dict(load_train_state(path)["optimizer"])
    for p in (reference, parameter): p.grad = torch.full_like(p, 0.25)
    optimizer.step(); reference_optimizer.step()
    assert torch.equal(parameter, reference)


def test_omnivoice_decoding_settings_do_not_mutate_index_controls(tmp_path):
    from indextts.training.decoding_sweep import apply_decoding_settings, load_decoding_settings, decoding_markdown
    import json
    folder = tmp_path / "analysis"; folder.mkdir()
    (folder / "decoding.json").write_text(json.dumps({"accepted":True,"settings":{"num_step":64,"guidance_scale":1.5}}))
    settings = load_decoding_settings(tmp_path)
    infer = {"temperature":0.8,"omnivoice":{"mode":"clone","num_step":32}}
    updated = apply_decoding_settings(infer, settings)
    assert updated["omnivoice"] == {"mode":"clone","num_step":64,"guidance_scale":1.5}
    assert infer["omnivoice"]["num_step"] == 32 and updated["temperature"] == .8
    assert "diffusion steps 64" in decoding_markdown({"base_settings":settings,"accepted":False})


def test_omnivoice_masked_metrics_ignore_padding_and_use_codebook_weights():
    from indextts.training.omnivoice_data import masked_audio_metrics
    class Model(torch.nn.Module):
        normalized_audio_codebook_weights = [1/8] * 8
        def forward(self, **batch):
            return SimpleNamespace(logits=batch["scores"])
    logits = torch.zeros(1, 8, 3, 4)
    labels = torch.full((1, 8, 3), -100, dtype=torch.long)
    labels[:,:,0] = 0
    logits[:,:,0,0] = 2
    logits[:,:,1:,3] = 100  # padding must contribute neither loss nor accuracy
    metrics = masked_audio_metrics(Model(), [{"scores":logits,"labels":labels}], torch.device("cpu"), dtype=torch.float32)
    expected = torch.nn.functional.cross_entropy(torch.tensor([[2.,0,0,0]]), torch.tensor([0])).item()
    assert metrics["loss"] == pytest.approx(expected)
    assert metrics["accuracy"] == 1


def test_invalid_voice_tags_fail_before_output_allocation(tmp_path):
    from ui.generation_tab import prepare_generation_request
    with pytest.raises(ValueError):
        prepare_generation_request({"app.model": "omnivoice", "omnivoice.mode": "design",
            "omnivoice.instruct": "A friendly documentary narrator"},
            prompt="", text="Test.", subtitle_file=None, image_path=None,
            emotion_audio=None, model_dir=str(tmp_path), output_root=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_grid_checkpoint_metadata_filters_the_selected_model(tmp_path):
    from safetensors.numpy import save_file
    from ui.grid_tab import _adapter_folders
    for name, model in (("omni", "k2-fsa/OmniVoice"), ("index", "IndexTTS2")):
        folder = tmp_path / name
        folder.mkdir()
        save_file({}, str(folder / f"{name}.safetensors"), metadata={
            "base_model": model, "adapter_type": "dora", "rank": "128"})
    assert [Path(path).name for _, path in _adapter_folders(tmp_path, "omnivoice")] == ["omni"]
    assert [Path(path).name for _, path in _adapter_folders(tmp_path, "indextts")] == ["index"]


def test_omnivoice_training_tiers_fit_measured_capacity():
    from ui.training_tab import training_tier_values
    from indextts.training.train_config import TrainConfig
    # 4096-token micro-batches x accumulation 2, checkpointing below 24 GB (round-2 probes: DoRA 2.7 GB and
    # full fine-tuning 11.5 GB with checkpointing; 17.1 and 19.5 GB without).
    low = training_tier_values("6", "cuda:0", "omnivoice", "dora")
    assert (low["base_variant"], low["omni_batch_tokens"], low["grad_accumulation"], low["learning_rate"]) == ("bf16", 4096, 2, 1e-4)
    assert low["gradient_checkpointing"] and not low["train_mel_embed_head"] and low["keep_last_n"] == 0
    full = training_tier_values("16", "cuda:0", "omnivoice", "full")
    assert (full["omni_batch_tokens"], full["grad_accumulation"], full["learning_rate"], full["keep_last_n"]) == (4096, 2, 2e-5, 3)
    assert full["gradient_checkpointing"] and full["train_mel_embed_head"]
    assert not training_tier_values("24", "cuda:0", "omnivoice", "full")["gradient_checkpointing"]
    with pytest.raises(ValueError, match="16 GB"):
        TrainConfig.from_dict({"dataset_dir": "data", "name": "test", "tts_model": "omnivoice",
                               "adapter_type": "full", "vram_tier": "12"})


def test_codec_cache_subset_and_cancellation_keep_complete_index(tmp_path, monkeypatch):
    import transformers
    from indextts.backends import omnivoice as backend
    from indextts.training import features
    from indextts.training.dataset_manifest import load_manifest, write_manifest
    from indextts.training.omnivoice_data import cache_omnivoice_features
    from indextts.training.features import FeatureCacheConfig
    model = tmp_path / "model"
    (model / "audio_tokenizer").mkdir(parents=True)
    (model / "audio_tokenizer/model.safetensors").write_bytes(b"codec")
    (model / "tokenizer.json").write_bytes(b"tokenizer")
    write_manifest(tmp_path / "manifest.jsonl", [{"id": name, "text": name, "audio": f"{name}.wav"}
                                               for name in ("one", "two")])
    for name in ("one", "two"):
        (tmp_path / f"{name}.wav").write_bytes(name.encode())
    monkeypatch.setattr(backend, "ensure_model", lambda *a, **k: (model, None))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: SimpleNamespace(encode=lambda *a, **k: [1]))
    class Extractor:
        sampling_rate = 24000
        def __call__(self, **kwargs):
            return {"input_values": torch.zeros(1, 2400)}
    codec = SimpleNamespace(encode=lambda audio: SimpleNamespace(audio_codes=torch.ones(1, 8, 20)))
    monkeypatch.setattr(transformers.AutoFeatureExtractor, "from_pretrained", lambda *a, **k: Extractor())
    monkeypatch.setattr(transformers.HiggsAudioV2TokenizerModel, "from_pretrained",
                        lambda *a, **k: SimpleNamespace(eval=lambda: codec))
    monkeypatch.setattr(features, "_read_audio", lambda *a: (torch.zeros(1, 2400), 24000))
    config = FeatureCacheConfig(str(tmp_path), device="cpu")
    cache_omnivoice_features(config)
    index = tmp_path / "cache/omnivoice/index.jsonl"
    original = index.read_bytes()
    config.max_items = 1
    cache_omnivoice_features(config)
    assert {row["id"] for row in load_manifest(index)} == {"one", "two"}
    config.max_items = 0
    calls = []
    def cancel(*args):
        calls.append(1)
        assert index.read_bytes() == original
        return len(calls) > 1
    cache_omnivoice_features(config, cancel_callback=cancel)
    assert index.read_bytes() == original
