"""AuK as the third speech model: registry, profiles, requests, instructions, durations, tasks and training contract.

None of these tests downloads or loads AuK weights.
"""

import math

import numpy as np
import pytest
import torch

from ui.model_profiles import capture_profile, is_model_only, profiled_keys, switch_profile
from ui.presets_store import PresetRegistry


def registry():
    result = PresetRegistry()
    for key, default in {
        "app.model": "indextts", "app.profiles": {}, "runtime.device": "cuda:0",
        "runtime.lora_path": "", "runtime.decoder_adapter": "auto",
        "generation.speaking_rate": 1.0, "generation.max_text_tokens_per_segment": 60,
        "omnivoice.num_step": 32, "auk.num_step": 32, "auk.mode": "clone",
        "runtime.auk_text_encoder_variant": "bf16", "runtime.auk_text_encoder_residency": "gpu",
    }.items():
        result.register(key, default=default)
    return result


def test_three_models_are_registered_and_checkpoints_belong_to_one():
    from indextts.backends import MODEL_IDS, checkpoint_matches_model, checkpoint_model, normalize_model

    assert MODEL_IDS == ("indextts", "omnivoice", "auk")
    assert normalize_model("AuK") == "auk"
    assert checkpoint_model({"base_model": "tencent/AuK"}) == "auk"
    assert checkpoint_model({"base_model": "k2-fsa/OmniVoice"}) == "omnivoice"
    assert checkpoint_model({}) == "indextts"
    assert checkpoint_matches_model({"base_model": "tencent/AuK"}, "auk")
    assert not checkpoint_matches_model({"base_model": "tencent/AuK"}, "indextts")
    assert not checkpoint_matches_model({"base_model": "tencent/AuK"}, "omnivoice")


def test_switch_roundtrip_keeps_each_models_profile_and_auk_only_values():
    controls = registry()
    index = capture_profile({**controls.defaults(), "runtime.lora_path": "index.safetensors",
                             "generation.speaking_rate": 1.05})
    auk = switch_profile(controls, "auk", index)
    assert auk["app.model"] == "auk"
    assert auk["runtime.lora_path"] == ""
    assert auk["generation.max_text_tokens_per_segment"] == 60
    auk.update({"runtime.lora_path": "auk.safetensors", "auk.num_step": 48,
                "runtime.auk_text_encoder_variant": "int8_convrot"})
    omni = switch_profile(controls, "omnivoice", auk)
    assert omni["runtime.lora_path"] == ""
    restored = switch_profile(controls, "auk", switch_profile(controls, "indextts", omni))
    assert restored["runtime.lora_path"] == "auk.safetensors"
    # Model-only controls keep one value across every switch.
    assert restored["auk.num_step"] == 48
    assert restored["runtime.auk_text_encoder_variant"] == "int8_convrot"
    index_again = switch_profile(controls, "indextts", restored)
    assert index_again["runtime.lora_path"] == "index.safetensors"
    assert index_again["generation.speaking_rate"] == 1.05


def test_auk_only_keys_are_never_profiled():
    keys = ["runtime.auk_text_encoder_variant", "runtime.auk_text_encoder_residency", "auk.mode",
            "auk_edit.task", "training.auk_prompt_fraction", "runtime.lora_path"]
    assert [key for key in keys if is_model_only(key)] == keys[:-1]
    assert profiled_keys(keys) == ["runtime.lora_path"]


def test_reference_free_auk_request_carries_settings_and_language(tmp_path):
    from ui.common import read_json
    from ui.generation_tab import RUNNER_REQUEST_KEYS, prepare_generation_request

    request = prepare_generation_request(
        {"app.model": "auk", "auk.mode": "design", "auk.voice_description": "a calm deep voice",
         "auk.num_step": 16, "auk.language": "EN", "generation.apply_pronunciation_dictionary": False},
        prompt="", text="Test voice.", subtitle_file=None, image_path=None, emotion_audio=None,
        model_dir=str(tmp_path), output_root=tmp_path / "out")
    assert "auk" in RUNNER_REQUEST_KEYS
    assert request["runtime"]["tts_model"] == "auk"
    assert request["auk"]["num_step"] == 16 and request["auk"]["mode"] == "design"
    assert "language" not in request["auk"] and request["language"] == "EN"
    assert request["omnivoice"] is None
    assert read_json(request["metadata_path"])["settings"]["request_values"]["auk.num_step"] == 16


def test_auk_clone_requires_reference_and_design_requires_description(tmp_path):
    from ui.generation_tab import prepare_generation_request

    with pytest.raises(ValueError, match="Reference Voice"):
        prepare_generation_request({"app.model": "auk", "auk.mode": "clone"}, prompt="", text="Test.",
                                   subtitle_file=None, image_path=None, emotion_audio=None,
                                   model_dir=str(tmp_path), output_root=tmp_path)
    with pytest.raises(ValueError, match="Describe the voice"):
        prepare_generation_request({"app.model": "auk", "auk.mode": "design"}, prompt="", text="Test.",
                                   subtitle_file=None, image_path=None, emotion_audio=None,
                                   model_dir=str(tmp_path), output_root=tmp_path)


def test_instructions_are_upstream_templates():
    from indextts.auk.conditioning import NO_PROMPT_AUDIO, user_message
    from indextts.auk.text import build_instruction

    assert build_instruction('Say "hi" now', "clone") == 'Say the following with the same voice: "Say “hi” now"'
    design = build_instruction("Hello.", "design", "a calm voice")
    assert design == 'Based on the following description: "a calm voice", generate speech content "Hello.".'
    assert build_instruction("你好", "design", "温柔", "zh").startswith("请基于下面的描述")
    # Auto voice speaks with the trained description (a default when none was saved).
    assert "trained speaker" in build_instruction("Hello.", "auto")
    assert NO_PROMPT_AUDIO == "|<no_prompt_audio>|"
    assert user_message("Hi.", with_audio=False)[0]["content"][0]["text"] == "Hi.|<no_prompt_audio>|"
    assert [item["type"] for item in user_message("Hi.", with_audio=True)[0]["content"]] == ["text", "audio"]


def test_duration_model_matches_upstream_prompt_enhancer():
    from indextts.auk.text import f5_seconds
    from indextts.backends.auk import AukEngine

    assert f5_seconds("abcd", "en") == pytest.approx(4 * 0.0656)
    assert f5_seconds("你好", "zh") == pytest.approx(6 * 0.0803)
    # Digits and spaces take the script before them.
    assert f5_seconds("你 2", "zh") == pytest.approx((3 + 1 + 1) * 0.0803)
    long_text = "This sentence has a fair number of letters in it."
    seconds = AukEngine.estimate_seconds(long_text, "en", 1.0)
    assert seconds == pytest.approx(math.ceil(f5_seconds(long_text) * 50) / 50)
    assert AukEngine.estimate_seconds(long_text, "en", 1.2) > seconds
    # Very short text: 0.3x speed without a reference, a one-second floor with one.
    assert AukEngine.estimate_seconds("Yes.", "en", 1.0) == pytest.approx(math.ceil(4 * 0.0656 / 0.3 * 50) / 50)
    assert AukEngine.estimate_seconds("Yes.", "en", 1.0, paced_by_reference=True) == 1.0
    assert AukEngine.estimate_seconds("x " * 400, "en", 1.0, max_seconds=12) == 12


def test_reference_trimming_and_pause_cut():
    from indextts.backends.auk import cut_at_pause, speech_span, split_at_pauses, trim_silence

    rate = 1000
    audio = np.concatenate([np.zeros(500), 0.5 * np.ones(2000), np.zeros(300), 0.5 * np.ones(1000), np.zeros(800)]).astype(np.float32)
    assert speech_span(audio, rate) == pytest.approx(3.3, abs=0.02)
    trimmed = trim_silence(audio, rate)
    assert len(trimmed) == pytest.approx(3300 + 200, abs=20)
    cut = cut_at_pause(audio, rate, 3.0)
    assert len(cut) <= 3000 and np.abs(cut[-50:]).max() == 0.0
    pieces = split_at_pauses(audio, rate, 2.6)
    assert sum(len(piece) for piece in pieces) == len(audio)
    assert max(len(piece) for piece in pieces) <= 2600


def test_task_catalog_renders_every_task_and_follows_duration_rules():
    from indextts.auk.tasks import FIELD_DEFAULTS, TASKS, render_instruction, target_seconds

    values = {**FIELD_DEFAULTS, "orig": "old words", "new": "brand new words", "text": "kindly", "anchor": "please",
              "target": "very", "spoken": "get what", "instruction": "Do it."}
    for key in TASKS:
        assert render_instruction(key, values).strip()
    assert render_instruction("speed", {"speed": 1.5}) == "Adjust the speech speed to 1.5x."
    assert render_instruction("pitch", {"direction": "lower", "semitones": 1}) == "Lower the pitch by 1 semitone."
    assert render_instruction("replace", values) == "Replace 'old words' with 'brand new words'."
    assert target_seconds("speed", {"speed": 2.0}, 10.0, 10.4) == 5.0
    assert target_seconds("emotion", {"emotion": "sad"}, 10.0, 10.4) == pytest.approx(12.2)
    assert target_seconds("from_whisper", {}, 10.0, 10.4) == 10.4
    assert target_seconds("nonverbal_add", {"sound": "laugh"}, 10.0, 10.0) == pytest.approx(10.75)
    # Content edits scale by spoken length: 0.30 s per English word.
    assert target_seconds("replace", values, 6.0, 6.0, "these are old words here") == pytest.approx(6.0 * (1.5 + 0.9 - 0.6) / 1.5)
    with pytest.raises(ValueError, match="Fill in"):
        render_instruction("replace", {"orig": "", "new": "x"})


def test_train_config_accepts_auk_methods_and_automatic_epochs():
    from indextts.training.train_config import TrainConfig

    config = TrainConfig.from_dict({"dataset_dir": "x", "name": "voice", "tts_model": "auk", "adapter_type": "full",
                                    "epochs": 0, "auk_prompt_fraction": 2, "auk_reference_seconds": 1})
    assert config.adapter_type == "full" and config.epochs == 0
    assert config.auk_prompt_fraction == 1.0 and config.auk_reference_seconds == 3.0
    with pytest.raises(ValueError):
        TrainConfig.from_dict({"dataset_dir": "x", "name": "voice", "tts_model": "auk", "adapter_type": "bitfit"})


def test_adapter_targets_cover_attention_feed_forward_and_adaln():
    from indextts.auk.cfm import AukModel
    from indextts.training.auk_trainer import adapter_targets

    arch = {"dim": 64, "heads": 2, "dim_head": 32, "ff_mult": 2, "text_hidden_dim": 32, "num_layers": 1,
            "num_single_layers": 1}
    model = AukModel(arch, latent_dim=8, num_text_layers=2)
    targets = adapter_targets(model)
    assert "transformer.transformer_blocks.0.attn.to_qkv_c" in targets
    assert "transformer.transformer_blocks.0.ff_x.linear_in" in targets
    assert "transformer.single_transformer_blocks.0.attn_norm.linear" in targets
    assert not any("proj_out" in name or "txt_proj" in name for name in targets)
    assert len(adapter_targets(model, adaln=False)) == len(targets) - 3


def test_flow_model_sampling_shapes_masks_and_cfg():
    from indextts.auk.cfm import AukModel

    torch.manual_seed(0)
    arch = {"dim": 64, "heads": 2, "dim_head": 32, "ff_mult": 2, "text_hidden_dim": 32, "num_layers": 1,
            "num_single_layers": 1}
    model = AukModel(arch, latent_dim=8, num_text_layers=2).eval()
    text = torch.randn(2, 5, 32)
    context = torch.ones(2, 5, dtype=torch.bool)
    ref = torch.randn(2, 4, 8)
    steps = []
    out = model.sample(text, context, ref, torch.tensor([4, 3]), torch.tensor([6, 3]), steps=4, cfg_strength=2.0,
                       step_callback=lambda step, total: steps.append(step))
    assert out.shape == (2, 6, 8) and steps == [1, 2, 3, 4]
    assert torch.all(out[1, 3:] == 0)
    single = model.sample(text[:1], context[:1], ref[:1], torch.tensor([4]), torch.tensor([6]), steps=4,
                          cfg_strength=0.0, noise=torch.zeros(1, 6, 8))
    assert torch.isfinite(single).all()
    loss = model.train()(torch.randn(2, 6, 8), text, context, ref_latent=ref, ref_lens=torch.tensor([4, 3]),
                         target_lens=torch.tensor([6, 3]))
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_vae_folds_weight_norm_like_upstream():
    from indextts.auk.vae import CausalConv1d

    conv = torch.nn.utils.weight_norm(torch.nn.Conv1d(3, 4, 5))
    folded = torch._weight_norm(conv.weight_v.detach(), conv.weight_g.detach(), 0)
    assert torch.allclose(folded, conv.weight.detach())
    causal = CausalConv1d(3, 4, 5, causal=True)
    assert causal(torch.randn(1, 3, 20)).shape[-1] == 20


def test_runtime_config_validates_auk_encoder_fields():
    from indextts.runtime.vram_presets import RuntimeConfig

    config = RuntimeConfig.from_dict({"auk_text_encoder_variant": "INT8_CONVROT", "auk_text_encoder_residency": "x"})
    assert config.auk_text_encoder_variant == "int8_convrot"
    assert config.auk_text_encoder_residency == "gpu"


def test_auk_tier_presets_shrink_memory_with_the_card():
    from indextts.runtime.auk_presets import resolve_preset, resolve_training_preset

    big, mid, ten, small = resolve_preset(32), resolve_preset(12), resolve_preset(10), resolve_preset(6)
    assert (big.model_variant, big.auk_text_encoder_variant, big.auk_text_encoder_residency) == ("bf16", "bf16", "gpu")
    assert mid.auk_text_encoder_variant == "int8_convrot" and mid.model_variant == "bf16"
    assert ten.model_variant == "int8_convrot" and ten.auk_text_encoder_residency == "gpu"
    # On demand the encoder and the transformer take turns: the BF16 transformer fits 8 GB again.
    assert small.auk_text_encoder_residency == "on_demand" and small.model_variant == "int8_convrot"
    assert resolve_preset(8).model_variant == "bf16" and resolve_preset(8).auk_text_encoder_residency == "on_demand"
    assert resolve_training_preset(32, "full")["base_variant"] == "bf16"
    assert resolve_training_preset(32, "full")["gradient_checkpointing"] is False
    assert resolve_training_preset(8, "dora")["base_variant"] == "int8_convrot"


def test_cpu_token_table_lookup_is_exact():
    import torch
    from indextts.auk.conditioning import _CpuEmbedding

    embedding = torch.nn.Embedding(50, 8, padding_idx=0)
    ids = torch.tensor([[3, 7, 0, 49]])
    moved = _CpuEmbedding(embedding)
    assert torch.equal(moved(ids), embedding(ids))
    assert not list(moved.parameters()) and moved.weight.shape == (50, 8)


def test_on_demand_residency_lends_one_model_at_a_time():
    from indextts.backends.auk import AukEngine

    class Loan:
        def __init__(self):
            self.active = False

        def activate(self, active):
            self.active = bool(active)

    class Conditioner:
        device = None

    engine = AukEngine.__new__(AukEngine)
    engine.device, engine.conditioner = "cuda:0", Conditioner()
    engine._text_offload, engine._dit_offload = Loan(), Loan()
    engine._residency("text")
    assert engine._text_offload.active and not engine._dit_offload.active
    assert str(engine.conditioner.device) == "cuda:0"
    engine._residency("dit")
    assert engine._dit_offload.active and not engine._text_offload.active
    engine._residency("idle")
    assert not engine._dit_offload.active and not engine._text_offload.active
    resident = AukEngine.__new__(AukEngine)
    resident._text_offload = None
    resident._residency("text")  # both models resident: nothing to lend


def test_trained_voices_keep_the_wording_they_learned():
    from indextts.auk.text import VOICE_TEMPLATE, build_instruction, voice_template
    from indextts.training.auk_data import training_instruction, voice_record

    learned = training_instruction("Hello.", "The trained speaker's natural voice", "en")
    assert learned.startswith("Generate speech based on the following description:")
    assert build_instruction("Hello.", "auto", "The trained speaker's natural voice", "en", voice_template("en")) == learned
    assert voice_template("en", "Custom {description}: {text}") == "Custom {description}: {text}"
    assert voice_template("zh").startswith("请基于下面的描述")
    rows = [{"cache": {"speech_seconds": 2.0}, "train_text": "Hello there friend.", "language": "en"}]
    assert voice_record(rows, "desc")["template"] == VOICE_TEMPLATE


def test_sections_longer_than_the_context_are_split_again():
    from indextts.backends.auk import AukEngine

    engine = AukEngine.__new__(AukEngine)

    def split(text, max_tokens, **_options):
        words = text.split()
        return [" ".join(words[start:start + max_tokens]) for start in range(0, len(words), max_tokens)]

    engine.split_text_by_tokens = split
    parts = engine._fit_context([" ".join(["word"] * 200)], 200, 30.0, "en", 1.0, False, 0.0, 1.0)
    assert len(parts) > 1 and sum(len(part.split()) for part in parts) == 200
    assert all(AukEngine.estimate_seconds(part, "en", 1.0, max_seconds=1000) <= 30.0 for part in parts)
    assert engine._fit_context(["A short sentence."], 60, 30.0, "en", 1.0, False, 0.0, 1.0) == ["A short sentence."]


def test_pronunciation_markup_becomes_plain_text_for_auk():
    from indextts.utils.pronunciation import plain_readings
    from ui import generation_tab

    assert plain_readings("A <DoRA|D AO1 . R AH0> run with <Qwen|chwen>.") == "A DoRA run with chwen."
    assert plain_readings("<行|XING2>人") == "行人"
    rows, _note = generation_tab.preview_segments("A <DoRA|D AO1 . R AH0> run.", "EN", 60, model_id="auk")
    assert rows[0][2] == "A DoRA run." and "reads" not in rows[0][3]


def test_batches_carry_their_epoch_to_persistent_workers():
    from indextts.training.auk_data import EpochTaggedBatches

    class Sampler:
        epoch = 0

        def set_epoch(self, epoch):
            self.epoch = epoch

        def __iter__(self):
            return iter([[2, 0], [1]])

        def __len__(self):
            return 2

    batches = EpochTaggedBatches(Sampler())
    assert list(batches) == [[(0, 2), (0, 0)], [(0, 1)]] and len(batches) == 2
    batches.set_epoch(3)
    assert list(batches)[0] == [(3, 2), (3, 0)]


def test_fully_trained_edges_stay_float_in_the_int8_transformer():
    import sys
    from pathlib import Path

    from indextts.auk.cfm import AukModel
    from indextts.training.auk_trainer import EDGE_MODULES, EDGE_PROJECTIONS, adapter_targets

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from quantize_auk import dit_targets

    arch = {"dim": 64, "heads": 2, "dim_head": 32, "ff_mult": 2, "text_hidden_dim": 32, "num_layers": 1,
            "num_single_layers": 1}
    model = AukModel(arch, latent_dim=8, num_text_layers=2)
    quantized = set(dit_targets(list(model.state_dict())))
    # An adapter must load into the BF16 and the INT8 transformer: fully trained modules are never quantized,
    # and the quantized edge projections are adapted instead.
    for name in EDGE_MODULES:
        assert not any(key.startswith(f"transformer.{name}") for key in quantized), name
    for name in EDGE_PROJECTIONS:
        assert f"transformer.{name}" in quantized, name
    with_edges = adapter_targets(model, edges=True)
    assert set(with_edges) - set(adapter_targets(model)) == {f"transformer.{name}" for name in EDGE_PROJECTIONS}
