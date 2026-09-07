"""Voice decoder (s2mel) adapters: configuration, file discovery, component tags, and target modules."""
from pathlib import Path

import pytest
import torch
from torch import nn

from indextts.lora.apply import inject_adapters
from indextts.lora.decoder import (
    DECODER_ADAPTER_SUFFIX,
    adapter_root,
    component_of_metadata,
    decoder_target_modules,
    find_decoder_adapter,
    is_decoder_adapter_path,
    lora_component,
)
from indextts.lora.io import LoraMetadata, save_lora
from indextts.runtime.vram_presets import RuntimeConfig
from indextts.training.decoder_adapter import DecoderAdapterConfig
from indextts.training.train_config import TrainConfig


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def test_decoder_adapter_is_found_for_every_checkpoint_of_a_training(tmp_path):
    run = tmp_path / "loras" / "voice"
    best = _touch(run / "best" / "voice.safetensors")
    epoch = _touch(run / "voice_epoch_003.safetensors")
    final = _touch(run / "voice.safetensors")
    assert adapter_root(best) == run.resolve() and adapter_root(epoch) == run.resolve()
    assert find_decoder_adapter(best) == ""  # nothing trained yet
    decoder = _touch(run / f"voice{DECODER_ADAPTER_SUFFIX}")
    for checkpoint in (best, epoch, final):
        assert Path(find_decoder_adapter(checkpoint)) == decoder.resolve()
    # A decoder file beside a specific checkpoint wins over the training folder's file.
    specific = _touch(run / "best" / f"voice{DECODER_ADAPTER_SUFFIX}")
    assert Path(find_decoder_adapter(best)) == specific.resolve()
    assert Path(find_decoder_adapter(epoch)) == decoder.resolve()
    # Decoder files are never treated as GPT adapters, and missing GPT files find nothing.
    assert find_decoder_adapter(decoder) == "" and find_decoder_adapter(run / "missing.safetensors") == ""
    assert is_decoder_adapter_path(decoder) and not is_decoder_adapter_path(best)


def test_component_tag_distinguishes_decoder_files_from_gpt_files(tmp_path):
    assert component_of_metadata({"component": "s2mel"}) == "s2mel"
    assert component_of_metadata('{"component": "s2mel"}') == "s2mel"
    assert component_of_metadata({}, "loras/v/v.safetensors") == "gpt"
    assert component_of_metadata(None, f"loras/v/v{DECODER_ADAPTER_SUFFIX}") == "s2mel"

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 8)

    toy = Toy()
    adapters = inject_adapters(toy, rank=2, alpha=2.0, dropout=0.0, use_dora=False, target_modules=["proj"])
    decoder_file = tmp_path / f"toy{DECODER_ADAPTER_SUFFIX}"
    save_lora(decoder_file, adapters, {}, LoraMetadata(adapter_type="lora", rank=2, alpha=2.0, target_modules=["proj"],
                                                       train_config={"component": "s2mel"}), dtype=torch.float32)
    gpt_file = tmp_path / "toy.safetensors"
    save_lora(gpt_file, adapters, {}, LoraMetadata(adapter_type="lora", rank=2, alpha=2.0, target_modules=["proj"]), dtype=torch.float32)
    assert lora_component(decoder_file) == "s2mel" and lora_component(gpt_file) == "gpt"


def test_target_modules_cover_attention_and_feed_forward_projections_of_every_block():
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.wqkv = nn.Linear(16, 48, bias=False)
            self.wo = nn.Linear(16, 16, bias=False)

    class FeedForward(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Linear(16, 32, bias=False)
            self.w3 = nn.Linear(16, 32, bias=False)
            self.w2 = nn.Linear(32, 16, bias=False)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = Attention()
            self.feed_forward = FeedForward()
            self.attention_norm = nn.LayerNorm(16)

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(Block() for _ in range(3))

    class Estimator(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = Transformer()
            self.x_embedder = nn.Linear(80, 16)

    estimator = Estimator()
    paths = decoder_target_modules(estimator)
    assert len(paths) == 15 and paths[0] == "transformer.layers.0.attention.wqkv" and "transformer.layers.2.feed_forward.w2" in paths
    assert all("x_embedder" not in path and "norm" not in path for path in paths)
    assert len(decoder_target_modules(estimator, mlp=False)) == 6 and len(decoder_target_modules(estimator, attention=False)) == 9
    adapters = inject_adapters(estimator, rank=4, alpha=4.0, dropout=0.0, use_dora=True, target_modules=paths)
    assert set(adapters) == set(paths)
    assert decoder_target_modules(nn.Linear(2, 2)) == []


def test_decoder_config_validates_and_round_trips():
    config = DecoderAdapterConfig(dataset_dir="datasets/voice", output_path="loras/voice/voice.s2mel.safetensors",
                                  rank=16, epochs=2, learning_rate=2e-4)
    restored = DecoderAdapterConfig.from_dict(config.to_dict())
    assert restored.name == "voice" and restored.rank == 16 and restored.epochs == 2 and restored.adapter_type == "dora"
    assert DecoderAdapterConfig(dataset_dir="d", output_path="loras/v/v.s2mel.safetensors").validate().rank == 128
    with pytest.raises(ValueError):
        DecoderAdapterConfig(dataset_dir="datasets/voice", output_path="loras/voice/voice.safetensors").validate()
    with pytest.raises(ValueError):
        DecoderAdapterConfig(dataset_dir="datasets/voice", output_path="loras/voice/voice.s2mel.safetensors", learning_rate=0).validate()
    with pytest.raises(ValueError):
        DecoderAdapterConfig(dataset_dir="", output_path="loras/voice/voice.s2mel.safetensors").validate()


def test_training_and_runtime_configs_carry_the_decoder_options():
    config = TrainConfig.from_dict({"dataset_dir": "d", "name": "n"})
    assert config.decoder_adapter_enabled and config.decoder_adapter_rank == 128 and config.decoder_adapter_epochs == 10
    assert config.decoder_adapter_alpha == 128.0
    custom = TrainConfig.from_dict({"dataset_dir": "d", "name": "n", "decoder_adapter_enabled": False, "decoder_adapter_rank": 8,
                                    "decoder_adapter_timeout_s": 10})
    assert not custom.decoder_adapter_enabled and custom.decoder_adapter_rank == 8 and custom.decoder_adapter_timeout_s == 60.0
    with pytest.raises(ValueError):
        TrainConfig.from_dict({"dataset_dir": "d", "name": "n", "decoder_adapter_learning_rate": 0})
    assert TrainConfig.from_dict(custom.to_dict()).decoder_adapter_rank == 8
    assert RuntimeConfig().decoder_adapter == "auto"
    assert RuntimeConfig.from_dict({"runtime": {"use_decoder_adapter": False}}).decoder_adapter == "none"  # older presets
    assert RuntimeConfig.from_dict({"lora_path": "x"}).to_dict()["decoder_adapter"] == "auto"
    assert RuntimeConfig.from_dict({"decoder_adapter": "NONE"}).decoder_adapter == "none"


def test_decoder_prompts_vary_per_target_and_epoch_and_stay_within_limits():
    """One fixed prompt would teach the adapter a prompt-independent offset; every target gets its own clip."""
    from types import SimpleNamespace
    from indextts.training.decoder_adapter import DecoderAdapterTrainer

    config = DecoderAdapterConfig(dataset_dir="d", output_path="loras/v/v.s2mel.safetensors", min_prompt_seconds=3.0,
                                  max_prompt_seconds=10.0, select_by="loss", identity_clips=0)
    trainer = DecoderAdapterTrainer.__new__(DecoderAdapterTrainer)
    trainer.config = config.validate()
    records = [{"id": f"clip{i:03d}", "speaker": "A", "duration_s": str(2.0 + i)} for i in range(12)]
    records.append({"id": "other", "speaker": "B", "duration_s": "5.0"})
    trainer._build_prompt_pool(SimpleNamespace(records=records))
    pool = trainer._pool
    assert [item["id"] for item in pool["A"]] == [f"clip{i:03d}" for i in range(1, 9)]  # 3.0 .. 10.0 seconds
    assert [item["id"] for item in pool["B"]] == ["other"]
    prompts_epoch0 = {record["id"]: trainer._sample_prompt(record, "train", 0, record["id"])["id"] for record in records[:12]}
    prompts_epoch1 = {record["id"]: trainer._sample_prompt(record, "train", 1, record["id"])["id"] for record in records[:12]}
    assert len(set(prompts_epoch0.values())) > 3, "targets should draw different prompts"
    assert prompts_epoch0 != prompts_epoch1, "a new epoch should draw new prompts"
    assert all(prompts_epoch0[key] != key for key in prompts_epoch0), "a clip never prompts itself"
    assert trainer._sample_prompt(records[3], "train", 0, records[3]["id"]) is trainer._sample_prompt(records[3], "train", 0, records[3]["id"])
    assert trainer._sample_prompt(records[12], "train", 0, "other") is None  # the only clip of its speaker
    assert trainer._sample_prompt({"id": "x", "speaker": "C"}, "train", 0, "x") is None


def test_rejected_decoder_adapter_is_parked_outside_the_searched_paths(tmp_path):
    from indextts.training.decoder_adapter import reject_decoder_adapter

    run = tmp_path / "loras" / "voice"
    gpt = _touch(run / "voice.safetensors")
    decoder = _touch(run / f"voice{DECODER_ADAPTER_SUFFIX}")
    assert Path(find_decoder_adapter(gpt)) == decoder.resolve()
    parked = reject_decoder_adapter(decoder)
    assert parked == (run / "analysis" / "voice.s2mel.rejected").resolve() and parked.is_file()
    assert not decoder.exists() and find_decoder_adapter(gpt) == ""
    # Parking again replaces the earlier parked file instead of failing.
    _touch(decoder)
    assert reject_decoder_adapter(decoder) == parked and not decoder.exists()


def test_decoder_config_selection_fields_validate():
    config = DecoderAdapterConfig(dataset_dir="d", output_path="loras/v/v.s2mel.safetensors").validate()
    assert config.select_by == "identity" and config.identity_clips == 8 and config.identity_min_gain == 0.005
    assert config.learning_rate == 2e-4 and config.max_prompt_seconds == 15.0
    with pytest.raises(ValueError):
        DecoderAdapterConfig(dataset_dir="d", output_path="loras/v/v.s2mel.safetensors", select_by="wer").validate()
    clamped = DecoderAdapterConfig(dataset_dir="d", output_path="loras/v/v.s2mel.safetensors", identity_steps=1,
                                   max_prompt_seconds=1.0, min_prompt_seconds=4.0).validate()
    assert clamped.identity_steps == 2 and clamped.max_prompt_seconds == 4.0


def test_decoder_test_markdown_states_the_verdict_and_reasons():
    from indextts.training.speech_eval import decoder_test_markdown

    report = {"accepted": False, "metric": "speaker_similarity_real", "speaker_gain": {"mean": -0.0749, "prompts": 12, "ci95": [-0.09, -0.06]},
              "clips": 36, "wer_increase": -0.0017, "reasons": ["speaker similarity to the real recordings changed by -0.0749"],
              "without": {"speaker_similarity_real": 0.81, "speaker_similarity": 0.91, "style_similarity_real": 0.70,
                          "corpus_error_rate": 0.059, "failure_count": 0},
              "with": {"speaker_similarity_real": 0.7356, "speaker_similarity": 0.84, "style_similarity_real": 0.63,
                       "corpus_error_rate": 0.057, "failure_count": 0},
              "checkpoint": "loras/v/v.safetensors", "baseline_report": "loras/v/analysis/speech_evaluation/final_test/report.json"}
    text = decoder_test_markdown(report)
    assert text.startswith("**Voice decoder adapter not installed.**") and "-0.0749" in text and "Reasons:" in text
    assert "| 0.8100 | 0.7356 |" in text and "5.90%" in text and "final_test" in text
    report.update(accepted=True, reasons=[])
    assert decoder_test_markdown(report).startswith("**Voice decoder adapter installed.**")


def test_decoder_adapter_dropdown_choices_and_selection(tmp_path):
    """Loading a LoRA / DoRA selects its own decoder adapter; None and other files stay selectable."""
    from indextts.lora.decoder import DECODER_CHOICE_AUTO, DECODER_CHOICE_NONE, decoder_adapter_choices, decoder_adapter_selection

    loras = tmp_path / "loras"
    gpt = _touch(loras / "voice" / "voice.safetensors")
    own = _touch(loras / "voice" / f"voice{DECODER_ADAPTER_SUFFIX}")
    other = _touch(loras / "second" / f"second{DECODER_ADAPTER_SUFFIX}")
    choices = decoder_adapter_choices(str(gpt), loras)
    assert choices[0][1] == DECODER_CHOICE_AUTO and own.name in choices[0][0]
    assert choices[1] == ("None (GPT adapter only)", DECODER_CHOICE_NONE)
    assert [value for _, value in choices[2:]] == [str(other.resolve())]  # the own file is not listed twice
    no_adapter = decoder_adapter_choices(str(_touch(loras / "bare" / "bare.safetensors")), loras)
    assert "none saved" in no_adapter[0][0] and len(no_adapter) == 4
    assert decoder_adapter_selection(DECODER_CHOICE_AUTO) == (True, "") and decoder_adapter_selection("") == (True, "")
    assert decoder_adapter_selection(DECODER_CHOICE_NONE) == (False, "") and decoder_adapter_selection("None") == (False, "")
    assert decoder_adapter_selection(str(other)) == (True, str(other.resolve()))


def test_runtime_values_map_the_decoder_choice_and_legacy_flag():
    from ui.common import runtime_config_from_values

    assert runtime_config_from_values({"runtime.decoder_adapter": "none"})["decoder_adapter"] == "none"
    assert runtime_config_from_values({"runtime.decoder_adapter": "auto"})["decoder_adapter"] == "auto"
    explicit = runtime_config_from_values({"runtime.decoder_adapter": "loras/v/v.s2mel.safetensors"})
    assert explicit["decoder_adapter"] == "loras/v/v.s2mel.safetensors"
    assert runtime_config_from_values({"runtime.use_decoder_adapter": False})["decoder_adapter"] == "none"  # older presets
    assert runtime_config_from_values({})["decoder_adapter"] == "auto"
    assert RuntimeConfig.from_dict({"decoder_adapter_path": "x.s2mel.safetensors"}).decoder_adapter == "x.s2mel.safetensors"
    assert RuntimeConfig.from_dict({"decoder_adapter_strength": "0.6"}).decoder_adapter_strength == 0.6
    assert RuntimeConfig.from_dict({"decoder_adapter_strength": "x"}).decoder_adapter_strength == 1.0
    assert runtime_config_from_values({"runtime.decoder_adapter_strength": 0.5})["decoder_adapter_strength"] == 0.5
    assert runtime_config_from_values({})["decoder_adapter_strength"] == 1.0


def test_recommended_decoder_strength_comes_from_the_full_pipeline_test(tmp_path):
    import json
    from indextts.lora.decoder import recommended_decoder_strength

    run = tmp_path / "loras" / "voice"
    gpt = _touch(run / "best" / "voice.safetensors")
    assert recommended_decoder_strength(gpt) is None  # no decoder adapter
    _touch(run / f"voice{DECODER_ADAPTER_SUFFIX}")
    assert recommended_decoder_strength(gpt) is None  # never measured
    report = run / "analysis" / "speech_evaluation" / "decoder_test" / "report.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"accepted": True, "strength": 0.6}), encoding="utf-8")
    assert recommended_decoder_strength(gpt) == 0.6
    report.write_text(json.dumps({"accepted": False, "strength": 0.6}), encoding="utf-8")
    assert recommended_decoder_strength(gpt) is None
    assert recommended_decoder_strength("") is None


def test_decoder_test_markdown_lists_strength_variants():
    from indextts.training.speech_eval import decoder_test_markdown

    report = {"accepted": True, "metric": "speaker_similarity_real", "speaker_gain": {"mean": 0.0433, "prompts": 12},
              "clips": 36, "wer_increase": 0.0032, "reasons": [], "strength": 0.6, "wer_weight": 4.0,
              "without": {"speaker_similarity_real": 0.7958, "corpus_error_rate": 0.0354, "failure_count": 0},
              "with": {"speaker_similarity_real": 0.8391, "corpus_error_rate": 0.0376, "failure_count": 0},
              "variants": [{"strength": 1.0, "speaker_gain": {"mean": 0.0777}, "wer_increase": 0.0133, "score": 0.0245, "passes": True, "selected": False},
                           {"strength": 0.6, "speaker_gain": {"mean": 0.0433}, "wer_increase": 0.0032, "score": 0.0305, "passes": True, "selected": True}],
              "checkpoint": "loras/v/v.safetensors", "baseline_report": "loras/v/analysis/speech_evaluation/final_test/report.json"}
    text = decoder_test_markdown(report)
    assert "| 0.6 (recommended) | +0.0433 | +0.32 points | +0.0305 | yes |" in text
    assert "| 1 | +0.0777 | +1.33 points | +0.0245 | yes |" in text and "minus 4 times" in text
