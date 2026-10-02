"""AuK INT8 layer selection and the slim Qwen2.5-Omni Thinker folder helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from indextts.auk import thinker_files  # noqa: E402
from indextts.quant.convrot_int8 import ConvRotInt8Linear, convert_gpt_checkpoint  # noqa: E402
from quantize_auk import dit_targets, text_targets  # noqa: E402


def _auk_keys() -> list[str]:
    keys = ["layer_weights", "layer_scale", "transformer.rotary_embed.inv_freq", "transformer.txt_norm.weight"]
    for name in ("txt_proj", "time_embed.time_mlp.0", "time_embed.time_mlp.2", "norm_out.linear", "proj_out",
                 "audio_embed.linear", "audio_embed.conv_pos_embed.conv1d.0", "audio_embed.conv_pos_embed.conv1d.2"):
        keys += [f"transformer.{name}.weight", f"transformer.{name}.bias"]
    for index in range(10):
        base = f"transformer.transformer_blocks.{index}"
        for name in ("attn.to_qkv", "attn.to_qkv_c", "attn.to_out.0", "attn.to_out_c", "attn_norm_c.linear",
                     "attn_norm_x.linear"):
            keys += [f"{base}.{name}.weight", f"{base}.{name}.bias"]
        keys += [f"{base}.attn.{norm}.weight" for norm in ("q_norm", "k_norm", "c_q_norm", "c_k_norm")]
        keys += [f"{base}.{ff}.{part}.weight" for ff in ("ff_c", "ff_x") for part in ("linear_in", "linear_out")]
    for index in range(20):
        base = f"transformer.single_transformer_blocks.{index}"
        for name in ("attn.to_qkv", "attn.to_out.0", "attn_norm.linear"):
            keys += [f"{base}.{name}.weight", f"{base}.{name}.bias"]
        keys += [f"{base}.attn.q_norm.weight", f"{base}.attn.k_norm.weight"]
        keys += [f"{base}.ff.linear_in.weight", f"{base}.ff.linear_out.weight"]
    return keys


def test_dit_layer_sets() -> None:
    keys = _auk_keys()
    blocks, adaln, every = (dit_targets(keys, name) for name in ("blocks", "blocks_adaln", "all"))
    assert (len(blocks), len(adaln), len(every)) == (160, 200, 204)
    assert set(blocks) < set(adaln) < set(every)
    assert set(adaln) - set(blocks) == {key.removesuffix(".weight") for key in keys if ".attn_norm" in key and key.endswith("linear.weight")}
    assert set(every) - set(adaln) == {"transformer.txt_proj", "transformer.time_embed.time_mlp.0",
                                       "transformer.time_embed.time_mlp.2", "transformer.norm_out.linear"}
    for kept in ("transformer.proj_out", "transformer.audio_embed.linear", "transformer.audio_embed.conv_pos_embed.conv1d.0"):
        assert kept not in every
    with pytest.raises(ValueError):
        dit_targets(keys, "everything")


def test_text_targets_select_the_decoder_projections_only() -> None:
    keys = ["thinker.model.embed_tokens.weight", "thinker.model.norm.weight", "thinker.lm_head.weight",
            "thinker.audio_tower.layers.0.self_attn.q_proj.weight"]
    for index in range(36):
        base = f"thinker.model.layers.{index}"
        keys += [f"{base}.self_attn.{name}.weight" for name in ("q_proj", "k_proj", "v_proj", "o_proj")]
        keys += [f"{base}.self_attn.{name}.bias" for name in ("q_proj", "k_proj", "v_proj")]
        keys += [f"{base}.mlp.{name}.weight" for name in ("gate_proj", "up_proj", "down_proj")]
        keys += [f"{base}.input_layernorm.weight", f"{base}.post_attention_layernorm.weight"]
    targets = text_targets(keys, "thinker.model.")
    assert len(targets) == 252 and targets[0].startswith("layers.")
    assert text_targets([key.removeprefix("thinker.") for key in keys], "model.") == targets


def _snapshot(folder: Path) -> dict[str, torch.Tensor]:
    """A miniature Qwen2.5-Omni snapshot with every component the slim folder sorts."""
    folder.mkdir(parents=True)
    state = {
        "thinker.model.embed_tokens.weight": torch.randn(16, 8).bfloat16(),
        "thinker.model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8).bfloat16(),
        "thinker.audio_tower.conv1.weight": torch.randn(4, 4).bfloat16(),
        "thinker.visual.patch_embed.proj.weight": torch.randn(4, 4).bfloat16(),
        "thinker.visual.merger.mlp.0.weight": torch.randn(4, 4).bfloat16(),
        "thinker.visual.blocks.0.attn.qkv.weight": torch.randn(4, 4).bfloat16(),
        "thinker.lm_head.weight": torch.randn(16, 8).bfloat16(),
        "talker.model.embed_tokens.weight": torch.randn(4, 4).bfloat16(),
        "token2wav.code2wav_dit_model.proj_out.weight": torch.randn(4, 4),
    }
    keys = sorted(state)
    weight_map = {}
    for index, names in enumerate((keys[:5], keys[5:]), 1):
        name = f"model-{index:05d}-of-00002.safetensors"
        save_file({key: state[key] for key in names}, str(folder / name))
        weight_map.update({key: name for key in names})
    (folder / thinker_files.WEIGHTS_INDEX).write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    config = {"model_type": "qwen2_5_omni", "transformers_version": "4.50.0",
              "thinker_config": {"model_type": "qwen2_5_omni_thinker", "_attn_implementation_autoset": True,
                                 "text_config": {"tie_word_embeddings": False, "hidden_size": 8},
                                 "vision_config": {"depth": 32, "fullatt_block_indexes": [7, 15]},
                                 "audio_config": {"d_model": 4}}}
    (folder / "config.json").write_text(json.dumps(config))
    for name in thinker_files.PROCESSOR_FILES + ("LICENSE", "spk_dict.pt"):
        (folder / name).write_text("{}")
    return state


def test_slim_folder_keeps_only_the_thinker_parts_auk_runs(tmp_path: Path) -> None:
    state = _snapshot(tmp_path / "full")
    slim = tmp_path / "slim"
    report = thinker_files.build_slim_folder(tmp_path / "full", slim, max_shard_bytes=200, progress=None)
    assert thinker_files.is_complete(slim) and report["tensors"] == 5
    stored = {}
    for path in thinker_files.weight_files(slim):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            stored.update({key: handle.get_tensor(key) for key in handle.keys()})
    assert sorted(stored) == ["audio_tower.conv1.weight", "model.embed_tokens.weight",
                              "model.layers.0.self_attn.q_proj.weight", "visual.merger.mlp.0.weight",
                              "visual.patch_embed.proj.weight"]
    for key, value in stored.items():
        assert torch.equal(value, state[f"thinker.{key}"])
    config = json.loads((slim / "config.json").read_text())
    assert config["model_type"] == "qwen2_5_omni_thinker"
    assert config["architectures"] == ["Qwen2_5OmniThinkerForConditionalGeneration"]
    assert config["tie_word_embeddings"] is True and config["text_config"]["tie_word_embeddings"] is True
    assert config["vision_config"]["depth"] == 0 and config["vision_config"]["fullatt_block_indexes"] == []
    assert "_attn_implementation_autoset" not in config and config[thinker_files.SLIM_MARKER]["version"] == 1
    assert all((slim / name).is_file() for name in thinker_files.PROCESSOR_FILES + ("LICENSE",))
    assert not (slim / "spk_dict.pt").exists()
    assert len(report["shards"]) > 1


def test_thinker_folder_prefers_a_complete_slim_folder(tmp_path: Path) -> None:
    models = tmp_path / "models"
    _snapshot(models / thinker_files.FULL_FOLDER)
    assert thinker_files.thinker_folder(models) == models / thinker_files.FULL_FOLDER
    slim = thinker_files.slim_folder(models)
    thinker_files.build_slim_folder(models / thinker_files.FULL_FOLDER, slim, progress=None)
    assert thinker_files.thinker_folder(models) == slim
    shard = thinker_files.weight_files(slim)[0]
    shard.write_bytes(shard.read_bytes()[:-4])  # an interrupted download
    assert not thinker_files.is_complete(slim)
    assert thinker_files.thinker_folder(models) == models / thinker_files.FULL_FOLDER


class _RopeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(64, 32)
        self.register_buffer("inv_freq", 1.0 / (10000 ** (torch.arange(0, 16, 2).float() / 16)), persistent=False)


def test_int8_text_loader_keeps_float32_rope_buffers(tmp_path: Path) -> None:
    source = tmp_path / "model.safetensors"
    model = _RopeModel()
    save_file({key: value.contiguous() for key, value in model.state_dict().items()}, str(source))
    destination = tmp_path / "int8.safetensors"
    convert_gpt_checkpoint(str(source), str(destination), device="cpu", progress=None, group_sizes=(16,),
                           linear_targets=["proj"])
    loaded = _RopeModel()
    expected = loaded.inv_freq.clone()
    thinker_files.load_int8_text_model(loaded, destination, device="cpu")
    assert isinstance(loaded.proj, ConvRotInt8Linear) and loaded.proj.bias.dtype == torch.bfloat16
    assert loaded.inv_freq.dtype == torch.float32 and torch.equal(loaded.inv_freq, expected)
    assert loaded.proj.kernel_mode == "w8a16"
