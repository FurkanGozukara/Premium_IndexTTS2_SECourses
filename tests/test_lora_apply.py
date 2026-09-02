from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from transformers.pytorch_utils import Conv1D

from indextts.lora.apply import (
    apply_lora,
    get_lora_handle,
    inject_adapters,
    list_target_modules,
    merge_lora_for_inference,
    merge_lora_into_model,
    remove_lora,
    set_lora_strength,
    set_training_mode,
    trainable_parameters,
    unmerge_lora_from_model,
)
from indextts.lora.io import LoraMetadata, save_lora
from indextts.lora.layers import LoRAAdapter


class _Attention(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.c_attn = Conv1D(3 * width, width)
        self.c_proj = Conv1D(width, width)


class _Mlp(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.c_fc = Conv1D(4 * width, width)
        self.c_proj = Conv1D(width, 4 * width)


class _Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.attn = _Attention(width)
        self.mlp = _Mlp(width)


class _Gpt(nn.Module):
    def __init__(self, width: int, layers: int) -> None:
        super().__init__()
        self.h = nn.ModuleList([_Block(width) for _ in range(layers)])


class TinyVoice(nn.Module):
    def __init__(self, width: int = 8, layers: int = 2) -> None:
        super().__init__()
        self.gpt = _Gpt(width, layers)
        self.spk_emb_proj = nn.Linear(3, width)


def _make_lora_file(path: Path, full_value: float, seed: int) -> None:
    torch.manual_seed(seed)
    source = TinyVoice()
    targets = list_target_modules(source)
    adapters = inject_adapters(source, 2, 4, 0.0, False, targets)
    for adapter in adapters.values():
        adapter.lora_A.weight.data.normal_()
        adapter.lora_B.weight.data.normal_()
    source.spk_emb_proj.weight.data.fill_(full_value)
    source.spk_emb_proj.bias.data.fill_(-full_value)
    save_lora(
        path,
        adapters,
        {"spk_emb_proj": source.spk_emb_proj},
        LoraMetadata(
            rank=2,
            alpha=4,
            target_modules=targets,
            dataset_name=f"seed-{seed}",
        ),
        dtype=torch.float32,
    )


def test_list_inject_trainables_and_remove() -> None:
    model = TinyVoice()
    targets = list_target_modules(model)
    assert len(targets) == 8
    assert list_target_modules(model, attention=True, mlp=False) == [
        "gpt.h.0.attn.c_attn",
        "gpt.h.0.attn.c_proj",
        "gpt.h.1.attn.c_attn",
        "gpt.h.1.attn.c_proj",
    ]
    original = {key: value.clone() for key, value in model.state_dict().items()}
    adapters = inject_adapters(model, 2, 4, 0.1, True, targets)
    assert all(isinstance(model.get_submodule(path), LoRAAdapter) for path in targets)
    parameters = trainable_parameters(
        model, adapters, {"spk_emb_proj": model.spk_emb_proj}
    )
    assert parameters
    assert all(parameter.requires_grad for parameter in parameters)
    assert not any(adapter.base.weight.requires_grad for adapter in adapters.values())
    remove_lora(model)
    assert all(not isinstance(model.get_submodule(path), LoRAAdapter) for path in targets)
    for key, value in original.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)


def test_apply_hot_swap_strength_and_full_restore(tmp_path: Path) -> None:
    first_path = tmp_path / "first.safetensors"
    second_path = tmp_path / "second.safetensors"
    _make_lora_file(first_path, 3.0, 1)
    _make_lora_file(second_path, 5.0, 2)
    model = TinyVoice()
    original = {key: value.clone() for key, value in model.state_dict().items()}

    first = apply_lora(model, str(first_path), strength=0.5)
    first_adapter_ids = {path: id(value) for path, value in first._adapters.items()}
    assert torch.all(model.spk_emb_proj.weight == 3.0)
    set_lora_strength(first, 1.25)
    assert all(adapter.strength == 1.25 for adapter in first._adapters.values())

    second = apply_lora(model, str(second_path), strength=0.75)
    assert first_adapter_ids == {path: id(value) for path, value in second._adapters.items()}
    assert torch.all(model.spk_emb_proj.weight == 5.0)
    assert get_lora_handle(model) is second
    assert second.targets == list_target_modules(TinyVoice())

    remove_lora(model)
    assert get_lora_handle(model) is None
    for key, value in original.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)


def test_merge_unwraps_adapters() -> None:
    model = TinyVoice(layers=1)
    targets = list_target_modules(model)
    adapters = inject_adapters(model, 2, 2, 0.0, False, targets)
    for adapter in adapters.values():
        adapter.lora_B.weight.data.normal_()
    merge_lora_into_model(model)
    assert all(not isinstance(model.get_submodule(path), LoRAAdapter) for path in targets)


def test_temporary_inference_merge_keeps_wrappers_and_unmerges_exactly() -> None:
    model = TinyVoice(layers=1).bfloat16()
    targets = list_target_modules(model)
    adapters = inject_adapters(model, 2, 2, 0.0, True, targets)
    for adapter in adapters.values():
        adapter.lora_A.weight.data.normal_(std=0.05)
        adapter.lora_B.weight.data.normal_(std=0.05)
        adapter.lora_magnitude.data.mul_(0.97)
    original = {
        path: adapter.base.weight.detach().clone()
        for path, adapter in adapters.items()
    }

    merge_lora_for_inference(model)
    assert all(adapter._merged for adapter in adapters.values())
    assert all(isinstance(model.get_submodule(path), LoRAAdapter) for path in targets)

    unmerge_lora_from_model(model)
    assert all(not adapter._merged for adapter in adapters.values())
    for path, adapter in adapters.items():
        torch.testing.assert_close(adapter.base.weight, original[path], rtol=0, atol=0)


def test_int8_injection_and_training_ste() -> None:
    try:
        from indextts.quant.convrot_int8 import ConvRotInt8Linear
    except ImportError:
        pytest.skip("ConvRotInt8Linear is unavailable")

    model = nn.Module()
    model.proj = ConvRotInt8Linear(4, 8, group_size=4)
    model.proj.weight_int8.copy_(torch.randint(-10, 10, (8, 4), dtype=torch.int8))
    model.proj.weight_scale.fill_(0.01)
    model.proj.bias.data.zero_()
    adapters = inject_adapters(model, 2, 2, 0.0, True, ["proj"])
    output = model.proj(torch.randn(2, 3, 4))
    assert output.shape == (2, 3, 8)
    set_training_mode(model, True)
    assert adapters["proj"].base.training_ste
    with pytest.raises(TypeError, match="int8"):
        merge_lora_into_model(model)
    remove_lora(model)
    assert not model.proj.training_ste
