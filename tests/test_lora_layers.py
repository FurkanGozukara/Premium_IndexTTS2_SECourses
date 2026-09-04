from __future__ import annotations

import pytest
import torch
from torch import nn
from transformers.pytorch_utils import Conv1D

from indextts.lora.layers import LoRAAdapter


def _linear_weight(base: nn.Module) -> torch.Tensor:
    return base.weight if isinstance(base, nn.Linear) else base.weight.T


@pytest.mark.parametrize("base_kind", ["linear", "conv1d"])
@pytest.mark.parametrize("use_dora", [False, True])
def test_forward_matches_reference(base_kind: str, use_dora: bool) -> None:
    torch.manual_seed(7)
    base = nn.Linear(5, 8) if base_kind == "linear" else Conv1D(8, 5)
    adapter = LoRAAdapter(base, rank=3, alpha=5.0, use_dora=use_dora)
    adapter.lora_A.weight.data.normal_(std=0.2)
    adapter.lora_B.weight.data.normal_(std=0.2)
    if use_dora:
        adapter.lora_magnitude.data.uniform_(0.5, 1.5)
    adapter.strength = 0.65
    x = torch.randn(2, 4, 5)

    weight = _linear_weight(base)
    delta_output = adapter.scaling * (x @ adapter.lora_A.weight.T) @ adapter.lora_B.weight.T
    if use_dora:
        delta_weight = adapter.scaling * adapter.lora_B.weight @ adapter.lora_A.weight
        magnitude_scale = adapter.lora_magnitude / (weight + delta_weight.detach()).norm(dim=1)
        effective_scale = 1 + adapter.strength * (magnitude_scale - 1)
        expected = base(x)
        expected = expected + (effective_scale - 1) * (x @ weight.T)
        expected = expected + effective_scale * adapter.strength * delta_output
    else:
        expected = base(x) + adapter.strength * delta_output

    torch.testing.assert_close(adapter(x), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("use_dora", [False, True])
@pytest.mark.parametrize("base", [nn.Linear(5, 7), Conv1D(7, 5)])
def test_strength_zero_and_merge(use_dora: bool, base: nn.Module) -> None:
    torch.manual_seed(11)
    adapter = LoRAAdapter(base, rank=2, alpha=3.0, use_dora=use_dora)
    adapter.lora_A.weight.data.normal_()
    adapter.lora_B.weight.data.normal_()
    if use_dora:
        adapter.lora_magnitude.data.uniform_(0.8, 1.2)
    x = torch.randn(3, 5)

    adapter.strength = 0.0
    torch.testing.assert_close(adapter(x), base(x), rtol=0, atol=0)
    adapter.strength = 0.7
    before = adapter(x).detach()
    adapter.merge_into_base()
    torch.testing.assert_close(adapter(x), before, rtol=1e-5, atol=2e-6)


def test_bf16_base_keeps_fp32_adapters_and_supports_nd_input() -> None:
    container = nn.Module()
    container.projection = LoRAAdapter(nn.Linear(8, 6).bfloat16(), rank=2, alpha=2)
    container.projection.lora_B.weight.data.normal_()
    container.bfloat16()
    assert container.projection.base.weight.dtype == torch.bfloat16
    assert container.projection.lora_A.weight.dtype == torch.float32
    assert container.projection.lora_B.weight.dtype == torch.float32
    output = container.projection(torch.randn(2, 3, 8, dtype=torch.bfloat16))
    assert output.shape == (2, 3, 6)
    assert output.dtype == torch.bfloat16


def test_gradients_only_flow_to_adapter_parameters() -> None:
    adapter = LoRAAdapter(nn.Linear(4, 6), rank=2, alpha=2, use_dora=True)
    adapter.lora_B.weight.data.normal_()
    adapter(torch.randn(3, 4)).square().mean().backward()

    assert all(parameter.grad is None for parameter in adapter.base.parameters())
    assert adapter.lora_A.weight.grad is not None
    assert adapter.lora_B.weight.grad is not None
    assert adapter.lora_magnitude.grad is not None


@pytest.mark.parametrize("base", [nn.Linear(5, 7), Conv1D(7, 5)])
def test_dora_magnitude_starts_at_base_row_norm(base: nn.Module) -> None:
    expected = _linear_weight(base).float().norm(dim=1)
    adapter = LoRAAdapter(base, rank=2, alpha=2, use_dora=True)
    torch.testing.assert_close(adapter.lora_magnitude, expected)
    assert adapter.state_dict_keys() == [
        "lora_A.weight",
        "lora_B.weight",
        "lora_magnitude",
    ]


def _warm_dora_cache(adapter: LoRAAdapter, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        result = adapter(x)
    assert adapter._effective_scale_cache is not None
    assert adapter._effective_scale_cache.dtype == torch.float32
    return result


def test_dora_cache_reuse_and_invalidation() -> None:
    torch.manual_seed(17)
    adapter = LoRAAdapter(nn.Linear(5, 7), rank=2, alpha=3, use_dora=True)
    adapter.lora_A.weight.data.normal_(std=0.2)
    adapter.lora_B.weight.data.normal_(std=0.2)
    adapter.eval()
    x = torch.randn(3, 5)

    _warm_dora_cache(adapter, x)
    first_cache = adapter._effective_scale_cache
    _warm_dora_cache(adapter, x)
    assert adapter._effective_scale_cache is first_cache

    # Storage-only rebinding is how block swap moves value-identical tensors.
    adapter.base.weight = nn.Parameter(
        adapter.base.weight.detach().clone(), requires_grad=False
    )
    _warm_dora_cache(adapter, x)
    assert adapter._effective_scale_cache is first_cache

    adapter.strength = 0.4
    assert adapter._effective_scale_cache is None
    _warm_dora_cache(adapter, x)

    state = {name: tensor.clone() for name, tensor in adapter.state_dict().items()}
    state["lora_magnitude"].mul_(1.1)
    adapter.load_state_dict(state)
    assert adapter._effective_scale_cache is None
    _warm_dora_cache(adapter, x)

    adapter.train()
    assert adapter._effective_scale_cache is None
    with torch.no_grad():
        adapter(x)
    assert adapter._effective_scale_cache is not None
    adapter.eval()
    assert adapter._effective_scale_cache is None
    _warm_dora_cache(adapter, x)

    adapter.enabled = False
    assert adapter._effective_scale_cache is None
    adapter.enabled = True
    _warm_dora_cache(adapter, x)


@pytest.mark.parametrize("base_kind", ["linear", "conv1d"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_dora", [False, True])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ],
)
def test_training_and_inference_paths_agree(
    base_kind: str, dtype: torch.dtype, use_dora: bool, device: str
) -> None:
    torch.manual_seed(23)
    if base_kind == "linear":
        base = nn.Linear(8, 6, bias=False).to(device=device, dtype=dtype)
    else:
        base = Conv1D(6, 8).to(device=device, dtype=dtype)
    adapter = LoRAAdapter(base, rank=3, alpha=4, use_dora=use_dora)
    adapter.lora_A.weight.data.normal_(std=0.08)
    adapter.lora_B.weight.data.normal_(std=0.08)
    if use_dora:
        adapter.lora_magnitude.data.mul_(0.9)
    adapter.strength = 0.7
    x = torch.randn(2, 4, 8, device=device, dtype=dtype)

    adapter.train()
    training_output = adapter(x).detach()
    adapter.eval()
    with torch.no_grad():
        inference_output = adapter(x)

    if dtype == torch.bfloat16:
        torch.testing.assert_close(
            inference_output, training_output, rtol=2e-2, atol=2e-2
        )
    else:
        torch.testing.assert_close(
            inference_output, training_output, rtol=2e-5, atol=2e-6
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("base_kind", ["linear", "conv1d"])
@pytest.mark.parametrize("use_dora", [False, True])
def test_cuda_bf16_token_fast_path_matches_training(
    base_kind: str, use_dora: bool
) -> None:
    torch.manual_seed(27)
    base = (
        nn.Linear(64, 96).cuda().bfloat16()
        if base_kind == "linear"
        else Conv1D(96, 64).cuda().bfloat16()
    )
    adapter = LoRAAdapter(base, rank=32, alpha=32, use_dora=use_dora)
    adapter.lora_A.weight.data.normal_(std=0.04)
    adapter.lora_B.weight.data.normal_(std=0.04)
    if use_dora:
        adapter.lora_magnitude.data.mul_(0.92)
    adapter.strength = 0.8
    x = torch.randn(3, 64, device="cuda", dtype=torch.bfloat16)

    adapter.train()
    training_output = adapter(x).detach()
    adapter.eval()
    with torch.no_grad():
        inference_output = adapter(x)

    torch.testing.assert_close(
        inference_output, training_output, rtol=3e-2, atol=3e-2
    )
    if use_dora:
        assert adapter._effective_scale_cache is not None


@pytest.mark.parametrize("base_kind", ["linear", "conv1d"])
@pytest.mark.parametrize("use_dora", [False, True])
def test_unmerge_from_bf16_base_round_trip(
    base_kind: str, use_dora: bool
) -> None:
    torch.manual_seed(29)
    base = (
        nn.Linear(8, 6).bfloat16()
        if base_kind == "linear"
        else Conv1D(6, 8).bfloat16()
    )
    adapter = LoRAAdapter(base, rank=2, alpha=2, use_dora=use_dora)
    adapter.lora_A.weight.data.normal_(std=0.1)
    adapter.lora_B.weight.data.normal_(std=0.1)
    if use_dora:
        adapter.lora_magnitude.data.mul_(0.95)
    adapter.strength = 0.6
    adapter.eval()
    x = torch.randn(3, 8, dtype=torch.bfloat16)
    original_weight = base.weight.detach().clone()

    with torch.no_grad():
        before = adapter(x)
    adapter.merge_into_base()
    assert adapter._original_base_weight is not None
    assert adapter._original_base_weight.device.type == "cpu"
    with torch.no_grad():
        merged = adapter(x)
    torch.testing.assert_close(merged, before, rtol=3e-2, atol=3e-2)

    adapter.unmerge_from_base()
    assert adapter.enabled
    assert not adapter._merged
    assert adapter._original_base_weight is None
    torch.testing.assert_close(base.weight, original_weight, rtol=0, atol=0)
    with torch.no_grad():
        after = adapter(x)
    torch.testing.assert_close(after, before, rtol=0, atol=0)
