import copy

import pytest
import torch
from torch import nn

from indextts.runtime.block_swap import (
    BlockSwapConfig,
    default_swap_tensor_selector,
    enable_block_swap,
    resolve_blocks_to_swap,
)


def test_resolve_blocks_to_swap_math():
    assert resolve_blocks_to_swap(7, 24, 100, 0, 2) == 7
    assert resolve_blocks_to_swap(99, 24, 100, 0, 2) == 24
    assert resolve_blocks_to_swap(-1, 24, 100, 2400, 2) == 0
    assert resolve_blocks_to_swap(-1, 24, 100, 1600, 2) == 10
    assert resolve_blocks_to_swap(-1, 24, 0, 0, 2) == 0


def test_int8_selector_uses_mixed_dtype_buffers():
    class Int8Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight_int8", torch.ones(3, 4, dtype=torch.int8))
            self.register_buffer("weight_scale", torch.ones(4, dtype=torch.float32))

    layer = Int8Layer()
    names = [name for _, name in default_swap_tensor_selector(layer)]
    assert names == ["weight_int8", "weight_scale"]


def test_selector_keeps_lora_branch_resident():
    from indextts.lora.layers import LoRAAdapter

    adapter = LoRAAdapter(nn.Linear(4, 4), rank=2, alpha=2)
    selected = default_swap_tensor_selector(adapter)
    assert (adapter.base, "weight") in selected
    assert all(module not in {adapter.lora_A, adapter.lora_B} for module, _ in selected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("swapped", [0, 4, 12, 23])
def test_gpt2_block_swap_forward_and_kv_decode(swapped):
    from transformers import GPT2Config, GPT2Model

    torch.manual_seed(123)
    config = GPT2Config(
        n_layer=24, n_embd=32, n_head=4, n_positions=32, vocab_size=97,
        bos_token_id=None, eos_token_id=None,
    )
    reference = GPT2Model(config).eval().to("cuda:0", dtype=torch.bfloat16)
    streamed = copy.deepcopy(reference)
    controller = enable_block_swap(
        list(streamed.h),
        swapped,
        BlockSwapConfig("cuda:0", ring_size=2, use_pinned_memory=True),
    )
    ids = torch.randint(0, config.vocab_size, (2, 6), device="cuda:0")
    reference_cache = streamed_cache = None
    for step in range(3):
        current = ids if step == 0 else ids[:, -1:]
        expected = reference(current, past_key_values=reference_cache, use_cache=True)
        actual = streamed(current, past_key_values=streamed_cache, use_cache=True)
        assert torch.allclose(expected.last_hidden_state, actual.last_hidden_state, atol=0, rtol=0)
        reference_cache = expected.past_key_values
        streamed_cache = actual.past_key_values
        ids = torch.cat([ids, torch.randint(0, config.vocab_size, (2, 1), device="cuda:0")], dim=1)
    assert controller.summary()["swapped_blocks"] == swapped
    controller.remove()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_block_swap_checkpointed_adapter_gradients_match():
    from torch.utils.checkpoint import checkpoint

    class AdapterBlock(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.base = nn.Linear(width, width)
            self.base.requires_grad_(False)
            self.adapter_a = nn.Parameter(torch.randn(width, 2) * 0.02)
            self.adapter_b = nn.Parameter(torch.randn(2, width) * 0.02)

        def forward(self, value):
            return torch.relu(self.base(value) + value @ self.adapter_a @ self.adapter_b)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([AdapterBlock(32) for _ in range(6)])

        def forward(self, value):
            for block in self.blocks:
                value = checkpoint(block, value, use_reentrant=True)
            return value

    torch.manual_seed(321)
    reference = Model().cuda()
    streamed = copy.deepcopy(reference)
    value = torch.randn(2, 5, 32, device="cuda:0", requires_grad=True)
    streamed_value = value.detach().clone().requires_grad_(True)
    reference(value).sum().backward()
    controller = enable_block_swap(
        list(streamed.blocks),
        5,
        BlockSwapConfig("cuda:0", supports_backward=True, ring_size=2),
    )
    streamed(streamed_value).sum().backward()
    assert torch.allclose(value.grad, streamed_value.grad, atol=1e-6, rtol=1e-5)
    for expected, actual in zip(reference.blocks, streamed.blocks):
        assert torch.allclose(expected.adapter_a.grad, actual.adapter_a.grad, atol=1e-6, rtol=1e-5)
        assert torch.allclose(expected.adapter_b.grad, actual.adapter_b.grad, atol=1e-6, rtol=1e-5)
    controller.remove()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_backward_mode_rejects_missing_checkpointing():
    blocks = nn.ModuleList([nn.Sequential(nn.Linear(8, 8), nn.ReLU()) for _ in range(3)]).cuda()
    for block in blocks:
        block[0].requires_grad_(False)
    controller = enable_block_swap(
        list(blocks),
        2,
        BlockSwapConfig("cuda:0", supports_backward=True, ring_size=1),
    )
    with pytest.raises(RuntimeError, match="gradient checkpointing"):
        blocks[0](torch.randn(2, 8, device="cuda:0", requires_grad=True))
    controller.remove()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_non_reentrant_checkpointing_can_be_declared():
    from torch.utils.checkpoint import checkpoint

    blocks = nn.ModuleList([nn.Sequential(nn.Linear(8, 8), nn.ReLU()) for _ in range(3)]).cuda()
    for block in blocks:
        block[0].requires_grad_(False)
    controller = enable_block_swap(
        list(blocks),
        2,
        BlockSwapConfig(
            "cuda:0", supports_backward=True, ring_size=1, gradient_checkpointing=True
        ),
    )
    value = torch.randn(2, 8, device="cuda:0", requires_grad=True)
    for block in blocks:
        value = checkpoint(block, value, use_reentrant=False)
    value.sum().backward()
    assert torch.isfinite(value).all()
    controller.remove()
