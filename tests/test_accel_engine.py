from types import SimpleNamespace

import torch
from torch import nn
from transformers import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

from indextts.accel.accel_engine import AccelInferenceEngine
from indextts.accel.gpt2_accel import GPT2AccelModel, share_gpt_layers
from indextts.accel.kv_manager import KVCacheManager, Seq
from indextts.gpt.model_v2 import UnifiedVoice


def test_accel_gpt_accepts_transformers_tensor_and_legacy_tuple_blocks() -> None:
    class Block:
        def __init__(self, returns_tuple: bool):
            self.returns_tuple = returns_tuple

        def __call__(self, value):
            result = value + 1
            return (result,) if self.returns_tuple else result

    model = SimpleNamespace(
        drop=nn.Identity(),
        h=[Block(False), Block(True)],
        ln_f=nn.Identity(),
    )
    result = GPT2AccelModel.forward(
        model,
        inputs_embeds=torch.zeros(1, 3, 4),
        return_dict=True,
    )
    assert result.last_hidden_state.shape == (1, 3, 4)
    assert torch.all(result.last_hidden_state == 2)


def test_accel_sampling_applies_repetition_top_k_and_top_p() -> None:
    request = Seq([0, 1])
    request.append_token(2)
    logits = torch.tensor([[9.0, 8.0, 7.0, 6.0]])

    filtered = AccelInferenceEngine._process_sampling_logits(
        logits,
        [request],
        temperature=1.0,
        top_k=2,
        top_p=0.7,
        repetition_penalty=2.0,
    )

    assert torch.isfinite(filtered).sum().item() == 1
    assert filtered.argmax(dim=-1).item() == 0
    assert filtered[0, 2].isneginf()


def test_accel_sampling_accepts_gui_disabled_top_k_and_zero_top_p() -> None:
    filtered = AccelInferenceEngine._process_sampling_logits(
        torch.tensor([[1.0, 3.0, 2.0]]),
        [Seq([0])],
        temperature=1.0,
        top_k=None,
        top_p=0.0,
        repetition_penalty=1.0,
    )

    assert torch.isfinite(filtered).sum().item() == 1
    assert filtered.argmax(dim=-1).item() == 1


def test_accel_generation_keeps_eos_in_returned_sequence() -> None:
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(hidden_size=4)

        def forward(self, input_ids=None, **_kwargs):
            batch, length = input_ids.shape
            return SimpleNamespace(
                last_hidden_state=torch.zeros(batch, length, 4)
            )

    class StopHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(1))

        def forward(self, hidden):
            logits = torch.zeros(hidden.shape[0], 8)
            logits[:, 7] = 10
            return logits

    class Cache:
        def allocate(self, _sequence):
            return None

        def remove_seq(self, _sequence):
            return None

    engine = AccelInferenceEngine.__new__(AccelInferenceEngine)
    engine.model = Model()
    engine.lm_head = StopHead()
    engine.kv_manager = Cache()
    engine.use_cuda_graph = False
    engine.graph_captured = False
    engine.current_sequences = []
    engine._prepare_prefill = lambda requests: (None, None)
    engine._prepare_sample = lambda requests, temperature: torch.ones(len(requests))

    output = engine.generate(
        torch.tensor([[4, 5]]),
        max_new_tokens=4,
        temperature=0,
        top_k=0,
        top_p=1.0,
        stop_tokens=[7],
    )

    assert output.tolist() == [[4, 5, 7]]


def test_kv_cache_preserves_requested_compute_dtype() -> None:
    cache = KVCacheManager(
        num_layers=1,
        num_heads=1,
        head_dim=4,
        block_size=8,
        num_blocks=2,
        dtype=torch.bfloat16,
    )
    assert cache.kv_cache.dtype == torch.bfloat16


def _tiny_gpt2_config() -> GPT2Config:
    return GPT2Config(vocab_size=16, n_positions=32, n_embd=32, n_layer=2, n_head=2)


def test_accel_model_runs_on_the_gpt_layers_not_a_copy() -> None:
    # A load_state_dict(strict=False) copy kept random weights for INT8 ConvRot layers (weight_int8) and
    # LoRA / DoRA wrappers. The acceleration model shares the live modules and follows a replacement.
    config = _tiny_gpt2_config()
    source, accel = GPT2Model(config), GPT2AccelModel(config)
    share_gpt_layers(accel, source)
    for accel_block, block in zip(accel.h, source.h):
        assert accel_block.ln_1 is block.ln_1 and accel_block.ln_2 is block.ln_2 and accel_block.mlp is block.mlp
        assert accel_block.attn.c_attn is block.attn.c_attn and accel_block.attn.c_proj is block.attn.c_proj
    assert accel.ln_f is source.ln_f

    wrapper = nn.Identity()
    source.h[0].attn.c_attn = wrapper
    share_gpt_layers(accel, source)
    assert accel.h[0].attn.c_attn is wrapper


def test_binding_accel_layers_drops_captured_graphs() -> None:
    config = _tiny_gpt2_config()
    source, accel = GPT2Model(config), GPT2AccelModel(config)
    engine = AccelInferenceEngine.__new__(AccelInferenceEngine)
    engine.model = accel
    engine.graphs, engine.graph_vars, engine.graph_pool, engine.graph_captured = {1: object()}, {}, object(), True

    UnifiedVoice.bind_accel_layers(SimpleNamespace(accel_engine=engine, gpt=source))

    assert engine.graphs == {} and engine.graph_vars is None and engine.graph_pool is None
    assert engine.graph_captured is False
    assert accel.h[1].mlp is source.h[1].mlp
    UnifiedVoice.bind_accel_layers(SimpleNamespace(accel_engine=None, gpt=source))  # no engine: nothing to bind
