"""The decoder adapter caches prompt features once and frees the semantic encoder."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from indextts.training.decoder_adapter import DecoderAdapterTrainer


def _trainer(device: str) -> DecoderAdapterTrainer:
    trainer = DecoderAdapterTrainer.__new__(DecoderAdapterTrainer)
    trainer.device = torch.device(device)
    trainer._prompt_cache = {}
    trainer._cache_bytes = 0
    trainer._cache_limit = 10 ** 9
    trainer._semantic_offloaded = False
    trainer.cancel_callback = lambda: False
    trainer.features = SimpleNamespace(semantic_model=torch.nn.Linear(4, 4))
    trainer.status = SimpleNamespace(log=lambda message: logs.append(message), write=lambda **kwargs: None)
    logs: list[str] = []
    trainer._logs = logs
    calls: list[str] = []
    trainer._calls = calls

    def prompt(record):
        calls.append(str(record["id"]))
        stored = (torch.zeros(1), torch.zeros(1), torch.zeros(1))
        trainer._prompt_cache[str(record["id"])] = stored
        trainer._cache_bytes += 6
        return stored

    trainer._prompt = prompt
    return trainer


def test_prompt_features_are_cached_once_per_clip_and_cpu_keeps_the_encoder():
    trainer = _trainer("cpu")
    records = [{"id": "a"}, {"id": "b"}, {"id": "a"}]
    trainer._prepare_prompt_features(records)
    assert trainer._calls == ["a", "b"]
    assert trainer._semantic_offloaded is False
    assert next(trainer.features.semantic_model.parameters()).device.type == "cpu"
    with trainer._semantic_encoder_on_device():
        assert next(trainer.features.semantic_model.parameters()).device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.gpu
def test_prompt_features_free_the_encoder_on_cuda_and_bring_it_back_on_demand():
    trainer = _trainer("cuda:0")
    trainer.features.semantic_model.to("cuda:0")
    trainer._prepare_prompt_features([{"id": "a"}])
    assert trainer._semantic_offloaded is True
    assert next(trainer.features.semantic_model.parameters()).device.type == "cpu"
    assert any("moved the semantic encoder to the CPU" in line for line in trainer._logs)
    with trainer._semantic_encoder_on_device():
        assert next(trainer.features.semantic_model.parameters()).device.type == "cuda"
    assert next(trainer.features.semantic_model.parameters()).device.type == "cpu"
