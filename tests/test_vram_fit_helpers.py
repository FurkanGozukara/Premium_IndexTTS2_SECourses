"""Memory-fit helpers used by dataset preparation, the audit and the feature cache."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from indextts.training.features import feature_batch_size_for_free_vram
from indextts.training.whisper_asr import install_lean_encoder, whisper_device_for_free_vram


def test_feature_batch_size_follows_free_vram():
    assert feature_batch_size_for_free_vram(30.0) == 4
    assert feature_batch_size_for_free_vram(10.5) == 4
    assert feature_batch_size_for_free_vram(10.4) == 2
    assert feature_batch_size_for_free_vram(7.0) == 2
    assert feature_batch_size_for_free_vram(6.9) == 1
    assert feature_batch_size_for_free_vram(0.0) == 1
    assert feature_batch_size_for_free_vram(None) == 4
    assert feature_batch_size_for_free_vram("bad") == 4
    assert feature_batch_size_for_free_vram(30.0, requested=2) == 2
    assert feature_batch_size_for_free_vram(8.0, requested=1) == 1


def test_second_opinion_device_moves_to_cpu_only_when_vram_is_short():
    assert whisper_device_for_free_vram("cuda:0", 5.0, required_gb=4.5) == "cuda:0"
    assert whisper_device_for_free_vram("cuda:0", 4.4, required_gb=4.5) == "cpu"
    assert whisper_device_for_free_vram("cuda:1", None, required_gb=4.5) == "cuda:1"
    assert whisper_device_for_free_vram("cpu", 0.0, required_gb=4.5) == "cpu"
    assert whisper_device_for_free_vram("cuda:0", "n/a", required_gb=4.5) == "cuda:0"


class _FakeEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    def forward(self, input_features, **kwargs):
        self.calls.append(dict(kwargs))
        return SimpleNamespace(last_hidden_state=input_features * 2, attentions=None)


class _FakeWhisper:
    def __init__(self) -> None:
        self.encoder = _FakeEncoder()

    def get_encoder(self):
        return self.encoder

    def generate(self, input_features):
        # transformers asks the whole model for attentions when word timestamps are requested.
        return self.encoder(input_features, output_attentions=True, return_dict=True)


def test_lean_encoder_drops_the_attention_request_and_keeps_the_outputs():
    model = _FakeWhisper()
    assert install_lean_encoder(model) is True
    assert install_lean_encoder(model) is False  # installed once
    features = torch.ones(1, 128, 3000)
    output = model.generate(features)
    assert torch.equal(output.last_hidden_state, features * 2)
    assert model.encoder.calls == [{"output_attentions": False, "return_dict": True}]


def test_lean_encoder_skips_models_without_an_encoder():
    assert install_lean_encoder(SimpleNamespace(generate=lambda *a, **k: None)) is False
