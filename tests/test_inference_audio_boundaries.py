"""A shorter utterance must not receive audio from a longer batch member."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from indextts.infer_v2_5 import IndexTTS2
from indextts.s2mel.modules.length_regulator import InterpolateRegulator


class TemporalCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 4)
        self.conv = nn.Conv1d(4, 4, 3, padding=1)

    def decode(self, codes):
        x = self.conv(self.embedding(codes).transpose(1, 2))
        return x.repeat_interleave(2, dim=2).transpose(1, 2)


def make_engine():
    torch.manual_seed(17)
    engine = IndexTTS2.__new__(IndexTTS2)
    engine.residency = SimpleNamespace(use=lambda _name: nullcontext())
    engine._use_s2mel = nullcontext
    engine.semantic_codec = TemporalCodec().eval()
    engine.s2mel = SimpleNamespace(models={
        "length_regulator": InterpolateRegulator(
            channels=4, sampling_ratios=(1,), in_channels=4,
        ).eval(),
    })
    engine.bigvgan = nn.Sequential(nn.Conv1d(4, 1, 3, padding=1), nn.Tanh()).eval()
    return engine


@pytest.mark.parametrize("duration_factor", [0.7, 1.0, 1.003, 1.4])
def test_conditioning_matches_individual_decode_at_real_endpoints(duration_factor):
    engine = make_engine()
    lengths = torch.tensor([5, 11, 5])
    codes = torch.randint(0, 32, (3, 11))
    # An EOS/pad id is outside the codec's vocabulary. It must never be decoded.
    codes[0, 5:] = 8193
    codes[2, 5:] = 8193
    batched, targets = engine._prepare_batched_conditioning(
        codes, lengths, duration_factor=duration_factor,
    )
    for row, length in enumerate(lengths.tolist()):
        with torch.inference_mode():
            semantic = engine.semantic_codec.decode(codes[row:row + 1, :length])
            target = max(1, round(semantic.size(1) * 1.72 * duration_factor))
            individual = engine.s2mel.models["length_regulator"](
                semantic, ylens=torch.tensor([target]), n_quantizers=3, f0=None,
            )[0]
        assert targets[row] == target
        torch.testing.assert_close(batched[row:row + 1, :target], individual)
        assert not batched[row, target:].any()


def test_vocoder_receives_only_real_mels_and_preserves_order():
    engine = make_engine()
    lengths = torch.tensor([5, 11, 5])
    mels = torch.randn(3, 4, 11)
    mels[0, :, 5:] = 100
    mels[2, :, 5:] = -100
    result = engine._vocode_batched_mels(mels, lengths)
    for row, length in enumerate(lengths.tolist()):
        with torch.inference_mode():
            expected = engine.bigvgan(mels[row:row + 1, :, :length]).squeeze(1)
            expected = torch.clamp(32767 * expected, -32767.0, 32767.0)
        torch.testing.assert_close(result[row], expected)


def test_empty_speech_is_reported_instead_of_decoding_a_fake_token():
    engine = make_engine()
    with pytest.raises(RuntimeError, match="no speech"):
        engine._prepare_batched_conditioning(
            torch.full((1, 1), 8193), torch.tensor([0]), duration_factor=1.0,
        )


def test_invalid_batch_length_is_rejected():
    engine = make_engine()
    with pytest.raises(ValueError, match="exceeds"):
        engine._vocode_batched_mels(torch.zeros(1, 4, 2), torch.tensor([3]))


def test_acoustic_decoder_ignores_values_beyond_each_utterance():
    from indextts.s2mel.modules.wavenet import WN

    torch.manual_seed(17)
    decoder = WN(4, kernel_size=3, dilation_rate=1, n_layers=2).eval()
    features = torch.randn(2, 4, 12)
    mask = (torch.arange(12)[None, None, :] < torch.tensor([5, 12])[:, None, None]).float()
    changed = features.clone()
    changed[0, :, 5:] = 100
    with torch.inference_mode():
        expected = decoder(features, mask)
        actual = decoder(changed, mask)
    torch.testing.assert_close(actual, expected)
