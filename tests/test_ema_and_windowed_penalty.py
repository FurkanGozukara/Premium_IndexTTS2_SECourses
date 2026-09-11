from __future__ import annotations

import pytest
import torch

from indextts.gpt.model_v2 import WindowedRepetitionPenaltyLogitsProcessor
from indextts.training.analysis import _EMA_RE
from indextts.training.ema import AdapterEMA, effective_decay, ema_checkpoint_name, ema_epoch_of
from indextts.training.naturalness_ab import DEFAULT_VARIANTS, REPETITION_VARIANTS, load_variants
from indextts.training.train_config import TrainConfig


def test_ema_tracks_updates_and_swaps_weights_for_saving() -> None:
    params = [torch.nn.Parameter(torch.ones(3)), torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.bfloat16))]
    ema = AdapterEMA(params, 0.9)
    assert ema.updates == 0 and all(shadow.dtype == torch.float32 for shadow in ema.shadows)
    with torch.no_grad():
        params[0].mul_(3.0)
        params[1].fill_(2.0)
    used = ema.update()
    # Warm-up: the first update uses (1 + 1) / (10 + 1) instead of 0.9.
    assert used == pytest.approx(effective_decay(0.9, 1)) == pytest.approx(2 / 11)
    expected = 1.0 * used + 3.0 * (1.0 - used)
    assert torch.allclose(ema.shadows[0], torch.full((3,), expected))
    assert torch.allclose(ema.shadows[1], torch.full((2, 2), 2.0 * (1.0 - used)))
    with ema.averaged_weights():
        assert torch.allclose(params[0].data, torch.full((3,), expected))
        assert params[1].dtype == torch.bfloat16
    assert torch.allclose(params[0].data, torch.full((3,), 3.0))  # live weights restored
    for _ in range(200):
        ema.update()
    assert ema.current_decay == pytest.approx(0.9)
    state = ema.state_dict()
    fresh = AdapterEMA(params, 0.9)
    assert fresh.load_state_dict(state) and fresh.updates == ema.updates
    assert torch.allclose(fresh.shadows[0], ema.shadows[0])
    assert not fresh.load_state_dict({"shadows": [torch.zeros(1)]})
    with pytest.raises(ValueError):
        AdapterEMA(params, 1.0)


def test_ema_file_names_are_recognized_and_config_validates() -> None:
    assert ema_checkpoint_name("voice") == "voice_ema"
    assert ema_checkpoint_name("voice", 3) == "voice_ema_epoch_003"
    assert ema_epoch_of("voice_ema_epoch_003") == (True, 3)
    assert ema_epoch_of("voice_ema") == (True, None)
    assert ema_epoch_of("voice_epoch_003") == (False, None)
    assert _EMA_RE.search("voice_ema_epoch_002") and _EMA_RE.search("voice_ema") and not _EMA_RE.search("voice_epoch_002")
    config = TrainConfig(dataset_dir="d", name="n", ema_decay=0.999).validate()
    assert config.ema_decay == 0.999
    assert TrainConfig(dataset_dir="d", name="n").validate().ema_decay == 0.0
    with pytest.raises(ValueError):
        TrainConfig(dataset_dir="d", name="n", ema_decay=1.0).validate()


def test_windowed_penalty_touches_only_recent_generated_codes() -> None:
    processor = WindowedRepetitionPenaltyLogitsProcessor(10.0, 3, prompt_length=2)
    ids = torch.tensor([[9, 9, 5, 6, 7, 8]])  # prompt 9 9, generated 5 6 7 8; the window covers 6 7 8
    scores = torch.zeros(1, 12)
    scores[0, 5] = 2.0
    scores[0, 6] = 2.0
    scores[0, 8] = -1.0
    scores[0, 9] = 3.0
    out = processor(ids, scores.clone())
    assert out[0, 5] == 2.0  # outside the window: untouched
    assert out[0, 6] == pytest.approx(0.2)  # positive score divided
    assert out[0, 8] == pytest.approx(-10.0)  # negative score multiplied
    assert out[0, 9] == 3.0  # prompt codes are never penalized
    beams = torch.tensor([[9, 9, 5, 6, 7, 8], [9, 9, 1, 1, 1, 1]])
    out = processor(beams, torch.ones(2, 12))
    assert out[1, 1] == pytest.approx(0.1) and out[1, 2] == 1.0
    assert processor(torch.tensor([[9, 9]]), torch.ones(1, 12)).sum() == 12
    with pytest.raises(ValueError):
        WindowedRepetitionPenaltyLogitsProcessor(10.0, 0, 0)


def test_repetition_variants_are_selectable_but_not_default() -> None:
    names = {variant.name for variant in load_variants(None)}
    assert names == {variant.name for variant in DEFAULT_VARIANTS}
    extra = {variant.name for variant in load_variants(None, include_extra=True)}
    assert {variant.name for variant in REPETITION_VARIANTS} <= extra
    window = next(variant for variant in REPETITION_VARIANTS if variant.name == "window_16")
    assert window.infer == {"repetition_window": 16}
