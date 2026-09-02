from __future__ import annotations

import pytest

from indextts.training.train_config import TrainConfig


def test_reference_sampling_and_evaluation_defaults() -> None:
    config = TrainConfig(dataset_dir="dataset", name="adapter").validate()

    assert config.rank == 128
    assert config.alpha == 129.0
    assert config.learning_rate == 5e-5
    assert config.epochs == 20
    assert config.speaker_ref_mode == "other"
    assert config.emo_ref_mode == "follow_speaker"
    assert config.val_reference_mode == "other"
    assert config.keep_last_n == 0
    assert config.epoch_train_state is False
    assert config.sample_language == "auto"
    assert config.sample_seed == -1
    assert config.sample_temperature == 0.8
    assert config.sample_top_p == 0.8
    assert config.sample_top_k == 30
    assert config.sample_repetition_penalty == 10.0
    assert config.sample_num_beams == 3
    assert config.sample_emo_alpha == 0.65
    assert config.sample_diffusion_steps == 25
    assert config.sample_inference_cfg_rate == 0.7
    assert config.sample_max_text_tokens == 60
    assert config.sample_length_penalty == 0.0
    assert config.sample_max_mel_tokens == 1500
    assert config.sample_speaking_rate == 1.0
    assert config.eval_train_subset == 48
    assert config.eval_strengths == "1.0"
    assert config.eval_include_base is True


def test_new_fields_round_trip_and_old_configs_receive_defaults() -> None:
    config = TrainConfig(
        dataset_dir="dataset",
        name="adapter",
        emo_ref_mode="follow_speaker",
        val_reference_mode="other",
        sample_language="ja",
        sample_seed=987,
        sample_temperature=1.1,
        sample_top_p=0.6,
        sample_top_k=0,
        sample_repetition_penalty=7.5,
        sample_num_beams=2,
        sample_emo_alpha=0.4,
        sample_diffusion_steps=31,
        sample_inference_cfg_rate=1.2,
        sample_max_text_tokens=77,
        sample_length_penalty=-0.25,
        sample_max_mel_tokens=999,
        sample_speaking_rate=0.81,
        epoch_train_state=True,
        eval_train_subset=9,
        eval_strengths="0.5, 1.5",
        eval_include_base=False,
    ).validate()

    loaded = TrainConfig.from_dict(config.to_dict())
    assert loaded.to_dict() == config.to_dict()
    assert loaded.sample_language == "JA"

    old = TrainConfig.from_dict({"dataset_dir": "dataset", "name": "old-adapter"})
    assert old.emo_ref_mode == "follow_speaker"
    assert old.val_reference_mode == "other"
    assert old.epoch_train_state is False
    assert old.sample_num_beams == 3
    assert old.sample_speaking_rate == 1.0
    assert old.eval_strengths == "1.0"


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("emo_ref_mode", "invalid"),
        ("val_reference_mode", "mixed"),
        ("sample_language", "FR"),
        ("sample_seed", -2),
        ("sample_temperature", 0),
        ("sample_top_p", 1.01),
        ("sample_top_k", -1),
        ("sample_repetition_penalty", 0),
        ("sample_num_beams", 0),
        ("sample_emo_alpha", 1.01),
        ("sample_diffusion_steps", 1),
        ("sample_inference_cfg_rate", -0.01),
        ("sample_max_text_tokens", 19),
        ("sample_length_penalty", float("nan")),
        ("sample_max_mel_tokens", 0),
        ("sample_speaking_rate", 0.49),
        ("sample_speaking_rate", 1.51),
        ("eval_train_subset", -1),
        ("eval_strengths", "1.0, 4.1"),
    ],
)
def test_invalid_new_config_values_raise(field_name: str, value: object) -> None:
    config = TrainConfig(dataset_dir="dataset", name="adapter")
    setattr(config, field_name, value)

    with pytest.raises((TypeError, ValueError)):
        config.validate()
