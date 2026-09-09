"""Deployment-score selection, checkpoint averaging, GPT-code decoder training, and the clip-length mix."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from indextts.lora.io import LoraMetadata, load_lora
from indextts.training.analysis import checkpoint_descriptor, discover_checkpoints
from indextts.training.checkpoint_average import (average_lora_files, averaged_epoch_span, averaged_members_of,
                                                   choose_average_members, write_averaged_checkpoint)
from indextts.training.decoder_adapter import DecoderAdapterConfig, _sample_codes
from indextts.training.speech_metrics import deployment_score, select_recommendation
from indextts.training.train_config import TrainConfig


def _measured(label, errors=0.1, speaker=0.8, pause=1.0, failure=False):
    return [{"checkpoint": label, "prompt_id": f"p{p}", "seed": seed, "errors": errors * 20, "units": 20,
             "error_rate": errors, "speaker_similarity": speaker, "speaker_similarity_real": speaker,
             "pause_s": pause, "real_pause_s": 1.0, "invalid_audio": False, "possible_truncation": failure,
             "possible_repetition": False, "start_matches": True, "end_matches": not failure}
            for p in range(4) for seed in [42, 104771, 209500]]


POLICY = {"max_wer_increase": .02, "max_speaker_drop": .03}


def test_deployment_score_weighs_identity_word_errors_and_pauses():
    base = {"pause_ratio_vs_real": 1.5}
    score = deployment_score({"pause_ratio_vs_real": 1.0}, base, 0.03, 0.005)
    assert score["speaker_gain"] == pytest.approx(0.03)
    assert score["wer_penalty"] == pytest.approx(0.02)  # half a point of word error costs 0.02
    assert score["pause_term"] == pytest.approx(0.05 * abs(__import__("math").log(1.5)))
    assert score["score"] == pytest.approx(0.03 - 0.02 + score["pause_term"])
    # A word-error decrease is not rewarded, and missing pause data contributes nothing.
    assert deployment_score({}, {}, 0.01, -0.02)["score"] == pytest.approx(0.01)


def test_selection_prefers_the_deployment_score_and_breaks_ties_by_loss():
    candidates = [{"label": "Base", "path": "", "val_loss": 6},
                  {"label": "identity", "path": "a.safetensors", "val_loss": 4.1},
                  {"label": "words", "path": "b.safetensors", "val_loss": 4.0}]
    # "identity" gains 0.03 similarity for 0.4 points of word error (score +0.014); "words" only
    # lowers word error, which the score does not reward, so its score is the pause term alone.
    rows = (_measured("Base", errors=.05, speaker=.80, pause=1.4) + _measured("identity", errors=.054, speaker=.83, pause=1.4)
            + _measured("words", errors=.02, speaker=.80, pause=1.4))
    report = select_recommendation(candidates, rows, POLICY)
    assert report["recommended_label"] == "identity"
    assert report["candidates"][1]["deployment_score"]["score"] == pytest.approx(0.03 - 4 * 0.004)
    assert "deployment score" in report["decision"]
    # Equal scores fall back to the lower validation loss, Base included.
    tie = select_recommendation(candidates, _measured("Base") + _measured("identity") + _measured("words"), POLICY)
    assert tie["recommended_label"] == "words"
    # The Base guards still reject a candidate before any score is considered.
    bad = select_recommendation(candidates[:2], _measured("Base") + _measured("identity", errors=.2, speaker=.9), POLICY)
    assert bad["recommended_kind"] == "base"


def _lora_file(path: Path, scale: float, *, steps: int, epochs: int) -> Path:
    tensors = {"gpt.h.0.attn.c_attn.lora_A.weight": torch.full((2, 4), scale), "gpt.h.0.attn.c_attn.lora_B.weight": torch.full((4, 2), 2 * scale),
               "gpt.h.0.attn.c_attn.lora_magnitude": torch.full((4,), 3 * scale), "full.spk_emb_proj.weight": torch.full((3, 2), scale)}
    metadata = LoraMetadata(adapter_type="dora", rank=2, alpha=2.0, target_modules=["gpt.h.0.attn.c_attn"], trained_steps=steps,
                            epochs=epochs, dataset_name="voice", train_config={"name": "voice"})
    save_file({k: v.to(torch.bfloat16) for k, v in tensors.items()}, str(path), metadata=metadata.to_header())
    return path


def test_checkpoint_averaging_means_every_tensor_and_records_its_members(tmp_path):
    run = tmp_path / "voice"
    run.mkdir()
    first = _lora_file(run / "voice_epoch_005.safetensors", 1.0, steps=500, epochs=5)
    second = _lora_file(run / "voice_epoch_006.safetensors", 3.0, steps=600, epochs=6)
    final = _lora_file(run / "voice.safetensors", 5.0, steps=650, epochs=7)
    output = average_lora_files([first, second, final], run / "voice_avg_ep5_7.safetensors")
    loaded = load_lora(output)
    assert torch.allclose(loaded.tensors["gpt.h.0.attn.c_attn.lora_A.weight"].float(), torch.full((2, 4), 3.0))
    assert torch.allclose(loaded.tensors["gpt.h.0.attn.c_attn.lora_magnitude"].float(), torch.full((4,), 9.0))
    assert torch.allclose(loaded.tensors["full.spk_emb_proj.weight"].float(), torch.full((3, 2), 3.0))
    assert loaded.metadata.trained_steps == 650 and loaded.metadata.epochs == 7
    assert averaged_members_of(output) == ["voice_epoch_005.safetensors", "voice_epoch_006.safetensors", "voice.safetensors"]
    descriptor = checkpoint_descriptor(output)
    assert descriptor["kind"] == "averaged" and descriptor["file_label"] == "avg_ep5_7" and descriptor["epoch"] == 7
    assert averaged_epoch_span(output) == (5, 7)
    kinds = {Path(item["path"]).name: item["kind"] for item in discover_checkpoints(run)}
    assert kinds["voice_avg_ep5_7.safetensors"] == "averaged" and kinds["voice.safetensors"] == "final"
    with pytest.raises(ValueError, match="distinct"):
        average_lora_files([first, first], run / "bad.safetensors")


def test_trainer_averages_the_last_distinct_updates_and_skips_duplicates(tmp_path):
    run = tmp_path / "voice"
    (run / "best").mkdir(parents=True)
    _lora_file(run / "voice_epoch_004.safetensors", 1.0, steps=400, epochs=4)
    _lora_file(run / "voice_epoch_005.safetensors", 2.0, steps=500, epochs=5)
    _lora_file(run / "voice_epoch_006.safetensors", 3.0, steps=600, epochs=6)
    _lora_file(run / "voice.safetensors", 3.0, steps=600, epochs=6)  # the final file is the same update as epoch 6
    _lora_file(run / "best" / "voice_best.safetensors", 2.0, steps=500, epochs=5)
    members = choose_average_members(discover_checkpoints(run), 3)
    assert [int(item["steps"]) for item in members] == [400, 500, 600]
    path, message = write_averaged_checkpoint(run, "voice", 3)
    assert path is not None and path.name == "voice_avg_ep4_6.safetensors" and "3 saved updates" in message
    assert torch.allclose(load_lora(path).tensors["gpt.h.0.attn.c_attn.lora_A.weight"].float(), torch.full((2, 4), 2.0))
    # A second averaged file is never a member of a later average.
    members = choose_average_members(discover_checkpoints(run), 5)
    assert all(item["kind"] != "averaged" for item in members)
    assert write_averaged_checkpoint(tmp_path / "empty", "voice", 3)[0] is None
    assert TrainConfig.from_dict({"dataset_dir": "d", "name": "n"}).average_last_checkpoints == 0
    assert TrainConfig.from_dict({"dataset_dir": "d", "name": "n", "average_last_checkpoints": 3}).average_last_checkpoints == 3
    assert TrainConfig.from_dict({"dataset_dir": "d", "name": "n", "average_last_checkpoints": -2}).average_last_checkpoints == 0


def test_decoder_code_source_is_validated_and_sampling_respects_the_limits():
    config = DecoderAdapterConfig.from_dict({"dataset_dir": "d", "output_path": "voice.s2mel.safetensors", "code_source": "gpt",
                                             "gpt_checkpoint": "voice.safetensors"})
    assert config.code_source == "gpt" and config.gpt_code_variants == 3
    with pytest.raises(ValueError, match="gpt_checkpoint"):
        DecoderAdapterConfig.from_dict({"dataset_dir": "d", "output_path": "voice.s2mel.safetensors", "code_source": "mixed"})
    with pytest.raises(ValueError, match="code_source"):
        DecoderAdapterConfig.from_dict({"dataset_dir": "d", "output_path": "voice.s2mel.safetensors", "code_source": "latents"})
    with pytest.raises(ValueError, match="decoder_adapter_code_source"):
        TrainConfig.from_dict({"dataset_dir": "d", "name": "n", "decoder_adapter_code_source": "other"})
    logits = torch.zeros(5, 50)
    logits[:, 7] = 10.0  # one dominant code per position
    logits[:, 8] = 9.0
    generator = torch.Generator().manual_seed(1)
    sampled = _sample_codes(logits, generator, temperature=0.8, top_k=2, top_p=0.8)
    assert sampled.shape == (5,) and set(sampled.tolist()) <= {7, 8}
    greedy = _sample_codes(logits, generator, temperature=0.01, top_k=1, top_p=1.0)
    assert greedy.tolist() == [7] * 5


def test_dataset_prep_records_the_clip_length_mix(tmp_path):
    from indextts.training.dataset_prep import DatasetPrepConfig
    config = DatasetPrepConfig(name="mix", inputs=["x"], short_clip_fraction=0.15, medium_clip_fraction=0.15)
    config.validate()
    assert config.to_dict()["medium_clip_fraction"] == 0.15
    assert json.loads(json.dumps(config.to_dict()))["short_clip_fraction"] == 0.15
