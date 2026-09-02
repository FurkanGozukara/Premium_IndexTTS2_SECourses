from __future__ import annotations

from pathlib import Path

import pytest
import torch

import indextts.training.checkpoint_eval as eval_module
from indextts.gpt.model_v2 import UnifiedVoice
from indextts.lora.apply import inject_adapters
from indextts.lora.io import LoraMetadata, save_lora
from indextts.training.checkpoint_eval import (
    CheckpointEvalConfig,
    evaluate_checkpoints,
    load_checkpoint_eval,
    write_checkpoint_eval,
)


def _tiny_voice() -> UnifiedVoice:
    return UnifiedVoice(
        layers=1,
        model_dim=32,
        heads=4,
        max_text_tokens=24,
        max_mel_tokens=32,
        number_text_tokens=128,
        number_mel_codes=66,
        start_text_token=0,
        stop_text_token=1,
        start_mel_token=64,
        stop_mel_token=65,
        spk_cond_mode="campplus",
        emo_condition_module={
            "output_size": 16,
            "linear_units": 32,
            "attention_heads": 4,
            "num_blocks": 1,
            "input_layer": "conv2d2",
            "perceiver_mult": 2,
        },
        attention_backend="eager",
    )


class _FakeDataset(torch.utils.data.Dataset):
    def __init__(self, *_args, split="train", **_kwargs):
        self.split = split

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            "id": f"{self.split}-{index}",
            "reference_id": f"{self.split}-{index}",
            "text_tokens": torch.tensor([5 + index, 6 + index]),
            "codes": torch.tensor([11 + index, 12 + index, 13 + index]),
            "lang_id": 0,
            "campplus": torch.randn(192),
            "emo_raw": torch.randn(1024),
            "emo_vec": torch.randn(32),
            "speaker": "speaker",
        }


def test_cpu_checkpoint_evaluation_and_report_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(33)
    adapter_dir = tmp_path / "voice"
    adapter_dir.mkdir()
    checkpoint = adapter_dir / "voice_epoch_002.safetensors"
    source_model = _tiny_voice()
    adapters = inject_adapters(
        source_model,
        rank=2,
        alpha=2,
        dropout=0,
        use_dora=False,
        target_modules=["gpt.h.0.attn.c_proj"],
    )
    adapters["gpt.h.0.attn.c_proj"].lora_B.weight.data.normal_(std=0.02)
    save_lora(
        checkpoint,
        adapters,
        {},
        LoraMetadata(
            rank=2,
            alpha=2,
            trained_steps=2,
            epochs=2,
            train_config={"dataset_dir": str(tmp_path / "dataset"), "val_fraction": 0.5, "seed": 7},
        ),
        dtype=torch.float32,
    )
    monkeypatch.setattr(eval_module, "LoraTrainDataset", _FakeDataset)
    monkeypatch.setattr(eval_module, "build_evaluation_model", lambda _config: _tiny_voice())
    config = CheckpointEvalConfig(
        adapter_dir=str(adapter_dir),
        dataset_dir=str(tmp_path / "dataset"),
        checkpoints=[str(checkpoint)],
        strengths=[1.0, 0.5],
        train_subset=1,
        batch_size=2,
        device="cpu",
        base_variant="bf16",
        base_dtype="fp32",
        model_dir=str(tmp_path),
        model_config=str(tmp_path / "config.yaml"),
        attention_backend="eager",
        val_fraction=0.5,
        seed=7,
    )

    report = evaluate_checkpoints(config)

    assert [row.phase for row in report.rows] == ["base", "best", "variant"]
    assert report.rows[0].kind == "base"
    assert report.rows[1].val_loss is not None
    assert report.rows[1].train_loss is not None
    assert "Without any adapter" in report.summary_markdown
    path = write_checkpoint_eval(report, adapter_dir)
    assert path.is_file()
    assert path.with_suffix(".md").is_file()
    loaded = load_checkpoint_eval(adapter_dir)
    assert loaded is not None and len(loaded.rows) == 3

