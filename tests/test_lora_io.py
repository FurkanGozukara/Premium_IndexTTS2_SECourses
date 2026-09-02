from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

from indextts.lora.io import (
    LoraMetadata,
    inspect_lora,
    load_lora,
    load_train_state,
    resume_state_path_for,
    save_lora,
    save_train_state,
    scan_lora_files,
)
from indextts.lora.layers import LoRAAdapter


def _adapter(use_dora: bool = True) -> LoRAAdapter:
    result = LoRAAdapter(nn.Linear(5, 7), rank=3, alpha=6, use_dora=use_dora)
    result.lora_A.weight.data.normal_()
    result.lora_B.weight.data.normal_()
    return result


def test_save_load_round_trip_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "speaker.safetensors"
    adapter = _adapter()
    full = nn.Linear(4, 5)
    metadata = LoraMetadata(
        adapter_type="dora",
        rank=3,
        alpha=6,
        dropout=0.1,
        target_modules=["gpt.h.0.attn.c_attn"],
        trained_steps=120,
        epochs=2,
        dataset_name="voice-set",
        created_at="2026-09-02T10:00:00+03:00",
        train_config={"lr": 1e-4},
    )
    save_lora(
        path,
        {"gpt.h.0.attn.c_attn": adapter},
        {"spk_emb_proj": full},
        metadata,
        dtype=torch.float32,
    )

    loaded = load_lora(path)
    assert loaded.adapter_type == "dora"
    assert loaded.rank == 3
    assert loaded.alpha == 6
    assert loaded.has_full
    assert loaded.metadata.train_config == {"lr": 1e-4}
    torch.testing.assert_close(
        loaded.tensors["gpt.h.0.attn.c_attn.lora_A.weight"],
        adapter.lora_A.weight,
    )
    torch.testing.assert_close(loaded.tensors["full.spk_emb_proj.weight"], full.weight)


def test_shape_fallback_and_peft_key_normalization(tmp_path: Path) -> None:
    path = tmp_path / "peft.safetensors"
    tensors = {
        "base_model.model.gpt.h.0.attn.c_proj.lora_A.default.weight": torch.randn(2, 5),
        "base_model.model.gpt.h.0.attn.c_proj.lora_B.default.weight": torch.randn(7, 2),
        "base_model.model.gpt.h.0.attn.c_proj.lora_magnitude_vector.weight": torch.randn(7),
    }
    save_file(tensors, str(path), metadata={})
    loaded = load_lora(path)
    assert loaded.rank == 2
    assert loaded.alpha == 2
    assert loaded.adapter_type == "dora"
    assert loaded.module_paths == ["gpt.h.0.attn.c_proj"]
    assert "gpt.h.0.attn.c_proj.lora_magnitude" in loaded.tensors


def test_inspect_scan_reference_and_train_state(tmp_path: Path) -> None:
    root = tmp_path / "loras"
    nested = root / "voices"
    nested.mkdir(parents=True)
    path = nested / "alice.safetensors"
    reference = nested / "alice_reference.wav"
    reference.write_bytes(b"RIFF")
    save_lora(
        path,
        {"gpt.h.0.mlp.c_fc": _adapter(use_dora=False)},
        {},
        LoraMetadata(
            adapter_type="lora",
            rank=3,
            alpha=6,
            trained_steps=20,
            dataset_name="alice",
        ),
    )
    save_file({"not_an_adapter": torch.ones(1)}, str(root / "skip.safetensors"))

    info = inspect_lora(path)
    assert info["adapter_type"] == "lora"
    assert info["recommended_reference"] == str(reference)
    entries = scan_lora_files([str(root)])
    assert [entry.name for entry in entries] == ["alice"]
    assert "LORA" in entries[0].metadata_summary

    state_path = resume_state_path_for(path)
    assert state_path.endswith("alice.train_state.pt")
    save_train_state(state_path, {"step": 20, "optimizer": {"lr": 1e-4}})
    assert load_train_state(state_path)["step"] == 20
