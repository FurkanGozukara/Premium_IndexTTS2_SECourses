from __future__ import annotations

import torch

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.lora.apply import inject_adapters, trainable_parameters
from indextts.training.model_forward import (
    build_gpt_training_inputs,
    enable_gradient_checkpointing,
    gpt_train_step_logits,
    gpt_train_step_loss,
)


def _tiny_voice(*, checkpointing: bool = False) -> UnifiedVoice:
    return UnifiedVoice(
        layers=2,
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
        checkpointing=checkpointing,
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


def _batch() -> dict[str, torch.Tensor]:
    return {
        "text_tokens": torch.tensor([[5, 6, 7, 1], [9, 10, 1, 1]]),
        "text_lengths": torch.tensor([3, 2]),
        "codes": torch.tensor([[11, 12, 13, 14, 65], [21, 22, 23, 65, 65]]),
        "code_lengths": torch.tensor([4, 3]),
        "lang_ids": torch.tensor([0, 3]),
        "campplus": torch.randn(2, 192),
        "emo_raw": torch.randn(2, 1024),
        "emo_vec": torch.randn(2, 32),
    }


def test_logits_shapes_and_masks() -> None:
    torch.manual_seed(12)
    model = _tiny_voice().eval()
    batch = _batch()
    text_logits, mel_logits, masks = gpt_train_step_logits(model, batch)
    assert text_logits.shape == (2, 6, 129)
    assert mel_logits.shape == (2, 7, 66)
    assert masks["attention_mask"].shape == (2, 16)
    assert masks["text_loss_mask"].sum(dim=1).tolist() == [4, 3]
    assert masks["mel_loss_mask"].sum(dim=1).tolist() == [5, 4]
    assert masks["attention_mask"][:, :3].all()
    # The short row keeps its first stop as a key and masks later right padding.
    assert masks["attention_mask"][1, 3:9].tolist() == [1, 1, 1, 1, 0, 0]


def test_single_sample_prompt_embeddings_match_inference() -> None:
    torch.manual_seed(13)
    model = _tiny_voice().eval()
    batch = {key: value[:1].clone() for key, value in _batch().items()}
    batch["text_tokens"] = batch["text_tokens"][:, :3]
    batch["codes"] = batch["codes"][:, :4]
    built = build_gpt_training_inputs(model, batch)

    raw_text = batch["text_tokens"]
    _, inference_prompt, _ = model.prepare_gpt_inputs(
        built["cond"], raw_text, batch["lang_ids"]
    )
    training_prompt = built["inputs_embeds"][:, : 3 + raw_text.shape[1] + 2]
    torch.testing.assert_close(training_prompt, inference_prompt, rtol=0, atol=1e-6)


def test_non_reentrant_checkpointing_produces_adapter_gradients() -> None:
    torch.manual_seed(14)
    model = _tiny_voice()
    model.requires_grad_(False)
    adapters = inject_adapters(
        model,
        rank=2,
        alpha=2,
        dropout=0.0,
        use_dora=True,
        target_modules=["gpt.h.0.attn.c_attn", "gpt.h.1.mlp.c_proj"],
    )
    parameters = trainable_parameters(model, adapters, {})
    enable_gradient_checkpointing(model, True)
    model.train()
    loss, metrics = gpt_train_step_loss(model, _batch())
    loss.backward()
    assert torch.isfinite(loss)
    assert 0 <= metrics["mel_accuracy"] <= 1
    assert any(parameter.grad is not None for parameter in parameters)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in parameters)
