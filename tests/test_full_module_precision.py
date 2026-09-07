from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

import indextts.training.trainer as trainer_module
from indextts.lora import LoraMetadata, apply_lora, save_lora
from indextts.training.model_forward import gpt_train_step_loss
from indextts.training.train_config import TrainConfig


@pytest.fixture
def build_tiny_training_model(tmp_path: Path, monkeypatch):
    """Exercise the builder with CUDA-like base storage, without using a GPU."""
    model_config = tmp_path / "config.yaml"
    OmegaConf.save(
        {
            "gpt_checkpoint": "gpt.pth",
            "gpt": {
                "layers": 1,
                "model_dim": 32,
                "heads": 4,
                "max_text_tokens": 24,
                "max_mel_tokens": 32,
                "number_text_tokens": 128,
                "number_mel_codes": 66,
                "start_text_token": 0,
                "stop_text_token": 1,
                "start_mel_token": 64,
                "stop_mel_token": 65,
                "emo_condition_module": {
                    "output_size": 16,
                    "linear_units": 32,
                    "attention_heads": 4,
                    "num_blocks": 1,
                    "input_layer": "conv2d2",
                    "perceiver_mult": 2,
                },
            },
        },
        model_config,
    )
    (tmp_path / "gpt.pth").touch()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)

    def build(*, fp32: bool = True, base_dtype: str = "bf16", **kwargs):
        def load_checkpoint(model, *_args, **_kwargs):
            # CPU training normally promotes the whole base to FP32. Deliberately
            # retain the requested CUDA storage here to cover the actual bug.
            model.to(dtype=trainer_module._dtype(base_dtype))
            return SimpleNamespace(seconds=0.0, quantized_layers=0, missing_keys=[], unexpected_keys=[])

        monkeypatch.setattr(trainer_module, "load_gpt_checkpoint", load_checkpoint)
        config = TrainConfig(
            dataset_dir=str(tmp_path / "dataset"),
            name="tiny",
            model_dir=str(tmp_path),
            model_config=str(model_config),
            device="cpu",
            base_dtype=base_dtype,
            train_full_modules_fp32=fp32,
            train_emo_layers=True,
            train_mel_embed_head=True,
            rank=2,
            alpha=2,
            dropout=0,
            attention_backend="eager",
            **kwargs,
        )
        return trainer_module.build_training_model(config), config

    yield build
    torch.set_num_threads(threads)


def _batch() -> dict[str, torch.Tensor]:
    return {
        "text_tokens": torch.tensor([[5, 6, 7]]),
        "text_lengths": torch.tensor([3]),
        "codes": torch.tensor([[11, 12, 13, 14]]),
        "code_lengths": torch.tensor([4]),
        "lang_ids": torch.tensor([0]),
        "campplus": torch.randn(1, 192, dtype=torch.bfloat16),
        "emo_raw": torch.randn(1, 1024, dtype=torch.bfloat16),
        "emo_vec": torch.randn(1, 32, dtype=torch.bfloat16),
    }


@pytest.mark.parametrize("fp32", [False, True])
@pytest.mark.parametrize("base_dtype", ["bf16", "fp16", "fp32"])
def test_selected_full_modules_keep_requested_storage(build_tiny_training_model, fp32, base_dtype):
    built, _ = build_tiny_training_model(fp32=fp32, base_dtype=base_dtype)
    expected = torch.float32 if fp32 else trainer_module._dtype(base_dtype)
    assert all(parameter.dtype == expected for module in built.full_modules.values() for parameter in module.parameters())
    assert built.model.gpt.h[0].ln_1.weight.dtype == trainer_module._dtype(base_dtype)
    assert all(adapter.lora_A.weight.dtype == torch.float32 for adapter in built.adapters.values())
    assert all(adapter.lora_B.weight.dtype == torch.float32 for adapter in built.adapters.values())


@pytest.mark.parametrize("fp32", [False, True])
def test_fp32_full_module_retains_small_adamw_updates(build_tiny_training_model, fp32):
    built, config = build_tiny_training_model(fp32=fp32)
    parameter = built.model.spk_emb_proj.weight
    with torch.no_grad():
        parameter.fill_(0.05)
    before = parameter.detach().float().clone()
    optimizer = trainer_module._optimizer(config, built.parameters)
    for _ in range(100):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
    difference = (before - parameter.detach().float()).abs().max().item()
    if fp32:
        assert difference > 0.003
        assert optimizer.state[parameter]["exp_avg"].dtype == torch.float32
        assert optimizer.state[parameter]["exp_avg_sq"].dtype == torch.float32
    else:
        assert difference == 0


@pytest.mark.parametrize("fp32", [False, True])
@pytest.mark.parametrize(
    ("base_dtype", "mixed_precision"),
    [
        ("bf16", "bf16"), ("bf16", "fp32"),
        ("fp16", "fp16"), ("fp16", "fp32"),
        ("fp32", "bf16"), ("fp32", "fp16"), ("fp32", "fp32"),
    ],
)
def test_full_module_storage_works_with_forward_precision(
    build_tiny_training_model, fp32, mixed_precision, base_dtype
):
    # CPU autocast does not support a BF16 base under FP16 autocast, or
    # vice versa. Those combinations need CUDA verification separately.
    torch.manual_seed(42)
    built, _ = build_tiny_training_model(fp32=fp32, base_dtype=base_dtype)
    with torch.autocast("cpu", dtype=trainer_module._dtype(mixed_precision), enabled=mixed_precision != "fp32"):
        loss, _ = gpt_train_step_loss(built.model, _batch())
    loss.backward()
    assert torch.isfinite(loss)
    for module in built.full_modules.values():
        gradients = [parameter.grad for parameter in module.parameters()]
        assert any(gradient is not None and gradient.abs().sum() > 0 for gradient in gradients)
        assert all(gradient is None or torch.isfinite(gradient).all() for gradient in gradients)


def test_fp32_weights_survive_resume_and_load_for_base_precision_inference(build_tiny_training_model, tmp_path):
    source, _ = build_tiny_training_model()
    with torch.no_grad():
        source.model.spk_emb_proj.weight.fill_(0.050071231)
    checkpoint = tmp_path / "voice.safetensors"
    save_lora(
        checkpoint,
        source.adapters,
        source.full_modules,
        LoraMetadata(adapter_type="dora", rank=2, alpha=2, target_modules=list(source.adapters)),
        dtype=torch.float32,
    )
    resumed, _ = build_tiny_training_model(resume_from=str(checkpoint), resume_mode="continue")
    torch.testing.assert_close(resumed.model.spk_emb_proj.weight, source.model.spk_emb_proj.weight, rtol=0, atol=0)

    inference, _ = build_tiny_training_model(fp32=False)
    apply_lora(inference.model, str(checkpoint), strength=1.0)
    assert inference.model.spk_emb_proj.weight.dtype == torch.bfloat16
    torch.testing.assert_close(inference.model.spk_emb_proj.weight, source.model.spk_emb_proj.weight.to(torch.bfloat16), rtol=0, atol=0)


def test_resumed_bf16_optimizer_moments_follow_fp32_full_module(build_tiny_training_model):
    old, config = build_tiny_training_model(fp32=False)
    old_optimizer = trainer_module._optimizer(config, old.parameters)
    parameter = old.model.spk_emb_proj.weight
    parameter.grad = torch.ones_like(parameter)
    old_optimizer.step()
    old_scheduler = torch.optim.lr_scheduler.LambdaLR(old_optimizer, lambda _step: 1.0)
    state = {"optimizer": old_optimizer.state_dict(), "scheduler": old_scheduler.state_dict()}

    resumed, config = build_tiny_training_model()
    optimizer = trainer_module._optimizer(config, resumed.parameters)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    trainer = trainer_module.LoraTrainer.__new__(trainer_module.LoraTrainer)
    trainer.log = lambda _message: None
    trainer._restore_resume_state(state, optimizer, scheduler, torch.amp.GradScaler("cpu", enabled=False))
    parameter = resumed.model.spk_emb_proj.weight
    assert optimizer.state[parameter]["exp_avg"].dtype == torch.float32
    assert optimizer.state[parameter]["exp_avg_sq"].dtype == torch.float32
    parameter.grad = torch.ones_like(parameter)
    before = parameter.detach().clone()
    optimizer.step()
    assert not torch.equal(parameter, before)
