from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from transformers.pytorch_utils import Conv1D

from indextts.quant.convrot_int8 import (
    COMFY_FORMAT,
    ConvRotInt8Linear,
    _build_hadamard,
    _int8_gemm_supported,
    _rotate_activation,
    _rotate_weight,
    comfy_quant_tensor,
    is_int8_convrot_checkpoint,
    load_gpt_checkpoint,
    quantize_best_convrot,
)


@pytest.mark.parametrize("size", [4, 16, 64, 256])
def test_hadamard_orthogonality_and_involution(size: int) -> None:
    h = _build_hadamard(size)
    identity = torch.eye(size)
    torch.testing.assert_close(h @ h.T, identity, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(h @ h, identity, atol=1.0e-6, rtol=1.0e-6)


def test_rotate_unrotate_round_trip() -> None:
    generator = torch.Generator().manual_seed(123)
    weight = torch.randn((19, 128), generator=generator)
    activation = torch.randn((2, 7, 128), generator=generator)
    h = _build_hadamard(64)
    restored_weight = _rotate_weight(_rotate_weight(weight, h, 64), h, 64)
    restored_activation = _rotate_activation(
        _rotate_activation(activation, h, 64), h, 64
    )
    torch.testing.assert_close(restored_weight, weight, atol=2.0e-5, rtol=2.0e-5)
    torch.testing.assert_close(
        restored_activation, activation, atol=2.0e-5, rtol=2.0e-5
    )


def test_quantize_dequantize_cosine() -> None:
    weight = torch.randn((48, 256), generator=torch.Generator().manual_seed(9))
    q, scale, group_size, metrics = quantize_best_convrot(
        weight, group_sizes=(256, 64, 16), device="cpu"
    )
    assert q.dtype == torch.int8
    assert scale.dtype == torch.float32
    assert group_size in (256, 64, 16)
    assert metrics["cosine_similarity"] > 0.99
    assert metrics["relative_weight_error_pct"] < 2.0


def _make_layer(weight: torch.Tensor, bias: torch.Tensor | None = None) -> ConvRotInt8Linear:
    q, scale, group_size, _ = quantize_best_convrot(
        weight, group_sizes=(64, 16), device="cpu"
    )
    layer = ConvRotInt8Linear(
        weight.shape[1], weight.shape[0], bias=bias is not None, group_size=group_size
    )
    layer.weight_int8.copy_(q)
    layer.weight_scale.copy_(scale)
    if bias is not None:
        layer.bias.data.copy_(bias)
    return layer


def test_cpu_fallback_matches_dequantized_linear() -> None:
    generator = torch.Generator().manual_seed(10)
    weight = torch.randn((24, 64), generator=generator)
    bias = torch.randn((24,), generator=generator)
    x = torch.randn((2, 5, 64), generator=generator)
    layer = _make_layer(weight, bias)
    layer.force_fallback = True
    actual = layer(x)
    expected = F.linear(x, layer.dequantize_weight(), layer.bias)
    torch.testing.assert_close(actual, expected, atol=2.0e-5, rtol=2.0e-5)


class _TinyConv1DModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c_attn = Conv1D(32, 16)
        self.c_proj = Conv1D(16, 32)
        self.norm = nn.LayerNorm(16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.c_proj(F.gelu(self.c_attn(x))))


def _tiny_checkpoint(model: _TinyConv1DModel, path: Path) -> None:
    tensors: dict[str, torch.Tensor] = {}
    layers: dict[str, dict[str, object]] = {}
    for base in ("c_attn", "c_proj"):
        module = getattr(model, base)
        linear_weight = module.weight.detach().T.contiguous()
        q, scale, group_size, _ = quantize_best_convrot(
            linear_weight, group_sizes=(16,), device="cpu"
        )
        tensors[f"{base}.weight"] = q
        tensors[f"{base}.weight_scale"] = scale
        tensors[f"{base}.comfy_quant"] = comfy_quant_tensor(group_size)
        tensors[f"{base}.bias"] = module.bias.detach().bfloat16()
        layers[base] = {
            "format": COMFY_FORMAT,
            "convrot": True,
            "convrot_groupsize": group_size,
        }
    tensors["norm.weight"] = model.norm.weight.detach().bfloat16()
    tensors["norm.bias"] = model.norm.bias.detach().bfloat16()
    metadata = {
        "_quantization_metadata": json.dumps(
            {"format_version": "1.0", "layers": layers}, separators=(",", ":")
        ),
        "indextts_variant": "int8_convrot",
    }
    save_file(tensors, str(path), metadata=metadata)


def test_save_load_round_trip_with_conv1d(tmp_path: Path) -> None:
    source = _TinyConv1DModel()
    checkpoint = tmp_path / "tiny.safetensors"
    _tiny_checkpoint(source, checkpoint)

    assert is_int8_convrot_checkpoint(checkpoint)
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        marker = handle.get_tensor("c_attn.comfy_quant")
        config = json.loads(bytes(marker.tolist()).decode("utf-8"))
        assert config == {
            "format": COMFY_FORMAT,
            "convrot": True,
            "convrot_groupsize": 16,
        }
        assert "_quantization_metadata" in (handle.metadata() or {})

    loaded = _TinyConv1DModel()
    report = load_gpt_checkpoint(loaded, str(checkpoint), device="cpu")
    assert report.quantized_layers == 2
    assert report.missing_keys == []
    assert report.unexpected_keys == []
    assert isinstance(loaded.c_attn, ConvRotInt8Linear)
    assert isinstance(loaded.c_proj, ConvRotInt8Linear)
    assert loaded.c_attn.weight_int8.dtype == torch.int8
    assert loaded.c_attn.weight_scale.dtype == torch.float32
    assert loaded.c_attn.bias.dtype == torch.bfloat16
    assert loaded.norm.weight.dtype == torch.bfloat16
    assert loaded.c_attn.nf == 32
    assert loaded.c_attn.weight.shape == (16, 32)


def test_apply_protects_quantized_dtypes() -> None:
    layer = ConvRotInt8Linear(64, 32, bias=True, group_size=64)
    int8_before = layer.weight_int8.clone()
    scale_before = layer.weight_scale.clone()
    rhs_before = layer._get_weight_int8_rhs().clone()
    layer.bfloat16()
    assert layer.weight_int8.dtype == torch.int8
    assert layer.weight_scale.dtype == torch.float32
    assert layer.weight_int8_rhs.dtype == torch.int8
    assert layer.bias.dtype == torch.bfloat16
    torch.testing.assert_close(layer.weight_int8, int8_before)
    torch.testing.assert_close(layer.weight_scale, scale_before)
    torch.testing.assert_close(layer.weight_int8_rhs, rhs_before)
    layer.half()
    assert layer.weight_int8.dtype == torch.int8
    assert layer.weight_scale.dtype == torch.float32
    assert layer.weight_int8_rhs.dtype == torch.int8
    assert layer.bias.dtype == torch.float16


@pytest.mark.parametrize("binding", ["attribute", "buffers"])
def test_rhs_cache_invalidates_when_weight_buffer_is_rebound(binding: str) -> None:
    layer = ConvRotInt8Linear(64, 24, bias=False, group_size=64)
    layer.weight_int8.copy_(
        torch.arange(layer.weight_int8.numel(), dtype=torch.int64)
        .remainder(255)
        .sub(127)
        .to(torch.int8)
        .reshape_as(layer.weight_int8)
    )
    first_rhs = layer._get_weight_int8_rhs()
    replacement = layer.weight_int8.flip(0).contiguous()
    if binding == "attribute":
        layer.weight_int8 = replacement
    else:
        layer._buffers["weight_int8"] = replacement

    second_rhs = layer._get_weight_int8_rhs()
    assert second_rhs.data_ptr() != first_rhs.data_ptr()
    torch.testing.assert_close(
        second_rhs[: layer.in_features, : layer.out_features], replacement.T
    )


def test_state_dict_round_trip_keeps_comfy_weight_layout() -> None:
    generator = torch.Generator().manual_seed(4321)
    source = ConvRotInt8Linear(64, 30, bias=True, group_size=64)
    source.weight_int8.random_(-127, 128, generator=generator)
    source.weight_scale.uniform_(0.0001, 0.01, generator=generator)
    source.bias.data.normal_(generator=generator)
    source._get_weight_int8_rhs()

    state = source.state_dict()
    assert state["weight_int8"].shape == (30, 64)
    assert state["weight_scale"].shape == (30, 1)
    assert "weight_int8_rhs" not in state

    restored = ConvRotInt8Linear(64, 30, bias=True, group_size=64)
    restored.weight_int8.zero_()
    restored._get_weight_int8_rhs()
    restored.load_state_dict(state)
    assert restored.weight_int8_rhs.numel() == 0
    torch.testing.assert_close(restored.weight_int8, source.weight_int8)
    torch.testing.assert_close(restored.weight_scale, source.weight_scale)
    torch.testing.assert_close(restored.bias, source.bias)
    restored_rhs = restored._get_weight_int8_rhs()
    torch.testing.assert_close(
        restored_rhs[: restored.in_features, : restored.out_features],
        source.weight_int8.T,
    )


def test_ste_input_gradient_matches_dequantized_reference() -> None:
    generator = torch.Generator().manual_seed(99)
    weight = torch.randn((20, 64), generator=generator)
    layer = _make_layer(weight)
    layer.training_ste = True
    x = torch.randn((3, 4, 64), generator=generator, requires_grad=True)
    output_gradient = torch.randn((3, 4, 20), generator=generator)
    (layer(x) * output_gradient).sum().backward()
    actual_gradient = x.grad.detach().clone()

    reference_x = x.detach().clone().requires_grad_(True)
    reference = F.linear(reference_x, layer.dequantize_weight())
    (reference * output_gradient).sum().backward()
    torch.testing.assert_close(
        actual_gradient, reference_x.grad, atol=3.0e-5, rtol=3.0e-5
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("rows", [1, 3, 17, 64, 257, 600])
def test_cuda_w8a16_and_w8a8_paths_match(rows: int) -> None:
    device = torch.device("cuda")
    if not _int8_gemm_supported(device):
        pytest.skip("torch._int_mm is unavailable")
    generator = torch.Generator(device=device).manual_seed(5000 + rows)
    layer = ConvRotInt8Linear(
        64,
        96,
        bias=True,
        group_size=64,
        device=device,
        dtype=torch.bfloat16,
    )
    layer.weight_int8.random_(-127, 128, generator=generator)
    layer.weight_scale.uniform_(0.0002, 0.002, generator=generator)
    layer.bias.data.normal_(generator=generator)
    x = torch.randn(
        (rows, 64), device=device, dtype=torch.bfloat16, generator=generator
    )

    with torch.inference_mode():
        layer.kernel_mode = "w8a16"
        w8a16 = layer(x)
        layer.kernel_mode = "w8a8"
        w8a8 = layer(x)
    torch.testing.assert_close(w8a8, w8a16, atol=0.04, rtol=0.04)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_block_swap_style_gpu_rebinding_rebuilds_rhs() -> None:
    device = torch.device("cuda")
    if not _int8_gemm_supported(device):
        pytest.skip("torch._int_mm is unavailable")
    generator = torch.Generator(device=device).manual_seed(7123)
    layer = ConvRotInt8Linear(
        64,
        96,
        bias=False,
        group_size=64,
        device=device,
        dtype=torch.bfloat16,
    )
    layer.kernel_mode = "w8a8"
    layer.weight_int8.random_(-127, 128, generator=generator)
    layer.weight_scale.fill_(0.001)
    x = torch.randn(
        (64, 64), device=device, dtype=torch.bfloat16, generator=generator
    )
    with torch.inference_mode():
        before = layer(x)
    old_rhs = layer.weight_int8_rhs

    layer._buffers["weight_int8"] = layer.weight_int8.clone()
    layer._buffers["weight_scale"] = layer.weight_scale.clone()
    with torch.inference_mode():
        after = layer(x)
    assert layer.weight_int8_rhs.data_ptr() != old_rhs.data_ptr()
    torch.testing.assert_close(after, before, atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("rows", [1, 3, 32, 257])
@pytest.mark.parametrize("in_features", [1280, 5120])
def test_cuda_fast_path_matches_fallback(rows: int, in_features: int) -> None:
    device = torch.device("cuda")
    if not _int8_gemm_supported(device):
        pytest.skip("torch._int_mm is unavailable")
    generator = torch.Generator(device=device).manual_seed(rows + in_features)
    layer = ConvRotInt8Linear(
        in_features,
        128,
        bias=True,
        group_size=256,
        device=device,
        dtype=torch.bfloat16,
    )
    layer.weight_int8.random_(-127, 128, generator=generator)
    layer.weight_scale.fill_(0.0005)
    layer.bias.data.normal_(generator=generator)
    x = torch.randn(
        (rows, in_features), device=device, dtype=torch.bfloat16, generator=generator
    )

    torch.cuda.synchronize()
    started = time.perf_counter()
    fast = layer(x)
    torch.cuda.synchronize()
    fast_seconds = time.perf_counter() - started
    layer.force_fallback = True
    fallback = layer(x)
    torch.testing.assert_close(fast, fallback, atol=0.20, rtol=0.10)
    assert math.isfinite(fast_seconds)
    assert fast_seconds < 30.0


def run_full_model_comparison(
    checkpoint: Path = Path("models/gpt_int8_convrot.safetensors"),
) -> dict[str, float]:
    """Run the bf16-versus-INT8 GPT logit comparison used by the GPU test."""

    from omegaconf import OmegaConf

    from indextts.gpt.model_v2 import UnifiedVoice

    device = torch.device("cuda")
    torch.manual_seed(2026)
    config = OmegaConf.load("models/config.yaml")
    kwargs = dict(config.gpt)

    reference_model = UnifiedVoice(
        **kwargs, use_accel=False, spk_cond_mode="campplus"
    )
    load_gpt_checkpoint(
        reference_model, "models/gpt.pth", device=device, dtype=torch.bfloat16
    )
    reference_model.eval()
    # model_v2.py is owned by the runtime task. Its current null-position helper
    # creates fp32 zeros, so keep this isolated comparison wholly in bf16.
    reference_model.gpt.wpe = lambda positions: torch.zeros(
        (*positions.shape, reference_model.model_dim),
        device=positions.device,
        dtype=torch.bfloat16,
    )
    token_ids = torch.randint(0, 8192, (1, 18), device=device)
    first_input = reference_model.mel_embedding(token_ids).detach()
    conditioning = (
        torch.randn((1, 4, reference_model.model_dim), device=device) * 0.02
    ).to(torch.bfloat16)
    with torch.inference_mode():
        reference_logits = reference_model.get_logits(
            conditioning, first_input, reference_model.mel_head
        ).float()
    del reference_model
    torch.cuda.empty_cache()

    quantized_model = UnifiedVoice(
        **kwargs, use_accel=False, spk_cond_mode="campplus"
    )
    report = load_gpt_checkpoint(
        quantized_model, str(checkpoint), device=device, dtype=torch.bfloat16
    )
    assert report.quantized_layers == 96
    quantized_model.eval()
    quantized_model.gpt.wpe = lambda positions: torch.zeros(
        (*positions.shape, quantized_model.model_dim),
        device=positions.device,
        dtype=torch.bfloat16,
    )
    with torch.inference_mode():
        quantized_logits = quantized_model.get_logits(
            conditioning, first_input, quantized_model.mel_head
        ).float()

    cosine = F.cosine_similarity(
        reference_logits.flatten(), quantized_logits.flatten(), dim=0
    ).item()
    agreement = (
        reference_logits.argmax(dim=1) == quantized_logits.argmax(dim=1)
    ).float().mean().item()
    return {"logits_cosine_similarity": cosine, "top1_agreement_rate": agreement}


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_full_gpt_logits_quality() -> None:
    checkpoint = Path("models/gpt_int8_convrot.safetensors")
    if not checkpoint.is_file():
        pytest.skip("converted checkpoint has not been generated")
    if torch.cuda.get_device_properties(0).total_memory < 12 * 1024**3:
        pytest.skip("full GPT comparison requires at least 12 GiB VRAM")
    metrics = run_full_model_comparison(checkpoint)
    print(json.dumps(metrics, indent=2))
    assert metrics["logits_cosine_similarity"] > 0.99


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the full model comparison")
    print(json.dumps(run_full_model_comparison(), indent=2))
