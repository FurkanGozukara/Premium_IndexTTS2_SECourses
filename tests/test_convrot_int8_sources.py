"""Sharded/directory sources and F32-kept tensors in the ConvRot INT8 converter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from indextts.quant.convrot_int8 import ConvRotInt8Linear, convert_gpt_checkpoint, load_gpt_checkpoint


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(64, 32)
        self.out = nn.Linear(32, 16, bias=False)
        self.norm = nn.RMSNorm(32)
        self.layer_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.norm(self.proj(x))) * self.layer_scale


def _source_state(prefix: str = "") -> dict[str, torch.Tensor]:
    torch.manual_seed(5)
    model = _Tiny()
    with torch.no_grad():
        model.norm.weight.uniform_(0.5, 1.5)
        model.layer_scale.fill_(1.2345678)
    state = {f"{prefix}{key}": value.detach().clone().contiguous() for key, value in model.state_dict().items()}
    if prefix:
        state["other.tower.weight"] = torch.randn(8, 8)
    return state


def _dtypes(path: Path) -> dict[str, str]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return {key: handle.get_slice(key).get_dtype() for key in handle.keys()}


def _tensors(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


def _convert(source, destination: Path, **kwargs) -> dict:
    return convert_gpt_checkpoint(source, str(destination), device="cpu", progress=None,
                                  group_sizes=(16,), linear_targets=["proj", "out"], **kwargs)


def test_default_writes_every_unquantized_tensor_as_bf16(tmp_path: Path) -> None:
    source = tmp_path / "model.safetensors"
    save_file(_source_state(), str(source))
    report = _convert(source, tmp_path / "int8.safetensors")
    dtypes = _dtypes(tmp_path / "int8.safetensors")
    assert dtypes["proj.weight"] == "I8" and dtypes["out.weight"] == "I8"
    assert dtypes["proj.weight_scale"] == "F32"
    assert dtypes["norm.weight"] == "BF16" and dtypes["layer_scale"] == "BF16" and dtypes["proj.bias"] == "BF16"
    assert "keep_float32" not in report and "source_files" not in report


def test_keep_float32_keys_and_suffixes(tmp_path: Path) -> None:
    source = tmp_path / "model.safetensors"
    state = _source_state()
    save_file(state, str(source))
    destination = tmp_path / "int8.safetensors"
    report = _convert(source, destination, keep_float32=("layer_scale", "norm.weight"))
    dtypes = _dtypes(destination)
    assert dtypes["norm.weight"] == "F32" and dtypes["layer_scale"] == "F32"
    assert dtypes["proj.bias"] == "BF16"
    assert report["f32_tensors"] == 2 and report["bf16_tensors"] == 1
    written = _tensors(destination)
    assert torch.equal(written["layer_scale"], state["layer_scale"])
    assert torch.equal(written["norm.weight"], state["norm.weight"])


def test_keep_float32_must_not_name_a_quantized_weight(tmp_path: Path) -> None:
    source = tmp_path / "model.safetensors"
    save_file(_source_state(), str(source))
    with pytest.raises(ValueError, match="keep_float32"):
        _convert(source, tmp_path / "int8.safetensors", keep_float32=("proj.weight",))


def _write_shards(folder: Path, state: dict[str, torch.Tensor]) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    keys = sorted(state)
    shards = [keys[::2], keys[1::2]]
    files, weight_map = [], {}
    for index, names in enumerate(shards, 1):
        path = folder / f"model-{index:05d}-of-00002.safetensors"
        save_file({key: state[key] for key in names}, str(path))
        files.append(path)
        weight_map.update({key: path.name for key in names})
    (folder / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    return files


def test_sharded_directory_and_file_list_match_a_single_file(tmp_path: Path) -> None:
    prefix = "thinker.model."
    state = _source_state(prefix)
    single = tmp_path / "single.safetensors"
    save_file(state, str(single))
    files = _write_shards(tmp_path / "shards", state)
    # A stray file the index does not name must be ignored for a directory source.
    save_file({"thinker.model.proj.weight": torch.zeros(32, 64)}, str(tmp_path / "shards" / "stray.safetensors"))

    outputs = {}
    for name, source in (("single", single), ("folder", tmp_path / "shards"), ("list", files)):
        destination = tmp_path / f"{name}.safetensors"
        report = _convert(source, destination, state_prefix=prefix, keep_float32=("layer_scale",))
        outputs[name] = _tensors(destination)
        if name != "single":
            assert report["source_files"] == [str(path.resolve()) for path in files]
        assert report["selected_source_bytes"] == sum(
            value.numel() * value.element_size() for key, value in state.items() if key.startswith(prefix))
    assert set(outputs["single"]) == set(outputs["folder"]) == set(outputs["list"])
    assert not any(key.startswith("other.") for key in outputs["folder"])
    for key, value in outputs["single"].items():
        assert torch.equal(value, outputs["folder"][key]), key
        assert torch.equal(value, outputs["list"][key]), key


def test_duplicate_tensor_across_shards_is_rejected(tmp_path: Path) -> None:
    state = _source_state()
    first, second = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    save_file(state, str(first))
    save_file({"norm.weight": state["norm.weight"]}, str(second))
    with pytest.raises(ValueError, match="more than one source file"):
        _convert([first, second], tmp_path / "int8.safetensors")


def test_sharded_conversion_loads_and_matches_the_float_model(tmp_path: Path) -> None:
    state = _source_state()
    files = _write_shards(tmp_path / "shards", state)
    destination = tmp_path / "int8.safetensors"
    _convert(files, destination, keep_float32=("layer_scale", "norm.weight"))
    reference = _Tiny()
    reference.load_state_dict(state)
    loaded = _Tiny()
    report = load_gpt_checkpoint(loaded, str(destination), device="cpu", dtype=torch.float32, strict=True)
    assert report.quantized_layers == 2
    assert isinstance(loaded.proj, ConvRotInt8Linear) and isinstance(loaded.out, ConvRotInt8Linear)
    x = torch.randn(4, 64, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        expected, actual = reference(x), loaded(x)
    relative = float((actual - expected).norm() / expected.norm())
    assert relative < 0.03


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("rows", [17, 33, 257, 1994])
def test_cuda_w8a8_accepts_unaligned_rows(rows: int) -> None:
    # AuK's transformer sees M = 2 x (text + reference + target frames), rarely a
    # multiple of 32; forced W8A8 must run and agree with W8A16.
    from indextts.quant.convrot_int8 import _int8_gemm_supported

    device = torch.device("cuda")
    if not _int8_gemm_supported(device):
        pytest.skip("torch._int_mm is unavailable")
    generator = torch.Generator(device=device).manual_seed(rows)
    layer = ConvRotInt8Linear(1536, 512, bias=True, group_size=256, device=device, dtype=torch.bfloat16)
    layer.weight_int8.random_(-127, 128, generator=generator)
    layer.weight_scale.uniform_(0.0002, 0.002, generator=generator)
    layer.bias.data.normal_(generator=generator)
    x = torch.randn((rows, 1536), device=device, dtype=torch.bfloat16, generator=generator)
    with torch.inference_mode():
        layer.kernel_mode = "w8a16"
        w8a16 = layer(x)
        layer.kernel_mode = "w8a8"
        w8a8 = layer(x)
    relative = float((w8a8.float() - w8a16.float()).norm() / w8a16.float().norm())
    assert relative < 0.02


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_auto_kernel_choice_of_w8a16_releases_the_w8a8_weight_copy(monkeypatch) -> None:
    from indextts.quant import convrot_int8

    device = torch.device("cuda")
    if not convrot_int8._int8_gemm_supported(device):
        pytest.skip("torch._int_mm is unavailable")
    monkeypatch.setattr(convrot_int8, "_cached_int8_kernel", lambda *args: None)
    monkeypatch.setattr(convrot_int8, "_choose_int8_kernel", lambda *args: "w8a16")
    layer = ConvRotInt8Linear(1536, 512, bias=False, group_size=256, device=device, dtype=torch.bfloat16)
    layer.weight_int8.random_(-127, 128)
    layer.weight_scale.fill_(0.001)
    with torch.inference_mode():
        layer(torch.randn((64, 1536), device=device, dtype=torch.bfloat16))
    assert layer.weight_int8_rhs.numel() == 0
    monkeypatch.setattr(convrot_int8, "_choose_int8_kernel", lambda *args: "w8a8")
    with torch.inference_mode():
        layer(torch.randn((600, 1536), device=device, dtype=torch.bfloat16))
    assert layer.weight_int8_rhs.numel() > 0


def test_meta_initialized_model_loads_and_moves(tmp_path: Path) -> None:
    # AuK builds its INT8 transformer on the meta device and lets the checkpoint
    # supply every tensor; the never-filled W8A8 cache must not block .to().
    state = _source_state()
    destination = tmp_path / "int8.safetensors"
    save_file(state, str(tmp_path / "model.safetensors"))
    _convert(tmp_path / "model.safetensors", destination, keep_float32=("layer_scale", "norm.weight"))
    with torch.device("meta"):
        loaded = _Tiny()
    load_gpt_checkpoint(loaded, str(destination), device="cpu", dtype=torch.bfloat16, strict=True)
    assert not any(tensor.is_meta for tensor in [*loaded.parameters(), *loaded.buffers()])
    assert loaded.proj.weight_int8_rhs.device.type == "cpu" and loaded.proj.weight_int8.dtype == torch.int8
    loaded.to(torch.float32)
    reference = _Tiny()
    reference.load_state_dict(state)
    x = torch.randn(4, 64, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        relative = float((loaded(x) - reference(x)).norm() / reference(x).norm())
    assert relative < 0.03
