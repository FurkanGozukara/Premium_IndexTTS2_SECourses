from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType

from indextts.runtime.vram_presets import RuntimeConfig
from indextts.training.train_config import TrainConfig
from indextts.training.trainer import LoraTrainer
from indextts.utils import model_downloads
from webui_generation_runner import create_tts


class _FakeIndexTTS2:
    def __init__(self, *, runtime, model_dir, cfg_path, **_kwargs):
        self.runtime = runtime
        self.model_dir = model_dir
        self.cfg_path = cfg_path


def _install_fake_engine(monkeypatch) -> None:
    module = ModuleType("indextts.infer_v2_5")
    module.IndexTTS2 = _FakeIndexTTS2
    monkeypatch.setitem(sys.modules, "indextts.infer_v2_5", module)


def _runtime_payload(model_dir: Path) -> dict:
    runtime = RuntimeConfig(
        device="cpu",
        model_variant="int8_convrot",
        gpt_dtype="fp32",
    )
    return {
        "runtime": runtime.to_dict(),
        "model_dir": str(model_dir),
        "cfg_path": str(model_dir / "config.yaml"),
    }


def test_generation_factory_auto_downloads_missing_int8_on_cpu(
    tmp_path: Path, monkeypatch
) -> None:
    _install_fake_engine(monkeypatch)
    calls: list[Path] = []

    def fake_ensure(models_dir, progress_cb):
        root = Path(models_dir)
        calls.append(root)
        progress_cb(
            0.5,
            desc="gpt_int8_convrot.safetensors: 50.0% (512.0 MB/1.0 GB)",
        )
        target = model_downloads.int8_gpt_path(root)
        target.write_bytes(b"int8")
        progress_cb(1.0, desc="IndexTTS 2.5 INT8 ConvRot GPT ready")
        return str(target)

    monkeypatch.setattr(model_downloads, "ensure_int8_gpt", fake_ensure)
    progress_path = tmp_path / "progress.json"

    engine = create_tts(
        _runtime_payload(tmp_path),
        progress_file=str(progress_path),
    )

    assert calls == [tmp_path]
    assert engine.runtime.model_variant == "int8_convrot"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["desc"].startswith("Downloading INT8 ConvRot GPT ")
    assert progress["desc"].endswith(" MB")


def test_generation_factory_falls_back_to_bf16_when_int8_download_fails(
    tmp_path: Path, monkeypatch
) -> None:
    _install_fake_engine(monkeypatch)

    def offline(_models_dir, _progress_cb):
        raise OSError("offline for test")

    monkeypatch.setattr(model_downloads, "ensure_int8_gpt", offline)
    progress_path = tmp_path / "progress.json"

    engine = create_tts(
        _runtime_payload(tmp_path),
        progress_file=str(progress_path),
    )

    assert engine.runtime.model_variant == "bf16"
    assert "falling back to the BF16 GPT" in engine.runtime_warning
    assert model_downloads.INT8_GPT_REPO_ID in engine.runtime_warning
    assert model_downloads.INT8_GPT_REMOTE_FILENAME in engine.runtime_warning
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["desc"] == engine.runtime_warning


def test_training_worker_preflight_auto_downloads_missing_int8(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[Path] = []

    def fake_ensure(models_dir, progress_cb):
        root = Path(models_dir)
        calls.append(root)
        progress_cb(0.25, desc="checkpoint (256.0 MB/1024.0 MB)")
        target = model_downloads.int8_gpt_path(root)
        target.write_bytes(b"int8")
        return str(target)

    monkeypatch.setattr(model_downloads, "ensure_int8_gpt", fake_ensure)
    config = TrainConfig(
        dataset_dir=str(tmp_path / "dataset"),
        name="int8_preflight",
        output_dir=str(tmp_path / "loras"),
        model_dir=str(tmp_path),
        model_config=str(tmp_path / "config.yaml"),
        base_variant="int8_convrot",
        device="cpu",
        base_dtype="fp32",
        mixed_precision="fp32",
    )
    trainer = LoraTrainer(config, state_dir=tmp_path / "state")

    warning = trainer._prepare_base_variant()

    assert warning == ""
    assert calls == [tmp_path]
    status = json.loads((tmp_path / "state" / "status.json").read_text(encoding="utf-8"))
    assert status["message"].startswith("Downloading INT8 ConvRot GPT ")
    assert status["message"].endswith(" MB")
