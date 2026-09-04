from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from indextts.runtime.vram_presets import RuntimeConfig
import indextts.training.sampling as sampling
from indextts.training.train_config import TrainConfig


def test_training_sample_request_uses_all_configured_inference_values(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "dataset_info.json").write_text(
        json.dumps({"language": "es"}), encoding="utf-8"
    )
    (dataset / "manifest.jsonl").write_text(
        json.dumps({"id": "first", "language": "JA"}) + "\n", encoding="utf-8"
    )
    reference = tmp_path / "reference.wav"
    adapter = tmp_path / "adapter.safetensors"
    reference.write_bytes(b"reference")
    adapter.write_bytes(b"adapter")
    captured: dict = {}

    monkeypatch.setattr(sampling, "gpu_free_gb", lambda _index: 24.0)
    monkeypatch.setattr(sampling, "gpu_total_gb", lambda _index: 32.0)
    monkeypatch.setattr(
        sampling,
        "resolve_preset",
        lambda *_args, **_kwargs: RuntimeConfig(
            device="cpu", cfm_cache_length=4321
        ),
    )

    def fake_run(command, **_kwargs):
        request_path = Path(command[command.index("--request-file") + 1])
        result_path = Path(command[command.index("--result-file") + 1])
        captured.update(json.loads(request_path.read_text(encoding="utf-8")))
        output = Path(captured["task_layout"]["final_wav_path"])
        output.write_bytes(b"wav")
        result_path.write_text(
            json.dumps({"status": "ok", "output_path": str(output)}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sampling.subprocess, "run", fake_run)
    config = TrainConfig(
        dataset_dir=str(dataset),
        name="adapter",
        device="cpu",
        sample_language="auto",
        sample_seed=999,
        sample_temperature=1.25,
        sample_top_p=0.55,
        sample_top_k=0,
        sample_repetition_penalty=6.5,
        sample_num_beams=4,
        sample_emo_alpha=0.35,
        sample_diffusion_steps=37,
        sample_inference_cfg_rate=1.1,
        sample_max_text_tokens=73,
        sample_length_penalty=-0.4,
        sample_max_mel_tokens=987,
        sample_speaking_rate=0.8,
    ).validate()

    result = sampling.generate_training_sample(
        config,
        adapter_path=adapter,
        reference_path=reference,
        output_path=tmp_path / "sample.wav",
        epoch=2,
        seed=123456,
    )

    assert result.generated
    assert captured["language"] == "ES"
    assert captured["seed"] == 123456
    assert captured["max_text_tokens"] == 73
    infer = captured["infer_kwargs"]
    assert infer["temperature"] == 1.25
    assert infer["top_p"] == 0.55
    assert infer["top_k"] is None
    assert infer["repetition_penalty"] == 6.5
    assert infer["num_beams"] == 4
    assert infer["emo_alpha"] == 0.35
    assert infer["diffusion_steps"] == 37
    assert infer["inference_cfg_rate"] == 1.1
    assert infer["max_text_tokens_per_segment"] == 73
    assert infer["length_penalty"] == -0.4
    assert infer["max_mel_tokens"] == 987
    assert infer["latent_multiplier"] == 2.15
    assert infer["emo_audio_prompt"] is None
    assert infer["use_emo_text"] is False
    assert infer["cfm_cache_length"] == 4321


def test_auto_sample_language_falls_back_to_manifest_then_english(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    manifest = dataset / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"id": "first", "language": "AR"}) + "\n", encoding="utf-8"
    )
    config = TrainConfig(dataset_dir=str(dataset), name="adapter").validate()

    assert sampling._sample_language(config) == "AR"
    empty_dataset = tmp_path / "empty-dataset"
    empty_dataset.mkdir()
    empty_config = TrainConfig(
        dataset_dir=str(empty_dataset), name="adapter"
    ).validate()
    assert sampling._sample_language(empty_config) == "EN"
