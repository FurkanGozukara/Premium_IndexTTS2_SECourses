import json
from pathlib import Path
from types import SimpleNamespace

import ui.batch_tab as batch_tab
import ui.generation_tab as generation_tab
from ui.app import build_app
from ui.dataset_tab import dataset_status_to_panel, dataset_status_updates
from ui.generation_tab import build_generation_request, recent_outputs
from ui.grid_tab import grid_status_updates
from ui.training_tab import training_status_updates


ROOT = Path(__file__).resolve().parents[1]


def _args():
    return SimpleNamespace(
        model_dir=str(ROOT / "models"),
        device="cpu",
        verbose=False,
        no_browser=True,
        port=7861,
        host="127.0.0.1",
        share=False,
    )


def test_cpu_ui_flow_mappings_without_loading_models(tmp_path):
    demo = build_app(_args())
    request = build_generation_request(
        demo.preset_registry.defaults(),
        prompt="reference.wav",
        text="A CPU-only request contract check.",
        model_dir=str(ROOT / "models"),
    )
    assert request["runtime"]["lora_merge_into_base"] is False
    assert request["lora_merge_into_base"] is False
    assert request["infer_kwargs"]["section_batch_size"] == 1

    paced_values = {
        **demo.preset_registry.defaults(),
        "generation.latent_multiplier": 2.05,
        "generation.speaking_rate": 0.8,
    }
    paced_request = build_generation_request(paced_values)
    assert paced_request["infer_kwargs"]["latent_multiplier"] == 2.5625

    reference = tmp_path / "batch-reference.wav"
    reference.write_bytes(b"batch reference placeholder")
    batch_request = batch_tab.prepare_generation_request(
        paced_values,
        prompt=str(reference),
        text="Batch uses the shared speaking-rate request builder.",
        subtitle_file=None,
        image_path=None,
        emotion_audio=None,
        model_dir=str(ROOT / "models"),
        output_root=tmp_path / "batch-output",
    )
    assert batch_request["infer_kwargs"]["latent_multiplier"] == 2.5625

    zero_strength_values = {
        **demo.preset_registry.defaults(),
        "runtime.lora_strength": 0.0,
    }
    zero_strength_request = build_generation_request(
        zero_strength_values,
        model_dir=str(ROOT / "models"),
    )
    assert zero_strength_request["lora_strength"] == 0.0
    assert zero_strength_request["runtime"]["lora_strength"] == 0.0

    prep_state = tmp_path / "prep"
    prep_state.mkdir()
    prep_status = {
        "phase": "segments",
        "file_i": 2,
        "file_n": 4,
        "segment_count": 7,
        "total_audio_seconds": 84.0,
        "message": "source.wav: segment 7/12",
        "fraction": 0.375,
        "elapsed_s": 12.5,
        "eta_s": 20.8,
        "speed": 0.56,
        "speed_unit": "segments/s",
        "vram_used_gb": 2.5,
        "vram_total_gb": 31.8,
        "updated_at": 1_788_300_000.0,
    }
    (prep_state / "status.json").write_text(json.dumps(prep_status), encoding="utf-8")
    panel, line = dataset_status_to_panel(prep_status)
    assert "37.5%" in panel
    assert "7 segments" in line
    assert "1.40 minutes" in line
    mapped = dataset_status_updates(str(prep_state), "")
    assert mapped[0] == panel
    assert mapped[1] == line

    real_training = ROOT / "outputs" / "training_runs" / "furkan_dora_r32" / "status.json"
    if real_training.is_file():
        training = training_status_updates(str(real_training.parent), 0.9)
        assert "Training" in training[0]
        assert "step" in training[1].lower()

    grid_state = tmp_path / "grids" / "voice_grid"
    grid_state.mkdir(parents=True)
    (grid_state / "status.json").write_text(
        json.dumps(
            {
                "phase": "generating",
                "message": "best / ref 1 / text 1",
                "completed": 1,
                "total": 4,
                "elapsed_s": 9.0,
            }
        ),
        encoding="utf-8",
    )
    mapped_grid = grid_status_updates(str(grid_state), output_root=tmp_path / "grids")
    assert "25.0%" in mapped_grid[1]
    assert "Attached to running grid" in mapped_grid[2]
    partial_grid = {
        "grid_dir": str(grid_state.resolve()),
        "grid_name": "voice_grid",
        "config": {},
        "seed": 1,
        "cells": [],
        "status": "running",
        "summary_markdown": "",
        "generated_at": "2026-09-04T00:00:00+00:00",
        "elapsed_s": 9.0,
    }
    (grid_state / "grid.json").write_text(
        json.dumps(partial_grid), encoding="utf-8"
    )
    mapped_partial = grid_status_updates(
        str(grid_state), output_root=tmp_path / "grids"
    )
    assert mapped_partial[4] == ""

    complete_status = {
        "phase": "complete",
        "message": "Listening grid complete",
        "completed": 4,
        "total": 4,
        "elapsed_s": 18.0,
    }
    (grid_state / "status.json").write_text(
        json.dumps(complete_status), encoding="utf-8"
    )
    mapped_complete = grid_status_updates(
        str(grid_state), output_root=tmp_path / "grids"
    )
    assert mapped_complete[4] == str(grid_state.resolve())


def _write_output(folder: Path, task_id: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    audio = folder / "final.wav"
    audio.write_bytes(b"RIFF" + b"\0" * 64)
    metadata = {
        "created_at": "2026-09-02T12:00:00+00:00",
        "status": "completed",
        "task": {"id": task_id},
        "outputs": {"final_audio_path": str(audio.resolve())},
    }
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_recent_outputs_only_returns_user_task_folders(tmp_path):
    _write_output(tmp_path / "user_project" / "task_001", "user-task")
    _write_output(tmp_path / "_quality_check" / "task_002", "underscore")
    _write_output(tmp_path / "worker_runtime_e2e" / "task_003", "worker")
    _write_output(tmp_path / ".sample_jobs" / "task_004", "sample")
    _write_output(tmp_path / "ui_batch_smoke_123" / "task_005", "ui-smoke")

    rows = recent_outputs(tmp_path, limit=20)
    assert [row[0] for row in rows] == ["user-task"]


def test_lora_selection_auto_applies_and_resets_calibrated_speaking_rate(
    monkeypatch,
) -> None:
    report = SimpleNamespace(
        recommended_speaking_rate=0.81,
        dataset_words_per_second=2.67,
        generated_words_per_second=3.30,
    )
    monkeypatch.setattr(
        generation_tab,
        "_lora_info",
        lambda _path: ("adapter info", None),
    )
    monkeypatch.setattr(
        generation_tab,
        "load_speaking_rate",
        lambda _path: report,
    )

    selected = generation_tab.lora_selection_updates(
        "voice.safetensors", None, True, True
    )
    assert selected[0] == "adapter info"
    assert selected[3] == 0.81
    assert "Applied calibrated speaking rate 0.81" in selected[2]

    cleared = generation_tab.lora_selection_updates("", None, True, True)
    assert cleared[3] == 1.0
    assert "natural pace" in cleared[2]
