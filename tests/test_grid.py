from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import wave

import pytest

import indextts.training.analysis as analysis_module
import indextts.training.grid as grid_module
from indextts.training.analysis import (
    BASE_CHECKPOINT_LABEL,
    BASE_GRID_HEADER_DETAIL,
    BASE_PHASE_LABEL,
)
from indextts.training.grid import (
    GridCell,
    GridCheckpoint,
    GridConfig,
    GridResult,
    build_grid_cells,
    list_grids,
    load_grid,
    run_grid,
)
from indextts.training.dataset_manifest import write_manifest
from indextts.training.speaking_rate import load_speaking_rate
from ui.generation_tab import GENERATION_DEFAULTS, INFER_KWARG_KEYS, recent_outputs
from ui.grid_tab import (
    GRID_DEFAULTS,
    _analysis_payload,
    _evaluation_reference_line,
    _grid_result_heading,
    _grid_rows,
    _renderable_grid_cells,
    build_grid_config_from_ui,
    calibrate_grid_speaking_rates,
)
from indextts.training.checkpoint_eval import CheckpointEvalReport, CheckpointEvalRow


def _wav(path: Path, frames: int = 2205) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(22050)
        handle.writeframes(b"\0\0" * frames)


def test_ui_grid_with_spaced_voice_name_completes_in_its_polled_folder(tmp_path, monkeypatch):
    from ui import app
    from ui import grid_tab as grid_ui
    from ui.presets_store import PresetStore
    from indextts.training import grid_worker

    monkeypatch.setattr(grid_ui, "ROOT", tmp_path)
    monkeypatch.setattr(app, "PresetStore", lambda registry, _root:
                        PresetStore(registry, tmp_path / "presets", detect_tier=lambda: 32))
    monkeypatch.setattr(app, "load_persisted_runtime", lambda: None)
    monkeypatch.setattr(app, "_runtime_summary", lambda *args, **kwargs: None)
    demo = app.build_app(SimpleNamespace(model_dir="models", device="cpu"))
    event = next(item for item in demo.fns.values() if item.api_name == "generate_checkpoint_grid")
    callback = getattr(event.fn, "original", event.fn)
    keys = {spec.component._id: spec.key for spec in demo.preset_registry.specs}
    voice = tmp_path / "Tutorial Omni Full"
    voice.mkdir()
    reference = tmp_path / "reference.wav"
    _wav(reference)
    overrides = {
        "grid.adapter_dir": str(voice), "grid.checkpoints": ["base"],
        "grid.references": str(reference), "grid.texts": "A complete comparison.",
    }
    args = [overrides.get(keys.get(component._id), component.value) for component in event.inputs]
    args[0] = {"base": {"label": BASE_CHECKPOINT_LABEL, "path": ""}}
    monkeypatch.setattr(grid_module, "create_tts", lambda runtime: object())

    def generate(request, engine):
        path = Path(request["task_layout"]["final_wav_path"])
        _wav(path)
        return {"output_path": str(path), "audio_seconds": 0.1, "seed": request["seed"]}

    monkeypatch.setattr(grid_module, "run_generation_request", generate)
    launched = []

    def start(kind, command, **kwargs):
        launched.append(Path(kwargs["state_dir"]))
        assert grid_worker.main(command[3:]) == 0

    monkeypatch.setattr(grid_ui.PROCESS_MANAGER, "start", start)
    state, *_ = callback(*args)
    folder = Path(state)
    assert launched == [folder]
    assert folder.name.startswith("Tutorial_Omni_Full_")
    assert json.loads((folder / "status.json").read_text())["phase"] == "complete"
    result = load_grid(folder)
    assert result.status == "complete" and len(result.cells) == 1
    assert Path(result.cells[0].audio_path).parent == folder
    assert list(folder.parent.iterdir()) == [folder]


def test_changing_adapters_preserves_comparison_text_and_reference(monkeypatch):
    import gradio as gr
    import ui.grid_tab as grid_ui

    monkeypatch.setattr(grid_ui, "_adapter_context", lambda path: {
        "info": "new adapter", "reference": "suggested.wav", "texts": "Suggested sample.",
    })
    updates = grid_ui.adapter_selection_updates(None, "reviewed.wav", "Same first prompt.\nSame second prompt.")
    assert updates[7] == gr.skip() and updates[8] == gr.skip()
    defaults = grid_ui.adapter_selection_updates(None, "", GRID_DEFAULTS["grid.texts"])
    assert defaults[7:9] == ("suggested.wav", "Suggested sample.")


def test_a_voice_saved_by_another_run_follows_the_selected_run(tmp_path, monkeypatch):
    # Compare in grid (and the preselection after training) kept the reference of the run shown before, so the
    # checkpoints of voice B were rendered cloning voice A. References the user entered still stay.
    import gradio as gr
    import ui.grid_tab as grid_ui

    monkeypatch.setattr(grid_ui, "ROOT", tmp_path)
    first, second = tmp_path / "loras" / "voice_a", tmp_path / "loras" / "voice_b"
    for folder in (first, second):
        folder.mkdir(parents=True)
        _wav(folder / f"{folder.name}_reference.wav")
    monkeypatch.setattr(grid_ui, "_adapter_context", lambda path: {
        "info": "run", "reference": str((Path(path) / f"{Path(path).name}_reference.wav").resolve()), "texts": "Sample.",
    })
    switched = grid_ui.adapter_selection_updates(str(second), str(first / "voice_a_reference.wav"), "Sample.")
    assert switched[7] == str((second / "voice_b_reference.wav").resolve())
    chosen = tmp_path / "my_clip.wav"
    _wav(chosen)
    kept = grid_ui.adapter_selection_updates(str(second), "\n".join([str(first / "voice_a_reference.wav"), str(chosen)]),
                                             "Sample.")
    assert kept[7] == gr.skip()


def test_dynamic_checkpoint_ids_survive_gradio_component_reconstruction():
    from ui.grid_tab import CheckpointCheckboxGroup
    original = CheckpointCheckboxGroup(choices=[("Base", "base")])
    rebuilt = original.__class__(choices=[("Base", "base"), ("New", "checkpoint-0")])
    assert rebuilt.get_block_name() == "checkboxgroup"
    # A pending callback may contain IDs from the previous folder. The grid's
    # checkpoint mapping filters them; Gradio must not reject unrelated events.
    assert rebuilt.preprocess(["base", "checkpoint-7"]) == ["base", "checkpoint-7"]


def _audible_wav(path: Path, seconds: float, sample_rate: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x40" * int(seconds * sample_rate))


def test_cell_order_names_seed_and_grid_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    checkpoint = adapter / "voice_epoch_003.safetensors"
    checkpoint.write_bytes(b"test checkpoint")
    reference_a = tmp_path / "a.wav"
    reference_b = tmp_path / "b.wav"
    _wav(reference_a)
    _wav(reference_b)
    config = GridConfig(
        adapter_dir=str(adapter),
        checkpoints=[
            GridCheckpoint("Base model", ""),
            GridCheckpoint("epoch 3", str(checkpoint)),
        ],
        strengths=[1.0, 0.5],
        references=[str(reference_a), str(reference_b)],
        texts=["first", "second"],
        seed=-1,
        output_root=str(tmp_path / "outputs" / "grids"),
        grid_name="test_grid",
    )
    cells = build_grid_cells(config)
    assert len(cells) == 12
    assert [cell.checkpoint_label for cell in cells[:4]] == [BASE_CHECKPOINT_LABEL] * 4
    assert cells[0].filename == "base__s1__ref1__t1.wav"
    assert cells[4].filename == "epoch_003__s1__ref1__t1.wav"

    engine = object()
    monkeypatch.setattr(grid_module, "create_tts", lambda _runtime: engine)
    monkeypatch.setattr(grid_module.secrets, "randbelow", lambda _limit: 123)

    def fake_request(request, received_engine):
        assert received_engine is engine
        path = Path(request["task_layout"]["final_wav_path"])
        _wav(path)
        return {"output_path": str(path), "audio_seconds": 0.1, "seed": request["seed"]}

    monkeypatch.setattr(grid_module, "run_generation_request", fake_request)
    result = run_grid(config)
    assert result.status == "complete"
    assert {cell.seed for cell in result.cells} == {123}
    assert all(Path(cell.audio_path).is_file() for cell in result.cells)
    assert (Path(result.grid_dir) / "grid.json").is_file()
    assert (Path(result.grid_dir) / "grid.md").is_file()
    progress = json.loads((Path(result.grid_dir) / "progress.json").read_text())
    assert progress["extra"]["audio_seconds"] == pytest.approx(sum(cell.audio_seconds for cell in result.cells))
    assert progress["speed"] == pytest.approx(1.2 / progress["elapsed_s"])
    assert BASE_CHECKPOINT_LABEL in result.summary_markdown
    assert BASE_PHASE_LABEL in result.summary_markdown
    assert "Base model (no adapter)" not in result.summary_markdown
    base_row = _grid_rows(result.grid_dir)[0]
    assert base_row[1] == BASE_CHECKPOINT_LABEL
    assert base_row[2] is None
    assert base_row[6] == BASE_PHASE_LABEL
    loaded = load_grid(result.grid_dir)
    assert loaded is not None and len(loaded.cells) == 12
    summaries = list_grids(config.output_root)
    assert [item.grid_name for item in summaries] == ["test_grid"]
    assert recent_outputs(tmp_path / "outputs", limit=20) == []


def test_grid_result_headers_explain_checkpoint_type_and_base_model() -> None:
    checkpoint = SimpleNamespace(
        checkpoint_label="best (epoch 10 DoRA Checkpoint)",
        checkpoint_path="voice.safetensors",
        checkpoint_kind="best",
        strength=1.0,
        verdict="best",
        val_loss=5.3628,
    )
    base = SimpleNamespace(
        checkpoint_label="Base model (no adapter)",
        checkpoint_path="",
        checkpoint_kind="base",
        strength=1.0,
        verdict="base",
        val_loss=None,
    )

    assert _grid_result_heading(checkpoint) == (
        "#### best (epoch 10 DoRA Checkpoint) @ 1 | Lowest validation loss | "
        "validation loss 5.3628"
    )
    assert _grid_result_heading(base) == (
        f"#### {BASE_CHECKPOINT_LABEL} | {BASE_GRID_HEADER_DETAIL}"
    )
    assert _evaluation_reference_line("other") == (
        "Evaluation references used by this report: **other (inference-like: a "
        "different clip of the same speaker)**."
    )


def test_renderable_grid_cells_omit_blank_and_missing_audio(tmp_path: Path) -> None:
    audio = tmp_path / "complete.wav"
    _wav(audio)
    complete = SimpleNamespace(audio_path=str(audio))
    pending = SimpleNamespace(audio_path="")
    missing = SimpleNamespace(audio_path=str(tmp_path / "missing.wav"))

    result = SimpleNamespace(cells=[complete, pending, missing])

    assert _renderable_grid_cells(result) == [complete]


def test_legacy_evaluation_and_grid_labels_upgrade_only_when_rendered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = tmp_path / "voice"
    checkpoint = adapter / "voice_epoch_030.safetensors"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        analysis_module,
        "inspect_lora",
        lambda _path: {
            "adapter_type": "dora",
            "rank": 128,
            "alpha": 129.0,
            "epochs": 30,
            "steps": 300,
            "train_config": {},
        },
    )
    report = CheckpointEvalReport(
        adapter_dir=str(adapter),
        dataset_dir=str(tmp_path / "dataset"),
        val_items=1,
        train_subset_items=0,
        rows=[
            CheckpointEvalRow(
                label="epoch 30 @0.5",
                path=str(checkpoint),
                kind="epoch",
                epoch=30,
                steps=300,
                strength=0.5,
                val_loss=5.4,
                val_mel_loss=5.4,
                val_text_loss=0.0,
                val_accuracy=0.5,
                train_loss=None,
                train_accuracy=None,
                gap=None,
                phase="best",
                elapsed_s=1.0,
            )
        ],
        best_label="epoch 30",
        best_path=str(checkpoint),
        recommended_checkpoint=str(checkpoint),
        summary_markdown="legacy report",
        device="cpu",
        generated_at="2026-09-02T00:00:00+00:00",
        elapsed_s=1.0,
    )
    analysis_dir = adapter / "analysis"
    analysis_dir.mkdir()
    (analysis_dir / "checkpoint_eval.json").write_text(
        json.dumps(report.to_dict()), encoding="utf-8"
    )

    payload = _analysis_payload(adapter)
    assert payload["rows"][0][0] == "epoch 30 (DoRA Checkpoint) @0.5"
    assert any(
        "epoch 30 (DoRA Checkpoint)" in label
        for label, _identifier in payload["choices"]
    )

    grid_dir = tmp_path / "legacy-grid"
    grid_dir.mkdir()
    cell = GridCell(
        index=1,
        label="legacy",
        filename="cell.wav",
        checkpoint_label="epoch 30",
        checkpoint_path=str(checkpoint),
        checkpoint_kind="epoch",
        strength=1.0,
        reference_index=1,
        text_index=1,
        reference="reference.wav",
        text="legacy grid text",
        seed=1,
    )
    grid = GridResult(
        grid_dir=str(grid_dir),
        grid_name="legacy-grid",
        config={},
        seed=1,
        cells=[cell],
        status="complete",
        summary_markdown="",
        generated_at="2026-09-02T00:00:00+00:00",
        elapsed_s=1.0,
    )
    (grid_dir / "grid.json").write_text(
        json.dumps(grid.to_dict()), encoding="utf-8"
    )
    assert _grid_rows(grid_dir)[0][1] == "epoch 30 (DoRA Checkpoint)"
    assert _grid_result_heading(cell).startswith(
        "#### epoch 30 (DoRA Checkpoint) @ 1"
    )


def test_grid_button_calibrates_rows_and_writes_the_recommended_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    write_manifest(
        dataset / "manifest.jsonl",
        [{"id": "one", "words": 4, "duration_s": 4.0}],
    )
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "train_config.json").write_text(
        json.dumps({"dataset_dir": str(dataset)}), encoding="utf-8"
    )
    first_checkpoint = adapter / "voice_epoch_001.safetensors"
    recommended_checkpoint = adapter / "voice_epoch_002.safetensors"
    first_checkpoint.write_bytes(b"first")
    recommended_checkpoint.write_bytes(b"second")
    monkeypatch.setattr(
        analysis_module,
        "inspect_lora",
        lambda path: {
            "adapter_type": "dora",
            "rank": 128,
            "alpha": 129.0,
            "epochs": 2 if "002" in str(path) else 1,
            "steps": 20 if "002" in str(path) else 10,
            "train_config": {},
        },
    )
    grid_dir = tmp_path / "grid-calibration"
    first_audio = grid_dir / "first.wav"
    second_audio = grid_dir / "second.wav"
    _audible_wav(first_audio, 2.0)
    _audible_wav(second_audio, 4.0)
    cells = [
        GridCell(
            index=1,
            label="first",
            filename=first_audio.name,
            checkpoint_label="epoch 1",
            checkpoint_path=str(first_checkpoint),
            checkpoint_kind="epoch",
            strength=1.0,
            reference_index=1,
            text_index=1,
            reference="reference.wav",
            text="one two three four",
            seed=1,
            audio_path=str(first_audio),
        ),
        GridCell(
            index=2,
            label="second",
            filename=second_audio.name,
            checkpoint_label="epoch 2",
            checkpoint_path=str(recommended_checkpoint),
            checkpoint_kind="epoch",
            strength=1.0,
            reference_index=1,
            text_index=1,
            reference="reference.wav",
            text="one two three four",
            seed=1,
            audio_path=str(second_audio),
        ),
    ]
    result = GridResult(
        grid_dir=str(grid_dir),
        grid_name="grid-calibration",
        config={"adapter_dir": str(adapter)},
        seed=1,
        cells=cells,
        status="complete",
        summary_markdown="",
        generated_at="2026-09-02T00:00:00+00:00",
        elapsed_s=1.0,
    )
    (grid_dir / "grid.json").write_text(
        json.dumps(result.to_dict()), encoding="utf-8"
    )

    table, status = calibrate_grid_speaking_rates(
        grid_dir, adapter, recommended_checkpoint
    )

    assert "epoch 1 (DoRA Checkpoint)" in table
    assert "epoch 2 (DoRA Checkpoint)" in table
    assert "| 2.00 | 1.00 | 0.500 |" in table
    assert "| 1.00 | 1.00 | 1.000 |" in table
    assert "Saved speaking rate 1.000 from epoch 2" in status
    saved = load_speaking_rate(adapter)
    assert saved is not None
    assert saved.method == "grid"
    assert saved.recommended_speaking_rate == 1.0

    _table, mismatch = calibrate_grid_speaking_rates(
        grid_dir, tmp_path / "different-adapter", recommended_checkpoint
    )
    assert "does not belong" in mismatch


def test_frontend_grid_config_carries_full_generation_request(tmp_path: Path) -> None:
    reference = tmp_path / "reference.wav"
    _wav(reference)
    grid_values = dict(GRID_DEFAULTS)
    grid_values.update(
        {
            "grid.adapter_dir": str(tmp_path / "voice"),
            "grid.checkpoints": ["base"],
            "grid.references": str(reference),
            "grid.texts": "A frontend-built grid request.",
            "grid.num_beams": 5,
            "grid.temperature": 0.55,
            "grid.speaking_rate": 0.8,
        }
    )
    generation_values = dict(GENERATION_DEFAULTS)
    generation_values.update(
        {
            "generation.cfm_temperature": 0.37,
            "generation.interval_silence": 777,
            "generation.segment_budget_scale_non_cjk": 1.25,
            "generation.reuse_spk_cond_for_emo": False,
            "generation.enable_pause_tags": False,
            "generation.trim_silence_ms_threshold": 125,
            "generation.target_duration_s": 4.5,
            "generation.target_duration_mode": "trim",
            "generation.latent_multiplier": 2.05,
            "generation.speaking_rate": 0.75,
            "generation.auto_retry_incomplete_speech": False,
            "generation.max_speech_retries": 0,
            "generation.max_speech_split_depth": 5,
        }
    )

    config = build_grid_config_from_ui(
        {"base": {"label": BASE_CHECKPOINT_LABEL, "path": ""}},
        grid_values,
        generation_values,
        model_dir=str(tmp_path / "models"),
        output_root=tmp_path / "outputs" / "grids",
        grid_name="frontend_grid",
    )

    assert INFER_KWARG_KEYS.issubset(config.infer_kwargs)
    assert config.infer_kwargs["cfm_temperature"] == pytest.approx(0.37)
    assert config.infer_kwargs["interval_silence"] == 777
    assert config.infer_kwargs["segment_budget_scale_non_cjk"] == pytest.approx(1.25)
    assert config.infer_kwargs["reuse_spk_cond_for_emo"] is False
    assert config.infer_kwargs["enable_pause_tags"] is False
    assert config.infer_kwargs["trim_silence_ms_threshold"] == 125
    assert config.infer_kwargs["target_duration_s"] == pytest.approx(4.5)
    assert config.infer_kwargs["target_duration_mode"] == "trim"
    assert config.infer_kwargs["num_beams"] == 5
    assert config.infer_kwargs["temperature"] == pytest.approx(0.55)
    assert config.infer_kwargs["latent_multiplier"] == pytest.approx(2.5625)
    assert config.runtime["lora_path"] == ""
    assert config.runtime["lora_strength"] == 1.0
    assert config.runtime["lora_merge_into_base"] is False

    cell = build_grid_cells(config)[0]
    cell.seed = 1234
    request = grid_module._request_for_cell(config, cell, tmp_path / "request_grid")
    # No reference entered: the grid compares with the run's own reference instead of stopping with
    # "add at least one reference audio path" (a run opened on page load never filled the field).
    import ui.grid_tab as grid_ui

    stored = tmp_path / "voice" / "voice_reference.wav"
    _wav(stored)
    empty = build_grid_config_from_ui(
        {"base": {"label": BASE_CHECKPOINT_LABEL, "path": ""}}, {**grid_values, "grid.references": ""},
        generation_values, model_dir=str(tmp_path / "models"), output_root=tmp_path / "outputs" / "grids",
        grid_name="frontend_grid_auto_reference",
    )
    assert empty.references == [str(stored.resolve())] == [grid_ui._adapter_context(str(tmp_path / "voice"))["reference"]]
    assert request["cfm_temperature"] == pytest.approx(0.37)
    assert request["segment_budget_scale_non_cjk"] == pytest.approx(1.25)
    assert request["reuse_spk_cond_for_emo"] is False
    assert request["enable_pause_tags"] is False
    assert request["trim_silence_ms_threshold"] == 125
    assert request["target_duration_s"] == pytest.approx(4.5)
    assert request["target_duration_mode"] == "trim"
    assert request["infer_kwargs"]["interval_silence"] == 777
    assert request["infer_kwargs"]["num_beams"] == 5
    assert request["infer_kwargs"]["latent_multiplier"] == pytest.approx(2.5625)
    assert request["infer_kwargs"]["auto_retry_incomplete_speech"] is False
    assert request["infer_kwargs"]["max_speech_retries"] == 0
    assert request["infer_kwargs"]["max_speech_split_depth"] == 5
