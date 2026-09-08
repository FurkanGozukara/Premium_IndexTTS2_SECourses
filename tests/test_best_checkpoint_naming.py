"""The lowest-validation-loss checkpoint carries ``_best`` in its file name; older trainings are migrated."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from indextts.training.analysis import checkpoint_descriptor, discover_checkpoints
from indextts.training.best_checkpoint import (
    best_checkpoint_path,
    describe_migration,
    is_best_checkpoint_name,
    legacy_best_checkpoint_path,
    migrate_legacy_best_checkpoints,
    migrate_run_best_checkpoints,
)
from indextts.training.selection import recommended_generation_value


def _adapter(path: Path, component: str = "gpt") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"dummy": np.zeros(1, dtype=np.float32)}, str(path), metadata={
        "adapter_type": "dora", "trained_steps": "100", "epochs": "6",
        "train_config": json.dumps({"component": component}),
    })
    return path


def _legacy_run(root: Path, name: str, *, phase: str = "complete", siblings=(".train_state.pt",)) -> Path:
    run = root / name
    best = _adapter(run / "best" / f"{name}.safetensors")
    for tail in siblings:
        (best.parent / f"{name}{tail}").write_bytes(b"sibling")
    (run / "status.json").write_text(
        json.dumps({"phase": phase, "recommended_checkpoint": str(best), "last_checkpoint": str(best)}),
        encoding="utf-8",
    )
    analysis = run / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    (analysis / "checkpoint_eval.json").write_text(
        json.dumps({"best_path": str(best), "rows": [{"path": str(best)}, {"path": str(run / f"{name}.safetensors")}]}),
        encoding="utf-8",
    )
    (analysis / "training_analysis.md").write_text(
        f"**Recommended checkpoint:** `best/{name}.safetensors`.\nAlso `best/{name}box.safetensors` stays.\n",
        encoding="utf-8",
    )
    return run


def test_best_checkpoint_path_carries_the_suffix(tmp_path: Path) -> None:
    path = best_checkpoint_path(tmp_path / "voice", "voice")
    assert path == tmp_path / "voice" / "best" / "voice_best.safetensors"
    assert is_best_checkpoint_name(path)
    assert not is_best_checkpoint_name(legacy_best_checkpoint_path(tmp_path / "voice", "voice"))
    assert not is_best_checkpoint_name(tmp_path / "voice" / "voice_best.safetensors")


def test_suffixed_file_is_still_discovered_as_the_best_kind(tmp_path: Path) -> None:
    best = _adapter(tmp_path / "best" / "voice_best.safetensors")
    final = _adapter(tmp_path / "voice.safetensors")
    descriptor = checkpoint_descriptor(best)
    assert descriptor["kind"] == "best"
    assert descriptor["file_label"] == "best_ep6"
    assert [row["path"] for row in discover_checkpoints(tmp_path)] == [str(best), str(final)]


def test_migration_renames_files_siblings_and_stored_references(tmp_path: Path) -> None:
    run = _legacy_run(tmp_path, "voice", siblings=(".train_state.pt", ".s2mel.safetensors", "_reference.wav"))
    records = migrate_run_best_checkpoints(run)
    assert [record.skipped for record in records] == [""]
    assert sorted(item.name for item in (run / "best").iterdir()) == [
        "voice_best.s2mel.safetensors",
        "voice_best.safetensors",
        "voice_best.train_state.pt",
        "voice_best_reference.wav",
    ]
    new_best = run / "best" / "voice_best.safetensors"
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    assert Path(status["recommended_checkpoint"]) == new_best
    assert Path(status["last_checkpoint"]) == new_best
    report = json.loads((run / "analysis" / "checkpoint_eval.json").read_text(encoding="utf-8"))
    assert Path(report["best_path"]) == new_best
    assert Path(report["rows"][0]["path"]) == new_best
    assert Path(report["rows"][1]["path"]) == run / "voice.safetensors", "the final checkpoint is untouched"
    markdown = (run / "analysis" / "training_analysis.md").read_text(encoding="utf-8")
    assert "`best/voice_best.safetensors`" in markdown
    assert "`best/voicebox.safetensors`" in markdown, "other stems are untouched"
    assert len(records[0].siblings) == 3
    assert {item.name for item in records[0].rewritten_files} == {
        "status.json", "checkpoint_eval.json", "training_analysis.md",
    }
    assert recommended_generation_value(run) == str(new_best.resolve())
    assert migrate_run_best_checkpoints(run) == [], "a migrated training is left alone"


def test_migration_waits_for_a_running_training(tmp_path: Path) -> None:
    run = _legacy_run(tmp_path, "voice", phase="training")
    records = migrate_run_best_checkpoints(run)
    assert records[0].skipped == "training is still running"
    assert (run / "best" / "voice.safetensors").is_file()
    forced = migrate_run_best_checkpoints(run, skip_active=False)
    assert forced[0].skipped == ""
    assert (run / "best" / "voice_best.safetensors").is_file()


def test_migration_keeps_both_files_when_the_new_name_exists(tmp_path: Path) -> None:
    run = _legacy_run(tmp_path, "voice")
    newer = _adapter(run / "best" / "voice_best.safetensors")
    before = newer.read_bytes()
    records = migrate_run_best_checkpoints(run)
    assert records[0].skipped.endswith("already exists")
    assert (run / "best" / "voice.safetensors").is_file()
    assert newer.read_bytes() == before
    assert (run / "best" / "voice.train_state.pt").is_file()


def test_root_migration_walks_runs_and_ignores_analysis_output(tmp_path: Path) -> None:
    root = tmp_path / "loras"
    _legacy_run(root, "alpha")
    _legacy_run(root / "nested", "beta")
    decoy = _adapter(root / "alpha" / "analysis" / "best" / "alpha.safetensors")
    _adapter(root / "gamma" / "best" / "gamma_best.safetensors")
    records = migrate_legacy_best_checkpoints(root)
    assert sorted(record.run for record in records) == ["alpha", "beta"]
    assert (root / "alpha" / "best" / "alpha_best.safetensors").is_file()
    assert (root / "nested" / "beta" / "best" / "beta_best.safetensors").is_file()
    assert decoy.is_file()
    lines = describe_migration(records)
    assert len(lines) == 2 and all(line.startswith("renamed ") for line in lines)
    assert migrate_legacy_best_checkpoints(tmp_path / "missing") == []


def test_a_training_named_with_the_suffix_is_still_migrated(tmp_path: Path) -> None:
    run = _legacy_run(tmp_path, "voice_best")
    records = migrate_run_best_checkpoints(run)
    assert [record.skipped for record in records] == [""]
    assert (run / "best" / "voice_best_best.safetensors").is_file()
    assert (run / "best" / "voice_best_best.train_state.pt").is_file()
    assert migrate_run_best_checkpoints(run) == []


def test_explicit_name_wins_over_the_folder_name(tmp_path: Path) -> None:
    run = _legacy_run(tmp_path, "voice")
    assert migrate_run_best_checkpoints(run, name="other") == []
    assert (run / "best" / "voice.safetensors").is_file()
    assert [record.skipped for record in migrate_run_best_checkpoints(run, name="voice")] == [""]
