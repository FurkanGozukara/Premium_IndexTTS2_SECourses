"""The benchmark comparison tool can compare against a run whose recommendation is the Base model."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_tool():
    path = Path(__file__).resolve().parents[1] / "tools" / "compare_checkpoint_benchmark.py"
    spec = importlib.util.spec_from_file_location("compare_checkpoint_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_run(root: Path) -> None:
    folder = root / "analysis" / "speech_evaluation"
    folder.mkdir(parents=True)
    (folder / "plan.json").write_text(json.dumps({"seeds": [42], "groups": []}), encoding="utf-8")
    report = {
        "status": "complete",
        "recommended_kind": "base",
        "recommended_checkpoint": "",
        "candidates": [
            {"label": "Base", "path": "", "mean_error_rate": 0.01},
            {"label": "epoch 6 (DoRA Checkpoint)", "path": str(root / "epoch_006.safetensors"), "sha256": "x"},
        ],
        "cells": [
            {"checkpoint": "Base", "prompt_id": "p1", "seed": 42, "error_rate": 0.0, "speaker_similarity_real": 0.8},
            {"checkpoint": "epoch 6 (DoRA Checkpoint)", "prompt_id": "p1", "seed": 42, "error_rate": 0.1, "speaker_similarity_real": 0.82},
        ],
        "inference": {"infer_kwargs": {"num_beams": 3}},
    }
    (folder / "report.json").write_text(json.dumps(report), encoding="utf-8")


def test_empty_or_base_baseline_uses_the_runs_base_measurement(tmp_path: Path) -> None:
    tool = _load_tool()
    _write_run(tmp_path)
    for value in ("", "base", "BASE"):
        plan, rows, label, inference, checkpoint = tool.resolve_baseline(tmp_path, value, "")
        assert label == "Base"
        assert [row["prompt_id"] for row in rows] == ["p1"]
        assert inference == {"infer_kwargs": {"num_beams": 3}}
        assert checkpoint == ""
        assert plan["seeds"] == [42]


def test_missing_base_measurement_is_reported(tmp_path: Path) -> None:
    tool = _load_tool()
    _write_run(tmp_path)
    folder = tmp_path / "analysis" / "speech_evaluation"
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    report["candidates"] = [row for row in report["candidates"] if row["path"]]
    (folder / "report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(SystemExit, match="no Base measurement"):
        tool.resolve_baseline(tmp_path, "", "")


def test_named_baseline_still_goes_through_the_verified_path(tmp_path: Path, monkeypatch) -> None:
    tool = _load_tool()
    _write_run(tmp_path)
    from indextts.training import speech_eval

    calls = []

    def fake(run_dir, checkpoint):
        calls.append((Path(run_dir), checkpoint))
        return {"seeds": [1]}, [{"checkpoint": "epoch 6 (DoRA Checkpoint)"}], "epoch 6 (DoRA Checkpoint)", Path("r"), {}

    monkeypatch.setattr(speech_eval, "development_baseline", fake)
    target = tmp_path / "epoch_006.safetensors"
    plan, rows, label, inference, checkpoint = tool.resolve_baseline(tmp_path, str(target), "")
    assert calls and calls[0][1] == str(target.resolve())
    assert label == "epoch 6 (DoRA Checkpoint)" and checkpoint == str(target.resolve())
