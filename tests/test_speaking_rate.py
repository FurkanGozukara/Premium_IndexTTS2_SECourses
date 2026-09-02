from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from indextts.training.dataset_manifest import write_manifest
from indextts.training.speaking_rate import (
    SpeakingRateReport,
    calibrate_from_grid,
    calibrate_from_samples,
    dataset_words_per_second,
    load_speaking_rate,
    words_per_second,
    write_speaking_rate,
)


def _wav(
    path: Path,
    active_seconds: float,
    *,
    leading_seconds: float = 0.25,
    trailing_seconds: float = 0.25,
    sample_rate: int = 8000,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    active = np.full(int(active_seconds * sample_rate), 0.5, dtype=np.float32)
    audio = np.concatenate(
        [
            np.zeros(int(leading_seconds * sample_rate), dtype=np.float32),
            active,
            np.zeros(int(trailing_seconds * sample_rate), dtype=np.float32),
        ]
    )
    sf.write(path, audio, sample_rate, subtype="FLOAT")


def _dataset(path: Path, *, words: int, duration_s: float) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    write_manifest(
        path / "manifest.jsonl",
        [
            {
                "id": "a",
                "audio": "a.wav",
                "text": "unused",
                "words": words // 2,
                "duration_s": duration_s / 2,
            },
            {
                "id": "b",
                "audio": "b.wav",
                "text": "unused",
                "words": words - words // 2,
                "duration_s": duration_s - duration_s / 2,
            },
        ],
    )
    return path


def test_words_per_second_removes_pause_tags_and_trims_40_db_edges(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "spoken.wav"
    _wav(audio, 2.0, leading_seconds=0.5, trailing_seconds=0.75)

    measured = words_per_second(
        "One two [pause:500ms] three <pause=0.2> 42",
        audio,
    )

    assert measured == pytest.approx(2.0, abs=0.001)
    short = tmp_path / "short.wav"
    _wav(short, 0.75)
    assert words_per_second("one two", short) == 0.0


def test_dataset_words_per_second_uses_aggregate_manifest_counts(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path / "dataset", words=8, duration_s=3.0)
    assert dataset_words_per_second(dataset) == pytest.approx(8 / 3)


@pytest.mark.parametrize(
    ("dataset_wps", "generated_wps", "expected"),
    [(0.5, 4.0, 0.5), (6.0, 2.0, 1.5)],
)
def test_sample_calibration_clamps_and_round_trips(
    tmp_path: Path,
    dataset_wps: float,
    generated_wps: float,
    expected: float,
) -> None:
    dataset = _dataset(
        tmp_path / f"dataset-{expected}", words=12, duration_s=12 / dataset_wps
    )
    adapter = tmp_path / f"adapter-{expected}"
    sample_text = "one two three four"
    _wav(adapter / "samples" / "epoch_001.wav", 4 / generated_wps)

    report = calibrate_from_samples(adapter, dataset, sample_text)

    assert report is not None
    assert report.method == "training_samples"
    assert report.clips_used == 1
    assert report.recommended_speaking_rate == expected
    path = write_speaking_rate(adapter, report)
    assert path == adapter.resolve() / "analysis" / "speaking_rate.json"
    assert load_speaking_rate(adapter) == report

    checkpoint = adapter / "voice.safetensors"
    best = adapter / "best" / "voice.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    best.parent.mkdir()
    best.write_bytes(b"best checkpoint")
    assert load_speaking_rate(checkpoint) == report
    assert load_speaking_rate(best) == report


def test_grid_calibration_uses_only_the_requested_checkpoint_cells(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path / "dataset", words=6, duration_s=3.0)
    grid = tmp_path / "grid"
    chosen = grid / "chosen.wav"
    ignored = grid / "ignored.wav"
    _wav(chosen, 2.0)
    _wav(ignored, 8.0)
    (grid / "grid.json").write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "checkpoint_label": "epoch 1",
                        "checkpoint_path": str(tmp_path / "epoch1.safetensors"),
                        "text": "one two three four",
                        "audio_path": str(chosen),
                    },
                    {
                        "checkpoint_label": "epoch 2",
                        "checkpoint_path": str(tmp_path / "epoch2.safetensors"),
                        "text": "one two three four",
                        "audio_path": str(ignored),
                    },
                    {
                        "checkpoint_label": "Base model",
                        "checkpoint_path": "",
                        "text": "one two three four",
                        "audio_path": str(ignored),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    report = calibrate_from_grid(grid, "epoch 1", dataset)

    assert isinstance(report, SpeakingRateReport)
    assert report.method == "grid"
    assert report.clips_used == 1
    assert report.dataset_words_per_second == 2.0
    assert report.generated_words_per_second == 2.0
    assert report.recommended_speaking_rate == 1.0
