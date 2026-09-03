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


def _tone_wav(
    path: Path,
    active_seconds: float,
    *,
    silence_seconds: float = 0.2,
    sample_rate: int = 8000,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(active_seconds * sample_rate)
    tone = 0.5 * np.cos(
        2.0 * np.pi * 220.0 * np.arange(frames, dtype=np.float32) / sample_rate
    )
    silence = np.zeros(int(silence_seconds * sample_rate), dtype=np.float32)
    sf.write(
        path,
        np.concatenate([silence, tone.astype(np.float32), silence]),
        sample_rate,
        subtype="FLOAT",
    )


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


def test_grid_calibration_uses_four_matched_dataset_sentences(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    manifest_texts = [
        '  "One TWO three four!"  ',
        "Five six seven eight.",
        "Nine ten eleven twelve!",
        "Thirteen fourteen fifteen sixteen?",
    ]
    cell_texts = [
        "one   two three FOUR.",
        "FIVE SIX SEVEN EIGHT",
        '"Nine ten eleven twelve"',
        "thirteen fourteen fifteen sixteen",
    ]
    rows = []
    cells = []
    grid = tmp_path / "grid-matched"
    checkpoint = tmp_path / "epoch1.safetensors"
    for index, (manifest_text, cell_text) in enumerate(
        zip(manifest_texts, cell_texts, strict=True), start=1
    ):
        recording = dataset / f"recording-{index}.wav"
        generated = grid / f"generated-{index}.wav"
        _tone_wav(recording, 2.0)
        _tone_wav(generated, 1.5)
        rows.append(
            {
                "id": str(index),
                "audio": recording.name,
                "text": manifest_text,
                "words": 4,
                # Deliberately includes much more pause than the trimmed WAV.
                "duration_s": 10.0,
            }
        )
        cells.append(
            {
                "checkpoint_label": "epoch 1",
                "checkpoint_path": str(checkpoint),
                "text": cell_text,
                "audio_path": str(generated),
            }
        )
    write_manifest(dataset / "manifest.jsonl", rows)
    (grid / "grid.json").write_text(
        json.dumps({"cells": cells}), encoding="utf-8"
    )

    report = calibrate_from_grid(grid, "epoch 1", dataset)

    assert isinstance(report, SpeakingRateReport)
    assert report.method == "grid_matched"
    assert report.clips_used == 4
    assert report.recommended_speaking_rate == pytest.approx(0.75, abs=0.001)
    assert report.dataset_words_per_second == pytest.approx(2.0, abs=0.001)
    assert report.generated_words_per_second == pytest.approx(8 / 3, abs=0.001)
    assert report.summary.startswith(
        "Across 4 sentences that exist in your recordings,"
    )
    assert "33 % faster than you" in report.summary


def test_grid_calibration_falls_back_when_fewer_than_four_sentences_match(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset-fallback"
    dataset.mkdir()
    manifest_texts = [
        "one two three four",
        "five six seven eight",
        "nine ten eleven twelve",
        "this sentence appears only in the manifest",
    ]
    rows = []
    for index, text in enumerate(manifest_texts, start=1):
        recording = dataset / f"recording-{index}.wav"
        _tone_wav(recording, 2.0)
        rows.append(
            {
                "id": str(index),
                "audio": recording.name,
                "text": text,
                "words": 4,
                "duration_s": 2.0,
            }
        )
    write_manifest(dataset / "manifest.jsonl", rows)

    grid = tmp_path / "grid-fallback"
    checkpoint = tmp_path / "epoch1.safetensors"
    cell_texts = manifest_texts[:3] + ["a grid sentence with no recording"]
    cells = []
    for index, text in enumerate(cell_texts, start=1):
        generated = grid / f"generated-{index}.wav"
        _tone_wav(generated, 1.5)
        cells.append(
            {
                "checkpoint_label": "epoch 1",
                "checkpoint_path": str(checkpoint),
                "text": text,
                "audio_path": str(generated),
            }
        )
    (grid / "grid.json").write_text(
        json.dumps({"cells": cells}), encoding="utf-8"
    )

    report = calibrate_from_grid(grid, "epoch 1", dataset)

    assert isinstance(report, SpeakingRateReport)
    assert report.method == "grid"
    assert report.clips_used == 4
    assert report.dataset_words_per_second == pytest.approx(2.0)
    assert report.generated_words_per_second == pytest.approx(3.0, abs=0.001)
    assert report.recommended_speaking_rate == pytest.approx(0.667, abs=0.001)


@pytest.mark.parametrize("method", ["training_samples", "grid"])
def test_old_speaking_rate_json_methods_still_load(
    tmp_path: Path, method: str
) -> None:
    adapter = tmp_path / method
    analysis = adapter / "analysis"
    analysis.mkdir(parents=True)
    (analysis / "speaking_rate.json").write_text(
        json.dumps(
            {
                "recommended_speaking_rate": 0.9,
                "dataset_words_per_second": 2.7,
                "generated_words_per_second": 3.0,
                "clips_used": 4,
                "method": method,
                "generated_at": "2026-09-02T00:00:00+00:00",
                "summary": "Legacy calibration.",
            }
        ),
        encoding="utf-8",
    )

    report = load_speaking_rate(adapter)

    assert report is not None
    assert report.method == method
    assert report.recommended_speaking_rate == 0.9
