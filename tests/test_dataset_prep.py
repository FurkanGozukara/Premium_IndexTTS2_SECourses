from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from indextts.training.dataset_manifest import load_manifest
from indextts.training.dataset_prep import DatasetPrepConfig, run_dataset_prep
from indextts.training.media import extract_audio


SOURCE2 = Path(r"G:\Index_TTS_v4\Lora_Training_Dataset\source2")
VIDEO2 = SOURCE2 / "video2.mp4"


@pytest.mark.skipif(not VIDEO2.is_file(), reason="SECourses media fixture is not installed")
def test_dataset_prep_end_to_end_sidecar_path(tmp_path: Path) -> None:
    config = DatasetPrepConfig(
        name="source2_test",
        inputs=[str(SOURCE2)],
        output_root=str(tmp_path),
        subtitle_policy="sidecar_only",
        align_with_whisper=False,
        segmentation_mode="cue_boundaries",
        max_segments=12,
        export_reference_candidates=2,
    )
    summary = run_dataset_prep(config)
    dataset_dir = tmp_path / "source2_test"
    rows = load_manifest(dataset_dir)
    assert summary.status == "complete"
    assert len(rows) == summary.segment_count == 12
    assert all(row["transcript_source"] == "sidecar_srt" for row in rows)
    assert all(config.min_s <= row["duration_s"] <= config.max_s for row in rows)
    assert (dataset_dir / "preview.csv").is_file()

    for row in rows:
        audio, sample_rate = sf.read(dataset_dir / row["audio"], always_2d=True)
        assert sample_rate == 24000
        assert audio.shape[1] == 1
        assert audio.shape[0] / sample_rate == pytest.approx(row["duration_s"], abs=1 / sample_rate)

    info = json.loads((dataset_dir / "dataset_info.json").read_text(encoding="utf-8"))
    assert info["status"] == "complete"
    assert info["segment_count"] == len(rows)
    assert info["sample_rate"] == 24000 if "sample_rate" in info else True
    assert info["cache"]["manifest_rewrite_required"] is False
    assert len(info["reference_candidates"]) == 2


def test_import_presegmented_metadata_without_recutting(tmp_path: Path) -> None:
    source = tmp_path / "import_source"
    source.mkdir()
    sample_rate = 22050
    time = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    stereo = np.stack(
        [0.03 * np.sin(2 * np.pi * 180 * time), 0.03 * np.sin(2 * np.pi * 220 * time)],
        axis=1,
    )
    sf.write(source / "clip.wav", stereo, sample_rate, subtype="PCM_16")
    (source / "metadata.csv").write_text(
        "clip.wav|A clean imported sentence.|Speaker A\n",
        encoding="utf-8",
    )
    config = DatasetPrepConfig(
        name="import_test",
        inputs=[str(source)],
        output_root=str(tmp_path),
        min_s=1.5,
        export_reference_candidates=0,
    )
    summary = run_dataset_prep(config)
    row = load_manifest(tmp_path / "import_test")[0]
    audio, output_rate = sf.read(tmp_path / "import_test" / row["audio"], always_2d=True)
    assert summary.segment_count == 1
    assert output_rate == 24000
    assert audio.shape[1] == 1
    assert row["speaker"] == "Speaker A"
    assert row["transcript_source"] == "metadata_csv"
    assert row["duration_s"] == pytest.approx(2.0, abs=2 / output_rate)


class _RecordingReporter:
    def __init__(self) -> None:
        self.logs: list[str] = []

    def update(self, *args, **kwargs) -> None:
        return None

    def log(self, message: str) -> None:
        self.logs.append(message)

    def set_stage(self, name: str) -> None:
        return None

    def finish(self) -> None:
        return None


def _run_duplicate_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    drop_duplicate_sentences: bool,
):
    import indextts.training.dataset_prep as dataset_prep

    source = tmp_path / "duplicate_source"
    source.mkdir()
    sample_rate = 8000
    fixture_rows = [
        ("a_near.wav", "Hello [music], WORLD! <pause=0.5>", 2.0, 0.70),
        ("b_aligned.wav", "hello world", 2.8, 0.95),
        ("c_unique.wav", "A unique training sentence.", 2.2, 0.80),
    ]
    for filename, _, duration_s, _ in fixture_rows:
        time = np.arange(round(sample_rate * duration_s), dtype=np.float32) / sample_rate
        sf.write(
            source / filename,
            0.05 * np.sin(2 * np.pi * 220 * time),
            sample_rate,
            subtype="PCM_16",
        )
    (source / "metadata.csv").write_text(
        "".join(f"{filename}|{text}|Speaker A\n" for filename, text, _, _ in fixture_rows),
        encoding="utf-8",
    )

    coverage_by_text = {text: coverage for _, text, _, coverage in fixture_rows}
    original_base_row = dataset_prep._base_row

    def base_row_with_alignment(**kwargs):
        row = original_base_row(**kwargs)
        row["alignment_coverage"] = coverage_by_text[kwargs["text"]]
        return row

    monkeypatch.setattr(dataset_prep, "_base_row", base_row_with_alignment)
    reporter = _RecordingReporter()
    config = DatasetPrepConfig(
        name="duplicates_on" if drop_duplicate_sentences else "duplicates_off",
        inputs=[str(source)],
        output_root=str(tmp_path),
        sample_rate=sample_rate,
        min_s=1.5,
        target_s=2.0,
        max_s=3.0,
        min_words_per_second=0.1,
        max_words_per_second=20.0,
        min_peak_dbfs=-80.0,
        loudness_normalize=False,
        export_reference_candidates=0,
        drop_duplicate_sentences=drop_duplicate_sentences,
    )
    summary = run_dataset_prep(config, reporter=reporter)
    return summary, Path(summary.output_dir), reporter


def test_dataset_prep_keeps_best_aligned_duplicate_sentence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary, dataset_dir, reporter = _run_duplicate_fixture(
        tmp_path, monkeypatch, drop_duplicate_sentences=True
    )
    rows = load_manifest(dataset_dir)

    assert summary.segment_count == len(rows) == 2
    assert {row["text"] for row in rows} == {
        "hello world",
        "A unique training sentence.",
    }
    assert next(row for row in rows if row["text"] == "hello world")[
        "alignment_coverage"
    ] == pytest.approx(0.95)
    assert summary.filter_drop_counts["duplicate_sentence"] == 1
    assert summary.filter_keep_counts == {"pause_boundary": 0}
    assert summary.subtitle_stats["duplicate_sentences_dropped"] == 1
    source_stats = {
        Path(source["source_media"]).name: source for source in summary.sources
    }
    assert source_stats["a_near.wav"]["filter_drop_counts"]["duplicate_sentence"] == 1
    assert source_stats["b_aligned.wav"]["filter_drop_counts"]["duplicate_sentence"] == 0
    assert all(
        source["filter_keep_counts"] == {"pause_boundary": 0}
        for source in source_stats.values()
    )
    info = json.loads((dataset_dir / "dataset_info.json").read_text(encoding="utf-8"))
    assert info["segment_count"] == 2
    assert info["filter_drop_counts"]["duplicate_sentence"] == 1
    assert info["filter_keep_counts"] == {"pause_boundary": 0}
    assert info["subtitle_stats"]["duplicate_sentences_dropped"] == 1
    assert {path.name for path in (dataset_dir / "segments").glob("*.wav")} == {
        Path(row["audio"]).name for row in rows
    }
    with (dataset_dir / "preview.csv").open(encoding="utf-8-sig", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 2
    assert (
        ">> dropped 1 duplicate sentence(s) (kept the best-aligned copy of each)"
        in reporter.logs
    )


def test_dataset_prep_can_keep_duplicate_sentences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary, dataset_dir, reporter = _run_duplicate_fixture(
        tmp_path, monkeypatch, drop_duplicate_sentences=False
    )

    assert summary.segment_count == len(load_manifest(dataset_dir)) == 3
    assert summary.filter_drop_counts["duplicate_sentence"] == 0
    assert summary.subtitle_stats["duplicate_sentences_dropped"] == 0
    assert len(list((dataset_dir / "segments").glob("*.wav"))) == 3
    assert not any("duplicate sentence(s)" in message for message in reporter.logs)


def test_segmentation_mode_auto_and_legacy_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    import indextts.training.dataset_prep as dataset_prep

    monkeypatch.setattr(dataset_prep, "_cuda_available", lambda: False)
    cpu_config = DatasetPrepConfig(name="cpu", inputs=["fixture"])
    assert cpu_config.segmentation_mode == "cue_boundaries"
    assert cpu_config.resolved_segmentation_mode() == "cue_boundaries"

    auto_config = DatasetPrepConfig(name="auto", inputs=["fixture"], segmentation_mode="auto")
    assert auto_config.resolved_segmentation_mode() == "cue_boundaries"
    alias_config = DatasetPrepConfig(
        name="alias",
        inputs=["fixture"],
        segmentation_mode="cue_boundaries",
        align_with_whisper=True,
    )
    assert alias_config.resolved_segmentation_mode() == "sentence_aligned"

    whisper_config = DatasetPrepConfig(
        name="whisper",
        inputs=["fixture"],
        subtitle_policy="sidecar_only",
        segmentation_mode="whisper_only",
    )
    assert whisper_config.resolved_segmentation_mode() == "whisper_only"


def test_prepare_dataset_cli_maps_pause_boundary_options() -> None:
    from tools.prepare_dataset import build_parser, config_from_args

    args = build_parser().parse_args(
        [
            "input.wav",
            "--name",
            "pause_fixture",
            "--boundary-mode",
            "sentence_or_pause",
            "--min-pause-boundary-ms",
            "525",
        ]
    )
    config = config_from_args(args)

    assert config.boundary_mode == "sentence_or_pause"
    assert config.min_pause_boundary_ms == 525


def test_prepare_dataset_cli_uses_dataclass_defaults() -> None:
    from tools.prepare_dataset import build_parser, config_from_args

    args = build_parser().parse_args(["input.wav", "--name", "default_fixture"])

    assert config_from_args(args).to_dict() == DatasetPrepConfig(
        name="default_fixture",
        inputs=["input.wav"],
    ).to_dict()


@pytest.mark.gpu
@pytest.mark.skipif(not VIDEO2.is_file(), reason="SECourses media fixture is not installed")
def test_whisper_word_timestamps_on_sixty_second_slice(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    clip = tmp_path / "video2_60s.wav"
    extract_audio(VIDEO2, clip, sample_rate=24000, mono=True, start_s=0.0, end_s=60.0)
    from indextts.training.whisper_asr import transcribe

    transcript = transcribe(clip, 24000, "EN", device="cuda:0")
    assert len(transcript.words) > 50
    assert all(word.start_s < word.end_s for word in transcript.words)
    assert all(
        transcript.words[index].start_s >= transcript.words[index - 1].start_s
        for index in range(1, len(transcript.words))
    )
    assert transcript.words[-1].end_s <= 61.0
