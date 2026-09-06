import json

import pytest

from indextts.training.prep_worker import WorkerReporter


def test_whisper_completion_preserves_outer_file_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(WorkerReporter, "_vram", staticmethod(lambda: (0.0, 0.0)))
    reporter = WorkerReporter(tmp_path)
    reporter.update(0, 17, "Extracting first.flac", {"phase": "extract", "file_i": 1, "file_n": 17})
    reporter.set_stage("whisper_alignment")
    reporter.update(0, 12, "Whisper chunk 1/12 (0-120s)")
    status = json.loads(reporter.status_path.read_text())
    assert status["fraction"] == 0
    reporter.update(11, 12, "Whisper chunk 12/12 (1265-1287s)")
    reporter.update(12, 12, "Whisper transcription complete")
    status = json.loads(reporter.status_path.read_text())
    assert (status["file_i"], status["file_n"]) == (1, 17)
    assert status["fraction"] == pytest.approx(1 / 17, abs=1e-6)
    assert status["eta_s"] > 0
    reporter.update(1, 17, "Extracting second.flac", {"phase": "extract", "file_i": 2, "file_n": 17})
    reporter.set_stage("whisper_alignment")
    reporter.update(1, 8, "Whisper chunk 2/8 (115-235s)")
    status = json.loads(reporter.status_path.read_text())
    assert status["fraction"] == pytest.approx(1.125 / 17, abs=1e-6)
    assert status["fraction"] < 1
    reporter.mark_finished("complete", "Prepared all recordings")
    assert json.loads(reporter.status_path.read_text())["fraction"] == 1


def test_cancellation_does_not_claim_all_files_completed(tmp_path, monkeypatch):
    monkeypatch.setattr(WorkerReporter, "_vram", staticmethod(lambda: (0.0, 0.0)))
    reporter = WorkerReporter(tmp_path)
    reporter.update(2, 17, "Writing segments", {"phase": "segments", "file_i": 3, "file_n": 17})
    before = json.loads(reporter.status_path.read_text())["fraction"]
    reporter.update(17, 17, "cancelled", {"phase": "cancelled", "file_i": 3, "file_n": 17})
    assert json.loads(reporter.status_path.read_text())["fraction"] == before
