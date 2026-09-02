import io
import json

from indextts.runtime.progress import ProgressReporter, format_duration, format_rate, read_progress_file


class PlainStream(io.StringIO):
    def isatty(self):
        return False


def test_format_helpers():
    assert format_duration(0) == "0s"
    assert format_duration(65) == "1m 05s"
    assert format_duration(3661) == "1h 01m 01s"
    assert format_rate(3.125, "x RT") == "3.12x RT"


def test_progress_file_and_final_console(monkeypatch, tmp_path):
    stream = PlainStream()
    monkeypatch.setattr("sys.stdout", stream)
    path = tmp_path / "progress.json"
    reporter = ProgressReporter("segments", total=2, progress_file=path)
    reporter.set_stage("synthesis")
    reporter.update(1, desc="first", extra={"audio_seconds": 2.0})
    reporter.update(2, desc="done", extra={"audio_seconds": 4.0})
    payload = read_progress_file(path)
    assert payload is not None
    assert payload["fraction"] == 1.0
    assert payload["stage"] == "synthesis"
    assert payload["speed_unit"] == "x RT"
    assert "100.0%" in stream.getvalue()
    assert json.loads(path.read_text(encoding="utf-8"))["completed"] == 2


def test_read_progress_file_tolerates_partial_json(tmp_path):
    path = tmp_path / "partial.json"
    path.write_text("{", encoding="utf-8")
    assert read_progress_file(path) is None
