"""Concurrency safety of the shared atomic JSON helpers (status/progress files polled by the UI)."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from indextts.runtime.progress import ProgressReporter, read_progress_file
from indextts.training.dataset_manifest import atomic_write_json
from indextts.utils.atomic_json import read_json_retry, write_json_atomic


def test_roundtrip_and_no_temp_leftovers(tmp_path: Path) -> None:
    target = tmp_path / "status.json"
    write_json_atomic(target, {"phase": "running", "step": 3})
    assert json.loads(target.read_text(encoding="utf-8")) == {"phase": "running", "step": 3}
    assert read_json_retry(target) == {"phase": "running", "step": 3}
    assert [p.name for p in tmp_path.iterdir()] == ["status.json"]


def test_writer_survives_reader_holding_the_file(tmp_path: Path) -> None:
    """On Windows ``os.replace`` fails while another handle keeps the destination open; the writer must retry."""

    target = tmp_path / "status.json"
    write_json_atomic(target, {"step": 0})
    errors: list[BaseException] = []

    def writer() -> None:
        try:
            for step in range(1, 6):
                write_json_atomic(target, {"step": step})
        except BaseException as exc:  # pragma: no cover - the assertion below reports it
            errors.append(exc)

    with target.open("r", encoding="utf-8") as held:
        thread = threading.Thread(target=writer)
        thread.start()
        time.sleep(0.15)
        held.read()
    thread.join(timeout=15)
    assert not thread.is_alive()
    assert errors == []
    assert read_json_retry(target) == {"step": 5}
    assert [p.name for p in tmp_path.iterdir()] == ["status.json"]


def test_read_retries_partial_file(tmp_path: Path) -> None:
    target = tmp_path / "progress.json"
    target.write_text('{"fraction": 0.', encoding="utf-8")

    def finish() -> None:
        time.sleep(0.05)
        write_json_atomic(target, {"fraction": 0.5}, indent=None, trailing_newline=False)

    threading.Thread(target=finish).start()
    assert read_json_retry(target, attempts=20, delay=0.02) == {"fraction": 0.5}
    assert read_progress_file(target) == {"fraction": 0.5}


def test_read_missing_file_returns_default(tmp_path: Path) -> None:
    assert read_json_retry(tmp_path / "missing.json", {"x": 1}) == {"x": 1}
    assert read_progress_file(tmp_path / "missing.json") is None


def test_manifest_and_progress_writers_use_helper(tmp_path: Path) -> None:
    status = tmp_path / "status.json"
    atomic_write_json(status, {"phase": "complete", "path": Path("a/b")})
    assert read_json_retry(status)["path"] == str(Path("a/b"))
    progress = tmp_path / "progress.json"
    reporter = ProgressReporter("items", total=4, progress_file=str(progress))
    reporter.update(2)
    value = read_progress_file(progress)
    assert value is not None and value.get("completed") == 2
    assert [p.name for p in tmp_path.iterdir()] == sorted(["status.json", "progress.json"])


@pytest.mark.parametrize("payload", [{"nan": float("nan")}])
def test_progress_writer_rejects_nan_like_before(tmp_path: Path, payload: dict) -> None:
    with pytest.raises(ValueError):
        write_json_atomic(tmp_path / "x.json", payload, allow_nan=False)
