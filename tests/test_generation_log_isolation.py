"""CPU-only concurrency checks for the Voice/Batch shared console tee."""

import contextlib
import io
import sys
import threading

import pytest

import ui.batch_tab as batch
import ui.generation_tab as generation


def _thread(target, errors, name="test-log-worker"):
    def checked():
        try:
            target()
        except BaseException as exc:
            errors.append(exc)
    worker = threading.Thread(target=checked, name=name, daemon=True)
    worker.start()
    return worker


def _join(worker, errors):
    worker.join(timeout=5)
    assert not worker.is_alive(), "Log operation deadlocked"
    assert not errors, repr(errors)


def test_batch_uses_the_generation_tee_class():
    assert batch._Tee is generation._Tee


def test_redirected_stdout_and_stderr_capture_only_worker_thread(tmp_path, monkeypatch):
    console, error_console = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", console)
    monkeypatch.setattr(sys, "stderr", error_console)
    ready, resume = threading.Event(), threading.Event()
    log_path = tmp_path / "generation.log"
    errors = []

    def owner():
        tee = generation._Tee(sys.stdout, log_path)
        try:
            # These are the same redirects used by Voice and Batch workers.
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                print("worker stdout", flush=True)
                print("worker stderr", file=sys.stderr, flush=True)
                ready.set()
                assert resume.wait(5)
                print("worker finished", flush=True)
        finally:
            tee.close()

    worker = _thread(owner, errors)
    try:
        assert ready.wait(5)
        print("foreign preparation stdout", flush=True)
        print("foreign preparation stderr", file=sys.stderr, flush=True)
    finally:
        resume.set()
        _join(worker, errors)
    assert log_path.read_text(encoding="utf-8") == "worker stdout\nworker stderr\nworker finished\n"
    # Existing behavior merges redirected stdout/stderr into the passed console;
    # console forwarding is not suppressed merely because a write is foreign.
    assert "foreign preparation stdout\n" in console.getvalue()
    assert "foreign preparation stderr\n" in console.getvalue()
    assert "worker stdout\n" in console.getvalue()
    assert "worker stderr\n" in console.getvalue()
    assert sys.stdout is console
    assert sys.stderr is error_console
    print("restored stderr", file=sys.stderr)
    assert error_console.getvalue() == "restored stderr\n"


def test_concurrent_foreign_writes_remain_console_only(tmp_path):
    console = io.StringIO()
    log_path = tmp_path / "generation.log"
    tee = generation._Tee(console, log_path)
    barrier = threading.Barrier(3)
    errors = []

    def foreign(label):
        barrier.wait(timeout=5)
        for index in range(50):
            value = f"{label}-{index}\n"
            assert tee.write(value) == len(value)
            tee.flush()

    workers = [_thread(lambda label=label: foreign(label), errors, name=label)
               for label in ("prep-pump", "other-worker")]
    try:
        barrier.wait(timeout=5)
        for index in range(50):
            value = f"owner-{index}\n"
            assert tee.write(value) == len(value)
        for worker in workers:
            _join(worker, errors)
    finally:
        tee.close()
    assert log_path.read_text().splitlines() == [f"owner-{index}" for index in range(50)]
    assert sorted(console.getvalue().splitlines()) == sorted(
        f"{label}-{index}" for label in ("owner", "prep-pump", "other-worker") for index in range(50))


def test_close_is_idempotent_and_late_owner_and_foreign_writes_still_forward(tmp_path):
    console = io.StringIO()
    log_path = tmp_path / "generation.log"
    tee = generation._Tee(console, log_path)
    tee.write("before close\n")
    tee.close()
    tee.close()
    assert tee.write("late owner\n") == len("late owner\n")
    tee.flush()
    errors = []

    def foreign():
        assert tee.write("late foreign\n") == len("late foreign\n")
        tee.flush()
        tee.close()

    _join(_thread(foreign, errors), errors)
    assert tee.handle.closed
    assert not console.closed
    assert log_path.read_text() == "before close\n"
    assert console.getvalue() == "before close\nlate owner\nlate foreign\n"


def test_close_waits_for_inflight_owner_write_without_closed_file_error(tmp_path):
    entered, release, close_started, close_done = (threading.Event() for _ in range(4))
    errors, box = [], {}
    log_path = tmp_path / "generation.log"

    class BlockingConsole(io.StringIO):
        def write(self, value):
            if value == "in-flight owner\n":
                entered.set()
                assert release.wait(5)
            return super().write(value)

    console = BlockingConsole()

    def owner():
        box["tee"] = generation._Tee(console, log_path)
        box["tee"].write("in-flight owner\n")

    def closer():
        close_started.set()
        box["tee"].close()
        close_done.set()

    worker = _thread(owner, errors)
    closer_thread = None
    try:
        assert entered.wait(5)
        closer_thread = _thread(closer, errors)
        assert close_started.wait(5)
        assert not close_done.wait(0.05), "Close raced past the active file writer"
    finally:
        release.set()
        _join(worker, errors)
        if closer_thread is not None:
            _join(closer_thread, errors)
    assert close_done.is_set()
    box["tee"].write("late pump write\n")
    box["tee"].flush()
    assert log_path.read_text() == "in-flight owner\n"
    assert console.getvalue() == "in-flight owner\nlate pump write\n"


def test_nested_tees_preserve_each_worker_ownership_and_console(tmp_path):
    console = io.StringIO()
    primary_path, secondary_path = tmp_path / "primary.log", tmp_path / "secondary.log"
    primary = generation._Tee(console, primary_path)
    ready, resume = threading.Event(), threading.Event()
    errors, box = [], {}

    def secondary_owner():
        secondary = generation._Tee(primary, secondary_path)
        box["tee"] = secondary
        try:
            secondary.write("secondary owner\n")
            ready.set()
            assert resume.wait(5)
        finally:
            secondary.close()

    worker = _thread(secondary_owner, errors)
    try:
        assert ready.wait(5)
        # A process-wide redirect may point at another worker's tee. The outer
        # tee still forwards to the primary owner without copying into its file.
        box["tee"].write("primary owner\n")
    finally:
        resume.set()
        _join(worker, errors)
        primary.close()
    assert primary_path.read_text() == "primary owner\n"
    assert secondary_path.read_text() == "secondary owner\n"
    assert console.getvalue() == "secondary owner\nprimary owner\n"
