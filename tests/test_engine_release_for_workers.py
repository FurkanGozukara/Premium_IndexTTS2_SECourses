"""GPU child workers start with the in-process inference engine released."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import ui.common as common


class _Engine:
    def __init__(self) -> None:
        self.unloaded = False

    def unload(self) -> None:
        self.unloaded = True


def _loaded_engine() -> tuple[common.LazyEngine, _Engine]:
    lazy = common.LazyEngine()
    instance = _Engine()
    lazy._instance = instance
    lazy._fingerprint = "loaded"
    return lazy, instance


def test_idle_engine_is_released_before_a_gpu_worker(capsys):
    lazy, instance = _loaded_engine()
    assert lazy.release_for_worker("training") is True
    assert instance.unloaded is True
    assert lazy.peek() is None
    out = capsys.readouterr().out
    assert "Released the in-process inference models before the training worker" in out


def test_nothing_to_release_when_no_engine_is_loaded(capsys):
    lazy = common.LazyEngine()
    assert lazy.release_for_worker("dataset_prep") is False
    assert "Released" not in capsys.readouterr().out


def test_running_generation_keeps_its_engine(capsys):
    lazy, instance = _loaded_engine()
    with lazy.in_use():
        assert lazy.busy is True
        assert lazy.release_for_worker("dataset_cache") is False
        assert instance.unloaded is False
        assert lazy.peek() is instance
    assert lazy.busy is False
    assert "Keeping the in-process inference models loaded" in capsys.readouterr().out
    assert lazy.release_for_worker("dataset_cache") is True


def test_model_load_in_progress_keeps_the_engine_without_waiting(capsys):
    import threading

    lazy, instance = _loaded_engine()
    holding = threading.Event()
    release = threading.Event()

    def load():
        with lazy._lock:
            holding.set()
            release.wait(5)

    loader = threading.Thread(target=load)
    loader.start()
    try:
        assert holding.wait(2)
        assert lazy.release_for_worker("grid_generation") is False
        assert instance.unloaded is False
        assert "a model load is in progress" in capsys.readouterr().out
    finally:
        release.set()
        loader.join(5)
    assert lazy.release_for_worker("grid_generation") is True


def test_in_use_is_reentrant_and_never_goes_negative():
    lazy = common.LazyEngine()
    with lazy.in_use():
        with lazy.in_use():
            assert lazy.busy is True
        assert lazy.busy is True
    assert lazy.busy is False
    lazy._busy = 0
    with lazy.in_use():
        pass
    assert lazy._busy == 0


@pytest.mark.parametrize(
    "kind, expected",
    [
        ("training", True),
        ("dataset_prep", True),
        ("dataset_cache", True),
        ("dataset_curation", True),
        ("checkpoint_eval", True),
        ("grid_generation", True),
        ("vram_benchmark", True),
        ("generation", True),
        ("batch_generation", True),
        ("analysis", False),
    ],
)
def test_process_manager_releases_the_engine_only_for_gpu_workers(tmp_path, monkeypatch, kind, expected):
    calls: list[str] = []
    monkeypatch.setattr(common, "LAZY_ENGINE", SimpleNamespace(release_for_worker=lambda name: calls.append(name)))
    manager = common.ProcessManager()
    job = manager.start(kind, [sys.executable, "-c", "pass"], state_dir=tmp_path / kind)
    job.process.wait(timeout=60)
    assert calls == ([kind] if expected else [])


def test_explicit_release_flag_overrides_the_kind_policy(tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(common, "LAZY_ENGINE", SimpleNamespace(release_for_worker=lambda name: calls.append(name)))
    manager = common.ProcessManager()
    job = manager.start("analysis", [sys.executable, "-c", "pass"], state_dir=tmp_path / "a", release_engine=True)
    job.process.wait(timeout=60)
    job = manager.start("training", [sys.executable, "-c", "pass"], state_dir=tmp_path / "b", release_engine=False)
    job.process.wait(timeout=60)
    assert calls == ["analysis"]


def test_process_manager_start_releases_the_real_engine_and_the_child_runs(tmp_path, monkeypatch, capsys):
    lazy, instance = _loaded_engine()
    monkeypatch.setattr(common, "LAZY_ENGINE", lazy)
    manager = common.ProcessManager()
    job = manager.start("dataset_prep", [sys.executable, "-c", "print('child ran')"], state_dir=tmp_path / "prep")
    assert job.process.wait(timeout=60) == 0
    assert instance.unloaded is True
    assert lazy.peek() is None
    assert "Released the in-process inference models before the dataset prep worker" in capsys.readouterr().out


def test_every_gpu_worker_kind_used_by_the_tabs_is_covered():
    import re
    from pathlib import Path

    root = Path(common.__file__).resolve().parent
    kinds: set[str] = set()
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r'PROCESS_MANAGER\.start\(\s*"([a-z_]+)"', text):
            kinds.add(match.group(1))
    assert kinds, "expected worker launches in the UI modules"
    assert kinds <= common.GPU_WORKER_KINDS
