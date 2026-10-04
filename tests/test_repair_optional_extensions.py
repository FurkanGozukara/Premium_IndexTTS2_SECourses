import importlib.util
from importlib import metadata
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "repair_optional_extensions", Path(__file__).resolve().parents[1] / "tools" / "repair_optional_extensions.py")
repair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repair)


def _versions(monkeypatch, versions):
    def version(name):
        if name not in versions:
            raise metadata.PackageNotFoundError(name)
        return versions[name]
    monkeypatch.setattr(repair.metadata, "version", version)


def test_a_companion_from_another_cuda_index_is_reported(monkeypatch):
    _versions(monkeypatch, {"torch": "2.14.1+cu130", "torchaudio": "2.11.0+cu128", "torchvision": "0.29.1+cu130"})
    monkeypatch.setattr(repair, "_imports", lambda module: True)
    assert repair.mismatched_companions() == ["torchaudio"]


def test_matching_companions_that_import_are_left_alone(monkeypatch):
    _versions(monkeypatch, {"torch": "2.14.1+cu130", "torchaudio": "2.11.0+cu130", "torchvision": "0.29.1+cu130"})
    monkeypatch.setattr(repair, "_imports", lambda module: True)
    assert repair.mismatched_companions() == []


def test_a_matching_build_that_fails_to_import_is_reported(monkeypatch):
    _versions(monkeypatch, {"torch": "2.14.1+cu130", "torchaudio": "2.11.0+cu130"})
    monkeypatch.setattr(repair, "_imports", lambda module: module != "torchaudio")
    assert repair.mismatched_companions() == ["torchaudio"]


def test_reinstall_pins_torch_and_uses_its_cuda_index(monkeypatch):
    _versions(monkeypatch, {"torch": "2.14.1+cu130"})
    calls = []
    monkeypatch.setattr(repair.subprocess, "run", lambda command, **kwargs: calls.append(command))
    monkeypatch.setattr(repair, "_imports", lambda module: True)
    repair.reinstall_companions(["torchaudio"])
    command = calls[0]
    assert command[1:4] == ["-m", "uv", "pip"]
    assert "--no-sources" in command
    assert "torch==2.14.1+cu130" in command
    assert command[command.index("--extra-index-url") + 1] == "https://download.pytorch.org/whl/cu130"
    assert command[command.index("--reinstall-package") + 1] == "torchaudio"


def test_reinstall_fails_loudly_when_the_import_still_breaks(monkeypatch):
    _versions(monkeypatch, {"torch": "2.14.1+cu130"})
    monkeypatch.setattr(repair.subprocess, "run", lambda command, **kwargs: None)
    monkeypatch.setattr(repair, "_imports", lambda module: False)
    with pytest.raises(SystemExit):
        repair.reinstall_companions(["torchaudio"])
