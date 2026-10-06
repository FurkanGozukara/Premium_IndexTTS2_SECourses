"""The dataset tab's Cache features now runs this tool with the speech model chosen in the header."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "cache_dataset_features.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("cache_dataset_features_tool", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("model", ["indextts", "omnivoice", "auk"])
def test_each_speech_model_caches_with_its_own_features(model, tmp_path: Path, monkeypatch) -> None:
    from indextts.training import auk_data, omnivoice_data

    tool = _load_tool()
    calls: list[str] = []

    def fake(name):
        def cache(config, reporter=None):
            calls.append(name)
            assert config.dataset_dir == str(tmp_path)
            return SimpleNamespace(cancelled=False, to_dict=lambda: {"model": name})
        return cache

    monkeypatch.setattr(tool, "cache_dataset_features", fake("indextts"))
    monkeypatch.setattr(omnivoice_data, "cache_omnivoice_features", fake("omnivoice"))
    monkeypatch.setattr(auk_data, "cache_auk_features", fake("auk"))
    monkeypatch.setattr(sys, "argv", ["cache_dataset_features.py", "--dataset-dir", str(tmp_path),
                                      "--device", "cpu", "--tts-model", model])
    assert tool.main() == 0
    assert calls == [model]
