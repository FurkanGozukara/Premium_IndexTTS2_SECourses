"""Start training caches an IndexTTS dataset's features first instead of failing (OmniVoice and AuK cache themselves)."""

import json
from pathlib import Path
from types import SimpleNamespace

from indextts.training import trainer
from indextts.training.train_config import TrainConfig


def _config(tmp_path: Path) -> TrainConfig:
    dataset = tmp_path / "voice"
    dataset.mkdir()
    (dataset / "manifest.jsonl").write_text("{}\n", encoding="utf-8")
    return TrainConfig.from_dict({"dataset_dir": str(dataset), "name": "voice", "device": "cpu", "epochs": 2})


def test_a_dataset_without_features_is_cached_before_training(tmp_path, monkeypatch):
    import indextts.training.features as features

    calls = []

    def fake_cache(config, reporter=None, cancel_callback=None):
        calls.append(config)
        cache = Path(config.dataset_dir) / "cache"
        cache.mkdir()
        (cache / "index.jsonl").write_text("", encoding="utf-8")
        return SimpleNamespace(cancelled=False)

    monkeypatch.setattr(features, "cache_dataset_features", fake_cache)
    config = _config(tmp_path)
    state = tmp_path / "run"
    state.mkdir()
    assert trainer.cache_features_before_training(config, state)
    assert calls and calls[0].dataset_dir == str(Path(config.dataset_dir).resolve())
    assert json.loads((state / "status.json").read_text(encoding="utf-8"))["phase"] == "caching"
    assert trainer.cache_features_before_training(config, state)  # cached now: nothing runs again
    assert len(calls) == 1


def test_stop_during_caching_ends_the_run_as_stopped(tmp_path, monkeypatch):
    import indextts.training.features as features

    monkeypatch.setattr(features, "cache_dataset_features", lambda *a, **k: SimpleNamespace(cancelled=True))
    state = tmp_path / "run"
    state.mkdir()
    assert not trainer.cache_features_before_training(_config(tmp_path), state)
    assert json.loads((state / "status.json").read_text(encoding="utf-8"))["phase"] == "stopped"
