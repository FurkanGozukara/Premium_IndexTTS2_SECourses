from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading

import pytest

from ui.presets_store import PresetRegistry, PresetStore, SYSTEM_PREFIX


def make_store(tmp_path: Path) -> PresetStore:
    registry = PresetRegistry()
    registry.register("generation.count", default=3, kind="int", minimum=1, maximum=8)
    registry.register("generation.enabled", default=True, kind="bool")
    registry.register("generation.mode", default="quality", kind="choice", choices=["quality", "fast"])
    registry.register("generation.text", default="hello", kind="str")
    return PresetStore(registry, tmp_path / "presets")


def test_save_load_delete_and_system_protection(tmp_path: Path):
    store = make_store(tmp_path)
    store.ensure_system_presets()
    assert store.list_presets()[0] == SYSTEM_PREFIX + "default"
    store.save(
        "my voice",
        {
            "generation.count": "999",
            "generation.enabled": "off",
            "generation.mode": "missing",
            "generation.text": 42,
            "future.unknown": "ignored",
        },
    )
    loaded = store.load("my voice")
    assert loaded == {
        "generation.count": 8,
        "generation.enabled": False,
        "generation.mode": "quality",
        "generation.text": "42",
    }
    assert store.get_last_used() == "my voice"
    assert store.delete("my voice")
    with pytest.raises(PermissionError):
        store.delete("default")
    with pytest.raises(PermissionError):
        store.save("default", {})


def test_default_regeneration_is_byte_identical(tmp_path: Path):
    store = make_store(tmp_path)
    store.ensure_system_presets()
    path = store.system_dir / "default.json"
    first = path.read_bytes()
    store.ensure_system_presets()
    assert path.read_bytes() == first


def test_legacy_nested_preset_migration(tmp_path: Path):
    registry = PresetRegistry()
    registry.register("generation.language", default="EN", kind="choice", choices=["EN", "ES"])
    registry.register("generation.section_batch_size", default=1, kind="int", minimum=1, maximum=8)
    registry.register("generation.top_p", default=0.8, kind="float", minimum=0, maximum=1)
    store = PresetStore(registry, tmp_path / "presets")
    payload = {
        "_meta": {"format": "indextts2_premium_ui"},
        "audio_generation": {"language": "ES", "autoregressive_batch_size": "4"},
        "advanced_parameters": {"top_p": "0.65", "unknown": 9},
    }
    (store.user_dir / "legacy.json").write_text(json.dumps(payload), encoding="utf-8")
    assert store.load("legacy") == {
        "generation.language": "ES",
        "generation.section_batch_size": 4,
        "generation.top_p": 0.65,
    }


def test_registry_rejects_duplicate_keys():
    registry = PresetRegistry()
    registry.register("same", default=1)
    with pytest.raises(ValueError, match="Duplicate"):
        registry.register("same", default=2)


def test_concurrent_last_used_updates_are_atomic(tmp_path: Path):
    store = make_store(tmp_path)
    store.ensure_system_presets()
    names = ("default", "quality", "fast", "low_vram_8gb")
    barrier = threading.Barrier(16)

    def update(worker: int) -> None:
        barrier.wait()
        for offset in range(12):
            assert store.set_last_used(names[(worker + offset) % len(names)])

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(update, range(16)))

    persisted = store.last_used_path.read_text(encoding="utf-8").strip()
    assert persisted in names
    assert store.get_last_used() in names
    assert not list(store.user_dir.glob(".*.tmp"))


def test_loading_survives_last_used_bookmark_failure(tmp_path: Path, monkeypatch):
    store = make_store(tmp_path)
    store.ensure_system_presets()
    original_write = store._write_atomic

    def fail_bookmark(path: Path, text: str) -> None:
        if path == store.last_used_path:
            raise PermissionError("simulated sharing violation")
        original_write(path, text)

    monkeypatch.setattr(store, "_write_atomic", fail_bookmark)
    assert store.load("quality") == store.registry.defaults()
    assert store.get_last_used() == "quality"
