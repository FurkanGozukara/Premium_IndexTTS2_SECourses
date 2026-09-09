from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading

import pytest

from ui.gpu_tier_presets import LEGACY_SYSTEM_PRESETS, tier_preset_name
from ui.presets_store import PresetRegistry, PresetStore, SYSTEM_PREFIX


def make_store(tmp_path: Path, *, detect_tier=lambda: 12) -> PresetStore:
    registry = PresetRegistry()
    registry.register("generation.count", default=3, kind="int", minimum=1, maximum=8)
    registry.register("generation.enabled", default=True, kind="bool")
    registry.register("generation.mode", default="quality", kind="choice", choices=["quality", "fast"])
    registry.register("generation.text", default="hello", kind="str")
    registry.register("generation.num_beams", default=3, kind="int", minimum=1, maximum=10)
    return PresetStore(registry, tmp_path / "presets", detect_tier=detect_tier)


def test_save_load_delete_and_system_protection(tmp_path: Path):
    store = make_store(tmp_path)
    store.ensure_system_presets()
    assert store.list_presets()[0] == SYSTEM_PREFIX + "6 GB GPU"
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
        "generation.num_beams": 3,
    }
    assert store.get_last_used() == "my voice"
    assert store.delete("my voice")
    # Deleting the last-used user preset returns to the GPU's own tier preset.
    assert store.get_last_used() == "12 GB GPU"
    with pytest.raises(PermissionError):
        store.delete("12 GB GPU")
    with pytest.raises(PermissionError):
        store.save("★ 12 GB GPU", {})


def test_system_presets_are_the_seven_tiers_in_size_order(tmp_path: Path):
    store = make_store(tmp_path)
    store.ensure_system_presets()
    assert store._system_names() == [
        "6 GB GPU", "8 GB GPU", "10 GB GPU", "12 GB GPU", "16 GB GPU", "24 GB GPU", "32 GB GPU",
    ]
    # Tier presets carry the tier decoding settings on top of the registry defaults.
    assert store.load("32 GB GPU")["generation.num_beams"] == 4
    assert store.load("6 GB GPU")["generation.num_beams"] == 2
    for name in store._system_names():
        payload = json.loads((store.system_dir / f"{name}.json").read_text(encoding="utf-8"))
        assert payload["_meta"]["read_only"] is True
        assert payload["_meta"]["scope"] == "system"


def test_default_regeneration_is_byte_identical(tmp_path: Path):
    store = make_store(tmp_path)
    store.ensure_system_presets()
    path = store.system_dir / "12 GB GPU.json"
    first = path.read_bytes()
    store.ensure_system_presets()
    assert path.read_bytes() == first


def test_fresh_install_selects_the_detected_gpu_tier(tmp_path: Path):
    store = make_store(tmp_path, detect_tier=lambda: 8)

    assert store.default_preset_name() == "8 GB GPU"
    assert store.stored_last_used() is None
    assert store.get_last_used() == "8 GB GPU"
    store.ensure_system_presets()
    assert store.get_last_used() == "8 GB GPU"
    assert store.stored_last_used() is None
    # Loading it once records it as the last-used preset like any other.
    store.load("8 GB GPU")
    assert store.stored_last_used() == "8 GB GPU"


def test_machines_without_a_gpu_or_a_failing_probe_fall_back_to_the_smallest_tier(tmp_path: Path):
    def broken() -> int:
        raise RuntimeError("no driver")

    assert make_store(tmp_path / "a", detect_tier=broken).default_preset_name() == "6 GB GPU"
    assert make_store(tmp_path / "b", detect_tier=lambda: 99).default_preset_name() == "6 GB GPU"


def test_saved_user_preset_and_last_used_selection_are_kept(tmp_path: Path):
    store = make_store(tmp_path, detect_tier=lambda: 32)
    store.ensure_system_presets()
    store.save("studio", {"generation.count": 5})
    reopened = make_store(tmp_path, detect_tier=lambda: 32)
    assert reopened.get_last_used() == "studio"
    reopened.load("8 GB GPU")
    assert make_store(tmp_path, detect_tier=lambda: 32).get_last_used() == "8 GB GPU"


def test_legacy_system_presets_are_retired_and_their_bookmark_falls_back(tmp_path: Path):
    store = make_store(tmp_path, detect_tier=lambda: 24)
    for legacy in LEGACY_SYSTEM_PRESETS:
        (store.system_dir / f"{legacy}.json").write_text("{}", encoding="utf-8")
    store.last_used_path.write_text("quality\n", encoding="utf-8")
    assert store.retire_legacy_system_presets() == list(LEGACY_SYSTEM_PRESETS)
    assert store.stored_last_used() is None
    assert store.get_last_used() == "24 GB GPU"
    store.ensure_system_presets()
    assert not any((store.system_dir / f"{legacy}.json").exists() for legacy in LEGACY_SYSTEM_PRESETS)
    assert store.list_presets() == [SYSTEM_PREFIX + tier_preset_name(tier) for tier in (6, 8, 10, 12, 16, 24, 32)]


def test_reset_returns_the_detected_tier_preset(tmp_path: Path):
    store = make_store(tmp_path, detect_tier=lambda: 16)
    store.ensure_system_presets()
    store.save("mine", {"generation.count": 7})
    assert store.reset()["generation.count"] == 3
    assert store.get_last_used() == "16 GB GPU"


def test_legacy_nested_preset_migration(tmp_path: Path):
    registry = PresetRegistry()
    registry.register("generation.language", default="EN", kind="choice", choices=["EN", "ES"])
    registry.register("generation.section_batch_size", default=1, kind="int", minimum=1, maximum=8)
    registry.register("generation.top_p", default=0.8, kind="float", minimum=0, maximum=1)
    store = PresetStore(registry, tmp_path / "presets", detect_tier=lambda: 12)
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
    names = ("6 GB GPU", "8 GB GPU", "12 GB GPU", "32 GB GPU")
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
    assert store.load("12 GB GPU")["generation.count"] == 3
    assert store.get_last_used() == "12 GB GPU"


@pytest.mark.parametrize("spelling", ["32 GB GPU", "★ 32 GB GPU", "32 gb gpu", "32GB GPU", "  32 Gb Gpu  "])
def test_tier_presets_cannot_be_overwritten_or_deleted_in_any_spelling(tmp_path: Path, spelling: str):
    store = make_store(tmp_path)
    store.ensure_system_presets()
    assert store.is_system(spelling)
    assert store.canonical_name(spelling) == "32 GB GPU"
    with pytest.raises(PermissionError, match="read-only and cannot be overwritten"):
        store.save(spelling, {"generation.count": 1})
    with pytest.raises(PermissionError, match="read-only and cannot be deleted"):
        store.delete(spelling)
    assert not list(store.user_dir.glob("*.json"))
    # Loading the same spelling opens the tier preset and bookmarks its canonical name.
    assert store.load(spelling)["generation.num_beams"] == store.load("32 GB GPU")["generation.num_beams"]
    assert store.get_last_used() == "32 GB GPU"


def test_a_user_file_named_like_a_tier_never_shadows_the_system_preset(tmp_path: Path):
    store = make_store(tmp_path)
    store.ensure_system_presets()
    shadow = store.user_dir / "16 gb gpu.json"
    shadow.write_text(json.dumps({"values": {"generation.num_beams": 9}}), encoding="utf-8")
    assert "16 gb gpu" not in store.list_presets()
    assert store.list_presets().count(SYSTEM_PREFIX + "16 GB GPU") == 1
    assert store.load("16 gb gpu")["generation.num_beams"] != 9
    assert store.is_system("16 gb gpu")


def test_system_preset_files_are_restored_at_startup_after_a_manual_edit(tmp_path: Path):
    store = make_store(tmp_path)
    store.ensure_system_presets()
    path = store.system_dir / "24 GB GPU.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["values"]["generation.num_beams"] = 1
    payload["_meta"]["read_only"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    store.ensure_system_presets()
    restored = json.loads(path.read_text(encoding="utf-8"))
    assert restored["_meta"]["read_only"] is True
    assert restored["values"]["generation.num_beams"] == store.tier_preset_values(24)["generation.num_beams"]


def test_non_tier_numbers_and_other_names_stay_available_to_users(tmp_path: Path):
    store = make_store(tmp_path)
    store.ensure_system_presets()
    for name in ("40 GB GPU", "my 32 GB voice", "GPU 32"):
        assert not store.is_system(name)
        assert store.save(name, {"generation.count": 2}) == name
        assert name in store.list_presets()
        assert store.delete(name)
