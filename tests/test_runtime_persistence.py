from dataclasses import fields

from indextts.runtime.vram_presets import RuntimeConfig
from ui.app import overlay_persisted_runtime
from ui.models_tab import (
    load_persisted_runtime,
    persist_runtime_config,
    runtime_registry_values,
)
from ui.presets_store import PresetRegistry, PresetStore


def _runtime_registry() -> PresetRegistry:
    registry = PresetRegistry()
    defaults = RuntimeConfig(device="auto").to_dict()
    for item in fields(RuntimeConfig):
        if item.name == "aux_residency":
            for name, value in defaults["aux_residency"].items():
                registry.register(f"runtime.aux_residency.{name}", default=value)
        else:
            registry.register(f"runtime.{item.name}", default=defaults[item.name])
    return registry


def test_applied_runtime_round_trips_and_only_overlays_system_presets(tmp_path):
    path = tmp_path / ".last_runtime.json"
    expected = RuntimeConfig(
        device="cpu",
        gpt_dtype="fp32",
        blocks_to_swap=7,
        lora_strength=0.65,
        lora_merge_into_base=True,
    )
    persist_runtime_config(expected, path)
    restored = load_persisted_runtime(path)
    assert restored is not None
    assert restored.to_dict() == expected.to_dict()

    registry = _runtime_registry()
    preset = registry.defaults()
    system_values = overlay_persisted_runtime(
        registry, preset, restored, system_preset=True
    )
    user_values = overlay_persisted_runtime(
        registry, preset, restored, system_preset=False
    )
    assert system_values["runtime.blocks_to_swap"] == 7
    assert system_values["runtime.lora_merge_into_base"] is True
    assert user_values == preset
    assert runtime_registry_values(restored)["runtime.gpt_dtype"] == "fp32"

    store = PresetStore(registry, tmp_path / "presets")
    persist_runtime_config(expected, store.user_dir / ".last_runtime.json")
    assert ".last_runtime" not in store.list_presets()
