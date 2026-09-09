"""GPU VRAM tier presets: naming, detection, preset contents and app wiring."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from indextts.runtime.vram_presets import RuntimeConfig, VRAM_TIERS
from indextts.training.sampling import resolve_sample_runtime
from indextts.training.train_config import TrainConfig
from ui import gpu_tier_presets as tiers
from ui.app import build_app, overlay_persisted_runtime
from ui.presets_store import PresetRegistry, SYSTEM_PREFIX
from ui.training_tab import train_config_from_values


def test_tier_preset_names_round_trip():
    assert tiers.tier_preset_name(8) == "8 GB GPU"
    assert tiers.TIER_PRESET_NAMES[32] == "32 GB GPU"
    assert tiers.tier_from_preset_name("★ 12 GB GPU") == 12
    assert tiers.tier_from_preset_name("12 gb gpu") == 12
    assert tiers.tier_from_preset_name("9 GB GPU") is None
    assert tiers.tier_from_preset_name("my voice") is None
    assert tiers.tier_from_preset_name("") is None


def test_detection_counts_from_half_a_gigabyte_below_each_tier(monkeypatch):
    assert tiers.detect_gpu_tier(31.82) == 32
    assert tiers.detect_gpu_tier(31.5) == 32
    assert tiers.detect_gpu_tier(31.49) == 24
    assert tiers.detect_gpu_tier(23.99) == 24
    assert tiers.detect_gpu_tier(9.5) == 10
    assert tiers.detect_gpu_tier(9.49) == 8
    assert tiers.detect_gpu_tier(0.0) == 6
    monkeypatch.setattr("indextts.runtime.gpu.list_gpus", lambda: [])
    assert tiers.detected_gpu_total_gb() == 0.0
    assert tiers.detect_gpu_tier() == 6
    monkeypatch.setattr(
        "indextts.runtime.gpu.list_gpus",
        lambda: [SimpleNamespace(index=0, total_gb=11.99, is_default=False), SimpleNamespace(index=1, total_gb=7.9, is_default=True)],
    )
    assert tiers.detected_gpu_total_gb() == 7.9
    assert tiers.detect_gpu_tier() == 8


def test_tier_overrides_cover_runtime_generation_and_training_but_not_voice_choices():
    values = tiers.tier_registry_overrides(8)
    assert values["runtime.vram_tier"] == "8"
    assert values["runtime.model_variant"] == "bf16"
    assert values["runtime.blocks_to_swap"] == 0
    assert values["runtime.vram_reserve_gb"] == 1.0
    assert values["runtime.aux_residency.semantic_model"] == "cpu"
    assert values["generation.num_beams"] == 4
    assert values["generation.low_memory_mode"] is True
    assert values["generation.diffusion_steps"] == 40
    assert values["grid.num_beams"] == 4
    assert values["training.vram_tier"] == "8"
    assert values["training.blocks_to_swap"] == 0
    assert tiers.tier_registry_overrides(6)["training.blocks_to_swap"] == 22
    # Training samples and the speech benchmark keep their measured defaults.
    assert "training.sample_num_beams" not in values
    assert "training.sample_diffusion_steps" not in values
    assert tiers.tier_registry_overrides(32)["generation.diffusion_steps"] == 50
    assert tiers.tier_registry_overrides(6)["generation.num_beams"] == 2
    for key in ("runtime.device", "runtime.lora_path", "runtime.lora_strength", "runtime.decoder_adapter",
                "runtime.use_qwen_emo", "runtime.attention_backend"):
        assert key not in values
    assert tiers.tier_registry_overrides(32)["runtime.aux_residency.qwen_emo"] == "gpu"
    assert tiers.tier_registry_overrides(32)["runtime.vram_reserve_gb"] == 2.0
    assert tiers.tier_registry_overrides(6)["runtime.aux_residency.campplus"] == "cpu"


def test_sample_runtime_follows_the_training_tier_and_shrinks_beside_a_running_model(monkeypatch):
    monkeypatch.setattr("indextts.training.sampling.gpu_total_gb", lambda index: 31.8)
    config = TrainConfig(dataset_dir="dataset", name="adapter", vram_tier="12").validate()
    standalone = resolve_sample_runtime(config, share_gpu=False, free_gb=30.0)
    assert standalone.vram_tier == "12"
    assert standalone.device == "cuda:0"
    shared = resolve_sample_runtime(config, share_gpu=True, free_gb=7.4)
    assert shared.vram_tier == "8"
    assert resolve_sample_runtime(config, share_gpu=True, free_gb=3.0).vram_tier == "6"
    explicit = TrainConfig(dataset_dir="dataset", name="adapter", vram_tier="32", sample_runtime_tier="10").validate()
    assert resolve_sample_runtime(explicit, share_gpu=False, free_gb=30.0).vram_tier == "10"
    detected = TrainConfig(dataset_dir="dataset", name="adapter").validate()
    assert resolve_sample_runtime(detected, share_gpu=False, free_gb=30.0).vram_tier == "32"


def test_persisted_runtime_only_overlays_a_preset_of_its_own_tier():
    registry = PresetRegistry()
    registry.register("runtime.vram_tier", default="auto")
    registry.register("runtime.blocks_to_swap", default=0, kind="int", minimum=-1, maximum=24)
    preset_8 = {"runtime.vram_tier": "8", "runtime.blocks_to_swap": 12}
    preset_32 = {"runtime.vram_tier": "32", "runtime.blocks_to_swap": 0}
    persisted = RuntimeConfig(vram_tier="32", blocks_to_swap=3)
    assert overlay_persisted_runtime(registry, preset_32, persisted, system_preset=True)["runtime.blocks_to_swap"] == 3
    assert overlay_persisted_runtime(registry, preset_8, persisted, system_preset=True)["runtime.blocks_to_swap"] == 12
    assert overlay_persisted_runtime(registry, preset_8, persisted, system_preset=False)["runtime.blocks_to_swap"] == 12
    auto_persisted = RuntimeConfig(vram_tier="auto", blocks_to_swap=5)
    assert overlay_persisted_runtime(registry, preset_32, auto_persisted, system_preset=True, detected_tier=32)["runtime.blocks_to_swap"] == 5
    assert overlay_persisted_runtime(registry, preset_8, auto_persisted, system_preset=True, detected_tier=32)["runtime.blocks_to_swap"] == 12
    custom = RuntimeConfig(vram_tier="custom", blocks_to_swap=9)
    assert overlay_persisted_runtime(registry, preset_8, custom, system_preset=True)["runtime.blocks_to_swap"] == 9


@pytest.fixture(scope="module")
def demo():
    return build_app(SimpleNamespace(
        model_dir="models", device="cpu", verbose=False, no_browser=True,
        port=7861, host="127.0.0.1", share=False,
    ))


def test_app_lists_the_tier_presets_first_and_builds_training_from_them(demo):
    store = demo.preset_store
    names = store.list_presets()
    assert names[:7] == [SYSTEM_PREFIX + tiers.tier_preset_name(tier) for tier in VRAM_TIERS]
    assert not any(name.removeprefix(SYSTEM_PREFIX) in tiers.LEGACY_SYSTEM_PRESETS for name in names)
    dropdown = next(
        component for component in demo.config["components"]
        if component["type"] == "dropdown" and component["props"].get("label") == "Universal preset"
    )
    listed = [choice[1] if isinstance(choice, (list, tuple)) else choice for choice in dropdown["props"]["choices"]]
    assert listed[:7] == names[:7]
    assert store.default_preset_name() in {tiers.tier_preset_name(tier) for tier in VRAM_TIERS}

    values = store.load("6 GB GPU")
    config = train_config_from_values(values)
    assert config.vram_tier == "6"
    assert config.blocks_to_swap == 22 and config.swap_ring_size == 1
    assert config.base_variant == "bf16" and config.gradient_checkpointing is True
    assert config.sample_min_free_vram_gb == 4.5
    assert config.rank == 128 and config.batch_size == 1
    assert store.load("32 GB GPU")["training.vram_tier"] == "32"

    user_values = dict(store.load("32 GB GPU"))
    user_values["training.vram_tier"] = "8"
    user_values["training.blocks_to_swap"] = 12
    store.save("tier-roundtrip", user_values)
    try:
        restored = store.load("tier-roundtrip")
        assert restored["training.vram_tier"] == "8"
        assert restored["training.blocks_to_swap"] == 12
    finally:
        store.delete("tier-roundtrip")
        store.load(store.default_preset_name())
