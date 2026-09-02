from indextts.runtime.vram_presets import (
    RuntimeConfig,
    VRAM_TIERS,
    auto_tier,
    estimate_vram_gb,
    generation_hints,
    resolve_preset,
)


def test_runtime_config_round_trip_and_extra_keys():
    original = RuntimeConfig(blocks_to_swap=8, aux_residency={"semantic_model": "on_demand"})
    payload = original.to_dict()
    payload["future_option"] = "ignored"
    restored = RuntimeConfig.from_dict(payload)
    assert restored.to_dict() == original.to_dict()
    assert set(restored.aux_residency) == {
        "semantic_model", "qwen_emo", "campplus", "semantic_codec", "s2mel", "bigvgan"
    }


def test_runtime_config_legacy_and_validation():
    config = RuntimeConfig.from_dict({
        "use_bf16": True,
        "use_cuda_kernel": "yes",
        "block_swap_ring_size": 99,
        "blocks_to_swap": -99,
        "attention_backend": "unknown",
        "lora_strength": 999,
    })
    assert config.gpt_dtype == "bf16"
    assert config.use_cuda_kernel_bigvgan is True
    assert config.swap_ring_size == 4
    assert config.blocks_to_swap == -1
    assert config.attention_backend == "sdpa"
    assert config.lora_strength == 4.0


def test_binding_preset_table():
    expected = {
        32: ("bf16", 0, "gpu", "gpu", 8192),
        24: ("bf16", 0, "gpu", "gpu", 8192),
        16: ("bf16", 0, "gpu", "on_demand", 8192),
        12: ("bf16", 0, "on_demand", "on_demand", 8192),
        10: ("bf16", 8, "on_demand", "on_demand", 6144),
        8: ("int8_convrot", 8, "on_demand", "on_demand", 4096),
        6: ("int8_convrot", 22, "cpu", "cpu", 2048),
    }
    assert VRAM_TIERS == [6, 8, 10, 12, 16, 24, 32]
    for tier, row in expected.items():
        config = resolve_preset(str(tier), float(tier), float(tier))
        assert (
            config.model_variant,
            config.blocks_to_swap,
            config.aux_residency["semantic_model"],
            config.aux_residency["qwen_emo"],
            config.cfm_cache_length,
        ) == row
        assert generation_hints(tier)["section_batch_size_max"] == config.max_section_batch_size_hint
        if tier == 6:
            assert config.aux_residency["campplus"] == "cpu"
            assert config.swap_ring_size == 1
            assert config.s2mel_estimator_autocast is True


def test_auto_tier_and_estimate_are_sane():
    assert auto_tier(5) == 6
    assert auto_tier(11.9) == 12
    assert auto_tier(31.8) == 32
    estimate = estimate_vram_gb(resolve_preset("12", 12), 12)
    assert estimate["estimated_peak_gb"] > estimate["resident_weights_gb"]
    assert isinstance(estimate["fits"], bool)
