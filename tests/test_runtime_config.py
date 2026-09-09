from indextts.runtime.vram_presets import (
    RuntimeConfig,
    TIER_BUDGET_GB,
    VRAM_TIERS,
    auto_tier,
    estimate_vram_gb,
    fit_tier_to_free_vram,
    generation_hints,
    generation_preset,
    preset_notes,
    resolve_preset,
    resolve_training_preset,
    tier_budget_gb,
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
        "use_qwen_emo": "false",
        "use_deepspeed": "yes",
    })
    assert config.gpt_dtype == "bf16"
    assert config.use_cuda_kernel_bigvgan is True
    assert config.swap_ring_size == 4
    assert config.blocks_to_swap == -1
    assert config.attention_backend == "sdpa"
    assert config.lora_strength == 4.0
    assert config.use_qwen_emo is False
    assert config.use_deepspeed is True


def test_binding_preset_table():
    # Every tier keeps the BF16 GPT; smaller cards move auxiliary models and
    # stream blocks before any decoding setting is reduced.
    expected = {
        32: ("bf16", 0, "gpu", "gpu", 8192, 2.0),
        24: ("bf16", 0, "gpu", "gpu", 8192, 2.0),
        16: ("bf16", 0, "gpu", "gpu", 8192, 1.0),
        12: ("bf16", 0, "gpu", "gpu", 8192, 1.0),
        10: ("bf16", 0, "on_demand", "on_demand", 6144, 1.0),
        8: ("bf16", 0, "cpu", "on_demand", 4096, 1.0),
        6: ("bf16", 22, "cpu", "cpu", 2048, 1.0),
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
            config.vram_reserve_gb,
        ) == row
        assert generation_hints(tier)["section_batch_size_max"] == config.max_section_batch_size_hint
        assert tier_budget_gb(tier) == tier - config.vram_reserve_gb
        if tier == 6:
            assert config.aux_residency["campplus"] == "cpu"
            assert config.swap_ring_size == 1
            assert config.s2mel_estimator_autocast is True


def test_tier_budgets_and_free_vram_fit():
    assert TIER_BUDGET_GB == {6: 5.0, 8: 7.0, 10: 9.0, 12: 11.0, 16: 15.0, 24: 22.0, 32: 30.0}
    # A card counts as a tier from 500 MB below its nominal size.
    assert auto_tier(31.5) == 32 and auto_tier(31.49) == 24
    assert auto_tier(9.5) == 10 and auto_tier(9.49) == 8
    assert auto_tier(23.99) == 24 and auto_tier(15.9) == 16 and auto_tier(11.99) == 12
    assert auto_tier(7.5) == 8 and auto_tier(5.5) == 6 and auto_tier(4.0) == 6
    # A second process beside a running job shrinks to the tier whose budget fits.
    assert fit_tier_to_free_vram("32", 30.3) == 32
    assert fit_tier_to_free_vram("32", 10.5) == 10
    assert fit_tier_to_free_vram("12", 7.4) == 8
    assert fit_tier_to_free_vram("12", 6.5) == 6
    assert fit_tier_to_free_vram("8", 3.0) == 6
    assert fit_tier_to_free_vram("6", 100.0) == 6


def test_generation_and_training_presets_keep_quality_first():
    for tier in VRAM_TIERS:
        generation = generation_preset(tier)
        training = resolve_training_preset(tier)
        assert generation["diffusion_steps"] == (50 if tier >= 24 else 40)
        assert generation["cfm_temperature"] == 0.9
        assert generation["section_batch_size"] == 1
        assert generation["max_text_tokens_per_segment"] == 60
        assert generation["cfm_cache_length"] == resolve_preset(tier, float(tier)).cfm_cache_length
        assert training["base_variant"] == "bf16"
        assert training["gradient_checkpointing"] is True
        assert training["vram_tier"] == str(tier)
        assert training["sample_runtime_tier"] == "auto"
    assert [generation_preset(tier)["num_beams"] for tier in VRAM_TIERS] == [2, 4, 4, 4, 4, 4, 4]
    assert [generation_preset(tier)["low_memory_mode"] for tier in VRAM_TIERS] == [True, True, False, False, False, False, False]
    assert [resolve_training_preset(tier)["blocks_to_swap"] for tier in VRAM_TIERS] == [22, 0, 0, 0, 0, 0, 0]
    assert resolve_training_preset("auto", 9.9)["vram_tier"] == "10"
    assert "GB" in preset_notes(8) and "beams" in preset_notes(8)


def test_auto_tier_and_estimate_are_sane():
    assert auto_tier(5) == 6
    assert auto_tier(11.9) == 12
    assert auto_tier(31.8) == 32
    estimate = estimate_vram_gb(resolve_preset("12", 12), 12)
    assert estimate["estimated_peak_gb"] > estimate["resident_weights_gb"]
    assert isinstance(estimate["fits"], bool)
