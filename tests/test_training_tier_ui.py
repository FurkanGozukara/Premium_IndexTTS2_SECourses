from types import SimpleNamespace

from ui.app import build_app


def test_training_tier_defaults_use_selected_device_and_set_both_precisions(monkeypatch):
    monkeypatch.setattr(
        "ui.training_tab._gpu_total",
        lambda device: {"cuda:0": 32.0, "cuda:1": 8.0, "cpu": 0.0}[device],
    )
    demo = build_app(SimpleNamespace(
        model_dir="models", device="cpu", verbose=False, no_browser=True,
        port=7861, host="127.0.0.1", share=False,
    ))
    training = demo.ui_tabs["LoRA / DoRA Training"]
    models = demo.ui_tabs["Models & Performance"]
    callback = next(
        fn for fn in demo.fns.values()
        if getattr(fn.fn, "__name__", "") == "apply_tier"
        and training.base_dtype in fn.outputs
    )

    assert callback.inputs == [models.tier, models.device]
    assert callback.outputs == [
        training.base_variant, training.base_dtype, training.mixed_precision,
        training.blocks_to_swap, training.swap_ring_size, training.pin_swap_memory,
    ]
    for tier in ("auto", "custom"):
        large = callback.fn(tier, "cuda:0")
        small = callback.fn(tier, "cuda:1")
        assert large == ("bf16", "bf16", "bf16", 0, 2, True)
        assert small == ("int8_convrot", "bf16", "bf16", 8, 2, True)
        assert len(small) == len(callback.outputs)
    assert callback.fn("32", "cuda:1") == ("bf16", "bf16", "bf16", 0, 2, True)
    assert callback.fn("auto", "cpu") == ("bf16", "fp32", "fp32", 0, 2, False)
