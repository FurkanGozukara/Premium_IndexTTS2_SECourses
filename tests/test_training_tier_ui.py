from types import SimpleNamespace

from ui.app import build_app
from ui.training_tab import TRAINING_TIER_FIELDS, OMNI_CAPACITY_FIELDS, training_tier_values


def _demo():
    return build_app(SimpleNamespace(
        model_dir="models", device="cpu", verbose=False, no_browser=True,
        port=7861, host="127.0.0.1", share=False,
    ))


def test_training_tier_dropdown_sits_beside_the_dataset_and_fills_the_vram_controls(monkeypatch):
    monkeypatch.setattr(
        "ui.training_tab._gpu_total",
        lambda device: {"cuda:0": 32.0, "cuda:1": 8.0, "cpu": 0.0}.get(str(device), 0.0),
    )
    demo = _demo()
    training = demo.ui_tabs["LoRA / DoRA Training"]
    assert "training.vram_tier" in demo.preset_registry.keys
    assert demo.preset_registry["training.vram_tier"].component is training.vram_tier
    assert [value for _, value in training.vram_tier.choices] == ["auto", "6", "8", "10", "12", "16", "24", "32"]

    callbacks = [
        fn for fn in demo.fns.values()
        if getattr(fn.fn, "__name__", "") == "apply_training_tier"
    ]
    assert {tuple(callback.targets[0]) for callback in callbacks} == {
        (training.vram_tier._id, "select"), (training.apply_tier_button._id, "click"),
    }
    expected_outputs = [training.controls[f"training.{name}"] for name in (*TRAINING_TIER_FIELDS, *OMNI_CAPACITY_FIELDS)] + [training.tier_note]
    for callback in callbacks:
        assert callback.inputs == [training.vram_tier, training.device, training.model_selector, training.controls["training.adapter_type"]]
        assert callback.outputs == expected_outputs
        large = callback.fn("auto", "cuda:0")
        small = callback.fn("auto", "cuda:1")
        assert len(large) == len(expected_outputs)
        assert large[:8] == ("bf16", "bf16", "bf16", True, 0, 2, True, "auto")
        assert small[:8] == ("bf16", "bf16", "bf16", True, 0, 2, True, "auto")
        assert "32 GB" in large[-1] and "8 GB" in small[-1]
        assert callback.fn("6", "cuda:0")[:8] == ("bf16", "bf16", "bf16", True, 22, 1, True, "auto")
        assert callback.fn("auto", "cpu")[:7] == ("bf16", "fp32", "fp32", True, 0, 2, False)
        assert "CPU" in callback.fn("auto", "cpu")[-1]

    note_callbacks = [fn for fn in demo.fns.values() if getattr(fn.fn, "__name__", "") == "_training_tier_note"]
    triggers = {tuple(target) for callback in note_callbacks for target in callback.targets}
    # Tier, device and method changes refresh the note directly; a model switch or
    # preset load refreshes it as a chained step (the selector's own change event
    # already starts the Models tab's deferring description).
    assert {(training.vram_tier._id, "change"), (training.device._id, "change"),
            (training.controls["training.adapter_type"]._id, "change")} <= triggers
    assert (training.model_selector._id, "change") not in triggers
    # The chained step refreshes every model-dependent display in one call.
    follow = [fn for fn in demo.fns.values() if getattr(fn.fn, "__name__", "") == "refresh_model_displays"]
    assert follow and all(training.tier_note in fn.outputs for fn in follow)
    assert all(target[1] == "then" for fn in follow for target in fn.targets)


def test_training_tier_values_follow_the_measured_tables(monkeypatch):
    monkeypatch.setattr("ui.training_tab._gpu_total", lambda device: 9.9 if device == "cuda:0" else 0.0)
    detected = training_tier_values("auto", "cuda:0")
    explicit = training_tier_values("10", "cuda:0")
    assert detected == explicit
    assert detected["blocks_to_swap"] == 0
    assert training_tier_values("6", "cuda:0")["blocks_to_swap"] == 22
    assert detected["sample_min_free_vram_gb"] == 4.5
    assert set(detected) == set(TRAINING_TIER_FIELDS)
    assert training_tier_values("32", "cuda:0")["blocks_to_swap"] == 0
