from types import SimpleNamespace

from ui.app import build_app


def test_runtime_tier_callback_matches_outputs_and_keeps_voice_settings(monkeypatch):
    monkeypatch.setattr("ui.models_tab._gpu_total", lambda _: 16.0)
    monkeypatch.setattr("ui.models_tab._gpu_free", lambda _: 12.0)
    demo = build_app(SimpleNamespace(
        model_dir="models", device="cpu", verbose=False, no_browser=True,
        port=7861, host="127.0.0.1", share=False,
    ))
    callback = next(fn for fn in demo.fns.values() if getattr(fn.fn, "__name__", "") == "apply_tier")
    for tier in ("6", "12", "32", "auto", "custom"):
        result = callback.fn(tier, "cuda:0")
        assert len(result) == len(callback.outputs)
    outputs = {component._id for component in callback.outputs}
    voice_keys = {
        "runtime.lora_path", "runtime.lora_strength", "runtime.lora_merge_into_base",
        "runtime.decoder_adapter", "runtime.decoder_adapter_strength",
    }
    for spec in demo.preset_registry.specs:
        if spec.key in voice_keys:
            assert spec.component._id not in outputs
