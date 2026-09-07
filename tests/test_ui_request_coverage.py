from dataclasses import fields
import json
from types import SimpleNamespace

import gradio as gr
import pytest

from indextts.runtime.vram_presets import RuntimeConfig
from indextts.utils.text_segmentation import SpeechRecoveryConfig
from ui.app import startup_request_self_check
from ui.generation_tab import (
    INFER_KWARG_KEYS,
    RUNNER_REQUEST_KEYS,
    build_default_generation_request,
    build_generation_request,
    build_generation_tab,
    bind_generation_events,
    validate_request_coverage,
)
from ui.presets_store import PresetRegistry, PresetStore


def test_default_request_has_exact_runner_and_infer_coverage():
    request = build_default_generation_request(model_dir="models")
    assert set(request) == RUNNER_REQUEST_KEYS
    assert set(request["infer_kwargs"]) == INFER_KWARG_KEYS
    assert validate_request_coverage(request) == (set(), set())


def test_runtime_config_fields_are_present_in_built_registry():
    # This light registry shape mirrors the flattened runtime representation.
    registry = PresetRegistry()
    for item in fields(RuntimeConfig):
        if item.name == "aux_residency":
            registry.register("runtime.aux_residency.semantic_model", default="gpu")
        else:
            registry.register(f"runtime.{item.name}", default=None)
    covered = {key.removeprefix("runtime.").split(".", 1)[0] for key in registry.keys}
    assert covered == {item.name for item in fields(RuntimeConfig)}


def test_gradio_recovery_controls_share_defaults_and_persist_zero_and_disabled_values(tmp_path):
    registry = PresetRegistry()
    with gr.Blocks():
        args = SimpleNamespace(model_dir="models")
        tab = build_generation_tab(args, registry)
        bind_generation_events(tab, args, registry)
    selected = {
        "generation.auto_retry_incomplete_speech": False,
        "generation.max_speech_retries": 0,
        "generation.max_speech_split_depth": 0,
    }
    defaults = registry.defaults()
    assert defaults["generation.auto_retry_incomplete_speech"] is SpeechRecoveryConfig.enabled
    assert defaults["generation.max_speech_retries"] == SpeechRecoveryConfig.max_attempts
    assert defaults["generation.max_speech_split_depth"] == SpeechRecoveryConfig.max_split_depth
    assert selected.keys() <= set(tab.request_keys)
    for key in selected:
        assert registry[key].component.visible is True
    store = PresetStore(registry, tmp_path / "presets")
    store.save("no retries", selected)
    restored = store.load("no retries")
    assert {key: restored[key] for key in selected} == selected


@pytest.mark.parametrize("batch_item", [False, True])
@pytest.mark.parametrize("enabled,attempts,depth", [(False, 0, 0), (True, 5, 2)])
def test_selected_recovery_values_reach_runner_inference_unchanged(tmp_path, batch_item, enabled, attempts, depth):
    from ui.batch_tab import _item_generation_values
    from webui_generation_runner import run_generation_request

    selected = {
        "generation.auto_retry_incomplete_speech": enabled,
        "generation.max_speech_retries": attempts,
        "generation.max_speech_split_depth": depth,
        "generation.section_batch_size": 2 if batch_item else 1,
    }
    if batch_item:
        selected = _item_generation_values(selected, {"subtitle": None})
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"outputs": {}, "processing": {}}), encoding="utf-8")
    request = build_generation_request(
        selected, prompt="reference.wav", text="A spoken section.",
        runtime={"device": "cpu", "gpt_dtype": "fp32"}, metadata_path=str(metadata),
        task_layout={
            "task_folder": str(tmp_path), "final_wav_path": str(tmp_path / "final.wav"),
            "final_mp3_path": str(tmp_path / "final.mp3"), "final_mp4_path": str(tmp_path / "final.mp4"),
        },
    )
    calls = []

    class CapturedRequest(Exception):
        pass

    class CaptureEngine:
        low_vram = False

        def infer(self, **kwargs):
            calls.append(("single", kwargs))
            raise CapturedRequest()

        def infer_texts(self, **kwargs):
            calls.append(("batch", kwargs))
            raise CapturedRequest()

    with pytest.raises(CapturedRequest):
        run_generation_request(request, CaptureEngine())
    assert len(calls) == 1
    path, kwargs = calls[0]
    assert path == ("batch" if batch_item else "single")
    assert kwargs["auto_retry_incomplete_speech"] is enabled
    assert kwargs["max_speech_retries"] == attempts
    assert kwargs["max_speech_split_depth"] == depth

