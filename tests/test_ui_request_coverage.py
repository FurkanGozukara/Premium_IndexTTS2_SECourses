from dataclasses import fields

from indextts.runtime.vram_presets import RuntimeConfig
from ui.app import startup_request_self_check
from ui.generation_tab import (
    INFER_KWARG_KEYS,
    RUNNER_REQUEST_KEYS,
    build_default_generation_request,
    validate_request_coverage,
)
from ui.presets_store import PresetRegistry


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

