from dataclasses import fields
from types import SimpleNamespace
import warnings

from indextts.runtime.vram_presets import RuntimeConfig
from indextts.training.dataset_prep import DatasetPrepConfig
from indextts.training.train_config import TrainConfig
from ui.app import build_app
from ui.batch_tab import _batch_items
from ui.generation_tab import GENERATION_DEFAULTS


def test_build_app_constructs_all_tabs_without_loading_models():
    args = SimpleNamespace(
        model_dir="models",
        device="cpu",
        verbose=False,
        no_browser=True,
        port=7861,
        host="127.0.0.1",
        share=False,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        demo = build_app(args)
    gradio_warnings = [
        item
        for item in caught
        if issubclass(item.category, UserWarning)
        and ("gradio" in item.filename.lower() or "gradio" in str(item.message).lower())
    ]
    assert not gradio_warnings
    assert list(demo.ui_tabs) == [
        "Voice Generation",
        "Batch Generation",
        "LoRA Dataset Preparation",
        "LoRA / DoRA Training",
        "Checkpoint Grid",
        "Models & Performance",
        "Help",
    ]
    keys = demo.preset_registry.keys
    assert len(keys) == len(set(keys))
    assert set(GENERATION_DEFAULTS).issubset(keys)
    for prefix, config_type in (
        ("dataset.", DatasetPrepConfig),
        ("training.", TrainConfig),
        ("runtime.", RuntimeConfig),
    ):
        registered = {
            key.removeprefix(prefix).split(".", 1)[0]
            for key in keys
            if key.startswith(prefix)
        }
        assert {item.name for item in fields(config_type)}.issubset(registered)
    assert demo.request_coverage["ok"]
    assert {
        "grid.adapter_dir",
        "grid.checkpoints",
        "grid.strengths",
        "grid.texts",
        "grid.references",
        "grid.seed",
    }.issubset(keys)
    default_path = demo.preset_store.system_dir / "default.json"
    before = default_path.read_bytes()
    demo.preset_store.ensure_system_presets()
    assert default_path.read_bytes() == before


def test_blank_batch_folder_does_not_scan_working_directory(tmp_path, monkeypatch):
    (tmp_path / "unrelated.txt").write_text("must not be included", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    items = _batch_items(None, "first\n\nsecond", "")

    assert [item["text"] for item in items] == ["first", "second"]


def test_confirmation_events_pass_a_real_boolean_to_backend_handlers():
    args = SimpleNamespace(
        model_dir="models",
        device="cpu",
        verbose=False,
        no_browser=True,
        port=7861,
        host="127.0.0.1",
        share=False,
    )
    demo = build_app(args)
    components = {component["id"]: component for component in demo.config["components"]}
    confirmations = [
        dependency
        for dependency in demo.config["dependencies"]
        if "window.confirm(" in (dependency.get("js") or "")
    ]

    assert len(confirmations) == 8
    for dependency in confirmations:
        confirmation_input = components[dependency["inputs"][0]]
        assert confirmation_input["type"] == "checkbox"
        assert confirmation_input["props"]["visible"] is False
