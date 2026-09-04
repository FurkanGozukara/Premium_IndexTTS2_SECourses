from dataclasses import fields
import json
from types import SimpleNamespace
import warnings

from indextts.runtime.vram_presets import RuntimeConfig
from indextts.training.dataset_prep import DatasetPrepConfig
from indextts.training.train_config import TrainConfig
from ui.app import build_app
from ui.batch_tab import _batch_items, _item_generation_values
from ui.generation_tab import GENERATION_DEFAULTS
from ui.training_tab import TRAIN_DEFAULTS


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
    components = {
        (component["type"], component["props"].get("label")): component
        for component in demo.config["components"]
    }
    reference_source = components[("file", "Reference Voice (audio or video)")]
    reference_audio = components[("audio", "Reference Voice audio preview")]
    reference_video = components[("video", "Reference Voice video preview")]
    assert {"audio", "video"}.issubset(reference_source["props"]["file_types"])
    assert reference_source["props"].get("value") is None
    assert reference_audio["props"].get("value") is None
    assert reference_audio["props"]["visible"] is False
    assert reference_audio["props"]["interactive"] is False
    assert reference_video["props"]["visible"] is False
    dependencies = {item.get("api_name"): item for item in demo.config["dependencies"]}
    assert dependencies["generate_voice"]["trigger_only_on_success"] is True
    assert dependencies["generate_voice"]["trigger_after"] is not None
    assert {
        "grid.adapter_dir",
        "grid.checkpoints",
        "grid.strengths",
        "grid.texts",
        "grid.references",
        "grid.seed",
        "grid.eval_reference_mode",
        "grid.eval_train_subset",
        "grid.eval_include_base",
        "grid.num_beams",
    }.issubset(keys)
    default_path = demo.preset_store.system_dir / "default.json"
    before = {
        path.name: path.read_bytes()
        for path in demo.preset_store.system_dir.glob("*.json")
    }
    demo.preset_store.ensure_system_presets()
    after = {
        path.name: path.read_bytes()
        for path in demo.preset_store.system_dir.glob("*.json")
    }
    assert after == before
    values = json.loads(before[default_path.name].decode("utf-8"))["values"]
    assert values["grid.eval_reference_mode"] == ""
    assert values["grid.eval_train_subset"] == 48
    assert values["grid.eval_include_base"] is True
    assert values["grid.num_beams"] == GENERATION_DEFAULTS["generation.num_beams"]
    assert values["dataset.boundary_mode"] == "sentence"
    assert values["dataset.min_pause_boundary_ms"] == 400
    for preset_path in demo.preset_store.system_dir.glob("*.json"):
        preset_values = json.loads(preset_path.read_text(encoding="utf-8"))["values"]
        assert preset_values["dataset.boundary_mode"] == "sentence"
        assert preset_values["dataset.min_pause_boundary_ms"] == 400
        assert preset_values["dataset.target_s"] == 14.0
        assert preset_values["dataset.min_s"] == 4.0
        assert preset_values["dataset.max_s"] == 20.0
    assert demo.preset_registry["dataset.target_s"].maximum == 30
    assert demo.preset_registry["dataset.min_s"].maximum == 15
    assert demo.preset_registry["dataset.max_s"].maximum == 40

    measured_fields = (
        "rank",
        "alpha",
        "batch_size",
        "grad_accumulation",
        "learning_rate",
        "epochs",
        "warmup_steps",
        "speaker_ref_mode",
        "emo_ref_mode",
        "val_reference_mode",
        "keep_last_n",
        "epoch_train_state",
        "sample_speaking_rate",
    )
    config_defaults = TrainConfig(
        dataset_dir="datasets/secourses_demo", name="voice_adapter"
    ).to_dict()
    registry_defaults = demo.preset_registry.defaults()
    for field_name in measured_fields:
        key = f"training.{field_name}"
        assert TRAIN_DEFAULTS[field_name] == config_defaults[field_name]
        assert registry_defaults[key] == config_defaults[field_name]
        for preset_path in demo.preset_store.system_dir.glob("*.json"):
            preset_values = json.loads(preset_path.read_text(encoding="utf-8"))["values"]
            assert preset_values[key] == config_defaults[field_name]
    assert registry_defaults["generation.speaking_rate"] == 1.0
    assert registry_defaults["generation.auto_lora_speaking_rate"] is True
    assert registry_defaults["grid.speaking_rate"] == 1.0

    checkpoint_group = demo.ui_tabs["Checkpoint Grid"].checkpoint_group
    assert checkpoint_group.get_block_name() == "checkboxgroup"
    assert checkpoint_group.choices == []
    assert checkpoint_group.preprocess(["base", "epoch_001"]) == [
        "base",
        "epoch_001",
    ]

    saved_grid = components[("dropdown", "Saved grids")]
    saved_grid_targets = [
        target
        for dependency in demo.config["dependencies"]
        for target in dependency["targets"]
        if target[0] == saved_grid["id"]
    ]
    assert (saved_grid["id"], "input") in saved_grid_targets
    assert (saved_grid["id"], "change") not in saved_grid_targets


def test_blank_batch_folder_does_not_scan_working_directory(tmp_path, monkeypatch):
    (tmp_path / "unrelated.txt").write_text("must not be included", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    items = _batch_items(None, "first\n\nsecond", "")

    assert [item["text"] for item in items] == ["first", "second"]


def test_plain_text_batch_item_disables_caption_timing() -> None:
    values = {"generation.use_caption_timing": True, "generation.temperature": 0.8}

    plain = _item_generation_values(values, {"subtitle": None})
    captioned = _item_generation_values(values, {"subtitle": "sample.srt"})

    assert plain["generation.use_caption_timing"] is False
    assert captioned["generation.use_caption_timing"] is True
    assert values["generation.use_caption_timing"] is True


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
