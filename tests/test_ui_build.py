from dataclasses import fields
import json
from types import SimpleNamespace
import warnings

from indextts.runtime.vram_presets import RuntimeConfig
from indextts.training.dataset_prep import DatasetPrepConfig
from indextts.training.train_config import TrainConfig
from ui.app import build_app
from ui.batch_tab import _batch_items, _batch_timer, _item_generation_values
from ui.common import APP_CSS
from ui.generation_tab import GENERATION_DEFAULTS
from ui.models_tab import _estimate_html, _gpu_total
from ui.training_tab import TRAIN_DEFAULTS, _refresh_dataset_updates


def test_cpu_runtime_does_not_report_a_gpu_vram_fit() -> None:
    config = RuntimeConfig(device="cpu", gpt_dtype="fp32")
    estimate = _estimate_html(config, _gpu_total("cpu"))

    assert _gpu_total("cpu") == 0.0
    assert "CPU diagnostics mode" in estimate
    assert "Fits selected GPU" not in estimate


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
        "📜 Changelog",
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
    assert components[("tabitem", "📜 Changelog")]["props"]["render_children"] is False
    reference_source = components[("file", "Reference Voice (audio or video)")]
    reference_recording = components[("audio", "Microphone reference")]
    reference_audio = components[("audio", "Reference Voice audio preview")]
    reference_video = components[("video", "Reference Voice video preview")]
    assert {"audio", "video"}.issubset(reference_source["props"]["file_types"])
    assert reference_recording["props"]["sources"] == ["microphone"]
    assert reference_recording["props"]["format"] == "wav"
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


def test_completed_batch_stops_item_polling_timer() -> None:
    assert _batch_timer(True).active is False
    assert _batch_timer(True).value == 1.0
    assert _batch_timer(False).active is False
    assert _batch_timer(False).value == 5.0


def test_mobile_header_actions_wrap_within_the_viewport() -> None:
    mobile_rules = APP_CSS.split("@media (max-width: 900px)", 1)[1]

    assert ".app-header { flex-wrap: wrap; }" in mobile_rules
    assert ".row.header-actions" in mobile_rules
    assert "flex: 1 1 100% !important;" in mobile_rules
    assert "width: 100% !important;" in mobile_rules
    assert ".header-actions button.ax" in mobile_rules
    assert "min-width: 0 !important;" in mobile_rules
    assert "white-space: normal;" in mobile_rules


def test_training_dataset_refresh_recomputes_cache_status(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "prepared"
    dataset.mkdir()
    (dataset / "dataset_info.json").write_text(
        json.dumps({"segment_count": 2, "total_duration_minutes": 0.5}),
        encoding="utf-8",
    )
    monkeypatch.setattr("ui.training_tab._dataset_choices", lambda: [])

    _, before = _refresh_dataset_updates(str(dataset))
    assert before.endswith("features **not cached**")
    (dataset / "cache").mkdir()
    (dataset / "cache" / "index.jsonl").write_text("", encoding="utf-8")
    _, after = _refresh_dataset_updates(str(dataset))
    assert after.endswith("features **cached**")


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
