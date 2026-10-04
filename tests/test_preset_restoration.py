"""Exercise the real preset UI handlers with an isolated, restartable store."""

from types import SimpleNamespace

import pytest

from ui import app
from ui.presets_store import PresetStore, SYSTEM_PREFIX


@pytest.fixture
def build_demo(tmp_path, monkeypatch):
    monkeypatch.setattr(
        app, "PresetStore",
        lambda registry, _root: PresetStore(registry, tmp_path / "presets", detect_tier=lambda: 32),
    )
    monkeypatch.setattr(app, "load_persisted_runtime", lambda: None)
    monkeypatch.setattr(app, "_runtime_summary", lambda *args, **kwargs: None)
    # These tests compare every stored value; the datasets prepared in this installation must not change them.
    monkeypatch.setattr(app, "existing_training_dataset", lambda values: dict(values))
    return lambda: app.build_app(SimpleNamespace(model_dir="models", device="cpu"))


def test_a_missing_preset_dataset_gives_way_to_the_newest_prepared_one(tmp_path, monkeypatch):
    # The GPU tier presets carry the default datasets/voice_dataset: after every restart the Training tab
    # opened on that missing folder ("Select a dataset to see the training plan") beside prepared datasets.
    import os

    from ui import training_tab

    older, newer = tmp_path / "a_voice", tmp_path / "b_voice_audited"
    for stamp, folder in enumerate((newer, older)):
        folder.mkdir()
        (folder / "dataset_info.json").write_text("{}", encoding="utf-8")
        os.utime(folder / "dataset_info.json", (2000 - stamp, 2000 - stamp))
    monkeypatch.setattr(training_tab, "_dataset_choices", lambda: [("a", str(older)), ("b", str(newer))])
    assert training_tab.existing_training_dataset({"training.dataset_dir": "datasets/voice_dataset", "x": 1}) == {
        "training.dataset_dir": str(newer), "x": 1}
    kept = {"training.dataset_dir": str(older)}
    assert training_tab.existing_training_dataset(kept) == kept
    monkeypatch.setattr(training_tab, "_dataset_choices", lambda: [])
    missing = {"training.dataset_dir": "datasets/voice_dataset"}
    assert training_tab.existing_training_dataset(missing) == missing


def handler(demo, name):
    return next(fn for fn in demo.fns.values() if fn.api_name == name)


def assert_loaded(demo, result, name, expected):
    from ui.model_profiles import capture_profile
    # Shared presets now also preserve the complete snapshot for each model.
    expected = capture_profile(expected)
    specs = demo.preset_registry.component_specs
    assert result[0]["value"] == (SYSTEM_PREFIX + name if demo.preset_store.is_system(name) else name)
    assert result[1] == name
    if len(result) == 5 + len(demo.preset_state_keys):
        # The page's own events: browser values travel in one index-addressed payload.
        payload, states = result[4], result[5:]
        loaded = {key: payload[str(index)] for index, key in enumerate(demo.preset_payload_keys)}
        loaded.update(zip(demo.preset_state_keys, states))
    else:
        loaded = dict(zip((spec.key for spec in specs), result[2:2 + len(specs)]))
    assert loaded == {spec.key: expected[spec.key] for spec in specs}
    assert demo.preset_store.get_last_used() == name


@pytest.mark.parametrize("action", ["select_preset", "load_preset", "save_preset"])
def test_new_sessions_and_restart_restore_all_values_from_the_latest_action(build_demo, action):
    demo = build_demo()
    store = demo.preset_store
    expected = demo.preset_registry.coerce({
        "generation.temperature": 0.73,
        "generation.speaking_rate": 1.12,
        "generation.auto_lora_speaking_rate": False,
        "batch.output_subfolder": "remembered-batch",
        "dataset.target_s": 9.0,
        "training.learning_rate": 0.000023,
        "runtime.lora_strength": 0.65,
        "grid.num_beams": 2,
    })
    store.save("remember me", expected)
    store.load("8 GB GPU")

    event = handler(demo, action)
    if action == "save_preset":
        event.fn("remember me", *[expected[spec.key] for spec in demo.preset_registry.component_specs])
    else:
        assert_loaded(demo, event.fn("remember me"), "remember me", expected)
    assert store.last_used_path.read_text(encoding="utf-8").strip() == "remember me"

    # A new page must use the latest bookmark, not the State baked into build_app.
    initial = handler(demo, "initial_load")
    assert initial.inputs == []
    assert_loaded(demo, initial.fn(), "remember me", expected)

    restarted = build_demo()
    dropdown = next(component for component in restarted.config["components"]
                    if component["props"].get("label") == "Universal preset")
    assert dropdown["props"]["value"] == "remember me"
    assert_loaded(restarted, handler(restarted, "initial_load").fn(), "remember me", expected)


def test_new_session_uses_another_process_latest_bookmark(build_demo):
    demo = build_demo()
    store = demo.preset_store
    other = PresetStore(demo.preset_registry, store.root, detect_tier=lambda: 32)
    expected = other.load("8 GB GPU")
    assert_loaded(demo, handler(demo, "initial_load").fn(), "8 GB GPU", expected)


@pytest.mark.parametrize("contents", [None, "deleted preset\n", "\n"])
def test_startup_without_a_valid_last_used_preset_loads_the_gpu_default(build_demo, contents):
    demo = build_demo()
    store = demo.preset_store
    if contents is None:
        store.last_used_path.unlink()
    else:
        store.last_used_path.write_text(contents, encoding="utf-8")
    # A fresh server has no in-memory fallback from the previous session.
    restarted = build_demo()
    expected = restarted.preset_store.tier_preset_values(32)
    assert_loaded(restarted, handler(restarted, "initial_load").fn(), "32 GB GPU", expected)
