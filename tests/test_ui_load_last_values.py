from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ui.app import build_app
from ui.grid_tab import GRID_EMPTY_HINT, latest_checkpoint_eval_state, latest_lora_folder


@pytest.fixture(scope="module")
def demo():
    return build_app(
        SimpleNamespace(
            model_dir="models",
            device="cpu",
            verbose=False,
            no_browser=True,
            port=7861,
            host="127.0.0.1",
            share=False,
        )
    )


def _descendant_ids(node: dict[str, Any]) -> set[int]:
    result: set[int] = set()
    for child in node.get("children") or []:
        if isinstance(child.get("id"), int):
            result.add(child["id"])
        result.update(_descendant_ids(child))
    return result


def _tab_component_ids(config: dict[str, Any]) -> dict[str, set[int]]:
    components = {item["id"]: item for item in config["components"]}
    result: dict[str, set[int]] = {}

    def visit(node: dict[str, Any]) -> None:
        component = components.get(node.get("id"), {})
        if component.get("type") == "tabitem":
            result[str(component["props"]["label"])] = _descendant_ids(node)
        for child in node.get("children") or []:
            visit(child)

    visit(config["layout"])
    return result


def _find_in_tab(
    config: dict[str, Any], tab: str, component_type: str, label: str
) -> dict[str, Any]:
    tab_ids = _tab_component_ids(config)[tab]
    matches = [
        component
        for component in config["components"]
        if component["id"] in tab_ids
        and component["type"] == component_type
        and component.get("props", {}).get("label") == label
    ]
    assert len(matches) == 1, (tab, component_type, label, len(matches))
    return matches[0]


def _table_data(component: dict[str, Any]) -> list[Any]:
    return list((component["props"].get("value") or {}).get("data") or [])


def _button_hue(component: dict[str, Any]) -> str:
    values = [
        item.removeprefix("ax-")
        for item in component["props"].get("elem_classes") or []
        if item.startswith("ax-")
    ]
    assert len(values) == 1, component["props"].get("value")
    return values[0]


def _button_icon(component: dict[str, Any]) -> str:
    label = str(component["props"].get("value") or "")
    leading = label.split(maxsplit=1)[0] if label else ""
    assert any(ord(character) > 127 for character in leading), label
    return leading


def test_only_presets_load_automatically_and_attachments_use_header_button(demo) -> None:
    config = demo.config
    load_dependencies = [
        dependency
        for dependency in config["dependencies"]
        if any(target[1] == "load" for target in dependency.get("targets") or [])
    ]
    assert [dependency.get("api_name") for dependency in load_dependencies] == [
        "initial_load"
    ]

    last_values = [
        component
        for component in config["components"]
        if component["type"] == "button"
        and component.get("props", {}).get("value") == "🕘  Load last values"
    ]
    assert len(last_values) == 1
    button_id = last_values[0]["id"]
    assert last_values[0]["props"]["interactive"] is False
    initial_load = load_dependencies[0]
    enable_dependency = next(
        dependency
        for dependency in config["dependencies"]
        if dependency["outputs"] == [button_id]
        and dependency.get("trigger_after") == initial_load["id"]
    )
    assert enable_dependency["outputs"] == [button_id]
    assert enable_dependency["trigger_after"] == initial_load["id"]
    sections_id = next(
        component["id"]
        for component in config["components"]
        if component["type"] == "button"
        and component.get("props", {}).get("value")
        == "⇕  Open / close all sections"
    )

    sibling_sequences: list[list[int]] = []

    def collect_siblings(node: dict[str, Any]) -> None:
        sibling_sequences.append(
            [child["id"] for child in node.get("children") or [] if "id" in child]
        )
        for child in node.get("children") or []:
            collect_siblings(child)

    collect_siblings(config["layout"])
    assert any(
        sequence[index : index + 2] == [button_id, sections_id]
        for sequence in sibling_sequences
        for index in range(max(0, len(sequence) - 1))
    )
    for api_name in (
        "attach_generation",
        "attach_batch",
        "attach_dataset",
        "attach_training",
        "attach_checkpoint_grid",
    ):
        matches = [
            dependency
            for dependency in config["dependencies"]
            if dependency.get("api_name") == api_name
        ]
        assert len(matches) == 1
        assert matches[0]["targets"] == [(button_id, "click")]

    checkpoint_load = next(
        dependency
        for dependency in config["dependencies"]
        if dependency.get("api_name") == "load_last_checkpoint_analysis"
    )
    assert checkpoint_load["targets"] == [(button_id, "click")]


def test_build_time_results_are_blank_even_with_existing_catalogs(demo) -> None:
    config = demo.config
    recent = _find_in_tab(
        config, "Voice Generation", "dataframe", "Recent outputs (last 10)"
    )
    assert _table_data(recent) == []
    for label, component_type in (
        ("Generated audio", "audio"),
        ("Generated MP4", "video"),
    ):
        component = _find_in_tab(
            config, "Voice Generation", component_type, label
        )
        assert component["props"].get("value") is None
        assert component["props"]["visible"] is False

    assert _table_data(
        _find_in_tab(config, "Batch Generation", "dataframe", "Batch results")
    ) == []
    assert _table_data(
        _find_in_tab(
            config, "LoRA Dataset Preparation", "dataframe", "Prepared segments"
        )
    ) == []
    assert _table_data(
        _find_in_tab(config, "LoRA / DoRA Training", "dataframe", "Checkpoints")
    ) == []
    assert (
        _find_in_tab(
            config, "LoRA / DoRA Training", "textbox", "Training log (last 60 lines)"
        )["props"].get("value")
        is None
    )

    folder = _find_in_tab(
        config, "Checkpoint Grid", "dropdown", "LoRA / DoRA folder"
    )
    assert folder["props"].get("value") is None
    assert _table_data(
        _find_in_tab(config, "Checkpoint Grid", "dataframe", "Checkpoints")
    ) == []
    assert _table_data(
        _find_in_tab(config, "Checkpoint Grid", "dataframe", "Grid cells")
    ) == []
    assert (
        _find_in_tab(config, "Checkpoint Grid", "checkboxgroup", "Checkpoints")[
            "props"
        ].get("value")
        == []
    )
    eval_references = _find_in_tab(
        config, "Checkpoint Grid", "dropdown", "Evaluation references"
    )
    assert eval_references["props"]["value"] == ""
    assert eval_references["props"]["choices"] == [
        ("Same as training validation", ""),
        ("self", "self"),
        (
            "other (inference-like: a different clip of the same speaker)",
            "other",
        ),
    ]
    assert _find_in_tab(
        config, "Checkpoint Grid", "number", "Training subset"
    )["props"]["value"] == 48
    assert _find_in_tab(
        config, "Checkpoint Grid", "checkbox", "Include base model"
    )["props"]["value"] is True
    beams = _find_in_tab(config, "Checkpoint Grid", "slider", "Beams")
    assert beams["props"]["value"] == 3
    assert "strongly affects quality" in beams["props"]["info"]
    grid_ids = _tab_component_ids(config)["Checkpoint Grid"]
    assert any(
        component["id"] in grid_ids
        and component["type"] == "markdown"
        and component.get("props", {}).get("value") == GRID_EMPTY_HINT
        for component in config["components"]
    )


def test_button_hues_and_icons_are_unique_within_every_tab(demo) -> None:
    config = demo.config
    tabs = _tab_component_ids(config)
    buttons = [
        component for component in config["components"] if component["type"] == "button"
    ]
    tab_button_ids = set().union(*tabs.values())
    global_buttons = [
        component for component in buttons if component["id"] not in tab_button_ids
    ]

    for tab_name, component_ids in tabs.items():
        present = global_buttons + [
            component for component in buttons if component["id"] in component_ids
        ]
        hues = [_button_hue(component) for component in present]
        icons = [_button_icon(component) for component in present]
        assert not {value: count for value, count in Counter(hues).items() if count > 1}, tab_name
        assert not {value: count for value, count in Counter(icons).items() if count > 1}, tab_name

    assert sum(_button_hue(component) == "bronze" for component in buttons) == 1
    assert sum(_button_icon(component) == "🕘" for component in buttons) == 1


def test_latest_lora_folder_and_evaluation_job_use_newest_mtime(
    tmp_path: Path,
) -> None:
    older = tmp_path / "older"
    newer = tmp_path / "newer"
    for folder in (older, newer):
        folder.mkdir()
        (folder / "status.json").write_text("{}", encoding="utf-8")
        (folder / "voice.safetensors").write_bytes(b"checkpoint")
    os.utime(older / "status.json", (100, 100))
    os.utime(older / "voice.safetensors", (300, 300))
    os.utime(newer / "status.json", (200, 200))
    os.utime(newer / "voice.safetensors", (200, 200))

    assert latest_lora_folder(tmp_path) == str(older.resolve())

    legacy = older / "analysis" / "eval_job" / "status.json"
    latest = older / "analysis" / "eval_jobs" / "new" / "status.json"
    legacy.parent.mkdir(parents=True)
    latest.parent.mkdir(parents=True)
    legacy.write_text("{}", encoding="utf-8")
    latest.write_text("{}", encoding="utf-8")
    os.utime(legacy, (400, 400))
    os.utime(latest, (500, 500))
    assert latest_checkpoint_eval_state(older) == str(latest.parent.resolve())
