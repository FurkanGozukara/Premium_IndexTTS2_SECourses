from __future__ import annotations

from types import SimpleNamespace

import gradio as gr

from indextts.training.dataset_manifest import DURATION_BUCKETS, duration_histogram
from ui import dataset_tab
from ui.presets_store import PresetRegistry


def test_histogram_frames_keep_numeric_counts_in_bucket_order() -> None:
    empty = dataset_tab._empty_histogram()
    assert list(empty.columns) == ["bucket", "count"]
    assert str(empty["count"].dtype) == "int64"

    stored = {">15s": 19, "<3s": 0, "3-6s": 12, "12-15s": 101, "6-9s": 22, "9-12s": 51}
    frame = dataset_tab._histogram_frame(stored)
    assert list(frame["bucket"]) == list(DURATION_BUCKETS)
    assert list(frame["count"]) == [0, 12, 22, 51, 101, 19]
    assert str(frame["count"].dtype) == "int64"
    assert list(duration_histogram([2.0, 4.0, 13.0, 16.0])) == list(DURATION_BUCKETS)


def test_duration_plot_starts_with_quantitative_counts_and_ordered_buckets(tmp_path, monkeypatch) -> None:
    # Gradio's BarPlot keeps the axis types of its first value: an untyped empty frame drew the
    # later counts as categories (floating bars on a 0, 12, 19, 22, 51, 101 axis).
    monkeypatch.setattr(dataset_tab, "DATASET_STATE", tmp_path / "jobs")
    with gr.Blocks() as demo:
        dataset_tab.build_dataset_tab(SimpleNamespace(device="cpu", model_dir="models"), PresetRegistry())
    plots = [block for block in demo.blocks.values() if isinstance(block, gr.BarPlot)]
    histogram = next(plot for plot in plots if plot.title == "Segment duration distribution")
    assert histogram.value["datatypes"] == {"bucket": "nominal", "count": "quantitative"}
    assert histogram.sort == list(DURATION_BUCKETS)
