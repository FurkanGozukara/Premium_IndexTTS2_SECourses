"""CPU-only regression coverage for peaks across engine counter resets."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tools import vram_benchmark as benchmark


def _run_fake_benchmark(
    monkeypatch,
    tmp_path,
    *,
    text_peaks=((5.0, 8.0), (3.0, 4.0)),
    final_peak=(2.0, 3.0),
    engine_peak=None,
    batch=1,
    subtitle=True,
    fail_after_texts=False,
    fail_final_snapshot=False,
    fail_load=False,
):
    # No model module, audio decoder, CUDA context, or subprocess is started.
    # The load peak is deliberately much larger than any measured text peak.
    allocator = {"peak": (90.0, 100.0), "snapshot_fails": False}
    calls = {"callbacks": 0, "unloaded": False}

    def memory_stats(device):
        assert device == "cuda:0"
        if allocator["snapshot_fails"]:
            raise RuntimeError("allocator telemetry unavailable")
        allocated, reserved = allocator["peak"]
        return {
            "allocated_gb": 1.0,
            "reserved_gb": 2.0,
            "peak_allocated_gb": allocated,
            "peak_reserved_gb": reserved,
        }

    class FakeTTS:
        def __init__(self, **kwargs):
            if fail_load:
                raise RuntimeError("model load failed")
            self.last_generation_stats = {}

        def finish(self):
            # Simulate allocations after the final callback and a reset after
            # prior texts, so the final CUDA snapshot alone is insufficient.
            allocator["peak"] = final_peak
            allocator["snapshot_fails"] = fail_final_snapshot
            if fail_after_texts:
                raise RuntimeError("later text ran out of memory")
            self.last_generation_stats = {
                "generated_tokens": 40,
                "gpt_time": 2.0,
                "peak_vram_gb": engine_peak,
            }

        def infer_texts(self, *, texts, section_batch_size, on_text_complete, **kwargs):
            calls["texts"] = list(texts)
            calls["batch"] = section_batch_size
            calls["kwargs"] = kwargs
            assert len(text_peaks) == len(texts)
            outputs = []
            for index, peak in enumerate(text_peaks):
                allocator["peak"] = peak
                audio = (10, SimpleNamespace(shape=(20, 1)))
                outputs.append(audio)
                # A failed final text never emits its completion callback.
                if fail_after_texts and index == len(text_peaks) - 1:
                    break
                on_text_complete(index, audio)
                calls["callbacks"] += 1
            self.finish()
            return outputs

        def infer(self, *, text, output_path, **kwargs):
            calls["single_text"] = text
            calls["kwargs"] = kwargs
            self.finish()

        def unload(self):
            calls["unloaded"] = True

    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    monkeypatch.setattr(benchmark, "_resolve_reference_audio", lambda _: tmp_path / "reference.wav")
    monkeypatch.setattr(benchmark, "_wait_for_idle", lambda _: {"used_gb": 0.0})
    monkeypatch.setattr(benchmark, "memory_stats", memory_stats)
    monkeypatch.setitem(sys.modules, "indextts.infer_v2_5", SimpleNamespace(IndexTTS2=FakeTTS))
    monkeypatch.setitem(sys.modules, "librosa", SimpleNamespace(get_duration=lambda **_: 2.5))
    for method in ("init", "reset_peak_memory_stats", "synchronize"):
        monkeypatch.setattr(torch.cuda, method, Mock())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    argv = ["--child", "--tier", "6", "--batch", str(batch), "--json-out", str(tmp_path / "result.json")]
    if subtitle:
        argv.append("--subtitle")
    result = benchmark.run_one(benchmark._parser().parse_args(argv))
    persisted = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result == persisted
    return result, calls


@pytest.mark.parametrize("batch,subtitle,peaks", [
    (1, True, ((5.0, 8.0), (3.0, 4.0))),
    (2, False, ((5.0, 8.0), (3.0, 4.0))),
    (2, True, ((5.0, 8.0), (6.0, 7.0), (3.0, 4.0))),
])
def test_all_text_peaks_survive_counter_resets(monkeypatch, tmp_path, batch, subtitle, peaks):
    result, calls = _run_fake_benchmark(
        monkeypatch, tmp_path, batch=batch, subtitle=subtitle, text_peaks=peaks,
    )
    assert result["fit"] is True
    assert result["error"] is None
    assert result["peak_allocated_gb"] == max(peak[0] for peak in peaks)
    assert result["peak_reserved_gb"] == max(peak[1] for peak in peaks)
    assert calls["callbacks"] == len(peaks)
    assert calls["batch"] == batch
    assert calls["kwargs"]["seed"] == 123
    assert calls["kwargs"]["do_sample"] is False
    assert calls["unloaded"] is True
    assert result["audio_seconds"] == len(peaks) * 2.0
    assert result["generated_tokens"] == 40
    assert result["tokens_per_s"] == 20.0
    # Preserve post-load residency fields without incorporating the 90/100 GiB
    # load peaks into the generation-only measurements.
    assert result["load_allocated_gb"] == 1.0
    assert result["load_reserved_gb"] == 2.0


def test_final_snapshot_can_raise_either_workload_peak(monkeypatch, tmp_path):
    result, _ = _run_fake_benchmark(monkeypatch, tmp_path, final_peak=(7.0, 9.0))
    assert result["peak_allocated_gb"] == 7.0
    assert result["peak_reserved_gb"] == 9.0


def test_engine_aggregate_allocated_peak_is_not_replaced_by_last_text(monkeypatch, tmp_path):
    result, _ = _run_fake_benchmark(monkeypatch, tmp_path, engine_peak=9.25)
    assert result["peak_allocated_gb"] == 9.25
    assert result["peak_reserved_gb"] == 8.0


@pytest.mark.parametrize("engine_peak", [None, -1.0, float("nan"), float("inf"), "unknown"])
def test_invalid_optional_engine_peak_keeps_valid_allocator_measurements(monkeypatch, tmp_path, engine_peak):
    result, _ = _run_fake_benchmark(monkeypatch, tmp_path, engine_peak=engine_peak)
    assert result["fit"] is True
    assert result["peak_allocated_gb"] == 5.0
    assert result["peak_reserved_gb"] == 8.0


def test_single_text_peak_contract_is_unchanged(monkeypatch, tmp_path):
    result, calls = _run_fake_benchmark(monkeypatch, tmp_path, subtitle=False, final_peak=(3.5, 4.5))
    assert calls["single_text"] == benchmark.TEXT
    assert calls["callbacks"] == 0
    assert result["fit"] is True
    assert result["audio_seconds"] == 2.5
    assert result["peak_allocated_gb"] == 3.5
    assert result["peak_reserved_gb"] == 4.5


def test_later_failure_preserves_earlier_completed_text_peaks(monkeypatch, tmp_path):
    result, calls = _run_fake_benchmark(
        monkeypatch, tmp_path, fail_after_texts=True, final_peak=(6.0, 4.0),
    )
    assert calls["callbacks"] == 1
    assert result["fit"] is False
    assert result["error"] == "RuntimeError: later text ran out of memory"
    assert result["peak_allocated_gb"] == 6.0
    assert result["peak_reserved_gb"] == 8.0


def test_later_telemetry_failure_does_not_erase_collected_peaks(monkeypatch, tmp_path):
    result, _ = _run_fake_benchmark(
        monkeypatch, tmp_path, fail_after_texts=True, fail_final_snapshot=True,
    )
    assert result["fit"] is False
    assert result["error"] == "RuntimeError: later text ran out of memory"
    assert result["peak_allocated_gb"] == 5.0
    assert result["peak_reserved_gb"] == 8.0


def test_load_failure_still_reports_available_load_peak(monkeypatch, tmp_path):
    result, _ = _run_fake_benchmark(monkeypatch, tmp_path, fail_load=True)
    assert result["fit"] is False
    assert result["error"] == "RuntimeError: model load failed"
    assert result["peak_allocated_gb"] == 90.0
    assert result["peak_reserved_gb"] == 100.0
