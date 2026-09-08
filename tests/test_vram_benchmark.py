from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools import vram_benchmark as benchmark
from tools.vram_benchmark import _parser, _resolve_reference_audio


def test_benchmark_reference_falls_back_to_reference_library(tmp_path: Path) -> None:
    fallback = tmp_path / "reference_audios" / "demo_voice.mp3"
    fallback.parent.mkdir(parents=True)
    fallback.write_bytes(b"audio")

    assert _resolve_reference_audio(None, root=tmp_path) == fallback.resolve()


def test_benchmark_reference_prefers_explicit_path(tmp_path: Path) -> None:
    explicit = tmp_path / "custom.wav"
    explicit.write_bytes(b"audio")

    parsed = _parser().parse_args(["--reference", str(explicit)])

    assert _resolve_reference_audio(parsed.reference, root=tmp_path) == explicit.resolve()


def test_benchmark_reference_error_lists_override(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="--reference"):
        _resolve_reference_audio(None, root=tmp_path)


def _smi(stdout, returncode=0, stderr=""):
    return SimpleNamespace(stdout=stdout, returncode=returncode, stderr=stderr)


def _memory(used=4.0):
    return dict(device_id="0", physical_index="0", total_gb=32.0, free_gb=28.0,
                used_gb=used, reserved_gb=0.4, usage_source="memory.used")


def test_idle_uses_device_memory_not_driver_reservation(monkeypatch, capsys):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    query = Mock(return_value=_smi("2, 32607, 3009, 29179, 420"))
    monkeypatch.setattr(benchmark.subprocess, "run", query)

    result = benchmark._wait_for_idle(0)

    assert result["used_gb"] == pytest.approx(3009 / 1024)
    assert result["usage_source"] == "memory.used"
    assert query.call_args.args[0][1:3] == ["--id", "2"]
    assert "Idle check passed" in capsys.readouterr().out


@pytest.mark.parametrize("used,reserved,expected,source", [
    ("N/A", "420", 3008, "total-free-reserved (fallback)"),
    ("N/A", "N/A", 3428, "total-free (conservative fallback; reserved may be included)"),
])
def test_idle_query_fallback_accounts_for_known_reservation(monkeypatch, used, reserved, expected, source):
    monkeypatch.setattr(benchmark.subprocess, "run", Mock(return_value=_smi(
        f"0, 32607, {used}, 29179, {reserved}")))
    result = benchmark._query_idle_memory("0")
    assert result["used_gb"] == pytest.approx(expected / 1024)
    assert result["usage_source"] == source


def test_idle_query_supports_driver_without_reserved_field(monkeypatch):
    query = Mock(side_effect=[_smi("", 1, "invalid field memory.reserved"),
                              _smi("0, 8192, 512, 7400")])
    monkeypatch.setattr(benchmark.subprocess, "run", query)
    result = benchmark._query_idle_memory("0")
    assert result["used_gb"] == 0.5
    assert result["reserved_gb"] is None
    assert "memory.reserved" not in query.call_args.args[0][3]


@pytest.mark.parametrize("response", [
    _smi("0, 8192, 512, 99999, 128"),
    _smi("0, N/A, N/A, N/A, N/A"),
    _smi(""), _smi("", 1, "unavailable"),
])
def test_idle_query_fails_closed_on_unavailable_memory(monkeypatch, response):
    monkeypatch.setattr(benchmark.subprocess, "run", Mock(return_value=response))
    with pytest.raises(RuntimeError, match="Cannot verify idle GPU"):
        benchmark._query_idle_memory("0")


def test_idle_query_has_bounded_driver_timeout(monkeypatch):
    query = Mock(side_effect=benchmark.subprocess.TimeoutExpired("nvidia-smi", 5))
    monkeypatch.setattr(benchmark.subprocess, "run", query)
    with pytest.raises(RuntimeError, match="nvidia-smi failed"):
        benchmark._query_idle_memory("0")
    assert query.call_args.kwargs["timeout"] == 5.0


@pytest.mark.parametrize("value", ["", "-1"])
def test_idle_check_rejects_hidden_cuda_without_query(monkeypatch, value):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
    query = Mock()
    monkeypatch.setattr(benchmark, "_query_idle_memory", query)
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        benchmark._wait_for_idle(0)
    query.assert_not_called()


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "-inf", "no"])
def test_idle_timeout_cli_rejects_invalid_values(value):
    with pytest.raises(SystemExit):
        _parser().parse_args(["--idle-timeout", value])


def test_idle_wait_obeys_custom_deadline_and_reports_actual_threshold(monkeypatch, capsys):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    clock = [0.0]
    sleeps = []
    monkeypatch.setattr(benchmark.time, "monotonic", lambda: clock[0])

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(benchmark.time, "sleep", sleep)
    monkeypatch.setattr(benchmark, "_query_idle_memory", lambda _: _memory())
    with pytest.raises(TimeoutError, match=r"exceeded 7s:.*idle limit 3.20 GiB"):
        benchmark._wait_for_idle(7)
    assert sleeps == [5.0, 2.0]
    assert "0.0s remaining" in capsys.readouterr().out


def test_idle_wait_resumes_when_busy_gpu_becomes_idle(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(benchmark.time, "sleep", lambda _: None)
    monkeypatch.setattr(benchmark, "_query_idle_memory", Mock(side_effect=[_memory(4), _memory(2)]))
    assert benchmark._wait_for_idle(30)["used_gb"] == 2


def test_idle_failure_does_not_query_cuda_memory(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    monkeypatch.setattr(benchmark, "_resolve_reference_audio", lambda _: tmp_path / "reference.wav")
    monkeypatch.setattr(benchmark, "_wait_for_idle", Mock(side_effect=TimeoutError("busy")))
    memory = Mock(side_effect=AssertionError("CUDA must stay untouched"))
    monkeypatch.setattr(benchmark, "memory_stats", memory)
    result = benchmark.run_one(_parser().parse_args(["--child", "--idle-timeout", "0"]))
    assert result["fit"] is False
    assert result["error"] == "TimeoutError: busy"
    memory.assert_not_called()


def test_all_tiers_forward_idle_timeout_and_visible_device(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    monkeypatch.setattr(benchmark, "VRAM_TIERS", (6, 8))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-selected")
    monkeypatch.setattr(benchmark, "_resolve_reference_audio", lambda _: tmp_path / "reference.wav")
    query = Mock(return_value=_smi(
        'VRAM_BENCHMARK_JSON={"tier":6,"variant":"bf16","blocks_to_swap":0,"fit":true}\n'))
    monkeypatch.setattr(benchmark.subprocess, "run", query)
    assert benchmark.run_all(_parser().parse_args(["--idle-timeout", "7"])) == 0
    for call in query.call_args_list:
        command = call.args[0]
        assert command[command.index("--idle-timeout") + 1] == "7.0"
        assert call.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-selected"
