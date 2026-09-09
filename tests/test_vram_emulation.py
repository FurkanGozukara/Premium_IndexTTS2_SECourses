"""INDEXTTS_VRAM_EMULATE_GB makes a process behave like a smaller card."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from indextts.runtime import gpu


ROOT = Path(__file__).resolve().parents[1]


def test_emulation_is_off_without_the_variable(monkeypatch):
    monkeypatch.delenv(gpu.EMULATE_ENV, raising=False)
    assert gpu.emulated_gpu_gb() is None
    assert gpu.emulated_cap_gb() is None
    assert gpu.apply_emulated_vram_cap() is None


@pytest.mark.parametrize("value,size,cap", [("8", 8.0, 7.0), ("6", 6.0, 5.0), ("32", 32.0, 30.0), ("11.9", 11.9, 11.0)])
def test_emulated_size_maps_to_the_tier_budget(monkeypatch, value, size, cap):
    monkeypatch.setenv(gpu.EMULATE_ENV, value)
    assert gpu.emulated_gpu_gb() == size
    assert gpu.emulated_cap_gb() == cap


@pytest.mark.parametrize("value", ["", "0", "-4", "abc"])
def test_invalid_emulation_values_are_ignored(monkeypatch, value):
    monkeypatch.setenv(gpu.EMULATE_ENV, value)
    assert gpu.emulated_gpu_gb() is None
    assert gpu.emulated_cap_gb() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.gpu
def test_emulated_process_sees_the_small_card_and_hits_its_budget():
    script = (
        "import torch, indextts.runtime as runtime\n"
        "from indextts.runtime import gpu\n"
        "print('total', gpu.gpu_total_gb(0))\n"
        "print('name', gpu.list_gpus()[0].name)\n"
        "held = torch.zeros(int(1.5 * 1024 ** 3) // 4, device='cuda')\n"
        "print('free', round(gpu.gpu_free_gb(0), 2))\n"
        "try:\n"
        "    torch.zeros(int(6 * 1024 ** 3) // 4, device='cuda')\n"
        "    print('oom', False)\n"
        "except torch.OutOfMemoryError:\n"
        "    print('oom', True)\n"
    )
    env = {**os.environ, gpu.EMULATE_ENV: "8", "CUDA_VISIBLE_DEVICES": "0"}
    completed = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    lines = dict(line.split(" ", 1) for line in completed.stdout.splitlines() if " " in line and not line.startswith(">>"))
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert float(lines["total"]) == 8.0
    assert "emulated 8 GB" in lines["name"]
    assert 5.0 <= float(lines["free"]) <= 5.6
    assert lines["oom"] == "True"
