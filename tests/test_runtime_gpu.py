import pytest
import torch

from indextts.runtime.gpu import apply_vram_cap, device_from_string, memory_stats


def test_gpu_inventory_shows_the_device_wide_free_memory(monkeypatch):
    # Under Windows (WDDM) CUDA's free memory leaves other processes out: a training worker held 15 GB and the
    # inventory still showed 30.26 of 31.82 GB free. The table takes the lower nvidia-smi figure.
    from types import SimpleNamespace

    from indextts.runtime import gpu
    from ui import models_tab

    monkeypatch.setattr(gpu.shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(gpu.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout="GPU-AAAA-1, 16000\nGPU-BBBB-2, 24000\n"))
    assert gpu.device_wide_free_gb() == {"aaaa-1": 16000 / 1024, "bbbb-2": 24000 / 1024}
    monkeypatch.setattr(models_tab, "list_gpus", lambda: [gpu.GpuInfo(0, "RTX", 31.82, 30.26, True)])
    monkeypatch.setattr(models_tab, "visible_gpu_uuid", lambda index: "aaaa-1")
    monkeypatch.setattr(models_tab, "device_wide_free_gb", gpu.device_wide_free_gb)
    assert models_tab._gpu_rows() == [["cuda:0", "RTX", 31.82, 15.62, "Yes"]]
    monkeypatch.setattr(models_tab, "device_wide_free_gb", lambda: {})
    assert models_tab._gpu_rows()[0][3] == 30.26  # without nvidia-smi the CUDA figure stays


def test_cpu_memory_stats_and_device_resolution():
    assert memory_stats("cpu") == {
        "allocated_gb": 0.0,
        "reserved_gb": 0.0,
        "peak_allocated_gb": 0.0,
        "peak_reserved_gb": 0.0,
    }
    assert device_from_string("cpu") == torch.device("cpu")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_apply_vram_cap():
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    fraction = apply_vram_cap("cuda:0", total)
    assert 0.99 <= fraction <= 1.0


def test_app_keeps_its_own_cuda_jit_cache(tmp_path, monkeypatch):
    # The driver's per-user kernel cache (1 GB, shared by every CUDA program) is smaller than the 1.3 GB one IndexTTS
    # generation needs, so every new process prepared its GPU kernels again: the first generation took 22 s, not 8 s.
    import os

    import webui

    monkeypatch.delenv("CUDA_CACHE_PATH", raising=False)
    monkeypatch.delenv("CUDA_CACHE_MAXSIZE", raising=False)
    webui.configure_cuda_jit_cache(tmp_path / "models")
    assert os.environ["CUDA_CACHE_PATH"] == str(tmp_path / "models" / "cuda_jit_cache")
    assert (tmp_path / "models" / "cuda_jit_cache").is_dir()
    assert os.environ["CUDA_CACHE_MAXSIZE"] == str(4 * 1024 ** 3)
    # Settings the user made win.
    monkeypatch.setenv("CUDA_CACHE_PATH", str(tmp_path / "mine"))
    monkeypatch.setenv("CUDA_CACHE_MAXSIZE", "1000")
    webui.configure_cuda_jit_cache(tmp_path / "models")
    assert (os.environ["CUDA_CACHE_PATH"], os.environ["CUDA_CACHE_MAXSIZE"]) == (str(tmp_path / "mine"), "1000")
