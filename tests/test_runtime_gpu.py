import pytest
import torch

from indextts.runtime.gpu import apply_vram_cap, device_from_string, memory_stats


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
