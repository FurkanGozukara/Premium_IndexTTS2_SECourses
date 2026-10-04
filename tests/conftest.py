from pathlib import Path
import os
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires a CUDA-capable GPU")


def pytest_collection_modifyitems(config, items):
    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False

    # GPU tests are opt-in even on CUDA workstations; the default suite is the
    # CPU-only verification contract used while a shared GPU may be occupied.
    if cuda_available and os.environ.get("INDEXTTS_RUN_GPU_TESTS") == "1":
        return

    skip_gpu = pytest.mark.skip(reason="GPU tests require INDEXTTS_RUN_GPU_TESTS=1 and CUDA")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture(scope="session", autouse=True)
def _isolated_presets(tmp_path_factory):
    """Trainings in tests end with the preset step (indextts.training.voice_preset); it must not write user presets."""
    folder = tmp_path_factory.mktemp("presets")
    previous = os.environ.get("INDEXTTS_PRESETS_DIR")
    os.environ["INDEXTTS_PRESETS_DIR"] = str(folder)
    yield folder
    if previous is None:
        os.environ.pop("INDEXTTS_PRESETS_DIR", None)
    else:
        os.environ["INDEXTTS_PRESETS_DIR"] = previous
