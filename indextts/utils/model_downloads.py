"""Bridge to the distribution-level model downloader.

The standalone downloader lives beside the application repository so the same
entry point can be used by the Windows and Linux installers.
"""

from __future__ import annotations

from functools import lru_cache
import importlib.util
from pathlib import Path
import re
from types import ModuleType
from typing import Callable, Optional


INT8_GPT_LOCAL_FILENAME = "gpt_int8_convrot.safetensors"
INT8_GPT_REPO_ID = "MonsterMMORPG/Wan_GGUF"
INT8_GPT_REMOTE_FILENAME = "IndexTTS-2.5_gpt_int8_convrot.safetensors"

_BYTE_PAIR_RE = re.compile(
    r"\(([0-9]+(?:\.[0-9]+)?)\s*(B|KB|MB|GB)\s*/\s*"
    r"([0-9]+(?:\.[0-9]+)?)\s*(B|KB|MB|GB)\)",
    re.IGNORECASE,
)


def int8_gpt_path(models_dir: str | Path) -> Path:
    return Path(models_dir).expanduser().resolve() / INT8_GPT_LOCAL_FILENAME


def int8_gpt_expected_source(models_dir: str | Path) -> str:
    return (
        f"{int8_gpt_path(models_dir)} from Hugging Face "
        f"{INT8_GPT_REPO_ID}/{INT8_GPT_REMOTE_FILENAME}"
    )


def int8_fallback_warning(models_dir: str | Path, error: BaseException | str) -> str:
    return (
        "WARNING: INT8 ConvRot GPT download failed; falling back to the BF16 GPT. "
        f"Expected {int8_gpt_expected_source(models_dir)}. Download error: {error}"
    )


def _megabytes(value: str, unit: str) -> float:
    factors = {"B": 1 / 1024**2, "KB": 1 / 1024, "MB": 1.0, "GB": 1024.0}
    return float(value) * factors[unit.upper()]


def int8_download_progress_message(
    fraction: float,
    description: str = "",
    *,
    models_dir: str | Path | None = None,
) -> str:
    """Normalize downloader output into the UI/console INT8 progress contract."""

    match = _BYTE_PAIR_RE.search(str(description or ""))
    if match:
        current_mb = _megabytes(match.group(1), match.group(2))
        total_mb = _megabytes(match.group(3), match.group(4))
        return f"Downloading INT8 ConvRot GPT {current_mb:.1f}/{total_mb:.1f} MB"

    if models_dir is not None:
        path = int8_gpt_path(models_dir)
        if path.is_file():
            size_mb = path.stat().st_size / 1024**2
            return f"Downloading INT8 ConvRot GPT {size_mb:.1f}/{size_mb:.1f} MB"

    percent = max(0.0, min(1.0, float(fraction))) * 100.0
    return f"Downloading INT8 ConvRot GPT ({percent:.1f}%)"


@lru_cache(maxsize=1)
def _load_distribution_downloader() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    downloader_path = repo_root.parent / "Models_Downloader.py"
    if not downloader_path.is_file():
        raise RuntimeError(
            "The distribution model downloader is missing. Expected it at "
            f"{downloader_path}. Keep Models_Downloader.py beside the "
            f"{repo_root.name} application folder."
        )

    spec = importlib.util.spec_from_file_location(
        "indextts_distribution_model_downloader",
        downloader_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load model downloader from {downloader_path}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to import distribution model downloader at {downloader_path}: {exc}"
        ) from exc
    return module


def ensure_base_models(
    models_dir: str | Path,
    progress_cb: Optional[Callable] = None,
) -> str:
    """Ensure the official IndexTTS 2.5 model set is available."""
    models_path = Path(models_dir).expanduser().resolve()
    downloader = _load_distribution_downloader()
    downloader.download_models(
        str(models_path),
        model_type="index_tts_2_5",
        progress_cb=progress_cb,
    )
    return str(models_path)


def ensure_int8_gpt(
    models_dir: str | Path,
    progress_cb: Optional[Callable] = None,
) -> str:
    """Ensure base models and the optional INT8 ConvRot GPT are available."""
    models_path = Path(models_dir).expanduser().resolve()
    downloader = _load_distribution_downloader()
    result = downloader.download_models(
        str(models_path),
        model_type="index_tts_2_5_int8",
        progress_cb=progress_cb,
    )
    return str(result["int8_gpt"])


__all__ = [
    "INT8_GPT_LOCAL_FILENAME",
    "INT8_GPT_REMOTE_FILENAME",
    "INT8_GPT_REPO_ID",
    "ensure_base_models",
    "ensure_int8_gpt",
    "int8_download_progress_message",
    "int8_fallback_warning",
    "int8_gpt_expected_source",
    "int8_gpt_path",
]
