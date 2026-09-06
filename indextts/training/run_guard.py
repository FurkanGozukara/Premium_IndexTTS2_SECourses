"""Protect fresh runs from accidentally combining previous training artifacts."""
from pathlib import Path
from typing import Any


def ensure_run_destination(config: Any, state_dir: str | Path | None = None) -> None:
    root = (Path(config.output_dir).expanduser().resolve() / config.name).resolve()
    continuing_here = bool(config.resume_from and config.resume_mode == "continue" and
                           Path(config.resume_from).expanduser().resolve().is_relative_to(root))
    roots = {root, Path(state_dir).resolve() if state_dir else root}
    for path in roots:
        trained = path.is_dir() and (any(path.rglob("*.safetensors")) or
                    any((path / name).is_file() and (path / name).stat().st_size for name in ("metrics.jsonl", "log.txt")))
        if trained and not continuing_here:
            raise FileExistsError(f"Training output already contains a run: {path}. Choose a new name for fresh training, or explicitly Continue run.")
