"""Subprocess entry point for listening-grid generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback
from typing import Any

from indextts.runtime.progress import ProgressReporter
from indextts.utils.atomic_json import read_json_retry, write_json_atomic

from .grid import GridConfig, run_grid


class GridWorkerReporter(ProgressReporter):
    def __init__(self, state_dir: Path) -> None:
        super().__init__("cells", progress_file=state_dir / "progress.json")
        self.status_path = state_dir / "status.json"
        self._status("initializing", "Preparing listening grid")

    def _status(self, phase: str, message: str) -> None:
        current = read_json_retry(self.status_path, {}) or {}
        current.update(
            {
                "phase": phase,
                "message": message,
                "completed": int(self.completed),
                "total": int(self.total or 0),
                "elapsed_s": time.perf_counter() - self.started_at,
                "updated_at": time.time(),
            }
        )
        write_json_atomic(self.status_path, current, indent=2, ensure_ascii=False)

    def set_stage(self, name: str) -> None:
        super().set_stage(name)
        phase = "generating" if name == "generating" else "initializing"
        self._status(phase, str(name).replace("_", " "))

    def update(
        self,
        completed: int | float,
        total: int | None = None,
        desc: str = "",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = super().update(completed, total=total, desc=desc, extra=extra)
        self._status("generating", desc or "Generating listening grid")
        return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IndexTTS listening-grid worker")
    parser.add_argument("--config", required=True, help="GridConfig JSON file")
    parser.add_argument("--state-dir", required=True, help="Grid output/status directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    state_dir = Path(args.state_dir).expanduser().resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    reporter = GridWorkerReporter(state_dir)
    try:
        config = GridConfig.from_json(args.config)
        config.output_root = str(state_dir.parent)
        config.grid_name = state_dir.name
        config.validate()
        result = run_grid(
            config,
            reporter=reporter,
            cancel_callback=lambda: (state_dir / "stop.flag").is_file(),
        )
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), flush=True)
        print(
            f">> Listening grid summary | status={result.status} | items="
            f"{sum(cell.status == 'complete' for cell in result.cells)}/{len(result.cells)} | "
            f"elapsed={result.elapsed_s:.3f}s | output={result.grid_dir}",
            flush=True,
        )
        return 0
    except BaseException as exc:
        detail = traceback.format_exc()
        with (state_dir / "log.txt").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(detail.rstrip() + "\n")
        current = read_json_retry(state_dir / "status.json", {}) or {}
        current.update(phase="failed", message=str(exc), updated_at=time.time())
        write_json_atomic(state_dir / "status.json", current, indent=2, ensure_ascii=False)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

