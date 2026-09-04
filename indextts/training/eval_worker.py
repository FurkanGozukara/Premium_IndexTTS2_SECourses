"""Subprocess entry point for measured checkpoint evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback
from typing import Any

from indextts.runtime.progress import ProgressReporter
from indextts.utils.atomic_json import read_json_retry, write_json_atomic
from indextts.utils.console_encoding import configure_console_output

from .checkpoint_eval import (
    CheckpointEvalConfig,
    evaluate_checkpoints,
    write_checkpoint_eval,
)


class EvalWorkerReporter(ProgressReporter):
    def __init__(self, state_dir: Path) -> None:
        super().__init__("checkpoints", progress_file=state_dir / "progress.json")
        self.state_dir = state_dir
        self.status_path = state_dir / "status.json"
        self.log_path = state_dir / "log.txt"
        self._status("initializing", "Preparing checkpoint evaluation")

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
        self._status("evaluating", str(name).replace("_", " "))

    def update(
        self,
        completed: int | float,
        total: int | None = None,
        desc: str = "",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = super().update(completed, total=total, desc=desc, extra=extra)
        self._status("evaluating", desc or self.stage or "Evaluating checkpoints")
        return payload

    def log(self, msg: str) -> None:
        line = str(msg)
        with self.log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
        super().log(line)

    def terminal(self, phase: str, message: str) -> None:
        self._status(phase, message)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IndexTTS checkpoint evaluation worker")
    parser.add_argument("--config", required=True, help="CheckpointEvalConfig JSON file")
    parser.add_argument("--state-dir", required=True, help="Directory for status and progress")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_console_output()
    args = parse_args(argv)
    state_dir = Path(args.state_dir).expanduser().resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    reporter = EvalWorkerReporter(state_dir)
    try:
        config = CheckpointEvalConfig.from_json(args.config)
        report = evaluate_checkpoints(
            config,
            reporter=reporter,
            cancel_callback=lambda: (state_dir / "stop.flag").is_file(),
        )
        report_path = write_checkpoint_eval(report, config.adapter_dir)
        message = (
            f"Evaluation complete: {len(report.rows)} rows; recommended "
            f"{report.best_label or 'not found'}"
        )
        reporter.terminal("complete", message)
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), flush=True)
        print(
            f">> Checkpoint evaluation summary | status=complete | items={len(report.rows)} | "
            f"elapsed={report.elapsed_s:.3f}s | output={report_path}",
            flush=True,
        )
        return 0
    except BaseException as exc:
        detail = traceback.format_exc()
        with (state_dir / "log.txt").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(detail.rstrip() + "\n")
        reporter.terminal("failed", str(exc))
        traceback.print_exc()
        print(
            f">> Checkpoint evaluation summary | status=failed | items={reporter.completed}/"
            f"{reporter.total or 0} | elapsed={time.perf_counter() - reporter.started_at:.3f}s | "
            f"output={state_dir}",
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
