"""Subprocess entry point for LoRA/DoRA training."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from .dataset_manifest import atomic_write_json
from .train_config import TrainConfig
from .trainer import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IndexTTS 2.5 adapter training worker")
    parser.add_argument("--config", required=True, help="TrainConfig JSON file")
    parser.add_argument("--state-dir", required=True, help="Directory for status and metrics")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state_dir = Path(args.state_dir).expanduser().resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    try:
        config = TrainConfig.from_json(args.config)
        result = run_training(config, state_dir=state_dir)
        print(json.dumps(result.__dict__, indent=2, ensure_ascii=False), flush=True)
        try:
            dataset_info = json.loads(
                (Path(config.dataset_dir) / "dataset_info.json").read_text(encoding="utf-8-sig")
            )
        except (OSError, UnicodeError, json.JSONDecodeError):
            dataset_info = {}
        audio_seconds = float(
            dataset_info.get(
                "total_duration_s",
                float(dataset_info.get("total_duration_minutes", 0.0) or 0.0) * 60.0,
            )
            or 0.0
        )
        print(
            f">> Training summary | status={result.status} | items={result.step}/{result.total_steps} | "
            f"audio={audio_seconds:.3f}s | elapsed={result.elapsed_s:.3f}s | "
            f"{result.avg_it_s:.3f} it/s | output={Path(result.output_path).parent}",
            flush=True,
        )
        return 0
    except BaseException as exc:
        traceback.print_exc()
        status_path = state_dir / "status.json"
        try:
            current = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            current = {}
        current.update(
            {
                "phase": "failed",
                "message": str(exc),
                "updated_at": time.time(),
            }
        )
        atomic_write_json(status_path, current)
        with (state_dir / "log.txt").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(f">> worker failed: {exc}\n")
        print(
            f">> Training summary | status=failed | items={int(current.get('step', 0) or 0)}/"
            f"{int(current.get('total_steps', 0) or 0)} | audio=0.000s | "
            f"elapsed={float(current.get('elapsed_s', 0.0) or 0.0):.3f}s | "
            f"{float(current.get('it_s', 0.0) or 0.0):.3f} it/s | output={state_dir}",
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
