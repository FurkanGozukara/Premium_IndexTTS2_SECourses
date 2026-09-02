from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from indextts.training.train_config import TrainConfig
from indextts.training.trainer import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an IndexTTS 2.5 LoRA/DoRA adapter")
    parser.add_argument("--config", required=True, help="TrainConfig JSON file")
    parser.add_argument("--state-dir", help="Optional status/metrics directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = TrainConfig.from_json(args.config)
    result = run_training(config, state_dir=Path(args.state_dir) if args.state_dir else None)
    print(json.dumps(result.__dict__, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
