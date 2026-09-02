from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from indextts.training.checkpoint_eval import (
    CheckpointEvalConfig,
    evaluate_checkpoints,
    write_checkpoint_eval,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate IndexTTS adapter checkpoints")
    parser.add_argument("--config", required=True, help="CheckpointEvalConfig JSON file")
    parser.add_argument("--state-dir", help="Optional progress directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.state_dir:
        from indextts.training.eval_worker import main as worker_main

        return worker_main(
            ["--config", str(Path(args.config)), "--state-dir", str(Path(args.state_dir))]
        )
    config = CheckpointEvalConfig.from_json(args.config)
    report = evaluate_checkpoints(config)
    path = write_checkpoint_eval(report, config.adapter_dir)
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    print(f">> report saved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
