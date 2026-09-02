from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from indextts.training.grid import GridConfig, run_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an IndexTTS checkpoint listening grid")
    parser.add_argument("--config", required=True, help="GridConfig JSON file")
    parser.add_argument("--state-dir", help="Optional exact grid output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = GridConfig.from_json(args.config)
    if args.state_dir:
        state = Path(args.state_dir).expanduser().resolve()
        config.output_root = str(state.parent)
        config.grid_name = state.name
    result = run_grid(config)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
