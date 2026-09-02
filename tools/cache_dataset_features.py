from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from indextts.training.features import FeatureCacheConfig, cache_dataset_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache IndexTTS 2.5 training features")
    parser.add_argument("dataset_dir_pos", nargs="?", help="Dataset directory containing manifest.jsonl")
    parser.add_argument("--dataset-dir", dest="dataset_dir", help="Dataset directory containing manifest.jsonl")
    parser.add_argument("--config", help="Optional FeatureCacheConfig JSON file")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--semantic-layer", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--verify-count", type=int, default=None)
    parser.add_argument("--verify-output-dir", default=None)
    parser.add_argument("--no-skip-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload: dict[str, object] = {}
    if args.config:
        with Path(args.config).open("r", encoding="utf-8-sig") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise TypeError("feature cache config JSON must contain an object")
        payload.update(loaded)
    dataset_dir = args.dataset_dir or args.dataset_dir_pos
    if dataset_dir:
        payload["dataset_dir"] = dataset_dir
    if not payload.get("dataset_dir"):
        raise SystemExit("dataset_dir is required (positional argument or config JSON)")
    for argument, field in (
        (args.model_dir, "model_dir"),
        (args.model_config, "model_config"),
        (args.device, "device"),
        (args.semantic_layer, "semantic_layer"),
        (args.batch_size, "batch_size"),
        (args.max_items, "max_items"),
        (args.verify_count, "verify_count"),
        (args.verify_output_dir, "verify_output_dir"),
    ):
        if argument is not None:
            payload[field] = argument
    if args.no_skip_existing:
        payload["skip_existing"] = False
    config = FeatureCacheConfig.from_dict(payload)
    summary = cache_dataset_features(config)
    print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False))
    return 0 if not summary.cancelled else 2


if __name__ == "__main__":
    raise SystemExit(main())
