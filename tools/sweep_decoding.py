"""Sweep decoding settings (temperature, guidance rate, beams) for an existing training folder.

Example:
  python tools/sweep_decoding.py --adapter-dir loras/my_voice
renders the training's speech benchmark with the recommended checkpoint at alternative settings, keeps a change
only when it scores better than the defaults, and saves the winner as loras/my_voice/analysis/decoding.json, which
Voice Generation applies automatically with that LoRA / DoRA.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    from indextts.training.decoding_sweep import run_decoding_sweep
    from indextts.training.speech_eval import load_speech_evaluation
    from indextts.training.train_config import TrainConfig

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter-dir", required=True, help="Training folder with a completed speech benchmark")
    parser.add_argument("--checkpoint", default="", help="GPT checkpoint to sweep (default: the speech recommendation)")
    args = parser.parse_args(argv)
    adapter_dir = Path(args.adapter_dir).expanduser().resolve()
    train_config = adapter_dir / "train_config.json"
    report = load_speech_evaluation(adapter_dir)
    if not train_config.is_file() or report is None:
        raise SystemExit("this training folder has no completed speech benchmark; run the automatic speech comparison first")
    checkpoint = args.checkpoint or str(report.get("recommended_checkpoint") or "")
    if not checkpoint or not Path(checkpoint).is_file():
        raise SystemExit("the speech benchmark recommends the base model; pass --checkpoint to sweep a checkpoint")
    config = TrainConfig.from_json(train_config)
    state_dir = adapter_dir / "analysis" / "speech_evaluation" / "decoding_sweep" / "sweep_job"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "stop.flag").unlink(missing_ok=True)
    result = run_decoding_sweep(config, state_dir, checkpoint_path=checkpoint)
    print(result["summary_markdown"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
