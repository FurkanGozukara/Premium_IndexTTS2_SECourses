"""Train a voice decoder (s2mel) adapter for an existing LoRA / DoRA training folder from its cached dataset.

Example:
  python tools/train_decoder_adapter.py --dataset-dir datasets/my_voice --adapter-dir loras/my_voice
writes loras/my_voice/my_voice.s2mel.safetensors, which Voice Generation loads automatically with any checkpoint
of that training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from indextts.lora.decoder import DECODER_ADAPTER_SUFFIX
from indextts.training.decoder_adapter import DecoderAdapterConfig, reject_decoder_adapter, train_decoder_adapter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True, help="Cached training dataset (manifest.jsonl and cache/)")
    parser.add_argument("--adapter-dir", required=True, help="Training folder of the GPT LoRA / DoRA; the decoder adapter is saved inside it")
    parser.add_argument("--name", default="", help="Adapter name (defaults to the folder name)")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--adapter-type", choices=("lora", "dora"), default="dora")
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=128.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--val-split-mode", choices=("record", "source"), default="source")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing decoder adapter file")
    parser.add_argument("--no-test", action="store_true",
                        help="Skip the full-pipeline test against the training's speech benchmark (the adapter is kept regardless)")
    parser.add_argument("--checkpoint", default="", help="GPT checkpoint for the full-pipeline test (default: the speech recommendation)")
    parser.add_argument("--code-source", choices=("real", "gpt", "mixed"), default="real",
                        help="Semantic codes the targets are rendered from: the recordings' (real), the GPT checkpoint's teacher-forced predictions (gpt), or half of each (mixed)")
    parser.add_argument("--gpt-checkpoint", default="", help="GPT checkpoint whose codes are predicted for --code-source gpt/mixed (default: --checkpoint)")
    args = parser.parse_args(argv)
    adapter_dir = Path(args.adapter_dir).expanduser().resolve()
    name = args.name or adapter_dir.name
    output = adapter_dir / f"{name}{DECODER_ADAPTER_SUFFIX}"
    if output.is_file() and not args.overwrite:
        raise SystemExit(f"decoder adapter already exists: {output} (pass --overwrite to replace it)")
    gpt_checkpoint = args.gpt_checkpoint or args.checkpoint
    if args.code_source != "real" and not gpt_checkpoint:
        raise SystemExit("--code-source gpt/mixed needs --gpt-checkpoint (or --checkpoint) to predict the codes with")
    config = DecoderAdapterConfig(
        dataset_dir=str(Path(args.dataset_dir).expanduser().resolve()), output_path=str(output), name=name,
        model_dir=args.model_dir, model_config=str(Path(args.model_dir) / "config.yaml"), device=args.device,
        adapter_type=args.adapter_type, rank=args.rank, alpha=args.alpha, epochs=args.epochs, max_steps=args.max_steps,
        learning_rate=args.learning_rate, val_split_mode=args.val_split_mode, val_fraction=args.val_fraction, seed=args.seed,
        code_source=args.code_source, gpt_checkpoint=str(Path(gpt_checkpoint).expanduser().resolve()) if gpt_checkpoint else "")
    state_dir = adapter_dir / "analysis" / "decoder_adapter_job"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "stop.flag").unlink(missing_ok=True)
    result = train_decoder_adapter(config, state_dir, cancel_callback=lambda: (state_dir / "stop.flag").is_file())
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    if result.status not in {"complete", "stopped"}:
        return 1
    if args.no_test:
        return 0
    verdict = full_pipeline_test(adapter_dir, output, args.checkpoint)
    if verdict is not None and not verdict["accepted"]:
        parked = reject_decoder_adapter(output)
        print(f">> decoder adapter not installed ({'; '.join(verdict['reasons'])}); parked at {parked}")
        return 2
    return 0


def full_pipeline_test(adapter_dir: Path, adapter_path: Path, checkpoint: str = "") -> dict | None:
    """Judge the adapter through the full pipeline when the training folder has a speech benchmark."""
    from indextts.training.speech_eval import load_speech_evaluation, run_decoder_test
    from indextts.training.train_config import TrainConfig
    train_config = adapter_dir / "train_config.json"
    report = load_speech_evaluation(adapter_dir)
    if not train_config.is_file() or report is None:
        print(">> no speech benchmark in this training folder; the decoder adapter is kept without a full-pipeline test")
        return None
    checkpoint = checkpoint or str(report.get("recommended_checkpoint") or "")
    if not checkpoint or not Path(checkpoint).is_file():
        print(">> the speech benchmark recommends the base model; pass --checkpoint to test the decoder adapter with a checkpoint")
        return None
    config = TrainConfig.from_json(train_config)
    state_dir = adapter_dir / "analysis" / "speech_evaluation" / "decoder_test" / "test_job"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "stop.flag").unlink(missing_ok=True)
    verdict = run_decoder_test(config, state_dir, checkpoint_path=checkpoint, adapter_path=str(adapter_path))
    print(verdict["summary_markdown"])
    return verdict


if __name__ == "__main__":
    raise SystemExit(main())
