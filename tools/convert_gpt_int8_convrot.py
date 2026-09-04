"""Command-line wrapper for the IndexTTS GPT INT8 ConvRot converter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from indextts.quant.convrot_int8 import convert_gpt_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert IndexTTS 2.5 gpt.pth to ComfyUI INT8 ConvRot safetensors."
    )
    parser.add_argument("--src", default="models/gpt.pth", help="source torch checkpoint")
    parser.add_argument(
        "--dst",
        default="models/gpt_int8_convrot.safetensors",
        help="destination safetensors file",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="JSON report path (default: destination with .report.json suffix)",
    )
    parser.add_argument(
        "--device", default="cuda", help="conversion device, for example cuda or cpu"
    )
    parser.add_argument(
        "--group-sizes",
        type=int,
        nargs="+",
        default=(256, 64, 16),
        help="regular-Hadamard group sizes to search",
    )
    parser.add_argument(
        "--no-mse-clip", action="store_true", help="use absmax scales instead of HQ MSE clipping"
    )
    parser.add_argument(
        "--quantize-emo-encoder",
        action="store_true",
        help="also quantize eligible Linear weights in emo_conditioning_encoder",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = convert_gpt_checkpoint(
        args.src,
        args.dst,
        group_sizes=tuple(args.group_sizes),
        mse_clip=not args.no_mse_clip,
        device=args.device,
        report_path=args.report,
        quantize_emo_encoder=args.quantize_emo_encoder,
    )
    summary = {key: value for key, value in report.items() if key not in {"layers", "metadata"}}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
