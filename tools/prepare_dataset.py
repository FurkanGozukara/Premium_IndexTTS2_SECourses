from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from indextts.training.dataset_prep import DatasetPrepConfig, run_dataset_prep


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare clean 24 kHz mono IndexTTS LoRA training segments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", help="Media files, folders, metadata.csv, or wav+txt folders")
    parser.add_argument("--name", required=True, help="Dataset directory name")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--language", default="EN")
    parser.add_argument("--output-root", default="datasets")
    parser.add_argument(
        "--subtitle-policy",
        choices=("prefer_sidecar", "whisper_only", "sidecar_only"),
        default="prefer_sidecar",
    )
    parser.add_argument("--whisper-model", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--whisper-device", default="cuda:0")
    parser.add_argument("--align-with-whisper", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--target-s", type=float, default=8.0)
    parser.add_argument("--min-s", type=float, default=1.5)
    parser.add_argument("--max-s", type=float, default=15.0)
    parser.add_argument("--max-gap-ms", type=int, default=700)
    parser.add_argument(
        "--boundary-mode",
        choices=("sentence", "sentence_or_pause"),
        default="sentence",
    )
    parser.add_argument("--min-pause-boundary-ms", type=int, default=400)
    parser.add_argument("--pad-ms", type=int, default=60)
    parser.add_argument("--snap-to-silence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--snap-window-ms", type=int, default=200)
    parser.add_argument("--trim-silence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trim-top-db", type=float, default=40.0)
    parser.add_argument("--loudness-normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-lufs", type=float, default=-20.0)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--min-words", type=int, default=2)
    parser.add_argument("--max-words", type=int, default=80)
    parser.add_argument(
        "--remove-bracket-annotations",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dedupe-rolling-captions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--export-reference-candidates", type=int, default=5)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speaker-name", default="")
    parser.add_argument("--speaker-from-folder", action=argparse.BooleanOptionalAction, default=False)
    return parser


def config_from_args(args: argparse.Namespace) -> DatasetPrepConfig:
    return DatasetPrepConfig(
        name=args.name,
        inputs=args.inputs,
        recursive=args.recursive,
        language=args.language,
        output_root=args.output_root,
        subtitle_policy=args.subtitle_policy,
        whisper_model=args.whisper_model,
        whisper_device=args.whisper_device,
        align_with_whisper=args.align_with_whisper,
        target_s=args.target_s,
        min_s=args.min_s,
        max_s=args.max_s,
        max_gap_ms=args.max_gap_ms,
        boundary_mode=args.boundary_mode,
        min_pause_boundary_ms=args.min_pause_boundary_ms,
        pad_ms=args.pad_ms,
        snap_to_silence=args.snap_to_silence,
        snap_window_ms=args.snap_window_ms,
        trim_silence=args.trim_silence,
        trim_top_db=args.trim_top_db,
        loudness_normalize=args.loudness_normalize,
        target_lufs=args.target_lufs,
        sample_rate=args.sample_rate,
        min_words=args.min_words,
        max_words=args.max_words,
        remove_bracket_annotations=args.remove_bracket_annotations,
        dedupe_rolling_captions=args.dedupe_rolling_captions,
        export_reference_candidates=args.export_reference_candidates,
        overwrite=args.overwrite,
        max_segments=args.max_segments,
        seed=args.seed,
        speaker_name=args.speaker_name,
        speaker_from_folder=args.speaker_from_folder,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_dataset_prep(config_from_args(args))
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    return 0 if summary.status in {"complete", "cancelled"} else 1


if __name__ == "__main__":
    sys.exit(main())
