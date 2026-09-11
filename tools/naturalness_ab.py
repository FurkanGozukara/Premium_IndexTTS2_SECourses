"""Render one trained voice under several decoding or conditioning variants and compare their naturalness.

Every variant renders the same held-out sentences (the run's final test by default) with the same
seeds, in the app's own generation path (the listening-grid worker). The clips are measured against
the person's real recordings: transcript error, speaker and style similarity, pause time, and the
prosody statistics of ``indextts.training.prosody_metrics`` (pitch variability, loudness dynamics,
articulation rate). A report and blind listening bundles are written to the output folder.

Examples
    python tools/naturalness_ab.py --run-dir loras/Furkan_EN_DoRA_r128_v11
    python tools/naturalness_ab.py --run-dir loras/Furkan_EN_DoRA_r128_v11 --variants sampling no_decoder no_commas
    python tools/naturalness_ab.py --run-dir loras/Furkan_EN_DoRA_r128_v11 --expressive-reference datasets/.../segments/clip.wav
    python tools/naturalness_ab.py --run-dir loras/Furkan_EN_DoRA_r128_v11 --checkpoints epoch_003 epoch_005 final
    python tools/naturalness_ab.py --measure-only outputs/naturalness_ab/20260910_180000
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _deployment(run_dir: Path) -> dict:
    """Checkpoint, decoder, speaking rate and reference the app would use for this run."""

    from indextts.lora.decoder import find_decoder_adapter, recommended_decoder_strength
    from indextts.training.speaking_rate import load_speaking_rate

    report_path = run_dir / "analysis" / "speech_evaluation" / "report.json"
    checkpoint = ""
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        checkpoint = str(report.get("recommended_checkpoint") or "")
    if not checkpoint or not Path(checkpoint).is_file():
        candidates = sorted(run_dir.glob("*.safetensors"))
        candidates = [item for item in candidates if not item.name.endswith(".s2mel.safetensors")]
        checkpoint = str(candidates[-1]) if candidates else ""
    rate = load_speaking_rate(checkpoint) if checkpoint else None
    # The speaker reference is <name>_reference.wav; the expressive clip (<name>_expressive_reference.wav) is not it.
    reference = next(iter(sorted(p for p in run_dir.glob("*_reference.wav") if "_expressive_reference" not in p.name)), None)
    return {
        "checkpoint": checkpoint,
        "decoder": find_decoder_adapter(checkpoint) if checkpoint else "",
        "decoder_strength": (recommended_decoder_strength(checkpoint) or 1.0) if checkpoint else 1.0,
        "speaking_rate": float(rate.recommended_speaking_rate) if rate is not None else 1.0,
        "reference": str(reference) if reference else "",
    }


def _resolve_checkpoints(run_dir: Path, names: list[str], deployed: str) -> list[tuple[str, str]]:
    if not names:
        return [("deployed", deployed)]
    resolved: list[tuple[str, str]] = []
    for name in names:
        if name in {"deployed", "final"} and name == "deployed":
            resolved.append(("deployed", deployed))
            continue
        if name.lower() == "base":
            resolved.append(("base", ""))  # the base model without an adapter, rendered with the run's reference
            continue
        path = Path(name)
        if not path.is_file():
            matches = [item for item in run_dir.glob("*.safetensors") if name in item.stem and not item.name.endswith(".s2mel.safetensors")]
            if name == "final":
                matches = [item for item in run_dir.glob("*.safetensors") if "epoch" not in item.stem and "avg" not in item.stem and not item.name.endswith(".s2mel.safetensors")]
            if not matches:
                raise SystemExit(f"no checkpoint matches {name!r} in {run_dir}")
            path = sorted(matches)[-1]
        resolved.append((path.stem, str(path.resolve())))
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", help="Training folder (loras/<name>) with train_config.json")
    parser.add_argument("--checkpoints", nargs="*", default=[], help="Checkpoint names or paths to render (default: the deployed one)")
    parser.add_argument("--variants", nargs="*", default=None, help="Variant names from the default table, or one JSON file; default: every default variant")
    parser.add_argument("--sentences", default="final_test", help="final_test | development | path to a JSON list with text and real_audio")
    parser.add_argument("--limit", type=int, default=0, help="Use only the first N sentences")
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 104771, 209500])
    parser.add_argument("--expressive-reference", default="", help="An expressive clean clip; adds the expressive_* variants")
    parser.add_argument("--reference", default="", help="Speaker reference to use instead of the run's saved one (for fair cross-run comparisons)")
    parser.add_argument("--output", default="", help="Output folder (default outputs/naturalness_ab/<stamp>)")
    parser.add_argument("--bundle-variants", nargs="*", default=None, help="Variants to put into the blind listening sets (default: all)")
    parser.add_argument("--max-sets", type=int, default=0, help="Limit the number of blind sets")
    parser.add_argument("--measure-only", default="", help="Re-measure an existing output folder without rendering")
    args = parser.parse_args()

    from indextts.training.naturalness_ab import (
        build_listening_bundles, load_sentences, load_variants, render_report_markdown, summarize_variant, transform_text,
    )
    from indextts.training.prosody_metrics import measure_prosody
    from indextts.utils.atomic_json import write_json_atomic

    if args.measure_only:
        out = Path(args.measure_only).expanduser().resolve()
        state = json.loads((out / "render_state.json").read_text(encoding="utf-8"))
        run_dir = Path(state["run_dir"])
        rows_by_variant = state["rows_by_variant"]
        report_meta = state["report_meta"]
    else:
        if not args.run_dir:
            raise SystemExit("--run-dir is required unless --measure-only is used")
        run_dir = Path(args.run_dir).expanduser().resolve()
        from indextts.runtime import ProgressReporter
        from indextts.training.grid import GridCheckpoint, GridConfig, run_grid
        from indextts.training.speech_eval import _benchmark_infer_kwargs, _benchmark_runtime
        from indextts.training.train_config import TrainConfig

        config = TrainConfig.from_json(run_dir / "train_config.json")
        deployment = _deployment(run_dir)
        if args.reference:
            deployment["reference"] = str(Path(args.reference).expanduser().resolve())
        if not deployment["checkpoint"]:
            raise SystemExit("no GPT checkpoint found in the run folder")
        if not deployment["reference"] or not Path(deployment["reference"]).is_file():
            raise SystemExit("no <name>_reference.wav found in the run folder (or --reference does not exist)")
        checkpoints = _resolve_checkpoints(run_dir, list(args.checkpoints), deployment["checkpoint"])
        sentences = load_sentences(run_dir, args.sentences, limit=args.limit)
        if not sentences:
            raise SystemExit("no held-out sentences with real recordings were found")
        variant_source = None
        if args.variants and len(args.variants) == 1 and Path(args.variants[0]).is_file():
            variant_source = args.variants[0]
        variants = load_variants(variant_source, expressive_reference=args.expressive_reference or None,
                                 include_extra=bool(args.variants) and variant_source is None)
        if args.variants and variant_source is None:
            wanted = set(args.variants)
            variants = [variant for variant in variants if variant.name in wanted]
            missing = wanted - {variant.name for variant in variants}
            if missing:
                raise SystemExit(f"unknown variants: {', '.join(sorted(missing))}")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out = Path(args.output).expanduser().resolve() if args.output else REPO_ROOT / "outputs" / "naturalness_ab" / stamp
        out.mkdir(parents=True, exist_ok=True)
        base_runtime = _benchmark_runtime(config).to_dict()
        base_infer = _benchmark_infer_kwargs(config)
        base_infer["latent_multiplier"] = round(1.72 / deployment["speaking_rate"], 4)
        language = sentences[0].get("language", "EN")
        rows_by_variant: dict[str, list[dict]] = {}
        started = time.perf_counter()
        for label, checkpoint in checkpoints:
            for variant in variants:
                name = variant.name if len(checkpoints) == 1 else f"{label}__{variant.name}"
                runtime = dict(base_runtime)
                decoder = variant.decoder if variant.decoder is not None else (deployment["decoder"] or "none")
                runtime["decoder_adapter"] = decoder if checkpoint else "none"
                runtime["decoder_adapter_strength"] = float(variant.decoder_strength if variant.decoder_strength is not None else deployment["decoder_strength"])
                infer = dict(base_infer)
                if not checkpoint:
                    infer["latent_multiplier"] = 1.72  # the base model speaks at its own pace
                infer.update(variant.infer)
                if variant.emotion_reference:
                    infer["emo_audio_prompt"] = variant.emotion_reference
                    infer["emo_alpha"] = float(variant.emo_alpha if variant.emo_alpha is not None else 0.65)
                reference = variant.reference or deployment["reference"]
                texts = [transform_text(item["text"], variant.text_transform) for item in sentences]
                print(f">> rendering {name}: {len(texts)} sentences x {len(args.seeds)} seeds", flush=True)
                grid_config = GridConfig(
                    adapter_dir=str(run_dir), checkpoints=[GridCheckpoint(label, checkpoint)], references=[reference],
                    texts=texts, language=language, seeds=list(args.seeds), seed=args.seeds[0],
                    output_root=str(out / "grids"), grid_name=name,
                    runtime={"runtime": runtime, "model_dir": config.model_dir, "cfg_path": config.model_config, "use_qwen_emo": False},
                    infer_kwargs=infer, include_verdicts=False,
                )
                result = run_grid(grid_config, reporter=ProgressReporter(name, progress_file=out / "progress.json"))
                if result.status != "complete":
                    raise SystemExit(f"{name}: grid {result.status}")
                rows: list[dict] = []
                for cell in result.cells:
                    sentence = sentences[cell.text_index - 1]
                    rows.append({
                        "audio": cell.audio_path, "reference": reference, "real_audio": sentence["real_audio"],
                        "text": sentence["text"], "spoken_text": texts[cell.text_index - 1], "language": language,
                        "kind": "matched", "prompt_id": sentence["id"], "source": "", "seed": cell.seed, "checkpoint": name,
                    })
                rows_by_variant[name] = rows
                gc.collect()
        report_meta = {
            "run_dir": str(run_dir), "checkpoint": deployment["checkpoint"], "checkpoints": checkpoints,
            "deployment": deployment, "seeds": list(args.seeds), "sentence_count": len(sentences),
            "variant_table": [variant.to_dict() for variant in variants], "render_elapsed_s": time.perf_counter() - started,
            "model_dir": config.model_dir, "model_config": config.model_config, "device": config.device,
        }
        write_json_atomic(out / "render_state.json", {"run_dir": str(run_dir), "rows_by_variant": rows_by_variant, "report_meta": report_meta}, indent=2)

    # ---- measurement -----------------------------------------------------------------------
    from indextts.training.speech_eval import _lenient_terms
    from indextts.training.speech_metrics import measure_clips
    from indextts.training.train_config import TrainConfig

    config = TrainConfig.from_json(run_dir / "train_config.json")
    try:
        plan = json.loads((run_dir / "analysis" / "speech_evaluation" / "plan.json").read_text(encoding="utf-8"))
        lenient = _lenient_terms(config, plan)
    except (OSError, ValueError):
        lenient = set()

    def update(message: str, completed: int, total: int) -> None:
        print(f">> {message}: {completed}/{total}", flush=True)

    real_prosody: dict[str, dict] = {}
    variants_summary: dict[str, dict] = {}
    measured_rows: dict[str, list[dict]] = {}
    for name, rows in rows_by_variant.items():
        clips = [dict(row) for row in rows]
        measured = measure_clips(clips, model_dir=report_meta["model_dir"], model_config=report_meta["model_config"],
                                 device=report_meta["device"], output_dir=out / "measure", update=update,
                                 cancelled=lambda: False, lenient_terms=lenient)
        by_key = {(row["prompt_id"], int(row["seed"])): row for row in measured}
        for row in rows:
            entry = by_key.get((row["prompt_id"], int(row["seed"])), {})
            row.update({key: value for key, value in entry.items() if key not in row or key in {"error_rate", "errors", "units"}})
            real = row["real_audio"]
            if real not in real_prosody:
                real_prosody[real] = measure_prosody(real, row["text"])
            row["real_prosody"] = real_prosody[real]
            row["prosody"] = measure_prosody(row["audio"], row["text"])
        variants_summary[name] = summarize_variant(rows)
        measured_rows[name] = rows
        print(f">> {name}: liveliness {variants_summary[name]['prosody'].get('liveliness_ratio')} | "
              f"pause ratio {variants_summary[name].get('pause_ratio_vs_real_mean')} | "
              f"word error {variants_summary[name]['speech'].get('corpus_error_rate')}", flush=True)
    report = dict(report_meta)
    report["variants"] = variants_summary
    report["rows"] = measured_rows
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(out / "report.json", report, indent=1)
    (out / "report.md").write_text(render_report_markdown(report), encoding="utf-8")
    keys = build_listening_bundles(measured_rows, out / "listening", variants=args.bundle_variants, max_sets=args.max_sets)
    print(f">> report: {out / 'report.md'} | blind sets: {len(keys)} under {out / 'listening' / 'blind'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
