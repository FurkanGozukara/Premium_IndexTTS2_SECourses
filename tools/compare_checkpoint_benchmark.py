"""Render a training's speech benchmark with another checkpoint and compare it with the run's own measurement.

The development speech comparison measured the run's recommended checkpoint on frozen validation
sentences, seeds and reference. This tool renders the same benchmark with a different GPT checkpoint
(for example an averaged one, or the same checkpoint with a different voice decoder adapter) and reports
the paired differences: speaker similarity to the real recordings, strict transcript error, pause time,
and the deployment score the trainer selects with. Nothing in the run is modified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def resolve_baseline(run_dir: Path, baseline_arg: str, recommended: str) -> tuple[dict, list[dict], str, dict, str]:
    """Return (plan, baseline rows, label, inference, baseline path) for the comparison.

    A run whose speech comparison recommended the Base model has no checkpoint file to
    hash-verify, so ``--baseline-checkpoint`` left empty (or given as ``base``) compares with
    the run's own Base measurement instead of failing.
    """

    raw = str(baseline_arg or recommended or "").strip()
    if raw.lower() in {"", "base"}:
        root = run_dir / "analysis" / "speech_evaluation"
        plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
        full = json.loads((root / "report.json").read_text(encoding="utf-8"))
        base = next((row for row in full.get("candidates", []) if not row.get("path")), None)
        if base is None:
            raise SystemExit("the development speech benchmark has no Base measurement to compare with")
        label = str(base["label"])
        rows = [row for row in full.get("cells", []) if row.get("checkpoint") == label]
        inference = full.get("inference") if isinstance(full.get("inference"), dict) else {}
        return plan, rows, label, inference, ""
    from indextts.training.speech_eval import development_baseline

    baseline_checkpoint = str(Path(raw).expanduser().resolve())
    plan, rows, label, _report_path, inference = development_baseline(run_dir, baseline_checkpoint)
    return plan, rows, label, inference, baseline_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Training folder with analysis/speech_evaluation")
    parser.add_argument("--checkpoint", required=True, help="GPT checkpoint to render")
    parser.add_argument("--baseline-checkpoint", default="", help="Measured checkpoint to compare with (default: the speech recommendation)")
    parser.add_argument("--label", default="", help="Row label for the rendered checkpoint")
    parser.add_argument("--decoder", default="none", help="Voice decoder adapter: none, auto, or a .s2mel.safetensors path")
    parser.add_argument("--decoder-strength", type=float, default=1.0)
    parser.add_argument("--output", default="", help="Directory for grids and the comparison JSON (default: <run>/analysis/speech_evaluation/comparisons/<stamp>)")
    args = parser.parse_args()

    from indextts.training.train_config import TrainConfig
    from indextts.training.speech_eval import (_benchmark_infer_kwargs, _benchmark_runtime, _lenient_terms,
                                                load_speech_evaluation, render_benchmark_rows)
    from indextts.training.speech_metrics import deployment_score, paired_difference, summarize
    from indextts.utils.atomic_json import write_json_atomic

    run_dir = Path(args.run_dir).expanduser().resolve()
    config = TrainConfig.from_json(run_dir / "train_config.json")
    report = load_speech_evaluation(run_dir)
    if report is None:
        raise SystemExit("the run has no completed speech evaluation")
    plan, baseline_rows, baseline_label, inference, baseline_checkpoint = resolve_baseline(
        run_dir, args.baseline_checkpoint, str(report.get("recommended_checkpoint") or ""),
    )
    candidate = str(Path(args.checkpoint).expanduser().resolve())
    label = args.label or Path(candidate).stem
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(args.output).expanduser().resolve() if args.output else run_dir / "analysis" / "speech_evaluation" / "comparisons" / stamp
    out.mkdir(parents=True, exist_ok=True)
    state = out / "state"
    state.mkdir(exist_ok=True)
    runtime = _benchmark_runtime(config).to_dict()
    runtime["decoder_adapter"] = args.decoder
    runtime["decoder_adapter_strength"] = float(args.decoder_strength)
    infer = dict(inference.get("infer_kwargs") or {}) or _benchmark_infer_kwargs(config)
    started = time.perf_counter()

    def update(message: str, completed: int, total: int) -> None:
        print(f">> {message}: {completed}/{total}", flush=True)

    measured, grids = render_benchmark_rows(config, plan, run_dir=run_dir, checkpoint=candidate, label=label, runtime=runtime,
                                            infer=infer, attempt=out / "grids", out=out, state=state, update=update,
                                            cancelled=lambda: False, lenient=_lenient_terms(config, plan), baseline=baseline_rows)
    base_rows = [row for row in report["cells"] if row.get("checkpoint") == "Base"]
    base_summary = summarize(base_rows)
    rows = {"candidate": measured, "baseline": baseline_rows}
    result = {"run_dir": str(run_dir), "candidate": candidate, "candidate_label": label, "baseline": baseline_checkpoint,
              "baseline_label": baseline_label, "decoder": args.decoder, "decoder_strength": float(args.decoder_strength),
              "grids": grids, "elapsed_s": time.perf_counter() - started, "generated_at": datetime.now(timezone.utc).isoformat()}
    for name, table in rows.items():
        summary = summarize(table)
        speaker = paired_difference(table, base_rows, "speaker_similarity_real")
        error = paired_difference(table, base_rows, "error_rate")
        summary["deployment_score_vs_base"] = deployment_score(summary, base_summary, speaker["mean"], error["mean"])
        summary["speaker_gain_vs_base"] = speaker
        summary["error_delta_vs_base"] = error
        result[name + "_summary"] = summary
    for key, name in (("speaker_similarity_real", "speaker similarity to real"), ("error_rate", "transcript error"),
                      ("style_similarity_real", "style similarity to real"), ("pause_ratio_vs_real", "pause ratio")):
        result["paired_" + key] = paired_difference(measured, baseline_rows, key)
    result["cells"] = measured
    write_json_atomic(out / "comparison.json", result)
    c, b = result["candidate_summary"], result["baseline_summary"]
    print(f"\n{label} vs {baseline_label} on {len(measured)} benchmark clips (decoder {args.decoder}):")
    print(f"  speaker similarity to real   {c['speaker_similarity_real']:.4f} vs {b['speaker_similarity_real']:.4f}  paired {result['paired_speaker_similarity_real']['mean']:+.4f} ci {result['paired_speaker_similarity_real']['ci95']}")
    print(f"  mean transcript error        {100*c['mean_error_rate']:.2f}% vs {100*b['mean_error_rate']:.2f}%  paired {100*result['paired_error_rate']['mean']:+.2f} points ci {[round(100*x,2) for x in result['paired_error_rate']['ci95']]}")
    print(f"  style similarity to real     {c['style_similarity_real']:.4f} vs {b['style_similarity_real']:.4f}")
    print(f"  pause time vs real           {c['pause_ratio_vs_real']:.3f} vs {b['pause_ratio_vs_real']:.3f}")
    print(f"  deployment score vs Base     {c['deployment_score_vs_base']['score']:+.4f} vs {b['deployment_score_vs_base']['score']:+.4f}")
    print(f"  worst clip / failures        {100*c['worst_error_rate']:.1f}% / {c['failure_count']} vs {100*b['worst_error_rate']:.1f}% / {b['failure_count']}")
    print(f"  written to {out / 'comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
