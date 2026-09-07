"""Run a frozen, dataset-specific speech benchmark after training releases memory."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback
from typing import Any, Mapping, Sequence

from .dataset_manifest import atomic_write_json


def load_speech_evaluation(run_dir: str | Path) -> dict[str, Any] | None:
    try:
        report = json.loads((Path(run_dir) / "analysis" / "speech_evaluation" / "report.json").read_text(encoding="utf-8"))
        return report if report.get("status") == "complete" else None
    except (OSError, ValueError, TypeError):
        return None


def shortlist_checkpoints(run_dir: str | Path, limit: int) -> list[dict[str, Any]]:
    from .analysis import discover_checkpoints
    from .checkpoint_eval import load_checkpoint_eval
    root = Path(run_dir).resolve()
    report = load_checkpoint_eval(root)
    measured = {str(Path(r.path).resolve()): r for r in report.rows
                if r.path and abs(r.strength - 1) < 1e-9 and r.val_loss is not None} if report else {}
    base_loss = next((r.val_loss for r in report.rows if r.kind == "base"), None) if report else None
    entries = []
    for item in discover_checkpoints(root):
        path = Path(item["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Speech candidates must belong to this training run")
        row = measured.get(str(path))
        entries.append({"path": str(path), "label": str(item["label"]), "steps": int(item.get("steps", 0)),
                        "val_loss": row.val_loss if row else item.get("val_loss")})
    entries.sort(key=lambda r: (r["val_loss"] if r["val_loss"] is not None else float("inf"), -r["steps"], r["path"]))
    distinct = []
    seen = set()
    for item in entries:
        # Best/final/epoch files can contain exactly the same training update.
        identity = item["steps"] if item["steps"] else item["path"]
        if identity not in seen:
            distinct.append(item)
            seen.add(identity)
    if not distinct:
        raise ValueError("No checkpoints from this run are available for speech evaluation")
    selected = distinct[:max(1, limit)]
    latest = max(distinct, key=lambda r: r["steps"])
    if limit > 1 and latest not in selected:
        selected[-1] = latest
    return [{"label": "Base", "path": "", "steps": 0, "val_loss": base_loss}, *selected]


def report_markdown(report: dict[str, Any]) -> str:
    heading = "Frozen selection on final test" if report.get("final_test") else "Speech recommendation"
    real_metric = report.get("speaker_metric") == "speaker_similarity_real"
    lines = [f"**{heading}: {report['recommended_label']}**", report["scope"], "",
             "| Candidate | Mean transcript error | Worst clip | Speaker similarity vs real | Speaker similarity vs reference | Pause time vs real | Flagged clips | Eligible |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in report["candidates"]:
        speaker = f"{row['speaker_similarity']:.3f}" if row.get("speaker_similarity") is not None else "unavailable"
        speaker_real = f"{row['speaker_similarity_real']:.3f}" if row.get("speaker_similarity_real") is not None else "unavailable"
        pause = f"{row['pause_ratio_vs_real']:.2f}" if row.get("pause_ratio_vs_real") is not None else "unavailable"
        lines.append(f"| {row['label']} | {row['mean_error_rate']:.1%} | {row['worst_error_rate']:.1%} | {speaker_real} | {speaker} | {pause} | {row['failure_count']}/{row['clips']} | {'yes' if row['eligible'] else 'no'} |")
    lines.extend(["", report["decision"], "Transcript error uses words for EN/ES/AR and characters for ZH/JA; the dataset's own spellings of names and terms are accepted.",
                  "Pause time vs real divides the generated clips' internal pause time (silences of at least 120 ms between words and sentences) by the real recordings' on the same sentences: 1.00 matches the person, above 1 pauses longer, below 1 rushes. Shown for information; it does not affect selection.",
                  ("The speaker guard compares each generated sentence with the real recording of that sentence, the speaker's actual identity; "
                   "similarity to the single reference clip rewards copying that prompt and is shown for information.") if real_metric else
                  "Too few matched real recordings for a real-recording speaker comparison; the guard uses similarity to the reference clip."])
    for row in report["candidates"]:
        if row["path"]:
            delta = row["error_delta_vs_base"]
            lines.append(f"- {row['label']}: paired error change {delta['mean']:+.1%}, prompt-bootstrap 95% interval "
                         f"[{delta['ci95'][0]:+.1%}, {delta['ci95'][1]:+.1%}]. " + "; ".join(row["rejection_reasons"]))
    real = report.get("real_recordings")
    if real:
        lines.append(f"Real held-out recordings have {real['mean_error_rate']:.1%} mean ASR error on the same texts; ASR itself is imperfect.")
    lines.extend(["", *report.get("warnings", []),
                  "Intervals describe the evaluated prompts, not unseen people or recording sessions. "
                  "ASR edge/repetition flags and embeddings are automated proxies; no listening ratings were supplied."])
    if report.get("final_test_status"):
        lines.extend(["", f"**Independent final test: {report['final_test_status']}**. " + report.get("final_test_message", "")])
    if report.get("listening_review"):
        lines.extend(["", "A blind listening form is saved as `listening_review.html` beside this report. "
                      "Open it to play each matched prompt and seed and export your ratings."])
    return "\n".join(lines)


def run_speech_evaluation(config: Any, state_dir: str | Path, *,
                          frozen_selection: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    import torch
    from indextts.runtime import ProgressReporter, gpu_free_gb, gpu_total_gb, resolve_preset
    from .grid import GridCheckpoint, GridConfig, run_grid
    from .sampling import SAMPLE_FIXED_INFER_KWARGS
    from .speech_metrics import measure_clips, select_recommendation, summarize
    run_dir = Path(config.output_dir).resolve() / config.name
    root = run_dir / "analysis" / "speech_evaluation"
    final_test = frozen_selection is not None
    if final_test:
        root = root / "final_test"
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    if not plan["groups"]:
        raise ValueError("The speech benchmark contains no held-out prompts")
    state = Path(state_dir)
    state.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    def cancelled() -> bool:
        return (state / "stop.flag").exists() or (run_dir / "stop.flag").exists()
    def update(message: str, completed: int, total: int) -> None:
        value = {"phase": "evaluating_speech", "message": message, "desc": message, "completed": completed,
                 "total": total, "fraction": completed / total if total else 0, "elapsed_s": time.perf_counter()-started,
                 "updated_at": time.time()}
        atomic_write_json(state / "status.json", value)
        atomic_write_json(state / "progress.json", value)
        print(f">> {message}: {completed}/{total}", flush=True)
    candidates = [dict(row) for row in frozen_selection] if final_test else shortlist_checkpoints(run_dir, plan["candidate_limit"])
    for candidate in candidates:
        if candidate["path"]:
            candidate["sha256"] = hashlib.sha256(Path(candidate["path"]).read_bytes()).hexdigest()
    runtime = _benchmark_runtime(config)
    infer = _benchmark_infer_kwargs(config)
    attempt = root / "grids" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    clips, real_clips, grids = [], [], []
    for group in plan["groups"]:
        update(f"Generating speech for {group['speaker']} ({group['language']})", 0, len(group["prompts"]) * len(plan["seeds"]) * len(candidates))
        if hashlib.sha256(Path(group["reference"]).read_bytes()).hexdigest() != group["reference_sha256"]:
            raise ValueError("The frozen training reference has changed")
        for prompt in group["prompts"]:
            if prompt.get("audio_sha256") and hashlib.sha256(Path(prompt["audio"]).read_bytes()).hexdigest() != prompt["audio_sha256"]:
                raise ValueError("A frozen evaluation recording has changed")
        grid_config = GridConfig(adapter_dir=str(run_dir),
            checkpoints=[GridCheckpoint(row["label"], row["path"]) for row in candidates],
            references=[group["reference"]], texts=[p["text"] for p in group["prompts"]], language=group["language"],
            seeds=plan["seeds"], seed=plan["seeds"][0], output_root=str(attempt), grid_name=group["id"],
            runtime={"runtime": runtime.to_dict(), "model_dir": config.model_dir, "cfg_path": config.model_config, "use_qwen_emo": False},
            infer_kwargs=infer, include_verdicts=False)
        result = run_grid(grid_config, reporter=ProgressReporter("speech clips", progress_file=state / "progress.json"), cancel_callback=cancelled)
        if result.status != "complete":
            raise InterruptedError(f"Speech generation {result.status}")
        grids.append(result.grid_dir)
        for cell in result.cells:
            prompt = group["prompts"][cell.text_index - 1]
            clips.append({"audio": cell.audio_path, "reference": group["reference"], "real_audio": prompt["audio"],
                          "text": prompt["text"], "language": group["language"], "kind": prompt["kind"],
                          "prompt_id": f"{group['id']}:{prompt['id']}", "source": prompt["source"],
                          "seed": cell.seed, "checkpoint": cell.checkpoint_label if cell.checkpoint_path else "Base"})
        for prompt in group["prompts"]:
            if prompt["audio"]:
                real_clips.append({"audio": prompt["audio"], "reference": group["reference"], "real_audio": "",
                                   "text": prompt["text"], "language": group["language"], "kind": "real",
                                   "prompt_id": f"{group['id']}:{prompt['id']}", "source": prompt["source"],
                                   "seed": 0, "checkpoint": "Real recordings"})
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    measured = measure_clips([*clips, *real_clips], model_dir=config.model_dir, model_config=config.model_config,
                             device=config.device, output_dir=root, update=update, cancelled=cancelled,
                             lenient_terms=_lenient_terms(config, plan))
    generated = [row for row in measured if row["kind"] != "real"]
    real = [row for row in measured if row["kind"] == "real"]
    report = select_recommendation(candidates, generated, plan["policy"])
    if final_test:
        chosen = candidates[-1]
        measured_chosen = next(row for row in report["candidates"] if row["label"] == chosen["label"])
        report.update(final_test=True, recommended_label=chosen["label"], recommended_checkpoint=chosen["path"],
                      recommended_kind="adapter" if chosen["path"] else "base",
                      final_test_status="passed observed regression guards" if measured_chosen["eligible"] else "regression detected",
                      final_test_message="; ".join(measured_chosen["rejection_reasons"]),
                      scope="Independent final-test measurements for the already selected checkpoint. These results do not reselect a checkpoint.",
                      decision="The development recommendation was frozen before any final-test audio was generated.")
    report.update(dataset_identity=plan["dataset_identity"], plan=str(root / "plan.json"), grids=grids,
                  cells=generated, real_cells=real, real_recordings=summarize(real) if real else None,
                  warnings=plan["warnings"], seeds=plan["seeds"], inference=grid_config.to_dict(),
                  generated_at=datetime.now(timezone.utc).isoformat(), elapsed_s=time.perf_counter()-started)
    if not final_test:
        # Write the development decision before touching independent final-test audio.
        report["summary_markdown"] = report_markdown(report)
        atomic_write_json(root / "report.json", report)
        report["final_test_status"] = "not configured"
        if (root / "final_test" / "plan.json").is_file():
            base = candidates[0]
            selected = next(row for row in candidates if row["path"] == report["recommended_checkpoint"])
            frozen = [base] if selected is base else [base, selected]
            atomic_write_json(root / "final_test" / "selection_frozen.json", {
                "recommended_checkpoint": selected["path"], "candidates": frozen,
                "frozen_at": datetime.now(timezone.utc).isoformat()})
            try:
                final_report = run_speech_evaluation(config, state_dir, frozen_selection=frozen)
                report.update(final_test_status=final_report["final_test_status"],
                              final_test_message=final_report["final_test_message"],
                              final_test_report=str(root / "final_test" / "report.json"))
            except Exception as exc:
                report.update(final_test_status="failed", final_test_message=str(exc))
                print(f">> independent final test did not complete: {exc}", flush=True)
        report["elapsed_s"] = time.perf_counter()-started
    report["summary_markdown"] = report_markdown(report)
    from .listening_review import write_listening_review
    report["listening_review"] = str(write_listening_review(root, report))
    report["summary_markdown"] = report_markdown(report)
    atomic_write_json(root / "report.json", report)
    (root / "report.md").write_text(report["summary_markdown"], encoding="utf-8")
    atomic_write_json(state / "status.json", {"phase": "complete", "message": "Speech evaluation complete", "elapsed_s": report["elapsed_s"]})
    return report


def _benchmark_runtime(config: Any) -> Any:
    from indextts.runtime import gpu_free_gb, gpu_total_gb, resolve_preset
    device_index = int(config.device.split(":")[-1]) if ":" in config.device else 0
    runtime = resolve_preset(config.sample_runtime_tier, gpu_total_gb(device_index), gpu_free_gb(device_index))
    runtime.device = config.device
    return runtime


def _benchmark_infer_kwargs(config: Any) -> dict[str, Any]:
    from .sampling import SAMPLE_FIXED_INFER_KWARGS
    infer = dict(SAMPLE_FIXED_INFER_KWARGS)
    infer.update(top_p=config.sample_top_p, top_k=config.sample_top_k or None, temperature=config.sample_temperature,
                 length_penalty=config.sample_length_penalty, num_beams=config.sample_num_beams,
                 repetition_penalty=config.sample_repetition_penalty, max_mel_tokens=config.sample_max_mel_tokens,
                 emo_alpha=config.sample_emo_alpha, max_text_tokens_per_segment=config.sample_max_text_tokens,
                 diffusion_steps=config.sample_diffusion_steps, inference_cfg_rate=config.sample_inference_cfg_rate,
                 latent_multiplier=round(float(infer["latent_multiplier"]) / config.sample_speaking_rate, 4))
    return infer


def _lenient_terms(config: Any, plan: dict[str, Any]) -> set[str]:
    # The dataset's own spellings of names and terms are not transcript errors,
    # for Base and adapters alike; this mirrors the voice and transcript audit.
    from .dataset_manifest import load_manifest
    from .dataset_quality import transcript_vocabulary
    from .speech_metrics import lenient_units
    try:
        vocabulary = transcript_vocabulary(row.get("text", "") for row in load_manifest(Path(config.dataset_dir)))
    except Exception:
        vocabulary = []
    lenient: set[str] = set()
    for language in {group["language"] for group in plan["groups"]}:
        lenient |= set(lenient_units(vocabulary, language))
    return lenient


DECODER_TEST_MIN_SPEAKER_GAIN = 0.005
# Strengths judged through the full pipeline, strongest first. A lower strength keeps part of the identity
# gain while costing less intelligibility; the score below trades a point of word error for four hundredths of
# speaker similarity, and the best-scoring strength that passes both gates becomes the recommended strength.
DECODER_TEST_STRENGTHS = (1.0, 0.6)
DECODER_TEST_WER_WEIGHT = 4.0


def load_decoder_test(run_dir: str | Path) -> dict[str, Any] | None:
    try:
        report = json.loads((Path(run_dir) / "analysis" / "speech_evaluation" / "decoder_test" / "report.json").read_text(encoding="utf-8"))
        return report if isinstance(report, dict) and report.get("status") == "complete" else None
    except (OSError, ValueError, TypeError):
        return None


def decoder_test_markdown(report: dict[str, Any]) -> str:
    without, with_ = report["without"], report["with"]
    metric_name = "speaker similarity to the real recordings" if report["metric"] == "speaker_similarity_real" else "speaker similarity to the reference clip"
    gain = report["speaker_gain"]
    verdict = "installed" if report["accepted"] else "not installed"
    lines = [f"**Voice decoder adapter {verdict}.** " + (
        f"Through the full pipeline, {metric_name} rose by {gain['mean']:+.4f} on {gain['prompts']} held-out sentences "
        f"({report['clips']} clips) and the word error rate moved by {100 * report['wer_increase']:+.2f} points."
        if gain.get("mean") is not None else "The paired comparison could not be measured."), ""]
    if report["reasons"]:
        lines.append("Reasons: " + "; ".join(report["reasons"]) + ".")
        lines.append("")
    lines.extend(["| Measure | Without adapter | With adapter |", "|---|---:|---:|"])
    def cell(value: Any, percent: bool = False) -> str:
        if value is None:
            return "-"
        return f"{100 * float(value):.2f}%" if percent else f"{float(value):.4f}"
    lines.append(f"| {metric_name} | {cell(without.get(report['metric']))} | {cell(with_.get(report['metric']))} |")
    lines.append(f"| speaker similarity to the reference clip | {cell(without.get('speaker_similarity'))} | {cell(with_.get('speaker_similarity'))} |")
    lines.append(f"| style similarity to the real recordings | {cell(without.get('style_similarity_real'))} | {cell(with_.get('style_similarity_real'))} |")
    lines.append(f"| word error rate | {cell(without.get('corpus_error_rate'), True)} | {cell(with_.get('corpus_error_rate'), True)} |")
    lines.append(f"| failed clips | {without.get('failure_count', 0)} | {with_.get('failure_count', 0)} |")
    if gain.get("ci95"):
        lines.extend(["", f"95 percent interval of the similarity gain: {gain['ci95'][0]:+.4f} to {gain['ci95'][1]:+.4f} "
                          "(bootstrap over sentences, all seeds of a sentence together)."])
    variants = report.get("variants") or []
    if len(variants) > 1:
        lines.extend(["", "| Decoder strength | Similarity gain | Word error change | Score | Passes |", "|---:|---:|---:|---:|---|"])
        for item in variants:
            item_gain = item["speaker_gain"].get("mean")
            lines.append(f"| {item['strength']:g}{' (recommended)' if item.get('selected') else ''} | "
                         f"{item_gain:+.4f} | {100 * item['wer_increase']:+.2f} points | {item['score']:+.4f} | "
                         f"{'yes' if item['passes'] else 'no'} |" if item_gain is not None else
                         f"| {item['strength']:g} | - | {100 * item['wer_increase']:+.2f} points | - | no |")
        lines.append(f"Score: similarity gain minus {report.get('wer_weight', DECODER_TEST_WER_WEIGHT):g} times the word-error increase. "
                     "Voice Generation applies the recommended strength when this LoRA / DoRA is selected.")
    lines.extend(["", f"Checkpoint `{Path(report['checkpoint']).name}`, same sentences, reference, and seeds as `{Path(report['baseline_report']).parent.name}`."])
    return "\n".join(lines) + "\n"


def render_benchmark_rows(config: Any, plan: dict[str, Any], *, run_dir: Path, checkpoint: str, label: str,
                          runtime: Mapping[str, Any], infer: Mapping[str, Any], attempt: Path, out: Path, state: Path,
                          update: Any, cancelled: Any, lenient: set[str], grid_suffix: str = "",
                          baseline: Sequence[Mapping[str, Any]] | None = None,
                          message: str = "Generating speech") -> tuple[list[dict[str, Any]], list[str]]:
    """Render every benchmark group with one checkpoint and one settings choice, then measure the clips.

    With ``baseline`` rows, the result is restricted to their sentences and seeds and must cover all of them,
    so callers can compare pairwise.
    """
    import torch
    from indextts.runtime import ProgressReporter
    from .grid import GridCheckpoint, GridConfig, run_grid
    from .speech_metrics import measure_clips
    clips: list[dict[str, Any]] = []
    grids: list[str] = []
    for group in plan["groups"]:
        if hashlib.sha256(Path(group["reference"]).read_bytes()).hexdigest() != group["reference_sha256"]:
            raise ValueError("The frozen training reference has changed")
        update(f"{message} for {group['speaker']} ({group['language']})", 0, len(group["prompts"]) * len(plan["seeds"]))
        grid_config = GridConfig(adapter_dir=str(run_dir), checkpoints=[GridCheckpoint(label, checkpoint)],
                                 references=[group["reference"]], texts=[p["text"] for p in group["prompts"]],
                                 language=group["language"], seeds=plan["seeds"], seed=plan["seeds"][0],
                                 output_root=str(attempt), grid_name=f"{group['id']}{grid_suffix}",
                                 runtime={"runtime": dict(runtime), "model_dir": config.model_dir, "cfg_path": config.model_config,
                                          "use_qwen_emo": False},
                                 infer_kwargs=dict(infer), include_verdicts=False)
        result = run_grid(grid_config, reporter=ProgressReporter("benchmark clips", progress_file=state / "progress.json"),
                          cancel_callback=cancelled)
        if result.status != "complete":
            raise InterruptedError(f"Speech generation {result.status}")
        grids.append(result.grid_dir)
        for cell in result.cells:
            prompt = group["prompts"][cell.text_index - 1]
            clips.append({"audio": cell.audio_path, "reference": group["reference"], "real_audio": prompt["audio"],
                          "text": prompt["text"], "language": group["language"], "kind": prompt["kind"],
                          "prompt_id": f"{group['id']}:{prompt['id']}", "source": prompt["source"],
                          "seed": cell.seed, "checkpoint": label})
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    measured = measure_clips(clips, model_dir=config.model_dir, model_config=config.model_config, device=config.device,
                             output_dir=out, update=update, cancelled=cancelled, lenient_terms=lenient)
    if baseline is not None:
        pairs = {(row["prompt_id"], row["seed"]) for row in baseline}
        measured = [row for row in measured if (row["prompt_id"], row["seed"]) in pairs]
        if len(measured) != len(pairs):
            raise ValueError("the benchmark render did not produce every sentence and seed of the earlier measurement")
    return measured, grids


def run_decoder_test(config: Any, state_dir: str | Path, *, checkpoint_path: str, adapter_path: str,
                     min_speaker_gain: float = DECODER_TEST_MIN_SPEAKER_GAIN,
                     strengths: tuple[float, ...] = DECODER_TEST_STRENGTHS,
                     wer_weight: float = DECODER_TEST_WER_WEIGHT) -> dict[str, Any]:
    """Judge a voice decoder adapter through the full pipeline and pick its strength.

    The speech benchmark is rendered again with the selected checkpoint and the adapter at each candidate
    strength, then compared pairwise with that checkpoint's earlier measurement without it (same sentences,
    reference, and seeds). Teacher-forced checks inside the decoder trainer cannot see what generated codes
    do to the adapter; this can, and it decides whether the adapter stays installed and at which strength.
    """
    from indextts.lora.decoder import find_decoder_adapter
    from .speech_metrics import MIN_REAL_SPEAKER_ROWS, paired_difference, summarize
    run_dir = Path(config.output_dir).resolve() / config.name
    root = run_dir / "analysis" / "speech_evaluation"
    out = root / "decoder_test"
    checkpoint = str(Path(checkpoint_path).expanduser().resolve())
    adapter = str(Path(adapter_path).expanduser().resolve())
    found = find_decoder_adapter(checkpoint)
    if not found or str(Path(found).resolve()) != adapter:
        raise ValueError(f"generation with {Path(checkpoint).name} would use {found or 'no decoder adapter'}, not {adapter}")
    plan: dict[str, Any] | None = None
    baseline_rows: list[dict[str, Any]] = []
    label = ""
    baseline_report = ""
    inference: dict[str, Any] = {}
    # The independent final test measured the selected checkpoint last; fall back to the development report.
    for base_root in (root / "final_test", root):
        try:
            report = json.loads((base_root / "report.json").read_text(encoding="utf-8"))
            candidate_plan = json.loads((base_root / "plan.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if not isinstance(report, dict) or report.get("status") != "complete":
            continue
        label = next((str(c["label"]) for c in report.get("candidates", [])
                      if c.get("path") and str(Path(c["path"]).resolve()) == checkpoint), "")
        rows = [row for row in report.get("cells", []) if label and row.get("checkpoint") == label]
        if rows:
            plan, baseline_rows, baseline_report = candidate_plan, rows, str(base_root / "report.json")
            inference = report.get("inference") if isinstance(report.get("inference"), dict) else {}
            break
    if plan is None:
        raise ValueError("the speech benchmark has no measurement of the selected checkpoint to compare the decoder adapter with")
    state = Path(state_dir)
    state.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def cancelled() -> bool:
        return (state / "stop.flag").exists() or (run_dir / "stop.flag").exists()

    def update(message: str, completed: int, total: int) -> None:
        value = {"phase": "testing_decoder", "message": message, "desc": message, "completed": completed, "total": total,
                 "fraction": completed / total if total else 0, "elapsed_s": time.perf_counter() - started, "updated_at": time.time()}
        atomic_write_json(state / "status.json", value)
        atomic_write_json(state / "progress.json", value)
        print(f">> {message}: {completed}/{total}", flush=True)

    runtime = _benchmark_runtime(config).to_dict()
    runtime["decoder_adapter"] = adapter  # the very file under test
    infer = dict(inference.get("infer_kwargs") or {}) or _benchmark_infer_kwargs(config)
    attempt = out / "grids" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    lenient = _lenient_terms(config, plan)
    baseline = [row for row in baseline_rows]
    without_summary = summarize(baseline)
    max_wer_increase = float((plan.get("policy") or {}).get("max_wer_increase", 0.02))
    real_rows = lambda rows: sum(row.get("speaker_similarity_real") is not None for row in rows)
    candidates = list(dict.fromkeys(max(0.05, min(4.0, float(item))) for item in strengths)) or [1.0]
    variants: list[dict[str, Any]] = []
    for strength in candidates:
        runtime["decoder_adapter_strength"] = float(strength)
        measured, grids = render_benchmark_rows(config, plan, run_dir=run_dir, checkpoint=checkpoint, label=label, runtime=runtime,
                                                infer=infer, attempt=attempt, out=out, state=state, update=update, cancelled=cancelled,
                                                lenient=lenient, grid_suffix=f"_strength_{strength:g}".replace(".", "_"), baseline=baseline,
                                                message=f"Generating speech with the voice decoder adapter at strength {strength:g}")
        metric = "speaker_similarity_real" if min(real_rows(measured), real_rows(baseline)) >= MIN_REAL_SPEAKER_ROWS else "speaker_similarity"
        gain = paired_difference(measured, baseline, metric)
        with_summary = summarize(measured)
        wer_increase = float(with_summary["corpus_error_rate"]) - float(without_summary["corpus_error_rate"])
        reasons: list[str] = []
        target = "the real recordings" if metric == "speaker_similarity_real" else "the reference clip"
        if gain.get("mean") is None:
            reasons.append(f"speaker similarity to {target} could not be compared")
        elif float(gain["mean"]) < min_speaker_gain:
            reasons.append(f"speaker similarity to {target} changed by {float(gain['mean']):+.4f} through the full pipeline at "
                           f"strength {strength:g}, less than the required +{min_speaker_gain:g}")
        if wer_increase > max_wer_increase:
            reasons.append(f"the word error rate rose by {100 * wer_increase:.2f} points at strength {strength:g}, "
                           f"more than the allowed {100 * max_wer_increase:.0f}")
        score = (float(gain["mean"]) - wer_weight * max(0.0, wer_increase)) if gain.get("mean") is not None else float("-inf")
        variants.append({"strength": float(strength), "metric": metric, "speaker_gain": gain, "with": with_summary,
                         "wer_increase": wer_increase, "reasons": reasons, "passes": not reasons, "score": score,
                         "cells": measured, "grids": grids, "selected": False})
        print(f">> decoder adapter at strength {strength:g}: similarity {gain.get('mean') if gain.get('mean') is None else round(float(gain['mean']), 4):+} "
              f"| word error {100 * wer_increase:+.2f} points | {'passes' if not reasons else '; '.join(reasons)}", flush=True)
    passing = [item for item in variants if item["passes"]]
    chosen = max(passing, key=lambda item: (item["score"], item["strength"])) if passing else max(variants, key=lambda item: (item["score"], item["strength"]))
    chosen["selected"] = True
    accepted = bool(passing)
    reasons = list(chosen["reasons"]) if not accepted else []
    report = {"status": "complete", "accepted": accepted, "reasons": reasons, "metric": chosen["metric"],
              "speaker_gain": chosen["speaker_gain"], "without": without_summary, "with": chosen["with"],
              "wer_increase": chosen["wer_increase"], "max_wer_increase": max_wer_increase, "min_speaker_gain": min_speaker_gain,
              "strength": chosen["strength"], "wer_weight": wer_weight, "variants": variants, "clips": len(chosen["cells"]),
              "checkpoint": checkpoint, "checkpoint_label": label, "adapter": adapter, "baseline_report": baseline_report,
              "cells": chosen["cells"], "baseline_cells": baseline, "grids": [grid for item in variants for grid in item["grids"]],
              "seeds": plan["seeds"], "generated_at": datetime.now(timezone.utc).isoformat(), "elapsed_s": time.perf_counter() - started}
    report["summary_markdown"] = decoder_test_markdown(report)
    atomic_write_json(out / "report.json", report)
    (out / "report.md").write_text(report["summary_markdown"], encoding="utf-8")
    atomic_write_json(state / "status.json", {"phase": "complete", "message": "Decoder test complete", "elapsed_s": report["elapsed_s"]})
    return report


def main() -> int:
    from indextts.utils.console_encoding import configure_console_output
    from .train_config import TrainConfig
    configure_console_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--decoder-test", default="", help="Voice decoder adapter file to judge through the full pipeline")
    parser.add_argument("--decoding-sweep", action="store_true", help="Sweep decoding settings for --checkpoint on the benchmark")
    parser.add_argument("--checkpoint", default="", help="GPT checkpoint the decoder test or decoding sweep generates with")
    args = parser.parse_args()
    try:
        if args.decoder_test:
            report = run_decoder_test(TrainConfig.from_json(args.config), args.state_dir,
                                      checkpoint_path=args.checkpoint, adapter_path=args.decoder_test)
        elif args.decoding_sweep:
            from .decoding_sweep import run_decoding_sweep
            report = run_decoding_sweep(TrainConfig.from_json(args.config), args.state_dir, checkpoint_path=args.checkpoint)
        else:
            report = run_speech_evaluation(TrainConfig.from_json(args.config), args.state_dir)
        print(report["summary_markdown"], flush=True)
        return 0
    except BaseException as exc:
        traceback.print_exc()
        atomic_write_json(Path(args.state_dir) / "status.json", {"phase": "cancelled" if isinstance(exc, InterruptedError) else "failed", "message": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
