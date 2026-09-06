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
from typing import Any

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
    lines = [f"**{heading}: {report['recommended_label']}**", report["scope"], "",
             "| Candidate | Mean transcript error | Worst clip | Speaker similarity | Flagged clips | Eligible |",
             "|---|---:|---:|---:|---:|---|"]
    for row in report["candidates"]:
        speaker = f"{row['speaker_similarity']:.3f}" if row.get("speaker_similarity") is not None else "unavailable"
        lines.append(f"| {row['label']} | {row['mean_error_rate']:.1%} | {row['worst_error_rate']:.1%} | {speaker} | {row['failure_count']}/{row['clips']} | {'yes' if row['eligible'] else 'no'} |")
    lines.extend(["", report["decision"], "Transcript error uses words for EN/ES/AR and characters for ZH/JA."])
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
    device_index = int(config.device.split(":")[-1]) if ":" in config.device else 0
    runtime = resolve_preset(config.sample_runtime_tier, gpu_total_gb(device_index), gpu_free_gb(device_index))
    runtime.device = config.device
    infer = dict(SAMPLE_FIXED_INFER_KWARGS)
    infer.update(top_p=config.sample_top_p, top_k=config.sample_top_k or None, temperature=config.sample_temperature,
                 length_penalty=config.sample_length_penalty, num_beams=config.sample_num_beams,
                 repetition_penalty=config.sample_repetition_penalty, max_mel_tokens=config.sample_max_mel_tokens,
                 emo_alpha=config.sample_emo_alpha, max_text_tokens_per_segment=config.sample_max_text_tokens,
                 diffusion_steps=config.sample_diffusion_steps, inference_cfg_rate=config.sample_inference_cfg_rate,
                 latent_multiplier=round(float(infer["latent_multiplier"]) / config.sample_speaking_rate, 4))
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
                             device=config.device, output_dir=root, update=update, cancelled=cancelled)
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


def main() -> int:
    from indextts.utils.console_encoding import configure_console_output
    from .train_config import TrainConfig
    configure_console_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--state-dir", required=True)
    args = parser.parse_args()
    try:
        report = run_speech_evaluation(TrainConfig.from_json(args.config), args.state_dir)
        print(report["summary_markdown"], flush=True)
        return 0
    except BaseException as exc:
        traceback.print_exc()
        atomic_write_json(Path(args.state_dir) / "status.json", {"phase": "cancelled" if isinstance(exc, InterruptedError) else "failed", "message": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
