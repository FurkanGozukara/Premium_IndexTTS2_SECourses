"""Run a frozen, dataset-specific speech benchmark after training releases memory."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import shutil
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


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def development_fingerprint(run_dir: str | Path) -> str:
    """Identify measured development data, excluding later final-test summary updates."""
    root = Path(run_dir) / "analysis" / "speech_evaluation"
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    payload = {"plan": plan, **{key: report.get(key) for key in
               ("dataset_identity", "candidates", "cells", "real_cells", "inference", "evaluation_partition")}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _validate_development_report(report: Mapping[str, Any]) -> None:
    if (report.get("status") != "complete" or report.get("final_test")
            or report.get("evaluation_partition") != "validation"):
        raise ValueError("Selection requires a completed validation-only speech report")
    inference = report.get("inference") or {}
    runtime = inference.get("runtime", {})
    runtime = runtime.get("runtime", runtime) if isinstance(runtime, dict) else {}
    if runtime.get("decoder_adapter") != "none":
        raise ValueError("The development baseline must have the voice decoder adapter explicitly disabled")


def development_baseline(run_dir: str | Path, checkpoint: str) -> tuple[dict[str, Any], list[dict[str, Any]], str, Path, dict[str, Any]]:
    """Only validation may select a decoder strength or decoding settings."""
    root = Path(run_dir) / "analysis" / "speech_evaluation"
    report_path = root / "report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("A completed development speech benchmark is required; final-test measurements cannot be used for tuning") from exc
    if not isinstance(report, dict):
        raise ValueError("A completed validation-only speech report is required")
    _validate_development_report(report)
    candidate = next((row for row in report.get("candidates", [])
                      if row.get("path") and str(Path(row["path"]).resolve()) == checkpoint), None)
    if not candidate or not Path(checkpoint).is_file() or candidate.get("sha256") != _file_sha256(checkpoint):
        raise ValueError("The selected checkpoint no longer matches its development measurement")
    label = str(candidate["label"])
    rows = [row for row in report.get("cells", []) if label and row.get("checkpoint") == label]
    if not rows:
        raise ValueError("The development speech benchmark has no measurement of the selected checkpoint")
    inference = report.get("inference") if isinstance(report.get("inference"), dict) else {}
    return plan, rows, label, report_path, inference


def freeze_deployment_selection(config: Any, checkpoint_path: str) -> dict[str, Any]:
    """Freeze development-selected weights and effective deployment settings before final testing."""
    from indextts.lora.decoder import find_decoder_adapter
    from .decoding_sweep import DECODING_KEYS, load_decoding_settings
    from .sampling import SAMPLE_FIXED_INFER_KWARGS
    from .speaking_rate import load_speaking_rate

    run_dir = Path(config.output_dir).resolve() / config.name
    root = run_dir / "analysis" / "speech_evaluation"
    report = load_speech_evaluation(run_dir)
    if report is None:
        raise ValueError("Freeze requires a completed development speech recommendation")
    _validate_development_report(report)
    checkpoint = str(Path(checkpoint_path).resolve()) if checkpoint_path else ""
    recommended = str(Path(report["recommended_checkpoint"]).resolve()) if report.get("recommended_checkpoint") else ""
    if checkpoint != recommended:
        raise ValueError("Final testing cannot change the development-selected checkpoint")
    if checkpoint and not Path(checkpoint).is_relative_to(run_dir):
        raise ValueError("The frozen checkpoint must belong to this training run")
    base = next((dict(row) for row in report["candidates"] if not row.get("path")), None)
    selected = next((dict(row) for row in report["candidates"]
                     if (str(Path(row["path"]).resolve()) if row.get("path") else "") == checkpoint), None)
    if base is None or selected is None:
        raise ValueError("The development report lacks Base or the selected checkpoint")
    runtime = _benchmark_runtime(config).to_dict()
    runtime.update(lora_path="", decoder_adapter="none", decoder_adapter_strength=1.0)
    infer = _benchmark_infer_kwargs(config)
    # With deployment settings the development report rendered Base and the adapters with their own
    # settings (emotion prompt, pauses, token target, pace); the final test keeps each candidate's.
    candidate_inference = report.get("candidate_inference") if isinstance(report.get("candidate_inference"), dict) else {}

    def infer_for(label: str) -> dict[str, Any]:
        entry = candidate_inference.get(label) if isinstance(candidate_inference, dict) else None
        if isinstance(entry, dict) and isinstance(entry.get("infer_kwargs"), dict) and entry["infer_kwargs"]:
            return dict(entry["infer_kwargs"])
        return dict(infer)

    base_infer = infer_for("Base")
    base.update(path="", runtime=dict(runtime), infer_kwargs=base_infer,
                speaking_rate=round(float(SAMPLE_FIXED_INFER_KWARGS["latent_multiplier"]) / float(base_infer.get("latent_multiplier") or SAMPLE_FIXED_INFER_KWARGS["latent_multiplier"]), 4))
    candidates = [base]
    artifacts = {str(root / "report.json"): _file_sha256(root / "report.json"),
                 str(root / "plan.json"): _file_sha256(root / "plan.json"),
                 str(root / "final_test" / "plan.json"): _file_sha256(root / "final_test" / "plan.json")}
    if checkpoint:
        selected_infer = infer_for(str(selected.get("label") or ""))
        selected.update(path=checkpoint, runtime=dict(runtime), infer_kwargs=selected_infer,
                        speaking_rate=round(float(SAMPLE_FIXED_INFER_KWARGS["latent_multiplier"]) / float(selected_infer.get("latent_multiplier") or SAMPLE_FIXED_INFER_KWARGS["latent_multiplier"]), 4))
        artifacts[checkpoint] = _file_sha256(checkpoint)
        if selected.get("sha256") != artifacts[checkpoint]:
            raise ValueError("The selected checkpoint changed after development evaluation")
        measured_development = development_fingerprint(run_dir)
        decoder_path = find_decoder_adapter(checkpoint)
        if decoder_path:
            decoder = load_decoder_test(run_dir)
            expected_baseline = root / "report.json"
            if (not decoder or not decoder.get("accepted") or decoder.get("evaluation_partition") != "validation"
                    or str(Path(str(decoder.get("checkpoint", ""))).resolve()) != checkpoint
                    or str(Path(str(decoder.get("adapter", ""))).resolve()) != str(Path(decoder_path).resolve())
                    or str(Path(str(decoder.get("baseline_report", ""))).resolve()) != str(expected_baseline.resolve())
                    or decoder.get("checkpoint_sha256") != artifacts[checkpoint]
                    or decoder.get("development_fingerprint") != measured_development
                    or decoder.get("adapter_sha256") != _file_sha256(decoder_path)):
                raise ValueError("An installed decoder lacks a matching completed validation gate; final testing is blocked")
            selected["runtime"].update(decoder_adapter=str(Path(decoder_path).resolve()),
                                       decoder_adapter_strength=float(decoder.get("strength", 1.0)))
            artifacts[str(Path(decoder_path).resolve())] = decoder["adapter_sha256"]
            gate_path = root / "decoder_test" / "report.json"
            artifacts[str(gate_path)] = _file_sha256(gate_path)
        rate = load_speaking_rate(checkpoint)
        if rate is not None:
            selected["speaking_rate"] = float(rate.recommended_speaking_rate)
            selected["infer_kwargs"]["latent_multiplier"] = round(
                float(SAMPLE_FIXED_INFER_KWARGS["latent_multiplier"]) / selected["speaking_rate"], 4)
            rate_path = run_dir / "analysis" / "speaking_rate.json"
            artifacts[str(rate_path)] = _file_sha256(rate_path)
        settings_path = run_dir / "analysis" / "decoding.json"
        if settings_path.is_file():
            settings_report = json.loads(settings_path.read_text(encoding="utf-8"))
            artifacts[str(settings_path)] = _file_sha256(settings_path)
            if settings_report.get("accepted"):
                current_decoder = selected["runtime"]["decoder_adapter"]
                if (settings_report.get("evaluation_partition") != "validation"
                        or str(Path(str(settings_report.get("checkpoint", ""))).resolve()) != checkpoint
                        or str(settings_report.get("decoder_adapter", "none")) != current_decoder
                        or float(settings_report.get("decoder_adapter_strength", 1.0)) != selected["runtime"]["decoder_adapter_strength"]):
                    raise ValueError("Adopted decoding settings do not match the validation-selected checkpoint and decoder")
                decoder_digest = artifacts.get(current_decoder, "")
                sweep_path = root / "decoding_sweep" / "report.json"
                if (settings_report.get("checkpoint_sha256") != artifacts[checkpoint]
                        or settings_report.get("decoder_adapter_sha256", "") != decoder_digest
                        or settings_report.get("development_fingerprint") != measured_development
                        or not sweep_path.is_file() or settings_report.get("report_sha256") != _file_sha256(sweep_path)):
                    raise ValueError("Adopted decoding settings lack matching validation provenance")
                sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
                if (sweep.get("status") != "complete" or not sweep.get("accepted")
                        or sweep.get("evaluation_partition") != "validation"
                        or sweep.get("settings") != settings_report.get("settings")
                        or sweep.get("checkpoint_sha256") != artifacts[checkpoint]
                        or sweep.get("decoder_adapter_sha256", "") != decoder_digest
                        or sweep.get("development_fingerprint") != measured_development):
                    raise ValueError("The complete decoding validation report does not match the frozen deployment")
                artifacts[str(sweep_path)] = settings_report["report_sha256"]
                settings = load_decoding_settings(checkpoint)
                if settings is None:
                    raise ValueError("The adopted decoding settings cannot be loaded")
                selected["infer_kwargs"].update({key: settings[key] for key in DECODING_KEYS})
        candidates.append(selected)
    for candidate in candidates:
        candidate["sha256"] = artifacts.get(candidate["path"], "")
    return {"version": 2, "evaluation_partition": "final_test", "selection_partition": "validation",
            "recommended_checkpoint": checkpoint, "candidates": candidates, "artifacts": artifacts,
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "scope": "Checkpoint, decoder and strength, calibrated speaking rate, and decoding settings frozen before final testing"}


def _validate_frozen_deployment(frozen: Mapping[str, Any]) -> None:
    for path, expected in frozen.get("artifacts", {}).items():
        if not Path(path).is_file() or _file_sha256(path) != expected:
            raise ValueError(f"A frozen deployment artifact changed: {path}")


def run_final_test(config: Any, state_dir: str | Path, *, checkpoint_path: str) -> dict[str, Any]:
    """Assess the already frozen deployment, without feeding final results back into tuning."""
    run_dir = Path(config.output_dir).resolve() / config.name
    root = run_dir / "analysis" / "speech_evaluation"
    final_root = root / "final_test"
    frozen = freeze_deployment_selection(config, checkpoint_path)
    # An explicit rerun preserves earlier evidence instead of silently replacing it.
    prior = [final_root / name for name in ("report.json", "report.md", "selection_frozen.json", "listening_review.html")
             if (final_root / name).is_file()]
    if prior:
        history = final_root / "history" / str(time.time_ns())
        history.mkdir(parents=True)
        for path in prior:
            shutil.copy2(path, history / path.name)
    # The development summary receives the final result later. Keep its exact
    # selection-time contents as immutable evidence for the frozen deployment.
    source_report = root / "report.json"
    evidence = final_root / "frozen_inputs" / str(time.time_ns()) / "development_report.json"
    evidence.parent.mkdir(parents=True)
    shutil.copy2(source_report, evidence)
    expected = frozen["artifacts"].pop(str(source_report))
    if _file_sha256(evidence) != expected:
        raise ValueError("The development recommendation changed while freezing deployment")
    frozen["artifacts"][str(evidence)] = expected
    frozen["development_report"] = str(evidence)
    atomic_write_json(final_root / "selection_frozen.json", frozen)
    try:
        report = run_speech_evaluation(config, state_dir, frozen_selection=frozen["candidates"], frozen_deployment=frozen)
    except Exception as exc:
        development = load_speech_evaluation(run_dir)
        if development:
            development.update(final_test_status="failed", final_test_message=str(exc))
            development["summary_markdown"] = report_markdown(development)
            atomic_write_json(root / "report.json", development)
            (root / "report.md").write_text(development["summary_markdown"], encoding="utf-8")
        raise
    development = load_speech_evaluation(run_dir)
    if development:
        development.update(final_test_status=report["final_test_status"], final_test_message=report["final_test_message"],
                           final_test_report=str(final_root / "report.json"), final_test_deployment_frozen=True)
        development["summary_markdown"] = report_markdown(development)
        atomic_write_json(root / "report.json", development)
        (root / "report.md").write_text(development["summary_markdown"], encoding="utf-8")
    return report


_EXTRA_CANDIDATE_KINDS = frozenset({"averaged", "ema"})


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
                        "val_loss": row.val_loss if row else item.get("val_loss"), "kind": str(item.get("kind") or "")})
    entries.sort(key=lambda r: (r["val_loss"] if r["val_loss"] is not None else float("inf"), -r["steps"], r["path"]))
    distinct = []
    seen = set()
    for item in entries:
        # Best/final/epoch files can contain exactly the same training update; an averaged file shares
        # the newest member's step count but is a different model, so it keeps its own identity.
        identity = item["path"] if item.get("kind") in _EXTRA_CANDIDATE_KINDS or not item["steps"] else item["steps"]
        if identity not in seen:
            distinct.append(item)
            seen.add(identity)
    if not distinct:
        raise ValueError("No checkpoints from this run are available for speech evaluation")
    members = [item for item in distinct if item.get("kind") not in _EXTRA_CANDIDATE_KINDS] or distinct
    selected = members[:max(1, limit)]
    latest = max(members, key=lambda r: r["steps"])
    if limit > 1 and latest not in selected:
        selected[-1] = latest
    # The update whose epoch probe scored best against Base during training is always judged too; it may be
    # the same training update as an epoch file, in which case that file already represents it.
    probe_steps = {item["steps"] for item in entries if item.get("kind") == "probe_best" and item["steps"]}
    for item in members:
        if item["steps"] in probe_steps and item not in selected:
            selected.append(item)
    for item in distinct:
        if item.get("kind") == "probe_best" and not item["steps"] and item not in selected:
            selected.append(item)
    # The averaged checkpoint and the EMA of the final update are always judged, in addition to the shortlist;
    # EMA epoch files stay available for the grid but are not rendered automatically.
    selected.extend(
        item for item in distinct
        if item.get("kind") in _EXTRA_CANDIDATE_KINDS and item not in selected
        and not (item.get("kind") == "ema" and "_ema_epoch_" in str(item["path"]).lower())
    )
    return [{"label": "Base", "path": "", "steps": 0, "val_loss": base_loss},
            *[{key: value for key, value in item.items() if key != "kind"} for item in selected]]


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
                         f"[{delta['ci95'][0]:+.1%}, {delta['ci95'][1]:+.1%}]. " + "; ".join(row["rejection_reasons"])
                         + ("; ".join(row.get("notes") or []) if row.get("notes") else ""))
    joint = report.get("joint_selection")
    if joint:
        lines.extend(["", f"**Deployment choice with the voice decoder adapter: {joint['recommended_label']}.** {joint['decision']}"])
        if joint.get("candidates"):
            lines.extend(["", "| Deployment | Deployment score vs Base | Speaker similarity vs real | Mean transcript error | Eligible |",
                          "|---|---:|---:|---:|---|"])
            for row in joint["candidates"]:
                speaker_real = f"{row['speaker_similarity_real']:.3f}" if row.get("speaker_similarity_real") is not None else "unavailable"
                lines.append(f"| {row['label']} | {row['deployment_score']['score']:+.4f} | {speaker_real} | {row['mean_error_rate']:.1%} | "
                             f"{'yes' if row['eligible'] else 'no'} |")
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
                          frozen_selection: list[dict[str, Any]] | None = None,
                          frozen_deployment: Mapping[str, Any] | None = None) -> dict[str, Any]:
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
        if frozen_deployment is None:
            raise ValueError("Final testing requires an explicit frozen deployment")
    if frozen_deployment is not None:
        _validate_frozen_deployment(frozen_deployment)
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    if not plan["groups"]:
        raise ValueError("The speech benchmark contains no held-out prompts")
    state = Path(state_dir)
    state.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    def cancelled() -> bool:
        return (state / "stop.flag").exists() or (run_dir / "stop.flag").exists()
    def update(message: str, completed: int, total: int) -> None:
        value = {"phase": "evaluating_final_test" if final_test else "evaluating_speech",
                 "message": message, "desc": message, "completed": completed,
                 "total": total, "fraction": completed / total if total else 0, "elapsed_s": time.perf_counter()-started,
                 "updated_at": time.time()}
        atomic_write_json(state / "status.json", value)
        atomic_write_json(state / "progress.json", value)
        print(f">> {message}: {completed}/{total}", flush=True)
    candidates = [dict(row) for row in frozen_selection] if final_test else shortlist_checkpoints(run_dir, plan["candidate_limit"])
    for candidate in candidates:
        if candidate["path"]:
            candidate["sha256"] = _file_sha256(candidate["path"])
    runtime = _benchmark_runtime(config)
    # A continued run can still have an older decoder installed. Development
    # compares GPT checkpoints without it; deployment is assessed separately.
    runtime.decoder_adapter = "none"
    infer = _benchmark_infer_kwargs(config)
    # Development renders every candidate the way Voice Generation deploys it by default: Base with the
    # language defaults, the run's adapters with their profile's token target, pauses, expressive clip and
    # calibrated pace, all at the GPU tier's beams and diffusion steps. Every checkpoint of one run shares
    # those settings, so Base and the adapters form two batches.
    deployment = bool(plan.get("deployment_settings", getattr(config, "speech_eval_deployment_settings", True))) and not final_test
    deployment_infer: dict[tuple[str, str], dict[str, Any]] = {}

    def deployment_settings_for(candidate: Mapping[str, Any], language: str) -> dict[str, Any]:
        # Every checkpoint of one run shares its profile, so Base and "an adapter" per language suffice.
        from .deployment_settings import deployment_infer_kwargs
        key = ("adapter" if candidate["path"] else "base", str(language).upper())
        if key not in deployment_infer:
            deployment_infer[key] = deployment_infer_kwargs(config, candidate["path"], language=key[1], tier=runtime.vram_tier)
        return dict(deployment_infer[key])
    attempt = root / "grids" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    clips, real_clips, grids = [], [], []
    candidate_inference = {}
    report_inference: dict[str, Any] | None = None
    for group in plan["groups"]:
        update(f"Generating speech for {group['speaker']} ({group['language']})", 0, len(group["prompts"]) * len(plan["seeds"]) * len(candidates))
        if hashlib.sha256(Path(group["reference"]).read_bytes()).hexdigest() != group["reference_sha256"]:
            raise ValueError("The frozen training reference has changed")
        for prompt in group["prompts"]:
            if prompt.get("audio_sha256") and hashlib.sha256(Path(prompt["audio"]).read_bytes()).hexdigest() != prompt["audio_sha256"]:
                raise ValueError("A frozen evaluation recording has changed")
        # Final deployment needs per-candidate settings: Base must never inherit
        # the selected adapter's decoder, speaking rate, or tuned decoding knobs.
        if final_test:
            batches = [[row] for row in candidates]
        elif deployment:
            batches = [rows for rows in ([row for row in candidates if not row["path"]], [row for row in candidates if row["path"]]) if rows]
        else:
            batches = [candidates]
        for index, batch in enumerate(batches):
            if frozen_deployment is not None:
                _validate_frozen_deployment(frozen_deployment)
            candidate = batch[0]
            selected_runtime = dict(candidate.get("runtime") or runtime.to_dict()) if final_test else runtime.to_dict()
            if final_test:
                selected_infer = dict(candidate.get("infer_kwargs") or infer)
            elif deployment:
                selected_infer = deployment_settings_for(candidate, group["language"])
            else:
                selected_infer = dict(infer)
            if not candidate["path"]:
                selected_runtime["decoder_adapter"] = "none"
            if final_test:
                grid_name = f"{group['id']}_candidate_{index}"
            elif deployment:
                grid_name = f"{group['id']}_{'adapters' if candidate['path'] else 'base'}"
            else:
                grid_name = group["id"]
            grid_config = GridConfig(adapter_dir=str(run_dir),
                checkpoints=[GridCheckpoint(row["label"], row["path"]) for row in batch],
                references=[group["reference"]], texts=[p["text"] for p in group["prompts"]], language=group["language"],
                seeds=plan["seeds"], seed=plan["seeds"][0], output_root=str(attempt), grid_name=grid_name,
                runtime={"runtime": selected_runtime, "model_dir": config.model_dir, "cfg_path": config.model_config, "use_qwen_emo": False},
                infer_kwargs=selected_infer, include_verdicts=False)
            result = run_grid(grid_config, reporter=ProgressReporter("speech clips", progress_file=state / "progress.json"), cancel_callback=cancelled)
            if result.status != "complete":
                raise InterruptedError(f"Speech generation {result.status}")
            grids.append(result.grid_dir)
            for row in batch:
                candidate_inference[row["label"]] = grid_config.to_dict()
            # The report's inference block describes how the adapters were rendered (the decoder gate and the
            # decoding sweep render the selected adapter again with exactly these settings).
            if report_inference is None or candidate["path"]:
                report_inference = grid_config.to_dict()
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
    if frozen_deployment is not None:
        _validate_frozen_deployment(frozen_deployment)
    report = select_recommendation(candidates, generated, plan["policy"])
    if final_test:
        chosen = candidates[-1]
        measured_chosen = next(row for row in report["candidates"] if row["label"] == chosen["label"])
        report.update(final_test=True, recommended_label=chosen["label"], recommended_checkpoint=chosen["path"],
                      recommended_kind="adapter" if chosen["path"] else "base",
                      final_test_status="passed observed regression guards" if measured_chosen["eligible"] else "regression detected",
                      final_test_message="; ".join(measured_chosen["rejection_reasons"]),
                      scope="Independent final-test measurements for the frozen deployed pipeline. These results do not reselect a checkpoint, decoder, or decoding settings.",
                      decision="The checkpoint, decoder, speaking rate, and decoding settings were frozen on development data before final-test generation.",
                      deployment_frozen=dict(frozen_deployment) if frozen_deployment else None)
    report.update(dataset_identity=plan["dataset_identity"], plan=str(root / "plan.json"), grids=grids,
                  cells=generated, real_cells=real, real_recordings=summarize(real) if real else None,
                  warnings=plan["warnings"], seeds=plan["seeds"], inference=report_inference or grid_config.to_dict(),
                  candidate_inference=candidate_inference, deployment_settings=deployment,
                  evaluation_partition="final_test" if final_test else "validation",
                  generated_at=datetime.now(timezone.utc).isoformat(), elapsed_s=time.perf_counter()-started)
    if not final_test:
        # Decoder and decoding selection still follow. Independent testing runs
        # in a separate final phase only after the whole deployment is frozen.
        report["final_test_status"] = ("pending deployment freeze" if (root / "final_test" / "plan.json").is_file()
                                      else "not configured")
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
    # The training model has been released before this phase, so the benchmark
    # renders with the training's own tier instead of shrinking to free memory.
    from .sampling import resolve_sample_runtime
    return resolve_sample_runtime(config, share_gpu=False)


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
    from .speech_metrics import GUARD_MODE_INTERVAL, MIN_REAL_SPEAKER_ROWS, paired_difference, source_regression, summarize
    run_dir = Path(config.output_dir).resolve() / config.name
    root = run_dir / "analysis" / "speech_evaluation"
    out = root / "decoder_test"
    checkpoint = str(Path(checkpoint_path).expanduser().resolve())
    adapter = str(Path(adapter_path).expanduser().resolve())
    found = find_decoder_adapter(checkpoint, allow_unverified=True)  # explicit gate association, not AUTO loading
    if not found or str(Path(found).resolve()) != adapter:
        raise ValueError(f"decoder associated with {Path(checkpoint).name} is {found or 'none'}, not {adapter}")
    plan: dict[str, Any] | None = None
    baseline_rows: list[dict[str, Any]] = []
    label = ""
    baseline_report = ""
    inference: dict[str, Any] = {}
    plan, baseline_rows, label, report_path, inference = development_baseline(run_dir, checkpoint)
    baseline_report = str(report_path)
    checkpoint_sha256, adapter_sha256 = _file_sha256(checkpoint), _file_sha256(adapter)
    measured_development = development_fingerprint(run_dir)
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
    guard_mode = str((plan.get("policy") or {}).get("guard_mode") or "mean").strip().lower()
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
        notes: list[str] = []
        error_delta = paired_difference(measured, baseline, "error_rate")
        if wer_increase > max_wer_increase:
            resolved = True
            if guard_mode == GUARD_MODE_INTERVAL:
                sources = source_regression(measured, baseline, "error_rate", worse_when_higher=True)
                ci = error_delta.get("ci95")
                resolved = bool(ci and float(ci[0]) > 0) or bool(sources.get("majority"))
                if not resolved:
                    notes.append(f"the word error rate rose by {100 * wer_increase:.2f} points at strength {strength:g}, more than the "
                                 f"allowed {100 * max_wer_increase:.0f}, but within the benchmark's noise (95% interval includes zero, "
                                 f"higher on {sources['regressed']} of {sources['sources']} recordings); the increase costs score instead")
            if resolved:
                reasons.append(f"the word error rate rose by {100 * wer_increase:.2f} points at strength {strength:g}, "
                               f"more than the allowed {100 * max_wer_increase:.0f}")
        score = (float(gain["mean"]) - wer_weight * max(0.0, wer_increase)) if gain.get("mean") is not None else float("-inf")
        variants.append({"strength": float(strength), "metric": metric, "speaker_gain": gain, "with": with_summary,
                         "wer_increase": wer_increase, "error_delta": error_delta, "reasons": reasons, "notes": notes,
                         "passes": not reasons, "score": score, "cells": measured, "grids": grids, "selected": False})
        print(f">> decoder adapter at strength {strength:g}: similarity {gain.get('mean') if gain.get('mean') is None else round(float(gain['mean']), 4):+} "
              f"| word error {100 * wer_increase:+.2f} points | {'passes' if not reasons else '; '.join(reasons)}", flush=True)
    passing = [item for item in variants if item["passes"]]
    chosen = max(passing, key=lambda item: (item["score"], item["strength"])) if passing else max(variants, key=lambda item: (item["score"], item["strength"]))
    chosen["selected"] = True
    accepted = bool(passing)
    reasons = list(chosen["reasons"]) if not accepted else []
    if (_file_sha256(checkpoint) != checkpoint_sha256 or _file_sha256(adapter) != adapter_sha256
            or development_fingerprint(run_dir) != measured_development):
        raise ValueError("The checkpoint or decoder adapter changed during its validation gate")
    report = {"status": "complete", "evaluation_partition": "validation",
              "accepted": accepted, "reasons": reasons, "notes": list(chosen.get("notes") or []), "metric": chosen["metric"],
              "guard_mode": guard_mode,
              "speaker_gain": chosen["speaker_gain"], "without": without_summary, "with": chosen["with"],
              "wer_increase": chosen["wer_increase"], "max_wer_increase": max_wer_increase, "min_speaker_gain": min_speaker_gain,
              "strength": chosen["strength"], "wer_weight": wer_weight, "variants": variants, "clips": len(chosen["cells"]),
              "checkpoint": checkpoint, "checkpoint_sha256": checkpoint_sha256,
              "checkpoint_label": label, "adapter": adapter, "adapter_sha256": adapter_sha256,
              "development_fingerprint": measured_development,
              "baseline_report": baseline_report,
              "cells": chosen["cells"], "baseline_cells": baseline, "grids": [grid for item in variants for grid in item["grids"]],
              "seeds": plan["seeds"], "generated_at": datetime.now(timezone.utc).isoformat(), "elapsed_s": time.perf_counter() - started}
    report["summary_markdown"] = decoder_test_markdown(report)
    atomic_write_json(out / "report.json", report)
    (out / "report.md").write_text(report["summary_markdown"], encoding="utf-8")
    atomic_write_json(state / "status.json", {"phase": "complete", "message": "Decoder test complete", "elapsed_s": report["elapsed_s"]})
    return report


DECODER_LABEL_SUFFIX = " + voice decoder"


def best_adapter_candidate(report: Mapping[str, Any]) -> dict[str, Any] | None:
    """The adapter candidate of a development report that scored best against Base, eligible or not.

    Used to gate the voice decoder adapter when the GPT comparison preferred Base: the decoder can only be
    judged with an adapter, and the adapter closest to Base on the deployment score is the one whose
    combination with the decoder has the best chance of beating Base.
    """

    adapters = [dict(row) for row in report.get("candidates") or []
                if row.get("path") and Path(str(row["path"])).is_file() and isinstance(row.get("deployment_score"), Mapping)]
    if not adapters:
        return None
    return max(adapters, key=lambda row: (bool(row.get("eligible")), float(row["deployment_score"].get("score", float("-inf"))),
                                          -float(row.get("val_loss") if row.get("val_loss") is not None else float("inf"))))


def joint_recommendation(report: Mapping[str, Any], decoder_report: Mapping[str, Any],
                         plan_policy: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Choose among Base, the measured GPT adapters, and the gated adapter with its voice decoder.

    The decoder gate renders the gated checkpoint on the same sentences, seeds and reference as the
    development benchmark, so its clips pair with Base's cells exactly like the GPT candidates' do. The
    combination is judged by the same Base guards and deployment score; a decoder that only improves the
    adapter relative to itself cannot win unless the whole deployment also beats Base and every plain
    adapter. Nothing in the returned mapping changes the report's measured cells or candidates.
    """

    from .speech_metrics import select_recommendation
    score_policy = dict(report.get("score_policy") or {})
    policy = {"max_wer_increase": 0.02, "max_speaker_drop": 0.03, **dict(plan_policy or {})}
    # The report's score policy is what selected the GPT candidates; the combination is judged the same way.
    policy.update({"guard_mode": score_policy.get("guard_mode", policy.get("guard_mode", "mean")),
                   "score_wer_weight": score_policy.get("wer_weight", policy.get("score_wer_weight", 4.0)),
                   "score_pause_weight": score_policy.get("pause_weight", policy.get("score_pause_weight", 0.05)),
                   "score_min_delta": score_policy.get("min_delta", policy.get("score_min_delta", 0.002))})
    plan_policy = policy
    candidates = [{key: value for key, value in row.items() if key in {"label", "path", "steps", "val_loss", "sha256"}}
                  for row in report.get("candidates") or []]
    cells = [dict(row) for row in report.get("cells") or []]
    gated = next((row for row in candidates if row.get("path")
                  and str(Path(str(row["path"])).resolve()) == str(Path(str(decoder_report.get("checkpoint", ""))).resolve())), None)
    if gated is None:
        raise ValueError("the decoder gate's checkpoint is not a candidate of the development report")
    combo_label = f"{gated['label']}{DECODER_LABEL_SUFFIX}"
    combo = {**gated, "label": combo_label, "decoder": str(decoder_report.get("adapter", "")),
             "decoder_strength": float(decoder_report.get("strength", 1.0) or 1.0)}
    combo_cells = [{**row, "checkpoint": combo_label} for row in decoder_report.get("cells") or []]
    if not combo_cells:
        raise ValueError("the decoder gate has no measured clips")
    joint = select_recommendation([*candidates, combo], [*cells, *combo_cells], plan_policy)
    chosen = next(row for row in joint["candidates"] if row["label"] == joint["recommended_label"])
    with_decoder = chosen["label"] == combo_label
    previous_label = str(report.get("recommended_label") or "")
    decision = (f"Base, the measured GPT checkpoints and {gated['label']} with its voice decoder adapter (strength "
                f"{combo['decoder_strength']:g}) were judged by the same Base guards and deployment score; "
                f"{chosen['label']} scored best ({chosen['deployment_score']['score']:+.4f} against Base).")
    return {"recommended_label": chosen["label"], "recommended_checkpoint": chosen["path"] if chosen["path"] else "",
            "recommended_kind": "adapter" if chosen["path"] else "base", "with_decoder": with_decoder,
            "decoder": combo["decoder"] if with_decoder else "", "decoder_strength": combo["decoder_strength"] if with_decoder else None,
            "gpt_label": gated["label"] if with_decoder else chosen["label"], "previous_label": previous_label,
            "changed": chosen["label"] != previous_label and not (with_decoder and previous_label == gated["label"]),
            "candidates": [{key: row.get(key) for key in ("label", "path", "eligible", "rejection_reasons", "notes", "deployment_score",
                                                          "mean_error_rate", "speaker_similarity_real", "speaker_similarity", "val_loss")}
                           for row in joint["candidates"]],
            "decision": decision, "speaker_metric": joint["speaker_metric"], "policy": plan_policy}


def apply_joint_recommendation(run_dir: str | Path, decoder_report: Mapping[str, Any]) -> dict[str, Any]:
    """Record the joint choice in the development report without touching its measured evidence.

    Only the recommendation fields, the decision text and the summary change; ``candidates``, ``cells``,
    ``inference`` and the other fingerprinted fields stay exactly as measured, so the decoder gate, the
    decoding sweep and the final-test freeze keep matching the report.
    """

    root = Path(run_dir) / "analysis" / "speech_evaluation"
    report_path = root / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "complete" or report.get("final_test"):
        raise ValueError("a completed development speech report is required")
    try:
        plan_policy = dict(json.loads((root / "plan.json").read_text(encoding="utf-8")).get("policy") or {})
    except (OSError, ValueError, TypeError):
        plan_policy = {}
    joint = joint_recommendation(report, decoder_report, plan_policy)
    fingerprint_before = development_fingerprint(run_dir)
    report["joint_selection"] = joint
    if joint["with_decoder"] or joint["changed"]:
        report["recommended_label"] = joint["recommended_label"] if not joint["with_decoder"] else joint["gpt_label"]
        report["recommended_checkpoint"] = joint["recommended_checkpoint"]
        report["recommended_kind"] = joint["recommended_kind"]
        report["recommended_deployment"] = {"checkpoint": joint["recommended_checkpoint"], "label": joint["recommended_label"],
                                            "decoder": joint["decoder"], "decoder_strength": joint["decoder_strength"]}
        report["decision"] = str(report.get("decision") or "") + " " + joint["decision"]
    report["summary_markdown"] = report_markdown(report)
    if development_fingerprint(run_dir) != fingerprint_before:
        raise ValueError("the joint recommendation must not change measured development evidence")
    atomic_write_json(report_path, report)
    (root / "report.md").write_text(report["summary_markdown"], encoding="utf-8")
    return joint


def main() -> int:
    from indextts.utils.console_encoding import configure_console_output
    from .train_config import TrainConfig
    configure_console_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--decoder-test", default="", help="Voice decoder adapter file to judge through the full pipeline")
    parser.add_argument("--decoding-sweep", action="store_true", help="Sweep decoding settings for --checkpoint on the benchmark")
    parser.add_argument("--final-test", action="store_true", help="Assess the frozen deployment only after all development tuning")
    parser.add_argument("--checkpoint", default="", help="GPT checkpoint the decoder test or decoding sweep generates with")
    args = parser.parse_args()
    try:
        if args.final_test:
            report = run_final_test(TrainConfig.from_json(args.config), args.state_dir, checkpoint_path=args.checkpoint)
        elif args.decoder_test:
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
