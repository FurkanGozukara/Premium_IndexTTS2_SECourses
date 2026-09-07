"""Per-adapter decoding sweep: temperature, guidance rate, and beams judged on the speech benchmark.

Every adapter so far generated with the same fixed decoding settings. After the decoder adapter is
judged, the selected checkpoint renders the benchmark sentences again with one setting changed at a
time; each render is compared pairwise with the checkpoint's measurement at the defaults (same
sentences, reference, seeds, and decoder adapter), and a change is adopted only when it raises the
score (speaker similarity gain minus four times the word-error increase) by a margin without
breaking the benchmark's regression policy. The winners are combined and checked once more, and
the result is saved as ``analysis/decoding.json`` so Voice Generation applies it with the adapter.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from indextts.utils.atomic_json import read_json_retry

from .dataset_manifest import atomic_write_json

DECODING_KNOBS: dict[str, tuple[float, ...]] = {
    "temperature": (0.6, 1.0),
    "inference_cfg_rate": (0.5, 1.0),
    "num_beams": (1, 5),
}
DECODING_WER_WEIGHT = 4.0
DECODING_MIN_SCORE_GAIN = 0.005
DECODING_KEYS = ("temperature", "inference_cfg_rate", "num_beams")


def _adapter_dir(path: str | Path) -> Path:
    from indextts.lora.decoder import adapter_root
    return adapter_root(path)


def decoding_settings_path(adapter_or_checkpoint_path: str | Path) -> Path:
    return _adapter_dir(adapter_or_checkpoint_path) / "analysis" / "decoding.json"


def load_decoding_settings(adapter_or_checkpoint_path: str | Path | None) -> dict[str, Any] | None:
    """The decoding settings a training's sweep adopted for its adapter, or None when none were."""
    if not adapter_or_checkpoint_path:
        return None
    value = read_json_retry(decoding_settings_path(adapter_or_checkpoint_path), None)
    if not isinstance(value, Mapping) or not value.get("accepted"):
        return None
    settings = value.get("settings")
    if not isinstance(settings, Mapping):
        return None
    try:
        return {"temperature": float(settings["temperature"]), "inference_cfg_rate": float(settings["inference_cfg_rate"]),
                "num_beams": int(settings["num_beams"]), "score": float(value.get("score", 0.0)),
                "generated_at": str(value.get("generated_at", ""))}
    except (KeyError, TypeError, ValueError):
        return None


def evaluate_variant(rows: Sequence[dict[str, Any]], baseline: Sequence[dict[str, Any]], *, policy: Mapping[str, Any],
                     wer_weight: float = DECODING_WER_WEIGHT, min_score_gain: float = DECODING_MIN_SCORE_GAIN) -> dict[str, Any]:
    """Pairwise judgement of one decoding variant against the baseline rows."""
    from .speech_metrics import MIN_REAL_SPEAKER_ROWS, paired_difference, summarize
    real_rows = lambda items: sum(row.get("speaker_similarity_real") is not None for row in items)
    metric = "speaker_similarity_real" if min(real_rows(rows), real_rows(baseline)) >= MIN_REAL_SPEAKER_ROWS else "speaker_similarity"
    gain = paired_difference(list(rows), list(baseline), metric)
    with_summary, without_summary = summarize(list(rows)), summarize(list(baseline))
    wer_increase = float(with_summary["corpus_error_rate"]) - float(without_summary["corpus_error_rate"])
    similarity = float(gain["mean"]) if gain.get("mean") is not None else None
    score = (similarity - wer_weight * wer_increase) if similarity is not None else float("-inf")
    reasons: list[str] = []
    if similarity is None:
        reasons.append("speaker similarity could not be compared")
    elif similarity < -float(policy.get("max_speaker_drop", 0.03)):
        reasons.append(f"speaker similarity fell by {-similarity:.4f}")
    if wer_increase > float(policy.get("max_wer_increase", 0.02)):
        reasons.append(f"the word error rate rose by {100 * wer_increase:.2f} points")
    if score < min_score_gain and not reasons:
        reasons.append(f"score {score:+.4f} is below the required +{min_score_gain:g}")
    return {"metric": metric, "speaker_gain": gain, "similarity": similarity, "wer_increase": wer_increase,
            "score": score, "summary": with_summary, "reasons": reasons, "passes": not reasons, "clips": len(rows)}


def select_decoding(base_settings: Mapping[str, Any], variants: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Per knob, the best passing single change; returns (combined settings, chosen single variants)."""
    winners: dict[str, dict[str, Any]] = {}
    for item in variants:
        if not item.get("passes") or item.get("kind") == "combined":
            continue
        knob = str(item["knob"])
        if knob not in winners or float(item["score"]) > float(winners[knob]["score"]):
            winners[knob] = dict(item)
    settings = dict(base_settings)
    for knob, item in winners.items():
        settings[knob] = item["value"]
    return settings, list(winners.values())


def decoding_markdown(report: Mapping[str, Any]) -> str:
    base = report["base_settings"]
    chosen = report.get("settings") or base
    verdict = "adopted" if report.get("accepted") else "kept the defaults"
    lines = [f"**Decoding sweep {verdict}.** Baseline temperature {base['temperature']:g}, guidance {base['inference_cfg_rate']:g}, "
             f"beams {int(base['num_beams'])}" + (f"; adopted temperature {chosen['temperature']:g}, guidance {chosen['inference_cfg_rate']:g}, "
                                                    f"beams {int(chosen['num_beams'])} (score {float(report.get('score', 0.0)):+.4f})." if report.get("accepted") else "."), ""]
    if report.get("reasons"):
        lines.extend(["Reasons: " + "; ".join(report["reasons"]) + ".", ""])
    lines.extend(["| Change | Similarity gain | Word error change | Score | Passes |", "|---|---:|---:|---:|---|"])
    for item in report.get("variants", []):
        label = (f"{item['knob']} {item['value']:g}" if item.get("kind") != "combined" else "combined winners")
        sim = item.get("similarity")
        lines.append(f"| {label}{' (chosen)' if item.get('chosen') else ''} | "
                     f"{sim:+.4f} | {100 * float(item['wer_increase']):+.2f} points | {float(item['score']):+.4f} | {'yes' if item.get('passes') else 'no'} |"
                     if sim is not None else f"| {label} | - | {100 * float(item['wer_increase']):+.2f} points | - | no |")
    lines.extend(["", f"Score: similarity gain minus {float(report.get('wer_weight', DECODING_WER_WEIGHT)):g} times the word-error increase; "
                      f"a change needs +{float(report.get('min_score_gain', DECODING_MIN_SCORE_GAIN)):g} and must stay within the benchmark's regression policy. "
                      "Same sentences, reference, seeds, and decoder adapter as the baseline measurement."])
    return "\n".join(lines) + "\n"


def run_decoding_sweep(config: Any, state_dir: str | Path, *, checkpoint_path: str,
                       knobs: Mapping[str, Sequence[float]] = DECODING_KNOBS, wer_weight: float = DECODING_WER_WEIGHT,
                       min_score_gain: float = DECODING_MIN_SCORE_GAIN) -> dict[str, Any]:
    """Sweep decoding settings for the selected checkpoint on the speech benchmark and save the winner."""
    from .speech_eval import (_benchmark_infer_kwargs, _benchmark_runtime, _lenient_terms, load_decoder_test,
                              render_benchmark_rows)
    run_dir = Path(config.output_dir).resolve() / config.name
    root = run_dir / "analysis" / "speech_evaluation"
    out = root / "decoding_sweep"
    checkpoint = str(Path(checkpoint_path).expanduser().resolve())
    state = Path(state_dir)
    state.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def cancelled() -> bool:
        return (state / "stop.flag").exists() or (run_dir / "stop.flag").exists()

    def update(message: str, completed: int, total: int) -> None:
        value = {"phase": "calibrating_decoding", "message": message, "desc": message, "completed": completed, "total": total,
                 "fraction": completed / total if total else 0, "elapsed_s": time.perf_counter() - started, "updated_at": time.time()}
        atomic_write_json(state / "status.json", value)
        atomic_write_json(state / "progress.json", value)
        print(f">> {message}: {completed}/{total}", flush=True)

    # Baseline: the checkpoint as it is deployed. With an accepted decoder adapter that is the decoder
    # test's chosen render; otherwise the benchmark's own measurement of the checkpoint.
    runtime = _benchmark_runtime(config).to_dict()
    decoder = load_decoder_test(run_dir)
    baseline_rows: list[dict[str, Any]] = []
    label = ""
    baseline_report = ""
    if decoder and decoder.get("accepted") and str(Path(str(decoder.get("checkpoint", ""))).resolve()) == checkpoint \
            and Path(str(decoder.get("adapter", ""))).is_file():
        baseline_rows = list(decoder["cells"])
        label = str(decoder["checkpoint_label"])
        baseline_report = str(decoder.get("baseline_report", ""))
        runtime["decoder_adapter"] = str(decoder["adapter"])
        runtime["decoder_adapter_strength"] = float(decoder.get("strength", 1.0))
    else:
        runtime["decoder_adapter"] = "none"
    plan: dict[str, Any] | None = None
    inference: dict[str, Any] = {}
    for base_root in (root / "final_test", root):
        try:
            report = json.loads((base_root / "report.json").read_text(encoding="utf-8"))
            candidate_plan = json.loads((base_root / "plan.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if not isinstance(report, dict) or report.get("status") != "complete":
            continue
        if baseline_report and str(Path(baseline_report).resolve()) != str((base_root / "report.json").resolve()):
            continue
        found = next((str(c["label"]) for c in report.get("candidates", [])
                      if c.get("path") and str(Path(c["path"]).resolve()) == checkpoint), "")
        rows = [row for row in report.get("cells", []) if found and row.get("checkpoint") == found]
        if rows:
            plan = candidate_plan
            inference = report.get("inference") if isinstance(report.get("inference"), dict) else {}
            if not baseline_rows:
                baseline_rows, label, baseline_report = rows, found, str(base_root / "report.json")
            break
    if plan is None or not baseline_rows:
        raise ValueError("the speech benchmark has no measurement of the selected checkpoint to sweep decoding settings against")
    infer = dict(inference.get("infer_kwargs") or {}) or _benchmark_infer_kwargs(config)
    base_settings = {"temperature": float(infer.get("temperature", 0.8)), "inference_cfg_rate": float(infer.get("inference_cfg_rate", 0.7)),
                     "num_beams": int(infer.get("num_beams", 3) or 1)}
    policy = dict(plan.get("policy") or {})
    lenient = _lenient_terms(config, plan)
    attempt = out / "grids" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    variants: list[dict[str, Any]] = []

    def render(settings: Mapping[str, Any], suffix: str) -> tuple[list[dict[str, Any]], list[str]]:
        trial = dict(infer)
        trial.update({key: (int(value) if key == "num_beams" else float(value)) for key, value in settings.items()})
        return render_benchmark_rows(config, plan, run_dir=run_dir, checkpoint=checkpoint, label=label, runtime=runtime,
                                     infer=trial, attempt=attempt, out=out, state=state, update=update, cancelled=cancelled,
                                     lenient=lenient, grid_suffix=suffix, baseline=baseline_rows)

    for knob, alternatives in knobs.items():
        for value in alternatives:
            current = base_settings.get(knob)
            if current is not None and float(value) == float(current):
                continue
            settings = {**base_settings, knob: value}
            rows, grids = render(settings, f"_{knob}_{float(value):g}".replace(".", "_"))
            verdict = evaluate_variant(rows, baseline_rows, policy=policy, wer_weight=wer_weight, min_score_gain=min_score_gain)
            variants.append({"kind": "single", "knob": knob, "value": value, "settings": settings, "grids": grids, "chosen": False, **verdict})
            print(f">> decoding {knob} {value:g}: similarity {verdict['similarity'] if verdict['similarity'] is None else round(verdict['similarity'], 4):+} "
                  f"| word error {100 * verdict['wer_increase']:+.2f} points | score {verdict['score']:+.4f} | "
                  f"{'passes' if verdict['passes'] else '; '.join(verdict['reasons'])}", flush=True)
    combined_settings, winners = select_decoding(base_settings, variants)
    candidates = list(winners)
    if len(winners) >= 2:
        rows, grids = render(combined_settings, "_combined")
        verdict = evaluate_variant(rows, baseline_rows, policy=policy, wer_weight=wer_weight, min_score_gain=min_score_gain)
        combined = {"kind": "combined", "knob": "combined", "value": 0.0, "settings": combined_settings, "grids": grids, "chosen": False, **verdict}
        variants.append(combined)
        candidates.append(combined)
    passing = [item for item in candidates if item.get("passes")]
    chosen = max(passing, key=lambda item: float(item["score"])) if passing else None
    if chosen is not None:
        chosen["chosen"] = True
    accepted = chosen is not None
    report = {"status": "complete", "accepted": accepted, "settings": dict(chosen["settings"]) if chosen else dict(base_settings),
              "base_settings": base_settings, "score": float(chosen["score"]) if chosen else 0.0,
              "reasons": [] if accepted else ["no single or combined change beat the defaults by the required margin"],
              "variants": [{key: value for key, value in item.items()} for item in variants], "wer_weight": wer_weight,
              "min_score_gain": min_score_gain, "policy": policy, "checkpoint": checkpoint, "checkpoint_label": label,
              "decoder_adapter": runtime.get("decoder_adapter", "none"), "decoder_adapter_strength": runtime.get("decoder_adapter_strength", 1.0),
              "baseline_report": baseline_report, "baseline_clips": len(baseline_rows), "seeds": plan["seeds"],
              "generated_at": datetime.now(timezone.utc).isoformat(), "elapsed_s": time.perf_counter() - started}
    report["summary_markdown"] = decoding_markdown(report)
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / "report.json", report)
    (out / "report.md").write_text(report["summary_markdown"], encoding="utf-8")
    atomic_write_json(run_dir / "analysis" / "decoding.json", {
        "accepted": accepted, "settings": report["settings"], "base_settings": base_settings, "score": report["score"],
        "checkpoint": checkpoint, "decoder_adapter": report["decoder_adapter"], "generated_at": report["generated_at"],
        "report": str(out / "report.json")})
    atomic_write_json(state / "status.json", {"phase": "complete", "message": "Decoding sweep complete", "elapsed_s": report["elapsed_s"]})
    return report


__all__ = [
    "DECODING_KEYS",
    "DECODING_KNOBS",
    "decoding_markdown",
    "decoding_settings_path",
    "evaluate_variant",
    "load_decoding_settings",
    "run_decoding_sweep",
    "select_decoding",
]
