"""Language-aware, paired speech measurements; no model is loaded on import."""
from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
import gc
import math
from pathlib import Path
import re
import unicodedata
from typing import Any, Callable, Iterable

import numpy as np


LANGUAGES = {"EN": "english", "ZH": "chinese", "JA": "japanese", "AR": "arabic", "ES": "spanish"}

# Spoken forms that subtitles and ASR write differently. Both sides receive the
# same expansion, so these never count as transcript errors.
_ENGLISH_CONTRACTIONS = {
    "i'm": "i am", "it's": "it is", "that's": "that is", "there's": "there is", "here's": "here is",
    "what's": "what is", "let's": "let us", "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "cannot", "won't": "will not", "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "wouldn't": "would not", "couldn't": "could not", "shouldn't": "should not",
    "you're": "you are", "we're": "we are", "they're": "they are",
    "i'll": "i will", "you'll": "you will", "we'll": "we will", "it'll": "it will", "they'll": "they will",
    "i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have",
    "i'd": "i would", "you'd": "you would", "we'd": "we would", "they'd": "they would",
    "gonna": "going to", "wanna": "want to", "ok": "okay",
}
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'"})
_CONTRACTION_RE = re.compile(r"\b[a-z]+(?:'[a-z]+)?\b")
_UNITS = {"gb": "gigabytes", "mb": "megabytes", "kb": "kilobytes", "tb": "terabytes", "ghz": "gigahertz", "mhz": "megahertz",
          "khz": "kilohertz", "ms": "milliseconds"}
_UNIT_RE = re.compile(r"(\d)\s*(gb|mb|kb|tb|ghz|mhz|khz|ms)\b")
_CURRENCY_RE = re.compile(r"\$\s*(\d+)(?:\.(\d{1,2}))?")
_DECIMAL_RE = re.compile(r"(\d)\.(\d)")
# A substituted project term still counts as agreement when the recognizer's
# spelling shares at least this much of the reference spelling.
LENIENT_TERM_RATIO = 0.45
# An edge word that is replaced by a similar-looking word or split into pieces
# is transcript noise; a missing or extra edge word is a real boundary problem.
EDGE_SUBSTITUTION_RATIO = 0.6
# Matched real recordings needed before speaker similarity is judged against them.
MIN_REAL_SPEAKER_ROWS = 4
# Internal pauses: quiet stretches at least this long between words or sentences.
PAUSE_HOP_MS = 10
MIN_PAUSE_MS = 120
PAUSE_THRESHOLD_DBFS = -40.0
PAUSE_RELATIVE_DB = 25.0
PAUSE_NO_SIGNAL_DBFS = -80.0


def _currency(match: re.Match) -> str:
    dollars, cents = int(match.group(1)), match.group(2)
    cents_value = int(cents.ljust(2, "0")) if cents else 0
    parts = []
    if dollars or not cents_value:
        parts.append(f"{dollars} dollar" + ("" if dollars == 1 else "s"))
    if cents_value:
        parts.append(f"{cents_value} cent" + ("" if cents_value == 1 else "s"))
    return " " + " ".join(parts) + " "


def _english_spoken_form(text: str) -> str:
    text = text.translate(_APOSTROPHES).replace("%", " percent ")
    text = _CURRENCY_RE.sub(_currency, text)
    text = _UNIT_RE.sub(lambda m: f"{m.group(1)} {_UNITS[m.group(2)]}", text)
    text = _DECIMAL_RE.sub(r"\1 point \2", text)
    return _CONTRACTION_RE.sub(lambda match: _ENGLISH_CONTRACTIONS.get(match.group(0), match.group(0)), text)


def internal_pause_metrics(samples: Any, sample_rate: int) -> dict[str, Any]:
    """Time spent in pauses inside a clip: total, count, and the longest run.

    Leading and trailing quiet audio is excluded, so the numbers describe the
    silences between words and sentences. A frame is quiet below
    ``PAUSE_THRESHOLD_DBFS``; in a recording quieter than that, it must be
    ``PAUSE_RELATIVE_DB`` under the loudest frame, so a quiet recording is not
    measured as one long pause. A clip with no signal has no pauses.
    Compared with the real recording of the same sentence, this shows whether a
    voice rushes between sentences or lingers longer than the person does.
    """
    array = np.asarray(samples, dtype=np.float32).reshape(-1)
    hop = max(1, int(round(sample_rate * PAUSE_HOP_MS / 1000)))
    frames = len(array) // hop
    empty = {"speech_span_s": 0.0, "pause_s": 0.0, "pause_time_fraction": None, "pause_count": 0, "longest_pause_ms": 0}
    if frames == 0:
        return empty
    rms = np.sqrt(np.mean(np.square(array[:frames * hop]).reshape(frames, hop), axis=1) + 1e-12)
    level = 20.0 * np.log10(rms)
    loudest = float(level.max())
    if loudest < PAUSE_NO_SIGNAL_DBFS:
        return empty
    quiet = level < min(PAUSE_THRESHOLD_DBFS, loudest - PAUSE_RELATIVE_DB)
    loud = np.flatnonzero(~quiet)
    if len(loud) == 0:
        return empty
    inner = quiet[loud[0]:loud[-1] + 1]
    edges = np.diff(np.pad(inner.astype(np.int8), (1, 1)))
    runs = [(end - start) * PAUSE_HOP_MS for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
            if (end - start) * PAUSE_HOP_MS >= MIN_PAUSE_MS]
    span = len(inner) * PAUSE_HOP_MS / 1000.0
    pause = sum(runs) / 1000.0
    return {"speech_span_s": span, "pause_s": pause, "pause_time_fraction": pause / span if span else None,
            "pause_count": len(runs), "longest_pause_ms": int(max(runs)) if runs else 0}


def transcript_units(text: str, language: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    if language == "EN":
        from .dataset_quality import normalized_words
        return normalized_words(_english_spoken_form(text))
    text = "".join(" " if unicodedata.category(c)[0] in {"P", "S"} else c for c in text)
    if language in {"ZH", "JA"}:
        return [c for c in text if not c.isspace()]
    return text.split()


def lenient_units(terms: Iterable[str], language: str) -> frozenset[str]:
    """Normalized units of transcript vocabulary whose ASR spellings may be forgiven."""
    units: set[str] = set()
    for term in terms:
        if str(term).strip().isdigit():
            continue  # Numbers are ordinary vocabulary, not project spellings.
        units.update(transcript_units(str(term), language))
    return frozenset(units)


def _alignment_blocks(ref: list[str], hyp: list[str]) -> list[tuple[list[str], list[str], bool, int]]:
    """Minimal edit alignment grouped into (reference, hypothesis, equal, operations) blocks."""
    n, m = len(ref), len(hyp)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i
    for j in range(1, m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(cost[i - 1][j] + 1, cost[i][j - 1] + 1, cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops: list[tuple[str, str | None, str | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("equal" if ref[i - 1] == hyp[j - 1] else "sub", ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, hyp[j - 1]))
            j -= 1
    ops.reverse()
    blocks: list[tuple[list[str], list[str], bool, int]] = []
    for tag, r, h in ops:
        if tag == "equal":
            blocks.append(([r], [h], True, 0))
            continue
        if blocks and not blocks[-1][2]:
            r_seg, h_seg, _, count = blocks[-1]
            blocks[-1] = (r_seg + ([r] if r is not None else []), h_seg + ([h] if h is not None else []), False, count + 1)
        else:
            blocks.append(([r] if r is not None else [], [h] if h is not None else [], False, 1))
    return blocks


def _block_status(r_seg: list[str], h_seg: list[str], equal: bool, lenient: frozenset[str] | None) -> str:
    if equal:
        return "equal"
    if r_seg and h_seg and "".join(r_seg) == "".join(h_seg):
        return "equal"  # "Swarm UI" and "SwarmUI" are one spoken word.
    if lenient and r_seg and h_seg and all(unit in lenient for unit in r_seg):
        if SequenceMatcher(None, "".join(r_seg), "".join(h_seg)).ratio() >= LENIENT_TERM_RATIO:
            return "forgiven"
    return "error"


def _edge_substitution(r_seg: list[str], h_seg: list[str]) -> bool:
    """A recognizer word replaced by a similar word or split into pieces, not a missing or extra word."""
    if not r_seg or not h_seg:
        return False
    if len(r_seg) == len(h_seg):
        return True
    return SequenceMatcher(None, "".join(r_seg), "".join(h_seg)).ratio() >= EDGE_SUBSTITUTION_RATIO


def _edge_matches(blocks: list[tuple[list[str], list[str], bool, int]], statuses: list[str], edge: int, *, from_end: bool) -> bool:
    """Both edges must keep every reference word.

    Missing or extra words at an edge fail: they indicate a cut or extra speech
    the transcript does not contain. A substituted edge word still counts as an
    error in the error rate, but the acoustic edge check guards against cuts, so
    it does not by itself reject the clip.
    """
    consumed = 0
    order = range(len(blocks) - 1, -1, -1) if from_end else range(len(blocks))
    for index in order:
        r_seg, h_seg, _, _ = blocks[index]
        if statuses[index] == "error" and not _edge_substitution(r_seg, h_seg):
            return False
        consumed += len(r_seg)
        if consumed >= edge:
            return True
    return consumed >= edge


def transcript_metrics(reference: str, hypothesis: str, language: str, *,
                       lenient_terms: Iterable[str] | None = None) -> dict[str, Any]:
    """Compare a transcript with recognized speech.

    ``lenient_terms`` holds normalized units (see :func:`lenient_units`) of the
    speaker's own spellings, such as product names. A recognizer that spells one
    of them differently does not produce an error; missing, extra, or otherwise
    different words still do.
    """
    ref, hyp = transcript_units(reference, language), transcript_units(hypothesis, language)
    if not ref:
        raise ValueError("Speech evaluation text contains no scoreable units")
    lenient = frozenset(lenient_terms) if lenient_terms else None
    blocks = _alignment_blocks(ref, hyp)
    statuses = [_block_status(r_seg, h_seg, equal, lenient) for r_seg, h_seg, equal, _ in blocks]
    errors = sum(count for (_, _, _, count), status in zip(blocks, statuses) if status == "error")
    forgiven = sum(count for (_, _, _, count), status in zip(blocks, statuses) if status == "forgiven")
    edge = min(2, len(ref))
    end_matches = _edge_matches(blocks, statuses, edge, from_end=True)
    n = 6 if language in {"ZH", "JA"} else 3
    def grams(units: list[str]) -> Counter:
        return Counter(tuple(units[i:i+n]) for i in range(len(units)-n+1))
    expected, observed = grams(ref), grams(hyp)
    repetition = any(count >= max(3, expected[gram] + 2) for gram, count in observed.items())
    return {"errors": errors, "units": len(ref), "error_rate": errors / len(ref),
            "error_unit": "character" if language in {"ZH", "JA"} else "word",
            "start_matches": _edge_matches(blocks, statuses, edge, from_end=False), "end_matches": end_matches,
            "possible_truncation": len(hyp) < 0.6 * len(ref) and not end_matches,
            "possible_repetition": repetition, "forgiven_units": forgiven}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No speech measurements to summarize")
    def mean(key: str) -> float | None:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return float(np.mean(values)) if values else None

    def pause_ratio() -> float | None:
        # Total pause time over the matched sentences, generated against real; a
        # per-clip mean would be dominated by sentences the person spoke without a pause.
        paired = [(float(row["pause_s"]), float(row["real_pause_s"])) for row in rows
                  if row.get("pause_s") is not None and row.get("real_pause_s")]
        if paired and sum(real for _, real in paired) > 0:
            return sum(generated for generated, _ in paired) / sum(real for _, real in paired)
        return mean("pause_ratio_vs_real")
    return {"clips": len(rows), "mean_error_rate": mean("error_rate"),
            "corpus_error_rate": sum(row["errors"] for row in rows) / sum(row["units"] for row in rows),
            "worst_error_rate": max(row["error_rate"] for row in rows),
            "speaker_similarity": mean("speaker_similarity"), "speaker_similarity_real": mean("speaker_similarity_real"),
            "style_similarity_real": mean("style_similarity_real"),
            "duration_ratio_vs_real": mean("duration_ratio_vs_real"),
            "pause_time_fraction": mean("pause_time_fraction"), "pause_ratio_vs_real": pause_ratio(),
            "failure_count": sum(bool(row["invalid_audio"] or row["possible_truncation"] or row["possible_repetition"]) for row in rows),
            "edge_mismatch_count": sum(not row["start_matches"] or not row["end_matches"] for row in rows),
            "invalid_audio_count": sum(bool(row["invalid_audio"]) for row in rows)}


def paired_difference(rows: list[dict[str, Any]], baseline: list[dict[str, Any]], key: str,
                      seed: int = 42) -> dict[str, Any]:
    """Bootstrap whole prompts, keeping a prompt's generation seeds together."""
    base = {(r["prompt_id"], r["seed"]): r for r in baseline}
    if len(base) != len(baseline) or len(rows) != len(base):
        raise ValueError("Paired speech comparisons require unique, complete prompt/seed coverage")
    differences: dict[str, list[float]] = defaultdict(list)
    seen = set()
    for row in rows:
        pair = (row["prompt_id"], row["seed"])
        if pair in seen or pair not in base:
            raise ValueError("Mismatched prompt/seed coverage in speech evaluation")
        seen.add(pair)
        if row.get(key) is not None and base[pair].get(key) is not None:
            differences[row["prompt_id"]].append(float(row[key]) - float(base[pair][key]))
    values = np.asarray([np.mean(group) for group in differences.values()], dtype=float)
    if not len(values):
        return {"mean": None, "ci95": None, "prompts": 0}
    sampled = np.random.default_rng(seed).choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
    return {"mean": float(values.mean()), "ci95": [float(x) for x in np.quantile(sampled, [0.025, 0.975])],
            "prompts": len(values), "cluster": "prompt (all generation seeds together)"}


# The deployment score ranks the candidates that pass the Base guards, the same way the voice decoder
# gate judges an adapter: the paired speaker-similarity gain over Base, minus SCORE_WER_WEIGHT times any
# paired word-error increase (one point of word error costs 0.04 of similarity), plus a pause term that
# rewards pausing more like the person than Base does (a total pause-time ratio moving from 1.5 to 1.0
# of the real recordings' earns about 0.02). Candidates within SCORE_MIN_DELTA are tied and the lower
# validation loss decides, so a run never selects on a difference the benchmark cannot resolve.
SCORE_WER_WEIGHT = 4.0
SCORE_PAUSE_WEIGHT = 0.05
SCORE_MIN_DELTA = 0.002
# Guard modes. "mean" rejects a candidate whose mean paired regression exceeds the margin. "interval" also
# needs the regression to be resolved by the benchmark: its prompt-bootstrap 95% interval must exclude zero,
# or a majority of the held-out recordings must show it. A margin crossed by a few sentences of one recording
# then costs deployment score instead of disqualifying the candidate outright.
GUARD_MODE_MEAN = "mean"
GUARD_MODE_INTERVAL = "interval"


def deployment_score(summary: dict[str, Any], base: dict[str, Any], speaker_gain: float | None,
                     error_increase: float | None, *, wer_weight: float = SCORE_WER_WEIGHT,
                     pause_weight: float = SCORE_PAUSE_WEIGHT) -> dict[str, float]:
    """Speaker gain minus weighted word-error increase plus the pause term, with its parts."""
    gain = float(speaker_gain) if speaker_gain is not None else 0.0
    penalty = float(wer_weight) * max(0.0, float(error_increase)) if error_increase is not None else 0.0
    pause_term = 0.0
    candidate_ratio, base_ratio = summary.get("pause_ratio_vs_real"), base.get("pause_ratio_vs_real")
    if candidate_ratio and base_ratio and float(candidate_ratio) > 0 and float(base_ratio) > 0:
        pause_term = float(pause_weight) * (abs(math.log(float(base_ratio))) - abs(math.log(float(candidate_ratio))))
    return {"score": gain - penalty + pause_term, "speaker_gain": gain, "wer_penalty": penalty, "pause_term": pause_term}


def source_regression(rows: list[dict[str, Any]], baseline: list[dict[str, Any]], key: str, *,
                      worse_when_higher: bool) -> dict[str, Any]:
    """How many held-out recordings show the regression: per-source mean paired difference in the bad direction."""
    base = {(r["prompt_id"], r["seed"]): r for r in baseline}
    per_source: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        pair = (row["prompt_id"], row["seed"])
        other = base.get(pair)
        if other is None or row.get(key) is None or other.get(key) is None:
            continue
        per_source[str(row.get("source") or row["prompt_id"])].append(float(row[key]) - float(other[key]))
    means = {source: float(np.mean(values)) for source, values in per_source.items() if values}
    regressed = [source for source, value in means.items() if (value > 0 if worse_when_higher else value < 0)]
    return {"sources": len(means), "regressed": len(regressed), "regressed_sources": sorted(regressed),
            "majority": bool(means) and len(regressed) * 2 > len(means), "per_source": means}


def _resolved_regression(delta: dict[str, Any], sources: dict[str, Any], *, worse_when_higher: bool) -> bool:
    ci = delta.get("ci95")
    if ci:
        if worse_when_higher and float(ci[0]) > 0:
            return True
        if not worse_when_higher and float(ci[1]) < 0:
            return True
    return bool(sources.get("majority"))


def regression_guards(measured: list[dict[str, Any]], base_rows: list[dict[str, Any]], *, policy: dict[str, Any],
                      speaker_metric: str, summary: dict[str, Any] | None = None,
                      base: dict[str, Any] | None = None) -> dict[str, Any]:
    """Base-regression guards for one candidate: reasons that reject it, notes about margins crossed by noise."""
    summary = summary or summarize(measured)
    base = base or summarize(base_rows)
    mode = str(policy.get("guard_mode") or GUARD_MODE_MEAN).strip().lower()
    max_wer = float(policy["max_wer_increase"])
    max_drop = float(policy["max_speaker_drop"])
    delta = paired_difference(measured, base_rows, "error_rate")
    speaker = paired_difference(measured, base_rows, speaker_metric)
    wer_sources = source_regression(measured, base_rows, "error_rate", worse_when_higher=True)
    speaker_sources = source_regression(measured, base_rows, speaker_metric, worse_when_higher=False)
    reasons: list[str] = []
    notes: list[str] = []
    if delta["mean"] is not None and delta["mean"] > max_wer:
        if mode != GUARD_MODE_INTERVAL or _resolved_regression(delta, wer_sources, worse_when_higher=True):
            detail = ""
            if mode == GUARD_MODE_INTERVAL:
                detail = (" (the 95% interval excludes zero)" if delta.get("ci95") and float(delta["ci95"][0]) > 0
                          else f" (higher on {wer_sources['regressed']} of {wer_sources['sources']} recordings)")
            reasons.append("transcript error exceeds the allowed increase over Base" + detail)
        else:
            notes.append(f"mean transcript error rose {100 * delta['mean']:+.2f} points, more than the {100 * max_wer:.0f}-point "
                         f"margin, but the 95% interval includes zero and only {wer_sources['regressed']} of "
                         f"{wer_sources['sources']} recordings got worse; the increase costs deployment score instead")
    if speaker["mean"] is None:
        reasons.append("speaker similarity could not be compared with Base")
    elif speaker["mean"] < -max_drop:
        if mode != GUARD_MODE_INTERVAL or _resolved_regression(speaker, speaker_sources, worse_when_higher=False):
            detail = ""
            if mode == GUARD_MODE_INTERVAL:
                detail = (" (the 95% interval excludes zero)" if speaker.get("ci95") and float(speaker["ci95"][1]) < 0
                          else f" (lower on {speaker_sources['regressed']} of {speaker_sources['sources']} recordings)")
            reasons.append("speaker similarity falls below the allowed Base margin" + detail)
        else:
            notes.append(f"mean speaker similarity fell {speaker['mean']:+.4f}, more than the {max_drop:g} margin, but the 95% "
                         f"interval includes zero and only {speaker_sources['regressed']} of {speaker_sources['sources']} "
                         "recordings got worse")
    if summary["failure_count"] > base["failure_count"]:
        reasons.append("more invalid, possibly truncated, or repetitive clips than Base")
    return {"reasons": reasons, "notes": notes, "error_delta": delta, "speaker_delta": speaker,
            "error_sources": wer_sources, "speaker_sources": speaker_sources, "guard_mode": mode}


def select_recommendation(candidates: list[dict[str, Any]], rows: list[dict[str, Any]],
                          policy: dict[str, Any]) -> dict[str, Any]:
    for row in rows:
        for key in ("error_rate", "speaker_similarity"):
            if row.get(key) is not None and not math.isfinite(float(row[key])):
                raise FloatingPointError(f"Non-finite {key}; refusing speech checkpoint selection")
    grouped = {item["label"]: [row for row in rows if row["checkpoint"] == item["label"]] for item in candidates}
    base_rows = grouped.get("Base", [])
    base = summarize(base_rows)
    if base["invalid_audio_count"] == base["clips"]:
        raise ValueError("Base produced no valid audio; the speech benchmark cannot make a reliable recommendation")
    # Similarity to the reference clip rewards copying that one prompt; similarity
    # to the real recording of the same sentence measures the speaker's identity.
    # Prefer the latter whenever enough matched real recordings exist.
    speaker_metric = "speaker_similarity"
    if sum(row.get("speaker_similarity_real") is not None for row in base_rows) >= MIN_REAL_SPEAKER_ROWS and all(
            sum(row.get("speaker_similarity_real") is not None for row in grouped[c["label"]]) >= MIN_REAL_SPEAKER_ROWS
            for c in candidates):
        speaker_metric = "speaker_similarity_real"
    wer_weight = float(policy.get("score_wer_weight", SCORE_WER_WEIGHT))
    pause_weight = float(policy.get("score_pause_weight", SCORE_PAUSE_WEIGHT))
    score_min_delta = float(policy.get("score_min_delta", SCORE_MIN_DELTA))
    guard_mode = str(policy.get("guard_mode") or GUARD_MODE_MEAN).strip().lower()
    results = []
    for candidate in candidates:
        measured = grouped[candidate["label"]]
        summary = summarize(measured)
        guards = regression_guards(measured, base_rows, policy=policy, speaker_metric=speaker_metric, summary=summary, base=base)
        delta, speaker = guards["error_delta"], guards["speaker_delta"]
        reasons = list(guards["reasons"]) if candidate["path"] else []
        notes = list(guards["notes"]) if candidate["path"] else []
        score = deployment_score(summary, base, speaker["mean"], delta["mean"], wer_weight=wer_weight, pause_weight=pause_weight)
        results.append({**candidate, **summary, "error_delta_vs_base": delta, "speaker_metric": speaker_metric,
                        "speaker_delta_vs_base": speaker, "deployment_score": score, "eligible": not reasons,
                        "rejection_reasons": reasons, "notes": notes,
                        "error_sources": guards["error_sources"], "speaker_sources": guards["speaker_sources"]})
    eligible = [r for r in results if r["eligible"]]
    top = max(r["deployment_score"]["score"] for r in eligible)
    # Candidates the score cannot separate are tied; the lower validation loss decides among them.
    tied = [r for r in eligible if r["deployment_score"]["score"] >= top - score_min_delta]
    best = min(tied, key=lambda r: (float(r.get("val_loss") if r.get("val_loss") is not None else float("inf")),
                                    r["mean_error_rate"], r["label"]))
    speaker_note = ("speaker similarity is measured against the real recording of each sentence"
                    if speaker_metric == "speaker_similarity_real" else "speaker similarity is measured against the reference clip")
    guard_note = (" A margin crossed only within the benchmark's noise (95% interval including zero, a minority of recordings) "
                  "costs score instead of disqualifying." if guard_mode == GUARD_MODE_INTERVAL else "")
    return {"status": "complete", "recommended_kind": "adapter" if best["path"] else "base",
            "recommended_checkpoint": best["path"], "recommended_label": best["label"],
            "candidates": results, "listening_status": "not_rated", "speaker_metric": speaker_metric,
            "score_policy": {"wer_weight": wer_weight, "pause_weight": pause_weight, "min_delta": score_min_delta,
                             "guard_mode": guard_mode},
            "decision": (f"Observed Base regression guards ({speaker_note}), then the deployment score: paired speaker-similarity "
                         f"gain over Base minus {wer_weight:g} times any paired word-error increase, plus a pause-time term "
                         f"(weight {pause_weight:g}); validation loss breaks ties within {score_min_delta:g}.{guard_note}"),
            "scope": "Provisional automatic recommendation for this development suite; human listening is still needed to judge naturalness."}


def measure_clips(clips: list[dict[str, Any]], *, model_dir: str, model_config: str, device: str,
                  output_dir: Path, update: Callable[[str, int, int], None],
                  cancelled: Callable[[], bool], lenient_terms: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Measure entire transcripts and distributed 20-second embedding windows.

    ``lenient_terms`` are normalized units of the dataset's own spellings (see
    :func:`lenient_units`); recognizer spellings of those words are not errors.
    """
    lenient = frozenset(lenient_terms) if lenient_terms else None
    import torch
    from transformers import pipeline
    from indextts.runtime import ProgressReporter
    from .features import FeatureCacheConfig, _FeatureModels, _load_audio_16k
    from .whisper_asr import _ensure_model
    from .dataset_manifest import atomic_write_json

    requests = {}
    for clip in clips:
        for key in ("audio", "reference", "real_audio"):
            if clip.get(key):
                requests[(clip[key], clip["language"])] = None
    features = {}
    models = _FeatureModels(FeatureCacheConfig(str(output_dir), model_dir=model_dir,
                                              model_config=model_config, device=device), ProgressReporter("speech clips"))
    for index, path in enumerate(sorted({path for path, _ in requests}), 1):
        if cancelled():
            raise InterruptedError("Speech evaluation canceled")
        wave, duration = _load_audio_16k(Path(path))
        array = wave.numpy()
        invalid = not np.isfinite(array).all() or duration < 0.25 or float(np.sqrt(np.mean(np.square(array)))) < 1e-5
        entry = {"duration_s": duration, "invalid_audio": invalid, "speaker": None, "style": None, "pauses": None}
        if not invalid:
            entry["pauses"] = internal_pause_metrics(array, 16000)
            length = min(wave.shape[-1], 20 * 16000)
            starts = sorted(set([0, (wave.shape[-1] - length) // 2, wave.shape[-1] - length]))
            speakers, styles = [], []
            for start in starts:
                snippet = wave[:, start:start+length]
                semantic = models.w2v_features([snippet])[0]
                _, speaker, _, style = models.item_features(snippet, semantic)
                if not torch.isfinite(speaker).all() or not torch.isfinite(style).all():
                    raise FloatingPointError(f"Non-finite speech embedding for {path}")
                speakers.append(torch.nn.functional.normalize(speaker, dim=0))
                styles.append(torch.nn.functional.normalize(style, dim=0))
            entry.update(speaker=torch.nn.functional.normalize(torch.stack(speakers).mean(0), dim=0),
                         style=torch.nn.functional.normalize(torch.stack(styles).mean(0), dim=0))
        features[path] = entry
        update("Measuring speaker and style", index, len({p for p, _ in requests}))
    del models
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if any(features[clip["reference"]]["invalid_audio"] for clip in clips):
        raise ValueError("A benchmark voice reference is empty, silent, or non-finite; choose a usable training reference")
    whisper = "openai/whisper-large-v3-turbo"
    pipe = pipeline("automatic-speech-recognition", model=str(_ensure_model(whisper)), device=device,
                    dtype=torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32)
    transcripts = {}
    for index, (path, language) in enumerate(sorted(requests), 1):
        if cancelled():
            raise InterruptedError("Speech evaluation canceled")
        text = ""
        if not features[path]["invalid_audio"]:
            wave, _ = _load_audio_16k(Path(path))
            result = pipe({"array": wave.squeeze().numpy(), "sampling_rate": 16000}, return_timestamps=True,
                          generate_kwargs={"language": LANGUAGES[language], "task": "transcribe", "do_sample": False})
            text = str(result["text"]).strip()
        transcripts[(path, language)] = text
        update("Transcribing evaluation audio", index, len(requests))
        atomic_write_json(output_dir / "transcriptions.json", [
            {"audio": p, "language": lang, "text": text} for (p, lang), text in transcripts.items()])
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    measured = []
    for clip in clips:
        feature, ref = features[clip["audio"]], features[clip["reference"]]
        text = transcripts[(clip["audio"], clip["language"])]
        def similarity(left: dict, right: dict, key: str) -> float | None:
            if left[key] is None or right[key] is None:
                return None
            return float(torch.dot(left[key], right[key]))
        row = {**clip, **transcript_metrics(clip["text"], text, clip["language"], lenient_terms=lenient), "asr_text": text,
               "duration_s": feature["duration_s"], "invalid_audio": feature["invalid_audio"],
               "speaker_similarity": similarity(feature, ref, "speaker"), "speaker_similarity_real": None}
        pauses = feature.get("pauses") or {}
        row.update(pause_time_fraction=pauses.get("pause_time_fraction"), pause_count=pauses.get("pause_count"),
                   longest_pause_ms=pauses.get("longest_pause_ms"), pause_s=pauses.get("pause_s"),
                   real_pause_s=None, pause_ratio_vs_real=None)
        if clip.get("real_audio"):
            real = features[clip["real_audio"]]
            real_pauses = real.get("pauses") or {}
            row.update(style_similarity_real=similarity(feature, real, "style"),
                       speaker_similarity_real=similarity(feature, real, "speaker"),
                       duration_ratio_vs_real=feature["duration_s"] / max(0.001, real["duration_s"]))
            if real_pauses.get("pause_s") is not None:
                row["real_pause_s"] = float(real_pauses["pause_s"])
            if real_pauses.get("pause_s"):
                # Above 1.0 the voice pauses longer than the person did on the same sentence; below 1.0 it rushes.
                row["pause_ratio_vs_real"] = float(pauses.get("pause_s", 0.0)) / float(real_pauses["pause_s"])
        measured.append(row)
    return measured
