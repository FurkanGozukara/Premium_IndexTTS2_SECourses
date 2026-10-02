"""Reference audition: the clip a trained voice clones best from, chosen by listening to the results.

A cloned voice copies its reference clip's timbre, recording conditions and delivery. In the round-3
comparison (October 2026) four clips of one speaker gave a fine-tuned OmniVoice voice speaker similarities
between 0.725 and 0.817 to the speaker's real recordings, and neither the clip length nor a typical pitch,
pace or intonation range predicted which clip was good. An audition therefore renders a few held-out
sentences with each candidate clip and measures every take against the speaker's own recording of the same
sentence (speaker and style similarity and Whisper word errors, the measurements of the training's speech
evaluation). The candidate whose takes sound most like the speaker wins.

The voice's original reference file is never changed (accepted work may cite it). A winner is copied to
``<run>/<run>_audition_choice.<ext>`` with its transcript, and ``analysis/reference_audition.json`` records
the audition; while its ``use`` flag is set, the voice's automatic reference is the winner.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

AUDITION_FILE = "reference_audition.json"
CHOICE_STEM = "audition_choice"
CANDIDATE_SECONDS = (6.0, 16.0)
SENTENCE_SECONDS = (5.0, 14.0)
TYPICALITY_POOL = 60  # clean clips measured for typical pitch and pace before candidates are picked
WER_GUARD = 0.05  # a winner may not have more than five points more word errors than the best candidate


def run_dir_of(adapter_path: str | Path) -> Path:
    source = Path(adapter_path).expanduser().resolve()
    return source.parent.parent if source.parent.name.casefold() == "best" else source.parent


def audition_record(adapter_path: str | Path | None) -> dict[str, Any] | None:
    if not adapter_path:
        return None
    path = run_dir_of(adapter_path) / "analysis" / AUDITION_FILE
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def audition_reference(adapter_path: str | Path | None) -> str | None:
    """The audition winner a voice clones from, while its use is switched on."""

    record = audition_record(adapter_path)
    if not record or not record.get("use"):
        return None
    reference = run_dir_of(adapter_path) / str(record.get("reference") or "")
    return str(reference) if reference.is_file() and reference.name else None


def _order(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _clean(row: Mapping[str, Any]) -> bool:
    try:
        wer = float(row.get("asr_wer", 0) or 0)
    except (TypeError, ValueError):
        return False
    return math.isfinite(wer) and wer == 0.0 and bool(row.get("boundary_words_match", True)) and bool(str(row.get("text") or "").strip())


def _duration(row: Mapping[str, Any]) -> float:
    try:
        return float(row.get("duration_s") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def split_rows(dataset_dir: Path, rows: Sequence[Mapping[str, Any]], run_dir: Path) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Training and held-out rows the way the voice was trained (its train_config), else the manifest's split labels."""

    from .plan import validation_record_ids

    try:
        config = json.loads((run_dir / "train_config.json").read_text(encoding="utf-8"))
        held_out = validation_record_ids(rows, float(config.get("val_fraction", 0.05)), int(config.get("seed", 42)),
                                         str(config.get("val_split_mode", "source")))
    except (OSError, ValueError, TypeError, KeyError):
        held_out = {str(row["id"]) for row in rows if str(row.get("split") or "") in {"val", "validation"}}
    training = [row for row in rows if str(row["id"]) not in held_out]
    validation = [row for row in rows if str(row["id"]) in held_out]
    return training, validation


def choose_candidates(dataset_dir: Path, training: Sequence[Mapping[str, Any]], count: int, *, seed: int = 42,
                      exclude: Sequence[str] = ()) -> list[Mapping[str, Any]]:
    """Clean training clips nearest the speaker's typical pitch and pace, alternating shorter and longer ones."""

    from .evaluation_plan import audio_path
    from .voice_profile import PitchCache, clip_pace, speaker_profile, typicality_distance

    pool = [row for row in training if _clean(row) and CANDIDATE_SECONDS[0] <= _duration(row) <= CANDIDATE_SECONDS[1]
            and str(row["id"]) not in set(exclude) and audio_path(dataset_dir, row).is_file()]
    pool.sort(key=lambda row: _order(seed, str(row["id"])))
    pool = pool[:TYPICALITY_POOL]
    if not pool:
        return []
    cache = PitchCache(dataset_dir)
    profile = speaker_profile(dataset_dir, list(training), cache=cache)
    ranked = sorted(pool, key=lambda row: (typicality_distance(cache.pitch_of(row), clip_pace(row), profile), str(row["id"])))
    cache.save()
    shorter = [row for row in ranked if _duration(row) < 10.0]
    longer = [row for row in ranked if _duration(row) >= 10.0]
    chosen: list[Mapping[str, Any]] = []
    while len(chosen) < count and (shorter or longer):
        for side in (longer, shorter):
            if side and len(chosen) < count:
                chosen.append(side.pop(0))
    return chosen


def choose_sentences(dataset_dir: Path, validation: Sequence[Mapping[str, Any]], training: Sequence[Mapping[str, Any]],
                     count: int, *, seed: int = 42, exclude: Sequence[str] = ()) -> list[Mapping[str, Any]]:
    """Held-out sentences with clean transcripts, one per recording first; training clips only without a split."""

    from .evaluation_plan import audio_path

    source_rows = list(validation) or [row for row in training if str(row["id"]) not in set(exclude)]
    pool = [row for row in source_rows if _clean(row) and SENTENCE_SECONDS[0] <= _duration(row) <= SENTENCE_SECONDS[1]
            and audio_path(dataset_dir, row).is_file()]
    pool.sort(key=lambda row: _order(seed + 1, str(row["id"])))
    chosen: list[Mapping[str, Any]] = []
    sources: set[str] = set()
    for row in pool:  # one sentence per recording first
        source = str(row.get("source_media") or row["id"])
        if source not in sources:
            chosen.append(row)
            sources.add(source)
    chosen = chosen[:count]
    for row in pool:  # then more from the same recordings when there are fewer recordings than sentences
        if len(chosen) >= count:
            break
        if row not in chosen:
            chosen.append(row)
    return chosen


def summarize(measured: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per candidate: mean speaker / style similarity, word errors and duration ratio; the winner first."""

    def mean(rows, key):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return round(statistics.fmean(values), 4) if values else None

    table = []
    for candidate in candidates:
        rows = [row for row in measured if row.get("candidate") == candidate["key"]]
        table.append({**{key: candidate[key] for key in ("key", "label", "audio", "text", "duration_s", "current")},
                      "takes": len(rows), "similarity": mean(rows, "speaker_similarity_real"),
                      "style": mean(rows, "style_similarity_real"), "word_errors": mean(rows, "error_rate"),
                      "duration_ratio": mean(rows, "duration_ratio_vs_real")})
    scored = [row for row in table if row["similarity"] is not None]
    if not scored:
        return table
    fewest = min(row["word_errors"] or 0.0 for row in scored)
    eligible = [row for row in scored if (row["word_errors"] or 0.0) <= fewest + WER_GUARD]
    winner = max(eligible, key=lambda row: (row["similarity"], row["style"] or 0.0))
    current = next((row for row in eligible if row["current"]), None)
    # Keep the voice's current reference unless another clip is clearly better.
    if current is not None and current is not winner and winner["similarity"] - current["similarity"] < 0.005:
        winner = current
    for row in table:
        row["winner"] = row is winner
    table.sort(key=lambda row: (not row["winner"], -(row["similarity"] or -1.0)))
    return table


def save_choice(adapter_path: str | Path, winner: Mapping[str, Any], report: Mapping[str, Any], *, use: bool) -> dict[str, Any]:
    """Copy the winner beside the voice (never over its original reference) and record the audition."""

    run_dir = run_dir_of(adapter_path)
    record: dict[str, Any] = {"version": 1, **dict(report), "use": bool(use), "reference": ""}
    if winner.get("current"):
        # The current reference won. When it is an earlier audition's winner, that choice stays the voice's
        # reference; an empty reference would quietly send the voice back to its original clip.
        current = Path(str(winner.get("audio") or ""))
        if (current.stem == f"{run_dir.name}_{CHOICE_STEM}" and current.is_file()
                and current.parent.resolve() == run_dir.resolve()):
            record["reference"] = current.name
    else:
        source = Path(str(winner["audio"]))
        target = run_dir / f"{run_dir.name}_{CHOICE_STEM}{source.suffix.lower() or '.wav'}"
        shutil.copy2(source, target)
        target.with_suffix(".txt").write_text(str(winner.get("text") or ""), encoding="utf-8")
        record["reference"] = target.name
    analysis = run_dir / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    temporary = analysis / f"{AUDITION_FILE}.tmp"
    temporary.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    temporary.replace(analysis / AUDITION_FILE)
    return record


def set_choice_use(adapter_path: str | Path, use: bool) -> dict[str, Any] | None:
    record = audition_record(adapter_path)
    if record is None:
        return None
    record["use"] = bool(use)
    path = run_dir_of(adapter_path) / "analysis" / AUDITION_FILE
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    return record


__all__ = ["AUDITION_FILE", "audition_record", "audition_reference", "choose_candidates", "choose_sentences",
           "run_dir_of", "save_choice", "set_choice_use", "split_rows", "summarize"]
