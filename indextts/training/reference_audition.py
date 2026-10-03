"""Reference audition: the clip a trained voice clones best from, chosen by listening to the results.

A cloned voice copies its reference clip's timbre, recording conditions and delivery. In the round-3
comparison (October 2026) four clips of one speaker gave a fine-tuned OmniVoice voice speaker similarities
between 0.725 and 0.817 to the speaker's real recordings, and neither the clip length nor a typical pitch,
pace or intonation range predicted which clip was good. An audition therefore renders a few held-out
sentences with each candidate clip and measures every take against the speaker's own recording of the same
sentence (speaker and style similarity and Whisper word errors, the measurements of the training's speech
evaluation). The candidate whose takes sound most like the speaker wins.

The search has two stages (``audition_search``): every candidate renders the screening sentences, then the
current reference and the most similar finalists render more sentences with the same seeds, and the winner is
chosen on all of them. A clip that looked good on a few sentences by luck rarely survives the confirmation, and
the many screened clips cost only the short first stage. Training runs it automatically after the voice is
finished (``audition_worker``); the voice panel's button runs the same search with the page's settings.

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
from typing import Any, Callable, Generator, Mapping, Sequence

AUDITION_FILE = "reference_audition.json"
CHOICE_STEM = "audition_choice"
CANDIDATE_SECONDS = (6.0, 16.0)
SENTENCE_SECONDS = (5.0, 14.0)
TYPICALITY_POOL = 60  # clean clips measured for typical pitch and pace before candidates are picked
WER_GUARD = 0.05  # a winner may not have more than five points more word errors than the best candidate
MARGIN = 0.005  # another clip replaces the current reference only when it is at least this much more similar
FINALISTS = 3  # clips besides the current reference that the confirmation stage renders again
SCREEN_SENTENCES = 3  # sentences every candidate renders in an automatic audition (the rest confirm)
CONFIRM_SENTENCES = 4  # extra sentences for the finalists of an audition started from the voice panel
FIRST_SEED = 1000


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
    if current is not None and current is not winner and winner["similarity"] - current["similarity"] < MARGIN:
        winner = current
    for row in table:
        row["winner"] = row is winner
    table.sort(key=lambda row: (not row["winner"], -(row["similarity"] or -1.0)))
    return table


def pick_finalists(summary: Sequence[Mapping[str, Any]], count: int = FINALISTS) -> list[str]:
    """The current reference and the ``count`` most similar other clips within the word-error guard."""

    scored = [row for row in summary if row.get("similarity") is not None]
    if not scored:
        return []
    fewest = min(row.get("word_errors") or 0.0 for row in scored)
    eligible = [row for row in scored if (row.get("word_errors") or 0.0) <= fewest + WER_GUARD and not row.get("current")]
    eligible.sort(key=lambda row: (-row["similarity"], -(row.get("style") or 0.0), str(row["key"])))
    keys = [str(row["key"]) for row in eligible[:max(1, int(count))]]
    return [str(row["key"]) for row in summary if row.get("current")] + keys


def sentence_language(sentence: Mapping[str, Any], fallback: str = "EN") -> str:
    """A held-out sentence's language for word errors: its dataset language, else ``fallback``, else English."""

    from .speech_metrics import LANGUAGES

    for value in (sentence.get("language"), fallback):
        code = str(value or "").strip().upper()
        if code in LANGUAGES:
            return code
    return "EN"


def take_clips(takes: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]], dataset_dir: str | Path,
               language: str = "EN") -> list[dict[str, Any]]:
    """Rendered takes as ``measure_clips`` input: each against the speaker's own recording of its sentence."""

    from .evaluation_plan import audio_path

    by_key = {str(candidate["key"]): candidate for candidate in candidates}
    return [{"audio": str(take["audio"]), "reference": str(by_key[str(take["candidate"])]["audio"]),
             "real_audio": str(audio_path(dataset_dir, take["sentence"])), "text": str(take["sentence"]["text"]),
             "language": sentence_language(take["sentence"], language), "candidate": str(take["candidate"]),
             "sentence_id": str(take["sentence"]["id"]), "stage": str(take.get("stage") or "")}
            for take in takes]


RenderStage = Callable[[str, Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]], int],
                       Generator[str, None, list[dict[str, Any]]]]


def audition_search(candidates: Sequence[Mapping[str, Any]], screen: Sequence[Mapping[str, Any]],
                    confirm: Sequence[Mapping[str, Any]], *, render_stage: RenderStage,
                    measure: Callable[[list[dict[str, Any]]], list[dict[str, Any]]], dataset_dir: str | Path,
                    language: str = "EN", finalists: int = FINALISTS) -> Generator[str, None, dict[str, Any]]:
    """Screen every candidate, confirm the finalists on more sentences, and return the result.

    ``render_stage(stage, candidates, sentences, first_seed)`` renders every candidate on every sentence (the
    sentence at position i with seed ``first_seed + i``, the same for every clip), yields progress messages and
    returns the takes (``candidate``, ``sentence``, ``audio``). ``measure`` is ``measure_clips``. Progress
    messages are yielded; the return value holds the summary table (winner first, then the other finalists,
    then the screened-out clips), the winner and every measured take.
    """

    takes = yield from render_stage("screen", candidates, screen, FIRST_SEED)
    yield f"Measuring {len(takes)} screening takes against the speaker's recordings..."
    measured = measure(take_clips([dict(take, stage="screen") for take in takes], candidates, dataset_dir, language))
    screened = summarize(measured, candidates)
    keys = pick_finalists(screened, finalists)
    if not confirm or len(keys) < 2:
        table = [dict(row, stage="final") for row in screened]
        return {"summary": table, "winner": next((row for row in table if row.get("winner")), None),
                "measured": measured, "finalists": keys}
    chosen = [candidate for candidate in candidates if str(candidate["key"]) in keys]
    yield (f"Confirming {len(chosen)} finalists ({', '.join(str(row['label']) for row in chosen)}) on "
           f"{len(confirm)} more sentences...")
    more = yield from render_stage("confirm", chosen, confirm, FIRST_SEED + len(screen))
    yield f"Measuring {len(more)} confirmation takes..."
    confirmed = measure(take_clips([dict(take, stage="confirm") for take in more], chosen, dataset_dir, language))
    combined = [row for row in measured if str(row.get("candidate")) in keys] + confirmed
    final = summarize(combined, chosen)
    table = [dict(row, stage="final") for row in final] + [
        dict(row, stage="screened out", winner=False) for row in screened if str(row["key"]) not in keys]
    return {"summary": table, "winner": next((row for row in table if row.get("winner")), None),
            "measured": measured + confirmed, "finalists": keys}


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


__all__ = ["AUDITION_FILE", "audition_record", "audition_reference", "audition_search", "choose_candidates",
           "choose_sentences", "pick_finalists", "run_dir_of", "save_choice", "sentence_language", "set_choice_use",
           "split_rows", "summarize", "take_clips"]
