"""Training-data fluency filters: keep the clips whose pauses fit their transcript.

A trained voice copies its speaker's delivery, including hesitations: a voice
trained on screen-demonstration speech also pauses mid-sentence. These filters
measure every training clip locally and train on the fluent ones only. Nothing
here uses a remote service, a speech recognizer or a model.

How a clip is measured
    Its internal pauses are the runs of at least 120 ms below -40 dBFS between
    the first and last loud frame (the same measurement as the dataset pause
    profile, cached in ``analysis/pause_cache.json``). Its transcript allows a
    pause at every sentence end after the first sentence and at every comma,
    semicolon, colon or dash. The longest pauses fill those places; every pause
    left over is a *hesitation*, and a hesitation of at least the long-hesitation
    length (250 ms by default) is a *long hesitation*. The pause share is the
    total pause time divided by the speaking time between the leading and
    trailing silence. Filler words and phrases are counted in the transcript.

How training uses a filter
    The filter keeps or drops training clips only; validation clips are never
    filtered, so every filter is validated on the same held-out recordings and
    the results stay comparable. The kept set is written as a sibling dataset
    ``datasets/<name>__fluency_<filter>`` whose audio and cached features are hard
    links to the original files (copies when the drive cannot link), so deleting
    it never touches the original dataset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from .dataset_manifest import atomic_write_json, load_manifest
from .pause_profile import PauseCache
from .plan import validation_record_ids

FLUENCY_VERSION = 1
NO_LIMIT = -1
VIEW_MARKER = "fluency_view.json"
VIEW_INFIX = "__fluency_"
REPORT_FILENAME = "fluency_report.json"
DEFAULT_LONG_HESITATION_MS = 250
DEFAULT_FILLER_WORDS = (
    "okay, ok, so, like, you know, you see, um, umm, uh, uhh, actually, basically, i mean, let's see, anyway"
)

_SENTENCE_END = re.compile(r"[.!?…](?:\s|$)")
_PAUSE_MARK = re.compile(r"[,;:–—]|\s-\s")


@dataclass(frozen=True)
class FluencyLimits:
    """Per-clip limits; -1 (counts) or 100 (percent) means no limit."""

    long_hesitation_ms: int = DEFAULT_LONG_HESITATION_MS
    max_long_hesitations: int = NO_LIMIT
    max_hesitations: int = NO_LIMIT
    max_pause_percent: float = 100.0
    max_fillers: int = NO_LIMIT
    filler_words: str = DEFAULT_FILLER_WORDS

    def normalized(self) -> "FluencyLimits":
        long_ms = int(round(float(self.long_hesitation_ms)))
        if not 120 <= long_ms <= 5000:
            raise ValueError("the long-hesitation length must be between 120 and 5000 ms")
        counts = {}
        for name in ("max_long_hesitations", "max_hesitations", "max_fillers"):
            value = int(round(float(getattr(self, name))))
            if value < NO_LIMIT:
                raise ValueError(f"{name.replace('_', ' ')} must be -1 (no limit) or 0 and above")
            counts[name] = value
        percent = float(self.max_pause_percent)
        if not 0.0 <= percent <= 100.0:
            raise ValueError("the maximum pause share must be between 0 and 100 percent")
        words = ", ".join(filler_items(self.filler_words))
        return FluencyLimits(long_ms, counts["max_long_hesitations"], counts["max_hesitations"],
                             round(percent, 3), counts["max_fillers"], words)


@dataclass(frozen=True)
class FluencyPreset:
    key: str
    label: str
    description: str
    limits: FluencyLimits | None  # None: no fluency filter


FLUENCY_PRESETS: tuple[FluencyPreset, ...] = (
    FluencyPreset(
        "all",
        "All curated clips (current system)",
        "No fluency filter. Every clip that passed dataset preparation trains the voice, as in every earlier "
        "release; hesitant clips teach the voice to hesitate too.",
        None,
    ),
    FluencyPreset(
        "no_long_hesitation",
        "No long hesitations",
        "Drops a clip when it pauses for the long-hesitation length or longer anywhere its transcript has no "
        "sentence end, comma, semicolon, colon or dash. Short hesitations and filler words stay.",
        FluencyLimits(max_long_hesitations=0),
    ),
    FluencyPreset(
        "fluent",
        "Fluent: at most one short hesitation",
        "No long hesitation, at most one shorter hesitation, and silence takes at most 12% of the speaking time.",
        FluencyLimits(max_long_hesitations=0, max_hesitations=1, max_pause_percent=12.0),
    ),
    FluencyPreset(
        "strict",
        "Strictly fluent: no hesitations or filler words",
        "Every pause sits at a sentence end or punctuation mark, and the transcript contains none of the filler "
        "words. Keeps the least data; check the analysis before training on it.",
        FluencyLimits(max_hesitations=0, max_fillers=0),
    ),
)
PRESETS_BY_KEY = {preset.key: preset for preset in FLUENCY_PRESETS}
FLUENCY_FILTER_KEYS = tuple(PRESETS_BY_KEY)


def preset(key: str) -> FluencyPreset:
    try:
        return PRESETS_BY_KEY[str(key or "all").strip().lower()]
    except KeyError as exc:
        raise ValueError(f"unknown fluency filter {key!r}; expected one of {', '.join(FLUENCY_FILTER_KEYS)}") from exc


def filler_items(words: str | Iterable[str] | None) -> list[str]:
    raw = words if isinstance(words, (list, tuple)) else re.split(r"[,\n]", str(words or ""))
    seen: dict[str, None] = {}
    for item in raw:
        text = " ".join(str(item).strip().lower().split())
        if text:
            seen.setdefault(text, None)
    return list(seen)


def filler_pattern(words: str | Iterable[str] | None) -> re.Pattern[str] | None:
    """Whole-word, case-insensitive matcher; longer phrases win, so "okay" is not also counted as "ok"."""
    items = sorted(filler_items(words), key=len, reverse=True)
    if not items:
        return None
    alternation = "|".join(re.escape(item).replace(r"\ ", r"\s+") for item in items)
    return re.compile(rf"(?<![\w'])(?:{alternation})(?![\w'])", re.IGNORECASE)


def clip_measurement(record: Mapping[str, Any], pauses: Mapping[str, Any]) -> dict[str, Any]:
    """Filter-independent facts of one clip: the pauses left after the transcript's pause places are filled."""
    text = str(record.get("text") or "")
    ordered = sorted((int(value) for value in pauses.get("pauses_ms") or []), reverse=True)
    sentences = max(1, len(_SENTENCE_END.findall(text.strip())))
    places = max(0, sentences - 1) + len(_PAUSE_MARK.findall(text))
    duration_s = float(record.get("duration_s") or 0.0) or float(pauses.get("duration_ms") or 0) / 1000.0
    edges_s = (float(record.get("leading_silence_ms") or 0.0) + float(record.get("trailing_silence_ms") or 0.0)) / 1000.0
    speaking_s = max(0.5, duration_s - edges_s)
    return {
        "duration_s": duration_s,
        "pause_places": places,
        "hesitations_ms": ordered[places:],
        "pause_percent": 100.0 * float(pauses.get("pause_ms") or sum(ordered)) / 1000.0 / speaking_s,
        "text": text,
    }


def clip_verdict(measurement: Mapping[str, Any], limits: FluencyLimits,
                 fillers: re.Pattern[str] | None = None) -> dict[str, Any]:
    """Counts for one clip under ``limits`` and whether it passes them."""
    hesitations = list(measurement["hesitations_ms"])
    long_count = sum(1 for value in hesitations if value >= limits.long_hesitation_ms)
    matcher = fillers if fillers is not None else filler_pattern(limits.filler_words)
    filler_count = len(matcher.findall(measurement["text"])) if matcher is not None else 0
    reasons = []
    if limits.max_long_hesitations >= 0 and long_count > limits.max_long_hesitations:
        reasons.append("long hesitation")
    if limits.max_hesitations >= 0 and len(hesitations) > limits.max_hesitations:
        reasons.append("hesitations")
    if limits.max_pause_percent < 100.0 and measurement["pause_percent"] > limits.max_pause_percent:
        reasons.append("pause share")
    if limits.max_fillers >= 0 and filler_count > limits.max_fillers:
        reasons.append("filler words")
    return {"hesitations": len(hesitations), "long_hesitations": long_count, "fillers": filler_count,
            "pause_percent": round(measurement["pause_percent"], 2), "passes": not reasons, "reasons": reasons}


def limits_from_values(values: Mapping[str, Any], prefix: str = "fluency_") -> FluencyLimits:
    """Limits from flat config or UI values (``fluency_long_hesitation_ms`` and so on)."""
    return FluencyLimits(
        long_hesitation_ms=values.get(prefix + "long_hesitation_ms", DEFAULT_LONG_HESITATION_MS),
        max_long_hesitations=values.get(prefix + "max_long_hesitations", NO_LIMIT),
        max_hesitations=values.get(prefix + "max_hesitations", NO_LIMIT),
        max_pause_percent=values.get(prefix + "max_pause_percent", 100.0),
        max_fillers=values.get(prefix + "max_fillers", NO_LIMIT),
        filler_words=values.get(prefix + "filler_words", DEFAULT_FILLER_WORDS),
    ).normalized()


def split_rows(rows: Sequence[Mapping[str, Any]], *, val_fraction: float, seed: int,
               val_split_mode: str) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """The training and validation rows exactly as training splits them (explicit labels win)."""
    usable = [row for row in rows if row.get("id")]
    val_ids = validation_record_ids(usable, val_fraction, seed, val_split_mode)
    train = [row for row in usable if str(row["id"]) not in val_ids]
    val = [row for row in usable if str(row["id"]) in val_ids]
    return train, val


def measure_rows(dataset_dir: str | Path, rows: Sequence[Mapping[str, Any]],
                 progress: Callable[[int, int], Any] | None = None) -> dict[str, dict[str, Any] | None]:
    """Pause measurements for ``rows`` from the dataset's pause cache, measuring clips it does not hold yet."""
    cache = PauseCache(dataset_dir)
    result: dict[str, dict[str, Any] | None] = {}
    total = len(rows)
    for index, row in enumerate(rows, 1):
        result[str(row["id"])] = cache.metrics_of(row)
        if progress is not None and (index == total or index % 250 == 0):
            progress(index, total)
    cache.save()
    return result


def _hours(seconds: float) -> float:
    return round(seconds / 3600.0, 2)


def analyze_dataset(
    dataset_dir: str | Path,
    *,
    val_fraction: float = 0.05,
    seed: int = 42,
    val_split_mode: str = "source",
    custom: FluencyLimits | None = None,
    progress: Callable[[int, int], Any] | None = None,
) -> dict[str, Any]:
    """Training time kept by every preset (and by ``custom`` limits), measured on the training split."""
    root = Path(dataset_dir).expanduser().resolve()
    rows = load_manifest(root)
    if not rows:
        raise FileNotFoundError(f"manifest.jsonl is empty or missing in {root}")
    train, val = split_rows(rows, val_fraction=val_fraction, seed=seed, val_split_mode=val_split_mode)
    pauses = measure_rows(root, train, progress)
    measured: dict[str, dict[str, Any]] = {}
    for row in train:
        metrics = pauses.get(str(row["id"]))
        if metrics:
            measured[str(row["id"])] = clip_measurement(row, metrics)
    seconds = {str(row["id"]): float(row.get("duration_s") or 0.0) for row in rows}
    total_s = sum(seconds[str(row["id"])] for row in train)

    def summarize(key: str, label: str, limits: FluencyLimits | None, *, is_custom: bool = False) -> dict[str, Any]:
        if limits is None:
            kept = [str(row["id"]) for row in train]
            reasons: dict[str, int] = {}
        else:
            matcher = filler_pattern(limits.filler_words)
            kept, reasons = [], {}
            for row_id, measurement in measured.items():
                verdict = clip_verdict(measurement, limits, matcher)
                if verdict["passes"]:
                    kept.append(row_id)
                for reason in verdict["reasons"]:
                    reasons[reason] = reasons.get(reason, 0) + 1
        kept_s = sum(seconds[row_id] for row_id in kept)
        return {"key": key, "label": label, "custom": is_custom, "clips": len(kept), "seconds": round(kept_s, 3),
                "hours": _hours(kept_s), "percent_of_time": round(100.0 * kept_s / total_s, 1) if total_s else 0.0,
                "removed_clips": len(train) - len(kept), "removed_by": reasons,
                "limits": asdict(limits) if limits is not None else None}

    report = {
        "version": FLUENCY_VERSION,
        "dataset": str(root),
        "generated_at": time.time(),
        "split": {"val_fraction": val_fraction, "seed": seed, "val_split_mode": val_split_mode},
        "training": {"clips": len(train), "seconds": round(total_s, 3), "hours": _hours(total_s)},
        "validation": {"clips": len(val), "hours": _hours(sum(seconds[str(row["id"])] for row in val))},
        "unmeasured_clips": len(train) - len(measured),
        "filters": [summarize(item.key, item.label, item.limits) for item in FLUENCY_PRESETS],
    }
    if custom is not None:
        report["filters"].append(summarize("custom", "Your current settings", custom.normalized(), is_custom=True))
    try:
        atomic_write_json(root / "analysis" / REPORT_FILENAME, report)
    except OSError:
        pass
    return report


def analysis_markdown(report: Mapping[str, Any]) -> str:
    train = report["training"]
    lines = [
        f"**Training split:** {train['clips']} clips, {train['hours']:.2f} h. "
        f"**Validation:** {report['validation']['clips']} clips ({report['validation']['hours']:.2f} h), never filtered.",
        "",
        "| Fluency filter | Training clips kept | Training time kept | Clips removed |",
        "|---|---:|---:|---:|",
    ]
    for item in report["filters"]:
        lines.append(f"| {item['label']} | {item['clips']} | {item['hours']:.2f} h ({item['percent_of_time']:.1f}%) | "
                     f"{item['removed_clips']} |")
    if report.get("unmeasured_clips"):
        lines += ["", f"{report['unmeasured_clips']} clip(s) could not be read and are removed by every filter."]
    lines += ["", "Measured locally from each clip's pauses and transcript; nothing is uploaded."]
    return "\n".join(lines)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def is_fluency_view(path: str | Path) -> bool:
    return (Path(path) / VIEW_MARKER).is_file()


def view_path(dataset_dir: str | Path, filter_key: str, limits: FluencyLimits) -> Path:
    """``<dataset>__fluency_<preset>``; edited limits get their own ``custom_<hash>`` folder."""
    root = Path(dataset_dir).expanduser().resolve()
    chosen = preset(filter_key)
    normalized = limits.normalized()
    if chosen.limits is not None and normalized == chosen.limits.normalized():
        suffix = chosen.key
    else:
        suffix = "custom_" + _fingerprint(asdict(normalized))[:8]
    return root.parent / f"{root.name}{VIEW_INFIX}{suffix}"


def _link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "link"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def build_fluency_view(
    dataset_dir: str | Path,
    filter_key: str,
    limits: FluencyLimits,
    *,
    val_fraction: float = 0.05,
    seed: int = 42,
    val_split_mode: str = "source",
    log: Callable[[str], Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Write (or reuse) the filtered sibling dataset for training; returns its path and a summary."""
    say = log or (lambda message: None)
    source = Path(dataset_dir).expanduser().resolve()
    if is_fluency_view(source):
        raise ValueError(f"{source.name} is already a fluency-filtered view; select its original dataset instead")
    chosen = preset(filter_key)
    if chosen.limits is None:
        raise ValueError("the 'all' filter trains on the dataset itself and needs no filtered view")
    limits = limits.normalized()
    manifest_path = source / "manifest.jsonl"
    rows = load_manifest(source)
    if not rows:
        raise FileNotFoundError(f"manifest.jsonl is empty or missing in {source}")
    fingerprint = _fingerprint({
        "version": FLUENCY_VERSION, "filter": chosen.key, "limits": asdict(limits),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "split": [val_fraction, seed, val_split_mode],
    })
    target = view_path(source, chosen.key, limits)
    marker = target / VIEW_MARKER
    if target.exists():
        if not marker.is_file():
            raise FileExistsError(f"{target} exists and is not a fluency view; it is never overwritten")
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous.get("fingerprint") == fingerprint and (target / "manifest.jsonl").is_file():
            say(f">> fluency filter: reusing {target.name} ({previous['summary']['kept_clips']} of "
                f"{previous['summary']['training_clips']} training clips)")
            return target, previous["summary"]

    train, val = split_rows(rows, val_fraction=val_fraction, seed=seed, val_split_mode=val_split_mode)
    say(f">> fluency filter '{chosen.label}': measuring {len(train)} training clips")
    pauses = measure_rows(source, train, lambda done, total: say(f">> fluency filter: measured {done}/{total}"))
    matcher = filler_pattern(limits.filler_words)
    kept_ids: set[str] = set()
    for row in train:
        metrics = pauses.get(str(row["id"]))
        if metrics and clip_verdict(clip_measurement(row, metrics), limits, matcher)["passes"]:
            kept_ids.add(str(row["id"]))
    if not kept_ids:
        raise ValueError(f"the '{chosen.label}' filter keeps no training clip; relax its limits")
    val_ids = {str(row["id"]) for row in val}
    keep = [row for row in rows if str(row.get("id")) in kept_ids or str(row.get("id")) in val_ids]

    staging = target.with_name(target.name + ".building")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    methods: dict[str, int] = {}
    view_rows = []
    for row in keep:
        row_id = str(row["id"])
        item = dict(row)
        item["split"] = "val" if row_id in val_ids else "train"
        audio = Path(str(row.get("audio") or ""))
        if str(row.get("audio") or "") and not audio.is_absolute():
            method = _link_or_copy(source / audio, staging / audio)
            methods[method] = methods.get(method, 0) + 1
        feature = source / "cache" / f"{row_id}.pt"
        if feature.is_file():
            method = _link_or_copy(feature, staging / "cache" / feature.name)
            methods[method] = methods.get(method, 0) + 1
        view_rows.append(item)
    with (staging / "manifest.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for item in view_rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    wanted = {str(row["id"]) for row in keep}
    index = source / "cache" / "index.jsonl"
    if index.is_file():
        lines = [line for line in index.read_text(encoding="utf-8").splitlines()
                 if line.strip() and str(json.loads(line).get("id")) in wanted]
        (staging / "cache" / "index.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    cache_index = source / "cache_index.json"
    if cache_index.is_file():
        value = json.loads(cache_index.read_text(encoding="utf-8-sig"))
        if isinstance(value, dict):
            if isinstance(value.get("records"), list):
                value["records"] = [item for item in value["records"] if str(item.get("id")) in wanted]
            if isinstance(value.get("ids"), list):
                value["ids"] = [item for item in value["ids"] if str(item) in wanted]
        atomic_write_json(staging / "cache_index.json", value)
    pause_cache = source / "analysis" / "pause_cache.json"
    if pause_cache.is_file():
        entries = json.loads(pause_cache.read_text(encoding="utf-8"))
        atomic_write_json(staging / "analysis" / "pause_cache.json",
                          {key: value for key, value in entries.items() if key in wanted})
    kept_s = sum(float(row.get("duration_s") or 0.0) for row in train if str(row["id"]) in kept_ids)
    total_s = sum(float(row.get("duration_s") or 0.0) for row in train)
    summary = {
        "filter": chosen.key, "label": chosen.label, "limits": asdict(limits),
        "training_clips": len(train), "kept_clips": len(kept_ids), "validation_clips": len(val_ids),
        "training_hours": _hours(total_s), "kept_hours": _hours(kept_s),
        "kept_percent_of_time": round(100.0 * kept_s / total_s, 1) if total_s else 0.0,
        "files": methods,
    }
    info_path = source / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8-sig")) if info_path.is_file() else {}
    view_s = sum(float(row.get("duration_s") or 0.0) for row in keep)
    info.update({
        "name": target.name, "segment_count": len(keep), "total_duration_s": round(view_s, 6),
        "total_duration_minutes": round(view_s / 60.0, 6), "derived_from": str(source),
        "fluency_filter": summary,
    })
    atomic_write_json(staging / "dataset_info.json", info)
    atomic_write_json(staging / VIEW_MARKER, {
        "version": FLUENCY_VERSION, "fingerprint": fingerprint, "source": str(source), "summary": summary,
        "created_at": time.time(),
    })
    if target.exists():
        shutil.rmtree(target)  # a stale view of this filter: only links, copies and small files
    staging.rename(target)
    say(f">> fluency filter '{chosen.label}': kept {summary['kept_clips']} of {summary['training_clips']} training "
        f"clips ({summary['kept_hours']:.2f} of {summary['training_hours']:.2f} h); all {len(val_ids)} validation "
        f"clips unchanged; dataset {target.name}")
    return target, summary


__all__ = [
    "DEFAULT_FILLER_WORDS", "DEFAULT_LONG_HESITATION_MS", "FLUENCY_FILTER_KEYS", "FLUENCY_PRESETS", "FluencyLimits",
    "FluencyPreset", "NO_LIMIT", "VIEW_MARKER", "analysis_markdown", "analyze_dataset", "build_fluency_view",
    "clip_measurement", "clip_verdict", "filler_items", "filler_pattern", "is_fluency_view", "limits_from_values",
    "measure_rows", "preset", "split_rows", "view_path",
]
