"""Speech-length profile of a training dataset and the text-length rules it implies.

A voice adapter reproduces the clip lengths it was trained on: a dataset packed
into 13-second clips speaks 13-second lines best, one cut into single sentences
speaks single sentences best. The profile measures the training split once
(durations, words, sentences, BPE tokens, words per second, vocabulary) and is
saved beside the adapter as ``analysis/dataset_profile.json``. The generation tab
reads it to show the optimal words and seconds per generated line for the
selected LoRA / DoRA and to choose the text-token budget whose segments land on
the dataset's typical clip length. Everything here runs on the CPU.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

from indextts.utils.atomic_json import read_json_retry, write_json_atomic
from indextts.utils.pause_tags import TextChunk, split_text_with_pauses
from indextts.utils.text_segmentation import CJK_LANGS, DEFAULT_NON_CJK_BUDGET_SCALE, normalize_language

from .dataset_manifest import duration_histogram, load_manifest


# Version 2 adds the pause statistics (``pauses``) and re-measures profiles written by version 1.
PROFILE_VERSION = 2
PROFILE_FILENAME = "dataset_profile.json"
VOCABULARY_FILENAME = "dataset_vocabulary.txt"
PERCENTILES = (5, 10, 25, 40, 50, 60, 75, 90, 95)
# Tokens the engine spends on the ``<|en|> `` language prefix before the text budget.
PREFIX_TOKENS = 2
MAX_TOKENS_MINIMUM = 20
MAX_TOKENS_MAXIMUM = 300
# Languages whose text the engine lowercases before tokenizing (see IndexTTS2._process_text_chunk).
_LOWERCASED_LANGUAGES = frozenset({"en", "zh", "zhen", "ja"})
_WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
_SENTENCE_END_RE = re.compile(r"(?<=[.!?…。！？])\s+")


def _percentile(sorted_values: Sequence[float], percent: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * (percent / 100.0)
    lower = math.floor(position)
    upper = min(len(sorted_values) - 1, lower + 1)
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _stats(values: Sequence[float], digits: int = 3) -> dict[str, float | int]:
    """Count, mean, min, max and the percentiles in PERCENTILES of a sample."""

    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return {"count": 0}
    result: dict[str, float | int] = {
        "count": len(clean),
        "mean": round(sum(clean) / len(clean), digits),
        "min": round(clean[0], digits),
        "max": round(clean[-1], digits),
    }
    for percent in PERCENTILES:
        result[f"p{percent:02d}"] = round(_percentile(clean, percent), digits)
    return result


def spoken_text(text: str) -> str:
    """Text without pause tags, as the words the voice will speak."""

    parts = (
        " ".join(chunk.text.split())
        for chunk in split_text_with_pauses(str(text or ""))
        if isinstance(chunk, TextChunk)
    )
    return " ".join(part for part in parts if part)


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(spoken_text(text)))


def sentence_word_counts(text: str) -> list[int]:
    """Words of every sentence in the text (sentence ends at . ! ? and their CJK forms)."""

    counts = [len(_WORD_RE.findall(part)) for part in _SENTENCE_END_RE.split(spoken_text(text).strip())]
    return [count for count in counts if count > 0]


def text_for_tokens(text: str, language: str) -> str:
    """The text as the engine tokenizes it: pause tags removed, lowercased for EN/ZH/JA."""

    value = spoken_text(text)
    return value.lower() if normalize_language(language) in _LOWERCASED_LANGUAGES else value


def _majority(values: Sequence[str], default: str) -> str:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "").strip()
        if key:
            counts[key] = counts.get(key, 0) + 1
    return max(counts, key=counts.get) if counts else default


def build_dataset_profile(
    dataset_dir: str | os.PathLike[str],
    *,
    split: str = "train",
    token_len: Callable[[str], int] | None = None,
    dataset_name: str | None = None,
    measure_pauses: bool = True,
) -> dict[str, Any] | None:
    """Measure the manifest rows of one split; None when the dataset has no usable rows.

    ``token_len`` counts text tokens the way the engine does (tiktoken with special
    tokens allowed); without it the token statistics are left out and the
    recommendation falls back to a 1.3 tokens-per-word estimate for English.
    ``measure_pauses`` reads every clip once (cached beside the dataset) for the
    speaker's sentence-pause and maximum-pause statistics.
    """

    root = Path(dataset_dir).expanduser()
    rows = load_manifest(root)
    if not rows:
        return None
    has_split = any(row.get("split") for row in rows)
    selected = [row for row in rows if not has_split or str(row.get("split") or "train") == split]
    if not selected:
        selected = rows
        split = "all"
    durations: list[float] = []
    words: list[int] = []
    tokens: list[int] = []
    sentences_per_clip: list[int] = []
    words_per_sentence: list[int] = []
    words_per_second: list[float] = []
    vocabulary: set[str] = set()
    aims: dict[str, int] = {}
    languages: list[str] = []
    for row in selected:
        try:
            duration = float(row.get("duration_s") or 0.0)
        except (TypeError, ValueError):
            continue
        text = str(row.get("text") or "")
        count = int(row.get("words") or 0) or word_count(text)
        if duration <= 0.0 or count <= 0:
            continue
        language = str(row.get("language") or "EN")
        languages.append(language)
        durations.append(duration)
        words.append(count)
        words_per_second.append(count / duration)
        sentence_counts = sentence_word_counts(text)
        sentences_per_clip.append(max(1, len(sentence_counts)))
        words_per_sentence.extend(sentence_counts)
        vocabulary.update(token.casefold() for token in _WORD_RE.findall(spoken_text(text)))
        aim = str(row.get("length_aim") or "").strip()
        if aim:
            aims[aim] = aims.get(aim, 0) + 1
        if token_len is not None:
            try:
                tokens.append(int(token_len(text_for_tokens(text, language))))
            except Exception:
                token_len = None
                tokens = []
    if not durations:
        return None
    language = _majority(languages, "EN")
    total_words = sum(words)
    total_duration = sum(durations)
    profile: dict[str, Any] = {
        "version": PROFILE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(root),
        "dataset_name": str(dataset_name or root.name),
        "split": split,
        "language": language,
        "clips": len(durations),
        "total_minutes": round(total_duration / 60.0, 3),
        "total_words": int(total_words),
        "words_per_second": round(total_words / total_duration, 4),
        "duration_s": _stats(durations),
        "words": _stats(words, digits=1),
        "clip_words_per_second": _stats(words_per_second),
        "sentences_per_clip": _stats(sentences_per_clip, digits=2),
        "words_per_sentence": _stats(words_per_sentence, digits=1),
        "duration_histogram": duration_histogram(durations),
        "length_aims": aims,
        "vocabulary_size": len(vocabulary),
    }
    if tokens:
        profile["text_tokens"] = _stats(tokens, digits=1)
        profile["tokens_per_word"] = round(sum(tokens) / total_words, 4)
    if measure_pauses:
        try:
            from .pause_profile import build_pause_profile

            pauses = build_pause_profile(root, selected)
        except Exception:
            pauses = None
        if pauses:
            profile["pauses"] = pauses
    profile["recommendation"] = recommend_line_rules(profile)
    profile["_vocabulary"] = sorted(vocabulary)
    return profile


def _tokens_per_word(profile: Mapping[str, Any]) -> float:
    value = profile.get("tokens_per_word")
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    if number > 0.0:
        return number
    # The tiktoken vocabulary spends about 1.3 tokens per English word including punctuation.
    return 1.3 if normalize_language(str(profile.get("language") or "EN")) not in CJK_LANGS else 1.0


def recommend_line_rules(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Word counts per generated line and per sentence that stay inside the training distribution.

    The generated line is one speech segment. The target is the middle of the
    training clips (p40 to p60 words), the acceptable range covers 80 percent of
    them (p10 to p90), the hard limits 90 percent (p05 and p95) and "never exceed"
    is the longest training clip. A single sentence should not stand alone below
    the hard minimum, and a sentence longer than the segment budget is cut in the
    middle, so the budget is also the sentence maximum.
    """

    words = profile.get("words") or {}
    sentence_words = profile.get("words_per_sentence") or {}
    if not words or int(words.get("count", 0) or 0) == 0:
        return {}

    def pick(stats: Mapping[str, Any], key: str, fallback: float) -> float:
        try:
            return float(stats.get(key, fallback))
        except (TypeError, ValueError):
            return float(fallback)

    p50 = pick(words, "p50", pick(words, "mean", 30.0))
    sentence_p50 = pick(sentence_words, "p50", max(6.0, p50 / 2.0))
    budget_words = min(pick(words, "p90", p50), max(p50, p50 + 0.5 * sentence_p50))
    text_tokens = profile.get("text_tokens") or {}
    target_tokens = int(math.ceil(pick(text_tokens, "p50", 0.0))) or int(math.ceil(p50 * _tokens_per_word(profile)))
    rules = {
        "target_words": [int(round(pick(words, "p40", p50))), int(round(pick(words, "p60", p50)))],
        "acceptable_words": [int(round(pick(words, "p10", p50))), int(round(pick(words, "p90", p50)))],
        "hard_min_words": int(round(pick(words, "p05", p50))),
        "hard_max_words": int(round(pick(words, "p95", p50))),
        "never_exceed_words": int(round(pick(words, "max", p50))),
        "sentence_min_alone_words": int(round(pick(words, "p05", p50))),
        "sentence_max_words": int(round(budget_words)),
        "budget_words": round(budget_words, 1),
        "budget_tokens": int(math.ceil(budget_words * _tokens_per_word(profile))),
        # The smart segmenter aims every segment at the median training clip, in text tokens.
        "target_tokens": max(1, target_tokens),
        "target_seconds": round(pick(profile.get("duration_s") or {}, "p50", 0.0), 2),
    }
    pause_rules = ((profile.get("pauses") or {}).get("recommendation")) or {}
    if pause_rules:
        rules["sentence_pause_ms"] = int(pause_rules.get("sentence_pause_ms", 0) or 0)
        rules["max_pause_ms"] = int(pause_rules.get("max_pause_ms", 0) or 0)
        rules["pause_source"] = str(pause_rules.get("sentence_pause_source") or "")
    return rules


def smart_target_tokens(profile: Mapping[str, Any] | None) -> int | None:
    """Text tokens the smart segmenter should aim for with this voice: the median training clip."""

    if not profile:
        return None
    recommendation = profile.get("recommendation") or recommend_line_rules(profile)
    value = int(recommendation.get("target_tokens") or 0)
    return value if value > 0 else None


def recommended_pauses(profile: Mapping[str, Any] | None) -> tuple[int, int] | None:
    """``(sentence_pause_ms, max_pause_ms)`` measured from the training recordings, or None."""

    if not profile:
        return None
    recommendation = ((profile.get("pauses") or {}).get("recommendation")) or {}
    sentence = int(recommendation.get("sentence_pause_ms", 0) or 0)
    ceiling = int(recommendation.get("max_pause_ms", 0) or 0)
    if ceiling <= 0:
        return None
    return sentence, ceiling


def budget_scale_for(language: str | None, budget_scale: float = DEFAULT_NON_CJK_BUDGET_SCALE) -> float:
    """The non-CJK token-budget scale applies to every language except the CJK ones."""

    return 1.0 if normalize_language(language) in CJK_LANGS else float(budget_scale)


def recommended_max_tokens(
    profile: Mapping[str, Any],
    *,
    language: str | None = None,
    budget_scale: float = DEFAULT_NON_CJK_BUDGET_SCALE,
    minimum: int = MAX_TOKENS_MINIMUM,
    maximum: int = MAX_TOKENS_MAXIMUM,
) -> int | None:
    """The **Max tokens per segment** whose usable budget equals the profile's line budget.

    The engine's budget is ``int((max_tokens - prefix) * scale)`` for non-CJK
    languages (``indextts.utils.text_segmentation.segment_token_budget``), so the
    inverse is taken here. Sentences are merged greedily up to that budget, which
    puts most segments between the budget minus one sentence and the budget, that
    is around the training clips' median length.
    """

    recommendation = profile.get("recommendation") or recommend_line_rules(profile)
    budget_tokens = int(recommendation.get("budget_tokens") or 0)
    if budget_tokens <= 0:
        return None
    scale = budget_scale_for(language or profile.get("language"), budget_scale)
    if not 0.0 < scale <= 1.0:
        scale = 1.0
    value = int(math.ceil(budget_tokens / scale)) + PREFIX_TOKENS
    return max(int(minimum), min(int(maximum), value))


def words_for_max_tokens(
    profile: Mapping[str, Any],
    max_tokens: int,
    *,
    language: str | None = None,
    budget_scale: float = DEFAULT_NON_CJK_BUDGET_SCALE,
) -> float:
    """Roughly how many words fit into one segment at a **Max tokens per segment** setting."""

    scale = budget_scale_for(language or profile.get("language"), budget_scale)
    budget = int(max(1, (int(max_tokens) - PREFIX_TOKENS)) * (scale if 0.0 < scale <= 1.0 else 1.0))
    return max(1.0, budget / _tokens_per_word(profile))


def seconds_for_words(words: float, words_per_second: float) -> float:
    return float(words) / words_per_second if words_per_second > 0.0 else 0.0


def _adapter_dir(adapter_or_checkpoint_path: str | os.PathLike[str]) -> Path:
    source = Path(adapter_or_checkpoint_path).expanduser().resolve()
    if source.is_dir():
        return source.parent if source.name.lower() == "best" else source
    return source.parent.parent if source.parent.name.lower() == "best" else source.parent


def profile_path(adapter_or_checkpoint_path: str | os.PathLike[str]) -> Path:
    return _adapter_dir(adapter_or_checkpoint_path) / "analysis" / PROFILE_FILENAME


def write_dataset_profile(adapter_dir: str | os.PathLike[str], profile: Mapping[str, Any]) -> Path:
    """Persist the profile (and its vocabulary as a text file) next to the adapter's other analyses."""

    analysis = Path(adapter_dir).expanduser().resolve() / "analysis"
    payload = {key: value for key, value in profile.items() if key != "_vocabulary"}
    destination = write_json_atomic(analysis / PROFILE_FILENAME, payload, indent=2, ensure_ascii=False, allow_nan=False)
    vocabulary = profile.get("_vocabulary")
    if isinstance(vocabulary, (list, tuple, set, frozenset)):
        words = "\n".join(sorted(str(word) for word in vocabulary))
        temporary = analysis / (VOCABULARY_FILENAME + ".tmp")
        temporary.write_text(words + ("\n" if words else ""), encoding="utf-8")
        os.replace(temporary, analysis / VOCABULARY_FILENAME)
    return Path(destination)


def load_dataset_profile(adapter_or_checkpoint_path: str | os.PathLike[str] | None) -> dict[str, Any] | None:
    if not adapter_or_checkpoint_path:
        return None
    value = read_json_retry(profile_path(adapter_or_checkpoint_path), None)
    if not isinstance(value, Mapping) or not value.get("duration_s"):
        return None
    profile = dict(value)
    if not profile.get("recommendation"):
        profile["recommendation"] = recommend_line_rules(profile)
    return profile


def load_dataset_vocabulary(adapter_or_checkpoint_path: str | os.PathLike[str] | None) -> frozenset[str]:
    """Every distinct word (casefolded) the adapter's training transcripts contain."""

    if not adapter_or_checkpoint_path:
        return frozenset()
    path = _adapter_dir(adapter_or_checkpoint_path) / "analysis" / VOCABULARY_FILENAME
    try:
        return frozenset(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return frozenset()


def dataset_dir_for_adapter(
    adapter_or_checkpoint_path: str | os.PathLike[str],
    *,
    datasets_root: str | os.PathLike[str] | None = None,
) -> Path | None:
    """The dataset an adapter was trained on: its train_config.json, its metadata, or datasets/<name>."""

    adapter_dir = _adapter_dir(adapter_or_checkpoint_path)
    candidates: list[Path] = []
    config = read_json_retry(adapter_dir / "train_config.json", None)
    if isinstance(config, Mapping) and config.get("dataset_dir"):
        candidates.append(Path(str(config["dataset_dir"])).expanduser())
    source = Path(adapter_or_checkpoint_path).expanduser()
    if source.is_file():
        try:
            from indextts.lora.io import inspect_lora

            info = inspect_lora(source)
        except Exception:
            info = {}
        train_config = info.get("train_config") if isinstance(info, Mapping) else None
        if isinstance(train_config, Mapping) and train_config.get("dataset_dir"):
            candidates.append(Path(str(train_config["dataset_dir"])).expanduser())
        name = str(info.get("dataset") or "") if isinstance(info, Mapping) else ""
        if name and datasets_root is not None:
            candidates.append(Path(datasets_root).expanduser() / name)
    for candidate in candidates:
        if (candidate / "manifest.jsonl").is_file():
            return candidate.resolve()
    return None


def ensure_dataset_profile(
    adapter_or_checkpoint_path: str | os.PathLike[str] | None,
    *,
    token_len: Callable[[str], int] | None = None,
    datasets_root: str | os.PathLike[str] | None = None,
    write: bool = True,
) -> dict[str, Any] | None:
    """Load the adapter's saved profile, or measure its dataset now when the dataset is still present."""

    if not adapter_or_checkpoint_path:
        return None
    existing = load_dataset_profile(adapter_or_checkpoint_path)
    if existing is not None and (int(existing.get("version") or 1) >= PROFILE_VERSION or existing.get("pauses")):
        return existing
    dataset_dir = dataset_dir_for_adapter(adapter_or_checkpoint_path, datasets_root=datasets_root)
    if dataset_dir is None:
        return existing
    try:
        profile = build_dataset_profile(dataset_dir, token_len=token_len)
    except (OSError, ValueError, TypeError):
        return existing
    if profile is None:
        return existing
    if existing is not None and existing.get("expressive_reference") and not profile.get("expressive_reference"):
        profile["expressive_reference"] = dict(existing["expressive_reference"])
    if write:
        try:
            write_dataset_profile(_adapter_dir(adapter_or_checkpoint_path), profile)
        except OSError:
            pass
    return profile


EXPRESSIVE_SUFFIX = "_expressive_reference"


def expressive_reference_path(adapter_or_checkpoint_path: str | os.PathLike[str] | None) -> str | None:
    """The expressive training clip saved beside the adapter (``<name>_expressive_reference.wav``), or None."""

    if not adapter_or_checkpoint_path:
        return None
    try:
        adapter_dir = _adapter_dir(adapter_or_checkpoint_path)
    except OSError:
        return None
    matches = sorted(path for path in adapter_dir.glob(f"*{EXPRESSIVE_SUFFIX}.*") if path.is_file())
    return str(matches[0]) if matches else None


def save_expressive_reference(
    adapter_dir: str | os.PathLike[str],
    adapter_name: str,
    dataset_dir: str | os.PathLike[str],
    choice: Mapping[str, Any],
) -> Path:
    """Copy the chosen clip next to the adapter and return the copy's path."""

    import shutil

    record = choice["record"]
    source = Path(str(record["audio"]))
    if not source.is_absolute():
        source = Path(dataset_dir) / source
    destination = Path(adapter_dir).expanduser().resolve() / f"{adapter_name}{EXPRESSIVE_SUFFIX}{source.suffix or '.wav'}"
    for stale in destination.parent.glob(f"{adapter_name}{EXPRESSIVE_SUFFIX}.*"):
        if stale != destination:
            stale.unlink(missing_ok=True)
    shutil.copy2(source, destination)
    return destination


def expressive_profile_entry(choice: Mapping[str, Any], saved: Path) -> dict[str, Any]:
    metrics = dict(choice.get("metrics") or {})
    return {
        "id": str(choice["record"].get("id", "")),
        "file": saved.name,
        "text": str(choice["record"].get("text", "")),
        "duration_s": float(choice["record"].get("duration_s") or 0.0),
        "score": float(choice.get("score", 0.0)),
        "pool_pitch_std_st": float(choice.get("pool_pitch_std_st", 0.0)),
        "pool_energy_std_db": float(choice.get("pool_energy_std_db", 0.0)),
        "candidates_measured": len(choice.get("candidates") or []),
        **metrics,
    }


def update_profile_expressive_reference(adapter_or_checkpoint_path: str | os.PathLike[str], entry: Mapping[str, Any]) -> None:
    """Record the expressive clip in the saved profile when one exists."""

    path = profile_path(adapter_or_checkpoint_path)
    value = read_json_retry(path, None)
    if not isinstance(value, Mapping):
        return
    payload = dict(value)
    payload["expressive_reference"] = dict(entry)
    write_json_atomic(path, payload, indent=2, ensure_ascii=False, allow_nan=False)


def describe_profile(profile: Mapping[str, Any]) -> str:
    """One line for logs: clips, minutes, median length and words per second."""

    duration = profile.get("duration_s") or {}
    words = profile.get("words") or {}
    return (
        f"{int(profile.get('clips', 0))} training clips, {float(profile.get('total_minutes', 0.0)):.1f} minutes; "
        f"median clip {float(duration.get('p50', 0.0)):.1f} s and {float(words.get('p50', 0.0)):.0f} words "
        f"(80 percent between {float(duration.get('p10', 0.0)):.1f} and {float(duration.get('p90', 0.0)):.1f} s); "
        f"{float(profile.get('words_per_second', 0.0)):.2f} words/s"
    )


__all__ = [
    "MAX_TOKENS_MAXIMUM",
    "MAX_TOKENS_MINIMUM",
    "PREFIX_TOKENS",
    "PROFILE_FILENAME",
    "PROFILE_VERSION",
    "VOCABULARY_FILENAME",
    "budget_scale_for",
    "build_dataset_profile",
    "dataset_dir_for_adapter",
    "describe_profile",
    "ensure_dataset_profile",
    "expressive_profile_entry",
    "expressive_reference_path",
    "load_dataset_profile",
    "load_dataset_vocabulary",
    "profile_path",
    "recommend_line_rules",
    "recommended_max_tokens",
    "recommended_pauses",
    "smart_target_tokens",
    "save_expressive_reference",
    "seconds_for_words",
    "sentence_word_counts",
    "spoken_text",
    "text_for_tokens",
    "update_profile_expressive_reference",
    "word_count",
    "words_for_max_tokens",
    "write_dataset_profile",
]
