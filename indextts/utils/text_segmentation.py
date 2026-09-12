"""Shared token-aware text segmentation for inference and previews.

Three modes split a text into speech segments, all under the same hard token
budget (``segment_token_budget``):

``budget``
    The original splitter: cut at any punctuation and pack greedily up to the
    budget. Fast, but it cuts sentences at commas and can leave a short tail.
``sentence``
    One sentence per segment. A sentence longer than the budget is cut at its
    clauses, then words, then characters (inside an oversized word only).
``smart``
    Whole sentences packed by dynamic programming so that every segment lands
    near a target length (the trained voice's typical clip, or 85 percent of
    the budget for the base model), never cuts inside a sentence unless the
    sentence itself exceeds the budget, and leaves no orphaned short tail.

Every mode preserves every character of the input: the segments concatenate
back to the original text, and pronunciation annotations are never split.
Sentence modes measure formatting whitespace as a single spoken space, matching
inference; subtitle line/cue breaks do not create sentence or clause boundaries.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass


CJK_LANGS = frozenset({"zh", "zhen", "ja", "ko", "yue"})
DEFAULT_NON_CJK_BUDGET_SCALE = 0.72
SEGMENTATION_MODES = ("budget", "sentence", "smart")
DEFAULT_SEGMENTATION_MODE = "smart"
# Without a dataset target the smart packer aims at this share of the budget, leaving room for
# one more sentence to finish before the hard limit.
SMART_TARGET_FRACTION = 0.85
_PROTECTED_PATTERN = re.compile(r"<\|SPECIAL_TOKEN_\d+\|>.*?<\|SPECIAL_TOKEN_\d+\|>")
_PUNCTUATION_SPLIT = re.compile(r"(?<=[\u3001\u3002\uff01\uff0c\uff1a\uff1b\uff1f,.!?;:\n])")
_CLAUSE_SPLIT = re.compile(r"(?<=[\u3001\u3002\uff01\uff0c\uff1a\uff1b\uff1f,.!?;:])")
_LANG_PREFIX = re.compile(r"<\|([^|]+)\|>")
_WORD_PIECES = re.compile(r"\s+|\S+")
_CJK_CHARACTER = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7a3]")
# A sentence ends at . ! ? or an ellipsis (plus closing quotes or brackets) followed by whitespace, at a
# CJK full stop, exclamation or question mark. Line/cue breaks alone are formatting.
_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?\u2026])[\"\u201d\u2019')\]]*\s+|(?<=[\u3002\uff01\uff1f])[\"\u201d\u2019')\]\u3011\u300d\u300f\uff09]*\s*"
)
_SENTENCE_END_CHARS = ".!?\u2026\u3002\uff01\uff1f"
_CLOSERS = "\"\u201d\u2019')]\u3011\u300d\u300f\uff09"
# "e.g. the model" or "Dr. Smith" are not sentence ends; single letters cover initials and "U.S.".
_ABBREVIATION_RE = re.compile(
    r"(?:\b(?:e\.g|i\.e|etc|vs|cf|mr|mrs|ms|dr|prof|sr|jr|st|no|fig|figs|approx|inc|ltd|co|dept|vol|ed|est)|(?<![A-Za-z])[A-Za-z])\.$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SpeechRecoveryConfig:
    """Shared UI/CLI defaults; explicit request values are never capped or replaced."""

    enabled: bool = True
    max_attempts: int = 14
    max_split_depth: int = 3

    def __post_init__(self):
        for field in ("max_attempts", "max_split_depth"):
            value = getattr(self, field)
            try:
                integer = int(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Speech recovery {field} must be a non-negative integer") from exc
            if integer < 0 or integer != value:
                raise ValueError(f"Speech recovery {field} must be a non-negative integer")
            object.__setattr__(self, field, integer)


def normalize_language(lang: str | None) -> str:
    value = str(lang or "").strip().lower()
    match = _LANG_PREFIX.match(value)
    return match.group(1).lower() if match else value


def default_segment_tokens(lang: str | None) -> int:
    """Return the UI-friendly default segment-token limit for a language."""

    language = normalize_language(lang)
    if language in {"en", "es"}:
        return 60
    if language == "ar":
        return 80
    if language == "ja":
        return 100
    if language in {"zh", "zhen", "yue"}:
        return 120
    return 120


def split_atomic_pieces(text: str) -> list[tuple[str, bool]]:
    """Split around pronunciation annotations without breaking an annotation."""

    pieces: list[tuple[str, bool]] = []
    position = 0
    for match in _PROTECTED_PATTERN.finditer(text):
        if match.start() > position:
            pieces.append((text[position:match.start()], False))
        pieces.append((match.group(0), True))
        position = match.end()
    if position < len(text):
        pieces.append((text[position:], False))
    return pieces


def segment_token_budget(
    max_tokens: int,
    capacity: int,
    lang_prefix: str,
    token_len: Callable[[str], int],
    segment_budget_scale_non_cjk: float = DEFAULT_NON_CJK_BUDGET_SCALE,
) -> int:
    """Calculate the usable text-token budget after prefix and model limits."""

    budget = min(int(max_tokens), int(capacity) - 2) - int(token_len(lang_prefix))
    language = normalize_language(lang_prefix)
    if language and language not in CJK_LANGS:
        scale = float(segment_budget_scale_non_cjk)
        if not 0.0 < scale <= 1.0:
            raise ValueError("segment_budget_scale_non_cjk must be in the range (0, 1]")
        budget = int(budget * scale)
    return max(1, budget)


def normalize_segmentation_mode(value: object) -> str:
    """``budget``, ``sentence`` or ``smart``; anything unknown falls back to the default mode."""

    mode = str(value or "").strip().lower()
    aliases = {
        "tokens": "budget", "token": "budget", "token budget": "budget", "greedy": "budget",
        "sentences": "sentence", "every sentence": "sentence", "one sentence": "sentence", "per sentence": "sentence",
        "intelligent": "smart", "smart sentences": "smart", "smart sentence": "smart",
    }
    mode = aliases.get(mode, mode)
    return mode if mode in SEGMENTATION_MODES else DEFAULT_SEGMENTATION_MODE


def _protected_spans(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in _PROTECTED_PATTERN.finditer(text)]


def _inside(position: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < position < end for start, end in spans)


def ends_sentence(piece: str) -> bool:
    """True when the piece ends with sentence-final punctuation (closing quotes and spaces allowed)."""

    stripped = str(piece).rstrip().rstrip(_CLOSERS)
    return bool(stripped) and stripped[-1] in _SENTENCE_END_CHARS


def normalize_sentence_whitespace(text: str) -> str:
    """One spoken space per whitespace run, retaining leading/trailing boundaries."""

    return re.sub(r"\s+", " ", str(text))


def split_sentences(text: str) -> list[str]:
    """Sentence pieces of the text; each keeps its trailing whitespace so they concatenate to the input.

    Boundaries inside pronunciation annotations and after common abbreviations
    or single-letter initials are not used. A newline (even a blank subtitle cue
    separator) is whitespace, not evidence that the sentence has ended.
    """

    source = str(text)
    if not source:
        return []
    spans = _protected_spans(source)
    pieces: list[str] = []
    position = 0
    for match in _SENTENCE_BOUNDARY.finditer(source):
        if match.end() >= len(source):
            break  # the final piece takes the rest, including trailing whitespace
        if _inside(match.start(), spans) or _inside(match.end(), spans):
            continue
        head = source[position:match.start()]
        if not head.strip():
            continue
        if head.endswith(".") and _ABBREVIATION_RE.search(head.rstrip(_CLOSERS)):
            continue
        pieces.append(source[position:match.end()])
        position = match.end()
    if position < len(source):
        pieces.append(source[position:])
    return pieces or [source]


def _budget_chunks(text: str, budget: int, token_len: Callable[[str], int]) -> list[tuple[str, str]]:
    """Cut a piece that exceeds the budget into ``(chunk, boundary)`` items, each within the budget.

    ``boundary`` names the kind of break at the chunk's end: ``clause`` (punctuation),
    ``word`` or ``char`` (inside an oversized word only). Annotations stay whole.
    """

    chunks: list[tuple[str, str]] = []
    for piece, atomic in split_atomic_pieces(text):
        if atomic:
            chunks.append((piece, "word"))
            continue
        for part in _CLAUSE_SPLIT.split(piece):
            if not part:
                continue
            if token_len(part) <= budget:
                chunks.append((part, "clause"))
                continue
            for word in _WORD_PIECES.findall(part):
                if token_len(word) <= budget:
                    chunks.append((word, "word"))
                    continue
                current = ""
                for character in word:
                    if current and token_len(current + character) > budget:
                        chunks.append((current, "char"))
                        current = character
                    else:
                        current += character
                if current:
                    chunks.append((current, "word"))
    # Merge consecutive chunks greedily so a clause split does not produce a dozen tiny pieces.
    merged: list[tuple[str, str]] = []
    current_text = ""
    current_kind = "clause"
    for chunk, kind in chunks:
        if current_text and token_len(current_text + chunk) > budget:
            merged.append((current_text, current_kind))
            current_text, current_kind = chunk, kind
        else:
            current_text += chunk
            current_kind = kind
    if current_text:
        merged.append((current_text, current_kind))
    return merged


def sentence_pieces(text: str, budget: int, token_len: Callable[[str], int]) -> list[tuple[str, str]]:
    """``(piece, boundary)`` items: whole sentences, or the budget-sized parts of an oversized sentence."""

    pieces: list[tuple[str, str]] = []
    for sentence in split_sentences(text):
        if token_len(sentence) <= budget:
            pieces.append((sentence, "sentence" if ends_sentence(sentence) else "clause"))
            continue
        parts = _budget_chunks(sentence, budget, token_len)
        for index, (chunk, kind) in enumerate(parts):
            last = index == len(parts) - 1
            pieces.append((chunk, ("sentence" if ends_sentence(sentence) else "clause") if last else kind))
    return pieces


_BOUNDARY_COST = {"sentence": 0.0, "clause": 0.35, "word": 0.8, "char": 1.2}
_SEGMENT_COST = 0.12  # every extra segment is one more join to render and to hear
_ORPHAN_FRACTION = 0.35
_ORPHAN_COST = 0.5


def _segment_cost(tokens: int, target: int, boundary: str, *, last: bool, only: bool) -> float:
    ratio = tokens / float(max(1, target))
    deviation = (ratio - 1.0) ** 2 if ratio <= 1.0 else ((ratio - 1.0) * 2.0) ** 2
    cost = deviation + _SEGMENT_COST
    if not last:
        cost += _BOUNDARY_COST.get(boundary, 0.35)
    if not only and ratio < _ORPHAN_FRACTION:
        cost += _ORPHAN_COST
    return cost


def pack_sentences(
    pieces: list[tuple[str, str]],
    budget: int,
    token_len: Callable[[str], int],
    *,
    target: int | None = None,
) -> list[str]:
    """Pack sentence pieces into segments near ``target`` tokens without exceeding ``budget``.

    Dynamic programming over the piece sequence: each segment pays for its distance
    from the target (overshooting costs more than falling short), for ending inside a
    sentence, and for being an orphaned short tail. Every segment is measured with the
    real tokenizer, so a segment never exceeds the budget unless a single piece already does.
    """

    if not pieces:
        return []
    aim = int(target) if target and int(target) > 0 else max(1, int(round(budget * SMART_TARGET_FRACTION)))
    aim = min(aim, max(1, budget))
    count = len(pieces)
    best: list[float] = [0.0] * (count + 1)
    choice: list[int] = [count] * (count + 1)
    piece_tokens = [token_len(piece) for piece, _ in pieces]
    for start in range(count - 1, -1, -1):
        best_cost = float("inf")
        best_end = start + 1
        joined = ""
        running = 0
        for end in range(start, count):
            joined += pieces[end][0]
            running += piece_tokens[end]
            if end > start and running > budget + 2:
                break
            tokens = token_len(joined) if end > start else piece_tokens[end]
            if end > start and tokens > budget:
                break
            cost = _segment_cost(
                tokens, aim, pieces[end][1], last=end == count - 1, only=(start == 0 and end == count - 1),
            ) + best[end + 1]
            if cost < best_cost:
                best_cost, best_end = cost, end + 1
        best[start], choice[start] = best_cost, best_end
    segments: list[str] = []
    position = 0
    while position < count:
        end = choice[position]
        segments.append("".join(piece for piece, _ in pieces[position:end]))
        position = end
    return segments


def split_text_by_sentences(
    text: str,
    budget: int,
    token_len: Callable[[str], int],
    *,
    mode: str = DEFAULT_SEGMENTATION_MODE,
    target_tokens: int | None = None,
) -> list[str]:
    """Sentence-aware segmentation under a token budget (``sentence`` or ``smart`` mode)."""

    source = str(text)
    if not source.strip():
        return [source] if source else []

    def spoken_token_len(value: str) -> int:
        return token_len(normalize_sentence_whitespace(value))

    pieces = sentence_pieces(source, budget, spoken_token_len)
    if normalize_segmentation_mode(mode) == "sentence":
        segments: list[str] = []
        current = ""
        for piece, boundary in pieces:
            if current and spoken_token_len(current + piece) > budget:
                segments.append(current)
                current = ""
            current += piece
            if boundary == "sentence":
                segments.append(current)
                current = ""
        if current:
            segments.append(current)
        return segments or [source]
    return pack_sentences(pieces, budget, spoken_token_len, target=target_tokens) or [source]


def split_text_by_tokens(
    text: str,
    max_tokens: int,
    *,
    capacity: int,
    token_len: Callable[[str], int],
    lang_prefix: str = "",
    segment_budget_scale_non_cjk: float = DEFAULT_NON_CJK_BUDGET_SCALE,
    mode: str = "budget",
    target_tokens: int | None = None,
) -> list[str]:
    """Split text into speech segments within the token budget.

    ``mode`` selects the strategy (see the module docstring); ``budget`` keeps the
    original greedy behavior exactly. ``target_tokens`` is the length the ``smart``
    packer aims for (the trained voice's typical clip); it defaults to 85 percent of
    the budget.
    """

    text = str(text)
    budget = segment_token_budget(
        max_tokens,
        capacity,
        lang_prefix,
        token_len,
        segment_budget_scale_non_cjk,
    )
    selected = normalize_segmentation_mode(mode) if mode != "budget" else "budget"
    if selected != "budget":
        return split_text_by_sentences(text, budget, token_len, mode=selected, target_tokens=target_tokens)
    if token_len(text) <= budget:
        return [text]

    chunks: list[str] = []
    for piece, atomic in split_atomic_pieces(text):
        if atomic:
            chunks.append(piece)
            continue
        for part in _PUNCTUATION_SPLIT.split(piece):
            if not part:
                continue
            if token_len(part) <= budget:
                chunks.append(part)
                continue
            for word in _WORD_PIECES.findall(part):
                if token_len(word) <= budget:
                    chunks.append(word)
                    continue
                # A single oversized word (or unspaced CJK text) is the only
                # case where a character boundary is necessary. Ordinary words
                # must not be cut just because a preceding clause filled up.
                current = ""
                for character in word:
                    if current and token_len(current + character) > budget:
                        chunks.append(current)
                        current = character
                    else:
                        current += character
                if current:
                    chunks.append(current)

    segments: list[str] = []
    current = ""
    for chunk in chunks:
        if current and token_len(current + chunk) > budget:
            segments.append(current)
            current = chunk
        else:
            current += chunk
    if current:
        segments.append(current)
    return segments or [text]


def split_text_for_recovery(text: str, token_len: Callable[[str], int]) -> list[str]:
    """Bisect a failed speech segment without cutting words or pronunciation tags.

    Recovery has no smaller text-token budget: it needs two shorter linguistic
    units. Return an empty list when there is no safe split (e.g. one word).
    Concatenating the returned parts always reproduces the original text.
    """

    protected = [(match.start(), match.end()) for match in _PROTECTED_PATTERN.finditer(text)]

    def safe(position: int) -> bool:
        return (
            bool(text[:position].strip())
            and bool(text[position:].strip())
            and not any(start < position < end for start, end in protected)
        )

    word_boundaries = {match.end() for match in re.finditer(r"\s+", text)}
    clause_boundaries = {
        match.end() for match in re.finditer(r"[\u3001\u3002\uff01\uff0c\uff1a\uff1b\uff1f,!?;:]|\.(?=\s|$)", text)
    }
    # CJK has no required spaces. Do not apply this fallback inside Latin words
    # embedded in a CJK sentence, or inside a protected pronunciation annotation.
    cjk_boundaries = {
        position for position in range(1, len(text))
        if _CJK_CHARACTER.fullmatch(text[position - 1])
        or _CJK_CHARACTER.fullmatch(text[position])
    }
    candidates = {position for position in word_boundaries | clause_boundaries | cjk_boundaries if safe(position)}
    if not candidates:
        return []
    lengths = {position: (token_len(text[:position]), token_len(text[position:])) for position in candidates}
    balanced_clauses = {
        position for position in candidates & clause_boundaries
        if min(lengths[position]) * 3 >= max(lengths[position])
    }
    position = min(
        balanced_clauses or candidates,
        key=lambda value: (abs(lengths[value][0] - lengths[value][1]), value),
    )
    return [text[:position], text[position:]]


__all__ = [
    "CJK_LANGS",
    "DEFAULT_NON_CJK_BUDGET_SCALE",
    "DEFAULT_SEGMENTATION_MODE",
    "SEGMENTATION_MODES",
    "SMART_TARGET_FRACTION",
    "SpeechRecoveryConfig",
    "default_segment_tokens",
    "ends_sentence",
    "normalize_language",
    "normalize_segmentation_mode",
    "normalize_sentence_whitespace",
    "pack_sentences",
    "segment_token_budget",
    "sentence_pieces",
    "split_atomic_pieces",
    "split_sentences",
    "split_text_by_sentences",
    "split_text_by_tokens",
    "split_text_for_recovery",
]
