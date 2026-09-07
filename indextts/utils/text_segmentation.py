"""Shared token-aware text segmentation for inference and previews."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass


CJK_LANGS = frozenset({"zh", "zhen", "ja", "ko", "yue"})
DEFAULT_NON_CJK_BUDGET_SCALE = 0.72
_PROTECTED_PATTERN = re.compile(r"<\|SPECIAL_TOKEN_\d+\|>.*?<\|SPECIAL_TOKEN_\d+\|>")
_PUNCTUATION_SPLIT = re.compile(r"(?<=[\u3001\u3002\uff01\uff0c\uff1a\uff1b\uff1f,.!?;:\n])")
_LANG_PREFIX = re.compile(r"<\|([^|]+)\|>")
_WORD_PIECES = re.compile(r"\s+|\S+")
_CJK_CHARACTER = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7a3]")


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


def split_text_by_tokens(
    text: str,
    max_tokens: int,
    *,
    capacity: int,
    token_len: Callable[[str], int],
    lang_prefix: str = "",
    segment_budget_scale_non_cjk: float = DEFAULT_NON_CJK_BUDGET_SCALE,
) -> list[str]:
    """Prefer punctuation and whole words; split characters only inside an oversized word."""

    text = str(text)
    budget = segment_token_budget(
        max_tokens,
        capacity,
        lang_prefix,
        token_len,
        segment_budget_scale_non_cjk,
    )
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
    "SpeechRecoveryConfig",
    "default_segment_tokens",
    "normalize_language",
    "segment_token_budget",
    "split_atomic_pieces",
    "split_text_by_tokens",
    "split_text_for_recovery",
]
