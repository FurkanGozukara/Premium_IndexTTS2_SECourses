"""Shared token-aware text segmentation for inference and previews."""

from __future__ import annotations

import re
from collections.abc import Callable


CJK_LANGS = frozenset({"zh", "zhen", "ja", "ko", "yue"})
DEFAULT_NON_CJK_BUDGET_SCALE = 0.72
_PROTECTED_PATTERN = re.compile(r"<\|SPECIAL_TOKEN_\d+\|>.*?<\|SPECIAL_TOKEN_\d+\|>")
_PUNCTUATION_SPLIT = re.compile(r"(?<=[\u3001\u3002\uff01\uff0c\uff1a\uff1b\uff1f,.!?;:\n])")
_LANG_PREFIX = re.compile(r"<\|([^|]+)\|>")


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
    """Split text at punctuation, then characters, without exceeding the budget."""

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
            current = ""
            for character in part:
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


__all__ = [
    "CJK_LANGS",
    "DEFAULT_NON_CJK_BUDGET_SCALE",
    "default_segment_tokens",
    "normalize_language",
    "segment_token_budget",
    "split_atomic_pieces",
    "split_text_by_tokens",
]
