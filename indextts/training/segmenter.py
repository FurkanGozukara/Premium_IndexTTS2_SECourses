from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re
from typing import Any, Iterable, Sequence

import numpy as np

from .subtitles import CaptionTranscript, Segment


_TERMINAL_RE = re.compile(r"[.?!;:。！？][\"')\]]*$")
_WORD_TOKEN_RE = re.compile(r"\b[\w]+(?:['’-][\w]+)*\b", flags=re.UNICODE)
_SENTENCE_TERMINAL_RE = re.compile(r"[.?!;。！？；][\"')\]]*$")
_BOUNDARY_MODES = frozenset({"sentence", "sentence_or_pause"})
_ALWAYS_NONTERMINAL_ABBREVIATIONS = {
    "dr",
    "mr",
    "mrs",
    "ms",
    "prof",
    "sr",
    "jr",
    "st",
    "vs",
}


@dataclass(frozen=True)
class SentenceSpan:
    char_start: int
    char_end: int
    word_start: int
    word_end: int
    starts_sentence: bool = True
    ends_sentence: bool = True


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _unit_times_ms(item: Any) -> tuple[int, int]:
    start_ms = _value(item, "start_ms")
    end_ms = _value(item, "end_ms")
    if start_ms is not None and end_ms is not None:
        return int(round(float(start_ms))), int(round(float(end_ms)))
    start_s = _value(item, "start_s")
    end_s = _value(item, "end_s")
    if start_s is not None and end_s is not None:
        return int(round(float(start_s) * 1000.0)), int(round(float(end_s) * 1000.0))
    timestamp = _value(item, "timestamp")
    if timestamp and len(timestamp) == 2:
        return int(round(float(timestamp[0]) * 1000.0)), int(round(float(timestamp[1]) * 1000.0))
    raise ValueError(f"Timed unit has no usable timestamps: {item!r}")


def _unit_text(item: Any) -> str:
    return str(_value(item, "text", "") or "").strip()


def _word_dict(item: Any) -> dict[str, Any]:
    start_ms, end_ms = _unit_times_ms(item)
    return {
        "text": _unit_text(item),
        "start_s": start_ms / 1000.0,
        "end_s": end_ms / 1000.0,
    }


def _join_units(units: Sequence[Any]) -> str:
    parts = [_unit_text(unit) for unit in units if _unit_text(unit)]
    if not parts:
        return ""
    if any(str(_value(unit, "text", "")).startswith(" ") for unit in units):
        text = "".join(str(_value(unit, "text", "")) for unit in units)
    else:
        text = " ".join(parts)
    text = re.sub(r"\s+([,.;:!?。！？])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def snap_boundaries_to_silence(
    segment: Segment,
    energy_envelope: np.ndarray,
    hop_ms: int,
    window_ms: int = 200,
    *,
    previous_end_ms: int | None = None,
    next_start_ms: int | None = None,
    start_upper_ms: int | None = None,
    end_lower_ms: int | None = None,
) -> Segment:
    """Move both edges to local energy minima without crossing adjacent segments."""

    energy = np.asarray(energy_envelope, dtype=np.float32).reshape(-1)
    if energy.size == 0 or hop_ms <= 0 or window_ms <= 0:
        return replace(segment)

    max_time_ms = int((energy.size - 1) * hop_ms)

    def minimum_near(boundary_ms: int, lower_ms: int, upper_ms: int) -> int:
        low = max(0, boundary_ms - window_ms, lower_ms)
        high = min(max_time_ms, boundary_ms + window_ms, upper_ms)
        if high < low:
            return max(lower_ms, min(boundary_ms, upper_ms))
        low_i = max(0, int(math.ceil(low / hop_ms)))
        high_i = min(energy.size - 1, int(math.floor(high / hop_ms)))
        if high_i < low_i:
            return max(lower_ms, min(boundary_ms, upper_ms))
        window = energy[low_i : high_i + 1]
        minimum = float(np.min(window))
        candidates = np.flatnonzero(np.isclose(window, minimum, rtol=1e-5, atol=1e-12)) + low_i
        original_i = boundary_ms / float(hop_ms)
        best_i = int(candidates[np.argmin(np.abs(candidates - original_i))])
        return best_i * hop_ms

    start_lower = max(0, int(previous_end_ms or 0))
    start_upper = max(
        start_lower,
        min(segment.end_ms - 1, int(start_upper_ms))
        if start_upper_ms is not None
        else segment.end_ms - 1,
    )
    snapped_start = minimum_near(segment.start_ms, start_lower, start_upper)
    end_lower = min(
        max_time_ms,
        max(snapped_start + 1, int(end_lower_ms))
        if end_lower_ms is not None
        else snapped_start + 1,
    )
    end_upper = max(end_lower, min(max_time_ms, int(next_start_ms or max_time_ms)))
    snapped_end = minimum_near(segment.end_ms, end_lower, end_upper)
    if snapped_end <= snapped_start:
        return replace(segment)
    return replace(segment, start_ms=int(snapped_start), end_ms=int(snapped_end))


def _next_text_character(text: str, position: int) -> str:
    while position < len(text) and (text[position].isspace() or text[position] in "\"')]}»”’"):
        position += 1
    return text[position] if position < len(text) else ""


def _is_sentence_boundary(text: str, position: int) -> bool:
    punctuation = text[position]
    if punctuation != ".":
        return punctuation in "?!;。！？；"
    if position and position + 1 < len(text) and text[position - 1].isdigit() and text[position + 1].isdigit():
        return False

    next_character = _next_text_character(text, position + 1)
    if next_character and next_character.islower():
        return False
    previous = re.search(r"([A-Za-z]+)\.$", text[: position + 1])
    abbreviation = previous.group(1).casefold() if previous else ""
    if abbreviation in _ALWAYS_NONTERMINAL_ABBREVIATIONS:
        return False
    if len(abbreviation) == 1 and next_character and next_character.isupper():
        return False
    # A dotted acronym has no whitespace after this period (for example U.S.).
    if position + 1 < len(text) and text[position + 1].isalpha():
        return False
    return True


def split_caption_sentences(caption: CaptionTranscript) -> list[SentenceSpan]:
    """Split the full caption stream without treating cue wrapping as syntax."""

    if not caption.text or not caption.words:
        return []
    ranges: list[tuple[int, int, bool]] = []
    start = 0
    position = 0
    while position < len(caption.text):
        if caption.text[position] in ".?!;。！？；" and _is_sentence_boundary(caption.text, position):
            end = position + 1
            while end < len(caption.text) and caption.text[end] in ".?!;。！？；\"')]}»”’":
                end += 1
            ranges.append((start, end, True))
            start = end
            position = end
            continue
        position += 1
    if caption.text[start:].strip():
        ranges.append((start, len(caption.text), False))

    spans: list[SentenceSpan] = []
    word_cursor = 0
    for raw_start, raw_end, terminal in ranges:
        char_start = raw_start
        while char_start < raw_end and caption.text[char_start].isspace():
            char_start += 1
        char_end = raw_end
        while char_end > char_start and caption.text[char_end - 1].isspace():
            char_end -= 1
        while word_cursor < len(caption.words) and caption.words[word_cursor].char_end <= char_start:
            word_cursor += 1
        sentence_word_start = word_cursor
        while word_cursor < len(caption.words) and caption.words[word_cursor].char_start < char_end:
            word_cursor += 1
        if word_cursor > sentence_word_start:
            spans.append(
                SentenceSpan(
                    char_start=char_start,
                    char_end=char_end,
                    word_start=sentence_word_start,
                    word_end=word_cursor,
                    starts_sentence=True,
                    ends_sentence=terminal,
                )
            )
    return spans


def _span_times_ms(span: SentenceSpan, words: Sequence[Any]) -> tuple[int, int]:
    start_ms, _ = _unit_times_ms(words[span.word_start])
    _, end_ms = _unit_times_ms(words[span.word_end - 1])
    return start_ms, end_ms


def _split_overlong_sentence_span(
    span: SentenceSpan,
    caption: CaptionTranscript,
    words: Sequence[Any],
    target_ms: int,
    min_ms: int,
    max_ms: int,
) -> list[SentenceSpan]:
    start_ms, end_ms = _span_times_ms(span, words)
    if end_ms - start_ms <= max_ms or span.word_end - span.word_start < 2:
        return [span]

    pieces: list[SentenceSpan] = []
    word_start = span.word_start
    char_start = span.char_start
    while word_start < span.word_end:
        current_start, _ = _unit_times_ms(words[word_start])
        _, remaining_end = _unit_times_ms(words[span.word_end - 1])
        if remaining_end - current_start <= max_ms:
            pieces.append(
                SentenceSpan(
                    char_start=char_start,
                    char_end=span.char_end,
                    word_start=word_start,
                    word_end=span.word_end,
                    starts_sentence=span.starts_sentence and word_start == span.word_start,
                    ends_sentence=span.ends_sentence,
                )
            )
            break

        eligible: list[int] = []
        for cut_after in range(word_start, span.word_end - 1):
            _, cut_end = _unit_times_ms(words[cut_after])
            duration = cut_end - current_start
            if duration > max_ms:
                break
            if duration >= min_ms or not eligible:
                eligible.append(cut_after)
        if not eligible:
            eligible = [word_start]

        target_end = current_start + target_ms
        clause_candidates: list[int] = []
        for cut_after in eligible:
            gap_text = caption.text[
                int(_value(words[cut_after], "char_end")) : int(_value(words[cut_after + 1], "char_start"))
            ]
            if re.search(r"[,,:\-–—]", gap_text):
                clause_candidates.append(cut_after)
        if clause_candidates:
            selected = min(
                clause_candidates,
                key=lambda index: abs(_unit_times_ms(words[index])[1] - target_end),
            )
        else:
            selected = max(
                eligible,
                key=lambda index: (
                    _unit_times_ms(words[index + 1])[0] - _unit_times_ms(words[index])[1],
                    -abs(_unit_times_ms(words[index])[1] - target_end),
                ),
            )

        next_word_start = int(_value(words[selected + 1], "char_start"))
        char_end = next_word_start
        while char_end > char_start and caption.text[char_end - 1].isspace():
            char_end -= 1
        pieces.append(
            SentenceSpan(
                char_start=char_start,
                char_end=char_end,
                word_start=word_start,
                word_end=selected + 1,
                starts_sentence=span.starts_sentence and word_start == span.word_start,
                ends_sentence=False,
            )
        )
        word_start = selected + 1
        char_start = next_word_start
    return pieces


def _group_duration_ms(group: Sequence[SentenceSpan], words: Sequence[Any]) -> int:
    start_ms, _ = _unit_times_ms(words[group[0].word_start])
    _, end_ms = _unit_times_ms(words[group[-1].word_end - 1])
    return end_ms - start_ms


def _starts_sentence_text(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    first = next((character for character in value if character.isalnum()), "")
    if not first:
        return False
    if first.isdigit() or first.isupper() or not first.isalpha():
        return True
    # Scripts such as Chinese and Japanese have sentence starts but no case.
    return first.lower() == first.upper()


def _ends_sentence_text(text: str) -> bool:
    return bool(_SENTENCE_TERMINAL_RE.search(str(text or "").strip()))


def is_sentence_aligned_text(text: str) -> bool:
    return _starts_sentence_text(text) and _ends_sentence_text(text)


def _validate_boundary_options(boundary_mode: str, min_pause_boundary_ms: int) -> tuple[str, int]:
    mode = str(boundary_mode)
    if mode not in _BOUNDARY_MODES:
        raise ValueError(f"Unsupported boundary_mode: {boundary_mode}")
    pause_ms = int(min_pause_boundary_ms)
    if pause_ms < 0:
        raise ValueError("min_pause_boundary_ms must be zero or positive")
    return mode, pause_ms


def _classify_aligned_boundary(
    group: Sequence[SentenceSpan],
    words: Sequence[Any],
    text: str,
    boundary_mode: str,
    min_pause_boundary_ms: int,
) -> tuple[bool, str | None]:
    word_start = group[0].word_start
    word_end = group[-1].word_end
    starts_sentence = group[0].starts_sentence and _starts_sentence_text(text)
    ends_sentence = group[-1].ends_sentence and _ends_sentence_text(text)
    sentence_aligned = starts_sentence and ends_sentence
    if sentence_aligned:
        return True, "sentence"
    if boundary_mode != "sentence_or_pause":
        return False, None

    start_ms, _ = _unit_times_ms(words[word_start])
    _, end_ms = _unit_times_ms(words[word_end - 1])
    preceded_by_pause = False
    if word_start > 0:
        _, previous_end_ms = _unit_times_ms(words[word_start - 1])
        preceded_by_pause = start_ms - previous_end_ms >= min_pause_boundary_ms
    followed_by_pause = False
    if word_end < len(words):
        next_start_ms, _ = _unit_times_ms(words[word_end])
        followed_by_pause = next_start_ms - end_ms >= min_pause_boundary_ms

    if (starts_sentence or preceded_by_pause) and (ends_sentence or followed_by_pause):
        return False, "pause"
    return False, None


def build_sentence_aligned_segments(
    caption: CaptionTranscript,
    aligned_words: Sequence[Any],
    target_s: float = 8.0,
    max_s: float = 12.0,
    min_s: float = 4.0,
    max_gap_ms: int = 700,
    *,
    boundary_mode: str = "sentence",
    min_pause_boundary_ms: int = 400,
) -> list[Segment]:
    """Pack caption spans and classify their aligned sentence or pause edges."""

    if not caption.words or len(aligned_words) != len(caption.words):
        return []
    boundary_mode, min_pause_boundary_ms = _validate_boundary_options(
        boundary_mode,
        min_pause_boundary_ms,
    )
    target_ms = max(1, int(round(target_s * 1000.0)))
    max_ms = max(1, int(round(max_s * 1000.0)))
    min_ms = max(0, int(round(min_s * 1000.0)))

    atomic: list[SentenceSpan] = []
    for sentence in split_caption_sentences(caption):
        atomic.extend(
            _split_overlong_sentence_span(
                sentence,
                caption,
                aligned_words,
                target_ms,
                min_ms,
                max_ms,
            )
        )
    if not atomic:
        return []

    groups: list[list[SentenceSpan]] = []
    current: list[SentenceSpan] = []
    for unit in atomic:
        unit_start, _ = _span_times_ms(unit, aligned_words)
        if current:
            _, previous_end = _span_times_ms(current[-1], aligned_words)
            projected = _group_duration_ms([*current, unit], aligned_words)
            if projected > max_ms or unit_start - previous_end > max_gap_ms:
                groups.append(current)
                current = []
        current.append(unit)
        if _group_duration_ms(current, aligned_words) >= target_ms:
            groups.append(current)
            current = []
    if current:
        groups.append(current)

    repaired: list[list[SentenceSpan]] = []
    index = 0
    while index < len(groups):
        group = groups[index]
        if _group_duration_ms(group, aligned_words) >= min_ms:
            repaired.append(group)
            index += 1
            continue
        if repaired and _group_duration_ms([*repaired[-1], *group], aligned_words) <= max_ms:
            repaired[-1].extend(group)
            index += 1
            continue
        if index + 1 < len(groups) and _group_duration_ms([*group, *groups[index + 1]], aligned_words) <= max_ms:
            groups[index + 1] = [*group, *groups[index + 1]]
            index += 1
            continue
        repaired.append(group)
        index += 1

    segments: list[Segment] = []
    for group in repaired:
        word_start = group[0].word_start
        word_end = group[-1].word_end
        selected_words = aligned_words[word_start:word_end]
        start_ms, _ = _unit_times_ms(selected_words[0])
        _, end_ms = _unit_times_ms(selected_words[-1])
        text = caption.text[group[0].char_start : group[-1].char_end].strip()
        cue_indices = tuple(dict.fromkeys(int(_value(word, "cue_index")) for word in selected_words))
        matched_count = sum(bool(_value(word, "matched", False)) for word in selected_words)
        coverage = matched_count / len(selected_words) if selected_words else 0.0
        sentence_aligned, boundary = _classify_aligned_boundary(
            group,
            aligned_words,
            text,
            boundary_mode,
            min_pause_boundary_ms,
        )
        segments.append(
            Segment(
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
                source_cue_indices=cue_indices,
                word_timestamps=[
                    {
                        "text": _unit_text(word),
                        "start_s": _unit_times_ms(word)[0] / 1000.0,
                        "end_s": _unit_times_ms(word)[1] / 1000.0,
                        "matched": bool(_value(word, "matched", False)),
                    }
                    for word in selected_words
                ],
                alignment_coverage=coverage,
                sentence_aligned=sentence_aligned,
                boundary=boundary,
            )
        )
    return segments


def build_segments_from_words(
    words: Sequence[Any],
    target_s: float = 8.0,
    max_s: float = 15.0,
    min_s: float = 1.5,
    max_gap_ms: int = 700,
) -> list[Segment]:
    """Build sentence-aware segments from Whisper word timestamps."""

    if not words:
        return []
    timed = sorted(words, key=lambda word: _unit_times_ms(word)[0])
    source_indices = {id(word): index for index, word in enumerate(words)}
    target_ms = max(1, int(round(target_s * 1000.0)))
    max_ms = max(1, int(round(max_s * 1000.0)))
    min_ms = max(0, int(round(min_s * 1000.0)))
    groups: list[list[Any]] = []
    current: list[Any] = []

    def flush() -> None:
        nonlocal current
        if current:
            groups.append(current)
            current = []

    for index, word in enumerate(timed):
        start_ms, end_ms = _unit_times_ms(word)
        if end_ms <= start_ms:
            continue
        if current:
            current_start, _ = _unit_times_ms(current[0])
            _, previous_end = _unit_times_ms(current[-1])
            if start_ms - previous_end > max_gap_ms or end_ms - current_start > max_ms:
                flush()
        current.append(word)
        current_start, _ = _unit_times_ms(current[0])
        duration_ms = end_ms - current_start
        next_word = timed[index + 1] if index + 1 < len(timed) else None
        next_gap = None
        next_duration = 0
        if next_word is not None:
            next_start, next_end = _unit_times_ms(next_word)
            next_gap = next_start - end_ms
            next_duration = next_end - current_start
        terminal = bool(_TERMINAL_RE.search(_unit_text(word)))
        close = duration_ms >= min_ms and terminal and (
            duration_ms >= target_ms * 0.55
            or next_duration > target_ms * 1.25
            or next_gap is not None and next_gap > max_gap_ms
        )
        close = close or next_word is None
        close = close or (next_gap is not None and next_gap > max_gap_ms)
        close = close or (next_word is not None and next_duration > max_ms)
        if close:
            flush()
    flush()

    segments: list[Segment] = []
    for group in groups:
        start_ms, _ = _unit_times_ms(group[0])
        _, end_ms = _unit_times_ms(group[-1])
        segments.append(
            Segment(
                start_ms=start_ms,
                end_ms=end_ms,
                text=_join_units(group),
                source_cue_indices=tuple(source_indices[id(word)] for word in group),
                word_timestamps=[_word_dict(word) for word in group],
            )
        )

    repaired: list[Segment] = []
    index = 0
    while index < len(segments):
        segment = segments[index]
        if segment.duration_ms >= min_ms:
            repaired.append(segment)
            index += 1
            continue
        if repaired:
            previous = repaired[-1]
            if segment.end_ms - previous.start_ms <= max_ms:
                repaired[-1] = Segment(
                    previous.start_ms,
                    segment.end_ms,
                    _join_units([{"text": previous.text}, {"text": segment.text}]),
                    previous.source_cue_indices + segment.source_cue_indices,
                    previous.word_timestamps + segment.word_timestamps,
                )
                index += 1
                continue
        if index + 1 < len(segments):
            following = segments[index + 1]
            if following.end_ms - segment.start_ms <= max_ms:
                segments[index + 1] = Segment(
                    segment.start_ms,
                    following.end_ms,
                    _join_units([{"text": segment.text}, {"text": following.text}]),
                    segment.source_cue_indices + following.source_cue_indices,
                    segment.word_timestamps + following.word_timestamps,
                )
        # An isolated sub-minimum fragment is intentionally dropped.
        index += 1
    return repaired


def apply_padding_and_limits(
    segments: Sequence[Segment],
    pad_ms: int,
    media_duration_ms: int,
) -> list[Segment]:
    padding = max(0, int(pad_ms))
    media_end = max(0, int(media_duration_ms))
    result: list[Segment] = []
    for segment in segments:
        start = max(0, int(segment.start_ms) - padding)
        end = min(media_end, int(segment.end_ms) + padding)
        if end > start:
            result.append(replace(segment, start_ms=start, end_ms=end))
    return result


def filter_segments(
    segments: Sequence[Segment],
    min_s: float,
    max_s: float,
    min_words: int,
    max_words: int,
    *,
    min_alignment_coverage: float | None = None,
    require_sentence_aligned: bool = False,
    boundary_mode: str = "sentence",
    reason_counts: dict[str, int] | None = None,
    keep_counts: dict[str, int] | None = None,
) -> list[Segment]:
    boundary_mode, _ = _validate_boundary_options(boundary_mode, 0)
    minimum_ms = int(round(min_s * 1000.0))
    maximum_ms = int(round(max_s * 1000.0))
    accepted: list[Segment] = []
    for segment in segments:
        word_count = len(_WORD_TOKEN_RE.findall(segment.text))
        if not minimum_ms <= segment.duration_ms <= maximum_ms:
            if reason_counts is not None:
                reason_counts["duration"] = reason_counts.get("duration", 0) + 1
            continue
        if not int(min_words) <= word_count <= int(max_words):
            if reason_counts is not None:
                reason_counts["word_count"] = reason_counts.get("word_count", 0) + 1
            continue
        if (
            min_alignment_coverage is not None
            and segment.alignment_coverage is not None
            and segment.alignment_coverage < float(min_alignment_coverage)
        ):
            if reason_counts is not None:
                reason_counts["alignment_coverage"] = reason_counts.get("alignment_coverage", 0) + 1
            continue
        if require_sentence_aligned and segment.sentence_aligned is False:
            if boundary_mode == "sentence_or_pause" and segment.boundary == "pause":
                if keep_counts is not None:
                    keep_counts["pause_boundary"] = keep_counts.get("pause_boundary", 0) + 1
            else:
                if reason_counts is not None:
                    reason_counts["sentence_boundary"] = reason_counts.get("sentence_boundary", 0) + 1
                continue
        if segment.text.strip():
            accepted.append(segment)
    return accepted


def _split_text_proportionally(segment: Segment, max_ms: int) -> list[Segment]:
    matches = list(_WORD_TOKEN_RE.finditer(segment.text))
    piece_count = max(2, int(math.ceil(segment.duration_ms / float(max_ms))))
    if len(matches) < piece_count:
        return []
    chunks: list[Segment] = []
    for piece_index in range(piece_count):
        word_start = round(piece_index * len(matches) / piece_count)
        word_end = round((piece_index + 1) * len(matches) / piece_count)
        if word_end <= word_start:
            continue
        char_start = matches[word_start].start()
        char_end = matches[word_end - 1].end()
        while char_end < len(segment.text) and segment.text[char_end] in ",.;:!?。！？\"') ]":
            char_end += 1
        text = segment.text[char_start:char_end].strip()
        start_ms = segment.start_ms + round(segment.duration_ms * word_start / len(matches))
        end_ms = segment.start_ms + round(segment.duration_ms * word_end / len(matches))
        chunks.append(
            Segment(start_ms, end_ms, text, segment.source_cue_indices, [])
        )
    return chunks


def split_long_segment(
    segment: Segment,
    words_or_cues: Sequence[Any],
    max_s: float,
) -> list[Segment]:
    """Split at timed word/cue edges, with proportional text splitting as a last resort."""

    max_ms = max(1, int(round(max_s * 1000.0)))
    if segment.duration_ms <= max_ms:
        return [replace(segment)]
    units: list[Any] = []
    for unit in words_or_cues:
        try:
            start_ms, end_ms = _unit_times_ms(unit)
        except (TypeError, ValueError):
            continue
        if end_ms > segment.start_ms and start_ms < segment.end_ms:
            units.append(unit)
    units.sort(key=lambda item: _unit_times_ms(item)[0])
    if len(units) < 2:
        proportional = _split_text_proportionally(segment, max_ms)
        return proportional or [replace(segment)]

    output: list[Segment] = []
    current: list[Any] = []
    for unit in units:
        unit_start, unit_end = _unit_times_ms(unit)
        if current:
            current_start, _ = _unit_times_ms(current[0])
            if unit_end - current_start > max_ms:
                previous_end = _unit_times_ms(current[-1])[1]
                output.append(
                    Segment(
                        max(segment.start_ms, current_start),
                        min(segment.end_ms, previous_end),
                        _join_units(current),
                        tuple(int(_value(item, "index", idx)) for idx, item in enumerate(current)),
                        [_word_dict(item) for item in current if _value(item, "start_s") is not None],
                    )
                )
                current = []
        current.append(unit)
    if current:
        start_ms, _ = _unit_times_ms(current[0])
        _, end_ms = _unit_times_ms(current[-1])
        output.append(
            Segment(
                max(segment.start_ms, start_ms),
                min(segment.end_ms, end_ms),
                _join_units(current),
                tuple(int(_value(item, "index", idx)) for idx, item in enumerate(current)),
                [_word_dict(item) for item in current if _value(item, "start_s") is not None],
            )
        )

    final: list[Segment] = []
    for piece in output:
        if piece.duration_ms > max_ms:
            proportional = _split_text_proportionally(piece, max_ms)
            final.extend(proportional or [piece])
        elif piece.duration_ms > 0 and piece.text:
            final.append(piece)
    return final


__all__ = [
    "SentenceSpan",
    "apply_padding_and_limits",
    "build_sentence_aligned_segments",
    "build_segments_from_words",
    "filter_segments",
    "is_sentence_aligned_text",
    "snap_boundaries_to_silence",
    "split_caption_sentences",
    "split_long_segment",
]
