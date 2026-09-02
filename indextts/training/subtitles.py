from __future__ import annotations

from dataclasses import dataclass, field, replace
import html
import re
from typing import Any, Iterable, Sequence

from indextts.utils.subtitle_utils import SubtitleCue, parse_subtitle_file


_TAG_RE = re.compile(r"<[^>]*>")
_TIMING_TAG_RE = re.compile(r"<\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}>")
_ALL_CAPS_SPEAKER_RE = re.compile(r"^\s*[A-Z][A-Z0-9 _.'-]{1,39}:\s*")
_ANNOTATION_RE = re.compile(
    r"(?:\[[^\]]+\]"
    r"|\([^)]*(?:music|applause|laugh|laughter|noise|silence|inaudible|cough|sigh|cheer|clap|"
    r"chuckle|gasp|whisper|speaking|screaming|singing)[^)]*\))",
    flags=re.IGNORECASE,
)
_TERMINAL_RE = re.compile(r"[.?!;:。！？][\"')\]]*$")
_WORD_RE = re.compile(r"[\w]+(?:['’-][\w]+)*", flags=re.UNICODE)


@dataclass
class Segment:
    start_ms: int
    end_ms: int
    text: str
    source_cue_indices: tuple[int, ...] = ()
    word_timestamps: list[dict[str, Any]] = field(default_factory=list)
    alignment_coverage: float | None = None
    sentence_aligned: bool | None = None

    @property
    def duration_ms(self) -> int:
        return max(0, int(self.end_ms) - int(self.start_ms))

    @property
    def duration_s(self) -> float:
        return self.duration_ms / 1000.0

    def with_times(self, start_ms: int, end_ms: int) -> "Segment":
        return replace(self, start_ms=int(start_ms), end_ms=int(end_ms))


@dataclass(frozen=True)
class CaptionWord:
    """One caption word with its exact text position and coarse cue timing."""

    text: str
    char_start: int
    char_end: int
    cue_index: int
    cue_start_ms: int
    cue_end_ms: int


@dataclass(frozen=True)
class CaptionCueSpan:
    cue_index: int
    char_start: int
    char_end: int
    word_start: int
    word_end: int
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class CaptionTranscript:
    """Clean caption text plus stable word and cue spans into that text."""

    text: str
    words: tuple[CaptionWord, ...]
    cue_spans: tuple[CaptionCueSpan, ...]


def _fix_punctuation_spacing(text: str) -> str:
    text = re.sub(r"\s+([,.;:!?。！？])", r"\1", text)
    text = re.sub(r"([,;:!?])(?=[^\s\d\"')\]])", r"\1 ", text)
    text = re.sub(r"([.!?。！？])(?=[A-Za-z])", r"\1 ", text)
    text = re.sub(r"([\[(])\s+", r"\1", text)
    text = re.sub(r"\s+([\])])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def dedupe_rolling_repeat(previous_text: str, current_text: str) -> str:
    """Remove a caption prefix that repeats the tail of the preceding cue."""

    previous_words = list(_WORD_RE.finditer(previous_text))
    current_words = list(_WORD_RE.finditer(current_text))
    if not previous_words or not current_words:
        return current_text
    previous_norm = [match.group(0).casefold() for match in previous_words]
    current_norm = [match.group(0).casefold() for match in current_words]
    maximum = min(len(previous_norm), len(current_norm), 40)
    overlap = 0
    for count in range(maximum, 0, -1):
        if previous_norm[-count:] == current_norm[:count]:
            if count >= 2 or len(current_norm[0]) >= 6:
                overlap = count
            break
    if not overlap:
        return current_text
    if overlap == len(current_words):
        return ""
    cut_at = current_words[overlap - 1].end()
    remainder = current_text[cut_at:]
    remainder = re.sub(r"^[\s,;:.-]+", "", remainder)
    return remainder.strip()


def clean_cue_text(
    text: str,
    *,
    remove_bracket_annotations: bool = True,
    previous_text: str | None = None,
    dedupe_rolling_captions: bool = True,
) -> str:
    """Normalize caption markup without rewriting the speaker's words."""

    value = html.unescape(str(text or ""))
    value = _TIMING_TAG_RE.sub(" ", value)
    value = _TAG_RE.sub(" ", value)
    if remove_bracket_annotations:
        value = _ANNOTATION_RE.sub(" ", value)

    cleaned_lines: list[str] = []
    for raw_line in value.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = re.sub(r"^\s*(?:>>+|»+)\s*", "", raw_line)
        line = _ALL_CAPS_SPEAKER_RE.sub("", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            cleaned_lines.append(line)
    value = _fix_punctuation_spacing(" ".join(cleaned_lines))
    if previous_text and dedupe_rolling_captions:
        value = dedupe_rolling_repeat(previous_text, value)
        value = _fix_punctuation_spacing(value)
    return value


def clean_cues(
    cues: Iterable[SubtitleCue],
    *,
    remove_bracket_annotations: bool = True,
    dedupe_rolling_captions: bool = True,
) -> list[SubtitleCue]:
    cleaned: list[SubtitleCue] = []
    previous = ""
    for cue in cues:
        text = clean_cue_text(
            cue.text,
            remove_bracket_annotations=remove_bracket_annotations,
            previous_text=previous,
            dedupe_rolling_captions=dedupe_rolling_captions,
        )
        if text:
            cleaned.append(
                SubtitleCue(index=cue.index, start_ms=cue.start_ms, end_ms=cue.end_ms, text=text)
            )
            previous = text
    return cleaned


def build_caption_transcript(cues: Sequence[SubtitleCue]) -> CaptionTranscript:
    """Join cleaned cues while retaining a coarse cue span for every word.

    ``cues`` should normally be the output of :func:`clean_cues`. Joining is
    performed once across the complete caption stream so wrapping at cue edges
    cannot become an artificial text boundary.
    """

    usable = [cue for cue in cues if str(cue.text or "").strip()]
    if not usable:
        return CaptionTranscript(text="", words=(), cue_spans=())

    text = _join_text([cue.text for cue in usable])
    matches = list(_WORD_RE.finditer(text))
    words: list[CaptionWord] = []
    spans: list[CaptionCueSpan] = []
    cursor = 0
    for position, cue in enumerate(usable):
        cue_word_count = len(_WORD_RE.findall(cue.text))
        word_start = cursor
        word_end = min(len(matches), cursor + cue_word_count)
        if position == len(usable) - 1:
            # Any count mismatch caused by unusual Unicode tokenization belongs
            # to the final cue rather than being silently discarded.
            word_end = len(matches)
        cue_matches = matches[word_start:word_end]
        for match in cue_matches:
            words.append(
                CaptionWord(
                    text=match.group(0),
                    char_start=match.start(),
                    char_end=match.end(),
                    cue_index=int(cue.index),
                    cue_start_ms=int(cue.start_ms),
                    cue_end_ms=int(cue.end_ms),
                )
            )
        if cue_matches:
            char_start = cue_matches[0].start()
            if position + 1 < len(usable) and word_end < len(matches):
                char_end = matches[word_end].start()
            else:
                char_end = len(text)
        else:
            char_start = spans[-1].char_end if spans else 0
            char_end = char_start
        spans.append(
            CaptionCueSpan(
                cue_index=int(cue.index),
                char_start=char_start,
                char_end=char_end,
                word_start=word_start,
                word_end=word_end,
                start_ms=int(cue.start_ms),
                end_ms=int(cue.end_ms),
            )
        )
        cursor = word_end

    return CaptionTranscript(text=text, words=tuple(words), cue_spans=tuple(spans))


def _join_text(parts: Sequence[str]) -> str:
    return _fix_punctuation_spacing(" ".join(part for part in parts if part))


def _strip_leading_conjunction(text: str) -> str:
    return re.sub(
        r"^(?:and|but|so|or|because|then|however|therefore|also)\b[, ]*",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    ).strip()


def merge_cues_into_sentences(
    cues: Sequence[SubtitleCue],
    max_gap_ms: int = 700,
    target_s: float = 8.0,
    max_s: float = 15.0,
    min_s: float = 1.5,
    *,
    remove_bracket_annotations: bool = True,
    dedupe_rolling_captions: bool = True,
    strip_leading_conjunction_fragments: bool = False,
) -> list[Segment]:
    """Join wrapped caption cues into bounded, sentence-aware speech segments."""

    normalized = clean_cues(
        cues,
        remove_bracket_annotations=remove_bracket_annotations,
        dedupe_rolling_captions=dedupe_rolling_captions,
    )
    if not normalized:
        return []
    max_ms = max(1, int(round(max_s * 1000.0)))
    min_ms = max(0, int(round(min_s * 1000.0)))
    target_ms = max(min_ms, int(round(target_s * 1000.0)))
    result: list[Segment] = []
    group: list[SubtitleCue] = []

    def flush() -> None:
        nonlocal group
        if not group:
            return
        text = _join_text([cue.text for cue in group])
        if strip_leading_conjunction_fragments:
            text = _strip_leading_conjunction(text)
        if text:
            result.append(
                Segment(
                    start_ms=int(group[0].start_ms),
                    end_ms=int(max(cue.end_ms for cue in group)),
                    text=text,
                    source_cue_indices=tuple(cue.index for cue in group),
                )
            )
        group = []

    for position, cue in enumerate(normalized):
        if group:
            gap = cue.start_ms - max(item.end_ms for item in group)
            projected = max(cue.end_ms, max(item.end_ms for item in group)) - group[0].start_ms
            if gap > max_gap_ms or (projected > max_ms and group[-1].end_ms > group[0].start_ms):
                flush()
        group.append(cue)
        duration_ms = max(item.end_ms for item in group) - group[0].start_ms
        next_cue = normalized[position + 1] if position + 1 < len(normalized) else None
        next_gap = (
            next_cue.start_ms - max(item.end_ms for item in group) if next_cue is not None else None
        )
        next_duration = (
            max(next_cue.end_ms, max(item.end_ms for item in group)) - group[0].start_ms
            if next_cue is not None
            else 0
        )
        terminal = bool(_TERMINAL_RE.search(group[-1].text))
        should_close = False
        if duration_ms >= min_ms and terminal:
            # When two legal sentence ends are available, retain the one closer
            # to the target instead of producing a needlessly short segment.
            projected_is_better = (
                next_cue is not None
                and next_duration <= max_ms
                and bool(_TERMINAL_RE.search(next_cue.text))
                and abs(next_duration - target_ms) < abs(duration_ms - target_ms)
            )
            should_close = not projected_is_better
        should_close = should_close or next_cue is None
        should_close = should_close or (next_gap is not None and next_gap > max_gap_ms)
        should_close = should_close or (next_cue is not None and next_duration > max_ms)
        if should_close:
            flush()
    flush()

    # Absorb tiny fragments where doing so preserves the maximum duration.
    repaired: list[Segment] = []
    index = 0
    while index < len(result):
        segment = result[index]
        if segment.duration_ms >= min_ms:
            repaired.append(segment)
            index += 1
            continue
        if repaired:
            previous = repaired[-1]
            combined_duration = max(previous.end_ms, segment.end_ms) - previous.start_ms
            if combined_duration <= max_ms:
                repaired[-1] = Segment(
                    previous.start_ms,
                    max(previous.end_ms, segment.end_ms),
                    _join_text([previous.text, segment.text]),
                    previous.source_cue_indices + segment.source_cue_indices,
                    previous.word_timestamps + segment.word_timestamps,
                )
                index += 1
                continue
        if index + 1 < len(result):
            following = result[index + 1]
            combined_duration = max(segment.end_ms, following.end_ms) - segment.start_ms
            if combined_duration <= max_ms:
                result[index + 1] = Segment(
                    segment.start_ms,
                    max(segment.end_ms, following.end_ms),
                    _join_text([segment.text, following.text]),
                    segment.source_cue_indices + following.source_cue_indices,
                    segment.word_timestamps + following.word_timestamps,
                )
                index += 1
                continue
        index += 1
    return repaired


__all__ = [
    "CaptionCueSpan",
    "CaptionTranscript",
    "CaptionWord",
    "Segment",
    "SubtitleCue",
    "clean_cue_text",
    "clean_cues",
    "build_caption_transcript",
    "dedupe_rolling_repeat",
    "merge_cues_into_sentences",
    "parse_subtitle_file",
]
