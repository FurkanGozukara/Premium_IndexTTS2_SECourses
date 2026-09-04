"""Parse explicit inline pauses without sending the tags to the tokenizer."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re


@dataclass(frozen=True, slots=True)
class TextChunk:
    text: str


@dataclass(frozen=True, slots=True)
class PauseChunk:
    duration_s: float

    @property
    def duration_ms(self) -> int:
        return int(round(self.duration_s * 1000.0))


Chunk = TextChunk | PauseChunk

_PAUSE_PATTERN = re.compile(
    r"(?:\[\s*pause\s*:\s*(?P<bracket>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>ms|s)\s*\]"
    r"|<\s*pause\s*=\s*(?P<angle>[0-9]+(?:\.[0-9]+)?)\s*>)",
    flags=re.IGNORECASE,
)


def _duration_seconds(match: re.Match[str]) -> float:
    if match.group("angle") is not None:
        value = float(match.group("angle"))
    else:
        value = float(match.group("bracket"))
        if match.group("unit").lower() == "ms":
            value /= 1000.0
    if not math.isfinite(value) or value < 0:
        raise ValueError("pause duration must be a finite non-negative number")
    return value


def split_text_with_pauses(text: str) -> list[Chunk]:
    """Return text and pause chunks in their original order."""

    source = str(text)
    chunks: list[Chunk] = []
    position = 0
    for match in _PAUSE_PATTERN.finditer(source):
        if match.start() > position:
            chunks.append(TextChunk(source[position:match.start()]))
        chunks.append(PauseChunk(_duration_seconds(match)))
        position = match.end()
    if position < len(source):
        chunks.append(TextChunk(source[position:]))
    if not chunks:
        chunks.append(TextChunk(source))
    return chunks


def describe_pauses(text: str) -> str:
    """Return a compact human-readable pause summary for preview surfaces."""

    pauses = [chunk for chunk in split_text_with_pauses(text) if isinstance(chunk, PauseChunk)]
    if not pauses:
        return "No inline pauses"
    durations = ", ".join(f"{chunk.duration_ms} ms" for chunk in pauses)
    total_ms = sum(chunk.duration_ms for chunk in pauses)
    noun = "pause" if len(pauses) == 1 else "pauses"
    return f"{len(pauses)} inline {noun}: {durations} ({total_ms} ms total)"


__all__ = ["Chunk", "PauseChunk", "TextChunk", "describe_pauses", "split_text_with_pauses"]
