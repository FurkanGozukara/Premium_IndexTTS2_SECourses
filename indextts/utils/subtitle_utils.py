"""Caption parsing for every common subtitle format plus subtitle-timed audio helpers.

Caption files arrive in many shapes: SRT and WebVTT from editors and YouTube, SBV
and SubViewer, Advanced SubStation Alpha, MicroDVD, LRC lyrics, TTML/DFXP, SAMI,
and the JSON/TSV transcripts written by speech recognizers. Every parser here is
tolerant of stray text, missing milliseconds, byte-order marks and Windows line
endings, and the format is detected from the content before the extension is
trusted, so a WebVTT file saved as ``.srt`` still loads.
"""

from __future__ import annotations

from dataclasses import dataclass
import codecs
import html
import json
import os
import re
import shutil
import subprocess
from typing import Callable, Iterable, List, Sequence, Tuple
import wave

import librosa
import numpy as np
from scipy.signal import resample

from indextts.utils.text_encoding import read_text_resilient


TIMECODE_RE = re.compile(
    r"^(?:(?P<hours>\d+):)?(?P<minutes>\d{1,2}):(?P<seconds>\d{1,2})(?:[,.](?P<milliseconds>\d{1,3}))?$"
)
_FRAME_TIMECODE_RE = re.compile(
    r"^(?P<hours>\d+):(?P<minutes>\d{1,2}):(?P<seconds>\d{1,2})[:;](?P<frames>\d{1,3})$"
)
_OFFSET_TIME_RE = re.compile(r"^([+-]?\d+(?:[.,]\d+)?)\s*(h|m|s|ms|t|f)?$", re.IGNORECASE)
_ARROW_SPLIT_RE = re.compile(r"\s*-{1,3}>\s*")
_SBV_TIMELINE_RE = re.compile(
    r"^(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3})\s*,\s*(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3})\s*$"
)
_MICRODVD_RE = re.compile(r"^\{(\d+)\}\{(\d*)\}(.*)$")
_LRC_TAGS_RE = re.compile(r"^((?:\[\d{1,3}:\d{2}(?:[.:]\d{1,3})?\])+)(.*)$")
_LRC_TIME_RE = re.compile(r"\[(\d{1,3}):(\d{2})(?:[.:](\d{1,3}))?\]")
_LRC_WORD_TIME_RE = re.compile(r"<\d{1,3}:\d{2}(?:[.:]\d{1,3})?>")
_SAMI_SYNC_RE = re.compile(r"<sync[^>]*?start\s*=\s*\"?(\d+)\"?[^>]*>", re.IGNORECASE)
_ASS_OVERRIDE_RE = re.compile(r"\{[^}]*\}")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_JSON_HINT_RE = re.compile(
    r"\"(?:segments|events|cues|subtitles|captions|items|transcript|utterances)\"\s*:\s*\["
    r"|^\s*\[\s*\{[^{}]*\"(?:start|start_ms|startMs|start_time|startTime|begin|from|offset|tStartMs)\"",
    re.DOTALL,
)

# Every caption extension accepted as a sidecar next to a recording, as an upload
# in Voice Generation and Batch Generation, and by the dataset preparation scan.
SUPPORTED_SUBTITLE_EXTENSIONS = (
    ".srt",
    ".vtt",
    ".sbv",
    ".ass",
    ".ssa",
    ".sub",
    ".lrc",
    ".ttml",
    ".dfxp",
    ".smi",
    ".sami",
    ".json",
    ".json3",
    ".tsv",
)
# Extensions that unrelated files also use (recognizer output, download metadata,
# binary VobSub). They count as captions only when the content looks like captions.
AMBIGUOUS_SUBTITLE_EXTENSIONS = frozenset({".json", ".json3", ".tsv", ".sub"})
SUBTITLE_FORMAT_LABELS = {
    ".srt": "SRT",
    ".vtt": "WebVTT",
    ".sbv": "SBV",
    ".ass": "ASS",
    ".ssa": "SSA",
    ".sub": "SUB",
    ".lrc": "LRC",
    ".ttml": "TTML",
    ".dfxp": "DFXP",
    ".xml": "TTML",
    ".smi": "SAMI",
    ".sami": "SAMI",
    ".json": "JSON",
    ".json3": "JSON3",
    ".tsv": "TSV",
}
SUBTITLE_FORMAT_SUMMARY = "SRT, VTT, SBV, ASS/SSA, SUB, LRC, TTML/DFXP, SAMI, JSON, TSV"
_DEFAULT_CUE_DURATION_MS = 5000


@dataclass(frozen=True)
class SubtitleCue:
    index: int
    start_ms: int
    end_ms: int
    text: str

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass(frozen=True)
class SubtitleRenderUnit:
    index: int
    start_ms: int
    end_ms: int
    text: str
    cue_indices: Tuple[int, ...]

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass(frozen=True)
class _Entry:
    """A parsed caption whose end may still have to be taken from the next cue."""

    start_ms: int
    end_ms: int | None
    text: str


def parse_srt_timestamp(value: str) -> int:
    """Parse ``HH:MM:SS,mmm``; hours, the separator style and the fraction are optional."""

    text = str(value).strip().replace(" ", "")
    match = TIMECODE_RE.match(text)
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {value}")

    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    fraction = match.group("milliseconds")
    milliseconds = int(fraction.ljust(3, "0")) if fraction else 0
    return (((hours * 60) + minutes) * 60 + seconds) * 1000 + milliseconds


def parse_timestamp_flexible(
    value: str | int | float,
    *,
    frame_rate: float = 25.0,
    tick_rate: float | None = None,
    unit: str = "s",
) -> int:
    """Parse a timecode, a SMPTE ``HH:MM:SS:FF`` value, or a bare number.

    Bare numbers are seconds unless ``unit`` is ``"ms"``; TTML offset suffixes
    (``12.5s``, ``1500ms``, ``2m``, ``1h``, ``30f``, ``250t``) are honored.
    """

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        scale = 1.0 if unit == "ms" else 1000.0
        return int(round(float(value) * scale))
    text = str(value).strip().replace(" ", "")
    if not text:
        raise ValueError("Timestamp is empty")
    frames = _FRAME_TIMECODE_RE.match(text)
    if frames:
        base = (
            (int(frames.group("hours")) * 60 + int(frames.group("minutes"))) * 60
            + int(frames.group("seconds"))
        ) * 1000
        return base + int(round(int(frames.group("frames")) * 1000.0 / max(float(frame_rate), 1e-6)))
    if TIMECODE_RE.match(text):
        return parse_srt_timestamp(text)
    offset = _OFFSET_TIME_RE.match(text)
    if offset:
        number = float(offset.group(1).replace(",", "."))
        suffix = (offset.group(2) or unit).lower()
        if suffix == "h":
            return int(round(number * 3_600_000))
        if suffix == "m":
            return int(round(number * 60_000))
        if suffix == "s":
            return int(round(number * 1000))
        if suffix == "ms":
            return int(round(number))
        if suffix == "f":
            return int(round(number * 1000.0 / max(float(frame_rate), 1e-6)))
        if suffix == "t":
            rate = float(tick_rate) if tick_rate else 1.0
            return int(round(number * 1000.0 / max(rate, 1e-9)))
    raise ValueError(f"Invalid timestamp: {value}")


def format_srt_timestamp(value_ms: int) -> str:
    total_ms = max(0, int(value_ms))
    hours, remainder = divmod(total_ms, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def read_subtitle_file(
    path: str,
    warning_callback: Callable[[str], None] | None = None,
) -> str:
    return read_text_resilient(path, warning_callback=warning_callback)


def read_srt_file(
    path: str,
    warning_callback: Callable[[str], None] | None = None,
) -> str:
    return read_subtitle_file(path, warning_callback=warning_callback)


def get_subtitle_extension(path: str | None) -> str:
    if not path:
        return ".srt"
    raw_value = str(path).strip().lower()
    extension = os.path.splitext(raw_value)[1]
    if not extension and raw_value.startswith("."):
        extension = raw_value
    return extension or ".srt"


def get_subtitle_format_label(path: str | None) -> str:
    extension = get_subtitle_extension(path)
    return SUBTITLE_FORMAT_LABELS.get(extension, extension.lstrip(".").upper() or "caption")


def _normalize_text(content: str) -> str:
    return str(content or "").lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")


def _blocks(content: str) -> List[List[str]]:
    normalized = _normalize_text(content).strip()
    if not normalized:
        return []
    return [block.split("\n") for block in re.split(r"\n\s*\n", normalized)]


def _split_timeline(line: str) -> Tuple[int, int] | None:
    """Return (start, end) for ``start --> end [settings]`` lines, else ``None``."""

    parts = _ARROW_SPLIT_RE.split(str(line).strip(), maxsplit=1)
    if len(parts) != 2:
        return None
    try:
        start_ms = parse_srt_timestamp(parts[0].split()[0])
        end_ms = parse_srt_timestamp(parts[1].split()[0])
    except (ValueError, IndexError):
        return None
    return start_ms, max(start_ms, end_ms)


def _build_cues(
    entries: Iterable[_Entry],
    *,
    sort: bool,
    default_duration_ms: int = _DEFAULT_CUE_DURATION_MS,
) -> List[SubtitleCue]:
    """Fill missing cue ends from the next start and drop end markers without text."""

    items = list(entries)
    if sort:
        items.sort(key=lambda entry: entry.start_ms)
    cues: List[SubtitleCue] = []
    for position, entry in enumerate(items):
        end_ms = entry.end_ms
        if end_ms is None:
            if position + 1 < len(items):
                end_ms = items[position + 1].start_ms
            else:
                end_ms = entry.start_ms + default_duration_ms
        end_ms = max(int(entry.start_ms), int(end_ms))
        text = str(entry.text or "").strip("\n")
        if not text.strip():
            continue
        cues.append(SubtitleCue(index=len(cues) + 1, start_ms=int(entry.start_ms), end_ms=end_ms, text=text))
    return cues


def parse_srt(content: str) -> List[SubtitleCue]:
    """Parse SubRip text; stray blocks and missing indices are skipped, not fatal."""

    cues: List[SubtitleCue] = []
    for lines in _blocks(content):
        stripped = [line.strip() for line in lines]
        if not any(stripped):
            continue
        timeline_index = next(
            (position for position, line in enumerate(stripped[:3]) if _split_timeline(line) is not None),
            None,
        )
        if timeline_index is None:
            continue
        start_ms, end_ms = _split_timeline(stripped[timeline_index])  # type: ignore[misc]
        cue_index = len(cues) + 1
        if timeline_index >= 1 and stripped[timeline_index - 1].isdigit():
            cue_index = int(stripped[timeline_index - 1])
        text = "\n".join(lines[timeline_index + 1 :]).strip("\n")
        cues.append(SubtitleCue(index=cue_index, start_ms=start_ms, end_ms=end_ms, text=text))
    return cues


def parse_vtt(content: str) -> List[SubtitleCue]:
    """Parse WebVTT including cue identifiers, cue settings, NOTE/STYLE/REGION blocks."""

    cues: List[SubtitleCue] = []
    for lines in _blocks(content):
        stripped = [line.strip() for line in lines]
        if not any(stripped):
            continue
        first_line = next(line for line in stripped if line)
        if first_line.upper().startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue
        timeline_index = next(
            (position for position, line in enumerate(stripped[:3]) if _split_timeline(line) is not None),
            None,
        )
        if timeline_index is None:
            continue
        start_ms, end_ms = _split_timeline(stripped[timeline_index])  # type: ignore[misc]
        text = "\n".join(lines[timeline_index + 1 :]).strip("\n")
        cues.append(SubtitleCue(index=len(cues) + 1, start_ms=start_ms, end_ms=end_ms, text=text))
    return cues


def parse_sbv(content: str) -> List[SubtitleCue]:
    """Parse YouTube SBV and SubViewer 2.0 (``[INFORMATION]`` header, ``[br]`` breaks)."""

    cues: List[SubtitleCue] = []
    for lines in _blocks(content):
        stripped = [line.strip() for line in lines]
        if not any(stripped):
            continue
        first_index = next(position for position, line in enumerate(stripped) if line)
        match = _SBV_TIMELINE_RE.match(stripped[first_index])
        if not match:
            continue
        start_ms = parse_srt_timestamp(match.group(1))
        end_ms = max(start_ms, parse_srt_timestamp(match.group(2)))
        text = "\n".join(lines[first_index + 1 :]).strip("\n")
        text = re.sub(r"\[br\]", "\n", text, flags=re.IGNORECASE)
        cues.append(SubtitleCue(index=len(cues) + 1, start_ms=start_ms, end_ms=end_ms, text=text))
    return cues


def parse_ass(content: str) -> List[SubtitleCue]:
    """Parse Advanced SubStation Alpha / SSA ``Dialogue:`` events without their styling."""

    default_fields = ["layer", "start", "end", "style", "name", "marginl", "marginr", "marginv", "effect", "text"]
    fields: List[str] | None = None
    in_events = False
    entries: List[_Entry] = []
    for raw_line in _normalize_text(content).split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            in_events = line.lower() == "[events]"
            fields = None
            continue
        if not in_events:
            continue
        lowered = line.lower()
        if lowered.startswith("format:"):
            fields = [field.strip().lower() for field in line.split(":", 1)[1].split(",")]
            continue
        if not lowered.startswith("dialogue:"):
            continue
        names = fields or default_fields
        parts = line.split(":", 1)[1].strip().split(",", len(names) - 1)
        if len(parts) < len(names):
            continue
        record = dict(zip(names, parts))
        try:
            start_ms = parse_srt_timestamp(record.get("start", ""))
            end_ms = parse_srt_timestamp(record.get("end", ""))
        except ValueError:
            continue
        text = _ASS_OVERRIDE_RE.sub("", record.get("text", ""))
        text = text.replace("\\N", "\n").replace("\\n", "\n").replace("\\h", " ")
        entries.append(_Entry(start_ms, end_ms, text))
    return _build_cues(entries, sort=True)


def parse_microdvd(content: str, frame_rate: float | None = None) -> List[SubtitleCue]:
    """Parse frame-based MicroDVD ``{start}{end}text`` lines; a leading ``{1}{1}fps`` sets the rate."""

    fps = float(frame_rate) if frame_rate else 25.0
    entries: List[_Entry] = []
    first = True
    for raw_line in _normalize_text(content).split("\n"):
        match = _MICRODVD_RE.match(raw_line.strip())
        if not match:
            continue
        start_frame = int(match.group(1))
        end_frame = int(match.group(2)) if match.group(2) else None
        text = match.group(3)
        if first:
            first = False
            declared = re.fullmatch(r"\s*(\d+(?:[.,]\d+)?)\s*", text)
            if declared and start_frame <= 1 and (end_frame is None or end_frame <= 1):
                fps = float(declared.group(1).replace(",", ".")) or fps
                continue
        text = _ASS_OVERRIDE_RE.sub("", text).replace("|", "\n")
        start_ms = int(round(start_frame * 1000.0 / fps))
        end_ms = int(round(end_frame * 1000.0 / fps)) if end_frame is not None else None
        entries.append(_Entry(start_ms, end_ms, text))
    return _build_cues(entries, sort=True)


def parse_sub(content: str) -> List[SubtitleCue]:
    """``.sub`` is either MicroDVD (frame braces) or SubViewer (SBV-like timelines)."""

    head = _normalize_text(content).lstrip()[:4000]
    if re.search(r"^\{\d+\}\{\d*\}", head, flags=re.MULTILINE):
        return parse_microdvd(content)
    return parse_sbv(content)


def parse_lrc(content: str) -> List[SubtitleCue]:
    """Parse LRC lyrics; each timed line ends where the next one starts."""

    entries: List[_Entry] = []
    for raw_line in _normalize_text(content).split("\n"):
        match = _LRC_TAGS_RE.match(raw_line.strip())
        if not match:
            continue
        text = _LRC_WORD_TIME_RE.sub(" ", match.group(2))
        text = re.sub(r"\s+", " ", text).strip()
        for stamp in _LRC_TIME_RE.finditer(match.group(1)):
            fraction = stamp.group(3) or ""
            start_ms = int(stamp.group(1)) * 60_000 + int(stamp.group(2)) * 1000
            start_ms += int(fraction.ljust(3, "0")) if fraction else 0
            entries.append(_Entry(start_ms, None, text))
    if not entries:
        raise ValueError("No timed LRC lines were found")
    return _build_cues(entries, sort=True)


def _local_name(tag: object) -> str:
    return str(tag).rsplit("}", 1)[-1].lower() if isinstance(tag, str) else ""


def _ttml_text(element) -> str:
    parts: List[str] = [element.text or ""]
    for child in element:
        if _local_name(child.tag) == "br":
            parts.append("\n")
        else:
            parts.append(_ttml_text(child))
        parts.append(child.tail or "")
    text = "".join(parts)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def parse_ttml(content: str) -> List[SubtitleCue]:
    """Parse TTML / DFXP / SMPTE-TT ``<p begin end|dur>`` paragraphs."""

    import xml.etree.ElementTree as ElementTree

    text = _normalize_text(content).strip()
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError as exc:
        raise ValueError(f"Invalid TTML/XML captions: {exc}") from exc

    frame_rate = 25.0
    frame_rate_declared = False
    multiplier = 1.0
    tick_rate: float | None = None
    for key, value in root.attrib.items():
        name = _local_name(key)
        try:
            if name == "framerate":
                frame_rate = float(value)
                frame_rate_declared = True
            elif name == "frameratemultiplier":
                numerator, denominator = value.split()
                multiplier = float(numerator) / float(denominator)
            elif name == "tickrate":
                tick_rate = float(value)
        except (TypeError, ValueError, ZeroDivisionError):
            continue
    frame_rate *= multiplier
    if tick_rate is None:
        tick_rate = frame_rate if frame_rate_declared else 1.0

    entries: List[_Entry] = []
    for element in root.iter():
        if _local_name(element.tag) != "p":
            continue
        begin = element.attrib.get("begin")
        if begin is None:
            continue
        try:
            start_ms = parse_timestamp_flexible(begin, frame_rate=frame_rate, tick_rate=tick_rate)
            end_value = element.attrib.get("end")
            duration_value = element.attrib.get("dur")
            if end_value is not None:
                end_ms: int | None = parse_timestamp_flexible(end_value, frame_rate=frame_rate, tick_rate=tick_rate)
            elif duration_value is not None:
                end_ms = start_ms + parse_timestamp_flexible(duration_value, frame_rate=frame_rate, tick_rate=tick_rate)
            else:
                end_ms = None
        except ValueError:
            continue
        entries.append(_Entry(start_ms, end_ms, _ttml_text(element)))
    return _build_cues(entries, sort=True)


def parse_sami(content: str) -> List[SubtitleCue]:
    """Parse SAMI ``<SYNC Start=ms>`` blocks; a ``&nbsp;`` block ends the previous caption."""

    text = _normalize_text(content)
    matches = list(_SAMI_SYNC_RE.finditer(text))
    if not matches:
        raise ValueError("No <SYNC> blocks were found")
    entries: List[_Entry] = []
    for position, match in enumerate(matches):
        stop = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        chunk = re.split(r"</body", text[match.end() : stop], flags=re.IGNORECASE)[0]
        paragraphs = re.findall(r"<p[^>]*>(.*?)(?=<p[^>]*>|$)", chunk, flags=re.IGNORECASE | re.DOTALL)
        body = paragraphs[0] if paragraphs else chunk
        body = re.sub(r"<br\s*/?>", "\n", body, flags=re.IGNORECASE)
        body = html.unescape(_HTML_TAG_RE.sub("", body)).replace("\xa0", " ")
        lines = [re.sub(r"\s+", " ", line).strip() for line in body.split("\n")]
        entries.append(_Entry(int(match.group(1)), None, "\n".join(line for line in lines if line)))
    return _build_cues(entries, sort=False)


def _json_time(record: dict, keys: Sequence[str]) -> int | None:
    for key in keys:
        if key not in record or record[key] is None:
            continue
        value = record[key]
        unit = "ms" if key.lower().endswith("ms") else "s"
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(round(float(value) * (1.0 if unit == "ms" else 1000.0)))
        return parse_timestamp_flexible(str(value), unit=unit)
    return None


def parse_json_subtitles(content: str) -> List[SubtitleCue]:
    """Parse recognizer JSON (Whisper ``segments``, YouTube ``json3`` events, or a list of cues)."""

    try:
        payload = json.loads(_normalize_text(content))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON captions: {exc}") from exc

    entries: List[_Entry] = []
    if isinstance(payload, dict) and isinstance(payload.get("events"), list):
        for event in payload["events"]:
            if not isinstance(event, dict) or not isinstance(event.get("segs"), list):
                continue
            text = "".join(str(seg.get("utf8", "")) for seg in event["segs"] if isinstance(seg, dict))
            start_ms = int(event.get("tStartMs", 0) or 0)
            duration = event.get("dDurationMs")
            end_ms = start_ms + int(duration) if duration is not None else None
            entries.append(_Entry(start_ms, end_ms, text))
        return _build_cues(entries, sort=True)

    records = None
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        for key in ("segments", "cues", "subtitles", "captions", "items", "transcript", "utterances", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                records = value
                break
            if isinstance(value, dict):
                nested = next(
                    (value[name] for name in ("segments", "utterances", "items") if isinstance(value.get(name), list)),
                    None,
                )
                if nested is not None:
                    records = nested
                    break
    if records is None:
        raise ValueError("JSON does not contain a caption or segment list")

    for record in records:
        if not isinstance(record, dict):
            continue
        text = record.get("text") or record.get("content") or record.get("transcript") or record.get("utf8") or ""
        if not text and isinstance(record.get("words"), list):
            text = " ".join(
                str(word.get("word") or word.get("text") or "").strip()
                for word in record["words"]
                if isinstance(word, dict)
            ).strip()
        start_ms = _json_time(
            record,
            ("start", "start_time", "startTime", "begin", "from", "offset", "start_ms", "startMs", "tStartMs"),
        )
        if start_ms is None:
            continue
        end_ms = _json_time(record, ("end", "end_time", "endTime", "to", "end_ms", "endMs"))
        if end_ms is None:
            duration = _json_time(record, ("duration", "dur", "duration_ms", "durationMs", "dDurationMs"))
            end_ms = start_ms + duration if duration is not None else None
        entries.append(_Entry(start_ms, end_ms, str(text)))
    return _build_cues(entries, sort=True)


def parse_tsv(content: str) -> List[SubtitleCue]:
    """Parse tab-separated transcripts such as Whisper's ``start<TAB>end<TAB>text`` (milliseconds)."""

    lines = [line for line in _normalize_text(content).split("\n") if line.strip()]
    if not lines:
        return []
    header = [cell.strip().lower() for cell in lines[0].split("\t")]
    columns = {"start": 0, "end": 1, "text": 2}
    body = lines
    if "start" in header or "text" in header:
        columns = {name: header.index(name) for name in ("start", "end", "text") if name in header}
        for alias in ("transcript", "content", "caption", "sentence"):
            if "text" not in columns and alias in header:
                columns["text"] = header.index(alias)
        body = lines[1:]
    if "start" not in columns or "text" not in columns:
        raise ValueError("TSV captions need start, end and text columns")

    rows: List[Tuple[str, str, str]] = []
    for line in body:
        cells = line.split("\t")
        if len(cells) <= max(columns["start"], columns["text"]):
            continue
        start_raw = cells[columns["start"]].strip()
        end_raw = cells[columns["end"]].strip() if "end" in columns and len(cells) > columns["end"] else ""
        if columns["text"] == max(columns.values()):
            text = "\t".join(cells[columns["text"] :])
        else:
            text = cells[columns["text"]]
        rows.append((start_raw, end_raw, text.strip()))

    integers_only = all(
        re.fullmatch(r"-?\d+", start) and (not end or re.fullmatch(r"-?\d+", end)) for start, end, _ in rows
    )

    def convert(raw: str) -> int:
        if integers_only:
            return int(raw)
        return parse_timestamp_flexible(raw)

    entries: List[_Entry] = []
    for start_raw, end_raw, text in rows:
        try:
            start_ms = convert(start_raw)
            end_ms = convert(end_raw) if end_raw else None
        except ValueError:
            continue
        entries.append(_Entry(start_ms, end_ms, text))
    return _build_cues(entries, sort=True)


_PARSERS: dict[str, Callable[[str], List[SubtitleCue]]] = {
    ".srt": parse_srt,
    ".vtt": parse_vtt,
    ".sbv": parse_sbv,
    ".ass": parse_ass,
    ".ssa": parse_ass,
    ".sub": parse_sub,
    ".lrc": parse_lrc,
    ".ttml": parse_ttml,
    ".dfxp": parse_ttml,
    ".xml": parse_ttml,
    ".smi": parse_sami,
    ".sami": parse_sami,
    ".json": parse_json_subtitles,
    ".json3": parse_json_subtitles,
    ".tsv": parse_tsv,
}


def detect_subtitle_format(content: str) -> str | None:
    """Recognize the caption format from the text itself; ``None`` when nothing matches."""

    head = _normalize_text(content).lstrip()[:65536]
    if not head:
        return None
    first_line = head.split("\n", 1)[0].strip()
    if first_line.upper().startswith("WEBVTT"):
        return ".vtt"
    if re.search(
        r"^\[Script Info\]|^\[V4\+? Styles\]|^\[Events\]|^Dialogue:\s*[^\n]*,\d+:\d{2}:\d{2}",
        head,
        re.IGNORECASE | re.MULTILINE,
    ):
        return ".ass"
    if re.search(r"<tt[\s>]", head, re.IGNORECASE):
        return ".ttml"
    if re.search(r"<sami[\s>]|<sync\s+start", head, re.IGNORECASE):
        return ".smi"
    if re.search(r"^\{\d+\}\{\d*\}", head, re.MULTILINE):
        return ".sub"
    if re.match(r"^\[INFORMATION\]", head, re.IGNORECASE):
        return ".sub"
    if re.search(r"^\[\d{1,3}:\d{2}(?:[.:]\d{1,3})?\]", head, re.MULTILINE):
        return ".lrc"
    if head[:1] in "[{" and _JSON_HINT_RE.search(head):
        return ".json"
    if re.search(r"\d{1,2}:\d{2}(?::\d{2})?(?:[,.]\d{1,3})?\s*-{1,3}>\s*\d", head):
        return ".srt"
    if re.search(r"^\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}\s*,\s*\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}\s*$", head, re.MULTILINE):
        return ".sbv"
    if "\t" in first_line:
        cells = [cell.strip().lower() for cell in first_line.split("\t")]
        if "start" in cells and ("text" in cells or "end" in cells):
            return ".tsv"
        if len(cells) >= 3:
            try:
                parse_timestamp_flexible(cells[0])
                parse_timestamp_flexible(cells[1])
                return ".tsv"
            except ValueError:
                pass
    return None


def parse_subtitle(content: str, extension: str | None = None) -> List[SubtitleCue]:
    """Parse captions, trusting the detected content format before the file extension."""

    normalized_extension = get_subtitle_extension(extension)
    detected = detect_subtitle_format(content)
    candidates: List[str] = []
    for candidate in (detected, normalized_extension):
        if candidate and candidate in _PARSERS and candidate not in candidates:
            candidates.append(candidate)
    if not candidates:
        raise ValueError(
            f"Unsupported caption format '{normalized_extension}'. Supported formats: "
            + ", ".join(SUPPORTED_SUBTITLE_EXTENSIONS)
        )
    if not _normalize_text(content).strip():
        return []

    errors: List[str] = []
    for candidate in candidates:
        try:
            cues = _PARSERS[candidate](content)
        except Exception as exc:  # a wrong guess must not hide the extension's own parser
            errors.append(f"{SUBTITLE_FORMAT_LABELS.get(candidate, candidate)}: {exc}")
            continue
        if cues:
            return cues
    if detected is not None and not errors:
        # A recognized caption file with no cues, such as a header-only WebVTT.
        return []
    detail = f" ({'; '.join(errors)})" if errors else ""
    raise ValueError(
        f"No caption cues were found in the {get_subtitle_format_label(normalized_extension)} content{detail}."
    )


def parse_subtitle_file(
    path: str | None,
    warning_callback: Callable[[str], None] | None = None,
) -> List[SubtitleCue]:
    if not path:
        return []
    return parse_subtitle(
        read_subtitle_file(path, warning_callback=warning_callback),
        get_subtitle_extension(path),
    )


def looks_like_subtitle_file(path: str | os.PathLike[str], *, max_bytes: int = 65536) -> bool:
    """Cheaply decide whether a file's content is a caption file (used for ambiguous extensions)."""

    try:
        with open(path, "rb") as handle:
            data = handle.read(max_bytes)
    except OSError:
        return False
    if not data.strip():
        return False
    if b"\x00" in data[:4096] and not data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return False
    text = ""
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if not text:
        text = data.decode("utf-8", errors="replace")
    return detect_subtitle_format(text) is not None


def subtitle_cues_to_text(cues: Sequence[SubtitleCue]) -> str:
    return "\n\n".join(cue.text for cue in cues)


def normalize_subtitle_text(text: str) -> str:
    return " ".join(part.strip() for part in text.splitlines() if part.strip())


def build_subtitle_render_units(cues: Sequence[SubtitleCue]) -> List[SubtitleRenderUnit]:
    if not cues:
        return []

    units: List[SubtitleRenderUnit] = []
    current_group: List[SubtitleCue] = []
    current_end_ms = -1

    def flush_group() -> None:
        if not current_group:
            return

        normalized_parts = [normalize_subtitle_text(cue.text) for cue in current_group]
        text = " ".join(part for part in normalized_parts if part).strip()
        units.append(
            SubtitleRenderUnit(
                index=len(units) + 1,
                start_ms=current_group[0].start_ms,
                end_ms=max(cue.end_ms for cue in current_group),
                text=text,
                cue_indices=tuple(cue.index for cue in current_group),
            )
        )

    for cue in cues:
        if not current_group:
            current_group = [cue]
            current_end_ms = cue.end_ms
            continue

        if cue.start_ms < current_end_ms:
            current_group.append(cue)
            current_end_ms = max(current_end_ms, cue.end_ms)
            continue

        flush_group()
        current_group = [cue]
        current_end_ms = cue.end_ms

    flush_group()
    return units


def ensure_audio_matrix(audio: np.ndarray) -> np.ndarray:
    matrix = np.asarray(audio)
    if matrix.ndim == 1:
        matrix = matrix[:, np.newaxis]
    elif matrix.ndim != 2:
        raise ValueError(f"Expected 1D or 2D audio array, got shape {matrix.shape}")

    if matrix.dtype != np.int16:
        matrix = matrix.astype(np.int16)

    return matrix


def write_pcm16_wav(audio: np.ndarray, sampling_rate: int, output_path: str) -> str:
    matrix = ensure_audio_matrix(audio)
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(matrix.shape[1])
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sampling_rate))
        wav_file.writeframes(matrix.tobytes())

    return output_path


def read_pcm16_wav(path: str) -> Tuple[int, np.ndarray]:
    with wave.open(path, "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sampling_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        frames = wav_file.readframes(frame_count)

    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, got sample width {sample_width} bytes: {path}")

    audio = np.frombuffer(frames, dtype=np.int16)
    if audio.size == 0:
        return sampling_rate, np.zeros((0, channels), dtype=np.int16)

    if audio.size % channels != 0:
        raise ValueError(f"PCM frame data is not divisible by channel count for {path}")

    return sampling_rate, audio.reshape(-1, channels).copy()


def pad_or_trim_audio_to_samples(audio: np.ndarray, target_samples: int) -> np.ndarray:
    matrix = ensure_audio_matrix(audio)
    target_samples = max(0, int(target_samples))
    current_samples = matrix.shape[0]

    if current_samples == target_samples:
        return matrix

    if target_samples == 0:
        return np.zeros((0, matrix.shape[1]), dtype=np.int16)

    if current_samples == 0:
        return np.zeros((target_samples, matrix.shape[1]), dtype=np.int16)

    if current_samples > target_samples:
        return matrix[:target_samples]

    padding = np.zeros((target_samples - current_samples, matrix.shape[1]), dtype=np.int16)
    return np.concatenate([matrix, padding], axis=0)


def samples_to_ms(sample_count: int, sampling_rate: int) -> int:
    return int(round(sample_count * 1000.0 / sampling_rate))


def ms_to_samples(value_ms: int, sampling_rate: int) -> int:
    return int(round(value_ms * sampling_rate / 1000.0))


def build_ffmpeg_atempo_chain(playback_rate: float) -> List[float]:
    rate = float(playback_rate)
    if rate <= 0:
        raise ValueError(f"Playback rate must be positive, got {playback_rate}")

    chain: List[float] = []
    while rate < 0.5:
        chain.append(0.5)
        rate /= 0.5
    while rate > 2.0:
        chain.append(2.0)
        rate /= 2.0

    if not chain or abs(rate - 1.0) > 1e-9:
        chain.append(rate)

    return chain or [1.0]


def retime_audio_file_with_ffmpeg(
    input_path: str,
    output_path: str,
    target_duration_ms: int,
) -> dict:
    sampling_rate, source_audio = read_pcm16_wav(input_path)
    source_audio = ensure_audio_matrix(source_audio)
    target_duration_ms = max(0, int(target_duration_ms))
    target_samples = ms_to_samples(target_duration_ms, sampling_rate)
    source_duration_ms = samples_to_ms(source_audio.shape[0], sampling_rate)

    info = {
        "method": "copy",
        "source_duration_ms": source_duration_ms,
        "target_duration_ms": target_duration_ms,
        "delta_ms_before_fit": int(source_duration_ms - target_duration_ms),
        "stretch_rate": 1.0,
        "output_duration_ms": target_duration_ms,
    }

    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    if source_audio.shape[0] == target_samples:
        shutil.copyfile(input_path, output_path)
        return info

    if target_samples == 0:
        info["method"] = "target_silence"
        write_pcm16_wav(np.zeros((0, source_audio.shape[1]), dtype=np.int16), sampling_rate, output_path)
        info["output_duration_ms"] = 0
        return info

    if source_audio.shape[0] == 0:
        info["method"] = "source_silence"
        write_pcm16_wav(
            np.zeros((target_samples, source_audio.shape[1]), dtype=np.int16),
            sampling_rate,
            output_path,
        )
        return info

    stretch_rate = source_audio.shape[0] / float(target_samples)
    atempo_chain = build_ffmpeg_atempo_chain(stretch_rate)
    target_seconds = target_samples / float(sampling_rate)
    filters = [f"atempo={factor:.10f}" for factor in atempo_chain]
    filters.append(f"apad=whole_dur={target_seconds:.10f}")
    filters.append(f"atrim=duration={target_seconds:.10f}")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-filter:a",
        ",".join(filters),
        "-ar",
        str(sampling_rate),
        "-ac",
        str(source_audio.shape[1]),
        "-acodec",
        "pcm_s16le",
        "-f",
        "wav",
        output_path,
        "-loglevel",
        "error",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg retime failed for {os.path.basename(output_path)}: {result.stderr.strip()}")

    output_sampling_rate, output_audio = read_pcm16_wav(output_path)
    exact_audio = pad_or_trim_audio_to_samples(output_audio, target_samples)
    if exact_audio.shape[0] != output_audio.shape[0]:
        write_pcm16_wav(exact_audio, output_sampling_rate, output_path)
    else:
        exact_audio = ensure_audio_matrix(output_audio)

    info["method"] = "ffmpeg_atempo"
    info["stretch_rate"] = float(stretch_rate)
    info["output_duration_ms"] = samples_to_ms(exact_audio.shape[0], output_sampling_rate)
    return info


def fit_audio_to_duration(
    audio: np.ndarray,
    sampling_rate: int,
    target_duration_ms: int,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    matrix = ensure_audio_matrix(audio)
    source_duration_ms = samples_to_ms(matrix.shape[0], sampling_rate)
    target_samples = max(0, ms_to_samples(target_duration_ms, sampling_rate))
    info = {
        "method": "none",
        "source_duration_ms": source_duration_ms,
        "target_duration_ms": int(target_duration_ms),
        "delta_ms_before_fit": int(source_duration_ms - target_duration_ms),
        "stretch_rate": 1.0,
    }

    if matrix.shape[0] == target_samples:
        return (matrix, info) if return_info else matrix

    if target_samples == 0:
        info["method"] = "target_silence"
        fitted = np.zeros((0, matrix.shape[1]), dtype=np.int16)
        return (fitted, info) if return_info else fitted

    if matrix.shape[0] == 0:
        info["method"] = "source_silence"
        fitted = np.zeros((target_samples, matrix.shape[1]), dtype=np.int16)
        return (fitted, info) if return_info else fitted

    tolerance_ms = min(250, max(80, int(round(target_duration_ms * 0.01))))
    tolerance_samples = ms_to_samples(tolerance_ms, sampling_rate)
    delta_samples = target_samples - matrix.shape[0]

    if abs(delta_samples) <= tolerance_samples:
        if delta_samples > 0:
            info["method"] = "pad_silence"
            padding = np.zeros((delta_samples, matrix.shape[1]), dtype=np.int16)
            fitted = np.concatenate([matrix, padding], axis=0)
        else:
            info["method"] = "trim_tail"
            fitted = matrix[:target_samples]
        return (fitted, info) if return_info else fitted

    stretch_rate = matrix.shape[0] / float(target_samples)
    info["method"] = "time_stretch"
    info["stretch_rate"] = float(stretch_rate)
    stretched_channels: List[np.ndarray] = []

    for channel_idx in range(matrix.shape[1]):
        samples = matrix[:, channel_idx].astype(np.float32) / 32768.0
        expected_samples = max(1, int(round(samples.shape[0] / stretch_rate)))
        if samples.shape[0] < 2048:
            stretched = resample(samples, expected_samples)
        else:
            stretched = librosa.effects.time_stretch(samples, rate=stretch_rate)
            if not np.isfinite(stretched).all():
                stretched = resample(samples, expected_samples)
        stretched_channels.append(stretched)

    max_len = max(channel.shape[0] for channel in stretched_channels)
    stretched_matrix = np.zeros((max_len, len(stretched_channels)), dtype=np.float32)
    for channel_idx, channel in enumerate(stretched_channels):
        stretched_matrix[: channel.shape[0], channel_idx] = channel

    if stretched_matrix.shape[0] > target_samples:
        stretched_matrix = stretched_matrix[:target_samples]
    elif stretched_matrix.shape[0] < target_samples:
        padding = np.zeros((target_samples - stretched_matrix.shape[0], stretched_matrix.shape[1]), dtype=np.float32)
        stretched_matrix = np.concatenate([stretched_matrix, padding], axis=0)

    stretched_matrix = np.clip(stretched_matrix, -1.0, 1.0)
    fitted = (stretched_matrix * 32767.0).astype(np.int16)
    return (fitted, info) if return_info else fitted


def assemble_subtitle_audio(
    rendered_cues: Sequence[Tuple[SubtitleCue, np.ndarray]],
    sampling_rate: int,
) -> Tuple[np.ndarray, List[dict]]:
    pieces: List[np.ndarray] = []
    issues: List[dict] = []
    cursor_samples = 0
    channel_count = None

    for cue, raw_audio in rendered_cues:
        audio = ensure_audio_matrix(raw_audio)
        if channel_count is None:
            channel_count = audio.shape[1]
        elif audio.shape[1] != channel_count:
            raise ValueError("All rendered subtitle cues must use the same channel count")

        cue_start_samples = ms_to_samples(cue.start_ms, sampling_rate)
        if cue_start_samples > cursor_samples:
            silence = np.zeros((cue_start_samples - cursor_samples, channel_count), dtype=np.int16)
            pieces.append(silence)
            cursor_samples = cue_start_samples
        elif cue_start_samples < cursor_samples:
            issues.append(
                {
                    "cue_index": cue.index,
                    "type": "late_start",
                    "delta_ms": samples_to_ms(cursor_samples - cue_start_samples, sampling_rate),
                }
            )

        if audio.shape[0] > 0:
            pieces.append(audio)
            cursor_samples += audio.shape[0]

        audio_duration_ms = samples_to_ms(audio.shape[0], sampling_rate)
        slot_overrun_ms = audio_duration_ms - cue.duration_ms
        if slot_overrun_ms > 0:
            issues.append(
                {
                    "cue_index": cue.index,
                    "type": "slot_overrun",
                    "delta_ms": slot_overrun_ms,
                }
            )

    if not pieces:
        channel_count = 1 if channel_count is None else channel_count
        return np.zeros((0, channel_count), dtype=np.int16), issues

    return np.concatenate(pieces, axis=0), issues
