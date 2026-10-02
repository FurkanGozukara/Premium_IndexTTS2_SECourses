"""Word timings of generated speech, and subtitles built from them.

Neither speech model reports when it says each word: IndexTTS generates audio codes one after another and
OmniVoice unmasks a whole utterance at once. The timings therefore come from the finished audio: Whisper's
word timestamps aligned to the words the user wrote (``align_caption_words``, the dataset-preparation
aligner). Recognizer spellings never replace the written words; words Whisper missed share the time between
their recognized neighbours. Files are written next to the audio: ``<name>.words.json`` (every word with its
start and end), ``<name>.srt`` and ``<name>.vtt`` (subtitle cues of at most two 42-character lines).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Sequence

_PAUSE_RE = re.compile(r"\[\s*pause\s*:[^\]]*\]|<\s*pause\s*=[^>]*>", re.IGNORECASE)
_ANNOTATION_RE = re.compile(r"<([^|>\n]+)\|([^>\n]+)>")
_BRACKET_RE = re.compile(r"\[[^\[\]\n]*\]")  # OmniVoice tags ([laughter]) and CMU phones
_CJK_RE = re.compile(r"[぀-ヿ㐀-鿿가-힣]")
_SENTENCE_END = (".", "!", "?", "。", "！", "？", "…")
_CLAUSE_END = (",", ";", ":", "，", "；", "：", "、")
_CLOSERS = "\"'”’)]»"
LINE_CHARS = 42
CJK_LINE_CHARS = 18
MAX_CUE_S = 7.0
MIN_CUE_S = 0.8
MAX_GAP_S = 1.2
ALIGNER = "openai/whisper-large-v3-turbo"


def written_words(text: str) -> list[str]:
    """The words as they should appear in subtitles: tags removed, readings back to the written word."""

    value = _PAUSE_RE.sub(" ", str(text or ""))
    value = _ANNOTATION_RE.sub(lambda match: match.group(1), value)
    value = _BRACKET_RE.sub(" ", value)
    words: list[str] = []
    for piece in value.split():
        if not _CJK_RE.search(piece):
            words.append(piece)
            continue
        # CJK text has no spaces: every character is a word; punctuation stays on the character before it.
        for char in piece:
            if words and not _CJK_RE.match(char) and not char.isalnum():
                words[-1] += char
            else:
                words.append(char)
    return [word for word in words if any(char.isalnum() for char in word)]


def whisper_language(language: str | None) -> str | None:
    """Whisper's code for an app language (``EN``, ``ZHEN``, OmniVoice codes); ``None`` lets Whisper detect it."""

    code = str(language or "").strip().lower()
    if code in {"", "auto"}:
        return None
    code = {"zhen": "zh", "zh-cn": "zh", "cmn": "zh", "yue": "yue"}.get(code, code)
    try:
        from transformers.models.whisper.tokenization_whisper import LANGUAGES
    except ImportError:  # pragma: no cover - transformers is a dependency
        return code
    return code if code in LANGUAGES else None


def align_words(text: str, recognized: Sequence[Any], duration_s: float) -> dict[str, Any]:
    """Written words with start and end seconds; ``recognized`` are Whisper words (``text``/``start_s``/``end_s``)."""

    from indextts.training.whisper_asr import align_caption_words

    words = written_words(text)
    total_ms = max(20, int(round(float(duration_s) * 1000)))
    caption, position = [], 0
    for word in words:
        caption.append({"text": word, "cue_index": 0, "cue_start_ms": 0, "cue_end_ms": total_ms,
                        "char_start": position, "char_end": position + len(word)})
        position += len(word) + 1
    alignment = align_caption_words(caption, list(recognized))
    rows = [{"text": item.text, "start_s": round(min(item.start_s, duration_s), 3),
             "end_s": round(min(max(item.end_s, item.start_s + 0.01), duration_s), 3), "matched": bool(item.matched)}
            for item in alignment.words]
    return {"words": rows, "matched_words": alignment.matched_words, "total_words": alignment.total_words,
            "coverage": round(alignment.coverage, 4)}


def _joined(words: Sequence[str]) -> str:
    text = ""
    for word in words:
        if text and not (_CJK_RE.match(word[0]) and _CJK_RE.search(text[-1])):
            text += " "
        text += word
    return text


def _split_points(text: str, limit: int) -> list[int]:
    """Spaces where the text breaks into two lines of at most ``limit`` characters."""

    return [index for index, char in enumerate(text) if char == " " and index <= limit and len(text) - index - 1 <= limit]


def _fits(text: str, limit: int) -> bool:
    return len(text) <= limit or bool(_split_points(text, limit))


def _wrapped(text: str, limit: int) -> str:
    """One line, or two lines split near the middle (after punctuation when close), each within the limit."""

    if len(text) <= limit:
        return text
    points = _split_points(text, limit) or [index for index, char in enumerate(text) if char == " "]
    if not points:
        return text
    middle = len(text) / 2
    best = min(points, key=lambda index: abs(index - middle) - (6 if text[index - 1] in _CLAUSE_END + _SENTENCE_END else 0))
    return text[:best] + "\n" + text[best + 1:]


def subtitle_cues(words: Sequence[dict[str, Any]], *, line_chars: int | None = None,
                  max_cue_s: float = MAX_CUE_S, max_gap_s: float = MAX_GAP_S) -> list[dict[str, Any]]:
    """Group timed words into cues of at most two lines, ending at sentences, long pauses or the length limit."""

    if not words:
        return []
    cjk = sum(1 for word in words if _CJK_RE.search(word["text"])) > len(words) / 2
    limit = int(line_chars or (CJK_LINE_CHARS if cjk else LINE_CHARS))
    cues: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []

    def close() -> None:
        if current:
            cues.append({"start_s": current[0]["start_s"], "end_s": current[-1]["end_s"],
                         "text": _wrapped(_joined([word["text"] for word in current]), limit)})
            current.clear()

    for word in words:
        if current:
            text = _joined([item["text"] for item in current] + [word["text"]])
            if (not _fits(text, limit) or word["end_s"] - current[0]["start_s"] > max_cue_s
                    or word["start_s"] - current[-1]["end_s"] > max_gap_s):
                close()
        current.append(word)
        tail = word["text"].rstrip(_CLOSERS)
        length = len(_joined([item["text"] for item in current]))
        if tail.endswith(_SENTENCE_END) or (tail.endswith(_CLAUSE_END) and length > limit):
            close()
    close()
    # Short cues stay on screen a little longer when the next cue leaves room.
    for index, cue in enumerate(cues):
        following = cues[index + 1]["start_s"] if index + 1 < len(cues) else None
        wanted = cue["start_s"] + MIN_CUE_S
        if cue["end_s"] < wanted:
            cue["end_s"] = round(min(wanted, following) if following is not None else wanted, 3)
    return cues


def _clock(seconds: float, separator: str) -> str:
    milliseconds = max(0, int(round(float(seconds) * 1000)))
    hours, rest = divmod(milliseconds, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def srt_text(cues: Sequence[dict[str, Any]]) -> str:
    return "".join(f"{index}\n{_clock(cue['start_s'], ',')} --> {_clock(cue['end_s'], ',')}\n{cue['text']}\n\n"
                   for index, cue in enumerate(cues, 1))


def vtt_text(cues: Sequence[dict[str, Any]]) -> str:
    return "WEBVTT\n\n" + "".join(f"{_clock(cue['start_s'], '.')} --> {_clock(cue['end_s'], '.')}\n{cue['text']}\n\n"
                                  for cue in cues)


def timestamp_paths(audio_path: str | os.PathLike[str]) -> dict[str, Path]:
    stem = Path(audio_path).with_suffix("")
    return {"json": stem.with_name(stem.name + ".words.json"), "srt": stem.with_suffix(".srt"),
            "vtt": stem.with_suffix(".vtt")}


def _write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_speech_timestamps(audio_path: str | os.PathLike[str], text: str, *, language: str | None = None,
                            device: str = "cuda:0", progress: Callable[..., Any] | None = None,
                            transcriber: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Transcribe ``audio_path``, align the written ``text`` and write the JSON, SRT and VTT files beside it."""

    import soundfile as sf

    from indextts.training.whisper_asr import transcribe, whisper_device_for_free_vram

    info = sf.info(str(audio_path))
    duration = info.frames / float(info.samplerate) if info.samplerate else 0.0
    if str(device).startswith("cuda"):
        try:
            import torch

            free = torch.cuda.mem_get_info(torch.device(device))[0] / 1024**3 if torch.cuda.is_available() else None
        except Exception:
            free = None
        # Whisper large-v3-turbo needs about 2 GB beside the loaded speech model; a full card falls back to the CPU.
        device = whisper_device_for_free_vram(device, free, required_gb=2.5)
    code = whisper_language(language)
    transcript = (transcriber or transcribe)(str(audio_path), language=code, device=device, progress_cb=progress)
    aligned = align_words(text, transcript.words, duration)
    cues = subtitle_cues(aligned["words"])
    paths = timestamp_paths(audio_path)
    payload = {"version": 1, "audio": str(Path(audio_path).resolve()), "duration_s": round(duration, 3),
               "language": code or "auto", "aligner": ALIGNER, **aligned, "cues": cues,
               "recognized_text": getattr(transcript, "text", "")}
    _write(paths["json"], json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    _write(paths["srt"], srt_text(cues))
    _write(paths["vtt"], vtt_text(cues))
    return {"paths": {key: str(path) for key, path in paths.items()}, "coverage": aligned["coverage"],
            "words": aligned["total_words"], "cues": len(cues)}


__all__ = ["align_words", "srt_text", "subtitle_cues", "timestamp_paths", "vtt_text", "whisper_language",
           "write_speech_timestamps", "written_words"]
