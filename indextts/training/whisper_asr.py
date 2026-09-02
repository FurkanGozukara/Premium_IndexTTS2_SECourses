from __future__ import annotations

from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
import gc
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence
import unicodedata

import numpy as np
import soundfile as sf

from .subtitles import CaptionWord, Segment


DEFAULT_WHISPER_MODEL = "openai/whisper-large-v3-turbo"
_MODEL_ALIASES = {
    "large-v3-turbo": "openai/whisper-large-v3-turbo",
    "whisper-large-v3-turbo": "openai/whisper-large-v3-turbo",
    "large-v3": "openai/whisper-large-v3",
    "whisper-large-v3": "openai/whisper-large-v3",
    "medium": "openai/whisper-medium",
    "whisper-medium": "openai/whisper-medium",
    "small": "openai/whisper-small",
    "whisper-small": "openai/whisper-small",
}


@dataclass(frozen=True)
class Word:
    text: str
    start_s: float
    end_s: float


@dataclass
class TranscriptSegment:
    text: str
    start_s: float
    end_s: float
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    words: list[Word]
    segments: list[TranscriptSegment]
    text: str


@dataclass(frozen=True)
class AlignedCaptionWord:
    """Caption text paired with a precise or interpolated Whisper timestamp."""

    text: str
    start_s: float
    end_s: float
    char_start: int
    char_end: int
    cue_index: int
    matched: bool


@dataclass(frozen=True)
class CaptionWordAlignment:
    words: tuple[AlignedCaptionWord, ...]
    matched_words: int
    total_words: int

    @property
    def coverage(self) -> float:
        return self.matched_words / self.total_words if self.total_words else 0.0


def _canonical_model_name(model_name: str) -> str:
    value = str(model_name or DEFAULT_WHISPER_MODEL).strip()
    return _MODEL_ALIASES.get(value.casefold(), value)


def _model_directory(model_name: str) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    leaf = _canonical_model_name(model_name).split("/")[-1]
    return repo_root / "models" / "hf_cache" / "whisper" / leaf


def _ensure_model(model_name: str) -> Path:
    canonical = _canonical_model_name(model_name)
    destination = _model_directory(canonical)
    weight_files = ("model.safetensors", "pytorch_model.bin")
    if (destination / "config.json").is_file() and any(
        (destination / name).is_file() for name in weight_files
    ):
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Whisper model {canonical} to {destination} ...", flush=True)
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=canonical, local_dir=destination, max_workers=8)
    if not (destination / "config.json").is_file():
        raise RuntimeError(f"Whisper download did not produce config.json in {destination}")
    print(f"Whisper model ready: {destination}", flush=True)
    return destination


def _notify_progress(
    callback: Any,
    completed: int,
    total: int,
    desc: str,
) -> None:
    if callback is None:
        return
    if hasattr(callback, "update"):
        callback.update(completed, total, desc)
        return
    try:
        callback(completed, total, desc)
    except TypeError:
        callback(completed / max(1, total), desc)


def _read_input(audio_path_or_array: str | Path | np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    if isinstance(audio_path_or_array, (str, Path)):
        audio, actual_sr = sf.read(str(audio_path_or_array), dtype="float32", always_2d=False)
        sr = int(actual_sr)
    else:
        audio = np.asarray(audio_path_or_array, dtype=np.float32)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"Expected a mono waveform or channel-last matrix, got {audio.shape}")
    return np.ascontiguousarray(audio, dtype=np.float32), int(sr)


def _result_words(result: dict[str, Any], offset_s: float) -> list[Word]:
    words: list[Word] = []
    chunks = result.get("chunks") or []
    for chunk in chunks:
        timestamp = chunk.get("timestamp") or chunk.get("timestamps")
        if not timestamp or timestamp[0] is None:
            continue
        start = max(0.0, float(timestamp[0]) + offset_s)
        raw_end = timestamp[1]
        end = float(raw_end) + offset_s if raw_end is not None else start + 0.2
        if end <= start:
            end = start + 0.02
        text = str(chunk.get("text") or "").strip()
        if text:
            words.append(Word(text=text, start_s=start, end_s=end))
    return words


def _join_words(words: Sequence[Word]) -> str:
    text = " ".join(word.text.strip() for word in words if word.text.strip())
    text = re.sub(r"\s+([,.;:!?。！？])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


_SMALL_NUMBERS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


def _integer_words(value: int) -> list[str]:
    if value < 20:
        return [_SMALL_NUMBERS[value]]
    if value < 100:
        tens, remainder = divmod(value, 10)
        return [_TENS[tens], *(_integer_words(remainder) if remainder else [])]
    if value < 1000:
        hundreds, remainder = divmod(value, 100)
        return [_SMALL_NUMBERS[hundreds], "hundred", *(_integer_words(remainder) if remainder else [])]
    if value < 1_000_000:
        thousands, remainder = divmod(value, 1000)
        return [*_integer_words(thousands), "thousand", *(_integer_words(remainder) if remainder else [])]
    return [_SMALL_NUMBERS[int(digit)] for digit in str(value)]


def normalize_alignment_token(text: str) -> tuple[str, ...]:
    """Normalize a visible word into comparable pieces for caption/ASR alignment."""

    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    pieces: list[str] = []
    for raw in re.findall(r"[^\W\d_]+|\d+", value, flags=re.UNICODE):
        if raw.isdigit():
            # Expanding common, reasonably small integers cheaply handles
            # caption "24" versus ASR "twenty four" without a dependency.
            number = int(raw)
            pieces.extend(_integer_words(number) if len(raw) <= 6 else list(raw))
        else:
            folded = "".join(
                char for char in unicodedata.normalize("NFKD", raw) if not unicodedata.combining(char)
            )
            if folded:
                pieces.append(folded)
    return tuple(pieces)


def _caption_value(word: CaptionWord | Mapping[str, Any], name: str) -> Any:
    if isinstance(word, Mapping):
        return word[name]
    return getattr(word, name)


def align_caption_words(
    caption_words: Sequence[CaptionWord | Mapping[str, Any]],
    whisper_words: Sequence[Word | Mapping[str, Any]],
) -> CaptionWordAlignment:
    """Align caption words to noisy Whisper words and fill timing gaps.

    Exact normalized token runs are anchored with ``SequenceMatcher``.
    Unmatched caption words between anchors share the available time; leading,
    trailing, and otherwise unusable gaps fall back to their cue timing.
    """

    if not caption_words:
        return CaptionWordAlignment(words=(), matched_words=0, total_words=0)

    caption_tokens: list[str] = []
    caption_owner: list[int] = []
    for index, word in enumerate(caption_words):
        for token in normalize_alignment_token(str(_caption_value(word, "text"))):
            caption_tokens.append(token)
            caption_owner.append(index)

    whisper_tokens: list[str] = []
    whisper_owner: list[int] = []
    for index, word in enumerate(whisper_words):
        for token in normalize_alignment_token(_word_text(word)):
            whisper_tokens.append(token)
            whisper_owner.append(index)

    anchors: dict[int, set[int]] = {}
    matcher = SequenceMatcher(None, caption_tokens, whisper_tokens, autojunk=False)
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            caption_index = caption_owner[block.a + offset]
            whisper_index = whisper_owner[block.b + offset]
            anchors.setdefault(caption_index, set()).add(whisper_index)

    coarse: list[tuple[float, float]] = [(0.0, 0.02)] * len(caption_words)
    cue_groups: dict[tuple[int, int, int], list[int]] = {}
    for index, word in enumerate(caption_words):
        key = (
            int(_caption_value(word, "cue_index")),
            int(_caption_value(word, "cue_start_ms")),
            int(_caption_value(word, "cue_end_ms")),
        )
        cue_groups.setdefault(key, []).append(index)
    for (_, cue_start_ms, cue_end_ms), indices in cue_groups.items():
        cue_start = max(0.0, cue_start_ms / 1000.0)
        cue_end = max(cue_start + 0.02, cue_end_ms / 1000.0)
        step = (cue_end - cue_start) / max(1, len(indices))
        for position, index in enumerate(indices):
            start = cue_start + position * step
            end = cue_start + (position + 1) * step
            coarse[index] = (start, max(start + 0.01, end))

    starts: list[float | None] = [None] * len(caption_words)
    ends: list[float | None] = [None] * len(caption_words)
    matched = [False] * len(caption_words)
    for caption_index, whisper_indices in anchors.items():
        selected = [whisper_words[index] for index in sorted(whisper_indices)]
        if not selected:
            continue
        starts[caption_index] = min(_word_times(word)[0] for word in selected)
        ends[caption_index] = max(_word_times(word)[1] for word in selected)
        matched[caption_index] = True

    cursor = 0
    while cursor < len(caption_words):
        if matched[cursor]:
            cursor += 1
            continue
        run_start = cursor
        while cursor < len(caption_words) and not matched[cursor]:
            cursor += 1
        run_end = cursor
        left = run_start - 1 if run_start else None
        right = run_end if run_end < len(caption_words) else None
        left_end = ends[left] if left is not None else None
        right_start = starts[right] if right is not None else None
        if left_end is not None and right_start is not None and right_start > left_end:
            step = (right_start - left_end) / (run_end - run_start)
            for offset, index in enumerate(range(run_start, run_end)):
                starts[index] = left_end + offset * step
                ends[index] = left_end + (offset + 1) * step
        else:
            for index in range(run_start, run_end):
                starts[index], ends[index] = coarse[index]

    resolved_starts = [max(0.0, float(value or 0.0)) for value in starts]
    resolved_ends = [max(resolved_starts[index] + 0.01, float(ends[index] or 0.0)) for index in range(len(ends))]
    for index in range(1, len(resolved_starts)):
        if resolved_starts[index] < resolved_ends[index - 1]:
            boundary = (resolved_starts[index] + resolved_ends[index - 1]) / 2.0
            boundary = max(resolved_starts[index - 1] + 0.005, boundary)
            boundary = min(resolved_ends[index] - 0.005, boundary)
            resolved_ends[index - 1] = max(resolved_starts[index - 1] + 0.005, boundary)
            resolved_starts[index] = min(resolved_ends[index] - 0.005, boundary)

    aligned = tuple(
        AlignedCaptionWord(
            text=str(_caption_value(word, "text")),
            start_s=resolved_starts[index],
            end_s=resolved_ends[index],
            char_start=int(_caption_value(word, "char_start")),
            char_end=int(_caption_value(word, "char_end")),
            cue_index=int(_caption_value(word, "cue_index")),
            matched=matched[index],
        )
        for index, word in enumerate(caption_words)
    )
    return CaptionWordAlignment(
        words=aligned,
        matched_words=sum(matched),
        total_words=len(caption_words),
    )


def _transcript_from_words(words: Sequence[Word]) -> Transcript:
    from .segmenter import build_segments_from_words

    accepted = list(words)
    base_segments = build_segments_from_words(accepted, 8.0, 15.0, 1.5, 700)
    segments = [
        TranscriptSegment(
            text=segment.text,
            start_s=segment.start_ms / 1000.0,
            end_s=segment.end_ms / 1000.0,
            words=[
                Word(
                    text=str(item["text"]),
                    start_s=float(item["start_s"]),
                    end_s=float(item["end_s"]),
                )
                for item in segment.word_timestamps
            ],
        )
        for segment in base_segments
    ]
    return Transcript(words=accepted, segments=segments, text=_join_words(accepted))


def save_word_timestamps(
    path: str | Path,
    transcript: Transcript,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically persist the small, reusable Whisper word-timing cache."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        **dict(metadata or {}),
        "text": transcript.text,
        "words": [
            {"text": word.text, "start_s": round(word.start_s, 6), "end_s": round(word.end_s, 6)}
            for word in transcript.words
        ],
    }
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return destination


def load_word_timestamps(path: str | Path) -> Transcript:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    raw_words = payload.get("words") if isinstance(payload, dict) else None
    if not isinstance(raw_words, list):
        raise ValueError(f"Whisper word cache has no words list: {path}")
    words = [
        Word(text=str(item["text"]), start_s=float(item["start_s"]), end_s=float(item["end_s"]))
        for item in raw_words
        if isinstance(item, dict) and item.get("text") is not None
    ]
    return _transcript_from_words(words)


def transcribe(
    audio_path_or_array: str | Path | np.ndarray,
    sr: int = 24000,
    language: str = "EN",
    model_name: str = DEFAULT_WHISPER_MODEL,
    device: str = "cuda:0",
    progress_cb: Callable[..., Any] | Any | None = None,
) -> Transcript:
    """Transcribe with lazy model loading and overlap-safe, word-level chunking."""

    import torch
    from transformers import pipeline

    waveform, sample_rate = _read_input(audio_path_or_array, sr)
    if waveform.size == 0:
        return Transcript(words=[], segments=[], text="")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested for Whisper but is unavailable: {device}")

    model_path = _ensure_model(model_name)
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    pipe = None
    try:
        print(f"Loading Whisper {_canonical_model_name(model_name)} on {device} ({dtype}) ...", flush=True)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=str(model_path),
            device=device,
            dtype=dtype,
            chunk_length_s=30,
        )
        total_s = waveform.size / float(sample_rate)
        manual_chunk_s = 120.0
        overlap_s = 5.0
        starts: list[float] = []
        cursor = 0.0
        while cursor < total_s:
            starts.append(cursor)
            if cursor + manual_chunk_s >= total_s:
                break
            cursor += manual_chunk_s - overlap_s

        accepted: list[Word] = []
        for index, start_s in enumerate(starts):
            end_s = min(total_s, start_s + manual_chunk_s)
            start_i = int(round(start_s * sample_rate))
            end_i = int(round(end_s * sample_rate))
            _notify_progress(
                progress_cb,
                index,
                len(starts),
                f"Whisper chunk {index + 1}/{len(starts)} ({start_s:.0f}-{end_s:.0f}s)",
            )
            result = pipe(
                {"array": waveform[start_i:end_i], "sampling_rate": sample_rate},
                return_timestamps="word",
                generate_kwargs={"language": str(language).lower(), "task": "transcribe"},
            )
            chunk_words = _result_words(result, start_s)
            accept_from = start_s if index == 0 else start_s + overlap_s / 2.0
            accept_to = end_s if index == len(starts) - 1 else end_s - overlap_s / 2.0
            for word in chunk_words:
                midpoint = (word.start_s + word.end_s) / 2.0
                if accept_from <= midpoint < accept_to + 1e-6:
                    if accepted and word.start_s < accepted[-1].end_s - 0.25:
                        if word.text.casefold() == accepted[-1].text.casefold():
                            continue
                    accepted.append(word)
        accepted.sort(key=lambda word: (word.start_s, word.end_s))
        _notify_progress(progress_cb, len(starts), len(starts), "Whisper transcription complete")

        return _transcript_from_words(accepted)
    finally:
        if pipe is not None:
            del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _word_times(word: Any) -> tuple[float, float]:
    if isinstance(word, dict):
        return float(word["start_s"]), float(word["end_s"])
    return float(word.start_s), float(word.end_s)


def _word_text(word: Any) -> str:
    return str(word.get("text", "") if isinstance(word, dict) else word.text)


def align_segments_with_words(
    segments: Sequence[Segment],
    words: Sequence[Word | dict[str, Any]],
) -> list[Segment]:
    """Refine subtitle edges to nearby Whisper word starts and ends."""

    if not segments or not words:
        return [replace(segment) for segment in segments]
    ordered_words = sorted(words, key=lambda word: _word_times(word)[0])
    aligned: list[Segment] = []
    for segment in segments:
        start_s = segment.start_ms / 1000.0
        end_s = segment.end_ms / 1000.0
        inside = [
            word
            for word in ordered_words
            if start_s - 0.75 <= sum(_word_times(word)) / 2.0 <= end_s + 0.75
        ]
        if not inside:
            aligned.append(replace(segment))
            continue
        start_word = min(inside, key=lambda word: abs(_word_times(word)[0] - start_s))
        end_word = min(inside, key=lambda word: abs(_word_times(word)[1] - end_s))
        new_start = _word_times(start_word)[0]
        new_end = _word_times(end_word)[1]
        if new_end <= new_start:
            new_start, new_end = start_s, end_s
        selected = [
            word
            for word in ordered_words
            if new_start - 1e-6 <= sum(_word_times(word)) / 2.0 <= new_end + 1e-6
        ]
        aligned.append(
            replace(
                segment,
                start_ms=max(0, int(round(new_start * 1000.0))),
                end_ms=max(1, int(round(new_end * 1000.0))),
                word_timestamps=[
                    {
                        "text": _word_text(word),
                        "start_s": _word_times(word)[0],
                        "end_s": _word_times(word)[1],
                    }
                    for word in selected
                ],
            )
        )

    for index in range(1, len(aligned)):
        previous = aligned[index - 1]
        current = aligned[index]
        if current.start_ms < previous.end_ms:
            midpoint = (current.start_ms + previous.end_ms) // 2
            aligned[index - 1] = replace(previous, end_ms=max(previous.start_ms + 1, midpoint))
            aligned[index] = replace(current, start_ms=min(current.end_ms - 1, midpoint))
    return aligned


__all__ = [
    "AlignedCaptionWord",
    "CaptionWordAlignment",
    "DEFAULT_WHISPER_MODEL",
    "Transcript",
    "TranscriptSegment",
    "Word",
    "align_caption_words",
    "align_segments_with_words",
    "load_word_timestamps",
    "normalize_alignment_token",
    "save_word_timestamps",
    "transcribe",
]
