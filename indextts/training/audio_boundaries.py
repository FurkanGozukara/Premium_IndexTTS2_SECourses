"""Recover complete caption sentences using acoustic pauses before packing."""
from __future__ import annotations

from typing import Any, Callable, Sequence, TYPE_CHECKING
import math

import numpy as np

from .segmenter import split_caption_sentences
from .subtitles import CaptionTranscript, Segment
from .media import measure_loudness_lufs

if TYPE_CHECKING:
    from .dataset_prep import DatasetPrepConfig


def _get(word: Any, key: str, default: Any = None) -> Any:
    return word.get(key, default) if isinstance(word, dict) else getattr(word, key, default)


def _ms(word: Any, key: str) -> int:
    return round(float(_get(word, key)) * 1000)


def _pause(
    energy: np.ndarray, low: int, high: int, preferred: int, minimum_ms: int,
    threshold_dbfs: float,
) -> tuple[int, int] | None:
    hop = 10
    first = max(0, math.ceil(low / hop))
    stop = min(len(energy), math.floor(high / hop))
    window = energy[first:stop]
    minimum = max(1, math.ceil(minimum_ms / hop))
    if len(window) < minimum:
        return None
    # The relative limit prevents quiet recordings being classified entirely
    # as silence. The absolute limit keeps background audio out of boundaries.
    context = energy[max(0, first - 50):min(len(energy), stop + 50)]
    threshold = min(10 ** (threshold_dbfs / 20), float(context.max(initial=0)) * 0.1)
    quiet = np.isfinite(window) & (window <= threshold)
    edges = np.diff(np.pad(quiet.astype(np.int8), (1, 1)))
    runs = [((first + start) * hop, (first + end) * hop)
            for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
            if end - start >= minimum]
    return min(runs, key=lambda pair: abs((pair[0] + pair[1]) / 2 - preferred)) if runs else None


def build_safe_sentence_segments(
    caption: CaptionTranscript,
    words: Sequence[Any],
    energy: np.ndarray,
    config: DatasetPrepConfig,
    *,
    audio: np.ndarray | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> tuple[list[Segment], list[dict[str, Any]]]:
    """Repack complete sentences across unsafe cuts, using original audio.

    A boundary with no sustained pause cannot become an output edge. Dynamic
    programming maximizes retained caption words, then favors the target clip
    duration. It can move a sentence to an adjacent clip or merge sentences
    across a bad boundary; every retained word appears exactly once.
    """
    spans = split_caption_sentences(caption)
    if not spans or len(words) != len(caption.words):
        return [], []
    edge_ms = max(10, math.ceil(config.min_edge_silence_ms / 10) * 10)
    pad = max(edge_ms, config.pad_ms)
    radius = config.snap_window_ms + pad
    media_end = len(energy) * 10
    boundary_cache: dict[tuple[int, float], tuple[int | None, int | None]] = {}

    def boundary(index: int, gain_db: float) -> tuple[int | None, int | None]:
        # Candidate clips can receive different gains. Check their source
        # pauses at the level they will have after loudness normalization.
        key = (index, math.ceil(gain_db * 10) / 10)
        if key in boundary_cache:
            return boundary_cache[key]
        threshold = config.silence_threshold_dbfs - key[1]
        pair: tuple[int | None, int | None] = (None, None)
        if index == 0:
            first = words[spans[0].word_start]
            if _get(first, "matched", False):
                first_start = _ms(first, "start_s")
                pause = _pause(energy, max(0, first_start - radius), first_start,
                               first_start - pad, edge_ms, threshold)
                if pause:
                    pair = (None, max(pause[0], pause[1] - pad))
        elif index == len(spans):
            last = words[spans[-1].word_end - 1]
            if _get(last, "matched", False):
                last_end = _ms(last, "end_s")
                pause = _pause(energy, last_end, min(media_end, last_end + radius),
                               last_end + pad, edge_ms, threshold)
                if pause:
                    pair = (min(pause[1], pause[0] + pad), None)
        else:
            pair = internal_boundary(index, threshold)
        boundary_cache[key] = pair
        return pair

    def internal_boundary(index: int, threshold: float) -> tuple[int | None, int | None]:
        previous = words[spans[index - 1].word_end - 1]
        following = words[spans[index].word_start]
        if not (_get(previous, "matched", False) and _get(following, "matched", False)):
            return None, None
        previous_end = _ms(previous, "end_s")
        next_start = _ms(following, "start_s")
        preferred = (previous_end + next_start) // 2
        # A late release can fall inside the next ASR word. Never search before
        # the last aligned word or beyond the next word's end.
        high = min(media_end, max(previous_end, next_start) + radius,
                   _ms(following, "end_s") - 1)
        pause = _pause(energy, previous_end, high, preferred,
                       2 * edge_ms, threshold)
        if pause:
            quiet_start, quiet_end = pause
            middle = (quiet_start + quiet_end) // 2
            return min(middle, quiet_start + pad), max(middle, quiet_end - pad)
        return None, None

    def gain_for_group(first_word: int, word_end: int) -> float:
        if audio is None or not config.loudness_normalize:
            return 0.0
        first_sample = max(0, round(float(_get(words[first_word], "start_s")) * config.sample_rate))
        last_sample = min(len(audio), round(float(_get(words[word_end - 1], "end_s")) * config.sample_rate))
        piece = audio[first_sample:last_sample]
        level = measure_loudness_lufs(piece, config.sample_rate)
        if not math.isfinite(level):
            return 0.0
        peak = float(np.max(np.abs(piece), initial=0.0))
        return min(config.target_lufs - level, 20 * math.log10(.999 / max(peak, 1e-12)))

    n = len(spans)
    groups: list[list[tuple[int, int]]] = [[] for _ in spans]
    boundary_gains = [float("-inf")] * (n + 1)
    for first_index in range(n - 1, -1, -1):
        if progress_cb is not None and (n - first_index) % 25 == 0:
            progress_cb(f"Checking safe sentence groups {n - first_index}/{n}")
        if not _get(words[spans[first_index].word_start], "matched", False):
            continue
        for last_index in range(first_index, n):
            if last_index > first_index:
                previous_word = words[spans[last_index - 1].word_end - 1]
                next_word = words[spans[last_index].word_start]
                if _ms(next_word, "start_s") - _ms(previous_word, "end_s") > config.max_gap_ms:
                    break
            first_word = spans[first_index].word_start
            word_end = spans[last_index].word_end
            word_count = word_end - first_word
            speech_ms = _ms(words[word_end - 1], "end_s") - _ms(words[first_word], "start_s")
            if speech_ms > config.max_s * 1000 or word_count > config.max_words:
                break
            if not spans[last_index].ends_sentence or not _get(words[word_end - 1], "matched", False):
                continue
            if speech_ms < config.min_s * 1000 - 2 * pad:
                continue
            if word_count < config.min_words:
                continue
            selected = words[first_word:word_end]
            coverage = sum(bool(_get(word, "matched", False)) for word in selected) / word_count
            if coverage < config.min_segment_alignment_coverage:
                continue
            gain_db = gain_for_group(first_word, word_end)
            groups[first_index].append((last_index, word_count))
            boundary_gains[first_index] = max(boundary_gains[first_index], gain_db)
            boundary_gains[last_index + 1] = max(boundary_gains[last_index + 1], gain_db)

    # A shared caption boundary gets ONE acoustic pause, safe for every
    # candidate's gain. Independent choices can select different nearby pauses
    # and duplicate a release in both neighbors even though both ends are quiet.
    boundaries = [boundary(index, gain) if math.isfinite(gain) else (None, None)
                  for index, gain in enumerate(boundary_gains)]
    scores: list[tuple[int, float]] = [(0, 0.0)] * (n + 1)
    choices: list[int | None] = [None] * n
    chosen_times: list[tuple[int, int] | None] = [None] * n
    for first_index in range(n - 1, -1, -1):
        scores[first_index] = scores[first_index + 1]
        for last_index, word_count in groups[first_index]:
            start = boundaries[first_index][1]
            end = boundaries[last_index + 1][0]
            if start is None or end is None:
                continue
            duration = (end - start) / 1000
            if not (config.min_s <= duration <= config.max_s):
                continue
            future = scores[last_index + 1]
            score = (future[0] + word_count, future[1] - abs(duration - config.target_s))
            if score > scores[first_index]:
                scores[first_index] = score
                choices[first_index] = last_index
                chosen_times[first_index] = (start, end)

    result: list[Segment] = []
    rejected: list[dict[str, Any]] = []
    index = 0
    while index < n:
        last_index = choices[index]
        if last_index is None:
            span = spans[index]
            rejected.append({
                "text": caption.text[span.char_start:span.char_end].strip(),
                "source_start_s": _ms(words[span.word_start], "start_s") / 1000,
                "source_end_s": _ms(words[span.word_end - 1], "end_s") / 1000,
                "reason": "no_safe_sentence_group",
            })
            index += 1
            continue
        selected = words[spans[index].word_start:spans[last_index].word_end]
        start, end = chosen_times[index]
        result.append(Segment(
            start_ms=int(start), end_ms=int(end),
            text=caption.text[spans[index].char_start:spans[last_index].char_end].strip(),
            source_cue_indices=tuple(dict.fromkeys(int(_get(word, "cue_index")) for word in selected)),
            word_timestamps=[{
                "text": str(_get(word, "text")),
                "start_s": _ms(word, "start_s") / 1000,
                "end_s": _ms(word, "end_s") / 1000,
                "matched": bool(_get(word, "matched", False)),
            } for word in selected],
            alignment_coverage=sum(bool(_get(word, "matched", False)) for word in selected) / len(selected),
            sentence_aligned=True, boundary="sentence",
        ))
        index = last_index + 1
    return result, rejected
