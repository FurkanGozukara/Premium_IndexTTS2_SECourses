"""Recover complete caption sentences using acoustic pauses before packing."""
from __future__ import annotations

from typing import Any, Callable, Sequence, TYPE_CHECKING
import hashlib
import math

import numpy as np

from .segmenter import SentenceSpan, split_caption_sentences
from .subtitles import CaptionTranscript, Segment
from .media import measure_loudness_lufs

if TYPE_CHECKING:
    from .dataset_prep import DatasetPrepConfig

# Whisper word end times frequently run into the following pause. When the
# strict window finds nothing, the search may start this far before the aligned
# end of the previous word, but never before that word's midpoint.
PAUSE_LOOKBACK_MS = 200
# A quiet run found inside the previous word must be longer than a stop
# consonant closure, so that closure cannot be mistaken for a pause.
CLOSURE_SAFE_QUIET_MS = 100
# Target used for the share of groups that should stay one short sentence.
SHORT_CLIP_TARGET_S = 6.0
# Target used for the share of groups that should stay one long sentence or two short ones.
MEDIUM_CLIP_TARGET_S = 10.0
# A group aimed at a short or medium length is only taken when each of its inner edges sits in a
# pause at least this wide after padding (about 200 ms of quiet with the default 60 ms padding),
# so the clip keeps the silence a spoken sentence has around it instead of being cut where the
# speaker ran on; a start that finds no such group falls back to the normal target.
SHORT_CLIP_MIN_PAUSE_MS = 80
# A pause must also be this fraction of the loudest nearby audio (-20 dB), so a quiet recording is
# not classified entirely as silence.
PAUSE_RELATIVE_LIMIT = 0.1


class RangeLoudness:
    """BS.1770 integrated loudness and peak of any sample range of one recording.

    Follows pyloudnorm measuring the cut piece (400 ms blocks every 100 ms, absolute and relative gates): the
    recording is K-weighted once, in chunks, and block energies come from 1 ms prefix sums, so a range costs
    about 0.04 ms instead of 1.5 ms. On speech the two differ by 0.004 dB on average and by up to 0.2 dB when a
    block lands on the other side of a gate. For packing that weighs thousands of overlapping candidate clips.
    """

    def __init__(self, audio: np.ndarray, rate: int) -> None:
        import pyloudnorm
        from scipy.signal import lfilter

        self.audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        self.rate = int(rate)
        self.unit = max(1, self.rate // 1000)
        stages = list(pyloudnorm.Meter(self.rate)._filters.values())
        states = [np.zeros(max(len(stage.a), len(stage.b)) - 1) for stage in stages]
        sums = np.zeros(-(-self.audio.size // self.unit), dtype=np.float64)
        chunk_size = self.unit * 40_000
        for start in range(0, self.audio.size, chunk_size):
            chunk = self.audio[start:start + chunk_size].astype(np.float64)
            for index, stage in enumerate(stages):
                chunk, states[index] = lfilter(stage.b, stage.a, chunk, zi=states[index])
                chunk = stage.passband_gain * chunk
            squared = np.square(chunk)
            squared = np.pad(squared, (0, (-squared.size) % self.unit))
            first = start // self.unit
            sums[first:first + squared.size // self.unit] = squared.reshape(-1, self.unit).sum(axis=1)
        self.prefix = np.concatenate(([0.0], np.cumsum(sums)))
        self.frame = self.unit * 10
        full = self.audio.size // self.frame
        self.frame_peaks = (np.abs(self.audio[:full * self.frame]).reshape(full, self.frame).max(axis=1)
                            if full else np.zeros(0, dtype=np.float32))

    def loudness(self, first: int, last: int) -> float:
        length = int(last) - int(first)
        if length < 0.4 * self.rate:
            return measure_loudness_lufs(self.audio[first:last], self.rate)  # pyloudnorm's short-piece fallback
        blocks = int(np.round((length / self.rate - 0.4) / (0.4 * 0.25))) + 1
        index = np.arange(blocks)
        lower = (0.4 * (index * 0.25) * self.rate).astype(np.int64)
        upper = np.minimum((0.4 * (index * 0.25 + 1) * self.rate).astype(np.int64), length)
        units = self.prefix.size - 1
        low = np.clip(np.rint((first + lower) / self.unit).astype(np.int64), 0, units)
        high = np.clip(np.rint((first + upper) / self.unit).astype(np.int64), 0, units)
        energy = np.maximum(self.prefix[high] - self.prefix[low], 0.0) / (0.4 * self.rate)
        with np.errstate(divide="ignore"):
            levels = -0.691 + 10.0 * np.log10(energy)
            absolute = energy[levels >= -70.0]
            if absolute.size:
                relative = -0.691 + 10.0 * math.log10(float(np.mean(absolute))) - 10.0
                gated = energy[(levels > relative) & (levels > -70.0)]
                if gated.size:
                    value = -0.691 + 10.0 * math.log10(float(np.mean(gated)))
                    if math.isfinite(value):
                        return value
        return measure_loudness_lufs(self.audio[first:last], self.rate)  # silent or ungated: the RMS fallback

    def peak(self, first: int, last: int) -> float:
        first, last = int(first), int(last)
        full_first, full_last = -(-first // self.frame), last // self.frame
        if full_last <= full_first:
            return float(np.max(np.abs(self.audio[first:last]), initial=0.0))
        return max(
            float(self.frame_peaks[full_first:full_last].max()),
            float(np.max(np.abs(self.audio[first:full_first * self.frame]), initial=0.0)),
            float(np.max(np.abs(self.audio[full_last * self.frame:last]), initial=0.0)),
        )


def _get(word: Any, key: str, default: Any = None) -> Any:
    return word.get(key, default) if isinstance(word, dict) else getattr(word, key, default)


def _ms(word: Any, key: str) -> int:
    return round(float(_get(word, key)) * 1000)


def _pause(
    energy: np.ndarray, low: int, high: int, preferred: int, minimum_ms: int,
    threshold_dbfs: float, relative_limit: float = PAUSE_RELATIVE_LIMIT,
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
    threshold = min(10 ** (threshold_dbfs / 20), float(context.max(initial=0)) * relative_limit)
    quiet = np.isfinite(window) & (window <= threshold)
    edges = np.diff(np.pad(quiet.astype(np.int8), (1, 1)))
    runs = [((first + start) * hop, (first + end) * hop)
            for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
            if end - start >= minimum]
    return min(runs, key=lambda pair: abs((pair[0] + pair[1]) / 2 - preferred)) if runs else None


def pause_phrase_spans(
    caption: CaptionTranscript,
    words: Sequence[Any],
    energy: np.ndarray,
    config: DatasetPrepConfig,
    *,
    gain_db: float = 0.0,
    pause_relative_limit: float = PAUSE_RELATIVE_LIMIT,
    split_long_sentences: bool = True,
) -> list[SentenceSpan]:
    """Sentence spans, where a stretch that cannot become a clip also ends at acoustic pauses.

    A recognizer sometimes returns long stretches without sentence punctuation (in noise it may return none at
    all, with word times that touch), and a sentence can be longer than a clip; neither can be packed. Such a
    stretch also ends at every word boundary where the pause search finds quiet audio, with ``gain_db``
    standing in for each clip's loudness normalization and 6 dB of tolerance (the repacking checks every edge
    again exactly). The transcript's last stretch ends with the recording. Sentences that fit a clip stay
    whole; with ``split_long_sentences`` off, so do longer punctuated sentences (a supplied transcript's own
    sentence boundaries are kept).
    """
    spans = split_caption_sentences(caption)
    if not spans or len(words) != len(caption.words):
        return spans
    edge_ms = max(10, math.ceil(config.min_edge_silence_ms / 10) * 10)
    radius = config.snap_window_ms + max(edge_ms, config.pad_ms)
    media_end = len(energy) * 10
    threshold = config.silence_threshold_dbfs - gain_db + 6.0
    result: list[SentenceSpan] = []
    for index, span in enumerate(spans):
        last_span = index == len(spans) - 1
        start = _ms(words[span.word_start], "start_s")
        end = _ms(words[span.word_end - 1], "end_s")
        if span.ends_sentence and (end - start <= config.max_s * 1000 or not split_long_sentences):
            result.append(span)
            continue
        piece_word, piece_char = span.word_start, span.char_start
        for cut in range(span.word_start, span.word_end - 1):
            previous, following = words[cut], words[cut + 1]
            previous_start, previous_end = _ms(previous, "start_s"), _ms(previous, "end_s")
            next_start, next_end = _ms(following, "start_s"), _ms(following, "end_s")
            high = min(media_end, max(previous_end, next_start) + radius, next_end - 1)
            preferred = (previous_end + next_start) // 2
            pause = _pause(energy, previous_end, high, preferred, 2 * edge_ms, threshold, pause_relative_limit)
            low = max((previous_start + previous_end) // 2, previous_end - PAUSE_LOOKBACK_MS)
            if pause is None and low < previous_end:
                pause = _pause(energy, low, high, preferred, max(CLOSURE_SAFE_QUIET_MS, 2 * edge_ms),
                               threshold, pause_relative_limit)
            if pause is None:
                continue
            next_char = int(caption.words[cut + 1].char_start)
            char_end = next_char
            while char_end > piece_char and caption.text[char_end - 1].isspace():
                char_end -= 1
            result.append(SentenceSpan(
                char_start=piece_char, char_end=char_end, word_start=piece_word, word_end=cut + 1,
                starts_sentence=span.starts_sentence and piece_word == span.word_start, ends_sentence=True,
            ))
            piece_word, piece_char = cut + 1, next_char
        result.append(SentenceSpan(
            char_start=piece_char, char_end=span.char_end, word_start=piece_word, word_end=span.word_end,
            starts_sentence=span.starts_sentence and piece_word == span.word_start,
            ends_sentence=span.ends_sentence or last_span,
        ))
    return result


def build_safe_sentence_segments(
    caption: CaptionTranscript,
    words: Sequence[Any],
    energy: np.ndarray,
    config: DatasetPrepConfig,
    *,
    audio: np.ndarray | None = None,
    progress_cb: Callable[[str], None] | None = None,
    pause_relative_limit: float = PAUSE_RELATIVE_LIMIT,
    spans: Sequence[SentenceSpan] | None = None,
    loudness: RangeLoudness | None = None,
) -> tuple[list[Segment], list[dict[str, Any]]]:
    """Repack complete sentences across unsafe cuts, using original audio.

    A boundary with no sustained pause cannot become an output edge. Dynamic
    programming maximizes retained caption words, then favors the target clip
    duration. It can move a sentence to an adjacent clip or merge sentences
    across a bad boundary; every retained word appears exactly once.
    ``pause_relative_limit`` is the loudness fraction of nearby audio a pause
    must stay under, in addition to the configured silence threshold.
    ``spans`` replaces the caption's sentences (see ``pause_phrase_spans``), and
    ``loudness`` measures candidate clips' normalization gains from one pass
    over the recording instead of each cut piece.
    """
    spans = list(spans) if spans is not None else split_caption_sentences(caption)
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
                               first_start - pad, edge_ms, threshold, pause_relative_limit)
                if pause:
                    pair = (None, max(pause[0], pause[1] - pad))
        elif index == len(spans):
            last = words[spans[-1].word_end - 1]
            if _get(last, "matched", False):
                last_end = _ms(last, "end_s")
                pause = _pause(energy, last_end, min(media_end, last_end + radius),
                               last_end + pad, edge_ms, threshold, pause_relative_limit)
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
                       2 * edge_ms, threshold, pause_relative_limit)
        previous_start = _ms(previous, "start_s")
        low = max((previous_start + previous_end) // 2, previous_end - PAUSE_LOOKBACK_MS)
        if pause is None and low < previous_end:
            pause = _pause(energy, low, high, preferred,
                           max(CLOSURE_SAFE_QUIET_MS, 2 * edge_ms), threshold, pause_relative_limit)
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
        if loudness is not None:
            level = loudness.loudness(first_sample, last_sample)
            if not math.isfinite(level):
                return 0.0
            peak = loudness.peak(first_sample, last_sample)
            return min(config.target_lufs - level, 20 * math.log10(.999 / max(peak, 1e-12)))
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

    short_fraction = min(1.0, max(0.0, float(getattr(config, "short_clip_fraction", 0.0) or 0.0)))
    medium_fraction = min(1.0 - short_fraction, max(0.0, float(getattr(config, "medium_clip_fraction", 0.0) or 0.0)))
    short_target_s = max(float(config.min_s), min(SHORT_CLIP_TARGET_S, float(config.target_s)))
    medium_target_s = max(float(config.min_s), min(MEDIUM_CLIP_TARGET_S, float(config.target_s)))

    def target_for(first_index: int) -> tuple[float, bool]:
        # A reproducible share of groups aims for one short sentence, and another share for a
        # medium clip, so the dataset also contains the sentence lengths generation typically
        # uses. The flag says the aim is shorter than the target: such a group is taken only
        # between clear pauses, so it keeps the natural silence around a sentence instead of
        # being clipped out of a run-on, and otherwise the start falls back to the target.
        if short_fraction <= 0.0 and medium_fraction <= 0.0:
            return float(config.target_s), False
        digest = hashlib.sha256(f"{config.seed}:{n}:{first_index}".encode("utf-8")).digest()
        unit = int.from_bytes(digest[:8], "big") / 2 ** 64
        if unit < short_fraction:
            return short_target_s, True
        if unit < short_fraction + medium_fraction:
            return medium_target_s, True
        return float(config.target_s), False

    def pause_width_ms(index: int) -> int | None:
        # Width of the shared pause a boundary sits in, after padding; None at the media edges.
        if index <= 0 or index >= n:
            return None
        end_previous, start_next = boundaries[index]
        if end_previous is None or start_next is None:
            return None
        return int(start_next) - int(end_previous)

    def clear_edges(first_index: int, last_index: int) -> bool:
        for index in (first_index, last_index + 1):
            width = pause_width_ms(index)
            if width is not None and width < SHORT_CLIP_MIN_PAUSE_MS:
                return False
        return True

    scores: list[tuple[int, float]] = [(0, 0.0)] * (n + 1)
    choices: list[int | None] = [None] * n
    chosen_times: list[tuple[int, int] | None] = [None] * n
    chosen_aims: list[str] = ["target"] * n
    for first_index in range(n - 1, -1, -1):
        scores[first_index] = scores[first_index + 1]
        aimed_target_s, shorter = target_for(first_index)
        attempts = ((aimed_target_s, True), (float(config.target_s), False)) if shorter else ((aimed_target_s, False),)
        for target_s, strict in attempts:
            for last_index, word_count in groups[first_index]:
                start = boundaries[first_index][1]
                end = boundaries[last_index + 1][0]
                if start is None or end is None:
                    continue
                if strict and not clear_edges(first_index, last_index):
                    continue
                duration = (end - start) / 1000
                if not (config.min_s <= duration <= config.max_s):
                    continue
                future = scores[last_index + 1]
                score = (future[0] + word_count, future[1] - abs(duration - target_s))
                if score > scores[first_index]:
                    scores[first_index] = score
                    choices[first_index] = last_index
                    chosen_times[first_index] = (start, end)
                    # Label the length class the clip actually reached, not the aim: a short aim that only
                    # found a clear-edged two-sentence group is a medium or target-length clip.
                    if not strict or duration > (medium_target_s + float(config.target_s)) / 2:
                        chosen_aims[first_index] = "target"
                    elif duration > (short_target_s + medium_target_s) / 2:
                        chosen_aims[first_index] = "medium"
                    else:
                        chosen_aims[first_index] = "short"
            if choices[first_index] is not None:
                break

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
            sentence_aligned=True, boundary="sentence", length_aim=chosen_aims[index],
        ))
        index = last_index + 1
    return result, rejected
