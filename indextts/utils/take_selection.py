"""Best of several takes: keep the candidate Whisper transcribes with the fewest word errors.

Both speech models sample, so takes of the same text differ; a take now and then drops, repeats or slurs a
word. Scoring every candidate against the written text and keeping the best one removed about a sixth to a quarter
of the remaining word errors when keeping the best of two or three takes (September 2026 narration study). Ties keep
the earlier candidate.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Mapping, Sequence

import numpy as np

_CJK_RE = re.compile(r"[぀-ヿ㐀-鿿]")


def metric_language(language: str | None, text: str) -> str:
    """The word-error language: characters for Chinese and Japanese, words otherwise."""

    code = str(language or "").strip().upper()
    if code in {"ZH", "ZHEN", "JA"}:
        return "JA" if code == "JA" else "ZH"
    if code in {"", "AUTO"} and _CJK_RE.search(text or ""):
        return "JA" if re.search(r"[぀-ヿ]", text) else "ZH"
    return "EN"


def _whisper(device: str) -> Callable[[Any, str], str]:
    """The built-in Whisper (``indextts.asr``) as ``run(source, language) -> text``."""

    from indextts.asr import recognize

    def run(source: Any, language: str) -> str:
        """Transcribe a file path or ``(samples, sample_rate)`` in the given language (never auto-detected)."""
        if isinstance(source, tuple):
            samples, rate = source
            source = (np.asarray(samples, dtype=np.float32).reshape(-1), int(rate))
        return recognize(source, language=language, device=device, words=False).text

    return run


def _release_whisper() -> None:
    """The judging is over: the built-in Whisper's weights wait in RAM and the speech model keeps the VRAM."""
    from indextts.asr import park

    park()


def candidate_word_errors(paths: Sequence[str], text: str, language: str | None, *, device: str = "cuda:0",
                          transcriber: Callable[[str, str | None], str] | None = None) -> list[dict[str, Any]]:
    """Word error rate of every candidate against the written text, in candidate order."""

    from indextts.training.speech_metrics import transcript_metrics
    from .speech_timestamps import whisper_language, written_words

    reference = " ".join(written_words(text))
    scoring = metric_language(language, reference)
    code = whisper_language(language, reference)
    run = transcriber or _whisper(device)
    rows = []
    try:
        for path in paths:
            heard = run(str(path), code)
            metrics = transcript_metrics(reference, heard, scoring)
            rows.append({"path": str(path), "error_rate": round(float(metrics["error_rate"]), 4), "heard": heard})
    finally:
        if transcriber is None:
            _release_whisper()
    return rows


# Readings the engines speak but no recognizer writes: IndexTTS special-token phone spans, OmniVoice brackets.
_READING_SPANS = re.compile(r"<\|SPECIAL_TOKEN_\d+\|>.*?<\|SPECIAL_TOKEN_\d+\|>|\[[^\[\]\n]*\]")
_SPECIAL_TOKEN = re.compile(r"<\|SPECIAL_TOKEN_\d+\|>")
# The <word|reading> annotations of the text the user wrote (the pronunciation dictionary writes the same form).
_ANNOTATION = re.compile(r"<([^|>\n]+)\|([^>\n]+)>")


def reading_key(reading: str) -> str:
    """A reading as both engines write it, for matching: upper case, single spaces, no syllable dots."""

    return " ".join(token for token in str(reading).upper().split() if token != ".")


def written_readings(text: str) -> dict[str, str]:
    """The written word of every ``<word|reading>`` annotation in the text, keyed by ``reading_key``."""

    words: dict[str, str] = {}
    for match in _ANNOTATION.finditer(str(text or "")):
        words.setdefault(reading_key(match.group(2)), match.group(1).strip())
    return words


def section_reference(section_text: str, words: Mapping[str, str] | None = None) -> str:
    """A section's text as Whisper should hear it: every phone reading the engine speaks becomes the written word
    of its annotation (``words``, from ``written_readings``); readings without one, and tags such as [laughter],
    are left out."""

    def replace(match: re.Match) -> str:
        span = _SPECIAL_TOKEN.sub(" ", match.group(0)).strip()
        if span.startswith("[") and span.endswith("]"):
            span = span[1:-1]
        word = (words or {}).get(reading_key(span))
        return f" {word} " if word else " "

    return " ".join(_READING_SPANS.sub(replace, str(section_text)).split())


class SectionTakeJudge:
    """Word errors of one section's takes, for "Takes per section" (both speech models).

    Whisper loads on the first take it scores, transcribes in the generation's language (``whisper_language``:
    the speech model's setting, OmniVoice's Auto resolved from the text) and stays loaded until ``close``.
    A section's reference is its own text with every phone reading written as the word it reads (from the
    ``<word|reading>`` annotations of ``text``; rare words forgive close spellings), so a take that speaks the
    reading right has no word error. Leaving the readings out made every take of such a section count the
    spoken word as an extra one ("<xformers|...> and <SageAttention|...>." scored 400 % on every take): each
    section with a dictionary word rendered all its takes and never counted as free of word errors.

    With ``rule="similar"`` the judge also measures how much each take sounds like the voice
    (``voice_similarity``: the trained voice's clips, else the reference clip) and the engine keeps the most
    similar take without word errors (``keep_most_similar_take``), checking at most ``checks`` takes.
    """

    def __init__(self, language: str | None, text: str, *, device: str = "cuda:0",
                 transcriber: Callable[[Any, str], str] | None = None, rule: str = "errors", checks: int = 5,
                 voice: str | None = None, reference: str | None = None, model_dir: str = "models",
                 embedder: Callable[[Any, int], Any] | None = None) -> None:
        from .speech_timestamps import whisper_language

        self.language = whisper_language(language, text)
        self.scoring = {"zh": "ZH", "yue": "ZH", "ja": "JA"}.get(self.language, "EN")
        self.readings = written_readings(text)
        self.device = device
        self._run = transcriber
        self._owned = transcriber is None
        self.history: list[dict[str, Any]] = []
        self.rule = "similar" if str(rule or "").strip().lower() == "similar" else "errors"
        self.checks = max(1, min(8, int(checks or 5)))
        self._voice, self._reference, self._model_dir = voice or None, reference or None, model_dir
        self._embed = embedder
        self._embed_owned = embedder is None
        self._target: Any = None
        self._target_ready = False
        self.target_source = ""

    def _embedder(self) -> Callable[[Any, int], Any]:
        if self._embed is None:
            from .voice_similarity import SpeakerEmbedder

            self._embed = SpeakerEmbedder(self._model_dir, self.device)
        return self._embed

    def _similarity_target(self) -> Any:
        if not self._target_ready:
            self._target_ready = True
            try:
                from .voice_similarity import voice_target

                self._target, self.target_source = voice_target(self._voice, self._embedder(),
                                                                reference=self._reference)
            except Exception as exc:  # no CAMPPlus weights or an unreadable dataset: fall back to word errors
                print(f">> Takes per section: voice similarity unavailable ({exc})", flush=True)
                self._target, self.target_source = None, ""
            if self._target is None and self.rule == "similar":
                print(">> Takes per section: nothing to compare the voice with; keeping the fewest word errors",
                      flush=True)
        return self._target

    def compares_voices(self) -> bool:
        """True when the most-similar rule is on and there is a voice to compare takes with."""
        return self.rule == "similar" and self._similarity_target() is not None

    def similarity(self, samples: Any, sample_rate: int) -> float:
        import numpy as np

        target = self._similarity_target()
        if target is None:
            return 0.0
        return round(float(np.dot(np.asarray(self._embedder()(samples, sample_rate)).reshape(-1),
                                  np.asarray(target).reshape(-1))), 4)

    def error_rate(self, section_text: str, samples: Any, sample_rate: int) -> float:
        from indextts.training.speech_metrics import transcript_metrics

        reference = section_reference(section_text, self.readings)
        if not any(char.isalnum() for char in reference):
            return 0.0
        if self._run is None:
            self._run = _whisper(self.device)
        heard = self._run((samples, sample_rate), self.language)
        try:
            return round(float(transcript_metrics(reference, heard, self.scoring,
                                                  lenient_terms=self._lenient(reference))["error_rate"]), 4)
        except ValueError:  # nothing scoreable in the reference
            return 0.0

    def _lenient(self, reference: str) -> frozenset[str] | None:
        """Rare and technical words (outside the English pronouncing dictionary): a close spelling counts as right.

        Whisper writes "Kohya" as "Koya" in every take, which would otherwise keep each take above zero errors.
        A clearly different word still counts (speech_metrics' lenient-term rule).
        """
        if self.scoring != "EN":
            return None
        from indextts.training.speech_metrics import lenient_units
        from indextts.utils.pronunciation import candidate_words, cmu_available, lookup_word

        if not cmu_available():
            return None
        terms = [word for word in candidate_words(reference) if not lookup_word(word.casefold())]
        return lenient_units(terms, "EN") if terms else None

    def record(self, section: int, rates: Sequence[float], kept: int) -> None:
        self.history.append({"section": int(section), "error_rates": [float(rate) for rate in rates], "kept": int(kept)})
        summary = ", ".join(f"take {index + 1} {100 * rate:.1f}%" for index, rate in enumerate(rates))
        print(f">> Section {section + 1}: {summary} -> kept take {kept + 1}", flush=True)

    def record_similar(self, section: int, outcome: Mapping[str, Any]) -> None:
        similarities = [float(value) for value in outcome["similarities"]]
        checked = [dict(item) for item in outcome["checked"]]
        kept = int(outcome["kept"])
        self.history.append({"section": int(section), "rule": "similar", "similarities": similarities,
                             "checked": checked, "kept": kept})
        summary = ", ".join(f"take {item['take'] + 1} ({similarities[item['take']]:.3f}) "
                            f"{100 * float(item['error_rate']):.1f}%" for item in checked)
        print(f">> Section {section + 1}: {len(similarities)} takes, most similar first: {summary} "
              f"-> kept take {kept + 1}", flush=True)

    def close(self) -> None:
        if self._owned and self._run is not None:
            self._run = None
            _release_whisper()
        if self._embed_owned and self._embed is not None:
            closer = getattr(self._embed, "close", None)
            self._embed = None
            if closer:
                closer()


def keep_best_take(first: Any, retake: Callable[[], Any], score: Callable[[Any], float], takes: int
                   ) -> tuple[Any, list[float], int]:
    """Up to ``takes`` renders of one section: the first with the fewest word errors (a perfect take ends the search)."""

    best, rates = first, [score(first)]
    kept = 0
    while rates[kept] > 0.0 and len(rates) < max(1, int(takes)):
        candidate = retake()
        rates.append(score(candidate))
        if rates[-1] < rates[kept]:
            best, kept = candidate, len(rates) - 1
    return best, rates, kept


def keep_most_similar_take(first: Any, render_more: Callable[[int], Sequence[Any]],
                           similarity: Callable[[Any], float], score: Callable[[Any], float], renders: int,
                           checks: int = 5) -> tuple[Any, dict[str, Any]]:
    """``renders`` takes of one section, most speaker-like first: Whisper checks up to ``checks`` of them in that
    order and the first without a word error wins, else the fewest errors (ties: the more similar take).

    ``render_more(n)`` returns ``n`` further takes (an engine may render them as one batch). On a 139-line narration
    ten cloned takes with five checks scored best on likeness, delivery style and word errors (October 2026).
    """

    takes = [first, *render_more(max(0, int(renders) - 1))] if int(renders) > 1 else [first]
    similarities = [float(similarity(take)) for take in takes]
    order = sorted(range(len(takes)), key=lambda index: (-similarities[index], index))
    checked: list[dict[str, Any]] = []
    for index in order[:max(1, int(checks))]:
        checked.append({"take": index, "error_rate": float(score(takes[index]))})
        if checked[-1]["error_rate"] <= 0.0:
            break
    kept = min(checked, key=lambda item: (item["error_rate"], order.index(item["take"])))["take"]
    return takes[kept], {"similarities": similarities, "checked": checked, "kept": kept}


def take_defaults(model: str, tier: Any, *, trained: bool = False) -> tuple[int, str, int]:
    """``(takes, rule, Whisper checks)`` of "Takes per section" in the presets, per speech model and GPU tier.

    Measured in October 2026 on 60 tutorial lines per setting (zero-shot cloning of the bundled demo voice with each
    base model; trained IndexTTS, AuK and OmniVoice voices): every take selection removed 60-75 % of the word errors.
    Ranking by voice similarity raised AuK's delivery style (+0.006 with 5 takes, +0.009 with 10) and OmniVoice's
    slightly, and a cloned OmniVoice voice with 10 takes ranked best of 14 narration versions; for IndexTTS it brought
    no likeness and made generation five times slower than fewest word errors. OmniVoice and AuK render the extra takes
    in batches. Larger tiers are faster GPUs, so they render more takes; AuK's on-demand tiers (6, 8 GB) move its models
    for every batch and keep to fewest word errors.
    """

    try:
        size = int(float(str(tier)))
    except (TypeError, ValueError):
        size = 32
    if model == "omnivoice":
        if size >= (12 if trained else 24):
            return 10, "similar", 5
        return (5, "similar", 3) if size >= (6 if trained else 12) else (3, "errors", 5)
    if model == "auk":
        if size >= 16:
            return (10, "similar", 5) if trained else (5, "similar", 3)
        return (5, "similar", 3) if size >= 10 else (3, "errors", 5)
    return (5, "errors", 5) if trained else (3, "errors", 5)


def take_preset_values(model: str, tier: Any, *, trained: bool = False) -> dict[str, Any]:
    takes, rule, checks = take_defaults(model, tier, trained=trained)
    return {"generation.section_takes": takes, "generation.section_take_rule": rule,
            "generation.section_take_checks": checks}


def best_take(rows: Sequence[dict[str, Any]]) -> int:
    """Index of the take with the fewest word errors; the first of equals."""

    return min(range(len(rows)), key=lambda index: (float(rows[index]["error_rate"]), index)) if rows else 0


__all__ = ["SectionTakeJudge", "best_take", "candidate_word_errors", "keep_best_take", "keep_most_similar_take",
           "metric_language", "take_defaults", "take_preset_values"]
