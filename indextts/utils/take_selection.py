"""Best of several takes: keep the candidate Whisper transcribes with the fewest word errors.

Both speech models sample, so takes of the same text differ; a take now and then drops, repeats or slurs a
word. Scoring every candidate against the written text and keeping the best one removed about a sixth to a quarter
of the remaining word errors when keeping the best of two or three takes (September 2026 narration study). Ties keep
the earlier candidate.
"""

from __future__ import annotations

import gc
import re
from typing import Any, Callable, Sequence

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


def _whisper(device: str) -> Callable[[str, str | None], str]:
    import torch
    from transformers import pipeline

    from indextts.training.whisper_asr import _ensure_model, whisper_device_for_free_vram

    if str(device).startswith("cuda"):
        try:
            free = torch.cuda.mem_get_info(torch.device(device))[0] / 1024**3 if torch.cuda.is_available() else None
        except Exception:
            free = None
        device = whisper_device_for_free_vram(device, free, required_gb=2.5)
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    pipe = pipeline("automatic-speech-recognition", model=str(_ensure_model("openai/whisper-large-v3-turbo")),
                    device=device, dtype=dtype, chunk_length_s=30)

    def run(source: Any, language: str) -> str:
        """Transcribe a file path or ``(samples, sample_rate)`` in the given language (never auto-detected)."""
        options = {"task": "transcribe", "do_sample": False, "language": language}
        if isinstance(source, tuple):
            samples, rate = source
            source = {"raw": np.asarray(samples, dtype=np.float32).reshape(-1), "sampling_rate": int(rate)}
        return str(pipe(source, generate_kwargs=options)["text"]).strip()

    run.pipe = pipe  # type: ignore[attr-defined]
    return run


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
            del run
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
    return rows


# Readings the engines speak but no recognizer writes: IndexTTS special-token phone spans, OmniVoice brackets.
_READING_SPANS = re.compile(r"<\|SPECIAL_TOKEN_\d+\|>.*?<\|SPECIAL_TOKEN_\d+\|>|\[[^\[\]\n]*\]")


class SectionTakeJudge:
    """Word errors of one section's takes, for "Takes per section" (both speech models).

    Whisper loads on the first take it scores, transcribes in the generation's language (``whisper_language``:
    the speech model's setting, OmniVoice's Auto resolved from the text) and stays loaded until ``close``.
    A section's reference is its own text without phone readings, so a word spoken from a reading costs every
    take of that section the same and the comparison between its takes stays fair.
    """

    def __init__(self, language: str | None, text: str, *, device: str = "cuda:0",
                 transcriber: Callable[[Any, str], str] | None = None) -> None:
        from .speech_timestamps import whisper_language

        self.language = whisper_language(language, text)
        self.scoring = {"zh": "ZH", "yue": "ZH", "ja": "JA"}.get(self.language, "EN")
        self.device = device
        self._run = transcriber
        self._owned = transcriber is None
        self.history: list[dict[str, Any]] = []

    def error_rate(self, section_text: str, samples: Any, sample_rate: int) -> float:
        from indextts.training.speech_metrics import transcript_metrics

        reference = " ".join(_READING_SPANS.sub(" ", str(section_text)).split())
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

    def close(self) -> None:
        if self._owned and self._run is not None:
            self._run = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


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


def best_take(rows: Sequence[dict[str, Any]]) -> int:
    """Index of the take with the fewest word errors; the first of equals."""

    return min(range(len(rows)), key=lambda index: (float(rows[index]["error_rate"]), index)) if rows else 0


__all__ = ["SectionTakeJudge", "best_take", "candidate_word_errors", "keep_best_take", "metric_language"]
