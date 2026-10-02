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

    def run(path: str, language: str | None) -> str:
        options = {"task": "transcribe", "do_sample": False, **({"language": language} if language else {})}
        return str(pipe(path, generate_kwargs=options)["text"]).strip()

    run.pipe = pipe  # type: ignore[attr-defined]
    return run


def candidate_word_errors(paths: Sequence[str], text: str, language: str | None, *, device: str = "cuda:0",
                          transcriber: Callable[[str, str | None], str] | None = None) -> list[dict[str, Any]]:
    """Word error rate of every candidate against the written text, in candidate order."""

    from indextts.training.speech_metrics import transcript_metrics
    from .speech_timestamps import whisper_language, written_words

    reference = " ".join(written_words(text))
    scoring = metric_language(language, reference)
    code = whisper_language(language)
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


def best_take(rows: Sequence[dict[str, Any]]) -> int:
    """Index of the take with the fewest word errors; the first of equals."""

    return min(range(len(rows)), key=lambda index: (float(rows[index]["error_rate"]), index)) if rows else 0


__all__ = ["best_take", "candidate_word_errors", "metric_language"]
