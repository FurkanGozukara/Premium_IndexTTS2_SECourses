"""Transcript agreement and reference-anchored speaker checks for training data."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torchaudio

from .features import _load_audio_16k
from .whisper_asr import normalize_alignment_token


_TERM_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-\.]*[A-Za-z0-9]|[A-Za-z0-9]")
_MIXED_CASE_RE = re.compile(r"[a-z][A-Z]|[A-Z][a-z]+[A-Z]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def normalized_words(text: str) -> list[str]:
    return [token for word in text.split() for token in normalize_alignment_token(word)]


def transcript_vocabulary(texts: Iterable[str], *, minimum_count: int = 2) -> list[str]:
    """Spellings the transcripts themselves use for names, acronyms, versions, and products.

    Mixed-case tokens, acronyms, tokens containing digits, and words that stay
    capitalized in the middle of sentences are returned most frequent first.
    These are the words a speech recognizer most often spells differently, and
    the user's subtitles are the authority on how they are written.
    """
    counts: Counter[str] = Counter()
    mid_sentence: Counter[str] = Counter()
    for text in texts:
        for sentence in _SENTENCE_SPLIT_RE.split(str(text or "")):
            for index, token in enumerate(_TERM_TOKEN_RE.findall(sentence)):
                counts[token] += 1
                if index:
                    mid_sentence[token] += 1

    def is_term(token: str) -> bool:
        if len(token) < 2 or token == "I" or token.startswith("I'"):
            return False
        if any(ch.isdigit() for ch in token) or token.isupper() or _MIXED_CASE_RE.search(token):
            return True
        return token[0].isupper() and mid_sentence[token] >= 3 and mid_sentence[token] >= 0.6 * counts[token]

    return [token for token, count in counts.most_common() if count >= minimum_count and is_term(token)]


def word_error_counts(reference: str, hypothesis: str) -> tuple[int, int]:
    ref, hyp = normalized_words(reference), normalized_words(hypothesis)
    distances = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        previous = distances
        distances = [i] + [0] * len(hyp)
        for j, b in enumerate(hyp, 1):
            distances[j] = min(previous[j] + 1, distances[j - 1] + 1, previous[j - 1] + (a != b))
    return distances[-1], len(ref)


def boundary_words_match(reference: str, hypothesis: str, count: int = 2) -> bool:
    """Check both transcript edges independently of whole-clip WER."""
    ref, hyp = normalized_words(reference), normalized_words(hypothesis)
    if not ref or not hyp:
        return False
    edge = min(max(1, count), len(ref))
    return len(hyp) >= edge and ref[:edge] == hyp[:edge] and ref[-edge:] == hyp[-edge:]


class TimedTranscript:
    def __init__(self, words: Sequence[Mapping[str, Any]]) -> None:
        ordered = sorted(words, key=lambda word: (float(word["start_s"]) + float(word["end_s"])) / 2)
        self.words = ordered
        self.midpoints = [(float(word["start_s"]) + float(word["end_s"])) / 2 for word in ordered]

    def between(self, start_s: float, end_s: float) -> str:
        start = bisect_left(self.midpoints, start_s)
        end = bisect_right(self.midpoints, end_s)
        return " ".join(str(word["text"]) for word in self.words[start:end])


class SpeakerVerifier:
    """Compare the whole clip and overlapping windows with reviewed references.

    Window scores catch an inserted second voice that a pooled embedding can
    conceal. Scores are similarities, not calibrated identity probabilities.
    """
    def __init__(self, references: Sequence[str | Path], model_dir: str | Path = "models", device: str = "cuda:0") -> None:
        from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus

        if not references:
            raise ValueError("at least one speaker reference is required")
        self.device = torch.device(device)
        self.model = CAMPPlus(feat_dim=80, embedding_size=192)
        path = Path(model_dir) / "hf_cache" / "campplus_cn_common.bin"
        self.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        self.model.to(self.device).eval()
        vectors = []
        for path in references:
            waveform, _ = _load_audio_16k(Path(path))
            vectors.append(self.embedding(waveform))
        self.centroid = torch.nn.functional.normalize(torch.stack(vectors).mean(0), dim=0)

    @torch.inference_mode()
    def embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        features = torchaudio.compliance.kaldi.fbank(waveform.cpu(), num_mel_bins=80, dither=0, sample_frequency=16000)
        features -= features.mean(dim=0, keepdim=True)
        vector = self.model(features.unsqueeze(0).to(self.device)).float().squeeze(0)
        return torch.nn.functional.normalize(vector, dim=0).cpu()

    def score(self, path: str | Path, window_s: float = 6.0) -> dict[str, Any]:
        waveform, duration = _load_audio_16k(Path(path))
        full = float(torch.dot(self.embedding(waveform), self.centroid))
        window = int(window_s * 16000)
        offsets = list(range(0, max(1, waveform.shape[-1] - window + 1), max(1, window // 2)))
        if waveform.shape[-1] > window:
            offsets.append(waveform.shape[-1] - window)
        scores = [float(torch.dot(self.embedding(waveform[:, start:start + window]), self.centroid))
                  for start in sorted(set(offsets))]
        return {"speaker_similarity": full, "speaker_window_min": min(scores),
                "speaker_window_mean": float(np.mean(scores)), "speaker_windows": scores, "duration_s": duration}
