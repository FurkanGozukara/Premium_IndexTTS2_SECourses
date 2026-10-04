"""How much generated speech sounds like a voice: CAMPPlus speaker vectors, as the speech evaluation measures them.

A take's vector is CAMPPlus (``models/hf_cache/campplus_cn_common.bin``, the evaluation's speaker embedding) of its
16 kHz audio, averaged over 20-second windows. A trained voice's target is the normalized mean of up to 60 of its
training clips (6-15 s, spread over the source recordings), cached beside the voice in ``analysis/voice_centroid.json``;
without one (a base model, or a voice whose dataset is gone), the cloning reference clip is the target.

"Takes per section" ranks a section's takes by this similarity (``take_selection.keep_most_similar_take``), and the
cloning preset after training builds the centroid ahead of time (``indextts.training.voice_preset``). On a 139-line
narration, keeping the most similar error-free of ten cloned takes scored best of 14 versions on likeness, delivery
style and word errors (October 2026).
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CENTROID_FILE = "voice_centroid.json"
CENTROID_VERSION = 1
CENTROID_CLIPS = 60
EMBEDDING = "campplus_cn_common"
WINDOW_S = 20.0


class SpeakerEmbedder:
    """Unit-length CAMPPlus vectors of audio (numpy samples at any rate, or a file)."""

    def __init__(self, model_dir: str | Path = "models", device: str = "cuda:0") -> None:
        import torch

        from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus

        weights = Path(model_dir) / "hf_cache" / f"{EMBEDDING}.bin"
        if not weights.is_file():
            raise FileNotFoundError(f"CAMPPlus weights not found: {weights}")
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        model = CAMPPlus(feat_dim=80, embedding_size=192)
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
        self.model = model.to(device=self.device, dtype=torch.float32).eval()

    def __call__(self, samples: Any, rate: int) -> np.ndarray:
        import torch
        import torchaudio

        wave = torch.as_tensor(np.asarray(samples, dtype=np.float32).reshape(-1))
        if int(rate) != 16000:
            wave = torchaudio.functional.resample(wave, int(rate), 16000)
        window = int(WINDOW_S * 16000)
        pieces = [wave[start:start + window] for start in range(0, max(1, wave.numel()), window)]
        pieces = [piece for piece in pieces if piece.numel() >= 8000] or [wave]  # a tail under 0.5 s adds nothing
        vectors = []
        with torch.inference_mode():
            for piece in pieces:
                fbank = torchaudio.compliance.kaldi.fbank(piece.unsqueeze(0), num_mel_bins=80, dither=0,
                                                          sample_frequency=16000)
                fbank = fbank - fbank.mean(dim=0, keepdim=True)
                vector = self.model(fbank.unsqueeze(0).to(self.device)).squeeze(0).float().cpu().numpy()
                vectors.append(vector / (np.linalg.norm(vector) + 1e-9))
        return unit(np.mean(vectors, axis=0))

    def file(self, path: str | Path) -> np.ndarray:
        import soundfile as sf

        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
        return self(data.mean(axis=1), rate)

    def close(self) -> None:
        import torch

        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def unit(vector: Any) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    return vector / (np.linalg.norm(vector) + 1e-9)


def centroid_path(adapter_path: str | Path) -> Path:
    from indextts.training.reference_audition import run_dir_of

    return run_dir_of(adapter_path) / "analysis" / CENTROID_FILE


def _source(row: Mapping[str, Any]) -> str:
    return str(row.get("source_media") or str(row.get("id") or "").rsplit("_", 1)[0])


def choose_centroid_clips(rows: Sequence[Mapping[str, Any]], count: int = CENTROID_CLIPS,
                          seed: int = 7) -> list[Mapping[str, Any]]:
    """Up to ``count`` clips of 6-15 s in a seeded order, spread evenly over the source recordings."""
    eligible = [row for row in rows if 6.0 <= float(row.get("duration_s") or 0.0) <= 15.0]
    random.Random(seed).shuffle(eligible)
    sources = {_source(row) for row in eligible}
    per_source = max(3, math.ceil(count / max(1, len(sources))))
    picked: list[Mapping[str, Any]] = []
    used: dict[str, int] = {}
    for row in eligible:
        if used.get(_source(row), 0) < per_source:
            used[_source(row)] = used.get(_source(row), 0) + 1
            picked.append(row)
        if len(picked) >= count:
            break
    return picked


def build_centroid(adapter_path: str | Path, embedder: SpeakerEmbedder, *,
                   datasets_root: str | Path | None = None) -> dict[str, Any] | None:
    """Measure the voice's training clips and cache their centroid beside the voice (None without its dataset)."""
    from indextts.training.dataset_manifest import atomic_write_json, load_manifest
    from indextts.training.dataset_profile import dataset_dir_for_adapter
    from indextts.training.evaluation_plan import audio_path
    from indextts.training.reference_audition import run_dir_of, split_rows

    dataset_dir = dataset_dir_for_adapter(adapter_path, datasets_root=datasets_root)
    if dataset_dir is None:
        return None
    rows = load_manifest(Path(dataset_dir))
    training, _ = split_rows(Path(dataset_dir), rows, run_dir_of(adapter_path))
    clips = [(row, audio_path(Path(dataset_dir), row)) for row in choose_centroid_clips(training or rows)]
    clips = [(row, path) for row, path in clips if path.is_file()]
    if len(clips) < 5:
        return None
    vectors = [embedder.file(path) for _, path in clips]
    record = {"version": CENTROID_VERSION, "embedding": EMBEDDING, "dataset": str(dataset_dir),
              "clips": [str(row.get("id") or path.name) for row, path in clips],
              "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "vector": [round(float(value), 6) for value in unit(np.mean(vectors, axis=0))]}
    target = centroid_path(adapter_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target, record)
    return record


def load_centroid(adapter_path: str | Path | None) -> np.ndarray | None:
    if not adapter_path:
        return None
    try:
        record = json.loads(centroid_path(adapter_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if record.get("version") != CENTROID_VERSION or record.get("embedding") != EMBEDDING:
        return None
    vector = np.asarray(record.get("vector") or [], dtype=np.float32)
    return unit(vector) if vector.size == 192 else None


def voice_target(adapter_path: str | Path | None, embedder: SpeakerEmbedder, *, reference: str | Path | None = None,
                 datasets_root: str | Path | None = None) -> tuple[np.ndarray | None, str]:
    """``(vector, description)`` of what takes are compared with: the voice's centroid (cached, else built from its
    dataset), else the reference clip, else ``(None, "")``."""
    if adapter_path and Path(adapter_path).is_file():
        vector = load_centroid(adapter_path)
        if vector is None:
            record = build_centroid(adapter_path, embedder, datasets_root=datasets_root)
            vector = load_centroid(adapter_path) if record else None
        if vector is not None:
            return vector, "the voice's training clips"
    if reference and Path(reference).is_file():
        return embedder.file(reference), "the reference clip"
    return None, ""


__all__ = ["CENTROID_FILE", "SpeakerEmbedder", "build_centroid", "centroid_path", "choose_centroid_clips",
           "load_centroid", "unit", "voice_target"]
