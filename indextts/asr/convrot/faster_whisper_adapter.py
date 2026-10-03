"""faster-whisper ``WhisperModel`` that runs on the ConvRot engine instead of CTranslate2.

Ported from Whisper-WebUI Premium (SECourses), modules/whisper/convrot at revision 613beba (September 2026).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import tokenizers
from faster_whisper import WhisperModel
from faster_whisper.feature_extractor import FeatureExtractor
from faster_whisper.utils import get_logger

from .engine import ConvRotWhisper


class ConvRotFasterWhisperModel(WhisperModel):
    """Same public API as ``faster_whisper.WhisperModel``; ``self.model`` is a ``ConvRotWhisper``."""

    def __init__(self, model_size_or_path: str, device: str = "cuda", device_index: int = 0,
                 compute_type: str = "int8_convrot", engine: Optional[ConvRotWhisper] = None,
                 use_cuda_graphs: bool = True, **_ignored):
        self.logger = get_logger()
        model_path = model_size_or_path
        self.model = engine or ConvRotWhisper(model_path, device=device, device_index=device_index,
                                              compute_type=compute_type, use_cuda_graphs=use_cuda_graphs)
        tokenizer_file = os.path.join(model_path, "tokenizer.json")
        self.hf_tokenizer = tokenizers.Tokenizer.from_file(tokenizer_file)
        self.feat_kwargs = self._get_feature_kwargs(model_path, None)
        self.feature_extractor = FeatureExtractor(**self.feat_kwargs)
        self.input_stride = 2
        self.num_samples_per_token = self.feature_extractor.hop_length * self.input_stride
        self.frames_per_second = self.feature_extractor.sampling_rate // self.feature_extractor.hop_length
        self.tokens_per_second = self.feature_extractor.sampling_rate // self.num_samples_per_token
        self.time_precision = 0.02
        self.max_length = 448

    def encode(self, features: np.ndarray):
        if features.ndim == 2:
            features = np.expand_dims(features, 0)
        return self.model.encode(np.ascontiguousarray(features, dtype=np.float32), to_cpu=False)
