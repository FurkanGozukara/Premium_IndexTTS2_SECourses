"""Qwen2.5-Omni Thinker conditioning for AuK (text instruction plus optional audio).

Upstream builds ChatML messages, lets ``qwen_omni_utils`` reload each audio path
with ``librosa.load(sr=16000)`` and keeps every hidden layer of the Thinker. Here
the caller supplies the audio already loaded; resampling uses the same librosa
default (soxr_hq), so the encoder sees identical input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from . import ENCODER_SAMPLE_RATE

NO_PROMPT_AUDIO = "|<no_prompt_audio>|"


def user_message(instruction: str, with_audio: bool) -> list[dict]:
    """One single-turn conversation in upstream's content order (text, then audio).

    Without audio the instruction carries upstream's ``|<no_prompt_audio>|`` marker,
    exactly as both the inference engine and the trainer append it.
    """
    text = str(instruction)
    if not with_audio and not text.endswith(NO_PROMPT_AUDIO):
        text += NO_PROMPT_AUDIO
    content = [{"type": "text", "text": text}]
    if with_audio:
        content.append({"type": "audio", "audio": "<audio>"})
    return [{"role": "user", "content": content}]


def to_encoder_rate(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Mono float32 audio at the Thinker's 16 kHz, resampled like ``librosa.load``."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=0 if audio.shape[0] <= 8 else -1)
    if int(sample_rate) != ENCODER_SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=int(sample_rate), target_sr=ENCODER_SAMPLE_RATE, res_type="soxr_hq")
    return np.ascontiguousarray(audio, dtype=np.float32)


class _HiddenOnly(torch.nn.Module):
    """Stands in for the Thinker's vocabulary head: AuK reads hidden states only."""

    def forward(self, hidden_states):
        return hidden_states[..., :0]


class QwenConditioner:
    """The frozen Qwen2.5-Omni Thinker (text and audio encoder; no vision tower)."""

    def __init__(self, model_path: str | Path, device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                 attn_implementation: str = "sdpa"):
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

        from transformers.utils import logging as hf_logging

        path = str(model_path)
        self.device = torch.device(device)
        # The full Qwen2.5-Omni snapshot also holds the talker and speech decoder, which AuK
        # does not use: their skipped weights and config checks are expected, not worth a report.
        verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            self.processor = Qwen2_5OmniProcessor.from_pretrained(path)
            thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                path, dtype=dtype, attn_implementation=attn_implementation, low_cpu_mem_usage=True)
        finally:
            hf_logging.set_verbosity(verbosity)
        if getattr(thinker, "visual", None) is not None:
            del thinker.visual
            thinker.visual = None
        thinker.lm_head = _HiddenOnly()
        thinker.requires_grad_(False).eval()
        self.model = thinker.to(self.device)
        self.num_layers = int(thinker.config.text_config.num_hidden_layers)
        self.hidden_size = int(thinker.config.text_config.hidden_size)

    def to(self, device):
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def inputs(self, instructions: Sequence[str], audios: Sequence[np.ndarray | None]):
        """Processor inputs for a batch; ``audios[i]`` is 16 kHz mono audio or None."""
        conversations = [user_message(text, audio is not None) for text, audio in zip(instructions, audios)]
        formatted = self.processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
        clips = [audio for audio in audios if audio is not None]
        kwargs = {"text": formatted, "padding": True, "return_tensors": "pt"}
        if clips:
            kwargs.update(audio=clips, use_audio_in_video=True)
        return self.processor(**kwargs)

    @torch.no_grad()
    def hidden_states(self, inputs):
        """All hidden layers (embeddings first) and the token mask of processor inputs.

        Upstream encodes inside its BF16 autocast region (inference and training),
        which also feeds the float32 audio features to the BF16 audio tower.
        """
        inputs = inputs.to(self.device)
        cuda = self.device.type == "cuda"
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
            outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
        return outputs.hidden_states, inputs["attention_mask"].bool()

    def unload(self):
        self.model = None
