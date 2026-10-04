"""Tencent AuK behind the application's shared generation/adapter contract.

AuK is a flow-matching transformer conditioned on a natural-language instruction
(encoded with the Qwen2.5-Omni Thinker) and, optionally, a reference recording
(prepended as VAE latents). Speech length is not predicted by the model: every
section is generated at a duration estimated here from the text, the reference
speaker's measured pace, or a trained voice's calibrated pace.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from indextts.auk import AUK_REPO, LATENT_RATE, MAX_CONTEXT_SECONDS, QWEN_OMNI_REPO, SAMPLE_RATE
from indextts.runtime.vram_presets import RuntimeConfig
from indextts.utils.audio_plan import assemble_audio_plan, fit_target_samples, trim_segment_silence
from indextts.utils.pause_tags import PauseChunk, TextChunk, split_text_with_pauses
from indextts.utils.pronunciation import plain_readings
from indextts.utils.text_segmentation import ends_sentence, normalize_sentence_whitespace, split_text_by_tokens

MODEL_REPO = AUK_REPO
TEXT_ENCODER_REPO = QWEN_OMNI_REPO
QUANT_REPO = "MonsterMMORPG/Wan_GGUF"
QUANT_FILES = {
    "dit": "AuK/auk_dit_convrot_int8.safetensors",
    "text": "AuK/qwen_omni_thinker_convrot_int8.safetensors",
}
ROOT = Path(__file__).resolve().parents[2]
REFERENCE_CACHE_DIR = ROOT / ".ui_state" / "auk_references"
REFERENCE_CACHE_VERSION = 1
HOP = SAMPLE_RATE // LATENT_RATE  # 480 samples per latent frame

from indextts.auk.text import (  # noqa: F401  (re-exported for callers of the engine module)
    CLONE_TEMPLATE, DESIGN_TEMPLATE, DESIGN_TEMPLATE_ZH, GENERATION_DEFAULTS, SECONDS_PER_BYTE, SHORT_TEXT_BYTES,
    SHORT_TEXT_SPEED, TRAINED_VOICE_DESCRIPTION, build_instruction, detect_language, f5_seconds, normalize_auk_text,
    preset_guidance, quote_text, text_units, trained_voice, validate_voice_settings, voice_template,
)

SEGMENT_SECONDS_LIMIT = MAX_CONTEXT_SECONDS


def speech_span(audio: np.ndarray, sample_rate: int, gate_db: float = -40.0) -> float:
    """Seconds from the first to the last frame within ``gate_db`` of the peak (a VAD span)."""
    frame = max(1, int(sample_rate * 0.01))
    rms = _quiet_frames(audio, frame)
    if not len(rms) or float(rms.max()) <= 0:
        return len(audio) / sample_rate
    loud = np.nonzero(rms >= float(rms.max()) * 10 ** (gate_db / 20.0))[0]
    return (int(loud[-1]) + 1 - int(loud[0])) * frame / sample_rate if len(loud) else len(audio) / sample_rate


def ensure_model(model_dir, *, dit_variant="bf16", text_variant="bf16", progress=None):
    """Download public weights on demand, retaining Hugging Face's resume cache."""
    from huggingface_hub import hf_hub_download, snapshot_download

    from indextts.auk.thinker_files import SLIM_REPO_FOLDER, full_folder, is_complete, slim_folder

    root = Path(model_dir)
    folder = root / "auk"
    if not all((folder / name).is_file() for name in ("config.yaml", "auk_base.safetensors", "vae.safetensors")):
        print(f">> Downloading public AuK model to {folder}", flush=True)
        if progress:
            progress(0, desc="Downloading AuK weights")
        snapshot_download(MODEL_REPO, local_dir=str(folder), allow_patterns=["*.safetensors", "config.yaml", "LICENSE"])
    # AuK runs only the Thinker (text model and audio tower): the slim folder (7.5 GB) gives
    # bitwise-identical conditioning to the 12 GB public snapshot, which keeps working when present.
    qwen = slim_folder(root)
    if not is_complete(qwen):
        if is_complete(full_folder(root)):
            qwen = full_folder(root)
        else:
            print(f">> Downloading the Qwen2.5-Omni-3B Thinker (text and audio encoder, 7.5 GB) to {qwen}", flush=True)
            if progress:
                progress(0, desc="Downloading the Qwen2.5-Omni-3B Thinker encoder")
            try:
                snapshot_download(QUANT_REPO, local_dir=str(root / "quantized"), allow_patterns=[f"{SLIM_REPO_FOLDER}/*"])
            except Exception as exc:  # the slim folder is unavailable: fall back to the public snapshot
                print(f">> Slim Thinker download failed ({exc}); downloading the full Qwen2.5-Omni-3B snapshot", flush=True)
            if not is_complete(qwen):
                qwen = full_folder(root)
                snapshot_download(TEXT_ENCODER_REPO, local_dir=str(qwen))
    quantized = {}
    for part, wanted in (("dit", dit_variant), ("text", text_variant)):
        if wanted != "int8_convrot":
            continue
        path = root / "quantized" / QUANT_FILES[part]
        if not path.is_file():
            print(f">> Downloading AuK ConvRot INT8 weights ({part})", flush=True)
            path = Path(hf_hub_download(QUANT_REPO, QUANT_FILES[part], local_dir=str(root / "quantized")))
        quantized[part] = path
    return folder, qwen, quantized


def _quiet_frames(audio: np.ndarray, frame: int) -> np.ndarray:
    count = len(audio) // frame
    if count == 0:
        return np.zeros(0, dtype=np.float32)
    frames = audio[: count * frame].reshape(count, frame)
    return np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)


def trim_silence(audio: np.ndarray, sample_rate: int, *, gate_db: float = -40.0, margin_s: float = 0.1) -> np.ndarray:
    """Trim leading and trailing audio quieter than ``gate_db`` below the peak, keeping a margin."""
    frame = max(1, int(sample_rate * 0.01))
    rms = _quiet_frames(audio, frame)
    if not len(rms):
        return audio
    peak = float(rms.max())
    if peak <= 0:
        return audio
    loud = np.nonzero(rms >= peak * 10 ** (gate_db / 20.0))[0]
    if not len(loud):
        return audio
    margin = int(margin_s * sample_rate)
    start = max(0, int(loud[0]) * frame - margin)
    end = min(len(audio), (int(loud[-1]) + 1) * frame + margin)
    return audio[start:end]


def cut_at_pause(audio: np.ndarray, sample_rate: int, max_seconds: float) -> np.ndarray:
    """Shorten audio to at most ``max_seconds``, cutting at the quietest point of the last 3 seconds."""
    limit = int(max_seconds * sample_rate)
    if len(audio) <= limit:
        return audio
    frame = max(1, int(sample_rate * 0.05))
    window_start = max(0, limit - 3 * sample_rate)
    rms = _quiet_frames(audio[window_start:limit], frame)
    if not len(rms):
        return audio[:limit]
    cut = window_start + int(np.argmin(rms)) * frame + frame // 2
    return audio[: max(frame, cut)]


def split_at_pauses(audio: np.ndarray, sample_rate: int, max_seconds: float) -> list[np.ndarray]:
    """Cut long audio into pieces of at most ``max_seconds``, each ending at the quietest
    point of its last 3 seconds (the same rule that shortens references)."""
    pieces, rest = [], audio
    limit = int(max_seconds * sample_rate)
    while len(rest) > limit:
        piece = cut_at_pause(rest, sample_rate, max_seconds)
        pieces.append(piece)
        rest = rest[len(piece):]
    if len(rest):
        pieces.append(rest)
    return pieces


def read_audio(path) -> tuple[np.ndarray, int]:
    """Mono float32 audio at its native rate; non-WAV containers go through librosa/FFmpeg."""
    try:
        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
    except Exception:
        import librosa

        audio, sample_rate = librosa.load(str(path), sr=None, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"The audio file is empty or damaged: {Path(path).name}")
    return audio, int(sample_rate)


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Upstream resamples the VAE input with torchaudio's windowed sinc; this is the same filter."""
    if int(source_rate) == int(target_rate):
        return np.asarray(audio, dtype=np.float32)
    try:
        import torchaudio.functional as AF

        tensor = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)
        return AF.resample(tensor, int(source_rate), int(target_rate)).squeeze(0).numpy()
    except Exception:
        import librosa

        return librosa.resample(audio, orig_sr=int(source_rate), target_sr=int(target_rate), res_type="soxr_hq")


def _rms(audio) -> float:
    array = audio.detach().float().cpu().numpy() if isinstance(audio, torch.Tensor) else np.asarray(audio, dtype=np.float32)
    array = array.reshape(-1)
    return float(np.sqrt(np.mean(array * array) + 1e-12)) if array.size else 0.0


def _park_recognizer():
    """The built-in Whisper keeps its weights on the GPU after a job; AuK's memory budget needs them back."""
    from indextts.asr import park

    park()


PEAK_CEILING = 0.99


def _limit_peak(audio: torch.Tensor, ceiling: float = PEAK_CEILING) -> torch.Tensor:
    """Scale a take down when it would exceed full scale: a restored or louder take keeps its
    shape instead of being hard-clipped by the 16-bit conversion."""
    peak = float(audio.abs().max()) if audio.numel() else 0.0
    return audio * (ceiling / peak) if peak > ceiling else audio


class OffloadedModule:
    """Keep a frozen module in pinned CPU memory and lend it to the GPU while it runs.

    Weights never change, so offloading only rebinds every tensor to its pinned
    CPU copy (no device-to-host copy); loading is one asynchronous host-to-device
    copy per tensor.
    """

    def __init__(self, module: torch.nn.Module, device, label="text encoder"):
        self.module, self.device, self.label = module, torch.device(device), label
        self.cpu = {}
        for owner in module.modules():
            for collection in (owner._parameters, owner._buffers):
                for name, tensor in collection.items():
                    if tensor is not None and tensor.device.type == "cpu":
                        pinned = tensor.detach().pin_memory() if torch.cuda.is_available() else tensor.detach()
                        self.cpu[(id(owner), name)] = pinned
                        collection[name] = torch.nn.Parameter(pinned, requires_grad=False) \
                            if isinstance(tensor, torch.nn.Parameter) else pinned
        self.active = False

    def activate(self, active: bool):
        if bool(active) == self.active:
            return
        started = time.perf_counter()
        for owner in self.module.modules():
            for collection in (owner._parameters, owner._buffers):
                for name, tensor in list(collection.items()):
                    pinned = self.cpu.get((id(owner), name))
                    if pinned is None or tensor is None:
                        continue
                    value = pinned.to(self.device, non_blocking=True) if active else pinned
                    collection[name] = torch.nn.Parameter(value, requires_grad=False) \
                        if isinstance(tensor, torch.nn.Parameter) else value
        if active:
            torch.cuda.synchronize(self.device)
        else:
            torch.cuda.empty_cache()
        self.active = bool(active)
        print(f">> AuK {self.label} {'on GPU' if active else 'offloaded'} in {time.perf_counter() - started:.2f}s", flush=True)


class PreparedReference:
    """A reference cut for the model: 24 kHz for the VAE, 16 kHz for the Qwen audio encoder."""

    def __init__(self, audio24: np.ndarray, audio16: np.ndarray, transcript: str, speech_seconds: float,
                 pace: float | None, source: str):
        self.audio24, self.audio16 = audio24, audio16
        self.transcript, self.speech_seconds = transcript, speech_seconds
        self.pace, self.source = pace, source
        self.seconds = len(audio24) / SAMPLE_RATE
        self.rms = _rms(audio24)
        self.latent = None


class AukEngine:
    sampling_rate = SAMPLE_RATE
    model_id = "auk"

    def __init__(self, model_dir="models", runtime=None, progress_callback=None):
        from indextts.auk.conditioning import QwenConditioner
        from indextts.auk.loader import build_model, build_vae, read_config, read_state

        self.runtime = RuntimeConfig.from_dict(runtime)
        self.device = self.runtime.device
        if self.device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        cuda = str(self.device).startswith("cuda")
        if cuda:
            torch.set_num_threads(min(torch.get_num_threads(), 8))
        self.dtype = torch.bfloat16 if cuda and self.runtime.gpt_dtype != "fp32" else torch.float32
        self.low_vram = self._runtime_low_vram = False
        self.progress_reporter = self.gr_progress = None
        self.last_generation_stats = {}
        self._lora_path, self._lora_strength, self._lora_merged = "", 1.0, False
        self._reference_key, self._reference = None, None
        self._voice = {}
        self.text_residency = self.runtime.auk_text_encoder_residency if cuda else "gpu"
        folder, qwen_dir, quantized = ensure_model(
            model_dir, dit_variant=self.runtime.model_variant, text_variant=self.runtime.auk_text_encoder_variant,
            progress=progress_callback)
        self.model_dir, self.folder = Path(model_dir), folder
        started = time.perf_counter()
        print(f">> Loading AuK | {self.device} | {self.dtype} | transformer {self.runtime.model_variant} | "
              f"text encoder {self.runtime.auk_text_encoder_variant} ({self.text_residency})", flush=True)
        self.config = read_config(folder)
        text_device = self.device if self.text_residency == "gpu" else "cpu"
        # The low-VRAM configurations (INT8 or on-demand text encoder) also keep the 0.6 GB token
        # table in CPU memory; the BF16 GPU-resident path stays exactly as upstream loads it.
        self.conditioner = QwenConditioner(qwen_dir, device=text_device, dtype=torch.bfloat16 if cuda else torch.float32,
                                           attn_implementation=self.runtime.attention_backend,
                                           int8_path=quantized.get("text"),
                                           cpu_embeddings=cuda and ("text" in quantized or self.text_residency != "gpu"))
        self._text_offload = OffloadedModule(self.conditioner.model, self.device) if self.text_residency != "gpu" else None
        num_layers = self.conditioner.num_layers
        # On demand, the GPU holds one model at a time: the text encoder while encoding and the
        # transformer while sampling, so the peak is the larger of the two instead of their sum.
        dit_device = self.device if self._text_offload is None else "cpu"
        if "dit" in quantized:
            from indextts.auk.loader import build_int8_model
            from indextts.quant.convrot_int8 import set_kernel_mode

            self.model = build_int8_model(self.config, quantized["dit"], device=dit_device, num_text_layers=num_layers)
            # W8A16 is faster than W8A8 for AuK's shapes (and never calls cuBLASLt's int8 GEMM).
            set_kernel_mode(self.model, "w8a16")
        else:
            state = read_state(folder / "auk_base.safetensors")
            self.model = build_model(self.config, state, device=dit_device, dtype=self.dtype, num_text_layers=num_layers)
            del state
        # The layer fusion runs right after encoding; its two small tensors stay on the GPU.
        for name in ("layer_weights", "layer_scale"):
            getattr(self.model, name).data = getattr(self.model, name).data.to(self.device)
        self._dit_offload = None
        # The VAE encodes references and decodes while the transformer samples, so it is lent with it.
        self.vae = build_vae(self.config, folder / "vae.safetensors", device=dit_device)
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
        print(f">> AuK ready in {time.perf_counter() - started:.1f}s", flush=True)
        self.set_lora(self.runtime.lora_path, self.runtime.lora_strength, merge_into_base=self.runtime.lora_merge_into_base)

    # ------------------------------------------------------------------ residency

    def _residency(self, phase: str):
        """Lend the GPU to the text encoder ("text"), the transformer ("dit") or neither ("idle", for
        Whisper); a no-op when both stay resident."""
        if self._text_offload is None:
            return
        lent = {"text": self._text_offload, "dit": self._dit_offload}.get(phase)
        for offload in (self._text_offload, self._dit_offload):
            if offload is not None and offload is not lent:
                offload.activate(False)
        if lent is not None:
            lent.activate(True)
        self.conditioner.device = torch.device(self.device if phase == "text" else "cpu")

    # ------------------------------------------------------------------ adapters

    def set_lora(self, path, strength=1.0, *, merge_into_base=False, **_kwargs):
        from indextts.lora import apply_lora, inspect_lora, merge_lora_for_inference, remove_lora

        path = os.path.abspath(str(path)) if path else ""
        metadata = {}
        if path:
            metadata = inspect_lora(path)
            base = str(metadata.get("base_model", "")).lower()
            if "auk" not in base:
                owner = "OmniVoice" if "omnivoice" in base else "IndexTTS"
                raise ValueError(f"This adapter belongs to {owner}. Select an AuK adapter or clear the adapter selection.")
            if metadata.get("adapter_type") == "full" and self.runtime.model_variant != "bf16":
                raise ValueError("Select BF16 before loading a full fine-tuning checkpoint.")
        if self._dit_offload is not None:
            # Adapters change the pinned weights themselves, which are re-pinned afterwards.
            self._dit_offload.activate(False)
        remove_lora(self.model)
        if path:
            try:
                apply_lora(self.model, path, strength=float(strength))
            except KeyError as exc:
                if "full-module tensor" not in str(exc):
                    raise
                remove_lora(self.model)
                # Adapters from builds before 7.1 trained these layers in the base's own precision.
                other = "BF16" if self.runtime.model_variant != "bf16" else "ConvRot INT8"
                raise ValueError("This adapter was trained by an earlier build with input and output layers stored "
                                 f"for the other transformer precision. Select the {other} transformer to use it, "
                                 "or train it again.") from exc
            if merge_into_base:
                merge_lora_for_inference(self.model)
        self._lora_path, self._lora_strength, self._lora_merged = path, float(strength), bool(merge_into_base)
        self.model.eval().requires_grad_(False)
        if self._text_offload is not None:
            self._dit_offload = OffloadedModule(torch.nn.ModuleList([self.model, self.vae]), self.device,
                                                label="transformer and VAE")
        self._voice = trained_voice(path)

    # ------------------------------------------------------------------ text

    def split_text_by_tokens(self, text, max_tokens, lang_prefix="", *, mode="budget", target_tokens=None):
        tokenizer = self.conditioner.processor.tokenizer
        return split_text_by_tokens(text, max_tokens, capacity=4096,
            token_len=lambda value: len(tokenizer.encode(value, add_special_tokens=False)),
            mode=mode, target_tokens=target_tokens, segment_budget_scale_non_cjk=1.0)

    # ------------------------------------------------------------------ reference

    def prepare_reference(self, path, settings, language="auto") -> PreparedReference:
        """Load, trim and transcribe a reference once; later calls reuse it until the file changes."""
        stat = Path(path).stat()
        transcript_file = Path(path).with_suffix(".txt")
        transcript = str(settings.get("reference_text") or "").strip()
        if not transcript and transcript_file.is_file():
            transcript = transcript_file.read_text(encoding="utf-8-sig").strip()
        key = (str(Path(path).resolve()), stat.st_mtime_ns, stat.st_size, transcript,
               float(settings["max_reference_seconds"]), bool(settings["trim_reference_silence"]))
        if key == self._reference_key and self._reference is not None:
            return self._reference
        audio, rate = read_audio(path)
        if settings["trim_reference_silence"]:
            audio = trim_silence(audio, rate)
        speech_seconds = speech_span(audio, rate)
        if not transcript:
            transcript = self._cached_transcript(path, audio, rate, language)
        transcript = normalize_auk_text(transcript, language)
        # The reference speaker's pace relative to upstream's byte model (upstream scales
        # the reference's speech span by the target/reference weighted-byte ratio).
        reference_f5 = f5_seconds(transcript, language) if transcript else 0.0
        pace = speech_seconds / reference_f5 if reference_f5 > 0.3 and speech_seconds > 0.5 else None
        audio = cut_at_pause(audio, rate, float(settings["max_reference_seconds"]))
        audio24 = resample(audio, rate, SAMPLE_RATE)
        audio24 = audio24[: len(audio24) // HOP * HOP]
        from indextts.auk.conditioning import to_encoder_rate

        audio16 = to_encoder_rate(audio, rate)
        reference = PreparedReference(audio24, audio16, transcript, speech_seconds, pace, str(path))
        self._reference_key, self._reference = key, reference
        return reference

    def _cached_transcript(self, path, audio, rate, language) -> str:
        """Transcribe a reference with the app's Whisper model once, cached by audio content."""
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        digest.update(json.dumps([REFERENCE_CACHE_VERSION]).encode("utf-8"))
        cached = REFERENCE_CACHE_DIR / f"{digest.hexdigest()[:32]}.json"
        try:
            return str(json.loads(cached.read_text(encoding="utf-8"))["transcript"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
        from indextts.training.whisper_asr import transcribe

        lang = str(language or "auto").lower()
        lang = lang if lang in {"en", "zh"} else "en"
        print(f">> Transcribing reference {Path(path).name} once with Whisper (pace estimate)", flush=True)
        try:
            transcript = transcribe(resample(audio, rate, 16000), sr=16000, language=lang,
                                    device=self._whisper_device()).text.strip()
        finally:
            _park_recognizer()
        try:
            REFERENCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            temporary = cached.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps({"transcript": transcript, "source": Path(path).name}), encoding="utf-8")
            os.replace(temporary, cached)
        except OSError as exc:
            print(f">> AuK reference transcript cache not written ({exc})", flush=True)
        print(f">> Reference transcript: {transcript}", flush=True)
        return transcript

    def _reference_latent(self, reference: PreparedReference):
        if reference.latent is None:
            audio = torch.from_numpy(reference.audio24).to(self.device).view(1, 1, -1)
            with torch.inference_mode():
                latent, _ = self.vae.encode(audio)
            reference.latent = latent[0]
        return reference.latent

    # ------------------------------------------------------------------ duration

    def pace(self, reference: PreparedReference | None) -> tuple[float, str]:
        """Speech time per unit of upstream's byte model for the active voice, and its source."""
        if reference is not None and reference.pace:
            return float(min(3.0, max(0.33, reference.pace))), "reference"
        voice_pace = self._voice.get("pace")
        if voice_pace:
            return float(min(3.0, max(0.33, float(voice_pace)))), "trained voice"
        return 1.0, "default"

    @staticmethod
    def estimate_seconds(text, language, pace, *, paced_by_reference=False, edge_seconds=0.0,
                         duration_factor=1.0, max_seconds=None) -> float:
        """Target length of one section, on the 50 Hz latent grid."""
        seconds = f5_seconds(text, language) * float(pace)
        if text_units(text) < SHORT_TEXT_BYTES:
            # Upstream slows very short text to 0.3x; with a reference that leaves a long
            # padded slot, so cloned short text keeps its pace with a one-second floor.
            seconds = max(1.0, seconds) if paced_by_reference else seconds / SHORT_TEXT_SPEED
        seconds = seconds * float(duration_factor) + float(edge_seconds)
        seconds = min(max(0.4, seconds), float(max_seconds or SEGMENT_SECONDS_LIMIT))
        return math.ceil(seconds * LATENT_RATE) / LATENT_RATE

    def _fit_context(self, segments, max_tokens, budget, language, pace, paced_by_reference, edge_seconds,
                     duration_factor):
        """Split again every section whose speech would not fit the context beside the reference.

        AuK renders a section in the context the reference leaves; a longer estimate would be
        clamped, which rushes the speech (word errors rose from 2 % to 8-54 % in docs/AUK.md).
        """
        fitted = []
        for segment in segments:
            seconds = self.estimate_seconds(segment, language, pace, paced_by_reference=paced_by_reference,
                                            edge_seconds=edge_seconds, duration_factor=duration_factor,
                                            max_seconds=10 * SEGMENT_SECONDS_LIMIT)
            tokens = int(max_tokens * budget / seconds) if seconds > budget else 0
            parts = [part for part in self.split_text_by_tokens(segment, max(4, tokens)) if part.strip()] if tokens else []
            if len(parts) > 1 and tokens < max_tokens:
                fitted.extend(self._fit_context(parts, max(4, tokens), budget, language, pace, paced_by_reference,
                                                edge_seconds, duration_factor))
            else:
                fitted.append(segment)
        return fitted

    # ------------------------------------------------------------------ synthesis core

    def _encode(self, instructions, audios16):
        with torch.inference_mode():
            inputs = self.conditioner.inputs(instructions, audios16)
            hidden, mask = self.conditioner.hidden_states(inputs)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(self.device).startswith("cuda")):
                fused = self.model.fuse([item.to(self.device) for item in hidden])
        return fused.float(), mask.to(self.device)

    def _sample(self, text, mask, ref_latents, ref_lens, target_lens, settings, seeds, step_callback=None):
        batch = text.shape[0]
        max_target = int(max(target_lens))
        noise = torch.zeros(batch, max_target, self.model.num_channels, device=self.device)
        for index, seed in enumerate(seeds):
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
            noise[index, : target_lens[index]] = torch.randn(int(target_lens[index]), self.model.num_channels,
                                                             device=self.device, generator=generator)
        with torch.inference_mode(), torch.autocast("cuda", dtype=self.dtype, enabled=str(self.device).startswith("cuda")):
            return self.model.sample(
                text, mask, ref_latents, ref_lens.to(self.device), torch.tensor(target_lens, device=self.device),
                steps=int(settings["num_step"]), cfg_strength=float(settings["guidance_scale"]),
                sway_sampling_coef=float(settings["sway_coef"]), noise=noise,
                method=str(settings.get("solver") or "euler"), step_callback=step_callback)

    def _decode(self, latent, frames):
        with torch.inference_mode():
            audio = self.vae.decode(self.vae.denormalize(latent[None, :frames]).permute(0, 2, 1))
        audio = audio.reshape(1, -1).float().cpu()
        if not audio.numel() or not torch.isfinite(audio).all():
            raise RuntimeError("AuK returned empty or non-finite audio.")
        return audio

    def encode_batches(self, batches):
        """Qwen conditioning of each (instructions, references) batch, kept on the CPU when lent on demand.

        With on-demand residency the text encoder is lent once for every batch, then the transformer.
        """
        self._residency("text")
        encoded = []
        for instructions, references in batches:
            text, mask = self._encode(instructions, [item.audio16 if item is not None else None for item in references])
            encoded.append((text.cpu(), mask.cpu()) if self._text_offload is not None else (text, mask))
        self._residency("dit")
        return encoded

    def generate_batch(self, instructions, references, seconds, settings, seeds, progress=None, conditioning=None):
        """Render one batch: instructions[i] with optional references[i] at seconds[i]."""
        text, mask = conditioning if conditioning is not None else self.encode_batches([(instructions, references)])[0]
        text, mask = text.to(self.device), mask.to(self.device)
        latents = [self._reference_latent(reference) if reference is not None else None for reference in references]
        ref_frames = [int(item.shape[0]) if item is not None else 0 for item in latents]
        ref_latents = torch.zeros(len(references), max(ref_frames), self.model.num_channels, device=self.device)
        for index, item in enumerate(latents):
            if item is not None:
                ref_latents[index, : ref_frames[index]] = item
        target_lens = [max(1, int(math.ceil(value * LATENT_RATE))) for value in seconds]
        generated = self._sample(text, mask, ref_latents, torch.tensor(ref_frames), target_lens, settings, seeds, progress)
        return [self._decode(generated[index], target_lens[index]) for index in range(len(instructions))]

    # ------------------------------------------------------------------ public API

    def infer(self, spk_audio_prompt, text, output_path=None, lang="EN", **kwargs):
        result = self.infer_texts(spk_audio_prompt, [text], lang=lang, **kwargs)[0]
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, result[1], result[0], subtype="PCM_16")
            return str(output_path)
        return result

    def infer_texts(self, spk_audio_prompt, texts, lang="EN", *, auk=None,
              max_text_tokens_per_segment=60, interval_silence=200, sentence_pause_ms=0,
              segmentation_mode="budget", segment_target_tokens=None, enable_pause_tags=True,
              text_normalization=True, duration_factor=1.0, seed=None, target_duration_s=None,
              target_duration_mode="off", section_batch_size=1, on_text_complete=None,
              trim_silence_ms_threshold=0, **_kwargs):
        started = time.perf_counter()
        cuda = str(self.device).startswith("cuda")
        if cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        settings = {**GENERATION_DEFAULTS, **(auk or {})}
        if "guidance_scale" not in (auk or {}):
            settings["guidance_scale"] = preset_guidance(settings["mode"])  # a request without its own guidance
        validate_voice_settings(settings)
        if target_duration_mode not in {"off", "natural", "pad", "trim"}:
            raise ValueError("Unknown target duration mode")
        mode = settings["mode"]
        language = str(lang or "auto").lower()
        if language not in {"en", "zh"}:
            language = detect_language(" ".join(texts))
        reference = None
        if mode == "clone":
            if not spk_audio_prompt or not Path(spk_audio_prompt).is_file():
                raise ValueError("Voice cloning requires reference audio. Choose Auto voice or Voice design to generate without it.")
            reference = self.prepare_reference(spk_audio_prompt, settings, language)
        pace, pace_source = self.pace(reference)
        description = settings["voice_description"] if mode == "design" else self._voice.get("description", "")
        # Upstream trains up to 30 s on each side, so a section may use the whole context whatever the
        # reference's length; subtracting the reference only rushed the speech (docs/AUK.md).
        budget = SEGMENT_SECONDS_LIMIT

        plans, speech, owners, natural = [], [], [], []
        for text_index, text in enumerate(texts):
            plan, first = [], len(speech)
            chunks = split_text_with_pauses(text) if enable_pause_tags else [TextChunk(text)]
            for chunk in chunks:
                if isinstance(chunk, PauseChunk):
                    plan.append(("pause", round(chunk.duration_s * self.sampling_rate)))
                    continue
                # Pronunciation markup: AuK reads plain text, so respellings replace words and phonemes are dropped.
                source = plain_readings(chunk.text).strip()
                if segmentation_mode != "budget":
                    source = normalize_sentence_whitespace(source)
                if text_normalization:
                    source = normalize_auk_text(source, language)
                segments = [part for part in self.split_text_by_tokens(source, max_text_tokens_per_segment,
                    mode=segmentation_mode, target_tokens=segment_target_tokens) if part.strip()]
                segments = self._fit_context(segments, max_text_tokens_per_segment, budget, language, pace,
                                             pace_source == "reference", settings["edge_seconds"], duration_factor)
                for index, segment in enumerate(segments):
                    plan.append(("segment", len(speech)))
                    speech.append(segment.strip())
                    owners.append(text_index)
                    if index < len(segments) - 1:
                        gap = sentence_pause_ms if sentence_pause_ms and ends_sentence(segment) else interval_silence
                        kind = "sentence_gap" if sentence_pause_ms and ends_sentence(segment) else "silence"
                        plan.append((kind, round(max(0, gap) * self.sampling_rate / 1000)))
            if len(speech) == first:
                raise ValueError("Enter some text to generate speech.")
            plans.append(plan)
            factor = float(duration_factor)
            estimated = [self.estimate_seconds(part, language, pace, paced_by_reference=pace_source == "reference",
                                               edge_seconds=settings["edge_seconds"], duration_factor=factor,
                                               max_seconds=budget) for part in speech[first:]]
            if target_duration_s and target_duration_mode == "natural":
                fixed = sum(value for kind, value in plan if kind != "segment") / self.sampling_rate
                scale = max(0.1, float(target_duration_s) - fixed) / max(1e-6, sum(estimated))
                estimated = [min(budget, max(0.6, value * scale)) for value in estimated]
            natural.extend(estimated)

        # A trained voice hears the wording it was trained with; the base model's Auto voice and
        # voice design use the canonical design wording.
        template = voice_template(language, self._voice.get("template", "")) if mode == "auto" and self._voice else ""
        instructions = [build_instruction(part, mode, description, language, template) for part in speech]
        base_seed = int(seed) if seed is not None else int(torch.randint(0, 2**31 - 1, (1,)).item())
        batch_size = 1 if getattr(self, "low_vram", False) else max(1, int(section_batch_size))
        total = math.ceil(len(speech) / batch_size)
        rendered, durations = [None] * len(speech), []
        results, protected_by_text = [None] * len(texts), {}

        def complete(text_index):
            plan = plans[text_index]
            if results[text_index] is not None or any(rendered[value] is None for kind, value in plan if kind == "segment"):
                return
            audio, protected = assemble_audio_plan(rendered, plan, self.sampling_rate)
            if target_duration_s and target_duration_mode in {"pad", "trim"}:
                audio = fit_target_samples(audio, round(float(target_duration_s) * self.sampling_rate), target_duration_mode)
            pcm = (audio.flatten().clamp(-1, 1) * 32767).round().to(torch.int16).numpy()
            results[text_index] = (self.sampling_rate, pcm)
            protected_by_text[text_index] = [(start / self.sampling_rate, end / self.sampling_rate) for start, end in protected]
            if on_text_complete:
                on_text_complete(text_index, results[text_index])

        prefetched = {}
        if self._text_offload is not None:
            # One loan of the text encoder for every batch, then one of the transformer.
            if self.progress_reporter:
                self.progress_reporter.update(0, total=total * max(1, int(settings["num_step"])), desc="AuK encoding the text")
            starts = list(range(0, len(speech), batch_size))
            encoded = self.encode_batches([(instructions[start:start + batch_size],
                                            [reference] * len(instructions[start:start + batch_size])) for start in starts])
            prefetched = dict(zip(starts, encoded))
        devices = [torch.device(self.device).index or 0] if cuda else []
        # Takes per section, as in the other speech models: fewest word errors, or the most similar take of all
        # renders without word errors (the extra takes render in batches with their own seeds).
        takes = max(1, int(getattr(self, "section_takes", 1) or 1))
        judge = getattr(self, "take_judge", None) if takes > 1 else None
        retakes = [0]

        def finish(audio):
            if settings.get("match_loudness") and reference is not None and reference.rms > 1e-4:
                audio = audio * min(4.0, reference.rms / max(1e-4, _rms(audio)))
            return trim_segment_silence(_limit_peak(audio), self.sampling_rate, trim_silence_ms_threshold)

        def render_more(index, count):
            more = []
            for begin in range(0, count, batch_size):
                size = min(batch_size, count - begin)
                more_seeds = [(base_seed + 7919 * index + 104729 * (retakes[0] + offset + 1)) % (2**31 - 1)
                              for offset in range(size)]
                retakes[0] += size
                more.extend(finish(audio) for audio in self.generate_batch(
                    [instructions[index]] * size, [reference] * size, [natural[index]] * size, settings, more_seeds))
            return more

        def samples(audio):
            return audio.detach().float().cpu().numpy().reshape(-1)

        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(base_seed)
            for start in range(0, len(speech), batch_size):
                batch_index = start // batch_size
                stop = min(len(speech), start + batch_size)

                def on_step(step, steps, batch_index=batch_index):
                    if self.progress_reporter:
                        self.progress_reporter.update(batch_index * steps + step, total=total * steps,
                            desc=f"AuK section batch {batch_index + 1}/{total}, step {step}/{steps}")

                if self.progress_reporter:
                    self.progress_reporter.update(batch_index * max(1, int(settings["num_step"])),
                        total=total * max(1, int(settings["num_step"])), desc=f"AuK encoding batch {batch_index + 1}/{total}")
                seeds = [(base_seed + 7919 * index) % (2**31 - 1) for index in range(start, stop)]
                audios = self.generate_batch(instructions[start:stop], [reference] * (stop - start), natural[start:stop],
                                             settings, seeds, progress=on_step, conditioning=prefetched.get(start))
                for offset, audio in enumerate(audios):
                    index = start + offset
                    take = finish(audio)
                    if judge is not None and getattr(judge, "compares_voices", lambda: False)():
                        from indextts.utils.take_selection import keep_most_similar_take

                        take, outcome = keep_most_similar_take(
                            take, lambda count, i=index: render_more(i, count),
                            lambda candidate: judge.similarity(samples(candidate), self.sampling_rate),
                            lambda candidate, i=index: judge.error_rate(speech[i], samples(candidate), self.sampling_rate),
                            takes, judge.checks)
                        judge.record_similar(index, outcome)
                    elif judge is not None:
                        from indextts.utils.take_selection import keep_best_take

                        take, rates, kept = keep_best_take(
                            take, lambda i=index: render_more(i, 1)[0],
                            lambda candidate, i=index: judge.error_rate(speech[i], samples(candidate), self.sampling_rate),
                            takes)
                        judge.record(index, rates, kept)
                    rendered[index] = take
                    durations.append(rendered[index].shape[-1] / self.sampling_rate)
                    complete(owners[index])
        self.last_generation_stats = {
            "segment_count": len(speech), "segments_count": len(speech),
            "total_duration_s": sum(len(result[1]) for result in results) / self.sampling_rate,
            "mean_duration_s": float(np.mean(durations)), "min_duration_s": min(durations),
            "max_duration_s": max(durations), "generation_time_s": time.perf_counter() - started,
            "model": self.model_id, "section_batch_size": batch_size,
            "pace": round(pace, 4), "pace_source": pace_source, "voice_mode": mode,
            "section_seconds": [round(value, 2) for value in natural],
            "peak_vram_gb": torch.cuda.max_memory_allocated(self.device) / 1024**3 if cuda else 0.0,
            "protected_pauses": protected_by_text.get(0, []) if len(texts) == 1 else [],
        }
        return results

    # Whisper conversion inputs: the paper normalizes them to -24 dBFS RMS (peak ceiling 0.95).
    WHISPER_INPUT_RMS = 10 ** (-24 / 20)
    EDIT_CHUNK_SECONDS = 24.0

    def edit_audio(self, source, task_key, values=None, *, instruction=None, transcript="", duration_mode="auto",
                   seconds=None, settings=None, seed=None, language="en"):
        """Apply one AuK editing, enhancement or separation task to a recording.

        ``source`` is a path or ``(audio, sample_rate)``. ``duration_mode``: "auto" (the
        task's rule), "source" (the source length) or "custom" (``seconds``). Same-length
        tasks on audio longer than a context run piece by piece, cut at pauses.
        Returns ``(24000, int16 pcm)``.
        """
        from indextts.auk.conditioning import to_encoder_rate
        from indextts.auk.tasks import TASKS, render_instruction, target_seconds

        task = TASKS[task_key]
        values = dict(values or {})
        settings = {**GENERATION_DEFAULTS, **(settings or {})}
        instruction = str(instruction or "").strip() or render_instruction(task_key, values, language)
        audio, rate = read_audio(source) if isinstance(source, (str, Path)) else (np.asarray(source[0], np.float32), int(source[1]))
        if audio.ndim > 1:
            audio = audio.mean(axis=0 if audio.shape[0] <= 8 else -1)
        full_seconds = len(audio) / rate
        base_seconds = full_seconds
        if task.trim:
            base_seconds = speech_span(audio, rate)
            audio = trim_silence(audio, rate)
        if task_key in {"to_whisper", "from_whisper"}:
            level = _rms(audio)
            if level > 1e-6:
                audio = audio * (self.WHISPER_INPUT_RMS / level)
                peak = float(np.abs(audio).max())
                if peak > 0.95:
                    audio = audio * (0.95 / peak)
        if duration_mode == "custom" and seconds:
            target = float(seconds)
        elif duration_mode == "source":
            target = len(audio) / rate
        else:
            if task.duration == "content" and not transcript and str(settings.get("transcribe_source", True)) != "False":
                transcript = self._transcribe_array(audio, rate, language)
            target = target_seconds(task_key, values, base_seconds, full_seconds, transcript)
        seed = int(seed) if seed is not None else int(torch.randint(0, 2**31 - 1, (1,)).item())
        started = time.perf_counter()
        long_source = len(audio) / rate > MAX_CONTEXT_SECONDS
        if long_source and not task.chunkable:
            raise ValueError(f"{task.label} works on up to {MAX_CONTEXT_SECONDS:.0f} seconds of audio; trim the source first.")
        pieces = split_at_pauses(audio, rate, self.EDIT_CHUNK_SECONDS) if long_source else [audio]
        scale = target / max(1e-6, len(audio) / rate)
        outputs = []
        devices = [torch.device(self.device).index or 0] if str(self.device).startswith("cuda") else []
        references = []
        for piece in pieces:
            audio24 = resample(piece, rate, SAMPLE_RATE)
            audio24 = audio24[: max(HOP, len(audio24) // HOP * HOP)]
            references.append(PreparedReference(audio24, to_encoder_rate(piece, rate), "", len(audio24) / SAMPLE_RATE,
                                                None, "edit source"))
        encoded = (self.encode_batches([([instruction], [reference]) for reference in references])
                   if self._text_offload is not None else [None] * len(pieces))
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            for index, (piece, reference) in enumerate(zip(pieces, references)):
                piece_seconds = target if len(pieces) == 1 else len(piece) / rate * scale

                def on_step(step, total, index=index):
                    if self.progress_reporter:
                        self.progress_reporter.update(index * total + step, total=len(pieces) * total,
                                                      desc=f"AuK edit, piece {index + 1}/{len(pieces)}, step {step}/{total}")

                outputs.append(self.generate_batch([instruction], [reference], [min(2 * MAX_CONTEXT_SECONDS,
                                                   max(0.3, piece_seconds))], settings, [seed + index], on_step,
                                                   conditioning=encoded[index])[0])
        result = _limit_peak(torch.cat(outputs, dim=-1) if len(outputs) > 1 else outputs[0])
        self.last_generation_stats = {
            "model": self.model_id, "task": task_key, "instruction": instruction, "pieces": len(pieces),
            "generation_time_s": time.perf_counter() - started, "total_duration_s": result.shape[-1] / SAMPLE_RATE,
            "target_seconds": round(target, 3), "source_seconds": round(full_seconds, 3), "transcript": transcript,
        }
        return SAMPLE_RATE, (result.flatten().clamp(-1, 1) * 32767).round().to(torch.int16).numpy()

    def _transcribe_array(self, audio, rate, language) -> str:
        from indextts.training.whisper_asr import transcribe

        lang = str(language or "en").lower()
        lang = lang if lang in {"en", "zh"} else "en"
        try:
            return transcribe(resample(audio, rate, 16000), sr=16000, language=lang,
                              device=self._whisper_device()).text.strip()
        finally:
            _park_recognizer()

    def _whisper_device(self) -> str:
        """Whisper runs beside AuK only where 3 GB stay free (on demand, both models step aside
        first); otherwise on the CPU, which is quick for a reference or one edit source."""
        if not str(self.device).startswith("cuda"):
            return "cpu"
        from indextts.training.whisper_asr import whisper_device_for_free_vram

        self._residency("idle")
        free, _total = torch.cuda.mem_get_info(torch.device(self.device))
        return whisper_device_for_free_vram(str(self.device), free / 1024**3, required_gb=3.0)

    def unload(self):
        self._reference = self._reference_key = None
        self.progress_reporter = self.gr_progress = None
        if getattr(self, "conditioner", None) is not None:
            self.conditioner.unload()
        self.model = self.vae = self.conditioner = self._text_offload = self._dit_offload = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
