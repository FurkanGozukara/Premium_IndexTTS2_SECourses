"""The app's built-in Whisper: one recognizer for every place the app listens to speech.

Whisper large-v3 in INT8 ConvRot runs on Whisper-WebUI Premium's engine (Triton INT8 kernels, CUDA graphs, beam
search on the GPU; ``indextts.asr.convrot``) through faster-whisper's decoding, with the settings Whisper-WebUI
measured as its best on 14 public English test sets ("Fast Whisper Best Quality", version 12.11): beam size 5,
word timestamps, the temperature fallback, the previous text as context except on English recordings over 30
seconds (where large-v3 repeats passages), and no words invented after the last 30-second window (the end-of-audio
tail guard, here applied only after a window that ended a sentence; see ``TAIL_GUARD_MODE``). The INT8 model
downloads once (1.6 GB) into ``models/hf_cache/whisper``; on the Open ASR Leaderboard's English sets it reached the
published accuracy of the full-precision model (7.22 % against 7.44 % average word error rate).

Measured against the Transformers large-v3-turbo the app used before (October 2026, the speaker's own narration):
the same word errors on 152 held-out recordings (164 of 4,702 words each), fewer on 320 generated takes with
known text (321 against 333 of 10,546; a lost final word in 3 takes and an extra one in 2, against 14 and 13)
and far fewer on 3-minute recordings, where turbo's 120-second chunks lost words at their joins (4.2 % against
11.1 %, twice as fast).

Between jobs the weights wait in RAM (``park``) so the speech models keep the VRAM; the next job moves them back
in about a second. GPUs older than the RTX 30 series, a missing Triton or a card without room for it use the
Transformers Whisper large-v3-turbo the app used before (on the CPU when the card is full).

Whisper transcribes in the language it is given; the app passes the speech model's own language
(``indextts.utils.speech_timestamps.whisper_language``), never Whisper's detection.
"""

from __future__ import annotations

import gc
import json
import math
import os
import shutil
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "models" / "hf_cache" / "whisper"
BUILTIN_MODEL = "large-v3-int8-convrot"
HOSTED_MODELS = {BUILTIN_MODEL: ("MonsterMMORPG/Wan_GGUF", "Whisper_INT8_ConvRot/large-v3-int8-convrot")}
BUILTIN_ALIASES = frozenset({BUILTIN_MODEL, "builtin", "built-in", "whisper-large-v3-int8-convrot"})
FALLBACK_MODEL = "openai/whisper-large-v3-turbo"
SAMPLE_RATE = 16000
# Free VRAM the engine needs to load: 1.6 GB of INT8 weights, the beam-search decoding session (KV caches and the
# cross-attention buffer) and the word-alignment pass.
REQUIRED_FREE_GB = 3.0

# Whisper-WebUI's "Fast Whisper Best Quality" preset for large-v3 INT8 ConvRot (12.11 benchmarks). faster-whisper's
# punctuation sets stand in for the preset's ASCII ones: the same for English, and they keep CJK punctuation on its
# word. Word timestamps are always on: they also place the next window (7.34 % against 7.54 % word errors without).
BEST_QUALITY = {
    "beam_size": 5, "best_of": 5, "patience": 1.0, "length_penalty": 1.0, "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0, "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0), "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0, "no_speech_threshold": 0.6, "prompt_reset_on_temperature": 0.5,
    "suppress_blank": True, "suppress_tokens": [-1], "max_initial_timestamp": 1.0, "word_timestamps": True,
    "chunk_length": 30, "vad_filter": False, "hallucination_silence_threshold": None,
}
ENGLISH_CONTEXT_LIMIT_S = 30.0
LONG_FORM_CONTEXT_WINDOWS = 60
# The end-of-audio tail guard (Whisper-WebUI 12.11): after the last window, a remainder of up to 2 s is decoded only
# when the voice detector hears 0.3 s of speech in it after its first half second (which holds the end of the last
# word). Decoding silence there invented "you", "Thank you." or a repeated sentence at the end of most files.
# "sentence" (this app) applies it only when the window's text ended a sentence: in its unpunctuated lowercase mode
# large-v3 sometimes stops a pass before the last words ("... but you can" for "... but you can try."), and the
# guard then dropped them. Measured on the narration sets above: Whisper-WebUI's rule ("always") 171 and 323 word
# errors, no guard ("off") 165 and 355 (invented endings such as "Taller." or a whole repeated sentence on
# generated takes), "sentence" 164 and 321.
TAIL_GUARD_S = 2.0
TAIL_IGNORED_S = 0.5
TAIL_SPEECH_S = 0.3
VAD_FRAME = 512
TAIL_GUARD_MODE = "sentence"
_SENTENCE_END = (".", "!", "?", "。", "！", "？", "…", "؟", "।")

_LOCK = threading.RLock()
_DOWNLOAD_LOCK = threading.Lock()
_STATE: dict[str, Any] = {"engine": None, "fallback": None, "unsupported": None}


@dataclass(frozen=True)
class RecognizedWord:
    text: str
    start_s: float
    end_s: float
    probability: float = 1.0


@dataclass(frozen=True)
class Recognition:
    text: str
    words: tuple[RecognizedWord, ...]
    language: str | None
    engine: str
    seconds: float


def is_builtin_model(name: Any) -> bool:
    """True for the built-in engine's names (an empty name means the built-in engine too)."""
    value = str(name or "").strip()
    return not value or value.casefold() in BUILTIN_ALIASES


def model_folder(name: str = BUILTIN_MODEL) -> Path:
    return MODEL_ROOT / name


def _is_model_folder(path: Path) -> bool:
    config = path / "config.json"
    if not (config.is_file() and (path / "model.safetensors").is_file() and (path / "tokenizer.json").is_file()):
        return False
    try:
        return "convrot" in json.loads(config.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False


def ensure_model(name: str = BUILTIN_MODEL) -> Path:
    """The model folder, downloaded on first use.

    Files arrive in a hidden staging folder (an interrupted download resumes there) and the finished folder is
    moved into place, so a partial download never looks like a usable model.
    """
    target = model_folder(name)
    if _is_model_folder(target):
        return target
    with _DOWNLOAD_LOCK:
        if _is_model_folder(target):
            return target
        from huggingface_hub import snapshot_download

        repo_id, subfolder = HOSTED_MODELS[name]
        staging = MODEL_ROOT / f".download-{name}"
        print(f">> Downloading the built-in Whisper {name} (1.6 GB, first use only) from {repo_id} to {target} ...",
              flush=True)
        snapshot_download(repo_id=repo_id, allow_patterns=[f"{subfolder}/*"], local_dir=staging, max_workers=8)
        downloaded = staging.joinpath(*subfolder.split("/"))
        if not _is_model_folder(downloaded):
            raise RuntimeError(f"The download of {name} from {repo_id}/{subfolder} is incomplete: {downloaded}")
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(downloaded, target)
        shutil.rmtree(staging, ignore_errors=True)
        print(f">> Built-in Whisper ready: {target}", flush=True)
        return target


def _device_index(device: str) -> int:
    text = str(device or "cuda").strip().lower()
    if ":" in text:
        return int(text.split(":", 1)[1] or 0)
    import torch

    return int(torch.cuda.current_device())


def convrot_supported(device: str = "cuda:0") -> tuple[bool, str]:
    """Whether the INT8 ConvRot engine can run on ``device``: a CUDA GPU of the RTX 30 series or newer, Triton."""
    if not str(device or "").lower().startswith("cuda"):
        return False, f"the requested device is {device}"
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is always installed with the app
        return False, f"PyTorch unavailable ({exc})"
    if not torch.cuda.is_available():
        return False, "no NVIDIA CUDA GPU"
    try:
        major, minor = torch.cuda.get_device_capability(_device_index(device))
    except Exception as exc:
        return False, f"{device} is not available ({exc})"
    if (major, minor) < (8, 0):
        return False, f"GPU compute capability {major}.{minor}; the RTX 30 series (8.0) or newer is required"
    for module in ("triton", "faster_whisper"):
        try:
            __import__(module)
        except Exception as exc:
            return False, f"{module} is not available ({type(exc).__name__}: {exc})"
    return True, ""


def _available_gb(index: int) -> float | None:
    """Free VRAM plus what this process's allocator holds unused (both can take the model)."""
    try:
        import torch

        free = torch.cuda.mem_get_info(index)[0]
        cached = torch.cuda.memory_reserved(index) - torch.cuda.memory_allocated(index)
        return (free + max(0, cached)) / 1024**3
    except Exception:
        return None


def _language_code(language: Any) -> str | None:
    """A Whisper language code, or None (the recognizer detects it) for no or an unknown language."""
    code = str(language or "").strip().lower()
    if code in {"", "auto"}:
        return None
    from transformers.models.whisper.tokenization_whisper import LANGUAGES, TO_LANGUAGE_CODE

    from indextts.utils.speech_timestamps import _WHISPER_ALIASES

    code = _WHISPER_ALIASES.get(code, TO_LANGUAGE_CODE.get(code, code))
    if code in LANGUAGES:
        return code
    print(f">> Whisper has no language '{language}'; it detects the language of this audio.", flush=True)
    return None


def audio_16k(source: Any) -> np.ndarray:
    """Mono float32 samples at 16 kHz from a file path, a ``(samples, sample_rate)`` pair or 16 kHz samples."""
    rate = SAMPLE_RATE
    if isinstance(source, (str, os.PathLike)):
        try:
            import soundfile as sf

            samples, rate = sf.read(str(source), dtype="float32", always_2d=False)
        except Exception:  # video and formats libsndfile cannot read: FFmpeg through PyAV
            from faster_whisper import decode_audio

            samples, rate = decode_audio(str(source), sampling_rate=SAMPLE_RATE), SAMPLE_RATE
    elif isinstance(source, tuple):
        samples, rate = source
    else:
        samples = source
    if hasattr(samples, "detach"):
        samples = samples.detach().float().cpu().numpy()
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=0 if samples.shape[0] <= samples.shape[-1] else -1)
    samples = samples.reshape(-1)
    if int(rate) != SAMPLE_RATE and samples.size:
        from scipy.signal import resample_poly

        divisor = math.gcd(int(rate), SAMPLE_RATE)
        samples = resample_poly(samples, SAMPLE_RATE // divisor, int(rate) // divisor)
    return np.ascontiguousarray(samples, dtype=np.float32)


class _ConvRotEngine:
    """The INT8 ConvRot model behind faster-whisper's decoding, parked in RAM between jobs."""

    def __init__(self, path: Path, index: int):
        from .convrot.faster_whisper_adapter import ConvRotFasterWhisperModel

        started = time.perf_counter()
        print(f">> Loading the built-in Whisper ({BUILTIN_MODEL}) on cuda:{index} ...", flush=True)
        self.index = index
        self.model = ConvRotFasterWhisperModel(str(path), device="cuda", device_index=index)
        self.resident = True
        print(f">> Built-in Whisper loaded in {time.perf_counter() - started:.1f} s", flush=True)

    def wake(self) -> None:
        if not self.resident:
            self.model.model.load_model()
            self.resident = True

    def park(self) -> None:
        if self.resident:
            self.model.model.unload_model(to_cpu=True)
            self.resident = False

    def close(self) -> None:
        self.model.model.unload_model(to_cpu=False)
        self.resident = False


def _convrot_engine(device: str) -> _ConvRotEngine | None:
    """The engine on ``device``, loaded or woken; None when this system or the free VRAM cannot take it."""
    supported, reason = convrot_supported(device)
    if not supported:
        if _STATE["unsupported"] != reason:
            print(f">> The built-in Whisper engine cannot run here ({reason}); using Transformers {FALLBACK_MODEL}.",
                  flush=True)
            _STATE["unsupported"] = reason
        return None
    index = _device_index(device)
    engine = _STATE["engine"]
    if engine is not None and engine.index != index:
        engine.close()
        engine = _STATE["engine"] = None
    if engine is not None and engine.resident:
        return engine
    available = _available_gb(index)
    if available is not None and available < REQUIRED_FREE_GB:
        print(f">> {available:.1f} GB of VRAM is free on cuda:{index}; the built-in Whisper needs {REQUIRED_FREE_GB:.0f} GB. "
              f"Using Transformers {FALLBACK_MODEL} on the CPU for this job.", flush=True)
        return None
    if engine is None:
        engine = _STATE["engine"] = _ConvRotEngine(ensure_model(), index)
    else:
        engine.wake()
    return engine


def _keep_context(language: str | None, duration_s: float) -> bool:
    """Previous text as context (Whisper-WebUI's rule): off for English over 30 s, where large-v3 repeated passages,
    and in any language beyond 60 windows (30 minutes)."""
    if language == "en" and duration_s > ENGLISH_CONTEXT_LIMIT_S:
        return False
    return math.ceil(duration_s / 30.0) < LONG_FORM_CONTEXT_WINDOWS


def _speech_probabilities(audio: np.ndarray) -> np.ndarray:
    """faster-whisper's Silero speech probability of every 512-sample frame of ``audio``."""
    from faster_whisper.vad import get_vad_model

    frames = -(-audio.shape[0] // VAD_FRAME)
    padded = np.zeros(frames * VAD_FRAME, dtype=np.float32)
    padded[:audio.shape[0]] = audio
    return np.asarray(get_vad_model()(padded), dtype=np.float32).reshape(-1)


def _tail_has_speech(audio: np.ndarray, start_s: float, end_s: float) -> bool:
    try:
        start = int(start_s * SAMPLE_RATE)
        end = min(int(audio.shape[-1]), int(end_s * SAMPLE_RATE))
        counted = start + int(TAIL_IGNORED_S * SAMPLE_RATE)
        if end - counted < TAIL_SPEECH_S * SAMPLE_RATE:
            return False
        # a couple of seconds before the tail let the detector settle on the recording's noise level
        first = max(0, start - 2 * SAMPLE_RATE)
        probabilities = _speech_probabilities(audio[first:end])
        speech = int(np.count_nonzero(probabilities[(counted - first) // VAD_FRAME:] >= 0.5))
        return speech * VAD_FRAME / float(SAMPLE_RATE) >= TAIL_SPEECH_S
    except Exception as exc:
        print(f">> Voice detection of the last {end_s - start_s:.1f} s failed ({exc}); the tail is skipped.", flush=True)
        return False


def _guard_tail(seek: int, window_end: int, clip_end: int, time_per_frame: float, audio: np.ndarray | None,
                sentence_ended: bool = True) -> int:
    """End the clip instead of decoding a short remainder the window just decoded already contained, unless the
    voice detector hears speech in it or (``TAIL_GUARD_MODE`` "sentence") the window stopped mid-sentence."""
    if window_end >= clip_end and seek < clip_end and (clip_end - seek) * time_per_frame <= TAIL_GUARD_S:
        if TAIL_GUARD_MODE == "off" or (TAIL_GUARD_MODE == "sentence" and not sentence_ended):
            return seek
        if audio is not None and _tail_has_speech(audio, seek * time_per_frame, clip_end * time_per_frame):
            return seek
        return clip_end
    return seek


def _generate_segments(model: Any, features: np.ndarray, tokenizer: Any, options: Any, log_progress: bool,
                       encoder_output: Any = None, *, audio: np.ndarray | None = None) -> Iterator[Any]:
    """faster-whisper 1.2.1's ``WhisperModel.generate_segments`` with Whisper-WebUI's end-of-audio tail guard.

    Copied from faster-whisper (MIT license, SYSTRAN) as Whisper-WebUI Premium does; only ``_guard_tail`` after
    each window (and the last segment text it reads) is new.
    """
    from faster_whisper.audio import pad_or_trim
    from faster_whisper.transcribe import Segment, Word
    from faster_whisper.utils import get_end
    from tqdm import tqdm

    content_frames = features.shape[-1] - 1
    content_duration = float(content_frames * model.feature_extractor.time_per_frame)

    if isinstance(options.clip_timestamps, str):
        options.clip_timestamps = [
            float(ts) for ts in (options.clip_timestamps.split(",") if options.clip_timestamps else [])
        ]
    seek_points = [round(ts * model.frames_per_second) for ts in options.clip_timestamps]
    if len(seek_points) == 0:
        seek_points.append(0)
    if len(seek_points) % 2 == 1:
        seek_points.append(content_frames)
    seek_clips = list(zip(seek_points[::2], seek_points[1::2]))

    punctuation = "\"'“¿([{-\"'.。,，!！?？:：”)]}、"

    idx = 0
    clip_idx = 0
    seek = seek_clips[clip_idx][0]
    all_tokens: list[int] = []
    prompt_reset_since = 0

    if options.initial_prompt is not None:
        if isinstance(options.initial_prompt, str):
            all_tokens.extend(tokenizer.encode(" " + options.initial_prompt.strip()))
        else:
            all_tokens.extend(options.initial_prompt)

    pbar = tqdm(total=content_duration, unit="seconds", disable=not log_progress)
    last_speech_timestamp = 0.0
    last_text = ""
    try:
        while clip_idx < len(seek_clips):
            seek_clip_start, seek_clip_end = seek_clips[clip_idx]
            if seek_clip_end > content_frames:
                seek_clip_end = content_frames
            if seek < seek_clip_start:
                seek = seek_clip_start
            if seek >= seek_clip_end:
                clip_idx += 1
                if clip_idx < len(seek_clips):
                    seek = seek_clips[clip_idx][0]
                continue
            time_offset = seek * model.feature_extractor.time_per_frame
            window_end_time = float((seek + model.feature_extractor.nb_max_frames) * model.feature_extractor.time_per_frame)
            segment_size = min(model.feature_extractor.nb_max_frames, content_frames - seek, seek_clip_end - seek)
            segment = features[:, seek: seek + segment_size]
            segment_duration = segment_size * model.feature_extractor.time_per_frame
            segment = pad_or_trim(segment)

            previous_tokens = all_tokens[prompt_reset_since:]
            if seek > 0 or encoder_output is None:
                encoder_output = model.encode(segment)

            if options.multilingual:
                results = model.model.detect_language(encoder_output)
                language_token, _probability = results[0][0]
                tokenizer.language = tokenizer.tokenizer.token_to_id(language_token)
                tokenizer.language_code = language_token[2:-2]

            prompt = model.get_prompt(tokenizer, previous_tokens, without_timestamps=options.without_timestamps,
                                      prefix=options.prefix if seek == 0 else None, hotwords=options.hotwords)
            result, avg_logprob, temperature, compression_ratio = model.generate_with_fallback(
                encoder_output, prompt, tokenizer, options)

            if options.no_speech_threshold is not None:
                # no voice activity check
                should_skip = result.no_speech_prob > options.no_speech_threshold
                if options.log_prob_threshold is not None and avg_logprob > options.log_prob_threshold:
                    # don't skip if the logprob is high enough, despite the no_speech_prob
                    should_skip = False
                if should_skip:
                    # fast-forward to the next segment boundary
                    seek += segment_size
                    continue

            tokens = result.sequences_ids[0]
            previous_seek = seek

            # anomalous words are very long/short/improbable
            def word_anomaly_score(word: dict) -> float:
                probability = word.get("probability", 0.0)
                duration = word["end"] - word["start"]
                score = 0.0
                if probability < 0.15:
                    score += 1.0
                if duration < 0.133:
                    score += (0.133 - duration) * 15
                if duration > 2.0:
                    score += duration - 2.0
                return score

            def is_segment_anomaly(segment: dict | None) -> bool:
                if segment is None or not segment["words"]:
                    return False
                words = [w for w in segment["words"] if w["word"] not in punctuation][:8]
                score = sum(word_anomaly_score(w) for w in words)
                return score >= 3 or score + 0.01 >= len(words)

            def next_words_segment(segments: list) -> dict | None:
                return next((s for s in segments if s["words"]), None)

            current_segments, seek, single_timestamp_ending = model._split_segments_by_timestamps(
                tokenizer=tokenizer, tokens=tokens, time_offset=time_offset, segment_size=segment_size,
                segment_duration=segment_duration, seek=seek)

            if options.word_timestamps:
                model.add_word_timestamps([current_segments], tokenizer, encoder_output, segment_size,
                                          options.prepend_punctuations, options.append_punctuations,
                                          last_speech_timestamp=last_speech_timestamp)
                if not single_timestamp_ending:
                    last_word_end = get_end(current_segments)
                    if last_word_end is not None and last_word_end > time_offset:
                        seek = round(last_word_end * model.frames_per_second)

                # skip silence before possible hallucinations
                if options.hallucination_silence_threshold is not None:
                    threshold = options.hallucination_silence_threshold
                    # if first segment might be a hallucination, skip leading silence
                    first_segment = next_words_segment(current_segments)
                    if first_segment is not None and is_segment_anomaly(first_segment):
                        gap = first_segment["start"] - time_offset
                        if gap > threshold:
                            seek = previous_seek + round(gap * model.frames_per_second)
                            continue
                    # skip silence before any possible hallucination that is surrounded by silence or more
                    # hallucinations
                    hal_last_end = last_speech_timestamp
                    for si in range(len(current_segments)):
                        segment = current_segments[si]
                        if not segment["words"]:
                            continue
                        if is_segment_anomaly(segment):
                            next_segment = next_words_segment(current_segments[si + 1:])
                            hal_next_start = (next_segment["words"][0]["start"] if next_segment is not None
                                              else time_offset + segment_duration)
                            silence_before = (segment["start"] - hal_last_end > threshold
                                              or segment["start"] < threshold
                                              or segment["start"] - time_offset < 2.0)
                            silence_after = (hal_next_start - segment["end"] > threshold
                                             or is_segment_anomaly(next_segment)
                                             or window_end_time - segment["end"] < 2.0)
                            if silence_before and silence_after:
                                seek = round(max(time_offset + 1, segment["start"]) * model.frames_per_second)
                                if content_duration - segment["end"] < threshold:
                                    seek = content_frames
                                current_segments[si:] = []
                                break
                        hal_last_end = segment["end"]

                last_word_end = get_end(current_segments)
                if last_word_end is not None:
                    last_speech_timestamp = last_word_end

            for segment in current_segments:
                tokens = segment["tokens"]
                text = tokenizer.decode(tokens)
                if segment["start"] == segment["end"] or not text.strip():
                    continue
                all_tokens.extend(tokens)
                idx += 1
                last_text = text
                yield Segment(id=idx, seek=previous_seek, start=segment["start"], end=segment["end"], text=text,
                              tokens=tokens, temperature=temperature, avg_logprob=avg_logprob,
                              compression_ratio=compression_ratio, no_speech_prob=result.no_speech_prob,
                              words=([Word(**word) for word in segment["words"]] if options.word_timestamps else None))

            if not options.condition_on_previous_text or temperature > options.prompt_reset_on_temperature:
                prompt_reset_since = len(all_tokens)

            seek = _guard_tail(seek, previous_seek + segment_size, seek_clip_end,
                               model.feature_extractor.time_per_frame, audio,
                               sentence_ended=last_text.rstrip().endswith(_SENTENCE_END))
            pbar.update((min(content_frames, seek) - previous_seek) * model.feature_extractor.time_per_frame)
    finally:
        pbar.close()


@contextmanager
def _tail_guarded(model: Any, audio: np.ndarray) -> Iterator[None]:
    """faster-whisper's ``transcribe`` decodes with the tail-guarded loop while the segments are consumed."""

    def generate_segments(self: Any, features: np.ndarray, tokenizer: Any, options: Any, log_progress: bool,
                          encoder_output: Any = None) -> Iterator[Any]:
        return _generate_segments(self, features, tokenizer, options, log_progress, encoder_output, audio=audio)

    model.generate_segments = MethodType(generate_segments, model)
    try:
        yield
    finally:
        del model.generate_segments


def _recognize_convrot(engine: _ConvRotEngine, audio: np.ndarray, language: str | None, initial_prompt: str | None,
                       progress: Callable[[float, float], None] | None) -> tuple[str, list[RecognizedWord], str | None]:
    from .convrot import triton_status

    duration = audio.shape[-1] / float(SAMPLE_RATE)
    options = dict(BEST_QUALITY, condition_on_previous_text=_keep_context(language, duration))
    tuning = triton_status.counts()
    pieces: list[str] = []
    words: list[RecognizedWord] = []
    with _tail_guarded(engine.model, audio):
        segments, info = engine.model.transcribe(audio, language=language, task="transcribe",
                                                 initial_prompt=initial_prompt or None, **options)
        reported = 0.0
        for segment in segments:
            pieces.append(segment.text)
            for word in segment.words or ():
                text = word.word.strip()
                if text:
                    words.append(RecognizedWord(text, float(word.start), float(word.end), float(word.probability)))
            if progress is not None and (segment.end - reported >= 30.0):
                reported = float(segment.end)
                progress(min(reported, duration), duration)
    summary = triton_status.summary_since(tuning)
    if summary:
        print(f">> {summary}", flush=True)
    return "".join(pieces).strip(), words, info.language


def _fallback_pipeline(device: str) -> Any:
    cached = _STATE["fallback"]
    if cached is not None and cached[0] == device:
        return cached[1]
    import torch
    from transformers import pipeline

    from indextts.training.whisper_asr import _ensure_model

    _STATE["fallback"] = None
    pipe = pipeline("automatic-speech-recognition", model=str(_ensure_model(FALLBACK_MODEL)), device=device,
                    dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32, chunk_length_s=30)
    _STATE["fallback"] = (device, pipe)
    return pipe


def _fallback_device(device: str) -> str:
    """The requested device when it has room for large-v3-turbo beside the loaded models, otherwise the CPU."""
    if not str(device).lower().startswith("cuda"):
        return str(device)
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"
    except Exception:
        return "cpu"
    available = _available_gb(_device_index(device))
    return str(device) if available is None or available >= 2.5 else "cpu"


def _recognize_fallback(audio: np.ndarray, language: str | None, device: str, words: bool,
                        progress: Callable[[float, float], None] | None) -> tuple[str, list[RecognizedWord]]:
    device = _fallback_device(device)
    if words:
        from indextts.training.whisper_asr import _transformers_transcribe

        transcript = _transformers_transcribe(audio, SAMPLE_RATE, language, FALLBACK_MODEL, device, None)
        return transcript.text, [RecognizedWord(word.text, word.start_s, word.end_s) for word in transcript.words]
    pipe = _fallback_pipeline(device)
    options = {"task": "transcribe", "do_sample": False, **({"language": language} if language else {})}
    text = str(pipe({"raw": audio, "sampling_rate": SAMPLE_RATE}, generate_kwargs=options)["text"]).strip()
    if progress is not None:
        progress(audio.shape[-1] / SAMPLE_RATE, audio.shape[-1] / SAMPLE_RATE)
    return text, []


def recognize(audio: Any, *, language: str | None, device: str = "cuda:0", initial_prompt: str | None = None,
              words: bool = True, progress: Callable[[float, float], None] | None = None) -> Recognition:
    """Transcribe ``audio`` (a file path, a ``(samples, sample_rate)`` pair or 16 kHz samples) in ``language``.

    ``language`` is a Whisper code or name; None or "auto" lets Whisper detect it (only the OmniVoice reference
    transcript does that, when its language is Auto). ``initial_prompt`` primes spellings. ``progress`` receives
    (seconds done, total seconds) about every 30 seconds of long audio. The built-in engine always measures word
    timings; ``words`` decides whether the fallback recognizer spends the extra pass on them.
    """
    samples = audio_16k(audio)
    code = _language_code(language)
    started = time.perf_counter()
    if not samples.size:
        return Recognition("", (), code, "none", 0.0)
    with _LOCK:
        engine = _convrot_engine(device)
        if engine is not None:
            try:
                text, timed, detected = _recognize_convrot(engine, samples, code, initial_prompt, progress)
            except Exception as exc:
                import torch

                if not isinstance(exc, torch.OutOfMemoryError):
                    raise
                # The speech model left too little room: this job falls back, the engine waits in RAM.
                print(f">> The built-in Whisper ran out of VRAM ({exc}); this job uses Transformers {FALLBACK_MODEL}.",
                      flush=True)
                engine.park()
                gc.collect()
                torch.cuda.empty_cache()
            else:
                label = f"whisper {BUILTIN_MODEL} (built-in engine, cuda:{engine.index})"
                return Recognition(text, tuple(timed), detected or code, label, time.perf_counter() - started)
        text, timed = _recognize_fallback(samples, code, device, words, progress)
        return Recognition(text, tuple(timed), code, f"{FALLBACK_MODEL} (Transformers)", time.perf_counter() - started)


def park() -> None:
    """Free the VRAM after a job: the built-in engine's weights wait in RAM, a fallback recognizer is dropped."""
    with _LOCK:
        engine = _STATE["engine"]
        released = _STATE["fallback"] is not None or (engine is not None and engine.resident)
        if engine is not None:
            engine.park()
        _STATE["fallback"] = None
    if released:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def unload() -> None:
    """Drop the recognizer from VRAM and RAM; the next job loads it from disk again."""
    with _LOCK:
        engine = _STATE["engine"]
        if engine is not None:
            engine.close()
        _STATE["engine"] = None
        _STATE["fallback"] = None
    gc.collect()


def description(device: str = "cuda:0") -> str:
    """Which recognizer the app uses on ``device`` (for logs and the interface)."""
    supported, reason = convrot_supported(device)
    if supported:
        return f"Whisper {BUILTIN_MODEL} on the built-in INT8 ConvRot engine"
    return f"Transformers {FALLBACK_MODEL} ({reason})"


__all__ = [
    "BEST_QUALITY", "BUILTIN_MODEL", "FALLBACK_MODEL", "Recognition", "RecognizedWord", "audio_16k", "convrot_supported",
    "description", "ensure_model", "is_builtin_model", "model_folder", "park", "recognize", "unload",
]
