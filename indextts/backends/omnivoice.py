"""OmniVoice behind the application's shared generation/adapter contract."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import threading
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch

from indextts.runtime.vram_presets import RuntimeConfig
from indextts.utils.pause_tags import PauseChunk, TextChunk, split_text_with_pauses
from indextts.utils.pronunciation import omnivoice_readings
from indextts.utils.text_segmentation import ends_sentence, split_text_by_tokens, normalize_sentence_whitespace
from indextts.utils.audio_plan import assemble_audio_plan, trim_segment_silence, fit_target_samples

MODEL_REPO = "k2-fsa/OmniVoice"
QUANT_REPO = "MonsterMMORPG/Wan_GGUF"
QUANT_FILENAME = "OmniVoice/omnivoice_convrot_int8.safetensors"
# Reference prompts (audio tokens plus transcript) keyed by audio content.
PROMPT_CACHE_DIR = Path(__file__).resolve().parents[2] / ".ui_state" / "omnivoice_prompts"
PROMPT_CACHE_VERSION = 1

GENERATION_DEFAULTS = {
    "mode": "clone", "reference_text": "", "instruct": "",
    "num_step": 32, "guidance_scale": 2.0, "t_shift": 0.1,
    "layer_penalty_factor": 5.0, "position_temperature": 5.0,
    "class_temperature": 0.0, "denoise": True, "preprocess_prompt": True,
    "postprocess_output": True, "audio_chunk_duration": 15.0,
    "audio_chunk_threshold": 30.0, "pad_duration": 0.1, "fade_duration": 0.1,
}


_NORMALIZER = None
_NORMALIZER_LOCK = threading.Lock()


def normalize_omnivoice_text(text, language="auto"):
    """The text normalization generation applies; training uses the same, so a
    fine-tuned voice hears transcripts written the way it will be asked to speak
    (numbers, versions and units spelled out)."""
    global _NORMALIZER
    from omnivoice.utils.text import normalize_text

    language = str(language or "auto").lower()
    if language == "auto":
        language = "zh" if re.search(r"[一-鿿]", text) else "en"
    if language not in {"en", "zh"}:
        return normalize_text(text, language)
    with _NORMALIZER_LOCK:
        if _NORMALIZER is None:
            from indextts.utils.front import TextNormalizer
            normalizer = TextNormalizer()
            normalizer.load()
            _NORMALIZER = normalizer
    # Reuse the app's Windows-compatible WeText path; preserve OmniVoice tags.
    parts = re.split(r"(\[[^\]\n]+\])", text)
    return "".join(part if part.startswith("[") or not part.strip() else
        part[:len(part)-len(part.lstrip())] + _NORMALIZER.normalize(part.strip(), lang=language)
        + part[len(part.rstrip()):] for part in parts)


def calibrated_voice_speed(adapter_path):
    """The speed calibrated for a trained voice without a reference (training writes it), or None."""
    if not adapter_path:
        return None
    source = Path(adapter_path)
    run_dir = source.parent.parent if source.parent.name.lower() == "best" else source.parent
    try:
        value = float(json.loads((run_dir / "omnivoice_voice.json").read_text(encoding="utf-8"))["speed_without_reference"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return value if 0.5 <= value <= 2.0 else None


def trained_voice_speed(adapter_path):
    """Speed that sizes unprompted speech at a trained voice's own pace (1.0 for none)."""
    value = calibrated_voice_speed(adapter_path)
    return 1.0 if value is None else value


# OmniVoice's fallback pace for speech without a reference: this text is one second (25 audio tokens).
_DEFAULT_PACE_TEXT, _DEFAULT_PACE_TOKENS = "Nice to meet you.", 25


def validate_voice_settings(settings):
    mode = settings.get("mode", "clone")
    if mode not in {"auto", "clone", "design"}:
        raise ValueError("Unknown OmniVoice voice mode")
    instruct = str(settings.get("instruct") or "").strip()
    if mode == "design" and not instruct:
        raise ValueError("Enter supported voice tags for Voice design, for example: male, middle-aged, moderate pitch.")
    if mode != "auto" and instruct:
        # Reuse the installed model's supported tags and conflict checks.
        from omnivoice.models.omnivoice import _resolve_instruct
        _resolve_instruct(instruct)


def ensure_model(model_dir, *, quantized=False, progress=None):
    """Download public weights on demand, retaining Hugging Face's resume cache."""
    from huggingface_hub import hf_hub_download, snapshot_download

    folder = Path(model_dir) / "omnivoice"
    required = [folder / name for name in (
        "config.json", "model.safetensors", "tokenizer.json",
        "audio_tokenizer/config.json", "audio_tokenizer/model.safetensors",
    )]
    if not all(path.is_file() for path in required):
        print(f">> Downloading public OmniVoice model to {folder}", flush=True)
        if progress:
            progress(0, desc="Downloading OmniVoice weights and audio tokenizer")
        snapshot_download(MODEL_REPO, local_dir=str(folder), allow_patterns=[
            "*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken",
            "audio_tokenizer/*",
        ])
    quant_path = folder / Path(QUANT_FILENAME).name
    if not quant_path.is_file():
        quant_path = Path(model_dir) / "quantized" / QUANT_FILENAME
    if quantized and not quant_path.is_file():
        print(">> Downloading OmniVoice ConvRot INT8 weights", flush=True)
        downloaded = hf_hub_download(QUANT_REPO, QUANT_FILENAME, local_dir=str(Path(model_dir) / "quantized"))
        quant_path = Path(downloaded)
    return folder, quant_path


class OmniVoiceEngine:
    sampling_rate = 24000
    model_id = "omnivoice"

    def __init__(self, model_dir="models", runtime=None, progress_callback=None):
        from indextts.utils.torch_compat import install_native_enum_pytree_compatibility

        install_native_enum_pytree_compatibility()
        from omnivoice import OmniVoice

        self.runtime = RuntimeConfig.from_dict(runtime)
        self.device = self.runtime.device
        if self.device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if self.device.startswith("cuda"):
            torch.set_num_threads(min(torch.get_num_threads(), 8))
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[self.runtime.gpt_dtype]
        if self.device == "cpu":
            dtype = torch.float32
        self.low_vram = self._runtime_low_vram = False
        self.progress_reporter = self.gr_progress = None
        self.last_generation_stats = {}
        self._lora_path, self._lora_strength, self._lora_merged = "", 1.0, False
        self._voice_key, self._voice_prompt = None, None
        from indextts.quant.finetune_int8 import is_int8_finetune

        # The INT8 version of a full fine-tune is the model itself: it replaces the public INT8 transformer.
        self._int8_finetune_path = os.path.abspath(self.runtime.lora_path) if is_int8_finetune(self.runtime.lora_path) else ""
        if self._int8_finetune_path:
            self.runtime.model_variant = "int8_convrot"
        folder, quant_path = ensure_model(model_dir, quantized=self.runtime.model_variant == "int8_convrot" and not self._int8_finetune_path,
                                          progress=progress_callback)
        print(f">> Loading OmniVoice | {self.device} | {dtype} | {self.runtime.model_variant}", flush=True)
        self.model = OmniVoice.from_pretrained(str(folder), device_map=self.device, dtype=dtype, load_asr=False)
        self.model.llm.config.use_cache = False
        self.model.llm.set_attn_implementation(self.runtime.attention_backend)
        self.model.eval()
        if self.runtime.model_variant == "int8_convrot":
            from indextts.quant.convrot_int8 import load_gpt_checkpoint
            from indextts.quant.finetune_int8 import OMNIVOICE_EXTRA_PREFIX

            # The checkpoint contains the transformer; tokenizer weights stay full precision. A fine-tuned model's
            # file also carries its trained audio embeddings and heads (BF16, under the "omnivoice." prefix).
            source = self._int8_finetune_path or str(quant_path)
            load_gpt_checkpoint(self.model.llm, source, device=self.device, dtype=dtype, strict=True,
                                skip_prefixes=(OMNIVOICE_EXTRA_PREFIX,))
            if self._int8_finetune_path:
                from safetensors import safe_open

                with safe_open(source, framework="pt", device="cpu") as handle, torch.no_grad():
                    for key in handle.keys():
                        if key.startswith(OMNIVOICE_EXTRA_PREFIX):
                            target = self.model.get_parameter(key[len(OMNIVOICE_EXTRA_PREFIX):])
                            target.copy_(handle.get_tensor(key).to(device=target.device, dtype=target.dtype))
                print(f">> INT8 ConvRot fine-tuned model: {source}", flush=True)
        self.set_lora(self.runtime.lora_path, self.runtime.lora_strength, merge_into_base=self.runtime.lora_merge_into_base)

    def set_lora(self, path, strength=1.0, *, merge_into_base=False, **_kwargs):
        from indextts.lora import apply_lora, inspect_lora, remove_lora, merge_lora_for_inference

        path = os.path.abspath(str(path)) if path else ""
        finetuned = getattr(self, "_int8_finetune_path", "")
        if finetuned or path.lower().endswith(".int8_convrot.safetensors"):
            from indextts.quant.finetune_int8 import is_int8_finetune

            if path != finetuned and (finetuned or is_int8_finetune(path)):
                raise ValueError("An INT8 fine-tuned model is the speech model itself; reload the model to change to or from it.")
            if path == finetuned:
                self._lora_path, self._lora_strength, self._lora_merged = path, 1.0, False
                self._voice_key, self._voice_prompt = None, None
                self._unprompted_speed = trained_voice_speed(path)
                self._pace_calibrated = calibrated_voice_speed(path) is not None
                return
        if path:
            metadata = inspect_lora(path)
            base = str(metadata.get("base_model", ""))
            if "omnivoice" not in base.lower():
                raise ValueError("This adapter belongs to IndexTTS. Select an OmniVoice adapter or clear the adapter selection.")
            if metadata.get("adapter_type") == "full" and self.runtime.model_variant != "bf16":
                raise ValueError("Select BF16 before loading a full fine-tuning checkpoint, or choose its INT8 ConvRot version.")
        remove_lora(self.model)
        if path:
            apply_lora(self.model, path, strength=float(strength))
            if merge_into_base:
                merge_lora_for_inference(self.model)
        self._lora_path, self._lora_strength, self._lora_merged = path, float(strength), bool(merge_into_base)
        self.model.eval().requires_grad_(False)
        self._voice_key, self._voice_prompt = None, None
        self._unprompted_speed = trained_voice_speed(path)
        self._pace_calibrated = calibrated_voice_speed(path) is not None

    def _generation_speed(self, voice):
        """Pace multiplier: a trained voice keeps its calibrated pace with or without a reference clip.

        OmniVoice sizes cloned speech at the reference clip's own pace (audio tokens per text weight), so a
        brisk or slow clip made every take brisk or slow (round 3: one 8-second clip gave 0.79 of the real
        durations). With a calibrated voice the clip still sets timbre and style, and the pace becomes the one
        calibrated for the voice; base-model cloning keeps the reference's pace.
        """
        calibrated = getattr(self, "_unprompted_speed", 1.0)
        if voice is None:
            return calibrated
        if not getattr(self, "_pace_calibrated", False):
            return 1.0
        estimator = getattr(self.model, "duration_estimator", None)
        try:
            tokens = int(voice.ref_audio_tokens.shape[-1])
            ref_weight = float(estimator.calculate_total_weight(voice.ref_text or ""))
            default_weight = float(estimator.calculate_total_weight(_DEFAULT_PACE_TEXT))
        except (AttributeError, TypeError, ValueError):
            return 1.0
        if tokens <= 0 or ref_weight <= 0 or default_weight <= 0:
            return 1.0
        return calibrated * (tokens / ref_weight) / (_DEFAULT_PACE_TOKENS / default_weight)

    def split_text_by_tokens(self, text, max_tokens, lang_prefix="", *, mode="budget", target_tokens=None):
        return split_text_by_tokens(text, max_tokens, capacity=2048,
            token_len=lambda value: len(self.model.text_tokenizer.encode(value, add_special_tokens=False)),
            mode=mode, target_tokens=target_tokens, segment_budget_scale_non_cjk=1.0)

    def _prompt(self, path, settings):
        if settings["mode"] != "clone":
            return None
        if not path or not Path(path).is_file():
            raise ValueError("Voice cloning requires reference audio. Choose Auto voice or Voice design to generate without it.")
        stat = Path(path).stat()
        settings = dict(settings)
        transcript = Path(path).with_suffix(".txt")
        if not settings["reference_text"] and transcript.is_file():
            settings["reference_text"] = transcript.read_text(encoding="utf-8-sig").strip()
        key = (str(Path(path).resolve()), stat.st_mtime_ns, stat.st_size,
               settings["reference_text"], settings["denoise"], settings["preprocess_prompt"])
        if key != self._voice_key:
            self._voice_prompt = self._cached_prompt(path, settings)
            self._voice_key = key
        return self._voice_prompt

    def _cached_prompt(self, path, settings):
        """The reference's audio tokens and transcript, reused across restarts.

        A reference without a transcript is transcribed by OmniVoice's Whisper
        model once; the prompt is then cached by audio content, so later runs
        need neither the speech recognizer nor the audio tokenizer for it.
        """
        from omnivoice.models.omnivoice import VoiceClonePrompt

        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        digest.update(json.dumps([settings["reference_text"], bool(settings["preprocess_prompt"]),
                                  PROMPT_CACHE_VERSION]).encode("utf-8"))
        cached = PROMPT_CACHE_DIR / f"{digest.hexdigest()[:32]}.pt"
        if cached.is_file():
            try:
                return VoiceClonePrompt.load(str(cached))
            except Exception as exc:  # a damaged cache entry is rebuilt below
                print(f">> OmniVoice reference cache ignored ({exc})", flush=True)
        transcribing = not settings["reference_text"]
        if transcribing:
            print(f">> Transcribing reference {Path(path).name} once with OmniVoice's Whisper model", flush=True)
        try:
            with torch.inference_mode():
                prompt = self.model.create_voice_clone_prompt(
                    ref_audio=str(path), ref_text=settings["reference_text"] or None,
                    preprocess_prompt=settings["preprocess_prompt"])
        finally:
            if transcribing and getattr(self.model, "_asr_pipe", None) is not None:
                # The recognizer is needed once per reference; release its VRAM.
                self.model._asr_pipe = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        try:
            PROMPT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            temporary = cached.with_suffix(f".{os.getpid()}.tmp")
            prompt.save(str(temporary))
            os.replace(temporary, cached)
        except OSError as exc:
            print(f">> OmniVoice reference cache not written ({exc})", flush=True)
        if transcribing:
            print(f">> Reference transcript: {prompt.ref_text}", flush=True)
        return prompt

    def _normalize_text(self, text, language):
        return normalize_omnivoice_text(text, language)

    def infer(self, spk_audio_prompt, text, output_path=None, lang="EN", **kwargs):
        result = self.infer_texts(spk_audio_prompt, [text], lang=lang, **kwargs)[0]
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, result[1], result[0], subtype="PCM_16")
            return str(output_path)
        return result

    def infer_texts(self, spk_audio_prompt, texts, lang="EN", *, omnivoice=None,
              max_text_tokens_per_segment=120, interval_silence=200, sentence_pause_ms=0,
              segmentation_mode="budget", segment_target_tokens=None, enable_pause_tags=True,
              text_normalization=True, duration_factor=1.0, seed=None, target_duration_s=None,
              target_duration_mode="off", section_batch_size=1, on_text_complete=None,
              trim_silence_ms_threshold=0, **_kwargs):
        started = time.perf_counter()
        if str(self.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)
        settings = {**GENERATION_DEFAULTS, **(omnivoice or {})}
        validate_voice_settings(settings)
        if target_duration_mode not in {"off", "natural", "pad", "trim"}:
            raise ValueError("Unknown target duration mode")
        voice = self._prompt(spk_audio_prompt, settings)
        plans, speech, owners, target_lengths = [], [], [], []
        for text_index, text in enumerate(texts):
            plan, first = [], len(speech)
            chunks = split_text_with_pauses(text) if enable_pause_tags else [TextChunk(text)]
            for chunk in chunks:
                if isinstance(chunk, PauseChunk):
                    plan.append(("pause", round(chunk.duration_s * self.sampling_rate)))
                    continue
                # Dictionary and hand-written <word|PHONES> readings in OmniVoice's own syntax.
                source = omnivoice_readings(chunk.text).strip()
                if segmentation_mode != "budget":
                    source = normalize_sentence_whitespace(source)
                if text_normalization:
                    source = self._normalize_text(source, lang)
                segments = [part for part in self.split_text_by_tokens(source, max_text_tokens_per_segment,
                    mode=segmentation_mode, target_tokens=segment_target_tokens) if part.strip()]
                for index, segment in enumerate(segments):
                    plan.append(("segment", len(speech)))
                    speech.append(segment)
                    owners.append(text_index)
                    if index < len(segments) - 1:
                        gap = sentence_pause_ms if sentence_pause_ms and ends_sentence(segment) else interval_silence
                        kind = "sentence_gap" if sentence_pause_ms and ends_sentence(segment) else "silence"
                        plan.append((kind, round(max(0, gap) * self.sampling_rate / 1000)))
            if len(speech) == first:
                raise ValueError("Enter some text to generate speech.")
            plans.append(plan)
            fixed_seconds = sum(value for kind, value in plan if kind != "segment") / self.sampling_rate
            total_chars = sum(len(part) for part in speech[first:])
            for part in speech[first:]:
                target_lengths.append(max(0.1, float(target_duration_s) - fixed_seconds) * len(part) / total_chars
                    if target_duration_s and target_duration_mode == "natural" else None)
        config = {key: settings[key] for key in GENERATION_DEFAULTS if key not in {"mode", "reference_text", "instruct"}}
        batch_size = 1 if getattr(self, "low_vram", False) else max(1, int(section_batch_size))
        # Check the shared cancellation reporter on every diffusion step.
        batch_index, batch_step = 0, 0
        total = math.ceil(len(speech) / batch_size)
        def on_forward(_module, _args):
            nonlocal batch_step
            batch_step += 1
            if self.progress_reporter:
                # Upstream may split a long section into several diffusion
                # passes. Count completed batches, not an assumed pass count,
                # so progress cannot reach 100% while audio is still rendering.
                self.progress_reporter.update(batch_index, total=total,
                    desc=f"OmniVoice batch {batch_index + 1}/{total}, model pass {batch_step}")
        hook = self.model.register_forward_pre_hook(on_forward)
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
        devices = [torch.device(self.device).index or 0] if str(self.device).startswith("cuda") else []

        def synthesize(texts_, lengths_):
            generated = self.model.generate(
                text=texts_ if len(texts_) > 1 else texts_[0],
                language=None if str(lang).lower() == "auto" else str(lang).lower(),
                voice_clone_prompt=voice, instruct=(settings["instruct"] or None) if settings["mode"] != "auto" else None,
                # A trained voice keeps its speaker's measured pace, with or without a reference.
                speed=self._generation_speed(voice) / max(0.1, float(duration_factor)),
                duration=lengths_ if len(texts_) > 1 else lengths_[0], normalize_text=False, **config)
            if len(generated) != len(texts_):
                raise RuntimeError("OmniVoice returned the wrong batch size.")
            outputs = []
            for audio in generated:
                audio = np.asarray(audio, dtype=np.float32).reshape(-1)
                if not len(audio) or not np.isfinite(audio).all():
                    raise RuntimeError("OmniVoice returned empty or non-finite audio.")
                outputs.append(trim_segment_silence(torch.from_numpy(audio).unsqueeze(0), self.sampling_rate,
                                                    trim_silence_ms_threshold))
            return outputs

        # Takes per section: Whisper scores every take; up to this many renders of a section, the fewest errors win.
        takes = max(1, int(getattr(self, "section_takes", 1) or 1))
        judge = getattr(self, "take_judge", None) if takes > 1 else None
        try:
            with torch.inference_mode(), torch.random.fork_rng(devices=devices):
                if seed is not None:
                    torch.manual_seed(int(seed))
                for start in range(0, len(speech), batch_size):
                    batch_index, batch_step = start // batch_size, 0
                    batch = speech[start:start + batch_size]
                    lengths = target_lengths[start:start + batch_size]
                    for offset, take in enumerate(synthesize(batch, lengths)):
                        index = start + offset
                        if judge is not None:
                            from indextts.utils.take_selection import keep_best_take

                            take, rates, kept = keep_best_take(
                                take, lambda i=index: synthesize([speech[i]], [target_lengths[i]])[0],
                                lambda audio, i=index: judge.error_rate(speech[i], audio.squeeze(0).numpy(), self.sampling_rate),
                                takes)
                            judge.record(index, rates, kept)
                        rendered[index] = take
                        durations.append(take.shape[-1] / self.sampling_rate)
                        complete(owners[index])
                    if self.progress_reporter:
                        self.progress_reporter.update(batch_index + 1, total=total,
                            desc=f"OmniVoice batch {batch_index + 1}/{total} complete")
        finally:
            hook.remove()
        self.last_generation_stats = {
            "segment_count": len(speech), "segments_count": len(speech),
            "total_duration_s": sum(len(result[1]) for result in results) / self.sampling_rate,
            "mean_duration_s": float(np.mean(durations)), "min_duration_s": min(durations),
            "max_duration_s": max(durations), "generation_time_s": time.perf_counter() - started,
            "model": self.model_id, "section_batch_size": batch_size,
            "peak_vram_gb": torch.cuda.max_memory_allocated(self.device) / 1024**3 if str(self.device).startswith("cuda") else 0.0,
            "protected_pauses": protected_by_text.get(0, []) if len(texts) == 1 else [],
        }
        return results

    def unload(self):
        self._voice_prompt = self._voice_key = None
        self.progress_reporter = self.gr_progress = None
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
