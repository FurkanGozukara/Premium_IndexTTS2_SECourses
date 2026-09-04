import os
from subprocess import CalledProcessError
from contextlib import contextmanager

import json
import math
import re
import secrets
import time
import librosa
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from omegaconf import OmegaConf

from indextts.codec.models import EnhancedCodec
from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.common import save_pcm_wav
from indextts.utils.front import TextNormalizer
from indextts.utils.tokenizer import get_tokenizer, lang_to_token
from indextts.utils.ja_g2p import JapaneseG2PProcessor
from indextts.utils.nemo_tn import normalize_text as nemo_text_normalize
from indextts.utils.pause_tags import PauseChunk, TextChunk, split_text_with_pauses
from indextts.utils.text_segmentation import (
    DEFAULT_NON_CJK_BUDGET_SCALE,
    default_segment_tokens,
    split_atomic_pieces,
    split_text_by_tokens as shared_split_text_by_tokens,
)
from indextts.runtime.block_swap import (
    BlockSwapConfig,
    default_swap_tensor_selector,
    enable_block_swap,
    resolve_blocks_to_swap,
)
from indextts.runtime.gpu import device_from_string, gpu_free_gb, memory_stats
from indextts.runtime.residency import ResidencyManager
from indextts.runtime.vram_presets import RuntimeConfig, describe, estimate_vram_gb

from indextts.s2mel.modules.commons import load_checkpoint2, MyModel
from indextts.s2mel.modules.bigvgan import bigvgan
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from indextts.s2mel.modules.audio import mel_spectrogram
from transformers import AutoTokenizer
from modelscope import AutoModelForCausalLM
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel
import random
import torch.nn.functional as F


SAMPLE_RATE = 22050
_TARGET_DURATION_MODES = {"off", "natural", "pad", "trim"}


def _batched_generation_kwargs(kwargs):
    """Remove duration options handled only by the sequential inference path."""
    values = dict(kwargs)
    values.pop("target_duration_s", None)
    values.pop("target_duration_mode", None)
    return values


def _seed_everything(seed):
    """Resolve a request seed and seed every RNG used by the inference path."""

    if seed is None or int(seed) == -1:
        actual_seed = secrets.randbelow(2**32)
    else:
        actual_seed = int(seed) % (2**32)
    random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)
    return actual_seed


def trim_segment_silence(wav, sampling_rate=SAMPLE_RATE, minimum_silence_ms=0):
    """Trim sufficiently long quiet leading/trailing runs using a fixed RMS gate."""

    minimum_ms = max(0.0, float(minimum_silence_ms or 0))
    if minimum_ms <= 0 or wav.numel() == 0:
        return wav
    audio = wav.detach().float()
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    peak_scale = 32767.0 if audio.abs().max().item() > 2.0 else 1.0
    mono = audio.abs().mean(dim=0) / peak_scale
    frame_size = max(1, int(round(float(sampling_rate) * 0.01)))
    frame_count = int(math.ceil(mono.numel() / frame_size))
    padded = F.pad(mono, (0, frame_count * frame_size - mono.numel()))
    rms = padded.view(frame_count, frame_size).square().mean(dim=1).sqrt()
    active = torch.nonzero(rms >= 10 ** (-45.0 / 20.0), as_tuple=False).flatten()
    if active.numel() == 0:
        return wav

    first = min(mono.numel(), int(active[0].item()) * frame_size)
    last = min(mono.numel(), (int(active[-1].item()) + 1) * frame_size)
    minimum_samples = int(round(float(sampling_rate) * minimum_ms / 1000.0))
    start = first if first >= minimum_samples else 0
    end = last if mono.numel() - last >= minimum_samples else mono.numel()
    if start >= end:
        return wav
    return wav[..., start:end]

PRONUNCIATION_ANNOTATION_PATTERN = re.compile(r'<([^|>\n]+)\|([^>\n]+)>')
"""
匹配发音标注格式：<文字|发音>
例如：<going|G OW1 . IH0 NG>，<行|XING2>
"""
def is_kana(s: str) -> bool:
    hira = re.compile(r'^[\u3040-\u309F]+$')
    kata = re.compile(r'^[\u30A0-\u30FF]+$')
    if hira.fullmatch(s):
        return True
    elif kata.fullmatch(s):
        return True
    return False

def apply_pronunciation_annotations(text: str) -> str:
    """
    处理发音标注格式 <文字|发音>，转换为特殊token包裹的发音
    - 文字含中文 -> <|SPECIAL_TOKEN_2|>发音<|SPECIAL_TOKEN_2|>
    - 文字为英文 -> <|SPECIAL_TOKEN_1|>发音<|SPECIAL_TOKEN_1|>
    发音内容统一转大写

    例如：
        <going|G OW1 . IH0 NG> -> <|SPECIAL_TOKEN_1|>G OW1 . IH0 NG<|SPECIAL_TOKEN_1|>
        <行|XING2>              -> <|SPECIAL_TOKEN_2|>XING2<|SPECIAL_TOKEN_2|>
    """
    def _replace(match):
        word = match.group(1)
        pronunciation = match.group(2).upper()
        has_chinese = bool(re.search(r"[\u4e00-\u9fff]", word))
        if is_kana(pronunciation):
            return f' {pronunciation} '
        else:
            token = 'SPECIAL_TOKEN_2' if has_chinese else 'SPECIAL_TOKEN_1'
            return f'<|{token}|>{pronunciation}<|{token}|>'
    return PRONUNCIATION_ANNOTATION_PATTERN.sub(_replace, text)


class IndexTTS2:
    def __init__(
            self, cfg_path="models/config.yaml", model_dir="models",
            runtime: RuntimeConfig | None = None, **legacy_kwargs
    ):
        """Load IndexTTS with a shared runtime contract and legacy keyword support."""
        runtime_was_supplied = runtime is not None
        self.runtime = RuntimeConfig.from_dict(runtime)
        if "device" in legacy_kwargs:
            legacy_device = legacy_kwargs.pop("device")
            if legacy_device is not None:
                self.runtime.device = str(legacy_device)
        if "use_bf16" in legacy_kwargs:
            self.runtime.gpt_dtype = "bf16" if bool(legacy_kwargs.pop("use_bf16")) else "fp32"
        if "use_fp16" in legacy_kwargs:
            self.runtime.gpt_dtype = "bf16" if bool(legacy_kwargs.pop("use_fp16")) else self.runtime.gpt_dtype
        if "use_cuda_kernel" in legacy_kwargs:
            legacy_cuda_kernel = legacy_kwargs.pop("use_cuda_kernel")
            if legacy_cuda_kernel is not None:
                self.runtime.use_cuda_kernel_bigvgan = bool(legacy_cuda_kernel)
        if "use_accel" in legacy_kwargs:
            self.runtime.use_accel = bool(legacy_kwargs.pop("use_accel"))
        if "use_torch_compile" in legacy_kwargs:
            self.runtime.torch_compile_s2mel = bool(legacy_kwargs.pop("use_torch_compile"))
        use_deepspeed = bool(legacy_kwargs.pop("use_deepspeed", False))
        load_qwen_emo = bool(legacy_kwargs.pop("use_qwen_emo", runtime_was_supplied))
        if legacy_kwargs:
            print(">> Ignoring unsupported legacy runtime options:", ", ".join(sorted(legacy_kwargs)))
        self.runtime.validate()

        resolved_device = device_from_string(self.runtime.device)
        self.device = str(resolved_device)
        self.runtime.device = self.device
        requested_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[self.runtime.gpt_dtype]
        if resolved_device.type == "cpu" and requested_dtype != torch.float32:
            print(">> CPU inference uses FP32 even though a reduced GPT dtype was requested.")
            requested_dtype = torch.float32
        if requested_dtype == torch.bfloat16 and resolved_device.type == "cuda" and not torch.cuda.is_bf16_supported():
            print(">> BF16 is unavailable on this GPU; GPT is falling back to FP32.")
            requested_dtype = torch.float32
        self.gpt_torch_dtype = requested_dtype
        self.use_bf16 = requested_dtype == torch.bfloat16
        self.dtype = None if requested_dtype == torch.float32 else requested_dtype
        self.use_cuda_kernel = bool(
            self.runtime.use_cuda_kernel_bigvgan and resolved_device.type == "cuda"
        )
        self.use_accel = bool(self.runtime.use_accel and resolved_device.type == "cuda")
        if self.use_accel:
            try:
                import flash_attn  # noqa: F401
            except (ImportError, OSError):
                print(">> GPT acceleration requested, but flash-attn is unavailable; using standard generation.")
                self.use_accel = False
        self.use_torch_compile = bool(self.runtime.torch_compile_s2mel)
        self.progress_reporter = None
        self.block_swap = None
        self._lora_handle = None
        self._lora_path = ""
        self._lora_strength = 1.0
        self._lora_merged = False
        self.runtime_warning = ""

        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        os.environ["HF_HUB_CACHE"] = os.path.join(self.model_dir, "hf_cache")
        self.stop_text_token = self.cfg.gpt.stop_text_token
        self.stop_mel_token = self.cfg.gpt.stop_mel_token
        print(">> Runtime:", describe(self.runtime))

        if self.runtime.model_variant == "int8_convrot":
            from indextts.utils import model_downloads

            int8_path = model_downloads.int8_gpt_path(self.model_dir)
            if not int8_path.is_file():
                print(
                    f">> INT8 ConvRot GPT is missing at {int8_path}; starting automatic download",
                    flush=True,
                )

                def download_progress(fraction, desc="", **_kwargs):
                    print(
                        ">> "
                        + model_downloads.int8_download_progress_message(
                            fraction, desc, models_dir=self.model_dir
                        ),
                        flush=True,
                    )

                try:
                    model_downloads.ensure_int8_gpt(
                        self.model_dir, download_progress
                    )
                    if not int8_path.is_file():
                        raise FileNotFoundError(
                            f"download returned without creating {int8_path}"
                        )
                except Exception as exc:
                    self.runtime_warning = model_downloads.int8_fallback_warning(
                        self.model_dir, exc
                    )
                    self.runtime.model_variant = "bf16"
                    print(">> " + self.runtime_warning, flush=True)

        # Detect low-VRAM GPUs (< 10 GB) to enable automatic text chunking
        self.low_vram = False
        if torch.cuda.is_available() and resolved_device.type == "cuda":
            dev_idx = int(self.device.split(":")[-1]) if ":" in self.device else 0
            total_vram_gb = torch.cuda.get_device_properties(dev_idx).total_memory / (1024 ** 3)
            if total_vram_gb < 10.0 or self.runtime.blocks_to_swap >= 12:
                self.low_vram = True
                reason = f"{total_vram_gb:.1f} GB GPU" if total_vram_gb < 10.0 else "aggressive block swap"
                print(f">> Low-VRAM mode enabled ({reason}); long text will be split into chunks")

        load_started = time.perf_counter()
        self.gpt = UnifiedVoice(
            **self.cfg.gpt,
            use_accel=self.use_accel,
            spk_cond_mode="campplus",
            attention_backend=self.runtime.attention_backend,
        )
        self.runtime.attention_backend = self.gpt.attention_backend
        self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        loaded_gpt_path = self.gpt_path
        if self.runtime.model_variant == "int8_convrot":
            int8_path = os.path.join(self.model_dir, "gpt_int8_convrot.safetensors")
            if not os.path.isfile(int8_path):
                raise FileNotFoundError(f"INT8 ConvRot checkpoint not found after download: {int8_path}")
            try:
                from indextts.quant.convrot_int8 import load_gpt_checkpoint
            except ImportError as exc:
                raise ImportError(
                    "The INT8 ConvRot runtime is unavailable; install/update the quant module before loading this variant."
                ) from exc
            self.gpt_load_report = load_gpt_checkpoint(
                self.gpt,
                int8_path,
                device=torch.device("cpu"),
                dtype=self.gpt_torch_dtype,
            )
            loaded_gpt_path = int8_path
        else:
            load_checkpoint(self.gpt, self.gpt_path)
            self.gpt_load_report = None
            if self.gpt_torch_dtype != torch.float32:
                self.gpt.to(dtype=self.gpt_torch_dtype)
        self.gpt.eval()
        print(">> GPT weights restored from:", loaded_gpt_path)

        if self.runtime.lora_path:
            self.set_lora(
                self.runtime.lora_path,
                self.runtime.lora_strength,
                merge_into_base=self.runtime.lora_merge_into_base,
            )

        blocks_to_swap = self.runtime.blocks_to_swap
        if blocks_to_swap == -1:
            if resolved_device.type != "cuda":
                blocks_to_swap = 0
            else:
                first_block_bytes = sum(
                    getattr(module, name).numel() * getattr(module, name).element_size()
                    for module, name in default_swap_tensor_selector(self.gpt.gpt.h[0])
                )
                free_bytes = int(gpu_free_gb(resolved_device.index or 0) * 1024**3)
                estimate = estimate_vram_gb(self.runtime, free_bytes / 1024**3)
                non_block_headroom = int(
                    (self.runtime.vram_reserve_gb + estimate["activations_gb"] + estimate["on_demand_peak_gb"])
                    * 1024**3
                )
                blocks_to_swap = resolve_blocks_to_swap(
                    -1,
                    len(self.gpt.gpt.h),
                    first_block_bytes,
                    max(0, free_bytes - non_block_headroom),
                    self.runtime.swap_ring_size,
                )
            print(f">> Automatic GPT block swap resolved to {blocks_to_swap}/24 blocks.")
        self.resolved_blocks_to_swap = blocks_to_swap

        if blocks_to_swap > 0 and resolved_device.type == "cuda":
            if self.use_accel:
                print(">> GPT acceleration engine is incompatible with block swap; using standard generation.")
                self.use_accel = False
                self.gpt.use_accel = False
            if use_deepspeed:
                print(">> DeepSpeed is incompatible with block swap; using standard generation.")
                use_deepspeed = False
            self.block_swap = enable_block_swap(
                list(self.gpt.gpt.h),
                blocks_to_swap,
                BlockSwapConfig(
                    device=resolved_device,
                    supports_backward=False,
                    use_pinned_memory=self.runtime.pin_swap_memory,
                    ring_size=self.runtime.swap_ring_size,
                ),
            )
            self._move_gpt_nonblocks(resolved_device)
            print(">> GPT block swap:", self.block_swap.summary())
        else:
            self.gpt.to(resolved_device)

        if use_deepspeed:
            try:
                import deepspeed
            except (ImportError, OSError, CalledProcessError) as e:
                use_deepspeed = False
                print(f">> Failed to load DeepSpeed. Falling back to normal inference. Error: {e}")

        self.gpt.post_init_gpt2_config(
            use_deepspeed=use_deepspeed,
            kv_cache=True,
            half=self.gpt_torch_dtype == torch.float16,
            dtype=self.gpt_torch_dtype,
        )
        self._log_model("gpt", self.gpt, load_started)

        if self.use_cuda_kernel:
            # preload the CUDA kernel for BigVGAN
            try:
                from indextts.s2mel.modules.bigvgan.alias_free_activation.cuda import activation1d

                print(">> Preload custom CUDA kernel for BigVGAN", activation1d.anti_alias_activation_cuda)
            except Exception as e:
                print(">> Failed to load custom CUDA kernel for BigVGAN. Falling back to torch.")
                print(f"{e!r}")
                self.use_cuda_kernel = False

        self.residency = ResidencyManager(resolved_device)

        qwen_started = time.perf_counter()
        if load_qwen_emo:
            qwen_policy = self.runtime.aux_residency["qwen_emo"]
            self.qwen_emo = QwenEmotion(
                os.path.join(self.model_dir, self.cfg.qwen_emo_path),
                dtype=(
                    torch.bfloat16
                    if resolved_device.type == "cuda" and qwen_policy != "cpu"
                    else torch.float32
                ),
            )
            self.residency.register("qwen_emo", self.qwen_emo, self.runtime.aux_residency["qwen_emo"])
            self._log_model("qwen_emo", self.qwen_emo.model, qwen_started)
        else:
            self.qwen_emo = None
            print(">> QwenEmotion not loaded (legacy use_qwen_emo=False)")

        w2v_started = time.perf_counter()
        w2v_bert_dir = os.path.join(self.model_dir, "hf_cache", "w2v-bert-2.0")
        if not os.path.isdir(w2v_bert_dir):
            from indextts.utils.model_download import ensure_models_available
            aux_paths = ensure_models_available(self.model_dir)
            w2v_bert_dir = aux_paths["w2v_bert"]
        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained(w2v_bert_dir, local_files_only=True)
        self.semantic_model = Wav2Vec2BertModel.from_pretrained(w2v_bert_dir, local_files_only=True)
        # Demo-audio validation found BF16 changed 47/525 semantic codec codes.
        # Keep this reference encoder in FP32; its normalized output also stays FP32.
        semantic_dtype = torch.float32
        self.semantic_model.to(dtype=semantic_dtype)
        self.semantic_model.eval()
        self.residency.register(
            "semantic_model", self.semantic_model, self.runtime.aux_residency["semantic_model"]
        )
        stat_mean_var = torch.load(os.path.join(self.model_dir, self.cfg.w2v_stat))
        self.semantic_mean = stat_mean_var["mean"].float().to(self.device)
        self.semantic_std = torch.sqrt(stat_mean_var["var"].float()).to(self.device)
        self._log_model("semantic_model", self.semantic_model, w2v_started)

        start_time = time.perf_counter()
        self.semantic_codec = EnhancedCodec(**self.cfg.semantic_codec, cfg=self.cfg.semantic_codec)
        codec_ckpt_path = os.path.join(self.model_dir, "codec.pth")
        self.semantic_codec.load_checkpoint(codec_ckpt_path)
        print('>> semantic_codec weights restored from:', codec_ckpt_path)
        self.semantic_codec.eval()
        self.residency.register(
            "semantic_codec", self.semantic_codec, self.runtime.aux_residency["semantic_codec"]
        )
        print('>> semantic_codec weights restored cost: ', time.perf_counter() - start_time)
        self._log_model("semantic_codec", self.semantic_codec, start_time)

        s2mel_started = time.perf_counter()
        s2mel_path = os.path.join(self.model_dir, self.cfg.s2mel_checkpoint)
        s2mel = MyModel(self.cfg.s2mel)
        s2mel, _, _, _ = load_checkpoint2(
            s2mel,
            None,
            s2mel_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        self.s2mel = s2mel.float()
        self.s2mel.eval()
        self.s2mel.models["cfm"].estimator_autocast_dtype = (
            torch.bfloat16 if self.runtime.s2mel_estimator_autocast else None
        )
        self.residency.register("s2mel", self.s2mel, self.runtime.aux_residency["s2mel"])
        if self.runtime.aux_residency["s2mel"] == "gpu" or resolved_device.type == "cpu":
            self.s2mel.models['cfm'].estimator.setup_caches(
                max_batch_size=1,
                max_seq_length=self.runtime.cfm_cache_length,
            )

        # Enable torch.compile optimization if requested
        if self.use_torch_compile:
            print(">> Enabling torch.compile optimization")
            self.s2mel.enable_torch_compile()
            print(">> torch.compile optimization enabled successfully")

        print(">> s2mel weights restored from:", s2mel_path)
        self._log_model("s2mel", self.s2mel, s2mel_started)

        # load campplus_model
        campplus_ckpt_path = os.path.join(self.model_dir, "hf_cache", "campplus_cn_common.bin")
        if not os.path.isfile(campplus_ckpt_path):
            from indextts.utils.model_download import ensure_models_available
            aux_paths = ensure_models_available(self.model_dir)
            campplus_ckpt_path = aux_paths["campplus"]
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        self.campplus_model = campplus_model
        self.campplus_model.eval()
        self.residency.register("campplus", self.campplus_model, self.runtime.aux_residency["campplus"])
        print(">> campplus_model weights restored from:", campplus_ckpt_path)
        self._log_model("campplus", self.campplus_model)

        bigvgan_started = time.perf_counter()
        bigvgan_dir = os.path.join(self.model_dir, "hf_cache", "bigvgan")
        if not os.path.isdir(bigvgan_dir):
            from indextts.utils.model_download import ensure_models_available
            aux_paths = ensure_models_available(self.model_dir)
            bigvgan_dir = aux_paths["bigvgan"]
        self.bigvgan = bigvgan.BigVGAN.from_pretrained(bigvgan_dir, use_cuda_kernel=self.use_cuda_kernel)
        self.bigvgan.remove_weight_norm()
        self.bigvgan.eval()
        self.residency.register("bigvgan", self.bigvgan, self.runtime.aux_residency["bigvgan"])
        print(">> bigvgan weights restored from:", bigvgan_dir)
        self._log_model("bigvgan", self.bigvgan, bigvgan_started)

        self.tokenizer = get_tokenizer(multilingual=True, model_dir=self.model_dir)
        self.ja_text_process = JapaneseG2PProcessor(g2p_ratio=0)
        self.text_process = TextNormalizer(enable_glossary=True)
        self.text_process.load()

        # 加载术语词汇表（如果存在）
        self.glossary_path = os.path.join(self.model_dir, "glossary.yaml")
        if os.path.exists(self.glossary_path):
            self.text_process.load_glossary_from_yaml(self.glossary_path)
            print(">> Glossary loaded from:", self.glossary_path)

        emo_matrix = torch.load(os.path.join(self.model_dir, self.cfg.emo_matrix))
        self.emo_matrix = emo_matrix.to(self.device)
        self.emo_num = list(self.cfg.emo_num)

        spk_matrix = torch.load(os.path.join(self.model_dir, self.cfg.spk_matrix))
        self.spk_matrix = spk_matrix.to(self.device)

        self.emo_matrix = torch.split(self.emo_matrix, self.emo_num)
        self.spk_matrix = torch.split(self.spk_matrix, self.emo_num)

        mel_fn_args = {
            "n_fft": self.cfg.s2mel['preprocess_params']['spect_params']['n_fft'],
            "win_size": self.cfg.s2mel['preprocess_params']['spect_params']['win_length'],
            "hop_size": self.cfg.s2mel['preprocess_params']['spect_params']['hop_length'],
            "num_mels": self.cfg.s2mel['preprocess_params']['spect_params']['n_mels'],
            "sampling_rate": self.cfg.s2mel["preprocess_params"]["sr"],
            "fmin": self.cfg.s2mel['preprocess_params']['spect_params'].get('fmin', 0),
            "fmax": None if self.cfg.s2mel['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
            "center": False
        }
        self.mel_fn = lambda x: mel_spectrogram(x, **mel_fn_args)

        # 缓存参考音频：
        self.cache_spk_cond = None
        self.cache_s2mel_style = None
        self.cache_s2mel_prompt = None
        self.cache_spk_audio_prompt = None
        self.cache_spk_prompt_key = None
        self.cache_emo_cond = None
        self.cache_emo_audio_prompt = None
        self.cache_emo_prompt_key = None
        self.cache_mel = None

        # 进度引用显示（可选）
        self.gr_progress = None
        self.model_version = self.cfg.version if hasattr(self.cfg, "version") else None
        self.residency.report()

    def _move_gpt_nonblocks(self, device):
        """Move UnifiedVoice except its managed GPT2 block list."""
        device = torch.device(device)
        for name, child in self.gpt.named_children():
            if name != "gpt":
                child.to(device)
                continue
            for gpt_name, gpt_child in child.named_children():
                if gpt_name != "h":
                    gpt_child.to(device)

    def _log_model(self, name, module, started_at=None):
        parameter = next(module.parameters(), None) if isinstance(module, torch.nn.Module) else None
        dtype = str(parameter.dtype).replace("torch.", "") if parameter is not None else "n/a"
        location = str(parameter.device) if parameter is not None else "n/a"
        elapsed = time.perf_counter() - started_at if started_at is not None else None
        stats = memory_stats(self.device)
        timing = f" | load {elapsed:.2f}s" if elapsed is not None else ""
        print(
            f">> {name}: dtype={dtype}, device={location}{timing} | "
            f"VRAM {stats['allocated_gb']:.2f} GB allocated / {stats['reserved_gb']:.2f} GB reserved"
        )

    def set_lora(self, path, strength=1.0, merge_into_base=None):
        """Replace the active LoRA / DoRA without reloading base weights."""
        requested = str(path or "")
        strength = max(0.0, min(4.0, float(strength)))
        merge_requested = (
            bool(self.runtime.lora_merge_into_base)
            if merge_into_base is None
            else bool(merge_into_base)
        )
        try:
            from indextts.lora.apply import (
                apply_lora,
                merge_lora_for_inference,
                move_adapters_to_device,
                remove_lora,
                set_lora_strength,
                unmerge_lora_from_model,
            )
        except ImportError:
            if requested:
                print(">> LoRA / DoRA support is not installed yet; continuing without one.")
            return None
        resolved = os.path.abspath(requested) if requested else ""
        if resolved and resolved == self._lora_path and self._lora_handle is not None:
            if self._lora_merged:
                unmerge_lora_from_model(self.gpt)
                self._lora_merged = False
            set_lora_strength(self._lora_handle, strength)
            self._lora_strength = strength
            self.runtime.lora_strength = strength
            self.runtime.lora_merge_into_base = merge_requested
            if merge_requested and self.runtime.model_variant == "bf16":
                merge_lora_for_inference(self.gpt)
                self._lora_merged = True
            return self._lora_handle
        if self._lora_handle is not None or self._lora_path:
            if self._lora_merged:
                unmerge_lora_from_model(self.gpt)
                self._lora_merged = False
            remove_lora(self.gpt)
            self._lora_handle = None
            self._lora_path = ""
        if not requested:
            self.runtime.lora_path = ""
            self.runtime.lora_strength = strength
            self.runtime.lora_merge_into_base = merge_requested
            return None
        if not os.path.isfile(requested):
            raise FileNotFoundError(f"LoRA / DoRA not found: {requested}")
        self._lora_handle = apply_lora(self.gpt, requested, strength)
        move_adapters_to_device(self.gpt, self.device)
        self._lora_path = resolved
        self._lora_strength = strength
        self.runtime.lora_path = requested
        self.runtime.lora_strength = strength
        self.runtime.lora_merge_into_base = merge_requested
        if merge_requested and self.runtime.model_variant == "bf16":
            merge_lora_for_inference(self.gpt)
            self._lora_merged = True
        elif merge_requested:
            print(">> LoRA / DoRA merge requested but skipped because the GPT base is INT8.")
        print(
            f">> LoRA / DoRA active: {requested} (strength {strength:.3f}, "
            f"merged={'yes' if self._lora_merged else 'no'})"
        )
        return self._lora_handle

    def unload(self):
        """Move all models to CPU, release CUDA caches, and return freed GiB."""
        before = memory_stats(self.device)["allocated_gb"]
        if hasattr(self, "s2mel"):
            transformer = self.s2mel.models['cfm'].estimator.transformer
            transformer.freqs_cis = None
            transformer.causal_mask = None
            transformer.max_batch_size = -1
            transformer.max_seq_length = -1
        if self.block_swap is not None:
            try:
                self.block_swap.remove(to_cpu=True)
            except TypeError:
                self.block_swap.remove()
            self.block_swap = None
        if hasattr(self, "residency"):
            self.residency.to_cpu_all()
        if hasattr(self, "gpt"):
            if hasattr(self.gpt, "inference_model"):
                self.gpt.inference_model.cached_mel_emb = None
            self.gpt.to("cpu")
        for name in (
            "cache_spk_cond",
            "cache_s2mel_style",
            "cache_s2mel_prompt",
            "cache_emo_cond",
            "cache_mel",
        ):
            setattr(self, name, None)
        if hasattr(self, "emo_matrix"):
            self.emo_matrix = tuple(tensor.cpu() for tensor in self.emo_matrix)
        if hasattr(self, "spk_matrix"):
            self.spk_matrix = tuple(tensor.cpu() for tensor in self.spk_matrix)
        if hasattr(self, "semantic_mean"):
            self.semantic_mean = self.semantic_mean.cpu()
            self.semantic_std = self.semantic_std.cpu()
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        after = memory_stats(self.device)["allocated_gb"]
        freed = max(0.0, before - after)
        print(f">> IndexTTS unloaded; freed {freed:.2f} GB of allocated VRAM.")
        return freed

    def _setup_s2mel_caches(self, max_batch_size, max_seq_length):
        estimator = self.s2mel.models['cfm'].estimator
        transformer = estimator.transformer
        model_device = next(estimator.parameters()).device
        freqs = getattr(transformer, "freqs_cis", None)
        if isinstance(freqs, torch.Tensor) and freqs.device != model_device:
            transformer.freqs_cis = None
            transformer.causal_mask = None
            transformer.max_batch_size = -1
            transformer.max_seq_length = -1
        estimator.setup_caches(
            max_batch_size=max(1, int(max_batch_size)),
            max_seq_length=max(1, int(max_seq_length)),
        )

    def _release_s2mel_caches_if_needed(self):
        if self.runtime.aux_residency.get("s2mel") != "on_demand":
            return
        transformer = self.s2mel.models['cfm'].estimator.transformer
        transformer.freqs_cis = None
        transformer.causal_mask = None
        transformer.max_batch_size = -1
        transformer.max_seq_length = -1

    @contextmanager
    def _use_s2mel(self):
        with self.residency.use("s2mel"):
            try:
                yield self.s2mel
            finally:
                self._release_s2mel_caches_if_needed()

    @torch.no_grad()
    def get_emb(self, input_features, attention_mask, semantic_layer=17):
        with self.residency.use("semantic_model"):
            model_dtype = next(self.semantic_model.parameters()).dtype
            model_device = next(self.semantic_model.parameters()).device
            vq_emb = self.semantic_model(
                input_features=input_features.to(device=model_device, dtype=model_dtype),
                attention_mask=attention_mask.to(model_device),
                output_hidden_states=True,
            )
        semantic_layer = max(1, min(int(semantic_layer), len(vq_emb.hidden_states) - 1))
        feat = vq_emb.hidden_states[semantic_layer].to(device=self.device, dtype=torch.float32)
        feat = (feat - self.semantic_mean) / self.semantic_std
        return feat

    def _campplus_fbank(self, audio_16k):
        reference_device = (
            torch.device("cpu")
            if self.residency.policy("campplus") == "cpu"
            else torch.device(self.device)
        )
        return torchaudio.compliance.kaldi.fbank(
            audio_16k.to(device=reference_device, dtype=torch.float32),
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000,
        )

    @torch.no_grad()
    def get_scode(self, inputs):
        with self.residency.use("semantic_codec"):
            semantic_code, feat = self.semantic_codec.quantize(inputs)
        # vq = self.semantic_codec.quantizer.vq2emb(semantic_code.unsqueeze(1))
        # vq = vq.transpose(1,2)
        return semantic_code

    def remove_long_silence(self, codes: torch.Tensor, silent_token=52, max_consecutive=30):
        """
        Shrink special tokens (silent_token and stop_mel_token) in codes
        codes: [B, T]
        """
        code_lens = []
        codes_list = []
        device = codes.device
        dtype = codes.dtype
        isfix = False
        for i in range(0, codes.shape[0]):
            code = codes[i]
            if not torch.any(code == self.stop_mel_token).item():
                len_ = code.size(0)
            else:
                stop_mel_idx = (code == self.stop_mel_token).nonzero(as_tuple=False)
                len_ = stop_mel_idx[0].item() if len(stop_mel_idx) > 0 else code.size(0)

            count = torch.sum(code == silent_token).item()
            if count > max_consecutive:
                # code = code.cpu().tolist()
                ncode_idx = []
                n = 0
                for k in range(len_):
                    assert code[
                               k] != self.stop_mel_token, f"stop_mel_token {self.stop_mel_token} should be shrinked here"
                    if code[k] != silent_token:
                        ncode_idx.append(k)
                        n = 0
                    elif code[k] == silent_token and n < 10:
                        ncode_idx.append(k)
                        n += 1
                    # if (k == 0 and code[k] == 52) or (code[k] == 52 and code[k-1] == 52):
                    #    n += 1
                # new code
                len_ = len(ncode_idx)
                codes_list.append(code[ncode_idx])
                isfix = True
            else:
                # shrink to len_
                codes_list.append(code[:len_])
            code_lens.append(len_)
        if isfix:
            if len(codes_list) > 1:
                codes = pad_sequence(codes_list, batch_first=True, padding_value=self.stop_mel_token)
            else:
                codes = codes_list[0].unsqueeze(0)
        else:
            # unchanged
            pass
        # clip codes to max length
        max_len = max(code_lens)
        if max_len < codes.shape[1]:
            codes = codes[:, :max_len]
        code_lens = torch.tensor(code_lens, dtype=torch.long, device=device)
        return codes, code_lens

    def interval_silence(self, wavs, sampling_rate=22050, interval_silence=200):
        """
        Silences to be insert between generated segments.
        """

        if not wavs or interval_silence <= 0:
            return wavs

        # get channel_size
        channel_size = wavs[0].size(0)
        # get silence tensor
        sil_dur = int(sampling_rate * interval_silence / 1000.0)
        return torch.zeros(channel_size, sil_dur)

    def insert_interval_silence(self, wavs, sampling_rate=22050, interval_silence=200):
        """
        Insert silences between generated segments.
        wavs: List[torch.tensor]
        """

        if not wavs or interval_silence <= 0:
            return wavs

        # get channel_size
        channel_size = wavs[0].size(0)
        # get silence tensor
        sil_dur = int(sampling_rate * interval_silence / 1000.0)
        sil_tensor = torch.zeros(channel_size, sil_dur)

        wavs_list = []
        for i, wav in enumerate(wavs):
            wavs_list.append(wav)
            if i < len(wavs) - 1:
                wavs_list.append(sil_tensor)

        return wavs_list

    def _set_gr_progress(self, value, desc):
        if self.gr_progress is not None:
            self.gr_progress(value, desc=desc)

    def _progress_log(self, message):
        if self.progress_reporter is not None:
            self.progress_reporter.log(message)
        else:
            print(message)

    def _load_and_cut_audio(self,audio_path,max_audio_length_seconds,verbose=False,sr=None):
        if not sr:
            audio, sr = librosa.load(audio_path)
        else:
            audio, _ = librosa.load(audio_path,sr=sr)
        audio = torch.tensor(audio).unsqueeze(0)
        max_audio_samples = int(max_audio_length_seconds * sr)

        if audio.shape[1] > max_audio_samples:
            if verbose:
                print(f"Audio too long ({audio.shape[1]} samples), truncating to {max_audio_samples} samples")
            audio = audio[:, :max_audio_samples]
        return audio, sr

    SPLIT_PROTECTED_PATTERN = re.compile(r'<\|SPECIAL_TOKEN_\d+\|>.*?<\|SPECIAL_TOKEN_\d+\|>')

    def _token_len(self, text):
        return len(self.tokenizer.encode(text, allowed_special='all'))

    def _split_atomic_pieces(self, text):
        return split_atomic_pieces(text)

    def split_text_by_tokens(
        self,
        text,
        max_tokens,
        lang_prefix="",
        segment_budget_scale_non_cjk=DEFAULT_NON_CJK_BUDGET_SCALE,
    ):
        capacity = self.gpt.text_pos_embedding.emb.num_embeddings
        return shared_split_text_by_tokens(
            text,
            max_tokens,
            capacity=capacity,
            token_len=self._token_len,
            lang_prefix=lang_prefix,
            segment_budget_scale_non_cjk=segment_budget_scale_non_cjk,
        )

    @staticmethod
    def split_text_by_punctuation(text, max_chars=40):
        """Split text into segments of at most `max_chars` characters,
        breaking at punctuation boundaries. If a segment exceeds the limit
        and contains no punctuation, it is kept as-is to avoid mid-word splits."""
        import re
        # Split while keeping the delimiter attached to the preceding segment
        parts = re.split(r'(?<=[，。！？、；：,\.!\?;:\n])', text)
        segments = []
        current = ""
        for part in parts:
            if not part:
                continue
            if len(current) + len(part) <= max_chars:
                current += part
            else:
                if current:
                    segments.append(current)
                current = part
        if current:
            segments.append(current)
        return segments

    def _process_text_chunk(self, text, lang, text_normalization):
        language = str(lang or "EN").lower()
        value = self.text_process.clean_pattern.sub(
            lambda match: self.text_process.char_rep_map[match.group()],
            str(text),
        )
        if text_normalization:
            if language in {"zh", "zhen", "en"}:
                value = self.text_process.normalize(value)
            elif language in {"ja", "es"}:
                value = nemo_text_normalize(value, language)
        if language in {"ja", "zh", "zhen", "en"}:
            value = value.lower()
        elif language == "es":
            value = value.upper()
        value = apply_pronunciation_annotations(value)
        if language == "ja":
            value = self.ja_text_process.process_ja_text(value)
        return re.sub(r"<\|([^|]+)\|>", lambda match: f"<|{match.group(1).upper()}|>", value)

    def _build_text_plan(
        self,
        text,
        lang,
        max_tokens,
        text_normalization,
        interval_silence,
        enable_pause_tags,
        segment_budget_scale_non_cjk,
    ):
        chunks = split_text_with_pauses(text) if enable_pause_tags else [TextChunk(str(text))]
        lang_prefix = f"<|{str(lang or 'EN').lower()}|> "
        segments = []
        plan = []
        interval_samples = max(0, int(round(SAMPLE_RATE * float(interval_silence) / 1000.0)))
        for chunk in chunks:
            if isinstance(chunk, PauseChunk):
                pause_samples = max(0, int(round(SAMPLE_RATE * chunk.duration_s)))
                if pause_samples:
                    plan.append(("silence", pause_samples))
                continue
            if not chunk.text.strip():
                continue
            processed = self._process_text_chunk(chunk.text, lang, text_normalization)
            chunk_segments = self.split_text_by_tokens(
                processed,
                max_tokens,
                lang_prefix,
                segment_budget_scale_non_cjk,
            )
            chunk_segments = [item for item in chunk_segments if item]
            for index, segment in enumerate(chunk_segments):
                segment_index = len(segments)
                segments.append(segment)
                plan.append(("segment", segment_index))
                if interval_samples and index < len(chunk_segments) - 1:
                    plan.append(("silence", interval_samples))
        return segments, plan, lang_prefix

    @staticmethod
    def _assemble_audio_plan(segment_wavs, plan, sampling_rate=SAMPLE_RATE):
        del sampling_rate
        template = next((item for item in segment_wavs if item is not None), None)
        channels = int(template.shape[0]) if template is not None else 1
        dtype = template.dtype if template is not None else torch.float32
        parts = []
        for kind, value in plan:
            if kind == "segment":
                parts.append(segment_wavs[value])
            else:
                parts.append(torch.zeros(channels, int(value), dtype=dtype))
        if not parts:
            return torch.zeros(channels, 0, dtype=dtype)
        return torch.cat(parts, dim=1)

    @staticmethod
    def _fit_target_samples(wav, target_samples, mode):
        target_samples = max(0, int(target_samples))
        current = int(wav.shape[-1])
        if mode == "pad" and current < target_samples:
            return F.pad(wav, (0, target_samples - current))
        if mode == "trim" and current > target_samples:
            return wav[..., :target_samples]
        return wav

    def _reset_peak_vram(self):
        device = torch.device(self.device)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

    def _peak_vram_gb(self):
        device = torch.device(self.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated(device) / 1024**3

    def _record_generation_stats(
        self,
        *,
        seed,
        segments_count,
        audio_seconds,
        wall_time,
        gpt_time,
        s2mel_time,
        vocoder_time,
        generated_tokens,
        **extra,
    ):
        rtf = wall_time / audio_seconds if audio_seconds > 0 else 0.0
        stats = {
            "seed": int(seed),
            "segments_count": int(segments_count),
            "audio_seconds": float(audio_seconds),
            "rtf": float(rtf),
            "gpt_time": float(gpt_time),
            "s2mel_time": float(s2mel_time),
            "vocoder_time": float(vocoder_time),
            "peak_vram_gb": float(self._peak_vram_gb()),
            "generated_tokens": int(generated_tokens),
            "mel_tokens_per_s": float(generated_tokens / gpt_time) if gpt_time > 0 else 0.0,
            "wall_time_s": float(wall_time),
            # Compatibility aliases used by the runtime benchmark task.
            "gpt_time_s": float(gpt_time),
            "s2mel_time_s": float(s2mel_time),
            "vocoder_time_s": float(vocoder_time),
        }
        stats.update(extra)
        self.last_generation_stats = stats
        self._progress_log(">> === Generation summary ===")
        self._progress_log(
            f">> seed={stats['seed']} | segments={stats['segments_count']} | "
            f"audio={stats['audio_seconds']:.3f}s | RTF={stats['rtf']:.4f}"
        )
        self._progress_log(
            f">> GPT={stats['gpt_time']:.3f}s | s2mel={stats['s2mel_time']:.3f}s | "
            f"vocoder={stats['vocoder_time']:.3f}s | peak VRAM={stats['peak_vram_gb']:.3f} GB"
        )
        self._progress_log(">> ==========================")
        return stats

    @torch.no_grad()
    def _render_codes_segment(
        self,
        codes,
        prompt_condition,
        ref_mel,
        style,
        *,
        duration_factor,
        cfm_cache_length,
        diffusion_steps,
        inference_cfg_rate,
        cfm_temperature,
        trim_silence_ms_threshold,
    ):
        codes = codes.to(self.device)
        s2mel_started = time.perf_counter()
        with self.residency.use("semantic_codec"):
            semantic = self.semantic_codec.decode(codes)
        target_length = max(1, int(round(semantic.shape[1] * 1.72 * float(duration_factor))))
        target_lengths = torch.tensor([target_length], dtype=torch.long, device=codes.device)
        with self._use_s2mel():
            cond = self.s2mel.models["length_regulator"](
                semantic,
                ylens=target_lengths,
                n_quantizers=3,
                f0=None,
            )[0]
            cat_condition = torch.cat([prompt_condition, cond], dim=1)
            required_cache_length = max(int(cfm_cache_length), int(cat_condition.size(1)))
            cfm_batch_size = 2 if inference_cfg_rate > 0 else 1
            self._setup_s2mel_caches(cfm_batch_size, required_cache_length)
            vc_target = self.s2mel.models["cfm"].inference(
                cat_condition,
                torch.tensor([cat_condition.size(1)], dtype=torch.long, device=cond.device),
                ref_mel,
                style,
                None,
                int(diffusion_steps),
                temperature=float(cfm_temperature),
                inference_cfg_rate=float(inference_cfg_rate),
            )
            vc_target = vc_target[:, :, ref_mel.size(-1):]
        s2mel_elapsed = time.perf_counter() - s2mel_started

        vocoder_started = time.perf_counter()
        with self.residency.use("bigvgan"):
            wav = self.bigvgan(vc_target.float()).squeeze().unsqueeze(0)
        vocoder_elapsed = time.perf_counter() - vocoder_started
        wav = torch.clamp(32767 * wav.squeeze(1), -32767.0, 32767.0).cpu()
        wav = trim_segment_silence(
            wav,
            sampling_rate=SAMPLE_RATE,
            minimum_silence_ms=trim_silence_ms_threshold,
        )
        return wav, s2mel_elapsed, vocoder_elapsed
    
    def normalize_emo_vec(self, emo_vector, apply_bias=True):
        # apply biased emotion factors for better user experience,
        # by de-emphasizing emotions that can cause strange results
        if apply_bias:
            # [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
            emo_bias = [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]
            emo_vector = [vec * bias for vec, bias in zip(emo_vector, emo_bias)]

        # the total emotion sum must be 0.8 or less
        emo_sum = sum(emo_vector)
        if emo_sum > 0.8:
            scale_factor = 0.8 / emo_sum
            emo_vector = [vec * scale_factor for vec in emo_vector]

        return emo_vector

    def _reset_generation_cache(self):
        """Release the autoregressive cache between segments when requested."""
        try:
            inference_model = self.gpt.inference_model
            cache = getattr(inference_model, "_cache", None)
            if cache is not None and hasattr(cache, "reset"):
                cache.reset()
            if hasattr(inference_model, "cached_mel_emb"):
                inference_model.cached_mel_emb = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # 原始推理模式
    def infer(self, spk_audio_prompt, text, output_path, lang="EN",
              emo_audio_prompt=None, emo_alpha=1.0,
              emo_vector=None, use_emo_text=False, emo_text=None, use_random=False, interval_silence=200,
              verbose=False, max_text_tokens_per_segment=120, stream_return=False, more_segment_before=0,
              duration_factor=1.0, text_normalization=True,
              max_speaker_audio_length=15, max_emotion_audio_length=15,
              max_consecutive_silence=0, semantic_layer=17, cfm_cache_length=None,
              diffusion_steps=25, inference_cfg_rate=0.7,
              reset_beam_cache_per_segment=False,
              segment_budget_scale_non_cjk=DEFAULT_NON_CJK_BUDGET_SCALE,
              cfm_temperature=1.0, seed=None, reuse_spk_cond_for_emo=False,
              enable_pause_tags=True, trim_silence_ms_threshold=0,
              target_duration_s=None, target_duration_mode="off",
              **generation_kwargs):
        if cfm_cache_length is None:
            cfm_cache_length = self.runtime.cfm_cache_length
        if stream_return:
            return self.infer_generator(
                spk_audio_prompt, text, output_path, lang,
                emo_audio_prompt, emo_alpha,
                emo_vector,
                use_emo_text, emo_text, use_random, interval_silence,
                verbose, max_text_tokens_per_segment, stream_return, more_segment_before,
                duration_factor=duration_factor,
                text_normalization=text_normalization,
                max_speaker_audio_length=max_speaker_audio_length,
                max_emotion_audio_length=max_emotion_audio_length,
                max_consecutive_silence=max_consecutive_silence,
                semantic_layer=semantic_layer,
                cfm_cache_length=cfm_cache_length,
                diffusion_steps=diffusion_steps,
                inference_cfg_rate=inference_cfg_rate,
                reset_beam_cache_per_segment=reset_beam_cache_per_segment,
                segment_budget_scale_non_cjk=segment_budget_scale_non_cjk,
                cfm_temperature=cfm_temperature,
                seed=seed,
                reuse_spk_cond_for_emo=reuse_spk_cond_for_emo,
                enable_pause_tags=enable_pause_tags,
                trim_silence_ms_threshold=trim_silence_ms_threshold,
                target_duration_s=target_duration_s,
                target_duration_mode=target_duration_mode,
                **generation_kwargs
            )
        else:
            try:
                return list(self.infer_generator(
                    spk_audio_prompt, text, output_path, lang,
                    emo_audio_prompt, emo_alpha,
                    emo_vector,
                    use_emo_text, emo_text, use_random, interval_silence,
                    verbose, max_text_tokens_per_segment, stream_return, more_segment_before,
                    duration_factor=duration_factor,
                    text_normalization=text_normalization,
                    max_speaker_audio_length=max_speaker_audio_length,
                    max_emotion_audio_length=max_emotion_audio_length,
                    max_consecutive_silence=max_consecutive_silence,
                    semantic_layer=semantic_layer,
                    cfm_cache_length=cfm_cache_length,
                    diffusion_steps=diffusion_steps,
                    inference_cfg_rate=inference_cfg_rate,
                    reset_beam_cache_per_segment=reset_beam_cache_per_segment,
                    segment_budget_scale_non_cjk=segment_budget_scale_non_cjk,
                    cfm_temperature=cfm_temperature,
                    seed=seed,
                    reuse_spk_cond_for_emo=reuse_spk_cond_for_emo,
                    enable_pause_tags=enable_pause_tags,
                    trim_silence_ms_threshold=trim_silence_ms_threshold,
                    target_duration_s=target_duration_s,
                    target_duration_mode=target_duration_mode,
                    **generation_kwargs
                ))[0]
            except IndexError:
                return None

    def infer_texts(self, spk_audio_prompt, texts, on_text_complete=None, lang="EN", **kwargs):
        """Generate text units in real micro-batches with shared reference conditioning."""
        kwargs = dict(kwargs)
        actual_seed = _seed_everything(kwargs.pop("seed", None))
        self._reset_peak_vram()
        request_started_at = time.perf_counter()
        section_batch_size = max(1, int(kwargs.pop("section_batch_size", 1)))
        console_progress_enabled = bool(kwargs.pop("console_progress_enabled", True))
        console_progress_label = kwargs.pop("console_progress_label", "Batch synthesis")
        kwargs.pop("console_progress_item_label", None)

        def run_sequential():
            results = []
            total = len(texts)
            reporter = self.progress_reporter
            audio_seconds = 0.0
            totals = {
                "segments_count": 0,
                "gpt_time": 0.0,
                "s2mel_time": 0.0,
                "vocoder_time": 0.0,
                "generated_tokens": 0,
                "peak_vram_gb": 0.0,
            }
            if reporter is not None:
                reporter.label = "segments"
                reporter.set_stage("synthesis")
                reporter.update(0, total=total, desc="preparing segments")
                self.progress_reporter = None
            try:
                for index, unit_text in enumerate(texts):
                    if console_progress_enabled and reporter is None:
                        print(f">> {console_progress_label} {index + 1}/{total}")
                    result = self.infer(
                        spk_audio_prompt=spk_audio_prompt,
                        text=unit_text,
                        output_path=None,
                        lang=lang,
                        seed=(actual_seed + index) % (2**32),
                        **dict(kwargs),
                    )
                    results.append(result)
                    unit_stats = dict(getattr(self, "last_generation_stats", {}))
                    for key in ("segments_count", "gpt_time", "s2mel_time", "vocoder_time", "generated_tokens"):
                        totals[key] += unit_stats.get(key, 0)
                    totals["peak_vram_gb"] = max(
                        totals["peak_vram_gb"],
                        float(unit_stats.get("peak_vram_gb", 0.0)),
                    )
                    if isinstance(result, tuple) and len(result) == 2:
                        audio_seconds += result[1].shape[0] / float(result[0])
                    if reporter is not None:
                        reporter.update(
                            index + 1,
                            total=total,
                            desc=f"speech synthesis {index + 1}/{total}",
                            extra={"audio_seconds": audio_seconds, "speed_unit": "x RT"},
                        )
                    if on_text_complete is not None:
                        on_text_complete(index, result)
            finally:
                if reporter is not None:
                    self.progress_reporter = reporter
                    reporter.finish()
            self._record_generation_stats(
                seed=actual_seed,
                segments_count=totals["segments_count"],
                audio_seconds=audio_seconds,
                wall_time=time.perf_counter() - request_started_at,
                gpt_time=totals["gpt_time"],
                s2mel_time=totals["s2mel_time"],
                vocoder_time=totals["vocoder_time"],
                generated_tokens=totals["generated_tokens"],
                peak_vram_gb=totals["peak_vram_gb"],
            )
            return results

        target_duration_mode = str(kwargs.get("target_duration_mode", "off") or "off").lower()
        if target_duration_mode not in _TARGET_DURATION_MODES:
            raise ValueError(f"target_duration_mode must be one of {sorted(_TARGET_DURATION_MODES)}")
        if section_batch_size == 1 or self.low_vram or target_duration_mode != "off":
            if self.low_vram and section_batch_size > 1:
                print(">> Low-VRAM mode uses section batch size 1")
            return run_sequential()

        kwargs = _batched_generation_kwargs(kwargs)

        use_emo_text = bool(kwargs.pop("use_emo_text", False))
        emo_text = kwargs.pop("emo_text", None)
        # A blank emotion prompt asks Qwen to analyze each unit independently.
        # Keep that behavior exact rather than sharing one vector across a batch.
        if use_emo_text and not emo_text:
            kwargs["use_emo_text"] = use_emo_text
            kwargs["emo_text"] = emo_text
            return run_sequential()

        emo_audio_prompt = kwargs.pop("emo_audio_prompt", None)
        emo_alpha = float(kwargs.pop("emo_alpha", 1.0))
        emo_vector = kwargs.pop("emo_vector", None)
        use_random = bool(kwargs.pop("use_random", False))
        interval_silence = int(kwargs.pop("interval_silence", 200))
        verbose = bool(kwargs.pop("verbose", False))
        max_text_tokens_per_segment = int(kwargs.pop("max_text_tokens_per_segment", 120))
        duration_factor = float(kwargs.pop("duration_factor", 1.0))
        text_normalization = bool(kwargs.pop("text_normalization", True))
        max_speaker_audio_length = float(kwargs.pop("max_speaker_audio_length", 15))
        max_emotion_audio_length = float(kwargs.pop("max_emotion_audio_length", 15))
        max_consecutive_silence = int(kwargs.pop("max_consecutive_silence", 0))
        semantic_layer = int(kwargs.pop("semantic_layer", 17))
        cfm_cache_length = int(kwargs.pop("cfm_cache_length", self.runtime.cfm_cache_length))
        diffusion_steps = int(kwargs.pop("diffusion_steps", 25))
        inference_cfg_rate = float(kwargs.pop("inference_cfg_rate", 0.7))
        reset_beam_cache_per_segment = bool(kwargs.pop("reset_beam_cache_per_segment", False))
        segment_budget_scale_non_cjk = float(
            kwargs.pop("segment_budget_scale_non_cjk", DEFAULT_NON_CJK_BUDGET_SCALE)
        )
        cfm_temperature = float(kwargs.pop("cfm_temperature", 1.0))
        reuse_spk_cond_for_emo = bool(kwargs.pop("reuse_spk_cond_for_emo", False))
        enable_pause_tags = bool(kwargs.pop("enable_pause_tags", True))
        trim_silence_ms_threshold = float(kwargs.pop("trim_silence_ms_threshold", 0) or 0)
        if not 0.0 < segment_budget_scale_non_cjk <= 1.0:
            raise ValueError("segment_budget_scale_non_cjk must be in the range (0, 1]")
        if not math.isfinite(cfm_temperature) or cfm_temperature < 0:
            raise ValueError("cfm_temperature must be a finite non-negative number")

        reuse_default_emotion = bool(
            reuse_spk_cond_for_emo
            and not emo_audio_prompt
            and emo_vector is None
            and not use_emo_text
        )

        if use_emo_text:
            if self.qwen_emo is None:
                raise RuntimeError(
                    "use_emo_text=True requires QwenEmotion, but it was not loaded at init "
                    "(use_qwen_emo=False). Re-construct IndexTTS2 with use_qwen_emo=True."
                )
            with self.residency.use("qwen_emo"):
                emo_dict = self.qwen_emo.inference(emo_text)
            print(f"detected emotion vectors from text: {emo_dict}")
            emo_vector = list(emo_dict.values())

        if use_emo_text or emo_vector is not None:
            emo_audio_prompt = None

        if emo_vector is not None:
            emo_vector_scale = max(0.0, min(1.0, emo_alpha))
            if emo_vector_scale != 1.0:
                emo_vector = [int(x * emo_vector_scale * 10000) / 10000 for x in emo_vector]
                print(f"scaled emotion vectors to {emo_vector_scale}x: {emo_vector}")

        if emo_audio_prompt is None and not reuse_default_emotion:
            emo_audio_prompt = spk_audio_prompt
            emo_alpha = 1.0

        # Build the speaker reference once. The cache keys include all controls
        # that alter the extracted conditioning so later calls remain correct.
        spk_prompt_key = (spk_audio_prompt, max_speaker_audio_length, semantic_layer)
        if self.cache_spk_cond is None or self.cache_spk_prompt_key != spk_prompt_key:
            extraction_started = time.perf_counter()
            if self.cache_spk_cond is not None:
                self.cache_spk_cond = None
                self.cache_s2mel_style = None
                self.cache_s2mel_prompt = None
                self.cache_mel = None
                torch.cuda.empty_cache()
            audio, sr = self._load_and_cut_audio(spk_audio_prompt, max_speaker_audio_length, verbose)
            audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
            audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)
            inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
            input_features = inputs["input_features"]
            attention_mask = inputs["attention_mask"]
            spk_cond_emb = self.get_emb(input_features, attention_mask, semantic_layer)
            ref_mel = self.mel_fn(audio_22k.to(self.device).float())
            ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)
            feat = self._campplus_fbank(audio_16k)
            feat = feat - feat.mean(dim=0, keepdim=True)
            with self.residency.use("campplus"):
                camp_device = next(self.campplus_model.parameters()).device
                style = self.campplus_model(feat.unsqueeze(0).to(camp_device)).to(self.device)
            with self._use_s2mel():
                prompt_condition = self.s2mel.models['length_regulator'](
                    spk_cond_emb,
                    ylens=ref_target_lengths,
                    n_quantizers=3,
                    f0=None,
                )[0]
            self.cache_spk_cond = spk_cond_emb
            self.cache_s2mel_style = style
            self.cache_s2mel_prompt = prompt_condition
            self.cache_spk_audio_prompt = spk_audio_prompt
            self.cache_spk_prompt_key = spk_prompt_key
            self.cache_mel = ref_mel
            print(f">> Reference extraction (speaker): {time.perf_counter() - extraction_started:.3f}s")
        else:
            style = self.cache_s2mel_style
            prompt_condition = self.cache_s2mel_prompt
            spk_cond_emb = self.cache_spk_cond
            ref_mel = self.cache_mel

        emovec_mat = None
        weight_vector = None
        if emo_vector is not None:
            weight_vector = torch.tensor(emo_vector, device=self.device)
            if use_random:
                random_index = [random.randint(0, count - 1) for count in self.emo_num]
            else:
                random_index = [find_most_similar_cosine(style, matrix) for matrix in self.spk_matrix]
            emo_matrix = [
                matrix[index].unsqueeze(0)
                for index, matrix in zip(random_index, self.emo_matrix)
            ]
            emovec_mat = torch.sum(weight_vector.unsqueeze(1) * torch.cat(emo_matrix, 0), 0).unsqueeze(0)

        if reuse_default_emotion:
            emo_cond_emb = spk_cond_emb
            print(">> Reusing speaker conditioning for the default emotion path.")
        else:
            emo_prompt_key = (emo_audio_prompt, max_emotion_audio_length, semantic_layer)
            if self.cache_emo_cond is None or self.cache_emo_prompt_key != emo_prompt_key:
                extraction_started = time.perf_counter()
                if self.cache_emo_cond is not None:
                    self.cache_emo_cond = None
                    torch.cuda.empty_cache()
                emo_audio, _ = self._load_and_cut_audio(
                    emo_audio_prompt,
                    max_emotion_audio_length,
                    verbose,
                    sr=16000,
                )
                emo_inputs = self.extract_features(emo_audio, sampling_rate=16000, return_tensors="pt")
                emo_cond_emb = self.get_emb(
                    emo_inputs["input_features"],
                    emo_inputs["attention_mask"],
                    semantic_layer,
                )
                self.cache_emo_cond = emo_cond_emb
                self.cache_emo_audio_prompt = emo_audio_prompt
                self.cache_emo_prompt_key = emo_prompt_key
                print(f">> Reference extraction (emotion): {time.perf_counter() - extraction_started:.3f}s")
            else:
                emo_cond_emb = self.cache_emo_cond

        lang_code = str(lang or "EN").upper()
        lang_prefix = f'<|{lang_code.lower()}|> '
        jobs = []
        expected_segments = []
        text_plans = []
        for text_index, unit_text in enumerate(texts):
            segments, plan, _ = self._build_text_plan(
                unit_text,
                lang_code,
                max_text_tokens_per_segment,
                text_normalization,
                interval_silence,
                enable_pause_tags,
                segment_budget_scale_non_cjk,
            )
            text_plans.append(plan)
            expected_segments.append(len(segments))
            for segment_index, segment_text in enumerate(segments):
                token_ids = self.tokenizer.encode(lang_prefix + segment_text, allowed_special='all')
                token_tensor = torch.tensor(token_ids, dtype=torch.long, device=self.device)
                token_tensor = F.pad(token_tensor, (0, 1), value=self.stop_text_token)
                jobs.append((text_index, segment_index, token_tensor))

        if len(jobs) < 2:
            kwargs.update({
                "emo_audio_prompt": emo_audio_prompt,
                "emo_alpha": emo_alpha,
                "emo_vector": emo_vector,
                "use_emo_text": False,
                "use_random": use_random,
                "interval_silence": interval_silence,
                "verbose": verbose,
                "max_text_tokens_per_segment": max_text_tokens_per_segment,
                "duration_factor": duration_factor,
                "text_normalization": text_normalization,
                "max_speaker_audio_length": max_speaker_audio_length,
                "max_emotion_audio_length": max_emotion_audio_length,
                "max_consecutive_silence": max_consecutive_silence,
                "semantic_layer": semantic_layer,
                "cfm_cache_length": cfm_cache_length,
                "diffusion_steps": diffusion_steps,
                "inference_cfg_rate": inference_cfg_rate,
                "reset_beam_cache_per_segment": reset_beam_cache_per_segment,
                "segment_budget_scale_non_cjk": segment_budget_scale_non_cjk,
                "cfm_temperature": cfm_temperature,
                "reuse_spk_cond_for_emo": reuse_spk_cond_for_emo,
                "enable_pause_tags": enable_pause_tags,
                "trim_silence_ms_threshold": trim_silence_ms_threshold,
            })
            return run_sequential()

        do_sample = bool(kwargs.pop("do_sample", True))
        top_p = kwargs.pop("top_p", 0.8)
        top_k = kwargs.pop("top_k", 30)
        temperature = kwargs.pop("temperature", 0.8)
        length_penalty = kwargs.pop("length_penalty", 0.0)
        num_beams = int(kwargs.pop("num_beams", 3))
        repetition_penalty = kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = int(kwargs.pop("max_mel_tokens", 1500))

        generated_parts = [[] for _ in texts]
        results = [None for _ in texts]
        emitted = set()
        sampling_rate = 22050
        total_batches = (len(jobs) + section_batch_size - 1) // section_batch_size
        lang_token = lang_to_token(lang_code)
        generated_audio_seconds = 0.0
        generated_code_tokens = 0
        batch_started_at = request_started_at
        gpt_time = 0.0
        s2mel_time = 0.0
        vocoder_time = 0.0
        shared_cond_lengths = torch.tensor([spk_cond_emb.shape[-1]], device=self.device)
        shared_emo_cond_lengths = torch.tensor([emo_cond_emb.shape[-1]], device=self.device)
        default_emovec = None
        if reuse_default_emotion:
            with torch.no_grad(), torch.amp.autocast(
                torch.device(self.device).type,
                enabled=self.dtype is not None,
                dtype=self.dtype,
            ):
                default_emovec = self.gpt.get_emovec(spk_cond_emb, shared_cond_lengths)
        if self.progress_reporter is not None:
            self.progress_reporter.label = "segments"
            self.progress_reporter.set_stage("synthesis")
            self.progress_reporter.update(0, total=len(jobs), desc="preparing micro-batches")

        def emit_completed_text(text_index):
            if text_index in emitted or len(generated_parts[text_index]) != expected_segments[text_index]:
                return
            ordered_parts = [part for _, part in sorted(generated_parts[text_index])]
            combined = self._assemble_audio_plan(ordered_parts, text_plans[text_index]).to(torch.int16)
            result = (sampling_rate, combined.numpy().T)
            results[text_index] = result
            emitted.add(text_index)
            if on_text_complete is not None:
                on_text_complete(text_index, result)

        for batch_index in range(total_batches):
            batch_jobs = jobs[
                batch_index * section_batch_size:(batch_index + 1) * section_batch_size
            ]
            if console_progress_enabled and self.progress_reporter is None:
                print(
                    f">> {console_progress_label} micro-batch {batch_index + 1}/{total_batches} "
                    f"({len(batch_jobs)} sections)"
                )
            text_tokens = pad_sequence(
                [job[2] for job in batch_jobs],
                batch_first=True,
                padding_value=self.stop_text_token,
            )
            batch_count = text_tokens.size(0)
            langs = torch.full(
                (batch_count,),
                lang_token,
                dtype=torch.long,
                device=self.device,
            )
            cond_lengths = shared_cond_lengths
            emo_cond_lengths = shared_emo_cond_lengths

            gpt_started = time.perf_counter()
            with torch.no_grad():
                with torch.amp.autocast(
                    text_tokens.device.type,
                    enabled=self.dtype is not None,
                    dtype=self.dtype,
                ):
                    emovec = default_emovec
                    if emovec is None:
                        emovec = self.gpt.merge_emovec(
                            spk_cond_emb,
                            emo_cond_emb,
                            cond_lengths,
                            emo_cond_lengths,
                            alpha=emo_alpha,
                        )
                    if emovec_mat is not None:
                        emovec = emovec_mat + (1 - torch.sum(weight_vector)) * emovec

                    generation_options = {
                        "do_sample": do_sample,
                        "num_beams": num_beams,
                        "repetition_penalty": repetition_penalty,
                    }
                    if do_sample:
                        generation_options.update({
                            "top_p": top_p,
                            "temperature": temperature,
                        })
                        if top_k is not None:
                            generation_options["top_k"] = top_k
                    if num_beams > 1:
                        generation_options["length_penalty"] = length_penalty
                    generation_options.update(kwargs)

                    codes, _ = self.gpt.inference_speech(
                        spk_cond_emb,
                        text_tokens,
                        langs,
                        emo_cond_emb,
                        cond_lengths=cond_lengths,
                        emo_cond_lengths=emo_cond_lengths,
                        emo_vec=emovec,
                        campplus_embedding=style,
                        wav=spk_audio_prompt,
                        num_return_sequences=1,
                        max_generate_length=max_mel_tokens,
                        **generation_options,
                    )
                    if reset_beam_cache_per_segment:
                        self._reset_generation_cache()
                gpt_time += time.perf_counter() - gpt_started

                code_lens = []
                for code in codes:
                    stop_positions = (code == self.stop_mel_token).nonzero(as_tuple=False)
                    code_lens.append(
                        stop_positions[0, 0].item() if stop_positions.numel() else code.numel()
                    )
                code_lens = torch.tensor(code_lens, dtype=torch.long, device=self.device)
                generated_code_tokens += int(code_lens.sum().item())
                if max_consecutive_silence > 0:
                    codes, code_lens = self.remove_long_silence(
                        codes,
                        silent_token=52,
                        max_consecutive=max_consecutive_silence,
                    )

                max_code_len = max(1, int(code_lens.max().item()))
                codes = codes[:, :max_code_len].clone()
                for row_index, code_len in enumerate(code_lens.tolist()):
                    if code_len < max_code_len:
                        codes[row_index, code_len:] = 52
                codes[(codes < 0) | (codes >= self.semantic_codec.codebook_size)] = 52

                s2mel_started = time.perf_counter()
                with torch.amp.autocast(
                    text_tokens.device.type,
                    enabled=False,
                    dtype=None,
                ):
                    with self.residency.use("semantic_codec"):
                        semantic = self.semantic_codec.decode(codes)
                    semantic_scale = semantic.size(1) / float(max_code_len)
                    semantic_lengths = torch.clamp(
                        torch.round(code_lens.float() * semantic_scale).long(),
                        min=1,
                    )
                    target_lengths = torch.clamp(
                        torch.round(semantic_lengths.float() * 1.72 * duration_factor).long(),
                        min=1,
                    )
                    with self._use_s2mel():
                        cond = self.s2mel.models['length_regulator'](
                            semantic,
                            ylens=target_lengths,
                            n_quantizers=3,
                            f0=None,
                        )[0]
                        batch_prompt = prompt_condition.expand(batch_count, -1, -1)
                        batch_ref_mel = ref_mel.expand(batch_count, -1, -1)
                        batch_style = style.expand(batch_count, -1)
                        cat_condition = torch.cat([batch_prompt, cond], dim=1)
                        cfm_lengths = target_lengths + batch_prompt.size(1)
                        required_cache_length = max(cfm_cache_length, cat_condition.size(1))
                        cfm_batch_size = batch_count * (2 if inference_cfg_rate > 0 else 1)
                        self._setup_s2mel_caches(cfm_batch_size, required_cache_length)
                        vc_target = self.s2mel.models['cfm'].inference(
                            cat_condition,
                            cfm_lengths,
                            batch_ref_mel,
                            batch_style,
                            None,
                            diffusion_steps,
                            temperature=cfm_temperature,
                            inference_cfg_rate=inference_cfg_rate,
                        )
                        vc_target = vc_target[:, :, ref_mel.size(-1):]
                    s2mel_time += time.perf_counter() - s2mel_started
                    vocoder_started = time.perf_counter()
                    with self.residency.use("bigvgan"):
                        wav_batch = self.bigvgan(vc_target.float()).squeeze(1)
                    vocoder_time += time.perf_counter() - vocoder_started
                    wav_batch = torch.clamp(32767 * wav_batch, -32767.0, 32767.0)

            samples_per_frame = wav_batch.size(-1) / float(vc_target.size(-1))
            completed_texts = set()
            for row_index, (text_index, segment_index, _) in enumerate(batch_jobs):
                sample_count = max(1, int(round(target_lengths[row_index].item() * samples_per_frame)))
                segment_wav = wav_batch[row_index, :sample_count].to(torch.int16).cpu().unsqueeze(0)
                segment_wav = trim_segment_silence(
                    segment_wav,
                    sampling_rate=sampling_rate,
                    minimum_silence_ms=trim_silence_ms_threshold,
                )
                generated_parts[text_index].append((segment_index, segment_wav))
                completed_texts.add(text_index)
                generated_audio_seconds += segment_wav.shape[-1] / float(sampling_rate)
            for text_index in completed_texts:
                emit_completed_text(text_index)
            if self.progress_reporter is not None:
                completed_jobs = min(len(jobs), (batch_index + 1) * section_batch_size)
                self.progress_reporter.update(
                    completed_jobs,
                    total=len(jobs),
                    desc=f"micro-batch {batch_index + 1}/{total_batches}",
                    extra={"audio_seconds": generated_audio_seconds, "speed_unit": "x RT"},
                )

        for text_index in range(len(texts)):
            emit_completed_text(text_index)
        total_audio_seconds = sum(
            result[1].shape[0] / float(result[0])
            for result in results
            if isinstance(result, tuple) and len(result) == 2
        )
        self._record_generation_stats(
            seed=actual_seed,
            segments_count=len(jobs),
            audio_seconds=total_audio_seconds,
            wall_time=time.perf_counter() - batch_started_at,
            gpt_time=gpt_time,
            s2mel_time=s2mel_time,
            vocoder_time=vocoder_time,
            generated_tokens=generated_code_tokens,
        )
        if self.progress_reporter is not None:
            self.progress_reporter.finish()
        return results


    def infer_generator(self, spk_audio_prompt, text, output_path, lang="EN",
              emo_audio_prompt=None, emo_alpha=1.0, emo_vector=None,
              use_emo_text=False, emo_text=None, use_random=False, interval_silence=200,
              verbose=False, max_text_tokens_per_segment=120, stream_return=False, quick_streaming_tokens=0,
              duration_factor=1.0, text_normalization=True,
              max_speaker_audio_length=15, max_emotion_audio_length=15,
              max_consecutive_silence=0, semantic_layer=17, cfm_cache_length=None,
              diffusion_steps=25, inference_cfg_rate=0.7,
              reset_beam_cache_per_segment=False,
              segment_budget_scale_non_cjk=DEFAULT_NON_CJK_BUDGET_SCALE,
              cfm_temperature=1.0, seed=None, reuse_spk_cond_for_emo=False,
              enable_pause_tags=True, trim_silence_ms_threshold=0,
              target_duration_s=None, target_duration_mode="off",
              **generation_kwargs):
        if cfm_cache_length is None:
            cfm_cache_length = self.runtime.cfm_cache_length
        actual_seed = _seed_everything(seed)
        self._reset_peak_vram()
        segment_budget_scale_non_cjk = float(segment_budget_scale_non_cjk)
        if not 0.0 < segment_budget_scale_non_cjk <= 1.0:
            raise ValueError("segment_budget_scale_non_cjk must be in the range (0, 1]")
        cfm_temperature = float(cfm_temperature)
        if not math.isfinite(cfm_temperature) or cfm_temperature < 0:
            raise ValueError("cfm_temperature must be a finite non-negative number")
        target_duration_mode = str(target_duration_mode or "off").strip().lower()
        if target_duration_mode not in _TARGET_DURATION_MODES:
            raise ValueError(f"target_duration_mode must be one of {sorted(_TARGET_DURATION_MODES)}")
        if target_duration_s is not None:
            target_duration_s = float(target_duration_s)
            if not math.isfinite(target_duration_s) or target_duration_s <= 0:
                raise ValueError("target_duration_s must be a finite value greater than zero")
        if target_duration_s is None:
            target_duration_mode = "off"
        reuse_default_emotion = bool(
            reuse_spk_cond_for_emo
            and not emo_audio_prompt
            and emo_vector is None
            and not use_emo_text
        )
        if self.progress_reporter is None:
            print(">> starting inference...")
        self._set_gr_progress(0, "starting inference...")
        if verbose:
            print(f"origin text:{text}, spk_audio_prompt:{spk_audio_prompt}, "
                  f"emo_audio_prompt:{emo_audio_prompt}, emo_alpha:{emo_alpha}, "
                  f"emo_vector:{emo_vector}, use_emo_text:{use_emo_text}, "
                  f"emo_text:{emo_text}")
        start_time = time.perf_counter()

        if use_emo_text or emo_vector is not None:
            # we're using a text or emotion vector guidance; so we must remove
            # "emotion reference voice", to ensure we use correct emotion mixing!
            emo_audio_prompt = None

        if use_emo_text:
            # automatically generate emotion vectors from text prompt
            if self.qwen_emo is None:
                raise RuntimeError(
                    "use_emo_text=True requires QwenEmotion, but it was not loaded at init "
                    "(use_qwen_emo=False). Re-construct IndexTTS2 with use_qwen_emo=True."
                )
            if emo_text is None:
                emo_text = text  # use main text prompt
            with self.residency.use("qwen_emo"):
                emo_dict = self.qwen_emo.inference(emo_text)
            print(f"detected emotion vectors from text: {emo_dict}")
            # convert ordered dict to list of vectors; the order is VERY important!
            emo_vector = list(emo_dict.values())

        if emo_vector is not None:
            # we have emotion vectors; they can't be blended via alpha mixing
            # in the main inference process later, so we must pre-calculate
            # their new strengths here based on the alpha instead!
            emo_vector_scale = max(0.0, min(1.0, emo_alpha))
            if emo_vector_scale != 1.0:
                # scale each vector and truncate to 4 decimals (for nicer printing)
                emo_vector = [int(x * emo_vector_scale * 10000) / 10000 for x in emo_vector]
                print(f"scaled emotion vectors to {emo_vector_scale}x: {emo_vector}")

        if emo_audio_prompt is None and not reuse_default_emotion:
            # we are not using any external "emotion reference voice"; use
            # speaker's voice as the main emotion reference audio.
            emo_audio_prompt = spk_audio_prompt
            # must always use alpha=1.0 when we don't have an external reference voice
            emo_alpha = 1.0

        # 如果参考音频改变了，才需要重新生成, 提升速度
        spk_prompt_key = (spk_audio_prompt, float(max_speaker_audio_length), int(semantic_layer))
        if self.cache_spk_cond is None or self.cache_spk_prompt_key != spk_prompt_key:
            extraction_started = time.perf_counter()
            if self.cache_spk_cond is not None:
                self.cache_spk_cond = None
                self.cache_s2mel_style = None
                self.cache_s2mel_prompt = None
                self.cache_mel = None
                torch.cuda.empty_cache()
            audio, sr = self._load_and_cut_audio(spk_audio_prompt, max_speaker_audio_length, verbose)
            audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
            audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)

            inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
            input_features = inputs["input_features"]
            attention_mask = inputs["attention_mask"]
            spk_cond_emb = self.get_emb(input_features, attention_mask, semantic_layer)

            # _, S_ref = self.semantic_codec.quantize(spk_cond_emb)
            S_ref = spk_cond_emb
            ref_mel = self.mel_fn(audio_22k.to(self.device).float())
            ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)

            feat = self._campplus_fbank(audio_16k)
            feat = feat - feat.mean(dim=0, keepdim=True)  # feat2另外一个滤波器能量组特征[922, 80]
            with self.residency.use("campplus"):
                camp_device = next(self.campplus_model.parameters()).device
                style = self.campplus_model(feat.unsqueeze(0).to(camp_device)).to(self.device)

            with self._use_s2mel():
                prompt_condition = self.s2mel.models['length_regulator'](
                    spk_cond_emb,
                    ylens=ref_target_lengths,
                    n_quantizers=3,
                    f0=None)[0]

            self.cache_spk_cond = spk_cond_emb
            self.cache_s2mel_style = style
            self.cache_s2mel_prompt = prompt_condition
            self.cache_spk_audio_prompt = spk_audio_prompt
            self.cache_spk_prompt_key = spk_prompt_key
            self.cache_mel = ref_mel
            print(f">> Reference extraction (speaker): {time.perf_counter() - extraction_started:.3f}s")
        else:
            style = self.cache_s2mel_style
            prompt_condition = self.cache_s2mel_prompt
            spk_cond_emb = self.cache_spk_cond
            ref_mel = self.cache_mel

        if emo_vector is not None:
            weight_vector = torch.tensor(emo_vector, device=self.device)
            if use_random:
                random_index = [random.randint(0, x - 1) for x in self.emo_num]
            else:
                random_index = [find_most_similar_cosine(style, tmp) for tmp in self.spk_matrix]

            emo_matrix = [tmp[index].unsqueeze(0) for index, tmp in zip(random_index, self.emo_matrix)]
            emo_matrix = torch.cat(emo_matrix, 0)
            emovec_mat = weight_vector.unsqueeze(1) * emo_matrix
            emovec_mat = torch.sum(emovec_mat, 0)
            emovec_mat = emovec_mat.unsqueeze(0)

        if reuse_default_emotion:
            emo_cond_emb = spk_cond_emb
            print(">> Reusing speaker conditioning for the default emotion path.")
        else:
            emo_prompt_key = (emo_audio_prompt, float(max_emotion_audio_length), int(semantic_layer))
            if self.cache_emo_cond is None or self.cache_emo_prompt_key != emo_prompt_key:
                extraction_started = time.perf_counter()
                if self.cache_emo_cond is not None:
                    self.cache_emo_cond = None
                    torch.cuda.empty_cache()
                emo_audio, _ = self._load_and_cut_audio(
                    emo_audio_prompt,
                    max_emotion_audio_length,
                    verbose,
                    sr=16000,
                )
                emo_inputs = self.extract_features(emo_audio, sampling_rate=16000, return_tensors="pt")
                emo_cond_emb = self.get_emb(
                    emo_inputs["input_features"],
                    emo_inputs["attention_mask"],
                    semantic_layer,
                )
                self.cache_emo_cond = emo_cond_emb
                self.cache_emo_audio_prompt = emo_audio_prompt
                self.cache_emo_prompt_key = emo_prompt_key
                print(f">> Reference extraction (emotion): {time.perf_counter() - extraction_started:.3f}s")
            else:
                emo_cond_emb = self.cache_emo_cond

        self._set_gr_progress(0.1, "text processing...")
        segments, audio_plan, lang_prefix = self._build_text_plan(
            text,
            lang,
            max_text_tokens_per_segment,
            text_normalization,
            interval_silence,
            enable_pause_tags,
            segment_budget_scale_non_cjk,
        )
        segments_count = len(segments)
        if self.progress_reporter is not None:
            self.progress_reporter.label = "segments"
            self.progress_reporter.set_stage("synthesis")
            self.progress_reporter.update(0, total=segments_count, desc="text processing")
        segment_tokens = []
        for seg_text in segments:
            toks = self.tokenizer.encode(lang_prefix + seg_text, allowed_special='all')
            toks = torch.IntTensor(toks).unsqueeze(0).to(self.device)
            segment_tokens.append(F.pad(toks, (0, 1), value=1))
        lang = torch.LongTensor([lang_to_token(lang)]).to(self.device)
        if verbose:
            print("segments count:", segments_count)
            print("max_text_tokens_per_segment:", max_text_tokens_per_segment)
            print(*segments, sep="\n")
        do_sample = generation_kwargs.pop("do_sample", True)
        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 0.8)
        autoregressive_batch_size = 1
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 1500)
        hf_generation_kwargs = dict(generation_kwargs)
        hf_generation_kwargs.update({
            "do_sample": do_sample,
            "num_beams": num_beams,
            "repetition_penalty": repetition_penalty,
        })
        if do_sample:
            hf_generation_kwargs.update({
                "top_p": top_p,
                "temperature": temperature,
            })
            if top_k is not None:
                hf_generation_kwargs["top_k"] = top_k
        if num_beams > 1:
            hf_generation_kwargs["length_penalty"] = length_penalty
        sampling_rate = 22050

        wavs = []
        rendered_codes = []
        gpt_gen_time = 0
        s2mel_time = 0
        bigvgan_time = 0
        has_warned = False
        generated_code_tokens = 0
        stream_plan_cursor = 0
        cond_lengths = torch.tensor([spk_cond_emb.shape[-1]], device=self.device)
        emo_cond_lengths = torch.tensor([emo_cond_emb.shape[-1]], device=self.device)
        default_emovec = None
        if reuse_default_emotion:
            with torch.no_grad(), torch.amp.autocast(
                torch.device(self.device).type,
                enabled=self.dtype is not None,
                dtype=self.dtype,
            ):
                default_emovec = self.gpt.get_emovec(spk_cond_emb, cond_lengths)
        for seg_idx, text_tokens in enumerate(segment_tokens):
            self._set_gr_progress(0.2 + 0.7 * seg_idx / segments_count,
                                  f"speech synthesis {seg_idx + 1}/{segments_count}...")
            if verbose:
                print(text_tokens)
                print(f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}")

            m_start_time = time.perf_counter()
            with torch.no_grad():
                with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    emovec = default_emovec
                    if emovec is None:
                        emovec = self.gpt.merge_emovec(
                            spk_cond_emb,
                            emo_cond_emb,
                            cond_lengths,
                            emo_cond_lengths,
                            alpha=emo_alpha,
                        )

                    if emo_vector is not None:
                        emovec = emovec_mat + (1 - torch.sum(weight_vector)) * emovec
                        # emovec = emovec_mat

                    codes, speech_conditioning_latent = self.gpt.inference_speech(
                        spk_cond_emb,
                        text_tokens,
                        lang,
                        emo_cond_emb,
                        cond_lengths=cond_lengths,
                        emo_cond_lengths=emo_cond_lengths,
                        emo_vec=emovec,
                        campplus_embedding=style,
                        wav=spk_audio_prompt,
                        num_return_sequences=autoregressive_batch_size,
                        max_generate_length=max_mel_tokens,
                        **hf_generation_kwargs
                    )
                    if reset_beam_cache_per_segment:
                        self._reset_generation_cache()

                gpt_gen_time += time.perf_counter() - m_start_time
                if not has_warned and (codes[:, -1] != self.stop_mel_token).any():
                    warnings.warn(
                        f"WARN: generation stopped due to exceeding `max_mel_tokens` ({max_mel_tokens}). "
                        f"Input text tokens: {text_tokens.shape[1]}. "
                        f"Consider reducing `max_text_tokens_per_segment`({max_text_tokens_per_segment}) or increasing `max_mel_tokens`.",
                        category=RuntimeWarning
                    )
                    has_warned = True

                code_lens = torch.tensor([codes.shape[-1]], device=codes.device, dtype=codes.dtype)
                #                 if verbose:
                #                     print(codes, type(codes))
                #                     print(f"codes shape: {codes.shape}, codes type: {codes.dtype}")
                #                     print(f"code len: {code_lens}")

                code_lens = []
                max_code_len = 0
                for code in codes:
                    if self.stop_mel_token not in code:
                        code_len = len(code)
                    else:
                        len_ = (code == self.stop_mel_token).nonzero(as_tuple=False)[0]
                        code_len = len_[0].item() if len_.numel() > 0 else len(code)
                    code_lens.append(code_len)
                    max_code_len = max(max_code_len, code_len)
                codes = codes[:, :max_code_len]
                code_lens = torch.LongTensor(code_lens)
                generated_code_tokens += int(code_lens.sum().item())
                code_lens = code_lens.to(self.device)
                if max_consecutive_silence > 0:
                    codes, code_lens = self.remove_long_silence(
                        codes,
                        silent_token=52,
                        max_consecutive=int(max_consecutive_silence),
                    )
                if verbose:
                    print(codes, type(codes))
                    print(f"fix codes shape: {codes.shape}, codes type: {codes.dtype}")
                    print(f"code len: {code_lens}")

                rendered_codes.append(codes.detach().cpu())
                wav, segment_s2mel_time, segment_vocoder_time = self._render_codes_segment(
                    codes,
                    prompt_condition,
                    ref_mel,
                    style,
                    duration_factor=duration_factor,
                    cfm_cache_length=cfm_cache_length,
                    diffusion_steps=diffusion_steps,
                    inference_cfg_rate=inference_cfg_rate,
                    cfm_temperature=cfm_temperature,
                    trim_silence_ms_threshold=trim_silence_ms_threshold,
                )
                s2mel_time += segment_s2mel_time
                bigvgan_time += segment_vocoder_time
                if verbose:
                    print(f"wav shape: {wav.shape}", "min:", wav.min(), "max:", wav.max())
                wavs.append(wav)
                if self.progress_reporter is not None:
                    generated_seconds = sum(item.shape[-1] for item in wavs) / float(sampling_rate)
                    self.progress_reporter.update(
                        seg_idx + 1,
                        total=segments_count,
                        desc=f"speech synthesis {seg_idx + 1}/{segments_count}",
                        extra={"audio_seconds": generated_seconds, "speed_unit": "x RT"},
                    )
                if stream_return and target_duration_mode == "off":
                    while stream_plan_cursor < len(audio_plan):
                        kind, value = audio_plan[stream_plan_cursor]
                        if kind == "segment" and value > seg_idx:
                            break
                        if kind == "segment":
                            yield wavs[value].cpu()
                        else:
                            yield torch.zeros(1, int(value), dtype=wav.dtype)
                        stream_plan_cursor += 1
        natural_duration_factor = float(duration_factor)
        if target_duration_mode == "natural" and wavs:
            fixed_samples = sum(value for kind, value in audio_plan if kind == "silence")
            natural_speech_samples = sum(item.shape[-1] for item in wavs)
            desired_speech_samples = max(1, int(round(target_duration_s * sampling_rate)) - fixed_samples)
            ratio = desired_speech_samples / max(1, natural_speech_samples)
            adjusted_factor = min(4.0, max(0.25, natural_duration_factor * ratio))
            self._progress_log(
                f">> Target duration natural pass: {natural_speech_samples / sampling_rate:.3f}s speech -> "
                f"{desired_speech_samples / sampling_rate:.3f}s, duration_factor={adjusted_factor:.4f}"
            )
            adjusted_wavs = []
            for codes in rendered_codes:
                adjusted_wav, segment_s2mel_time, segment_vocoder_time = self._render_codes_segment(
                    codes,
                    prompt_condition,
                    ref_mel,
                    style,
                    duration_factor=adjusted_factor,
                    cfm_cache_length=cfm_cache_length,
                    diffusion_steps=diffusion_steps,
                    inference_cfg_rate=inference_cfg_rate,
                    cfm_temperature=cfm_temperature,
                    trim_silence_ms_threshold=trim_silence_ms_threshold,
                )
                adjusted_wavs.append(adjusted_wav)
                s2mel_time += segment_s2mel_time
                bigvgan_time += segment_vocoder_time
            wavs = adjusted_wavs
            natural_duration_factor = adjusted_factor

        self._set_gr_progress(0.9, "saving audio...")
        wav = self._assemble_audio_plan(wavs, audio_plan, sampling_rate)
        if target_duration_mode in {"pad", "trim"}:
            wav = self._fit_target_samples(
                wav,
                int(round(target_duration_s * sampling_rate)),
                target_duration_mode,
            )
        if stream_return:
            if target_duration_mode == "off":
                for kind, value in audio_plan[stream_plan_cursor:]:
                    if kind == "segment":
                        yield wavs[value].cpu()
                    else:
                        yield torch.zeros(1, int(value), dtype=wav.dtype)
            elif target_duration_mode == "natural":
                for kind, value in audio_plan:
                    if kind == "segment":
                        yield wavs[value].cpu()
                    else:
                        yield torch.zeros(1, int(value), dtype=wav.dtype)
            else:
                yield wav.cpu()
        end_time = time.perf_counter()

        wav_length = wav.shape[-1] / sampling_rate
        self._record_generation_stats(
            seed=actual_seed,
            segments_count=segments_count,
            audio_seconds=wav_length,
            wall_time=end_time - start_time,
            gpt_time=gpt_gen_time,
            s2mel_time=s2mel_time,
            vocoder_time=bigvgan_time,
            generated_tokens=generated_code_tokens,
            target_duration_mode=target_duration_mode,
            target_duration_s=target_duration_s,
            duration_factor_used=natural_duration_factor,
        )
        if self.progress_reporter is not None:
            self.progress_reporter.finish()

        # save audio
        wav = wav.cpu()  # to cpu
        if output_path:
            # 直接保存音频到指定路径中
            if os.path.isfile(output_path):
                os.remove(output_path)
                print(">> remove old wav file:", output_path)
            if os.path.dirname(output_path) != "":
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

            save_pcm_wav(output_path, wav, sampling_rate)
            print(">> wav file saved to:", output_path)
            if stream_return:
                return None
            yield output_path
        else:
            if stream_return:
                return None
            # 返回以符合Gradio的格式要求
            wav_data = wav.type(torch.int16)
            wav_data = wav_data.numpy().T
            yield (sampling_rate, wav_data)


def find_most_similar_cosine(query_vector, matrix):
    query_vector = query_vector.float()
    matrix = matrix.float()

    similarities = F.cosine_similarity(query_vector, matrix, dim=1)
    most_similar_index = torch.argmax(similarities)
    return most_similar_index

class QwenEmotion:
    def __init__(self, model_dir, dtype=torch.bfloat16):
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            dtype=dtype,
            device_map=None,
        )
        self.prompt = "文本情感分类"
        self.cn_key_to_en = {
            "高兴": "happy",
            "愤怒": "angry",
            "悲伤": "sad",
            "恐惧": "afraid",
            "反感": "disgusted",
            # TODO: the "低落" (melancholic) emotion will always be mapped to
            # "悲伤" (sad) by QwenEmotion's text analysis. it doesn't know the
            # difference between those emotions even if user writes exact words.
            # SEE: `self.melancholic_words` for current workaround.
            "低落": "melancholic",
            "惊讶": "surprised",
            "自然": "calm",
        }
        self.desired_vector_order = ["高兴", "愤怒", "悲伤", "恐惧", "反感", "低落", "惊讶", "自然"]
        self.melancholic_words = {
            # emotion text phrases that will force QwenEmotion's "悲伤" (sad) detection
            # to become "低落" (melancholic) instead, to fix limitations mentioned above.
            "低落",
            "melancholy",
            "melancholic",
            "depression",
            "depressed",
            "gloomy",
        }
        self.max_score = 1.2
        self.min_score = 0.0

    def clamp_score(self, value):
        return max(self.min_score, min(self.max_score, value))

    def normalize_content(self, content):
        if isinstance(content, dict):
            normalized = dict(content)
        else:
            normalized = {}

        def label_to_cn_key(value):
            if not isinstance(value, str):
                return None

            value = value.strip()
            if value in self.cn_key_to_en:
                return value

            value_lower = value.lower()
            for cn_key, en_key in self.cn_key_to_en.items():
                if value_lower == en_key:
                    return cn_key
            return None

        detected_key = label_to_cn_key(content) if isinstance(content, str) else None
        if detected_key is None:
            for alias in ("emotion", "emotion_label", "label", "情感", "情绪"):
                detected_key = label_to_cn_key(normalized.get(alias))
                if detected_key is not None:
                    break
        if detected_key is not None and all(key not in normalized for key in self.desired_vector_order):
            normalized[detected_key] = 1.0

        for cn_key in self.desired_vector_order:
            detected_key = label_to_cn_key(normalized.get(cn_key))
            if detected_key is not None:
                normalized[cn_key] = 1.0 if detected_key == cn_key else 0.0
                if detected_key != cn_key:
                    normalized[detected_key] = 1.0

        return normalized

    def convert(self, content):
        # generate emotion vector dictionary:
        # - insert values in desired order (Python 3.7+ `dict` remembers insertion order)
        # - convert Chinese keys to English
        # - clamp all values to the allowed min/max range
        # - use 0.0 for any values that were missing in `content`
        content = self.normalize_content(content)
        emotion_dict = {
            self.cn_key_to_en[cn_key]: self.clamp_score(content.get(cn_key, 0.0))
            for cn_key in self.desired_vector_order
        }

        # default to a calm/neutral voice if all emotion vectors were empty
        if all(val <= 0.0 for val in emotion_dict.values()):
            print(">> no emotions detected; using default calm/neutral voice")
            emotion_dict["calm"] = 1.0

        return emotion_dict

    def inference(self, text_input):
        start = time.time()
        messages = [
            {"role": "system", "content": f"{self.prompt}"},
            {"role": "user", "content": f"{text_input}"}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=32768,
            pad_token_id=self.tokenizer.eos_token_id
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True)

        # decode the JSON emotion detections as a dictionary
        try:
            content = json.loads(content)
        except json.decoder.JSONDecodeError:
            # invalid JSON; fallback to manual string parsing
            # print(">> parsing QwenEmotion response", content)
            content = {
                m.group(1): float(m.group(2))
                for m in re.finditer(r'([^\s":.,]+?)"?\s*:\s*([\d.]+)', content)
            }
            # print(">> dict result", content)

        # workaround for QwenEmotion's inability to distinguish "悲伤" (sad) vs "低落" (melancholic).
        # if we detect any of the IndexTTS "melancholic" words, we swap those vectors
        # to encode the "sad" emotion as "melancholic" (instead of sadness).
        text_input_lower = text_input.lower()
        if any(word in text_input_lower for word in self.melancholic_words):
            # print(">> before vec swap", content)
            content["悲伤"], content["低落"] = content.get("低落", 0.0), content.get("悲伤", 0.0)
            # print(">>  after vec swap", content)

        return self.convert(content)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default="models/config.yaml")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--prompt_wav", type=str, default="examples/voice_01.wav")
    parser.add_argument("--text", type=str, default="欢迎大家来体验indextts2，并给予我们意见与反馈，谢谢大家。")
    parser.add_argument("--lang", type=str, default="ZH")
    parser.add_argument("--output", type=str, default="gen.wav")
    parser.add_argument("--text_normalization", action="store_true", default=True)
    args = parser.parse_args()

    tts = IndexTTS2(
        cfg_path=args.cfg_path,
        model_dir=args.model_dir,
        use_bf16=True,
        use_cuda_kernel=False,
        use_torch_compile=False,
        use_qwen_emo=True,
    )

    tts.infer(spk_audio_prompt=args.prompt_wav, text=args.text, lang=args.lang, output_path=args.output, verbose=True, text_normalization=args.text_normalization)
