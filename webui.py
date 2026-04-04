import html
import json
import os
import sys
import threading
import time
import gc
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import glob
from pathlib import Path
import platform
import signal
import subprocess
import tempfile
import shutil
from typing import Any, Dict, List, Optional

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# pandas removed - not needed, using native list format instead

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse
parser = argparse.ArgumentParser(
    description="IndexTTS WebUI",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the web UI on")
parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 for inference if available")
parser.add_argument("--deepspeed", action="store_true", default=False, help="Use DeepSpeed to accelerate if available")
parser.add_argument("--cuda_kernel", action="store_true", default=False, help="Use CUDA kernel for inference if available")
parser.add_argument("--gui_seg_tokens", type=int, default=80, help="GUI: Max tokens per generation segment")
parser.add_argument("--share", action="store_true", default=False, help="Enable Gradio live sharing to create a public link")
cmd_args = parser.parse_args()

if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist. Please download the model first.")
    sys.exit(1)

for file in [
    "bpe.model",
    "gpt.pth",
    "config.yaml",
    "s2mel.pth",
    "wav2vec2bert_stats.pt"
]:
    file_path = os.path.join(cmd_args.model_dir, file)
    if not os.path.exists(file_path):
        print(f"Required file {file_path} does not exist. Please download it.")
        sys.exit(1)

import gradio as gr
from omegaconf import OmegaConf
from indextts.utils.front import TextNormalizer, TextTokenizer
from indextts.utils.subtitle_utils import (
    SUPPORTED_SUBTITLE_EXTENSIONS,
    assemble_subtitle_audio,
    build_subtitle_render_units,
    ensure_audio_matrix,
    fit_audio_to_duration,
    format_srt_timestamp,
    get_subtitle_extension,
    get_subtitle_format_label,
    parse_subtitle_file,
    read_pcm16_wav,
    retime_audio_file_with_ffmpeg,
    subtitle_cues_to_text,
    write_pcm16_wav,
)
from indextts.utils.task_output_utils import (
    build_segment_output_path,
    create_task_output_layout,
    write_metadata_file,
)
from tools.i18n.i18n import I18nAuto
from webui_generation_runner import create_tts as create_generation_tts, run_generation_request

i18n = I18nAuto(language="Auto")
MODE = 'local'


class LazyTTSProxy:
    def __init__(self, factory):
        self._factory = factory
        self._instance = None
        self._lock = threading.Lock()

    def get_instance(self):
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = self._factory()
        return self._instance

    def release_instance(self):
        with self._lock:
            instance = self._instance
            self._instance = None
        return instance

    def is_loaded(self):
        return self._instance is not None

    def __getattr__(self, item):
        return getattr(self.get_instance(), item)

    def __setattr__(self, key, value):
        if key in {"_factory", "_instance", "_lock"}:
            object.__setattr__(self, key, value)
            return
        setattr(self.get_instance(), key, value)


def _build_tts_runtime_options():
    return {
        "model_dir": cmd_args.model_dir,
        "cfg_path": os.path.join(cmd_args.model_dir, "config.yaml"),
        "use_fp16": bool(cmd_args.fp16),
        "use_deepspeed": bool(cmd_args.deepspeed),
        "use_cuda_kernel": bool(cmd_args.cuda_kernel),
    }


PREVIEW_CFG = OmegaConf.load(os.path.join(cmd_args.model_dir, "config.yaml"))
PREVIEW_MAX_TEXT_TOKENS = int(PREVIEW_CFG.gpt.max_text_tokens)
MODEL_VERSION = str(getattr(PREVIEW_CFG, "version", "1.0"))
PREVIEW_BPE_PATH = os.path.join(cmd_args.model_dir, PREVIEW_CFG.dataset["bpe_model"])
PREVIEW_TEXT_NORMALIZER = TextNormalizer()
PREVIEW_TEXT_NORMALIZER.load()
PREVIEW_TEXT_TOKENIZER = TextTokenizer(PREVIEW_BPE_PATH, PREVIEW_TEXT_NORMALIZER)


def _create_inprocess_tts():
    return create_generation_tts(_build_tts_runtime_options())


tts = LazyTTSProxy(_create_inprocess_tts)


def unload_inprocess_tts():
    instance = tts.release_instance()
    if instance is None:
        return False

    for attr in (
        "qwen_emo",
        "gpt",
        "semantic_model",
        "semantic_codec",
        "s2mel",
        "campplus_model",
        "bigvgan",
        "emo_matrix",
        "spk_matrix",
        "mel_fn",
        "extract_features",
        "semantic_mean",
        "semantic_std",
        "cache_spk_cond",
        "cache_s2mel_style",
        "cache_s2mel_prompt",
        "cache_spk_audio_prompt",
        "cache_spk_prompt_key",
        "cache_emo_cond",
        "cache_emo_audio_prompt",
        "cache_emo_prompt_key",
        "cache_mel",
        "gr_progress",
    ):
        if hasattr(instance, attr):
            setattr(instance, attr, None)

    gc.collect()
    torch_module = sys.modules.get("torch")
    if torch_module is not None:
        try:
            if torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()
                if hasattr(torch_module.cuda, "ipc_collect"):
                    torch_module.cuda.ipc_collect()
            elif hasattr(torch_module, "mps") and torch_module.backends.mps.is_available():
                torch_module.mps.empty_cache()
        except Exception:
            pass
    return True
# 支持的语言列表
LANGUAGES = {
    "中文": "zh_CN",
    "English": "en_US"
}
EMO_CHOICES_ALL = ["Same as speaker voice",
                "Use emotion reference audio",
                "Use emotion vector control",
                "Use emotion text description"]

os.makedirs("outputs/tasks",exist_ok=True)
os.makedirs("prompts",exist_ok=True)
os.makedirs("outputs/used_audios",exist_ok=True)
PRESETS_DIR = os.path.join(current_dir, "presets")
os.makedirs(PRESETS_DIR, exist_ok=True)

UI_PRESET_VERSION = "1.0"
UI_PRESET_FORMAT = "indextts2_premium_ui"
DEFAULT_UI_PRESET_NAME = "default"
_LAST_USED_UI_PRESET_FILE = ".last_used_ui_preset.txt"

MAX_LENGTH_TO_USE_SPEED = 70
APP_TITLE = "Index TTS2 Premium SECourses App"
APP_ASSETS_DIR = os.path.join(current_dir, "ui_assets")
APP_FAVICON_PATH = os.path.join(APP_ASSETS_DIR, "indextts_premium_favicon.svg")
SUBTITLE_TIMING_INTERVAL_SILENCE_MS = 0
APP_HEAD = """
<meta name="theme-color" content="#a11236">
<script>
(() => {
  let sectionCountTimer = null;

  function scheduleSectionCountRefresh() {
    const signal = document.querySelector("#section-count-refresh-signal textarea, #section-count-refresh-signal input");
    if (!signal) {
      return;
    }
    if (sectionCountTimer) {
      clearTimeout(sectionCountTimer);
    }
    sectionCountTimer = setTimeout(() => {
      signal.value = String(Date.now());
      signal.dispatchEvent(new Event("input", { bubbles: true }));
      signal.dispatchEvent(new Event("change", { bubbles: true }));
    }, 500);
  }

  document.addEventListener("input", (event) => {
    const target = event.target;
    if (!target) {
      return;
    }
    if (target.closest("#input-text-source") || target.closest("#max-tokens-segment-source")) {
      scheduleSectionCountRefresh();
    }
  }, true);
})();
</script>
"""
MEDIA_FILE_TYPES = [
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv",
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".wma", ".aac", ".opus",
]
CAPTION_TIMING_HELP = """
**What cue timing does**

The app generates separate caption timing units, then auto-retimes each finished unit to the matching caption duration before timeline assembly. In most files, each caption block becomes its own unit. If cues overlap, overlapping cues are merged into a larger timing unit first.

**Example**

`00:00:01.000 --> 00:00:03.000   Hello there.`

`00:00:04.500 --> 00:00:06.000   Welcome back.`

With cue timing **off**, the app treats the text like normal paragraphs and decides pacing on its own.

With cue timing **on**, `Hello there.` is generated as its own timing unit, retimed to fit the `2.0s` cue slot, the `1.5s` gap is preserved, and `Welcome back.` starts at `4.5s`.

**Impact on the result**

This is useful for subtitle-aligned narration, dubbing, and scene-matched timing. Because each finished timing unit is retimed to its target slot before assembly, subtitle timing stays aligned much more reliably than the old cue-fitting approach. If cues overlap, they are synthesized as merged timing units so the final timeline still matches the caption file structure.
"""
APP_CSS = """
.top-input-panel {
    border: 0 !important;
    border-radius: 0;
    padding: 0;
    background: transparent !important;
    box-shadow: none !important;
}

.ui-hidden-signal {
    display: none !important;
}

.top-input-panel > div {
    border: 1px solid var(--block-border-color, rgba(255, 255, 255, 0.08)) !important;
    border-radius: var(--radius-lg, 18px) !important;
    padding: 0.95rem !important;
    background: var(--block-background-fill, transparent) !important;
    box-shadow: none !important;
}

.top-input-panel h3,
.top-input-panel .prose h3 {
    margin-top: 0 !important;
    margin-bottom: 0.8rem !important;
    padding-bottom: 0.7rem;
    border-bottom: 1px solid var(--block-border-color, rgba(255, 255, 255, 0.08));
    color: var(--body-text-color, inherit) !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
}

.caption-timing-help > div,
.caption-timing-help .prose {
    padding: 0.45rem 0 0.3rem !important;
}

.reference-subsection-title > div,
.reference-subsection-title .prose {
    padding: 0.1rem 0 0.35rem !important;
}

.reference-subsection h4,
.reference-subsection-title h4,
.reference-subsection-title .prose h4 {
    color: var(--body-text-color, inherit) !important;
    font-weight: 700 !important;
    margin: 0 !important;
}

.reference-subsection {
    margin: 0 0 0.9rem;
}

.reference-subsection > div {
    border: 1px solid var(--block-border-color, rgba(255, 255, 255, 0.08)) !important;
    border-radius: calc(var(--radius-lg, 18px) - 4px) !important;
    padding: 0.85rem !important;
    background: var(--background-fill-secondary, transparent) !important;
    box-shadow: none !important;
}

.top-section-flat {
    border-width: 0 !important;
    border-style: none !important;
    box-shadow: none !important;
    background: transparent !important;
}

.top-section-flat > .wrap,
.top-section-flat > .block,
.top-section-flat .block {
    border-width: 0 !important;
    border-style: none !important;
    box-shadow: none !important;
    background: transparent !important;
}

:is(button.action-button, .action-button button) {
    position: relative;
    overflow: hidden;
    min-height: 48px;
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    color: #fdf8ff !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em;
    text-shadow: 0 1px 0 rgba(15, 23, 42, 0.28);
    transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease !important;
}

:is(button.action-button, .action-button button)::before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(120deg, transparent 18%, rgba(255, 255, 255, 0.22) 38%, transparent 56%);
    transform: translateX(-160%);
    transition: transform 0.55s ease;
    pointer-events: none;
}

:is(button.action-button, .action-button button):hover {
    transform: translateY(-1px);
    filter: saturate(1.06) brightness(1.03);
}

:is(button.action-button, .action-button button):hover::before {
    transform: translateX(160%);
}

:is(button.action-button, .action-button button):active {
    transform: translateY(1px);
}

:is(button.action-button, .action-button button):focus-visible {
    outline: 2px solid rgba(255, 255, 255, 0.82);
    outline-offset: 2px;
}

:is(button#extract-audio-button, #extract-audio-button button) {
    background: linear-gradient(180deg, #ffd4a4 0%, #ffb05f 18%, #e67c26 58%, #9a4a0d 100%) !important;
    border-color: rgba(160, 77, 14, 0.65) !important;
    box-shadow:
        0 14px 28px rgba(230, 124, 38, 0.28),
        0 1px 0 rgba(255, 247, 234, 0.34) inset,
        0 -3px 0 rgba(113, 50, 5, 0.28) inset !important;
}

:is(button#load-audio-button, #load-audio-button button) {
    background: linear-gradient(180deg, #9df2db 0%, #47d6ab 16%, #0f9f7f 56%, #0a5f4e 100%) !important;
    border-color: rgba(8, 100, 82, 0.65) !important;
    box-shadow:
        0 14px 28px rgba(15, 159, 127, 0.25),
        0 1px 0 rgba(229, 255, 248, 0.34) inset,
        0 -3px 0 rgba(5, 73, 60, 0.28) inset !important;
}

:is(button#generate-speech-button, #generate-speech-button button) {
    min-height: 54px;
    letter-spacing: 0.03em;
    color: #ffe7eb !important;
    text-shadow:
        0 0 8px rgba(255, 222, 228, 0.65),
        0 0 18px rgba(255, 131, 157, 0.48),
        0 1px 0 rgba(107, 13, 35, 0.9);
    background:
        radial-gradient(circle at 18% 0%, rgba(255, 197, 210, 0.36), transparent 34%),
        linear-gradient(180deg, #ff9aae 0%, #ff6383 16%, #d91f4d 55%, #7f102c 100%) !important;
    border-color: rgba(135, 17, 48, 0.72) !important;
    box-shadow:
        0 0 0 1px rgba(255, 180, 196, 0.12),
        0 16px 34px rgba(217, 31, 77, 0.34),
        0 0 26px rgba(255, 77, 116, 0.26),
        0 1px 0 rgba(255, 234, 239, 0.34) inset,
        0 -3px 0 rgba(95, 9, 31, 0.34) inset !important;
    animation: premium-button-glow 2.8s ease-in-out infinite;
}

:is(button#generate-speech-button, #generate-speech-button button)::before {
    background: linear-gradient(120deg, transparent 15%, rgba(255, 255, 255, 0.28) 36%, transparent 58%);
    animation: premium-button-sheen 3.6s ease-in-out infinite;
}

:is(button#generate-speech-button, #generate-speech-button button):hover {
    box-shadow:
        0 0 0 1px rgba(255, 188, 202, 0.18),
        0 20px 38px rgba(217, 31, 77, 0.42),
        0 0 34px rgba(255, 77, 116, 0.34),
        0 1px 0 rgba(255, 238, 242, 0.38) inset,
        0 -3px 0 rgba(95, 9, 31, 0.38) inset !important;
}

:is(button#generate-speech-button, #generate-speech-button button):active {
    animation-play-state: paused;
}

:is(button#open-outputs-button, #open-outputs-button button) {
    background: linear-gradient(180deg, #b5e8ff 0%, #69c8ff 18%, #238cd8 58%, #12518d 100%) !important;
    border-color: rgba(19, 85, 145, 0.68) !important;
    box-shadow:
        0 14px 28px rgba(35, 140, 216, 0.26),
        0 1px 0 rgba(234, 248, 255, 0.34) inset,
        0 -3px 0 rgba(14, 60, 106, 0.28) inset !important;
}

:is(button#preset-save-button, #preset-save-button button) {
    background: linear-gradient(180deg, #d6bcff 0%, #b084ff 18%, #7a41d8 58%, #4c1f96 100%) !important;
    border-color: rgba(80, 31, 151, 0.68) !important;
    box-shadow:
        0 14px 28px rgba(122, 65, 216, 0.27),
        0 1px 0 rgba(246, 239, 255, 0.34) inset,
        0 -3px 0 rgba(60, 20, 118, 0.28) inset !important;
}

:is(button#preset-load-button, #preset-load-button button) {
    background: linear-gradient(180deg, #c1cbff 0%, #8ea2ff 18%, #4c65e2 58%, #2c3a97 100%) !important;
    border-color: rgba(42, 58, 151, 0.7) !important;
    box-shadow:
        0 14px 28px rgba(76, 101, 226, 0.26),
        0 1px 0 rgba(241, 244, 255, 0.34) inset,
        0 -3px 0 rgba(28, 40, 112, 0.3) inset !important;
}

:is(button#preset-reset-button, #preset-reset-button button) {
    background: linear-gradient(180deg, #ffe9b0 0%, #ffd463 18%, #e0a61f 58%, #8f6200 100%) !important;
    border-color: rgba(145, 99, 1, 0.68) !important;
    color: #fffdf5 !important;
    box-shadow:
        0 14px 28px rgba(224, 166, 31, 0.26),
        0 1px 0 rgba(255, 251, 231, 0.34) inset,
        0 -3px 0 rgba(109, 74, 2, 0.28) inset !important;
}

:is(button#preset-delete-button, #preset-delete-button button) {
    background: linear-gradient(180deg, #ffbdd1 0%, #ff7aa2 18%, #d62f6b 58%, #7b163d 100%) !important;
    border-color: rgba(125, 20, 62, 0.72) !important;
    box-shadow:
        0 14px 28px rgba(214, 47, 107, 0.28),
        0 1px 0 rgba(255, 238, 244, 0.34) inset,
        0 -3px 0 rgba(92, 10, 43, 0.32) inset !important;
}

@keyframes premium-button-glow {
    0%, 100% {
        box-shadow:
            0 14px 30px rgba(226, 58, 94, 0.26),
            0 1px 0 rgba(255, 255, 255, 0.35) inset,
            0 -3px 0 rgba(104, 10, 30, 0.32) inset;
    }
    50% {
        box-shadow:
            0 18px 38px rgba(226, 58, 94, 0.4),
            0 1px 0 rgba(255, 255, 255, 0.38) inset,
            0 -3px 0 rgba(104, 10, 30, 0.36) inset;
    }
}

@keyframes premium-button-sheen {
    0%, 100% {
        transform: translateX(-150%);
    }
    45%, 55% {
        transform: translateX(150%);
    }
}

@media (prefers-reduced-motion: reduce) {
    :is(button#generate-speech-button, #generate-speech-button button),
    :is(button#generate-speech-button, #generate-speech-button button)::before {
        animation: none !important;
    }
}
"""

# Try to import pydub for MP3 export
try:
    from pydub import AudioSegment
    MP3_AVAILABLE = True
except ImportError:
    MP3_AVAILABLE = False
    print("Warning: pydub not installed. MP3 export will not be available.")
    print("To enable MP3 export, install pydub: pip install pydub")

# Check if FFmpeg is available
def check_ffmpeg():
    try:
        result = subprocess.run(['ffmpeg', '-version'],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

FFMPEG_AVAILABLE = check_ffmpeg()
if not FFMPEG_AVAILABLE:
    print("Warning: FFmpeg not found in PATH. Video/audio processing will not work.")
    print("Please install FFmpeg: https://ffmpeg.org/download.html")

REFERENCE_WAVEFORM_OPTIONS = gr.WaveformOptions(
    waveform_color="#f7c0cb",
    waveform_progress_color="#a11436",
    trim_region_color="#e23a5e",
    sample_rate=24000,
)

def get_next_file_number(output_dir="outputs", target_folder=None, prefix=""):
    """Get the next available file number in sequence."""
    if target_folder:
        output_dir = target_folder

    os.makedirs(output_dir, exist_ok=True)

    # Find all existing files with our naming pattern
    existing_files = glob.glob(os.path.join(output_dir, f"{prefix}[0-9][0-9][0-9][0-9].*"))

    if not existing_files:
        return 1

    # Extract numbers from filenames
    numbers = []
    for filepath in existing_files:
        filename = os.path.basename(filepath)
        # Remove prefix if present
        if prefix:
            filename = filename[len(prefix):]
        # Extract the 4-digit number
        try:
            num_str = filename[:4]
            if num_str.isdigit():
                numbers.append(int(num_str))
        except:
            continue

    if numbers:
        return max(numbers) + 1
    else:
        return 1

def open_outputs_folder():
    """Open the outputs folder in the system's file manager (cross-platform)."""
    output_dir = os.path.abspath("outputs")
    os.makedirs(output_dir, exist_ok=True)

    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(output_dir)
        elif system == "Darwin":  # macOS
            subprocess.run(["open", output_dir])
        else:  # Linux and other Unix-like systems
            subprocess.run(["xdg-open", output_dir])
        print(f"Opened outputs folder: {output_dir}")
    except Exception as e:
        print(f"Failed to open outputs folder: {str(e)}")

def generate_output_path(target_folder=None, filename=None, save_as_mp3=False, prefix=""):
    """Generate output file path with sequential numbering."""
    output_dir = target_folder if target_folder else "outputs"
    os.makedirs(output_dir, exist_ok=True)

    if filename:
        # Use provided filename
        extension = ".mp3" if save_as_mp3 and MP3_AVAILABLE else ".wav"
        if not filename.endswith(('.wav', '.mp3')):
            filename = filename + extension
        return os.path.join(output_dir, filename)
    else:
        # Use sequential numbering
        next_num = get_next_file_number(output_dir, target_folder, prefix)
        extension = ".mp3" if save_as_mp3 and MP3_AVAILABLE else ".wav"
        filename = f"{prefix}{next_num:04d}{extension}"
        return os.path.join(output_dir, filename)

def convert_wav_to_mp3(wav_path, mp3_path, bitrate="256k"):
    """Convert WAV file to MP3 using pydub."""
    if not MP3_AVAILABLE:
        print("Warning: MP3 conversion not available. Keeping WAV format.")
        return wav_path

    try:
        audio = AudioSegment.from_wav(wav_path)
        audio.export(mp3_path, format="mp3", bitrate=bitrate)
        # Remove the original WAV file
        os.remove(wav_path)
        return mp3_path
    except Exception as e:
        print(f"Error converting to MP3: {e}")
        return wav_path

def extract_audio_from_media(media_path, output_path=None, sample_rate=24000):
    """Extract audio from video/audio file and convert to acceptable format using FFmpeg."""
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".wav")

    try:
        # Use FFmpeg subprocess directly for cross-platform compatibility
        cmd = [
            'ffmpeg', '-i', media_path,
            '-ar', str(sample_rate),
            '-ac', '1',  # mono
            '-acodec', 'pcm_s16le',
            '-f', 'wav',
            output_path,
            '-y',  # overwrite output
            '-loglevel', 'error'  # only show errors
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return None

        if os.path.exists(output_path):
            return output_path
        else:
            return None

    except FileNotFoundError:
        print("Error: FFmpeg not found. Please ensure FFmpeg is installed and in PATH.")
        return None
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

def extract_time_ranges(audio_path, time_ranges_str, sample_rate=24000):
    """Extract and merge audio segments based on time ranges using FFmpeg.
    Time ranges format: '1:3; 3:7; 11:15'
    """
    try:
        # Parse time ranges
        segments = []
        for range_str in time_ranges_str.split(';'):
            range_str = range_str.strip()
            if ':' in range_str:
                parts = range_str.split(':')
                if len(parts) == 2:
                    start, end = parts
                    try:
                        start_sec = float(start.strip())
                        end_sec = float(end.strip())
                        duration = end_sec - start_sec
                        if duration > 0:
                            segments.append((start_sec, duration))
                    except ValueError:
                        print(f"Invalid time range: {range_str}")
                        continue

        if not segments:
            return None

        # Create a temporary directory for segment files
        temp_dir = tempfile.mkdtemp()
        segment_files = []

        try:
            # Extract each segment using FFmpeg
            for i, (start, duration) in enumerate(segments):
                segment_file = os.path.join(temp_dir, f"segment_{i:03d}.wav")

                cmd = [
                    'ffmpeg', '-i', audio_path,
                    '-ss', str(start),  # start time
                    '-t', str(duration),  # duration
                    '-ar', str(sample_rate),
                    '-ac', '1',  # mono
                    '-acodec', 'pcm_s16le',
                    '-f', 'wav',
                    segment_file,
                    '-y',
                    '-loglevel', 'error'
                ]

                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                if result.returncode == 0 and os.path.exists(segment_file):
                    segment_files.append(segment_file)
                else:
                    print(f"Failed to extract segment {start}-{start+duration}: {result.stderr}")

            if not segment_files:
                return None

            # Merge all segments using FFmpeg concat
            output_path = tempfile.mktemp(suffix=".wav")

            if len(segment_files) == 1:
                # If only one segment, just copy it
                shutil.copy2(segment_files[0], output_path)
            else:
                # Create a concat file list
                concat_file = os.path.join(temp_dir, "concat_list.txt")
                with open(concat_file, 'w') as f:
                    for seg_file in segment_files:
                        # Use forward slashes for FFmpeg compatibility
                        seg_path = seg_file.replace(os.sep, '/')
                        f.write(f"file '{seg_path}'\n")

                # Concatenate using FFmpeg
                cmd = [
                    'ffmpeg', '-f', 'concat',
                    '-safe', '0',
                    '-i', concat_file,
                    '-ar', str(sample_rate),
                    '-ac', '1',
                    '-acodec', 'pcm_s16le',
                    '-f', 'wav',
                    output_path,
                    '-y',
                    '-loglevel', 'error'
                ]

                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                if result.returncode != 0:
                    print(f"Failed to merge segments: {result.stderr}")
                    return None

            return output_path if os.path.exists(output_path) else None

        finally:
            # Clean up temporary files
            for seg_file in segment_files:
                if os.path.exists(seg_file):
                    os.remove(seg_file)
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass

    except Exception as e:
        print(f"Error extracting time ranges: {e}")
        return None

def load_audio_from_path(audio_path):
    """Load audio from a file path."""
    if os.path.exists(audio_path):
        return audio_path
    else:
        return None

def save_pcm16_wav(audio_matrix, sampling_rate, output_path):
    """Save a mono/stereo int16 numpy array as a WAV file."""
    audio_matrix = ensure_audio_matrix(audio_matrix)

    if os.path.isfile(output_path):
        os.remove(output_path)
        print(">> remove old wav file:", output_path)
    if os.path.dirname(output_path) != "":
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    write_pcm16_wav(audio_matrix, sampling_rate, output_path)
    print(">> wav file saved to:", output_path)
    return output_path

def current_timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def format_elapsed_duration(elapsed_seconds):
    elapsed_ms = max(0, int(round(float(elapsed_seconds) * 1000.0)))
    hours, remainder_ms = divmod(elapsed_ms, 3600000)
    minutes, remainder_ms = divmod(remainder_ms, 60000)
    seconds, milliseconds = divmod(remainder_ms, 1000)

    if hours:
        return f"{hours}h {minutes}m {seconds}.{milliseconds:03d}s"
    if minutes:
        return f"{minutes}m {seconds}.{milliseconds:03d}s"
    return f"{seconds}.{milliseconds:03d}s"


def print_console_progress(label, completed, total, started_at, processed_audio_seconds=None, item_label="item"):
    total = max(1, int(total))
    completed = min(total, max(0, int(completed)))
    elapsed_seconds = max(0.0, time.perf_counter() - started_at)
    eta_seconds = ((elapsed_seconds / completed) * (total - completed)) if completed else None
    percent = 100.0 * completed / total
    parts = [
        f">> {label} {completed}/{total} {item_label}{'' if completed == 1 else 's'} ({percent:.1f}%)",
        f"elapsed {format_elapsed_duration(elapsed_seconds)}",
    ]
    if eta_seconds is not None:
        parts.append(f"eta {format_elapsed_duration(eta_seconds)}")
    if processed_audio_seconds is not None and processed_audio_seconds > 0 and elapsed_seconds > 0:
        parts.append(f"speed {processed_audio_seconds / elapsed_seconds:.2f}x RT")
        parts.append(f"audio {processed_audio_seconds:.2f}s")
    print(" | ".join(parts))


def audio_duration_ms(audio, sampling_rate):
    matrix = np.asarray(audio)
    if matrix.ndim == 0:
        return 0
    return int(round(matrix.shape[0] * 1000.0 / sampling_rate))


def finalize_subtitle_segment_audio(unit, unit_audio, sampling_rate, segment_path, temp_dir=None):
    natural_audio = ensure_audio_matrix(unit_audio)
    natural_duration_ms = audio_duration_ms(natural_audio, sampling_rate)

    fit_info = {
        "method": "copy",
        "source_duration_ms": int(natural_duration_ms),
        "target_duration_ms": int(unit.duration_ms),
        "delta_ms_before_fit": int(natural_duration_ms - unit.duration_ms),
        "stretch_rate": 1.0,
        "output_duration_ms": int(natural_duration_ms),
    }

    if natural_audio.shape[0] == 0 and unit.duration_ms <= 0:
        final_audio = natural_audio
        save_pcm16_wav(final_audio, sampling_rate, segment_path)
    elif natural_duration_ms == unit.duration_ms:
        final_audio = natural_audio
        save_pcm16_wav(final_audio, sampling_rate, segment_path)
    elif FFMPEG_AVAILABLE:
        if temp_dir:
            os.makedirs(temp_dir, exist_ok=True)
            raw_path = os.path.join(temp_dir, f"{unit.index:04d}_raw.wav")
        else:
            raw_path = f"{segment_path}.raw.wav"
        try:
            save_pcm16_wav(natural_audio, sampling_rate, raw_path)
            fit_info = retime_audio_file_with_ffmpeg(
                input_path=raw_path,
                output_path=segment_path,
                target_duration_ms=unit.duration_ms,
            )
            fitted_sampling_rate, final_audio = read_pcm16_wav(segment_path)
            if fitted_sampling_rate != sampling_rate:
                raise ValueError(
                    f"Subtitle unit {unit.index} retimed at {fitted_sampling_rate} Hz instead of {sampling_rate} Hz"
                )
        finally:
            if os.path.exists(raw_path):
                os.remove(raw_path)
    else:
        final_audio, fit_info = fit_audio_to_duration(
            natural_audio,
            sampling_rate=sampling_rate,
            target_duration_ms=unit.duration_ms,
            return_info=True,
        )
        fit_info = dict(fit_info)
        fit_info["method"] = f"python_{fit_info['method']}"
        fit_info["output_duration_ms"] = audio_duration_ms(final_audio, sampling_rate)
        save_pcm16_wav(final_audio, sampling_rate, segment_path)

    print(
        f">> Subtitle unit saved | unit {unit.index} | "
        f"target {unit.duration_ms / 1000.0:.2f}s | natural {natural_duration_ms / 1000.0:.2f}s | "
        f"final {fit_info['output_duration_ms'] / 1000.0:.2f}s | method {fit_info['method']} | "
        f"stretch {fit_info['stretch_rate']:.4f}"
    )
    return {
        "audio": final_audio,
        "segment_path": segment_path,
        "natural_duration_ms": int(natural_duration_ms),
        "fit_info": fit_info,
    }


def abs_path_or_none(path):
    return os.path.abspath(path) if path else None


def build_subtitle_status_message(cues, issues=None, sample_count=None, sampling_rate=None,
                                  task_folder=None, segments_dir=None, subtitle_file=None):
    if not cues:
        return "No subtitle cues loaded."

    format_label = get_subtitle_format_label(subtitle_file)
    render_units = build_subtitle_render_units(cues)
    message_parts = [
        f"Loaded {len(cues)} {format_label} cue(s).",
        f"Timeline end: {format_srt_timestamp(cues[-1].end_ms)}.",
        f"Synthesis units: {len(render_units)}.",
        (
            "Subtitle timing uses cue start times, merges overlapping cues into larger render units when needed, "
            "avoids extra section-gap silence inside each unit, and retimes each finished unit to its target slot."
            if FFMPEG_AVAILABLE
            else "Subtitle timing uses cue start times, merges overlapping cues into larger render units when needed, "
            "avoids extra section-gap silence inside each unit, and falls back to in-process duration fitting because FFmpeg is unavailable."
        ),
    ]

    if len(render_units) < len(cues):
        message_parts.append(
            f"Detected {len(cues) - len(render_units)} overlapping cue transition(s); overlapping cues will be synthesized as merged units."
        )

    if sample_count is not None and sampling_rate:
        message_parts.append(f"Generated output length: {sample_count / float(sampling_rate):.2f}s.")

    if issues is not None:
        late_starts = [issue["delta_ms"] for issue in issues if issue["type"] == "late_start"]
        overruns = [issue["delta_ms"] for issue in issues if issue["type"] == "slot_overrun"]
        if late_starts:
            message_parts.append(
                f"{len(late_starts)} cue(s) started late because earlier speech ran long. Max late start: {max(late_starts)}ms."
            )
        if overruns:
            message_parts.append(
                f"{len(overruns)} cue(s) exceeded their subtitle duration. Max overrun: {max(overruns)}ms."
            )
        if not late_starts and not overruns:
            message_parts.append("All subtitle cue starts were preserved without timing overruns.")

    if task_folder:
        message_parts.append(f"Task folder: {os.path.abspath(task_folder)}.")
    if segments_dir:
        message_parts.append(f"Separate cue WAVs: {os.path.abspath(segments_dir)}.")

    return " ".join(message_parts)


def normalize_emo_vector(emo_vector, apply_bias=True, max_emotion_sum=0.8, custom_biases=None):
    if apply_bias:
        emo_bias = custom_biases or [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]
        emo_vector = [vec * bias for vec, bias in zip(emo_vector, emo_bias)]

    emo_sum = sum(emo_vector)
    if emo_sum > max_emotion_sum and emo_sum > 0:
        scale_factor = max_emotion_sum / emo_sum
        emo_vector = [vec * scale_factor for vec in emo_vector]

    return emo_vector


_SUBPROCESS_STATE_LOCK = threading.Lock()
_SUBPROCESS_STATE = {
    "process": None,
    "request_file": None,
    "result_file": None,
    "metadata_path": None,
    "task_id": None,
    "canceled": False,
    "cancel_reason": None,
}


def _clear_subprocess_state(expected_process=None):
    with _SUBPROCESS_STATE_LOCK:
        process = _SUBPROCESS_STATE.get("process")
        if expected_process is not None and process is not expected_process:
            return None

        snapshot = dict(_SUBPROCESS_STATE)
        _SUBPROCESS_STATE.update(
            {
                "process": None,
                "request_file": None,
                "result_file": None,
                "metadata_path": None,
                "task_id": None,
                "canceled": False,
                "cancel_reason": None,
            }
        )
        return snapshot


def _register_subprocess_state(process, request_file, result_file, metadata_path, task_id):
    with _SUBPROCESS_STATE_LOCK:
        current_process = _SUBPROCESS_STATE.get("process")
        if current_process is not None and current_process.poll() is None:
            _terminate_process_tree(process)
            raise gr.Error("A subprocess generation is already running.")

        _SUBPROCESS_STATE.update(
            {
                "process": process,
                "request_file": request_file,
                "result_file": result_file,
                "metadata_path": metadata_path,
                "task_id": task_id,
                "canceled": False,
                "cancel_reason": None,
            }
        )


def _cleanup_temp_file(path):
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


def _parse_started_at_timestamp(started_at):
    if not started_at:
        return None
    try:
        return datetime.strptime(started_at, "%Y-%m-%dT%H:%M:%S%z")
    except Exception:
        return None


def _mark_metadata_canceled(metadata_path, reason):
    if not metadata_path or not os.path.exists(metadata_path):
        return

    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except Exception:
        return

    if metadata.get("status") == "completed":
        return

    now = current_timestamp()
    metadata["status"] = "canceled"
    metadata["updated_at"] = now
    metadata["error"] = reason
    metadata.setdefault("processing", {})
    metadata["processing"]["ended_at"] = now
    started_at = _parse_started_at_timestamp(metadata["processing"].get("started_at"))
    if started_at is not None:
        elapsed_seconds = max(0.0, (datetime.now(started_at.tzinfo) - started_at).total_seconds())
        metadata["processing"]["elapsed_ms"] = int(round(elapsed_seconds * 1000.0))
        metadata["processing"]["elapsed_seconds"] = round(elapsed_seconds, 3)
        metadata["processing"]["elapsed_human"] = format_elapsed_duration(elapsed_seconds)
    write_metadata_file(metadata_path, metadata)


def _terminate_process_tree(process):
    if process is None or process.poll() is not None:
        return

    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


def _prepare_generation_request(
    emo_control_method,
    prompt,
    text,
    subtitle_mode,
    subtitle_file,
    save_used_audio,
    output_filename,
    emo_ref_path,
    emo_weight,
    vec1,
    vec2,
    vec3,
    vec4,
    vec5,
    vec6,
    vec7,
    vec8,
    emo_text,
    emo_random,
    max_text_tokens_per_segment,
    save_as_mp3,
    diffusion_steps,
    inference_cfg_rate,
    interval_silence,
    max_speaker_audio_length,
    max_emotion_audio_length,
    autoregressive_batch_size,
    apply_emo_bias,
    max_emotion_sum,
    latent_multiplier,
    max_consecutive_silence,
    mp3_bitrate,
    do_sample,
    top_p,
    top_k,
    temperature,
    length_penalty,
    num_beams,
    repetition_penalty,
    max_mel_tokens,
    low_memory_mode,
    prevent_vram_accumulation,
    semantic_layer,
    cfm_cache_length,
    emo_bias_joy,
    emo_bias_anger,
    emo_bias_sad,
    emo_bias_fear,
    emo_bias_disgust,
    emo_bias_depression,
    emo_bias_surprise,
    emo_bias_calm,
    use_subprocess_system,
):
    subtitle_mode = bool(subtitle_mode)
    if not prompt:
        raise gr.Error("Speaker reference audio is required before you can generate speech.")

    processing_started_at = current_timestamp()
    subtitle_extension = get_subtitle_extension(subtitle_file) if subtitle_mode else None
    task_layout = create_task_output_layout(
        output_root="outputs",
        filename=output_filename,
        subtitle_mode=subtitle_mode,
        subtitle_extension=subtitle_extension,
    )
    metadata_path = task_layout["metadata_path"]
    task_folder = task_layout["task_folder"]

    if not isinstance(emo_control_method, int) and hasattr(emo_control_method, "value"):
        emo_control_method = emo_control_method.value
    emo_control_method = int(emo_control_method)

    if emo_control_method == 0:
        emo_ref_path = None
    if emo_control_method == 2:
        vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        custom_emo_biases = [
            emo_bias_joy,
            emo_bias_anger,
            emo_bias_sad,
            emo_bias_fear,
            emo_bias_disgust,
            emo_bias_depression,
            emo_bias_surprise,
            emo_bias_calm,
        ]
        vec = normalize_emo_vector(
            vec,
            apply_bias=apply_emo_bias,
            max_emotion_sum=max_emotion_sum,
            custom_biases=custom_emo_biases if apply_emo_bias else None,
        )
    else:
        vec = None

    if emo_text == "":
        emo_text = None

    print(f"Emo control mode:{emo_control_method},weight:{emo_weight},vec:{vec}")

    max_tokens = resolve_max_text_tokens(max_text_tokens_per_segment)
    infer_kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": int(num_beams),
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
        "emo_audio_prompt": emo_ref_path,
        "emo_alpha": float(emo_weight),
        "emo_vector": vec,
        "use_emo_text": (emo_control_method == 3),
        "emo_text": emo_text,
        "use_random": bool(emo_random),
        "verbose": bool(cmd_args.verbose),
        "max_text_tokens_per_segment": max_tokens,
        "interval_silence": int(interval_silence),
        "diffusion_steps": int(diffusion_steps),
        "inference_cfg_rate": float(inference_cfg_rate),
        "max_speaker_audio_length": float(max_speaker_audio_length),
        "max_emotion_audio_length": float(max_emotion_audio_length),
        "section_batch_size": int(autoregressive_batch_size),
        "max_emotion_sum": float(max_emotion_sum),
        "latent_multiplier": float(latent_multiplier),
        "max_consecutive_silence": int(max_consecutive_silence),
        "semantic_layer": int(semantic_layer),
        "cfm_cache_length": int(cfm_cache_length),
        "reset_beam_cache_per_segment": bool(prevent_vram_accumulation),
    }

    subtitle_cues = parse_subtitle_file(subtitle_file) if subtitle_mode else []
    subtitle_render_units = build_subtitle_render_units(subtitle_cues) if subtitle_mode else []
    resolved_settings = {
        "emotion_control_method_index": emo_control_method,
        "emotion_control_method_label": EMO_CHOICES_ALL[emo_control_method]
        if 0 <= emo_control_method < len(EMO_CHOICES_ALL)
        else str(emo_control_method),
        "save_used_audio": bool(save_used_audio),
        "save_as_mp3_requested": bool(save_as_mp3),
        "save_as_mp3_enabled": bool(save_as_mp3 and MP3_AVAILABLE),
        "mp3_bitrate": mp3_bitrate,
        "subtitle_mode": subtitle_mode,
        "subtitle_format": get_subtitle_format_label(subtitle_file) if subtitle_mode and subtitle_file else None,
        "low_memory_mode": bool(low_memory_mode),
        "prevent_vram_accumulation": bool(prevent_vram_accumulation),
        "use_subprocess_system": bool(use_subprocess_system),
        "execution_mode": "subprocess" if use_subprocess_system else "main_process",
        "resolved_generation_kwargs": infer_kwargs,
        "subtitle_timing_overrides": (
            {
                "interval_silence": SUBTITLE_TIMING_INTERVAL_SILENCE_MS,
                "ffmpeg_timing_fit_enabled": bool(FFMPEG_AVAILABLE),
                "timing_fit_background_workers": max(1, min(4, os.cpu_count() or 1)),
                "duration_retry_enabled": False,
            }
            if subtitle_mode
            else None
        ),
        "normalized_emotion_vector": vec,
    }

    metadata = {
        "status": "in_progress",
        "created_at": processing_started_at,
        "updated_at": processing_started_at,
        "task": {
            "id": task_layout["task_id"],
            "folder": abs_path_or_none(task_folder),
            "mode": "subtitle" if subtitle_mode else "text",
            "requested_output_filename": output_filename or "",
            "resolved_output_basename": task_layout["final_basename"],
        },
        "inputs": {
            "text": text,
            "speaker_reference_audio": abs_path_or_none(prompt),
            "emotion_reference_audio": abs_path_or_none(emo_ref_path),
            "subtitle_file": abs_path_or_none(subtitle_file),
        },
        "settings": resolved_settings,
        "outputs": {
            "final_audio_path": None,
            "final_wav_path": abs_path_or_none(task_layout["final_wav_path"]),
            "final_mp3_path": abs_path_or_none(task_layout["final_mp3_path"]) if save_as_mp3 and MP3_AVAILABLE else None,
            "metadata_path": abs_path_or_none(metadata_path),
            "segments_dir": abs_path_or_none(task_layout["segments_dir"]),
            "speaker_reference_copy_path": None,
            "subtitle_copy_path": None,
        },
        "processing": {
            "started_at": processing_started_at,
            "ended_at": None,
            "elapsed_ms": None,
            "elapsed_seconds": None,
            "elapsed_human": None,
        },
        "subtitle": None,
        "error": None,
    }

    if subtitle_mode:
        metadata["subtitle"] = {
            "format": get_subtitle_format_label(subtitle_file) if subtitle_file else None,
            "cue_count": len(subtitle_cues),
            "render_unit_count": len(subtitle_render_units),
            "timeline_end_ms": subtitle_cues[-1].end_ms if subtitle_cues else 0,
            "cues": [
                {
                    "index": cue.index,
                    "start_ms": cue.start_ms,
                    "end_ms": cue.end_ms,
                    "duration_ms": cue.duration_ms,
                    "text": cue.text,
                    "segment_file": None,
                    "generated_duration_ms": None,
                }
                for cue in subtitle_cues
            ],
            "render_units": [
                {
                    "index": unit.index,
                    "start_ms": unit.start_ms,
                    "end_ms": unit.end_ms,
                    "duration_ms": unit.duration_ms,
                    "text": unit.text,
                    "source_cue_indices": list(unit.cue_indices),
                    "segment_file": None,
                    "natural_duration_ms": None,
                    "generated_duration_ms": None,
                    "duration_delta_before_fit_ms": None,
                    "fit_method": None,
                    "fit_stretch_rate": None,
                    "selected_latent_multiplier": float(infer_kwargs["latent_multiplier"]),
                    "retry_attempted": False,
                    "retry_selected": False,
                    "retry_latent_multiplier": None,
                }
                for unit in subtitle_render_units
            ],
            "timing_issues": [],
        }

    if subtitle_mode and subtitle_file and task_layout["subtitle_copy_path"]:
        shutil.copy2(subtitle_file, task_layout["subtitle_copy_path"])
        metadata["outputs"]["subtitle_copy_path"] = abs_path_or_none(task_layout["subtitle_copy_path"])

    write_metadata_file(metadata_path, metadata)

    return {
        "runtime": _build_tts_runtime_options(),
        "task_layout": task_layout,
        "metadata_path": metadata_path,
        "task_id": task_layout["task_id"],
        "prompt": prompt,
        "text": text,
        "subtitle_mode": subtitle_mode,
        "subtitle_file": subtitle_file,
        "save_used_audio": bool(save_used_audio),
        "save_as_mp3": bool(save_as_mp3),
        "mp3_bitrate": mp3_bitrate,
        "infer_kwargs": infer_kwargs,
        "low_memory_mode": bool(low_memory_mode),
        "max_text_tokens": max_tokens,
    }


def _run_generation_subprocess(request):
    request_fd, request_path = tempfile.mkstemp(prefix="indextts_request_", suffix=".json")
    os.close(request_fd)
    result_fd, result_path = tempfile.mkstemp(prefix="indextts_result_", suffix=".json")
    os.close(result_fd)

    try:
        with open(request_path, "w", encoding="utf-8") as handle:
            json.dump(request, handle, indent=2, ensure_ascii=False)

        cmd = [
            sys.executable,
            os.path.join(current_dir, "webui_subprocess_worker.py"),
            "--request-file",
            request_path,
            "--result-file",
            result_path,
        ]
        popen_kwargs = {
            "cwd": current_dir,
            "env": {**os.environ, "PYTHONUNBUFFERED": "1"},
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            popen_kwargs["start_new_session"] = True

        process = subprocess.Popen(cmd, **popen_kwargs)
        _register_subprocess_state(
            process,
            request_path,
            result_path,
            request["metadata_path"],
            request["task_id"],
        )

        return_code = process.wait()
        state_snapshot = _clear_subprocess_state(process) or {}
        canceled = bool(state_snapshot.get("canceled"))
        cancel_reason = state_snapshot.get("cancel_reason") or "Generation was canceled."

        if canceled:
            raise gr.Error(cancel_reason)

        if not os.path.exists(result_path):
            if return_code != 0:
                raise gr.Error("Generation subprocess exited unexpectedly. Check the console output.")
            raise gr.Error("Generation subprocess finished without a result file.")

        with open(result_path, "r", encoding="utf-8") as handle:
            result = json.load(handle)

        if result.get("status") != "ok":
            raise gr.Error(result.get("error") or "Generation subprocess failed.")

        subtitle_status_message = result.get("subtitle_status") or ""
        return (
            gr.update(value=result["output_path"], visible=True),
            gr.update(value=subtitle_status_message, visible=bool(subtitle_status_message)),
        )
    finally:
        _cleanup_temp_file(request_path)
        _cleanup_temp_file(result_path)


def cancel_generation_process(use_subprocess_system, cancel_confirmed):
    if not cancel_confirmed:
        return gr.update()

    with _SUBPROCESS_STATE_LOCK:
        process = _SUBPROCESS_STATE.get("process")
        metadata_path = _SUBPROCESS_STATE.get("metadata_path")
        task_id = _SUBPROCESS_STATE.get("task_id")
        if process is None or process.poll() is not None:
            _SUBPROCESS_STATE.update(
                {
                    "process": None,
                    "request_file": None,
                    "result_file": None,
                    "metadata_path": None,
                    "task_id": None,
                    "canceled": False,
                    "cancel_reason": None,
                }
            )
            message = (
                "Subprocess mode is disabled and no subprocess generation is running."
                if not use_subprocess_system
                else "No subprocess generation is currently running."
            )
            return gr.update(value=message)

        _SUBPROCESS_STATE["canceled"] = True
        _SUBPROCESS_STATE["cancel_reason"] = "Generation canceled by user."

    _mark_metadata_canceled(metadata_path, "Generation canceled by user.")
    _terminate_process_tree(process)
    task_label = f" task {task_id}" if task_id else ""
    return gr.update(value=f"Cancel signal sent to subprocess for{task_label}.")


def on_subprocess_mode_change(use_subprocess_system):
    if use_subprocess_system:
        unload_inprocess_tts()


def resolve_max_text_tokens(max_text_tokens_per_segment):
    if not max_text_tokens_per_segment:
        return 120

    try:
        max_tokens = int(float(str(max_text_tokens_per_segment).strip()))
        return max(20, min(max_tokens, PREVIEW_MAX_TEXT_TOKENS))
    except (ValueError, TypeError):
        return 120


def get_text_processing_sections(text, max_text_tokens_per_segment):
    if not text:
        return []

    max_tokens = resolve_max_text_tokens(max_text_tokens_per_segment)
    text_tokens_list = PREVIEW_TEXT_TOKENIZER.tokenize(text)
    return PREVIEW_TEXT_TOKENIZER.split_segments(text_tokens_list, max_text_tokens_per_segment=max_tokens)


def build_section_count_message(text, max_text_tokens_per_segment, subtitle_mode=False, subtitle_file=None):
    if subtitle_mode and subtitle_file:
        try:
            cues = parse_subtitle_file(subtitle_file)
            render_units = build_subtitle_render_units(cues)
            processing_sections = 0
            for unit in render_units:
                if unit.text.strip():
                    processing_sections += len(get_text_processing_sections(unit.text, max_text_tokens_per_segment))

            if processing_sections == len(render_units):
                return (
                    f"**Current Sections:** {processing_sections} subtitle timing "
                    f"{'unit' if processing_sections == 1 else 'units'} from {len(cues)} cue"
                    f"{'' if len(cues) == 1 else 's'}"
                )

            return (
                f"**Current Sections:** {processing_sections} processing section"
                f"{'' if processing_sections == 1 else 's'} across {len(render_units)} subtitle timing unit"
                f"{'' if len(render_units) == 1 else 's'} from {len(cues)} cue"
                f"{'' if len(cues) == 1 else 's'}"
            )
        except Exception as e:
            return f"**Current Sections:** Unable to read subtitle file: {html.escape(str(e))}"

    sections = get_text_processing_sections(text, max_text_tokens_per_segment)
    return f"**Current Sections:** {len(sections)} text section{'' if len(sections) == 1 else 's'}"


def get_preview_rows(text, max_text_tokens_per_segment, subtitle_mode=False, subtitle_file=None):
    if subtitle_mode and subtitle_file:
        try:
            cues = parse_subtitle_file(subtitle_file)
            data = []
            cue_label = f"{get_subtitle_format_label(subtitle_file)} Cue"
            for cue in cues:
                details = f"{format_srt_timestamp(cue.start_ms)} -> {format_srt_timestamp(cue.end_ms)} ({cue.duration_ms} ms)"
                data.append([cue.index, cue_label, cue.text, details])
            return data
        except Exception as e:
            return [[0, "Caption Error", str(e), ""]]

    if not text:
        return []

    segments = get_text_processing_sections(text, max_text_tokens_per_segment)

    data = []
    for i, segment_tokens in enumerate(segments):
        segment_str = ''.join(segment_tokens)
        tokens_count = len(segment_tokens)
        data.append([i, "Text Segment", segment_str, f"{tokens_count} tokens"])
    return data


def gen_single(emo_control_method,prompt, text, subtitle_mode, subtitle_file, save_used_audio, output_filename,
               emo_ref_path, emo_weight,
               vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
               emo_text,emo_random,
               max_text_tokens_per_segment,
               save_as_mp3,
               # Expert params (in order from expert_params list)
               diffusion_steps,
               inference_cfg_rate,
               interval_silence,
               max_speaker_audio_length,
               max_emotion_audio_length,
               autoregressive_batch_size,
               apply_emo_bias,
               max_emotion_sum,
               latent_multiplier,
               max_consecutive_silence,
               mp3_bitrate,
               # Advanced params (in order from advanced_params list)
               do_sample,
               top_p,
               top_k,
               temperature,
               length_penalty,
               num_beams,
               repetition_penalty,
               max_mel_tokens,
               low_memory_mode,
               prevent_vram_accumulation,
               # Model params (semantic layer, cache, emotion biases)
               semantic_layer,
               cfm_cache_length,
               emo_bias_joy,
               emo_bias_anger,
               emo_bias_sad,
               emo_bias_fear,
               emo_bias_disgust,
               emo_bias_depression,
               emo_bias_surprise,
               emo_bias_calm,
               use_subprocess_system=True,
               progress=gr.Progress()):
    request = _prepare_generation_request(
        emo_control_method,
        prompt,
        text,
        subtitle_mode,
        subtitle_file,
        save_used_audio,
        output_filename,
        emo_ref_path,
        emo_weight,
        vec1,
        vec2,
        vec3,
        vec4,
        vec5,
        vec6,
        vec7,
        vec8,
        emo_text,
        emo_random,
        max_text_tokens_per_segment,
        save_as_mp3,
        diffusion_steps,
        inference_cfg_rate,
        interval_silence,
        max_speaker_audio_length,
        max_emotion_audio_length,
        autoregressive_batch_size,
        apply_emo_bias,
        max_emotion_sum,
        latent_multiplier,
        max_consecutive_silence,
        mp3_bitrate,
        do_sample,
        top_p,
        top_k,
        temperature,
        length_penalty,
        num_beams,
        repetition_penalty,
        max_mel_tokens,
        low_memory_mode,
        prevent_vram_accumulation,
        semantic_layer,
        cfm_cache_length,
        emo_bias_joy,
        emo_bias_anger,
        emo_bias_sad,
        emo_bias_fear,
        emo_bias_disgust,
        emo_bias_depression,
        emo_bias_surprise,
        emo_bias_calm,
        bool(use_subprocess_system),
    )

    if use_subprocess_system:
        return _run_generation_subprocess(request)

    result = run_generation_request(request, tts.get_instance(), progress_callback=progress)
    subtitle_status_message = result.get("subtitle_status") or ""
    return (
        gr.update(value=result["output_path"], visible=True),
        gr.update(value=subtitle_status_message, visible=bool(subtitle_status_message)),
    )

def update_prompt_audio():
    update_button = gr.update(interactive=True)
    return update_button


def _sanitize_preset_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(name))
    return safe.strip("._") or "default"


def _ui_preset_path(preset_name: str) -> Path:
    return Path(PRESETS_DIR) / f"{_sanitize_preset_name(preset_name)}.json"


def _list_ui_presets() -> List[str]:
    root = Path(PRESETS_DIR)
    saved = sorted(
        p.stem
        for p in root.glob("*.json")
        if p.is_file() and p.stem != DEFAULT_UI_PRESET_NAME
    )
    return [DEFAULT_UI_PRESET_NAME] + saved


def _set_last_used_ui_preset(preset_name: str) -> None:
    try:
        Path(PRESETS_DIR).mkdir(parents=True, exist_ok=True)
        (Path(PRESETS_DIR) / _LAST_USED_UI_PRESET_FILE).write_text(
            _sanitize_preset_name(preset_name),
            encoding="utf-8",
        )
    except Exception:
        pass


def _get_last_used_ui_preset() -> Optional[str]:
    path = Path(PRESETS_DIR) / _LAST_USED_UI_PRESET_FILE
    if not path.exists():
        return None

    try:
        name = path.read_text(encoding="utf-8").strip()
    except Exception:
        return None

    return name if name in _list_ui_presets() else None


def _save_ui_preset(preset_name: str, config: Dict[str, Any]) -> str:
    if not preset_name or not str(preset_name).strip():
        raise ValueError("Preset name cannot be empty.")

    safe_name = _sanitize_preset_name(preset_name)
    if safe_name == DEFAULT_UI_PRESET_NAME:
        raise ValueError(f"Preset name '{DEFAULT_UI_PRESET_NAME}' is reserved.")

    cfg = dict(config)
    cfg.setdefault("_meta", {})
    cfg["_meta"]["version"] = UI_PRESET_VERSION
    cfg["_meta"]["format"] = UI_PRESET_FORMAT
    cfg["_meta"]["last_modified"] = datetime.now().isoformat()
    if "created_at" not in cfg["_meta"]:
        cfg["_meta"]["created_at"] = cfg["_meta"]["last_modified"]

    out_path = _ui_preset_path(safe_name)
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(out_path)
    _set_last_used_ui_preset(safe_name)
    return safe_name


def _load_ui_preset(preset_name: str) -> Optional[Dict[str, Any]]:
    if not preset_name or str(preset_name).strip() == DEFAULT_UI_PRESET_NAME:
        return None

    path = _ui_preset_path(preset_name)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    _set_last_used_ui_preset(preset_name)
    return data


def _delete_ui_preset(preset_name: str) -> bool:
    if not preset_name or str(preset_name).strip() == DEFAULT_UI_PRESET_NAME:
        return False

    path = _ui_preset_path(preset_name)
    if not path.exists():
        return False

    try:
        path.unlink()
        return True
    except Exception:
        return False


theme = gr.themes.Soft()
theme.font = [gr.themes.GoogleFont("Inter"), "Tahoma", "ui-sans-serif", "system-ui", "sans-serif"]
with gr.Blocks(title=APP_TITLE) as demo:
    mutex = threading.Lock()
    gr.Markdown("## Index TTS2 Premium SECourses App : https://www.patreon.com/posts/139297407")

    with gr.Tab("Audio Generation"):
        with gr.Row(equal_height=False):
            os.makedirs("prompts",exist_ok=True)

            with gr.Column(scale=1, min_width=280):
                with gr.Group(elem_classes="top-input-panel"):
                    gr.Markdown("### Reference Media (Mandatory!) (Recommended 15 seconds)")
                    with gr.Group(elem_classes="reference-subsection"):
                        media_upload = gr.File(
                            label="Upload Speaker Reference / Audio / Video",
                            file_count="single",
                            file_types=MEDIA_FILE_TYPES,
                            type="filepath",
                            height=120,
                            elem_classes="top-section-flat",
                        )
                        time_ranges_input = gr.Textbox(
                            label="Extract Audio Segments (optional)",
                            placeholder="e.g., 1:3; 3:7; 11:15",
                            value="",
                            info="Optional. Extract and merge only these time ranges from the uploaded audio/video before using it as the speaker reference."
                        )
                        extract_button = gr.Button(
                            "Extract and Use Audio",
                            variant="secondary",
                            elem_id="extract-audio-button",
                            elem_classes=["action-button"],
                        )
                        gr.Markdown("##### Load Audio from Path")
                        with gr.Row():
                            audio_path_input = gr.Textbox(
                                label="Audio File Path",
                                placeholder="Enter full path to audio or video file",
                                value=""
                            )
                            load_audio_button = gr.Button(
                                "Load Audio",
                                variant="secondary",
                                elem_id="load-audio-button",
                                elem_classes=["action-button"],
                            )
                    gr.Markdown("#### Record Speaker Audio", elem_classes="reference-subsection-title")
                    with gr.Group(elem_classes="reference-subsection"):
                        prompt_audio = gr.Audio(
                            label="Active Speaker Reference Audio (Required, 3-90 seconds)",
                            key="prompt_audio",
                            sources=["microphone"],
                            type="filepath",
                            format="wav",
                            elem_id="speaker-reference-audio",
                            waveform_options=REFERENCE_WAVEFORM_OPTIONS,
                            elem_classes="top-section-flat",
                        )
                    reference_status = gr.Textbox(
                        label="Reference Audio Status",
                        value="",
                        interactive=False,
                        visible=False
                    )

                prompt_list = os.listdir("prompts")
                default = ''
                if prompt_list:
                    default = prompt_list[0]

            with gr.Column(scale=1, min_width=280):
                with gr.Group(elem_classes=["subtitle-controls-group", "top-input-panel"]):
                    gr.Markdown("### Captions")
                    subtitle_file = gr.File(
                        label="Caption File (.srt, .vtt, .sbv)",
                        file_count="single",
                        file_types=list(SUPPORTED_SUBTITLE_EXTENSIONS),
                        type="filepath",
                        height=120,
                        elem_classes="top-section-flat",
                    )
                    subtitle_mode = gr.Checkbox(
                        label="Use Caption Cue Timing",
                        value=False,
                        info="Generate separate caption timing units, then auto-retime each finished unit to the caption duration before timeline assembly."
                    )
                    gr.Markdown(CAPTION_TIMING_HELP, elem_classes="caption-timing-help")
                    subtitle_status = gr.Textbox(
                        label="Caption Timing Status",
                        value="",
                        interactive=False,
                        visible=False
                    )

            with gr.Column(scale=1, min_width=300):
                use_subprocess_system = gr.Checkbox(
                    label="Use Subprocess System",
                    value=True,
                    info="When enabled, each generation runs in a separate process so the main UI can stay clean and releasing the child process frees its RAM and VRAM."
                )
                input_text_single = gr.TextArea(
                    label="Text to Synthesize",
                    key="input_text_single",
                    elem_id="input-text-source",
                    placeholder="Enter the text you want to convert to speech",
                    info=f"Model v{MODEL_VERSION} | Supports multiple languages. Long texts are automatically segmented. Upload a caption file (.srt/.vtt/.sbv) when you want cue-by-cue timing."
                )
                section_count_refresh_signal = gr.Textbox(
                    value="",
                    show_label=False,
                    container=False,
                    elem_id="section-count-refresh-signal",
                    elem_classes=["ui-hidden-signal"],
                )
                with gr.Row():
                    gen_button = gr.Button(
                        "Generate Speech",
                        key="gen_button",
                        elem_id="generate-speech-button",
                        elem_classes=["action-button"],
                        interactive=True,
                        variant="primary"
                    )
                    open_outputs_button = gr.Button(
                        "Open Outputs Folder",
                        key="open_outputs_button",
                        elem_id="open-outputs-button",
                        elem_classes=["action-button"],
                    )

                section_count_label = gr.Markdown("**Current Sections:** 0")
                autoregressive_batch_size = gr.Slider(
                    label="Section Batch Size",
                    value=1,
                    minimum=1,
                    maximum=8,
                    step=1,
                    info="Real micro-batch size for processing multiple text/subtitle sections together with shared reference conditioning. Higher values increase throughput with a smaller VRAM increase than parallel runs, but still use more memory. Start with 2."
                )

                # Output filename and save used audio options
                with gr.Row():
                    output_filename = gr.Textbox(
                        label="Output Filename (optional)",
                        placeholder="Optional final filename inside the numbered task folder",
                        value=""
                    )
                    save_used_audio = gr.Checkbox(
                        label="Save Used Reference Audio",
                        value=False,
                        info="Copy the speaker reference audio into this generation's numbered task folder"
                    )

            with gr.Column(scale=1, min_width=280):
                output_audio = gr.Audio(
                    label="Generated Result (click to play/download)",
                    visible=True,
                    key="output_audio"
                )
                with gr.Accordion("Config Presets (Save / Load)", open=True):
                    gr.Markdown(
                        "Saves and loads tunable controls from the Audio Generation and Advanced Parameters tabs. Working content like Text to Synthesize, uploaded subtitle/reference files, and active reference-media inputs is intentionally not included."
                    )
                    ui_preset_dropdown = gr.Dropdown(
                        label="Select Preset",
                        choices=_list_ui_presets(),
                        value=(_get_last_used_ui_preset() or DEFAULT_UI_PRESET_NAME),
                        allow_custom_value=False,
                    )
                    ui_preset_name = gr.Textbox(
                        label="New Preset Name",
                        placeholder="Enter a preset name to save",
                        value="",
                    )
                    with gr.Row():
                        ui_preset_save_btn = gr.Button(
                            "Save",
                            variant="primary",
                            elem_id="preset-save-button",
                            elem_classes=["action-button"],
                        )
                        ui_preset_load_btn = gr.Button(
                            "Load Selected",
                            elem_id="preset-load-button",
                            elem_classes=["action-button"],
                        )
                    with gr.Row():
                        ui_preset_reset_btn = gr.Button(
                            "Reset Defaults",
                            variant="secondary",
                            elem_id="preset-reset-button",
                            elem_classes=["action-button"],
                        )
                        ui_preset_delete_btn = gr.Button(
                            "Delete",
                            variant="stop",
                            elem_id="preset-delete-button",
                            elem_classes=["action-button"],
                        )
                    ui_preset_status = gr.Markdown("")
                cancel_process_button = gr.Button(
                    "Cancel Running Process",
                    variant="stop",
                    elem_id="cancel-generation-button",
                    elem_classes=["action-button"],
                )
                cancel_process_note = gr.Markdown("Small note: works only when subprocess mode is enabled.")
                cancel_confirm_signal = gr.Checkbox(value=False, visible=False)
                cancel_process_status = gr.Markdown("")

        with gr.Accordion("Function Settings"):
            # 情感控制选项部分 - now showing ALL options including experimental
            with gr.Row():
                emo_control_method = gr.Radio(
                    choices=EMO_CHOICES_ALL,
                    type="index",
                    value=EMO_CHOICES_ALL[0],
                    label="Emotion Control Method",
                    info="Choose how to control emotions: Speaker's natural emotion, reference audio emotion, manual vector control, or text description"
                )
        # 情感参考音频部分
        with gr.Group(visible=False) as emotion_reference_group:
            with gr.Row():
                emo_upload = gr.Audio(
                    label="Upload Emotion Reference Audio",
                    type="filepath"
                )

        # 情感随机采样
        with gr.Row(visible=False) as emotion_randomize_group:
            emo_random = gr.Checkbox(
                label="Random Emotion Sampling",
                value=False,
                info="Enable random sampling from emotion matrix for more varied emotional expression"
            )

        # 情感向量控制部分
        with gr.Group(visible=False) as emotion_vector_group:
            with gr.Row():
                with gr.Column():
                    vec1 = gr.Slider(label="Joy", minimum=0.0, maximum=1.0, value=0.0, step=0.05, info="Happiness and cheerfulness in voice")
                    vec2 = gr.Slider(label="Anger", minimum=0.0, maximum=1.0, value=0.0, step=0.05, info="Aggressive and forceful tone")
                    vec3 = gr.Slider(label="Sadness", minimum=0.0, maximum=1.0, value=0.0, step=0.05, info="Melancholic and sorrowful expression")
                    vec4 = gr.Slider(label="Fear", minimum=0.0, maximum=1.0, value=0.0, step=0.05, info="Anxious and worried tone")
                with gr.Column():
                    vec5 = gr.Slider(label="Disgust", minimum=0.0, maximum=1.0, value=0.0, step=0.05, info="Repulsed and disgusted expression")
                    vec6 = gr.Slider(label="Depression", minimum=0.0, maximum=1.0, value=0.0, step=0.05, info="Low energy and melancholic mood")
                    vec7 = gr.Slider(label="Surprise", minimum=0.0, maximum=1.0, value=0.0, step=0.05, info="Shocked and amazed reaction")
                    vec8 = gr.Slider(label="Calm", minimum=0.0, maximum=1.0, value=0.0, step=0.05, info="Neutral and peaceful tone")

        with gr.Group(visible=False) as emo_text_group:
            with gr.Row():
                emo_text = gr.Textbox(label="Emotion Description Text",
                                      placeholder="Enter emotion description (or leave empty to automatically use target text as emotion description)",
                                      value="",
                                      info="e.g.: feeling wronged, danger is approaching quietly")

        with gr.Row(visible=False) as emo_weight_group:
            emo_weight = gr.Slider(
                label="Emotion Weight",
                minimum=0.0,
                maximum=1.0,
                value=0.65,
                step=0.01,
                info="Controls the strength of emotion blending. 0 = no emotion, 1 = full emotion from reference. Default: 0.65"
            )

        with gr.Accordion("Advanced Generation Parameter Settings", open=True, visible=True) as advanced_settings_group:
            # Row 1: Diffusion Steps and CFG Rate
            with gr.Row():
                diffusion_steps = gr.Slider(
                    label="Diffusion Steps",
                    value=25,
                    minimum=10,
                    maximum=100,
                    step=1,
                    info="Number of denoising steps in the diffusion model. Higher = better quality but slower. Default: 25"
                )
                inference_cfg_rate = gr.Slider(
                    label="CFG Rate (Classifier-Free Guidance)",
                    value=0.7,
                    minimum=0.0,
                    maximum=2.0,
                    step=0.05,
                    info="Controls how strongly the model follows the voice, emotion, and style characteristics from reference audio. Higher values = stricter adherence to reference, lower = more variation. 0.0 = no guidance (random), 0.7 = balanced (default), >1.0 = very strong adherence to reference characteristics."
                )

            # Row 2: Reference Audio Processing Limits
            with gr.Row():
                with gr.Column():
                    max_speaker_audio_length = gr.Slider(
                        label="Max Speaker Reference Length (seconds)",
                        value=30,
                        minimum=3,
                        maximum=90,
                        step=1,
                        info="How much of the speaker reference audio to use. Model works best with 5-15 seconds. Maximum set to 90 seconds for safety. Default: 30s"
                    )
                with gr.Column():
                    max_emotion_audio_length = gr.Slider(
                        label="Max Emotion Reference Length (seconds)",
                        value=30,
                        minimum=3,
                        maximum=90,
                        step=1,
                        info="How much of the emotion reference audio to use. Model works best with 5-15 seconds. Maximum set to 90 seconds for safety. Default: 30s"
                    )

            # Row 3: Enable Sampling and Temperature
            with gr.Row():
                do_sample = gr.Checkbox(
                    label="Enable Sampling",
                    value=True,
                    info="When ON: Uses random sampling for natural, varied speech. When OFF: Always picks most likely tokens for consistent but potentially robotic output. Keep ON for natural speech."
                )
                temperature = gr.Slider(
                    label="Temperature",
                    minimum=0.1,
                    maximum=2.0,
                    value=0.8,
                    step=0.1,
                    info="Controls speech expressiveness. Higher (0.9-1.2) = more varied intonation and expression. Lower (0.3-0.7) = flatter but more stable speech. Default 0.8 is balanced."
                )

            # Row 4: Beam Search Beams and Max Tokens per Segment
            with gr.Row():
                num_beams = gr.Slider(
                    label="Beam Search Beams",
                    value=3,
                    minimum=1,
                    maximum=10,
                    step=1,
                    info="Explores multiple generation paths simultaneously. Higher (5-10) = better quality but slower. Lower (1-3) = faster but potentially worse quality. Default 3 balances speed and quality. Bigger also uses more VRAM."
                )
                initial_value = max(20, min(PREVIEW_MAX_TEXT_TOKENS, cmd_args.gui_seg_tokens))
                max_text_tokens_per_segment = gr.Textbox(
                    label="Max Tokens per Segment",
                    value=str(initial_value),
                    key="max_text_tokens_per_segment",
                    elem_id="max-tokens-segment-source",
                    info=f"Splits long text into chunks for processing. Valid range: 20-{PREVIEW_MAX_TEXT_TOKENS}. Smaller (80-120) = more natural pauses and consistent quality but slower. Larger (150-200) = faster but may have quality variations. Default: {initial_value}. Bigger value uses more VRAM."
                )

            # Row 5: Save as MP3 and Low Memory Mode
            with gr.Row():
                save_as_mp3 = gr.Checkbox(
                    label="Save as MP3",
                    value=False,
                    visible=MP3_AVAILABLE,
                    info="Save audio as MP3 format instead of WAV" if MP3_AVAILABLE else "Requires pydub: pip install pydub"
                )
                low_memory_mode = gr.Checkbox(
                    label="Low Memory Mode",
                    value=False,
                    info="Enable low memory mode for systems with limited GPU memory (inference will be slower)"
                )
                prevent_vram_accumulation = gr.Checkbox(
                    label="Prevent VRAM Accumulation",
                    value=False,
                    info="Reset beam search cache after each segment. Helps avoid VRAM growth at higher beams (e.g., 8). Slight performance impact."
                )

        with gr.Accordion("Preview Sentence Segmentation Results", open=True) as segments_settings:
            segments_preview = gr.Dataframe(
                headers=["Index", "Type", "Content", "Details"],
                key="segments_preview",
                wrap=True,
            )

    with gr.Tab("Advanced Parameters"):
        gr.Markdown("### 🎯 Advanced Audio Generation Parameters")
        gr.Markdown("_Fine-tune generation parameters for expert control over audio synthesis._")

        with gr.Row():
            with gr.Column():
                mp3_bitrate = gr.Dropdown(
                    label="MP3 Bitrate",
                    choices=["128k", "192k", "256k", "320k"],
                    value="256k",
                    info="Audio quality when saving as MP3. 128k = smaller files but lower quality. 320k = best quality but larger files. 256k = good balance for most uses."
                )
            with gr.Column():
                latent_multiplier = gr.Slider(
                    label="Latent Length Multiplier",
                    value=1.72,
                    minimum=1.0,
                    maximum=3.0,
                    step=0.01,
                    info="Controls speech pacing speed. Higher (2.0-3.0) = slower, more stretched speech. Lower (1.0-1.5) = faster, more compressed speech. Default 1.72 is natural pacing."
                )

        with gr.Row():
            with gr.Column():
                top_p = gr.Slider(
                    label="Top-p (Nucleus Sampling)",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.8,
                    step=0.01,
                    info="Limits token selection to most probable options. Higher (0.9-1.0) = more varied and expressive speech. Lower (0.3-0.7) = more predictable, conservative speech. Default 0.8 balances variety and stability."
                )
            with gr.Column():
                top_k = gr.Slider(
                    label="Top-k",
                    minimum=0,
                    maximum=100,
                    value=30,
                    step=1,
                    info="Limits selection to k most probable tokens. Higher (50-100) = more speech variety. Lower (10-30) = more consistent speech. 0 = disabled. Default 30 avoids unlikely tokens while maintaining variety."
                )

        with gr.Row():
            with gr.Column():
                repetition_penalty = gr.Number(
                    label="Repetition Penalty",
                    precision=None,
                    value=10.0,
                    minimum=1.0,
                    maximum=20.0,
                    step=0.1,
                    info="Prevents speech from getting stuck in loops. Higher (10-15) = strongly avoids repetition. Lower (1-5) = allows natural repetition. Default 10.0 effectively prevents stuttering."
                )
            with gr.Column():
                length_penalty = gr.Number(
                    label="Length Penalty",
                    precision=None,
                    value=0.0,
                    minimum=-2.0,
                    maximum=2.0,
                    step=0.1,
                    info="Influences speech segment length. Positive (0.5-2.0) = longer segments. Negative (-2.0 to -0.5) = shorter segments. Zero = natural length based on content."
                )

        with gr.Row():
            with gr.Column():
                max_consecutive_silence = gr.Slider(
                    label="Max Consecutive Silent Tokens (0=disabled)",
                    value=0,
                    minimum=0,
                    maximum=100,
                    step=5,
                    info="Removes long pauses in speech. Higher (30-50) = allows longer natural pauses. Lower (5-20) = tighter, more continuous speech. 0 = no pause removal. Try 30 if output has awkward long silences."
                )
            with gr.Column():
                interval_silence = gr.Slider(
                    label="Silence Between Segments (ms)",
                    value=200,
                    minimum=0,
                    maximum=1000,
                    step=50,
                    info="Pause length between text segments. Higher (500-1000ms) = formal presentation style with clear breaks. Lower (50-200ms) = conversational flow. Default 200ms is natural for most content."
                )

        gr.Markdown("### 🎯 Emotion Control Parameters")
        with gr.Row():
            with gr.Column():
                apply_emo_bias = gr.Checkbox(
                    label="Apply Emotion Bias Correction",
                    value=True,
                    info="Prevents emotions from becoming too extreme or unnatural. Keeps emotional expression balanced and realistic. Recommended: Keep ON."
                )
            with gr.Column():
                max_emotion_sum = gr.Slider(
                    label="Max Total Emotion Strength",
                    value=0.8,
                    minimum=0.1,
                    maximum=2.0,
                    step=0.05,
                    info="Limits overall emotional intensity. Higher (1.0-2.0) = stronger emotions allowed. Lower (0.3-0.7) = more subtle emotions. Default 0.8 keeps emotions natural."
                )

        with gr.Row():
            with gr.Column():
                max_mel_tokens = gr.Slider(
                    label="Max Mel Tokens",
                    value=1500,
                    minimum=50,
                    maximum=1815,
                    step=10,
                    info=f"Maximum speech length per segment. 1815 tokens ≈ 84 seconds. Lower values may cut off long segments. Default 1500 works for most content.",
                    key="max_mel_tokens"
                )

        gr.Markdown("### 🧠 Advanced Model Architecture Settings (Expert Only!)")
        gr.Markdown("⚠️ **WARNING**: These settings directly affect model internals. Only change if you understand the architecture!")

        with gr.Row():
            with gr.Column():
                semantic_layer = gr.Slider(
                    label="Semantic Feature Extraction Layer",
                    value=17,
                    minimum=1,
                    maximum=24,
                    step=1,
                    info="Which layer of the semantic model to use. Higher layers (15-20) = more expressive, emotion-aware speech. Lower layers (5-12) = clearer pronunciation. Default 17 balances both."
                )
            with gr.Column():
                cfm_cache_length = gr.Slider(
                    label="CFM Max Cache Sequence Length",
                    value=8192,
                    minimum=1024,
                    maximum=16384,
                    step=512,
                    info="Memory allocation for processing speech. Higher (12000-16000) = handles longer segments better but uses more VRAM. Lower (4000-8000) = less memory usage. Default 8192 works for most."
                )

        gr.Markdown("### 🎛️ Custom Emotion Bias Weights")
        gr.Markdown("Fine-tune individual emotion channel biases in normalize_emo_vec() when Apply Emotion Bias is enabled:")
        with gr.Row():
            emo_bias_joy = gr.Slider(label="Joy Bias", value=0.9375, minimum=0.5, maximum=1.5, step=0.0625,
                                     info="Adjusts how much joy/happiness is expressed. <1.0 = less joyful, >1.0 = more joyful")
            emo_bias_anger = gr.Slider(label="Anger Bias", value=0.875, minimum=0.5, maximum=1.5, step=0.0625,
                                       info="Adjusts anger intensity. <1.0 = less angry, >1.0 = more angry")
            emo_bias_sad = gr.Slider(label="Sadness Bias", value=1.0, minimum=0.5, maximum=1.5, step=0.0625,
                                     info="Adjusts sadness expression. <1.0 = less sad, >1.0 = more sad")
            emo_bias_fear = gr.Slider(label="Fear Bias", value=1.0, minimum=0.5, maximum=1.5, step=0.0625,
                                      info="Adjusts fear/anxiety expression. <1.0 = less fearful, >1.0 = more fearful")
        with gr.Row():
            emo_bias_disgust = gr.Slider(label="Disgust Bias", value=0.9375, minimum=0.5, maximum=1.5, step=0.0625,
                                         info="Adjusts disgust expression. <1.0 = less disgusted, >1.0 = more disgusted")
            emo_bias_depression = gr.Slider(label="Depression Bias", value=0.9375, minimum=0.5, maximum=1.5, step=0.0625,
                                           info="Adjusts melancholic/depressed tone. <1.0 = less depressed, >1.0 = more depressed")
            emo_bias_surprise = gr.Slider(label="Surprise Bias", value=0.6875, minimum=0.5, maximum=1.5, step=0.0625,
                                          info="Adjusts surprise/amazement expression. <1.0 = less surprised, >1.0 = more surprised")
            emo_bias_calm = gr.Slider(label="Calm Bias", value=0.5625, minimum=0.5, maximum=1.5, step=0.0625,
                                      info="Adjusts calm/neutral tone. <1.0 = less calm, >1.0 = more calm and peaceful")

        # Define parameter lists for function calls
        advanced_params = [
            do_sample, top_p, top_k, temperature,
            length_penalty, num_beams, repetition_penalty, max_mel_tokens,
            low_memory_mode, prevent_vram_accumulation,
        ]

        expert_params = [
            diffusion_steps, inference_cfg_rate, interval_silence,
            max_speaker_audio_length, max_emotion_audio_length,
            autoregressive_batch_size, apply_emo_bias, max_emotion_sum,
            latent_multiplier, max_consecutive_silence, mp3_bitrate
        ]

        model_params = [semantic_layer, cfm_cache_length,
                       emo_bias_joy, emo_bias_anger, emo_bias_sad, emo_bias_fear,
                       emo_bias_disgust, emo_bias_depression, emo_bias_surprise, emo_bias_calm]

        _CONFIG_FIELDS = [
            {"section": "audio_generation", "key": "autoregressive_batch_size", "component": autoregressive_batch_size, "default": 1, "kind": "int", "min": 1, "max": 8},
            {"section": "audio_generation", "key": "output_filename", "component": output_filename, "default": "", "kind": "str"},
            {"section": "audio_generation", "key": "save_used_audio", "component": save_used_audio, "default": False, "kind": "bool"},
            {"section": "audio_generation", "key": "use_subprocess_system", "component": use_subprocess_system, "default": True, "kind": "bool"},
            {"section": "audio_generation", "key": "emo_control_method", "component": emo_control_method, "default": 0, "kind": "emotion_method"},
            {"section": "audio_generation", "key": "emo_random", "component": emo_random, "default": False, "kind": "bool"},
            {"section": "audio_generation", "key": "vec1", "component": vec1, "default": 0.0, "kind": "float", "min": 0.0, "max": 1.0},
            {"section": "audio_generation", "key": "vec2", "component": vec2, "default": 0.0, "kind": "float", "min": 0.0, "max": 1.0},
            {"section": "audio_generation", "key": "vec3", "component": vec3, "default": 0.0, "kind": "float", "min": 0.0, "max": 1.0},
            {"section": "audio_generation", "key": "vec4", "component": vec4, "default": 0.0, "kind": "float", "min": 0.0, "max": 1.0},
            {"section": "audio_generation", "key": "vec5", "component": vec5, "default": 0.0, "kind": "float", "min": 0.0, "max": 1.0},
            {"section": "audio_generation", "key": "vec6", "component": vec6, "default": 0.0, "kind": "float", "min": 0.0, "max": 1.0},
            {"section": "audio_generation", "key": "vec7", "component": vec7, "default": 0.0, "kind": "float", "min": 0.0, "max": 1.0},
            {"section": "audio_generation", "key": "vec8", "component": vec8, "default": 0.0, "kind": "float", "min": 0.0, "max": 1.0},
            {"section": "audio_generation", "key": "emo_text", "component": emo_text, "default": "", "kind": "str"},
            {"section": "audio_generation", "key": "emo_weight", "component": emo_weight, "default": 0.65, "kind": "float", "min": 0.0, "max": 1.0},
            {"section": "audio_generation", "key": "diffusion_steps", "component": diffusion_steps, "default": 25, "kind": "int", "min": 10, "max": 100},
            {"section": "audio_generation", "key": "inference_cfg_rate", "component": inference_cfg_rate, "default": 0.7, "kind": "float", "min": 0.0, "max": 2.0},
            {"section": "audio_generation", "key": "max_speaker_audio_length", "component": max_speaker_audio_length, "default": 30, "kind": "int", "min": 3, "max": 90},
            {"section": "audio_generation", "key": "max_emotion_audio_length", "component": max_emotion_audio_length, "default": 30, "kind": "int", "min": 3, "max": 90},
            {"section": "audio_generation", "key": "do_sample", "component": do_sample, "default": True, "kind": "bool"},
            {"section": "audio_generation", "key": "temperature", "component": temperature, "default": 0.8, "kind": "float", "min": 0.1, "max": 2.0},
            {"section": "audio_generation", "key": "num_beams", "component": num_beams, "default": 3, "kind": "int", "min": 1, "max": 10},
            {
                "section": "audio_generation",
                "key": "max_text_tokens_per_segment",
                "component": max_text_tokens_per_segment,
                "default": str(initial_value),
                "kind": "int_text",
                "min": 20,
                "max": PREVIEW_MAX_TEXT_TOKENS,
            },
            {"section": "audio_generation", "key": "save_as_mp3", "component": save_as_mp3, "default": False, "kind": "bool"},
            {"section": "audio_generation", "key": "low_memory_mode", "component": low_memory_mode, "default": False, "kind": "bool"},
            {"section": "audio_generation", "key": "prevent_vram_accumulation", "component": prevent_vram_accumulation, "default": False, "kind": "bool"},
            {
                "section": "advanced_parameters",
                "key": "mp3_bitrate",
                "component": mp3_bitrate,
                "default": "256k",
                "kind": "choice",
                "choices": ["128k", "192k", "256k", "320k"],
            },
            {"section": "advanced_parameters", "key": "latent_multiplier", "component": latent_multiplier, "default": 1.72, "kind": "float", "min": 1.0, "max": 3.0},
            {"section": "advanced_parameters", "key": "top_p", "component": top_p, "default": 0.8, "kind": "float", "min": 0.0, "max": 1.0},
            {"section": "advanced_parameters", "key": "top_k", "component": top_k, "default": 30, "kind": "int", "min": 0, "max": 100},
            {"section": "advanced_parameters", "key": "repetition_penalty", "component": repetition_penalty, "default": 10.0, "kind": "float", "min": 1.0, "max": 20.0},
            {"section": "advanced_parameters", "key": "length_penalty", "component": length_penalty, "default": 0.0, "kind": "float", "min": -2.0, "max": 2.0},
            {"section": "advanced_parameters", "key": "max_consecutive_silence", "component": max_consecutive_silence, "default": 0, "kind": "int", "min": 0, "max": 100},
            {"section": "advanced_parameters", "key": "interval_silence", "component": interval_silence, "default": 200, "kind": "int", "min": 0, "max": 1000},
            {"section": "advanced_parameters", "key": "apply_emo_bias", "component": apply_emo_bias, "default": True, "kind": "bool"},
            {"section": "advanced_parameters", "key": "max_emotion_sum", "component": max_emotion_sum, "default": 0.8, "kind": "float", "min": 0.1, "max": 2.0},
            {"section": "advanced_parameters", "key": "max_mel_tokens", "component": max_mel_tokens, "default": 1500, "kind": "int", "min": 50, "max": 1815},
            {"section": "advanced_parameters", "key": "semantic_layer", "component": semantic_layer, "default": 17, "kind": "int", "min": 1, "max": 24},
            {"section": "advanced_parameters", "key": "cfm_cache_length", "component": cfm_cache_length, "default": 8192, "kind": "int", "min": 1024, "max": 16384},
            {"section": "advanced_parameters", "key": "emo_bias_joy", "component": emo_bias_joy, "default": 0.9375, "kind": "float", "min": 0.5, "max": 1.5},
            {"section": "advanced_parameters", "key": "emo_bias_anger", "component": emo_bias_anger, "default": 0.875, "kind": "float", "min": 0.5, "max": 1.5},
            {"section": "advanced_parameters", "key": "emo_bias_sad", "component": emo_bias_sad, "default": 1.0, "kind": "float", "min": 0.5, "max": 1.5},
            {"section": "advanced_parameters", "key": "emo_bias_fear", "component": emo_bias_fear, "default": 1.0, "kind": "float", "min": 0.5, "max": 1.5},
            {"section": "advanced_parameters", "key": "emo_bias_disgust", "component": emo_bias_disgust, "default": 0.9375, "kind": "float", "min": 0.5, "max": 1.5},
            {"section": "advanced_parameters", "key": "emo_bias_depression", "component": emo_bias_depression, "default": 0.9375, "kind": "float", "min": 0.5, "max": 1.5},
            {"section": "advanced_parameters", "key": "emo_bias_surprise", "component": emo_bias_surprise, "default": 0.6875, "kind": "float", "min": 0.5, "max": 1.5},
            {"section": "advanced_parameters", "key": "emo_bias_calm", "component": emo_bias_calm, "default": 0.5625, "kind": "float", "min": 0.5, "max": 1.5},
        ]
        _CONFIG_COMPONENTS = [field["component"] for field in _CONFIG_FIELDS]
        _CONFIG_SECTIONS = tuple(dict.fromkeys(field["section"] for field in _CONFIG_FIELDS))
        _PRESET_AUX_OUTPUTS = [
            emotion_reference_group,
            emotion_randomize_group,
            emotion_vector_group,
            emo_text_group,
            emo_weight_group,
            segments_preview,
            section_count_label,
            subtitle_status,
        ]

        def _default_ui_config() -> Dict[str, Any]:
            cfg: Dict[str, Any] = {
                "_meta": {
                    "version": UI_PRESET_VERSION,
                    "format": UI_PRESET_FORMAT,
                }
            }
            for section in _CONFIG_SECTIONS:
                cfg[section] = {}
            for field in _CONFIG_FIELDS:
                cfg[field["section"]][field["key"]] = field["default"]
            return cfg

        def _merge_ui_config(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            merged = _default_ui_config()
            if not isinstance(cfg, dict):
                return merged

            meta = cfg.get("_meta")
            if isinstance(meta, dict):
                merged["_meta"].update(meta)

            for section in _CONFIG_SECTIONS:
                section_data = cfg.get(section)
                if isinstance(section_data, dict):
                    merged[section].update(section_data)

            return merged

        def _normalize_bool(value: Any, default: bool) -> bool:
            if isinstance(value, bool):
                return value
            if value is None:
                return bool(default)
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"1", "true", "yes", "y", "on"}:
                    return True
                if normalized in {"0", "false", "no", "n", "off", ""}:
                    return False
            return bool(default)

        def _normalize_int(value: Any, default: int, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
            try:
                normalized = int(float(value))
            except Exception:
                normalized = int(default)
            if min_value is not None:
                normalized = max(min_value, normalized)
            if max_value is not None:
                normalized = min(max_value, normalized)
            return normalized

        def _normalize_float(value: Any, default: float, min_value: Optional[float] = None, max_value: Optional[float] = None) -> float:
            try:
                normalized = float(value)
            except Exception:
                normalized = float(default)
            if min_value is not None:
                normalized = max(min_value, normalized)
            if max_value is not None:
                normalized = min(max_value, normalized)
            return normalized

        def _normalize_text(value: Any, default: str) -> str:
            if value is None:
                return str(default)
            return str(value)

        def _normalize_emotion_method(value: Any, default: int = 0) -> int:
            if hasattr(value, "value"):
                value = value.value
            if isinstance(value, str):
                stripped = value.strip()
                if stripped in EMO_CHOICES_ALL:
                    return EMO_CHOICES_ALL.index(stripped)
            normalized = _normalize_int(value, default, 0, len(EMO_CHOICES_ALL) - 1)
            return normalized

        def _normalize_field_value(field: Dict[str, Any], value: Any) -> Any:
            kind = field["kind"]
            default = field["default"]
            min_value = field.get("min")
            max_value = field.get("max")

            if kind == "str":
                return _normalize_text(value, default)
            if kind == "bool":
                return _normalize_bool(value, default)
            if kind == "int":
                return _normalize_int(value, default, min_value, max_value)
            if kind == "float":
                return _normalize_float(value, default, min_value, max_value)
            if kind == "int_text":
                return str(_normalize_int(value, int(default), min_value, max_value))
            if kind == "choice":
                normalized = _normalize_text(value, default)
                return normalized if normalized in field.get("choices", []) else default
            if kind == "emotion_method":
                return _normalize_emotion_method(value, default)
            return value if value is not None else default

        def _normalize_ui_config(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            merged = _merge_ui_config(cfg)
            for field in _CONFIG_FIELDS:
                section = field["section"]
                key = field["key"]
                merged[section][key] = _normalize_field_value(field, merged[section].get(key))
            return merged

        def _values_to_ui_config(*values: Any) -> Dict[str, Any]:
            cfg = _default_ui_config()
            for field, value in zip(_CONFIG_FIELDS, values):
                cfg[field["section"]][field["key"]] = _normalize_field_value(field, value)
            return cfg

        def _ui_config_to_values(cfg: Optional[Dict[str, Any]]) -> List[Any]:
            normalized = _normalize_ui_config(cfg)
            return [normalized[field["section"]][field["key"]] for field in _CONFIG_FIELDS]

        def _component_output_value(field: Dict[str, Any], value: Any) -> Any:
            if field["kind"] == "emotion_method":
                index_value = _normalize_emotion_method(value, field["default"])
                return EMO_CHOICES_ALL[index_value]
            return value

        def _build_subtitle_status_for_preset(subtitle_mode_value: bool, current_subtitle_file: Optional[str]):
            if not subtitle_mode_value or not current_subtitle_file:
                return gr.update(value="", visible=False)
            try:
                cues = parse_subtitle_file(current_subtitle_file)
                return gr.update(
                    value=build_subtitle_status_message(cues, subtitle_file=current_subtitle_file),
                    visible=True,
                )
            except Exception as e:
                return gr.update(value=f"Failed to load caption file: {str(e)}", visible=True)

        def _preset_component_updates(
            cfg: Optional[Dict[str, Any]],
            current_text: str,
            current_subtitle_mode: bool,
            current_subtitle_file: Optional[str],
        ) -> List[Any]:
            normalized = _normalize_ui_config(cfg)
            values = [
                _component_output_value(field, normalized[field["section"]][field["key"]])
                for field in _CONFIG_FIELDS
            ]
            text_value = current_text
            subtitle_mode_value = bool(current_subtitle_mode)
            max_tokens_value = normalized["audio_generation"]["max_text_tokens_per_segment"]
            preview_rows = get_preview_rows(
                text_value,
                max_tokens_value,
                subtitle_mode_value,
                current_subtitle_file,
            )
            section_count = build_section_count_message(
                text_value,
                max_tokens_value,
                subtitle_mode_value,
                current_subtitle_file,
            )
            return values + list(on_method_change(normalized["audio_generation"]["emo_control_method"])) + [
                gr.update(value=preview_rows, visible=True, type="array"),
                gr.update(value=section_count),
                _build_subtitle_status_for_preset(subtitle_mode_value, current_subtitle_file),
            ]

        def _save_preset_ui(preset_name: str, *values: Any):
            try:
                saved = _save_ui_preset(preset_name, _values_to_ui_config(*values))
                return (
                    gr.update(choices=_list_ui_presets(), value=saved),
                    gr.update(value=saved),
                    f"✅ Saved preset **{saved}**",
                )
            except Exception as e:
                return (
                    gr.update(choices=_list_ui_presets()),
                    gr.update(),
                    f"[ERROR] Save failed: {e}",
                )

        def _load_preset_ui(
            preset_name: str,
            current_text: str,
            current_subtitle_mode: bool,
            current_subtitle_file: Optional[str],
        ):
            requested = (preset_name or "").strip()
            if not requested or requested == DEFAULT_UI_PRESET_NAME:
                _set_last_used_ui_preset(DEFAULT_UI_PRESET_NAME)
                return (
                    *_preset_component_updates(
                        _default_ui_config(),
                        current_text,
                        current_subtitle_mode,
                        current_subtitle_file,
                    ),
                    "INFO: Loaded default settings.",
                )

            cfg = _load_ui_preset(requested)
            if not cfg:
                _set_last_used_ui_preset(DEFAULT_UI_PRESET_NAME)
                return (
                    *_preset_component_updates(
                        _default_ui_config(),
                        current_text,
                        current_subtitle_mode,
                        current_subtitle_file,
                    ),
                    f"WARNING: Preset **{requested}** not found (loaded defaults).",
                )

            return (
                *_preset_component_updates(
                    cfg,
                    current_text,
                    current_subtitle_mode,
                    current_subtitle_file,
                ),
                f"✅ Loaded preset **{requested}**",
            )

        def _reset_defaults_ui(current_text: str, current_subtitle_mode: bool, current_subtitle_file: Optional[str]):
            _set_last_used_ui_preset(DEFAULT_UI_PRESET_NAME)
            return (
                gr.update(choices=_list_ui_presets(), value=DEFAULT_UI_PRESET_NAME),
                *_preset_component_updates(
                    _default_ui_config(),
                    current_text,
                    current_subtitle_mode,
                    current_subtitle_file,
                ),
                "✅ Reset to defaults",
            )

        def _delete_preset_ui(
            preset_name: str,
            current_text: str,
            current_subtitle_mode: bool,
            current_subtitle_file: Optional[str],
        ):
            requested = (preset_name or "").strip()
            if not requested or requested == DEFAULT_UI_PRESET_NAME:
                _set_last_used_ui_preset(DEFAULT_UI_PRESET_NAME)
                return (
                    gr.update(choices=_list_ui_presets(), value=DEFAULT_UI_PRESET_NAME),
                    *_preset_component_updates(
                        _default_ui_config(),
                        current_text,
                        current_subtitle_mode,
                        current_subtitle_file,
                    ),
                    f"INFO: Built-in preset **{DEFAULT_UI_PRESET_NAME}** cannot be deleted",
                )

            ok = _delete_ui_preset(requested)
            _set_last_used_ui_preset(DEFAULT_UI_PRESET_NAME)
            return (
                gr.update(choices=_list_ui_presets(), value=DEFAULT_UI_PRESET_NAME),
                *_preset_component_updates(
                    _default_ui_config(),
                    current_text,
                    current_subtitle_mode,
                    current_subtitle_file,
                ),
                f"✅ Deleted preset **{requested}**" if ok else f"WARNING: Could not delete preset **{requested}**",
            )

    def process_media_to_reference(media_path, time_ranges="", require_time_ranges=False):
        if not media_path:
            if require_time_ranges:
                return None, "Upload an audio or video file first."
            return None, ""

        try:
            temp_audio = tempfile.mktemp(suffix=".wav")
            extracted_audio = extract_audio_from_media(media_path, temp_audio)
            if not extracted_audio:
                return None, f"Failed to read audio from {os.path.basename(media_path)}."

            has_ranges = bool(time_ranges and time_ranges.strip())
            if has_ranges:
                segments_audio = extract_time_ranges(extracted_audio, time_ranges)
                if segments_audio:
                    if os.path.exists(extracted_audio):
                        os.remove(extracted_audio)
                    extracted_audio = segments_audio
                    return (
                        extracted_audio,
                        f"Loaded extracted reference audio from {os.path.basename(media_path)} using ranges: {time_ranges.strip()}."
                    )
                if os.path.exists(extracted_audio):
                    os.remove(extracted_audio)
                return None, "No valid time ranges were found. Use a format like 1:3; 3:7; 11:15."

            if require_time_ranges:
                if os.path.exists(extracted_audio):
                    os.remove(extracted_audio)
                return None, "Enter time ranges like 1:3; 3:7 before extracting segments."

            return extracted_audio, f"Loaded reference audio from {os.path.basename(media_path)}."
        except Exception as e:
            print(f"Error processing media: {e}")
            return None, f"Error while processing media: {str(e)}"

    def process_media_upload(media_file, time_ranges):
        """Process uploaded media file and extract audio."""
        extracted_audio, status = process_media_to_reference(media_file, time_ranges, require_time_ranges=False)
        if not extracted_audio:
            if not status:
                return gr.update(), gr.update(value="", visible=False)
            return gr.update(), gr.update(value=status, visible=True)
        return gr.update(value=extracted_audio), gr.update(value=status, visible=True)

    def extract_audio_segments(media_file, time_ranges):
        """Extract specific time segments from uploaded media."""
        extracted_audio, status = process_media_to_reference(media_file, time_ranges, require_time_ranges=True)
        if not extracted_audio:
            return gr.update(), gr.update(value=status, visible=True)
        return gr.update(value=extracted_audio), gr.update(value=status, visible=True)

    def clear_reference_audio():
        """Clear the merged reference-media inputs."""
        return (
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=""),
            gr.update(value="", visible=False),
        )

    def load_audio_from_path_ui(audio_path, time_ranges):
        """Load audio from the specified file path."""
        if not audio_path:
            return gr.update(), gr.update(value="Please enter a file path", visible=True)

        audio_path = audio_path.strip()
        if not os.path.exists(audio_path):
            return gr.update(), gr.update(value=f"File not found: {audio_path}", visible=True)

        extracted_audio, status = process_media_to_reference(audio_path, time_ranges, require_time_ranges=False)
        if extracted_audio:
            return gr.update(value=extracted_audio), gr.update(value=status, visible=True)
        return gr.update(), gr.update(value=status or "Failed to load audio file", visible=True)

    def load_subtitle_file(subtitle_file_path, current_text, subtitle_mode, max_text_tokens_per_segment):
        if not subtitle_file_path:
            preview_rows = get_preview_rows(current_text, max_text_tokens_per_segment, False, None)
            section_count = build_section_count_message(current_text, max_text_tokens_per_segment, False, None)
            return (
                current_text,
                gr.update(value=False),
                gr.update(value="", visible=False),
                gr.update(value=preview_rows, visible=True, type="array"),
                gr.update(value=section_count),
            )

        try:
            cues = parse_subtitle_file(subtitle_file_path)
            subtitle_text = subtitle_cues_to_text(cues)
            use_subtitle_timing = bool(subtitle_mode)
            preview_rows = get_preview_rows(
                subtitle_text,
                max_text_tokens_per_segment,
                use_subtitle_timing,
                subtitle_file_path,
            )
            section_count = build_section_count_message(
                subtitle_text,
                max_text_tokens_per_segment,
                use_subtitle_timing,
                subtitle_file_path,
            )
            return (
                subtitle_text,
                gr.update(value=use_subtitle_timing),
                gr.update(value=build_subtitle_status_message(cues, subtitle_file=subtitle_file_path), visible=True),
                gr.update(value=preview_rows, visible=True, type="array"),
                gr.update(value=section_count),
            )
        except Exception as e:
            preview_rows = [[0, "Caption Error", str(e), ""]]
            return (
                current_text,
                gr.update(value=False),
                gr.update(value=f"Failed to load caption file: {str(e)}", visible=True),
                gr.update(value=preview_rows, visible=True, type="array"),
                gr.update(value=f"**Current Sections:** Unable to read subtitle file: {html.escape(str(e))}"),
            )

    def on_segmentation_inputs_change(text, max_text_tokens_per_segment, subtitle_mode, subtitle_file_path):
        data = get_preview_rows(text, max_text_tokens_per_segment, subtitle_mode, subtitle_file_path)
        section_count = build_section_count_message(text, max_text_tokens_per_segment, subtitle_mode, subtitle_file_path)
        return (
            gr.update(value=data, visible=True, type="array"),
            gr.update(value=section_count),
        )

    def on_method_change(emo_control_method):
        if emo_control_method == 1:  # emotion reference audio
            return (gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True)
                    )
        elif emo_control_method == 2:  # emotion vectors
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True)
                    )
        elif emo_control_method == 3:  # emotion text description
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True)
                    )
        else:  # 0: same as speaker voice
            return (gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                    )

    emo_control_method.change(on_method_change,
        inputs=[emo_control_method],
        outputs=[emotion_reference_group,
                 emotion_randomize_group,
                 emotion_vector_group,
                 emo_text_group,
                 emo_weight_group]
    )


    section_count_refresh_signal.change(
        on_segmentation_inputs_change,
        inputs=[input_text_single, max_text_tokens_per_segment, subtitle_mode, subtitle_file],
        outputs=[segments_preview, section_count_label],
        queue=False,
        show_progress="hidden"
    )

    subtitle_mode.change(
        on_segmentation_inputs_change,
        inputs=[input_text_single, max_text_tokens_per_segment, subtitle_mode, subtitle_file],
        outputs=[segments_preview, section_count_label]
    )

    subtitle_file.change(
        load_subtitle_file,
        inputs=[subtitle_file, input_text_single, subtitle_mode, max_text_tokens_per_segment],
        outputs=[input_text_single, subtitle_mode, subtitle_status, segments_preview, section_count_label]
    )

    prompt_audio.upload(update_prompt_audio,
                         inputs=[],
                         outputs=[gen_button],
                         queue=False,
                         show_progress="hidden")

    media_upload.upload(
        process_media_upload,
        inputs=[media_upload, time_ranges_input],
        outputs=[prompt_audio, reference_status],
        queue=False,
        show_progress="hidden",
        trigger_mode="always_last"
    )

    extract_button.click(
        extract_audio_segments,
        inputs=[media_upload, time_ranges_input],
        outputs=[prompt_audio, reference_status],
        queue=False,
        show_progress="hidden"
    )

    prompt_audio.clear(
        clear_reference_audio,
        outputs=[media_upload, prompt_audio, audio_path_input, reference_status],
        queue=False,
        show_progress="hidden"
    )

    load_audio_button.click(
        load_audio_from_path_ui,
        inputs=[audio_path_input, time_ranges_input],
        outputs=[prompt_audio, reference_status],
        queue=False,
        show_progress="hidden"
    )

    use_subprocess_system.change(
        fn=on_subprocess_mode_change,
        inputs=[use_subprocess_system],
        queue=False,
        show_progress="hidden",
    )

    gen_button.click(gen_single,
                     inputs=[emo_control_method,prompt_audio, input_text_single, subtitle_mode, subtitle_file, save_used_audio, output_filename, emo_upload, emo_weight,
                             vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                              emo_text,emo_random,
                              max_text_tokens_per_segment,
                              save_as_mp3,
                              *expert_params,
                              *advanced_params,
                              *model_params,
                              use_subprocess_system,
                      ],
                      outputs=[output_audio, subtitle_status])

    open_outputs_button.click(open_outputs_folder)

    cancel_process_button.click(
        fn=cancel_generation_process,
        inputs=[use_subprocess_system, cancel_confirm_signal],
        outputs=[cancel_process_status],
        queue=False,
        show_progress="hidden",
        js="""
        (use_subprocess_system, _signal) => [
            use_subprocess_system,
            window.confirm("Cancel the running generation subprocess?")
        ]
        """,
    )

    ui_preset_save_btn.click(
        fn=_save_preset_ui,
        inputs=[ui_preset_name] + _CONFIG_COMPONENTS,
        outputs=[ui_preset_dropdown, ui_preset_name, ui_preset_status],
        queue=False,
        show_progress="hidden",
    )
    ui_preset_load_btn.click(
        fn=_load_preset_ui,
        inputs=[ui_preset_dropdown, input_text_single, subtitle_mode, subtitle_file],
        outputs=_CONFIG_COMPONENTS + _PRESET_AUX_OUTPUTS + [ui_preset_status],
        queue=False,
        show_progress="hidden",
    )
    ui_preset_dropdown.change(
        fn=_load_preset_ui,
        inputs=[ui_preset_dropdown, input_text_single, subtitle_mode, subtitle_file],
        outputs=_CONFIG_COMPONENTS + _PRESET_AUX_OUTPUTS + [ui_preset_status],
        queue=False,
        show_progress="hidden",
    )
    ui_preset_reset_btn.click(
        fn=_reset_defaults_ui,
        inputs=[input_text_single, subtitle_mode, subtitle_file],
        outputs=[ui_preset_dropdown] + _CONFIG_COMPONENTS + _PRESET_AUX_OUTPUTS + [ui_preset_status],
        queue=False,
        show_progress="hidden",
    )
    ui_preset_delete_btn.click(
        fn=_delete_preset_ui,
        inputs=[ui_preset_dropdown, input_text_single, subtitle_mode, subtitle_file],
        outputs=[ui_preset_dropdown] + _CONFIG_COMPONENTS + _PRESET_AUX_OUTPUTS + [ui_preset_status],
        queue=False,
        show_progress="hidden",
    )
    demo.load(
        fn=_load_preset_ui,
        inputs=[ui_preset_dropdown, input_text_single, subtitle_mode, subtitle_file],
        outputs=_CONFIG_COMPONENTS + _PRESET_AUX_OUTPUTS + [ui_preset_status],
        queue=False,
        show_progress="hidden",
    )



if __name__ == "__main__":
    demo.queue(20)
    demo.launch(
        share=cmd_args.share,
        inbrowser=True,
        theme=theme,
        css=APP_CSS,
        head=APP_HEAD,
        favicon_path=APP_FAVICON_PATH,
    )
