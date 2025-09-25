import html
import json
import os
import sys
import threading
import time
import glob
import platform
import subprocess
import tempfile
import shutil

import warnings

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
from indextts.infer_v2 import IndexTTS2
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language="Auto")
MODE = 'local'
tts = IndexTTS2(model_dir=cmd_args.model_dir,
                cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
                use_fp16=cmd_args.fp16,
                use_deepspeed=cmd_args.deepspeed,
                use_cuda_kernel=cmd_args.cuda_kernel,
                )
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

MAX_LENGTH_TO_USE_SPEED = 70

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

def gen_single(emo_control_method,prompt, text, save_used_audio, output_filename,
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
               progress=gr.Progress()):
    # Generate output path with sequential numbering or use custom filename
    temp_wav_path = generate_output_path(filename=output_filename, save_as_mp3=False)  # Always generate WAV first
    output_path = temp_wav_path
    # set gradio progress
    tts.gr_progress = progress

    # Update the low memory mode setting
    tts.hybrid_model_device = bool(low_memory_mode)

    kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
        # "typical_sampling": bool(typical_sampling),
        # "typical_mass": float(typical_mass),
    }
    if type(emo_control_method) is not int:
        emo_control_method = emo_control_method.value
    if emo_control_method == 0:  # emotion from speaker
        emo_ref_path = None  # remove external reference audio
    if emo_control_method == 1:  # emotion from reference audio
        pass
    if emo_control_method == 2:  # emotion from custom vectors
        vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        # Use custom emotion biases if provided
        custom_emo_biases = [emo_bias_joy, emo_bias_anger, emo_bias_sad, emo_bias_fear,
                            emo_bias_disgust, emo_bias_depression, emo_bias_surprise, emo_bias_calm]
        vec = tts.normalize_emo_vec(vec, apply_bias=apply_emo_bias, max_emotion_sum=max_emotion_sum,
                                   custom_biases=custom_emo_biases if apply_emo_bias else None)
    else:
        # don't use the emotion vector inputs for the other modes
        vec = None

    if emo_text == "":
        # erase empty emotion descriptions; `infer()` will then automatically use the main prompt
        emo_text = None

    print(f"Emo control mode:{emo_control_method},weight:{emo_weight},vec:{vec}")

    # Ensure max_text_tokens_per_segment is within valid range
    # Handle all input types gracefully (now it's a string from Textbox)
    if not max_text_tokens_per_segment:
        max_tokens = 120
    else:
        try:
            # Convert string to int and clamp to valid range
            max_tokens = int(float(str(max_text_tokens_per_segment).strip()))
            max_tokens = max(20, min(max_tokens, tts.cfg.gpt.max_text_tokens))
        except (ValueError, TypeError):
            # Use default if conversion fails
            max_tokens = 120

    # Pass new parameters to infer
    output = tts.infer(spk_audio_prompt=prompt, text=text,
                       output_path=output_path,
                       emo_audio_prompt=emo_ref_path, emo_alpha=emo_weight,
                       emo_vector=vec,
                       use_emo_text=(emo_control_method==3), emo_text=emo_text,use_random=emo_random,
                       verbose=cmd_args.verbose,
                       max_text_tokens_per_segment=max_tokens,
                       interval_silence=int(interval_silence),
                       diffusion_steps=int(diffusion_steps),
                       inference_cfg_rate=float(inference_cfg_rate),
                       max_speaker_audio_length=float(max_speaker_audio_length),
                       max_emotion_audio_length=float(max_emotion_audio_length),
                       autoregressive_batch_size=int(autoregressive_batch_size),
                       max_emotion_sum=float(max_emotion_sum),
                       latent_multiplier=float(latent_multiplier),
                       max_consecutive_silence=int(max_consecutive_silence),
                       semantic_layer=int(semantic_layer),
                       cfm_cache_length=int(cfm_cache_length),
                       reset_beam_cache_per_segment=bool(prevent_vram_accumulation),
                       **kwargs)

    # Save used audio if requested
    if save_used_audio and prompt:
        try:
            # Extract base filename from output path
            base_name = os.path.basename(output).rsplit('.', 1)[0]
            used_audio_path = os.path.join("outputs/used_audios", f"{base_name}_reference.wav")
            shutil.copy2(prompt, used_audio_path)
            print(f"Saved used reference audio to: {used_audio_path}")
        except Exception as e:
            print(f"Error saving used audio: {e}")

    # Convert to MP3 if requested
    if save_as_mp3 and MP3_AVAILABLE:
        mp3_path = output.replace('.wav', '.mp3')
        output = convert_wav_to_mp3(output, mp3_path, bitrate=mp3_bitrate)

    return gr.update(value=output,visible=True)

def update_prompt_audio():
    update_button = gr.update(interactive=True)
    return update_button


theme = gr.themes.Soft()
theme.font = [gr.themes.GoogleFont("Inter"), "Tahoma", "ui-sans-serif", "system-ui", "sans-serif"]
with gr.Blocks(title="SECourses IndexTTS2 Premium App", theme=theme) as demo:
    mutex = threading.Lock()
    gr.Markdown("## SECourses Index TTS2 Premium App V3 : https://www.patreon.com/posts/139297407")

    with gr.Tab("Audio Generation"):
        with gr.Row():
            os.makedirs("prompts",exist_ok=True)

            # Left column for reference audio
            with gr.Column():
                # Video/Audio upload panel (initially visible)
                with gr.Group(visible=True) as media_upload_group:
                    media_upload = gr.File(
                        label="Upload Video/Audio File",
                        file_count="single",
                        file_types=[".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv",
                                    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".wma", ".aac", ".opus"],
                        type="filepath"
                    )

                    # Time range extraction section
                    with gr.Column():
                        time_ranges_input = gr.Textbox(
                            label="Extract Audio Segments (optional)",
                            placeholder="e.g., 1:3; 3:7; 11:15 (extracts and merges segments from 1-3s, 3-7s, 11-15s)",
                            value="",
                            info="Enter time ranges to extract specific parts of the audio"
                        )
                        extract_button = gr.Button("Extract and Use Audio", variant="secondary")

                # Speaker reference audio (initially hidden, shown after upload)
                prompt_audio = gr.Audio(
                    label="Speaker Reference Audio (3-90 seconds)",
                    key="prompt_audio",
                    sources=["upload","microphone"],
                    type="filepath",
                    visible=False
                )

                prompt_list = os.listdir("prompts")
                default = ''
                if prompt_list:
                    default = prompt_list[0]

            # Middle column for text input
            with gr.Column():
                input_text_single = gr.TextArea(
                    label="Text to Synthesize",
                    key="input_text_single",
                    placeholder="Enter the text you want to convert to speech",
                    info=f"Model v{tts.model_version or '1.0'} | Supports multiple languages. Long texts will be automatically segmented."
                )
                with gr.Row():
                    gen_button = gr.Button("Generate Speech", key="gen_button", interactive=True, variant="primary")
                    open_outputs_button = gr.Button("📁 Open Outputs Folder", key="open_outputs_button")

                # Output filename and save used audio options
                with gr.Row():
                    output_filename = gr.Textbox(
                        label="Output Filename (optional)",
                        placeholder="Leave empty for auto-numbering (e.g., 0001.wav)",
                        value=""
                    )
                    save_used_audio = gr.Checkbox(
                        label="Save Used Reference Audio",
                        value=False,
                        info="Save reference audio to outputs/used_audios"
                    )

            # Right column for output and load from path
            with gr.Column():
                output_audio = gr.Audio(
                    label="Generated Result (click to play/download)",
                    visible=True,
                    key="output_audio"
                )

                # Load audio from path section
                with gr.Group():
                    gr.Markdown("### Load Audio from Path")
                    with gr.Row():
                        audio_path_input = gr.Textbox(
                            label="Audio File Path",
                            placeholder="Enter full path to audio file",
                            value=""
                        )
                        load_audio_button = gr.Button("Load Audio", variant="secondary")
                    load_status = gr.Textbox(
                        label="Status",
                        value="",
                        interactive=False,
                        visible=False
                    )

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
                initial_value = max(20, min(tts.cfg.gpt.max_text_tokens, cmd_args.gui_seg_tokens))
                max_text_tokens_per_segment = gr.Textbox(
                    label="Max Tokens per Segment",
                    value=str(initial_value),
                    key="max_text_tokens_per_segment",
                    info=f"Splits long text into chunks for processing. Valid range: 20-{tts.cfg.gpt.max_text_tokens}. Smaller (80-120) = more natural pauses and consistent quality but slower. Larger (150-200) = faster but may have quality variations. Default: {initial_value}. Bigger value uses more VRAM."
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
                headers=["Index", "Segment Content", "Token Count"],
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
                autoregressive_batch_size = gr.Slider(
                    label="Autoregressive Batch Size",
                    value=1,
                    minimum=1,
                    maximum=8,
                    step=1,
                    info="Generates multiple speech variations simultaneously. Higher (3-8) = more options, potentially better quality but much slower. 1 = single fast generation."
                )
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

    def process_media_upload(media_file, time_ranges):
        """Process uploaded media file and extract audio."""
        if media_file is None:
            return gr.update(visible=True), gr.update(visible=False), gr.update(value=None)

        try:
            # Extract audio from media file
            temp_audio = tempfile.mktemp(suffix=".wav")
            extracted_audio = extract_audio_from_media(media_file, temp_audio)

            if not extracted_audio:
                return gr.update(visible=True), gr.update(visible=False), gr.update(value=None)

            # If time ranges specified, extract and merge segments
            if time_ranges and time_ranges.strip():
                segments_audio = extract_time_ranges(extracted_audio, time_ranges)
                if segments_audio:
                    os.remove(extracted_audio)
                    extracted_audio = segments_audio

            # Hide upload panel, show audio panel with extracted audio
            return (
                gr.update(visible=False),  # Hide media upload
                gr.update(visible=True, value=extracted_audio),  # Show and update prompt_audio
                gr.update(value=extracted_audio)  # Return path
            )
        except Exception as e:
            print(f"Error processing media: {e}")
            return gr.update(visible=True), gr.update(visible=False), gr.update(value=None)

    def extract_audio_segments(media_file, time_ranges):
        """Extract specific time segments from uploaded media."""
        if media_file is None:
            return gr.update(), gr.update(), gr.update()

        try:
            # First extract full audio
            temp_audio = tempfile.mktemp(suffix=".wav")
            extracted_audio = extract_audio_from_media(media_file, temp_audio)

            if not extracted_audio:
                return gr.update(), gr.update(), gr.update()

            # Extract and merge segments if specified
            if time_ranges and time_ranges.strip():
                segments_audio = extract_time_ranges(extracted_audio, time_ranges)
                if segments_audio:
                    os.remove(extracted_audio)
                    extracted_audio = segments_audio

            # Update audio component
            return (
                gr.update(visible=False),  # Hide upload panel
                gr.update(visible=True, value=extracted_audio),  # Update audio
                gr.update(value=extracted_audio)  # Return path
            )
        except Exception as e:
            print(f"Error extracting segments: {e}")
            return gr.update(), gr.update(), gr.update()

    def clear_reference_audio():
        """Clear reference audio and show upload panel again."""
        return gr.update(visible=True), gr.update(visible=False, value=None)

    def load_audio_from_path_ui(audio_path):
        """Load audio from the specified file path."""
        if not audio_path:
            return gr.update(value=None), gr.update(value="Please enter a file path", visible=True)

        audio_path = audio_path.strip()
        if not os.path.exists(audio_path):
            return gr.update(value=None), gr.update(value=f"File not found: {audio_path}", visible=True)

        try:
            # Convert to acceptable format if needed
            temp_audio = tempfile.mktemp(suffix=".wav")
            extracted_audio = extract_audio_from_media(audio_path, temp_audio)

            if extracted_audio:
                return (
                    gr.update(visible=False),  # Hide upload panel
                    gr.update(visible=True, value=extracted_audio),  # Update prompt audio
                    gr.update(value="Audio loaded successfully!", visible=True)  # Status
                )
            else:
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(value="Failed to load audio file", visible=True)
                )
        except Exception as e:
            return (
                gr.update(),
                gr.update(),
                gr.update(value=f"Error: {str(e)}", visible=True)
            )

    def on_input_text_change(text, max_text_tokens_per_segment):
        if text and len(text) > 0:
            # Ensure max_text_tokens_per_segment is within valid range
            # This prevents errors when users are typing values (now it's a string from Textbox)
            if not max_text_tokens_per_segment:
                max_tokens = 120
            else:
                try:
                    # Convert string to int and clamp to valid range
                    max_tokens = int(float(str(max_text_tokens_per_segment).strip()))
                    max_tokens = max(20, min(max_tokens, tts.cfg.gpt.max_text_tokens))
                except (ValueError, TypeError):
                    # Use default if conversion fails
                    max_tokens = 120

            text_tokens_list = tts.tokenizer.tokenize(text)

            segments = tts.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment=max_tokens)
            data = []
            for i, s in enumerate(segments):
                segment_str = ''.join(s)
                tokens_count = len(s)
                data.append([i, segment_str, tokens_count])
            return {
                segments_preview: gr.update(value=data, visible=True, type="array"),
            }
        else:
            # Return empty data with headers
            empty_data = []
            return {
                segments_preview: gr.update(value=empty_data, visible=True),
            }

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


    input_text_single.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview]
    )

    max_text_tokens_per_segment.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview]
    )

    prompt_audio.upload(update_prompt_audio,
                         inputs=[],
                         outputs=[gen_button])

    # New UI callbacks
    media_upload.change(
        process_media_upload,
        inputs=[media_upload, time_ranges_input],
        outputs=[media_upload_group, prompt_audio, prompt_audio]
    )

    extract_button.click(
        extract_audio_segments,
        inputs=[media_upload, time_ranges_input],
        outputs=[media_upload_group, prompt_audio, prompt_audio]
    )

    prompt_audio.clear(
        clear_reference_audio,
        outputs=[media_upload_group, prompt_audio]
    )

    load_audio_button.click(
        load_audio_from_path_ui,
        inputs=[audio_path_input],
        outputs=[media_upload_group, prompt_audio, load_status]
    )

    gen_button.click(gen_single,
                     inputs=[emo_control_method,prompt_audio, input_text_single, save_used_audio, output_filename, emo_upload, emo_weight,
                            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                             emo_text,emo_random,
                             max_text_tokens_per_segment,
                             save_as_mp3,
                             *expert_params,
                             *advanced_params,
                             *model_params,
                     ],
                     outputs=[output_audio])

    open_outputs_button.click(open_outputs_folder)



if __name__ == "__main__":
    demo.queue(20)
    demo.launch(
        share=cmd_args.share,
        inbrowser=True
    )
