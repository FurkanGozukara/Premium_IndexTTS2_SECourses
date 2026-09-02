# IndexTTS2 SECourses Premium Voice Cloning and Generation App - 1-Click to Install on Windows, RunPod and Massed Compute - Generate Entire Audiobooks With Consistent High Quality Voice

## 2 September 2026 V5 IndexTTS 2.5 Upgrade

- IndexTTS 2.5-only inference using the official multilingual model stack, with all 1.x and 2.0 paths removed.
- GPU VRAM presets for 6, 8, 10, 12, 16, 24, and 32 GB cards, including GPT block swapping for lower-VRAM GPUs.
- Optional INT8 ConvRot GPT model for memory-efficient generation while retaining the required 2.5 codec and vocoder models.
- LoRA / DoRA dataset preparation and training tabs, including subtitle-driven segmentation and training progress charts.
- Checkpoint Grid tab for listening to Base model (no LoRA / DoRA), the recommended checkpoint, final file, saved epochs, and strength variants with identical text, references, and seed.
- Automatic generalization analysis identifies the lowest validation-loss epoch, marks the sustained overfitting region, and makes Voice Generation prefer the recommended checkpoint.
- Optional validation-based early stopping and automatic measured checkpoint evaluation after the training model releases its memory.
- Evaluation can use inference-like references from a different clip of the same speaker, matching the normal voice-cloning workflow more closely.
- Measured quality-first training defaults use rank 128, alpha 129, learning rate 5e-5, 20 epochs, and inference-like conditioning (`speaker_ref_mode=other`, `emo_ref_mode=follow_speaker`, `val_reference_mode=other`).
- Per-voice speaking-rate calibration compares generated words/s with the training recordings and auto-applies the saved pace in Voice Generation; the Checkpoint Grid can calibrate older runs too.
- Every LoRA / DoRA keeps machine-readable and plain-language reports under `loras/<lora-or-dora>/analysis/`; listening grids are saved under `outputs/grids/`.
- The header's **🕘  Load last values** button restores the last run of every tab; earlier results stay hidden until it is clicked.
- Batch voice generation with reusable system and user presets.
- Live generation, download, dataset, and training progress with speed and ETA reporting.
- Cross-platform one-click installers and launchers for Windows, Massed Compute, RunPod, SimplePod, and other Linux hosts.
- Models & Performance tab: pick your GPU VRAM tier (6/8/10/12/16/24/32 GB); every tier was calibrated to keep about 2 GB of VRAM free. Lower tiers switch to the INT8 ConvRot GPT and stream GPT blocks from CPU RAM (block swap); the INT8 file is downloaded automatically from Hugging Face when needed.
- Presets: system presets live in `presets/system` (read-only, marked with a star) and your own presets in `presets/user`; a preset stores every parameter of every tab, and the last-used preset is restored at startup.
- LoRA Dataset Preparation: drop any audio/video files (with optional SRT/VTT/SBV subtitles); audio is extracted with FFmpeg, transcribed with Whisper (word timestamps), cut into sentence-aligned segments, loudness normalized, and cached as training features.
- LoRA / DoRA Training: rank/alpha/dropout, DoRA toggle, learning-rate schedule, gradient checkpointing, block swap for low VRAM, validation split, periodic audio samples, live loss/learning-rate charts, resume from an existing LoRA / DoRA, single `.safetensors` output that the Voice Generation tab loads (with a strength slider) from the LoRA / DoRA dropdown.
- Cancel/Stop buttons ask for confirmation and really stop the worker process; every long task reports progress, it/s, ETA, and VRAM in both the UI and the console.
- Resume training from any saved LoRA / DoRA: "Weights only" starts a fresh schedule from the LoRA / DoRA weights, "Continue run" restores the optimizer, scheduler and step counter and trains the extra epochs/steps you configure.
- Reloading starts with clean result panels; finished jobs and outputs are not restored automatically.
- Selecting the INT8 ConvRot model downloads it automatically on first use (with progress); if the download fails the app falls back to the BF16 model and tells you why.
- Tested on Windows 11 with an RTX 5090 (all seven VRAM tiers verified with GPU 0 otherwise idle), Python 3.12, PyTorch CUDA 13.0, transformers 5, Gradio 6.

## This app is made only for SECourses Patreon users : https://www.patreon.com/posts/139297407

### Download app from here > https://www.patreon.com/posts/139297407

## 1-Click Installers 

<img height="600" alt="image" src="https://github.com/user-attachments/assets/9d7b363c-aecf-4c7c-bd96-b336a599d4d8" />

## 11 May 2026 Update V4.1

- New feature provide image and generate video automatically implemented

- Just run installer bat file, zip file is still same, to update

<img  height="600" alt="image" src="https://github.com/user-attachments/assets/1ccb7828-5fce-44e6-ae8e-b8e472c000e6" />

<img  height="600" alt="image" src="https://github.com/user-attachments/assets/477f5510-8da1-418b-a1ac-a82c57be9bc7" />



## 5 April 2026 Update V4

- This is a massive update of the app so please make a fresh install

- Installers upgraded to uv thus now up to 100x faster

- - Tested on Windows, RunPod and Massed Compute

- - Linux users please use Massed Compute installers

- Now all models auto downloaded with our app

- How app has title on browser tabs and a custom favicon

- <img width="371" height="98" alt="image" src="https://github.com/user-attachments/assets/b3f77604-c5cf-402c-8287-c123a997ecce" />

- Interface completely upgraded

- <img height="600" alt="image" src="https://github.com/user-attachments/assets/7a3c9fa1-fcbd-4d14-9a8a-fdadaf9ae5ab" />

- Now supports caption / subtitle srt files upload and processing it automatically

- You can also enable Use Caption Cue Timing to generate exactly same timing as caption

- <img height="600" alt="image" src="https://github.com/user-attachments/assets/8f04f372-60e6-4595-a422-4e55371226f1" />

- This app works with reference audio file and now you can provide a video or an audio file or record audio from your microphone (recommended 15 seconds)

- <img height="600" alt="image" src="https://github.com/user-attachments/assets/c701ba3c-aa0f-4cd8-92fc-da260a20da76" />

- Now supports sub-process processing thus absolutely 0 VRAM or RAM usage after processing is done

- Now supports real batch-size processing thus the speed gain is immerse if fits into your VRAM (this is fully my custom implementation no one else has this feature)

- <img height="600" alt="image" src="https://github.com/user-attachments/assets/c43757da-c6f2-495f-b48d-ff98e6ba1be3" />

- Now supports full preset save / load and remember system

- Now supports cancel generation

- <img width="898" height="1480" alt="image" src="https://github.com/user-attachments/assets/5c5b7e61-d567-4fa3-b366-271b0d575b51" />

- Now when generating more information on the CMD will be shown like status, speed, ETA, how much left, etc.

- <img width="2098" height="435" alt="image" src="https://github.com/user-attachments/assets/c43c8683-804e-424b-8eff-1381e5f255a0" />

- Other features

- <img width="3531" height="1542" alt="image" src="https://github.com/user-attachments/assets/98341bc6-0c45-48b5-ad17-9ef43cb14880" />

- Advanced parameters

- <img width="3555" height="1633" alt="image" src="https://github.com/user-attachments/assets/4b909f90-6dd1-4ef4-8a00-7225f1a28390" />

- A tutorial video coming soon hopefully

- <img width="3568" height="1553" alt="image" src="https://github.com/user-attachments/assets/84ec9dbc-1aff-4b16-9d1e-fe0ce7bd1ca0" />







