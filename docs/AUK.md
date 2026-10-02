# AuK in the shared speech application

Select **AuK** beside the app heading. [Tencent AuK](https://huggingface.co/tencent/AuK) is a 1.5B flow-matching transformer conditioned on the frozen Thinker of Qwen2.5-Omni-3B, with a 24 kHz VAE. It clones voices, designs voices from a description, speaks a fine-tuned voice without a reference, and edits, restores and separates recordings. Model switching saves the current profile, restores AuK's and unloads the previous engine; weights load on first use. The Flash (distilled) variant is not used.

## Generation

- **Voice cloning:** one clean speaker. The reference is trimmed of edge silence and cut at a pause to at most **15 s** (longer references raised speaker similarity; 6 s lowered it clearly). The reference transcript only sets the speaking pace: leave it blank and Whisper transcribes the reference once (cached), or place a `.txt` beside the audio.
- **Voice design:** describe the voice in plain words, for example *a calm middle-aged man with a deep, warm voice*. English and Chinese only.
- **Auto voice:** a fine-tuned AuK voice speaks without a reference, with the description and pace it was trained with (stored beside the checkpoint). The base model picks a random voice.
- **Duration:** AuK needs every section's length up front. The app follows upstream's prompt-enhancer byte model, scaled by the reference's (or trained voice's) measured pace. Sections default to 60 text tokens (about 15 s); a section that would not fit the 30 s context is split again rather than rushed.
- **Sampling:** the official defaults are 32 Euler steps, guidance 2.0 and sway −1. The Max speed and Max quality choices select 16 and 64 steps. Loudness matching to the reference is on by default.

Decoding choices were compared on 24 held-out English sentences × 2 seeds, scored against real recordings of the same sentences (speaker and style similarity, Whisper word errors), with paired bootstrap intervals and a 72-sentence confirmation set.

| Setting | Result |
| --- | --- |
| Voice design wording | Upstream's canonical prompt-enhancer form gave 1.8 % word errors; a longer paraphrase gave 14 % (the model spoke parts of the description). The app uses the canonical form. |
| Reference length | 15 s: speaker similarity +0.006 to +0.009 over ~12 s, confirmed on new sentences; 6 s: −0.053. |
| Section length | 40–60 tokens best; 120 tokens +5.7 points word errors when the section was squeezed into the context. Sections now use the full 30 s context (upstream trains up to 30 s per side) and split when longer. |
| Steps | 16 slightly lower similarity; 24 = 32; 48 and 64 no gain. |
| Guidance | 1.5–2.5 equivalent; 3.0 lowered similarity on the confirmation set. |
| Sway, midpoint solver, duration factor, edge padding, trimming, loudness | No reliable gain over the defaults. |

## Editing tab

**AuK Audio Editing** appears while AuK is selected. Choose a task, fill its fields, and the instruction AuK receives is shown before running: content edits (replace, insert, remove words), lyrics, pitch, speed, volume, emotion, timbre, accent removal, nonverbal sounds, whisper conversion, enhancement, restoration, speaker and music separation, or a custom instruction. Output length follows the task's own rule (upstream's prompt enhancer), the source length, or a custom value. Content edits work on up to 30 seconds; acoustic tasks run on longer recordings piece by piece, cut at pauses. Community testing found description TTS, cloning, volume, enhancement and separation the most stable tasks and pitch, speed, accent, nonverbal and content edits less reliable; try another seed when an edit reproduces the source.

## Memory, precision and VRAM tiers

The app downloads AuK from [tencent/AuK](https://huggingface.co/tencent/AuK). The Qwen Thinker comes from [MonsterMMORPG/Wan_GGUF/AuK](https://huggingface.co/MonsterMMORPG/Wan_GGUF/tree/main/AuK): 7.5 GB with only the text model and audio tower, bitwise-identical hidden states to the 12 GB public snapshot, which is used when already present or as fallback. The same folder holds the optional ConvRot INT8 transformer (204 projections) and INT8 Thinker (252 projections), application-specific checkpoints converted by `tools/quantize_auk.py` with per-row MSE clipping and per-layer Hadamard group sizes.

INT8 measured within seed noise against BF16 (speaker similarity −0.001, word errors +0.1 to +0.3 points on 48 clips) and saves memory, not time: both models INT8 run 13–28 % slower. INT8 uses W8A16 kernels; cuBLASLt's INT8 GEMM faulted on some shapes with PyTorch 2.14 and CUDA 13. **On demand**, the text encoder and the transformer with the VAE take turns on the GPU, one loan each per request; audio is bit-identical, and the token table stays in CPU memory.

Whole-process peaks for a 34 s cloned request on an RTX A6000 (CUDA context included):

| Tier | Transformer / encoder / residency | Section batch | Peak GiB | Speed (RTF) |
| --- | --- | ---: | ---: | ---: |
| 16, 24, 32 GB | BF16 / BF16 / GPU | 8 | 12.0 | 0.34 |
| 12 GB | BF16 / INT8 / GPU | 4 | 8.7 | 0.35 |
| 10 GB | INT8 / INT8 / GPU | 4 | 7.3 | 0.37 |
| 8 GB | BF16 / INT8 / on demand | 2 | 5.1 | 0.45 |
| 6 GB | INT8 / INT8 / on demand | 1 | 4.6 | 0.45 |

Batching sections changes each take slightly and did not speed up cloning in these requests; long Auto-voice texts gained most (RTF 0.18 at batch 8 against 0.28 at batch 2). Whisper runs beside AuK only while 3 GB stay free, otherwise on the CPU.

## Training and selection

Training uses the prepared manifest dataset. AuK caches VAE posteriors and the frozen Qwen conditioning under `cache/auk` (reused across runs; tiers that train an INT8 base encode with the INT8 Thinker). The default objective teaches the voice to speak without a reference (Auto voice); the reference-prompt fraction mixes in cloning examples, encoding each reference live. Full fine-tuning (FP32 master weights, BF16 compute) needs the 32 GB tier; adapters (LoRA / DoRA on attention, feed-forward and adaLN projections, with the small input and output layers trained fully) fit every tier with an INT8 base below 16 GB. Micro-batches are measured in latent frames (50 per second) and accumulate to about 108 s of audio per update. A run writes `auk_voice.json` beside its checkpoints with the voice's pace, description and instruction wording.

Measured on an RTX A6000 at 2,400 frames per micro-batch: full fine-tuning peaks at 28.75 GiB with or without gradient checkpointing (the optimizer step dominates), 26.8 GiB with fused AdamW, and runs 65 % faster without it (2.41 updates/s). DoRA r32 needs 17.6 GiB without checkpointing (1.06/s) or 4.7 GiB with it (0.56/s); LoRA r32 10.9 GiB (1.72/s). Reference-prompt objectives load the Qwen encoder beside the model and need checkpointing.

Validation reports AuK's flow-matching loss on held-out clips with fixed noise at flow times 0.0–0.9; it is not comparable to the other models' losses and is not a speech-quality guarantee. Checkpoint evaluation, the speech comparison against real recordings and the checkpoint grid work as for the other models.

## Upstream and community findings

The port matches upstream's reference implementation: the BF16 transformer is bit-identical to upstream's FP32 storage with BF16 autocast, including the checkpoint's BF16-rounded rotary frequencies and FP32 normalization epsilon. The Thinker's 37 hidden states match upstream's encoder under Transformers 5.18. Torchcodec, `audioread` and `qwen_omni_utils` are not needed: audio is read with soundfile and resampled with soxr.

Useful community findings: the ComfyUI ports' Thinker-only encoder files and ConvRot layer selection (adopted); never keeping the encoder and the transformer resident together on small cards (adopted as on-demand residency); prompt-enhancer duration rules without an LLM (adopted); instruct-voice timbre depends strongly on the seed; only English and Chinese are supported. Claims from forks were checked on this machine before use.

## Licenses

AuK's code and weights are MIT-licensed. The Qwen2.5-Omni-3B Thinker, its slim copy and its INT8 conversion are under the **Qwen Research License** (research and non-commercial use; `models/quantized/AuK/qwen2_5_omni_thinker/LICENSE`). Personal training data and voices are never uploaded.
