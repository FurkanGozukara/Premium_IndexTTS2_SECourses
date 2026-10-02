# OmniVoice in the shared speech application

Select **OmniVoice** beside the app heading. Model switching saves the current profile, restores the chosen profile and unloads the previous engine. A job must finish or be canceled before switching. Legacy presets remain IndexTTS presets; universal presets retain both profiles. Model weights load on first use.

## Generation

- **Voice cloning:** use one clean speaker and an exact reference transcript. Leaving the transcript blank downloads automatic transcription on demand. A saved training reference can carry its transcript beside the WAV.
- **Auto voice:** generate without reference audio.
- **Consistency of a trained voice:** measured on 10 held-out sentences with 6 seeds each. Without a reference a full fine-tune came closest to the real recordings on average (speaker similarity 0.809, style 0.776, 4.6 % word errors) but varied more between takes than IndexTTS: the spread of speaker similarity was 0.032 against 0.021, of style similarity 0.050 against 0.025. Cloning a training clip of the voice was steadier than IndexTTS for every clip tried (spreads about 0.014 and 0.018), but the clip decided the similarity: four clips of the same speaker gave 0.725 to 0.817, which neither length nor typical pitch, pace or intonation predicted. **Position temperature** 0.5 instead of 5 (Advanced OmniVoice sampling) steadied takes without a reference by about a third (spreads 0.022 and 0.033) with no significant change in similarity (0.804) or word errors (5.3 %). **Reference audition** (voice panel) solves the clip choice by listening: it renders held-out sentences with each candidate clip and keeps the most speaker-like one; on the fine-tuned voice its winner reached 0.855 against 0.827 for the 15-second training reference. **Takes per section** removes the occasional slipped word. A trained voice keeps its calibrated pace when cloning: the clip sets timbre and style, not the speed (one brisk clip had made every take 21 % too short).
- **Voice design:** enter supported comma-separated voice tags, for example `male, middle-aged, moderate pitch, british accent`. The panel lists the model's supported tags; arbitrary descriptions are rejected before generation starts. Language, pace, text segmentation, pause tags, batch processing, subtitle timelines and WAV/MP3/MP4 export use the shared controls.
- **Sampling:** **Best quality**, the default, is the official 32 diffusion steps with guidance 2, which scored best on held-out sentences (word errors, speaker and style similarity); guidance 1.5 or 3.0, 64 steps, time shift 0.3 and class temperature 0.5 all scored lower. **Fast** selects 16 steps. Advanced sampling controls keep upstream defaults.
- **Pronunciation:** dictionary readings and hand-written `<word|PHONES>` annotations are converted to OmniVoice's own syntax before normalization: ARPAbet phones become bracketed CMU phones (`<DoRA|D AO1 . R AH0>` → `[D AO1 R AH0]`), Chinese pinyin readings become tone markers in place of the characters (`<行|XING2>` → `XING2`), kana and respellings replace the word. OmniVoice tags such as `[laughter]` and `[sigh]` pass through unchanged, and segmentation never splits a bracketed reading.
- **Memory:** choose the detected GPU tier and increase section batch size only after measuring the intended workload. The transformer and audio tokenizer stay resident; IndexTTS auxiliary models and block swapping do not apply. Output is 24 kHz.

## BF16 and ConvRot INT8

The application downloads the original public model from [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) and the optional converted transformer from [MonsterMMORPG/Wan_GGUF/OmniVoice](https://huggingface.co/MonsterMMORPG/Wan_GGUF/tree/main/OmniVoice).

The conversion quantizes 196 transformer projections using per-output-channel INT8 scales and grouped Hadamard rotation. It tests group sizes 16, 64 and 256 with three-stage MSE clipping, retaining embeddings, normalization, audio heads and the audio tokenizer in floating point. The checkpoint is application-specific, not GGUF. The conversion tool is `tools/quantize_omnivoice.py`.

A four-sentence warm voice-cloning check on an RTX 5090 at 32 steps measured 0.870 s mean BF16 synthesis versus 1.108 s INT8, with peak PyTorch allocation of 2.03 and 1.65 GiB. These exclude driver/context memory and are workload-specific: INT8 saved memory here, while BF16 was faster. Forced W8A8 was slower and less accurate than the automatic W8A16 choice for these shapes. The uploaded model folder includes the measured report and source revision.

The pretrained model card specifies **CC-BY-NC** for weights; converted weights retain those terms. Upstream application code has a separate Apache-2.0 license. Personal training assets are separate from the public quantized model.

## Training and selection

Use the existing prepared manifest dataset. OmniVoice audio tokens are cached separately under `cache/omnivoice`; IndexTTS caches remain independent. Training supports:

- **Full:** FP32 trainable model parameters with mixed-precision forward/backward computation. Use the BF16 base and checkpoint strength 1.0.
- **LoRA / DoRA:** transformer attention/MLP adapters, optionally with trainable audio embeddings and heads; BF16 or a frozen INT8 base.

Training GPU presets group clips into micro-batches of up to **4,096 tokens** (padded audio frames at 25 per second plus text) and accumulate two of them, the upstream recipe's 8,192 tokens (about four minutes of speech) per optimizer update. Tiers below 24 GB enable gradient checkpointing; every tier trains on the BF16 base. Full fine-tuning is the default from the 16 GB tier: learning rate 2e-5, audio embeddings and heads trained, the last three checkpoints kept (1.2 GB each). Smaller tiers default to rank-32 DoRA at learning rate 1e-4 with frozen embeddings and heads. Epochs 0, the OmniVoice default, sizes the run from the training audio: 25 epochs for 14 hours, about 65 for 2 hours, at most 100 (a 2-hour voice measured clearly better at 75 than at 25 epochs); early stopping and the best checkpoint guard the rest, samples render every fifth epoch, and the Training plan shows the resolved length and counts the token batches of the selected dataset. The audio prompt fraction defaults to 0, which teaches one speaker's voice to speak without a reference; set 0.3, the upstream value, for multi-speaker datasets that should keep cloning from a prompt.

Peak PyTorch allocations at 4,096 tokens with clips up to 16 seconds (30-step runs on an RTX 5090): full fine-tuning 19.5 GB at 3.1 updates per second, or 11.5 GB with checkpointing (2.2/s); rank-32 DoRA 17.1 GB (1.1/s), or 2.7 GB with checkpointing (0.7/s). A frozen ConvRot INT8 base lowers checkpointed DoRA to about 1.4 GB. Full fine-tuning is about three times faster per update than DoRA and measured better on a held-out voice comparison, so DoRA is the choice only where full fine-tuning does not fit. This is a bounded workload check, not a promise for longer clips or custom trainable modules; driver memory is additional.

After a full fine-tune, the best and the recommended checkpoint are also saved as INT8 ConvRot (`<checkpoint>.int8_convrot.safetensors`, unless **Save an INT8 ConvRot version after training** is off): the transformer in the public INT8 format plus the fine-tuned audio embeddings and heads in BF16. Selecting it in Voice Generation loads the fine-tuned model in INT8; switching to or from it reloads the model.

Batch size, accumulation, learning rate, warmup, validation cadence, checkpoint saving, early stopping, EMA and live progress share the existing training controls. Reference transcripts are saved with automatic samples. Continue mode restores optimizer/scheduler, RNG, data position and full-precision training parameters from the sidecar. Older sidecars retain their saved deployment precision and are reported as legacy continuation. Weights-only mode starts a fresh optimization run.

OmniVoice validation measures masked audio tokens with fixed masks at each checkpoint. Its loss is not comparable to IndexTTS next-token loss and is not a speech-quality guarantee. Checkpoint grids and optional speech evaluation compare candidates against Base. The shared decoding sweep varies OmniVoice diffusion steps and guidance on development recordings; a separate final-test dataset is assessed only after the deployment is frozen. IndexTTS's semantic-to-mel decoder adaptation is hidden because it does not apply to OmniVoice.

## Upstream and fork findings

Implementation was checked against [upstream revision 08be0b4](https://github.com/k2-fsa/OmniVoice/tree/08be0b4ccbac3e13e374e86fbfead4b4cac343e2). The fork inventory contained 2,118 unique forks; 49 distinct or active candidates were collected for source comparison. Useful findings included:

- [groxaxo/OmniVoice-Streaming](https://github.com/groxaxo/OmniVoice-Streaming): reference-prompt caching and attention-mask details. Prompt caching is used; the installed upstream already includes the needed bidirectional padding mask.
- [hangry-labs/OmniVoiceTTS](https://github.com/hangry-labs/OmniVoiceTTS): lazy ASR, cached voice prompts, serialized GPU work and workload-specific attention measurements. This app keeps ASR lazy and uses SDPA by default; FlashAttention is not assumed faster.
- [kawshikbuet17/OmniVoice-LoRA](https://github.com/kawshikbuet17/OmniVoice-LoRA) and [rikabi89/OmniVoiceGUI-Training](https://github.com/rikabi89/OmniVoiceGUI-Training): training workflows and checkpoint handling. The integration uses this application's existing adapters and monitoring rather than a second training UI.

Fork suggestions informed design; their performance or quality claims were not treated as measurements on this machine.

## Current dependencies

The installer updates ordinary requirements without version pins. SDPA and ConvRot do not require external FlashAttention, SageAttention, MSLK, TorchAO or xFormers binaries. The optional IndexTTS acceleration engine still requires FlashAttention compiled for the installed PyTorch ABI. After a PyTorch update, the installer removes custom optional wheels whose recorded build targets another PyTorch minor version, preventing eager dependency imports from breaking both models. Install a matching optional wheel to use that accelerator; do not reuse a 2.13 CUDA extension with PyTorch 2.14.

Startup prints module loading, panel construction and time to the ready URL. The normal BAT launcher opens Google Chrome; `--no-browser --port 7861` is available for supervised validation.
