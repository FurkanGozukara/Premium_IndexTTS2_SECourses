"""Public product release history; keep personal training results in local run reports."""

from __future__ import annotations

import gradio as gr


CHANGELOG_ENTRIES: list[tuple[str, str, str]] = [
    (
        "v6.5",
        "2026-09-07",
        """
### Consistent 15-second automatic training references

- Automatic references now target 15 seconds across training speaker/emotion conditioning, validation, checkpoint evaluation, saved adapter references, epoch samples, and new speech comparisons.
- Selection preserves transcript agreement and matching word boundaries, then chooses the nearest eligible duration from the same speaker's training split. Other-reference modes exclude the current target clip; equally suitable alternatives can still vary reproducibly with the seed and epoch.
- Known durations take priority over missing duration metadata. New speech plans record the target and selected duration, and resume fingerprints include the reference-selection policy and metadata.
- Training controls and startup logs explain the 15-second target and nearest-reference fallback.

Restart after updating. Existing adapters and frozen comparison references remain compatible. Explicit self-reference modes and custom sample references retain their requested behavior. The 15-second target applies to reference selection; it does not shorten or filter training utterances.
""".strip(),
    ),
    (
        "v6.4",
        "2026-09-07",
        """
### Automatic checkpoint selection for each dataset

- Fresh training starts from the installed base model and the selected dataset. Speech references and comparison plans come from that run's own data, and a new run cannot overwrite an existing training history. Earlier adapters and reports are not required.
- Automatic stopping now spaces patience checks by optimizer updates, so nearby epoch boundaries do not consume extra patience. One optional lower-learning-rate trial can continue a plateau within the original budget while preserving the best checkpoint and resume state.
- Automatic speech comparison freezes balanced validation prompts and several generation seeds before training, then compares up to three current-run checkpoints with freshly generated Base speech. It measures transcript errors, speaker similarity, possible truncation/repetition, and invalid audio; paired uncertainty estimates help resolve transcript comparisons.
- **Use best checkpoint** follows the completed speech recommendation. Base can win, and unresolved transcript ties use measured validation loss. A missing recommended file raises a clear error.
- Added an optional independent **Final-test dataset**. Its sources and audio must be separate from training/validation. The selected model is frozen before this test, and test results do not choose another checkpoint.
- Reports retain settings, file hashes, coverage warnings, per-clip results, and a blind listening form. Automated measurements are screening signals; human naturalness ratings remain separate.
- Subtitle sidecars remain part of the fresh preparation workflow. The voice/transcript audit now uses character errors for Chinese and Japanese and word errors for English, Spanish, and Arabic.
- The dashboard shows audio and text validation losses separately, reports stalled checks and the learning-rate trial, and distinguishes the lowest-loss update from other checkpoints in the same epoch. Fixed evaluation startup percentages, cumulative grid throughput, and cramped checkpoint headers.

Restart after updating. Existing adapters remain compatible. New training runs enable automatic speech comparison by default; this adds evaluation time after optimization. The optional independent test adds further generation work. Newly saved adapters now record the same release version shown in the app.

[Training and checkpoint-selection guide](https://github.com/FurkanGozukara/Premium_IndexTTS2_SECourses/blob/master/docs/TRAINING_SELECTION.md).
""".strip(),
    ),
    (
        "v6.3",
        "2026-09-06",
        """
### More reliable audio endings and safer training clips

- Batched generation now respects each clip's actual length through acoustic decoding. Padding for longer clips no longer leaks into shorter clips through the semantic decoder, length regulator, acoustic convolutions, or vocoder.
- Dataset preparation preserves word padding, retains the final analysis frame, and refines touching word timestamps against sustained quiet intervals. Shared cuts stay within the boundary words and never move an ending before its aligned final word.
- Sentence-aligned preparation now recovers late word releases from the original recording and repacks whole sentences around verified pauses. Neighboring clips share one cut decision, including when their loudness gains differ. The default 30 ms quiet-edge check inspects actual output samples after cleanup; unresolved sentences and cuts are recorded for review.
- Added a Voice and transcript audit in the dataset tab: reference-speaker checks, whole-recording validation/test holdouts, fresh clip transcription, and separate first/last-word checks before selecting the audited training dataset. Source clips are preserved.
- Fixed dataset progress reaching 100% when an individual Whisper transcription finished; stopping a partial run now preserves its actual progress.
- Feature-cache progress now reads the worker's actual clip counts, percentage, speed, and ETA instead of staying at zero until completion.
- Silence detection keeps frame timestamps aligned at 22.05 kHz, avoiding accumulated timing drift from repeatedly rounding fractional sample counts.
- Checkpoint comparisons now preserve entered text and reference recordings when switching adapters or refreshing analysis. Empty forms still use the selected run's suggestions, and **Use LoRA / DoRA reference** remains available.
- Refresh reloads the selected run's checkpoint list and analysis as well as the folder labels. Training samples show their actual filenames, and checkpoint summaries distinguish different updates within the same epoch.
- Fixed flashing progress and stats panels during voice generation, batch generation, dataset preparation, feature caching, and training. Live values, progress bars, charts, and logs update without the pulsing or fading overlay.

Restart the app after updating. Existing adapters remain compatible. The dataset fixes apply when preparing new clips; adapters trained on clipped audio need a rebuilt dataset, refreshed feature cache, and retraining to benefit from corrected training boundaries. Sampled pronunciation can still vary.

""".strip(),
    ),
    (
        "v6.2",
        "2026-09-06",
        """
### Fix for unfinished final words

- Fixed a speech-decoder position error that could stop the final word before it finished, including with LoRA / DoRA voices.
- Corrected both standard and accelerated decoding, including accelerated batches with different prompt lengths. Existing voice adapters work without retraining.
- Added regression checks that compare cached decoding against full-sequence decoding and verify accelerated speech positions on CUDA.

Restart the app after updating. Fixed seeds can produce different audio because generation now follows the positions used during training.
""".strip(),
    ),
    (
        "v6.1",
        "2026-09-06",
        """
### More reliable voice training and automatic stopping

- **Automatically stop when progress stalls** is now enabled by default. Training can finish before its epoch or step limit while keeping the best checkpoint. The default allows six validation checks without a meaningful gain after warmup, at least 1,000 updates, and two dataset passes; you can adjust these controls or disable automatic stopping.
- Validation now holds out complete source recordings by default and uses training-only clips for reference conditioning. It checks the full validation set every 250 updates and at epoch boundaries, with token-weighted scores for more consistent comparisons. Invalid validation references produce a clear error before model loading instead of silently disabling validation.
- Speech-token feature extraction now keeps the semantic encoder in FP32. Cache fingerprints detect changed audio, transcripts, model assets, and extraction settings so stale features are rebuilt when caching is requested.
- Best-checkpoint selection includes every validation check, including improvements between epoch boundaries. Resume state preserves the best step and stopping counter; recoverable FP16 overflow skips an update without advancing the learning-rate schedule.
- Added optional command-line tools for collecting main video/subtitle pairs, auditing speaker consistency and transcript agreement, and measuring generated checkpoint comparisons.
- Updated training presets, help, and the guide to explain validation, automatic stopping, and checkpoint selection.

For existing datasets, run feature caching again to rebuild older caches with the corrected extractor.
""".strip(),
    ),
    (
        "v6.0",
        "2026-09-04",
        """
### Reliable reference previews, restored results, and release history

- Record a reference voice directly from the microphone, or use audio and video from anywhere on disk. External media is staged safely for Gradio, and incompatible video is converted to a browser-playable preview without changing the source.
- Generated candidates and prepared-dataset reference clips now render reliably after a task completes or a page reconnects. Dataset feature caching has live progress, refreshes the selected dataset, and immediately updates the Training tab.
- Fixed competing batch progress updates, preserved the source sample rate during optional audio tuning, skipped automatic checkpoint evaluation cleanly when no validation items exist, and made CPU diagnostics independent of GPU VRAM estimates.
- The accelerated decoder now honors disabled `top-k` and `top-p` limits instead of rejecting valid settings.
- Added this lazy-rendered **Changelog** tab with the public IndexTTS Premium release history and project links.
""".strip(),
    ),
    (
        "v5.1",
        "2026-09-04",
        """
### End-to-end reliability pass

- Fresh installations now select the quality preset, while normal generation runs in the main process by default so the loaded model can be reused between jobs. Isolated subprocess generation remains available as an option.
- Fixed accelerated generation regressions involving sampling, attention masks, end-of-sequence handling, and cache state, with parity checks covering the standard and accelerated paths.
- Corrected the hosted INT8 ConvRot checkpoint name and kept automatic download with a clear BF16 fallback when the optimized file is unavailable.
- Checkpoint Grid now ignores incomplete audio cells, avoids overlapping renders, and supports a custom reference file in the VRAM benchmark.
- Improved mobile header wrapping, dataset-to-training refresh behavior, task reattachment, Unicode text handling, and compatibility with current PyTorch enum registration.
""".strip(),
    ),
    (
        "v5.0",
        "2026-09-04",
        """
### IndexTTS 2.5, LoRA / DoRA training, and a complete app rebuild

- Moved the application entirely to the official multilingual IndexTTS 2.5 model stack and removed the legacy IndexTTS 1.x and 2.0 execution paths.
- Added calibrated 6, 8, 10, 12, 16, 24, and 32 GB VRAM tiers, GPT block swapping for smaller GPUs, and an optional INT8 ConvRot GPT checkpoint that downloads automatically when selected.
- Rebuilt the interface on Gradio 6 with Voice Generation, Batch Generation, Models & Performance, reusable read-only system presets, editable user presets, live progress, real cancellation, and last-run value recovery.
- Added subtitle-aware dataset preparation from audio or video, Whisper word timestamps, sentence and pause boundaries, duplicate removal, loudness normalization, cached training features, and dataset statistics.
- Added full LoRA / DoRA training with quality-first defaults, resume modes, validation, early stopping, progress charts, periodic samples, and low-VRAM block swapping.
- Added Checkpoint Grid listening comparisons, automatic generalization analysis, measured checkpoint evaluation, recommended-checkpoint selection, and per-voice speaking-rate calibration.
- Unified reference audio and video handling across generation, training, and grid workflows, including automatic reference discovery and consistent preset behavior.
""".strip(),
    ),
    (
        "v4.2",
        "2026-09-02",
        """
### Initial IndexTTS 2.5 migration

- Added the official `IndexTeam/IndexTTS-2.5` multilingual inference stack and switched model loading to the repository-local `models` directory.
- Added true section micro-batching, main-process and subprocess generation, cooperative cancellation, multilingual text segmentation, and caption-timing support.
- Added speaker and emotion reference modes, emotion text and vector controls, optional audio tuning, and system/user preset separation so shipped defaults cannot be overwritten.
- Centralized model downloads and local Hugging Face caches so required model components can be reused without unnecessary downloads.
""".strip(),
    ),
    (
        "v4.1",
        "2026-05-11",
        """
### Turn generated speech into a ready-to-share video

- Added an optional image input that combines a still image with the generated voice as a 1080p MP4.
- The source image and final video are stored alongside the numbered task output, recorded in metadata, and shown in a dedicated video preview.
- MP3 export can run in the same job without deleting the WAV before MP4 rendering completes.
""".strip(),
    ),
    (
        "v4.0",
        "2026-04-05",
        """
### Major generation workflow and installer upgrade

- Rebuilt the interface with a browser title and favicon, automatic model downloads, full preset save/load, cancellation, richer console status, speed, progress, and ETA reporting.
- Added SRT subtitle input and optional cue-timed speech generation for matching existing caption timing.
- Reference voices can come from uploaded audio, uploaded video, or a microphone recording.
- Added subprocess generation that releases RAM and VRAM after completion, plus real section batch-size processing for higher throughput when memory allows.
- Moved the one-click installers to `uv` and verified the workflow on Windows, RunPod, and Massed Compute.
""".strip(),
    ),
]


def build_changelog_tab() -> None:
    """Render newest-first release notes and SECourses project details."""

    gr.Markdown("## Release history")
    for index, (version, release_date, markdown) in enumerate(CHANGELOG_ENTRIES):
        with gr.Accordion(f"{version} · {release_date}", open=index == 0):
            gr.Markdown(markdown)

    gr.Markdown(
        """
### About IndexTTS 2.5 Premium SECourses

Built by **SECourses** for local voice cloning, long-form speech generation, and LoRA / DoRA voice training.

[Support SECourses on Patreon](https://www.patreon.com/SECourses) · [GitHub repository](https://github.com/FurkanGozukara/Premium_IndexTTS2_SECourses)
""".strip(),
    )


__all__ = ["CHANGELOG_ENTRIES", "build_changelog_tab"]
