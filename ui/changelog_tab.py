"""Public product release history; keep personal training results in local run reports."""

from __future__ import annotations

import gradio as gr


CHANGELOG_ENTRIES: list[tuple[str, str, str]] = [
    (
        "v6.9",
        "2026-09-08",
        """
### Voice decoder adaptation: a second adapter for timbre, trained automatically after every run

- Generation has three stages, and voice adapters so far touched only the first. The GPT LoRA / DoRA decides what is said and when; the semantic-to-mel decoder decides how the voice sounds and knew a voice only through the reference clip of each generation. Training now adapts that decoder too: after checkpoint selection and the speech comparison, a DoRA of rank 128 (alpha 128, learning rate 2e-4 from a five-rate sweep) is trained on the decoder's transformer blocks with its own flow-matching objective, using a randomly drawn other clip of the speaker as the in-context prompt for every target clip and every epoch. A fixed prompt taught an early build a prompt-independent offset toward the dataset's average voice that made every generated sentence thinner and higher; the random prompt removes that failure by construction.
- Checkpoints are selected by speaker identity, not by loss: every 500 updates and at each epoch boundary, eight held-out clips are re-rendered from their own semantic codes and compared with the real recordings by CAMPPlus similarity, with the held-out flow loss as a guard against drifting from the decoder's objective. Early stopping and a one-time learning-rate halving follow the GPT trainer's rules.
- An adapter is installed only when it wins twice: its re-rendered identity must beat the pretrained decoder's, and the selected checkpoint must render the speech benchmark closer to the real recordings with the adapter than without it (same sentences, reference, and seeds), without a word-error-rate regression. The test judges strengths 1.0 and 0.6 and records the better-scoring one as the adapter's recommended strength, which Voice Generation applies when the LoRA / DoRA is selected. A rejected file is parked as `analysis/<name>.s2mel.rejected` and the training summary says why; generation keeps the pretrained decoder.
- The adapter is saved as `<name>.s2mel.safetensors` in the training folder, shared by every checkpoint of that training, and reported in `analysis/decoder_adapter.json` and `analysis/speech_evaluation/decoder_test/report.md`. Voice Generation applies it automatically whenever that LoRA / DoRA is selected; with classifier-free guidance the adapter renders the conditioned branch while the unconditional branch keeps the pretrained decoder. Selecting a LoRA / DoRA sets the new **Voice decoder adapter** dropdown to the adapter saved with it; **None** plays the GPT adapter alone for an A/B listen, any other decoder adapter file can be chosen, and **Voice decoder adapter strength** starts at the recommended strength and scales it independently of the LoRA / DoRA strength. Decoder files are tagged so they can never be applied to the GPT by mistake.
- Automatic references now prefer, among the cleanest training clips near 15 seconds, the one nearest the speaker's median pitch and pace (**Prefer a reference near the speaker's median pitch and pace**, on by default). The old rule picked a 119 Hz, 2.3 words-per-second clip for a 144 Hz, 2.6 words-per-second speaker, and every generation started from that low, slow baseline. Per-clip pitch is cached in the dataset's `analysis/pitch_cache.json`.
- **Sweep decoding settings after training** (on by default): the recommended checkpoint renders the speech benchmark at other temperatures, guidance rates, and beam counts; a change is kept only when it beats the defaults on speaker similarity and word error, and the winner is saved as `analysis/decoding.json`. Voice Generation applies it together with the calibrated speaking rate (the auto-apply switch now covers both), and `tools/sweep_decoding.py` runs the sweep for an existing training folder.
- New training option **Adapt the voice decoder after training** (enabled by default) with rank, alpha, epochs, learning rate, and timeout; the training dashboard shows the new phase and the verdict. `tools/train_decoder_adapter.py` adds a decoder adapter to an existing training folder without retraining the GPT adapter and runs the same full-pipeline test (`--no-test` skips it). A speaking-rate change with `inference_cfg_rate` 0 no longer fails in the flow-matching solver.
- Measured on the V8 voice: the installed decoder adapter (DoRA rank 128, early-stopped at 13,000 updates) raised speaker similarity to the real recordings on the speech benchmark from 0.804 to 0.865 (95 percent interval +0.047 to +0.077 over 13 sentences and three seeds) and lowered the word error rate from 2.34 to 1.93 percent, so strength 1.0 was recommended over 0.6. Its one-epoch predecessor gained similarity but cost 1.3 points of word error and tied a path-blind listening test at full strength while leading 7 to 5 at strength 0.6. The first, fixed-prompt build lost in 32 of 36 comparisons. A V9 training started from the browser with every new option at its default (median-matched reference, decoder adapter, decoding sweep) matched the hand-tuned V8 as deployed: word error 3.47 against 4.42 percent, speaker similarity 0.893 against 0.901, and a path-blind listener preferring V9 in 19 of 36 pairs. See `VOICE_DECODER_ADAPTER_2026-09-08.md` for every measurement.

Restart after updating. Existing adapters keep working without a decoder adapter; add one with the tool or by training again.
""".strip(),
    ),
    (
        "v6.8",
        "2026-09-07",
        """
### Unique grid cell files across runs, and a four-adapter comparison report

- Checkpoint Grid cell files were named by checkpoint kind and epoch only, so two checkpoints of the same kind and epoch from different runs (for example the "best" files of two trainings that both stopped in epoch 7) wrote the same files and the later one silently replaced the earlier one's audio and measurements. Cell names now add the checkpoint label when they would collide, so grids that compare adapters from several runs keep every clip.
- `ADAPTER_COMPARISON_V5_V8_2026-09-07.md` compares the automatically selected checkpoints of four trainings of one voice on a recording none of them trained on, at speaking rate 1.0 and at each adapter's calibrated rate, with objective measurements and a blind five-way listening test. With each adapter at its calibrated speaking rate, as the app runs them, the listener ranked V8 first in 18 of 36 groups and above V6 in 30, V7 second, V5 last of the four; at rate 1.0 V7 led with V8 second. V8 with its saved speaking rate is now the recommended adapter for that voice, V7 when the narrator's own pausing matters most, and V6 stays a fallback.

Restart after updating. Existing grids, adapters, and reports remain readable; only newly generated grids use the new file names.
""".strip(),
    ),
    (
        "v6.7",
        "2026-09-07",
        """
### Pause measurement in speech comparisons, and a measured V8 training pass

- Speech comparison reports and the grid measurement tool now show **Pause time vs real**: the generated clips' internal pause time (silences of at least 120 ms between words and sentences, clip edges excluded) divided by the real recordings' on the same sentences. 1.00 matches the person, above 1 pauses longer, below 1 rushes. It is shown for information and does not change which checkpoint is selected. It was added because a blind listening test preferred the V6 adapter over V7 in 26 of 36 comparisons while every existing proxy favored V7; measured, V7 matched the narrator's pauses (median ratio 0.84) and V6 paused 45 percent longer, and the listener preferred the longer pauses.
- A V8 adapter was trained with the complete v6.6 preparation and audit (whole file names such as "update.bat", relaxed edge rule, spoken-form normalization, whisper-large-v3 second opinion) and the single-sentence share at 0, so only the data changed against V6. Measured on the same 12 held-out sentences as the V6 and V7 adapters, V8 gained 17 percent more training audio but landed on V6's numbers for identity, pitch, pace and pauses and slightly behind it on strict word error; a blind listener preferred V6 over V8 in 22 of 36 comparisons and V8 over V7 in 20 of 36. V6 remains the recommended adapter for that voice, the audit improvements stay because they keep clean speech, and the single-sentence share stays at 0.
- The training guide, README, and help text describe the new column. `V8_TRAINING_REPORT_2026-09-07.md` contains the full comparison.

Restart after updating. Existing adapters, datasets, and saved reports remain compatible; older reports simply have no pause column.
""".strip(),
    ),
    (
        "v6.6",
        "2026-09-07",
        """
### Transcript audits that trust your subtitles, and a corrected speaking-rate calibration

- The voice and transcript audit now treats the transcripts you supplied as the authority on how names and terms are written. It collects mixed-case words, acronyms, version numbers, and mid-sentence capitalized names from your subtitles, decodes fresh clip transcriptions with a small beam search, and no longer counts a recognizer spelling of one of those terms as a transcript error. Contractions, joined compounds such as "Swarm UI", and okay/OK are normalized on both sides.
- The first/last-word check now fails only for missing, extra, or different ordinary words at a clip edge, so cut endings and music bleed are still rejected while spelling variants of your terms pass. On a 17-recording English narration dataset these rules accepted about 60 percent of the clips the previous audit had rejected on transcript grounds, without changing any clip it had accepted. Prompting Whisper with the dataset's terms is available as a command-line option; it recovered a few more clips but made some clean clips fail, so it stays off.
- Pause search for sentence-aligned clips may look up to 200 ms back into the previous word when Whisper's word end runs into the pause, but only a quiet stretch longer than a stop-consonant closure counts. Measured recovery is modest; the 30 ms quiet-edge check on exported audio is unchanged.
- New experimental dataset option **Share of single-sentence clips** aims a reproducible share of clips at one short sentence of about 6 seconds. It defaults to 0 and leaves existing presets unchanged. In one blind listening comparison a 0.3 share made the adapter rush between sentences and lowered naturalness ratings even though identity and pace measurements improved, so leave it at 0 unless you are experimenting.
- The saved speaking rate is now calibrated from matched held-out sentences once the automatic speech comparison completes. The earlier estimate from the short epoch sample compared a ten-word sentence with long multi-sentence recordings and overstated how slow a voice is; it is kept beside the new value for reference. Voice Generation shows the stored value in an editable **Saved speaking rate for this LoRA / DoRA** field with a **Save speaking rate** button, so you can override any estimate per adapter.
- Speech comparison reports use the same normalized transcript comparison, so split compounds and contractions no longer count as errors there either, and the dataset's own spellings of names and terms are accepted for Base and adapters alike.
- The speaker guard in automatic checkpoint selection now compares each generated sentence with the real recording of that sentence whenever at least four matched recordings exist. Similarity to the single reference clip rewards copying that prompt, which favored the unadapted model even when independent listening rated the adapted voice far closer to the speaker; both values are reported.
- Spoken forms of currency, storage and frequency units, and decimals are normalized on both sides of every transcript comparison ("$0.61" and "61 cents", "6 GB" and "6 gigabytes", "1.5" and "one point five"). An edge word that the recognizer merely replaces with a similar word or splits into pieces no longer fails the first/last-word check; missing or extra edge words still do, and the quiet-edge measurement continues to guard against cut audio.
- The audit gives clips that failed only the transcript checks a second opinion from the full whisper-large-v3 model and keeps them when it agrees with your transcript. Measured on the same narration dataset, the full model recovered a third of the clips the fast turbo model had rejected and agreed with an independent listener more often; it is 2.5 times slower, so it runs only on those clips. A new audit checkbox controls it.
- Caption cleanup no longer inserts a space inside file names and dotted acronyms such as "update.bat", "main.py", or "U.S."; run-on sentences like "done.Next" are still separated. Previously prepared clips containing such names spelled them as "update. bat", which taught models to pause before the extension.

Restart after updating. Existing adapters and datasets remain compatible; re-run the audit to benefit from the transcript changes. Generated speech and audit decisions still depend on automatic recognition, which remains imperfect on technical vocabulary.
""".strip(),
    ),
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
