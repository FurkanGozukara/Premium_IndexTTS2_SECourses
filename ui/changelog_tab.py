"""Public product release history; keep personal training results in local run reports."""

from __future__ import annotations

import gradio as gr


CHANGELOG_ENTRIES: list[tuple[str, str, str]] = [
    (
        "v6.11",
        "2026-09-09",
        """
### Selection by one deployment score, clip-length mix by default, and two new training options

- **Checkpoint selection:** among the checkpoints that pass the Base guards, the speech comparison now selects by the same score the voice decoder gate uses: the paired speaker-similarity gain over Base, minus four times any paired word-error increase, plus a small term for pausing more like the person than Base does; validation loss only breaks ties within 0.002. The report lists the score and its parts for every candidate. Base wins when no adapter beats it on that score.
- **Clip lengths:** dataset preparation gains **Share of medium clips** (about 10 seconds) beside **Share of single-sentence clips** (about 6 seconds). A shorter clip is only cut where each inner edge sits in a clear pause (about 200 ms of quiet with the default padding); a start that finds none keeps the target length. Each clip records which aim produced it and `dataset_info.json` counts them. Both shares now default to 0.25, so a dataset built from long narration also covers single sentences and short paragraphs; set both to 0 for the previous behavior.
- **Average the last saved checkpoints** (training, default 0): averages the last N saved updates in parameter space into one more speech-comparison candidate, `<name>_avg_ep<first>_<last>.safetensors`, chosen only when it scores best.
- **Decoder training codes** (training, default `real`): the voice decoder adapter can train on the selected checkpoint's own teacher-forced code predictions (`gpt`) or half real and half predicted codes (`mixed`) instead of the recordings' codes; `tools/train_decoder_adapter.py --code-source` does the same for an existing folder.
- **New tool** `tools/compare_checkpoint_benchmark.py`: renders a training's frozen speech benchmark with any checkpoint or decoder adapter and compares it pairwise with the run's own measurement.
- The learning-rate default stays 4e-5; a sweep found that higher rates reach a lower token loss but generate more word errors.

Restart after updating. Existing adapters, datasets and saved outputs remain compatible; existing presets keep their own clip-share values.
""".strip(),
    ),
    (
        "v6.10",
        "2026-09-08",
        """
### Safer adapters, reliable generation controls, and stronger training safeguards

- **Voice decoder loading:** cold startup now applies the requested decoder after the acoustic model is initialized. AUTO selection verifies the adapter's association and approval state, including when switching voices or reusing a loaded model; incompatible or unverified adapters are not silently installed.
- **Checkpoint consistency:** GPT selectors exclude decoder files by filename and metadata. Training and Checkpoint Grid use the same speech-based recommendation, respect a Base recommendation, report missing files, and clear stale speaking-rate calibration when the selected voice changes.
- **Progress and cancellation:** generation shows progress before model loading, keeps validation errors visible, and reconnects to active tasks without restoring stale success cards. Cancellation targets the intended task and waits for cooperative workers to exit. Training displays decoder-gate and final-test progress in their own phases.
- **Batch generation:** malformed caption files are handled per item, so Continue on errors can proceed. Upload order, emotion references, and image inputs are preserved; saved metadata records the actual execution policy, and per-item worker logs remain available after completion.
- **Media exports:** high-bitrate MP3 requests use a compatible sample rate; still-image MP4 duration follows the audio instead of adding a padded tail. Audio tuning reports when filter frequencies must be limited to the source sample rate.
- **Runtime and VRAM benchmarks:** per-request low-memory overrides are reversible, and loading a preset does not trigger user-only memory-tier changes. Benchmarks distinguish active GPU usage from driver-reserved memory, use bounded idle waits, retain the selected GPU, support cancellation/reconnection, and record peak memory across the full generation workload.
- **Training and final assessment:** checkpoint, decoder-strength, and decoding choices use validation data only. An optional independent final test runs after the deployment's checkpoint, decoder, rate, and sampling settings are frozen and hash-verified; its results do not retune that deployment. Failed, skipped, or rejected decoder gates keep their evidence outside automatic loading, with collision-safe quarantine filenames.
- **Diagnostics and release notes:** concurrent in-process jobs no longer mix unrelated output into generation logs. Decoding summaries distinguish retained defaults from an accepted sweep override. Added regression coverage for the repaired paths; public release notes describe product behavior without personal training results or dataset-specific comparisons.

Restart after updating. Existing adapters and saved outputs remain compatible. Automatic metrics and passing regression checks do not guarantee flawless speech or support for every hardware/backend combination.
""".strip(),
    ),
    (
        "v6.9",
        "2026-09-08",
        """
### Voice decoder adaptation: a second adapter for timbre, trained automatically after every run

- Training can adapt the semantic-to-mel decoder as a second phase after GPT checkpoint selection. The decoder DoRA uses its own flow-matching objective and a randomly drawn other clip of the speaker as the conditioning prompt for each target, instead of sharing one fixed training prompt.
- Checkpoints are selected by speaker identity, not by loss: every 500 updates and at each epoch boundary, eight held-out clips are re-rendered from their own semantic codes and compared with the real recordings by CAMPPlus similarity, with the held-out flow loss as a guard against drifting from the decoder's objective. Early stopping and a one-time learning-rate halving follow the GPT trainer's rules.
- An adapter is installed only when it wins twice: its re-rendered identity must beat the pretrained decoder's, and the selected checkpoint must render the speech benchmark closer to the real recordings with the adapter than without it (same sentences, reference, and seeds), without a word-error-rate regression. The test judges strengths 1.0 and 0.6 and records the better-scoring one as the adapter's recommended strength, which Voice Generation applies when the LoRA / DoRA is selected. A rejected file is parked as `analysis/<name>.s2mel.rejected` and the training summary says why; generation keeps the pretrained decoder.
- The adapter is saved as `<name>.s2mel.safetensors` in the training folder, shared by every checkpoint of that training, and reported in `analysis/decoder_adapter.json` and `analysis/speech_evaluation/decoder_test/report.md`. Voice Generation applies it automatically whenever that LoRA / DoRA is selected; with classifier-free guidance the adapter renders the conditioned branch while the unconditional branch keeps the pretrained decoder. Selecting a LoRA / DoRA sets the new **Voice decoder adapter** dropdown to the adapter saved with it; **None** plays the GPT adapter alone for an A/B listen, any other decoder adapter file can be chosen, and **Voice decoder adapter strength** starts at the recommended strength and scales it independently of the LoRA / DoRA strength. Decoder files are tagged so they can never be applied to the GPT by mistake.
- Automatic references now prefer, among the cleanest training clips near 15 seconds, the one nearest the speaker's median pitch and pace (**Prefer a reference near the speaker's median pitch and pace**, on by default). Per-clip pitch is cached in the dataset's `analysis/pitch_cache.json`.
- **Sweep decoding settings after training** (on by default): the recommended checkpoint renders the speech benchmark at other temperatures, guidance rates, and beam counts; a change is kept only when it beats the defaults on speaker similarity and word error, and the winner is saved as `analysis/decoding.json`. Voice Generation applies it together with the calibrated speaking rate (the auto-apply switch now covers both), and `tools/sweep_decoding.py` runs the sweep for an existing training folder.
- New training option **Adapt the voice decoder after training** (enabled by default) with rank, alpha, epochs, learning rate, and timeout; the training dashboard shows the new phase and the verdict. `tools/train_decoder_adapter.py` adds a decoder adapter to an existing training folder without retraining the GPT adapter and runs the same full-pipeline test (`--no-test` skips it). A speaking-rate change with `inference_cfg_rate` 0 no longer fails in the flow-matching solver.

Restart after updating. Existing adapters keep working without a decoder adapter; add one with the tool or by training again.
""".strip(),
    ),
    (
        "v6.8",
        "2026-09-07",
        """
### Unique grid cell files across runs

- Checkpoint Grid cell files were named by checkpoint kind and epoch only, so similarly labeled checkpoints from different training runs could overwrite one another's audio and measurements. Cell names now add the checkpoint label when they would collide, so grids that compare adapters from several runs keep every clip.

Restart after updating. Existing grids, adapters, and reports remain readable; only newly generated grids use the new file names.
""".strip(),
    ),
    (
        "v6.7",
        "2026-09-07",
        """
### Pause measurement in speech comparisons

- Speech comparison reports and the grid measurement tool now show **Pause time vs real**: the generated clips' internal pause time (silences of at least 120 ms between words and sentences, clip edges excluded) divided by the real recordings' on the same sentences. 1.00 matches the person, above 1 pauses longer, below 1 rushes. It is shown for information and does not change which checkpoint is selected.
- The training guide, README, and help text describe the new column. Objective pause measurements and subjective listening preferences remain separate observations.

Restart after updating. Existing adapters, datasets, and saved reports remain compatible; older reports simply have no pause column.
""".strip(),
    ),
    (
        "v6.6",
        "2026-09-07",
        """
### Transcript audits that trust your subtitles, and a corrected speaking-rate calibration

- The voice and transcript audit now treats the transcripts you supplied as the authority on how names and terms are written. It collects mixed-case words, acronyms, version numbers, and mid-sentence capitalized names from your subtitles, decodes fresh clip transcriptions with a small beam search, and no longer counts a recognizer spelling of one of those terms as a transcript error. Contractions, joined compounds such as "Swarm UI", and okay/OK are normalized on both sides.
- The first/last-word check now fails only for missing, extra, or different ordinary words at a clip edge, so cut endings and music bleed are still rejected while spelling variants of your terms pass. Prompting Whisper with the dataset's terms is available as a command-line option and stays off by default.
- Pause search for sentence-aligned clips may look up to 200 ms back into the previous word when Whisper's word end runs into the pause, but only a quiet stretch longer than a stop-consonant closure counts. The 30 ms quiet-edge check on exported audio is unchanged.
- New experimental dataset option **Share of single-sentence clips** aims a reproducible share of clips at one short sentence of about 6 seconds. It defaults to 0 and leaves existing presets unchanged; change it only when experimenting with the training utterance mix.
- The saved speaking rate is now calibrated from matched held-out sentences once the automatic speech comparison completes, rather than comparing an unrelated short epoch sample with longer recordings. The earlier estimate is retained separately for reference. Voice Generation shows the stored value in an editable **Saved speaking rate for this LoRA / DoRA** field with a **Save speaking rate** button, so you can override any estimate per adapter.
- Speech comparison reports use the same normalized transcript comparison, so split compounds and contractions no longer count as errors there either, and the dataset's own spellings of names and terms are accepted for Base and adapters alike.
- The speaker guard in automatic checkpoint selection now compares each generated sentence with the real recording of that sentence whenever at least four matched recordings exist. Similarity to the single conditioning reference is also reported, but is kept separate from resemblance to matched real speech.
- Spoken forms of currency, storage and frequency units, and decimals are normalized on both sides of every transcript comparison ("$0.61" and "61 cents", "6 GB" and "6 gigabytes", "1.5" and "one point five"). An edge word that the recognizer merely replaces with a similar word or splits into pieces no longer fails the first/last-word check; missing or extra edge words still do, and the quiet-edge measurement continues to guard against cut audio.
- The audit gives clips that failed only the transcript checks a second opinion from the full whisper-large-v3 model and keeps them when it agrees with your transcript. This adds processing time only for those rejected clips. A new audit checkbox controls it.
- Caption cleanup no longer inserts a space inside file names and dotted acronyms such as "update.bat", "main.py", or "U.S."; run-on sentences like "done.Next" are still separated. This avoids artificial text breaks inside file extensions.

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
