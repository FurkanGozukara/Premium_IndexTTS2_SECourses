# IndexTTS 2.5 Premium - The Complete Beginner-to-Advanced Guide

## App Download Link
- You can get the app installer from here https://www.patreon.com/SECourses/posts/indextts-2-5-and-139297407

## Quick Info

Voice cloning, long-form narration, caption-timed audio and MP4, batch production, dataset preparation, LoRA/DoRA training, checkpoint evaluation, listening grids, speaking-rate calibration, and low-VRAM operation - all in one tested workflow.

**V6.2 final-word fix:** corrected a speech-decoder position error that could leave the last word unfinished, including with LoRA/DoRA voices. Standard and accelerated generation now follow the positions used during training, with regression checks for cached decoding and variable-length accelerated prompts. Existing adapters work without retraining. Restart the app after updating; the same seed may produce different audio with the corrected decoder.

**V6.1 training quality update:** train with more reliable validation, preserve the best checkpoint, and stop automatically when progress stalls.

- The automatic-stop checkbox is enabled by default, with adjustable patience and a warmup/minimum-training grace period. The live status shows the best update and stopping counter.
- Validation holds out complete source recordings, uses training-only reference clips, and checks the full validation set every 250 updates and at epoch boundaries by default.
- FP32 semantic feature extraction and content-aware cache fingerprints prevent stale or lower-precision training targets from being reused when features are recached.
- Checkpoint analysis includes improvements between epoch boundaries; resume preserves the stopping state, and recoverable FP16 overflow does not advance the update schedule.
- Optional command-line tools collect main video/SRT pairs, audit narration data, and measure generated checkpoint comparisons. The [completed training report](TRAINING_QUALITY_REPORT_2026-09-06.md) documents the 7.85-hour V4 experiment, 3.28% lower independent test loss versus V3, and the limits of the audio measurements.

**V6.0 maintenance update:** the main workflows and registered settings remain compatible with the screenshots, while the following user-visible behavior is new or corrected.

- Record a reference directly from the browser microphone, alongside upload, local-path, library, adapter, and recent-output sources.
- Local media paths are staged safely for browser playback, and unsupported video codecs receive an FFmpeg-generated browser preview without changing the audio used for cloning.
- Voice Generation now defaults to in-process model reuse; isolated subprocess mode remains available when hard cancellation and complete VRAM release matter more than repeat-run speed.
- Candidate players, dataset reference players, feature-cache handoff, completed batch summaries, CPU diagnostics, and narrow-screen header controls now restore or update reliably.
- A lazy-rendered Changelog tab keeps the public release history and official SECourses project links inside the app.

**Choose a route:**

- Beginner route: install, choose a reference, enter text, and generate a first voice.
- Production route: control segmentation, pauses, subtitles, emotion, duration, formats, and batches.
- Voice-training route: prepare clean data, cache features, train LoRA or DoRA, evaluate checkpoints, and deploy the best adapter.
- Hardware route: choose a VRAM tier, BF16 or INT8, block swapping, auxiliary-model residency, and verify the setup with the isolated benchmark.

## 1. Install and Start the App

### Windows one-click install or update

![Installer Files](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/jXo3_Htdr-kvok9AA3dVF.png)

Place the downloaded package in a short writable path such as `G:\Index_TTS_v4`. Before the first install, provide Python 3.12.10 or newer, Git, FFmpeg, CUDA 13.0, cuDNN 9.17 or newer, and Visual Studio Community with the C++ workload. Then run `Windows_Install_or_Update.bat`.

The installer clones or updates `Premium_IndexTTS2_SECourses`, creates its Python 3.12 virtual environment, installs `uv`, resolves the requirements, and runs the model downloader. If it detects local source changes, it deliberately skips `git pull` rather than overwriting them.

1. Run `Windows_Install_or_Update.bat` once for a fresh install and again whenever an update is announced.
2. Run `Windows_Start_App.bat` for normal use. It activates the environment, configures the CUDA allocator and model cache, and launches the Gradio UI.
3. Wait for the local URL, normally `http://127.0.0.1:7860`, and open it in a browser.
4. For a specific GPU, add `set CUDA_VISIBLE_DEVICES=1` to the start BAT or launch `python webui.py --device cuda:1` from the activated environment.

The Windows BAT opens Google Chrome when installed and works regardless of the current working directory. Use `--browser default` to choose your system browser instead.

Useful launch flags are `--port`, `--host`, `--share`, `--model_dir`, `--verbose`, `--browser`, `--no-browser`, and `--device`. For example:

```bat
Windows_Start_App.bat --port 7861 --device cuda:0 --no-browser
```

If another Gradio app previously used the same address, its old tabs can send events to IndexTTS after the port is reused. IndexTTS rejects these requests, as well as requests from a previous launch, before running a handler. Updated IndexTTS tabs show **Reload IndexTTS**; copy any unsaved text before reloading. For an old tab from a different app, reopen that app at the URL printed by its own launcher. Unknown event IDs, incorrect input counts, and missing table-selection data return clear errors instead of the reported Gradio tracebacks. Native API clients remain supported; after a restart, reconnect the client so it loads the current API definitions.

If model loading or worker startup fails, the generation task is saved as failed so polling and page reloads retain its error instead of showing it as still running.

### Linux, RunPod, SimplePod, and Massed Compute

Use the supplied `RunPod_Install_IndexTTS.sh` or `Massed_Compute_Install.sh`, then follow the matching instruction text file in the package. The current Linux installers require root or `sudo`, install aria2, xz, Git LFS, and FFmpeg/ffprobe n9.0, provision a current Python 3.12 environment, install the application requirements, and download the IndexTTS 2.5 models. The Linux launcher then enters the app folder, activates `venv/bin/activate`, clears a conflicting `LD_LIBRARY_PATH`, and starts Gradio with `--share`. Keep the generated public Gradio link private because anyone who has it can use the running GPU.

```bash
bash Linux_Start_App.sh --device cuda:0
```

If startup reports missing model files, run `Windows_Model_Download_and_Fix.bat` on Windows or use **Models & Performance > Download / verify base models** after the interface opens.

## 2. Know the Workspace Before Generating

The header controls reusable settings; the tab row separates generation, batch work, dataset preparation, training, checkpoint comparison, performance, help, and release history. A fresh V6 install selects the quality preset automatically; an updated installation restores its last-used preset. Start in **Voice Generation** and keep the quality preset for the first successful output.

![Annotated 4K overview of the IndexTTS 2.5 Premium workspace](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/97GX_9lELPBHRPwdwMxVg.png)

*Figure 1. The first screen is a working console, not a landing page: reference and text are on the left and center, while run controls, progress, results, and logs stay on the right. **Open / close all sections** is the fastest way to expose or collapse advanced controls.*

At startup, earlier result panels stay clean. **Load last values** restores the most recently saved values across every tab. A system preset is marked with a star and is read-only; a user preset can be created, overwritten, loaded, or deleted. **Reset** returns the registered controls to the system default preset without deleting model files or generated outputs.

- Voice Generation: one script, one reference workflow, optional candidates and media output.
- Batch Generation: many TXT, SRT, VTT, or SBV jobs with shared or per-file references.
- LoRA Dataset Preparation: turn raw audio/video/captions into quality-controlled training clips.
- LoRA / DoRA Training: train, validate, sample, save, resume, analyze, and manage adapters.
- Checkpoint Grid: compare base, recommended, final, and epoch checkpoints with identical inputs.
- Models & Performance: fit the model to the GPU, verify files, and benchmark real VRAM use.
- Help: embedded quick starts, parameter guidance, pause syntax, and recovery steps.
- Changelog: read newest-first release notes and open the official Patreon or GitHub project pages.

## 3. Make the First Voice Clone

### Choose the speaker and write the script

Use a clean, single-speaker reference with little music, reverb, or room noise. Eight to fifteen seconds is a strong starting point. Audio, video, and direct microphone recording are accepted; video is decoded through FFmpeg and the audio preview shows the exact material generation will use.

![Annotated 4K reference voice, script, and run controls](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/beVmKAuttaFV54obG9iOG.png)

*Figure 2. Upload or drop the reference, type the script, choose its language, and press **Generate voice**. V6 also places a **Record from microphone** accordion directly beneath the upload surface. The right column exposes cancellation, progress, current item, speed, ETA, the generated player, and the last 60 log lines.*

1. Drop an audio or video file into **Reference Voice**, expand **Record from microphone** and record a WAV in the browser, or enter an existing local path in **Reference media path** and press **Load path**. Stop the microphone recording before generating; the recorded WAV becomes the active Reference Voice.
2. If only part of a long source is clean, enter ranges such as `1:4;7.5:12` or `01:02-01:08`, then press **Extract ranges**. Ranges are joined in the order written.
3. Type the words to synthesize. Match **Language** to the script, not necessarily to the reference speaker.
4. Keep **Max tokens per segment** on Auto or the language-aware default for the first run.
5. Press **Generate voice** and do not close the terminal while the worker is active.

The **Reference audio library** scans `reference_audios`. **Refresh** rescans it and **Load path** applies the selected entry. When no manual reference is present, an enabled LoRA/DoRA can supply its recommended reference automatically; otherwise the newest compatible library file is the final fallback. **Clear** removes uploads, recordings, previews, the path field, and the current manual selection.

For a local path outside the app's normal output, dataset, adapter, reference, or temporary folders, V6 stages a browser-safe hard link or copy under `.ui_state/reference_media`. A video whose codec cannot play directly in the browser receives a cached preview under `.ui_state/reference_video_previews`; generation still uses the audio extracted from the original source.

### Understand automatic text segmentation

Long input is split before synthesis so each section stays inside the model and VRAM budget. The live preview shows section number, type, text or pause, and details before you commit GPU time.

![Annotated 4K language-aware text segmentation preview](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/R7bzHNe69iMhSCjDS0Qht.png)

*Figure 3. Shorter segments are safer on low VRAM and make failures easier to retry; longer segments can preserve phrasing but reserve more semantic and diffusion memory. The non-CJK budget scale leaves room for English, Arabic, and Spanish subword expansion.*

Use punctuation to create natural boundaries. Enable pause tags when you need exact silence: `[pause:500ms]`, `[pause:0.8s]`, and `<pause=0.5>` are accepted forms. A pause becomes its own section and is inserted without asking the speech model to invent silence.

**Text normalization** expands text before phonetic processing and is recommended for ordinary numbers and punctuation. Disable it only when you have a deliberate pre-normalized script and have checked pronunciation. If the WeText backend rejects an unusual English or Chinese fragment, V6 keeps the original text and logs a concise warning instead of dropping the fragment.

### Use captions and create an MP4

Upload SRT, VTT, or SBV captions to replace or organize the script. With **Use caption cue timing**, each caption unit is retimed to its cue slot and cue start times are preserved. Add a still image only when an MP4 is required.

![Annotated 4K captions, cue timing, and still-image MP4 controls](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/8eVvUoZoJ2GkHOnzgRJfy.png)

*Figure 4. Caption timing is ideal for localization, dubbing drafts, and slide narration. Without the timing checkbox, caption text is still synthesized but flows naturally; without a still image, the job remains audio-only.*

- Plain captions: use subtitle text as a structured script while allowing natural timing.
- Cue-timed captions: preserve subtitle starts and fit each unit to its slot.
- Still-image video: combine the completed audio and uploaded image into an MP4.
- Caption plus pause tags: pause parsing still occurs inside caption text when enabled.

## 4. Shape Emotion, Sampling, and Timing

### Four emotion sources

Emotion is independent from speaker identity. Select **Same as speaker** for the simplest clone, **Emotion reference audio** to transfer delivery from another clip, **Manual vectors** for direct channel control, or **Emotion text** to describe the desired performance.

![Annotated 4K emotion source and vector controls](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/r_76jdiZGWBWm0cMKr_4e.png)

*Figure 5. **Emotion weight** blends the chosen emotion source with the speaker tone. Manual mode exposes joy, anger, sadness, fear, disgust, depression, surprise, and calm; tuned biases and the maximum vector sum keep a combination from becoming unnaturally extreme.*

For emotion-reference work, use a clip whose delivery is clear even if the voice is different: the speaker reference supplies identity and the emotion reference supplies style. **Random emotion exemplar** varies the internal exemplar used with manual vectors. For reproducible comparisons, turn it off and keep the generation seed fixed.

Emotion-text mode needs the Qwen emotion model enabled in Models & Performance. Enter a short direction such as `calm, reassuring, and quietly optimistic`. If the field is blank, the app analyzes the speech text itself.

### Autoregressive and diffusion controls

The autoregressive stage decides semantic tokens; the diffusion stage turns those tokens and conditioning into acoustic detail. Change one family at a time so you can hear what caused the difference.

![Annotated 4K sampling and diffusion settings](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/8FDtrP5a41xLLeQOSIEEi.png)

*Figure 6. The quality preset uses sampling, temperature 0.8, top-p 0.8, four beams, CFM temperature 0.9, and 40 diffusion steps. The system default preset uses three beams, CFM temperature 1.0, and 25 steps. A fixed seed plus a single candidate is the cleanest diagnostic setup.*

- Temperature: lower is steadier; higher is more varied but can be less stable.
- Top-p and top-k: restrict the token candidate pool. Top-k 0 disables that filter.
- Beams: can improve stability, but multiplies time and VRAM. Optional acceleration expects beams 1.
- Repetition penalty: prevents semantic-token loops; keep the established default unless diagnosing repeats.
- Length penalty: affects beam search only; 0 is neutral.
- Max mel tokens: a safety ceiling, not a requested duration.
- Candidates: consecutive seeded alternatives from one request; each costs another generation.
- Diffusion steps: 12-16 is a faster draft range, 25 is the registered/system-default value, and the quality preset uses 40; 35-50 can refine difficult material.
- CFG rate and CFM temperature: control conditioning strength and diffusion variation.
- CFM cache length: lower it only when reserved VRAM is the problem.

### Pacing, silence, and reference processing

Use **Speaking rate** for the voice's pace, **Section silence** for joins, and **Target duration** only when the entire assembled output must approach a known length. A trained voice can store a measured speaking rate and apply it automatically.

![Annotated 4K timing and reference-processing settings](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/z5fvNLN66kO4poWTTaDgY.png)

*Figure 7. Reference limits normally stay at 15 seconds. Semantic layer 17 is the trained recommendation. Reusing speaker conditioning for emotion is the fast default when there is no separate emotion source.*

- Off: no whole-output duration target.
- Natural: regenerate timing toward the target instead of mechanically editing the finished waveform.
- Pad: append silence only when output is shorter; it never speeds up or truncates longer speech.
- Trim: cut a longer assembled result to the exact target.
- Max consecutive silence tokens: 0 disables token trimming; use it only to suppress unusual model silences.
- Latent multiplier: the natural-duration factor passed to the engine; leave 1.72 unless running a controlled timing experiment.

### Formats, audio finishing, and execution mode

Give the task an optional safe filename, keep the used reference for reproducibility, and choose WAV, MP3, or still-image MP4 behavior. **Bypass** preserves model audio; the other tuning presets use FFmpeg post-processing with optional explicit overrides.

![Annotated 4K output, audio tuning, and execution settings](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/ethlsqm7ApVAs2Z-kO4pF.png)

*Figure 8. V6 system presets start with in-process reuse for faster repeat runs. Enable isolated subprocess mode when Cancel must terminate the complete model process and release its VRAM after the job; isolation trades away resident-model reuse.*

- MP3 bitrate 256k is a strong voice quality and file-size balance; WAV candidates always remain available.
- Edge-silence trimming removes only sufficiently long leading or trailing silence.
- Low cut, high cut, gain, LUFS, and de-ess values override the chosen tuning preset when supplied.
- Every non-bypass tuning preset preserves the source sample rate in V6.
- In-process reuse is the V6 default; isolated subprocess is an intentional per-preset opt-in.
- Section batch size 1 is safest. Raise it only within the active VRAM tier hint.
- Low-memory mode chooses sequential, aggressive-memory paths.
- Prevent VRAM accumulation clears autoregressive caches between sections; it is slower but useful for long books.
- Verbose logging exposes model inputs, timing, and detailed diagnostics in the terminal and live log.

## 5. Run, Cancel, Review, and Reuse Outputs

During a run, watch item number, elapsed time, ETA, speed, current stage, and the live log. **Cancel** stops new work and, in subprocess mode, can terminate the worker cleanly instead of leaving GPU memory occupied.

![Annotated 4K completed voice generation with progress and log](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/PsiHtcAc38qlJnRRrP0Hp.png)

*Figure 9. A successful run ends with a playable result and a final summary. **Open outputs folder** jumps to the task directory containing audio, media, reference copies, metadata, and logs selected by the output settings.*

If generation seems stuck, read the last log line before cancelling. First-time model loads and optional compilation can be much slower than later runs. If cancellation leaves a model resident in reuse mode, use **Models & Performance > Unload model / free VRAM**.

When **Candidates** is above one, listen to every player before choosing. The displayed seed lets you repeat or compare a run with the same stochastic starting point; settings and output metadata explain exactly what created each file.

![Annotated 4K candidate players, seed, metadata, and recent outputs](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/Ukxh_dDwKEkPqcIEGtoch.png)

*Figure 10. **Recent outputs** lists the last ten generated tasks. Select a good result and press **Load selected output into reference** to perform iterative voice cloning, continuation work, or a clean second-generation reference test.*

## 6. Batch Generation for Many Scripts

Build a queue from uploaded TXT/caption files, pasted text, or a local folder. The batch uses the shared Voice Generation settings, so first prove one representative item in the single-generation tab.

![Annotated 4K batch queue setup and naming controls](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/_ABFoS-mtJur8UEMAH-82.png)

*Figure 11. Naming supports `{index}`, `{name}`, and `{stem}`. Keep output in a safe subfolder under `outputs`, choose shared or same-stem per-file references, then choose cancellable subprocess execution or faster in-process reuse.*

1. Load mixed TXT, SRT, VTT, and SBV sources or point the folder field at a local collection.
2. Choose **Shared reference** to use the active Voice Generation reference for every item.
3. Choose **Per-file reference** when each text/caption file has a same-stem audio file beside it.
4. Enable **Continue after item errors** for unattended queues; failed rows are recorded while later items continue.
5. Press **Generate batch**, monitor the table and log, and use **Open batch folder** when complete.

The tested queue mixed a plain TXT item with an SRT item and completed both. Caption timing is now applied per item, so a subtitle entry can use its cues without incorrectly forcing the TXT entry through a caption-only path.

![Annotated 4K completed mixed TXT and subtitle batch](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/2clUP_6ItXDHc2_9Cvlat.png)

*Figure 12. Confirm that every requested item has a completed row, playable output, path, elapsed time, and no hidden error in the log. **Cancel batch** is safe at an item boundary and terminates the active subprocess when needed.*

## 7. Prepare a High-Quality Voice Dataset

### Inputs, names, and scanning

Dataset preparation accepts media files, folders, `metadata.csv`, or already segmented WAV+TXT folders. Use one safe dataset name and confirm the output root before scanning, especially when **Overwrite dataset** will later be enabled.

![Annotated 4K dataset source and destination controls](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/qn9JURSoVuQ45K7pannFH.png)

*Figure 13. Recursion discovers nested media. A fixed speaker name labels every segment; **Speaker from folder** instead uses each source parent folder, which is useful for a multi-speaker collection.*

Press **Scan inputs** before processing. The discovered-media table and statistics expose supported files, sidecar captions, durations, and warnings without spending time on transcription or cutting.

![Annotated 4K discovered-media scan report](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/zD92Ww9ij2YJBJKAK7STR.png)

*Figure 14. Fix missing paths, unsupported extensions, or accidental extra speakers at this stage. **Refresh** updates existing datasets, and **Open dataset folder** inspects a selected prepared dataset.*

### Transcripts and sentence-aligned segmentation

**Prefer sidecars** uses SRT/VTT/SBV text and timing when available and lets Whisper fill missing alignment. Sentence-aligned mode with Whisper word times is the recommended CUDA workflow because it preserves complete phrases instead of arbitrary waveform chunks.

![Annotated 4K transcript, Whisper, and segmentation controls](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/d5-Whc2PWD4iOAGIhUNab.png)

*Figure 15. Remove bracket notes such as `[music]`, deduplicate rolling captions, and drop repeated spoken sentences. The measured quality window is 4 to 20 seconds with a 14-second target; edge padding and silence snapping avoid clipped consonants.*

- Sentence boundaries require punctuation and are the cleanest option.
- Sentence-or-pause recovers more material at aligned-word pauses, with a small risk of less natural cuts.
- Maximum cue gap controls when nearby caption cues may merge.
- Minimum pause boundary controls how much silence is needed when punctuation does not provide an edge.
- Minimum and maximum word counts remove fragments and implausibly dense transcripts.

### Cleanup and objective quality gates

Trim leading and trailing silence before filtering, normalize loudness for consistent gradients, and retain the required 24 kHz sample rate. The remaining gates reject weak alignment, impossible speaking rates, very quiet audio, clipping, and excessive silence.

![Annotated 4K dataset cleanup and quality filters](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/4bWC_-ZXcnTPiOoslcPIC.png)

*Figure 16. Defaults are conservative voice-training safeguards: -20 LUFS leaves headroom, 40 dB is a gentle trim threshold, and a 0.001 clipping ratio permits at most 0.1 percent clipped samples. Leave maximum silence ratio blank to disable that optional whole-segment filter.*

Do not loosen several gates at once. Review rejected counts and playable segments, then change the single rule that is demonstrably excluding good speech. Bad transcripts or clipped recordings cannot be repaired by more training epochs.

### Destination, references, and deterministic smoke tests

Export reference candidates so training samples, validation, and later Voice Generation can reuse the cleanest clips. **Maximum segments** set to 0 processes everything; a small number is ideal for validating paths and settings before a long run.

![Annotated 4K dataset output, reference, and processing options](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/6JfZTfx3gymz71HvEjeEh.png)

*Figure 17. The preparation seed makes candidate ranking and randomized work reproducible. **Overwrite dataset** replaces the same named directory, so leave it off until the scan and destination are unquestionably correct.*

Press **Prepare dataset** and follow item progress, speed, ETA, stats, and the log. A successful summary reports accepted and rejected segments, total duration, manifests, reference candidates, and the output path.

![Annotated 4K completed dataset summary](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/OkyOnh9fLB8sgwy3C-Kyf.png)

*Figure 18. Completion means the files were produced, not that every segment is automatically good. Use the inspection panel next; **Cancel** is available throughout long extraction, alignment, and segmentation jobs.*

Sort through the prepared-segments table, inspect durations and warnings, and click representative rows to play the exact training waveform. Look for clipped words, wrong captions, background speakers, music, and long silence.

![Annotated 4K prepared segment inspection and audio player](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/iZpn1Qd4wlG-2pKpUYuYm.png)

*Figure 19. The waveform and duration distribution make outliers visible. A smaller, clean, correctly transcribed dataset normally beats a larger noisy one for a single voice.*

### Precompute the feature cache

For single-speaker narration that includes demonstrations or music, `tools/curate_voice_dataset.py` can audit a prepared dataset before caching. It checks subtitle/ASR agreement and CAMPPlus speaker similarity over whole clips and overlapping six-second windows. Clean speaker matches with disputed source transcripts receive a fresh transcription of the extracted clip. Every rejection is recorded; output must use a new directory. Reserve complete source recordings for validation and optionally final testing:

```text
python tools/curate_voice_dataset.py datasets/raw datasets/clean --reference reference.wav --validation-source held_out_tutorial --test-source final_test_tutorial
```

Source names are the media filename stems from the manifest. Similarity thresholds screen for contamination and can also reject quiet or atypical clean speech; inspect `quality_audit.jsonl` and `quality_summary.json`. The original dataset remains available. `tools/measure_grid_quality.py` measures a completed listening grid with ASR, speaker/style similarity, pitch and matched-text speaking rate when real target recordings are supplied.

Feature caching runs the expensive base-model preprocessing once and lets training consume compact cached samples. Choose the intended model directory and device, then press **Cache features now**. The semantic encoder uses FP32 to match inference. Cache entries track the audio, transcript and extraction assets; caching regenerates entries made with an older format or changed inputs.

![Annotated 4K training feature-cache workflow](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/9jmwh812EicQJF8oMPO2v.png)

*Figure 20. Monitor cache progress and failures just like preparation. If a cached sample exceeds text or semantic-code limits, correct or remove the source segment rather than hiding the failure.*

When caching completes in V6, the prepared dataset remains selected, its cached status refreshes immediately, and the Training tab receives the same dataset automatically. You no longer need to reselect it just to make the training summary notice the new cache.

## 8. Train a LoRA or DoRA Voice

### Adapter architecture

Select a prepared cached dataset, enter a safe adapter name, and choose LoRA or DoRA. DoRA is the measured quality default; LoRA uses slightly less compute. The recommended measured setup is rank 128 with alpha 129.

![Annotated 4K LoRA and DoRA adapter setup](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/WNDpzWV-lwGmgxlXtqVcf.png)

*Figure 21. Attention and MLP targets are recommended for voice fidelity. Speaker projection is a small fully trained module; emotion layers and the mel embedding/head are advanced additions that expand the trainable surface.*

Start with the default trainable modules. Increasing rank or enabling extra modules does not guarantee better speech; it increases file size, compute, and overfitting risk. The app inspects type, rank, and alpha when resuming so incompatible weights are rejected early.

### Optimizer, schedule, and effective updates

The measured batch-1 baseline uses AdamW, cosine decay, learning rate `4e-5`, 200 warmup steps, 10 epochs, batch size 1, and accumulation 1. With that setup, every training clip produces one optimizer update per epoch.

![Annotated 4K optimizer and learning-rate schedule](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/JCYJ1fqw-ZrSDKMJdd1Oh.png)

*Figure 22. `max_steps=0` derives the plan from epochs; use a tiny nonzero maximum for a smoke test. Gradient clipping limits spikes, while label smoothing and stronger weight decay should be changed only in response to measured validation behavior.*

- Fused AdamW can be faster on compatible CUDA builds; ordinary AdamW is the portable choice.
- Betas `0.9,0.99` and the default epsilon are established stable optimizer values.
- Mel loss is the main acoustic-token objective; text loss is auxiliary.
- Speaker reference `other` and emotion reference `follow_speaker` imitate normal inference better than self-conditioning.

### Validation and early stopping

Keep a validation split so the app measures unseen speech. The default holds out complete source recordings, requesting five percent of clips. With few recordings, the actual fraction can differ substantially; the training plan shows the resulting counts. A single recording falls back to a clip split. Explicit `split=train` and `split=val` labels in every manifest row override these settings.

![Annotated 4K validation split and early-stop controls](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/0FAKnqFq1Vk3rI9ITrUrH.png)

*Figure 23. Validation runs every 250 updates and at epoch boundaries by default. Maximum validation batches 0 evaluates the whole holdout; a positive cap uses a fixed shuffled subset.*

**Automatically stop when progress stalls** starts checked. After at least 1,000 updates, completed warmup and two dataset passes, six validation checks without a loss improvement greater than 0.005 stop training, even before the chosen epochs or steps. The lowest-loss checkpoint is preserved, including smaller improvements that do not reset patience. Uncheck the control to disable automatic stopping; patience 0 also disables it. Saved training state preserves the best step and patience counter when continuing a run. Both training and validation draw alternative references only from training clips.

When validation uses another clip as its reference, each held-out speaker needs at least one training clip. An invalid split reports the missing speakers instead of silently using the validation target as its own reference. Adjust the split or explicitly select self-reference validation if that is the intended evaluation.

Validation reference mode **other** uses a different same-speaker clip for both vectors and most closely matches real cloning. The training smoke test used four segments with a 25 percent split, producing three training items and one unseen validation item. If a tiny dataset produces zero validation items, V6 safely skips automatic measured checkpoint evaluation and records the reason in the log instead of starting an invalid evaluation job.

### Precision and low-VRAM training

BF16 base weights and BF16 mixed precision are recommended on modern NVIDIA cards. Gradient checkpointing trades extra computation for much lower activation memory and is required when training with GPT block swapping.

![Annotated 4K training precision, checkpointing, and block swap](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/rG72FzR-anPWpyqPrcR6I.png)

*Figure 24. INT8 ConvRot reduces frozen base-weight memory. Block swapping streams selected frozen GPT blocks through a CPU ring; pinned memory improves transfer speed, while ring size 1 uses the least VRAM.*

Use **Apply VRAM tier defaults** instead of guessing a low-memory combination. If training still OOMs, reduce batch size first, keep gradient checkpointing enabled, add block swap, and close other GPU applications. Do not compensate by raising gradient accumulation unless you intentionally want a different optimization regime.

### Saving, resume modes, and automatic evaluation

Save an epoch checkpoint so the best-sounding voice is not forced to be the final epoch. Keep-last 0 retains every epoch; **Save best** also tracks the lowest validation loss. BF16 adapter files are smaller, while FP32 preserves full update precision.

![Annotated 4K checkpoint saving, resume, and automatic analysis](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/-fcDUuIR348pOP-EU9lGc.png)

*Figure 25. **Weights only** loads an adapter into a fresh schedule at step 0. **Continue run** restores optimizer, scheduler, step, RNG, and data position when train state exists. Automatic analysis is CPU-only; measured checkpoint evaluation starts only after the training model releases GPU memory.*

- Save train state for exact continuation from best, final, or interrupted checkpoints.
- Save train state with every epoch only when you need to continue from an arbitrary epoch; it costs substantially more disk.
- Automatic evaluation may include the base model, a deterministic training subset, selected strengths, and a timeout that does not invalidate completed training.
- The recommended checkpoint is written into machine-readable and plain-language analysis files.

### Per-epoch listening samples

Enable training samples to hear progress at a fixed epoch interval. Keep one representative short sentence, one reference, one language, and one seed so differences come from the checkpoint rather than changing inputs.

![Annotated 4K primary training-sample settings](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/dbhbeyY5i_gQy6I7EmH_Y.png)

*Figure 26. The isolated sampling worker uses its own runtime tier, free-VRAM threshold, and timeout. Blank reference selects the dataset's best candidate automatically; Auto language follows the prepared dataset.*

Training samples mirror the important Voice Generation parameters: beams, temperature, top-p, top-k, repetition, emotion weight, diffusion steps, CFG, text-token ceiling, length penalty, mel-token ceiling, and speaking rate.

![Annotated 4K advanced per-epoch sampling controls](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/l1tfMBPsZhtO55dXgd6cU.png)

*Figure 27. Lock these values before training if you intend to compare epochs by ear. Otherwise a changed decoding setting can sound like a training improvement or regression.*

### Paths, seed, workers, and diagnostics

The final group sets output root, model directory and YAML, training device, attention backend, deterministic seed, data workers, and log frequency. Relative paths resolve from the app directory.

![Annotated 4K training paths, device, seed, and logging controls](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/TmOUZM2rM5AMIJWDOwN2L.png)

*Figure 28. Two workers is a safe Windows/Linux default; use zero when debugging loader problems. Log every step for a short run and less frequently only when an extremely fast long run makes logging expensive.*

### Start, stop, and read the live dashboard

Press **Start training** only after the optimizer-update plan looks sensible. The live dashboard reports epoch, optimizer step, loss, validation, learning rate, gradient norm, throughput, elapsed time, ETA, VRAM, samples, and the worker log.

![Annotated 4K live DoRA training dashboard](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/D5RqDxQcubEHWzZqHLo5M.png)

*Figure 29. **Stop** requests a graceful checkpointed stop. **Force stop** terminates a nonresponsive worker. Both paths were exercised; use force only after the normal stop path has had time to save.*

The real smoke run completed two optimizer steps on a DoRA rank-8 adapter, with a three-item train split and one-item unseen validation split. The final dashboard exposed its checkpoint, sample, measured values, and analysis actions.

![Annotated 4K completed training run and next actions](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/07bi-MH7-FANV_iN14zKr.png)

*Figure 30. Use **Open output folder** for artifacts, **Compare in grid** for controlled listening, and **Use best checkpoint** to send the recommendation to Voice Generation. Completion alone is not a reason to choose the final epoch.*

The completed run remains selected while analysis compares training and unseen loss. Read the recommendation, sustained-overfitting marker, and checkpoint table before deploying an adapter.

![Annotated 4K training-to-checkpoint-analysis handoff](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/TITZWaenDBXo9Uv8FMSuO.png)

*Figure 31. The handoff buttons avoid manual paths: compare the run in Checkpoint Grid or use its recommended checkpoint directly in generation.*

The manager inventories adapter type, rank, alpha, steps, dataset, recommendation, speaking rate, path, and health. **Refresh** rescans, **Open folder** inspects files, and **Delete** removes only after the UI confirmation flow.

![Annotated 4K LoRA and DoRA manager](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/eUEHK55Pd5ft5mBJqTGpq.png)

*Figure 32. Use the manager to distinguish a final adapter from epoch checkpoints and interrupted runs. A valid entry should have metadata and a readable safetensors file before it appears in Voice Generation.*

## 9. Analyze and Compare Checkpoints

### Generalization analysis and measured evaluation

Select one adapter folder and press **Analyze training log**. The CPU-only analysis compares train and validation trends, identifies the lowest unseen loss, and detects a sustained overfitting region rather than assuming the newest file is best.

![Annotated 4K checkpoint generalization analysis](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/PxQJJkB_8QuEh5MaVtgfr.png)

*Figure 33. A recommendation based on validation is a shortlist, not the final listening decision. Use the next measured and audible comparisons with the same references and text.*

The checkpoint table combines epoch/step identity, paths, validation values, verdicts, and the recommended marker. Base model provides a reference-only baseline when enabled.

![Annotated 4K checkpoint metrics and recommendation table](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/82AWPaUUt729FIHtW8dTC.png)

*Figure 34. **Use best checkpoint** updates the Voice Generation adapter choice. **Evaluate checkpoints now** starts an isolated inference-like measurement over the configured validation and training subsets.*

Evaluation references can match training validation or use a different clip of the same speaker, closer to ordinary voice cloning. The job fixes its examples and settings across every checkpoint.

![Annotated 4K live checkpoint evaluation](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/4anEExatEorUfut3PZir-.png)

*Figure 35. Live progress shows which row is loading and scoring. The test executed a real three-row comparison, not a mocked UI state.*

The completed measurement reports comparable scores and a best-checkpoint recommendation. Inspect failed rows and reference selection before trusting a ranking.

![Annotated 4K completed checkpoint evaluation](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/u73fKB0RmaQGy4zJIx9Bt.png)

*Figure 36. Use measured evaluation to narrow the field, then rely on the listening grid for intelligibility, identity, pacing, emotion, noise, and artifacts that one scalar cannot fully express.*

### Create a fair listening grid

Choose base, recommended, final, and selected epoch checkpoints; add strengths from 0 to 4; then provide one text per line and one reference path per line. Every cell receives the same language, text segmentation, reference, and seed.

![Annotated 4K listening-grid setup](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/o-G0UJNe52bahnm4Yhnwy.png)

*Figure 37. **Use LoRA / DoRA reference** inserts the stored reference. **Add dataset reference candidates** adds clean prepared clips. The runtime estimate grows with checkpoints x strengths x texts x references.*

Lock temperature, top-p, top-k, beams, repetition penalty, diffusion steps, CFG, segment budget, emotion weight, and speaking rate before running the grid.

![Annotated 4K shared listening-grid sampling settings](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/FRf0UoSkkfbK8tz61clNH.png)

*Figure 38. Seed -1 draws one random seed at grid start and then reuses it for every cell. Enter an explicit seed when a grid must be recreated later.*

Press **Generate grid** and watch cell-by-cell progress. The worker saves state under `outputs/grids`, so the log, table, audio, and metadata stay together.

![Annotated 4K listening-grid generation in progress](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/KiDQJ_WDVHyVt0rojeI3R.png)

*Figure 39. **Cancel** stops the active grid worker; completed cells remain useful evidence. Do not change runtime/model settings in another tab while an isolated comparison is loading.*

Completion summarizes every generated cell, its checkpoint, strength, text/reference index, seed, duration, and path. Use **Open grid folder** to inspect the saved Markdown, metrics, log, and audio files.

![Annotated 4K completed listening-grid table](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/TCOccvx3QM5LwWfFk6yd5.png)

*Figure 40. A complete table verifies coverage; the audible result view is where the final voice decision happens.*

Listen horizontally with the same text/reference and vertically across base, best, final, epochs, or strengths. The saved production grid was reloaded successfully and displayed all 12 audio players.

![Annotated 4K side-by-side grid audio players](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/f-PwFx1LuQG_bjxRXePMj.png)

*Figure 41. Rate each row for identity, pronunciation, prosody, background noise, breath artifacts, repeated syllables, and consistency. Prefer the earliest clean checkpoint when later gains are negligible and overfitting begins.*

### Calibrate the voice's real speaking pace

Press **Calibrate speaking rate from this grid** after generating representative audio. The app compares generated words per second with the voice's training recordings and saves a per-adapter multiplier.

![Annotated 4K speaking-rate calibration result](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/yXNqQKNzturKNPbl-y8hM.png)

*Figure 42. Voice Generation can automatically apply the stored value when that adapter is selected. The production calibration measured about 0.944, while a separate smoke calibration proved the full write-and-reload path.*

## 10. Fit Runtime to the GPU

### Inventory, tiers, and applying settings

Refresh GPU inventory, select the actual device, and choose Auto or a 6, 8, 10, 12, 16, 24, or 32 GB tier. Every tier is a coordinated set of model, swapping, cache, and batch hints designed to retain roughly 2 GB of headroom.

![Annotated 4K GPU inventory and VRAM tier controls](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/Mm1439DIwxoKjRNJJpYYY.png)

*Figure 43. **Apply runtime** commits the current model and memory settings. Changing a dropdown alone does not rebuild an already loaded model; unload before a major variant or device switch.*

### BF16, INT8, attention, and acceleration

BF16 is the official quality path on modern NVIDIA GPUs. INT8 ConvRot reduces GPT weight memory while retaining the required IndexTTS 2.5 codec and vocoder stack. FP32 is primarily a CPU-compatible fallback.

![Annotated 4K model variant and compute controls](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/A6H70wFnvDZNnH54D7C8H.png)

*Figure 44. SDPA is the compatible attention default. FlashAttention 2 needs its optional package. The acceleration engine uses the fast CUDA-graph/flash path and expects beams 1; torch compilation trades a slower first run for repeated-workload speed.*

- Enable the Qwen emotion model for Emotion text mode; on-demand residency avoids eager startup cost.
- BigVGAN CUDA kernel uses the optional fused activation kernel when available.
- CFM estimator BF16 autocast saves activation VRAM and can slightly change output.
- DeepSpeed is a legacy optional loader and should stay off unless installed intentionally.

### GPT block swapping

Zero keeps every GPT block resident. Minus one lets the runtime fit automatically; positive values up to 24 stream that many blocks from CPU RAM. More swapping reduces VRAM but increases transfer overhead.

![Annotated 4K GPT block swap and memory controls](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/JrtHVqiD4R10m54RU7Cbd.png)

*Figure 45. Ring size 2 overlaps copies and compute; ring size 1 minimizes memory. Keep pinned swap memory on for faster CPU-to-GPU transfer unless the host cannot lock enough RAM.*

Runtime CFM cache length sets the upper reservation, VRAM reserve protects against allocator spikes, and the section-batch hint informs Voice Generation. Treat the hint as an advisory maximum, not a promise when another process is using the GPU.

Selecting `cpu` now shows **CPU diagnostics mode** rather than a misleading GPU fit estimate. GPU VRAM limits do not apply there, and FP32 is the compatibility-oriented choice; CPU synthesis is primarily a diagnostic fallback and is much slower than CUDA.

### Place auxiliary models independently

Semantic model, Qwen emotion, CampPlus, semantic codec, S2Mel, and BigVGAN can each stay on GPU, stay on CPU, or load on demand according to the choices provided by the active tier.

![Annotated 4K auxiliary-model residency controls](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/HUamK54i2mJPqv4cy07z_.png)

*Figure 46. Keep frequently used synthesis stages resident when VRAM allows. Move optional or infrequent models off GPU first when reducing idle footprint.*

### Verify and download model files

The model table checks required base and optional INT8 files. **Download / verify base models** repairs the official stack; **Download INT8 model** fetches the memory-efficient GPT; **Refresh file status** rechecks disk; **Open model folder** opens the configured directory.

![Annotated 4K model-file verification and download actions](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/MXMRejn7FJX5ZoRc3NSAc.png)

*Figure 47. The INT8 path now has a verified fallback: if the hosted artifact is unavailable, the downloader can convert the local BF16 model instead of presenting a feature that cannot be obtained.*

### Run the isolated VRAM benchmark

Select a tier and press **Run VRAM benchmark** when the chosen GPU is otherwise idle. The UI uses a short deterministic calibration - beams 1, 60 text tokens, batch 1 - while the standalone CLI can still run a heavier stress profile.

![Annotated 4K VRAM benchmark setup](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/kaPbvlrCPR1v0LIi5cKQX.png)

*Figure 48. The benchmark uses a bundled reference voice, runs in an isolated subprocess, and checks whether the selected tier truly fits rather than estimating from file size alone.*

Watch the subprocess log for model load, warmup, synthesis, peak allocation, and fit status. On Windows WDDM, the idle gate allows the larger of 1 GB or 10 percent of physical VRAM so driver accounting does not falsely block an idle card.

![Annotated 4K live isolated VRAM benchmark](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/qebkCrvOtopB2eO57K7Yi.png)

*Figure 49. A benchmark can be cancelled without contaminating the main app model because it owns a separate process.*

The real 32 GB BF16 test fit successfully on the RTX 5090: 28.714 seconds of audio, real-time factor about 0.458, and peak allocated VRAM about 6.212 GB for the short calibration workload.

![Annotated 4K completed benchmark with measured fit and VRAM](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/0K0NsJJcIyxf-kCi2_6W5.png)

*Figure 50. Read **fit**, peak VRAM, RTF, and any error together. Apply the tier after a pass, then use **Unload model / free VRAM** whenever you want to release the in-process runtime.*

## 11. Built-in Help and Recovery

The Help tab repeats the shortest successful path for a first clone and explains the reference-only and trained-adapter workflows. Use it as an in-app checklist while this longer guide stays open separately.

![Annotated 4K built-in quick-start help](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/JT_eJ_Ms_NKD4OmB-GXhk.png)

*Figure 51. The best diagnostic habit is to return to one clean reference, one short sentence, the quality preset, base model, and section batch size 1.*

Workflow help covers speaking-rate calibration, checkpoint comparison, VRAM tiers, and the main parameter families. Follow a complete workflow rather than changing unrelated controls until the symptom disappears.

![Annotated 4K workflow and parameter guidance](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/LMSfG53C_GaT2mmSSYa_4.png)

*Figure 52. Presets make experiments repeatable: duplicate a known-good user preset and change one variable for each A/B comparison.*

The final help area documents pause syntax, reference guidance, links, and recovery steps. Copy a supported pause form exactly and check the live section preview before generation.

![Annotated 4K pause syntax and troubleshooting reference](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/WHA-Z6ilALhbSRxLlvIZx.png)

*Figure 53. For OOM: unload, lower batch size, choose a lower tier or INT8, add block swap, shorten sections, and rerun the benchmark. For bad speech: first inspect reference quality, language, text segmentation, seed, and adapter strength.*

### Read the V6 release history

The lazy-rendered **Changelog** tab follows Help. Open it to read the newest-first v6.2 through v4.0 release notes, including fixes that may affect an older workflow, and to reach the official [SECourses Patreon](https://www.patreon.com/SECourses) and [GitHub repository](https://github.com/FurkanGozukara/Premium_IndexTTS2_SECourses). The tab was added after the original V5 screenshot set, so it is documented here rather than shown in those captures.

## 12. Presets, Themes, and Repeatable Work

A universal preset stores every registered setting across every tab. Select a system preset for a protected baseline, or type a new name and press **Save** to create a user preset. Selecting an existing user preset allows an intentional overwrite.

![Annotated 4K universal preset management](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/G7FIvkLqWtjxZbcnQKPJm.png)

*Figure 54. **Load** applies a preset, **Delete** requires the confirmation flow and only removes user presets, **Reset** restores defaults, and **Load last values** recovers the most recent working state. Fresh installs select the quality system preset; upgrades retain the last-used preset. Unknown old keys are ignored and missing new keys receive defaults.*

Press **Light / dark theme** for the preferred presentation. Theme changes the interface only; it does not change audio, model precision, presets, or output files.

![Annotated 4K light-theme workspace](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/gFrBF57-R_W5CkGaSJXJm.png)

*Figure 55. The 4K capture confirms the same dense workspace remains legible in light mode. Switch back at any time without restarting a job.*

## 13. Deploy a Trained Voice and Use Advanced Controls

### Load, scale, and merge an adapter

Choose an adapter in Voice Generation. Its summary exposes type, rank, alpha, steps, recommendation, and saved metadata. Strength 1.0 is the trained scale; lower values are subtler and higher values are stronger.

![Annotated 4K trained LoRA/DoRA voice controls](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/4yCLpfHrrLta-jlYHsX-2.png)

*Figure 56. Auto-reference and auto-speaking-rate remove two common deployment mistakes. BF16 can temporarily merge the adapter into base weights for speed; the app restores base weights before switching, and INT8 stays on the unmerged path.*

### Emotion-reference mode

Select **Emotion reference audio**, upload a short expressive clip, and set emotion weight. Keep the identity reference separate: this mode is for transferring delivery, not replacing the speaker.

![Annotated 4K emotion-reference audio mode](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/iCS_5s4SCfv0ub4O52mVn.png)

*Figure 57. Use the maximum emotion-reference length to discard unneeded material and compare at a fixed seed before increasing the blend.*

### Manual-vector mode

Select **Manual vectors** and blend the eight channels. Start with one dominant channel, apply tuned biases, and keep the sum limit at 0.8 before attempting compound emotions.

![Annotated 4K manual emotion vector mode](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/1byx2ELE2lNwk6Xu4hnC_.png)

*Figure 58. A vector is continuous rather than a label: 0.2 calm plus 0.1 sadness is a different conditioning target from a full-strength sadness instruction.*

### Natural-language emotion mode

Select **Emotion text** and describe the intended performance in plain language. Keep it specific and short, such as `restrained concern, then quiet relief`, rather than writing a second script.

![Annotated 4K natural-language emotion instruction](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/fgTqeAz3Lc418cRrlgpbr.png)

*Figure 59. Qwen emotion must be enabled. The manual vector limits and biases remain available because the analyzed instruction ultimately becomes bounded emotion conditioning.*

The tested app produced real outputs for emotion-reference, manual-vector, and emotion-text modes in addition to same-speaker emotion. Inspect the log to confirm which source loaded and listen against a neutral fixed-seed control.

![Annotated 4K successfully completed emotional generation](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/s5SoQ3I78ChuRwZIt4oJ5.png)

*Figure 60. A completed player, output path, duration, settings summary, and clean final status prove the selected emotion mode reached actual synthesis rather than only changing the UI.*

### Whole-output target duration

Enter a whole-output target only when the application has a real timing requirement. **Natural** changes synthesis timing, **Pad** only adds tail silence, **Trim** only cuts, and **Off** leaves the result unconstrained.

![Annotated 4K target-duration mode controls](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/TqPyJIBONN2pEgP2RLgRF.png)

*Figure 61. Combine duration targets cautiously with caption cue timing, speaking rate, section silence, and explicit pause tags because all of them influence the assembled timeline.*

All three active paths were generated. A 3-second Natural target produced about 2.995 seconds; a 1.5-second Trim target produced exactly 1.500 seconds; a 4-second Pad target remained about 4.156 seconds because the spoken source was already longer, which is correct pad-only behavior.

![Annotated 4K completed duration-controlled output](https://cdn-uploads.huggingface.co/production/uploads/6345bd89fe134dfd7a0dba40/sapqcjV86Ocp52JWcrZEF.png)

*Figure 62. Always read the final measured duration rather than assuming every mode forces equality. Choose Natural for adaptive timing, Trim for a hard upper bound, and Pad for a hard minimum without speeding or cutting speech.*

## 14. Practical Recipes

### Clean first clone

1. Select the quality system preset and Base model.
2. Use one clean 8-15 second speaker reference and one short script in the correct language.
3. Keep same-speaker emotion, natural rate 1.0, seed fixed, one candidate, and section batch 1.
4. Generate, inspect the log, and save the good result plus used reference.

### Long audiobook or course narration

1. Calibrate the trained voice's speaking rate and enable automatic rate loading.
2. Use punctuation, pause tags, and a conservative per-segment token budget.
3. For long unattended jobs where complete post-job VRAM release matters, enable subprocess mode; also enable prevent-VRAM-accumulation when segment-to-segment cache growth is the problem.
4. Test one chapter, save a user preset, then run folders through Batch with deterministic naming and continue-on-error.

### Subtitle-timed localization video

1. Upload the translated SRT/VTT/SBV and enable caption cue timing.
2. Choose Natural duration behavior for model-based timing or Trim/Pad only when mechanical bounds are acceptable.
3. Add a still image for MP4, generate one representative cue set, and check starts and final duration.
4. Batch the remaining caption files only after the representative item passes.

### Train and deploy a new voice

1. Scan media and captions, prepare 4-20 second sentence-aligned 24 kHz clips, inspect them, and cache features.
2. Train the measured DoRA baseline with validation, samples, epoch checkpoints, train state, and automatic analysis.
3. Evaluate base plus saved checkpoints with inference-like references, then generate a fixed-seed listening grid.
4. Select the best audible checkpoint, calibrate speaking rate, and enable automatic reference/rate in Voice Generation.

### Recover from low VRAM or OOM

1. Cancel the job and use **Unload model / free VRAM**.
2. Close other GPU workloads and select the matching conservative VRAM tier.
3. Use INT8, more block swap, ring 1, lower CFM cache, on-demand auxiliary models, and section batch 1.
4. Shorten text sections, keep beams 1, apply runtime, and pass the isolated benchmark before retrying.

### Reproducible A/B test

1. Save a baseline user preset.
2. Fix reference, text, language, seed, checkpoint, strength, emotion source, and runtime.
3. Change exactly one control, generate one or more candidates, and record the output metadata.
4. Use a listening grid when comparing checkpoints or strengths so every cell is locked automatically.

## 15. Verified Behavior and Repairs Included in This Build

This tutorial was not produced from screenshots alone. The original audit completed real synthesis, batching, preparation, feature caching, training, checkpoint evaluation, listening grids, calibration, model verification, and benchmarking. The V6 maintenance audit rebuilt the live interface, checked the changed controls and Changelog in Chrome, and passed **288 tests**, with 37 environment-specific cases skipped and 15 upstream PyTorch deprecation warnings.

The V6.1 training update passed **307 tests**, with 37 optional GPU cases skipped. Separate CUDA checks, the full V4 training run, independent checkpoint evaluation, and 93 generated audio comparisons also completed. The [training quality report](TRAINING_QUALITY_REPORT_2026-09-06.md) records the dataset, evaluation protocol, results, and limitations.

- Mixed TXT/SRT batches now apply caption timing only to caption items instead of crashing the plain-text item.
- Batched inference no longer leaks sequential target-duration arguments into the batch engine.
- Saved listening grids retain and render their audio players, and idle polling no longer clears a loaded result.
- Dynamically loaded checkpoint selections now survive dependent events and timer polling instead of failing Gradio validation against an empty startup choice list.
- INT8 download now has a local BF16-to-INT8 fallback when the hosted artifact is absent.
- The Windows benchmark idle gate tolerates normal WDDM reservation, uses a bundled reference, and the UI runs a practical short calibration profile.
- V6 restores microphone reference recording and makes arbitrary local audio/video paths browser-safe, including cached previews for codecs the browser cannot play directly.
- Dynamic candidate and dataset-reference players render after reload, feature caching refreshes the training handoff, and completed batch summaries are no longer overwritten by a polling race.
- Acceleration now honors disabled top-k/top-p limits, preserves stop tokens and compute dtype, and surfaces internal failures instead of silently returning an empty result.
- Audio tuning preserves sample rate, text-normalization failures retain the original fragment, zero-item validation skips automatic evaluation cleanly, and CPU mode no longer claims a GPU VRAM fit.
- The Changelog tab renders only when opened and presents the public v6.2-to-v4.0 history plus official project links without slowing initial tab rendering.
- Every final annotated image passed an exact 3840 x 2160 dimension gate and was individually uploaded to the dedicated Hugging Face discussion.
- The repaired selectable copy source passed a complete Patreon paste: all 62 hosted images became full-width native image blocks with all 62 alt texts, and the headings, lists, links, and final paragraph were retained.

> A feature is treated as complete only when its control exists, its workflow reaches a real result, its failure/cancel path is understandable, and the tutorial shows where the user acts and where proof appears.

## 16. Complete Action Reference

These are the non-setting actions and result surfaces a regular user will encounter. The numbered guide above shows each in context.

**Header:** Load last values; open/close all sections; switch theme; save, load, delete, or reset a universal preset.

**Voice Generation:** Upload/load/clear a reference; record a microphone reference; extract time ranges; refresh the reference library; auto-select token budget; upload captions and a still image; generate; cancel; open outputs; refresh adapters; play generated/candidate audio; inspect recent outputs; load a selected output as a new reference.

**Batch Generation:** Upload text/caption files; generate the queue; cancel; open the batch folder; read progress; play results; inspect the per-item table.

**Dataset Preparation:** Refresh/open existing datasets; upload media; scan; prepare; cancel; cache features; open output; inspect discovered media and stats; select and play prepared segments.

**Training:** Refresh datasets; apply tier defaults; refresh resume sources; start; graceful stop; force stop; open output; compare in grid; use best checkpoint; inspect charts/sample/checkpoints; refresh/open/delete adapter-manager entries.

**Checkpoint Grid:** Refresh adapters; analyze logs; evaluate checkpoints; use best; use stored reference; add dataset candidates; upload a reference; generate/cancel a grid; open saved grid; calibrate speaking rate; play every result cell.

**Models & Performance:** Refresh GPU inventory; apply runtime; unload/free VRAM; download INT8; download/verify base models; refresh file status; open model folder; run the isolated benchmark.

**Help:** Read the quick starts, workflow guidance, parameter glossary, pause syntax, troubleshooting steps, and launch arguments.

**Changelog:** Open the newest-first v6.2-to-v4.0 release history and follow the official Patreon or GitHub project links.

## 17. Every Registered Setting

The appendix below is generated directly from the app's preset registry. It covers all 253 registered controls, including current defaults, ranges, choices, and the help text shown by the UI. A preset stores these values across tabs.

### Voice Generation - 70 settings

**Language** - `generation.language`. Language code used by text normalization and pronunciation. *(default "EN"; choices "ZH", "EN", "JA", "AR", "ES")*

**Max tokens per segment** - `generation.max_text_tokens_per_segment`. Per-language defaults are recommended; shorter segments use less VRAM. *(default 60; minimum 20; maximum 300)*

**Use caption cue timing** - `generation.use_caption_timing`. Retimes each caption unit to its cue slot and preserves cue start times. *(default false)*

**Auto-load the LoRA / DoRA recommended reference audio** - `generation.auto_lora_reference`. Loads the LoRA / DoRA's saved reference whenever no manual Reference Voice is selected. *(default true)*

**Auto-apply the LoRA / DoRA calibrated speaking rate** - `generation.auto_lora_speaking_rate`. Uses the selected voice's measured pace; selecting None resets speaking rate to 1.0. *(default true)*

**Emotion source** - `generation.emotion_mode`. Use the speaker tone, another reference, eight manual vectors, or emotion text analysis. *(default "Same as speaker voice"; choices "Same as speaker voice", "Emotion reference audio", "Emotion vector", "Emotion text")*

**Emotion weight** - `generation.emotion_weight`. 0 keeps more speaker emotion; 1 follows the selected emotion source fully. *(default 0.65; minimum 0; maximum 1)*

**Random emotion exemplar** - `generation.emotion_random`. Randomizes the internal exemplar used with manual emotion vectors. *(default false)*

**Emotion description** - `generation.emotion_text`. Used only in Emotion text mode; blank analyzes the speech text itself.

**Joy** - `generation.emotion_joy`. Manual joy strength. *(default 0; minimum 0; maximum 1)*

**Anger** - `generation.emotion_anger`. Manual anger strength. *(default 0; minimum 0; maximum 1)*

**Sadness** - `generation.emotion_sad`. Manual sadness strength. *(default 0; minimum 0; maximum 1)*

**Fear** - `generation.emotion_fear`. Manual fear strength. *(default 0; minimum 0; maximum 1)*

**Disgust** - `generation.emotion_disgust`. Manual disgust strength. *(default 0; minimum 0; maximum 1)*

**Depression** - `generation.emotion_depression`. Manual depression strength. *(default 0; minimum 0; maximum 1)*

**Surprise** - `generation.emotion_surprise`. Manual surprise strength. *(default 0; minimum 0; maximum 1)*

**Calm** - `generation.emotion_calm`. Manual calm strength. *(default 0; minimum 0; maximum 1)*

**Apply tuned emotion biases** - `generation.apply_emotion_bias`. Recommended balancing prevents several emotion channels from dominating. *(default true)*

**Maximum vector sum** - `generation.max_emotion_sum`. 0.8 is the model-tuned recommendation; larger values can sound exaggerated. *(default 0.8; minimum 0.1; maximum 2)*

**Joy bias** - `generation.emotion_bias_joy`. Multiplier applied before the vector sum limit. *(default 0.9375; minimum 0.5; maximum 1.5)*

**Anger bias** - `generation.emotion_bias_anger`. Multiplier applied before the vector sum limit. *(default 0.875; minimum 0.5; maximum 1.5)*

**Sadness bias** - `generation.emotion_bias_sad`. Multiplier applied before the vector sum limit. *(default 1; minimum 0.5; maximum 1.5)*

**Fear bias** - `generation.emotion_bias_fear`. Multiplier applied before the vector sum limit. *(default 1; minimum 0.5; maximum 1.5)*

**Disgust bias** - `generation.emotion_bias_disgust`. Multiplier applied before the vector sum limit. *(default 0.9375; minimum 0.5; maximum 1.5)*

**Depression bias** - `generation.emotion_bias_depression`. Multiplier applied before the vector sum limit. *(default 0.9375; minimum 0.5; maximum 1.5)*

**Surprise bias** - `generation.emotion_bias_surprise`. Multiplier applied before the vector sum limit. *(default 0.6875; minimum 0.5; maximum 1.5)*

**Calm bias** - `generation.emotion_bias_calm`. Multiplier applied before the vector sum limit. *(default 0.5625; minimum 0.5; maximum 1.5)*

**Sample** - `generation.do_sample`. Recommended for natural variation; disable for greedy/beam decoding. *(default true)*

**Temperature** - `generation.temperature`. 0.8 balances expressiveness and stability. *(default 0.8; minimum 0.1; maximum 2)*

**Top-p** - `generation.top_p`. Nucleus sampling threshold; 0.8 is recommended. *(default 0.8; minimum 0; maximum 1)*

**Top-k** - `generation.top_k`. Candidate token cutoff; 0 disables top-k filtering. *(default 30; minimum 0; maximum 100)*

**Beams** - `generation.num_beams`. More beams can improve stability but increase time and VRAM. *(default 3; minimum 1; maximum 10)*

**Repetition penalty** - `generation.repetition_penalty`. 10 is the established model default. *(default 10; minimum 1; maximum 20)*

**Length penalty** - `generation.length_penalty`. Only affects beam search; 0 is neutral. *(default 0; minimum -2; maximum 2)*

**Max mel tokens** - `generation.max_mel_tokens`. Upper limit on generated semantic tokens per section. *(default 1500; minimum 50; maximum 1815)*

**Seed** - `generation.seed`. -1 chooses a fresh random seed; reuse a shown seed for repeatability. *(default -1; minimum -1; maximum 4294967295)*

**Candidates** - `generation.num_candidates`. Generates consecutive seeded alternatives; each adds generation time. *(default 1; minimum 1; maximum 8)*

**Diffusion steps** - `generation.diffusion_steps`. 25 is the quality default; 12-16 is faster and 35-50 can refine difficult audio. *(default 25; minimum 2; maximum 100)*

**CFG rate** - `generation.inference_cfg_rate`. 0.7 is recommended; high values follow conditioning more aggressively. *(default 0.7; minimum 0; maximum 2)*

**CFM temperature** - `generation.cfm_temperature`. 1.0 is the best-quality default; lower values reduce diffusion variation. *(default 1; minimum 0; maximum 2)*

**CFM cache length** - `generation.cfm_cache_length`. 8192 fits typical sections; lower values reduce reserved VRAM. *(default 8192; minimum 1024; maximum 32768)*

**Non-CJK token budget scale** - `generation.segment_budget_scale_non_cjk`. 0.72 leaves room for subword expansion in English, Arabic, and Spanish. *(default 0.72; minimum 0.3; maximum 1)*

**Section silence (ms)** - `generation.interval_silence`. Silence inserted between generated text sections; cue timing overrides this to zero. *(default 200; minimum 0; maximum 2000)*

**Max consecutive silence tokens** - `generation.max_consecutive_silence`. 0 disables token trimming; use only to suppress unusually long model silences. *(default 0; minimum 0; maximum 200)*

**Latent multiplier** - `generation.latent_multiplier`. 1.72 is natural duration; the runner converts this to the engine duration factor. *(default 1.72; minimum 0.5; maximum 3)*

**Speaking rate** - `generation.speaking_rate`. 1.0 is the model's natural pace; below 1.0 speaks slower, above 1.0 faster. A trained LoRA / DoRA can carry a calibrated value that matches the speaker's real pace. *(default 1; minimum 0.5; maximum 1.5)*

**Target duration (seconds)** - `generation.target_duration_s`. Leave blank unless a whole-output duration target is needed. *(default blank; minimum 0.1; maximum 3600)*

**Target duration mode** - `generation.target_duration_mode`. Natural regenerates timing; pad/trim only adjust the assembled result. *(default "off"; choices "off", "natural", "pad", "trim")*

**Enable pause tags** - `generation.enable_pause_tags`. Parses inline pause tags before tokenization. *(default true)*

**Text normalization** - `generation.text_normalization`. Recommended: expands and normalizes text before phonetic processing. *(default true)*

**Maximum speaker audio length (s)** - `generation.max_speaker_audio_length`. 15 seconds preserves enough identity without wasting reference compute. *(default 15; minimum 3; maximum 90)*

**Maximum emotion audio length (s)** - `generation.max_emotion_audio_length`. 15 seconds is recommended for an emotion reference. *(default 15; minimum 3; maximum 90)*

**Semantic layer** - `generation.semantic_layer`. Layer 17 is trained and recommended; changing it alters reference embeddings. *(default 17; minimum 1; maximum 24)*

**Reuse speaker conditioning for emotion** - `generation.reuse_spk_cond_for_emo`. Faster default-emotion path; enable when no separate emotion source is used. *(default false)*

**Output filename** - `generation.output_filename`. Optional safe basename; task numbering is used when blank.

**Save used reference** - `generation.save_used_audio`. Copies the active Reference Voice into the task folder for reproducibility. *(default false)*

**Save MP3** - `generation.save_as_mp3`. Converts the final output to MP3; WAV candidates remain available. *(default false)*

**MP3 bitrate** - `generation.mp3_bitrate`. 256k is a strong quality/size balance for voice. *(default "256k"; choices "128k", "192k", "256k", "320k")*

**Audio tuning preset** - `generation.audio_tuning_preset`. Bypass preserves model audio exactly; other presets use FFmpeg post-processing. *(default "bypass"; choices "bypass", "voice_clarity", "clear_narration", "deharsh", "warm", "normalize")*

**Trim edge silence threshold (ms)** - `generation.trim_silence_ms_threshold`. 0 disables trimming; only edge silence at least this long is removed. *(default 0; minimum 0; maximum 3000)*

**Low cut (Hz)** - `generation.tuning_low_cut_hz`. Optional high-pass cutoff; leave blank to use the preset. *(default blank; minimum 20; maximum 500)*

**High cut (Hz)** - `generation.tuning_high_cut_hz`. Optional low-pass cutoff; leave blank to use the preset. *(default blank; minimum 1000; maximum 24000)*

**Gain (dB)** - `generation.tuning_gain_db`. Optional final gain before limiting. *(default blank; minimum -24; maximum 24)*

**Loudness target (LUFS)** - `generation.tuning_loudnorm_i`. Optional integrated loudness normalization target. *(default blank; minimum -30; maximum -5)*

**De-ess amount** - `generation.tuning_deess`. Optional attenuation around sibilance frequencies. *(default blank; minimum 0; maximum 12)*

**Use isolated subprocess** - `generation.use_subprocess`. Recommended: cancellation can terminate the complete model process and release VRAM. *(default false)*

**Section batch size** - `generation.section_batch_size`. 1 is safest; use the active VRAM tier hint before increasing this. *(default 1; minimum 1; maximum 16)*

**Low memory mode** - `generation.low_memory_mode`. Uses sequential paths and aggressive memory behavior for constrained GPUs. *(default false)*

**Prevent VRAM accumulation** - `generation.prevent_vram_accumulation`. Clears autoregressive caches between segments; slower but useful for long jobs. *(default false)*

**Verbose logging** - `generation.verbose`. Prints detailed model inputs and timing diagnostics to console and the live log. *(default false)*

### Batch Generation - 5 settings

**Naming pattern** - `batch.naming_pattern`. Supports {index}, {name}, and {stem}; the extension is selected by Output settings. *(default "{index:03d}_{name}")*

**Output subfolder** - `batch.output_subfolder`. A safe subfolder below outputs; task folders are created inside it. *(default "batch")*

**Reference mode** - `batch.reference_mode`. Per-file mode looks for a same-stem audio file beside each text/caption file. *(default "One reference for all"; choices "One reference for all", "Per-file reference")*

**Batch execution** - `batch.execution`. Subprocess is easiest to cancel; reuse is fastest after the first in-process load. *(default "Subprocess per item"; choices "Subprocess per item", "Reuse loaded model between items", "Reload in-process model per item")*

**Continue after item errors** - `batch.continue_errors`. Records a failed row and proceeds to the next item instead of ending the batch. *(default true)*

### Dataset Preparation - 45 settings

**Input files or folders** - `dataset.inputs`. Accepts media files, folders, metadata.csv, or pre-segmented WAV+TXT folders.

**Dataset name** - `dataset.name`. A single safe directory name created below Output root. *(default "voice_dataset")*

**Output root** - `dataset.output_root`. Dataset parent directory; relative paths resolve from the application folder. *(default "datasets")*

**Scan recursively** - `dataset.recursive`. Includes supported media in nested input folders. *(default true)*

**Language** - `dataset.language`. Language stored in manifest rows and used for Whisper. *(default "EN"; choices "ZH", "EN", "JA", "AR", "ES")*

**Speaker name** - `dataset.speaker_name`. Optional fixed speaker label stored in each segment.

**Speaker from folder** - `dataset.speaker_from_folder`. Uses each source parent folder name as the speaker label. *(default false)*

**Transcript policy** - `dataset.subtitle_policy`. Prefer sidecars is recommended; Whisper fills missing timing/text. *(default "prefer_sidecar"; choices "prefer_sidecar", "whisper_only", "sidecar_only")*

**Whisper model** - `dataset.whisper_model`. Hugging Face model id or local path used when transcription/alignment is needed. *(default "openai/whisper-large-v3-turbo")*

**Whisper device** - `dataset.whisper_device`. CUDA device recommended for sentence alignment; CPU works but is much slower. *(default "cuda:0")*

**Segmentation mode** - `dataset.segmentation_mode`. Sentence aligned uses Whisper word times with caption sentences and is recommended on CUDA. *(default "sentence_aligned"; choices "auto", "sentence_aligned", "cue_boundaries", "whisper_only")*

**Force Whisper alignment** - `dataset.align_with_whisper`. Compatibility alias that forces sentence_aligned mode. *(default false)*

**Remove bracket annotations** - `dataset.remove_bracket_annotations`. Removes caption notes such as [music] and [applause]. *(default true)*

**Deduplicate rolling captions** - `dataset.dedupe_rolling_captions`. Removes repeated text from live/rolling subtitle cues. *(default true)*

**Drop duplicate sentences** - `dataset.drop_duplicate_sentences`. Keeps one copy (best aligned) of every sentence that is spoken more than once, e.g. repeated intros or outros; recommended for voice training. *(default true)*

**Target seconds** - `dataset.target_s`. 14 seconds packs whole sentences into inference-length clips; measured best with a 20 second maximum. *(default 14; minimum 1; maximum 30)*

**Minimum seconds** - `dataset.min_s`. 4 seconds keeps enough voice context while retaining the measured quality range. *(default 4; minimum 0.5; maximum 15)*

**Maximum seconds** - `dataset.max_s`. 20 seconds covers 12-15-second inference segments and retains more source audio; 30 seconds measured worse. *(default 20; minimum 2; maximum 40)*

**Maximum cue gap (ms)** - `dataset.max_gap_ms`. Cues closer than this can merge into one sentence segment. *(default 700; minimum 0; maximum 3000)*

**Edge padding (ms)** - `dataset.pad_ms`. Small context padding avoids clipped consonants. *(default 60; minimum 0; maximum 500)*

**Snap to silence** - `dataset.snap_to_silence`. Moves segment boundaries toward nearby low-energy points. *(default true)*

**Silence snap window (ms)** - `dataset.snap_window_ms`. Search radius around a proposed boundary. *(default 200; minimum 0; maximum 1000)*

**Minimum words** - `dataset.min_words`. Drops fragments with too little transcript context. *(default 2; minimum 0; maximum 30)*

**Maximum words** - `dataset.max_words`. Drops transcript segments that are implausibly dense. *(default 80; minimum 10; maximum 200)*

**Segment boundaries** - `dataset.boundary_mode`. Sentence requires punctuation; sentence_or_pause recovers more audio at aligned-word pauses with a small risk of less natural cuts. *(default "sentence"; choices "sentence", "sentence_or_pause")*

**Minimum pause for a boundary (ms)** - `dataset.min_pause_boundary_ms`. A fragment edge must have at least this much silence when punctuation does not provide the boundary. *(default 400; minimum 0; maximum 60000)*

**Trim silence** - `dataset.trim_silence`. Trims leading/trailing low-energy audio before filtering. *(default true)*

**Trim top dB** - `dataset.trim_top_db`. 40 dB is a conservative silence threshold. *(default 40; minimum 10; maximum 80)*

**Normalize loudness** - `dataset.loudness_normalize`. Recommended for consistent training gradients across sources. *(default true)*

**Target LUFS** - `dataset.target_lufs`. -20 LUFS leaves headroom and matches voice training defaults. *(default -20; minimum -30; maximum -10)*

**Sample rate** - `dataset.sample_rate`. 24000 Hz is required by the IndexTTS training pipeline. *(default 24000; minimum 8000; maximum 48000)*

**Minimum file alignment coverage** - `dataset.min_file_alignment_coverage`. Below 0.60, sentence alignment falls back or rejects unreliable timing. *(default 0.6; minimum 0; maximum 1)*

**Minimum segment alignment coverage** - `dataset.min_segment_alignment_coverage`. Drops individual caption segments with weak word alignment. *(default 0.7; minimum 0; maximum 1)*

**Minimum words / second** - `dataset.min_words_per_second`. Drops unusually sparse transcript/audio matches. *(default 1; minimum 0.1; maximum 5)*

**Maximum words / second** - `dataset.max_words_per_second`. Drops implausibly dense or misaligned speech. *(default 5.5; minimum 1; maximum 12)*

**Minimum peak dBFS** - `dataset.min_peak_dbfs`. Drops audio too quiet to train reliably. *(default -35; minimum -80; maximum 0)*

**Maximum clipping ratio** - `dataset.max_clipping_ratio`. 0.001 allows at most 0.1% clipped samples. *(default 0.001; minimum 0; maximum 1)*

**Clipping threshold** - `dataset.clipping_threshold`. Absolute normalized sample level counted as clipping. *(default 0.999; minimum 0.5; maximum 1)*

**Maximum silence ratio** - `dataset.max_silence_ratio`. Optional; blank disables whole-segment silence-ratio filtering. *(default blank; minimum 0; maximum 1)*

**Silence threshold dBFS** - `dataset.silence_threshold_dbfs`. Frames below this level count as silence. *(default -40; minimum -80; maximum -10)*

**Silence frame (ms)** - `dataset.silence_frame_ms`. 20 ms gives stable silence estimates for speech. *(default 20; minimum 5; maximum 200)*

**Reference candidates** - `dataset.export_reference_candidates`. Exports the cleanest segments for training samples and LoRA / DoRA use. *(default 5; minimum 0; maximum 20)*

**Overwrite dataset** - `dataset.overwrite`. Replaces an existing dataset directory with the same name. *(default false)*

**Maximum segments** - `dataset.max_segments`. 0 processes all segments; use a small value for smoke tests. *(default 0; minimum 0; maximum 10000000)*

**Preparation seed** - `dataset.seed`. Controls deterministic candidate ranking and randomized operations. *(default 0; minimum 0; maximum 4294967295)*

### LoRA / DoRA Training - 87 settings

**Dataset** - `training.dataset_dir`. Prepared manifest dataset used for cached-feature training. *(default "datasets/secourses_demo")*

**LoRA / DoRA name** - `training.name`. Safe output folder and final safetensors basename. *(default "voice_adapter")*

**LoRA / DoRA type** - `training.adapter_type`. DoRA is the quality default; LoRA uses slightly less compute. *(default "dora"; choices "lora", "dora")*

**Rank** - `training.rank`. 128 with alpha 129 learned the voice fastest in measured runs; rank 32 reached the same floor more slowly. *(default 128; minimum 1; maximum 256)*

**Alpha** - `training.alpha`. 129 is the measured companion scale for the recommended rank 128. *(default 129; minimum 1; maximum 1024)*

**Dropout** - `training.dropout`. 0.05 remains the quality default; stronger measured regularization gave no benefit. *(default 0.05; minimum 0; maximum 0.5)*

**Target attention** - `training.target_attention`. Adapts GPT attention projections; recommended. *(default true)*

**Target MLP** - `training.target_mlp`. Adapts GPT feed-forward projections; recommended for voice fidelity. *(default true)*

**Train speaker projection** - `training.train_spk_proj`. Fully trains the small speaker projection module. *(default true)*

**Train emotion layers** - `training.train_emo_layers`. Advanced: trains small emotion modules in addition to LoRA / DoRA layers. *(default false)*

**Train mel embedding head** - `training.train_mel_embed_head`. Advanced: trains the mel token embedding/head modules. *(default false)*

**Learning rate** - `training.learning_rate`. 4e-5 is the robust batch-1 default: it reached 5.061 held-out loss, while 8e-5 overfit within two epochs. *(default 0.00004; minimum 1e-8; maximum 1)*

**Optimizer** - `training.optimizer`. AdamW is portable; fused AdamW is faster on supported CUDA builds. *(default "adamw"; choices "adamw", "adamw_fused", "prodigy")*

**Scheduler** - `training.lr_scheduler`. Cosine decay is recommended for multi-epoch voice adaptation. *(default "cosine"; choices "cosine", "linear", "constant", "constant_with_warmup")*

**Warmup steps** - `training.warmup_steps`. 200 steps is the measured batch-1 default, easing training into the 4e-5 learning rate. *(default 200; minimum 0; maximum 1000000)*

**Weight decay** - `training.weight_decay`. 0.01 is a mild regularizer; 0.05 showed no measured benefit. *(default 0.01; minimum 0; maximum 1)*

**Adam betas** - `training.betas`. Two comma-separated momentum coefficients; 0.9, 0.99 is recommended. *(default "0.9, 0.99")*

**Adam epsilon** - `training.eps`. Numerical stability term for Adam-family optimizers. *(default 1e-8; minimum 1e-12; maximum 0.1)*

**Epochs** - `training.epochs`. 10 epochs is the measured batch-1 default; 20 epochs overfit after the held-out optimum around epoch 6. *(default 10; minimum 1; maximum 10000)*

**Maximum steps** - `training.max_steps`. 0 derives steps from epochs; set 5 for a quick smoke run. *(default 0; minimum 0; maximum 100000000)*

**Batch size** - `training.batch_size`. 1 is the measured quality default: every training clip becomes an optimizer update each epoch. *(default 1; minimum 1; maximum 128)*

**Gradient accumulation** - `training.grad_accumulation`. 1 is the measured default; accumulation 2 or 4 removed updates and performed worse at the same learning rate. *(default 1; minimum 1; maximum 128)*

**Gradient clip** - `training.max_grad_norm`. 1.0 limits unstable gradient spikes; 0 disables clipping. *(default 1; minimum 0; maximum 100)*

**Label smoothing** - `training.label_smoothing`. 0 is recommended; increase only for overconfident large datasets. *(default 0; minimum 0; maximum 0.5)*

**Mel loss weight** - `training.mel_loss_weight`. Primary autoregressive acoustic-token loss weight. *(default 1; minimum 0; maximum 100)*

**Text loss weight** - `training.text_loss_weight`. Auxiliary text modeling loss weight. *(default 0.1; minimum 0; maximum 100)*

**Speaker reference mode** - `training.speaker_ref_mode`. self uses the target clip, other uses a deterministic different same-speaker clip, and mixed alternates between them; other is the measured quality default. *(default "other"; choices "self", "other", "mixed")*

**Emotion reference mode** - `training.emo_ref_mode`. self uses the target emotion, other uses another same-speaker clip, mixed alternates, and follow_speaker reuses the speaker-reference clip; follow_speaker is the measured inference-like default. *(default "follow_speaker"; choices "self", "other", "mixed", "follow_speaker")*

**Maximum codes** - `training.max_codes`. Cached samples longer than this semantic-code limit are rejected. *(default 1500; minimum 1; maximum 100000)*

**Maximum text tokens** - `training.max_text_tokens`. Cached text length safety limit. *(default 600; minimum 1; maximum 100000)*

**Validation fraction** - `training.val_fraction`. Requested holdout fraction; whole-recording splits may differ substantially. The training plan shows actual counts. Explicit manifest splits take precedence. *(default 0.05; minimum 0; maximum 0.5)*

**Validation split** - `training.val_split_mode`. source holds out whole recordings; a single recording falls back to record splitting. *(default "source"; choices "source", "record")*

**Validate every steps** - `training.val_every_steps`. 0 disables step validation; epoch validation still runs when a split exists. *(default 250; minimum 0; maximum 1000000)*

**Maximum validation batches** - `training.val_max_batches`. 0 evaluates the entire holdout. A positive cap uses a fixed shuffled subset; loss is weighted by valid tokens. *(default 0; minimum 0; maximum 1000000)*

**Validation reference** - `training.val_reference_mode`. self validates each target with itself, while other uses a different same-speaker clip for both vectors; other is inference-like and measured more accurately. *(default "other"; choices "self", "other")*

**Automatically stop when progress stalls** - `training.early_stop_enabled`. Stop after validation stalls beyond the initial learning period and retain the best checkpoint. *(default true)*

**Early-stop patience** - `training.early_stop_patience`. 0 disables early stopping; otherwise stop after this many validations without a meaningful improvement. *(default 6; minimum 0; maximum 1000000)*

**Early-stop minimum improvement** - `training.early_stop_min_delta`. Validation loss must fall by more than this amount to reset patience. *(default 0.005; minimum 0; maximum 1000000)*

**Minimum steps before early stopping** - `training.early_stop_min_steps`. Wait for this many updates and completed warmup before counting stalled checks. *(default 1000; minimum 0)*

**Minimum epochs before early stopping** - `training.early_stop_min_epochs`. Give the dataset this many passes before counting stalled checks. *(default 2; minimum 0)*

**Base variant** - `training.base_variant`. BF16 is the quality default; INT8 ConvRot reduces frozen base weight memory. *(default "bf16"; choices "bf16", "int8_convrot")*

**Base dtype** - `training.base_dtype`. Compute/storage dtype for the BF16 base variant. *(default "bf16"; choices "bf16", "fp16", "fp32")*

**Mixed precision** - `training.mixed_precision`. BF16 is recommended on modern GPUs and avoids FP16 overflow. *(default "bf16"; choices "bf16", "fp16", "fp32")*

**Gradient checkpointing** - `training.gradient_checkpointing`. Recommended; recomputes activations to save substantial VRAM. *(default true)*

**Blocks to swap** - `training.blocks_to_swap`. Streams this many frozen GPT blocks from CPU; requires gradient checkpointing. *(default 0; minimum 0; maximum 24)*

**Swap ring size** - `training.swap_ring_size`. 2 balances overlap and VRAM; 1 uses the least memory. *(default 2; minimum 1; maximum 4)*

**Pinned swap memory** - `training.pin_swap_memory`. Recommended for faster CPU-to-GPU transfers. *(default true)*

**Output root** - `training.output_dir`. LoRA / DoRA parent folder; relative paths resolve from the app directory. *(default "loras")*

**Save every epochs** - `training.save_every_epochs`. 1 keeps an epoch checkpoint; 0 disables epoch checkpoints. *(default 1; minimum 0; maximum 100000)*

**Save every steps** - `training.save_every_steps`. 0 disables step checkpoints. *(default 0; minimum 0; maximum 10000000)*

**Keep last N** - `training.keep_last_n`. 0 keeps every epoch checkpoint so measured checkpoint comparison can choose the best voice. *(default 0; minimum 0; maximum 10000)*

**Save best** - `training.save_best`. Keeps the checkpoint with the lowest validation loss. *(default true)*

**LoRA / DoRA save dtype** - `training.save_dtype`. BF16 halves LoRA / DoRA file size; FP32 preserves full update precision. *(default "bf16"; choices "bf16", "fp32")*

**Save train state** - `training.save_train_state`. Saves optimizer, scheduler, scaler, RNG, and data position for exact resume from best, final, and interrupted checkpoints. *(default true)*

**Save train state with every epoch checkpoint** - `training.epoch_train_state`. Only needed to Continue run from a specific epoch; costs ~4x disk per checkpoint. *(default false)*

**Resume from** - `training.resume_from`. Select a LoRA / DoRA checkpoint; rank, alpha, and type are inspected before launch.

**Resume mode** - `training.resume_mode`. Weights only starts a fresh schedule at step 0; Continue run restores train state when available. *(default "weights_only"; choices "weights_only", "continue")*

**Analyze generalization automatically** - `training.auto_analyze`. Reads the CPU-only training log after complete or stopped runs and recommends a checkpoint. *(default true)*

**Evaluate checkpoints automatically** - `training.auto_evaluate_checkpoints`. After training releases its model, measures saved checkpoints on validation and a small training subset. *(default true)*

**Evaluate Base model (no LoRA / DoRA)** - `training.eval_include_base`. Measures the reference-only baseline before checkpoints for an automatic comparison. *(default true)*

**Evaluation training subset** - `training.eval_train_subset`. Deterministic training items measured during automatic evaluation; 0 disables the training subset. *(default 48; minimum 0; maximum 100000)*

**Evaluation strengths** - `training.eval_strengths`. Comma-separated LoRA / DoRA strengths from 0 to 4 for automatic checkpoint evaluation. *(default "1.0")*

**Evaluation timeout (s)** - `training.eval_timeout_s`. Stops automatic checkpoint evaluation after this many seconds without failing the completed training run. *(default 900; minimum 1; maximum 100000)*

**Generate training samples** - `training.sample_enabled`. Renders a short sample at the configured epoch interval. *(default true)*

**Sample every epochs** - `training.sample_every_epochs`. 1 provides a sample after each completed epoch. *(default 1; minimum 1; maximum 10000)*

**Sample runtime tier** - `training.sample_runtime_tier`. Memory tier for the isolated sampling process. *(default "auto"; choices "auto", "6", "8", "10", "12", "16", "24", "32")*

**Minimum free VRAM (GB)** - `training.sample_min_free_vram_gb`. Skips sampling rather than risking training OOM below this free-memory threshold. *(default 6; minimum 0; maximum 128)*

**Sample timeout (s)** - `training.sample_timeout_s`. Kills a stuck sampling subprocess after this time. *(default 300; minimum 1; maximum 100000)*

**Sample text** - `training.sample_text`. Short representative phrase used to compare epochs. *(default "This is a training progress sample for the adapted voice.")*

**Custom sample reference** - `training.sample_reference`. Optional audio path; blank uses the dataset's best reference candidate automatically.

**Sample language** - `training.sample_language`. Mirrors Voice Generation for per-epoch samples; auto uses the prepared dataset language. *(default "auto"; choices "auto", "ZH", "EN", "JA", "AR", "ES")*

**Sample seed** - `training.sample_seed`. Mirrors Voice Generation for per-epoch samples; -1 chooses one seed at training start and reuses it across epochs. *(default -1; minimum -1; maximum 4294967295)*

**Sample beams** - `training.sample_num_beams`. Mirrors Voice Generation beam search for every per-epoch sample. *(default 3; minimum 1; maximum 10)*

**Sample temperature** - `training.sample_temperature`. Mirrors Voice Generation temperature for every per-epoch sample. *(default 0.8; minimum 0.1; maximum 2)*

**Sample top-p** - `training.sample_top_p`. Mirrors Voice Generation nucleus sampling for every per-epoch sample. *(default 0.8; minimum 0; maximum 1)*

**Sample top-k** - `training.sample_top_k`. Mirrors Voice Generation token filtering for per-epoch samples; 0 disables it. *(default 30; minimum 0; maximum 100)*

**Sample repetition penalty** - `training.sample_repetition_penalty`. Mirrors Voice Generation repetition control for every per-epoch sample. *(default 10; minimum 1; maximum 20)*

**Sample emotion weight** - `training.sample_emo_alpha`. Mirrors Voice Generation emotion weight for every per-epoch sample. *(default 0.65; minimum 0; maximum 1)*

**Sample diffusion steps** - `training.sample_diffusion_steps`. Mirrors Voice Generation diffusion quality for every per-epoch sample. *(default 25; minimum 2; maximum 100)*

**Sample CFG rate** - `training.sample_inference_cfg_rate`. Mirrors Voice Generation conditioning strength for every per-epoch sample. *(default 0.7; minimum 0; maximum 2)*

**Sample maximum text tokens** - `training.sample_max_text_tokens`. Mirrors Voice Generation segment length for every per-epoch sample. *(default 60; minimum 20; maximum 300)*

**Sample length penalty** - `training.sample_length_penalty`. Mirrors Voice Generation beam length control for every per-epoch sample. *(default 0; minimum -2; maximum 2)*

**Sample maximum mel tokens** - `training.sample_max_mel_tokens`. Mirrors Voice Generation output limit for every per-epoch sample. *(default 1500; minimum 1; maximum 1815)*

**Sample speaking rate** - `training.sample_speaking_rate`. 1.0 is the model's natural pace; below 1.0 speaks slower and above 1.0 faster. *(default 1; minimum 0.5; maximum 1.5)*

**Seed** - `training.seed`. Controls split, sampler, initialization, and training randomness. *(default 42; minimum -2147483648; maximum 4294967295)*

**Data workers** - `training.num_workers`. 2 is a safe Windows/Linux default; use 0 to debug worker issues. *(default 2; minimum 0; maximum 64)*

**Log every steps** - `training.log_every_steps`. 1 gives fully live charts; increase slightly for very fast runs. *(default 1; minimum 1; maximum 100000)*

**Training device** - `training.device`. CUDA device used by the training worker. *(default "cuda:0")*

**Attention backend** - `training.attention_backend`. SDPA is the compatible default. *(default "sdpa"; choices "sdpa", "eager", "flash_attention_2")*

**Model directory** - `training.model_dir`. Base IndexTTS 2.5 model directory. *(default "models")*

**Model config** - `training.model_config`. IndexTTS 2.5 YAML configuration path. *(default "models/config.yaml")*

### Checkpoint Grid - 20 settings

**LoRA / DoRA folder** - `grid.adapter_dir`. Choose one training run to analyze and compare.

**Evaluation references** - `grid.eval_reference_mode`. Same as training validation inherits that run's validation mode; other uses a different same-speaker clip like inference. *(choices "", "self", "other")*

**Training subset** - `grid.eval_train_subset`. Deterministic training items measured beside validation; 0 disables the training subset. *(default 48; minimum 0; maximum 100000)*

**Include base model** - `grid.eval_include_base`. Adds Base model (no LoRA / DoRA), the reference-only comparison baseline. *(default true)*

**Checkpoints** - `grid.checkpoints`. Base model (no LoRA / DoRA), the recommended checkpoint, and the final checkpoint are selected first. *(default [])*

**Strengths** - `grid.strengths`. Comma-separated LoRA / DoRA strengths from 0 to 4; Base model (no LoRA / DoRA) is generated once without a strength. *(default "1.0")*

**Texts (one per line)** - `grid.texts`. Each non-empty line is generated for every selected row and reference. *(default "This is a training progress sample for the adapted voice.\nA calm, natural voice should remain clear on a sentence it never heard during training.")*

**Reference audio paths (one per line)** - `grid.references`. Every checkpoint uses these same references in the same order.

**Seed** - `grid.seed`. -1 draws one random seed at grid start and then fixes it across cells. *(default -1; minimum -1; maximum 4294967295)*

**Language** - `grid.language`. Use the language of every sentence in this grid. *(default "EN"; choices "ZH", "EN", "JA", "AR", "ES")*

**Temperature** - `grid.temperature`. 0.8 matches Voice Generation defaults. *(default 0.8; minimum 0.1; maximum 2)*

**Top-p** - `grid.top_p`. Nucleus sampling threshold shared by every cell. *(default 0.8; minimum 0; maximum 1)*

**Top-k** - `grid.top_k`. Token candidate cutoff; 0 disables it. *(default 30; minimum 0; maximum 100)*

**Beams** - `grid.num_beams`. Beam count strongly affects quality as well as generation time and VRAM. *(default 3; minimum 1; maximum 10)*

**Repetition penalty** - `grid.repetition_penalty`. Keeps semantic-token loops under control. *(default 10; minimum 1; maximum 20)*

**Diffusion steps** - `grid.diffusion_steps`. 25 is the quality default. *(default 25; minimum 2; maximum 100)*

**CFG rate** - `grid.inference_cfg_rate`. Diffusion conditioning strength. *(default 0.7; minimum 0; maximum 2)*

**Max text tokens per segment** - `grid.max_text_tokens_per_segment`. Use the same segment size for a fair comparison. *(default 60; minimum 20; maximum 300)*

**Emotion weight (alpha)** - `grid.emotion_weight`. Shared emotion-conditioning blend for every cell. *(default 0.65; minimum 0; maximum 1)*

**Speaking rate** - `grid.speaking_rate`. 1.0 is the model's natural pace; below 1.0 speaks slower, above 1.0 faster. A trained LoRA / DoRA can carry a calibrated value that matches the speaker's real pace. *(default 1; minimum 0.5; maximum 1.5)*

### Models & Performance - 26 settings

**LoRA / DoRA** - `runtime.lora_path`. Select a trained LoRA / DoRA, or None for Base model (no LoRA / DoRA), which clones from the reference only.

**LoRA / DoRA strength** - `runtime.lora_strength`. 1.0 is the trained strength; lower is subtler and higher is stronger. *(default 1; minimum 0; maximum 2)*

**Merge LoRA / DoRA into base weights for speed (BF16 only)** - `runtime.lora_merge_into_base`. Temporarily folds the selected LoRA / DoRA into floating GPT weights and restores them before switching. *(default false)*

**Device** - `runtime.device`. Auto selects the first available accelerator; choose CPU only for diagnostics. *(default "auto"; choices "cuda:0", "cuda:1", "auto", "cpu")*

**VRAM tier** - `runtime.vram_tier`. Auto detects physical VRAM; selecting a named tier fills conservative runtime settings. *(default "auto"; choices "auto", "6", "8", "10", "12", "16", "24", "32", "custom")*

**GPT model variant** - `runtime.model_variant`. BF16 gives the official quality path; INT8 ConvRot reduces GPT weight memory. *(default "bf16"; choices "bf16", "int8_convrot")*

**GPT dtype** - `runtime.gpt_dtype`. BF16 is recommended on modern NVIDIA GPUs; FP32 is the CPU-compatible fallback. *(default "bf16"; choices "bf16", "fp16", "fp32")*

**Attention backend** - `runtime.attention_backend`. SDPA is the compatible default; FlashAttention 2 requires its optional package. *(default "sdpa"; choices "sdpa", "flash_attention_2", "eager")*

**Use acceleration engine** - `runtime.use_accel`. Enables the optional CUDA-graph/flash-attention path; use beams=1. *(default false)*

**Enable emotion-text model** - `runtime.use_qwen_emo`. Required for Emotion text mode; on-demand residency keeps startup lazy. *(default true)*

**Compile s2mel** - `runtime.torch_compile_s2mel`. Uses torch.compile for repeated workloads; first generation takes longer. *(default false)*

**BigVGAN CUDA kernel** - `runtime.use_cuda_kernel_bigvgan`. Uses the optional fused activation kernel when available. *(default false)*

**CFM estimator BF16 autocast** - `runtime.s2mel_estimator_autocast`. Reduces activation VRAM; useful at 6 GB and sometimes slightly changes output. *(default false)*

**Use DeepSpeed loader** - `runtime.use_deepspeed`. Optional legacy loader; leave off unless DeepSpeed is installed. *(default false)*

**GPT blocks to swap** - `runtime.blocks_to_swap`. 0 keeps all blocks resident; -1 lets runtime fit automatically; up to 24 streams from CPU. *(default 0; minimum -1; maximum 24)*

**Swap ring size** - `runtime.swap_ring_size`. 2 overlaps transfer and compute; 1 uses least VRAM. *(default 2; minimum 1; maximum 4)*

**Pinned swap memory** - `runtime.pin_swap_memory`. Recommended for faster CPU-to-GPU block transfers. *(default true)*

**Runtime CFM cache length** - `runtime.cfm_cache_length`. Upper cache reservation used when generation does not request a larger value. *(default 8192; minimum 1024; maximum 32768)*

**VRAM reserve (GB)** - `runtime.vram_reserve_gb`. 2 GB is recommended to absorb allocator and generation peaks. *(default 2; minimum 0; maximum 12)*

**Section batch hint** - `runtime.max_section_batch_size_hint`. Advisory maximum shown to generation controls for this runtime. *(default 8; minimum 1; maximum 64)*

**Semantic Model** - `runtime.aux_residency.semantic_model`. Residency policy for this auxiliary model. *(default "gpu"; choices "gpu", "on_demand", "cpu")*

**Qwen Emo** - `runtime.aux_residency.qwen_emo`. Residency policy for this auxiliary model. *(default "on_demand"; choices "gpu", "on_demand", "cpu")*

**Campplus** - `runtime.aux_residency.campplus`. Residency policy for this auxiliary model. *(default "gpu"; choices "gpu", "on_demand", "cpu")*

**Semantic Codec** - `runtime.aux_residency.semantic_codec`. Residency policy for this synthesis model. *(default "gpu"; choices "gpu", "on_demand")*

**S2Mel** - `runtime.aux_residency.s2mel`. Residency policy for this synthesis model. *(default "gpu"; choices "gpu", "on_demand")*

**Bigvgan** - `runtime.aux_residency.bigvgan`. Residency policy for this synthesis model. *(default "gpu"; choices "gpu", "on_demand")*

## 18. Final Checklist

- One short base-model generation succeeds before advanced tuning.
- The selected language matches the script and the reference is clean.
- The active adapter, strength, auto-reference, and speaking rate are intentional.
- The section preview, caption timing, pauses, and target-duration semantics match the use case.
- The selected VRAM tier has passed the benchmark and section batch size stays within its hint.
- Dataset segments were listened to before caching and training.
- A checkpoint was chosen by validation plus fixed-input listening, not file recency alone.
- The working state is saved as a named user preset and the output folder contains reproducibility metadata.

You now have a complete path from an untrained reference clip to repeatable production output, plus the tools to build and validate a dedicated voice adapter when reference-only cloning is not enough.
