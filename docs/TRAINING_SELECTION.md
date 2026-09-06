# Automatic training and checkpoint selection

Each fresh run uses the installed base model, the selected dataset, and checkpoints created by that run. It does not need an earlier adapter, training report, reference recording, or user preset. Previous experiments informed the starting defaults; they are not runtime inputs. A fresh run name cannot overwrite or append to an existing training history. Explicit **Continue run** and **Weights only** remain available when resuming is intended.

## Browser workflow

1. Prepare the user's recordings. **Prefer sidecar** uses matching SRT subtitles when available; sentence alignment checks their timing against the audio. Missing sidecars can use ASR. Inspect preparation warnings and clip boundaries.
2. Optionally run **Voice and transcript audit** with a clean reference of the intended speaker. Reserve complete source recordings for validation and, when enough material is available, a separate final test. The audit creates new datasets and preserves the prepared originals.
3. Cache the training dataset's features. Select it in **LoRA / DoRA Training**, choose a new name, and leave **Resume from** set to **Start fresh**.
4. Keep validation and automatic speech comparison enabled. Optionally enter the independently reserved dataset under **Final-test dataset**. This speech test reads audio directly and does not require a feature cache.
5. Start training. The dashboard reports optimization, checkpoint evaluation, and generated-speech comparison as separate phases. After evaluation, **Use best checkpoint** follows the speech recommendation, including Base when appropriate.
6. Open `analysis/speech_evaluation/listening_review.html` inside the training folder for a blind listening comparison. Select a prompt and seed, play the candidate clips, and optionally download your ratings. Ratings are not filled in by the model or silently used to change a checkpoint.

## How training stops

The epoch/update budget is a maximum, not a target quality score. By default, validation runs every 250 optimizer updates and at epoch boundaries. Every absolute improvement may replace the separately saved best checkpoint.

Patience requires six checks without a loss improvement greater than 0.005. It begins after warmup, 1,000 updates, and two dataset passes. **Minimum updates between patience checks** defaults to the validation interval: nearby epoch-end checks can save a better checkpoint without consuming extra patience.

When patience first runs out and enough budget remains, the optional refinement trial multiplies the current and remaining learning-rate schedule by 0.5. It gives the lower rate 1,000 grace updates before patience resumes. The original maximum budget remains in force. The best score, patience spacing, refinement count, grace period, optimizer, and scheduler are saved for continuation. The trial is an experiment within a run; it is not guaranteed to improve every dataset.

Validation audio-token loss and auxiliary text-token loss are recorded separately and shown in the live status. Audio-token accuracy measures exact next-code prediction, not the percentage of correctly spoken words. A validation regression is a warning, not proof of memorization.

## How generated speech affects selection

Before optimization, the run freezes `analysis/speech_evaluation/plan.json`:

- Up to 12 validation texts balanced across available source recordings, speakers, languages, and clip lengths; a longer concatenated prompt is added for each voice with enough texts.
- A reference selected from that voice's actual training split and copied into the run. Validation targets never supply their own voice references.
- Three deterministic generation seeds per prompt, plus the comparison margins. Audio hashes detect changed reference and target files.

After training releases its model, teacher-forced evaluation measures the saved checkpoints and Base. Speech comparison shortlists up to three distinct training updates: candidates with the lowest measured losses and the latest saved update. It generates Base again using the same prompts, references, seeds, and inference settings. It never loads a preferred adapter from a previous run.

Whisper transcribes the complete generated audio and matched real recordings. English, Spanish, and Arabic use word errors; Chinese and Japanese use character errors. Speaker and style embeddings sample up to three distributed 20-second windows. Reports include means, worst clips, possible truncation/repetition, invalid audio, transcript-edge disagreements, and matched-text duration ratios. ASR errors on real recordings show that the recognizer itself is imperfect.

The default selection policy rejects a candidate with an observed mean transcript-error increase over Base greater than 0.02, a mean speaker-similarity drop greater than 0.03, or more invalid/possibly truncated/repetitive clips. Among the remaining candidates, paired transcript comparisons identify distinguishable differences; validation loss breaks unresolved ties. Base can win. The transcript comparison uses 2,000 bootstrap draws over whole prompts, keeping a prompt's generation seeds together instead of treating them as independent observations.

The margins are configurable screening choices. These intervals describe the sampled prompts and do not establish performance on other speakers or recording sessions. Few validation sources, overlapping training/validation recordings, and unevaluated speakers are reported. Style, pace, edge checks, ASR, and embedding similarities are proxies; naturalness still requires listening. The result is a provisional automatic recommendation, not a guarantee of the universally best checkpoint.

## Independent final test

An optional final-test dataset must use separate source recordings. The app rejects overlapping source names, shared audio paths, and byte-identical copies of training/validation files. Curated test datasets may include training-reference rows; only their `split=val` rows become test targets.

The final-test plan is frozen before training. After development selection, `final_test/selection_frozen.json` records the decision. Only that selected model and Base are generated on the final test. Results report whether the frozen selection passes the observed regression guards; they do not choose another checkpoint after looking at the test set. Repeated tuning against this test would turn it into development data.

## Results and recovery

The training folder retains the loss-best checkpoint separately from the final file. `analysis/checkpoint_eval.json` stores measured token losses. `analysis/speech_evaluation/report.json` and `report.md` store the speech recommendation, per-clip results, settings, hashes, uncertainty, and coverage warnings. `grids/` contains the actual comparison WAV files; `final_test/` contains independent confirmation when configured.

Evaluation runs in a bounded child process after training frees its model. Cancellation, timeouts, missing resources, and evaluation failures are shown in the dashboard and log; they do not invalidate the saved training weights. A failed speech comparison leaves the loss-based fallback visible instead of claiming that audio quality passed. A missing recommended file produces an error rather than silently selecting the final checkpoint.

Whisper is downloaded through the existing model downloader if absent. The normal installed speech models provide the speaker/style measurements. The feature adds no dependency on a particular computer path or past training. Real GPU/UI verification is still needed on each supported hardware configuration; unit tests alone do not establish memory fit or audio quality.
