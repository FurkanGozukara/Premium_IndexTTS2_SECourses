# IndexTTS 2.5 Premium — bigger dataset, clip length and batch-size-1 defaults (3 September 2026)

Follow-up to `TRAINING_QUALITY_REPORT_2026-09-02.md`. Everything below was measured on your 10 videos (about 5 hours of speech) with one whole video held out for evaluation, and is committed in `021332a`, `aa51707` and `609f20f`.

## 1. Summary
- **Dataset preparation accepts your `Lora_Training_Dataset` layout** (videos with sidecar subtitles, one video without subtitles, per-video sub-folders), non-ASCII names and odd subtitle encodings, on Windows and Linux; the dataset was rebuilt from all 10 videos.
- **Longer clips are better up to 20 s**: 4–20 s clips (14 s target) match 4–12 s clips on short held-out speech and beat them on long speech, while keeping 11 % more audio; 30 s clips are worse. New default: 4–20 s.
- **Batch size 1 is now the default**, and the sweep showed the learning rate must go *up* (4e-5), not down, with 10 epochs; accumulation only hurts. The Training tab shows a live plan (optimizer updates) and suggests an epoch count for any dataset size.
- **New deliverable `loras/SECourses_Furkan_EN_DoRA_r128_v3`** speaks at your real pace (v2 was 17 % too fast) with the closest style match measured so far.

## 2. Dataset and clip length
### Source data and robustness
`G:\Index_TTS_v4\Lora_Training_Dataset` now mixes top-level videos with sidecar `.srt` files, one video without any subtitle (`gguf.mp4`), and five `sourceN/` sub-folders each holding one video plus its subtitle. The dataset tab and `tools/prepare_dataset.py` handle this layout recursively: sidecar subtitles are used when present, Whisper large-v3-turbo transcribes the rest, and the Whisper word cache is reused across rebuilds. Non-ASCII file names, BOM / UTF-16 / legacy-codepage subtitles and Windows console encodings were hardened and tested in commits c0bd63a and 3a419f3.

### Clip-length experiment (held-out video, inference-like references)
Training clips were built three ways from 9 of the 10 videos; the tenth video (video5) was held out and cut into 4–12 s and 4–20 s evaluation clips. Every run used rank 128 / alpha 129 DoRA, batch 4 × accumulation 2, learning rate 5e-5, 8 epochs, and validation with a *different* clip of the speaker supplying both speaker and emotion vectors (the way inference works).

| Training clips | Clips | Audio kept | Steps | Held-out 12 s clips (38) | Held-out 20 s clips (23) |
|---|---:|---:|---:|---:|---:|
| 4–12 s (target 8) | 1665 | 244.3 min | 1592 | **5.124** | 5.126 |
| 4–20 s (target 14) | 1064 | 270.8 min | 1064 | 5.153 | 5.122 |
| 4–30 s (target 20) | 723 | 264.8 min | 723 | 5.190 | 5.156 |
| 4–20 s, 12 epochs (step-matched) | 1064 | 270.8 min | 1524 | 5.125 | **5.095** |

Base model without LoRA / DoRA: 6.578 (12 s) and 6.433 (20 s). Reading: at equal step counts 4–20 s clips are as good as 4–12 s on short clips and better on long ones, and the 20 s cut keeps 11 % more of the source audio (fewer sentences are dropped at boundaries). 30 s clips are worse on both sets. New default: 4–20 s with a 14 s packing target. Pause-based boundaries (optional `sentence_or_pause` mode) recovered only 20 extra clips and were not adopted as the default.

### Full dataset rebuild
All 10 videos with the new defaults: 1112 clips, 282.7 min of speech, mean 15.3 s, longest 19.7 s (`datasets/furkan_full_20s`). The old 4–12 s cut of the same videos gave 1738 clips but only 254.7 min.

## 3. Batch-size-1 sweep
### Runs (4–12 s training clips, 1665 clips, held-out video5)
All runs: DoRA rank 128 / alpha 129, cosine schedule, 200 warmup steps, inference-like references for validation. "Auto-best" is the checkpoint the app selects by its own validation split; "lowest" is the best epoch on the held-out video, which the app cannot see.

| Run | Batch × accumulation | LR | Epochs | Optimizer updates | In-dataset val (best ep) | Held-out 12 s: lowest (ep) / auto-best / final | Held-out 20 s: lowest (ep) / final |
|---|---|---:|---:|---:|---:|---|---|
| reference (old default) | 4 × 2 | 5e-5 | 8 | 1,592 | 5.188 (8) | 5.124 (8) / 5.124 / 5.124 | 5.126 (8) / 5.126 |
| P1 | 1 × 1 | 1e-5 | 10 | 15,910 | 5.163 (10) | 5.143 (10) / 5.143 / 5.143 | 5.142 (10) / 5.142 |
| P4 | 1 × 2 | 1e-5 | 10 | 7,960 | 5.210 (9) | 5.169 (10) / 5.169 / 5.169 | 5.167 (10) / 5.167 |
| P2 | 1 × 1 | 2e-5 | 10 | 15,910 | 5.084 (9) | 5.101 (6) / 5.104 / 5.105 | 5.107 (6) / 5.109 |
| P3 | 1 × 4 | 2e-5 | 10 | 3,980 | 5.177 (10) | 5.149 (10) / 5.149 / 5.149 | 5.147 (9) / 5.147 |
| P6 | 1 × 1 | 2e-5 | 20 | 31,820 | 5.023 (14) | 5.084 (12) / 5.095 / 5.101 | 5.089 (12) / 5.102 |
| **P5** | 1 × 1 | **4e-5** | 10 | 15,910 | 5.006 (9) | **5.061 (6)** / 5.078 / 5.080 | **5.065 (6)** / 5.079 |
| P7 | 1 × 1 | 8e-5 | 10 | 15,910 | 4.969 (5) | 5.040 (4) / 5.075 / 5.216 | 5.034 (4) / 5.175 |
| P8 | 1 × 1 | 4e-5 | 20 | 31,820 | 4.983 (9) | 5.053 (6) / 5.102 / 5.260 | 5.053 (6) / 5.214 |

Reading:
- Batch 1 without accumulation wins because every clip becomes an optimizer update; accumulation (P3, P4) throws those updates away and lands close to the old batch-4 numbers.
- The learning rate must go *up* when the batch shrinks, not down: 1e-5 → 2e-5 → 4e-5 improves the held-out loss at every step (5.143 → 5.101 → 5.061). Doubling the epochs at 2e-5 instead (P6) does not reach 4e-5 × 10 epochs and doubles the training time.
- 8e-5 reaches the lowest single held-out number (5.040 at epoch 4) but overfits immediately afterwards; the checkpoint the app would pick by its own validation split (epoch 5) is no better than P5's, and the final file is much worse (5.216). Too fragile for a default.
- Twenty epochs at 4e-5 (P8) bottom out at the same point (epoch 6, 5.053) and then overfit hard: the app-selected checkpoint (epoch 9) is worse than the 10-epoch run's and the final file much worse (5.260). Ten epochs at 4e-5 is therefore the default, and the Training plan readout now suggests an epoch count that lands near 10,000 optimizer updates for any dataset size.
- The in-dataset validation split keeps improving after the held-out loss has bottomed out (P5: in-dataset best at epoch 9, held-out best at epoch 6). Validation clips come from the same recordings as the training clips, so part of that late improvement is memorising recording-specific detail. Keeping every epoch and comparing them in the Checkpoint Grid with new text remains the right way to choose the file to ship.

## 4. New defaults and pipeline changes
### New defaults (commits 021332a and aa51707)

| Setting | Old default | New default | Why |
|---|---|---|---|
| Dataset clip range | 4–12 s, target 8 s | **4–20 s, target 14 s** | Equal quality on short clips, better on long clips, 11 % more audio kept; 30 s was worse |
| Batch size × accumulation | 4 × 2 | **1 × 1** | Every clip becomes an optimizer update; accumulation only removed updates |
| Learning rate | 5e-5 | **4e-5** | Best held-out loss at batch 1 (1e-5 → 2e-5 → 4e-5 kept improving; 8e-5 overfit within two epochs) |
| Epochs | 20 | **10** | Held-out optimum after ~10,000 updates; 20 epochs at 4e-5 overfit and the auto-selected checkpoint got worse |
| Warmup steps | 50 | **200** | Batch-1 steps are noisier; all sweep runs used 200 |
| Rank / alpha, DoRA, dropout 0.05, weight decay 0.01, cosine, references `other` / `follow_speaker` / validation `other`, keep every epoch | unchanged | unchanged | Measured in session 1 |

With batch 1 an epoch means one optimizer update per training clip, so the epoch count is now much lower than the old batch-4 default while the number of updates is far higher (the old 20-epoch batch-4 default did about 550 updates on a 220-clip dataset; the new default does 2,090, and the readout suggests 48 epochs to reach the measured sweet spot).

### Pipeline changes
- **Training plan readout** (LoRA / DoRA Training → Optimization): shows clips, training / validation split, micro-batches and optimizer updates per epoch and for the whole run, recomputed live when the dataset, batch size, accumulation, epochs, maximum steps, validation fraction or seed change. It now also gives a data-backed advisory: below 5,000 updates it suggests the epoch count that reaches about 10,000 updates; inside 5,000–20,000 it confirms the plan is in the measured range; above 20,000 it points at the Checkpoint Grid and a reduced epoch count. The trainer logs the same plan line at start.
- `tools/prepare_dataset.py` now takes every default from `DatasetPrepConfig` (it used to carry its own, different numbers: target 8 s, minimum 1.5 s, maximum 15 s), so CLI, dataclass, UI and system presets can no longer drift apart. A test checks the UI slider defaults and bounds.
- System presets (`presets/system/*.json`) regenerated with the new values; the idempotence test guards them.
- README, ARCHITECTURE_NOTES and the Help tab describe the new defaults.

### How the numbers were produced
- Clip-length runs: `run_experiments.py` queues (batch 4 × 2, learning rate 5e-5, 8 or 12 epochs) on `exp_train_12s` / `exp_train_20s` / `exp_train_30s`, evaluated with `checkpoint_eval` on the held-out video cut into 4–12 s and 4–20 s clips, reference mode `other`, validation fraction 0.5, seed 42.
- Batch-1 sweep: eight runs P1–P8 on `exp_train_12s` (see table), same held-out evaluation for every second epoch plus the app-selected best and final checkpoints.
- All experiment outputs are under `G:\Index_TTS_v4\lora_experiments` (`results.jsonl`, `<run>/analysis/heldout_eval.json`, `<run>/analysis/heldout20_eval.json`).

## 5. Final deliverable
### `loras/SECourses_Furkan_EN_DoRA_r128_v3`
Trained from the app UI with the shipped defaults on `datasets/furkan_full_20s` (1,058 training / 54 validation clips, 10,580 optimizer updates, 77 minutes on the RTX 5090 including per-epoch samples). Every epoch is kept.

| | Validation loss (inference-like references) |
|---|---:|
| Base model (no LoRA / DoRA) | 6.719 |
| epoch 6 | 5.099 |
| **best = epoch 9** | **5.085** (trainer split: 5.039) |
| final = epoch 10 | 5.088 |

The app's automatic evaluation picked epoch 9 and reported a plateau without an overfitting rise, matching the sweep (held-out optimum around 10,000 updates).

**Listening grid** (`G:\Index_TTS_v4\lora_experiments\_grids\final_v3_rate1.0`): 12 validation sentences that have real recordings plus 3 unseen generic sentences, one shared reference clip, beams 3, speaking rate 1.0, same seed for every checkpoint. Objective metrics per checkpoint (WER via Whisper large-v3-turbo; CAMPPlus speaker similarity to the reference; style similarity to the real recording in the GPT emotion/style space; words per second; f0):

| Checkpoint | WER | Speaker sim | Style ~ real | Words / s | Speed vs real recording | f0 (Hz) |
|---|---:|---:|---:|---:|---:|---:|
| Real recordings | – | 0.851 | 1.000 | 2.79 | 1.00 | 140 |
| Base model | 0.042 | 0.920 | 0.590 | 2.85 | 1.04 | 138 |
| v2 (previous deliverable, epoch 11) | 0.034 | 0.924 | 0.655 | 3.18 | **1.17** | 141 |
| v3 epoch 6 | 0.040 | 0.905 | **0.717** | 2.89 | 1.06 | 140 |
| **v3 best (epoch 9)** | 0.032 | 0.908 | 0.698 | 2.83 | **1.04** | 144 |
| v3 final (epoch 10) | 0.031 | 0.902 | 0.706 | 2.89 | 1.07 | 144 |

What changed for the "sounds like me but rushed" problem: v2 spoke 17 % faster than your real recordings of the same sentences; v3 is within 4-7 % at speaking rate 1.0, and its style vectors are the closest to your real recordings of any checkpoint (0.70-0.72 vs 0.655). Intelligibility is unchanged (about 3 % WER) and pitch matches (140-144 Hz vs 140 Hz real). Speaker similarity to the reference clip is marginally lower than v2's (0.908 vs 0.924) but still above the 0.851 that your own real recordings score against the same reference, so it is not a fidelity loss.

Speaking rate: the app's grid calibration now compares each generated sentence with your real recording of the same sentence when the grid uses dataset sentences (commit `609f20f`; the old dataset-wide method was biased by the pauses inside long packed clips and said 0.90). Across the 12 matched sentences v3 speaks 3 % faster than you, so `analysis/speaking_rate.json` stores **0.97** and the Voice Generation tab applies it automatically when you pick this LoRA / DoRA. The same measurement gives 0.87 for v2 (14 % too fast), which is the number behind the "different speaking style" complaint.

## 6. Verification performed
- CPU test suite after each merge: 237 → 243 tests passing (37 GPU tests skipped); presets idempotence and UI build tests included.
- Browser checks on a clean app start (port 7861): training tab defaults (batch 1, accumulation 1, 4e-5, 10 epochs, warmup 200), live Training plan readout (updates on epoch and dataset changes, shows the epoch suggestion), dataset tab defaults (14 / 4 / 20 s) and their info texts; startup shows fresh defaults, not the last run.
- The final LoRA was trained from the UI, evaluated by the app's automatic checkpoint evaluation, compared in a 75-cell grid against v2 and the base model with objective metrics, and its speaking rate was calibrated from that grid.

## 7. Notes and housekeeping
- Experiment outputs are under `G:\Index_TTS_v4\lora_experiments` (41 runs, about 47 GB, every epoch kept) and `G:\Index_TTS_v4\lora_experiments\_grids`; they can be deleted. `G:\Index_TTS_v4\_unicode_test` (18 MB, the Unicode/robustness fixture) can be deleted too.
- Experiment datasets `datasets/exp_train_*`, `datasets/exp_eval_video5_*` and `datasets/furkan_full_12s` are only needed to reproduce the numbers above; `datasets/furkan_full_20s` is the one to keep.
- The old deliverable `loras/SECourses_Furkan_EN_DoRA_r128_v2` is kept for A/B listening; `loras/SECourses_Furkan_EN_DoRA_r32` can go.
