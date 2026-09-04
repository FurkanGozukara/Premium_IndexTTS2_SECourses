# IndexTTS 2.5 Premium — training quality and UI update (2 September 2026)

All changes below are in the working tree of `Premium_IndexTTS2_SECourses` and are **not committed** (nothing was asked to be committed). `venv\Scripts\python.exe -m pytest tests -q` passes: 226 passed, 37 skipped (GPU tests are opt-in).

## 1. Interface changes

- **`🕘 Load last values`** header button (new bronze colour, left of "⇕ Open / close all sections"). Nothing from earlier runs is shown at startup any more: every tab starts like a fresh install (empty progress panel, live log, generated audio, recent outputs, batch results, dataset dashboard, training dashboard/charts/checkpoints, grid results, no LoRA / DoRA folder pre-selected). One click restores the last run of every tab, selects the newest LoRA / DoRA folder in the Checkpoint Grid tab, and re-attaches to a job that is still running. Universal presets and the persisted runtime still load at startup because they are settings with their own Save/Load/Reset controls, not run results. The button is disabled until the page finished its initial load.
- "Adapter" is now "LoRA / DoRA" in every visible label, info text, table, dialog, report, Help section and README bullet. Internal keys and file formats are unchanged.
- Checkpoint labels carry the file type: `best (epoch 10 DoRA Checkpoint) @ 1 | Best generalization | validation loss 5.3628`. The base row reads `Base model (no LoRA / DoRA) | Plain voice clone: only the reference audio shapes the voice` with verdict `Reference-only baseline (no LoRA / DoRA)` and no strength value. Old saved grids and evaluation reports are upgraded at display time.
- Every button on a tab has a distinct colour and icon (enforced by a test; refresh/open/delete buttons that shared icons were separated).
- Nothing inference-related is hard-coded in the workers any more: per-epoch training samples, the listening grid (now including beams) and the automatic checkpoint evaluation (training subset, strengths, base row, reference mode) take every value from the front end. The Checkpoint Grid tab exposes the evaluation reference mode.
- Bug fixed: `Keep last N = 0` deleted every epoch checkpoint while the UI promised to keep them all.
- New **speaking rate** control (Voice Generation, Batch via the same request, Checkpoint Grid, training samples) with per-LoRA calibration (section 4).

## 2. Training pipeline changes

- The old trainer always fed the GPT the *target clip's own* emotion vector, while inference uses the *reference clip's* vector. New `emo_ref_mode` (`self | other | mixed | follow_speaker`), inference-like validation (`val_reference_mode = other`: a different clip of the same speaker supplies both the voice and the emotion vectors) and the same choice for the automatic and manual checkpoint evaluation.
- Per-epoch checkpoints no longer store the 4x larger optimizer state by default (`epoch_train_state`); `best/`, final and interrupted files keep it so "Continue run" still works.
- Trainer writes `analysis/speaking_rate.json` from the epoch samples; the grid tab can recalibrate from any completed grid.

## 3. Measured training defaults

Every run used your 31.7-minute dataset (220 sentence-aligned clips, 5 % validation split), DoRA, BF16, batch 4 × accumulation 2, cosine schedule, warm-up 50, dropout 0.05, weight decay 0.01, on the RTX 5090. "self" validation = the old measurement; "other" = inference-like. Lower is better; best = lowest end-of-epoch validation loss.

| run | rank/alpha | LR | epochs | speaker / emotion / validation refs | best epoch | best val | next-token acc | final val |
|---|---|---:|---:|---|---:|---:|---:|---:|
| E01 | 32/32 | 1e-4 | 20 | mixed / self / self | 8 | 5.3674 | 9.2 % | 5.4107 |
| E02 | 128/129 | 1e-4 | 20 | mixed / self / self | 6 | 5.3643 | 9.4 % | 5.6239 |
| E03 | 128/129 | 5e-5 | 20 | mixed / self / self | 8 | **5.3586** | 9.7 % | 5.4068 |
| E04 | 128/129 | 2e-4 | 20 | mixed / self / self | 4 | 5.3715 | 9.4 % | 6.1393 |
| E05 | 128/129, dropout 0.1, wd 0.05 | 1e-4 | 20 | mixed / self / self | 6 | 5.3626 | 9.2 % | 5.6169 |
| E06 | 128/129 | 1e-4 | 20 | self / self / self | 6 | 5.3620 | 9.4 % | 5.6263 |
| E07 | 128/129 | 1e-4 | 20 | other / self / self | 6 | 5.3667 | 9.2 % | 5.6310 |
| E08 | 128/129 + emotion layers | 1e-4 | 20 | mixed / self / self | 5 | 5.3690 | 8.8 % | 5.6483 |
| E15 | 128/129 | 1e-4 | 8 | mixed / self / self | 8 | 5.3606 | 9.6 % | 5.3606 |
| E16 | 128/129 | 5e-5 | 12 | mixed / self / self | 12 | 5.3749 | 9.4 % | 5.3749 |
| E17 | 128/129, dropout 0.1, wd 0.05 | 1e-4 | 8 | mixed / self / self | 8 | 5.3615 | 9.5 % | 5.3615 |
| E18 | 128/129 | 3e-5 | 20 | mixed / self / self | 20 | 5.3733 | 9.5 % | 5.3733 |
| E19 | 128/129, dropout 0.1, wd 0.05 | 5e-5 | 20 | mixed / self / self | 8 | 5.3598 | 9.7 % | 5.4055 |
| R2 control | 128/129 | 1e-4 | 20 | mixed / self / **other** | 5 | 5.6197 | 7.1 % | 5.9575 |
| R2 | 128/129 | 1e-4 | 20 | mixed / follow_speaker / other | 7 | 5.4829 | 8.4 % | 5.7344 |
| R2 | 128/129 | 1e-4 | 20 | other / follow_speaker / other | 6 | **5.4700** | 8.4 % | 5.7135 |
| R2 | 128/129 | 1e-4 | 20 | mixed / mixed / other | 7 | 5.4893 | 8.3 % | 5.7392 |
| R2 | 128/129 | 1e-4 | 20 | mixed / other / other | 6 | 5.4714 | 8.5 % | 5.7165 |
| R2 candidate | 128/129 | 5e-5 | 12 | other / follow_speaker / other | 10 | 5.4897 | 8.7 % | 5.4911 |
| R2 candidate | 128/129 | 5e-5 | 12 | mixed / follow_speaker / other | 10 | 5.5043 | 8.4 % | 5.5048 |
| R2 candidate | 128/129 | 5e-5 | 20 | other / follow_speaker / other | 11 | **5.4689** | 8.6 % | 5.5044 |

Base model without any LoRA / DoRA: 6.58 (self) / 6.83 (other).

What the numbers say:

- Rank 128 / alpha 129 reaches the same floor as rank 32 two epochs earlier but overfits harder at LR 1e-4; LR 5e-5 gives the best floor and a flat tail; 2e-4 overfits severely; 3e-5 is still improving after 20 epochs. Extra dropout / weight decay and training the emotion layers do not help.
- Under inference-like conditioning every new emotion-reference mode beats the old behaviour by about 0.14 nats (5.47–5.49 vs 5.62); speaker `other` + emotion `follow_speaker` is the best.
- A 20-epoch cosine schedule reaches a lower best checkpoint than 12 epochs (5.469 vs 5.490); with every epoch kept and the best one recommended automatically, the longer schedule only costs time.

New defaults: **DoRA, rank 128, alpha 129, LR 5e-5, 20 epochs, speaker reference `other`, emotion reference `follow_speaker`, validation reference `other`, keep every epoch checkpoint (without per-epoch optimizer state)**; everything else unchanged (dropout 0.05, weight decay 0.01, batch 4 × 2, warm-up 50, BF16, gradient checkpointing, samples every epoch, automatic analysis + evaluation).

## 4. Why the voice sounded like a different speaking style, and what fixes it

Objective comparison on 14 held-out sentences (11 from your validation split with your real recordings for reference, 3 generic), one reference clip, fixed seed, Voice Generation defaults (beams 3, temperature 0.8, repetition penalty 10):

| | words / s | duration vs your recording | pitch variation (semitones) | style similarity to your recording | speaker similarity | WER |
|---|---:|---:|---:|---:|---:|---:|
| your real recordings | 2.67 | 1.00 | 3.05 | 1.000 | 0.891 | – |
| base model | 3.30 | 0.81 | 2.37 | 0.797 | 0.924 | 3.0 % |
| old DoRA r32 best (epoch 10) | 3.27 | 0.83 | 1.97 | 0.824 | 0.945 | 4.0 % |
| new DoRA r128 best (epoch 11) | 3.18 | 0.85 | 2.25 | 0.839 | 0.928 | 2.9 % |
| new DoRA r128 at speaking rate 0.84 | 2.67 | 1.01 | 2.31 | 0.836 | 0.917 | 3.1 % |

- Pause fractions are identical (7–8 %): the model articulates about 20 % faster than you and with flatter intonation. That, not timbre, is the "wrong style". The old adapter made intonation even flatter (1.97 semitones).
- No decoding setting changes tempo (beams 1 vs 3, repetition penalty 1–10, length penalty) and no training setting fixed it either. The semantic-code-to-frame ratio (latent multiplier) is the lever: at the calibrated rate the new voice matches your pace exactly with unchanged WER and higher style similarity.
- The new **speaking rate** control folds into the existing latent multiplier (`1.72 / rate`). Each LoRA / DoRA gets a calibrated value: the trainer estimates it from the epoch samples, and **🐢 Calibrate speaking rate from this grid** in the Checkpoint Grid tab computes a better estimate from several sentences (recommended; the one-line training sample alone estimated 1.02 for the new adapter while the 14-sentence grid gives 0.84). Voice Generation auto-applies the calibrated value when the LoRA / DoRA is selected.

## 5. Your new LoRA / DoRA

`loras/SECourses_Furkan_EN_DoRA_r128_v2` was trained from the web UI with nothing but the new defaults (rank 128, alpha 129, LR 5e-5, 20 epochs, inference-like conditioning). Best epoch 11 (inference-like validation 5.477, base 6.83, old r32 best 5.64 under the same measurement), no overfitting afterwards, all 20 epoch checkpoints kept, a sample per epoch, automatic evaluation with inference-like references, and a calibrated speaking rate (0.944 from the in-app four-sentence grid: 3.06 generated words/s vs 2.89 in your recordings; the 14-sentence objective grid suggests about 0.84, so re-calibrate from a longer grid if it still feels fast). Listening grids: `outputs/grids/SECourses_Furkan_EN_DoRA_r128_v2_*` (in-app, base / best / final on four sentences) and the objective grids under `G:\Index_TTS_v4\lora_experiments\_grids\final_rate1.0` and `final_rate0.84` (base / old r32 / new r128, 14 sentences, natural and calibrated pace).

## 6. Verification performed

- CPU test-suite: 226 passed, 37 skipped; `compileall` clean; startup request self-check OK; `build_app` without Gradio warnings.
- Live app on port 7871 driven from Chrome: clean startup on every tab; Load last values restores all tabs and attaches to running jobs; presets save; Voice Generation, Batch (2 items), Dataset scan, Training (a 3-step run and the full 20-epoch run above, both from the UI with the new fields reaching the worker), Checkpoint Grid generation with the new labels, checkpoint evaluation in `self` and `other` modes, Models & Performance apply, Help, speaking-rate calibration from a grid, and auto-apply on LoRA selection.
- Known behaviour: Gradio pauses its timers while the browser tab is hidden, so live panels only refresh on a visible tab (unchanged, upstream behaviour).

## 7. Notes

- Experiment outputs (20 training runs with all checkpoints, objective grids) are under `G:\Index_TTS_v4\lora_experiments` (about 23 GB) and can be deleted.
- `speechbrain`, `hyperpyyaml` and `ruamel.yaml` were installed into the venv for an accent-classifier probe that turned out to be useless (it scores every clip the same); they are not used by the app and can be uninstalled.
- Round-1 experiments E09–E14 (rank 64/256, LoRA vs DoRA, attention-only, text loss 0, accumulation 4) were dropped for time; the decisions above do not depend on them.
