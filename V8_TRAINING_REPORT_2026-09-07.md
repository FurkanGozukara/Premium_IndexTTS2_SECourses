# V8 training pass: v6.6 audit rules with V6 clip lengths, measured against V6 and V7 (7 September 2026)

V7 (trained the same night) changed two things at once: the vocabulary-aware transcript audit and a 30 percent share of single-sentence clips. A blind listening test then preferred V6 over V7 in 26 of 36 comparisons even though every objective proxy favored V7. V8 isolates the data change: it uses the complete v6.6 preparation and audit (caption cleanup that keeps `update.bat` whole, relaxed edge rule, spoken-form normalization, and the whisper-large-v3 second opinion) with the single-sentence share back at 0, so its clip lengths match V6. Training settings are identical to V6 and V7. Everything below was run from the command-line tools the app's tabs launch (`tools/prepare_dataset` configuration reused from V7, `tools/curate_voice_dataset.py`, `tools/cache_dataset_features.py`, `indextts.training.train_worker`, `tools/generate_grid.py`, `tools/measure_grid_quality.py`).

## Dataset

Same 17 narration recordings, the same reference clip (`cuda13_tutorial_0020`), the same validation recordings (`captioner_tutorial`, `nvfpv2`, `qwen_2511_tutorial`) and test recordings (`acestep_remix_tutorial`, `qwe2512`, `upscaler_pro_tutorial`), the same thresholds (word error 0.15, speaker similarity 0.70, window 0.60, 30 ms quiet edges).

| Stage | V6 | V7 | V8 |
|---|---:|---:|---:|
| Prepared clips / minutes | 2,546 / 602.9 | 2,875 / 610.6 | 2,576 / 609.7 |
| Single-sentence share | 0 | 0.3 | 0 |
| Audit rejections | 503 | 315 | 151 |
| Training clips / minutes / words | 1,734 / 411.6 / 65,473 | 2,147 / 458.3 / 72,743 | 2,031 / 481.3 / 76,200 |
| Training clips under 9 s | 27 | 386 | 31 |
| Validation clips / minutes | 204 / 47.6 | 272 / 58.1 | 257 / 60.9 |
| Test clips / minutes | 105 / 25.1 | 141 / 30.5 | 137 / 32.3 |

V8 rejections: 106 boundary-word mismatches, 47 speaker windows, 40 transcript disagreements, 16 different speaker or music. The second opinion checked 146 clips and recovered 46. The caption fix changed the text of 49 clips that contain file names (`update.bat`, `main.py`), which V6 and V7 had trained on as `update. bat`. Whisper word timings from V7 were reused; they are identical inputs.

### Independent check of the recovered clips (Gemini 3.8 Flash through the Antigravity CLI, not part of the app)

Blind native listening of 30 randomly chosen clips that V8 kept only because whisper-large-v3 agreed with the subtitle, no subtitle shown, judged with the audit's own comparison rules: 26 of 30 transcripts agree with the subtitle (14 exactly equal after normalization), 27 of 30 have both edges clean, mean audio quality 4.87 of 5, no music or second speaker. Three of the four disagreements are an extra or trailing word at the end of the clip that the subtitle omits ("open", "hopefully", a possibly clipped "speed"); the fourth is "100%" spoken as "100 percentage". The earlier V7 check of 65 previously accepted clips found 65 of 65 clean, so the recovered clips are of the same quality as the rest of the dataset.

## Training

Identical settings to V6 and V7: DoRA rank 128, alpha 129, dropout 0.05, attention and MLP adapters plus speaker projection, BF16, AdamW 4e-5, cosine, 200 warmup updates, batch 1, accumulation 1, 10-epoch budget (20,310 updates), full validation every 250 updates, patience 6 with one lower-learning-rate trial.

Training stopped automatically at update 13,750 of 20,310 after six stalled checks; the learning-rate trial fired at 12,500. The best checkpoint is the final one, update 13,750 (epoch 7), validation loss 4.8334 on 257 held-out clips (Base 6.6176). Optimization took 73 minutes at about 5.6 updates per second; the whole worker including checkpoint evaluation, the speech comparison and the independent final test took 2 hours 0 minutes.

| Checkpoint | Validation loss on 257 held-out clips | Next-token accuracy |
|---|---:|---:|
| Base | 6.6176 | 4.07% |
| Epoch 4 | 4.8704 | 11.05% |
| Epoch 6 | 4.8361 | 11.25% |
| Best and final, update 13,750 (epoch 7) | 4.8332 | 11.27% |

Validation sets differ between runs (each audit changes the held-out clips), so these losses are not directly comparable with V6 (4.866) or V7 (4.842).

The epoch-sample pace estimate was 1.297; the matched-sentence calibration replaced it with **1.105** from 36 held-out sentences (9 percent slower than the recordings at rate 1.0). V6 was calibrated to 1.075 and V7 to 1.049.

## The app's own speech comparison, now with a pause measurement

This release adds **Pause time vs real** to the speech comparison reports: the generated clips' internal pause time (silences of at least 120 ms between words and sentences, edges excluded) divided by the real recordings' on the same sentences. It was added because the V6-versus-V7 listening test disagreed with every existing proxy; measured on that head-to-head grid, V7 matched the narrator's pauses (median ratio 0.84, mean pause-time fraction 0.109 versus 0.106 real) while V6 paused 45 percent longer (median 1.45) and Base 28 percent longer. The listener preferred the longer pauses. The metric is informational and does not affect selection.

Development comparison on 12 frozen held-out prompts with three seeds each (39 clips per candidate):

| Candidate | Mean transcript error | Worst clip | Speaker similarity vs real | vs reference | Pause time vs real |
|---|---:|---:|---:|---:|---:|
| Base | 2.3% | 8.0% | 0.734 | 0.913 | 1.60 |
| Final, update 13,750 (selected) | 2.3% | 24.1% | 0.837 | 0.911 | 1.75 |
| Epoch 6 | 2.2% | 17.2% | 0.838 | 0.915 | 1.44 |
| Epoch 5 | 3.8% | 54.5% | 0.836 | 0.915 | 1.62 |

Real held-out recordings score 3.2 percent on the same texts. The pause column uses the release definition, total pause time over the 36 matched sentences (the V8 run's own saved report was written by an earlier build that averaged per-clip ratios and shows 2.09, 2.31, 1.91 and 2.25); the narrator spends 9.2 percent of these sentences in pauses, Base 13.1 percent and the selected V8 checkpoint 14.5 percent. The worst clips are number readings: "2509" and "2511" are transcribed as "25.09", "2050.09" or "25.11" by the recognizer whatever the model says, and the epoch-5 outlier read "NVFP4" as "VFP4".

Independent final test, 12 prompts from the three held-out test recordings, frozen before generation: Base 2.2 percent mean transcript error and 0.734 speaker similarity versus the real recording; selected V8 2.8 percent and 0.804; paired error change +0.6 points (95 percent interval -1.6 to +3.2); pause time vs real 1.44 (Base) and 1.51 (V8) by the release definition; the selection passed the regression guards. The one repeatable defect is "download" at the start of one test sentence, spoken like "to unlock" in two of three seeds.

## Head-to-head: Base, V6 best, V7 best, V8 best on the 12 independent test sentences

Same shared reference clip (V6's frozen final-test reference), seeds 42, 104771 and 209500, strength 1.0, speaking rate 1.0, the fixed comparison settings. The 36 V8 clips (`outputs/grids/v8_final_test`) were generated after the 108 Base/V6/V7 clips (`outputs/grids/v6_vs_v7_final_test`) with an identical configuration, and all 144 were measured with `tools/measure_grid_quality.py` (strict word error without vocabulary leniency, CAMPPlus similarity to the reference clip, style similarity to the real recording, matched pace, pitch and pauses).

| Measure, 12 sentences x 3 seeds | Base | V6 best | V7 best | V8 best | Real recordings |
|---|---:|---:|---:|---:|---:|
| Corpus word error, strict recognizer | 1.70% | 2.51% | 1.92% | 3.10% | 2.21% (recognizer on real speech) |
| Speaker similarity vs reference clip | 0.923 | 0.912 | 0.909 | 0.909 | |
| Style similarity vs real recording | 0.723 | 0.791 | 0.774 | 0.771 | |
| Words per second | 2.59 | 2.68 | 2.82 | 2.61 | 2.84 |
| Matched speaking-rate ratio (real / generated duration) | 0.917 | 0.951 | 0.998 | 0.926 | |
| Median generated / real duration | 1.089 | 1.030 | 0.987 | 1.063 | |
| Median pitch | 117 Hz | 139 Hz | 143 Hz | 142 Hz | 148 Hz |
| Pause time fraction | 0.144 | 0.161 | 0.113 | 0.154 | 0.109 |
| Pause time vs real (total, matched sentences) | 1.29 | 1.39 | 0.92 | 1.38 | 1.00 |

Paired by sentence and seed (36 pairs): against V6, V8 has the lower word error in 8 pairs, ties in 16 and loses 12 (mean +0.7 points); V6 has the higher style similarity in 29 of 36 pairs (mean difference 0.021) and the higher speaker similarity in 21 of 36 (difference 0.003); V8 is closer to the real duration in 24 of 36. Against V7, V8 loses 12 word-error pairs and wins 4 (mean +1.4 points), splits speaker (20 to 16) and style (18 to 18) similarity, and is closer to the real duration in 30 of 36 because V7 runs slightly fast on these prompts.

Reading: with the same clip lengths as V6, the 70 extra minutes of audited audio and the corrected file-name texts did not move the objective proxies; V8 lands on V6's numbers for identity, pitch, pace and pauses and is slightly behind V6 on strict word error and style similarity. The pause and pace columns separate V7 from both: V7 alone speaks and pauses at the narrator's rate, which is the single-sentence share at work. V8's saved speaking rate of 1.105 (7 percent slow here) is the app's automatic correction for that.

## Blind listening

Offline, Gemini 3.8 Flash through the Antigravity CLI, labels hidden, the real recording of each sentence as the reference, all adapters at speaking rate 1.0. Two pairwise tests on the 36 head-to-head clips (12 sentences x 3 seeds), plus the three-way Base/V6/V7 test run earlier the same night for context.

| Judgment | V6 vs V8 | V7 vs V8 | Base / V6 / V7 (three-way, earlier) |
|---|---|---|---|
| Preferred | V6 22, V8 14 | V8 20, V7 16 | ranked first: 0 / 26 / 10 |
| Naturalness 1-5 | 4.39 vs 4.06 | 4.17 vs 4.08 | 2.83 / 4.61 / 4.22 |
| Pronunciation 1-5 | 4.72 vs 4.67 | 4.69 vs 4.61 | 3.81 / 4.75 / 4.69 |
| Voice similarity to the speaker 1-5 | 4.67 vs 4.58 | 4.42 vs 4.53 | 3.03 / 4.83 / 4.58 |
| Pace judged same / slower / faster than real | V6 20/13/3, V8 13/20/3 | V7 13/11/12, V8 11/22/3 | 10/23/3, 18/15/3, 15/9/12 |
| Clips with artifacts noted (of 36) | V6 22, V8 25 | V7 27, V8 25 | 36 / 19 / 25 |

The listener's notes on V8 repeat one theme: prolonged or oddly placed mid-sentence pauses and a sluggish cadence ("staccato and choppy pauses", "unnatural long pauses between phrases"), which is what the pause column measures (1.38 times the narrator's pause time, the same as V6's 1.39, while V7 sits at 0.92). Word-level defects were few and shared: the ". bat" caption artifact in two prompts makes every model drop the "dot", V8 misread "4096" in two seeds ("four to ninety-six", "fourteen ninety-six") where V6 and V7 read it correctly, and "features" leaned toward "futures" in all three adapters on one prompt. V8 preferred over V7 in 20 of 36 and V6 preferred over V8 in 22 of 36 are both within what a single listener and 36 pairs can resolve; the three-way V6 margin over V7 (26 of 36) is the only clear preference in the night's data.

Gemini also blind-transcribed 30 of the 46 clips the second-opinion pass recovered: 26 agree with the subtitle under the audit's rules and 27 have clean edges, so the audit change adds clean speech; it just did not change the adapter in this run.

## Recommendation

**Update, later the same day:** a larger comparison of V5, V6, V7 and V8 on a recording none of them trained on, with each adapter at its calibrated speaking rate, reversed the preference below: the listener ranked V8 first in 18 of 36 groups and above V6 in 30, with V7 second. See `ADAPTER_COMPARISON_V5_V8_2026-09-07.md`; V8 with its saved speaking rate is now the recommended adapter, and the points below stand as the rate-1.0 result on V6's test sentences.

- **Keep V6 (`loras/Furkan_EN_DoRA_r128_v6_fresh`, update 12,000) as the production adapter.** It is the listener's choice against both V7 and V8, and its saved speaking rate of 1.075 sits between the narrator's real pace and the slower delivery the listener rewards. V7 is the adapter to use when matching the narrator's tempo matters more than the smoother delivery; V8 offers nothing over V6.
- **The v6.6 audit and caption fixes stay.** They keep clean speech (Gemini agreed with the subtitle on 26 of 30 recovered clips and 65 of 65 accepted clips), they matter more for smaller datasets where 15 percent of the audio is the difference between a usable and an unusable set, and the corrected file-name texts teach the right thing for user prompts such as `update.bat`. On this 8-hour dataset one run with 17 percent more audio landed on V6's numbers, which says the speaker's voice was already well covered.
- **Leave the single-sentence share at 0.** It reproduces the narrator's tempo (V7) but costs the inter-sentence pauses listeners prefer; the pace it fixes is better handled by the saved speaking rate, which the app now calibrates from matched sentences (V6 1.075, V7 1.049, V8 1.105).
- **What would actually settle V6 versus V8** is not another metric but repeated runs: the objective differences between them are a few tenths of a point of word error and 0.02 of style similarity, below the variation one training seed can produce, and the listening margins are 22 to 14. A second V8 run with a different seed (about 2 hours end to end with this pipeline) would show whether V6's edge is real. The blind listening forms in each adapter's `analysis/speech_evaluation/listening_review.html` and the 144 head-to-head clips in `outputs/grids` are there for a human decision.

## Limits

Gemini's judgments are automated too; its preferences are one listener's, and it favors longer pauses than the narrator actually takes. Speaker and style similarities are proxies; the pause ratio counts silence, not prosody. The head-to-head grid uses one shared reference clip, three seeds and the fixed comparison settings, and its 12 sentences come from V6's test split; two of them still contain the "comfyui. bat" caption artifact that the fix removes from new datasets. Human listening was not performed.
