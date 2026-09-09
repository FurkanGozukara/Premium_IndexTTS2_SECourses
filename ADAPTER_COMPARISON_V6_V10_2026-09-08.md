# Best checkpoints of V6 to V10 compared on one recording none of them trained on (8 September 2026)

**Update, 9 September 2026:** five training-pipeline changes were tested against V10 and a new run, V11, trained on a preparation with deliberate short and medium clips; on the same 12 sentences as deployed, V11 has the highest speaker and style similarity of any run and pauses within one percent of the narrator's. See `TRAINING_PIPELINE_IMPROVEMENTS_2026-09-09.md`.

This extends `ADAPTER_COMPARISON_V5_V8_2026-09-07.md` with the two runs trained since: **V9** (the first run of the v6.9 automatic pipeline on the V8 dataset) and **V10** (a fresh preparation of the same recordings on the current build). The test is the same: the 12 audited clips of `qwen_2511_tutorial`, the one recording that no run from V5 to V10 trained on (a validation recording for V6 to V10, used for their loss checks and checkpoint selection but never for optimization), three seeds per sentence, the fixed comparison settings (3 beams, temperature and top-p 0.8, top-k 30, repetition penalty 10, 25 diffusion steps, CFG 0.7), and the narrator's real recording of every sentence for the matched measurements. The V5 to V8 clips are the ones the earlier report measured; V9 and V10 were generated today with the identical configuration on the RTX 5090 (`outputs/grids/v10cmp_*`).

Three conditions:

- **Speaking rate 1.0, shared reference, GPT adapter only** (V6's frozen final-test reference, no voice decoder adapter): the model-intrinsic comparison, 36 clips per adapter.
- **Each adapter at its calibrated rate, shared reference, GPT adapter only**: the rates come from the first condition with the app's matched-sentence method, so all adapters are calibrated on the same sentences.
- **As the app deploys each run**: the run's own saved reference, saved speaking rate, installed voice decoder adapter and accepted decoding settings. V8's and V9's clips in this condition are the ones measured for the decoder report (`outputs/grids/v9_vs_v8_*`); V7's and V10's were generated today.

Every clip was measured with `tools/measure_grid_quality.py` (strict word error without vocabulary leniency, CAMPPlus similarity to the reference clip and to the narrator's real recording of the sentence, style similarity to the real recording, matched pace, pitch, pauses). The V5 to V8 grids were re-measured with the current tool so that every row has the similarity to the real recording, which the earlier report did not have; their other numbers reproduce that report to the last digit. Blind listening was done offline by Gemini 3.8 Flash through the Antigravity CLI, one fresh conversation per sentence and seed, clips copied to neutral names, labels hidden, the real recording as reference, every clip rated and ranked; the bundles, prompts, raw responses and answer keys are under `outputs/native_review/v10cmp_listening`. It is not part of the app.

## The adapters

| | V6 | V7 | V8 | V9 | V10 |
|---|---:|---:|---:|---:|---:|
| Dataset | `furkan_v6_curated_20s` | `furkan_v7_curated_20s` | `furkan_v8_curated_20s` | `furkan_v8_curated_20s` (same as V8) | `furkan_v10_curated_20s` (fresh preparation) |
| Training audio | 1,734 clips / 411.6 min | 2,147 clips / 458.3 min | 2,031 clips / 481.3 min / 76,200 words | 2,031 clips / 481.3 min / 76,200 words | 2,034 clips / 482.0 min / 76,267 words |
| Recordings held out | 6 (3 validation, 3 test) | 6 | 6 | 3 validation (no final-test dataset given) | 6 (3 validation, 3 test) |
| Reference clip for training and generation | cleanest clip near 15 s (`qwen_fine_tuning_0385`, 119 Hz) | same clip | same clip | median-matched (`qwen_fine_tuning_0185`, 143 Hz, 2.67 words/s) | same median-matched clip |
| Selected checkpoint | update 12,000 (epoch 7) | update 12,882 (epoch 6) | update 13,750 (epoch 7, final) | update 12,186 (epoch 6) | update 14,000 (epoch 7, final) |
| Validation loss, own split (Base on the same split) | 4.866 (6.555), 204 clips | 4.842 (6.649), 272 clips | 4.833 (6.618), 257 clips | epoch 6: 4.835; best 4.832 at 13,750 (6.507), 257 clips | final: 4.833; best 4.833 at 12,204 (6.506), 256 clips |
| GPT optimization time | 89 min | 94 min | 73 min | 91 min | 88 min |
| Voice decoder adapter | none | none | trained afterwards, installed (+0.062 similarity on the final test) | installed at strength 1.0 (+0.031 similarity, +0.6 points word error on the benchmark), 35 min | installed at strength 1.0 (+0.033 similarity, -0.06 points word error), 28 min |
| Decoding settings the app applies | defaults | defaults | defaults | guidance rate 1.0 adopted by the sweep (score +0.0097) | defaults kept (no change met the margin) |
| Saved speaking rate in the app | 1.075 | 1.049 | 1.105 | 1.066 | 1.075 |

All five use DoRA rank 128, alpha 129, dropout 0.05, attention and MLP adapters plus the speaker projection, BF16 base, AdamW 4e-5 with cosine decay and 200 warmup updates, batch 1, a 10-epoch budget with automatic stopping (patience 6, one lower-learning-rate trial). V10 additionally stores the fully trained speaker projection in FP32 (the v6.9 default "Train speaker/extra modules in FP32"), which is why its optimizer ran at 4.3 updates per second against V8's 4.7. Validation losses are on different held-out sets and are not comparable across columns.

### What V9 and V10 changed

- **V9** is V8's dataset trained again on the v6.9 pipeline with every new option at its default: the reference clip is chosen near the speaker's median pitch and pace instead of being the cleanest clip near 15 seconds (V6 to V8 all conditioned on a 119 Hz clip of a 144 Hz speaker), a voice decoder adapter is trained and gated after the GPT adapter, and the decoding settings are swept once. Its speech comparison rejected the final epoch-7 file (3.3 percent transcript error against Base's 1.2, worst clip 36 percent) and selected epoch 6 (1.5 percent, speaker similarity to the real recordings 0.855 against Base's 0.834). The decoder report (`VOICE_DECODER_ADAPTER_2026-09-08.md`) already compared V9 with V8 as deployed on these sentences; that comparison is repeated below with V7 and V10 added.
- **V10** started from the raw recordings again on the current build (the `qa_v10_*` documents under `docs/` record every step through the browser): the fresh preparation reproduced V8's segmentation byte for byte, the audit rejected 150 clips instead of 151 (102 boundary-word mismatches, 36 speaker windows, 38 transcript disagreements, 19 different speaker or music; the whisper-large-v3 second opinion recovered 47 of 150), and training ran with the FP32 storage of the speaker projection that Codex's quality fixes added. Training stopped at update 14,000 of 20,340 after the learning-rate trial at 12,500; the loss-best checkpoint is epoch 6 (update 12,204, 4.8328) but the speech comparison chose the final file (1.3 percent transcript error and 0.863 similarity to the real recordings against epoch 6's 2.1 percent and 0.854), and the epoch-sample pace estimate of 1.235 was replaced by 1.075 from 36 matched sentences. The decoder adapter (early stop at 10,500, best re-rendered identity 0.9505 at update 8,000) passed the full-pipeline gate at both strengths; the sweep kept the defaults; the frozen pipeline passed its independent final test on the three test recordings (Base 1.5 percent error and 0.829 similarity to the real recordings, V10 1.6 percent and 0.877, pause time 1.41 against 1.20 of the real).

## Condition 1: speaking rate 1.0, shared reference, GPT adapter only, 12 sentences x 3 seeds

| Model | Corpus word error | Mean word error | Worst clip | Speaker similarity vs reference | Speaker similarity vs real | Style similarity vs real | Words per second | Matched rate ratio | Median duration ratio | Median pitch | Pause time fraction | Pause time vs real |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 2.73% | 2.93% | 16.1% | 0.923 | 0.733 | 0.630 | 2.50 | 0.902 | 1.111 | 118 Hz | 0.162 | 1.42 |
| V5 best | 4.28% | 4.69% | 20.0% | 0.911 | 0.783 | 0.714 | 2.67 | 0.962 | 1.051 | 135 Hz | 0.182 | 1.47 |
| V6 best | 3.54% | 3.81% | 12.9% | 0.915 | 0.777 | 0.710 | 2.60 | 0.937 | 1.084 | 133 Hz | 0.179 | 1.48 |
| V7 best | 3.32% | 3.66% | 16.1% | 0.910 | 0.784 | 0.690 | 2.67 | 0.960 | 1.052 | 141 Hz | 0.140 | 1.13 |
| V8 best | 4.20% | 4.62% | 19.4% | 0.910 | 0.786 | 0.692 | 2.56 | 0.922 | 1.073 | 141 Hz | 0.167 | 1.40 |
| V9 best | 4.06% | 4.38% | 16.1% | 0.921 | 0.760 | 0.694 | 2.66 | 0.957 | 1.067 | 128 Hz | 0.161 | 1.32 |
| V10 best | 3.32% | 3.63% | 16.2% | 0.917 | 0.768 | 0.697 | 2.62 | 0.944 | 1.087 | 130 Hz | 0.165 | 1.38 |
| Real recordings | 5.09% | 5.24% | | 0.767 | | | 2.79 | | | 147 Hz | 0.131 | |

Speaker similarity to the reference clip rewards copying the prompt; speaker similarity to the real recording of each sentence is the speaker's actual identity and is the column the training pipeline gates on. The recognizer makes more errors on the real recording (5.09 percent) than on any model, so word error here measures intelligibility to Whisper rather than correctness.

Paired per clip, same sentence and seed (wins / ties / losses for the first-named adapter):

- Word error: Base vs V5 15/15/6; Base vs V6 10/20/6; Base vs V7 10/21/5; Base vs V8 13/19/4; Base vs V9 12/21/3; Base vs V10 9/19/8; V5 vs V6 2/26/8; V5 vs V7 5/20/11; V5 vs V8 6/25/5; V5 vs V9 8/22/6; V5 vs V10 1/25/10; V6 vs V7 4/28/4; V6 vs V8 8/26/2; V6 vs V9 9/23/4; V6 vs V10 1/30/5; V7 vs V8 9/22/5; V7 vs V9 10/19/7; V7 vs V10 3/26/7; V8 vs V9 4/27/5; V8 vs V10 2/24/10; V9 vs V10 3/22/11.
- Style similarity to the real recording: Base vs V5 0/0/36; Base vs V6 0/0/36; Base vs V7 4/0/32; Base vs V8 4/0/32; Base vs V9 1/0/35; Base vs V10 1/0/35; V5 vs V6 22/0/14; V5 vs V7 28/0/8; V5 vs V8 26/0/10; V5 vs V9 23/0/13; V5 vs V10 24/0/12; V6 vs V7 25/0/11; V6 vs V8 23/0/13; V6 vs V9 23/0/13; V6 vs V10 22/0/14; V7 vs V8 17/0/19; V7 vs V9 17/0/19; V7 vs V10 12/0/24; V8 vs V9 17/0/19; V8 vs V10 17/0/19; V9 vs V10 17/0/19.
- Speaker similarity to the real recording: Base vs V5 4/0/32; Base vs V6 5/0/31; Base vs V7 4/0/32; Base vs V8 2/0/34; Base vs V9 9/0/27; Base vs V10 6/0/30; V5 vs V6 21/0/15; V5 vs V7 17/0/19; V5 vs V8 16/0/20; V5 vs V9 26/0/10; V5 vs V10 22/0/14; V6 vs V7 15/0/21; V6 vs V8 13/0/23; V6 vs V9 23/0/13; V6 vs V10 23/0/13; V7 vs V8 17/0/19; V7 vs V9 28/0/8; V7 vs V10 27/0/9; V8 vs V9 27/0/9; V8 vs V10 24/0/12; V9 vs V10 14/0/22.
- Speaker similarity to the reference clip: Base vs V5 29/0/7; Base vs V6 28/0/8; Base vs V7 26/0/10; Base vs V8 28/0/8; Base vs V9 19/0/17; Base vs V10 21/0/15; V5 vs V6 13/0/23; V5 vs V7 19/0/17; V5 vs V8 20/0/16; V5 vs V9 9/0/27; V5 vs V10 10/0/26; V6 vs V7 22/0/14; V6 vs V8 23/0/13; V6 vs V9 13/0/23; V6 vs V10 16/0/20; V7 vs V8 20/0/16; V7 vs V9 11/0/25; V7 vs V10 11/0/25; V8 vs V9 12/0/24; V8 vs V10 9/0/27; V9 vs V10 20/0/16.

At rate 1.0 every adapter is slower than the narrator (V7 and V5 by 4 percent, V9 by 4, V10 by 6, V6 by 6, V8 by 8, Base by 10). V10 has the lowest word error of the six adapters (3.32 percent corpus, 3.63 mean; only Base, which copies the prompt, is lower, and it beats V8 in 10 of 36 paired clips against 2 and V9 in 11 against 3), with V7 level on corpus error. V7 alone pauses like the narrator (1.13 of the real pause time; the others 1.32 to 1.48).

The pitch column shows what the shared reference does to the two new runs. V6, V7 and V8 were trained with this clip (119 Hz, one of the speaker's lowest) as their conditioning reference and render it at 133 to 141 Hz; V9 and V10 were trained with the median-matched 143 Hz clip, and given the low clip at inference they follow it down to 128 to 130 Hz, which raises their similarity to the reference clip (0.917 to 0.921, the highest of any adapter) and lowers their similarity to the real recordings (0.760 to 0.768 against 0.777 to 0.786), the narrator being a 147 Hz speaker on these sentences. This condition and the next one therefore hand V9 and V10 a prompt outside their training distribution, exactly the swap effect the decoder report measured on V8 in the other direction; the third condition removes it.

### Blind listening at rate 1.0, five-way ranking against the real recording

Each of the 36 sentence-and-seed groups was judged once, all five clips shuffled behind letters A to E and the real recording as reference; the listener rated every clip and ranked the five. Two groups needed a second conversation because the first returned its reasoning instead of the structured answer; every judgment was returned with high confidence.

| Model | Mean rank (1 is best) | Ranked first | Ranked last | Pronunciation 1-5 | Naturalness 1-5 | Voice similarity to the speaker 1-5 | Audio quality 1-5 | Pace same / slower / faster than real | Clips with wrong words | Clips with artifacts noted |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| V6 best | 3.36 | 3 | 7 | 4.69 | 4.03 | 4.64 | 4.42 | 22 / 11 / 3 | 1 | 23 |
| V7 best | 2.17 | 17 | 3 | 4.86 | 4.47 | 4.81 | 4.78 | 27 / 8 / 1 | 2 | 13 |
| V8 best | 3.00 | 8 | 6 | 4.81 | 4.14 | 4.75 | 4.58 | 25 / 9 / 2 | 0 | 21 |
| V9 best | 3.44 | 2 | 13 | 4.89 | 4.06 | 4.75 | 4.47 | 23 / 10 / 3 | 0 | 25 |
| V10 best | 3.03 | 6 | 7 | 4.83 | 4.19 | 4.75 | 4.56 | 25 / 9 / 2 | 0 | 23 |

Pairwise, from the rankings (times the row adapter was ranked above the column adapter, of 36; ties not counted):

| | V6 | V7 | V8 | V9 | V10 |
|---|---:|---:|---:|---:|---:|
| V6 | - | 11 | 15 | 19 | 14 |
| V7 | 25 | - | 25 | 25 | 27 |
| V8 | 21 | 11 | - | 23 | 17 |
| V9 | 17 | 11 | 13 | - | 15 |
| V10 | 22 | 9 | 19 | 21 | - |

V7 is ranked above each of the other four in 25 to 27 of 36 groups and takes sole first place in 17; the listener's notes on it are mostly "none" ("exceptionally fluid", "seamless phrasing"). V10 and V8 are level in second place (mean rank 3.03 against 3.00; V10 above V8 in 19 of 36), and V10 is above V6 in 22 and above V9 in 21. V9 is last: its notes are pace and cadence ("choppy cadence, micro-stutters", "hesitation after 'This is'", "rushed tempo"), the same complaints the earlier report recorded for V5 and V6, and it is ranked last in 13 groups. Word-level defects are almost gone at this stage: the only ones noted are V6's "2559" for "2509" in one clip and V7's "cop" for "cap" and "presets" for "preset"; V8, V9 and V10 have no wrong-word clip in 36. Today's rubric told the listener that a number read in another valid form is not a wrong word, so the "3090" reading that yesterday's test counted against every adapter is not counted here; the V6, V7 and V8 clips are the same files as yesterday.

## Condition 2: each adapter at its calibrated rate, shared reference, GPT adapter only

| Adapter | Generated words per second at rate 1.0 | Calibrated rate | Saved rate in the app |
|---|---:|---:|---:|
| V5 | 2.514 | 1.051 | 1.049 (epoch samples) |
| V6 | 2.455 | 1.076 | 1.075 |
| V7 | 2.515 | 1.050 | 1.049 |
| V8 | 2.414 | 1.094 | 1.105 |
| V9 | 2.497 | 1.058 | 1.066 |
| V10 | 2.452 | 1.077 | 1.075 |

| Model | Corpus word error | Mean word error | Worst clip | Speaker similarity vs reference | Speaker similarity vs real | Style similarity vs real | Words per second | Matched rate ratio | Median duration ratio | Median pitch | Pause time fraction | Pause time vs real |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base @ rate 1.0 | 2.73% | 2.93% | 16.1% | 0.923 | 0.733 | 0.630 | 2.50 | 0.902 | 1.111 | 118 Hz | 0.162 | 1.42 |
| V5 best @ 1.051 | 3.91% | 4.33% | 20.0% | 0.913 | 0.784 | 0.703 | 2.81 | 1.011 | 1.001 | 135 Hz | 0.177 | 1.36 |
| V6 best @ 1.076 | 4.06% | 4.33% | 16.2% | 0.917 | 0.780 | 0.702 | 2.80 | 1.007 | 1.008 | 134 Hz | 0.170 | 1.31 |
| V7 best @ 1.050 | 3.69% | 4.04% | 12.9% | 0.910 | 0.791 | 0.688 | 2.80 | 1.007 | 1.001 | 140 Hz | 0.133 | 1.03 |
| V8 best @ 1.094 | 3.61% | 3.98% | 19.4% | 0.911 | 0.786 | 0.684 | 2.80 | 1.008 | 0.980 | 140 Hz | 0.159 | 1.22 |
| V9 best @ 1.058 | 3.54% | 3.88% | 16.1% | 0.924 | 0.768 | 0.691 | 2.81 | 1.012 | 1.009 | 128 Hz | 0.155 | 1.20 |
| V10 best @ 1.077 | 3.54% | 3.85% | 20.0% | 0.920 | 0.772 | 0.686 | 2.82 | 1.016 | 1.010 | 130 Hz | 0.159 | 1.24 |
| Real recordings | 5.09% | 5.24% | | 0.767 | | | 2.79 | | | 147 Hz | 0.131 | |

Paired per clip, same sentence and seed (wins / ties / losses for the first-named adapter):

- Word error: Base vs V5 13/18/5; Base vs V6 12/18/6; Base vs V7 11/21/4; Base vs V8 10/23/3; Base vs V9 11/20/5; Base vs V10 12/16/8; V5 vs V6 4/24/8; V5 vs V7 4/24/8; V5 vs V8 6/25/5; V5 vs V9 3/27/6; V5 vs V10 3/26/7; V6 vs V7 6/25/5; V6 vs V8 7/24/5; V6 vs V9 4/27/5; V6 vs V10 4/27/5; V7 vs V8 6/25/5; V7 vs V9 5/25/6; V7 vs V10 5/24/7; V8 vs V9 4/26/6; V8 vs V10 6/22/8; V9 vs V10 6/25/5.
- Style similarity to the real recording: Base vs V5 0/0/36; Base vs V6 0/0/36; Base vs V7 5/0/31; Base vs V8 4/0/32; Base vs V9 0/0/36; Base vs V10 2/0/34; V5 vs V6 17/0/19; V5 vs V7 25/0/11; V5 vs V8 26/0/10; V5 vs V9 23/0/13; V5 vs V10 26/0/10; V6 vs V7 21/0/15; V6 vs V8 23/0/13; V6 vs V9 23/0/13; V6 vs V10 23/0/13; V7 vs V8 22/0/14; V7 vs V9 18/0/18; V7 vs V10 17/0/19; V8 vs V9 14/0/22; V8 vs V10 14/0/22; V9 vs V10 19/0/17.
- Speaker similarity to the real recording: Base vs V5 3/0/33; Base vs V6 4/0/32; Base vs V7 3/0/33; Base vs V8 2/0/34; Base vs V9 4/0/32; Base vs V10 3/0/33; V5 vs V6 18/0/18; V5 vs V7 17/0/19; V5 vs V8 15/0/21; V5 vs V9 27/0/9; V5 vs V10 21/0/15; V6 vs V7 11/0/25; V6 vs V8 15/0/21; V6 vs V9 25/0/11; V6 vs V10 21/0/15; V7 vs V8 19/0/17; V7 vs V9 27/0/9; V7 vs V10 27/0/9; V8 vs V9 27/0/9; V8 vs V10 23/0/13; V9 vs V10 15/0/21.
- Speaker similarity to the reference clip: Base vs V5 28/0/8; Base vs V6 25/0/11; Base vs V7 31/0/5; Base vs V8 29/0/7; Base vs V9 15/0/21; Base vs V10 19/0/17; V5 vs V6 19/0/17; V5 vs V7 23/0/13; V5 vs V8 22/0/14; V5 vs V9 8/0/28; V5 vs V10 13/0/23; V6 vs V7 25/0/11; V6 vs V8 23/0/13; V6 vs V9 9/0/27; V6 vs V10 15/0/21; V7 vs V8 16/0/20; V7 vs V9 7/0/29; V7 vs V10 11/0/25; V8 vs V9 6/0/30; V8 vs V10 10/0/26; V9 vs V10 25/0/11.

The V5 to V8 rates and clips are the earlier report's. V9's rate comes from all 36 matched clips of its rate-1.0 grid; V10's from the same 36 sentences matched against the V8 manifest, because one of the twelve (text 9, `qwen_2511_tutorial_0070`) is not in V10's own audited manifest and the app's method on V10's manifest would have used 33 sentences and given 1.068. The values land within 0.008 of what the app had already saved for both runs from different sentences.

With their own rates all six adapters speak within 2 percent of the narrator (2.80 to 2.82 words per second against 2.79) and their median duration ratios sit at 1.00, which is what the calibration is for. V9 and V10 now share the lowest strict word error of the adapters (3.54 percent corpus; means 3.88 and 3.85), but the paired counts say the margin is a seed's worth: V10 over V8 8 wins to 6, V9 over V8 6 to 4, V9 and V10 6 to 5. Pauses tighten with the rate for every run (V7 1.03 of the narrator's pause time, V9 1.20, V8 1.22, V10 1.24, V6 1.31). The reference effect of condition 1 is unchanged by the rate: V9 and V10 keep the highest similarity to the reference clip (0.920 to 0.924) and the lowest similarity to the real recordings of the adapters (0.768 to 0.772 against 0.780 to 0.791), and their pitch stays at 128 to 130 Hz against 140 for V7 and V8. V7 has the best identity to the real recordings of the six in this condition (0.791; above V9 and V10 in 27 of 36 clips each) and the narrator's pauses; V6 and V5 keep the highest style similarity.

### Blind listening at calibrated rates, five-way ranking against the real recording

Each group judged once, five clips behind letters A to E, three groups needing a second conversation; every judgment returned with high confidence.

| Model | Mean rank (1 is best) | Ranked first | Ranked last | Pronunciation 1-5 | Naturalness 1-5 | Voice similarity to the speaker 1-5 | Audio quality 1-5 | Pace same / slower / faster than real | Clips with wrong words | Clips with artifacts noted |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| V6 best @ 1.076 | 2.58 | 12 | 4 | 4.81 | 4.14 | 4.64 | 4.47 | 26 / 9 / 1 | 1 | 23 |
| V7 best @ 1.050 | 2.67 | 9 | 3 | 4.83 | 4.19 | 4.69 | 4.47 | 26 / 7 / 3 | 0 | 23 |
| V8 best @ 1.094 | 3.11 | 4 | 6 | 4.83 | 3.97 | 4.58 | 4.31 | 27 / 6 / 3 | 0 | 27 |
| V9 best @ 1.058 | 3.47 | 5 | 12 | 4.78 | 3.83 | 4.58 | 4.39 | 23 / 11 / 2 | 0 | 25 |
| V10 best @ 1.077 | 3.17 | 6 | 11 | 4.78 | 3.92 | 4.61 | 4.42 | 25 / 8 / 3 | 2 | 26 |

Pairwise, from the rankings (times the row adapter was ranked above the column adapter, of 36; ties not counted):

| | V6 | V7 | V8 | V9 | V10 |
|---|---:|---:|---:|---:|---:|
| V6 | - | 20 | 21 | 25 | 21 |
| V7 | 16 | - | 22 | 24 | 22 |
| V8 | 15 | 14 | - | 23 | 16 |
| V9 | 11 | 12 | 13 | - | 19 |
| V10 | 15 | 14 | 20 | 17 | - |

The pace differences are gone (the listener judges 23 to 27 of 36 clips of every adapter as "same" pace), and the ranking is V6 and V7 first (2.58 and 2.67; V6 above V7 in 20 of 36), V8 and V10 next (3.11 and 3.17; V10 above V8 in 20 of 36), V9 last (3.47, ranked last in 12 groups). V9 and V10 are even against each other (19 to 17 for V9). The V6, V7 and V8 clips in this condition are the same files the earlier report's listener ranked V8 first, V7 second and V6 third (V8 above V6 in 30 of 36); today, with a different rubric, two different companions in each group and a fresh listener on a different day, the same three files come out V6, V7, V8 (V6 above V8 in 21 of 36). The order of V6, V7 and V8 is therefore inside what this listener can resolve, and the same bound applies to the gaps between V8, V10 and V9 here (0.06 and 0.30 of a rank). The notes for V9 and V10 are the pause and cadence remarks of condition 1 ("segmented delivery between phrases", "hesitation after 'I mean'", "rushed tempo") in one seed and "none; clean" in the next; V10 has two wrong-word clips ("to" for "the", a distorted "yielding"), V6 one ("twenty-fifth of the nine" for 2509), V7, V8 and V9 none.

## Condition 3: as the app deploys each run

This is what a user gets by selecting each adapter in Voice Generation: the run's own saved reference (V7 and V8 the 119 Hz clip they were trained with; V9 and V10 the median-matched 143 Hz clip they were trained with), the saved speaking rate, the installed voice decoder adapter at strength 1.0 for V8, V9 and V10 (V7 has none), and V9's swept guidance rate of 1.0. Same sentences, seeds and remaining settings. The V8 and V9 clips are the ones the decoder report measured (`outputs/grids/v9_vs_v8_*`); the V7 and V10 clips were generated today.

| Model | Corpus word error | Mean word error | Worst clip | Speaker similarity vs reference | Speaker similarity vs real | Style similarity vs real | Words per second | Matched rate ratio | Median duration ratio | Median pitch | Pause time fraction | Pause time vs real |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V7 @ 1.049, own reference, no decoder | 4.50% | 4.78% | 16.2% | 0.915 | 0.832 | 0.828 | 2.86 | 1.027 | 0.987 | 135 Hz | 0.115 | 0.87 |
| V8 @ 1.105, own reference, decoder | 4.42% | 4.66% | 16.1% | 0.837 | 0.901 | 0.877 | 2.84 | 1.024 | 1.006 | 145 Hz | 0.142 | 1.10 |
| V9 @ 1.066, own reference, decoder, CFG 1.0 | 3.47% | 3.80% | 17.1% | 0.897 | 0.893 | 0.867 | 2.82 | 1.014 | 1.015 | 144 Hz | 0.140 | 1.07 |
| V10 @ 1.075, own reference, decoder | 4.13% | 4.50% | 16.2% | 0.913 | 0.902 | 0.870 | 2.80 | 1.010 | 1.005 | 144 Hz | 0.141 | 1.09 |
| Real recordings | 5.09% | 5.24% | | 0.875 | | | 2.79 | | | 147 Hz | 0.131 | |

The similarity of the real recordings to the reference clip differs by reference: 0.803 for the clip V7 and V8 use, 0.875 for the clip V9 and V10 use, which is what "median-matched" means in practice.

Paired per clip, same sentence and seed (wins / ties / losses for the first-named adapter):

- Word error: V7 vs V8 5/27/4; V7 vs V9 5/19/12; V7 vs V10 6/23/7; V8 vs V9 3/23/10; V8 vs V10 4/26/6; V9 vs V10 6/28/2.
- Style similarity to the real recording: V7 vs V8 2/0/34; V7 vs V9 4/0/32; V7 vs V10 5/0/31; V8 vs V9 19/0/17; V8 vs V10 19/0/17; V9 vs V10 15/0/21.
- Speaker similarity to the real recording: V7 vs V8 0/0/36; V7 vs V9 1/0/35; V7 vs V10 1/0/35; V8 vs V9 22/0/14; V8 vs V10 17/0/19; V9 vs V10 13/0/23.
- Speaker similarity to the reference clip: V7 vs V8 36/0/0; V7 vs V9 25/0/11; V7 vs V10 18/0/18; V8 vs V9 4/0/32; V8 vs V10 1/0/35; V9 vs V10 8/0/28.

Three things separate this condition from the two before it. First, identity: with the decoder adapter and their own references, V8, V9 and V10 sit at 0.893 to 0.902 speaker similarity to the real recordings, above every adapter in the shared-reference conditions (0.76 to 0.79) and above V7's 0.832, and their median pitch is the narrator's (144 to 145 Hz against 147; V7 135). V10 is closer to the real recording than V9 in 23 of 36 clips and level with V8 (17 to 19). Second, what the reference does: V8's similarity to its own reference falls from 0.911 to 0.837 when the decoder renders the speaker instead of the low clip it was prompted with, while V10 keeps 0.913 because its reference already is the speaker's median voice; a V10 generation therefore matches both the prompt and the person. Third, intelligibility: V9 has the lowest strict word error (3.47 percent; below V10 in 6 paired clips against 2 and below V8 in 10 against 3), which is its adopted guidance rate of 1.0 at work, and V10 (4.13) is a few tenths ahead of V8 (4.42) and V7 (4.50). All four speak within 1 to 3 percent of the narrator's tempo; V7 alone rushes the pauses (0.87 of the real pause time), the three decoder runs pause 7 to 10 percent longer than the narrator, closer than any shared-reference condition.

### Blind listening as deployed, four-way ranking against the real recording

Each of the 36 groups was judged once with the four clips shuffled behind letters A to D; two groups needed a second conversation for the same reason as before, and one judgment came back without a confidence statement.

| Model | Mean rank (1 is best) | Ranked first | Ranked last | Pronunciation 1-5 | Naturalness 1-5 | Voice similarity to the speaker 1-5 | Audio quality 1-5 | Pace same / slower / faster than real | Clips with wrong words | Clips with artifacts noted |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| V7 @ 1.049, no decoder | 2.50 | 11 | 11 | 4.67 | 4.25 | 4.61 | 4.53 | 27 / 3 / 6 | 1 | 21 |
| V8 @ 1.105, decoder | 2.44 | 13 | 14 | 4.78 | 4.14 | 4.83 | 4.56 | 29 / 6 / 1 | 2 | 19 |
| V9 @ 1.066, decoder, CFG 1.0 | 2.31 | 10 | 8 | 4.83 | 4.14 | 4.78 | 4.56 | 27 / 7 / 2 | 1 | 23 |
| V10 @ 1.075, decoder | 2.33 | 8 | 11 | 4.75 | 4.03 | 4.78 | 4.58 | 28 / 7 / 1 | 1 | 20 |

Pairwise, from the rankings (times the row adapter was ranked above the column adapter, of 36; ties not counted):

| | V7 | V8 | V9 | V10 |
|---|---:|---:|---:|---:|
| V7 | - | 15 | 16 | 17 |
| V8 | 19 | - | 13 | 16 |
| V9 | 18 | 20 | - | 15 |
| V10 | 17 | 17 | 18 | - |

As deployed, the four runs are within a quarter of a rank of each other and no pair is separated by more than 20 to 13. V9 has the best mean rank (2.31) and the fewest last places (8); V10 is a hundredth behind (2.33) and is ranked above V9 in 18 of 36 groups against 15, above V8 in 17 against 16, and level with V7 (17 to 17, two ties). V9 is above V8 in 20 of 36, the largest margin in the table, and V8 is above V7 in 19. V8 collects the most first places (13, 11 of them outright) and the most last places (14): the listener either liked its clip best or least. On the rated scales the four differ by at most 0.2: V8 has the highest voice-similarity score (4.83), V9 the highest pronunciation (4.83), V7 the highest naturalness (4.25). Pace is judged "same" for 27 to 29 of 36 clips of every run, so the rankings rest on pauses and articulation, and the notes are the same for all four: an unnatural gap inside "handles this task amazingly" or "4K resolution" in one seed, "none" in the next. Word-level defects are one sentence: "2509" read as "twenty fifty nine" by V7 and V8 in one or two seeds and split into "fifty, ninety" by V9 in one clip; V10's only wrong word is "wholly" for "fully" in one clip.

Read with the objective table, the deployed comparison says the three decoder deployments (V8 with its later-trained decoder, V9, V10) are equally good to this listener and all three carry the speaker's identity better than V7; among them V10 and V9 are ahead of V8 by small margins, in opposite orders on the two measurements that separate them (V10 on identity to the real recordings, V9 on word error).

## Reading the results

- **As the app deploys them, V8, V9 and V10 are equally good to this listener, and all three carry the speaker's identity better than V7.** Mean ranks 2.31 to 2.50 in a four-way test, no pair further apart than 20 to 13, V10 above V9 in 18 of 36 and above V8 in 17. The measurements agree: V10 and V8 tie on similarity to the real recordings (0.902 and 0.901) with V9 at 0.893, all three at the narrator's pitch, and V9 has the lowest strict word error (3.47 percent against V10's 4.13 and V8's 4.42). The separate comparison of the same four deployments on the three test recordings with a Pro listener (`docs/qa_v10_native_comparison.md`) reached the same result: V10 preferred over V7 18 to 13, split with V8 (16 to 15) and V9 (15 to 16).
- **V10's GPT adapter is the most intelligible of the runs on strict word error** (3.32 percent at rate 1.0, the best of the six adapters, and level with V9 at 3.54 at calibrated rates), with no wrong-word clip at rate 1.0 in 36 and a clean listening record in the deployed condition (one substituted word in 36). What it does not do is beat V7 on the shared-reference listening tests, where V7 is first at rate 1.0 by a wide margin (17 sole first places, above every other adapter in 25 to 27 groups) and joint first with V6 at calibrated rates.
- **The shared reference is the wrong prompt for V9 and V10.** They were trained on the 143 Hz median-matched clip and the comparison clip is the 119 Hz clip V6 to V8 were trained on; given it, they render at 128 to 130 Hz, score highest on similarity to that clip and lowest on similarity to the person, and the listener ranks them below V6 and V7. With their own reference and decoder they render at 144 Hz and have the best identity numbers in the report. Conditions 1 and 2 are the right test of the GPT adapter alone; condition 3 is the right test of what a user hears.
- **The listening tests bound themselves.** The same V6, V7 and V8 files were ranked V8, V7, V6 yesterday and V6, V7, V8 today at calibrated rates by the same model family with a different rubric and different companions; the V6-versus-V7 preference had already flipped between sentence sets. Differences of less than about half a rank in a 36-group test are listener variation, and every gap between V8, V9 and V10 in this report is smaller than that. V7's lead at rate 1.0 and V9's last place in the shared-reference conditions are the only listening results larger than it.
- **Recommendation:** make **V10** the default adapter for this voice, with everything its run saved (reference `qwen_fine_tuning_0185`, speaking rate 1.075, decoder adapter at strength 1.0, default decoding). It is the output of the current pipeline end to end, it passed an independent final test that V9 did not have, it matches both its own prompt (0.913) and the person (0.902) where V8 has to choose (0.837 and 0.901), and no test in this report or in the test-recording comparison ranks it below V8 or V9 by more than listener noise. Keep **V9** (rate 1.066, guidance 1.0) as the alternative when strict word accuracy matters most, and **V8** as the fallback that yesterday's report recommended; **V7** remains the choice when a user wants the narrator's own pausing at speaking rate 1.0 without a decoder. Retire V6 and V5 for deployment. The claim this report supports is that V10 is at least as good as V8 and V9 as deployed, not that it is audibly better; a human should confirm on the 144 deployed clips under `outputs/grids/v10cmp_app_*` and `outputs/grids/v9_vs_v8_*`.

## Limits

One recording, 12 sentences, three seeds and one automated listener (a Flash model, chosen to match the earlier report; the separate v7-to-v10 comparison on the three test recordings used a Pro model and one fresh conversation per set as well). The sentences are multi-sentence narration clips of 8 to 16 seconds, like the training data; short prompts and other speaking styles are not covered. Speaker and style similarities are embedding proxies, the pause ratio counts silence rather than prosody, and the recognizer's own error on the real recordings (5.09 percent) bounds how much word error can say; the strict word error here differs from the app's own speech comparison, which accepts the dataset's spellings of names. The shared-reference conditions favor the three adapters trained with that clip; the deployed condition removes that but changes several things at once (reference, decoder, guidance), so it ranks deployments rather than isolating one training change. The V8 and V9 deployed grids were generated last night for the decoder report and their console logs were not kept; their decoder use rests on that report and on their measured signature (similarity to the reference falling while similarity to the real recording rises). V10's calibrated rate for condition 2 was matched against the V8 manifest, as explained above. Human listening was not performed; every clip is under `outputs/grids/v10cmp_*`, `outputs/grids/v5_to_v8_qwen2511_*` and `outputs/grids/v9_vs_v8_*`, and the blind bundles with their answer keys under `outputs/native_review/v10cmp_listening`, for that.
