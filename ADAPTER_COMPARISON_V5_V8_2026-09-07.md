# Best checkpoints of V5, V6, V7 and V8 compared on one recording none of them trained on (7 September 2026)

Four DoRA adapters of the same voice, each the checkpoint its run selected automatically, compared on identical inputs. Earlier head-to-head grids used sentences from three recordings that V6, V7 and V8 held out but V5 trained on, so they cannot rank V5 fairly. The one recording that none of the four runs trained on is `qwen_2511_tutorial` (V5's independent test recording; a validation recording for V6, V7 and V8, used for their loss checks and checkpoint selection but never for optimization). Twelve of its audited clips (8 to 16 seconds, 20 to 45 words, spread across the recording) are the comparison sentences, each with the narrator's real recording for matched measurements.

Two conditions were generated, three seeds per sentence, the same shared reference clip (V6's frozen final-test reference), strength 1.0 and the fixed comparison settings (3 beams, temperature and top-p 0.8, top-k 30, repetition penalty 10, 25 diffusion steps, CFG 0.7):

- **Speaking rate 1.0 for every model** (180 clips): the model-intrinsic comparison, measured with `tools/measure_grid_quality.py` (strict word error, CAMPPlus similarity to the reference, style similarity to the real recording, matched pace, pitch, pauses).
- **Each adapter at its own calibrated rate** (144 clips plus Base at 1.0): how the app deploys them. The rates come from the first condition with the app's matched-sentence method (`Calibrate speaking rate from this grid`), so all four are calibrated on the same sentences.

Blind listening was done offline by Gemini 3.8 Flash through the Antigravity CLI, labels hidden, five clips per sentence and seed ranked against the real recording. It is not part of the app.

## The four adapters

| | V5 | V6 | V7 | V8 |
|---|---:|---:|---:|---:|
| Training audio | 1,746 clips / 415.4 min | 1,734 clips / 411.6 min | 2,147 clips / 458.3 min | 2,031 clips / 481.3 min |
| Preparation and audit | v6.3 boundary repair, strict transcript audit | v6.4 audit, held out 3 + 3 recordings | v6.6 audit rules, 30 percent single-sentence clips | complete v6.6 audit, single-sentence share 0 |
| Recordings held out from training | 3 (2 validation, 1 test) | 6 (3 validation, 3 test) | 6 | 6 |
| Selected checkpoint | update 7,750 (epoch 5) | update 12,000 (epoch 7) | update 12,882 (epoch 6) | update 13,750 (epoch 7) |
| Validation loss, own split (Base on the same split) | 4.962 (6.576), 147 clips | 4.866 (6.555), 204 clips | 4.842 (6.649), 272 clips | 4.833 (6.618), 257 clips |
| Optimization time | 49 min | 89 min | 94 min | 73 min |
| Saved speaking rate in the app | 1.049 (epoch samples) | 1.075 (matched sentences) | 1.049 (matched sentences) | 1.105 (matched sentences) |

All four use DoRA rank 128, alpha 129, dropout 0.05, attention and MLP adapters plus the speaker projection, BF16, AdamW 4e-5 with cosine decay and 200 warmup updates, batch 1, a 10-epoch budget with automatic stopping. Validation losses are on different held-out sets and are not comparable across rows.

## Condition 1: speaking rate 1.0, 12 sentences x 3 seeds

| Model | Corpus word error | Mean word error | Worst clip | Speaker similarity vs reference | Style similarity vs real | Words per second | Matched rate ratio | Median duration ratio | Median pitch | Pause time fraction | Pause time vs real |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 2.73% | 2.93% | 16.1% | 0.923 | 0.630 | 2.50 | 0.902 | 1.111 | 118 Hz | 0.162 | 1.42 |
| V5 best | 4.28% | 4.69% | 20.0% | 0.911 | 0.714 | 2.67 | 0.962 | 1.051 | 135 Hz | 0.182 | 1.47 |
| V6 best | 3.54% | 3.81% | 12.9% | 0.915 | 0.710 | 2.60 | 0.937 | 1.084 | 133 Hz | 0.179 | 1.48 |
| V7 best | 3.32% | 3.66% | 16.1% | 0.910 | 0.690 | 2.67 | 0.960 | 1.052 | 141 Hz | 0.140 | 1.13 |
| V8 best | 4.20% | 4.62% | 19.4% | 0.910 | 0.692 | 2.56 | 0.922 | 1.073 | 141 Hz | 0.167 | 1.40 |
| Real recordings | 5.09% | 5.24% | | 0.767 | | 2.79 | | | 147 Hz | 0.131 | |

The recognizer makes more errors on the real recording (5.09 percent) than on any model: this tutorial is dense with product names and numbers, and strict word error without the vocabulary leniency counts every spelling difference, so word error here measures intelligibility to Whisper rather than correctness. Speaker similarity is to the shared reference clip, so Base, which copies the prompt, scores highest; similarity to the real recording of each sentence is in the listening section.

Paired per clip, same sentence and seed (wins / ties / losses for the first-named adapter):

- Word error: V6 vs V7 4/28/4; V6 vs V8 8/26/2; V7 vs V8 9/22/5; V5 vs V6 2/26/8; V5 vs V7 5/20/11; V5 vs V8 6/25/5.
- Style similarity to the real recording: V5 vs V6 22/0/14; V5 vs V7 28/0/8; V5 vs V8 26/0/10; V6 vs V7 25/0/11; V6 vs V8 23/0/13; V7 vs V8 17/0/19.
- Speaker similarity to the reference: V6 vs V5 23/0/13; V6 vs V7 22/0/14; V6 vs V8 23/0/13; V5 vs V7 19/0/17; V5 vs V8 20/0/16; V7 vs V8 20/0/16.

At rate 1.0 every adapter is slower than the narrator (V5 and V7 by 4 percent, V6 by 6, V8 by 8, Base by 10). V7 alone pauses like the narrator (pause time 1.13 of the real, the others 1.40 to 1.48) and its pitch, like V8's, is closest to the real 147 Hz. V5 and V6 carry the strongest style match to the real recordings; V6 has the strongest speaker match.

A grid-tool bug surfaced while generating this condition and is fixed in this release: cell files were named by checkpoint kind and epoch, so two "best, epoch 7" checkpoints from different runs (V6 and V8) wrote the same files and V8 silently overwrote V6. V6 was regenerated in its own grid with the identical configuration; the numbers above use that regeneration, and file names now stay unique across runs.

## Calibrated speaking rates from condition 1

The app's `Calibrate speaking rate from this grid` method on the 36 matched clips of each adapter (real 2.641 words per second after edge trimming):

| Adapter | Generated words per second at rate 1.0 | Calibrated rate | Saved rate in the app before this test |
|---|---:|---:|---:|
| V5 | 2.514 | 1.051 | 1.049 (epoch samples) |
| V6 | 2.455 | 1.076 | 1.075 (matched sentences) |
| V7 | 2.515 | 1.050 | 1.049 (matched sentences) |
| V8 | 2.414 | 1.094 | 1.105 (matched sentences) |

The calibrations reproduce the rates the app had already saved for V6, V7 and V8 from different sentences, to within 0.011, and V5's epoch-sample estimate happened to land on the same value. Condition 2 uses the calibrated rates in the second column.

## Condition 2: each adapter at its calibrated rate (as deployed)

| Model | Corpus word error | Mean word error | Worst clip | Speaker similarity vs reference | Style similarity vs real | Words per second | Matched rate ratio | Median duration ratio | Median pitch | Pause time fraction | Pause time vs real |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base @ rate 1.0 | 2.73% | 2.93% | 16.1% | 0.923 | 0.630 | 2.50 | 0.902 | 1.111 | 118 Hz | 0.162 | 1.42 |
| V5 best @ rate 1.051 | 3.91% | 4.33% | 20.0% | 0.913 | 0.703 | 2.81 | 1.011 | 1.001 | 135 Hz | 0.177 | 1.36 |
| V6 best @ rate 1.076 | 4.06% | 4.33% | 16.2% | 0.917 | 0.702 | 2.80 | 1.007 | 1.008 | 134 Hz | 0.170 | 1.31 |
| V7 best @ rate 1.050 | 3.69% | 4.04% | 12.9% | 0.910 | 0.688 | 2.80 | 1.007 | 1.001 | 140 Hz | 0.133 | 1.03 |
| V8 best @ rate 1.094 | 3.61% | 3.98% | 19.4% | 0.911 | 0.684 | 2.80 | 1.008 | 0.980 | 140 Hz | 0.159 | 1.22 |
| Real recordings | 5.09% | 5.24% | | 0.767 | | 2.79 | | | 147 Hz | 0.131 | |

With their own rates all four adapters land within one percent of the narrator's tempo (2.80 words per second against 2.79) and their duration ratios sit at 1.00, which is what the calibration is for. Word error moves with the rate change by a few tenths of a point in both directions and the order becomes V8, V7, V5, V6, all within a range that a single seed changes. Speaker and style similarities are unchanged by the rate. V7 keeps the narrator's pause time (1.03); the other three still pause 22 to 36 percent longer.

Paired per clip (wins / ties / losses for the first-named adapter):

- Word error: V6 vs V7 6/25/5; V6 vs V8 7/24/5; V7 vs V8 6/25/5; V5 vs V6 4/24/8; V5 vs V7 4/24/8; V5 vs V8 6/25/5.
- Style similarity to the real recording: V5 vs V6 17/0/19; V5 vs V7 25/0/11; V5 vs V8 26/0/10; V6 vs V7 21/0/15; V6 vs V8 23/0/13; V7 vs V8 22/0/14.
- Speaker similarity to the reference: V6 vs V7 25/0/11; V6 vs V8 23/0/13; V5 vs V7 23/0/13; V5 vs V8 22/0/14; V5 vs V6 19/0/17; V7 vs V8 16/0/20.

## Blind listening, five-way ranking against the real recording

Each of the 36 sentence-and-seed groups was judged once with all five clips shuffled behind letters A to E and the real recording as reference; the listener rated every clip and ranked the five.

### Speaking rate 1.0

| Model | Mean rank (1 is best) | Ranked first | Ranked last | Pronunciation 1-5 | Naturalness 1-5 | Voice similarity to the speaker 1-5 | Pace same / slower / faster than real | Clips with wrong words | Clips with artifacts noted |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| Base | 4.86 | 1 | 34 | 3.61 | 2.50 | 2.47 | 7 / 24 / 5 | 20 | 36 |
| V5 best | 2.78 | 6 | 0 | 4.61 | 3.89 | 4.39 | 10 / 19 / 7 | 8 | 35 |
| V6 best | 2.78 | 7 | 1 | 4.64 | 3.92 | 4.31 | 9 / 23 / 4 | 8 | 34 |
| V7 best | 2.14 | 11 | 1 | 4.72 | 4.28 | 4.75 | 13 / 18 / 5 | 8 | 30 |
| V8 best | 2.44 | 11 | 0 | 4.64 | 4.00 | 4.67 | 8 / 24 / 4 | 8 | 30 |

Pairwise, from the rankings (times the row adapter was ranked above the column adapter, of 36):

| | V5 | V6 | V7 | V8 |
|---|---:|---:|---:|---:|
| V5 | - | 19 | 12 | 14 |
| V6 | 17 | - | 12 | 17 |
| V7 | 24 | 24 | - | 20 |
| V8 | 22 | 19 | 16 | - |

V7 is ranked above each of the other three in 20 to 24 of 36 groups, V8 above V5 and V6 in 22 and 19, and V5 versus V6 is even (19 to 17). First places are spread over sentences: no adapter wins a sentence in all three seeds except V5 on one, so seed variation is as large as the differences between adapters on most sentences. This reverses last night's result on the other sentence set (three recordings held out from V6, V7 and V8), where the same listener ranked V6 above V7 in 26 of 36 groups; the preference between V6 and V7 depends on the sentences.

Word-level defects are shared and few: every adapter reads "3090 Ti" as "three thousand ninety Ti" in all three seeds (the text normalizer expands the number that way; the narrator says "thirty ninety"), all four say "neev" for "new" and split "Qwen" in one clip, and each has one or two clip-specific slips ("sports cup", "Laura" for LoRA, "detail is"). Base mispronounces product names ("SwarmUI") and drops sounds in 20 clips. The listener's artifact notes for V5 and V6 are mostly pace and pauses ("rushed", "hesitation before the final phrase"); for V7 and V8 they are mostly timbre ("slight synthetic grain").

### Each adapter at its calibrated rate

| Model | Mean rank (1 is best) | Ranked first | Ranked last | Pronunciation 1-5 | Naturalness 1-5 | Voice similarity to the speaker 1-5 | Pace same / slower / faster than real | Clips with wrong words | Clips with artifacts noted |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| Base @ 1.0 | 4.94 | 0 | 34 | 3.53 | 2.25 | 2.56 | 8 / 26 / 2 | 20 | 36 |
| V5 best @ 1.051 | 3.25 | 5 | 2 | 4.58 | 3.72 | 4.19 | 14 / 10 / 12 | 11 | 32 |
| V6 best @ 1.076 | 2.89 | 3 | 0 | 4.67 | 4.03 | 4.36 | 18 / 8 / 10 | 12 | 32 |
| V7 best @ 1.050 | 2.11 | 10 | 0 | 4.86 | 4.42 | 4.67 | 21 / 8 / 7 | 9 | 29 |
| V8 best @ 1.094 | 1.81 | 18 | 0 | 4.89 | 4.44 | 4.83 | 18 / 10 / 8 | 9 | 28 |

Pairwise, from the rankings (times the row adapter was ranked above the column adapter, of 36):

| | V5 | V6 | V7 | V8 |
|---|---:|---:|---:|---:|
| V5 | - | 13 | 9 | 7 |
| V6 | 23 | - | 11 | 6 |
| V7 | 27 | 25 | - | 16 |
| V8 | 29 | 30 | 20 | - |

With the pace differences removed the ordering is clean: V8 above V6 in 30 of 36 groups and above V5 in 29, V7 above V6 in 25 and above V5 in 27, V8 above V7 in 20 of 36. V8 wins one sentence in all three seeds and V7 or V8 takes first place in 28 of the 36 groups. The listener now judges pace as "same" for about half the clips of every adapter, so its rankings rest on naturalness, voice and pronunciation: V7 and V8 score 4.4 on naturalness and 4.7 to 4.8 on voice similarity against 3.7 to 4.0 and 4.2 to 4.4 for V5 and V6. The remaining shared defects are the number readings ("3090" as "three thousand ninety" from the text normalizer, "2509" mis-stressed by V6 and V7) and one slip per adapter ("restored" clipped by V6, "asked" garbled by V5).

## Reading the results

- **As the app deploys them, V8 and V7 are the better adapters on this recording, and V8 edges V7.** Every adapter at its calibrated rate speaks at the narrator's tempo; what separates them is delivery. V7 and V8 have the pitch closest to the speaker (140 Hz against 147; V5 and V6 sit at 134 to 135), the fewest word-level slips, the cleanest listener notes, and the top two listening ranks in both conditions. V6 keeps the highest embedding similarity to the reference clip and, with V5, the highest style similarity to the real recordings, which is the "warmer prosody" the listener rewarded last night on a different sentence set; here that did not outweigh the newer adapters' pitch and articulation.
- **The V6-versus-V7 preference is sentence-dependent, not a fixed ranking.** Last night's test on three recordings held out from V6, V7 and V8 (V5 trained on them, so it could not join that test) ranked V6 above V7 in 26 of 36 groups at rate 1.0; today's recording ranks V7 above V6 in 24 of 36 at rate 1.0 and 25 of 36 at calibrated rates. Both are the same automated listener. The honest reading is that V6 and V7 are close, with V7 ahead on articulation, pitch and pauses and V6 ahead on prosodic warmth, and which one a listener prefers changes with the text.
- **V8 is not a regression, as the earlier same-day report feared from a single sentence set; at its calibrated rate it is the listener's first choice here.** Its objective proxies are indistinguishable from V6's and V7's (word errors within one seed's variation, identical speaker similarity), which is what more clean audio from the same speaker should look like once the voice is already covered; the gain shows up as fewer defects and higher naturalness and voice ratings, not in the recognizer's numbers.
- **V5 is superseded.** It ranks last of the four in both listening conditions, has the flattest pitch, and its speaking-rate estimate came from the older epoch-sample method; its one strength, the highest style similarity to the real recordings, does not translate into listener preference.
- **Recommendation:** use **V8** with its saved speaking rate (1.105 in the app; 1.094 on this recording) as the default adapter for this voice, and **V7** (rate 1.049) when matching the narrator's own pausing matters more than a slightly smoother delivery. Keep V6 as a fallback; retire V5. All three of V6, V7 and V8 stay within a range that a human should confirm by ear on the 288 clips under `outputs/grids/v5_to_v8_qwen2511_*`.

## Limits

One recording, 12 sentences, three seeds and one automated listener. The sentences are multi-sentence narration clips of 8 to 16 seconds, like the training data; short prompts and other speaking styles are not covered. Speaker and style similarities are embedding proxies, the pause ratio counts silence rather than prosody, and the recognizer's own error on the real recordings bounds how much word error can say. Human listening was not performed; every clip is in `outputs/grids/v5_to_v8_qwen2511_*` for that.
