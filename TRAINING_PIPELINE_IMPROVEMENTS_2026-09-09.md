# Five training-pipeline changes tested against V10 (8 to 9 September 2026)

Five candidate improvements to the automatic training pipeline were implemented, each tested on the recordings V10 was trained from (`Lora_Training_Dataset_V2`, the `furkan_v10_curated_20s` audit, or a fresh preparation of the same 17 recordings), and each set as the default only where it measured better than V10's recipe. Every measurement below is either V10's own frozen speech benchmark (12 validation sentences, three seeds, the median-matched reference `qwen_fine_tuning_0185`, identical for every run trained on that dataset), the 12 `qwen_2511_tutorial` sentences of the adapter comparison reports generated as the app deploys each candidate, or the new 33-sentence per-length set. Everything ran on the RTX 5090 (`cuda:0`).

| Change | Where it lives | Result | Default |
|---|---|---|---|
| Select checkpoints by the decoder gate's score, with a pause term | `speech_metrics.select_recommendation` | same picks for V9 and V10; picks V8's epoch 6, which pauses closer to the narrator with no regression | **on** (it is now the selection rule) |
| Average the last saved checkpoints | `checkpoint_average.py`, `average_last_checkpoints` | averages of the last two and three V10 updates both lose to the final file | off (0); available |
| Train the decoder adapter on the GPT's own codes | `decoder_adapter_code_source` (`real`, `gpt`, `mixed`) | GPT codes trade identity for intelligibility in the wrong direction at both strengths | `real`; `gpt`/`mixed` available |
| Sweep the GPT learning rate | training runs at 2e-5, 8e-5, 1.6e-4 | higher rates reach a lower token loss but make more word errors; 1.6e-4 matches V10 after a quarter of the updates, then degrades | 4e-5 stays |
| Deliberate short and medium clips | `medium_clip_fraction`, clear-pause rule, `length_aim` tags | V11: higher identity and style at every length, pauses closer to the narrator, rate at the narrator's pace, same word error; listener split 51 to 48 | **0.25 short, 0.25 medium** |

## 1. Selection by the deployment score

The speech comparison used to pick, among candidates that pass the Base guards, the lowest mean transcript error, treat candidates within the bootstrap interval as ties, and break ties by validation loss. Since every candidate of a converged run lands inside that interval, the loss was doing the choosing, and the loss is flat across the last epochs. The candidates are now ranked by the score the voice decoder gate already uses, plus a pause term:

    score = paired speaker-similarity gain over Base
            - 4 x max(0, paired mean word-error increase)
            + 0.05 x (|log(Base pause ratio)| - |log(candidate pause ratio)|)

One point of word error costs 0.04 of similarity; a total pause-time ratio moving from 1.5 to 1.0 of the narrator's earns about 0.02. Candidates within 0.002 are tied and the lower validation loss decides. The report lists the score and its three parts for every candidate. Replayed on the saved reports it keeps V9's pick (epoch 6, +0.010 against final's ineligible -0.064) and V10's (final, +0.026 against epoch 6's -0.014) and changes V8's from the final file (+0.098) to epoch 6 (+0.109), the pause term deciding (+0.0045 against -0.0051).

That V8 change was tested as deployed on the 12 `qwen_2511_tutorial` sentences, own reference, rate 1.105, decoder adapter, three seeds:

| V8 checkpoint | Corpus word error | Speaker similarity vs real | Style vs real | Pause time vs real |
|---|---:|---:|---:|---:|
| final, update 13,750 (old rule) | 4.42% | 0.901 | 0.877 | 1.10 |
| epoch 6, update 12,000 (new rule) | 4.42% | 0.897 | 0.872 | 1.06 |

Paired per clip, epoch 6 against final: word error 10 wins, 19 ties, 7 losses; speaker similarity 18 to 18; style 16 to 20; pause time closer to the narrator's in 23 of 36. The new rule's pick is not worse on any measure and is closer on the one the pause term targets. The same rule also handled the sweep runs correctly (section 4): it recommended Base for a run whose adapters gained identity but cost more word error than the gain was worth.

## 2. Checkpoint averaging

`indextts/training/checkpoint_average.py` averages the adapter parameters (low-rank factors, DoRA magnitudes, the fully trained speaker projection) of the last N saved updates into `<name>_avg_ep<first>_<last>.safetensors`, which the trainer writes after training, the checkpoint evaluation measures, and the speech comparison always includes as one more candidate. On V10, rendered on its own benchmark without a decoder and compared pairwise with the final file:

| Averaged members | Speaker similarity vs real | Mean transcript error | Worst clip | Pause time vs real | Deployment score vs Base |
|---|---:|---:|---:|---:|---:|
| final only (V10 as selected) | 0.8633 | 1.33% | 8.0% | 1.60 | +0.026 |
| epochs 5, 6 and final (10,170 / 12,204 / 14,000) | 0.8577 (paired -0.0056, interval -0.013 to +0.000) | 2.43% (+1.10 points) | 27.3% | 1.65 | -0.025 |
| epoch 6 and final (12,204 / 14,000) | 0.8588 (paired -0.0044, interval -0.009 to -0.000) | 1.90% (+0.56 points) | 24.0% | 1.58 | -0.000 |

Both averages are worse than the final file on identity and intelligibility, the three-member one badly. The mean of nearby DoRA checkpoints is not a better DoRA here. The option stays (some voices may differ) with default 0, and the two averaged files were removed from the V10 folder.

## 3. Decoder adapter trained on the GPT's codes

At inference the decoder renders codec features of the codes the GPT generates, but the adapter was trained on codes quantized from the recordings. `decoder_adapter_code_source` adds `gpt`: before the decoder trains, the selected checkpoint predicts every clip's codes teacher-forced (aligned with the real mel frame by frame), greedily plus two sampled variants with the generation settings, and each clip renders from a different variant every epoch; `mixed` uses real codes for half the examples. On V10 the greedy prediction agrees with the real code at 13 percent of positions. Trained in an isolated copy of the V10 run with the same rank, alpha, learning rate and gate:

| Decoder adapter, V10 final, speech benchmark through the full pipeline | Speaker similarity gain vs no decoder | Word error change | Gate score | Verdict |
|---|---:|---:|---:|---|
| real codes (shipped), strength 1.0 | +0.0329 | -0.06 points | +0.0329 | installed |
| real codes, strength 0.6 | +0.0247 | -0.18 points | +0.0247 | passes |
| GPT codes, strength 1.0 | +0.0371 | +0.49 points | +0.0175 | passes |
| GPT codes, strength 0.6 | +0.0280 | -0.25 points | +0.0280 | passes, recommended |

As deployed on the 12 `qwen_2511_tutorial` sentences (own reference, rate 1.075, three seeds):

| V10 decoder | Corpus word error | Speaker similarity vs real | Style vs real | Median pitch |
|---|---:|---:|---:|---:|
| none | 4.06% | 0.875 | 0.832 | 145 Hz |
| real codes, 1.0 (shipped) | 4.13% | 0.902 | 0.870 | 144 Hz |
| GPT codes, 1.0 | 5.38% | 0.900 | 0.864 | 140 Hz |
| GPT codes, 0.6 | 3.54% | 0.885 | 0.841 | 140 Hz |

At strength 1.0 the GPT-code adapter costs 1.25 points of word error (worse in 10 paired clips, better in 1) for no identity gain; at 0.6 it keeps only +0.010 of the +0.027 the real-code adapter delivers (lower identity in 34 of 36 clips). The hypothesis that the decoder should see the GPT's distribution did not survive contact with the gate: the real codes carry cleaner content, and the pretrained decoder already generalizes to the GPT's codes. `real` stays the default; the `mixed` variant was not measured, since both strengths of the pure variant move in the wrong direction and a half-and-half mix cannot beat the real-code adapter on the gate's score.

## 4. Learning-rate sweep

V10's recipe at 8e-5, 1.6e-4 and 2e-5, everything else identical (alpha 129, FP32 speaker projection, same dataset, seed and frozen benchmark), GPT phase and speech comparison only:

| Learning rate | Stopped at update | Best validation loss (update) | Speech candidate the policy selects | Its mean transcript error / speaker similarity vs real / pause time vs real | Score vs Base |
|---:|---:|---:|---|---|---:|
| 2e-5 | 15,750 | 4.911 (15,750, epoch 8) | none, Base recommended (best adapter: epoch 7) | 2.1% / 0.860 / 1.54 | -0.004 |
| 4e-5 (V10) | 14,000 | 4.833 (12,204, epoch 6) | final, update 14,000 | 1.3% / 0.863 / 1.60 | +0.026 |
| 8e-5 | 11,750 | 4.789 (10,000, epoch 5) | none, Base recommended (best adapter: update 10,000) | 2.0% / 0.857 / 1.77; the final file loops (321% error, 11 of 39 clips flagged) | -0.014 |
| 1.6e-4 | 7,750 | 4.772 (6,102, epoch 3) | epoch 2, update 4,068 | 1.5% / 0.863 / 1.52 | +0.022 |

Base on this benchmark: 1.2 percent, 0.834, 1.62.

The validation loss orders the four runs one way (1.6e-4 lowest, then 8e-5, 4e-5, 2e-5) and generated speech orders them another. Both higher rates reach a lower token loss and then make more word errors: at 8e-5 every candidate costs more intelligibility than its identity gain is worth and the last file degenerates into repetition; at 1.6e-4 the loss-best epoch 3 is ineligible (3.7 percent error) and only epoch 2 is usable. That epoch 2 is the one interesting result: V10-level speech (score +0.022 against +0.026, a tie within the policy's margin) after 4,068 updates instead of 14,000, so a run at 1.6e-4 with its checkpoint chosen by the speech comparison gets the same voice in a quarter of the optimization time. But every later epoch is worse and the loss-based early stopping does not know it, so it is a speed option for someone watching the run, not a default. The lower rate, 2e-5, is worse on every measure. The default stays 4e-5.

Two side findings. The 8e-5 run's first speech comparison ran inside a code snapshot whose `models` junction pointed at itself, failed before measuring, and was rerun from the main checkout; the numbers above are from that rerun. And the new selection policy recommended Base for two of the three sweep runs where the old rule would have deployed an adapter that trades a point of word error for two hundredths of similarity.

## 5. Short and medium clips

Every dataset so far packed the recordings into clips near 14 seconds, so a voice trained on them had seen almost no single sentences: V10's training split has one clip under 6 seconds and 31 under 9. Preparation now has two shares, **Share of single-sentence clips** (aim 6 seconds) and the new **Share of medium clips** (aim 10 seconds), and a shorter clip is only cut where each inner edge sits in a clear pause (about 200 ms of quiet with the default padding); a start that finds none keeps the target. Each clip records which aim produced it and the dataset summary counts them. A first attempt with a 370 ms requirement changed nothing on this narrator (11 more short clips), because his sentence boundaries rarely carry that much silence; the 200 ms rule with shares of 0.25 and 0.25 gave:

| | V10 preparation | V11 preparation (0.25 short, 0.25 medium) |
|---|---:|---:|
| Prepared clips / minutes / words | 2,576 / 609.7 / 96,553 | 2,749 / 610.4 / 96,815 |
| Clips 3 to 6 s / 6 to 9 s / 9 to 12 s / 12 to 15 s / over 15 s | 2 / 43 / 307 / 1,382 / 842 | 82 / 234 / 450 / 1,143 / 840 |
| Clips cut at the short / medium / target aim | 0 / 0 / 2,576 | 189 / 282 / 2,278 |
| Audit rejections (same six held-out recordings, same rules) | 150 | 162 |
| Training clips / minutes / words | 2,034 / 482.0 / 76,267 | 2,180 / 483.3 / 76,582 |
| Training clips under 6 s / 6 to 9 s | 1 / 31 | 65 / 190 |

Same audio, more and shorter clips; the twelve extra rejections are short clips failing the boundary-word and speaker-window checks. **V11** trained on it with V10's recipe unchanged (4e-5, alpha 129, FP32 speaker projection, the same reference rule, decoder adapter on real codes, sweep, final test):

| | V10 | V11 |
|---|---|---|
| GPT phase | 14,000 updates, best loss 4.833 at 12,204 (epoch 6), final selected | 16,500 updates, best loss 4.825 at 15,000 (epoch 7), final (16,500, epoch 8) selected |
| Reference chosen (median-matched) | `qwen_fine_tuning_0185`, 143 Hz | `qwen_fine_tuning_0199`, 143 Hz |
| Own speech benchmark, selected file vs Base | 1.3% vs 1.2% error, 0.863 vs 0.834 identity, pause 1.60 vs 1.62 (score +0.026) | 0.6% vs 0.6% error, 0.849 vs 0.827 identity, pause 1.14 vs 1.55 (score +0.037); four of its twelve prompts are under 8 s |
| Calibrated speaking rate | 1.075 (7 percent slow at rate 1.0) | 0.978 (at the narrator's pace at rate 1.0) |
| Decoder adapter gate, strength 1.0 | +0.033 identity, -0.06 points word error | +0.038 identity (interval +0.028 to +0.049), +0.06 points |
| Decoding sweep | defaults kept | defaults kept |
| Independent final test, selected deployment vs Base | 1.6% vs 1.5% error, 0.877 vs 0.829 identity, pause 1.20 vs 1.41 | 1.1% vs 2.7% error, 0.878 vs 0.823 identity, pause 1.49 vs 1.77; its prompts include four under 7 s |

The two benchmarks and final tests draw different prompts (V11's include short ones), so the direct comparison is a new set of 33 held-out sentences, 9 short (4 to 7 s), 12 medium (7 to 11 s) and 12 long (11 to 20 s), from the three validation recordings neither run trained on, generated as the app deploys each run with three seeds:

| Bucket | Run | Corpus word error | Mean | Worst clip | Speaker similarity vs real | Style vs real | Matched rate ratio | Pause time vs real | Median pitch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| short (27 clips) | V10 | 2.92% | 4.17% | 27.3% | 0.879 | 0.822 | 0.937 | 1.70 | 140 Hz |
| | V11 | 4.29% | 6.33% | 54.5% | 0.893 | 0.856 | 0.935 | 1.47 | 141 Hz |
| medium (36) | V10 | 7.88% | 8.77% | 29.2% | 0.871 | 0.839 | 1.036 | 1.35 | 146 Hz |
| | V11 | 7.09% | 8.04% | 26.7% | 0.879 | 0.858 | 1.024 | 1.25 | 145 Hz |
| long (36) | V10 | 3.75% | 4.15% | 17.9% | 0.896 | 0.850 | 1.013 | 1.26 | 147 Hz |
| | V11 | 4.05% | 4.44% | 21.4% | 0.905 | 0.856 | 0.968 | 1.34 | 145 Hz |
| all (99) | V10 | 4.93% | 5.84% | 29.2% | 0.882 | 0.838 | 1.001 | 1.35 | 145 Hz |
| | V11 | 5.07% | 6.26% | 54.5% | 0.892 | 0.857 | 0.980 | 1.33 | 144 Hz |

The recognizer's own error on the real recordings is 2.0 percent (short), 8.0 (medium) and 4.8 (long): the medium sentences are the acronym-dense ones. Paired per clip, V11 against V10: word error 16 wins, 64 ties, 19 losses; speaker similarity 56 to 43; style 64 to 35; pause time closer to the narrator's in 58 of 99 (17 of 27 short, 23 of 36 medium). The worst clips of both runs are one short sentence, "the new FLUX SRPO model NVFP4 mixed precision", whose acronyms the strict recognizer spells its own way; V11 also read "model" as "module" in two seeds there. Nothing loops or truncates.

On the 12 `qwen_2511_tutorial` sentences of the adapter comparison reports, as the app deploys each run (own reference, saved rate, decoder adapter, swept decoding), V11 joins the four runs measured there yesterday:

| Model | Corpus word error | Mean word error | Worst clip | Speaker similarity vs reference | Speaker similarity vs real | Style similarity vs real | Words per second | Matched rate ratio | Median duration ratio | Median pitch | Pause time fraction | Pause time vs real |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V7 @ 1.049, own reference, no decoder | 4.50% | 4.78% | 16.2% | 0.915 | 0.832 | 0.828 | 2.86 | 1.027 | 0.987 | 135 Hz | 0.115 | 0.87 |
| V8 @ 1.105, own reference, decoder | 4.42% | 4.66% | 16.1% | 0.837 | 0.901 | 0.877 | 2.84 | 1.024 | 1.006 | 145 Hz | 0.142 | 1.10 |
| V9 @ 1.066, own reference, decoder, CFG 1.0 | 3.47% | 3.80% | 17.1% | 0.897 | 0.893 | 0.867 | 2.82 | 1.014 | 1.015 | 144 Hz | 0.140 | 1.07 |
| V10 @ 1.075, own reference, decoder | 4.13% | 4.50% | 16.2% | 0.913 | 0.902 | 0.870 | 2.80 | 1.010 | 1.005 | 144 Hz | 0.141 | 1.09 |
| V11 @ 0.978, own reference, decoder | 4.06% | 4.29% | 12.9% | 0.901 | 0.917 | 0.894 | 2.70 | 0.972 | 1.020 | 144 Hz | 0.126 | 1.01 |

Paired per clip, V11 against V10: word error 7 wins, 23 ties, 6 losses; speaker similarity to the real recordings 26 to 10; style 27 to 9. Against V8: 9/22/5, 28 to 8, 26 to 10. Against V9: 3/26/7, 27 to 9, 30 to 6. V11 has the highest identity and style of any run on these sentences, the lowest worst clip, and pauses within one percent of the narrator's total, at a word error level with V10 and V8; it runs three percent slower than the narrator here because its rate was calibrated on a benchmark that mixes short and long sentences, on which it sits at the narrator's pace.

### Blind listening

Gemini 3.8 Flash through the Antigravity CLI, one fresh conversation per sentence and seed, clips renamed A and B (or A to D), the real recording as reference, every clip rated and ranked (the harness and rubric of the adapter comparison report; bundles, prompts, raw responses and answer keys under `outputs/native_review/v10cmp_listening/lengthmix` and `app4`).

V10 against V11 on the 33 per-length sentences, 99 pairs:

| Bucket | V11 preferred | V10 preferred | Pronunciation V10 / V11 | Naturalness | Voice similarity | Audio quality | Pace judged same | Wrong-word clips |
|---|---:|---:|---|---|---|---|---|---|
| short (27) | 13 | 14 | 4.85 / 4.85 | 4.37 / 4.30 | 4.93 / 4.85 | 4.67 / 4.63 | 22 / 21 of 27 | 0 / 1 |
| medium (36) | 21 | 15 | 4.81 / 4.92 | 4.33 / 4.53 | 4.97 / 4.97 | 4.81 / 4.75 | 31 / 30 of 36 | 3 / 2 |
| long (36) | 17 | 19 | 4.81 / 4.86 | 4.31 / 4.36 | 4.89 / 4.94 | 4.69 / 4.72 | 30 / 30 of 36 | 2 / 3 |
| all (99) | 51 | 48 | 4.82 / 4.88 | 4.33 / 4.40 | 4.93 / 4.93 | 4.73 / 4.71 | 83 / 81 of 99 | 5 / 6 |

Artifacts were noted in 52 V10 clips and 44 V11 clips. A tie: the listener cannot separate the two on short and long sentences and leans to V11 on medium ones, which are the two-sentence clips the medium share adds.

V8, V9, V10 and ### Verdict

On the objective measures V11 is the better deployment: identity and style closer to the speaker at every prompt length (56 of 99 and 64 of 99 paired clips on the per-length set, 26 of 36 and 27 of 36 on the twelve sentences), pauses closer to the narrator's in 58 of 99 clips and within one percent on the twelve sentences, a speaking rate that no longer needs a seven percent correction, the lowest worst clip of any run on the twelve sentences, and corpus word error level with V10 (16 wins, 64 ties, 19 losses on the per-length set; 7, 23 and 6 on the twelve sentences). Under the pipeline's own deployment score, V11 beats V10 as a baseline by about +0.02 on the twelve sentences and by less than +0.01 on the per-length set, against the +0.005 margin the gates use. The listener hears the two as equals on both sets (51 to 48 on 99 pairs, 17 to 18 on the twelve sentences), likes V11's medium two-sentence clips and its voice, and gives it the lowest naturalness score in the four-way test, where V9 is ahead of it by more than V9 was ahead of V10 yesterday. That is the pattern the earlier reports called "not a regression, measurably closer to the speaker, not audibly better", and it comes from the same audio with a different cut, which costs nothing at training time. The clip-length shares therefore become the preparation defaults (0.25 and 0.25), and **V11 is the recommended adapter for this voice** with everything its run saved: reference `qwen_fine_tuning_0199`, speaking rate 0.978, decoder adapter at strength 1.0, default decoding. The claim is the same one the adapter comparison report made for V10: at least as good as V10 as deployed and measurably closer to the speaker, not audibly better. V10 stays as the fallback, and V9 remains the choice when strict word accuracy matters most (it is the only run of the four with one wrong-word clip in 36 and the best four-way listening rank today). For long-form narration where V11's three percent slower pace on long sentences matters, a rate of 1.0 instead of the saved 0.978 removes it. A human should confirm on the 108 deployed clips under `outputs/grids/v10cmp_app_v11` and `outputs/grids/lengthmix_v10` and `lengthmix_v11`.

ed of sentence 6, which the recognizer also counts), "2509" read as "twenty fifty-nine" in two seeds of sentence 5 (V8 reads it the same way in two seeds; V10 and V9 read it correctly), "I will" for "I mean" and "you're" for "you are". Counted by the recognizer, 19 of V11's 36 clips have at least one word error against 17 of V10's and 19 of V8's, the same sentences for all three (the "3090" and "Qwen image edit" sentences); the corpus totals are equal because V11's errors are shorter. The naturalness notes are the ones every run gets on these sentences, a halting "handles this task amazingly" or a pitch spike on one stressed word in one seed and "none" in the next.

### Verdict

On the objective measures V11 is the better deployment: identity and style closer to the speaker at every prompt length (56 of 99 and 64 of 99 paired clips on the per-length set, 26 of 36 and 27 of 36 on the twelve sentences), pauses closer to the narrator's in 58 of 99 clips and within one percent on the twelve sentences, a speaking rate that no longer needs a seven percent correction, the lowest worst clip of any run on the twelve sentences, and word error level with V10 (16 wins, 64 ties, 19 losses; the two extra medium-sentence wins and the two extra short-sentence losses cancel). Under the pipeline's own deployment score, V11 beats V10 as a baseline by about +0.02 on the twelve sentences and by less than +0.01 on the per-length set, with V10's +0.005 margin the threshold the gates use. The listener hears them as equals. That is the pattern the earlier reports called "not a regression, measurably better on identity, inaudible to one listener", and it comes from the same audio with a different cut, which costs nothing. The clip-length shares therefore become the preparation defaults (0.25 and 0.25) and **V11 is the recommended adapter for this voice** with everything its run saved: reference `qwen_fine_tuning_0199`, speaking rate 0.978, decoder adapter at strength 1.0, default decoding. V10 stays as the fallback. For long-form narration where V11's three percent slower pace on long sentences matters, a rate of 1.0 instead of the saved 0.978 removes it.

## What changed in the code

- `indextts/training/speech_metrics.py`: `deployment_score` and the new `select_recommendation` (policy keys `score_wer_weight`, `score_pause_weight`, `score_min_delta`; defaults 4, 0.05, 0.002).
- `indextts/training/checkpoint_average.py` (new), `analysis.checkpoint_descriptor` (kind `averaged`), `speech_eval.shortlist_checkpoints` (the averaged file is always a candidate), `trainer._write_averaged_checkpoint`, `TrainConfig.average_last_checkpoints` (default 0), the training tab control, presets.
- `indextts/training/decoder_adapter.py`: `code_source`, `gpt_checkpoint`, `gpt_code_variants`, sampling settings, `_precompute_gpt_codes`, per-epoch variants; `TrainConfig.decoder_adapter_code_source` (default `real`), trainer wiring, `tools/train_decoder_adapter.py --code-source/--gpt-checkpoint`, the training tab control.
- `indextts/training/audio_boundaries.py`: `medium_clip_fraction` (10-second aim), the clear-pause rule for short and medium aims (`SHORT_CLIP_MIN_PAUSE_MS`, 80 ms after padding), `length_aim` tags; `DatasetPrepConfig.medium_clip_fraction`; the manifest and `dataset_info.json` record the aims; dataset tab slider, presets.
- `tools/compare_checkpoint_benchmark.py` (new): render a run's frozen benchmark with any checkpoint or decoder and compare it pairwise with the run's own measurement.
- Tests: `tests/test_pipeline_improvements.py` (new), additions to `tests/test_acoustic_sentence_repacking.py`; the UI build test covers the new controls.

## An incident during this work, and what it cost

The learning-rate sweep was trained from a frozen copy of the code (a git worktree) so that the edits above could not disturb the running jobs, and that worktree held two junctions, `models` and `datasets`, pointing at the main checkout's folders. Removing the worktree afterwards with `git worktree remove --force` followed both junctions and emptied the real `models/` and `datasets/` folders (02:35 on 9 September). Nothing else was touched: every adapter under `loras/`, every grid and listening bundle under `outputs/`, and the raw recordings were intact. `models/` was restored within minutes from a complete copy in another install on the same drive (`G:\tt2\Premium_IndexTTS2_SECourses\models`), all eleven checkpoint files verified against the recorded SHA-256 hashes. The datasets were regenerated from the raw recordings with the same deterministic preparation and the same audit (the rebuilt `furkan_v10_curated_20s` has the same 2,576 prepared clips, 609.670456 minutes and 96,553 words, the same 150 audit rejections by the same reasons, the same 2,034-clip training split, and the identity hash over every clip's id, path, text, split, duration and token counts equals the one V10's frozen speech plan recorded, `89f199df…`). Only the datasets V10 and V11 depend on were rebuilt (`furkan_v10_fresh_20s`, `furkan_v10_curated_20s` with its test split, `furkan_v11_fresh_20s`, `furkan_v11_curated_20s` with its test split); the older `furkan_v4` to `furkan_v9` datasets, the `exp_*` experiments and the tutorial datasets were not, and can be recreated with the commands in their reports if ever needed. The v11 run was in its decoder gate when the recordings disappeared, so the gate, the decoding sweep and the final test were rerun after the restore with the trainer's own commands; the v11 numbers above come from those reruns. The cleanup lesson is recorded here so it is not repeated: never place junctions to shared folders inside a worktree, or remove the worktree by hand after unlinking them.

## Limits

One speaker, one set of recordings, and one training seed per configuration: the sweep runs and V11 are single runs, and the earlier reports put run-to-run variation at a few tenths of a point of word error and about 0.01 of similarity, which is the size of several differences above. The learning-rate and averaging conclusions rest on V10's frozen benchmark (12 validation sentences, three seeds) and its own recognizer; the decoder-codes and V11 conclusions add the deployed grids and the automated listener, whose day-to-day variation the adapter comparison report measured. The per-length set has only nine short sentences (all the validation split holds), and the recognizer's error on the real medium sentences (8 percent) means their word error mostly measures acronyms. V11's rate calibration, taken on a benchmark that mixes lengths, makes it slightly slow on long sentences and V10's makes it slow on short ones; a rate calibrated per length would separate that from the training change. The datasets used by V10 and V11 were rebuilt after the incident above and verified identical; the older datasets were not rebuilt. No human listening was performed.
