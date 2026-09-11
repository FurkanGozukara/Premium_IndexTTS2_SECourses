# Why the cloned voice sounds less natural than the recordings, and what to do about it (10 September 2026)

Scope: the V11 DoRA (rank 128, alpha 129, 16,500 updates, decoder adapter, saved speaking rate 0.978) trained on 483 minutes of the narrator's tutorial recordings. Identity is excellent; delivery is judged less natural than the recordings. This report collects the CPU-side evidence gathered today, the literature and upstream findings, the ranked causes, the changes shipped in v6.13, and the GPU experiment plan that waits for the card. Nothing in the app was tied to this voice: every new feature derives its numbers from whichever dataset an adapter was trained on.

## 1. Evidence

### 1.1 Objective prosody of the generated clips against the real recordings (CPU, librosa pyin)

Matched sentences of the independent final test (12 sentences, 3 seeds, Base and V11) and the 33-sentence length-mix set (V11 as deployed, 99 clips), each generated clip measured against the person's recording of the same text:

| Metric | Base (36 clips) | V11 final test (36) | V11 deployed, length mix (99) |
|---|---|---|---|
| words per second | 2.55 vs 2.76 | 2.68 vs 2.76 | 2.65 vs 2.75 |
| articulation rate, pauses excluded | 2.96 vs 3.01 | 3.03 vs 3.01 | 3.00 vs 3.02 |
| voiced fraction | 0.72 vs 0.74 | 0.71 vs 0.74 | 0.72 vs 0.76 |
| pitch std (semitones) | 2.91 vs 2.95 | 3.11 vs 2.95 | 3.12 vs 2.81 |
| pitch 5 to 95 range (semitones) | 9.28 vs 9.47 | 9.89 vs 9.47 | 9.97 vs 8.87 |
| pitch movement per 10 ms frame (semitones) | 0.28 vs 0.31 | 0.34 vs 0.31 | 0.33 vs 0.30 |
| loudness std (dB) | 8.96 vs 8.86 | 8.69 vs 8.86 | 8.62 vs 8.70 |
| loudness 5 to 95 range (dB) | 27.3 vs 28.2 | 27.3 vs 28.2 | 27.2 vs 27.2 |
| pauses per clip | 5.72 vs 3.92 | 5.25 vs 3.92 | 4.62 vs 4.51 |
| pause time fraction | 0.14 vs 0.08 | 0.12 vs 0.08 | 0.12 vs 0.09 |
| longest pause (ms) | 466 vs 323 | 373 vs 323 | 381 vs 292 |

Three facts follow. The voice is not flat in the statistical sense: its pitch range and variance are equal to or above the narrator's. It pauses more often and longer (12 percent of the time against 8) while articulating at exactly the narrator's rate, so the slower pace that the speaking-rate calibration corrects is entirely pause time, not slower speech. And its loudness contrast is slightly smaller. The training clips themselves pause 12.6 percent of the time (1.8 pauses per 10 words, 3.3 sentences per clip, longest pause 390 ms on average): the model reproduces the dataset's average pausing on every sentence, including single sentences the narrator speaks with far fewer pauses.

### 1.2 Blind expert listening (Gemini 3.1 Pro through the Antigravity CLI, native audio, real versus generated pairs)

Six matched pairs, two fresh conversations, the model told which clip was real and asked to describe the differences. Both conversations reached the same diagnosis, ranked by impact:

1. Rhythm: syllables spaced too evenly, no speeding up through dependent clauses or lengthening for emphasis ("metronomic").
2. Intonation: missing pitch accents on the words the narrator stresses ("100", "overwrite", "out of VRAM error"), missing question and transition rises ("how to use ComfyUI?", "Moreover"), repetitive phrase-final falls.
3. Emphasis and loudness: stress spread evenly, constant volume, no fade at phrase ends.
4. Texture: no breaths, lip noise or micro-hesitations; "dead" digital silences between phrases; a slight buzz on sustained syllables.

Both listeners judged the pattern as an averaged, flattened delivery ("the model falls back to a safe mean prosody") rather than wrong-but-varied prosody, with a smaller audio-quality component (texture and buzz). Reconciled with 1.1: the pitch moves as much as the narrator's overall but not where meaning demands it, and the extra movement per frame is wobble rather than accents.

### 1.3 Training-time conditioning is constant

Since v6.9 (`reference_typical=True`) the training pool for the speaker prompt is led by one clip, the one nearest the speaker's median pitch and pace, and `_other_reference` takes the first non-empty group, so nearly every training example is conditioned on the same clip. With `emo_ref_mode=follow_speaker` the emotion vector is that clip's vector too, so it is a constant during fine-tuning. A constant input carries no information: the adapter learns to predict the codes from text alone and the base model's ability to take delivery from the prompt is unused and gradually overwritten. The IndexTTS2 paper pairs each target with a *different, randomly drawn* utterance of the same speaker, which keeps the vector varying; the community trainers (JarodMica training_v2, instavar) do the same. This is the leading candidate cause, and it is cheap to test (section 4, run B).

### 1.4 Decoding is mode-seeking

The deployed settings are `num_beams=3` with sampling (Hugging Face beam-search multinomial sampling returns the highest cumulative log-probability hypothesis), `repetition_penalty=10.0`, temperature 0.8, top-p 0.8, top-k 30. Upstream ships the same values. The literature is consistent that likelihood-maximising decoding compresses prosody: TRAD-BS (arXiv 2408.16373) concedes sampled outputs "are more expressive due to sampling-produced prosody variations"; CER- or NLL-driven optimisation "collapses prosodic variation into monotone speech" (arXiv 2509.18531, 2509.19928); no production AR TTS (CosyVoice 2/3, Fish, Higgs, CSM, Llasa, Spark, Chatterbox) uses beams, and their repetition penalties are 1.1 to 1.3 with windowed repetition-aware sampling. The app's decoding sweep only compared 3 against 5 beams and scored word error and identity, never naturalness, so pure sampling was never measured. Cheap to test (section 4, part 1).

### 1.5 Speaking rate is a mel stretch

`generation.speaking_rate` becomes `latent_multiplier / rate`, which changes the semantic-to-mel target length: the flow-matching decoder is asked for more or fewer mel frames per code. It stretches phones and pauses alike and does not touch the GPT's timing. Upstream's `duration_factor` is the same mechanism; the paper's token-count duration control is not in the public weights. Since the pace gap is pause time (1.1), stretching everything is the wrong correction; the fix is fewer or shorter pauses.

## 2. Upstream, papers and other models (what is usable)

- **IndexTTS-2.5** (weights 10 August 2026): inference only, no training code; defaults unchanged; new `<word|CMU PHONES>` pronunciation control with `.` syllable separators and stress digits, and a case-insensitive `glossary.yaml` for respellings. Issue #775: 60 to 80 text tokens per segment fixed collapse for English; issues #759 and #801 report 2.5 sounding less emotional than 2.0 in blind tests.
- **Training recipe in the paper** (arXiv 2506.21619): prompts and targets are different utterances of the same speaker; the emotion perceiver was trained only in stage two with a gradient-reversal speaker classifier on 135 hours of emotional data; stage three froze every conditioner. Issue #369 reports that a replication of stage three (conditioners frozen, 500 to 600 hours) lost emotion transfer after one or two epochs. The app already freezes the conditioners; what differs is the constant prompt.
- **Fine-tuning literature**: quality decouples from loss (MOS falls while validation loss improves, arXiv 2603.10904; VoxCPM2 LoRA MOS peaks at rank 64 while loss is lowest at 128); LoRA forgets less than weight decay or dropout and lower ranks forget less (arXiv 2405.09673); EMA of the adapter raised UTMOS 3.06 to 3.18 on a TTS fine-tune (arXiv 2605.23859); DPO with 250 to 1,000 self-generated pairs ranked by UTMOS, word error and identity raised UTMOS 3.80 to 4.23 (arXiv 2409.12403) but WER-only rewards make speech "slower and more clearly pronounced" and collapse pitch variability (Seed-TTS, arXiv 2509.18531); human or prosody-aware preferences restore it.
- **Metrics**: MOS predictors barely track prosodic naturalness (UTMOS 0.21, UTMOSv2 0.24 Spearman with human naturalness across systems; near chance on stress placement), so this work measures pitch, loudness, rate and pause statistics against the narrator's own recordings and keeps a blind listener in the loop.
- **Other open models** (September 2026): Fish S2 Pro (arena Elo 1131, non-commercial license, ARPAbet phoneme tags, GRPO with a quality reward), Qwen3-TTS 1.7B (Apache, official single-speaker SFT), MOSS-TTS Local (Apache, IPA input), VoxCPM2 and dots.tts (Apache, 48 kHz, official fine-tuning). None is a drop-in for the trained DoRA; the transferable lessons are the phoneme control this app now uses, pure sampling with windowed repetition control, and preference optimisation with a naturalness-aware reward.

## 3. Ranked causes

| Rank | Cause | Evidence | Fix and cost |
|---|---|---|---|
| 1 | Constant speaker and emotion prompt during fine-tuning (pinned typical reference) | 1.3, paper recipe, listening verdict "mean prosody" | run B: `reference_typical=false`; run C: `emo_ref_mode=self`; three epochs each, about 40 minutes |
| 2 | Beam sampling and repetition penalty 10 at inference | 1.4, literature, never measured for naturalness | part 1 A/B: 1 beam, temperature 0.9 to 1.0, penalty 1.5 to 3; minutes |
| 3 | Pause density copied from multi-sentence training clips and triggered by commas | 1.1: 12 percent pause time against 8; articulation rate equal | A/B: commas removed, silence cap; training-side: the 16-second maximum and the 25 percent short and medium shares already shift the clip mix |
| 4 | Decoder adapter at strength 1.0 (identity gain, possible buzz and compressed dynamics) | 1.2 texture notes; gate measured identity and word error only | A/B: decoder off and at 0.6 |
| 5 | Reference clip chosen for typicality (median pitch and pace) | by construction the least expressive clean clip | A/B: expressive reference and expressive emotion prompt (candidates listed in section 5) |
| 6 | Uniform mel stretch for pace | 1.5 | keep rate at 1.0 when pauses are fixed; panel now shows what each rate does |
| 7 | Later checkpoints trade naturalness for loss | literature | A/B across epoch checkpoints, no training needed |

## 4. GPU plan (in order, shortest first; nothing runs until the card is free)

**Part 1, inference only, about one hour.** `tools/naturalness_ab.py --run-dir loras/<voice> --expressive-reference <clip>` renders the final-test sentences with the deployed settings and eleven variants (pure sampling, warmer sampling, repetition penalty 3 and 1.5, decoder off and 0.6, commas removed, silence cap, guidance 0.5 with 40 steps, rate 1.0, expressive reference, expressive emotion prompt), measures word error, identity, style, pause time and the prosody statistics against the recordings, and writes blind listening sets. Then `--checkpoints epoch_002 epoch_004 epoch_006 final --variants deployed` for the epoch comparison. Adopt as a default only what raises liveliness and the blind ranking without more than 0.02 word error or 0.03 identity loss.

**Part 2, three-epoch training runs, about 40 minutes each** (`docs/naturalness_experiments/`): A control, B random prompts, C emotion from the target clip, D random prompts with batch 4. Each ends with its own speech comparison; the same A/B tool compares them on the development sentences. If B or C wins, retrain the full ten-epoch voice with that setting; the change is one flag in the training tab.

**Part 3, if parts 1 and 2 leave a gap:** EMA of the adapter during training; a short DPO pass on self-generated pairs ranked by the prosody statistics plus word error and identity (250 to 1,000 pairs); a windowed repetition-aware sampler instead of the global penalty.

## 5. Expressive reference candidates for this voice (CPU ranking, for part 1)

Clean 9 to 16 second training clips (speaker similarity at least 0.86, transcript error at most 5 percent) ranked by pitch and loudness variability; the top of the list, by combined z-score: `wan22_training_tutorial_0100` (12.6 s, pitch std 5.3 st, range 18.4 st), `wan22_training_tutorial_0118`, `cuda13_tutorial_0115`, `cuda13_tutorial_0055`, `qwen_fine_tuning_0278`, `qwen_fine_tuning_0253`. The dataset mean is about 4 st pitch std; the deployed reference `qwen_fine_tuning_0199` was chosen for being median. These are inputs to the A/B, not settings.

## 6. Pronunciation of words the voice never saw

The engine already reads `<word|PHONES>` (ARPAbet with stress digits and dots between syllables) through `<|SPECIAL_TOKEN_1|>`, trained on CMU-dictionary words in IndexTTS-2.5. v6.13 builds on that: the **Pronunciation check & dictionary** finds the words in a text that the selected voice never spoke in training (from the saved training vocabulary) and the base model has no CMU reading for, proposes ARPAbet (CMU lookup, CamelCase and acronym splitting, dictionary compounds such as run+pod, letter-to-sound rules as the last resort, digits left to the number normalizer), and inserts the annotations before synthesis. Words the voice learned from its recordings are never rewritten (scope `unseen`), because the fine-tuned text reading is better evidence than a phoneme guess. What still needs the GPU: confirming that phoneme annotations render at least as well as the text spelling for a set of technical words, with and without the syllable dots, and that annotations inside a fine-tuned voice do not shift identity.

## 7. Shipped today (v6.13, all voice-independent)

Adapter panel with calibrated and current speaking rates and live seconds; per-adapter dataset profile (`analysis/dataset_profile.json`, saved by training or measured on first selection); words-per-line and per-sentence rules from the training percentiles; automatic **Max tokens per segment** from the profile (inverse of the engine's token budget); original calibration kept through manual speaking-rate edits; pronunciation check and dictionary with native annotations; dataset maximum back to 16 seconds; slider bounds guard that clamps typed values instead of printing tracebacks; the prosody metrics module and the A/B harness with tests. Full CPU test suite passes.

## 9. GPU results, part 1 (RTX 3090, 10 September evening)

Twelve final-test sentences, two seeds, thirteen variants of the deployed V11, measured against the narrator's recordings (`outputs/naturalness_ab/part1_gpu1/report.md`). Liveliness is the mean ratio of the generated pitch and loudness variability to the recording's; the pitch columns are paired per-clip wins/losses against the deployed settings.

| Variant | Word error | Identity vs real | Style vs real | Pause time vs real | Liveliness | Pitch std wins | Pitch range wins | Pitch movement wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deployed (3 beams, penalty 10, decoder 1.0) | 1.19% | 0.876 | 0.812 | 1.65 | 1.088 | – | – | – |
| sampling (1 beam) | 1.32% | 0.884 | 0.812 | 1.77 | 1.085 | 12/12 | 15/9 | 9/15 |
| sampling, temperature 1.0, top-p 0.95, top-k 50 | 1.46% | 0.875 | 0.805 | 1.93 | 1.112 | 16/8 | 17/7 | 10/14 |
| 3 beams, penalty 3 | 1.72% | 0.883 | 0.822 | 1.76 | 1.077 | 8/13 | 11/10 | 8/13 |
| 1 beam, penalty 1.5 | identical to sampling (same clips) | | | | | | | |
| no decoder adapter | 0.93% | 0.843 | 0.801 | 1.64 | 0.972 | 0/24 | 0/23 | 0/24 |
| decoder adapter 0.6 | 0.93% | 0.867 | 0.820 | 1.62 | 1.057 | 3/20 | 6/18 | 4/20 |
| commas removed from the text | 1.32% | 0.879 | 0.825 | 1.59 | 1.073 | 6/12 | 9/9 | 7/11 |
| silence-token cap 10 | identical to deployed (the control is inert on the 2.5 codec) | | | | | | | |
| diffusion guidance 0.5, 40 steps | 0.93% | 0.880 | 0.809 | 1.68 | 1.089 | 9/14 | 12/11 | 6/17 |
| expressive clip as speaker and emotion prompt | 0.53% | 0.869 | 0.802 | 1.87 | 1.190 | 22/1 | 22/2 | 20/4 |
| deployed reference, expressive clip as emotion prompt (alpha 0.65) | 0.79% | 0.877 | 0.802 | 1.70 | 1.123 | 16/8 | 15/9 | 11/13 |
| same, with 1 beam | 1.19% | 0.879 | 0.815 | 1.71 | 1.129 | 19/5 | 17/7 | 15/9 |

What the numbers say:

- **Decoding is not the lever.** Beams, temperature, penalty and diffusion guidance move liveliness by at most 0.02 and leave the pause excess at 1.6 to 1.9 times the narrator's. Warmer sampling adds pitch variation but slows the pace (2.62 words/s) and adds pauses.
- **The repetition penalty is a hard ban above about 1.3.** Penalty 1.5 and 10 rendered byte-identical clips with one beam: under top-k 30 / top-p 0.8 any previously used code is already out of the sampled set at 1.5. With beams the magnitude changes the ranking (penalty 3 costs 0.5 points of word error). Because the 2.5 codec never repeats a code consecutively (71,710 codes of real speech, longest identical run under five), the penalty punishes the reuse of *any* earlier code in the utterance, not stuttering; the follow-up run tests penalty 1.0.
- **`Max consecutive silence tokens` is inert on the 2.5 codec** (it trims code 52, which appears once in 71,710 real codes). The panel text now says so; a working replacement is the audio-level **Maximum pause (ms)** control added today.
- **The decoder adapter adds pitch variance in every clip** (0/24 without it) while raising identity from 0.843 to 0.876 and costing 0.26 points of word error. Given the listeners' "buzz on sustained syllables", that added variance is more likely jitter than accents; strength 0.6 keeps most of the identity (0.867) with a third of the extra variance and the lowest word error. The blind ranking decides.
- **The emotion pathway is alive and is the strongest inference lever.** A lively clip as the emotion prompt (identity reference unchanged) raises pitch variability in 16 to 19 of 24 clips, keeps identity at 0.877 to 0.879, and lowers word error; as the speaker prompt too it raises liveliness to 1.19 at a small identity cost. Whether that movement lands as accents is what the blind listening measures.
- **Pauses remain a training-side problem**: removing commas trims the pause ratio from 1.65 to 1.59 only. The three-epoch runs (A control, B random prompts, C emotion from target) are queued on the same card, and the pause cap covers users in the meantime.

## 10. Blind listening of part 1 (Gemini 3.1 Pro through the Antigravity CLI, 24 sets, 7 clips each, real recording as reference)

| Variant | Mean rank (1 best of 7) | Sets ranked first | Naturalness (1-5) | Similarity to the real speaker | Pronunciation | Above / below the deployed clip of the same set |
|---|---:|---:|---:|---:|---:|---|
| expressive clip as speaker and emotion prompt | 3.62 | 7 | 3.92 | 4.42 | 4.96 | 17 / 7 |
| deployed reference, expressive clip as emotion prompt | 3.67 | 4 | 3.96 | 4.42 | 4.92 | 18 / 6 |
| commas removed | 3.67 | 3 | 3.79 | 4.42 | 5.00 | 18 / 6 |
| 1 beam | 3.88 | 5 | 3.75 | 4.38 | 4.79 | 16 / 8 |
| 1 beam, temperature 1.0, top-p 0.95, top-k 50 | 3.96 | 3 | 3.71 | 4.21 | 4.83 | 15 / 9 |
| no decoder adapter | 4.12 | 2 | 3.75 | 4.33 | 4.92 | 14 / 10 |
| deployed (3 beams, penalty 10, decoder 1.0, speaker prompt as emotion prompt) | 5.08 | 0 | 3.29 | 3.96 | 4.83 | – |

The deployed configuration ranks last in mean rank and never first; its notes read "strained rhythm", "hesitant", "robotic cadence". The listener also rates its voice as *less* similar (3.96) even though embedding similarity is the highest of the set (0.876), so the objective identity metric does not capture what the listener hears as the person. Every change beats it, and the two that also raise the naturalness score by more than half a point are the expressive emotion prompt (3.96) and the expressive speaker prompt (3.92). Combined with the objective table this settles the inference-side plan:

1. **Emotion prompt = the adapter's liveliest clean training clip, speaker prompt unchanged** (shipped in v6.13 as the default: training saves the clip, *Pick expressive clip* finds it for older adapters, the checkbox feeds it to the emotion pathway). Identity unchanged (0.877 embedding, 4.42 by the listener), word error down, naturalness up in 18 of 24 sets.
2. **Fewer pauses**: the *Maximum pause (ms)* control (shipped) and, for the text, removing commas (18 of 24 sets above deployed, pronunciation 5.0). The three-epoch training runs test whether varied prompts remove the excess at the source.
3. **One beam instead of three** is preferred 16 to 8 with a small word-error cost (1.19 to 1.32 percent); the default stays at the tier value pending the second round, which pits the emotion prompt with and without sampling against each other on the same clips.
4. The decoder adapter's extra pitch variance is not what the listener misses: without it the clips still rank above deployed (14 to 10) but score lower on naturalness than the emotion-prompt variants and lose identity; strength 0.6 is in round two.

### 10.1 Second blind round on the same clips (5 clips per set, 24 sets)

| Variant | Mean rank (1 best of 5) | Sets ranked first | Naturalness | Similarity | Above / below deployed in the same set |
|---|---:|---:|---:|---:|---|
| 1 beam | 2.71 | 7 | 3.83 | 4.38 | 13 / 11 |
| deployed reference, expressive clip as emotion prompt | 2.96 | 4 | 3.54 | 4.29 | 12 / 12 |
| deployed | 2.96 | 3 | 3.88 | 4.38 | – |
| decoder adapter 0.6 | 3.12 | 4 | 3.88 | 4.42 | 12 / 12 |
| expressive emotion prompt with 1 beam | 3.25 | 6 | 3.62 | 4.04 | 10 / 14 |

The same deployed clips that scored 3.29 in round one score 3.95 here, and the expressive emotion prompt that won 18 of 24 pairings in round one is even in round two. The model listener's absolute scores depend on the company a clip keeps in its set, and its pairwise preferences carry a variance of several sets in 24. Pooled over both rounds the emotion prompt is ahead of deployed 30 to 18 and one beam 29 to 19: a modest preference, not the decisive one round one suggested. Two forced-choice rounds (deployed against the emotion prompt, and deployed against repetition penalty 1.0, 24 pairs each) were run as tie-breakers; see 10.2. The objective side is unchanged (more pitch movement, lower word error, identity kept), so the expressive prompt stays the default with softened wording in the product texts, and the user's own ears decide; the blind sets under `outputs/naturalness_ab/part1_gpu1/listening*/blind/` with their `keys.json` are ready for a human pass.

### 10.2 Repetition penalty 1.0 (follow-up render, same sentences and seeds)

| Variant | Word error | Worst clip | Identity vs real | Style | Pause time vs real | Liveliness | Pitch std wins vs deployed |
|---|---:|---:|---:|---:|---:|---:|---:|
| deployed (3 beams, penalty 10) | 1.19% | – | 0.876 | 0.812 | 1.65 | 1.088 | – |
| 3 beams, penalty 1.0 | 0.79% | 5.3% | 0.880 | 0.812 | 1.90 | 1.063 | 9/15 |
| 1 beam, penalty 1.0 | 0.93% | 5.7% | 0.876 | 0.812 | 2.06 | 1.090 | 14/10 |
| 1 beam, temperature 1.0, top-p 0.95, top-k 50, penalty 1.0 | 1.19% | 6.2% | 0.873 | 0.803 | 2.03 | 1.082 | 15/9 |

Removing the penalty lets the model reuse codes it already emitted in the utterance: with three beams that lowers word error by a third and nudges identity up, at the price of longer pauses (which the pause cap can trim). Nothing looped or truncated (no failures in 72 clips). Whether it sounds better is left to the forced-choice round.

### 10.3 Forced choice, deployed against the expressive emotion prompt (24 pairs, real recording as reference)

Preferred: expressive emotion prompt 14, deployed 10. Scores: naturalness 3.92 against 3.67, similarity to the real speaker 4.71 against 4.50, pronunciation 4.92 against 4.88; pace judged the same as the reference in 12 of 24 for the emotion prompt (slower in 8) and 10 of 24 for deployed. Pooled over the three rounds the emotion prompt is ahead 44 to 28 (61 percent), which is the consistent modest preference the objective metrics predicted (more pitch movement, lower word error, identity kept). Decision: the expressive clip stays the default emotion prompt for trained voices, described in the product texts as a measured default to judge by ear.

### 10.4 Forced choice, deployed against repetition penalty 1.0 (24 pairs)

Preferred: deployed 14, penalty 1.0 10; naturalness 3.75 against 3.67, similarity 4.54 against 4.42. The lower word error and the small identity gain of penalty 1.0 do not translate into a preference, and its longer pauses (1.90 against 1.65 times the narrator's) are the likely reason. Decision: the repetition penalty default stays at 10 with three beams; the finding that any penalty above about 1.3 is a hard ban under top-k 30 / top-p 0.8 is recorded for the decoding sweep, which should test 1.0 and 1.2 rather than 3 and 5 when it next runs.

## 11. Pronunciation annotations, measured (GPU 1, 16 words, Base and V11, 2 seeds; recognizer and blind listener)

Each word in one carrier sentence three ways: plain spelling, `<word|PHONES>` with syllable dots (the documented format), the same phones without dots. The recognizer's "word heard" is a weak judge for rare words (it spells Qwen as Quen or Kwan whatever the audio does), so 64 three-clip sets were also graded blind for pronunciation correctness (1 to 5) by Gemini 3.1 Pro.

| Checkpoint / condition | Recognizer heard the word | Listener pronunciation score | Ranked best in the set |
|---|---:|---:|---:|
| Base, plain | 19/32 | 3.44 | 12 |
| Base, phones with dots | 18/32 | 3.56 | 10 |
| Base, phones without dots | 18/32 | 3.34 | 10 |
| V11, plain | 23/32 | 3.59 | 10 |
| V11, phones with dots | 21/32 | 3.59 | 11 |
| V11, phones without dots | 17/32 | 3.59 | 11 |

Per word, the annotation with dots rescued the words the base model mangles when the reading came from dictionary parts: SwarmUI 1.0 to 5.0, SageAttention 2.0 to 5.0, ComfyUI 2.5 to 4.5, Qwen 3.5 to 5.0, Nunchaku 3.0 to 4.0. It made words worse when the reading was a guess or wrong: Zorbulax (letter-to-sound rules) 3.0 to 2.5, and my seed readings for Krea, Hunyuan, RunPod, GGUF and Musubi (plain spelling already scored 4 to 5). On the trained voice the average is unchanged, which is what the `unseen` scope is for; its plain readings are its own (it says Qwen the way the narrator does, scored 1.0 by a listener expecting "kwen"). Dots beat no dots throughout (V11: 21 against 17 heard). Changes made from this: the seed dictionary drops GGUF, RunPod and Musubi and corrects Krea and Hunyuan; **Add suggestions and save** adds only dictionary-backed readings (medium or high confidence) and lists letter-rule guesses for hand editing.

### 10.5 Validation render of the saved expressive clip (first attempt confounded, repeated)

The first render of the saved clip (`Furkan_EN_DoRA_r128_v11_expressive_reference.wav`) used it as the *speaker* reference for every variant, because the A/B tool took the first `*_reference.wav` in the folder alphabetically and the new file sorts before `<name>_reference.wav`. The tell was a byte-identical pair (the alpha-1.0 emotion variant equals "deployed"). Read as an experiment on the expressive speaker reference it still says something: identity 0.877 (unchanged from the typical reference), word error 0.79 percent, liveliness 1.148, pause time 1.79 times the narrator's, and adding the same clip as emotion prompt at 0.65 on top changes little. The resolver in both tools now excludes the expressive file (the app itself resolves the reference by its exact recorded name and was never affected), and the render was repeated with the typical reference as speaker prompt. The repeated "deployed" clips are byte-identical to part one's, so the harness is reproducible, and the automatically chosen clip (`qwen_fine_tuning_0439`, pitch std 4.86 st against 4.13 for the pool) behaves like the hand-picked one:

| Variant (speaker prompt = typical reference) | Word error | Identity vs real | Style | Pause time vs real | Liveliness | Pitch std / range wins vs deployed |
|---|---:|---:|---:|---:|---:|---:|
| deployed | 1.19% | 0.876 | 0.812 | 1.65 | 1.088 | – |
| saved expressive clip as emotion prompt, weight 0.65 (the v6.13 default) | 0.79% | 0.876 | 0.803 | 1.72 | 1.113 | 14/10, 18/6 |
| same, commas removed | 1.06% | 0.883 | 0.813 | 1.64 | 1.122 | 15/8, 16/8 |
| same, weight 1.0 | 1.06% | 0.883 | 0.799 | 1.71 | 1.112 | 13/11, 15/8 |
| same, 1 beam | 1.32% | 0.876 | 0.818 | 2.01 | 1.087 | 15/9, 14/8 |

Weight 0.65 keeps the lowest word error; weight 1.0 gains a little identity for a little style; one beam on top adds pauses and errors and is not recommended with the emotion prompt.

### 10.6 Forced choice, deployed against the automatically chosen expressive clip (24 pairs)

Preferred: deployed 14, expressive emotion prompt with `qwen_fine_tuning_0439` 10; naturalness 3.92 against 3.75, similarity 4.58 against 4.54. With the hand-picked clip (`wan22_training_tutorial_0100`, 10.3) the same test went 14 to 10 the other way. Across all four rounds the emotion prompt stands at 54 to 42 against the plain prompt, a listener verdict that depends on the clip and sits inside the listener's own variance, while the objective measurements never moved against it (word error 0.79 against 1.19 percent, identity 0.876 both, pitch range wins 18 to 6). Decision unchanged: default on, described as an objective gain with a split listener verdict; the wording in the checkbox, changelog, README and help was updated to say so. Open question for later: whether ranking the candidates by pitch range alone (the hand-picked clip's strength) chooses a clip the listener prefers.

## 12. Three-epoch training runs (GPU 1, 11 September, about 70 minutes each including the speech comparison)

Same dataset, recipe and seed as V11, three epochs (6,540 updates), no decoder adapter, sweep or final test. Each run's own speech comparison (13 held-out development sentences, 2 seeds; identity and style against the real recordings; Base rendered with the run's reference):

| Run | Change | Best validation loss | Recommended checkpoint | Word error | Identity vs real | Style vs real | Pause time vs real | Saved speaking rate |
|---|---|---:|---|---:|---:|---:|---:|---:|
| A control | none (pinned typical reference, emotion follows the speaker prompt) | 4.959 | final, epoch 3 | 0.57% | 0.845 | 0.819 | 1.13 | 0.949 |
| B random prompts | `reference_typical=false` | 4.984 | epoch 2 | 0.86% | 0.831 | 0.836 | 1.03 | 0.946 |
| C emotion from target | `emo_ref_mode=self` | 5.219 | epoch 2 | 1.33% | 0.844 | 0.802 | 1.39 | 1.049 |
| Base (with A's reference) | – | – | – | 1.05% | 0.820 | 0.769 | 1.63 | – |

Reading: **C is rejected.** Training the adapter with the target clip's own emotion vector and then prompting with a different clip is the train-inference mismatch the paper warns about: higher loss, twice the word error of the control, more pauses, lower style similarity, a speaking rate that overshoots. **B is the interesting one.** With prompts drawn from any clean clip the emotion vector varies during training, and the run pauses within three percent of the narrator's total pause time on sentences it never saw (A: 13 percent over, V11 at ten epochs: 65 percent over) while style similarity is the highest of the three. Its identity number is measured with a different reference clip (the random-prompt rule also changes the recommended reference), so the cross-run render below repeats all three with one shared speaker reference before any conclusion about identity. Three epochs at this learning rate are not the full recipe, so a full-length B run is the next step if the render and the blind listening hold.

### 12.1 The three runs rendered with one shared speaker reference (12 development sentences, 2 seeds, deployed settings)

| Run | Word error | Identity vs real | Style vs real | Pause time vs real (per-clip mean) | Liveliness | Pitch std gen vs real (st) | Pitch std / range / movement wins vs A |
|---|---:|---:|---:|---:|---:|---|---|
| A control | 1.27% | 0.846 | 0.819 | 1.58 | 0.963 | 2.55 vs 2.69 | – |
| B random prompts | 0.51% | 0.834 | 0.774 | 1.57 | 1.033 | 2.86 vs 2.69 | 16/8, 19/5, 20/4 |
| C emotion from target | 0.51% | 0.840 | 0.791 | 1.88 | 0.948 | 2.42 vs 2.69 | 10/14, 8/16, 14/9 |

With the same speaker prompt for all three, B is the only run whose pitch moves at least as much as the narrator's (A and C are flatter than the recordings); it halves word error against the control and pauses no more, but it sits 0.012 lower on identity and 0.045 lower on style similarity to the real sentence. C confirms its rejection: flatter and pausier than the control. Adding the expressive emotion prompt on top of the shared reference helps A (word error 1.27 to 0.76 percent, identity 0.846 to 0.851, liveliness 0.963 to 1.003) and hurts B's word error (0.51 to 1.14 percent), so the two remedies are not additive on B. A blind three-way ranking of the deployed renders decides whether B's extra movement sounds like the narrator or like drift; a full ten-epoch B run is the candidate next training if it does.

### 12.2 Blind three-way ranking of the runs (24 sets, shared reference, real recording as reference)

| Run | Mean rank (1 best of 3) | Sets ranked first | Naturalness | Similarity | Head to head |
|---|---:|---:|---:|---:|---|
| A control | 1.83 | 9 | 3.67 | 4.42 | above B 16 to 8, even with C 12 to 12 |
| C emotion from target | 1.96 | 10 | 3.50 | 4.46 | above B 13 to 11 |
| B random prompts | 2.21 | 5 | 3.29 | 4.21 | – |

The listener hears B's extra pitch movement as "slightly unnatural emphasis" and "rushed", and rates its voice less similar, in line with the objective identity and style drop. So the leading training hypothesis of section 1.3 does not survive contact with the listener at three epochs: the pinned typical reference (the v6.9 rule) is not what flattens the delivery, and `reference_typical=True` with `emo_ref_mode=follow_speaker` stays the training default. C is rejected on every measure. What the runs did show is that the pause excess is smaller at three epochs than at ten on the runs' own benchmarks (1.13 against 1.65 times the narrator's), which points at training length rather than conditioning as the next training-side question: a run stopped by a naturalness measure rather than by validation loss. That is the candidate for the next full-length experiment, not B.

## 13. Where this leaves the naturalness question (11 September, 04:00)

Measured on this voice, ordered by evidence:

1. **Shipped and measured positive on the objective side:** the expressive training clip as emotion prompt (word error 1.19 to 0.79 percent, pitch range wins 18 to 6, identity unchanged; listener split 54 to 42 over four rounds), the dataset-derived line-length rules and token budget, and the pronunciation dictionary with dictionary-backed readings (SwarmUI, SageAttention, ComfyUI fixed for the base model; guesses excluded from automatic adding).
2. **Shipped as a remedy for the one defect every decoding and training variant left untouched:** the audio-level *Maximum pause* control (its listening check follows in 13.1).
3. **Not adopted:** pure sampling (mild listener preference 29 to 19, small word-error cost; available as a setting), repetition penalty 1.0 (better word error, listener prefers the default 14 to 10), decoder adapter 0.6 (identity cost, no listener gain), random-prompt training (livelier but judged less natural and less similar), emotion from the target clip (worse on everything).
4. **Corrected along the way:** `Max consecutive silence tokens` is inert on the 2.5 codec; the repetition penalty is a hard ban above about 1.3; the harness now excludes the expressive clip when it resolves the speaker reference.

Open next steps, in order of expected value: a full-length run stopped by a naturalness measure (pause ratio and liveliness against the recordings) instead of validation loss; a windowed repetition-aware sampler in place of the global penalty; and a human listening pass over the blind sets left under `outputs/naturalness_ab/*/listening*/blind`, since the model listener's verdicts moved by several sets between rounds on identical clips.

### 13.1 The pause cap, measured (deployed settings, 12 final-test sentences, 2 seeds)

The first render of this check exposed a bug: the cap read PCM16 samples as integers and compared them with dBFS thresholds, so real pauses (a low noise floor, never digital zero) were never detected and the capped clips were byte-identical to the originals. The detector now scales integer samples to full scale; a regression test renders a noisy pause and expects it to be cut. With the fix:

| Variant | Word error | Identity vs real | Style vs real | Pause time vs real (per-clip mean) | Words/s (real 2.76) | Pauses shortened | Audio removed per clip |
|---|---:|---:|---:|---:|---:|---:|---:|
| deployed | 1.19% | 0.876 | 0.812 | 1.65 | 2.74 | – | – |
| Maximum pause 300 ms | 1.19% | 0.876 | 0.814 | 1.51 | 2.77 | 111 of 223 (with the expressive variant) | about 145 ms |
| Maximum pause 200 ms | 1.19% | 0.877 | 0.817 | 1.15 | 2.84 | 73 of 114 | about 390 ms |
| expressive emotion prompt plus 300 ms | 1.06% | 0.877 | 0.806 | 1.47 | 2.83 | – | – |

Trimming the middle of long pauses touches nothing the recognizer or the speaker embedding can hear, and style similarity to the real sentence rises slightly as the timing gets closer. At 200 ms the total pause time lands within 15 percent of the narrator's and the pace slightly above his, so 200 to 300 ms is the useful range for a narrator who pauses briefly; the product text said 300 to 450 and is corrected to 200 to 400. The blind pairwise rounds (deployed against 300 ms, deployed against 200 ms) are in 13.2.

### 13.2 Pause cap, forced choice against the uncapped clips (24 pairs each)

| Cap | Preferred | Naturalness | Similarity |
|---|---:|---:|---:|
| 300 ms | cap 16, uncapped 8 | 3.75 vs 3.75 | 4.50 vs 4.46 |
| 200 ms | cap 8, uncapped 16 | 3.71 vs 3.71 | 4.50 vs 4.50 |

The 300 ms cap is the one inference-side change in this study that the listener prefers by a clear margin with no objective cost, and the 200 ms cap shows the limit: pause time within 15 percent of the narrator's is heard as rushed. The control ships with default 0 because a cap would damage deliberately paused speech (drama, poetry, dictation); the product texts recommend 300 ms for brisk narration. Combined with the expressive emotion prompt (13.1: word error 1.06 percent, pause time 1.47), this is the recommended inference setup for this voice until a training change earns its place.

## 8. Limits

One voice, one recording style, one run per configuration so far; the prosody statistics are corpus level and cannot place accents, which is why the blind listener stays in the loop. The listening judgments come from one model listener on six pairs. Part 1 and part 2 are designed to be small enough to repeat with three seeds before any default changes.

## 14. Second round (11 September, afternoon): sentence-aware splitting, speaker-derived pauses, repetition window, EMA weights

### 14.1 Shipped in v6.14 (all voice-independent)

- **Text splitting** with three modes. *Smart sentences* (new default) packs whole sentences into each segment by dynamic programming with a target of the trained voice's median clip in text tokens (85 percent of the budget for the base model), a hard budget, a 0.35 cost for ending inside a sentence, 0.12 per segment and 0.5 for a tail under 35 percent of the target; *Every sentence* renders one sentence per segment; *Token budget* is the former greedy splitter. Sentence ends: `. ! ?`, ellipsis, CJK stops, closing quotes, line breaks; abbreviations, initials, decimals and pronunciation annotations are not split.
- **Sentence pause (ms)**: the pause between two split sentences measured from the last loud 10 ms frame of one segment to the first of the next (edge quiet counts, excess is trimmed, the rest is inserted). **Maximum pause** moved next to it; explicit `[pause:…]` tags are reported by the engine and skipped by the cap.
- **Auto pauses from LoRA / DoRA dataset**: the dataset profile (version 2) measures every internal pause of the training clips and recommends the median sentence pause and the 90th percentile of sentence pauses (see 14.2).
- **Repetition window (codes)**: the Hugging Face penalty limited to the last N generated codes (see 14.4). Default 0.
- **Max consecutive silence tokens** hidden for the 2.5 codec; **EMA of the adapter weights** as a training option (see 14.5); the Text & Timing block redesigned into three rows and verified in Chrome (the panel fills Sentence pause 360, Maximum pause 450 and Max tokens 72 when the reference voice is selected, and the live preview reports "Smart sentences, aiming at 44 tokens per section").

### 14.2 The speaker's pauses, measured (2,180 training clips, 6.4 s on the CPU, cached)

| Statistic | Count | p25 | p50 | p75 | p90 | p95 |
|---|---:|---:|---:|---:|---:|---:|
| All internal pauses (ms) | 14,092 | 150 | 230 | 350 | 400 | 450 |
| Pauses at sentence boundaries (longest `sentences - 1` per clip) | 4,851 | 270 | 360 | 400 | 450 | 600 |
| Pauses inside sentences | 9,241 | 140 | 180 | 270 | 360 | 390 |
| Longest pause per clip | 2,170 | 350 | 380 | 420 | 590 | 630 |

12.5 percent of clip time is pause (mean; 7 to 18 percent for 80 percent of the clips). The recommendation rule gives **Sentence pause 360 ms** and **Maximum pause 450 ms**. The 300 ms cap preferred in 13.2 trims below the speaker's own median sentence pause; the 450 ms cap only touches pauses longer than nine in ten of his sentence pauses, so it is the "match the recordings" setting and 300 stays the "brisker than the recordings" setting. The paragraph A/B in 14.6 compares both.

### 14.3 The three splitters on a seven-sentence paragraph (real tokenizer, budget 50 tokens = the voice's automatic 72)

Token budget: 45 words ending at "same flat delivery," then 35 words ending at "livelier lines," then a 10-word tail. Every sentence: eight lines of 3 to 22 words ("e.g." and "Dr." were not treated as sentence ends). Smart sentences: three whole-sentence lines of 25, 36 and 29 words, all under the 41-word limit and within the voice's 21-to-47-word acceptable range.

### 14.4 Windowed repetition penalty (GPU 1, 12 final-test sentences, 2 seeds, deployed settings otherwise)

| Variant | Word error | Identity vs real | Style vs real | Pause time vs real | Liveliness | Pitch std (st) |
|---|---:|---:|---:|---:|---:|---:|
| deployed (penalty 10 over the whole segment) | 1.19% | 0.876 | 0.812 | 1.65 | 1.088 | 3.27 |
| window 8 | 1.19% | 0.876 | 0.808 | 1.78 | 1.085 | 3.29 |
| window 16 | 1.32% | 0.880 | 0.819 | 1.66 | 1.065 | 3.23 |
| window 32 | 1.06% | 0.880 | 0.815 | 1.75 | 1.089 | 3.30 |
| window 64 | 1.19% | 0.882 | 0.814 | 1.85 | 1.091 | 3.33 |
| window 16, penalty 3 | 0.93% | 0.880 | 0.817 | 1.71 | 1.072 | 3.24 |

No loop or stutter appeared in any arm, so the global ban was not what held the voice back; the windows move every measure by less than the seed-to-seed spread and add a little pause time. Blind forced choice against deployed (24 pairs each, real recording as reference): window 16 with penalty 3 **12 to 12** (naturalness 3.71 against 3.75), window 32 **12 to 12** (3.88 against 3.92). Verdict: neutral on this voice; the control ships with default 0 for voices that do loop, where a window keeps the penalty from starving a long segment.

### 14.5 EMA of the adapter weights (GPU 0, three epochs, decay 0.999, one run gives both arms)

The trainer's own speech comparison on the development sentences (13 prompts, 2 seeds; identity against the real recording of each sentence):

| Candidate | Word error | Worst clip | Identity vs real | Pause time vs real |
|---|---:|---:|---:|---:|
| Base | 0.6% | 5.0% | 0.825 | 1.60 |
| Raw final (= best, epoch 3) | 0.8% | 5.0% | 0.850 | 1.08 |
| Epoch 2 | 0.9% | 5.0% | 0.847 | 1.09 |
| EMA of the final update | 1.0% | 7.1% | 0.850 | 1.22 |

The raw final update was selected. The EMA file keeps the identity but adds pause time (1.22 against 1.08) and one worse clip; the averaged weights lag the trajectory by roughly a thousand updates, which at three epochs is half an epoch. The listening and prosody results follow in 14.5.1.

#### 14.5.1 EMA against the raw final update: listening and prosody

Blind forced choice on the 24 development clips of the speech comparison (real recording as reference): **raw final 13, EMA 11**; naturalness 3.62 against 3.54, similarity 4.50 against 4.42, pronunciation 4.67 against 4.83; the listener heard the EMA clips as faster in 14 of 24 pairs. The prosody A/B (GPU 0, both files rendered with the same reference, development sentences, 2 seeds):

| File | Word error | Identity vs real | Style vs real | Pause time vs real | Liveliness | Pitch std (st, real 2.69) |
|---|---:|---:|---:|---:|---:|---:|
| Raw final | 1.02% | 0.847 | 0.827 | 1.56 | 0.956 | 2.51 |
| EMA of the final update | 0.89% | 0.850 | 0.817 | 1.73 | 0.935 | 2.39 |

The EMA file reads the words slightly better and keeps the identity, but it is flatter (pitch std 2.39 against 2.51 semitones, liveliness 0.935 against 0.956) and pauses longer (1.73 against 1.56), which is the direction this study is trying to leave. **Not adopted as a default**; the option ships at 0 for users who want the smoother average, and the speech comparison judges it automatically whenever it is on.

### 14.6 Splitting and pauses on paragraphs (GPU 1, eight three-clip paragraphs of 66 to 115 words, 2 seeds, deployed settings otherwise)

The real reference of each paragraph is its three real recordings joined with the speaker's median sentence pause (360 ms). All arms use the voice's automatic token limit (72); the smart arms aim at the median clip (44 tokens).

| Arm | Word error | Identity vs real | Style vs real | Pause time vs real | Liveliness | Words/s (real 2.69) | Pauses per paragraph (real 15.1) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Token budget (former default) | 1.02% | 0.928 | 0.908 | 1.15 | 1.043 | 2.86 | 15.2 |
| Smart sentences | 0.82% | 0.930 | 0.909 | 1.20 | 1.051 | 2.82 | 15.1 |
| Smart sentences, sentence pause 360, maximum pause 450 (the new defaults for a trained voice) | 0.57% | 0.931 | 0.908 | 1.21 | 1.051 | 2.81 | 15.3 |
| Every sentence, sentence pause 360, maximum pause 450 | 0.95% | 0.928 | 0.904 | 1.76 | 1.093 | 2.44 | 21.7 |
| Smart sentences, sentence pause 300, maximum pause 300 | 0.89% | 0.931 | 0.910 | 1.04 | 1.054 | 2.86 | 15.3 |

Whole-sentence lines halve the word error against the comma-cut lines (0.57 against 1.02 percent with the pause defaults) at equal identity and style, and per clip they win the pitch-range comparison 10 to 5. One sentence per segment is the outlier: every join adds a full sentence pause on top of the model's own quiet edges, so the paragraph gains six pauses, 50 percent more pause time and a slower pace, while its pitch moves more (a fresh prompt for every sentence). The 300 ms cap brings pause time to 1.04 times the recordings. The blind pairwise rounds follow in 14.6.1.

The same paragraphs with the **base model** (no adapter, the voice's reference clip, token limit 60, GPU 1): Token budget 1.33 percent word error, Smart sentences 0.57 percent, Smart with the pause values 0.70 percent, identity 0.869 to 0.870 and liveliness 1.00 to 0.99 in all three; pause time rose from 1.49 to 1.63 times the recordings because whole-sentence lines let the base model pause at the sentence ends it now sees inside a segment. The default therefore holds for the base model as well: fewer word errors at unchanged identity.

#### 14.6.1 Blind pairwise rounds on the paragraphs (16 pairs each, joined real recording as reference)

| Pair | Preferred | Naturalness | Similarity |
|---|---:|---:|---:|
| Smart sentences + pauses 360/450 against Token budget | **10 to 6** | 3.69 vs 3.56 | 4.62 vs 4.44 |
| Smart sentences alone against Token budget | 7 to 8 (15 graded) | 3.60 vs 3.80 | 4.60 vs 4.73 |
| Smart + pauses 360/450 against Smart + 300/300 | **9 to 7** | 3.88 vs 3.69 | 4.69 vs 4.50 |

Whole-sentence lines alone are a wash for the listener (the word-error gain is not heard on paragraphs), the measured sentence pause at the joins is what carries the preference, and the speaker's own pause values beat the brisker 300 ms cap on paragraph-length text. The shipped defaults (Smart sentences, Auto pauses) are the arm the listener preferred.

#### 14.6.2 A third seed (GPU 0) and the pooled picture (24 clips per arm)

| Arm | Word error | Identity vs real | Style vs real | Pause time vs real | Liveliness | Words/s |
|---|---:|---:|---:|---:|---:|---:|
| Token budget | 0.85% | 0.929 | 0.907 | 1.15 | 1.049 | 2.87 |
| Smart sentences | 1.27% | 0.929 | 0.908 | 1.14 | 1.046 | 2.86 |
| Smart + pauses 360/450 | 1.06% | 0.929 | 0.907 | 1.17 | 1.045 | 2.85 |
| Every sentence + pauses 360/450 | 1.06% | 0.929 | 0.905 | 1.77 | 1.141 | 2.44 |
| Smart + 300/300 | 1.23% | 0.929 | 0.909 | 1.00 | 1.046 | 2.90 |

Word error swings by seed more than by arm (Token budget 1.52 / 0.51 / 0.51 percent over the three seeds, Smart 0.51 / 1.14 / 2.16 percent), so the halving seen at two seeds does not survive the third: pooled, the arms are equal within noise on every objective measure except the deliberate pause differences. The third seed shows the one cost of whole-sentence lines: when the voice picks a wrong reading ("models" heard as "modules"), it repeats it for the rest of the longer segment, while a comma-cut segment resets sooner; four of the seven errors in the worst paragraph are the same word. Identity and style are identical across arms. The decision therefore rests on the listener: Smart with the speaker's pauses was preferred over the comma-cut default 10 to 6 and over the 300 ms cap 9 to 7 (14.6.1), which is why it ships as the default; the remaining pairwise rounds (third seed, and the expressive prompt on both arms) follow in 14.6.3.

### 14.7 Epoch checkpoints of the reference voice (GPU 0, 12 final-test sentences, 2 seeds, deployed settings)

| Checkpoint | Word error | Identity vs real | Style vs real | Pause time vs real | Liveliness | Pitch std (st, real 2.95) |
|---|---:|---:|---:|---:|---:|---:|
| Epoch 2 | 1.72% | 0.880 | 0.811 | 1.68 | 1.071 | 3.18 |
| Epoch 4 | 1.59% | 0.877 | 0.804 | 1.80 | 1.050 | 3.18 |
| Epoch 6 | 0.40% | 0.880 | 0.814 | 1.80 | 1.058 | 3.15 |
| Final (deployed) | 1.85% | 0.873 | 0.811 | 1.81 | 1.065 | 3.19 |

Identity, style and pitch statistics are flat from epoch 2 on; pause time grows slightly with training (1.68 to 1.81) and the earliest checkpoint is the liveliest. Word error is lowest at epoch 6 and highest at the final update, so on this run the last update is not the best reading of the text. Blind pairs (final against epoch 6, final against epoch 2) are queued for the listener; if epoch 6 is also preferred by ear, the deployment should move to it, and the speech comparison's shortlist rule (lowest validation losses plus the latest update) deserves a naturalness term.

#### 14.6.3 With the expressive emotion prompt on (the app's default emotion source; GPU 1, 2 seeds)

| Arm | Word error | Identity vs real | Style vs real | Pause time vs real | Liveliness | Pitch std (st) |
|---|---:|---:|---:|---:|---:|---:|
| Token budget + expressive prompt | 1.08% | 0.929 | 0.900 | 1.17 | 1.072 | 3.12 |
| Smart + pauses 360/450 + expressive prompt (shipped defaults) | 0.89% | 0.928 | 0.899 | 1.19 | 1.074 | 3.16 |
| Smart + 300/300 + expressive prompt | 1.08% | 0.928 | 0.902 | 1.02 | 1.074 | 3.13 |

The expressive prompt lifts liveliness from about 1.05 to 1.07 on every splitter (pitch std 3.12 to 3.16 against 3.00), and the splitters again tie on identity and style; the shipped-default arm wins the per-clip pitch-std comparison 10 to 6 against the comma-cut arm. The blind pairs for these arms, the third-seed pairs and two more seeds of the decisive pair are queued behind a listener outage (the Antigravity service returned "stream interrupted" on every call from 15:05); the watcher grades them as soon as it answers.


#### 14.6.4 Five seeds pooled for the decisive pair (40 paragraph renders per arm)

| Arm | Clips | Word error | Identity vs real | Style vs real | Pause time vs real | Liveliness | Pitch std (st) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Token budget | 40 | 1.12% | 0.930 | 0.908 | 1.14 | 1.041 | 2.98 |
| Smart + pauses 360/450 | 40 | 1.07% | 0.930 | 0.908 | 1.18 | 1.053 | 3.02 |

| Arm | Clips | Word error | Identity vs real | Style vs real | Pause time vs real | Liveliness | Pitch std (st) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Token budget + expressive prompt | 40 | 1.04% | 0.929 | 0.900 | 1.13 | 1.090 | 3.13 |
| Smart + pauses 360/450 + expressive prompt | 40 | 1.22% | 0.926 | 0.901 | 1.23 | 1.100 | 3.16 |

Across five seeds the two splitters are indistinguishable on every objective measure with or without the expressive prompt; the differences of a single seed (14.6.2) average out, and the pause values add about 0.05 to the pause ratio because the join pause is enforced where the comma-cut splitter sometimes ran two sentences together. The choice between them is therefore a listening choice, and the listener's 10 to 6 for the shipped arm stands until the queued rounds add to it.

### 14.7.1 Epoch 6 against the final update, four seeds pooled (48 final-test clips per arm)

| Arm | Clips | Word error | Identity vs real | Style vs real | Pause time vs real | Liveliness | Pitch std (st) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Final (deployed) | 48 | 1.39% | 0.873 | 0.809 | 1.79 | 1.107 | 3.20 |
| Epoch 6 | 48 | 0.53% | 0.879 | 0.813 | 1.92 | 1.103 | 3.19 |

The word-error edge of epoch 6 holds on the second seed pair (0.66 against 0.93 percent), identity and style stay level or slightly higher, and the pause ratio is now mixed (2.03 against 1.77 on these seeds, 1.80 against 1.81 on the first pair). The development-sentence check and the blind pairs decide whether the deployment moves.


### 14.7.2 Epoch 6 against the final update on the development sentences (GPU 0, 12 multi-sentence clips, 2 seeds)

| Arm | Clips | Word error | Identity vs real | Style vs real | Pause time vs real | Liveliness | Pitch std (st) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Epoch 6 | 24 | 0.51% | 0.890 | 0.860 | 2.12 | 1.155 | 3.00 |
| Final (deployed) | 24 | 0.25% | 0.890 | 0.861 | 1.79 | 1.172 | 3.04 |

On the multi-sentence development clips the order reverses: the final update reads the text better (0.25 against 0.51 percent) and pauses less (1.79 against 2.12), with identity and style equal. The epoch 6 advantage is therefore text-set dependent rather than a property of the checkpoint, and the deployment stays on the final update; the queued blind pairs (final-test and development) will say whether either is heard as more natural. The general lesson stands: word error and pause time move between epochs while identity and pitch statistics do not, so a naturalness term in the checkpoint shortlist would be judging exactly these two quantities.
