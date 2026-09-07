# Voice decoder adaptation: measured passes on the V8 adapter (8 September 2026)

Generation has three stages. The GPT turns text into semantic tokens and decides what is said and when; a LoRA / DoRA on it is what every voice adapter so far has been. The semantic-to-mel decoder (the s2mel flow-matching DiT, 98M parameters) turns those tokens into a mel spectrogram and owns timbre and spectral detail; until this release it was frozen and knew a voice only through the reference clip of each generation. The vocoder renders audio and stays frozen.

v6.9 adds an automatic second phase to training: after checkpoint selection and the speech comparison, a DoRA is trained on the decoder's transformer blocks with the pretrained flow-matching objective, on the same cached dataset, with a clip of the speaker as the in-context prompt for every target clip and the prompt's CAMPPlus style vector as conditioning, exactly as inference conditions the decoder. In this build the decoder does not read the GPT's latents, so the decoder adapter is independent of the GPT checkpoint and is shared by every checkpoint of a training. It is saved as `<name>.s2mel.safetensors`, Voice Generation applies it automatically with that adapter, and a switch turns it off for comparison.

This report records two passes on the V8 dataset (`datasets/furkan_v8_curated_20s`: 2,031 training clips, 481 minutes; 257 held-out clips of the `qwen_2511_tutorial` recording, which no adapter trained on). The first pass failed in a way that the trainer's own measurements could not see, and the fix changed how every decoder adapter is trained and judged.

## Pass 1: one fixed prompt, and why re-rendering recordings misled

The first adapter (LoRA rank 32, alpha 32, learning rate 1e-4, 7,500 updates) lowered the held-out flow-matching loss from 0.5527 to 0.5294 and, when held-out recordings were re-rendered from their own semantic codes, raised their CAMPPlus similarity to the real recordings from 0.871 to 0.946 in 16 of 16 clips while cutting the mel distance by a quarter. Every measurement inside the trainer said it worked. Generated speech said the opposite.

| V8 best, 12 `qwen_2511_tutorial` sentences, one seed | Speaker similarity to the real recordings | to the reference clip | Median pitch | Adapter better |
|---|---:|---:|---:|---:|
| Calibrated speaking rate 1.094, without / with | 0.7958 / 0.7192 | 0.9118 / 0.8424 | 141 / 150 Hz | 0 of 12 |
| Speaking rate 1.0, without / with | 0.7870 / 0.7085 | 0.9097 / 0.8341 | 142 / 152 Hz | 0 of 12 |
| Guidance 0 (no classifier-free guidance), rate 1.094 | 0.8174 / 0.7922 | 0.8993 / 0.8572 | 140 / 150 Hz | 4 of 12 |
| In-domain held-out sentences, rate 1.0 | 0.8105 / 0.7356 | 0.9162 / 0.8399 | 140 / 151 Hz | 0 of 12 |

A pairwise listening test (Gemini 3.8 Flash, 36 pairs, three seeds, real recording as the reference) preferred V8 without the adapter in 32 of 36 pairs and found its timbre closer in 32 of 36; audio quality fell from 4.47 to 3.83 of 5. Every judgement described the adapted clip the same way: thinner, brighter, energy shifted to the high frequencies, sibilants harsher, pitch higher. This test was not fully blind: the clip paths named their grid folders, which carried the condition, and one judgement cited it. The measurements above carry the conclusion; the pass-2 tests below copy clips to neutral names first. Speaking rate, guidance, and sentence domain were each ruled out by the rows above.

The cause was the prompt. The dataset code that supplies "a different clip of the speaker" ranks the training clips and returns the single best reference (`qwen_fine_tuning_0357`, median pitch 119 Hz) for every target, in every epoch; it was written for the GPT trainer, which conditions on one deployed reference by design. The decoder adapter therefore saw one prompt for all 2,031 targets, whose average pitch is 142 Hz, and learned the cheapest solution: a prompt-independent offset of about +9 Hz and a brighter spectrum. Re-rendering recordings with that same prompt made the offset look like a cure, because the pretrained decoder renders those recordings 10 Hz too low from that prompt (133 Hz against 143 Hz real) and the offset lands on the target. Generated speech is different: the GPT's codes already carry the speaker's pitch, the pretrained decoder renders them at the right 142 Hz, and the adapter's offset then overshoots to 151 Hz and thins the voice.

| Teacher-forced re-rendering, 12 held-out clips, fixed prompt | Speaker similarity to real | Mel distance | Median pitch (real 142.8 Hz) | Log-mel energy low / mid / high (real -3.56 / -4.94 / -6.10) |
|---|---:|---:|---:|---|
| Pretrained decoder | 0.874 | 0.972 | 133.1 Hz | -2.94 / -4.44 / -5.75 |
| Pass 1 adapter (LoRA r32) | 0.946 | 0.724 | 141.1 Hz | -3.33 / -4.66 / -5.77 |
| DoRA r128, learning rate 2e-4, one epoch, same fixed prompt | 0.945 | 0.733 | 141.0 Hz | -3.32 / -4.67 / -5.82 |

Two general lessons went into the trainer. A decoder adapter must be trained with a randomly drawn prompt for every target and every epoch, so that only what holds relative to the prompt can be learned. And re-rendering recordings from their own codes cannot show what generated codes do to an adapter, so the decision to install one must come from the full pipeline.

## Learning rate for the rank-128 DoRA

Five one-epoch runs (2,031 updates each, the fixed prompt of pass 1, so only relative ranking matters), measured by held-out flow loss and by teacher-forced identity on 16 held-out clips:

| Learning rate | Held-out flow loss | Speaker similarity to real | Mel distance |
|---:|---:|---:|---:|
| 5e-5 | 0.5331 | 0.936 | 0.800 |
| 1e-4 | 0.5311 | 0.940 | 0.783 |
| 2e-4 | 0.5295 | 0.942 | 0.740 |
| 4e-4 | 0.5295 | 0.941 | 0.754 |
| 8e-4 | 0.5328 | 0.941 | 0.806 |

2e-4 is the best on every column and sits in the middle of a flat region, so it is the default; 8e-4 already loses.

## Pass 2: random prompts, identity-based selection, and a full-pipeline gate

The second pass trained the rank-128 DoRA (alpha 128, learning rate 2e-4) with a randomly drawn prompt for every target: one epoch, 2,031 updates, 8 minutes. On eight held-out clips re-rendered with random training clips as prompts, speaker similarity to the real recordings rose from 0.885 to 0.935 and the held-out flow loss from 0.553 to 0.531. With the old fixed prompt the adapter renders the held-out clips at 139.5 Hz (pretrained 133.1 Hz, real 142.8 Hz): it now follows the codes' pitch rather than adding a constant.

Through the full pipeline the picture reversed from pass 1. Same 12 `qwen_2511_tutorial` sentences, one seed, the shared reference, the calibrated speaking rate:

| V8 best with the pass-2 adapter | Speaker similarity to the real recordings | to the reference clip | Style similarity to real | Median pitch | Word error rate | Adapter better |
|---|---:|---:|---:|---:|---:|---:|
| Calibrated rate 1.094, without / with | 0.7958 / 0.8735 | 0.9118 / 0.8752 | 0.689 / 0.796 | 141 / 145 Hz | 3.54% / 4.87% | 12 of 12 (similarity) |
| In-domain held-out sentences, rate 1.0, without / with | 0.8105 / 0.8786 | 0.9162 / 0.8813 | 0.702 / 0.794 | 140 / 147 Hz | 5.91% / 5.91% | 12 of 12 (similarity) |

The gain to the real recordings (+0.078 and +0.068) is as large as pass 1's loss, style similarity rises by a tenth, and the pitch moves 4 to 7 Hz toward the recordings. Similarity to the reference clip falls, as it should: that clip is one of the speaker's lowest-pitched (115 Hz against a 142 Hz dataset average), and the adapter renders the speaker rather than that clip. The cost is intelligibility on the out-of-domain sentences: 1.3 points of word error rate at the calibrated rate, none on in-domain sentences; the full-pipeline gate allows 2 points.

A path-blind pairwise listening test (Gemini 3.8 Flash, the same 12 sentences, clips copied to neutral names) was a tie: 6 to 6 on overall preference and on closer timbre, voice similarity 4.50 against 4.50, audio quality 4.50 against 4.50, naturalness 4.58 without the adapter against 4.42 with it, and high-frequency buzz or sibilant harshness noted in 5 adapted clips against 2. A first run of the same test whose clip paths still named the grid folders had preferred the adapter 11 to 1, which is why the paths are now hidden. The measurements say the adapted voice is closer to the speaker; the listener hears no overall difference and slightly more high-frequency texture.

Two knobs were then measured on the same sentences against the same baseline. Guidance mode: rendering the unconditional branch of classifier-free guidance through the adapter as well (instead of the pretrained decoder) gave +0.067 similarity and the same word-error cost, slightly below the default's +0.078, so the default stays. Strength: the new **Voice decoder adapter strength** control at 0.6 kept +0.043 of similarity (12 of 12) and +0.054 of style while the word-error rate moved only 0.2 points (3.54 to 3.76 percent) and similarity to the reference clip stayed at 0.902.

| Pass-2 adapter, calibrated rate | Speaker similarity to real | Style similarity to real | Word error rate | Adapter better (similarity) |
|---|---:|---:|---:|---:|
| Without adapter | 0.7958 | 0.689 | 3.54% | |
| Strength 1.0, default guidance | 0.8735 | 0.796 | 4.87% | 12 of 12 |
| Strength 1.0, adapted-branch guidance | 0.8626 | 0.761 | 4.87% | 12 of 12 |
| Strength 0.6, default guidance | 0.8391 | 0.743 | 3.76% | 12 of 12 |

At strength 0.6 the path-blind listener leaned toward the adapter: preferred in 7 of 12 pairs and closer in timbre in 7 of 12, naturalness 5.00 against 4.67, voice similarity 4.58 against 4.42, audio quality 4.58 against 4.42, with the same single wrong-word clip on both sides. A score of similarity gain minus four times the word-error increase orders the two strengths the way the listener did (+0.034 at 0.6 against +0.025 at 1.0), so the full-pipeline gate now judges both strengths and records the better-scoring one as the adapter's recommended strength.

FINAL_PLACEHOLDER

## What the trainer now does

Every target clip is paired with a randomly drawn other clip of its speaker (3 to 15 seconds, a new draw every epoch, never validation audio). Every 500 updates and at each epoch boundary the trainer measures the held-out flow loss with repeatable noise and speaker identity: eight held-out clips re-rendered from their own codes with random training clips as prompts, 16 solver steps and the app's guidance, compared with the real recordings by CAMPPlus similarity. Identity selects the checkpoint; a checkpoint whose flow loss is above the pretrained decoder's is not kept. Patience, a one-time learning-rate halving, and early stopping follow the GPT trainer's rules. The adapter is installed only if its identity beats the pretrained decoder's by 0.005 and, when the run has a speech benchmark, the selected checkpoint renders the benchmark closer to the real recordings with the adapter than without it (same sentences, reference, and seeds; speaker similarity up by at least 0.005, word error rate within the benchmark's policy). A rejected file is parked as `analysis/<name>.s2mel.rejected` and generation keeps the pretrained decoder. With classifier-free guidance the conditioned branch runs through the adapter and the unconditional branch through the pretrained decoder.

## Limits

One voice, two decoder adapter runs, 36 clip pairs per listening test and one automated listener. The flow-matching loss measures spectrogram regression, not perceived quality, and CAMPPlus similarity to a recording rewards matching that recording's session as much as the voice. The decoder adapter was trained after the GPT adapter from the same cached features; its effect on other speakers, other languages, and low-VRAM tiers is untested, and the vocoder remains frozen.
