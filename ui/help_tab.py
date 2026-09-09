"""Static in-app help assembled from the runtime and generation contracts."""

from __future__ import annotations

import gradio as gr


HELP_MARKDOWN = r"""
## Quick Start

1. In **Voice Generation**, load a clean 3-15 second **Reference Voice**.
2. Enter text, choose its language, and keep the per-language segment-token default.
3. Leave the quality defaults in place and select **Generate voice**. Models load only on this first run.
4. Watch section progress, elapsed time, ETA, realtime speed, VRAM, and the live console tail. Every task is saved below `outputs/`.

Load last values restores the last run of every tab; nothing from earlier runs is shown until you click it in the header.

## Reference Audio

Use one speaker, natural pacing, little echo, and no music. A representative clip is better than a very long one. The media extractor accepts common audio/video formats and can merge ranges such as `1:4; 8:13`. A LoRA / DoRA can carry a recommended reference; the LoRA / DoRA selector loads it only when the current reference is empty.

## LoRA / DoRA Workflow

Automatic checkpoint, decoder-strength, and decoding choices use validation recordings only. Among the checkpoints that pass the Base guards, the speech comparison selects by one deployment score, the same one the decoder gate uses: the paired speaker-similarity gain over Base, minus four times any paired word-error increase, plus a small term for pausing more like the person than Base does; validation loss only breaks ties. A decoder that cannot complete its full-pipeline validation gate is preserved outside automatic loading, with a failed or skipped status. If a separate final-test dataset is configured, a final phase freezes the checkpoint, decoder, calibrated speaking rate, and decoding settings before comparing the deployed pipeline with Base; final-test results never retune those choices.

1. **Prepare:** add media and sidecar captions in **LoRA Dataset Preparation**, scan the files, then prepare clean 24 kHz segments with the measured 4-20 second range and 14 second target. Sentence alignment automatically repacks complete sentences at source pauses and checks for at least 30 ms of quiet audio at both output edges after cleanup.
2. **Audit:** for speaker and transcript screening, open **Voice and transcript audit**. Enter a clean reference recording and source filename stems to reserve for validation and optional final testing. **Audit and create training dataset** checks speaker consistency, transcribes the exported clips, and checks their first and last words. Your transcripts are the authority on how names and terms are spelled: the recognizer's spelling of a word your subtitles use is not an error, while missing or extra words still are. Clips that fail only the transcript checks get a second opinion from the full whisper-large-v3 model and are kept when it agrees with your transcript. It selects the separate audited dataset when finished. Rejection reports preserve the reasons for review; speech-recognition errors can also flag clean clips.
3. **Cache:** inspect segment statistics and audio, then select **Cache features now**. Training requires the cache index.
4. **Train:** choose the dataset in **LoRA / DoRA Training**. The measured quality defaults are DoRA, rank 128, alpha 128, batch size 1, gradient accumulation 1, learning rate 4e-5, 10 epochs, 200 warmup steps, speaker reference `other`, emotion reference `follow_speaker`, validation reference `other`, and every epoch checkpoint kept (`keep_last_n=0`) without its large optimizer sidecar (`epoch_train_state=False`); BF16 and gradient checkpointing stay on. With batch size 1 and accumulation 1, each epoch gives one optimizer update per training clip.
5. **Adapt the decoder:** with **Adapt the voice decoder after training** enabled (the default), the run then trains a second small adapter on the semantic-to-mel decoder, which owns timbre and spectral detail, and keeps it as `<name>.s2mel.safetensors` beside the GPT adapter only if it measures closer to the speaker twice: on re-rendered held-out clips, and through the full pipeline on the speech benchmark against the same checkpoint without it. A rejected adapter is parked in `analysis/` and the summary says why. **Decoder training codes** chooses what the adapter learns to render: the recordings' own semantic codes (`real`, the default), the selected checkpoint's teacher-forced predictions for the same clips (`gpt`), or half of each (`mixed`); on the measured voice the GPT's codes traded identity for intelligibility in the wrong direction, so `real` stays the default. Voice Generation applies an installed adapter automatically whenever that LoRA / DoRA is selected: the **Voice decoder adapter** dropdown shows which file is in use, **None** plays the GPT adapter alone for comparison, and **Voice decoder adapter strength** starts at the strength the full-pipeline test recommended (1.0 or 0.6) and can be changed.
6. **Sweep decoding:** with **Sweep decoding settings after training** enabled (the default), the run renders the speech benchmark at other temperatures, guidance rates, and beam counts and keeps a change only when it beats the defaults on speaker similarity and word error; Voice Generation applies the winner with the calibrated speaking rate. Automatic references prefer the clean clip nearest the speaker's median pitch and pace (**Prefer a reference near the speaker's median pitch and pace**).
7. **Use:** wait for automatic post-training phases and any configured final test to finish, inspect their separate statuses, then select **Use best checkpoint**, which opens Voice Generation with the speech-recommended checkpoint. A saved weight file alone does not mean every quality check completed. Strength 1.0 reproduces the trained scale.

**Speaker reference mode:** `self` uses the target clip, `other` uses a deterministic different clip from the same speaker, and `mixed` alternates between them.

**Emotion reference mode:** `self`, `other`, and `mixed` choose emotion independently, while `follow_speaker` uses the exact clip selected for the speaker embedding and matches inference.

**Validation reference mode:** `self` validates with the target clip, while `other` uses a different same-speaker clip for both speaker and emotion conditioning to measure inference-like generalization.

Validation defaults to complete source recordings and the whole holdout, every 250 updates and at epoch boundaries. Alternative references come only from training clips. The **Automatically stop when progress stalls** checkbox starts checked: after 1,000 updates, completed warmup and two dataset passes, six checks without an improvement greater than 0.005 stop the run and retain its best checkpoint. Uncheck it to train through the configured limit.

## Speaking Rate

Speaking rate 1.0 is the model's natural pace; values below 1.0 speak more slowly and values above 1.0 speak faster. Training first estimates a completed LoRA / DoRA's pace from its epoch samples, then replaces that estimate with a calibration from matched held-out sentences once the automatic speech comparison finishes; **Calibrate speaking rate from this grid** can also measure a saved listening grid. Voice Generation auto-applies the saved value so the trained voice matches the words-per-second pace of its recordings, and the **Saved speaking rate for this LoRA / DoRA** field lets you override it per adapter with **Save speaking rate**.

## Which checkpoint should I use?

Validation loss checks sentences the LoRA / DoRA never saw during training, and lower is better. Training loss checks the clips it is actively learning. When training loss keeps falling but validation loss rises, the LoRA / DoRA is memorizing those clips instead of learning a voice that transfers cleanly to new text. The app calls that overfitting.

After training, the app analyzes the log and recommends the checkpoint with the lowest end-of-epoch validation loss. Use **Checkpoint Grid** to compare that checkpoint with **Base model (no LoRA / DoRA)**, the final file, other saved epochs, and optional strength values. Keep the text, reference, and seed fixed, then listen down the rows. A measured checkpoint evaluation can add unseen-text and training-text accuracy to the verdict before you generate the grid.

The automatic speech comparison report also lists **Pause time vs real** for each candidate: the generated clips' internal pause time divided by the real recordings' on the same sentences. 1.00 matches the person, above 1 pauses longer, below 1 rushes between sentences. Listeners often prefer slightly longer pauses than a fast narrator takes, so treat it as a description, not a score; it does not change the recommendation.

The **Base model (no LoRA / DoRA)** row is a plain voice clone: only the reference audio shapes the voice. Its verdict is **Reference-only baseline (no LoRA / DoRA)** and it has no strength value.

For **Evaluation references**, **Same as training validation** reuses the run's validation setting, **self** conditions each validation clip on itself, and **other (inference-like: a different clip of the same speaker)** measures the more realistic different-clip workflow.

Set **Keep last N** to 0 when you want every epoch available for comparison. Early stopping can end a run after validation stops improving, while the `analysis/` folder preserves the automatic verdict and any measured comparison.

## GPU VRAM Presets

The read-only system presets in the header are the seven card sizes. On first start the app selects the preset of the detected GPU; a card counts as a tier from 500 MB below its nominal size, so 31.5 GB and above is a 32 GB card and 9.5 GB and above a 10 GB card. A saved user preset, or whichever preset was loaded last, is always restored instead, and **Reset** returns to the detected tier. The seven presets cannot be overwritten or deleted from the preset interface (Save and Delete refuse their names in any spelling, and their files are rewritten at every start); save your own settings under a new name. Each preset sets the inference runtime, the generation decoding and the training settings together. Every tier keeps the BF16 GPT, sampling, CFM temperature 0.9 and at least the 40 diffusion steps of the former quality preset; the 24 GB and 32 GB presets refine with 50 steps. Four beams are the measured optimum (the decoding sweeps of real training runs scored five beams worse than three), so every preset from 8 GB up uses four and only 6 GB drops to two. A smaller card pays with speed first: reference encoders on demand or on CPU, a shorter CFM cache, and on 6 GB frozen GPT blocks streamed from CPU.

| Preset | Whole-GPU peak | GPT blocks streamed | Reference and emotion models | CFM cache | Beams | Diffusion steps | Training |
|---|---:|---:|---|---:|---:|---:|---|
| 6 GB GPU | within 5 GB | 22 / 24, one ring slot | CPU | 2048 | 2 | 40 | BF16 base, 22 blocks streamed |
| 8 GB GPU | within 7 GB | 0 | Semantic encoder on CPU, emotion model on demand | 4096 | 4 | 40 | BF16 base, resident |
| 10 GB GPU | within 9 GB | 0 | On demand | 6144 | 4 | 40 | BF16 base, resident |
| 12 GB GPU | within 11 GB | 0 | GPU resident | 8192 | 4 | 40 | BF16 base, resident |
| 16 GB GPU | within 15 GB | 0 | GPU resident | 8192 | 4 | 40 | BF16 base, resident |
| 24 GB GPU | within 22 GB | 0 | GPU resident | 8192 | 4 | 50 | BF16 base, resident |
| 32 GB GPU | within 30 GB | 0 | GPU resident | 8192 | 4 | 50 | BF16 base, resident |

The peak is the memory the whole card reports, every process and CUDA context included, measured while generating (base voice, a LoRA / DoRA with its decoder adapter, emotion-text mode), preparing and auditing a dataset, caching features and training with samples, evaluation and the decoder adapter (`tools/gpu_tier_calibration.py`). Training keeps rank 128 and batch size 1 on every tier; the per-epoch sample renders in its own process with the largest tier that fits into the memory left beside the training model, and the speech comparison after training uses the training's own tier with the measured benchmark settings (3 beams, 25 steps) so runs stay comparable. The **GPU VRAM preset** dropdown beside the training dataset fills the training controls for one tier without changing the rest of the preset; **Models & Performance** does the same for the runtime alone, and its estimate is a planning aid while the live panel reports actual process VRAM. **Prevent VRAM accumulation** stays off in every preset: on the 6 GB tier it raised the transient peak by about 1 GB instead of lowering it. Dataset preparation adapts on its own: Whisper alignment keeps only the decoder attentions it needs for word timestamps, the feature cache picks the largest clip batch that fits the free VRAM, and the audit's second-opinion Whisper model runs on the CPU when the card cannot hold it beside the first model. Models left resident by an in-process generation are released automatically when training, dataset preparation, feature caching, an audit, a grid, an evaluation, the isolated benchmark or an isolated generation starts, so the budget belongs to that worker; they reload at the next generation (a generation that is still running keeps them).

## Parameter Glossary

- **Temperature / top-p / top-k:** control autoregressive variation. The defaults favor natural speech without excessive instability.
- **Beams:** broaden decoding search. More beams cost time and VRAM; low-memory tiers cap the useful range.
- **Repetition penalty:** discourages repeated semantic tokens. Keep the model default unless speech loops.
- **Diffusion steps:** trade CFM refinement for speed. 25 is the quality default.
- **CFG rate:** strength of diffusion conditioning. 0.7 is balanced.
- **CFM temperature:** diffusion noise scale. 1.0 is the best-quality default.
- **Segment tokens:** bounds each autoregressive section. EN/ES 60, AR 80, JA 100, and ZH 120 are recommended.
- **Latent multiplier:** converts semantic-code length to acoustic duration; speaking rate divides this value before inference, so normal use can leave 1.72 unchanged.
- **Semantic layer:** reference encoder layer used for speaker/emotion conditioning. Layer 17 is trained and recommended.
- **Section batch size:** number of text sections generated together. Increase only within the active VRAM tier hint.
- **Audio tuning:** optional FFmpeg post-processing. Bypass preserves the model waveform exactly.

## Pause Tags

Inline pauses are removed before tokenization and inserted exactly in the output plan:

```text
This is deliberate.[pause:500ms] Now continue.
Wait [pause:0.8s] and speak.
An alternate form is <pause=0.5>.
```

Caption cue timing owns its own gaps and target slots. It overrides section silence while each timing unit is generated.

## Troubleshooting

- **No output / model file error:** use **Models & Performance** to download or verify the base files. Selecting INT8 downloads its GPT automatically; when the Hugging Face file is unavailable, the run clearly warns and uses BF16.
- **CUDA out of memory:** load the next smaller GPU VRAM preset (it lowers generation, training and runtime together), close other GPU applications, or lower section batch size and beams, enable low-memory mode, and increase blocks to swap by hand.
- **Training cannot start:** verify `manifest.jsonl` and `cache/index.jsonl` exist in the selected dataset.
- **Training stop takes a moment:** graceful stop finishes one optimizer step and writes an interrupted checkpoint. **Force stop** terminates the entire subprocess tree.
- **Caption timing sounds stretched:** use sentence-length cues where possible and avoid extremely short slots for long phrases.
- **A backend error occurs:** the UI remains available; the error is shown in the tab and the full traceback stays in the console/log tail.

## Links

- [IndexTTS project](https://github.com/index-tts/index-tts)
- [Premium SECourses release notes and support](https://www.patreon.com/posts/139297407)
"""


def build_help_tab() -> None:
    with gr.Tab("Help", id="help"):
        # Long-form prose is capped to a comfortable measure; the rest of the app
        # uses the full width because it is dense controls, not reading.
        gr.Markdown(HELP_MARKDOWN, elem_classes=["help-prose"])


__all__ = ["HELP_MARKDOWN", "build_help_tab"]
