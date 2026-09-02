"""Static in-app help assembled from the runtime and generation contracts."""

from __future__ import annotations

import gradio as gr


HELP_MARKDOWN = r"""
## Quick Start

1. In **Voice Generation**, load a clean 3-15 second speaker reference.
2. Enter text, choose its language, and keep the per-language segment-token default.
3. Leave the quality defaults in place and select **Generate voice**. Models load only on this first run.
4. Watch section progress, elapsed time, ETA, realtime speed, VRAM, and the live console tail. Every task is saved below `outputs/`.

Load last values restores the last run of every tab; nothing from earlier runs is shown until you click it in the header.

## Reference Audio

Use one speaker, natural pacing, little echo, and no music. A representative clip is better than a very long one. The media extractor accepts common audio/video formats and can merge ranges such as `1:4; 8:13`. A LoRA / DoRA can carry a recommended reference; the LoRA / DoRA selector loads it only when the current reference is empty.

## LoRA / DoRA Workflow

1. **Prepare:** add media and sidecar captions in **LoRA Dataset Preparation**, scan the files, then prepare clean 24 kHz segments.
2. **Cache:** inspect segment statistics and audio, then select **Cache features now**. Training requires the cache index.
3. **Train:** choose the dataset in **LoRA / DoRA Training**. The measured quality defaults are DoRA, rank 128, alpha 129, learning rate 5e-5, 20 epochs, speaker reference `other`, emotion reference `follow_speaker`, validation reference `other`, and every epoch checkpoint kept (`keep_last_n=0`) without its large optimizer sidecar (`epoch_train_state=False`); BF16 and gradient checkpointing stay on.
4. **Use:** after a checkpoint is saved, select **Use best checkpoint**, which opens the Voice Generation tab with that LoRA / DoRA selected. Strength 1.0 reproduces the trained scale.

**Speaker reference mode:** `self` uses the target clip, `other` uses a deterministic different clip from the same speaker, and `mixed` alternates between them.

**Emotion reference mode:** `self`, `other`, and `mixed` choose emotion independently, while `follow_speaker` uses the exact clip selected for the speaker embedding and matches inference.

**Validation reference mode:** `self` validates with the target clip, while `other` uses a different same-speaker clip for both speaker and emotion conditioning to measure inference-like generalization.

## Speaking Rate

Speaking rate 1.0 is the model's natural pace; values below 1.0 speak more slowly and values above 1.0 speak faster. Training samples calibrate a completed LoRA / DoRA automatically, and **Calibrate speaking rate from this grid** can measure a saved listening grid. Voice Generation auto-applies the saved value so the trained voice matches the words-per-second pace of its recordings.

## Which checkpoint should I use?

Validation loss checks sentences the LoRA / DoRA never saw during training, and lower is better. Training loss checks the clips it is actively learning. When training loss keeps falling but validation loss rises, the LoRA / DoRA is memorizing those clips instead of learning a voice that transfers cleanly to new text. The app calls that overfitting.

After training, the app analyzes the log and recommends the checkpoint with the lowest end-of-epoch validation loss. Use **Checkpoint Grid** to compare that checkpoint with **Base model (no LoRA / DoRA)**, the final file, other saved epochs, and optional strength values. Keep the text, reference, and seed fixed, then listen down the rows. A measured checkpoint evaluation can add unseen-text and training-text accuracy to the verdict before you generate the grid.

The **Base model (no LoRA / DoRA)** row is a plain voice clone: only the reference audio shapes the voice. Its verdict is **Reference-only baseline (no LoRA / DoRA)** and it has no strength value.

For **Evaluation references**, **Same as training validation** reuses the run's validation setting, **self** conditions each validation clip on itself, and **other (inference-like: a different clip of the same speaker)** measures the more realistic different-clip workflow.

Set **Keep last N** to 0 when you want every epoch available for comparison. Early stopping can end a run after validation stops improving, while the `analysis/` folder preserves the automatic verdict and any measured comparison.

## VRAM Tiers

| Tier | GPT | Block swap | Large reference models | CFM cache | Typical section batch |
|---:|---|---:|---|---:|---:|
| 6 GB | INT8 ConvRot | 22 / 24 | CPU | 2048 | 1 |
| 8 GB | INT8 ConvRot | 8 / 24 | On demand | 4096 | 2 |
| 10 GB | BF16 | 8 / 24 | On demand | 6144 | 2 |
| 12 GB | BF16 | 0 | Semantic on demand | 8192 | 4 |
| 16 GB | BF16 | 0 | Emotion model on demand | 8192 | 4 |
| 24 GB | BF16 | 0 | GPU resident | 8192 | 8 |
| 32 GB | BF16 | 0 | GPU resident | 8192 | 8 |

Named tiers are conservative and reserve about 2 GB. Select a tier in **Models & Performance**, then customize individual controls if needed. The estimate is a planning aid; the live panel reports actual process VRAM.

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
- **CUDA out of memory:** select the next lower VRAM tier, lower section batch size and beams, enable low-memory mode, or increase blocks to swap.
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
