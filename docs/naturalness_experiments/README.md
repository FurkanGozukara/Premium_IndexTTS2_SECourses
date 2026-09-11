# Naturalness experiments (prepared 2026-09-10, GPU runs pending)

Everything here is generic: the configs are the app's measured training defaults with three
epochs and only the phases needed for a comparison, and the A/B tool works on any trained voice
folder. Edit `dataset_dir` in each JSON before running (paths are relative to the app folder).

## Part 1: inference-only A/B on an existing voice (about one GPU hour)

No training. Renders the run's held-out final-test sentences with every variant in
`indextts/training/naturalness_ab.py` (pure sampling, lower repetition penalty, decoder
adapter off or weaker, commas removed, silence cap, diffusion guidance, speaking rate 1.0) and,
when a clip is given, with an expressive reference or emotion prompt:

```
venv\Scripts\python.exe tools\naturalness_ab.py --run-dir loras\<voice> ^
    --expressive-reference datasets\<dataset>\segments\<lively_clip>.wav
```

Outputs under `outputs/naturalness_ab/<stamp>/`: `report.md` (word error, identity, style,
pause time and the prosody statistics of every variant against the real recordings),
`report.json`, the grids, and `listening/blind/set_NNN/` folders with `Reference.wav` (the real
recording) and shuffled lettered clips for blind ranking (`listening/keys.json` holds the answers).

Epoch comparison of the same run (does naturalness peak before the loss minimum?):

```
venv\Scripts\python.exe tools\naturalness_ab.py --run-dir loras\<voice> --variants deployed ^
    --checkpoints epoch_002 epoch_004 epoch_006 final
```

Pick the expressive reference with the highest pitch and loudness variability among clean 9 to
16 second training clips (the profile's `analysis/dataset_vocabulary.txt` neighbour,
`dataset_profile.json`, lists the clips; `tools/naturalness_ab.py --help` shows the flags).

## Part 2: three-epoch training runs (about 40 minutes each on an RTX 5090)

| Config | What changes | Hypothesis |
|---|---|---|
| `A_baseline_3ep.json` | nothing (control) | the three-epoch reference point for B to D |
| `B_random_prompts_3ep.json` | `reference_typical: false` | a pinned single reference makes the emotion vector a constant during training, so the model stops using it and falls back to an average delivery; varied prompts keep the pathway alive |
| `C_emotion_from_target_3ep.json` | `emo_ref_mode: "self"` | the base model learned the emotion vector from the target clip (stage two of its training); restoring that lets an expressive reference transfer its delivery |
| `D_random_prompts_batch4_3ep.json` | B plus `batch_size: 4`, `learning_rate: 8e-5` | smoother optimization keeps more of the base model's expressiveness |
| `E_ema_3ep.json` | `ema_decay: 0.999` (the control's settings otherwise) | the running average of the adapter weights is smoother than the last update; one run yields both the raw and the EMA files, saved side by side |

Run one at a time from the app folder (each finishes with its own speech comparison):

```
venv\Scripts\python.exe tools\train_lora.py --config docs\naturalness_experiments\B_random_prompts_3ep.json
```

Then compare the runs with the same sentences and seeds:

```
venv\Scripts\python.exe tools\naturalness_ab.py --run-dir loras\B_random_prompts_3ep --variants deployed sampling --sentences development
```

Decision rule: a change is adopted as a default only when it raises the prosody liveliness ratio
and the blind ranking against the real recordings without raising word error by more than the
gates already used by the trainer (0.02 absolute) or dropping speaker similarity by more than 0.03.

## Outcome on the reference voice (11 September 2026)

Runs A, B and C were trained (about 70 minutes each on an RTX 3090) and rendered with one shared speaker reference on the same development sentences. **A (control) was preferred by the blind listener over B 16 to 8 and tied with C**; B was livelier by every pitch measure and halved word error but lost identity (0.834 against 0.846) and style similarity (0.774 against 0.819) and was heard as "unnatural emphasis"; C was flatter and pausier than the control on every measure. The training defaults therefore stay as they are. Details and the inference-side results are in `NATURALNESS_RESEARCH_2026-09-10.md`, sections 9 to 13.

Run E (EMA decay 0.999, three epochs, 36 minutes on an RTX 5090) was added on the same day: the EMA of the final update kept the identity (0.850) and read the words slightly better (0.89 against 1.02 percent word error on the prosody A/B), but was flatter (pitch std 2.39 against 2.51 semitones) and pausier (pause time 1.73 against 1.56 times the recordings), and lost the blind forced choice 11 to 13. The option stays off by default.
