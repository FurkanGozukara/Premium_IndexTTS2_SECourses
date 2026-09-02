# Checkpoint Grid Implementation Report

## Outcome

Implemented the complete Checkpoint Grid and generalization-analysis workflow described in
`CHECKPOINT_GRID_SPEC.md` v5.1. Training runs can now produce a CPU-only log analysis, optionally
measure saved checkpoints in an isolated worker after training memory is released, stop early when
validation no longer improves, and generate fixed-input listening grids through the normal generation
engine. The Gradio app exposes the workflow in a seventh tab and routes the recommended checkpoint back
to Voice Generation.

The reference run at `outputs/training_runs/furkan_dora_r32` resolves to best epoch 10, validation loss
5.3648, sustained overfitting from epoch 14 at the configured 1% tolerance, and the epoch-10 file under
`loras/SECourses_Furkan_EN_DoRA_r32/best/`. Epoch 14 is within the requested two-epoch tolerance around
the manually diagnosed epoch 12.

## Implemented Areas

- CPU-only metrics analysis, checkpoint discovery, phase classification, recommendations, chart frames,
  and atomic JSON/Markdown reports under `<adapter>/analysis/`.
- Measured base/checkpoint comparison on the original validation split and deterministic training subset,
  including CPU fallback, adapter hot swapping, progress reporting, report persistence, worker, and CLI.
- Trainer integration for automatic analysis, bounded subprocess evaluation, VRAM gating, terminal-status
  restoration, and validation-based early stopping with a normal final adapter.
- Listening-grid planning and generation across checkpoints, strengths, references, and texts. One engine
  is reused, one seed is shared by default, every cell keeps reproducibility artifacts, and partial grids
  remain readable after cancellation or failure.
- Checkpoint Grid UI with adapter discovery, analysis/evaluation controls, verdict table and chart, runtime
  summary, grid setup, process management, cancellation, reload adoption, saved-grid playback, and routing
  to and from Training and Voice Generation.
- Universal-preset coverage for all `grid.*` controls and the new training settings.
- Atomic request/metadata writes and exclusion of `outputs/grids/` from normal recent-output scans.

## Verdict Rules

The analysis uses the last validation event recorded for each epoch. The earliest epoch with the minimum
validation loss is `best`. Epochs before it are `improving`. Later epochs within 1% of the best are
`plateau`. A later epoch starts `overfitting` when its validation loss is at least 1% above the best and no
subsequent epoch returns below that threshold. With no validation data, verdicts remain `unknown` and the
final saved checkpoint is recommended.

Measured evaluation applies the same epoch rules to strength-1.0 rows, marks the adapter-free row as
`base`, and marks extra strengths as `variant`. The base model is always evaluated before any adapter is
applied. Recommendations prefer the best measured strength-1.0 checkpoint; log-only recommendations
prefer a matching `best/` file, then a retained epoch file, then the final saved file.

## Files

New feature modules and entry points:

- `indextts/training/analysis.py`
- `indextts/training/checkpoint_eval.py`
- `indextts/training/eval_worker.py`
- `indextts/training/grid.py`
- `indextts/training/grid_worker.py`
- `tools/evaluate_checkpoints.py`
- `tools/generate_grid.py`
- `ui/grid_tab.py`

New tests:

- `tests/test_training_analysis.py`
- `tests/test_checkpoint_eval.py`
- `tests/test_grid.py`

Updated integration and behavior:

- `indextts/training/trainer.py`, `indextts/training/train_config.py`, `indextts/training/sampling.py`
- `indextts/lora/apply.py`, `indextts/lora/io.py`, `indextts/lora/__init__.py`
- `indextts/infer_v2_5.py`, `indextts/utils/task_output_utils.py`
- `ui/app.py`, `ui/training_tab.py`, `ui/generation_tab.py`, `ui/common.py`, `ui/help_tab.py`
- `presets/system/default.json`, `presets/system/fast.json`, `presets/system/low_vram_8gb.json`,
  `presets/system/quality.json`
- `tests/conftest.py`, `tests/test_trainer_cpu.py`, `tests/test_ui_build.py`, `tests/test_ui_flows.py`
- `ARCHITECTURE_NOTES.md`, `README.md`

## Verification

- `venv\Scripts\python.exe -m pytest tests -q`: **165 passed, 37 skipped, 2 warnings** in 14.67 seconds.
  GPU-marked tests are now opt-in with `INDEXTTS_RUN_GPU_TESTS=1`, so the acceptance run stayed CPU-only.
- App construction: seven expected tabs were printed in the required order and request coverage printed
  `True` (`30` runner keys, `29` inference kwargs, `19` runtime fields).
- Python compilation completed for `indextts`, `ui`, `tools`, and `tests`.
- `git diff --check` completed without whitespace errors.

## Open Points

No functional scope remains open. Real checkpoint evaluation, real listening-grid generation, and web UI
model loading were intentionally not run because GPU verification is reserved for a separate pass.

During an early baseline suite run, before the CPU-only opt-in guard was added, the repository's previous
marker policy allowed existing GPU-marked tests to execute because CUDA was visible. The final required
suite above was rerun with those tests skipped by default; no feature training, checkpoint-evaluation, or
grid-generation job was launched.

## Post-implementation verification (GPU + Chrome, 2026-09-02)

Verified in a real Chrome session against the app on port 7870 with the demo adapter
`loras/SECourses_Furkan_EN_DoRA_r32` (its run logs were copied next to the adapter so the tab can analyze it):

- Checkpoint Grid tab: adapter selection, **Analyze training log** (best epoch 10, overfitting from epoch 14,
  chart turns red from epoch 14), **Evaluate checkpoints now** (worker progress, log tail, verdict table refresh),
  a 6-cell grid (base / best / final x 2 texts, seed 1234) generated in 45 s, audio players rendered per row with
  verdict badges and served through the Gradio file route, saved-grid reopening, and
  **Use recommended checkpoint in Voice Generation** (selects `best/` in the adapter dropdown and switches tabs).
- Training tab: attaches to the run, checkpoints table shows Epoch / Validation loss / Verdict, generalization
  summary and chart, **Compare checkpoints in grid** switches tabs with the adapter selected. A 5-step smoke run
  started from the UI went through training -> automatic analysis -> "Evaluating" phase -> complete with
  `analysis/training_analysis.*` and `analysis/checkpoint_eval.*` written automatically.
- Grid cancellation through `stop.flag` (what the Cancel button writes): status `cancelled`, completed cells kept.
- CLI: `tools/evaluate_checkpoints.py` reproduces the diagnosis numbers (base 6.58, best 5.36, final 5.70).

Fixes applied after the Codex pass:

1. `ui/grid_tab.py`: the grid result folder is a hidden `gr.Textbox` instead of `gr.State`. Gradio only dispatches
   `State.change` (which drives `@gr.render`) for queued events, and the polling timer runs unqueued, so the audio grid
   never rendered.
2. `ui/grid_tab.py`: `grid.adapter_dir` and `grid.checkpoints` are registered with `preset=False`; a preset load used
   to blank the adapter dropdown on page load.
3. `ui/training_tab.py`: training-state discovery ignores `analysis/`, `eval_jobs/`, `samples/` and `.sample_jobs/`
   folders. The evaluation job folders carry their own `status.json`, and the training dashboard attached to them
   instead of the run (regression test added in `tests/test_ui_reattach.py`).
4. `indextts/training/checkpoint_eval.py`: the measured summary now reports the overfitting onset from the training
   log and separately names the earliest saved checkpoint that is already overfitted (the two differ when epoch
   checkpoints were pruned). `eval_worker.py`: clearer completion message.

Known behaviour worth noting: Gradio pauses its timers while the browser tab is hidden, so live progress panels only
update while the tab is visible (normal Gradio behaviour, not specific to this feature).

