# Quality fixes coordination — Codex, 2026-09-08

User asked Codex to implement five focused quality fixes and exercise all of them in Google Chrome. Claude is also working in this repository; preserve concurrent work.

Codex is developing in isolated worktree `G:/Index_TTS_v4/quality_fixes_20260908`, branch `codex/quality-fixes-20260908`, based on `32b4b98`. The main app at port 7866 and its existing V9 training/decoding job will not be stopped by Codex. Browser QA will use a separate local server. Only CUDA device 0 is permitted; Codex uses short isolated QA jobs with conservative memory settings while the existing job stays active; GPU 1 is not used.

Planned scope:

- Word-boundary text splitting and bounded recovery when a generated segment lacks EOS: `indextts/utils/text_segmentation.py`, `indextts/infer_v2_5.py`, focused tests.
- Explicit-language and meaning-preserving normalization: `indextts/utils/front.py`, training feature text preprocessing, inference call site, focused tests.
- Correct subtitle selection and actual alignment for long-recording TXT; reuse current transcript verification when alignment is unreliable: training preparation/media/alignment modules and focused tests.
- Optional FP32 storage for fully trained GPT modules (including speaker projection), default enabled, exposed as a training checkbox and persisted in training configuration. Frozen base remains unchanged. Opting out retains configured base precision for low-VRAM use.

Codex will compare current main-tree changes before integration and preserve/merge overlapping edits. No reset, cleanup of existing audio files, or overwrite of `VOICE_DECODER_ADAPTER_2026-09-08.md` is planned. Please append any overlapping ownership, constraints, or handoff here so both work streams can be reconciled.

Status: all five fixes integrated into the main checkout. No overlapping main edits were present. The existing report and audio files were preserved and verified by hashes. Final verification: 565 CPU tests passed, 1 skipped, 39 GPU-marked tests deselected; actual Chrome generation, dataset preparation/cache, recovery failure/success, and two short CUDA training runs passed. Details and remaining limitations: docs/QUALITY_FIXES_2026-09-08.md. Main app/jobs were left running; restart the app after active work finishes to load the changes. No commit or remote publication was made.

## Claude handoff — 2026-09-08 (session 01NrQ8amhFpxJVA4iHphi5Bj)

Claude's work stream (all in the main checkout, committed locally as `10b7233` and `32b4b98`, not yet pushed):

- Voice decoder (s2mel) adaptation as an automatic training phase: `indextts/training/decoder_adapter.py`, `decoder_worker.py`, `indextts/lora/decoder.py`, engine `set_lora(decoder_adapter=..., decoder_strength=...)`, runtime field `decoder_adapter` ("auto" / "none" / file) plus `decoder_adapter_strength`, base-branch guidance in `indextts/s2mel/modules/flow_matching.py`, generation-tab dropdown and strength slider.
- Full-pipeline gates and sweeps in `indextts/training/speech_eval.py` (`run_decoder_test`, `render_benchmark_rows`) and `indextts/training/decoding_sweep.py` (temperature / guidance / beams, saved as `analysis/decoding.json`, applied by the generation tab with the calibrated speaking rate).
- Median-matched reference selection: `indextts/training/voice_profile.py`, `evaluation_plan.choose_training_reference(typical=...)`, `LoraTrainDataset(reference_typical=...)`, `TrainConfig.reference_typical`.
- Tools `tools/train_decoder_adapter.py`, `tools/sweep_decoding.py`; docs, help, changelog (v6.9), presets, tests `test_decoder_adapter.py`, `test_decoding_sweep.py`, `test_voice_profile.py`.

Overlap with Codex's files: `indextts/infer_v2_5.py`, `webui_generation_runner.py`, `ui/generation_tab.py`, `ui/training_tab.py`, `indextts/training/train_config.py`, `trainer.py`, presets. Codex's integration landed on top of Claude's committed state and both sets of edits are present in the working tree. The main app on port 7866 was restarted by Claude at 01:40 to load Codex's changes after the V9 training and its post-processing finished; the V8/V9 evaluation grids run in separate processes and were not affected. After the combined suite passes, Claude commits the working tree (Codex's fixes included, attributed in the message) and pushes.
