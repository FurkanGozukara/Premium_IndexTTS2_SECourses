# Focused quality fixes — 2026-09-08

Implemented the five accepted fixes as general application behavior. No voice-specific, dataset-specific, or machine-specific tuning was added.

## Changes

1. **Whole-word segmentation.** Oversized clauses split at word boundaries. Character splitting is reserved for an individual word that cannot fit, or unspaced text. Pronunciation annotations remain protected. Preview and inference share the segmenter.
2. **Meaning-preserving normalization.** Explicit English/Chinese selection controls numeric-only text. Ambiguous contractions such as “she's been” and “it's already been” are preserved. Generation and training feature preprocessing both pass the selected language.
3. **Trustworthy transcript preparation.** Language-tagged sidecars are ordered by the selected language; unsuccessful alignment tries alternatives using the same ASR result. Long-recording TXT is lexically aligned before segmentation, replacing proportional word assignment. Weak subtitle alignment only retains cue ranges passing the selected transcript checks, which run again after trimming. Original transcript files are preserved. Explicit Whisper-only and cue-boundary selections are respected.
4. **Incomplete-speech handling.** Only a real, nonempty end-of-speech sequence is decoded. Failed sections can be split at safe boundaries and retried within the selected budget. Exhaustion produces a failed job rather than a truncated success. Recovery preserves the original section slots, explicit pauses, and healthy batch results.
5. **Optional FP32 full-module training.** `Train speaker/extra modules in FP32` defaults to enabled. Selected fully trained speaker, emotion, and mel modules are promoted before checkpoint restoration and optimizer creation. Disabling it retains the configured base precision on CUDA. Frozen base weights retain their chosen precision; LoRA/DoRA parameters remain FP32. CPU's existing FP32 requirement remains. Export/save dtype is a separate setting.

## Controls and settings propagation

| Location | Setting | Default |
|---|---|---:|
| Training → LoRA / DoRA | Train speaker/extra modules in FP32 | On |
| Generation → Segmentation & Timing | Automatically retry incomplete speech | On |
| Generation → Segmentation & Timing | Maximum speech retries per original section | 14 |
| Generation → Segmentation & Timing | Maximum recovery split depth | 3 |
| Dataset → Transcripts | Recover weakly aligned subtitle cues | On |
| Dataset → Transcripts | Maximum cue fallback transcript error | 0.15 |
| Dataset → Transcripts | Check cue fallback boundary words | On |
| Dataset → Transcripts | ASR timing margin for cue fallback | 40 ms |

These settings are registered in universal presets and passed through request/worker configuration. Zero retries permits no extra attempts; zero split depth permits same-text retries only. Backend recovery does not impose a second hidden cap. Grid requests accept the same recovery settings. Older presets missing the new keys use their defaults.

The settings audit also corrected Prodigy's forced learning rate, truncated warmup counts, runtime-tier callback output mismatch, training tier selection of the wrong GPU, and lost Qwen/DeepSpeed runtime checkbox values. Training tier defaults update both base and mixed precision. Checkpoint resume retains the saved adapter shape/layout and, in Continue mode, saved optimizer state; the UI now explains those constraints.

## Verification

Final frozen-source automated run:

```text
python -m pytest -q -m "not gpu"
565 passed, 1 skipped, 39 deselected in 262.59 seconds
```

Coverage includes real Gradio callback configuration capture, presets, worker propagation, normal/batch recovery, finite retry budgets, protected boundaries, transcript matching/rejection, optimizer settings, FP32 updates, checkpoint resume, and dtype compatibility. GPU-marked tests were excluded from this CPU suite; the following real GPU jobs were separately launched through Google Chrome.

Chrome used an isolated app server and generic synthetic narration. Inputs, settings, buttons, progress, errors, waveform previews, and playback were exercised in the actual application.

| Chrome case | Observed result |
|---|---|
| `outputs/0001`: long clause, contractions, English `123` and `42` separated by pauses | Four complete sections, 28.0887 seconds. Actual normalized segments preserve `across`, `she's been`, and `it's already been`; numbers become English. Captured Gradio values reconstruct the saved request exactly. Both 500 ms pauses and the 200 ms section gap are exact. |
| `datasets/qa_language`: German and English sidecars, English selected | English sidecar chosen with 100% word alignment, despite the German filename sorting first. Three accepted clips. |
| `datasets/qa_txt`: recording plus TXT, no timed captions | Four correctly paired clips. “Set the values to 123.” and “Then click save and close the window.” remain separate. All four features cached through Chrome. |
| `datasets/qa_bad`: deliberately unrelated captions | Zero accepted clips; visible explanation that no subtitle/cue matched recognized speech. Source subtitles preserved. |
| `outputs/0002`: 50 mel tokens, recovery disabled | Failed immediately on missing EOS; zero retries and no WAV saved. |
| `outputs/0003`: batch size 2, 150 mel tokens, retry limit 10, split depth 4 | Four retries, then completion. 14.4758-second WAV. Offline cached Whisper on CPU recognized every number 1–16 followed by “Thank you”, without omissions or duplication. |
| `outputs/0004`: 50 mel tokens, retry limit 1, split depth 0 | Exactly one additional attempt, then an explicit failure; no WAV saved. |
| Training `qa_fp32_on` / `qa_fp32_off` | Both completed two CUDA updates with BF16 base/autocast. Saved configs differ only in name and the checkbox value. Actual speaker Adam moments are FP32 when enabled and BF16 when disabled. All saved tensors and moments are finite. |
| Precision presets and runtime tier | Checkbox off/on save/load roundtrip passed; default is enabled. Selecting the 12 GB runtime tier updates its controls without the previous callback error. Loading the default system preset restores the new defaults. |
| Final startup | Restarted the isolated server with the final source, verified request coverage for 32 runner keys, 32 inference kwargs and 23 runtime fields, and confirmed the FP32 checkbox is enabled after loading the default preset. Empty text was rejected before creating a generation job. |

In the two otherwise matching GPU training checks, all 245,760 speaker weight values changed with FP32 enabled; 40,768 changed with BF16 storage. Bias counts were 1,280/1,280 versus 64/1,280. This demonstrates preservation of small updates, not an audible quality comparison. Both runs used the same separately selected FP32 export format.

## Scope and limitations

- Browser QA used Windows, Chrome, and CUDA device 0. Linux and other GPU architectures were not exercised live.
- Visual review used the available desktop browser viewport. A separate small-screen layout was not verified.
- ASR, logs, PCM validation, and working playback controls were verified. Subjective timbre, naturalness, and prosody were not assessed by listening.
- The TXT fixture used the visible `Snap to silence` option off to isolate lexical boundaries. With safe snapping enabled, existing boundary logic rejected its first sentence because Whisper placed the first word at time zero. Default behavior was not relaxed.
- The dataset UI still calls a finished preparation pass “complete” when it accepts zero clips; the zero count and rejection reason are displayed. A failed generation may leave the previous successful audio visible, but creates no new output audio.
- New preparation/alignment behavior applies when preparing data again. Existing datasets and model artifacts were not rewritten.

## Local QA evidence and integration

QA artifacts are retained outside the main checkout under `G:/Index_TTS_v4/quality_fixes_20260908/`: the `outputs`, `datasets`, and `loras` paths above, plus `tests/_tmp/quality_qa/full_suite.log`, `chrome_generation_artifact_audit.json`, `chrome0003_whisper_cpu.json`, and `integration_result.json`. These synthetic QA artifacts are not shipped as source or presets.

The tested source was integrated into the main checkout after checking for overlapping edits. Integrated files were compared with the tested copies, and hashes confirmed that the ten pre-existing concurrent files were preserved. The existing app on port 7866 and its ongoing work were not stopped. Restart that app when its active work finishes to load the new UI and code.
