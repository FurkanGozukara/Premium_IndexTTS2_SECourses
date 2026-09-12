# Smart sentences and subtitle wrapping

Updated 2026-09-12.

Smart sentences and Every sentence recognize punctuation, not line or caption breaks.
Single newlines, CRLF, blank cue separators and other whitespace runs count as one
spoken space. Sentences can span subtitle cues when **Use caption cue timing** is off.
Unpunctuated captions fall back to bounded clauses/words; no punctuation is invented.
The shared splitter preserves the original text exactly, while inference normalizes
formatting whitespace before text processing, even with text normalization disabled.
Token budget retains its legacy splitting. Explicit pause tags and timed cue slots
retain their existing behavior.

Implementation: `indextts/utils/text_segmentation.py` owns sentence boundaries,
spoken whitespace and packing. `indextts/infer_v2_5.py::_build_text_plan` uses the same
whitespace rule. `ui/generation_tab.py::preview_segments` counts spoken whitespace;
the caption upload/clear event reuses `update_preview` and its complete inputs so the
selected mode, voice target, dictionary and speaking rate apply immediately. Both
single-file and batch subtitle text reach this shared splitter.

Verified regression: the search/embeddings regression fixture had 8 sentences but wrapping
created extra fragments. Before the fix, 14 of the first 18 regressions failed. The
final CPU suite passed **153 tests**, including the real model tokenizer at limits
20, 60, 66, 120 and 220; both sentence modes; SRT/VTT sentences spanning cues; upload
and clear callbacks; explicit pauses; normalized and unnormalized inference plans;
legacy splitting; caption timing; recovery; batch handling and UI construction.
No speech was generated or listened to for this change.

Run from the app directory with `CUDA_VISIBLE_DEVICES` empty for CPU verification:

```powershell
$env:CUDA_VISIBLE_DEVICES = ''
./venv/Scripts/python.exe -m pytest tests/test_sentence_wrapping.py tests/test_segmentation_modes.py tests/test_timing_controls.py tests/test_generation_recovery.py tests/test_subtitle_formats.py tests/test_batch_workflow.py tests/test_ui_build.py tests/test_ui_request_guard.py -q
```
