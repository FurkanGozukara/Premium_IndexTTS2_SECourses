# UI integration notes

The UI task did not modify `webui_generation_runner.py`, `webui_subprocess_worker.py`, or any file under `indextts/`.

The current runner contract is fully covered by `ui.generation_tab.build_generation_request`: 29 top-level request keys and 29 `infer_kwargs`. The UI sends the eight inference extras that the runner otherwise defaults (`segment_budget_scale_non_cjk`, `cfm_temperature`, `seed`, `reuse_spk_cond_for_emo`, `enable_pause_tags`, `trim_silence_ms_threshold`, `target_duration_s`, and `target_duration_mode`). The startup source self-check reports missing or unknown keys without loading a model.

No engine-side integration patch is currently required. `runtime` contains the 18 `RuntimeConfig` fields plus the runner-supported construction options `model_dir`, `cfg_path`, `use_qwen_emo`, and `use_deepspeed`; `RuntimeConfig.from_dict()` correctly ignores those construction-only entries while `create_tts()` consumes them.

