# IndexTTS 2.5 Premium SECourses App — v5 Architecture Notes (shared contract for all implementation tasks)

This file is the single source of truth for how the v5 upgrade fits together. Every implementation task
must follow these contracts so independently developed modules integrate without friction.

## Environment facts (do not fight them)
- Python 3.12 venv at `venv/` (Windows: `venv\Scripts\python.exe`). torch 2.13.0+cu130, transformers 5.16.1,
  gradio 6.26.0, safetensors 0.8, huggingface_hub 1.29, librosa 1.0, numpy 2.5, pandas 3, matplotlib 3.11.
- The app must run on Windows AND Linux. Never add a platform-only dependency. Never pin new library versions in
  `../index_TTS_requirements.txt` (unpinned names only). `peft`, `bitsandbytes`, `deepspeed` are NOT installed and
  must not be required at runtime.
- GPUs in the dev machine: cuda:0 = RTX 5090 32GB (use this one), cuda:1 = RTX 3090 (busy, do not use).
  Set `CUDA_VISIBLE_DEVICES=0` when running anything on GPU.
- Model files live in `models/` (repo-relative): `gpt.pth` (fp32 state dict, 815M params, UnifiedVoice),
  `s2mel.pth`, `codec.pth`, `feat1.pt`, `feat2.pt`, `config.yaml` (version 2.5), `qwen0.6bemo4-merge/`,
  `hf_cache/w2v-bert-2.0/`, `hf_cache/campplus_cn_common.bin`, `hf_cache/bigvgan/`.
- transformers 5.x: KV caches are `DynamicCache` objects (not tuples). Beam search currently CRASHES in
  `GPT2InferenceModel._reorder_cache` (model_v2.py) because it assumes tuples. Must be fixed (use
  `past.reorder_cache(beam_idx)` when available).
- The GPT (`UnifiedVoice` in `indextts/gpt/model_v2.py`) wraps a stock HF `GPT2Model` (`self.gpt`) with
  24 `GPT2Block`s in `self.gpt.h`; each block has HF `Conv1D` layers `attn.c_attn` (1280->3840),
  `attn.c_proj` (1280->1280), `mlp.c_fc` (1280->5120), `mlp.c_proj` (5120->1280). `Conv1D.weight` is stored
  as `[in_features, out_features]` (transposed relative to nn.Linear).

## Package layout (new code goes here)
```
indextts/runtime/            # runtime configuration & memory management (task RUNTIME)
    __init__.py
    gpu.py                   # device inventory, total/free VRAM, vram cap, memory stats helpers
    vram_presets.py          # tiers, RuntimeConfig dataclass, preset resolution, auto tier detection
    block_swap.py            # block streaming / swapping for the GPT2 block stack (inference + training)
    residency.py             # on-demand GPU residency for auxiliary models (move to GPU for use, back to CPU)
    progress.py              # ProgressReporter (console single-line progress + JSON progress file + callback)
indextts/quant/              # int8 convrot (task INT8)
    __init__.py
    convrot_int8.py          # conversion, custom Linear, checkpoint detection/loading, STE training op
indextts/lora/               # LoRA / DoRA (task LORA)
    __init__.py
    layers.py                # LoRALinear / DoRA wrappers for nn.Linear, HF Conv1D and ConvRotInt8Linear bases
    io.py                    # safetensors save/load with JSON metadata, scanning, resume detection
    apply.py                 # inject/remove adapters on UnifiedVoice, set strength, list targets
indextts/training/           # dataset preparation + training (task TRAINING)
    __init__.py
    media.py                 # ffmpeg/ffprobe helpers (any audio/video -> mono wav), duration probing
    subtitles.py             # SRT/VTT/SBV parsing reuse (indextts.utils.subtitle_utils) + cue merging into sentences
    whisper_asr.py           # transformers Whisper transcription with word timestamps (optional, auto-download)
    segmenter.py             # cue/word based segmentation, silence snapping, min/max duration rules
    dataset_prep.py          # DatasetPrepConfig + run_dataset_prep(config, reporter) -> manifest
    features.py              # feature caching (text tokens, semantic codes, campplus emb, emo vec) using the loaded models
    dataset.py               # torch Dataset/collate over cached features
    trainer.py               # TrainConfig + LoRA/DoRA trainer loop (block swap aware, grad checkpointing, val, samples)
    speaking_rate.py         # CPU-only words/s measurement and per-voice speaking-rate calibration
    analysis.py              # CPU-only metrics analysis, phase verdicts, recommended checkpoint
    checkpoint_eval.py       # teacher-forced base/checkpoint comparison on matching train/validation splits
    eval_worker.py           # isolated checkpoint-evaluation worker and status/progress contract
    grid.py                  # deterministic checkpoint/strength/reference/text listening grids
    grid_worker.py           # isolated listening-grid worker and status/progress contract
    train_worker.py          # subprocess entry: python -m indextts.training.train_worker --config cfg.json --state-dir DIR
    prep_worker.py           # subprocess entry for dataset preparation
    charts.py                # helpers that turn metrics.jsonl into pandas frames for gr.LinePlot
ui/                          # Gradio UI modules (task UI)
    __init__.py
    common.py                # theme, CSS, JS head, shared widgets (progress panel, confirm-cancel JS), helpers
    presets_store.py         # system (read-only, presets/system) + user (presets/user) preset store used by all tabs
    generation_tab.py        # Voice Generation tab
    batch_tab.py             # Batch Generation tab
    dataset_tab.py           # LoRA Dataset Preparation tab
    training_tab.py          # LoRA / DoRA Training tab (+ LoRA manager)
    grid_tab.py              # Checkpoint Grid, generalization verdicts, evaluation and saved-grid playback
    models_tab.py            # Models & Performance tab (VRAM presets, model variant download, device, block swap)
    help_tab.py              # Help / About
webui.py                     # thin entry point: CLI args, builds tabs, launches
webui_generation_runner.py   # generation orchestration (in-process or subprocess); already exists, extended
webui_subprocess_worker.py   # generation subprocess entry; already exists, extended (runtime + lora + progress)
tests/                       # pytest unit tests (CPU-only by default; GPU tests require an explicit opt-in)
tools/                       # CLI tools, including evaluate_checkpoints.py and generate_grid.py
```
Delete obsolete legacy code only in task CLEANUP (see that spec). Other tasks must not delete files.

## RuntimeConfig (indextts/runtime/vram_presets.py) — the contract between UI, runner, worker and the engine
```python
@dataclass
class RuntimeConfig:
    device: str = "cuda:0"                 # "cuda:N" | "cpu" (UI dropdown of detected GPUs)
    model_variant: str = "bf16"            # "bf16" (official gpt.pth cast to bf16) | "int8_convrot" (models/gpt_int8_convrot.safetensors)
    gpt_dtype: str = "bf16"                # "bf16" | "fp16" | "fp32" compute/storage dtype for bf16 variant
    blocks_to_swap: int = 0                # 0 = no swapping; N = stream N of the 24 GPT blocks from CPU; -1 = auto (fit budget)
    swap_ring_size: int = 2                # GPU ring slots for streamed blocks (1..4)
    pin_swap_memory: bool = True           # pinned host memory for streamed blocks (faster)
    aux_residency: dict = field(default_factory=lambda: {   # "gpu" | "on_demand"; ref/Qwen also allow "cpu"
        "semantic_model": "gpu",           # w2v-bert-2.0 (580M) used only for reference feature extraction
        "qwen_emo": "on_demand",           # Qwen 0.6B emotion text model, only for emotion-text mode
        "campplus": "gpu",
        "semantic_codec": "gpu",
        "s2mel": "gpu",
        "bigvgan": "gpu",
    })
    attention_backend: str = "sdpa"        # "sdpa" | "flash_attention_2" | "eager" for the GPT2 stack
    use_accel: bool = False                # flash-attn CUDA-graph accel engine (only if flash_attn importable; beams=1 only)
    torch_compile_s2mel: bool = False
    use_cuda_kernel_bigvgan: bool = False
    s2mel_estimator_autocast: bool = False # BF16 autocast for the CFM DiT only; vocoder stays FP32
    cfm_cache_length: int = 8192           # s2mel CFM estimator cache length
    vram_reserve_gb: float = 2.0           # VRAM to keep free (presets are designed to leave 2 GB free)
    vram_tier: str = "auto"                # "auto" | "6" | "8" | "10" | "12" | "16" | "24" | "32" | "custom"
    lora_path: str = ""                    # optional LoRA/DoRA safetensors to apply on the GPT
    lora_strength: float = 1.0
    lora_merge_into_base: bool = False      # fold a BF16 adapter into base weights for faster inference
    max_section_batch_size_hint: int = 8   # UI can read this to cap the section batch size slider for the tier
```
`RuntimeConfig.to_dict()/from_dict()` must round-trip JSON. `resolve_preset(tier: str, gpu_total_gb: float,
gpu_free_gb: float) -> RuntimeConfig` returns the preset for a tier (tiers: 6, 8, 10, 12, 16, 24, 32).
`auto_tier(gpu_total_gb)` picks the largest tier <= physical VRAM. Presets are conservative and leave
`vram_reserve_gb` (2 GB) free on a GPU of exactly the tier size.

| tier | variant | swap/ring | semantic | CAMPPlus | Qwen | CFM | DiT BF16 | beams/batch/text hints |
|------|---------|-----------|----------|----------|------|-----|----------|------------------------|
| 32 | bf16 | 0/2 | gpu | gpu | gpu | 8192 | no | 8 / 8 / 200 |
| 24 | bf16 | 0/2 | gpu | gpu | gpu | 8192 | no | 6 / 8 / 160 |
| 16 | bf16 | 0/2 | gpu | gpu | on_demand | 8192 | no | 4 / 4 / 120 |
| 12 | bf16 | 0/2 | on_demand | gpu | on_demand | 8192 | no | 3 / 4 / 120 |
| 10 | bf16 | 8/2 | on_demand | gpu | on_demand | 6144 | no | 3 / 2 / 100 |
| 8 | int8_convrot | 8/2 | on_demand | gpu | on_demand | 4096 | no | 2 / 2 / 80 |
| 6 | int8_convrot | 22/1 | cpu | cpu | cpu | 2048 | yes | 1 / 1 / 40 |

Calibration was run on idle GPU 0 with `tools/vram_benchmark.py --all --emulate` (seed 123, 2 GB
allocator reserve) on 2026-09-02. All tiers fit. Values are per-process GiB; high-tier reserved peaks
reach the deliberate allocator cap.

| tier | load alloc | peak alloc | peak reserved | audio s | wall s | RTF | mel tok/s |
|------|-----------:|-----------:|--------------:|--------:|-------:|----:|----------:|
| 6 | 1.703 | 2.253 | 2.629 | 28.403 | 35.665 | 1.256 | 23.774 |
| 8 | 2.016 | 4.939 | 4.984 | 48.489 | 29.729 | 0.613 | 53.168 |
| 10 | 2.364 | 5.287 | 5.357 | 49.533 | 27.910 | 0.563 | 59.523 |
| 12 | 2.609 | 5.532 | 8.191 | 85.449 | 18.422 | 0.216 | 210.157 |
| 16 | 4.774 | 7.099 | 12.084 | 85.449 | 17.453 | 0.204 | 192.137 |
| 24 | 5.884 | 9.952 | 21.996 | 170.899 | 24.359 | 0.143 | 343.382 |
| 32 | 5.884 | 11.180 | 29.998 | 179.908 | 26.830 | 0.149 | 316.423 |

The 6 GB tier therefore keeps its strict 2 GB reserve: its 4 GB allocator budget peaked at only 2.629 GiB
reserved. Its DiT-only BF16 path was compared with FP32 using identical GPT codes, conditioning, CFM noise,
and 25 diffusion steps on the demo text: mel MSE `0.000255775`, relative MSE `1.6762e-5`, maximum absolute
mel delta `0.141567` (FP32 mel mean-square `15.2592`).

The engine (`IndexTTS2`) takes a `RuntimeConfig` and must honour every field. The UI must expose every field.

## Progress protocol (indextts/runtime/progress.py)
`ProgressReporter(label, total=None, progress_file=None, gr_progress=None)` with methods
`update(completed, total=None, desc="", extra: dict=None)`, `set_stage(name)`, `log(msg)`, `finish()`.
It (1) prints a single updating console line `[ 42.7%] 3/7 segments | elapsed 12s | ETA 16s | 3.1x RT | <desc>`
(SwarmUI style: metrics first; on non-TTY prints at most once per second), (2) writes `progress_file` JSON
`{"fraction":0.427,"completed":3,"total":7,"desc":...,"stage":...,"elapsed_s":..,"eta_s":..,"speed":..,"speed_unit":"x RT"|"it/s","vram_used_gb":..,"vram_total_gb":..,"updated_at":..,"extra":{...}}`
atomically (write tmp + os.replace) at most every 0.3 s, and (3) forwards to `gr_progress(fraction, desc=...)`
if given. The UI polls the file with `gr.Timer(0.5)` while a subprocess job runs.

## Generation request contract (webui_generation_runner.py)
The request dict gains `"runtime": RuntimeConfig.to_dict()`, `"progress_file": <path>`, `"lora_path"`,
`"lora_strength"`, and `"lora_merge_into_base"`.
`create_tts(runtime_options)` builds `IndexTTS2(cfg_path, model_dir, runtime=RuntimeConfig.from_dict(...))`.
`IndexTTS2` caches the loaded LoRA: switching LoRA path/strength must not reload base weights.

`generation.speaking_rate` and `grid.speaking_rate` are UI-only values (0.5 through 1.5, default 1.0), and
`generation.auto_lora_speaking_rate=True` is also UI-only; none is a runner or engine key. Request builders fold
them into the existing contract as
`infer_kwargs["latent_multiplier"] = round(latent_multiplier / speaking_rate, 4)`. Batch Generation reuses the
Voice Generation request builder and therefore inherits the same fold.

Inference extras may be supplied as top-level request keys (which override `infer_kwargs`) or directly in
`infer_kwargs`: `segment_budget_scale_non_cjk=0.72` (`(0,1]`, `1.0` disables scaling),
`cfm_temperature=1.0` (finite, `>=0`), `seed=None` (`None`/`-1` random),
`reuse_spk_cond_for_emo=False`, `enable_pause_tags=True`, `trim_silence_ms_threshold=0` (milliseconds,
`0` disables), `target_duration_s=None` (positive seconds), and
`target_duration_mode="off"` (`off|natural|pad|trim`, plain-text mode only). Natural duration reuses GPT codes
and clamps the adjusted duration factor to `0.25..4.0`.

Runner-level request keys are `num_candidates=1` (clamped to `1..32`),
`audio_tuning_preset="bypass"` (`bypass|voice_clarity|clear_narration|deharsh|warm|normalize`), and
`audio_tuning_overrides={}`. Override ranges are `low_cut_hz=20..500`, `high_cut_hz=1000..24000`,
`gain_db=-24..24`, `loudnorm_i=-30..-5`, and `deess=0..12`. Candidate 1 is the primary output; every candidate
is retained as `candidate_XX.wav`, and active tuning preserves `<name>_raw.wav` before MP3/MP4 conversion.
Results and metadata include `seed`, `segments_count`, `audio_seconds`, `rtf`, `gpt_time`, `s2mel_time`,
`vocoder_time`, and `peak_vram_gb`.

The shared preview helpers are `indextts.utils.text_segmentation.default_segment_tokens` (EN/ES 60, AR 80,
JA 100, ZH 120), `split_text_by_tokens`, and `indextts.utils.pause_tags.describe_pauses`.

Decode speed on the same demo text (`max_text_tokens_per_segment=60`, batch 1, seed 123) was:

| GPT path | beams 1 mel tok/s | beams 3 mel tok/s |
|----------|------------------:|------------------:|
| BF16, no swap | 58.055 | 50.462 |
| BF16, swap 12 | 27.800 | 27.472 |
| INT8, no swap | 44.976 | 37.290 |
| INT8, swap 12 | 34.875 | 33.971 |
| BF16 + `secourses_demo_dora_smoke` LoRA, no swap | 33.354 | 30.238 |

## Training/preparation worker contract (indextts/training)
Workers are launched as `python -m indextts.training.train_worker --config <json> --state-dir <dir>`.
Dataset preparation drops duplicate normalized sentences across all sources by default, retaining the best-aligned copy closest to the target duration.
Dataset preparation accepts BOM-aware UTF-8, BOM-marked UTF-16/UTF-32, encodings detected by `charset_normalizer` (including Windows Turkish CP1254), and a CP1252/replacement fallback for captions, transcripts, and `metadata.csv`. Sanitized source keys remain ASCII and gain a stable six-hex hash suffix whenever the original stem required replacement, so segment IDs, Whisper caches, and reference-candidate names are repeatable. Mixed folders are handled per media file: SRT/VTT/SBV sidecars are preferred where present, caption-less media falls back to Whisper under `prefer_sidecar`, and orphan subtitles produce warnings without blocking the other inputs.
Sentence-aligned preparation defaults to `boundary_mode="sentence"`. The optional `sentence_or_pause` mode also accepts a non-sentence fragment when each edge is either a sentence edge or has at least `min_pause_boundary_ms=400` between adjacent aligned Whisper words. Pause-accepted rows carry `boundary="pause"` (`"sentence"` otherwise), and per-source and dataset summaries record them in `filter_keep_counts.pause_boundary`; rejected fragments remain in `filter_drop_counts.sentence_boundary`.
Preparation targets 14-second clips within a 4-20 second range. This covers the model's 12-15 second inference segments, preserves more source audio than the old 12-second maximum, and avoids the measured quality loss at 30 seconds.
They write `state_dir/status.json` (phase, step, total_steps, epoch, total_epochs, loss, avg_loss, val_loss, lr,
grad_norm, it_s, eta_s, elapsed_s, vram_used_gb, message, updated_at, last_checkpoint, last_sample),
append `state_dir/metrics.jsonl` (one JSON per logged step: step, epoch, loss, avg_loss, lr, grad_norm, it_s, val_loss?),
append `state_dir/log.txt`, and honour `state_dir/stop.flag` (graceful: finish step, save, exit) and process kill.
The parent UI uses the same `_terminate_process_tree` approach as webui.py for hard cancel.

Quality-first `TrainConfig` defaults are rank 128, alpha 129, learning rate 4e-5, 10 epochs, speaker reference
`other`, emotion reference `follow_speaker`, validation reference `other`, and `keep_last_n=0`. Dropout remains
0.05, weight decay 0.01, batch size 1 with accumulation 1, warmup 200, cosine scheduling, BF16, gradient
checkpointing, validation fraction 0.05 every 250 steps, samples every epoch,
and automatic analysis/evaluation. Validation defaults to complete source recordings
(`val_split_mode="source"`), evaluates the entire holdout (`val_max_batches=0`), and weights
losses by valid tokens. Explicit `split=train|val` labels override the fraction and mode.
The early-stop checkbox defaults to enabled: patience 6, minimum improvement 0.005,
at least 1,000 updates, completed warmup, and two dataset passes before counting stalls.
The absolute best validation checkpoint is retained even when a small improvement does
not reset patience. Its exact step and resumable stopping state are saved in checkpoint
state and `analysis/checkpoint_selection.json`.
With batch size 1 and accumulation 1, each epoch gives one optimizer update per training clip.

Feature cache format 2 runs the semantic encoder in FP32, matching inference, and binds each
entry to the source audio/transcript and extraction assets/configuration. Requesting caching
regenerates old or stale entries. FP16 overflow skips the affected update and reduces the
gradient scale without advancing the scheduler; non-finite unscaled training fails explicitly.

`TrainConfig.epoch_train_state=False` omits the optimizer/scheduler/RNG `train_state.pt` sidecar from periodic
epoch and step checkpoints, avoiding about 4x extra disk per checkpoint. Best, final, and interrupted checkpoints
still carry train state when `save_train_state=True`, so Continue run remains available from those files.

### Training reference conditioning

`TrainConfig.speaker_ref_mode` controls the CAMPPlus source (`self`, `other`, or deterministic `mixed`).
`TrainConfig.emo_ref_mode` independently controls the emotion-vector source:

- `self` uses the target clip and preserves the legacy training behavior.
- `other` uses a deterministic different clip from the same speaker. Training can fall back to its target when
  no other training clip exists; validation rejects a speaker with no training reference instead of falling back.
- `mixed` deterministically chooses self or other per item and epoch.
- `follow_speaker` uses exactly the clip selected by `speaker_ref_mode` for both CAMPPlus and emotion. This is the
  inference-aligned mode because one reference clip supplies both vectors at generation time.

`TrainConfig.val_reference_mode` controls both validation vectors. `self` maps to speaker/emotion `self/self`;
`other` maps to `other/follow_speaker`, so validation measures inference-like generalization from a different clip
of the same speaker. Checkpoint evaluation uses the same mapping for its validation split and deterministic training
subset. `CheckpointEvalConfig.reference_mode=""` inherits `val_reference_mode` from the adapter's
`train_config.json` and falls back to `self` for older adapters. Reports persist the resolved reference mode and state
the conditioning method in their Markdown summary.

Reference candidates for both training and validation are restricted to training records.
Older saved configurations without `val_split_mode` retain their historical record split
in checkpoint evaluation. Positive validation caps use a fixed shuffled subset.

### Training sampling and evaluation settings

The sampling fields are `sample_language="auto"`, `sample_seed=-1`, `sample_temperature=0.8`,
`sample_top_p=0.8`, `sample_top_k=30`, `sample_repetition_penalty=10.0`, `sample_num_beams=3`,
`sample_emo_alpha=0.65`, `sample_diffusion_steps=25`, `sample_inference_cfg_rate=0.7`,
`sample_max_text_tokens=60`, `sample_length_penalty=0.0`, `sample_max_mel_tokens=1500`, and
`sample_speaking_rate=1.0` (validated from 0.5 through 1.5). Their defaults mirror
Voice Generation. `auto` language resolves from `dataset_info.json`, then the first manifest row, then `EN`.
A seed of `-1` is resolved once when training starts and the resolved seed is reused for every epoch sample.
`indextts.training.sampling` must build all user-controlled inference values from `TrainConfig`; only the documented
`SAMPLE_FIXED_INFER_KWARGS` structural worker settings may be fixed in that module.

Automatic checkpoint evaluation is controlled by `eval_train_subset=48` (`0` disables the training subset),
`eval_strengths="1.0"` (comma-separated finite values from 0 through 4), and `eval_include_base=True`.
Training `status.json` additionally persists the resolved `sample_seed`, `val_reference_mode`, and calibrated
`recommended_speaking_rate` when available; every validation
event in `metrics.jsonl` carries `reference_mode`.

Completed and gracefully stopped training runs can write `loras/<name>/analysis/training_analysis.json`
and `training_analysis.md`. Measured evaluation adds `checkpoint_eval.json` and `checkpoint_eval.md` in the
same folder. `training_analysis` is derived only from `metrics.jsonl`; `checkpoint_eval` loads the base GPT,
reconstructs the saved validation split, evaluates the base model first, and then hot-swaps adapters. Status
records expose `analysis_path`, `evaluation_path`, and `recommended_checkpoint` when available.

After a complete or graceful-stop run, epoch samples are measured by the CPU-only
`indextts.training.speaking_rate` module. It strips pause tags, trims leading/trailing audio below 40 dB of peak,
and compares aggregate generated words/s with `manifest.jsonl` words/duration. The recommended speaking rate is
`round(clamp(dataset_words_per_second / generated_words_per_second, 0.5, 1.5), 3)`. Reports are atomically stored at
`loras/<name>/analysis/speaking_rate.json`; Voice Generation can load the report from an adapter folder, a normal
checkpoint, or a checkpoint below `best/`. A Checkpoint Grid can produce the same report with method `grid`.
When grid text comes from the dataset, calibration instead compares each generated sentence with its matched,
identically trimmed recording and stores method `grid_matched`.

Checkpoint evaluation workers run as
`python -m indextts.training.eval_worker --config <json> --state-dir <dir>`. Their state directory contains
atomic `status.json` and `progress.json`, plus `log.txt`; status moves through `initializing`, `evaluating`,
and `complete` or `failed`. The report itself always lands in the adapter's `analysis/` folder.

Listening-grid workers run as
`python -m indextts.training.grid_worker --config <json> --state-dir <grid-dir>`. Each grid is stored at
`outputs/grids/<grid-name>/` with atomic `grid.json`, readable `grid.md`, `status.json`, `progress.json`,
`log.txt`, one top-level WAV per cell, and reproducibility artifacts below `.cells/<cell>/`. Cell order is
checkpoint, strength, reference, then text. The generation engine is constructed once and adapters are hot
swapped between cells. `outputs/grids/` is never treated as an ordinary recent Voice Generation task.

## LoRA file contract (indextts/lora/io.py)
Single `.safetensors` file. Tensor keys: `<module_path>.lora_A.weight`, `<module_path>.lora_B.weight`,
`<module_path>.lora_magnitude` (DoRA only), plus optional full tensors `full.<module_path>.<param>` for fully
fine-tuned small modules (e.g. `spk_emb_proj`). Metadata (safetensors header `__metadata__`, all strings):
`format="indextts2_premium_lora"`, `version="1"`, `adapter_type="lora"|"dora"`, `rank`, `alpha`, `dropout`,
`target_modules` (JSON list), `base_model="IndexTeam/IndexTTS-2.5"`, `base_variant`, `trained_steps`, `epochs`,
`dataset_name`, `created_at`, `app_version`, `train_config` (JSON), `recommended_reference` (relative wav path or ""),
`sample_rate`. Loading must auto-detect rank/alpha/DoRA from metadata (and from tensor shapes as fallback).

## Presets contract (ui/presets_store.py)
`presets/system/*.json` are read-only defaults shipped with the app (never written by user actions);
`presets/user/*.json` are user presets. Each tab registers its components with a key; one universal preset
stores all tabs' values under `{"_meta": {...}, "values": {key: value}}`. Last used preset name is remembered in
`presets/user/.last_used_preset.txt`. Loading coerces/clamps values and skips unknown keys.

## Console + UI information policy
Everything the user might want to know is printed to the console AND shown in the UI: model load times and VRAM,
per-segment progress with speed (x realtime) and ETA, training step/epoch/loss/lr/it/s/ETA, download progress
with MB/s and ETA. No silent long operations.

## Testing policy
- `tests/` uses pytest; CPU-only tests must pass with `venv\Scripts\python.exe -m pytest tests -q`.
- GPU tests are marked `@pytest.mark.gpu` and are opt-in with
  `INDEXTTS_RUN_GPU_TESTS=1`; they are also skipped when CUDA is unavailable.
- Each task adds tests for its modules.

## Atomic status/progress files (added after a Windows race was observed)
- All worker status/progress JSON files (`status.json`, `progress.json`, `dataset_info.json`, manifests) are written through
  `indextts.utils.atomic_json.write_json_atomic` (unique temp file + `os.replace` with retries on WinError 5/32, direct
  overwrite as a last resort) and read through `read_json_retry` (retries on locked or partially written files).
  `indextts.training.dataset_manifest.atomic_write_json`, `indextts.runtime.progress.ProgressReporter._write_file`,
  `ui.common.write_json_atomic` and `ui.common.read_json` all delegate to it. A UI training run previously died at step 10
  with `PermissionError: [WinError 5]` when the 1 Hz UI poll held `status.json` open during the rename.
