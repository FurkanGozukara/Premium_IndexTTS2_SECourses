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
    vram_reserve_gb: float = 2.0           # VRAM to keep free (tier presets set 1 GB up to 16 GB cards, 2 GB above)
    vram_tier: str = "auto"                # "auto" | "6" | "8" | "10" | "12" | "16" | "24" | "32" | "custom"
    lora_path: str = ""                    # optional LoRA/DoRA safetensors to apply on the GPT
    lora_strength: float = 1.0
    lora_merge_into_base: bool = False      # fold a BF16 adapter into base weights for faster inference
    max_section_batch_size_hint: int = 8   # UI can read this to cap the section batch size slider for the tier
```
`RuntimeConfig.to_dict()/from_dict()` must round-trip JSON. `resolve_preset(tier: str, gpu_total_gb: float,
gpu_free_gb: float) -> RuntimeConfig` returns the preset for a tier (tiers: 6, 8, 10, 12, 16, 24, 32).
`auto_tier(gpu_total_gb)` picks the largest tier <= advertised VRAM + 0.5 GB (`AUTO_TIER_TOLERANCE_GB`):
31.5 GB and above is a 32 GB card, 9.5 GB and above a 10 GB card. `TIER_BUDGET_GB` is the whole-GPU peak a
tier may reach (6: 5, 8: 7, 10: 9, 12: 11, 16: 15, 24: 22, 32: 30 GiB), `tier_reserve_gb(tier)` the memory
it leaves free (1 GB up to 16 GB, 2 GB above) and the value `resolve_preset` writes into `vram_reserve_gb`.
`fit_tier_to_free_vram(tier, free_gb)` shrinks a tier until its budget fits into the free memory; the trainer
uses it for the per-epoch sample process that runs beside the training model. `generation_preset(tier)` and
`resolve_training_preset(tier)` return the decoding and `TrainConfig` values a tier selects.

## GPU VRAM tier presets (ui/gpu_tier_presets.py, presets/system)
The read-only system presets are one per tier, named `6 GB GPU` ... `32 GB GPU`; `tier_registry_overrides(tier)`
flattens the three tables onto registry keys and `PresetStore.ensure_system_presets()` writes them (and removes
the retired `default`, `quality`, `fast`, `low_vram_8gb` files). Every tier keeps the BF16 GPT, sampling,
40 diffusion steps and CFM temperature 0.9; a smaller card pays with speed first (block streaming, on-demand
or CPU auxiliary models, a shorter CFM cache) and only then with fewer beams. Voice, device and optional-loader
runtime keys (`device`, `lora_*`, `decoder_adapter*`, `use_qwen_emo`, `use_deepspeed`, `attention_backend`,
`use_accel`, `torch_compile_s2mel`, `use_cuda_kernel_bigvgan`, `gpt_dtype`) stay at the registry defaults.

| tier | GPT | swap/ring | semantic | CAMPPlus | Qwen | CFM | DiT BF16 | beams | steps | low-memory mode | training swap/ring | sample gate |
|------|-----|-----------|----------|----------|------|-----|----------|-------|-------|-----------------|--------------------|-------------|
| 32 | bf16 | 0/2 | gpu | gpu | gpu | 8192 | no | 4 | 50 | off | 0/2 | 4.5 GB |
| 24 | bf16 | 0/2 | gpu | gpu | gpu | 8192 | no | 4 | 50 | off | 0/2 | 4.5 GB |
| 16 | bf16 | 0/2 | gpu | gpu | gpu | 8192 | no | 4 | 40 | off | 0/2 | 4.5 GB |
| 12 | bf16 | 0/2 | gpu | gpu | gpu | 8192 | no | 4 | 40 | off | 0/2 | 4.5 GB |
| 10 | bf16 | 0/2 | on_demand | gpu | on_demand | 6144 | no | 4 | 40 | off | 0/2 | 4.5 GB |
| 8 | bf16 | 0/2 | cpu | gpu | on_demand | 4096 | no | 4 | 40 | on | 0/2 | 4.5 GB |
| 6 | bf16 | 22/1 | cpu | cpu | cpu | 2048 | yes | 2 | 40 | on | 22/1 | 4.5 GB |

The budgets assume the worker owns the card. `ProcessManager.start` (ui/common.py) therefore calls
`LAZY_ENGINE.release_for_worker(kind)` for every kind in `GPU_WORKER_KINDS` (training, dataset_prep,
dataset_cache, dataset_curation, checkpoint_eval, grid_generation, vram_benchmark, generation,
batch_generation) before spawning the child, which unloads models an earlier in-process generation left
resident (a 32 GB run keeps about 7 GB loaded after one generation; an 8 GB card would otherwise start
training with 4.6 GB already taken). `LazyEngine.in_use()` wraps the in-process generation and batch
paths so a running generation is never unloaded underneath; the release then logs that it kept the
models. `release_engine=` overrides the kind policy for a single start. Measured from the browser on the
32 GB tier (RTX 5090, one in-process generation followed by a one-epoch training run): the generation
peaked at 7.15 GiB of whole-GPU use and left 7.1 GiB resident; without the release the training pipeline
then peaked at 17.71 GiB (resident models + trainer + epoch sample), with it at 11.40 GiB.

Section batch size is 1, text tokens per segment 60 and CFM temperature 0.9 on every tier; `_HINTS` keeps
the stress maxima the CLI benchmark and the section-batch hint use. Beams stop at four because the decoding
sweeps of three real training runs scored five beams worse than three (word error +0.1 to +0.8 points,
identity flat or lower). Training keeps rank 128, alpha 128, batch size 1 and the BF16 base on every tier,
and its sample, speech-benchmark and decoding-sweep settings stay at the measured defaults (3 beams, 25
steps) so runs stay comparable across releases; `TrainConfig.vram_tier` records the tier, and
`indextts.training.sampling.resolve_sample_runtime(config, share_gpu=...)` resolves `sample_runtime_tier`
"auto" to it (shrinking to the free memory when the sample renders beside the training model, keeping the
tier for the speech comparison, decoder test and decoding sweep after the model is released).

Calibration is the whole-GPU `nvidia-smi memory.used` peak (every process and CUDA context included) sampled
every 200 ms by `tools/gpu_tier_calibration.py` while it runs the real workers with a tier's preset values:
the generation worker (base voice, a LoRA / DoRA with its decoder adapter, emotion-text mode), the dataset
preparation worker, the voice audit, the feature cache and a one-epoch training run with its sample,
checkpoint evaluation, speech comparison and decoder adaptation phases. Measured on idle GPU 0 (RTX 5090,
31.8 GiB) on 2026-09-09 with the tables above; generation of the 29 s calibration passage in a separate
worker process, so two CUDA contexts (about 1 GB) are included and an in-process generation uses about
0.5 GB less:

| tier | budget | base voice | LoRA + decoder adapter | emotion text | RTF base / LoRA | notes |
|------|-------:|-----------:|-----------------------:|-------------:|----------------:|-------|
| 32 | 30 | 7.89 | 8.68 | 7.88 | 0.58 / 1.04 | |
| 24 | 22 | 7.89 | 8.68 | 7.88 | 0.56 / 1.05 | same runtime as 32 |
| 16 | 15 | 7.89 | 8.68 | 7.88 | 0.56 / 1.01 | same runtime as 32 |
| 12 | 11 | 7.89 | 8.68 | 7.88 | 0.63 / 1.01 | same runtime as 32 |
| 10 | 9 | 6.62 | 6.90 | | 0.72 / 1.07 | peak while the on-demand semantic encoder is on the GPU |
| 8 | 7 | 4.60 | 5.64 | 4.80 | 0.61 / 1.07 | streaming 12 blocks with the encoder on demand measured 5.54 / 6.48 at RTF 1.08 / 1.20 |
| 6 | 5 | 3.62 | 4.34 | 3.58 | 1.92 / 2.19 | a passage twice as long peaked at 3.79; **Prevent VRAM accumulation** raised it to 4.82 |

The training pipeline was measured the same way on a 98-clip, two-recording dataset (one epoch, four
speech-benchmark prompts, one decoder epoch, no decoding sweep). Whole-GPU peaks per phase, in GiB; the
"sample beside training" column is the epoch sample rendered while the training model is still loaded,
which the free-VRAM gate skips on the real card when the two do not fit:

| tier (training swap/ring) | training steps | sample beside training | checkpoint evaluation | speech comparison | decoder adaptation |
|------|---:|---:|---:|---:|---:|
| 32, 24, 16, 12 (0/2) | 3.99 | 10.88 | 4.08 | 8.5 to 9.1 | 7.97 |
| 10 (8/2) | 3.7 (5.8 for 0.4 s at the first validation) | 10.07 | 4.11 | 7.16 | 7.99 |
| 8 (12/2) | 3.59 | 8.04 | 4.11 | 6.64 | 7.99 |
| 6 (22/1) | 3.15 | 6.45 | 4.11 | 7.55 | 7.99 |

Training keeps the same speed with or without gradient checkpointing on this model (3.4 versus 3.5 it/s,
2.8 GB allocated either way), so checkpointing stays on for every tier. Dataset preparation of the two
recordings peaked at 9.03 GB with the former Whisper alignment path (encoder attention maps), the audit at
6.70 GB with both Whisper models on the GPU, and the feature cache at 8.90 GB with four clips per batch;
the fixes described in the release notes (lean encoder, free-VRAM batch size, CPU second opinion) exist for
those three stages, and the emulated small-card runs below record what remains. The lean encoder
(`whisper_asr.install_lean_encoder`) was checked on a 150 s excerpt: same 348 words with identical times,
peak allocated 2.09 GB instead of 4.78 GB, 8.9 s instead of 23.0 s.

Whole-GPU `memory.used` on a large card overstates what a small card needs, because the caching allocator
keeps freed blocks instead of reclaiming them (the 6 GB speech comparison showed 7.5 GB used while the
engine's allocated peak stayed at 2.6 GB). `INDEXTTS_VRAM_EMULATE_GB=<tier size>` therefore makes every
process that imports `indextts.runtime` behave like a card of that size: `apply_emulated_vram_cap()` caps
its allocator at the tier budget (`torch.cuda.set_per_process_memory_fraction`), `gpu_total_gb()` and
`list_gpus()` report the tier size (so `auto_tier` detects that tier), and `gpu_free_gb()` returns the
size minus a driver reserve, an application-context allowance and the process's own reserved memory (so
the epoch-sample gate and `fit_tier_to_free_vram` decide as they would on the real card). An allocation
beyond the budget raises `torch.OutOfMemoryError` instead of spilling into shared system memory, which is
the definitive "does not fit". `tools/gpu_tier_calibration.py --emulate` sets it per tier for every worker
(dataset preparation, audit and feature cache emulate the smallest requested tier); the app never sets it.

Emulated small-card runs (whole-GPU peaks in GiB, including the calibration process's own context, which
stands in for the app's; every stage completed under its cap unless noted):

| stage (emulated card) | peak | notes |
|------|---:|---|
| dataset preparation, two recordings (6 GB) | 3.67 | lean Whisper encoder; same 104 segments as the unemulated run |
| voice audit (6 GB) | 4.18 | second-opinion Whisper on the CPU; same 98 retained clips |
| feature cache (6 GB) | 5.25 | one clip per batch |
| training 8 GB: steps / sample beside training / checkpoint evaluation / speech comparison / decoder | 3.59 / 6.88 / 4.11 / 8.83 / 5.83 | decoder adaptation completed with the semantic encoder offloaded; the sample ran at exactly the old 4.0 GB gate, which is why the gate is now 4.5 GB |
| training 10 GB: steps / sample / evaluation / speech / decoder | 3.74 / 6.96 / 4.11 / 7.16 / 5.83 | sample rendered with the 6 GB tier |
| inference 6 GB: base / LoRA + decoder adapter / emotion text | 3.62 / 4.34 / 3.58 | RTF 1.9 to 2.2 |
| inference 8 GB: base / LoRA + decoder adapter / emotion text | 4.60 / 5.64 / 4.80 | RTF 0.6 to 1.1 |
| inference 10 GB: base / LoRA + decoder adapter | 5.95 / 6.90 | RTF 0.6 to 1.1 |

The first emulated 6 GB training run reached the decoder phase with the FP32 semantic encoder still resident
and ran out of memory there even at the lowest frame limit; `DecoderAdapterTrainer._prepare_prompt_features`
now caches every prompt clip's features before training and moves the encoder to the CPU, after which the
decoder phase of the 8 GB and 10 GB emulations peaked at 5.83 GB. With that change the emulated 6 GB
training pipeline completes end to end: steps 3.8 GB (block streaming), the epoch sample skipped by the
4.5 GB gate (2.9 GB were free), checkpoint evaluation 4.1 GB, the speech comparison 7.2 GB and the decoder
adaptation 5.8 GB of whole-GPU use. The post-training phases run in child processes while the trainer keeps
its CUDA context (about 0.7 GB) beside the application's, so their whole-GPU figure exceeds the 5 GB target
of the 6 GB tier even though every process stayed under its 5 GB cap; what a real 6 GB card needs is the
sum of the two idle contexts and the child's allocated peak (about 5.4 GB), which still fits the card.
The emulated 8 GB re-run with the final table (resident training, 4.5 GB sample gate) measured steps
3.99 GB, sample skipped, checkpoint evaluation 4.08 GB, speech comparison 6.62 GB and decoder adaptation
5.80 GB, all within the 7 GB budget.

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
`presets/system/*.json` are the read-only GPU VRAM tier presets, regenerated from the registry at every start
(never written by user actions); `presets/user/*.json` are user presets. Each tab registers its components
with a key; one universal preset stores all tabs' values under `{"_meta": {...}, "values": {key: value}}`.
The last used preset name is remembered in `presets/user/.last_used_preset.txt`; when it names a preset that
no longer exists (a fresh install, or the retired system presets) `PresetStore.get_last_used()` returns
`default_preset_name()`, the tier preset detected for the GPU (`detect_tier` is injectable for tests).
Reset and the fallback after deleting a user preset load that preset too. Loading coerces/clamps values and
skips unknown keys. The runtime saved by **Apply runtime** (`presets/user/.last_runtime.json`) is overlaid
at start only on a system preset of the same tier (or when it was applied as `custom`), never on a user preset.

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
