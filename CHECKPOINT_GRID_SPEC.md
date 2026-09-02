# Checkpoint Grid + Generalization Analysis — implementation spec (v5.1)

This document is the contract for the "Checkpoint Grid" feature. Read `ARCHITECTURE_NOTES.md` first;
every rule there (Windows + Linux, no new pinned dependencies, atomic JSON, worker/status contracts,
console + UI information policy, tests under `tests/`) still applies.

## 1. Why this feature exists (background from the 2026-09-02 diagnosis)

A user trained `loras/SECourses_Furkan_EN_DoRA_r32` for 40 epochs on 31 minutes of audio. The training
log (`outputs/training_runs/furkan_dora_r32/metrics.jsonl`) shows validation loss bottoming out at
epoch 10 and rising afterwards while training loss kept falling:

| checkpoint | validation loss | validation next-token accuracy | training-subset accuracy |
|---|---|---|---|
| base model, no adapter | 6.58 | 4.3% | 4.5% |
| epoch 10 (`best/`) | 5.36 | 9.9% | 14.5% |
| epoch 30 | 5.68 | 8.2% | 22.7% |
| epoch 40 (final file) | 5.70 | 8.0% | 23.4% |

The user listened only to the epoch-40 sample and judged the voice wrong. Users cannot read this table.
The app must therefore (a) tell users in plain language which checkpoint generalizes best and where
overfitting starts, (b) save that analysis automatically after every training run, and (c) let users
generate a listening grid (same text, same reference, same seed) across checkpoints and strengths so
they can hear the difference themselves. A manual version of that grid lives in
`outputs/lora_diagnosis/` (see its README.md and diag_report.json) and is the reference for the
expected output shape.

Facts about existing artifacts you will consume:

- `metrics.jsonl` (written by `indextts/training/trainer.py`): one JSON object per optimizer step
  (`step, epoch, loss, avg_loss, moving_avg_loss, mel_accuracy, lr, grad_norm, it_s, elapsed_s, eta_s,
  vram_used_gb, vram_peak_gb`) plus validation events
  `{"event": "validation", "step", "epoch", "val_loss", "val_mel_accuracy"}`. Validation runs every
  `val_every_steps` and at the end of every epoch; the end-of-epoch event is the last validation event
  with that epoch number. `indextts.training.charts.load_metrics(state_dir)` already parses the file.
- The training state dir is the adapter folder when launched from the UI (`loras/<name>/status.json`,
  `metrics.jsonl`, `log.txt`, `train_config.json`), but CLI runs may use another `--state-dir`.
  Analysis functions must accept an explicit state dir and fall back to the adapter dir.
- Checkpoint files in an adapter folder: `<name>.safetensors` (final), `<name>_epoch_NNN.safetensors`
  (pruned to `keep_last_n`), `<name>_step_NNNNNN.safetensors`, `<name>_interrupted.safetensors`,
  `best/<name>.safetensors`. Temporary sample adapters start with a dot
  (`.<name>_sample_epoch_NNN.safetensors`) and must be ignored. Every file carries safetensors
  metadata (`indextts.lora.io.inspect_lora` / `load_lora`): `trained_steps`, `epochs`, `train_config`
  (JSON with `dataset_dir`, `val_fraction`, `seed`, `base_variant`, `base_dtype`, ...), and
  `recommended_reference`.
- `LoraTrainDataset(dataset_dir, split="val", val_fraction, seed, speaker_ref_mode="self")` reproduces
  the exact validation split used during training when given the training config's `val_fraction`
  and `seed`. `gpt_train_step_loss` (indextts/training/model_forward.py) returns
  `(total, {"loss", "mel_loss", "text_loss", "mel_accuracy"})`.
- Loading the GPT for evaluation: see `build_training_model` in trainer.py (UnifiedVoice with
  `spk_cond_mode="campplus"`, `load_gpt_checkpoint(model, path, device="cpu", dtype=..., strict=False)`).
  Hot swapping adapters: `indextts.lora.apply.apply_lora(model, path, strength)` and `remove_lora(model)`;
  after `apply_lora` the adapter tensors must be moved to the model device (see
  `IndexTTS2.set_lora` in `indextts/infer_v2_5.py` for the loop that moves `lora_A`, `lora_B`,
  `lora_magnitude`). Add a small helper for that in `indextts/lora/apply.py`
  (`move_adapters_to_device(model, device)`) and use it in both places.
- The reference implementation of the evaluation is the throwaway script used for the diagnosis; its
  logic to copy: build model → for each checkpoint `remove_lora`, `apply_lora(strength)`, move adapter
  tensors, `eval()` → iterate `DataLoader(dataset, batch_size=4, collate_fn=collate)` under
  `torch.autocast("cuda", dtype=torch.bfloat16)` (no autocast on CPU) with `torch.no_grad()` →
  average `loss`, `mel_loss`, `text_loss`, `mel_accuracy`. The base model is evaluated with no adapter
  at all (fresh weights, before any `apply_lora`, or after `remove_lora` on a model whose full-module
  tensors were never overwritten — note that `apply_lora` overwrites `spk_emb_proj` from the file and
  `remove_lora` restores the values that were present before the first `apply_lora`, so evaluate the base
  model first).
- Generation of one audio file through the normal engine: `webui_generation_runner.create_tts(runtime)`
  once, then `run_generation_request(request, tts)` per cell. `indextts/training/sampling.py` builds
  exactly such a request for the training samples (copy its request shape, including `task_layout`,
  `metadata_path`, `infer_kwargs`); a fixed seed goes in `request["seed"]`. Switching `lora_path` /
  `lora_strength` between requests hot-swaps the adapter without reloading base weights; `lora_path=""`
  means the base model. Every cell output is a 22050 Hz WAV written to `task_layout.final_wav_path`.

## 2. Deliverables

### 2.1 `indextts/training/analysis.py` — CPU-only training-run analysis (no torch import needed)

```python
@dataclass
class EpochSummary:
    epoch: int
    steps: int
    train_loss: float | None         # mean of step losses in the epoch
    train_accuracy: float | None     # mean mel_accuracy of the epoch
    val_loss: float | None           # end-of-epoch validation loss (last validation event of the epoch)
    val_accuracy: float | None
    gap: float | None                # train_loss - val_loss (more negative = more memorization)
    lr: float | None
    phase: str                       # "improving" | "best" | "plateau" | "overfitting" | "unknown"

@dataclass
class TrainingAnalysis:
    adapter_dir: str
    state_dir: str
    status: str                      # "no_validation" | "still_improving" | "best_found" | "plateau" | "empty"
    epochs: list[EpochSummary]
    best_epoch: int | None
    best_step: int | None
    best_val_loss: float | None
    final_epoch: int | None
    final_val_loss: float | None
    overfit_start_epoch: int | None  # first epoch of the sustained rise after the best epoch
    tolerance: float                 # relative tolerance used (default 0.01)
    recommended_checkpoint: str      # absolute path or ""
    recommended_label: str           # e.g. "best/ (epoch 10)" or "final (epoch 15)"
    checkpoints: list[dict]          # discovered files: path, label, kind, epoch, steps, val_loss (from metrics), phase
    summary_markdown: str            # plain-language verdict, see below
    generated_at: str
    def to_dict(self) -> dict: ...

def analyze_training_run(adapter_dir, state_dir=None, *, tolerance=0.01) -> TrainingAnalysis
def write_training_analysis(analysis) -> Path       # loras/<name>/analysis/training_analysis.json + .md, atomic
def load_training_analysis(adapter_dir) -> TrainingAnalysis | None
def analysis_epoch_frame(analysis) -> pandas.DataFrame   # columns step(=epoch), value, series for gr.LinePlot
```

Phase rules (make them a pure function that is unit-tested):

- Use end-of-epoch validation values. If there are no validation events at all → `status="no_validation"`,
  every phase `"unknown"`, summary says validation was disabled and the app cannot judge overfitting; the
  recommended checkpoint is the final file.
- `best_epoch` = epoch with the minimum end-of-epoch validation loss (ties → earliest).
- An epoch after the best is "overfitting" when its validation loss is ≥ `best * (1 + tolerance)` and no
  later epoch drops back below that threshold; the first such epoch is `overfit_start_epoch`. Epochs after
  the best that stay within the tolerance are "plateau". Epochs before the best are "improving".
- `status`: `"still_improving"` when the best epoch is the last epoch (the summary must say training could
  run longer / the final file is the best one), `"best_found"` when an overfit start exists, `"plateau"`
  otherwise.
- `recommended_checkpoint`: when `best/<name>.safetensors` exists and its metadata `epochs` equals the
  best epoch (or best epoch < final epoch) → that file; else the epoch checkpoint file whose metadata
  `epochs` equals the best epoch if it still exists; else the final file. Always absolute paths.
- `summary_markdown` must read like this (numbers formatted with 2 decimals, percentages with 1 decimal):

  > **Best generalization: epoch 10** (validation loss 5.36, 9.9% next-token accuracy on unseen
  > sentences).
  > **Overfitting starts at epoch 12.** From there validation loss rises while training loss keeps
  > falling, which means the adapter memorizes the training clips instead of learning the voice.
  > The final file (epoch 40) is overfitted: validation loss 5.70 (+6.3% vs best) while training-set
  > accuracy climbed from 14.5% to 23.4%.
  > **Recommended checkpoint:** `best/SECourses_Furkan_EN_DoRA_r32.safetensors` (epoch 10).
  > Tip: stop training around epoch 10–12 next time, or keep every epoch checkpoint (Keep last N = 0)
  > and compare them in the Checkpoint Grid tab.

  Include a one-line legend the UI shows next to the chart: "Validation loss measures how well the adapter
  predicts sentences it never saw during training (lower is better). Training loss measures the clips
  it trains on. When training loss keeps falling but validation loss rises, the adapter is memorizing
  (overfitting)."

- `analysis_epoch_frame` returns rows for series `"train loss"`, `"validation (improving)"`,
  `"validation (overfitting)"` with `step` = epoch so the training tab LinePlot can colour the overfit part
  red (the `plateau` epochs go to the improving series). Use the existing `empty_series_frame` pattern.

### 2.2 `indextts/training/checkpoint_eval.py` — measured checkpoint comparison (GPU or CPU)

```python
@dataclass
class CheckpointEvalConfig:
    adapter_dir: str
    dataset_dir: str = ""            # default: train_config.json in the adapter dir, else adapter metadata train_config
    checkpoints: list[str] = []      # default: every non-hidden *.safetensors in adapter_dir and adapter_dir/best
    include_base: bool = True
    strengths: list[float] = [1.0]   # extra strengths evaluate the same file again (label "epoch 40 @0.5")
    train_subset: int = 48           # deterministic first N training records (seeded), 0 disables
    batch_size: int = 4
    max_batches: int = 0             # 0 = all validation batches
    device: str = "cuda:0"           # falls back to cpu (fp32) when CUDA is unavailable
    base_variant/base_dtype/model_dir/model_config/attention_backend: same semantics as TrainConfig; default from train_config.json
    val_fraction/seed: default from train_config.json so the split matches training

@dataclass
class CheckpointEvalRow: label, path, kind ("base"|"best"|"epoch"|"final"|"step"|"interrupted"), epoch, steps, strength,
    val_loss, val_mel_loss, val_text_loss, val_accuracy, train_loss, train_accuracy, gap, phase, elapsed_s
@dataclass
class CheckpointEvalReport: adapter_dir, dataset_dir, val_items, train_subset_items, rows, best_label, best_path,
    recommended_checkpoint, summary_markdown, device, generated_at, elapsed_s
    def to_dict(self)

def evaluate_checkpoints(config, reporter: ProgressReporter | None = None, cancel_callback=None) -> CheckpointEvalReport
def write_checkpoint_eval(report, adapter_dir) -> Path      # loras/<name>/analysis/checkpoint_eval.json + .md, atomic
def load_checkpoint_eval(adapter_dir) -> CheckpointEvalReport | None
```

- Progress: `reporter.set_stage("load model")`, then one `reporter.update(i, total=n, desc=label)` per
  checkpoint; write `progress.json` in the state dir like the other workers; print a console line per
  checkpoint (`>> epoch 10 [best]: val 5.363 acc 9.9% | train 4.576 acc 14.5% (2.1s)`).
- Phase assignment for measured rows reuses the analysis phase rules but on the measured validation losses,
  ordered by epoch (base row phase "base"; strength variants inherit the epoch and get phase "variant").
- `summary_markdown` in the same plain language as 2.1 but based on measured numbers, and it must mention
  the base model row ("Without any adapter the base model scores 6.58; every checkpoint below that number
  learned something from your data.").
- Worker entry `indextts/training/eval_worker.py`: `python -m indextts.training.eval_worker --config <json>
  --state-dir <dir>`; writes `status.json` (`phase`: initializing → evaluating → complete | failed,
  `message`, `completed`, `total`, `elapsed_s`, `updated_at`), `progress.json`, `log.txt`, and the report
  files; exit code 0/1 like `train_worker.py`.
- `tools/evaluate_checkpoints.py` CLI wrapper (mirrors `tools/train_lora.py`).

### 2.3 Trainer integration (`indextts/training/trainer.py`, `train_config.py`)

- New `TrainConfig` fields (add to the UI in the Saving & Resume accordion; keep the UI ↔ TrainConfig
  field check in `build_training_tab` passing):
  - `auto_analyze: bool = True` — after a run ends (complete or stopped, not failed) call
    `analyze_training_run` and `write_training_analysis`; log the summary lines to `log.txt`; add
    `analysis_path` and `recommended_checkpoint` to `status.json`. Never let a failure here fail the run.
  - `auto_evaluate_checkpoints: bool = True` — after the analysis, free the training model
    (`del`, `gc.collect()`, `torch.cuda.empty_cache()`) and run the evaluation through the eval worker
    subprocess (same pattern as `sampling.generate_training_sample`: subprocess, timeout
    `eval_timeout_s: float = 900`, skipped with a logged message when free VRAM is below
    `sample_min_free_vram_gb`). Write `status.json` phase `"evaluating"` with progress while it runs so
    the dashboard shows it, then restore phase `"complete"`/`"stopped"`.
  - `early_stop_patience: int = 0` — 0 disables. When > 0, stop the run (same path as `stop.flag`, but the
    saved file is the normal final file and status message says why) once `patience` consecutive
    validations have not improved `best_val_loss` by at least `early_stop_min_delta: float = 0.0`.
- The status/log messages must be explicit, e.g. `>> Best validation loss 5.3648 at epoch 10; the final
  epoch 40 adapter is overfitted (validation 5.7064). Recommended: best/SECourses_Furkan_EN_DoRA_r32.safetensors`.

### 2.4 `indextts/training/grid.py` — listening grid generator

```python
@dataclass
class GridCheckpoint: label: str; path: str        # path "" = base model (no adapter)
@dataclass
class GridConfig:
    adapter_dir: str
    checkpoints: list[GridCheckpoint]
    strengths: list[float] = [1.0]                  # ignored for the base model (one cell)
    references: list[str] = []                      # absolute wav paths, at least one
    texts: list[str] = []                           # at least one
    language: str = "EN"
    seed: int = -1                                  # -1: draw one random seed at start, then fixed for every cell
    same_seed_for_all_cells: bool = True
    output_root: str = "outputs/grids"
    grid_name: str = ""                             # default "<adapter>_<YYYYmmdd_HHMMSS>"
    runtime: dict = {}                              # RuntimeConfig.to_dict() + model_dir/cfg_path/use_qwen_emo, from the UI
    infer_kwargs: dict = {}                         # overrides merged over the sampling.py defaults
    include_verdicts: bool = True                   # attach phases from analysis/checkpoint_eval to rows

def build_grid_cells(config) -> list[GridCell]      # deterministic order: checkpoint → strength → reference → text
def run_grid(config, reporter=None, cancel_callback=None) -> GridResult
def load_grid(grid_dir) -> GridResult | None
def list_grids(output_root) -> list[GridSummary]    # newest first
```

- Output folder `outputs/grids/<grid_name>/`: `<cell>.wav` per cell, `.cells/<cell>/` task folders (request,
  metadata, progress, generation log), `grid.json` (config, seed, cells with label/path/checkpoint/strength/
  reference/text/audio_seconds/generation stats/verdict), `grid.md` (a markdown table users can read),
  `status.json`, `progress.json`, `log.txt`. `outputs/grids` must be excluded from `recent_outputs` and
  `latest_output_task` scans (they skip `_`/`.` prefixed folders; treat `grids` explicitly).
- Cell names: `{checkpoint_label}__s{strength:g}__ref{ri}__t{ti}.wav` where checkpoint labels are derived from
  metadata: `base`, `best_ep10`, `epoch_030`, `final_ep40`, `interrupted_ep07`, `step_000500`.
- The engine is created once with `create_tts(runtime)` and reused for every cell; adapters are swapped by
  passing `lora_path`/`lora_strength` in each request. Cancel: `cancel_callback` checked between cells.
- Progress: `ProgressReporter("cells", total=n, progress_file=...)`, `desc` = cell label, `extra` with audio
  seconds; console line per cell.
- Worker entry `indextts/training/grid_worker.py` (`--config <json> --state-dir <dir>`) with the same status
  contract as the other workers; `tools/generate_grid.py` CLI wrapper.

### 2.5 UI — new tab "Checkpoint Grid" (`ui/grid_tab.py`), id `checkpoint-grid`

Insert it after "LoRA / DoRA Training" and before "Models & Performance" in `ui/app.py`
(`build_grid_tab(options, registry, load_hook=demo.load)`, `bind_grid_events(grid, training, generation,
models, main_tabs)`), add it to `demo.ui_tabs` as `"Checkpoint Grid"`, and fix the CSS rule in
`ui/common.py` that hides main-strip tab buttons beyond the sixth (`:nth-child(n+7)` → `n+8`, comment
updated). Register grid controls in the preset registry under `grid.*` keys (they take part in universal
presets like every other tab; keep registry keys unique).

Layout, top to bottom:

1. **Adapter row**: dropdown of adapter folders under `loras/` (folder label + adapter type/rank/steps),
   Refresh button, Markdown with dataset name, epochs, steps, recommended reference, analysis availability.
   Selecting the folder fills everything below. When the training tab finishes a run or the user presses the
   new training-tab button "Compare checkpoints in grid", the tab is selected with that adapter.
2. **Generalization panel** (Accordion "Which checkpoint generalizes best?", open):
   - `gr.LinePlot` from `analysis_epoch_frame` (x epoch, series train loss / validation (improving) /
     validation (overfitting) with colours grey / green `#1ca881` / red `#df345b`), height 300.
   - Markdown: `summary_markdown` from `checkpoint_eval.json` when present, else from
     `training_analysis.json`, else "Run 'Analyze training log' or train with validation enabled." plus the
     legend line.
   - `gr.Dataframe` "Checkpoints" with columns
     `Checkpoint | Epoch | Validation loss (lower = better) | Unseen-text accuracy | Training-text accuracy |
     Verdict | Path`. Verdict strings are user-facing: `Best generalization`, `Improving`, `Plateau`,
     `Overfitting (memorizes training clips)`, `Base model (no adapter)`, `Not measured`. Rows come from the
     measured report when present, otherwise from the metrics-based analysis (accuracy columns "–").
   - Buttons: "Analyze training log" (instant, CPU), "Evaluate checkpoints now" (runs
     `eval_worker` via `PROCESS_MANAGER` kind `checkpoint_eval`, progress panel + log tail polled with a
     `gr.Timer`, same wiring style as the training tab), "Use recommended checkpoint in Voice Generation"
     (sets `runtime.lora_path` in the generation tab like `use_adapter` in training_tab.py and switches tab).
3. **Grid setup**:
   - `gr.CheckboxGroup` of checkpoints (labels like `best (epoch 10) — Best generalization`,
     `epoch 30 — Overfitting`, `final (epoch 40)`, `Base model (no adapter)`), default selection: base +
     recommended + final (+ every remaining epoch file when ≤ 6 files).
   - Strengths textbox (comma separated, default `1.0`; validate floats in 0–4), Textbox "Texts (one per
     line)" with two default sentences (the training sample text of the adapter when known plus one neutral
     sentence), Textbox "Reference audio paths (one per line)" plus buttons "Use adapter reference",
     "Add dataset reference candidates", and a `gr.Audio(type="filepath")` uploader whose value is appended
     when set; Number "Seed" (-1 random once per grid), Language dropdown (same choices as generation tab).
   - Accordion "Sampling": temperature, top-p, top-k, repetition penalty, diffusion steps, CFG rate, max text
     tokens per segment, emotion weight (alpha) — defaults equal to `GENERATION_DEFAULTS`.
   - Runtime comes from the current Models & Performance values via `runtime_config_from_values` exactly like
     the generation tab (no adapter path inside; the grid sets it per cell). Show a Markdown line with the
     resolved device/tier and the number of cells + a rough time estimate (cells × 9 s at 32 GB tier).
   - Buttons: "Generate grid" (primary, emerald action button), "Cancel" (confirm dialog; update the confirm-dialog
     count in `tests/test_ui_build.py`), "Open grid folder".
   - Progress panel (`progress_panel_html`), status Markdown, log tail Textbox (`log-tail` class), polled by a
     `gr.Timer` while the `grid_generation` job runs; page reload re-attaches to a running grid job (same
     adopt pattern as training: newest `outputs/grids/*/status.json` that is not terminal).
4. **Results**: a dropdown "Saved grids" (newest first, from `list_grids`) so any earlier grid can be reopened,
   then the grid itself rendered with `@gr.render` (like the candidate list in `ui/generation_tab.py`): one
   Markdown header per checkpoint × strength row (label + verdict badge + validation loss when known), and
   under it one `gr.Audio(type="filepath", buttons=["download"])` per reference × text cell with the cell
   label (`ref 1 · text 2`) and the text as `info`. A `gr.Dataframe` below lists every cell (`Row | Reference |
   Text | Seconds | File`). Users must be able to compare rows top-to-bottom without scrolling sideways; put
   cells of one row in a `gr.Row` when ≤ 4 cells, otherwise wrap.

Training-tab additions (`ui/training_tab.py`):

- After the checkpoints table add a Markdown "Generalization summary" fed by `training_analysis.md` /
  `checkpoint_eval` for the displayed run (poll output), and the LinePlot from `analysis_epoch_frame` (this is
  the same chart as in the grid tab; reuse one helper).
- The checkpoints Dataframe gets the columns `Epoch | Validation loss | Verdict` filled from the analysis.
- New buttons "Compare checkpoints in grid" (switch to the grid tab with this adapter selected) and change
  "Use this LoRA in Voice Generation" to prefer `recommended_checkpoint` from the analysis when it exists
  (fall back to `last_checkpoint`); rename its label to "Use recommended checkpoint in Voice Generation".
- Saving & Resume accordion: checkboxes for `auto_analyze`, `auto_evaluate_checkpoints`, number
  `eval_timeout_s`, and in Optimization: `early_stop_patience`, `early_stop_min_delta` with clear `info`
  texts. Update the "Keep last N" info: "0 keeps every epoch checkpoint, which is what the Checkpoint Grid
  needs to compare epochs".

Help tab (`ui/help_tab.py`): add a section "Which checkpoint should I use?" explaining validation loss,
overfitting, the recommended checkpoint, and the Checkpoint Grid workflow in the same plain language.
README.md: add a bullet list under the V5 section describing the Checkpoint Grid tab, the automatic
generalization analysis, early stopping, and the auto-saved `analysis/` folder.

### 2.6 Tests (CPU only, `venv\Scripts\python.exe -m pytest tests -q` must pass)

- `tests/test_training_analysis.py`: synthetic metrics (best in the middle → `best_found` with the right
  `overfit_start_epoch`; monotonic decrease → `still_improving`; no validation events → `no_validation`;
  plateau within tolerance → `plateau`), `summary_markdown` contains the epoch numbers, `analysis_epoch_frame`
  series names, recommended checkpoint resolution with fake safetensors files created through
  `indextts.lora.io.save_lora` (see `tests/test_lora_io.py` for how tests build tiny adapters) or with a
  monkeypatched `inspect_lora`.
- `tests/test_checkpoint_eval.py`: run `evaluate_checkpoints` on CPU with a tiny `UnifiedVoice`
  (`tests/test_train_forward.py::_tiny_voice`) — monkeypatch the model builder, a two-item fake cached dataset
  (see `tests/test_trainer_cpu.py` fixtures) and tiny adapters saved with `save_lora`; assert rows, phases,
  the base row, and that the JSON/MD files are written.
- `tests/test_grid.py`: cell enumeration order/names, seed handling, `run_grid` with a fake engine
  (monkeypatch `create_tts` and `run_generation_request` to write a small WAV) → `grid.json`/`grid.md` and
  files exist; `list_grids`/`load_grid` round trip; `recent_outputs` ignores `outputs/grids`.
- `tests/test_ui_build.py`: update the expected tab list (seven tabs) and the confirmation-dialog count;
  add the `grid.*` registry keys check; `tests/test_ui_flows.py`: extend with the grid tab status mapping.
- Trainer: extend `tests/test_trainer_cpu.py` with early stopping (patience 1 with a worsening validation
  loss stops early; status message mentions early stopping) and with `auto_analyze` writing
  `analysis/training_analysis.json` (set `auto_evaluate_checkpoints=False` in that test).
- Keep `tests/test_ui_request_coverage.py` passing (the grid tab must not add unknown runner request keys;
  build grid requests from the same `infer_kwargs` shape as `sampling.py`).

## 3. Constraints and verification

- Do not add dependencies. Windows and Linux paths (`pathlib`, no shell-specific code). Gradio 6.26 API
  (`gr.Timer`, `gr.render`, `buttons=[...]`, `gr.skip()`), transformers 5, torch 2.13.
- All JSON written by workers goes through the atomic helpers (`indextts.utils.atomic_json` /
  `dataset_manifest.atomic_write_json`).
- Console + UI information policy: every long step prints progress with speed/ETA and is visible in the tab.
- Do NOT run GPU jobs (training, evaluation, grids, or the web UI with models) in this task; the GPU is in
  use. GPU verification happens separately afterwards. The CPU test-suite must pass:
  `venv\Scripts\python.exe -m pytest tests -q`.
- `ui.app.build_app` must construct without warnings (the existing test checks this) and
  `startup_request_self_check` must still report OK.
- Update `ARCHITECTURE_NOTES.md` with the new modules, worker contracts (`eval_worker`, `grid_worker`),
  the `analysis/` folder layout, and the `outputs/grids/` layout.
- When finished, write `CHECKPOINT_GRID_REPORT.md` at the repo root: what was implemented, file list, how the
  verdict rules work, test results, and anything left open.
