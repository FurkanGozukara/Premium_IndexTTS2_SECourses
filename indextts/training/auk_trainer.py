"""AuK flow-matching fine-tuning with the app's checkpoints, monitoring and selection.

Trainable: the 1.5B Flux2Edit transformer (full fine-tuning) or LoRA / DoRA adapters on
its attention, feed-forward and (optionally) adaLN modulation projections, plus the small
input and output layers. The Qwen2.5-Omni encoder, its layer fusion and the VAE stay frozen,
as in upstream's fine-tuning recipe. The loss is upstream's: velocity MSE on logistic-normal
times, with classifier-free-guidance dropout of the reference (0.3) and of all conditioning (0.2).
"""
from __future__ import annotations

from collections import deque
from dataclasses import replace
import gc
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from indextts.lora import apply_lora, inject_adapters, trainable_parameters
from indextts.lora.io import load_train_state, resume_state_path_for
from indextts.runtime.progress import ProgressReporter
from .auk_data import (AukDataset, EpochTaggedBatches, VOICE_FILE, cache_auk_conditions, cache_auk_features, collate,
                       flow_validation_loss, fuse, fusion_state, normalised_means, sample_latents, voice_record)
from .dataset import LengthBucketBatchSampler, TokenBudgetBatchSampler
from .dataset_manifest import atomic_write_json
from .early_stopping import EarlyStopping
from .ema import AdapterEMA
from .plan import automatic_epochs
from .trainer import (BuiltTrainingModel, LoraTrainer, TrainingResult, _dtype, _optimizer, _scheduler,
                      _seed_everything, reduce_learning_rate)

AUK_BASE_MODEL = "tencent/AuK"
ATTENTION_PROJECTIONS = ("to_qkv", "to_qkv_c", "to_out.0", "to_out_c")
FEED_FORWARD_PROJECTIONS = ("linear_in", "linear_out")
ADALN_PROJECTIONS = ("attn_norm.linear", "attn_norm_x.linear", "attn_norm_c.linear")
# Small layers trained in full by "Train input and output layers" (a few million parameters).
# Input and output layers outside the blocks. The ones the ConvRot INT8 transformer keeps in floating
# point are trained fully; the ones it quantizes are adapted like the block projections, so an adapter
# loads into the BF16 and the INT8 transformer alike.
EDGE_MODULES = ("txt_norm", "audio_embed", "proj_out")
EDGE_PROJECTIONS = ("txt_proj", "time_embed.time_mlp.0", "time_embed.time_mlp.2", "norm_out.linear")


def adapter_targets(model, *, attention=True, feed_forward=True, adaln=True, edges=False) -> list[str]:
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) and type(module).__name__ != "ConvRotInt8Linear":
            continue
        if edges and name.removeprefix("transformer.") in EDGE_PROJECTIONS:
            targets.append(name)
            continue
        if not name.startswith(("transformer.transformer_blocks.", "transformer.single_transformer_blocks.")):
            continue
        tail = name.split(".", 3)[-1]
        if attention and tail.startswith("attn.") and tail.removeprefix("attn.") in ATTENTION_PROJECTIONS:
            targets.append(name)
        elif feed_forward and tail.split(".")[0] in {"ff", "ff_c", "ff_x"} and tail.rsplit(".", 1)[-1] in FEED_FORWARD_PROJECTIONS:
            targets.append(name)
        elif adaln and tail in ADALN_PROJECTIONS:
            targets.append(name)
    return targets


def build_auk_training_model(config) -> BuiltTrainingModel:
    from indextts.auk.loader import build_int8_model, build_model, read_config, read_state
    from indextts.backends.auk import ensure_model
    from indextts.lora import inspect_lora

    full = config.adapter_type == "full"
    if full and str(config.device).startswith("cuda"):
        total = torch.cuda.get_device_properties(config.device).total_memory / 1024**3
        if total < 31:
            raise ValueError("AuK full fine-tuning keeps FP32 weights, gradients and optimizer state of 1.5B "
                             "parameters (about 25 GB); choose LoRA / DoRA on cards under 32 GB.")
    if config.blocks_to_swap:
        raise ValueError("AuK training keeps its weights resident; set block swap to zero.")
    folder, _, quantized = ensure_model(config.model_dir, dit_variant=config.base_variant)
    model_config = read_config(folder)
    model_config["arch"] = {**model_config["arch"], "checkpoint_activations": bool(config.gradient_checkpointing),
                            "checkpoint_every_n_layers": 1}
    if config.base_variant == "int8_convrot":
        from indextts.quant.convrot_int8 import ConvRotInt8Linear

        model = build_int8_model(model_config, quantized["dit"], device=config.device)
        for module in model.modules():
            if isinstance(module, ConvRotInt8Linear):
                module.training_ste = True
    else:
        dtype = torch.float32 if full else _dtype(config.base_dtype)
        model = build_model(model_config, read_state(folder / "auk_base.safetensors"), device=config.device, dtype=dtype)
    if config.resume_from:
        metadata = inspect_lora(config.resume_from)
        if metadata.get("base_model") != AUK_BASE_MODEL:
            raise ValueError("The resume checkpoint belongs to a different speech model.")
        if metadata["adapter_type"] != config.adapter_type:
            raise ValueError("The resume checkpoint's training method differs from this run.")
        apply_lora(model, config.resume_from)
    if full:
        adapters = {}
        modules = {"transformer": model.transformer}
    else:
        targets = adapter_targets(model, attention=config.target_attention, feed_forward=config.target_mlp,
                                  adaln=config.auk_target_adaln, edges=config.train_mel_embed_head)
        adapters = inject_adapters(model, config.rank, config.alpha, config.dropout, config.adapter_type == "dora", targets)
        modules = {}
        if config.train_mel_embed_head:
            modules = {f"transformer.{name}": getattr(model.transformer, name) for name in EDGE_MODULES}
            if config.train_full_modules_fp32:
                for module in modules.values():
                    module.to(torch.float32)
    parameters = trainable_parameters(model, adapters, modules)
    model.train()
    return BuiltTrainingModel(model, adapters, modules, parameters)


class AukTrainer(LoraTrainer):
    def _metadata(self, step, epochs, targets):
        return replace(super()._metadata(step, epochs, targets), base_model=AUK_BASE_MODEL)

    def _train_state(self, **kwargs):
        state = super()._train_state(**kwargs)
        state["last_validation_loss"] = getattr(self, "last_validation_loss", None)
        if self.ema is not None:
            state["ema"] = self.ema.state_dict()
        return state

    def _prepare_reference(self):
        from .evaluation_plan import choose_training_reference

        reference = super()._prepare_reference()
        row = choose_training_reference(self.training_records, self.dataset_dir, typical=self.config.reference_typical)
        if reference and row:
            reference.with_suffix(".txt").write_text(str(row["text"]), encoding="utf-8")
        return reference

    # ------------------------------------------------------------------ conditioning

    def _encoder(self):
        """The frozen Qwen2.5-Omni encoder, loaded only for reference-prompt objectives."""
        if getattr(self, "_qwen", None) is None:
            from .auk_data import load_text_encoder

            self.log(">> Loading the frozen Qwen2.5-Omni encoder for reference-prompt training")
            self._qwen = load_text_encoder(self.config)
        return self._qwen

    def _encode(self, batch, *, sample, generator=None):
        device = torch.device(self.config.device)
        global_mean, global_var = self.vae_stats
        if "text" in batch:
            text = batch["text"].to(device)
            lengths = batch["text_lens"].to(device)
            context = torch.arange(text.shape[1], device=device)[None, :] < lengths[:, None]
        else:
            encoder = self._encoder()
            audios = batch.get("ref_audio") or [None] * len(batch["instructions"])
            with torch.no_grad():
                hidden, context = encoder.hidden_states(encoder.inputs(batch["instructions"], audios))
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    text = fuse(hidden, self.fusion)
            context = context.to(device)
        if "ref_mean" in batch:
            mean, log_std = batch["ref_mean"].to(device), batch["ref_log_std"].to(device)
            ref = sample_latents(mean, log_std, global_mean, global_var, generator) if sample else \
                normalised_means(mean, global_mean, global_var)
            ref_lens = batch["ref_lens"].to(device)
            ref = ref * (torch.arange(ref.shape[1], device=device)[None, :] < ref_lens[:, None])[..., None]
        else:
            ref = torch.zeros(len(batch["ids"]), 0, 64, device=device)
            ref_lens = torch.zeros(len(batch["ids"]), dtype=torch.long, device=device)
        return text.float(), context, ref, ref_lens

    # ------------------------------------------------------------------ run

    def run(self):
        from indextts.backends.auk import ensure_model
        from safetensors import safe_open
        from .evaluation_plan import build_final_test_plan, build_speech_plan
        from .features import FeatureCacheConfig
        from .sampling import generate_training_sample

        config = self.config
        _seed_everything(config.seed)
        device = torch.device(config.device)
        if device.type == "cuda":
            torch.set_num_threads(min(torch.get_num_threads(), 8))
        self.reporter = self.reporter or ProgressReporter("AuK training", progress_file=self.state_dir / "progress.json")
        self.ema, self._qwen = None, None
        self.write_status(phase="caching", message="Checking AuK latent features")
        cache_auk_features(FeatureCacheConfig(dataset_dir=str(self.dataset_dir), model_dir=config.model_dir,
                                              device=config.device), self.reporter,
                           cancel_callback=lambda *_: self.stop_path.exists())
        if self.stop_path.exists():
            self.write_status(phase="stopped", message="Stopped during feature caching")
            return TrainingResult("stopped", 0, 0, 0, str(self.adapter_dir), "", None, None, None, 0, 0,
                                  time.perf_counter() - self.started_perf)
        folder, _, _ = ensure_model(config.model_dir)
        with safe_open(str(folder / "vae.safetensors"), framework="pt", device="cpu") as handle:
            self.vae_stats = (handle.get_tensor("global_mean").float().to(device),
                              handle.get_tensor("global_log_std").float().to(device))
        self.fusion = fusion_state(folder)
        train_data, val_data = [AukDataset(config, split, log=self.log) for split in ("train", "val")]
        self.training_records = train_data.records
        if config.auk_prompt_fraction < 1.0:
            # The no-reference instructions' conditioning is fixed: encode it once.
            self.write_status(phase="caching", message="Encoding AuK training instructions")
            for data in (train_data, val_data):
                if len(data):
                    data.conditions = cache_auk_conditions(config, data.records, data.instructions, self.reporter,
                                                           cancel_callback=self.stop_path.exists)
        record = voice_record(train_data.records, train_data.description)
        if record:
            # Auto voice reads the pace and the description this voice was trained with.
            atomic_write_json(self.adapter_dir / VOICE_FILE, record)
            self.log(f">> Voice pace {record['pace']:.3f} x upstream's estimate; description: {record['description']}")
        reference = self._prepare_reference()
        if config.speech_eval_enabled and len(val_data):
            plan = build_speech_plan(config, train_data.records, val_data.records, self.adapter_dir)
            self.speech_plan_ready = bool(plan["groups"])
            build_final_test_plan(config, train_data.records, val_data.records, self.adapter_dir)
        if config.auk_batch_frames:
            sampler = TokenBudgetBatchSampler(train_data.lengths, config.auk_batch_frames, seed=config.seed)
            val_sampler = TokenBudgetBatchSampler(val_data.lengths, config.auk_batch_frames, shuffle=False, seed=config.seed)
        else:
            sampler = LengthBucketBatchSampler(train_data.lengths, config.batch_size, seed=config.seed)
            val_sampler = None
        # Workers read the cached latents and reference audio ahead of the GPU (about 75 ms per micro-batch
        # in the main process) and stay for the whole run; each batch carries its epoch.
        train_loader = DataLoader(train_data, batch_sampler=EpochTaggedBatches(sampler), collate_fn=collate,
                                  num_workers=config.num_workers, persistent_workers=config.num_workers > 0)
        loader_args = {"collate_fn": collate, "num_workers": 0}
        val_loader = (DataLoader(val_data, batch_sampler=val_sampler, **loader_args) if val_sampler is not None
                      else DataLoader(val_data, batch_size=config.batch_size, shuffle=False, **loader_args))
        if not config.epochs:
            seconds = sum(float(row.get("duration_s") or 0.0) for row in train_data.records)
            config.epochs = automatic_epochs(seconds)
            self.log(f">> Automatic length: {config.epochs} epochs for {seconds / 3600:.1f} hours of training audio")
        total_steps = int(min(config.max_steps or math.inf,
                              config.epochs * math.ceil(len(train_loader) / config.grad_accumulation)))
        self.write_status(phase="initializing", total_steps=total_steps, message="Loading AuK training weights")
        if device.type == "cuda":
            # Expandable segments keep the allocator from reserving far more than training uses
            # (full fine-tuning: 26.6 instead of 28.4 GiB reserved), which decides whether it fits 32 GB.
            torch.cuda.memory._set_allocator_settings("expandable_segments:True")
        built = build_auk_training_model(config)
        model = built.model
        optimizer = _optimizer(config, built.parameters)
        scheduler = _scheduler(config, optimizer, total_steps)
        scaler = torch.amp.GradScaler(device.type, enabled=False)
        self.ema = AdapterEMA(built.parameters, config.ema_decay) if config.ema_decay else None
        self.log(f">> AuK {config.adapter_type.upper()} | {len(train_data)} training / {len(val_data)} validation clips | "
                 f"{sum(p.numel() for p in built.parameters):,} trainable parameters | {total_steps} updates | "
                 f"reference prompts {config.auk_prompt_fraction:.0%}")
        atomic_write_json(self.adapter_dir / "train_config.json", config.to_dict())
        state = {}
        if config.resume_from and config.resume_mode == "continue":
            path = Path(resume_state_path_for(config.resume_from))
            if not path.is_file():
                raise FileNotFoundError(f"Continue run requires optimizer state: {path}")
            state = load_train_state(path)
            if state.get("dataset_fingerprint") != train_data.fingerprint:
                raise ValueError("Dataset content or split changed; use weights-only resume for a new dataset.")
            self._restore_resume_state(state, optimizer, scheduler, scaler)
            if self.ema is not None and state.get("ema"):
                self.ema.load_state_dict(state["ema"])
        step = int(state.get("step", 0))
        epoch_index = int(state.get("epoch", 0))
        next_batch = int(state.get("batch_in_epoch", 0))
        self.early_stopping = EarlyStopping.from_state(state.get("early_stopping"))
        moving = deque(state.get("moving_losses", []), maxlen=50)
        initial_loss, final_loss = None, None
        val_loss = state.get("last_validation_loss")
        self.last_validation_loss = val_loss
        speed, stop, early = 0.0, False, False
        started_step = step
        train_started = time.perf_counter()
        self.resolved_sample_seed = config.seed if config.sample_seed < 0 else config.sample_seed
        generator = torch.Generator(device=device).manual_seed(config.seed + step)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        def save(path, epoch, batch):
            return self.save_checkpoint(path, built, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                step=step, epochs_completed=epoch, next_epoch=epoch_index, next_batch=batch,
                dataset_fingerprint=train_data.fingerprint, best_val_loss=self.early_stopping.best_loss,
                ema_loss=sum(moving) / len(moving) if moving else None, moving_losses=moving)

        def validate(epoch):
            nonlocal val_loss, early
            self.write_status(phase="validating", step=step, message="Validating with fixed noise and times")
            loss, per_t = flow_validation_loss(model, val_loader, device, lambda batch: self._encode(batch, sample=False),
                                               self.vae_stats, max_batches=config.val_max_batches,
                                               cancel_callback=self.stop_path.exists)
            if loss is None:
                return
            val_loss = loss
            self.last_validation_loss = val_loss
            is_best, early = self.early_stopping.observe(val_loss, step=step, epoch=epoch,
                enabled=config.early_stop_enabled, patience=config.early_stop_patience,
                min_delta=config.early_stop_min_delta, min_steps=max(config.early_stop_min_steps, config.warmup_steps),
                min_epochs=config.early_stop_min_epochs, check_interval=config.early_stop_check_steps)
            self.metric({"event": "validation", "step": step, "epoch": math.ceil(epoch), "fractional_epoch": epoch,
                         "val_loss": val_loss, "val_loss_per_t": per_t, "elapsed_s": time.perf_counter() - self.started_perf})
            if is_best and config.save_best and step:
                save(self.best_path, math.ceil(epoch), next_batch)
            atomic_write_json(self.adapter_dir / "analysis/checkpoint_selection.json", {
                **self.early_stopping.to_dict(), "checkpoint": str(self.best_path) if self.best_path.is_file() else "",
                "metric": "flow_matching_validation_loss", "validation_items": len(val_data),
                "max_validation_batches": config.val_max_batches, "val_split_mode": config.val_split_mode})
            self.log(f">> validation step {step} | flow loss {val_loss:.5f} | per t "
                     + " ".join(f"{value:.3f}" for value in per_t))
            if early and config.plateau_lr_enabled and not self.early_stopping.lr_reductions \
                    and total_steps - step > config.plateau_lr_grace_steps:
                reduce_learning_rate(optimizer, scheduler, config.plateau_lr_factor)
                self.early_stopping.begin_refinement(step, config.plateau_lr_grace_steps)
                early = False
                self.log(">> validation plateau; reduced learning rate for refinement")

        if not state and len(val_data):
            validate(0.0)
        optimizer.zero_grad(set_to_none=True)
        global_mean, global_var = self.vae_stats
        try:
            while epoch_index < config.epochs and step < total_steps and not early:
                train_data.set_epoch(epoch_index)
                sampler.set_epoch(epoch_index)
                losses = []
                update_started = time.perf_counter()
                for batch_index, batch in enumerate(train_loader):
                    if batch_index < next_batch:
                        continue
                    if self.stop_path.exists():
                        stop = True
                        break
                    group_start = (batch_index // config.grad_accumulation) * config.grad_accumulation
                    group_size = min(config.grad_accumulation, len(train_loader) - group_start)
                    text, context, ref, ref_lens = self._encode(batch, sample=True, generator=generator)
                    target = sample_latents(batch["target_mean"].to(device), batch["target_log_std"].to(device),
                                            global_mean, global_var, generator)
                    with torch.autocast(device.type, dtype=_dtype(config.mixed_precision),
                                        enabled=device.type == "cuda" and config.mixed_precision != "fp32"):
                        loss = model(target, text, context, ref_latent=ref, ref_lens=ref_lens,
                                     target_lens=batch["target_lens"].to(device))
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite AuK loss at step {step}")
                    losses.append(float(loss.detach()))
                    (loss / group_size).backward()
                    del loss
                    if (batch_index + 1) % config.grad_accumulation and batch_index + 1 < len(train_loader):
                        continue
                    norm = torch.nn.utils.clip_grad_norm_(built.parameters, config.max_grad_norm or math.inf)
                    if not torch.isfinite(norm):
                        raise FloatingPointError(f"Non-finite AuK gradients at step {step}")
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    if self.ema is not None:
                        self.ema.update()
                    step += 1
                    next_batch = batch_index + 1
                    final_loss = sum(losses) / len(losses)
                    losses.clear()
                    if initial_loss is None:
                        initial_loss = final_loss
                    moving.append(final_loss)
                    speed = 1.0 / max(1e-6, time.perf_counter() - update_started)
                    elapsed = time.perf_counter() - self.started_perf
                    peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
                    row = {"step": step, "epoch": epoch_index + 1, "loss": final_loss, "avg_loss": sum(moving) / len(moving),
                           "moving_avg_loss": sum(moving) / len(moving), "lr": optimizer.param_groups[0]["lr"],
                           "grad_norm": float(norm), "it_s": speed, "elapsed_s": elapsed,
                           "eta_s": (total_steps - step) / speed, "vram_peak_gb": peak, "vram_used_gb": peak}
                    self.metric(row)
                    self.write_status(phase="training", total_steps=total_steps, val_loss=val_loss,
                                      last_checkpoint=self.last_checkpoint, last_sample=self.last_sample,
                                      sample_seed=self.resolved_sample_seed, message="AuK flow-matching training", **row)
                    if step % config.log_every_steps == 0:
                        self.log(f">> step {step}/{total_steps} | loss {final_loss:.4f} | {speed:.2f} it/s | VRAM {peak:.2f} GiB")
                    self.reporter.update(step, total=total_steps, desc=f"AuK step {step}/{total_steps}",
                                         extra={"speed": speed, "eta_s": (total_steps - step) / speed})
                    if config.val_every_steps and step % config.val_every_steps == 0:
                        validate(epoch_index + next_batch / len(train_loader))
                    if config.save_every_steps and step % config.save_every_steps == 0:
                        save(self.adapter_dir / f"{config.name}_step_{step:06d}.safetensors", epoch_index + 1, next_batch)
                    if early or step >= total_steps:
                        break
                    update_started = time.perf_counter()
                if stop:
                    break
                completed_epoch = next_batch >= len(train_loader)
                if len(val_data) and self.early_stopping.last_step != step:
                    validate(epoch_index + next_batch / len(train_loader))
                if completed_epoch:
                    epoch_index += 1
                    next_batch = 0
                    if config.save_every_epochs and epoch_index % config.save_every_epochs == 0:
                        checkpoint = save(self.adapter_dir / f"{config.name}_epoch_{epoch_index:03d}.safetensors", epoch_index, 0)
                        if config.sample_enabled and epoch_index % config.sample_every_epochs == 0 and reference:
                            sample = generate_training_sample(config, adapter_path=checkpoint, reference_path=reference,
                                output_path=self.adapter_dir / "samples" / f"epoch_{epoch_index:03d}.wav", epoch=epoch_index,
                                seed=self.resolved_sample_seed, log=self.log)
                            if sample.generated:
                                self.last_sample = sample.path
                if step >= total_steps or early:
                    break
            status = "stopped" if stop else "complete"
            path = self.adapter_dir / f"{config.name}{'_interrupted' if stop else ''}.safetensors"
            save(path, epoch_index + (next_batch > 0), next_batch)
        except BaseException:
            save(self.adapter_dir / f"{config.name}_interrupted.safetensors", epoch_index + (next_batch > 0), next_batch)
            raise
        peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
        result = TrainingResult(status, step, total_steps, epoch_index, str(path),
                                str(self.best_path) if self.best_path.is_file() else "", self.early_stopping.best_loss,
                                initial_loss, final_loss,
                                (step - started_step) / max(1e-6, time.perf_counter() - train_started), peak,
                                time.perf_counter() - self.started_perf)
        del optimizer, scheduler, built, model
        self.ema = None
        if self._qwen is not None:
            self._qwen.unload()
            self._qwen = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        self._write_averaged_checkpoint()
        self._write_automatic_analysis()
        message = self.early_stopping.reason or ("Stopped; resumable checkpoint saved" if stop else "AuK training complete")
        recommended = result.best_path
        if not stop:
            actions = [("checkpoint evaluation", self._run_automatic_evaluation),
                       ("speech evaluation", self._run_automatic_speech_evaluation)]
            for label, action in actions:
                try:
                    recommended = action(terminal_phase="post_training", terminal_message=message,
                                         recommended_checkpoint=recommended)
                except Exception as exc:
                    self.log(f">> {label} failed but training weights are safe: {exc}")
                    message += f"; {label} failed: {exc}"
            if bool(config.final_test_dataset) and not self.stop_path.exists():
                try:
                    self._run_final_test_assessment(terminal_phase="post_training", terminal_message=message,
                                                    recommended_checkpoint=recommended)
                except Exception as exc:
                    self.log(f">> final test failed but training weights are safe: {exc}")
                    message += f"; final test failed: {exc}"
            try:
                self._run_reference_audition(terminal_phase="post_training", terminal_message=message,
                                             recommended_checkpoint=recommended)
            except Exception as exc:
                self.log(f">> reference audition failed but training weights are safe: {exc}")
                message += f"; reference audition failed: {exc}"
        self.write_status(phase=status, message=message, step=step, total_steps=total_steps, last_checkpoint=str(path),
                          last_sample=self.last_sample, sample_seed=self.resolved_sample_seed,
                          recommended_checkpoint=recommended, elapsed_s=time.perf_counter() - self.started_perf)
        atomic_write_json(self.adapter_dir / "result.json", result.__dict__)
        return result
