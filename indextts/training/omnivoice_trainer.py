"""Masked-diffusion training with the app's checkpoints, monitoring and selection."""
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
from .dataset import LengthBucketBatchSampler, TokenBudgetBatchSampler
from .plan import automatic_epochs
from .dataset_manifest import atomic_write_json
from .early_stopping import EarlyStopping
from .ema import AdapterEMA
from .omnivoice_data import OmniVoiceDataset
from .trainer import (BuiltTrainingModel, LoraTrainer, TrainingResult, _dtype,
                      _optimizer, _scheduler, _seed_everything, _restore_rng,
                      reduce_learning_rate)


def build_omnivoice_training_model(config):
    if config.adapter_type == "full" and str(config.device).startswith("cuda"):
        from indextts.runtime.vram_presets import auto_tier
        total = torch.cuda.get_device_properties(config.device).total_memory / 1024**3
        if auto_tier(total) < 16:
            raise ValueError("OmniVoice full fine-tuning needs a GPU with at least 16 GB; choose LoRA / DoRA on this card.")
    from indextts.backends.omnivoice import MODEL_REPO, ensure_model
    from indextts.utils.torch_compat import install_native_enum_pytree_compatibility
    install_native_enum_pytree_compatibility()
    from omnivoice import OmniVoice
    from indextts.lora import inspect_lora
    folder, quant = ensure_model(config.model_dir, quantized=config.base_variant == "int8_convrot")
    dtype = torch.float32 if config.adapter_type == "full" else _dtype(config.base_dtype)
    model = OmniVoice.from_pretrained(str(folder), train=True, device_map=config.device, dtype=dtype)
    model.llm.set_attn_implementation(config.attention_backend)
    model.llm.config.use_cache = False
    if config.blocks_to_swap:
        raise ValueError("OmniVoice training uses resident weights. Set block swap to zero and reduce batch size or choose INT8 DoRA for lower VRAM.")
    if config.base_variant == "int8_convrot":
        from indextts.quant.convrot_int8 import load_gpt_checkpoint, ConvRotInt8Linear
        load_gpt_checkpoint(model.llm, str(quant), device=config.device, dtype=dtype, strict=True)
        for module in model.modules():
            if isinstance(module, ConvRotInt8Linear):
                module.training_ste = True
    if config.resume_from:
        metadata = inspect_lora(config.resume_from)
        if metadata.get("base_model") != MODEL_REPO:
            raise ValueError("The resume checkpoint belongs to a different speech model.")
        if metadata["adapter_type"] != config.adapter_type:
            raise ValueError("The resume checkpoint's training method differs from this run.")
        apply_lora(model, config.resume_from)
    if config.adapter_type == "full":
        adapters = {}
        full = {name: getattr(model, name) for name in ("llm", "audio_embeddings", "audio_heads")}
    else:
        projections = []
        if config.target_attention:
            projections += ["q_proj", "k_proj", "v_proj", "o_proj"]
        if config.target_mlp:
            projections += ["gate_proj", "up_proj", "down_proj"]
        targets = [name for name, _ in model.named_modules() if name.startswith("llm.layers.") and name.rsplit(".",1)[-1] in projections]
        adapters = inject_adapters(model, config.rank, config.alpha, config.dropout,
                                   config.adapter_type == "dora", targets)
        full = {name: getattr(model, name) for name in ("audio_embeddings", "audio_heads")} if config.train_mel_embed_head else {}
        if config.train_full_modules_fp32:
            for module in full.values():
                module.to(torch.float32)
    parameters = trainable_parameters(model, adapters, full)
    if config.gradient_checkpointing:
        model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return BuiltTrainingModel(model, adapters, full, parameters)


class OmniVoiceTrainer(LoraTrainer):
    def _prepare_reference(self):
        # The training reference and the exact transcript of the clip it was copied from.
        reference = super()._prepare_reference()
        row = getattr(self, "_reference_row", None)
        if reference and row:
            reference.with_suffix(".txt").write_text(str(row["text"]), encoding="utf-8")
        return reference

    def _metadata(self, step, epochs, targets):
        return replace(super()._metadata(step, epochs, targets), base_model="k2-fsa/OmniVoice")

    def _train_state(self, **kwargs):
        state = super()._train_state(**kwargs)
        state["last_validation_loss"] = getattr(self, "last_validation_loss", None)
        if self.ema is not None:
            state["ema"] = self.ema.state_dict()
        return state

    def _validate_omni(self, model, loader, device):
        from .omnivoice_data import masked_audio_metrics
        metrics = masked_audio_metrics(model, loader, device, dtype=_dtype(self.config.mixed_precision),
                                       max_batches=self.config.val_max_batches, cancel_callback=self.stop_path.exists)
        model.train()
        return (metrics["loss"], metrics["accuracy"]) if metrics["loss"] is not None else None

    def run(self):
        from transformers import AutoTokenizer
        from omnivoice.data.collator import PaddingDataCollator
        from indextts.backends.omnivoice import ensure_model
        from .features import FeatureCacheConfig
        from .omnivoice_data import cache_omnivoice_features
        from .evaluation_plan import build_speech_plan, build_final_test_plan
        from .sampling import generate_training_sample

        config = self.config
        _seed_everything(config.seed)
        device = torch.device(config.device)
        if device.type == "cuda":
            torch.set_num_threads(min(torch.get_num_threads(), 8))
        self.reporter = self.reporter or ProgressReporter("OmniVoice training", progress_file=self.state_dir / "progress.json")
        self.write_status(phase="caching", message="Checking OmniVoice codec features")
        cache_omnivoice_features(FeatureCacheConfig(dataset_dir=str(self.dataset_dir),model_dir=config.model_dir,
            device=config.device), self.reporter,
            cancel_callback=lambda *_: self.stop_path.exists())
        if self.stop_path.exists():
            self.write_status(phase="stopped", message="Stopped during feature caching")
            return TrainingResult("stopped",0,0,0,str(self.adapter_dir),"",None,None,None,0,0,time.perf_counter()-self.started_perf)
        folder, _ = ensure_model(config.model_dir)
        tokenizer = AutoTokenizer.from_pretrained(str(folder))
        train_data, val_data = [OmniVoiceDataset(config, tokenizer, split, log=self.log) for split in ("train", "val")]
        self.training_records = train_data.records
        from .omnivoice_data import VOICE_CALIBRATION_FILE, speaking_rate_calibration
        calibration = speaking_rate_calibration(train_data.records)
        if calibration:
            # Generation without a reference reads this to speak at the speaker's pace.
            atomic_write_json(self.adapter_dir / VOICE_CALIBRATION_FILE, calibration)
            self.log(f">> Voice pace calibration: speed {calibration['speed_without_reference']:.3f} without a reference")
        reference = self._prepare_reference()
        if config.speech_eval_enabled and len(val_data):
            plan = build_speech_plan(config, train_data.records, val_data.records, self.adapter_dir)
            self.speech_plan_ready = bool(plan["groups"])
            build_final_test_plan(config, train_data.records, val_data.records, self.adapter_dir)
        collator = PaddingDataCollator(train_data.processor, batch_tokens=0)
        if config.omni_batch_tokens:
            # Length-grouped batches of a token budget, as the upstream recipe trains.
            sampler = TokenBudgetBatchSampler(train_data.lengths, config.omni_batch_tokens, seed=config.seed)
            val_sampler = TokenBudgetBatchSampler(val_data.lengths, config.omni_batch_tokens, shuffle=False, seed=config.seed)
        else:
            sampler = LengthBucketBatchSampler(train_data.lengths, config.batch_size, seed=config.seed)
            val_sampler = None
        # The per-record seeded processor is cheap; zero workers also gives exact
        # continuation without worker-prefetch state and avoids Windows spawn copies.
        train_loader = DataLoader(train_data, batch_sampler=sampler, collate_fn=collator, num_workers=0,
                                  generator=torch.Generator().manual_seed(config.seed))
        if val_sampler is not None:
            val_loader = DataLoader(val_data, batch_sampler=val_sampler, collate_fn=collator, num_workers=0)
        else:
            val_loader = DataLoader(val_data, batch_size=config.batch_size, collate_fn=collator, num_workers=0,
                                    generator=torch.Generator().manual_seed(config.seed))
        if not config.epochs:
            seconds = sum(float(row.get("duration_s") or 0.0) for row in train_data.records)
            config.epochs = automatic_epochs(seconds)
            self.log(f">> Automatic length: {config.epochs} epochs for {seconds / 3600:.1f} hours of training audio")
        total_steps = min(config.max_steps or math.inf, config.epochs * math.ceil(len(train_loader)/config.grad_accumulation))
        total_steps = int(total_steps)
        self.write_status(phase="initializing", total_steps=total_steps, message="Loading OmniVoice training weights")
        built = build_omnivoice_training_model(config)
        model = built.model
        optimizer = _optimizer(config, built.parameters)
        scheduler = _scheduler(config, optimizer, total_steps)
        scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda" and config.mixed_precision == "fp16")
        self.ema = AdapterEMA(built.parameters, config.ema_decay) if config.ema_decay else None
        self.log(f">> OmniVoice {config.adapter_type.upper()} | {len(train_data)} training / {len(val_data)} validation clips | "
                 f"{sum(p.numel() for p in built.parameters):,} trainable parameters | {total_steps} updates")
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
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        def save(path, epoch, batch):
            return self.save_checkpoint(path, built, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                step=step, epochs_completed=epoch, next_epoch=epoch_index, next_batch=batch,
                dataset_fingerprint=train_data.fingerprint, best_val_loss=self.early_stopping.best_loss,
                ema_loss=sum(moving)/len(moving) if moving else None, moving_losses=moving)

        def validate(epoch):
            nonlocal val_loss, early
            self.write_status(phase="validating", step=step, message="Validating with fixed masks")
            values = self._validate_omni(model, val_loader, device)
            if values is None:
                return
            val_loss, accuracy = values
            self.last_validation_loss = val_loss
            is_best, early = self.early_stopping.observe(val_loss, step=step, epoch=epoch,
                enabled=config.early_stop_enabled, patience=config.early_stop_patience,
                min_delta=config.early_stop_min_delta, min_steps=max(config.early_stop_min_steps,config.warmup_steps),
                min_epochs=config.early_stop_min_epochs, check_interval=config.early_stop_check_steps)
            self.metric({"event":"validation","step":step,"epoch":math.ceil(epoch),"fractional_epoch":epoch,"val_loss":val_loss,"val_mel_accuracy":accuracy,
                         "elapsed_s":time.perf_counter()-self.started_perf})
            if is_best and config.save_best and step:
                save(self.best_path, math.ceil(epoch), next_batch)
            atomic_write_json(self.adapter_dir/"analysis/checkpoint_selection.json", {
                **self.early_stopping.to_dict(), "checkpoint":str(self.best_path) if self.best_path.is_file() else "",
                "metric":"masked_audio_validation_loss", "validation_items":len(val_data),
                "max_validation_batches":config.val_max_batches, "val_split_mode":config.val_split_mode})
            self.log(f">> validation step {step} | masked audio loss {val_loss:.5f} | accuracy {accuracy:.4f}")
            if early and config.plateau_lr_enabled and not self.early_stopping.lr_reductions and total_steps-step > config.plateau_lr_grace_steps:
                reduce_learning_rate(optimizer, scheduler, config.plateau_lr_factor)
                self.early_stopping.begin_refinement(step,config.plateau_lr_grace_steps)
                early=False
                self.log(">> validation plateau; reduced learning rate for refinement")

        if not state and len(val_data):
            validate(0.0)
        optimizer.zero_grad(set_to_none=True)
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
                        stop=True
                        break
                    batch = {k:v.to(device) for k,v in batch.items()}
                    # A short last accumulation group is scaled by its actual size.
                    group_start = (batch_index // config.grad_accumulation) * config.grad_accumulation
                    group_size = min(config.grad_accumulation,len(train_loader)-group_start)
                    with torch.autocast(device.type,dtype=_dtype(config.mixed_precision),
                                        enabled=device.type == "cuda" and config.mixed_precision != "fp32"):
                        output = model(**batch)
                        loss = output.loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite OmniVoice loss at step {step}")
                    losses.append(float(loss.detach()))
                    scaler.scale(loss / group_size).backward()
                    del output, loss
                    if (batch_index + 1) % config.grad_accumulation and batch_index + 1 < len(train_loader):
                        continue
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(built.parameters,config.max_grad_norm or math.inf)
                    if not torch.isfinite(norm):
                        raise FloatingPointError(f"Non-finite OmniVoice gradients at step {step}")
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    if self.ema is not None:
                        self.ema.update()
                    step += 1
                    next_batch = batch_index + 1
                    final_loss = sum(losses)/len(losses)
                    losses.clear()
                    if initial_loss is None:
                        initial_loss=final_loss
                    moving.append(final_loss)
                    speed = 1.0 / max(1e-6,time.perf_counter()-update_started)
                    elapsed = time.perf_counter()-self.started_perf
                    peak = torch.cuda.max_memory_allocated(device)/1024**3 if device.type == "cuda" else 0.0
                    row = {"step":step,"epoch":epoch_index+1,"loss":final_loss,"avg_loss":sum(moving)/len(moving),
                           "moving_avg_loss":sum(moving)/len(moving),"lr":optimizer.param_groups[0]["lr"],
                           "grad_norm":float(norm),"it_s":speed,"elapsed_s":elapsed,
                           "eta_s":(total_steps-step)/speed,"vram_peak_gb":peak,"vram_used_gb":peak}
                    self.metric(row)
                    self.write_status(phase="training",total_steps=total_steps,val_loss=val_loss,
                                      last_checkpoint=self.last_checkpoint,last_sample=self.last_sample,
                                      sample_seed=self.resolved_sample_seed,message="OmniVoice masked-diffusion training",**row)
                    if step % config.log_every_steps == 0:
                        self.log(f">> step {step}/{total_steps} | loss {final_loss:.4f} | {speed:.2f} it/s | VRAM {peak:.2f} GiB")
                    self.reporter.update(step,total=total_steps,desc=f"OmniVoice step {step}/{total_steps}",
                                         extra={"speed": speed, "eta_s": (total_steps-step)/speed})
                    if config.val_every_steps and step % config.val_every_steps == 0:
                        validate(epoch_index + next_batch/len(train_loader))
                    if config.save_every_steps and step % config.save_every_steps == 0:
                        save(self.adapter_dir/f"{config.name}_step_{step:06d}.safetensors",epoch_index+1,next_batch)
                    if early or step >= total_steps:
                        break
                    update_started=time.perf_counter()
                if stop:
                    break
                completed_epoch = next_batch >= len(train_loader)
                if len(val_data) and self.early_stopping.last_step != step:
                    validate(epoch_index + next_batch/len(train_loader))
                if completed_epoch:
                    epoch_index += 1
                    next_batch = 0
                    if config.save_every_epochs and epoch_index % config.save_every_epochs == 0:
                        checkpoint=save(self.adapter_dir/f"{config.name}_epoch_{epoch_index:03d}.safetensors",epoch_index,0)
                        if config.sample_enabled and epoch_index % config.sample_every_epochs == 0 and reference:
                            sample=generate_training_sample(config,adapter_path=checkpoint,reference_path=reference,
                                output_path=self.adapter_dir/"samples"/f"epoch_{epoch_index:03d}.wav",epoch=epoch_index,
                                seed=self.resolved_sample_seed,log=self.log)
                            if sample.generated:
                                self.last_sample=sample.path
                if step >= total_steps or early:
                    break
            status = "stopped" if stop else "complete"
            path=self.adapter_dir/f"{config.name}{'_interrupted' if stop else ''}.safetensors"
            save(path,epoch_index+(next_batch>0),next_batch)
        except BaseException:
            # Retain the last complete optimizer update. Interrupted partial
            # gradients are intentionally discarded on continuation.
            save(self.adapter_dir/f"{config.name}_interrupted.safetensors",epoch_index+(next_batch>0),next_batch)
            raise
        peak=torch.cuda.max_memory_allocated(device)/1024**3 if device.type == "cuda" else 0.0
        result=TrainingResult(status,step,total_steps,epoch_index,str(path),str(self.best_path) if self.best_path.is_file() else "",
            self.early_stopping.best_loss,initial_loss,final_loss,(step-started_step)/max(1e-6,time.perf_counter()-train_started),
            peak,time.perf_counter()-self.started_perf)
        del optimizer, scheduler, built, model
        self.ema=None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        self._write_averaged_checkpoint()
        self._write_automatic_analysis()
        message=self.early_stopping.reason or ("Stopped; resumable checkpoint saved" if stop else "OmniVoice training complete")
        recommended=result.best_path
        if not stop:
            for label, action in (("checkpoint evaluation", self._run_automatic_evaluation),
                                  ("speech evaluation", self._run_automatic_speech_evaluation)):
                try:
                    recommended=action(terminal_phase="post_training",terminal_message=message,recommended_checkpoint=recommended)
                except Exception as exc:
                    self.log(f">> {label} failed but training weights are safe: {exc}")
                    message += f"; {label} failed: {exc}"
            for label, enabled, action in (("decoding sweep", config.decoding_sweep_enabled, self._run_decoding_sweep),
                                           ("final test", bool(config.final_test_dataset), self._run_final_test_assessment)):
                if not enabled or self.stop_path.exists():
                    continue
                try:
                    action(terminal_phase="post_training",terminal_message=message,recommended_checkpoint=recommended)
                except Exception as exc:
                    self.log(f">> {label} failed but training weights are safe: {exc}")
                    message += f"; {label} failed: {exc}"
            self._write_int8_finetune(recommended)
        self.write_status(phase=status,message=message,step=step,total_steps=total_steps,last_checkpoint=str(path),
                          last_sample=self.last_sample,sample_seed=self.resolved_sample_seed,
                          recommended_checkpoint=recommended,elapsed_s=time.perf_counter()-self.started_perf)
        atomic_write_json(self.adapter_dir/"result.json",result.__dict__)
        return result
