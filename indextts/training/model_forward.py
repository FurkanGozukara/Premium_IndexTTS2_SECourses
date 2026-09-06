"""Inference-aligned training forward for the IndexTTS 2.5 GPT."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch.nn import functional as F


def enable_gradient_checkpointing(gpt: torch.nn.Module, enabled: bool = True) -> None:
    """Enable transformers 5.x non-reentrant checkpointing for inputs_embeds."""

    transformer = gpt.gpt
    if enabled:
        # UnifiedVoice deliberately removes GPT2Model.wte because training and
        # inference both supply inputs_embeds.  Transformers 5 otherwise tries
        # to attach its PEFT input-grad hook to that missing embedding.  Naming
        # inputs_embeds as the main input while enabling checkpointing avoids
        # the irrelevant hook; build_gpt_training_inputs supplies the explicit
        # grad-bearing input below.
        original_main_input = transformer.main_input_name
        transformer.main_input_name = "inputs_embeds"
        try:
            transformer.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        finally:
            transformer.main_input_name = original_main_input
        transformer.config.use_cache = False
    else:
        transformer.gradient_checkpointing_disable()


def _required(batch: Mapping[str, Any], name: str) -> torch.Tensor:
    value = batch.get(name)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"batch[{name!r}] must be a tensor")
    return value


def build_gpt_training_inputs(
    gpt: torch.nn.Module, batch: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    """Build ``[cond(3)][text][mel]`` exactly like campplus inference.

    Cached text excludes start/stop markers. The returned target masks include
    the first stop target and exclude the extra aligned stop target.
    """

    text = _required(batch, "text_tokens").long()
    text_lengths = _required(batch, "text_lengths").long()
    codes = _required(batch, "codes").long()
    code_lengths = _required(batch, "code_lengths").long()
    lang_ids = _required(batch, "lang_ids").long()
    campplus = _required(batch, "campplus")
    emo_raw = _required(batch, "emo_raw")
    cached_emo_vec = _required(batch, "emo_vec")

    if text.ndim != 2 or codes.ndim != 2:
        raise ValueError("text_tokens and codes must be [batch, time]")
    batch_size = text.shape[0]
    if codes.shape[0] != batch_size:
        raise ValueError("text/code batch sizes differ")

    device = text.device
    compute_dtype = gpt.spk_emb_proj.weight.dtype
    campplus = campplus.to(device=device, dtype=compute_dtype)
    speaker = gpt.spk_emb_proj(campplus).unsqueeze(1)

    emo_trainable = any(parameter.requires_grad for parameter in gpt.emovec_layer.parameters()) or any(
        parameter.requires_grad for parameter in gpt.emo_layer.parameters()
    )
    if emo_trainable:
        emo = gpt.emo_layer(
            gpt.emovec_layer(emo_raw.to(device=device, dtype=gpt.emovec_layer.weight.dtype))
        )
    else:
        emo = cached_emo_vec.to(device=device, dtype=speaker.dtype)
    emo = emo.to(dtype=speaker.dtype)
    shared = speaker + emo.unsqueeze(1)
    cond = torch.cat((shared, torch.zeros_like(shared).expand(-1, 2, -1)), dim=1)

    # Match set_*_padding, the explicit first stop, and aligned start/targets.
    text_padded = gpt.set_text_padding(text.clone(), text_lengths)
    text_padded = F.pad(text_padded, (0, 1), value=gpt.stop_text_token)
    text_inputs, text_targets = gpt.build_aligned_inputs_and_targets(
        text_padded, gpt.start_text_token, gpt.stop_text_token
    )
    mel_padded = gpt.set_mel_padding(codes.clone(), code_lengths)
    mel_padded = F.pad(mel_padded, (0, 1), value=gpt.stop_mel_token)
    mel_inputs, mel_targets = gpt.build_aligned_inputs_and_targets(
        mel_padded, gpt.start_mel_token, gpt.stop_mel_token
    )

    text_positions = torch.arange(text_inputs.shape[1], device=device)
    text_emb = gpt.text_embedding(text_inputs)
    text_emb = text_emb + gpt.text_pos_embedding.emb(text_positions)
    text_emb = text_emb + gpt.lang_embedding(lang_ids).unsqueeze(1)
    mel_emb = gpt.mel_embedding(mel_inputs) + gpt.mel_pos_embedding(mel_inputs)

    inputs_embeds = torch.cat((cond, text_emb, mel_emb), dim=1)
    text_key_mask = text_positions.unsqueeze(0) <= (text_lengths.unsqueeze(1) + 1)
    mel_positions = torch.arange(mel_inputs.shape[1], device=device)
    mel_key_mask = mel_positions.unsqueeze(0) <= (code_lengths.unsqueeze(1) + 1)
    attention_mask = torch.cat(
        (
            torch.ones((batch_size, 3), dtype=torch.bool, device=device),
            text_key_mask,
            mel_key_mask,
        ),
        dim=1,
    ).long()
    text_loss_mask = text_positions.unsqueeze(0) <= text_lengths.unsqueeze(1)
    mel_loss_mask = mel_positions.unsqueeze(0) <= code_lengths.unsqueeze(1)

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "text_inputs": text_inputs,
        "text_targets": text_targets,
        "text_loss_mask": text_loss_mask,
        "mel_inputs": mel_inputs,
        "mel_targets": mel_targets,
        "mel_loss_mask": mel_loss_mask,
        "cond": cond,
        "text_embeds": text_emb,
        "mel_embeds": mel_emb,
    }


def gpt_train_step_logits(
    gpt: torch.nn.Module, batch: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Return batch-major text/mel logits plus targets and masks."""

    values = build_gpt_training_inputs(gpt, batch)
    inputs_embeds = values["inputs_embeds"]
    if getattr(gpt.gpt, "is_gradient_checkpointing", False) and not inputs_embeds.requires_grad:
        # Non-reentrant checkpointing supports frozen inputs, but an explicit
        # grad-bearing input also keeps older transformers builds from pruning
        # adapter gradients when every embedding module is frozen.
        inputs_embeds.requires_grad_(True)
    output = gpt.gpt(
        inputs_embeds=inputs_embeds,
        attention_mask=values["attention_mask"],
        use_cache=False,
        return_dict=True,
    )
    hidden = gpt.final_norm(output.last_hidden_state[:, 3:])
    text_width = values["text_inputs"].shape[1]
    text_logits = gpt.text_head(hidden[:, :text_width])
    mel_logits = gpt.mel_head(hidden[:, text_width:])
    return text_logits, mel_logits, values


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    losses = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
        label_smoothing=float(label_smoothing),
    )
    weights = mask.reshape(-1).to(dtype=losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def gpt_train_step_loss(
    gpt: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    mel_loss_weight: float = 1.0,
    text_loss_weight: float = 0.1,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    text_logits, mel_logits, masks = gpt_train_step_logits(gpt, batch)
    text_loss = masked_cross_entropy(
        text_logits,
        masks["text_targets"],
        masks["text_loss_mask"],
        label_smoothing=label_smoothing,
    )
    mel_loss = masked_cross_entropy(
        mel_logits,
        masks["mel_targets"],
        masks["mel_loss_mask"],
        label_smoothing=label_smoothing,
    )
    total = float(mel_loss_weight) * mel_loss + float(text_loss_weight) * text_loss
    with torch.no_grad():
        correct = mel_logits.argmax(dim=-1).eq(masks["mel_targets"])
        valid = masks["mel_loss_mask"]
        mel_accuracy = (correct & valid).sum().float() / valid.sum().clamp_min(1)
    metrics = {
        "loss": total,
        "mel_loss": mel_loss,
        "text_loss": text_loss,
        "mel_accuracy": mel_accuracy,
        "mel_tokens": masks["mel_loss_mask"].sum(),
        "text_tokens": masks["text_loss_mask"].sum(),
    }
    return total, metrics


class TokenMetrics:
    """Aggregate each objective over its valid tokens, independent of batching."""

    def __init__(self) -> None:
        self.mel_sum = self.text_sum = self.correct = 0.0
        self.mel_tokens = self.text_tokens = 0

    def update(self, metrics: Mapping[str, Any]) -> None:
        mel_tokens = int(metrics["mel_tokens"])
        text_tokens = int(metrics["text_tokens"])
        self.mel_sum += float(metrics["mel_loss"]) * mel_tokens
        self.text_sum += float(metrics["text_loss"]) * text_tokens
        self.correct += float(metrics["mel_accuracy"]) * mel_tokens
        self.mel_tokens += mel_tokens
        self.text_tokens += text_tokens

    def result(self, mel_weight: float = 1.0, text_weight: float = 0.1) -> dict[str, float | None]:
        if not self.mel_tokens or not self.text_tokens:
            return dict.fromkeys(("loss", "mel_loss", "text_loss", "accuracy"))
        mel = self.mel_sum / self.mel_tokens
        text = self.text_sum / self.text_tokens
        return {"loss": mel_weight * mel + text_weight * text, "mel_loss": mel,
                "text_loss": text, "accuracy": self.correct / self.mel_tokens}


__all__ = [
    "build_gpt_training_inputs",
    "enable_gradient_checkpointing",
    "gpt_train_step_logits",
    "gpt_train_step_loss",
    "masked_cross_entropy",
    "TokenMetrics",
]
