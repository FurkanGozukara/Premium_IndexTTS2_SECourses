"""AuK's conditional flow matching around the Flux2Edit transformer.

``AukModel`` holds the trainable part of the upstream ``CFMEdit`` checkpoint: the
transformer and the learned fusion of the Qwen2.5-Omni hidden layers
(``layer_weights``, ``layer_scale``). The frozen Qwen encoder lives in
``conditioning.QwenConditioner``.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torch import nn

from .dit import Flux2Edit


def lens_to_mask(lengths: torch.Tensor, length: int | None = None) -> torch.Tensor:
    length = int(lengths.amax()) if length is None else int(length)
    return torch.arange(length, device=lengths.device)[None, :] < lengths[:, None]


def sway_time_grid(steps: int, sway_coef: float | None, device=None) -> torch.Tensor:
    t = torch.linspace(0, 1, int(steps) + 1, device=device, dtype=torch.float32)
    if sway_coef is not None:
        t = t + float(sway_coef) * (torch.cos(torch.pi / 2 * t) - 1 + t)
    return t


class AukModel(nn.Module):
    def __init__(self, arch: dict, latent_dim: int = 64, num_text_layers: int = 36,
                 audio_drop_prob: float = 0.3, cond_drop_prob: float = 0.2,
                 t_sampling: str = "logistic_normal", P_mean: float = -0.8, P_std: float = 0.8, **_ignored):
        super().__init__()
        self.transformer = Flux2Edit(**arch, latent_dim=latent_dim)
        self.num_channels = latent_dim
        self.layer_weights = nn.Parameter(torch.zeros(num_text_layers))
        self.layer_scale = nn.Parameter(torch.ones(1))
        self.audio_drop_prob, self.cond_drop_prob = float(audio_drop_prob), float(cond_drop_prob)
        self.t_sampling, self.P_mean, self.P_std = t_sampling, float(P_mean), float(P_std)

    def fuse(self, hidden_states) -> torch.Tensor:
        """ELMo-style weighted average of the layer-normalised Qwen layers (embeddings excluded)."""
        d_llm = hidden_states[0].shape[-1]
        stacked = torch.stack([F.layer_norm(h, [d_llm]) for h in hidden_states[1:]], dim=0)
        weights = F.softmax(self.layer_weights, dim=0)
        return (stacked * weights[:, None, None, None]).sum(dim=0) * self.layer_scale

    # ------------------------------------------------------------------ inference

    @torch.no_grad()
    def sample(self, text, context_mask, ref_latent, ref_lens, target_lens, *, steps=32, cfg_strength=2.0,
               sway_sampling_coef=-1.0, t_grid=None, noise=None, method="euler", step_callback=None):
        """Integrate the flow from noise to target latents.

        text: fused text conditioning [b, nt, 2048]; context_mask [b, nt] bool;
        ref_latent [b, np, 64] normalised reference latents (np may be 0);
        ref_lens / target_lens [b]. Returns the target latents [b, max_target, 64]
        (float32). ``noise`` overrides the initial Gaussian state.
        """
        self.eval()
        device = text.device
        batch = text.shape[0]
        ref_latent = ref_latent.to(device=device, dtype=torch.float32)
        ref_lens = ref_lens.to(device)
        target_lens = target_lens.to(device).clamp(min=1)
        ref_mask = lens_to_mask(ref_lens, length=ref_latent.shape[1]) if ref_latent.shape[1] else None
        max_target = int(target_lens.amax())
        target_mask = lens_to_mask(target_lens, length=max_target)
        # Masks are skipped when no sequence is padded; this is decided once per
        # call rather than on every attention call (each check would wait for the GPU).
        dense = bool(context_mask.all()) and bool(target_mask.all()) and (ref_mask is None or bool(ref_mask.all()))
        if noise is None:
            noise = torch.randn(batch, max_target, self.num_channels, device=device, dtype=torch.float32)
        y = noise.to(device=device, dtype=torch.float32) * target_mask[..., None]
        ref = ref_latent if ref_latent.shape[1] else None
        transformer = self.transformer

        def velocity(t, x):
            kwargs = dict(x=x, text=text, time=t, mask=target_mask, c_mask=context_mask, ref=ref, ref_mask=ref_mask,
                          cache=True, dense=dense)
            if cfg_strength < 1e-5:
                return transformer(**kwargs).float()
            v_cond, v_uncond = torch.chunk(transformer(cfg_infer=True, **kwargs).float(), 2, dim=0)
            return v_cond + (v_cond - v_uncond) * cfg_strength

        grid = (torch.tensor(t_grid, device=device, dtype=torch.float32) if t_grid is not None
                else sway_time_grid(steps, sway_sampling_coef, device))
        try:
            for index in range(len(grid) - 1):
                t0, t1 = grid[index], grid[index + 1]
                dt = t1 - t0
                if method == "midpoint":
                    half = y + 0.5 * dt * velocity(t0, y)
                    y = y + dt * velocity(t0 + 0.5 * dt, half)
                else:
                    y = y + dt * velocity(t0, y)
                if step_callback is not None:
                    step_callback(index + 1, len(grid) - 1)
        finally:
            transformer.clear_cache()
        return y * target_mask[..., None]

    # ------------------------------------------------------------------ training

    def sample_time(self, batch, device):
        if self.t_sampling == "uniform":
            return torch.rand((batch,), device=device)
        z = torch.randn((batch,), device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, target, text, context_mask, *, ref_latent, ref_lens, target_lens, time=None, x0=None,
                apply_cond_drop=True):
        """Flow-matching loss of normalised target latents [b, n, 64] (masked mean MSE)."""
        batch, seq_len = target.shape[:2]
        device = target.device
        mask = lens_to_mask(target_lens.to(device), length=seq_len)
        x1 = target.float()
        x0 = torch.randn_like(x1) if x0 is None else x0
        time = self.sample_time(batch, device) if time is None else time
        t = time.unsqueeze(-1).unsqueeze(-1)
        phi = (1 - t) * x0 + t * x1
        flow = x1 - x0
        drop_audio_cond = drop_text = False
        if apply_cond_drop:
            drop_audio_cond = random.random() < self.audio_drop_prob
            if random.random() < self.cond_drop_prob:
                drop_audio_cond = drop_text = True
        ref_mask = lens_to_mask(ref_lens.to(device), length=ref_latent.shape[1]) if ref_latent.shape[1] else None
        v_pred = self.transformer(x=phi, text=text, time=time, drop_audio_cond=drop_audio_cond, drop_text=drop_text,
                                  mask=mask, c_mask=context_mask, ref=ref_latent if ref_latent.shape[1] else None,
                                  ref_mask=ref_mask)
        loss = F.mse_loss(v_pred.float(), flow, reduction="none")
        return loss[mask].mean()
