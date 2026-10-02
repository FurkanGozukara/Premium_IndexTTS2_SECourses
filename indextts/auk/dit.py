"""AuK's Flux2Edit diffusion transformer (double-stream MMDiT then single-stream DiT).

Parameter names match the upstream checkpoint (``transformer.*``). Shapes:
b batch, n audio sequence (reference prompt then noised target), nt text tokens.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


class RotaryEmbedding(nn.Module):
    """x_transformers' rotary embedding (interleaved pairs, base 10000, no xpos)."""

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    @torch.autocast("cuda", enabled=False)
    def forward_from_seq_len(self, seq_len: int) -> torch.Tensor:
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        return torch.stack((freqs, freqs), dim=-1).flatten(-2)  # [n, dim]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


@torch.autocast("cuda", enabled=False)
def apply_rotary(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Rotate [b, h, n, d] by [n, d] frequencies in float32, returning t's dtype."""
    orig_dtype = t.dtype
    freqs = freqs[-t.shape[-2]:, :]
    rotated = t * freqs.cos() + _rotate_half(t) * freqs.sin()
    return rotated.to(orig_dtype)


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, scale: float = 1000) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 31, groups: int = 16):
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        if mask is not None:
            mask = mask.unsqueeze(1)
            x = x.masked_fill(~mask, 0.0)
        for index, block in enumerate(self.conv1d):
            x = block(x)
            if mask is not None and isinstance(block, nn.Conv1d):
                x = x.masked_fill(~mask, 0.0)
        return x.permute(0, 2, 1)


class AdaLayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormFinal(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)
        return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim: int, mult: float = 3.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.linear_in = nn.Linear(dim, inner_dim * 2, bias=False)
        self.linear_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x1, x2 = self.linear_in(x).chunk(2, dim=-1)
        return self.linear_out(F.silu(x1) * x2)


def _sdpa(query, key, value, mask):
    """Attention over [b, h, n, d] with an optional [b, n_key] padding mask."""
    if mask is not None:
        mask = mask[:, None, None, :]
    return F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=0.0, is_causal=False)


class Attention(nn.Module):
    """Self attention (DiT) or joint audio-text attention (MMDiT, ``context_dim``)."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0,
                 context_dim: int | None = None, mask_enabled: bool = True):
        super().__init__()
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.mask_enabled = mask_enabled
        self.context_dim = context_dim
        self.to_qkv = nn.Linear(dim, 3 * self.inner_dim)
        self.q_norm = nn.RMSNorm(dim_head, elementwise_affine=True)
        self.k_norm = nn.RMSNorm(dim_head, elementwise_affine=True)
        if context_dim is not None:
            self.to_qkv_c = nn.Linear(context_dim, 3 * self.inner_dim)
            self.c_q_norm = nn.RMSNorm(dim_head, elementwise_affine=True)
            self.c_k_norm = nn.RMSNorm(dim_head, elementwise_affine=True)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, dim), nn.Dropout(dropout)])
        if context_dim is not None:
            self.to_out_c = nn.Linear(self.inner_dim, context_dim)

    def _heads(self, projected):
        batch = projected.shape[0]
        query, key, value = projected.chunk(3, dim=-1)
        shape = (batch, -1, self.heads, query.shape[-1] // self.heads)
        return (query.view(shape).transpose(1, 2), key.view(shape).transpose(1, 2),
                value.view(shape).transpose(1, 2))

    def forward(self, x, c=None, mask=None, rope=None, c_rope=None, c_mask=None):
        batch = x.shape[0]
        query, key, value = self._heads(self.to_qkv(x))
        query, key = self.q_norm(query), self.k_norm(key)
        if rope is not None:
            query, key = apply_rotary(query, rope), apply_rotary(key, rope)
        if c is None:
            attn_mask = mask if self.mask_enabled else None
            out = _sdpa(query, key, value, attn_mask)
            out = out.transpose(1, 2).reshape(batch, -1, self.inner_dim).to(query.dtype)
            out = self.to_out[1](self.to_out[0](out))
            if mask is not None:
                out = out.masked_fill(~mask.unsqueeze(-1), 0.0)
            return out

        c_query, c_key, c_value = self._heads(self.to_qkv_c(c))
        c_query, c_key = self.c_q_norm(c_query), self.c_k_norm(c_key)
        if c_rope is not None:
            c_query, c_key = apply_rotary(c_query, c_rope), apply_rotary(c_key, c_rope)
        query = torch.cat([query, c_query], dim=2)
        key = torch.cat([key, c_key], dim=2)
        value = torch.cat([value, c_value], dim=2)
        joint_mask = None
        if self.mask_enabled and mask is not None:
            joint_mask = torch.cat([mask, c_mask], dim=1) if c_mask is not None else F.pad(mask, (0, c.shape[1]), value=True)
        out = _sdpa(query, key, value, joint_mask)
        out = out.transpose(1, 2).reshape(batch, -1, self.inner_dim).to(query.dtype)
        audio_len = x.shape[1]
        x_out, c_out = out[:, :audio_len], out[:, audio_len:]
        x_out = self.to_out[1](self.to_out[0](x_out))
        c_out = self.to_out_c(c_out)
        if mask is not None:
            x_out = x_out.masked_fill(~mask.unsqueeze(-1), 0.0)
        if c_mask is not None:
            c_out = c_out.masked_fill(~c_mask.unsqueeze(-1), 0.0)
        return x_out, c_out


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1, mask_enabled=True):
        super().__init__()
        self.attn_norm = AdaLayerNorm(dim)
        self.attn = Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout, mask_enabled=mask_enabled)
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = SwiGLUFeedForward(dim=dim, mult=ff_mult)

    def forward(self, x, t, mask=None, rope=None):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
        x = x + gate_msa.unsqueeze(1) * self.attn(x=norm, mask=mask, rope=rope)
        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        return x + gate_mlp.unsqueeze(1) * self.ff(norm)


class MMDiTBlock(nn.Module):
    """Joint text-audio block; ``_c`` is the text context, ``_x`` the audio."""

    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1, mask_enabled=True):
        super().__init__()
        self.attn_norm_c = AdaLayerNorm(dim)
        self.attn_norm_x = AdaLayerNorm(dim)
        self.attn = Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout,
                              context_dim=dim, mask_enabled=mask_enabled)
        self.ff_norm_c = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_c = SwiGLUFeedForward(dim=dim, mult=ff_mult)
        self.ff_norm_x = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_x = SwiGLUFeedForward(dim=dim, mult=ff_mult)

    def forward(self, x, c, t, mask=None, rope=None, c_rope=None, c_mask=None):
        norm_c, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.attn_norm_c(c, emb=t)
        norm_x, x_gate_msa, x_shift_mlp, x_scale_mlp, x_gate_mlp = self.attn_norm_x(x, emb=t)
        x_attn, c_attn = self.attn(x=norm_x, c=norm_c, mask=mask, rope=rope, c_rope=c_rope, c_mask=c_mask)
        c = c + c_gate_msa.unsqueeze(1) * c_attn
        norm_c = self.ff_norm_c(c) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        c = c + c_gate_mlp.unsqueeze(1) * self.ff_c(norm_c)
        x = x + x_gate_msa.unsqueeze(1) * x_attn
        norm_x = self.ff_norm_x(x) * (1 + x_scale_mlp[:, None]) + x_shift_mlp[:, None]
        x = x + x_gate_mlp.unsqueeze(1) * self.ff_x(norm_x)
        return c, x


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, freq_embed_dim: int = 256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep):
        hidden = self.time_embed(timestep).to(timestep.dtype)
        return self.time_mlp(hidden)


class AudioPromptEmbedding(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(out_dim)

    def embed(self, x, mask=None):
        x = self.linear(x)
        return self.conv_pos_embed(x, mask=mask) + x


class Flux2Edit(nn.Module):
    def __init__(self, *, dim, heads=8, dim_head=64, dropout=0.1, ff_mult=4, latent_dim=64,
                 text_hidden_dim=2048, checkpoint_activations=False, checkpoint_every_n_layers=1,
                 attn_mask_enabled=True, num_layers=8, num_single_layers=24, **_ignored):
        super().__init__()
        self.dim = dim
        self.time_embed = TimestepEmbedding(dim)
        self.txt_norm = nn.RMSNorm(dim, elementwise_affine=True)
        self.txt_proj = nn.Linear(text_hidden_dim, dim)
        self.audio_embed = AudioPromptEmbedding(latent_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim_head)
        self.transformer_blocks = nn.ModuleList([
            MMDiTBlock(dim, heads, dim_head, ff_mult=ff_mult, dropout=dropout, mask_enabled=attn_mask_enabled)
            for _ in range(num_layers)])
        self.single_transformer_blocks = nn.ModuleList([
            DiTBlock(dim, heads, dim_head, ff_mult=ff_mult, dropout=dropout, mask_enabled=attn_mask_enabled)
            for _ in range(num_single_layers)])
        self.norm_out = AdaLayerNormFinal(dim)
        self.proj_out = nn.Linear(dim, latent_dim)
        self.checkpoint_activations = bool(checkpoint_activations)
        self.checkpoint_every_n_layers = max(1, int(checkpoint_every_n_layers))
        self._text_cache = None

    def clear_cache(self):
        self._text_cache = None

    def project_text(self, text, drop_text=False):
        c = self.txt_norm(self.txt_proj(text))
        return torch.zeros_like(c) if drop_text else c

    def _embed_audio(self, x, ref, drop_audio_cond, mask, ref_mask):
        x_emb = self.audio_embed.embed(x, mask=mask)
        if ref is None or ref.shape[1] == 0:
            return x_emb, mask, 0
        if drop_audio_cond:
            ref = torch.zeros_like(ref)
        ref_emb = self.audio_embed.embed(ref, mask=ref_mask)
        audio_mask = None
        if mask is not None or ref_mask is not None:
            batch, n = x_emb.shape[:2]
            if mask is None:
                mask = torch.ones(batch, n, dtype=torch.bool, device=x_emb.device)
            if ref_mask is None:
                ref_mask = torch.ones(batch, ref_emb.shape[1], dtype=torch.bool, device=x_emb.device)
            audio_mask = torch.cat([ref_mask, mask], dim=1)
        return torch.cat([ref_emb, x_emb], dim=1), audio_mask, ref_emb.shape[1]

    def _run_block(self, index, block, *args):
        if self.checkpoint_activations and self.training and index % self.checkpoint_every_n_layers == 0:
            return checkpoint(block, *args, use_reentrant=False)
        return block(*args)

    def forward(self, x, text, time, mask=None, c_mask=None, drop_audio_cond=False, drop_text=False,
                cfg_infer=False, cache=False, ref=None, ref_mask=None, dense=False):
        """``dense`` declares that no sequence is padded (one sample, or equal
        lengths): every mask is then all True and is skipped, which lets SDPA
        use its fastest kernel; masking with all-True masks changes nothing."""
        batch = x.shape[0]
        if time.ndim == 0:
            time = time.repeat(batch)
        t = self.time_embed(time)
        if dense:
            mask = ref_mask = c_mask = None
        elif c_mask is None:
            c_mask = text.abs().sum(-1) > 0
        if cfg_infer:
            # Conditional and unconditional halves in one batch; the projected
            # text of both is constant across ODE steps and cached.
            if cache and self._text_cache is not None:
                c_cond, c_uncond = self._text_cache
            else:
                c_cond = self.project_text(text)
                c_uncond = torch.zeros_like(c_cond)
                if cache:
                    self._text_cache = (c_cond, c_uncond)
            x_cond, mask_cond, prompt_len = self._embed_audio(x, ref, False, mask, ref_mask)
            x_uncond, mask_uncond, _ = self._embed_audio(x, ref, True, mask, ref_mask)
            x = torch.cat((x_cond, x_uncond), dim=0)
            c = torch.cat((c_cond, c_uncond), dim=0)
            t = torch.cat((t, t), dim=0)
            audio_mask = torch.cat((mask_cond, mask_uncond), dim=0) if mask_cond is not None else None
            c_mask = torch.cat((c_mask, c_mask), dim=0) if c_mask is not None else None
        else:
            if cache and self._text_cache is not None and not drop_text:
                c = self._text_cache[0]
            else:
                c = self.project_text(text, drop_text=drop_text)
                if cache and not drop_text:
                    self._text_cache = (c, torch.zeros_like(c))
            x, audio_mask, prompt_len = self._embed_audio(x, ref, drop_audio_cond, mask, ref_mask)

        seq_len, text_len = x.shape[1], c.shape[1]
        rope_audio = self.rotary_embed.forward_from_seq_len(seq_len)
        rope_text = self.rotary_embed.forward_from_seq_len(text_len)
        for index, block in enumerate(self.transformer_blocks):
            c, x = self._run_block(index, block, x, c, t, audio_mask, rope_audio, rope_text, c_mask)
        x = torch.cat([c, x], dim=1)
        rope = self.rotary_embed.forward_from_seq_len(text_len + seq_len)
        single_mask = torch.cat([c_mask, audio_mask], dim=1) if audio_mask is not None else None
        for index, block in enumerate(self.single_transformer_blocks):
            x = self._run_block(index, block, x, t, single_mask, rope)
        x = x[:, text_len + prompt_len:]
        return self.proj_out(self.norm_out(x, t))
