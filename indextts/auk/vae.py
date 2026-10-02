"""AuK's causal BigVGAN VAE: 24 kHz audio <-> 50 Hz, 64-dimensional latents.

Adapted from NVIDIA BigVGAN (MIT), alias-free-torch (Apache-2.0), julius (MIT)
and snake (MIT) as shipped with Tencent AuK. Weight normalisation is folded into
plain weights when the checkpoint loads (the forward pass computes the same
weights upstream recomputes on every call). The posterior flow is only used to
train the VAE and is not built.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    attenuation = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    time = torch.arange(-half_size, half_size) + 0.5 if even else torch.arange(kernel_size) - half_size
    filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    filter_ /= filter_.sum()
    return filter_.view(1, 1, kernel_size)


class LowPassFilter1d(nn.Module):
    def __init__(self, cutoff=0.5, half_width=0.6, stride=1, kernel_size=12, causal=False):
        super().__init__()
        if causal:
            self.pad_left, self.pad_right = kernel_size - 1, 0
        else:
            even = kernel_size % 2 == 0
            self.pad_left, self.pad_right = kernel_size // 2 - int(even), kernel_size // 2
        self.stride = stride
        self.register_buffer("filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size))

    def forward(self, x):
        channels = x.shape[1]
        x = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")
        return F.conv1d(x, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels)


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * ratio + (self.kernel_size - ratio) // 2
        self.pad_right = self.pad * ratio + (self.kernel_size - ratio + 1) // 2
        self.register_buffer("filter", kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, self.kernel_size))

    def forward(self, x):
        channels = x.shape[1]
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(x, self.filter.expand(channels, -1, -1), stride=self.ratio, groups=channels)
        return x[..., self.pad_left:-self.pad_right]


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None, causal=False):
        super().__init__()
        kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(0.5 / ratio, 0.6 / ratio, stride=ratio, kernel_size=kernel_size, causal=causal)

    def forward(self, x):
        return self.lowpass(x)


class SnakeBeta(nn.Module):
    """x + 1/beta * sin^2(alpha * x) with log-scale alpha and beta."""

    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        alpha = torch.exp(self.alpha.unsqueeze(0).unsqueeze(-1))
        beta = torch.exp(self.beta.unsqueeze(0).unsqueeze(-1))
        return x + (1.0 / (beta + 1e-9)) * torch.pow(torch.sin(x * alpha), 2)


class Activation1d(nn.Module):
    def __init__(self, activation, causal=False):
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(2, 12)
        self.downsample = DownSample1d(2, 12, causal=causal)

    def forward(self, x):
        return self.downsample(self.act(self.upsample(x)))


class CausalConv1d(nn.Conv1d):
    """Conv1d with upstream's padding rule: left padding only when causal."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, bias=True, causal=False):
        self.causal = causal
        self.left_padding = dilation * (kernel_size - 1) if causal else 0
        padding = 0 if causal else int((kernel_size * dilation - dilation) / 2)
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

    def forward(self, x):
        if self.causal:
            x = F.pad(x, (self.left_padding, 0))
        return super().forward(x)


class CausalConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, causal=False):
        padding = 0 if causal else (kernel_size - stride) // 2
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.causal = causal
        self.trim = int(stride)

    def forward(self, x):
        x = super().forward(x)
        return x[:, :, :-self.trim] if self.causal else x


class EncoderConv(nn.Module):
    """Upstream ``Conv1d_S``: symmetric padding; the weight lives in ``.layer``."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super().__init__()
        self.layer = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=(kernel_size - 1) // 2)

    def forward(self, x):
        return self.layer(x)


class ResStack(nn.Module):
    def __init__(self, channel, kernel_size=3, base=3, nums=4):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(),
                nn.Conv1d(channel, channel, kernel_size=kernel_size, dilation=base ** i, padding=base ** i),
                nn.LeakyReLU(),
                nn.Conv1d(channel, channel, kernel_size=kernel_size, dilation=1, padding=1),
            ) for i in range(nums)])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self, out_channels, channels, down_sample_factors, base_channels=12):
        super().__init__()
        layers = [EncoderConv(1, base_channels, kernel_size=3), nn.LeakyReLU(0.2, True)]
        for (in_c, out_c), factor in zip(zip(channels[:-1], channels[1:]), down_sample_factors):
            layers += [EncoderConv(in_c, out_c, kernel_size=factor * 2, stride=factor),
                       ResStack(out_c, 3, 2, 6), nn.LeakyReLU(0.2, True)]
        layers += [EncoderConv(channels[-1], out_channels * 2, 3)]
        self.generator = nn.Sequential(*layers)

    def forward(self, x):
        return self.generator(x)


class AMPBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), causal=True, act_causal=False):
        super().__init__()
        self.convs1 = nn.ModuleList([CausalConv1d(channels, channels, kernel_size, 1, dilation=d, causal=causal)
                                     for d in dilation])
        self.convs2 = nn.ModuleList([CausalConv1d(channels, channels, kernel_size, 1, dilation=1, causal=causal)
                                     for _ in dilation])
        self.activations = nn.ModuleList([Activation1d(SnakeBeta(channels), causal=act_causal)
                                          for _ in range(len(self.convs1) + len(self.convs2))])

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            x = c2(a2(c1(a1(x)))) + x
        return x


DEFAULT_CONFIG = {
    "upsample_rates": [5, 4, 3, 2, 2, 2], "upsample_kernel_sizes": [10, 8, 6, 4, 4, 4],
    "upsample_initial_channel": 1536, "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "downsample_rates": [2, 2, 2, 3, 4, 5], "downsample_channels": [12, 24, 48, 96, 192, 384, 768],
    "latent_dim": 64, "causal": True, "act_causal": True,
}


class AukVAE(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        h = {**DEFAULT_CONFIG, **{key: value for key, value in (config or {}).items() if key in DEFAULT_CONFIG}}
        self.latent_dim = int(h["latent_dim"])
        self.hop_size = math.prod(h["downsample_rates"])
        causal, act_causal = bool(h["causal"]), bool(h["act_causal"])
        self.register_buffer("global_mean", torch.zeros(self.latent_dim))
        self.register_buffer("global_log_std", torch.ones(self.latent_dim))
        self.audio_encoder = Encoder(self.latent_dim, h["downsample_channels"], h["downsample_rates"])
        self.num_kernels = len(h["resblock_kernel_sizes"])
        initial = int(h["upsample_initial_channel"])
        self.conv_pre = CausalConv1d(self.latent_dim, initial, 7, 1, causal=False)
        self.ups = nn.ModuleList([
            nn.ModuleList([CausalConvTranspose1d(initial // 2 ** i, initial // 2 ** (i + 1), k, u, causal=causal)])
            for i, (u, k) in enumerate(zip(h["upsample_rates"], h["upsample_kernel_sizes"]))])
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            channels = initial // 2 ** (i + 1)
            for k, d in zip(h["resblock_kernel_sizes"], h["resblock_dilation_sizes"]):
                self.resblocks.append(AMPBlock1(channels, k, d, causal=causal, act_causal=act_causal))
        self.activation_post = Activation1d(SnakeBeta(channels), causal=act_causal)
        self.conv_post = CausalConv1d(channels, 1, 7, 1, bias=False, causal=causal)

    def load_checkpoint(self, state_dict):
        """Load an upstream checkpoint, folding ``weight_g``/``weight_v`` pairs."""
        folded = {}
        for key, value in state_dict.items():
            if key.startswith("flow."):
                continue
            if key.endswith(".weight_g"):
                continue
            if key.endswith(".weight_v"):
                stem = key[: -len("_v")]
                folded[stem] = torch._weight_norm(value.float(), state_dict[stem + "_g"].float(), 0)
                continue
            folded[key] = value
        missing, unexpected = self.load_state_dict(folded, strict=False)
        missing = [key for key in missing if not key.endswith(".filter")]
        if missing or unexpected:
            raise RuntimeError(f"AuK VAE checkpoint mismatch: missing {missing[:8]}, unexpected {unexpected[:8]}")
        return self

    @torch.autocast("cuda", enabled=False)
    def encode(self, audio, lengths=None, *, sample=True):
        """[b, 1, samples] float audio -> normalised latents [b, frames, 64] and frame lengths."""
        stats = self.audio_encoder(audio.float())
        if lengths is None:
            lengths = torch.full((audio.shape[0],), audio.shape[-1], dtype=torch.long, device=audio.device)
        latent_lengths = lengths // self.hop_size
        mean, log_std = stats.chunk(2, 1)
        latents = mean + torch.randn_like(mean) * torch.exp(log_std) if sample else mean
        latents = latents.transpose(1, 2).float()
        latents = (latents - self.global_mean.float()) / torch.sqrt(self.global_log_std.float())
        return latents, torch.clamp(latent_lengths, max=latents.shape[1])

    def denormalize(self, latents):
        return latents.float() * torch.sqrt(self.global_log_std.float()) + self.global_mean.float()

    @torch.autocast("cuda", enabled=False)
    def decode(self, latents):
        """Denormalised latents [b, 64, frames] -> audio [b, 1, frames * 480] in [-1, 1]."""
        x = self.conv_pre(latents.float())
        for i, ups in enumerate(self.ups):
            for up in ups:
                x = up(x)
            xs = None
            for j in range(self.num_kernels):
                y = self.resblocks[i * self.num_kernels + j](x)
                xs = y if xs is None else xs + y
            x = xs / self.num_kernels
        x = self.conv_post(self.activation_post(x))
        return torch.clamp(x, min=-1.0, max=1.0)
