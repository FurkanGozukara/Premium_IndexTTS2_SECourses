"""Build AuK's transformer and VAE from the public checkpoint folder."""

from __future__ import annotations

from pathlib import Path

import torch

from .cfm import AukModel
from .vae import AukVAE

DEFAULT_ARCH = {
    "dim": 1536, "heads": 24, "ff_mult": 2, "text_hidden_dim": 2048, "attn_mask_enabled": True,
    "checkpoint_activations": False, "checkpoint_every_n_layers": 4, "num_layers": 10, "num_single_layers": 20,
}
DEFAULT_SCHEDULE = {"t_sampling": "logistic_normal", "P_mean": -0.8, "P_std": 0.8}


def read_config(folder: str | Path) -> dict:
    """The ``model`` section of ``config.yaml`` with upstream defaults filled in."""
    path = Path(folder) / "config.yaml"
    section = {}
    if path.is_file():
        from omegaconf import OmegaConf

        section = OmegaConf.to_container(OmegaConf.load(str(path)).model, resolve=False)
    arch = {**DEFAULT_ARCH, **{key: value for key, value in (section.get("arch") or {}).items()
                               if key not in {"attn_backend"}}}
    vae = dict(section.get("vae") or {})
    vae_kwargs = dict(vae.get("model_init_kwargs") or {})
    vae_kwargs["latent_dim"] = int(vae.get("latent_dim", 64))
    return {"name": section.get("name", "AuK"), "arch": arch,
            "schedule": {**DEFAULT_SCHEDULE, **(section.get("schedule") or {})},
            "latent_dim": int(vae.get("latent_dim", 64)), "vae": vae_kwargs}


def read_state(path: str | Path) -> dict:
    """Weights of a released ``.safetensors`` file or the EMA of an upstream ``.pt`` checkpoint."""
    path = str(path)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path, device="cpu")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    ema = checkpoint.get("ema_model_state_dict") or checkpoint.get("model_state_dict") or checkpoint
    return {key.replace("ema_model.", ""): value for key, value in ema.items() if key not in {"initted", "step"}}


def build_model(config: dict, state: dict | None = None, *, device="cpu", dtype=torch.float32,
                num_text_layers: int = 36) -> AukModel:
    model = AukModel(config["arch"], latent_dim=config["latent_dim"], num_text_layers=num_text_layers,
                     **config["schedule"])
    if state is not None:
        state = {key: value for key, value in state.items() if not key.startswith("text_encoder.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [key for key in missing if not key.endswith("rotary_embed.inv_freq")]
        if missing or unexpected:
            raise RuntimeError(f"AuK checkpoint mismatch: missing {missing[:8]}, unexpected {unexpected[:8]}")
    model.requires_grad_(False).eval()
    return cast_weights(model.to(device), dtype)


def cast_weights(model: torch.nn.Module, dtype=torch.float32) -> torch.nn.Module:
    """Store linear and convolution weights in ``dtype``.

    Upstream keeps the transformer in float32 and computes under BF16 autocast,
    which casts these weights to BF16 on every call. Stored BF16 weights give the
    same matrix products without the per-call cast; normalisation weights, the
    rotary frequencies and the layer-fusion weights stay float32.
    """
    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
            module.to(dtype)
    return model


def build_vae(config: dict, path: str | Path, *, device="cpu") -> AukVAE:
    from safetensors.torch import load_file

    vae = AukVAE(config["vae"]).load_checkpoint(load_file(str(path), device="cpu"))
    vae.requires_grad_(False).eval()
    return vae.to(device)
