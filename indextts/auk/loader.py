"""Build AuK's transformer and VAE from the public checkpoint folder."""

from __future__ import annotations

from pathlib import Path

import torch

from .cfm import AukModel
from .dit import RotaryEmbedding
from .vae import AukVAE

DEFAULT_ARCH = {
    "dim": 1536, "heads": 24, "ff_mult": 2, "text_hidden_dim": 2048, "attn_mask_enabled": True,
    "checkpoint_activations": False, "checkpoint_every_n_layers": 4, "num_layers": 10, "num_single_layers": 20,
}
DEFAULT_SCHEDULE = {"t_sampling": "logistic_normal", "P_mean": -0.8, "P_std": 0.8}
# Small tensors kept in float32 by the INT8 conversion and by BF16 storage.
FLOAT32_SUFFIXES = ("layer_weights", "layer_scale", "rotary_embed.inv_freq", "q_norm.weight", "k_norm.weight",
                    "c_q_norm.weight", "c_k_norm.weight", "txt_norm.weight")


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


def empty_model(config: dict, num_text_layers: int = 36) -> AukModel:
    """The model structure without allocated weights (meta tensors)."""
    with torch.device("meta"):
        return AukModel(config["arch"], latent_dim=config["latent_dim"], num_text_layers=num_text_layers,
                        **config["schedule"])


def _finish(model: AukModel) -> AukModel:
    rotary = model.transformer.rotary_embed
    rotary.inv_freq = RotaryEmbedding(rotary.inv_freq.numel() * 2).inv_freq
    remaining = [name for name, tensor in [*model.named_parameters(), *model.named_buffers()] if tensor.is_meta]
    if remaining:
        raise RuntimeError(f"AuK checkpoint is missing {len(remaining)} tensors: {remaining[:6]}")
    model.requires_grad_(False).eval()
    return model


def build_model(config: dict, state: dict, *, device="cpu", dtype=torch.float32,
                num_text_layers: int = 36) -> AukModel:
    """Assign checkpoint tensors into the model (no second copy of the 6 GB weights)."""
    model = empty_model(config, num_text_layers)
    state = {key: value for key, value in state.items() if not key.startswith("text_encoder.")}
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    missing = [key for key in missing if not key.endswith("rotary_embed.inv_freq")]
    if missing or unexpected:
        raise RuntimeError(f"AuK checkpoint mismatch: missing {missing[:8]}, unexpected {unexpected[:8]}")
    return cast_weights(_finish(model), dtype).to(device)


def build_int8_model(config: dict, path: str | Path, *, device="cuda", num_text_layers: int = 36) -> AukModel:
    """The ConvRot INT8 transformer; tensors stored in float32 by the converter stay float32."""
    from safetensors import safe_open

    from indextts.quant.convrot_int8 import load_gpt_checkpoint

    model = empty_model(config, num_text_layers)
    load_gpt_checkpoint(model, str(path), device=device, dtype=torch.bfloat16, strict=False)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.endswith(FLOAT32_SUFFIXES) and handle.get_slice(key).get_dtype() == "F32":
                module_path, _, leaf = key.rpartition(".")
                owner = model.get_submodule(module_path) if module_path else model
                value = handle.get_tensor(key).to(device)
                if leaf in owner._parameters:
                    owner._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)
                else:
                    owner._buffers[leaf] = value
    model = _finish(model)
    model.transformer.rotary_embed.inv_freq = model.transformer.rotary_embed.inv_freq.to(device)
    return model


def cast_weights(model: torch.nn.Module, dtype=torch.float32) -> torch.nn.Module:
    """Store linear and convolution weights in ``dtype``.

    Upstream keeps the transformer in float32 and computes under BF16 autocast,
    which casts these weights to BF16 on every call. Stored BF16 weights give the
    same matrix products without the per-call cast (measured identical output);
    normalisation weights, the rotary frequencies and the layer-fusion weights stay float32.
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
