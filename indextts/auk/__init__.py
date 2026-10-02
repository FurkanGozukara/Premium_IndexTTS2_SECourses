"""Tencent AuK speech generation and editing model, vendored for this application.

Source: https://github.com/Tencent-Hunyuan/AuK (MIT, Copyright (C) 2026 Tencent),
revision da1f31b. The VAE adapts NVIDIA BigVGAN (MIT), alias-free-torch
(Apache-2.0), julius (MIT) and snake (MIT); see ``LICENSES.md`` beside this file.

Changes from upstream: audio is read by the application (no torchaudio or
torchcodec I/O, no qwen_omni_utils), rotary embeddings and the fixed-grid ODE
solver are implemented here (no x_transformers or torchdiffeq), attention uses
PyTorch SDPA, weight normalisation is folded into the VAE weights at load time,
and no module configures the root logger. Numerics follow the upstream model.

Importing this package does not import PyTorch model code; the submodules do.
"""

AUK_REPO = "tencent/AuK"
QWEN_OMNI_REPO = "Qwen/Qwen2.5-Omni-3B"
SAMPLE_RATE = 24000
LATENT_RATE = 50  # 24 kHz audio / 480x VAE downsampling
ENCODER_SAMPLE_RATE = 16000  # the Qwen2.5-Omni audio encoder's input rate
MAX_CONTEXT_SECONDS = 30.0  # upstream training clips: 0.3-30 s per side


def text_encoder_folder(model_dir):
    """The Qwen2.5-Omni folder to load: the slim Thinker-only copy when present, else the full snapshot."""
    from pathlib import Path

    root = Path(model_dir)
    slim = root / "quantized" / "AuK" / "qwen2_5_omni_thinker"
    return slim if (slim / "config.json").is_file() and (slim / "tokenizer.json").is_file() else root / "qwen2_5_omni_3b"
