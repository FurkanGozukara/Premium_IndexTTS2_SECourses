"""Convert public OmniVoice transformer weights using the shared HQ ConvRot converter."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    from safetensors import safe_open
    from indextts.backends.omnivoice import ensure_model, MODEL_REPO, QUANT_FILENAME
    from indextts.quant.convrot_int8 import convert_gpt_checkpoint
    folder, _ = ensure_model(args.model_dir)
    source = folder / "model.safetensors"
    projections = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                   "gate_proj.weight", "up_proj.weight", "down_proj.weight")
    with safe_open(str(source), framework="pt") as handle:
        targets = [key.removeprefix("llm.").removesuffix(".weight") for key in handle.keys()
                   if key.startswith("llm.layers.") and key.endswith(projections)]
    output = args.output or args.model_dir / "quantized" / QUANT_FILENAME
    return convert_gpt_checkpoint(str(source), str(output), device=args.device,
        linear_targets=targets, state_prefix="llm.", model_id=MODEL_REPO,
        progress=lambda message: print(message, flush=True))


if __name__ == "__main__":
    main()
