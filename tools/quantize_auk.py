"""Convert Tencent AuK's diffusion transformer and its Qwen2.5-Omni Thinker text model to ConvRot INT8.

python tools/quantize_auk.py                       # both INT8 files into models/quantized/AuK
python tools/quantize_auk.py --part dit --dit-layers blocks_adaln --dit-output <file>
python tools/quantize_auk.py --part thinker        # the slim BF16 Thinker folder (no INT8)

The DiT file holds the whole AukModel state (``transformer.*``, ``layer_weights``,
``layer_scale``) with the float32 tensors of ``indextts.auk.loader.FLOAT32_SUFFIXES``
kept F32; the text file holds ``Qwen2_5OmniThinkerTextModel`` (keys relative to
``thinker.model``). A JSON report is written beside each file.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DIT_FILE = "auk_dit_convrot_int8.safetensors"
TEXT_FILE = "qwen_omni_thinker_convrot_int8.safetensors"
# Attention and SwiGLU projections of the 10 MMDiT and 20 DiT blocks.
_BLOCK = re.compile(r"^transformer\.(transformer_blocks|single_transformer_blocks)\.\d+\."
                    r"(attn\.(to_qkv|to_qkv_c|to_out\.0|to_out_c)|ff(_c|_x)?\.(linear_in|linear_out))\.weight$")
# The blocks' AdaLayerNorm modulation (dim -> 6 dim, about 37% of the transformer's weights).
_ADALN = re.compile(r"^transformer\.(transformer_blocks|single_transformer_blocks)\.\d+\.attn_norm(_c|_x)?\.linear\.weight$")
# Text projection, timestep MLP and final modulation; the audio input projection,
# the convolutional position embedding and the output projection always stay BF16.
_EXTRA = ("transformer.txt_proj.weight", "transformer.time_embed.time_mlp.0.weight",
          "transformer.time_embed.time_mlp.2.weight", "transformer.norm_out.linear.weight")
DIT_LAYER_SETS = {
    "blocks": "block attention and feed-forward projections",
    "blocks_adaln": "block projections and the blocks' AdaLayerNorm modulation",
    "all": "block projections, AdaLayerNorm modulation, text projection, timestep MLP and final modulation",
}
DEFAULT_DIT_LAYERS = "all"
_TEXT_PROJECTIONS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                     "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def dit_targets(keys, layers=DEFAULT_DIT_LAYERS):
    """Linear module paths of the AuK transformer quantized for a layer set."""
    if layers not in DIT_LAYER_SETS:
        raise ValueError(f"Unknown DiT layer set {layers!r}; choose one of {sorted(DIT_LAYER_SETS)}")
    selected = [key for key in keys if _BLOCK.match(key)]
    if layers in ("blocks_adaln", "all"):
        selected += [key for key in keys if _ADALN.match(key)]
    if layers == "all":
        selected += [key for key in _EXTRA if key in keys]
    return sorted(key.removesuffix(".weight") for key in selected)


def text_targets(keys, prefix):
    """The 36 decoder layers' attention and MLP projections (embeddings and norms stay BF16)."""
    pattern = re.compile(rf"^{re.escape(prefix)}layers\.\d+\.({'|'.join(map(re.escape, _TEXT_PROJECTIONS))})\.weight$")
    return sorted(key[len(prefix):].removesuffix(".weight") for key in keys if pattern.match(key))


def _source_keys(source: Path) -> list:
    from safetensors import safe_open
    from indextts.auk.thinker_files import weight_files

    files = weight_files(source) if source.is_dir() else [source]
    keys = []
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys.extend(handle.keys())
    return keys


def _annotate(report: dict, **fields) -> dict:
    report.update(fields)
    path = Path(report["report"])
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)
    return report


def convert_dit(model_dir: Path, output: Path, *, layers=DEFAULT_DIT_LAYERS, device="cuda:0", progress=print):
    from indextts.auk import AUK_REPO
    from indextts.auk.loader import FLOAT32_SUFFIXES
    from indextts.quant.convrot_int8 import convert_gpt_checkpoint

    source = model_dir / "auk" / "auk_base.safetensors"
    keys = [key for key in _source_keys(source) if not key.startswith("text_encoder.")]
    targets = dit_targets(keys, layers)
    report = convert_gpt_checkpoint(str(source), str(output), device=device, linear_targets=targets,
                                    state_prefix="", model_id=AUK_REPO, keep_float32=FLOAT32_SUFFIXES,
                                    progress=progress)
    return _annotate(report, auk_part="dit", dit_layers=layers, dit_layers_description=DIT_LAYER_SETS[layers])


def convert_text(model_dir: Path, output: Path, *, source: Path | None = None, device="cuda:0", progress=print):
    from indextts.auk import QWEN_OMNI_REPO
    from indextts.auk.thinker_files import thinker_folder
    from indextts.quant.convrot_int8 import convert_gpt_checkpoint

    source = Path(source) if source else thinker_folder(model_dir)
    keys = _source_keys(source)
    # The public snapshot names the text model thinker.model.*; the slim folder model.*.
    prefix = "thinker.model." if any(key.startswith("thinker.model.") for key in keys) else "model."
    targets = text_targets(keys, prefix)
    if len(targets) != 36 * len(_TEXT_PROJECTIONS):
        raise ValueError(f"Expected {36 * len(_TEXT_PROJECTIONS)} Thinker projections in {source}, found {len(targets)}")
    report = convert_gpt_checkpoint(str(source), str(output), device=device, linear_targets=targets,
                                    state_prefix=prefix, model_id=QWEN_OMNI_REPO, progress=progress)
    return _annotate(report, auk_part="text", text_model="Qwen2_5OmniThinkerTextModel (thinker.model)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=("dit", "text", "int8", "thinker", "all"), default="int8",
                        help="int8: both INT8 files; thinker: the slim BF16 Thinker folder; all: everything")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--output-dir", type=Path, help="default: <model-dir>/quantized/AuK")
    parser.add_argument("--dit-layers", choices=sorted(DIT_LAYER_SETS), default=DEFAULT_DIT_LAYERS)
    parser.add_argument("--dit-output", type=Path)
    parser.add_argument("--text-output", type=Path)
    parser.add_argument("--text-source", type=Path, help="Qwen2.5-Omni snapshot or slim Thinker folder")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output_dir = args.output_dir or args.model_dir / "quantized" / "AuK"
    printer = lambda message: print(message, flush=True)  # noqa: E731
    if args.part in ("thinker", "all"):
        from indextts.auk.thinker_files import SLIM_FOLDER, build_slim_folder, full_folder

        build_slim_folder(full_folder(args.model_dir), args.model_dir / SLIM_FOLDER, progress=printer)
    if args.part in ("dit", "int8", "all"):
        convert_dit(args.model_dir, args.dit_output or output_dir / DIT_FILE, layers=args.dit_layers,
                    device=args.device, progress=printer)
    if args.part in ("text", "int8", "all"):
        convert_text(args.model_dir, args.text_output or output_dir / TEXT_FILE, source=args.text_source,
                     device=args.device, progress=printer)


if __name__ == "__main__":
    main()
