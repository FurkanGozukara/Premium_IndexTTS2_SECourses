import importlib
import importlib.util

import pytest


REMOVED_MODULES = [
    "indextts.infer",
    "indextts.infer_v2",
    "indextts.cli",
    "indextts.codec.maskgct_codec",
    "indextts.gpt.model",
    "indextts.gpt.model_v2_5",
    "indextts.gpt.transformers_gpt2",
    "indextts.gpt.transformers_beam_search",
    "indextts.gpt.transformers_generation_utils",
    "indextts.gpt.transformers_modeling_utils",
    "indextts.vqvae",
    "indextts.BigVGAN",
    "indextts.utils.presets",
    "indextts.utils.webui_utils",
    "indextts.utils.text_utils",
    "indextts.utils.utils",
    "indextts.utils.feature_extractors",
    "indextts.utils.hf_cache_utils",
    "indextts.utils.maskgct",
    "indextts.utils.maskgct_utils",
    "indextts.s2mel.modules.openvoice",
    "indextts.s2mel.modules.hifigan",
    "indextts.s2mel.modules.rmvpe",
    "indextts.s2mel.optimizers",
    "indextts.s2mel.wav2vecbert_extract",
    "indextts.s2mel.hf_utils",
]


@pytest.mark.parametrize("module_name", REMOVED_MODULES)
def test_legacy_module_is_removed(module_name):
    assert importlib.util.find_spec(module_name) is None


def test_indextts_2_5_inference_imports():
    module = importlib.import_module("indextts.infer_v2_5")

    assert hasattr(module, "IndexTTS2")
