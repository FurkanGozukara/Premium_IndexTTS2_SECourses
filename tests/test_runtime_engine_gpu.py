import os
from pathlib import Path
import time

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
DEMO = Path(
    os.environ.get(
        "INDEXTTS_TEST_REFERENCE_AUDIO", ROOT.parent / "demo_voice_for_test.mp3"
    )
)
DEMO_LORA = Path(
    os.environ.get(
        "INDEXTTS_TEST_DORA",
        ROOT
        / "loras"
        / "secourses_demo_dora_smoke"
        / "best"
        / "secourses_demo_dora_smoke.safetensors",
    )
)
DEMO_TEXT = (
    "Every voice carries a history of tiny choices: a pause before an important word, a smile hidden inside a "
    "sentence, and a rhythm learned over many years. This benchmark asks the model to preserve those details "
    "while reading a practical passage about clear speech, patient listening, and quiet confidence."
)


@pytest.fixture(scope="module")
def swapped_engine():
    if not torch.cuda.is_available() or not DEMO.is_file():
        pytest.skip("CUDA and the demo reference audio are required")
    from indextts.infer_v2_5 import IndexTTS2
    from indextts.runtime.vram_presets import resolve_preset

    runtime = resolve_preset("32", 32)
    runtime.blocks_to_swap = 12
    runtime.aux_residency["qwen_emo"] = "on_demand"
    engine = IndexTTS2(
        str(ROOT / "models" / "config.yaml"),
        str(ROOT / "models"),
        runtime=runtime,
        use_qwen_emo=False,
    )
    yield engine
    engine.unload()


@pytest.mark.gpu
@pytest.mark.parametrize("beams", [1, 3])
def test_engine_generates_with_dynamic_cache_and_block_swap(swapped_engine, tmp_path, beams):
    output = tmp_path / f"beam_{beams}.wav"
    result = swapped_engine.infer(
        str(DEMO),
        "A short runtime smoke test.",
        str(output),
        lang="EN",
        num_beams=beams,
        do_sample=False,
        max_mel_tokens=96,
        diffusion_steps=2,
    )
    assert result == str(output)
    assert output.stat().st_size > 44
    assert swapped_engine.block_swap.stats()["loads"] > 0


@pytest.mark.gpu
def test_int8_engine_smoke_when_checkpoint_exists(tmp_path):
    checkpoint = ROOT / "models" / "gpt_int8_convrot.safetensors"
    if not torch.cuda.is_available() or not DEMO.is_file() or not checkpoint.is_file():
        pytest.skip("INT8 ConvRot checkpoint is not installed")
    from indextts.infer_v2_5 import IndexTTS2
    from indextts.runtime.vram_presets import resolve_preset

    runtime = resolve_preset("8", 8)
    runtime.device = "cuda:0"
    engine = IndexTTS2(
        str(ROOT / "models" / "config.yaml"),
        str(ROOT / "models"),
        runtime=runtime,
        use_qwen_emo=False,
    )
    try:
        output = tmp_path / "int8.wav"
        engine.infer(
            str(DEMO),
            "An INT8 smoke test.",
            str(output),
            lang="EN",
            num_beams=1,
            do_sample=False,
            max_mel_tokens=64,
            diffusion_steps=2,
        )
        assert output.stat().st_size > 44
    finally:
        engine.unload()


@pytest.mark.gpu
def test_bf16_adapter_merge_matches_unmerged_output_and_reports_rtf():
    if not torch.cuda.is_available() or not DEMO.is_file() or not DEMO_LORA.is_file():
        pytest.skip("CUDA, demo reference audio, and the demo DoRA are required")
    from indextts.infer_v2_5 import IndexTTS2
    from indextts.runtime.vram_presets import resolve_preset

    runtime = resolve_preset("32", 32)
    runtime.device = "cuda:0"
    runtime.gpt_dtype = "fp32"
    runtime.lora_path = str(DEMO_LORA)
    runtime.lora_strength = 1.0
    runtime.lora_merge_into_base = False
    engine = IndexTTS2(
        str(ROOT / "models" / "config.yaml"),
        str(ROOT / "models"),
        runtime=runtime,
        use_qwen_emo=False,
    )
    options = {
        "lang": "EN",
        "seed": 123,
        "do_sample": False,
        "num_beams": 1,
        "max_text_tokens_per_segment": 60,
        "max_mel_tokens": 512,
        "diffusion_steps": 2,
        "cfm_temperature": 1.0,
        "reuse_spk_cond_for_emo": True,
        "interval_silence": 0,
    }
    try:
        comparison_options = dict(options)
        comparison_options["max_mel_tokens"] = 96
        unmerged_rate, unmerged_audio = engine.infer(
            str(DEMO), "A short deterministic adapter merge check.", None, **comparison_options
        )
        engine.set_lora(str(DEMO_LORA), 1.0, merge_into_base=True)
        assert engine._lora_merged is True
        merged_rate, merged_audio = engine.infer(
            str(DEMO), "A short deterministic adapter merge check.", None, **comparison_options
        )
        assert merged_rate == unmerged_rate
        np.testing.assert_allclose(merged_audio, unmerged_audio, rtol=2e-2, atol=96)

        engine.set_lora(str(DEMO_LORA), 1.0, merge_into_base=False)
        started = time.perf_counter()
        unmerged_rate, unmerged_audio = engine.infer(
            str(DEMO), DEMO_TEXT, None, **options
        )
        unmerged_elapsed = time.perf_counter() - started
        unmerged_rtf = unmerged_elapsed / (unmerged_audio.shape[0] / unmerged_rate)

        engine.set_lora(str(DEMO_LORA), 1.0, merge_into_base=True)
        assert engine._lora_merged is True
        started = time.perf_counter()
        merged_rate, merged_audio = engine.infer(
            str(DEMO), DEMO_TEXT, None, **options
        )
        merged_elapsed = time.perf_counter() - started
        merged_rtf = merged_elapsed / (merged_audio.shape[0] / merged_rate)

        print(
            f"MERGE_RTF unmerged={unmerged_rtf:.6f} merged={merged_rtf:.6f} "
            f"difference={merged_rtf - unmerged_rtf:+.6f}",
            flush=True,
        )

        engine.set_lora(str(DEMO_LORA), 0.8, merge_into_base=False)
        assert engine._lora_merged is False
        assert engine._lora_strength == 0.8
        engine.set_lora("", 1.0, merge_into_base=False)
        assert engine._lora_handle is None
    finally:
        engine.unload()
