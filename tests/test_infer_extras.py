import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import wave

import numpy as np
import pytest
import torch

from indextts.utils.audio_tuning import apply_audio_tuning
from indextts.utils.pause_tags import PauseChunk, TextChunk, describe_pauses, split_text_with_pauses
from indextts.utils.text_segmentation import (
    default_segment_tokens,
    segment_token_budget,
    split_text_by_tokens,
)


ROOT = Path(__file__).resolve().parents[1]
DEMO = Path(
    os.environ.get(
        "INDEXTTS_TEST_REFERENCE_AUDIO", ROOT.parent / "demo_voice_for_test.mp3"
    )
)


def test_pause_tag_parsing_and_description():
    chunks = split_text_with_pauses("One[pause:500ms]two [pause:0.8s]<pause=0.5>three")
    assert [type(chunk) for chunk in chunks] == [
        TextChunk,
        PauseChunk,
        TextChunk,
        PauseChunk,
        PauseChunk,
        TextChunk,
    ]
    assert [chunk.duration_ms for chunk in chunks if isinstance(chunk, PauseChunk)] == [500, 800, 500]
    assert describe_pauses("a[pause:500ms]b") == "1 inline pause: 500 ms (500 ms total)"
    assert describe_pauses("plain text") == "No inline pauses"


def test_non_cjk_segmentation_budget_uses_real_tokenizer():
    from indextts.utils.tokenizer import get_tokenizer

    tokenizer = get_tokenizer(multilingual=True, model_dir=str(ROOT / "models"))
    token_len = lambda value: len(tokenizer.encode(value, allowed_special="all"))
    prefix = "<|en|> "
    unscaled = segment_token_budget(120, 602, prefix, token_len, 1.0)
    scaled = segment_token_budget(120, 602, prefix, token_len, 0.72)
    assert scaled == int(unscaled * 0.72)

    sentence = "A measured sentence keeps the synthesis alignment stable. "
    text = sentence
    while token_len(text) <= scaled:
        text += sentence
    assert token_len(text) <= unscaled
    assert len(
        split_text_by_tokens(
            text,
            120,
            capacity=602,
            token_len=token_len,
            lang_prefix=prefix,
            segment_budget_scale_non_cjk=0.72,
        )
    ) > 1
    assert split_text_by_tokens(
        text,
        120,
        capacity=602,
        token_len=token_len,
        lang_prefix=prefix,
        segment_budget_scale_non_cjk=1.0,
    ) == [text]
    assert default_segment_tokens("EN") == 60
    assert default_segment_tokens("ES") == 60
    assert default_segment_tokens("AR") == 80
    assert default_segment_tokens("JA") == 100
    assert default_segment_tokens("ZH") == 120


@pytest.mark.parametrize(
    "preset", ["voice_clarity", "clear_narration", "deharsh", "warm", "normalize"]
)
def test_audio_tuning_processes_synthetic_wav(tmp_path, preset):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required")
    sample_rate = 22050
    samples = np.arange(sample_rate // 4)
    audio = (0.2 * np.sin(2 * np.pi * 220 * samples / sample_rate) * 32767).astype("<i2")
    source = tmp_path / "source.wav"
    output = tmp_path / "tuned.wav"
    with wave.open(str(source), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(audio.tobytes())

    assert apply_audio_tuning(source, output, preset, gain_db=-1.0) == str(output)
    with wave.open(str(output), "rb") as handle:
        assert handle.getframerate() == sample_rate
        assert handle.getnchannels() == 1
        assert abs(handle.getnframes() - audio.size) <= 2


def test_non_streaming_infer_does_not_hide_internal_index_errors() -> None:
    from indextts.infer_v2_5 import IndexTTS2

    engine = IndexTTS2.__new__(IndexTTS2)
    engine.runtime = SimpleNamespace(cfm_cache_length=8192)

    def broken_generator(*_args, **_kwargs):
        yield from ()
        raise IndexError("internal accelerator failure")

    engine.infer_generator = broken_generator
    with pytest.raises(IndexError, match="internal accelerator failure"):
        engine.infer("reference.wav", "hello", "output.wav")


def test_null_position_embeddings_follow_embedding_dtype_and_device():
    from indextts.gpt.model_v2 import build_hf_gpt_transformer, null_position_embeddings

    embedding = torch.nn.Embedding(4, 8, dtype=torch.bfloat16)
    positions = torch.zeros((2, 3), dtype=torch.long)
    result = null_position_embeddings(positions, 8, embedding=embedding)
    assert result.shape == (2, 3, 8)
    assert result.dtype == torch.bfloat16
    assert result.device == positions.device
    assert not result.any()

    gpt, *_ = build_hf_gpt_transformer(1, 8, 1, 16, 16, False)
    gpt.to(dtype=torch.bfloat16)
    actual = gpt.wpe(positions)
    assert actual.dtype == torch.bfloat16
    assert actual.device == positions.device


def test_batched_generation_drops_sequential_duration_options() -> None:
    from indextts.infer_v2_5 import _batched_generation_kwargs

    source = {
        "target_duration_s": None,
        "target_duration_mode": "off",
        "temperature": 0.8,
    }

    result = _batched_generation_kwargs(source)

    assert result == {"temperature": 0.8}
    assert source["target_duration_mode"] == "off"


def test_generation_request_plumbs_candidates_and_result_metadata(tmp_path):
    from webui_generation_runner import run_generation_request

    task_folder = tmp_path / "task"
    task_folder.mkdir()
    metadata_path = task_folder / "metadata.json"
    metadata_path.write_text(
        '{"status":"in_progress","outputs":{},"processing":{},"error":null}',
        encoding="utf-8",
    )

    class FakeTTS:
        low_vram = False

        def infer(self, spk_audio_prompt, text, output_path, lang="EN", **kwargs):
            del spk_audio_prompt, text, lang
            seed = kwargs["seed"]
            samples = np.full(1000, seed % 100, dtype="<i2")
            with wave.open(str(output_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(22050)
                handle.writeframes(samples.tobytes())
            self.last_generation_stats = {
                "seed": seed,
                "segments_count": 1,
                "audio_seconds": samples.size / 22050.0,
                "rtf": 0.5,
                "gpt_time": 0.1,
                "s2mel_time": 0.2,
                "vocoder_time": 0.05,
                "peak_vram_gb": 1.25,
            }
            return str(output_path)

    final_wav = task_folder / "final.wav"
    request = {
        "prompt": "speaker.wav",
        "text": "hello",
        "subtitle_mode": False,
        "subtitle_file": None,
        "language": "EN",
        "save_used_audio": False,
        "save_as_mp3": False,
        "mp3_bitrate": "256k",
        "image_path": None,
        "infer_kwargs": {},
        "runtime": {"device": "cpu", "gpt_dtype": "fp32"},
        "low_memory_mode": False,
        "metadata_path": str(metadata_path),
        "num_candidates": 2,
        "seed": 99,
        "task_layout": {
            "task_folder": str(task_folder),
            "final_wav_path": str(final_wav),
            "final_mp3_path": str(task_folder / "final.mp3"),
            "final_mp4_path": str(task_folder / "final.mp4"),
            "speaker_reference_copy_path": str(task_folder / "speaker.wav"),
            "segments_dir": None,
        },
    }
    result = run_generation_request(request, FakeTTS())
    assert result["seed"] == 99
    assert result["candidate_seeds"] == [99, 100]
    assert result["segments_count"] == 1
    assert result["peak_vram_gb"] == 1.25
    assert final_wav.is_file()
    assert (task_folder / "candidate_01.wav").is_file()
    assert (task_folder / "candidate_02.wav").is_file()


@pytest.fixture(scope="module")
def extras_engine():
    if not torch.cuda.is_available() or not DEMO.is_file():
        pytest.skip("CUDA and the demo reference audio are required")
    from indextts.infer_v2_5 import IndexTTS2
    from indextts.runtime.vram_presets import resolve_preset

    runtime = resolve_preset("32", 32)
    runtime.device = "cuda:0"
    engine = IndexTTS2(
        cfg_path=str(ROOT / "models" / "config.yaml"),
        model_dir=str(ROOT / "models"),
        runtime=runtime,
        use_qwen_emo=False,
    )
    yield engine
    engine.unload()


def _generate(engine, text, seed=123, **kwargs):
    options = {
        "lang": "EN",
        "seed": seed,
        "do_sample": False,
        "num_beams": 1,
        "cfm_temperature": 0.75,
        "reuse_spk_cond_for_emo": True,
        "diffusion_steps": 2,
        "max_mel_tokens": 96,
        "interval_silence": 0,
    }
    options.update(kwargs)
    result = engine.infer(str(DEMO), text, None, **options)
    assert result is not None
    return result


@pytest.mark.gpu
def test_seed_cfm_greedy_and_reused_emotion_are_deterministic(extras_engine):
    first_rate, first = _generate(extras_engine, "A short deterministic test.")
    second_rate, second = _generate(extras_engine, "A short deterministic test.")
    assert first_rate == second_rate == 22050
    assert first.shape[0] > first_rate // 5
    assert np.array_equal(first, second)
    assert extras_engine.last_generation_stats["seed"] == 123
    assert extras_engine.last_generation_stats["segments_count"] == 1


@pytest.mark.gpu
def test_inline_pause_adds_exact_half_second(extras_engine):
    sample_rate, baseline = _generate(extras_engine, "Pause check.", seed=456)
    pause_rate, paused = _generate(extras_engine, "Pause check.[pause:500ms]", seed=456)
    assert pause_rate == sample_rate
    assert abs((paused.shape[0] - baseline.shape[0]) - sample_rate // 2) <= 1


@pytest.mark.gpu
def test_natural_target_duration_reuses_gpt_codes(extras_engine):
    sample_rate, audio = _generate(
        extras_engine,
        "Natural duration check.",
        seed=789,
        target_duration_s=2.0,
        target_duration_mode="natural",
    )
    assert abs(audio.shape[0] / sample_rate - 2.0) < 0.35
    stats = extras_engine.last_generation_stats
    assert stats["target_duration_mode"] == "natural"
    assert stats["duration_factor_used"] != 1.0
