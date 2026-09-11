"""CPU coverage for word-safe segmentation and recovery before acoustic decoding."""

from contextlib import nullcontext
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from indextts.infer_v2_5 import IndexTTS2
from indextts.utils.text_segmentation import (
    SpeechRecoveryConfig,
    segment_token_budget,
    split_text_by_tokens,
    split_text_for_recovery,
)


ROOT = Path(__file__).resolve().parents[1]
EOS = 8193
WORDS = {word: (index + 1) * 10 for index, word in enumerate(
    ["alpha", "beta", "gamma", "delta", "echo", "foxtrot", "golf", "hotel"]
)}


def test_default_english_segmentation_never_splits_an_ordinary_word():
    from indextts.utils.tokenizer import get_tokenizer

    tokenizer = get_tokenizer(multilingual=True, model_dir=str(ROOT / "models"))
    size = lambda text: len(tokenizer.encode(text, allowed_special="all"))
    text = (
        "when you are ready to start recording make sure that the microphone is connected "
        "and the room is quiet then read this entire sentence slowly and naturally so that "
        "the system can learn your pronunciation and recognize your voice consistently across "
        "the whole recording"
    )
    parts = split_text_by_tokens(text, 60, capacity=602, token_len=size, lang_prefix="<|en|> ")
    assert len(parts) > 1
    assert "".join(parts) == text
    assert all(size(part) <= segment_token_budget(60, 602, "<|en|> ", size) for part in parts)
    assert [word for part in parts for word in part.split()] == text.split()
    assert any("across" in part.split() for part in parts)


@pytest.mark.parametrize("text", ["abcdefghijklmno", "one abcdefghijklmno two", "一二三四五六七八九十"])
def test_a_single_oversized_word_or_unspaced_cjk_can_still_be_split(text):
    parts = split_text_by_tokens(text, 5, capacity=100, token_len=len)
    assert "".join(parts) == text
    assert all(len(part) <= 5 for part in parts)


def test_words_that_exactly_fill_the_budget_are_not_cut_by_following_spaces():
    text = "abcde fghij"
    parts = split_text_by_tokens(text, 5, capacity=100, token_len=len)
    assert "".join(parts) == text
    assert [word for part in parts for word in part.split()] == ["abcde", "fghij"]


def test_recovery_splits_at_a_balanced_clause_without_changing_text():
    text = "alpha beta, gamma delta"
    assert split_text_for_recovery(text, len) == ["alpha beta,", " gamma delta"]
    assert split_text_for_recovery("uninterrupted", len) == []


def test_recovery_preserves_pronunciation_annotations_and_mixed_cjk_words():
    annotation = "<|SPECIAL_TOKEN_1|>a long pronunciation<|SPECIAL_TOKEN_2|>"
    assert split_text_for_recovery(annotation, len) == []
    text = f"first {annotation} last"
    parts = split_text_for_recovery(text, len)
    assert "".join(parts) == text
    assert any(annotation in part for part in parts)
    mixed = "中文 international 中文"
    assert all("international" not in part or "international" in part.split() for part in split_text_for_recovery(mixed, len))


class CharacterTokenizer:
    def encode(self, text, **_kwargs):
        value = re.sub(r"<\|[a-z]+\|> ", "", text)
        return [ord(character) + 2 for character in value]

    def text(self, row):
        return "".join(chr(int(value) - 2) for value in row if int(value) > 1).strip()


def make_engine(*, always_incomplete=()):
    engine = IndexTTS2.__new__(IndexTTS2)
    engine.device = "cpu"
    engine.dtype = None
    engine.low_vram = False
    engine.runtime = SimpleNamespace(cfm_cache_length=8)
    engine.stop_text_token = 1
    engine.stop_mel_token = EOS
    engine.tokenizer = CharacterTokenizer()
    engine.progress_reporter = None
    engine.gr_progress = None
    engine.cache_spk_prompt_key = ("reference.wav", 15.0, 17)
    engine.cache_spk_cond = torch.ones(1, 1, 1)
    engine.cache_s2mel_style = torch.zeros(1, 1)
    engine.cache_s2mel_prompt = torch.zeros(1, 1, 1)
    engine.cache_mel = torch.zeros(1, 1, 1)
    engine.text_process = SimpleNamespace(
        clean_pattern=re.compile(r"\r"), char_rep_map={"\r": " "},
        normalize=lambda text, lang=None: text,
    )
    engine._use_s2mel = nullcontext
    engine._setup_s2mel_caches = lambda *_args: None
    engine.logs = []
    engine._progress_log = engine.logs.append
    engine.rendered = []
    engine.batched_decoded = []

    class GPT:
        text_pos_embedding = SimpleNamespace(emb=SimpleNamespace(num_embeddings=2000))

        def __init__(self):
            self.calls = []

        def get_emovec(self, *_args):
            return torch.zeros(1, 1)

        def inference_speech(self, _speaker, tokens, _lang, _emotion, **kwargs):
            rows = []
            for row in tokens:
                text = engine.tokenizer.text(row)
                self.calls.append((text, dict(kwargs)))
                limit = kwargs["max_generate_length"]
                if text in always_incomplete:
                    result = [99] * limit
                else:
                    result = ([WORDS[word] for word in text.split()] + [EOS])[:limit]
                rows.append(torch.tensor(result, dtype=torch.long))
            return torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=EOS), None

    engine.gpt = GPT()

    def render(codes, _prompt, _mel, _style, **kwargs):
        values = codes.flatten().tolist()
        assert EOS not in values and 99 not in values, "incomplete codes reached the decoder"
        engine.rendered.append((values, dict(kwargs)))
        repeats = max(1, round(10 * kwargs["duration_factor"]))
        return codes.float().repeat_interleave(repeats, dim=-1), 0.01, 0.01

    engine._render_codes_segment = render

    def conditioning(codes, lengths, *, duration_factor):
        rows = []
        for code, length in zip(codes, lengths):
            valid = code[:int(length)]
            values = valid.tolist()
            assert EOS not in values and 99 not in values, "incomplete codes reached batched decoding"
            engine.batched_decoded.append(values)
            rows.append(valid.float().repeat_interleave(max(1, round(10 * duration_factor))))
        targets = torch.tensor([len(row) for row in rows])
        return torch.nn.utils.rnn.pad_sequence(rows, batch_first=True).unsqueeze(-1), targets

    engine._prepare_batched_conditioning = conditioning
    engine.s2mel = SimpleNamespace(models={
        "cfm": SimpleNamespace(inference=lambda condition, *_args, **_kwargs: condition.transpose(1, 2)),
    })
    engine._vocode_batched_mels = lambda mels, lengths: [mels[row:row + 1, 0, :int(length)] for row, length in enumerate(lengths)]
    return engine


def options(**overrides):
    result = dict(
        lang="EN", seed=37, max_mel_tokens=4, max_text_tokens_per_segment=1000,
        reuse_spk_cond_for_emo=True, do_sample=False, num_beams=3,
        temperature=0.7, repetition_penalty=7.0, diffusion_steps=7,
        cfm_temperature=0.6, inference_cfg_rate=0.8,
    )
    result.update(overrides)
    return result


def expected_audio(words, repeats=10):
    return np.repeat([WORDS[word] for word in words.split()], repeats).astype(np.int16)[:, None]


def test_exact_budget_words_do_not_create_whitespace_only_synthesis_jobs():
    engine = make_engine()
    segments, plan, _ = engine._build_text_plan(
        "alpha delta", "EN", 5, False, 200, True, 1.0,
    )
    assert segments == ["alpha", "delta"]
    assert plan == [("segment", 0), ("silence", 4410), ("segment", 1)]


def test_complete_sequential_speech_does_not_generate_extra_takes():
    engine = make_engine()
    sample_rate, audio = engine.infer("reference.wav", "alpha beta", None, **options())
    assert sample_rate == 22050
    np.testing.assert_array_equal(audio, expected_audio("alpha beta"))
    assert len(engine.gpt.calls) == 1


@pytest.mark.parametrize("batched", [False, True])
def test_only_incomplete_segments_are_recovered_and_all_words_remain_in_order(batched):
    engine = make_engine()
    text = "alpha beta gamma delta"
    if batched:
        callbacks = []
        results = engine.infer_texts(
            "reference.wav", [text, "echo"], section_batch_size=2,
            on_text_complete=lambda index, result: callbacks.append((index, result)), **options(),
        )
        assert sorted(index for index, _ in callbacks) == [0, 1]
        np.testing.assert_array_equal(results[1][1], expected_audio("echo"))
        assert engine.batched_decoded == [[WORDS["echo"]]]
        audio = results[0][1]
    else:
        _, audio = engine.infer("reference.wav", text, None, **options())
    np.testing.assert_array_equal(audio, expected_audio(text))
    calls = [text for text, _ in engine.gpt.calls]
    assert calls.count(text) == 1
    assert calls[-2:] == ["alpha beta", "gamma delta"]
    assert any("Recovering incomplete speech" in message for message in engine.logs)
    for _, kwargs in engine.gpt.calls:
        assert kwargs["max_generate_length"] == 4
        assert kwargs["do_sample"] is False
        assert kwargs["num_beams"] == 3
        assert kwargs["repetition_penalty"] == 7.0
        assert not {"auto_retry_incomplete_speech", "max_speech_retries", "max_speech_split_depth"} & kwargs.keys()
    for _, kwargs in engine.rendered:
        assert kwargs["diffusion_steps"] == 7
        assert kwargs["cfm_temperature"] == 0.6
        assert kwargs["inference_cfg_rate"] == 0.8


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("enabled,attempts", [(False, 4), (True, 0), (True, 1), (True, 4)])
def test_recovery_uses_the_selected_attempt_budget_and_never_decodes_incomplete_audio(batched, enabled, attempts):
    engine = make_engine(always_incomplete={"alpha"})
    completed = []
    with pytest.raises(RuntimeError, match="Incomplete speech.*max_mel_tokens=4"):
        if batched:
            engine.infer_texts(
                "reference.wav", ["alpha", "echo"], section_batch_size=2,
                on_text_complete=lambda *args: completed.append(args),
                **options(auto_retry_incomplete_speech=enabled, max_speech_retries=attempts),
            )
        else:
            engine.infer(
                "reference.wav", "alpha", None,
                **options(auto_retry_incomplete_speech=enabled, max_speech_retries=attempts),
            )
    assert [text for text, _ in engine.gpt.calls].count("alpha") == 1 + (attempts if enabled else 0)
    assert not engine.rendered
    assert not engine.batched_decoded
    assert not completed


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("depth", [0, 1, 2])
def test_selected_split_depth_is_honored_in_both_generation_paths(batched, depth):
    engine = make_engine()
    text = "alpha beta gamma delta"
    settings = options(max_mel_tokens=2, max_speech_retries=6, max_speech_split_depth=depth)

    def generate():
        if batched:
            return engine.infer_texts("reference.wav", [text, "echo"], section_batch_size=2, **settings)[0]
        return engine.infer("reference.wav", text, None, **settings)

    if depth < 2:
        with pytest.raises(RuntimeError, match="stopped after 6 retries"):
            generate()
        assert not engine.rendered
    else:
        _, audio = generate()
        np.testing.assert_array_equal(audio, expected_audio(text))
    calls = [value for value, _ in engine.gpt.calls if value != "echo"]
    assert len(calls) == 7
    if depth == 0:
        assert calls == [text] * 7
    elif depth == 1:
        assert all(len(value.split()) >= 2 for value in calls)
    else:
        assert calls == [text, "alpha beta", "alpha", "beta", "gamma delta", "gamma", "delta"]


@pytest.mark.parametrize("value", [-1, 0.5, float("nan")])
def test_invalid_recovery_budgets_are_rejected_instead_of_replaced(value):
    with pytest.raises(ValueError, match="non-negative integer"):
        SpeechRecoveryConfig(max_attempts=value)
    with pytest.raises(ValueError, match="non-negative integer"):
        SpeechRecoveryConfig(max_split_depth=value)


def test_larger_selected_budgets_are_not_limited_to_the_original_backend_defaults():
    engine = make_engine()
    text = " ".join(["alpha"] * 16)
    _, audio = engine.infer(
        "reference.wav", text, None,
        **options(max_mel_tokens=2, max_speech_retries=30, max_speech_split_depth=4),
    )
    np.testing.assert_array_equal(audio, expected_audio(text))
    assert len(engine.gpt.calls) == 31


def test_an_impossibly_small_token_limit_fails_with_a_bounded_number_of_attempts():
    engine = make_engine()
    with pytest.raises(RuntimeError, match="max_mel_tokens=1.*no incomplete audio was accepted"):
        engine.infer("reference.wav", "alpha beta gamma delta echo foxtrot golf hotel", None, **options(max_mel_tokens=1))
    assert len(engine.gpt.calls) <= 15
    assert not engine.rendered


def test_streamed_recovery_preserves_explicit_pause_and_does_not_replay_completed_audio():
    engine = make_engine()
    stream = engine.infer(
        "reference.wav", "echo[pause:100ms]alpha beta gamma delta", None,
        stream_return=True, **options(),
    )
    pieces = list(stream)
    expected = np.concatenate([
        expected_audio("echo"), np.zeros((2205, 1), dtype=np.int16),
        expected_audio("alpha beta gamma delta"),
    ])
    np.testing.assert_array_equal(torch.cat(pieces, dim=-1).numpy().T, expected)
    assert len(pieces) == 3
    assert [text for text, _ in engine.gpt.calls].count("echo") == 1


def test_streaming_failure_emits_no_partial_audio_from_the_failed_segment():
    engine = make_engine(always_incomplete={"alpha"})
    stream = engine.infer(
        "reference.wav", "echo[pause:100ms]alpha", None, stream_return=True, **options(),
    )
    np.testing.assert_array_equal(next(stream).numpy().T, expected_audio("echo"))
    assert not next(stream).any()
    with pytest.raises(RuntimeError, match="Incomplete speech"):
        next(stream)
    assert [codes for codes, _ in engine.rendered] == [[WORDS["echo"]]]


def test_batched_recovery_keeps_the_original_multisegment_text_and_pause_plan():
    engine = make_engine()
    completed = []
    results = engine.infer_texts(
        "reference.wav", ["echo[pause:100ms]alpha beta gamma delta", "foxtrot"],
        section_batch_size=2, on_text_complete=lambda index, _result: completed.append(index),
        **options(),
    )
    expected = np.concatenate([
        expected_audio("echo"), np.zeros((2205, 1), dtype=np.int16),
        expected_audio("alpha beta gamma delta"),
    ])
    np.testing.assert_array_equal(results[0][1], expected)
    np.testing.assert_array_equal(results[1][1], expected_audio("foxtrot"))
    assert completed == [0, 1]
    assert engine.last_generation_stats["segments_count"] == 3


def test_natural_duration_pass_rerenders_recovered_parts_without_more_gpt_generation():
    engine = make_engine()
    _, audio = engine.infer(
        "reference.wav", "alpha beta gamma delta", None,
        **options(target_duration_s=80 / 22050, target_duration_mode="natural"),
    )
    np.testing.assert_array_equal(audio, expected_audio("alpha beta gamma delta", repeats=20))
    assert len(engine.gpt.calls) == 3
    assert [kwargs["duration_factor"] for _, kwargs in engine.rendered] == [1.0, 1.0, 2.0, 2.0]


def test_runner_records_failure_instead_of_a_successful_truncated_file(tmp_path):
    from webui_generation_runner import run_generation_request

    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"status": "in_progress", "outputs": {}, "processing": {}}), encoding="utf-8")
    request = {
        "prompt": "reference.wav", "text": "alpha", "subtitle_mode": False,
        "language": "EN", "save_used_audio": False, "save_as_mp3": False,
        "mp3_bitrate": "256k", "infer_kwargs": options(max_mel_tokens=1),
        "runtime": {"device": "cpu", "gpt_dtype": "fp32"}, "low_memory_mode": False,
        "metadata_path": str(metadata),
        "task_layout": {
            "task_folder": str(tmp_path), "final_wav_path": str(tmp_path / "final.wav"),
            "final_mp3_path": str(tmp_path / "final.mp3"), "final_mp4_path": str(tmp_path / "final.mp4"),
        },
    }
    request["infer_kwargs"].pop("lang")
    with pytest.raises(RuntimeError, match="Incomplete speech"):
        run_generation_request(request, make_engine())
    saved = json.loads(metadata.read_text(encoding="utf-8"))
    assert saved["status"] == "failed"
    assert "Incomplete speech" in saved["error"]
    assert not (tmp_path / "final.wav").exists()
    assert not (tmp_path / "candidate_01.wav").exists()


def test_text_plan_marks_sentence_gaps_and_pause_tags():
    engine = make_engine()
    segments, plan, _ = engine._build_text_plan(
        "alpha delta. beta gamma. [pause:200ms] omega", "EN", 1000, False, 200, True, 1.0, "sentence", None, 500,
    )
    assert [segment.strip() for segment in segments] == ["alpha delta.", "beta gamma.", "omega"]
    # Sentence ends inside a text chunk get the sentence gap; the pause tag keeps its own kind.
    assert plan == [
        ("segment", 0), ("sentence_gap", 11025), ("segment", 1), ("pause", 4410), ("segment", 2),
    ]
    # Without a sentence pause the section silence is used, and a cut inside a sentence never gets the gap.
    _segments, plan, _ = engine._build_text_plan(
        "alpha delta. beta gamma.", "EN", 1000, False, 200, True, 1.0, "sentence", None, 0,
    )
    assert plan == [("segment", 0), ("silence", 4410), ("segment", 1)]
    segments, plan, _ = engine._build_text_plan("alpha delta", "EN", 5, False, 200, True, 1.0, "budget", None, 500)
    assert segments == ["alpha", "delta"] and plan == [("segment", 0), ("silence", 4410), ("segment", 1)]
