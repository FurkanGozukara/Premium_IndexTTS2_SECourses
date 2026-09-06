"""Cached speech decoding must use the same positions as full-sequence training."""

import pytest
import torch
from torch import nn

from indextts.gpt.model_v2 import GPT2InferenceModel, build_hf_gpt_transformer


@pytest.mark.parametrize("attention_backend", ["eager", "sdpa"])
@pytest.mark.parametrize("left_padding", [0, 3])
def test_cached_speech_logits_match_full_sequence(attention_backend, left_padding):
    torch.manual_seed(42)
    transformer, mel_positions, *_ = build_hf_gpt_transformer(
        2, 32, 4, 32, 32, False, attention_backend=attention_backend
    )
    model = GPT2InferenceModel(
        transformer.config,
        transformer,
        mel_positions,
        nn.Embedding(32, 32),
        nn.LayerNorm(32),
        nn.Linear(32, 32),
        kv_cache=True,
    ).eval()
    prompt_length = 8
    prompt = torch.randn(2, prompt_length, 32)
    prompt[0, :left_padding] = 0
    model.store_mel_emb(prompt)
    # Position zero is the start-mel token. Subsequent positions must be 1, 2, ...
    speech_tokens = torch.tensor([[30, 2, 5, 8, 11], [30, 3, 6, 9, 12]])
    all_ids = torch.cat((torch.ones(2, prompt_length, dtype=torch.long), speech_tokens), dim=1)
    mask = torch.ones_like(all_ids)
    mask[0, :left_padding] = 0

    with torch.no_grad():
        expected = model(input_ids=all_ids, attention_mask=mask, use_cache=False).logits
        past = None
        for step in range(speech_tokens.shape[1]):
            length = prompt_length + step + 1
            inputs = model.prepare_inputs_for_generation(
                all_ids[:, :length], past_key_values=past,
                attention_mask=mask[:, :length], use_cache=True,
            )
            actual = model(**inputs)
            torch.testing.assert_close(
                actual.logits[:, -1], expected[:, length - 1], atol=1e-6, rtol=1e-5,
                msg=f"Cached speech token {step} differs from the training/full-sequence position",
            )
            past = actual.past_key_values


@pytest.mark.gpu
@pytest.mark.parametrize("tts_mode", [True, False])
def test_accelerated_decode_positions_follow_each_sequences_own_prompt(tts_mode):
    from indextts.accel.accel_engine import AccelInferenceEngine
    from indextts.accel.kv_manager import Seq
    from indextts.accel.attention import get_forward_context, reset_forward_context

    engine = AccelInferenceEngine.__new__(AccelInferenceEngine)
    engine._tts_mode = tts_mode
    engine._tts_prompt_len = 9
    engine.block_size = 256
    sequences = [Seq([1] * 9), Seq([1] * 6)]
    for index, sequence in enumerate(sequences):
        sequence.block_table = [index]

    try:
        for step in range(1, 4):
            for sequence in sequences:
                sequence.append_token(10 + step)
            token_ids, positions = engine._prepare_decode(sequences)
            expected = [step, step] if tts_mode else [len(seq) - 1 for seq in sequences]
            assert positions.tolist() == expected
            assert token_ids.tolist() == [10 + step, 10 + step]
            assert get_forward_context().context_lens.tolist() == [len(seq) for seq in sequences]
    finally:
        reset_forward_context()
