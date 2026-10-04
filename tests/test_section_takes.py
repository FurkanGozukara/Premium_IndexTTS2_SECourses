"""Takes per section: up to N renders of each section, the one Whisper hears with the fewest word errors kept."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from indextts.utils.take_selection import SectionTakeJudge, keep_best_take


def test_keep_best_take_stops_at_a_perfect_take_and_keeps_the_first_of_equals():
    takes = iter(["b", "c", "d"])
    scores = {"a": 0.2, "b": 0.0, "c": 0.0, "d": 0.5}
    best, rates, kept = keep_best_take("a", lambda: next(takes), scores.__getitem__, 5)
    assert (best, rates, kept) == ("b", [0.2, 0.0], 1)
    best, rates, kept = keep_best_take("a", lambda: "x", {"a": 0.1, "x": 0.1}.__getitem__, 3)
    assert (best, rates, kept) == ("a", [0.1, 0.1, 0.1], 0)
    assert keep_best_take("a", lambda: pytest.fail("no retake"), {"a": 0.3}.__getitem__, 1)[1] == [0.3]


def test_judge_uses_the_generation_language_and_ignores_phone_readings():
    calls = []

    def transcriber(source, language):
        calls.append((language, source[1]))
        return "Otherwise a LoRA or Dora reference is preferred."

    judge = SectionTakeJudge("EN", "Otherwise a LoRA or DoRA reference.", device="cpu", transcriber=transcriber)
    rate = judge.error_rate("Otherwise a LoRA or [D AO1 R AH0] reference is preferred.", np.zeros(24000), 24000)
    assert calls == [("en", 24000)]
    assert 0 < rate < 0.2  # only the spoken reading itself differs, the same for every take of the section
    assert SectionTakeJudge("AUTO", "Bu bir deneme cümlesidir, sesi kontrol ediyoruz.", device="cpu",
                            transcriber=transcriber).language == "tr"
    assert judge.error_rate("[laughter]", np.zeros(10), 24000) == 0.0
    judge.record(0, [0.1, 0.0], 1)
    assert judge.history == [{"section": 0, "error_rates": [0.1, 0.0], "kept": 1}]


def test_judge_scores_dictionary_readings_as_their_written_words():
    # The engines speak a dictionary word from its reading (an IndexTTS special-token span, OmniVoice brackets)
    # and Whisper writes the word. Leaving the reading out of the reference made every take of
    # "<xformers|...> and <SageAttention|...>." score 400 %, so such sections rendered all their takes.
    text = ("We deploy it using <xformers|EH1 K S . F AO1 R . M ER0 Z> and "
            "<SageAttention|S EY1 JH . AH0 . T EH1 N . SH AH0 N>.")
    heard = SectionTakeJudge("EN", text, device="cpu", transcriber=lambda source, language: "X formers and Sage Attention.")
    index_section = ("<|SPECIAL_TOKEN_1|>EH1 K S . F AO1 R . M ER0 Z<|SPECIAL_TOKEN_1|> and "
                     "<|SPECIAL_TOKEN_1|>S EY1 JH . AH0 . T EH1 N . SH AH0 N<|SPECIAL_TOKEN_1|>.")
    omni_section = "[EH1 K S F AO1 R M ER0 Z] and [S EY1 JH AH0 T EH1 N SH AH0 N]."
    assert heard.error_rate(index_section, np.zeros(10), 24000) == 0.0
    assert heard.error_rate(omni_section, np.zeros(10), 24000) == 0.0
    dropped = SectionTakeJudge("EN", text, device="cpu", transcriber=lambda source, language: "and.")
    assert dropped.error_rate(index_section, np.zeros(10), 24000) > 0.5  # a word the take left out still counts


def test_omnivoice_retakes_a_section_until_whisper_hears_it_right():
    from indextts.backends.omnivoice import OmniVoiceEngine

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.text_tokenizer = SimpleNamespace(encode=lambda text, **kwargs: text.split())
            self.calls = 0

        def generate(self, **kwargs):
            self.calls += 1
            return [np.full(2400, 0.1 * self.calls, dtype=np.float32)]

    heard = iter(["wrong words entirely", "First sentence here."])

    class Judge:
        history = []

        def error_rate(self, text, samples, rate):
            return 0.0 if next(heard) == "First sentence here." else 1.0

        def record(self, section, rates, kept):
            self.history.append((section, rates, kept))

    engine = OmniVoiceEngine.__new__(OmniVoiceEngine)
    engine.model, engine.device, engine.progress_reporter = Model(), "cpu", None
    engine.section_takes, engine.take_judge = 5, Judge()
    rate, audio = engine.infer("", "First sentence here.", omnivoice={"mode": "auto"}, text_normalization=False, seed=3)
    assert engine.model.calls == 2  # the second take had no errors, so the search stopped
    assert engine.take_judge.history == [(0, [1.0, 0.0], 1)]
    assert np.allclose(audio[:100] / 32767, 0.2, atol=1e-3)  # the kept take is the second render


def test_runner_attaches_and_releases_the_judge(monkeypatch):
    import webui_generation_runner as runner

    engine = SimpleNamespace(device="cpu")
    assert runner._attach_take_judge({"runtime": {}}, engine, 1, "Hello there friend.", "EN") is None
    assert engine.section_takes == 1 and engine.take_judge is None
    judge = runner._attach_take_judge({"runtime": {}}, engine, 4, "Bu bir deneme cümlesidir, sesi kontrol ediyoruz.", "AUTO")
    assert engine.section_takes == 4 and engine.take_judge is judge and judge.language == "tr"
    runner._release_take_judge(engine, judge)
    assert engine.section_takes == 1 and engine.take_judge is None


@pytest.mark.parametrize("settings, compared", [
    ({}, "ref.wav"),  # IndexTTS always clones the reference
    ({"omnivoice": {"mode": "clone"}}, "ref.wav"),
    ({"omnivoice": {"mode": "design"}}, None),
    ({"omnivoice": {"mode": "auto"}}, None),
    ({"auk": {"mode": "clone"}}, "ref.wav"),
    ({"auk": {"mode": "design"}}, None),
    ({"auk": {"mode": "auto"}}, None),
])
def test_most_similar_compares_the_reference_only_when_it_is_cloned(settings, compared):
    import webui_generation_runner as runner

    engine = SimpleNamespace(device="cpu")
    request = {"runtime": {}, "prompt": "ref.wav", "lora_path": "", **settings}
    judge = runner._attach_take_judge(request, engine, 5, "Hello there friend.", "EN", rule="similar")
    assert judge._reference == compared and judge._voice is None
    voiced = runner._attach_take_judge({**request, "lora_path": "loras/voice"}, engine, 5, "Hello there friend.", "EN",
                                       rule="similar")
    assert voiced._voice == "loras/voice"  # a trained voice is always compared with its own clips


def test_request_carries_section_takes(tmp_path):
    from ui.generation_tab import prepare_generation_request

    request = prepare_generation_request({"app.model": "omnivoice", "omnivoice.mode": "auto", "generation.section_takes": 5},
                                         prompt="", text="Test voice.", subtitle_file=None, image_path=None, emotion_audio=None,
                                         model_dir=str(tmp_path), output_root=tmp_path / "out")
    assert request["section_takes"] == 5
