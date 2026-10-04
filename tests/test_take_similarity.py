"""Takes per section "most similar without word errors" and the cloning preset after training, without model weights."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from indextts.utils.take_selection import SectionTakeJudge, keep_most_similar_take


def test_most_similar_error_free_take_wins_and_checks_stop_early():
    similarity = {"a": 0.80, "b": 0.95, "c": 0.90, "d": 0.95}
    errors = {"a": 0.0, "b": 0.1, "c": 0.0, "d": 0.0}
    checked = []

    def score(take):
        checked.append(take)
        return errors[take]

    take, outcome = keep_most_similar_take("a", lambda count: ["b", "c", "d"][:count], similarity.get, score, 4, 5)
    assert take == "d"  # b and d tie on similarity: the earlier (b) is checked first, has an error; d is clean
    assert checked == ["b", "d"]
    assert outcome["kept"] == 3 and outcome["similarities"] == [0.80, 0.95, 0.90, 0.95]


def test_fewest_errors_then_more_similar_when_no_take_is_clean():
    similarity = {"a": 0.9, "b": 0.8, "c": 0.7}
    errors = {"a": 0.2, "b": 0.1, "c": 0.1}
    take, outcome = keep_most_similar_take("a", lambda count: ["b", "c"], similarity.get, errors.get, 3, 3)
    assert take == "b" and [item["take"] for item in outcome["checked"]] == [0, 1, 2]
    take, outcome = keep_most_similar_take("a", lambda count: ["b", "c"], similarity.get, errors.get, 3, 1)
    assert take == "a" and len(outcome["checked"]) == 1  # one check: the most similar is kept


def test_judge_falls_back_to_word_errors_without_a_voice_or_reference():
    judge = SectionTakeJudge("EN", "Hello there.", device="cpu", rule="similar", checks=12,
                             embedder=lambda samples, rate: np.ones(192))
    assert judge.checks == 8
    assert not judge.compares_voices()
    assert SectionTakeJudge("EN", "Hello.", device="cpu", rule="anything").rule == "errors"


def test_judge_compares_with_the_reference_clip(tmp_path):
    import soundfile as sf

    reference = tmp_path / "reference.wav"
    sf.write(str(reference), np.full(16000, 0.1, dtype=np.float32), 16000)

    class Embedder:
        def __call__(self, samples, rate):
            vector = np.zeros(192, dtype=np.float32)
            vector[0], vector[1] = 1.0, float(np.mean(samples))
            return vector / np.linalg.norm(vector)

        def file(self, path):
            return self(sf.read(str(path), dtype="float32")[0], 16000)

    judge = SectionTakeJudge("EN", "Hello.", device="cpu", rule="similar", reference=str(reference), embedder=Embedder())
    assert judge.compares_voices() and judge.target_source == "the reference clip"
    assert judge.similarity(np.full(100, 0.1), 16000) > judge.similarity(np.full(100, 0.9), 16000)


def test_omnivoice_renders_every_take_in_batches_and_keeps_the_most_similar_clean_one():
    from indextts.backends.omnivoice import OmniVoiceEngine

    class Model(torch.nn.Module):
        text_tokenizer = SimpleNamespace(encode=lambda text, **kwargs: text.split())

        def __init__(self):
            super().__init__()
            self.batches, self.count = [], 0

        def generate(self, **kwargs):
            texts = kwargs["text"] if isinstance(kwargs["text"], list) else [kwargs["text"]]
            self.batches.append(len(texts))
            out = []
            for _ in texts:
                self.count += 1
                out.append(np.full(2400, self.count / 100.0, dtype=np.float32))  # take k has level k/100
            return out

    class Judge:
        checks, history = 5, []

        def compares_voices(self):
            return True

        def similarity(self, samples, rate):
            level = round(float(np.mean(samples)) * 100)
            return {7: 0.99, 3: 0.95}.get(level, 0.5)

        def error_rate(self, text, samples, rate):
            return 0.2 if round(float(np.mean(samples)) * 100) == 7 else 0.0

        def record_similar(self, section, outcome):
            self.history.append(outcome)

    engine = OmniVoiceEngine.__new__(OmniVoiceEngine)
    engine.model, engine.device, engine.progress_reporter = Model(), "cpu", None
    engine.section_takes, engine.take_judge = 10, Judge()
    _, audio = engine.infer("", "Test.", omnivoice={"mode": "auto"}, text_normalization=False, section_batch_size=4)
    assert engine.model.batches == [1, 4, 4, 1]  # the first take, then nine more in batches of the section batch size
    assert engine.model.count == 10
    outcome = engine.take_judge.history[0]
    assert outcome["kept"] == 2  # take 7 is most similar but has an error; take 3 is next and clean
    assert [item["take"] for item in outcome["checked"]] == [6, 2]
    assert abs(audio.astype(np.float32).mean() / 32767 - 0.03) < 1e-3


def test_runner_attaches_the_rule_and_the_voice():
    from webui_generation_runner import _attach_take_judge, _release_take_judge

    tts = SimpleNamespace(device="cpu", model_dir="models")
    request = {"lora_path": "voice.safetensors", "prompt": "reference.wav", "runtime": {}}
    judge = _attach_take_judge(request, tts, 10, "Hello there.", "EN", rule="similar", checks=5)
    assert (tts.section_takes, judge.rule, judge.checks) == (10, "similar", 5)
    assert judge._voice == "voice.safetensors" and judge._reference == "reference.wav"
    _release_take_judge(tts, judge)
    assert tts.take_judge is None


def test_speech_span_and_preset_rate():
    from indextts.training.voice_preset import RATE_RANGE, preset_rate, speech_span_s

    rate = 16000
    tone = 0.3 * np.sin(2 * np.pi * 200 * np.arange(2 * rate) / rate)
    signal = np.concatenate([np.zeros(rate // 2), tone, np.zeros(rate // 2)]).astype(np.float32)
    assert abs(speech_span_s(signal, rate) - 2.0) < 0.05
    assert preset_rate([9.6, 9.5], [10.0, 10.0]) == (1.10, 0.955)  # within 10 %: the narration rate
    assert preset_rate([12.0], [10.0]) == (1.3, 1.2)  # 20 % slower than the speaker: 1.32, capped at 1.3
    assert RATE_RANGE[1] == 1.3
    assert preset_rate([8.0], [10.0]) == (0.88, 0.8)  # 20 % faster than the speaker slows down
    assert preset_rate([0.0], [1.0]) == (1.10, None)


def test_compose_preset_activates_omnivoice_cloning_with_ten_similar_takes():
    from indextts.training.voice_preset import compose_preset, preset_name

    base = {"app.model": "indextts", "generation.speaking_rate": 0.95, "omnivoice.mode": "auto",
            "omnivoice.position_temperature": 5.0, "generation.section_takes": 1,
            "app.profiles": {"_active": "indextts",
                             "omnivoice": {"generation.speaking_rate": 1.0, "runtime.lora_path": "",
                                           "generation.auto_lora_reference": False},
                             "indextts": {"generation.speaking_rate": 0.95}}}
    values = compose_preset(base, voice_path="loras/v/best/v_best.safetensors", speaking_rate=1.09, pauses=(310, 440))
    assert values["app.model"] == "omnivoice" and values["omnivoice.mode"] == "clone"
    assert values["generation.speaking_rate"] == 1.09 and values["generation.auto_lora_speaking_rate"] is False
    assert (values["generation.section_takes"], values["generation.section_take_rule"],
            values["generation.section_take_checks"]) == (10, "similar", 5)
    assert values["omnivoice.position_temperature"] == 0.5 and values["generation.auto_lora_reference"] is True
    assert (values["generation.sentence_pause_ms"], values["generation.max_pause_ms"]) == (310, 440)
    profile = values["app.profiles"]["omnivoice"]
    assert profile == {"generation.speaking_rate": 1.09, "runtime.lora_path": "loras/v/best/v_best.safetensors",
                       "generation.auto_lora_reference": True}
    assert values["app.profiles"]["indextts"] == {"generation.speaking_rate": 0.95}
    assert values["app.profiles"]["_active"] == "omnivoice"
    assert base["app.profiles"]["omnivoice"]["generation.speaking_rate"] == 1.0  # the base is not modified
    assert preset_name("Omni Voice/R2") == "Omni_Voice_R2_Clone_Best_of_10"


def test_write_preset_keeps_a_users_preset_of_the_same_name(tmp_path):
    from indextts.training.voice_preset import GENERATED_BY, PRESET_FORMAT, PRESET_VERSION, write_preset
    from ui.presets_store import PRESET_FORMAT as STORE_FORMAT, PRESET_VERSION as STORE_VERSION

    assert (PRESET_FORMAT, PRESET_VERSION) == (STORE_FORMAT, STORE_VERSION)
    (tmp_path / "user").mkdir()
    mine = tmp_path / "user" / "V_Clone_Best_of_10.json"
    mine.write_text(json.dumps({"_meta": {"name": "V_Clone_Best_of_10"}, "values": {"x": 1}}), encoding="utf-8")
    first = write_preset(tmp_path, "V_Clone_Best_of_10", {"app.model": "omnivoice"}, {"speaking_rate": 1.1})
    assert first.name == "V_Clone_Best_of_10_2.json" and json.loads(mine.read_text())["values"] == {"x": 1}
    again = write_preset(tmp_path, "V_Clone_Best_of_10", {"app.model": "omnivoice"}, {"speaking_rate": 1.05})
    assert again == first  # the generated preset is updated in place
    payload = json.loads(again.read_text(encoding="utf-8"))
    assert payload["_meta"]["generated"] == {"by": GENERATED_BY, "speaking_rate": 1.05}
    assert payload["_meta"]["format"] == STORE_FORMAT and payload["values"] == {"app.model": "omnivoice"}


def test_voice_preset_options_are_clamped():
    from indextts.training.train_config import TrainConfig

    config = TrainConfig.from_dict({"dataset_dir": "d", "name": "n", "voice_preset_sentences": 100,
                                    "voice_preset_timeout_s": 1})
    assert config.voice_preset_enabled is True
    assert (config.voice_preset_sentences, config.voice_preset_timeout_s) == (40, 60.0)


def test_centroid_clips_spread_over_sources():
    from indextts.utils.voice_similarity import choose_centroid_clips

    rows = [{"id": f"s{source}_{index}", "duration_s": 10.0, "source_media": f"s{source}"}
            for source in range(3) for index in range(40)] + [{"id": "short_1", "duration_s": 3.0}]
    picked = choose_centroid_clips(rows, count=60)
    assert len(picked) == 60 and all(row["duration_s"] >= 6 for row in picked)
    assert max(sum(row["source_media"] == f"s{source}" for row in picked) for source in range(3)) == 20


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


class _FakeProcess:
    """A finished child: writes its status like voice_preset.main and exits 0."""

    def __init__(self, args, **kwargs):
        from pathlib import Path

        self.args, self.returncode = args, 0
        state = Path(args[args.index("--state-dir") + 1])
        (state / "status.json").write_text(json.dumps({"phase": "complete", "message": "saved preset V_Clone_Best_of_10",
                                                       "preset": "presets/user/V_Clone_Best_of_10.json"}),
                                           encoding="utf-8")
        import io

        self.stdout = io.StringIO("")

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0


def _trainer_stub(tmp_path, model, enabled=True):
    from indextts.training.trainer import LoraTrainer

    statuses, logs = [], []
    best = tmp_path / "best" / "v_best.safetensors"
    best.parent.mkdir(parents=True)
    best.write_bytes(b"x")
    stub = SimpleNamespace(config=SimpleNamespace(tts_model=model, voice_preset_enabled=enabled, voice_preset_sentences=16,
                                                  voice_preset_timeout_s=60.0, to_dict=lambda: {"name": "v"}),
                           stop_path=tmp_path / "stop.flag", best_path=best, adapter_dir=tmp_path,
                           write_status=lambda **values: statuses.append(values), log=logs.append)
    run = lambda: LoraTrainer._run_voice_preset(stub, terminal_phase="done", terminal_message="m",  # noqa: E731
                                                recommended_checkpoint="")
    return run, statuses, logs


def test_trainer_builds_the_preset_for_every_model_unless_switched_off(tmp_path, monkeypatch):
    import indextts.training.trainer as trainer

    started = []
    monkeypatch.setattr(trainer.subprocess, "Popen", lambda args, **kwargs: started.append(args) or _FakeProcess(args))
    run, statuses, _ = _trainer_stub(tmp_path / "omni", "omnivoice")
    run()
    assert started and started[0][1:3] == ["-m", "indextts.training.voice_preset"]
    assert started[0][started[0].index("--checkpoint") + 1].endswith("v_best.safetensors")
    final = [item for item in statuses if item.get("voice_preset_status")][-1]
    assert final["voice_preset_status"] == "complete" and final["voice_preset"].endswith("V_Clone_Best_of_10.json")
    assert statuses[-1]["phase"] == "done"

    for model in ("indextts", "auk"):
        started.clear()
        run, statuses, _ = _trainer_stub(tmp_path / model, model)
        run()
        assert started and [item for item in statuses if item.get("voice_preset_status")][-1]["voice_preset_status"] == "complete"
    started.clear()
    run, statuses, _ = _trainer_stub(tmp_path / "off", "auk", enabled=False)
    run()
    assert not started and statuses[-1] == {"voice_preset_status": "skipped", "voice_preset_message": "disabled"}


def test_auk_and_indextts_presets_after_training():
    from indextts.training.voice_preset import MODELS, compose_preset, preset_name, trained_takes

    base = {"app.model": "omnivoice", "generation.speaking_rate": 1.0, "auk.mode": "auto",
            "app.profiles": {"_active": "omnivoice", "omnivoice": {"generation.speaking_rate": 1.0},
                             "auk": {"generation.speaking_rate": 1.0, "runtime.lora_path": "",
                                     "generation.section_takes": 1},
                             "indextts": {"generation.speaking_rate": 0.95, "runtime.lora_path": "",
                                          "generation.auto_lora_speaking_rate": False}}}
    auk = compose_preset(base, voice_path="a.safetensors", model="auk", speaking_rate=1.1, takes=(10, "similar", 5))
    assert (auk["app.model"], auk["auk.mode"], auk["auk.guidance_scale"]) == ("auk", "clone", 2.0)
    assert auk["generation.speaking_rate"] == 1.1 and auk["app.profiles"]["_active"] == "auk"
    assert auk["app.profiles"]["auk"]["generation.section_takes"] == 10
    index = compose_preset(base, voice_path="i.safetensors", model="indextts", takes=(5, "errors", 5))
    assert index["app.model"] == "indextts" and index["generation.speaking_rate"] == 0.95  # the profile's rate
    assert index["generation.auto_lora_speaking_rate"] is True and index["generation.auto_lora_max_tokens"] is True
    assert "omnivoice.mode" not in index and index["app.profiles"]["indextts"]["runtime.lora_path"] == "i.safetensors"
    assert preset_name("V", "auk", (10, "similar", 5)) == "V_Clone_Best_of_10"
    assert preset_name("V", "indextts", (5, "errors", 5)) == "V_Takes_5"
    assert preset_name("V", "indextts", (10, "similar", 5)) == "V_Best_of_10"
    assert set(MODELS) == {"omnivoice", "auk", "indextts"}
    assert trained_takes("omnivoice", "32") == (10, "similar", 5) and trained_takes("indextts", 6) == (5, "errors", 5)


def test_take_defaults_follow_the_measurements():
    from indextts.utils.take_selection import take_defaults

    assert take_defaults("indextts", 32) == take_defaults("indextts", 6) == (3, "errors", 5)
    assert take_defaults("omnivoice", "32") == (10, "similar", 5)
    assert take_defaults("omnivoice", 12) == (5, "similar", 3) and take_defaults("omnivoice", 8) == (3, "errors", 5)
    assert take_defaults("auk", 24) == (5, "similar", 3) and take_defaults("auk", 10) == (5, "similar", 3)
    assert take_defaults("auk", 8) == (3, "errors", 5)  # on-demand tiers move the models for every batch
    assert take_defaults("auk", 32, trained=True) == (10, "similar", 5)
    assert take_defaults("omnivoice", 8, trained=True) == (5, "similar", 3)


def test_tier_presets_carry_each_models_takes(tmp_path):
    from types import SimpleNamespace as NS

    from ui.app import build_app

    demo = build_app(NS(model_dir="models", device="cpu", verbose=False, no_browser=True, port=7861, host="127.0.0.1",
                        share=False))
    store = demo.preset_store
    for tier, omni, auk, index in (("32", (10, "similar", 5), (5, "similar", 3), (3, "errors", 5)),
                                   ("6", (3, "errors", 5), (3, "errors", 5), (3, "errors", 5))):
        values = store.tier_preset_values(tier)
        got = lambda source: (source["generation.section_takes"], source["generation.section_take_rule"],  # noqa: E731
                              source["generation.section_take_checks"])
        assert values["app.model"] == "omnivoice" and got(values) == omni
        assert got(values["app.profiles"]["omnivoice"]) == omni
        assert got(values["app.profiles"]["auk"]) == auk and got(values["app.profiles"]["indextts"]) == index
    # An older preset without these keys keeps one take per section.
    assert demo.preset_registry.defaults()["generation.section_takes"] == 1


def test_every_trainer_runs_the_preset_after_its_reference_audition():
    import inspect

    from indextts.training.auk_trainer import AukTrainer
    from indextts.training.omnivoice_trainer import OmniVoiceTrainer
    from indextts.training.trainer import LoraTrainer

    # OmniVoice and AuK run their own post-training sequence; each must end with the preset step like IndexTTS.
    for trainer in (LoraTrainer, OmniVoiceTrainer, AukTrainer):
        source = inspect.getsource(trainer)
        assert "self._run_reference_audition" in source and "self._run_voice_preset" in source, trainer.__name__
        assert source.index("self._run_voice_preset") > source.index("self._run_reference_audition"), trainer.__name__
