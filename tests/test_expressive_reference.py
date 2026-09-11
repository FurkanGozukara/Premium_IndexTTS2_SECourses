"""The expressive training clip: selection, saving beside the adapter, and use as the emotion prompt."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from indextts.training.dataset_manifest import write_manifest
from indextts.training.dataset_profile import (
    expressive_profile_entry,
    expressive_reference_path,
    save_expressive_reference,
    update_profile_expressive_reference,
    write_dataset_profile,
)
from indextts.training.voice_profile import (
    ExpressivenessCache,
    choose_expressive_reference,
    describe_expressive_choice,
    measure_clip_expressiveness,
)
from ui import generation_tab
from ui.generation_tab import GENERATION_DEFAULTS, build_generation_request


def _voice(path: Path, seconds: float, *, vibrato: float, tremolo: float, sr: int = 16000) -> None:
    t = np.arange(int(seconds * sr)) / sr
    frequency = 140.0 * (1.0 + vibrato * np.sin(2 * np.pi * 1.5 * t))
    phase = 2 * np.pi * np.cumsum(frequency) / sr
    wave = sum(np.sin(k * phase) / k for k in range(1, 5)) * 0.2 * (1.0 - tremolo + tremolo * np.sin(2 * np.pi * 3.0 * t) ** 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wave.astype(np.float32), sr)


def _dataset(tmp_path: Path) -> tuple[Path, list[dict]]:
    dataset = tmp_path / "datasets" / "voice"
    rows = []
    specs = [("flat_a", 0.0, 0.0), ("flat_b", 0.005, 0.05), ("lively", 0.12, 0.8), ("flat_c", 0.0, 0.1)]
    for index, (name, vibrato, tremolo) in enumerate(specs):
        _voice(dataset / "segments" / f"{name}.wav", 9.0, vibrato=vibrato, tremolo=tremolo)
        rows.append({"id": name, "audio": f"segments/{name}.wav", "duration_s": 9.0, "words": 24, "text": "word " * 24,
                     "speaker": "A", "asr_wer": 0.0, "boundary_words_match": True, "split": "train", "language": "EN"})
    write_manifest(dataset / "manifest.jsonl", rows)
    return dataset, rows


def test_expressiveness_measurement_and_choice(tmp_path: Path) -> None:
    dataset, rows = _dataset(tmp_path)
    flat = measure_clip_expressiveness(dataset / "segments" / "flat_a.wav")
    lively = measure_clip_expressiveness(dataset / "segments" / "lively.wav")
    assert flat is not None and lively is not None
    assert lively["pitch_std_st"] > flat["pitch_std_st"] and lively["energy_std_db"] > flat["energy_std_db"]
    cache = ExpressivenessCache(dataset)
    choice = choose_expressive_reference(dataset, rows, cache=cache)
    assert choice is not None and choice["record"]["id"] == "lively"
    assert choice["score"] > 0 and len(choice["candidates"]) == 4
    assert (dataset / "analysis" / "expressiveness_cache.json").is_file()
    assert "expressive clip lively" in describe_expressive_choice(choice)
    # Cached measurements are reused and a missing clip is skipped.
    rows.append({"id": "missing", "audio": "segments/missing.wav", "duration_s": 9.0, "words": 20, "asr_wer": 0.0, "boundary_words_match": True})
    again = choose_expressive_reference(dataset, rows, cache=ExpressivenessCache(dataset))
    assert again is not None and again["record"]["id"] == "lively"
    assert choose_expressive_reference(dataset, []) is None
    assert measure_clip_expressiveness(tmp_path / "nope.wav") is None


def test_saved_clip_is_found_and_used_as_emotion_prompt(tmp_path: Path) -> None:
    dataset, rows = _dataset(tmp_path)
    adapter = tmp_path / "loras" / "voice"
    adapter.mkdir(parents=True)
    checkpoint = adapter / "voice.safetensors"
    checkpoint.write_bytes(b"x")
    assert expressive_reference_path(checkpoint) is None
    choice = choose_expressive_reference(dataset, rows)
    saved = save_expressive_reference(adapter, "voice", dataset, choice)
    assert saved.name == "voice_expressive_reference.wav" and saved.is_file()
    assert expressive_reference_path(checkpoint) == str(saved)
    assert expressive_reference_path(adapter / "best" / "voice_best.safetensors") == str(saved)
    write_dataset_profile(adapter, {"clips": 4, "duration_s": {"p50": 9.0}, "_vocabulary": ["word"]})
    update_profile_expressive_reference(checkpoint, expressive_profile_entry(choice, saved))
    profile = json.loads((adapter / "analysis" / "dataset_profile.json").read_text(encoding="utf-8"))
    assert profile["expressive_reference"]["id"] == "lively" and profile["expressive_reference"]["file"] == saved.name

    values = dict(GENERATION_DEFAULTS)
    values["runtime.lora_path"] = str(checkpoint)
    request = build_generation_request(values, prompt="ref.wav", text="Hello.")
    assert request["infer_kwargs"]["emo_audio_prompt"] == str(saved)
    assert request["infer_kwargs"]["emo_alpha"] == GENERATION_DEFAULTS["generation.emotion_weight"]
    values["generation.auto_lora_emotion_reference"] = False
    assert build_generation_request(values, prompt="ref.wav", text="Hello.")["infer_kwargs"]["emo_audio_prompt"] is None
    values["generation.auto_lora_emotion_reference"] = True
    values["generation.emotion_mode"] = generation_tab.EMOTION_MODES[2]
    assert build_generation_request(values, prompt="ref.wav", text="Hello.")["infer_kwargs"]["emo_audio_prompt"] is None
    values["runtime.lora_path"] = ""
    values["generation.emotion_mode"] = generation_tab.EMOTION_MODES[0]
    assert build_generation_request(values, prompt="ref.wav", text="Hello.")["infer_kwargs"]["emo_audio_prompt"] is None


def test_pick_expressive_clip_from_the_generation_tab(tmp_path: Path, monkeypatch) -> None:
    dataset, _rows = _dataset(tmp_path)
    adapter = tmp_path / "loras" / "voice"
    adapter.mkdir(parents=True)
    checkpoint = adapter / "voice.safetensors"
    checkpoint.write_bytes(b"x")
    (adapter / "train_config.json").write_text(json.dumps({"dataset_dir": str(dataset)}), encoding="utf-8")
    monkeypatch.setattr(generation_tab, "_token_len_for_profiles", lambda: (lambda text: len(text.split()) + 1))
    generation_tab._PROFILE_CACHE.clear()
    message = generation_tab.pick_expressive_clip(str(checkpoint))
    assert message.startswith("Saved voice_expressive_reference.wav") and "lively" in message
    assert expressive_reference_path(checkpoint) is not None
    profile = json.loads((adapter / "analysis" / "dataset_profile.json").read_text(encoding="utf-8"))
    assert profile["expressive_reference"]["id"] == "lively"
    assert "Select a LoRA / DoRA" in generation_tab.pick_expressive_clip("")
    orphan = tmp_path / "orphan.safetensors"
    orphan.write_bytes(b"x")
    assert "not found" in generation_tab.pick_expressive_clip(str(orphan))


def test_reference_fallbacks_never_pick_the_expressive_clip(tmp_path: Path) -> None:
    adapter = tmp_path / "loras" / "voice"
    adapter.mkdir(parents=True)
    checkpoint = adapter / "voice.safetensors"
    checkpoint.write_bytes(b"x")
    expressive = adapter / "voice_expressive_reference.wav"
    _voice(expressive, 1.0, vibrato=0.1, tremolo=0.5)
    # Only the expressive clip exists: the fallback must not treat it as the speaker reference.
    assert generation_tab._resolve_lora_reference_path(checkpoint, "") is None
    other = adapter / "other_reference.wav"
    _voice(other, 1.0, vibrato=0.0, tremolo=0.0)
    assert generation_tab._resolve_lora_reference_path(checkpoint, "") == str(other.resolve())
    assert expressive_reference_path(checkpoint) == str(expressive)
