from argparse import Namespace
import json

import numpy as np
import pytest
import soundfile as sf
import torch

from indextts.training.dataset_manifest import load_manifest, write_manifest
from indextts.training.dataset_quality import boundary_words_match, word_error_counts
from ui.dataset_curation import CURATION_DEFAULTS, curation_command


def test_boundary_error_is_detected_even_when_overall_word_error_passes():
    reference = "The selected or automatically resolved file always appears in Reference Voice. Audio and video are decoded by FFmpeg, and the audio preview is exactly what generation uses."
    hypothesis = reference.replace("uses.", "use.")
    errors, words = word_error_counts(reference, hypothesis)
    assert errors / words < .15
    assert not boundary_words_match(reference, hypothesis)
    assert boundary_words_match("Hello, complete sentence!", "hello complete sentence")
    assert not boundary_words_match("Hello complete sentence.", "complete sentence")


def test_curation_checks_real_clip_edges_and_keeps_source_holdouts(tmp_path, monkeypatch):
    from tools import curate_voice_dataset as curate
    import transformers
    source = tmp_path / "prepared"
    source.mkdir()
    sr = 24000
    waveform = np.sin(2 * np.pi * 220 * np.arange(sr * 2) / sr).astype(np.float32) * .1
    waveform[:sr // 10] = waveform[-sr // 10:] = 0
    reference = tmp_path / "reference.wav"
    sf.write(reference, waveform, sr)
    texts = [
        "This clean training sentence has a complete ending.",
        "One small final word error can hide inside a longer sentence with a nearly perfect overall score and complete ending.",
        "This separate validation sentence has a complete ending.",
        "This independent test sentence has a complete ending.",
        "This training example ends in active speech.",
    ]
    topics = ["train", "train", "validation", "test", "train"]
    rows = []
    for index, (text, topic) in enumerate(zip(texts, topics)):
        audio = waveform.copy()
        if index == 4:
            audio[-sr // 10:] = .1
        filename = f"clip_{index}.wav"
        sf.write(source / filename, audio, sr)
        rows.append({"id": str(index), "audio": filename, "text": text, "duration_s": 2.0,
                     "source_media": topic + ".flac", "source_start_s": 0, "source_end_s": 2,
                     "speaker": "Speaker", "language": "EN"})
    write_manifest(source / "manifest.jsonl", rows)

    class Verifier:
        def __init__(self, *args, **kwargs):
            pass

        def score(self, path):
            return {"speaker_similarity": .95, "speaker_window_min": .9, "speaker_window_mean": .93,
                    "speaker_windows": [.9, .96], "duration_s": 2}

    native = iter([texts[0], texts[1].replace("ending.", "and."), texts[2], texts[3]])
    monkeypatch.setattr(curate, "SpeakerVerifier", Verifier)
    monkeypatch.setattr(curate, "_ensure_model", lambda model: tmp_path)
    monkeypatch.setattr(curate, "_load_audio_16k", lambda path: (torch.zeros(1, 32000), 2.0))
    monkeypatch.setattr(transformers, "pipeline", lambda *args, **kwargs: lambda *a, **k: {"text": next(native)})
    args = Namespace(dataset=source, output=tmp_path / "curated", reference=[str(reference)],
                     validation_source=["validation"], test_source=["test"], max_wer=.15,
                     min_speaker_similarity=.7, min_window_similarity=.6, device="cpu", model_dir=tmp_path,
                     whisper="fixture", no_asr_recheck=False, transcribe_all=True, check_boundary_words=True,
                     min_edge_silence_ms=30, state_dir=tmp_path / "state")
    curate.run_curation(args)
    train_val = load_manifest(args.output)
    assert [(row["id"], row["split"]) for row in train_val] == [("0", "train"), ("2", "val")]
    test = load_manifest(tmp_path / "curated_test")
    assert [row["id"] for row in test if row["split"] == "val"] == ["3"]
    assert {row["id"] for row in train_val}.isdisjoint({"3"})
    audit = load_manifest(args.output / "quality_audit.jsonl")
    assert "transcript_boundary_mismatch" in audit[1]["reasons"]
    assert audit[1]["asr_wer"] < .15
    assert "unsafe_audio_boundary" in audit[4]["reasons"]
    assert not audit[4]["asr_rechecked"]
    assert len(load_manifest(source)) == 5
    assert json.loads((args.state_dir / "status.json").read_text())["phase"] == "complete"


def test_ui_audit_rejects_invalid_holdout_before_starting_worker(tmp_path):
    dataset = tmp_path / "prepared"
    dataset.mkdir()
    (dataset / "dataset_info.json").write_text(json.dumps({"status": "complete"}))
    write_manifest(dataset / "manifest.jsonl", [{"id": "0", "source_media": "train.flac"},
                                                {"id": "1", "source_media": "validation.flac"}])
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"placeholder")
    values = {**CURATION_DEFAULTS, "references": str(reference), "validation_sources": "missing"}
    with pytest.raises(ValueError, match="Unknown source recordings"):
        curation_command(str(dataset), values)
    assert not (tmp_path / "voice_curated").exists()
    values["validation_sources"] = "validation.flac"
    command, output = curation_command(str(dataset), values)
    assert command[command.index("--validation-source") + 1] == "validation"
    assert command[command.index("--min-edge-silence-ms") + 1] == "30"
    assert "--check-boundary-words" in command and "--transcribe-all" in command
    assert output == tmp_path / "voice_curated"
