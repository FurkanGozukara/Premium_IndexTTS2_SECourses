from __future__ import annotations

from pathlib import Path
import re

import pytest

from indextts.training.dataset_manifest import load_manifest
from indextts.training.dataset_prep import DatasetPrepConfig, run_dataset_prep
from indextts.training.media import extract_audio
from indextts.training.segmenter import (
    build_sentence_aligned_segments,
    is_sentence_aligned_text,
    split_caption_sentences,
)
from indextts.training.subtitles import (
    SubtitleCue,
    build_caption_transcript,
    clean_cues,
    parse_subtitle_file,
)
from indextts.training.whisper_asr import Word, align_caption_words
from indextts.utils.subtitle_utils import format_srt_timestamp


VIDEO2 = Path(r"G:\Index_TTS_v4\Lora_Training_Dataset\source2\video2.mp4")
VIDEO2_SRT = VIDEO2.with_suffix(".srt")


def test_caption_words_align_through_whisper_insertions_and_deletions() -> None:
    cues = clean_cues(
        [
            SubtitleCue(1, 0, 3600, "Hello world. This is version 2.12 and"),
            SubtitleCue(2, 3600, 7200, "it works well. Next sentence ends here!"),
        ]
    )
    caption = build_caption_transcript(cues)
    whisper = [
        Word(" hello", 0.10, 0.40),
        Word(" noisy", 0.45, 0.65),  # ASR insertion
        Word(" world", 0.80, 1.10),
        Word(" this", 1.40, 1.70),
        Word(" is", 1.75, 1.90),
        Word(" version", 2.00, 2.40),
        Word(" two", 2.45, 2.65),
        Word(" twelve", 2.70, 3.00),
        Word(" and", 3.25, 3.55),
        # Caption "it" is deliberately absent from ASR.
        Word(" works", 4.10, 4.45),
        Word(" really", 4.46, 4.65),  # another ASR insertion
        Word(" well", 4.70, 5.00),
        Word(" next", 5.40, 5.70),
        Word(" sentence", 5.75, 6.20),
        Word(" ends", 6.25, 6.55),
        Word(" here", 6.60, 7.00),
    ]

    alignment = align_caption_words(caption.words, whisper)
    assert alignment.coverage == pytest.approx(14 / 15)
    missing = next(word for word in alignment.words if word.text == "it")
    assert not missing.matched
    assert 3.55 <= missing.start_s < missing.end_s <= 4.10

    segments = build_sentence_aligned_segments(
        caption,
        alignment.words,
        target_s=3.0,
        min_s=1.0,
        max_s=5.5,
    )
    assert [segment.text for segment in segments] == [
        "Hello world. This is version 2.12 and it works well.",
        "Next sentence ends here!",
    ]
    assert all(segment.alignment_coverage is not None for segment in segments)
    assert all(segment.sentence_aligned for segment in segments)


def test_sentence_split_ignores_decimals_and_lowercase_caption_artifacts() -> None:
    caption = build_caption_transcript(
        clean_cues(
            [
                SubtitleCue(
                    1,
                    0,
                    5000,
                    "ComfyUI. is still one thought in version 2.12. Next sentence works!",
                )
            ]
        )
    )
    sentences = split_caption_sentences(caption)
    assert [caption.text[item.char_start : item.char_end] for item in sentences] == [
        "ComfyUI. is still one thought in version 2.12.",
        "Next sentence works!",
    ]


def test_sentence_alignment_accepts_uncased_scripts() -> None:
    assert is_sentence_aligned_text("你好。")
    assert not is_sentence_aligned_text("lowercase fragment.")


def _write_three_minute_sidecar(destination: Path) -> None:
    cues = [cue for cue in parse_subtitle_file(str(VIDEO2_SRT)) if cue.start_ms < 180_000]
    blocks = []
    for index, cue in enumerate(cues, start=1):
        end_ms = min(cue.end_ms, 180_000)
        blocks.append(
            f"{index}\n{format_srt_timestamp(cue.start_ms)} --> "
            f"{format_srt_timestamp(end_ms)}\n{cue.text.strip()}"
        )
    destination.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


@pytest.mark.gpu
@pytest.mark.skipif(
    not VIDEO2.is_file() or not VIDEO2_SRT.is_file(),
    reason="SECourses media fixture is not installed",
)
def test_first_three_minutes_are_sentence_aligned(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    clip = tmp_path / "video2_180s.wav"
    extract_audio(VIDEO2, clip, sample_rate=24000, mono=True, start_s=0.0, end_s=180.0)
    _write_three_minute_sidecar(clip.with_suffix(".srt"))
    config = DatasetPrepConfig(
        name="video2_sentence_alignment",
        inputs=[str(clip)],
        output_root=str(tmp_path),
        subtitle_policy="sidecar_only",
        segmentation_mode="sentence_aligned",
        target_s=8.0,
        min_s=4.0,
        max_s=12.0,
        export_reference_candidates=0,
        overwrite=True,
    )
    summary = run_dataset_prep(config)
    rows = load_manifest(summary.output_dir)
    assert rows

    def aligned(text: str) -> bool:
        first = next((character for character in text.strip() if character.isalnum()), "")
        terminal = re.search(r"[.?!;。！？；][\"')\]]*$", text.strip())
        return bool(first and (first.isupper() or first.isdigit()) and terminal)

    exceptions = [row for row in rows if not aligned(str(row["text"]))]
    # Over-long sentences may require a clause split; the accepted exception
    # budget keeps that documented fallback from hiding cue-boundary regressions.
    assert len(exceptions) / len(rows) <= 0.05
    assert summary.alignment["files"][0]["coverage"] >= 0.60
