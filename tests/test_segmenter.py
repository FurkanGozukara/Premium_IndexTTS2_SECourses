from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pytest

from indextts.training.segmenter import snap_boundaries_to_silence
from indextts.training.subtitles import (
    Segment,
    clean_cue_text,
    clean_cues,
    merge_cues_into_sentences,
    parse_subtitle_file,
)


VIDEO1_SRT = Path(r"G:\Index_TTS_v4\Lora_Training_Dataset\source1\video1.srt")


def _space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def test_clean_cue_text_markup_wrapping_and_annotations() -> None:
    raw = ">> HOST: <i>Hello&nbsp;&amp;</i> welcome !\n<b>Wrapped line</b> [Music] (laughs)"
    assert clean_cue_text(raw) == "Hello & welcome! Wrapped line"


def test_clean_cue_text_dedupes_rolling_caption_tail() -> None:
    previous = "A longer introduction finishes with these repeated words"
    current = "these repeated words and now the caption advances."
    assert clean_cue_text(current, previous_text=previous) == "and now the caption advances."


@pytest.mark.skipif(not VIDEO1_SRT.is_file(), reason="SECourses subtitle fixture is not installed")
def test_real_subtitles_merge_in_order_and_stay_bounded() -> None:
    cues = parse_subtitle_file(str(VIDEO1_SRT))
    cleaned = clean_cues(cues)
    segments = merge_cues_into_sentences(cues, min_s=1.5, target_s=8.0, max_s=15.0)
    assert segments
    assert all(1.5 <= segment.duration_s <= 15.0 for segment in segments)
    assert _space(" ".join(segment.text for segment in segments)) == _space(
        " ".join(cue.text for cue in cleaned)
    )
    flattened_indices = [index for segment in segments for index in segment.source_cue_indices]
    assert flattened_indices == [cue.index for cue in cleaned]


def test_snap_boundaries_to_synthetic_silence() -> None:
    envelope = np.ones(500, dtype=np.float32)
    envelope[90] = 0.001
    envelope[310] = 0.002
    segment = Segment(1000, 3000, "A synthetic sentence.", (1,))
    snapped = snap_boundaries_to_silence(segment, envelope, hop_ms=10, window_ms=200)
    assert snapped.start_ms == 900
    assert snapped.end_ms == 3110
