from pathlib import Path
import os

import pytest

from indextts.utils.subtitle_utils import parse_subtitle_file


def _reference_dataset_root() -> Path:
    candidates = [
        os.environ.get("INDEXTTS_TEST_DATASET_ROOT"),
        r"G:\Index_TTS_v4\Lora_Training_Dataset",
        str(Path(__file__).resolve().parents[2] / "Lora_Training_Dataset"),
    ]
    for value in candidates:
        if value and Path(value).is_dir():
            return Path(value)
    pytest.skip(
        "Reference subtitle dataset is unavailable; set INDEXTTS_TEST_DATASET_ROOT"
    )


@pytest.mark.parametrize(
    ("relative_path", "expected_count"),
    [
        ("source1/video1.srt", 241),
        ("source1/video1.sbv", 241),
        ("source2/video2.vtt", 132),
    ],
)
def test_reference_subtitles_parse_with_monotonic_timestamps(
    relative_path,
    expected_count,
):
    subtitle_path = _reference_dataset_root() / relative_path
    assert subtitle_path.is_file(), f"Missing subtitle fixture: {subtitle_path}"

    cues = parse_subtitle_file(str(subtitle_path))

    assert len(cues) == expected_count
    assert [cue.start_ms for cue in cues] == sorted(cue.start_ms for cue in cues)
    assert [cue.end_ms for cue in cues] == sorted(cue.end_ms for cue in cues)
    assert all(cue.start_ms <= cue.end_ms for cue in cues)
