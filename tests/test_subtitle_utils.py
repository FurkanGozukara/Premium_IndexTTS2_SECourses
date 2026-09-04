from pathlib import Path

import pytest

from indextts.utils.subtitle_utils import parse_subtitle_file


DATASET_ROOT = Path(__file__).resolve().parents[2] / "Lora_Training_Dataset"


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
    subtitle_path = DATASET_ROOT / relative_path
    assert subtitle_path.is_file(), f"Missing subtitle fixture: {subtitle_path}"

    cues = parse_subtitle_file(str(subtitle_path))

    assert len(cues) == expected_count
    assert [cue.start_ms for cue in cues] == sorted(cue.start_ms for cue in cues)
    assert [cue.end_ms for cue in cues] == sorted(cue.end_ms for cue in cues)
    assert all(cue.start_ms <= cue.end_ms for cue in cues)
