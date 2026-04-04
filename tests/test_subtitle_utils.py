import unittest

import numpy as np

from indextts.utils.subtitle_utils import (
    SubtitleCue,
    assemble_subtitle_audio,
    format_srt_timestamp,
    parse_srt,
    parse_srt_timestamp,
    subtitle_cues_to_text,
)


class SubtitleUtilsTests(unittest.TestCase):
    def test_parse_srt_preserves_multiline_text(self):
        content = """1
00:00:01,000 --> 00:00:03,250
First line
Second line

2
00:00:05,000 --> 00:00:06,000
Third line
"""
        cues = parse_srt(content)

        self.assertEqual(2, len(cues))
        self.assertEqual(1000, cues[0].start_ms)
        self.assertEqual(3250, cues[0].end_ms)
        self.assertEqual("First line\nSecond line", cues[0].text)
        self.assertEqual("00:00:03,250", format_srt_timestamp(cues[0].end_ms))
        self.assertEqual("First line\nSecond line\n\nThird line", subtitle_cues_to_text(cues))

    def test_parse_srt_timestamp_accepts_comma_or_dot_separator(self):
        self.assertEqual(3723004, parse_srt_timestamp("01:02:03,004"))
        self.assertEqual(3723004, parse_srt_timestamp("01:02:03.004"))

    def test_assemble_subtitle_audio_inserts_silence_from_cue_start_times(self):
        cues = [
            SubtitleCue(index=1, start_ms=100, end_ms=200, text="A"),
            SubtitleCue(index=2, start_ms=400, end_ms=600, text="B"),
        ]
        rendered = [
            (cues[0], np.full((50,), 100, dtype=np.int16)),
            (cues[1], np.full((100,), 200, dtype=np.int16)),
        ]

        combined, issues = assemble_subtitle_audio(rendered, sampling_rate=1000)

        self.assertEqual((500, 1), combined.shape)
        self.assertEqual(0, combined[:100].sum())
        self.assertTrue(np.all(combined[100:150, 0] == 100))
        self.assertEqual(0, combined[150:400].sum())
        self.assertTrue(np.all(combined[400:500, 0] == 200))
        self.assertEqual([], issues)

    def test_assemble_subtitle_audio_reports_overruns_and_late_starts(self):
        cues = [
            SubtitleCue(index=1, start_ms=0, end_ms=200, text="A"),
            SubtitleCue(index=2, start_ms=250, end_ms=400, text="B"),
        ]
        rendered = [
            (cues[0], np.full((300,), 100, dtype=np.int16)),
            (cues[1], np.full((50,), 200, dtype=np.int16)),
        ]

        _, issues = assemble_subtitle_audio(rendered, sampling_rate=1000)

        self.assertEqual(
            [
                {"cue_index": 1, "type": "slot_overrun", "delta_ms": 100},
                {"cue_index": 2, "type": "late_start", "delta_ms": 50},
            ],
            issues,
        )


if __name__ == "__main__":
    unittest.main()
