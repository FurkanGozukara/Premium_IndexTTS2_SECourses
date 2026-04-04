import unittest
import os
import tempfile

import numpy as np

from indextts.utils.subtitle_utils import (
    SUPPORTED_SUBTITLE_EXTENSIONS,
    SubtitleCue,
    build_subtitle_render_units,
    assemble_subtitle_audio,
    fit_audio_to_duration,
    format_srt_timestamp,
    get_subtitle_format_label,
    parse_sbv,
    parse_srt,
    parse_srt_timestamp,
    parse_subtitle_file,
    parse_vtt,
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
        self.assertEqual(62004, parse_srt_timestamp("01:02.004"))

    def test_parse_vtt_accepts_header_identifiers_and_settings(self):
        content = """WEBVTT

intro
00:00:01.000 --> 00:00:03.250 align:start position:0%
First line
Second line

00:00:05.000 --> 00:00:06.000
Third line
"""
        cues = parse_vtt(content)

        self.assertEqual(2, len(cues))
        self.assertEqual(1000, cues[0].start_ms)
        self.assertEqual(3250, cues[0].end_ms)
        self.assertEqual("First line\nSecond line", cues[0].text)

    def test_parse_sbv_preserves_multiline_text(self):
        content = """0:00:01.000,0:00:03.250
First line
Second line

0:00:05.000,0:00:06.000
Third line
"""
        cues = parse_sbv(content)

        self.assertEqual(2, len(cues))
        self.assertEqual(1000, cues[0].start_ms)
        self.assertEqual(3250, cues[0].end_ms)
        self.assertEqual("First line\nSecond line", cues[0].text)

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

    def test_build_subtitle_render_units_merges_overlapping_cues(self):
        cues = [
            SubtitleCue(index=1, start_ms=0, end_ms=1000, text="Hello"),
            SubtitleCue(index=2, start_ms=500, end_ms=1500, text="world"),
            SubtitleCue(index=3, start_ms=2000, end_ms=2500, text="Again"),
        ]

        units = build_subtitle_render_units(cues)

        self.assertEqual(2, len(units))
        self.assertEqual((1, 2), units[0].cue_indices)
        self.assertEqual(0, units[0].start_ms)
        self.assertEqual(1500, units[0].end_ms)
        self.assertEqual("Hello world", units[0].text)
        self.assertEqual((3,), units[1].cue_indices)

    def test_fit_audio_to_duration_matches_target_sample_count(self):
        audio = np.full((200, 1), 1000, dtype=np.int16)

        fitted = fit_audio_to_duration(audio, sampling_rate=1000, target_duration_ms=350)

        self.assertEqual((350, 1), fitted.shape)
        self.assertEqual(np.int16, fitted.dtype)
        self.assertNotEqual(0, int(np.abs(fitted).sum()))

    def test_parse_subtitle_file_uses_extension_to_pick_parser(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subtitle_path = os.path.join(temp_dir, "sample.vtt")
            with open(subtitle_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHello there\n"
                )

            cues = parse_subtitle_file(subtitle_path)

        self.assertEqual(1, len(cues))
        self.assertEqual("Hello there", cues[0].text)

    def test_supported_extensions_have_labels(self):
        self.assertEqual({".srt", ".vtt", ".sbv"}, set(SUPPORTED_SUBTITLE_EXTENSIONS))
        self.assertEqual("WebVTT", get_subtitle_format_label("sample.vtt"))


if __name__ == "__main__":
    unittest.main()
