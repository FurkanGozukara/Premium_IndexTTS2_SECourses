"""Every caption format the app accepts, content sniffing, lenient parsing and sidecar discovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from indextts.training import media
from indextts.utils import subtitle_utils as su


def _cues(text: str, extension: str):
    return su.parse_subtitle(text, extension)


def test_supported_extension_lists_are_shared() -> None:
    assert media.SUPPORTED_SUBTITLE_EXTENSIONS == su.SUPPORTED_SUBTITLE_EXTENSIONS
    for extension in (".ass", ".ssa", ".sub", ".lrc", ".ttml", ".dfxp", ".smi", ".json", ".tsv"):
        assert extension in su.SUPPORTED_SUBTITLE_EXTENSIONS
    for extension in (".ogv", ".mxf", ".mka", ".vob", ".3g2", ".m4b", ".dvr-ms", ".webm", ".opus"):
        assert extension in media.SUPPORTED_MEDIA_EXTENSIONS
    assert len(set(media.SUPPORTED_MEDIA_EXTENSIONS)) == len(media.SUPPORTED_MEDIA_EXTENSIONS)


def test_srt_tolerates_stray_blocks_missing_milliseconds_and_positioning() -> None:
    text = (
        "﻿Some editor note that is not a cue\n\n"
        "1\n00:00:01,000 --> 00:00:02,500 X1:10 X2:20\nFirst line\nsecond line\n\n"
        "2\n00:00:03 --> 00:00:04.5\nNo milliseconds and dot separator\n\n"
        "00:00:05,000 --> 00:00:04,000\nEnd before start is clamped\n\n"
        "3\n\n"
    )
    cues = _cues(text, ".srt")
    assert [(cue.index, cue.start_ms, cue.end_ms) for cue in cues] == [
        (1, 1000, 2500),
        (2, 3000, 4500),
        (3, 5000, 5000),
    ]
    assert cues[0].text == "First line\nsecond line"


def test_vtt_with_header_notes_identifiers_and_settings() -> None:
    text = (
        "WEBVTT\nKind: captions\nLanguage: en\n\nNOTE this is a comment\n\n"
        "STYLE\n::cue { color: red }\n\n"
        "intro\n00:01.000 --> 00:03.000 align:start position:0%\n<c.yellow>Hello</c> there&nbsp;\n\n"
        "00:00:03.000 --> 00:00:04.000\nSecond\n"
    )
    cues = _cues(text, ".vtt")
    assert [(cue.start_ms, cue.end_ms) for cue in cues] == [(1000, 3000), (3000, 4000)]
    assert cues[0].text == "<c.yellow>Hello</c> there&nbsp;"


def test_content_detection_beats_extension() -> None:
    vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHi\n"
    assert su.detect_subtitle_format(vtt) == ".vtt"
    assert [cue.text for cue in _cues(vtt, ".srt")] == ["Hi"]
    assert _cues("WEBVTT\n", ".vtt") == []
    assert _cues("", ".srt") == []
    with pytest.raises(ValueError, match="No caption cues"):
        _cues("This is not a caption file.", ".srt")
    with pytest.raises(ValueError, match="Unsupported caption format"):
        _cues("plain transcript text", ".txt")


def test_sbv_and_subviewer() -> None:
    sbv = "0:00:00.160,0:00:05.200\nGreetings everyone.\n\n0:00:05.200,0:00:12.480\nSecond cue\n"
    cues = _cues(sbv, ".sbv")
    assert [(cue.start_ms, cue.end_ms) for cue in cues] == [(160, 5200), (5200, 12480)]
    subviewer = "[INFORMATION]\n[TITLE]demo\n[END INFORMATION]\n\n00:00:01.00,00:00:02.50\nline one[br]line two\n"
    cues = _cues(subviewer, ".sub")
    assert [(cue.start_ms, cue.end_ms, cue.text) for cue in cues] == [(1000, 2500, "line one\nline two")]


def test_ass_dialogue_strips_override_tags_and_sorts() -> None:
    text = (
        "[Script Info]\nScriptType: v4.00+\n\n[V4+ Styles]\nFormat: Name, Fontname\nStyle: Default,Arial\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:05.00,0:00:06.50,Default,,0,0,0,,{\\an8}Later line, with a comma\n"
        "Comment: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,ignored\n"
        "Dialogue: 0,0:00:01.00,0:00:02.50,Default,,0,0,0,,First\\Nsecond {\\i1}word{\\i0}\n"
    )
    cues = _cues(text, ".ass")
    assert [(cue.start_ms, cue.end_ms, cue.text) for cue in cues] == [
        (1000, 2500, "First\nsecond word"),
        (5000, 6500, "Later line, with a comma"),
    ]
    assert su.detect_subtitle_format(text) == ".ass"
    assert _cues(text, ".ssa") == cues


def test_microdvd_frame_rate_declaration() -> None:
    text = "{1}{1}25.000\n{25}{50}Hello|world\n{100}{}{y:i}Open ended\n{200}{250}Last\n"
    cues = _cues(text, ".sub")
    assert [(cue.start_ms, cue.end_ms, cue.text) for cue in cues] == [
        (1000, 2000, "Hello\nworld"),
        (4000, 8000, "Open ended"),
        (8000, 10000, "Last"),
    ]


def test_lrc_lines_end_at_next_timestamp() -> None:
    text = "[ar:Someone]\n[00:01.00]First line\n[00:04.50]<00:04.60>Second <00:05.00>line\n[00:09.00]\n"
    cues = _cues(text, ".lrc")
    assert [(cue.start_ms, cue.end_ms, cue.text) for cue in cues] == [
        (1000, 4500, "First line"),
        (4500, 9000, "Second line"),
    ]


def test_ttml_paragraphs_with_breaks_offsets_and_frames() -> None:
    text = (
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
        "<tt xmlns=\"http://www.w3.org/ns/ttml\" xmlns:ttp=\"http://www.w3.org/ns/ttml#parameter\" ttp:frameRate=\"25\">\n"
        "<body><div>\n"
        "<p begin=\"00:00:01.000\" end=\"00:00:02.500\">First<br/>line</p>\n"
        "<p begin=\"3s\" dur=\"1500ms\">Second <span>span</span></p>\n"
        "<p begin=\"00:00:05:12\" end=\"00:00:06:00\">Frames</p>\n"
        "</div></body></tt>"
    )
    cues = _cues(text, ".ttml")
    assert [(cue.start_ms, cue.end_ms, cue.text) for cue in cues] == [
        (1000, 2500, "First\nline"),
        (3000, 4500, "Second span"),
        (5480, 6000, "Frames"),
    ]
    assert _cues(text, ".dfxp") == cues


def test_sami_sync_blocks() -> None:
    text = (
        "<SAMI><HEAD><TITLE>demo</TITLE></HEAD><BODY>\n"
        "<SYNC Start=1000><P Class=ENCC>Hello<br>there</P>\n"
        "<SYNC Start=3000><P Class=ENCC>&nbsp;</P>\n"
        "<SYNC Start=4000><P Class=ENCC>Second</P>\n"
        "</BODY></SAMI>"
    )
    cues = _cues(text, ".smi")
    assert [(cue.start_ms, cue.end_ms, cue.text) for cue in cues] == [
        (1000, 3000, "Hello\nthere"),
        (4000, 9000, "Second"),
    ]


def test_json_whisper_youtube_and_generic_lists() -> None:
    whisper = json.dumps({"text": "x", "segments": [{"id": 0, "start": 0.5, "end": 2.25, "text": " Hello there."}]})
    assert [(cue.start_ms, cue.end_ms, cue.text) for cue in _cues(whisper, ".json")] == [(500, 2250, " Hello there.")]
    json3 = json.dumps(
        {
            "wireMagic": "pb3",
            "events": [
                {"tStartMs": 0, "dDurationMs": 100, "wsWinId": 1},
                {"tStartMs": 1000, "dDurationMs": 2000, "segs": [{"utf8": "Hello "}, {"utf8": "world"}]},
            ],
        }
    )
    assert [(cue.start_ms, cue.end_ms, cue.text) for cue in _cues(json3, ".json3")] == [(1000, 3000, "Hello world")]
    generic = json.dumps(
        [
            {"start_ms": 100, "end_ms": 900, "text": "a"},
            {"startTime": "00:00:02.000", "endTime": "00:00:03.000", "text": "b"},
        ]
    )
    assert [(cue.start_ms, cue.end_ms) for cue in _cues(generic, ".json")] == [(100, 900), (2000, 3000)]
    with pytest.raises(ValueError):
        _cues(json.dumps({"created": "today", "sources": [{"video": "a.webm"}]}), ".json")


def test_tsv_whisper_milliseconds_and_timecodes() -> None:
    whisper = "start\tend\ttext\n0\t2500\tHello there\n2500\t4000\tSecond\n"
    assert [(cue.start_ms, cue.end_ms, cue.text) for cue in _cues(whisper, ".tsv")] == [
        (0, 2500, "Hello there"),
        (2500, 4000, "Second"),
    ]
    timecodes = "00:00:01.000\t00:00:02.000\tA\n00:00:02.000\t00:00:03.000\tB\n"
    assert [(cue.start_ms, cue.end_ms) for cue in _cues(timecodes, ".tsv")] == [(1000, 2000), (2000, 3000)]


def test_sidecar_discovery_covers_new_formats_and_skips_metadata(tmp_path: Path) -> None:
    media_file = tmp_path / "talk.webm"
    media_file.write_bytes(b"\x1aE\xdf\xa3")
    (tmp_path / "talk.en.vtt").write_text("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHi\n", encoding="utf-8")
    (tmp_path / "talk.ass").write_text(
        "[Script Info]\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,Hi\n",
        encoding="utf-8",
    )
    (tmp_path / "talk.info.json").write_text(json.dumps({"id": "abc", "title": "talk"}), encoding="utf-8")
    (tmp_path / "talk.json").write_text(
        json.dumps({"segments": [{"start": 0.0, "end": 1.0, "text": "Hi"}]}), encoding="utf-8"
    )
    (tmp_path / "talk.tsv").write_text("just\tsome\tcolumns\n", encoding="utf-8")
    (tmp_path / "talk.sub").write_bytes(b"\x00\x00\x01\xba" + b"\x00" * 64)  # binary VobSub
    (tmp_path / "other.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nOther\n", encoding="utf-8")
    names = [Path(path).name for path in media.find_sidecar_subtitles(media_file, language="EN")]
    assert names == ["talk.en.vtt", "talk.ass", "talk.json"]


def test_find_media_files_probes_unknown_explicit_extensions(tmp_path: Path, monkeypatch) -> None:
    unknown = tmp_path / "recording.xyz"
    unknown.write_bytes(b"data")
    text_file = tmp_path / "notes.txt"
    text_file.write_text("hello", encoding="utf-8")
    probed: list[str] = []

    def fake_probe(path):
        probed.append(Path(path).name)
        return media.MediaInfo(duration_s=1.0, has_audio=True, has_video=False, sample_rate=24000, channels=1, codec="pcm")

    monkeypatch.setattr(media, "probe_media", fake_probe)
    found = media.find_media_files([str(unknown), str(text_file)])
    assert [Path(path).name for path in found] == ["recording.xyz"]
    assert probed == ["recording.xyz"]
    assert media.find_media_files([str(tmp_path)]) == []  # folder scans stay extension-based
