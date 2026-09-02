from __future__ import annotations

import logging
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from indextts.training.dataset_prep import (
    DatasetPrepConfig,
    _orphan_subtitle_warnings,
    _safe_key,
)
from indextts.training.subtitles import parse_subtitle_file
from ui import dataset_tab


def test_subtitle_encodings_parse_to_identical_cues(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    content = (
        "1\n00:00:00,000 --> 00:00:01,500\n"
        "Türkçe: ğüşİıöç, İstanbul'da yağmur.\n\n"
        "2\n00:00:01,750 --> 00:00:03,000\n"
        "Güzel bir gün, değil mi?\n"
    )
    utf8_path = tmp_path / "altyazı_utf8.srt"
    cp1254_path = tmp_path / "altyazı_cp1254.srt"
    utf16_path = tmp_path / "altyazı_utf16.srt"
    utf8_path.write_text(content, encoding="utf-8")
    cp1254_path.write_bytes(content.encode("cp1254"))
    utf16_path.write_text(content, encoding="utf-16")

    with caplog.at_level(logging.WARNING):
        expected = parse_subtitle_file(str(utf8_path))
        assert parse_subtitle_file(str(cp1254_path)) == expected
        assert parse_subtitle_file(str(utf16_path)) == expected

    warning_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "cp1254" in warning_text
    assert "utf-16" in warning_text


def test_safe_key_is_ascii_stable_and_unique_across_iteration_order() -> None:
    paths = [
        Path("видео ğüş 测试.mp4"),
        Path("音频 ğüş 测试.mp4"),
        Path("video_ğüş_x.mp4"),
    ]

    def build(order: list[Path]) -> dict[str, str]:
        used: set[str] = set()
        return {path.stem: _safe_key(path, used) for path in order}

    forward = build(paths)
    reverse = build(list(reversed(paths)))
    assert forward == reverse
    assert len(set(forward.values())) == len(paths)
    assert all(value.isascii() for value in forward.values())
    assert all(re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in forward.values())
    assert forward[paths[0].stem].startswith("source_")
    assert re.fullmatch(r"video_[0-9a-f]{6}", forward[paths[2].stem])


def test_unicode_mixed_folder_scan_and_orphan_reporting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "набор_ğüş_测试"
    source.mkdir()
    captioned = source / "урок_ğüş.mp4"
    uncaptioned = source / "测试.wav"
    matching_subtitle = source / "урок_ğüş.srt"
    orphan_subtitle = source / "сирота_测试.srt"
    captioned.write_bytes(b"")
    uncaptioned.write_bytes(b"")
    matching_subtitle.write_text("", encoding="utf-8")
    orphan_subtitle.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        dataset_tab,
        "probe_media",
        lambda path: SimpleNamespace(duration_s=12.5),
    )
    rows = dataset_tab.scan_input_rows(str(source), None, True)
    by_name = {row[0]: row for row in rows}

    assert set(by_name) == {captioned.name, uncaptioned.name}
    assert by_name[captioned.name][1] == 12.5
    assert by_name[captioned.name][3] == matching_subtitle.name
    assert by_name[uncaptioned.name][3] == "None"

    config = DatasetPrepConfig(name="mixed", inputs=[str(source)])
    warnings = _orphan_subtitle_warnings(config, [row[4] for row in rows])
    assert len(warnings) == 1
    assert orphan_subtitle.stem.casefold() in warnings[0].casefold()
    assert matching_subtitle.stem.casefold() not in warnings[0].casefold()
