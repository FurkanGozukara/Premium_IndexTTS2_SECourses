"""Resilient text decoding for user-supplied captions and metadata."""

from __future__ import annotations

import codecs
import logging
from pathlib import Path
from typing import Callable


_LOGGER = logging.getLogger(__name__)
_UTF8_ENCODINGS = frozenset({"ascii", "utf8", "utf_8", "utf-8", "utf_8_sig", "utf-8-sig"})
_TURKISH_BYTES = frozenset({0xD0, 0xDD, 0xDE, 0xF0, 0xFD, 0xFE})
_TURKISH_CHARS = frozenset("ĞğİıŞş")
_TURKISH_AMBIGUOUS_ENCODINGS = frozenset(
    {
        "cp1250",
        "cp1252",
        "cp1254",
        "cp1257",
        "cp1258",
        "hp_roman8",
        "iso8859_4",
        "iso8859_10",
        "iso8859_14",
        "iso8859_16",
        "mac_latin2",
    }
)


def _charset_normalizer_decode(data: bytes) -> tuple[str, str] | None:
    try:
        from charset_normalizer import from_bytes
    except ImportError:
        return None

    try:
        matches = from_bytes(data)
        match = matches.best()
    except Exception:
        return None
    if match is None or not match.encoding:
        return None

    encoding = str(match.encoding)
    try:
        text = data.decode(encoding)
    except (LookupError, UnicodeError):
        try:
            text = str(match)
        except Exception:
            return None

    # The Windows Turkish code page is byte-compatible with several Western
    # encodings except for these six letters. Short captions are otherwise too
    # small for statistical detection, so prefer CP1254 when those bytes form
    # Turkish-specific characters.
    if encoding.casefold() in _TURKISH_AMBIGUOUS_ENCODINGS and any(
        byte in _TURKISH_BYTES for byte in data
    ):
        try:
            turkish_text = data.decode("cp1254")
        except UnicodeError:
            pass
        else:
            if any(char in _TURKISH_CHARS for char in turkish_text):
                text, encoding = turkish_text, "cp1254"
    return text, encoding


def read_text_resilient(
    path: str | Path,
    *,
    warning_callback: Callable[[str], None] | None = None,
) -> str:
    """Read user text without ever failing solely because of its encoding."""

    source = Path(path)
    data = source.read_bytes()
    decoded: tuple[str, str] | None = None

    if data.startswith(codecs.BOM_UTF8):
        try:
            decoded = data.decode("utf-8-sig"), "utf-8-sig"
        except UnicodeError:
            decoded = None
    else:
        bom_encodings = (
            (codecs.BOM_UTF32_LE, "utf-32", "utf-32-le"),
            (codecs.BOM_UTF32_BE, "utf-32", "utf-32-be"),
            (codecs.BOM_UTF16_LE, "utf-16", "utf-16-le"),
            (codecs.BOM_UTF16_BE, "utf-16", "utf-16-be"),
        )
        for bom, decoder, label in bom_encodings:
            if not data.startswith(bom):
                continue
            try:
                decoded = data.decode(decoder), label
            except UnicodeError:
                decoded = None
            break

    if decoded is None:
        try:
            decoded = data.decode("utf-8"), "utf-8"
        except UnicodeError:
            decoded = _charset_normalizer_decode(data)

    if decoded is None:
        try:
            decoded = data.decode("cp1252"), "cp1252"
        except UnicodeError:
            decoded = data.decode("utf-8", errors="replace"), "utf-8-replacement"

    text, encoding = decoded
    if encoding.casefold() not in _UTF8_ENCODINGS:
        message = f"Decoded text file {source} using detected encoding {encoding}."
        if warning_callback is not None:
            warning_callback(message)
        else:
            _LOGGER.warning(message)
    return text


__all__ = ["read_text_resilient"]
