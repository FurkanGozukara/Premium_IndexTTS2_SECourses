"""Cross-platform atomic JSON helpers that tolerate Windows sharing violations.

Workers (generation, dataset preparation, training) rewrite small status/progress JSON files about once
per second while the UI polls them.  On Windows ``os.replace`` fails with ``PermissionError`` (WinError 5 or
32) whenever another process still holds the destination open for reading, which previously killed a
training run in the middle of a step.  These helpers retry the rename with a short back-off, fall back to
a direct overwrite as a last resort, and give readers a matching retry for partially written files.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping


def _is_transient(exc: OSError) -> bool:
    winerror = getattr(exc, "winerror", None)
    if winerror in (5, 32, 33):
        return True
    if isinstance(exc, PermissionError):
        return True
    return exc.errno in (11, 13, 16)  # EAGAIN, EACCES, EBUSY


def replace_with_retry(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    attempts: int = 60,
    delay: float = 0.02,
    max_delay: float = 0.25,
) -> None:
    """``os.replace`` that retries transient sharing/permission errors (about 10 s worst case)."""

    src, dst = str(source), str(destination)
    wait = delay
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except OSError as exc:
            if attempt >= attempts - 1 or not _is_transient(exc):
                raise
            time.sleep(wait)
            wait = min(max_delay, wait * 1.5)


def write_json_atomic(
    path: str | os.PathLike[str],
    payload: Mapping[str, Any] | Any,
    *,
    indent: int | None = 2,
    ensure_ascii: bool = False,
    default: Callable[[Any], Any] | None = None,
    allow_nan: bool = True,
    fsync: bool = False,
    trailing_newline: bool = True,
) -> Path:
    """Serialise ``payload`` to ``path`` through a unique temporary file and an atomic rename.

    Never raises for a transient rename failure: after the retries are exhausted the file is overwritten in
    place, which keeps a long job alive even if a reader held the file open at the wrong moment.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    text = json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent, default=default, allow_nan=allow_nan)
    if trailing_newline:
        text += "\n"
    try:
        with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            if fsync:
                os.fsync(handle.fileno())
        try:
            replace_with_retry(temporary, destination)
        except OSError:
            with open(destination, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
    finally:
        try:
            os.unlink(temporary)
        except OSError:
            pass
    return destination


def read_json_retry(
    path: str | os.PathLike[str] | None,
    default: Any = None,
    *,
    attempts: int = 6,
    delay: float = 0.02,
    encoding: str = "utf-8-sig",
) -> Any:
    """Read a JSON file, retrying briefly when it is locked or only partially written."""

    if not path:
        return default
    target = Path(path)
    wait = delay
    for attempt in range(attempts):
        try:
            return json.loads(target.read_text(encoding=encoding))
        except FileNotFoundError:
            return default
        except (OSError, UnicodeError, json.JSONDecodeError):
            if attempt >= attempts - 1:
                return default
            time.sleep(wait)
            wait = min(0.2, wait * 1.5)
    return default


__all__ = ["read_json_retry", "replace_with_retry", "write_json_atomic"]
