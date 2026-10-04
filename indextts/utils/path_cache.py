"""Remembered path resolution and folder walks for the adapter lists."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any
import threading
import time


@lru_cache(maxsize=8192)
def _resolved(path: str) -> Path:
    return Path(path).expanduser().resolve()


def resolved_path(path: str | os.PathLike[str]) -> Path:
    """``Path(path).expanduser().resolve()``, remembered per spelling.

    Adapter lists resolve the same few hundred checkpoint paths whenever a
    training run saves a checkpoint, and on Windows each resolution queries
    every folder of the path.
    """

    return _resolved(os.fspath(path))


# A training run keeps tens of thousands of evaluation and sample files below
# these folders. Adapters, decoder adapters and run status files never live
# there, so walking them made every adapter list take most of a second.
RUN_ARTIFACT_DIRS = frozenset({"analysis", "samples", ".sample_jobs", "eval_jobs", "eval_job", "__pycache__"})
_TREE_TTL_S = 2.0
_TREE_CACHE: dict[str, tuple[float, Any, tuple[Path, ...], dict[str, tuple[int, int]]]] = {}
_TREE_LOCK = threading.Lock()


def _walk(top: str) -> tuple[tuple[Path, ...], dict[str, tuple[int, int]]]:
    """``os.walk`` order and pruning, plus each file's modification time and size.

    The directory listing already carries both on Windows, so recording them
    here spares a ``stat`` call per checkpoint in every list built afterwards.
    """

    files: list[Path] = []
    stats: dict[str, tuple[int, int]] = {}
    stack = [top]
    while stack:
        directory = stack.pop()
        dirs: list[str] = []
        listed: list[tuple[Path, tuple[int, int] | None]] = []
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        is_dir = entry.is_dir()
                    except OSError:
                        is_dir = False
                    if is_dir:
                        dirs.append(entry.name)
                        continue
                    try:
                        stat = entry.stat()
                        listed.append((Path(directory, entry.name), (stat.st_mtime_ns, stat.st_size)))
                    except OSError:
                        listed.append((Path(directory, entry.name), None))
        except OSError:
            continue
        for path, stat in listed:
            files.append(path)
            if stat is not None:
                stats[str(path)] = stat
        if directory != top:
            dirs = [name for name in dirs if name.lower() not in RUN_ARTIFACT_DIRS]
        for name in reversed(dirs):
            child = os.path.join(directory, name)
            if not os.path.islink(child):
                stack.append(child)
    return tuple(files), stats


def tree_files(root: str | os.PathLike[str]) -> tuple[Path, ...]:
    """Every file below ``root`` outside run artifact folders.

    One interface action asks for several adapter lists at once; they share a
    single walk for two seconds, unless a run folder was added or removed below
    ``root`` meanwhile. Concurrent callers wait for the walk in progress instead
    of starting their own. Paths keep the caller's spelling so ``relative_to()``
    and saved values still match.
    """

    top = os.fspath(root)
    try:
        # The names of the run folders join the folder's own time: NTFS does not always
        # update a folder's modification time when a subfolder is created in it (a new
        # run folder went unseen for the whole two seconds), while listing it is one call.
        modified = (os.stat(top).st_mtime_ns, frozenset(os.listdir(top)))
    except OSError:
        modified = None
    with _TREE_LOCK:
        cached = _TREE_CACHE.get(top)
        if cached is not None and cached[1] == modified and time.monotonic() - cached[0] < _TREE_TTL_S:
            return cached[2]
        files, stats = _walk(top)
        _TREE_CACHE[top] = (time.monotonic(), modified, files, stats)
        return files


def walked_stat(path: str | os.PathLike[str]) -> tuple[int, int] | None:
    """Modification time (ns) and size of ``path`` from a walk of the last two
    seconds, or ``None`` when no such walk listed it."""

    # Walk keys are ``str(Path)`` spellings; other spellings simply miss.
    key = os.fspath(path)
    now = time.monotonic()
    with _TREE_LOCK:
        for stamp, _modified, _files, stats in _TREE_CACHE.values():
            if now - stamp < _TREE_TTL_S and key in stats:
                return stats[key]
    return None


def file_state(path: str | os.PathLike[str]) -> tuple[int, int]:
    """Modification time (ns) and size, from the recent walk when it listed ``path``."""

    known = walked_stat(path)
    if known is not None:
        return known
    stat = os.stat(path)
    return stat.st_mtime_ns, stat.st_size


def invalidate_tree() -> None:
    with _TREE_LOCK:
        _TREE_CACHE.clear()
