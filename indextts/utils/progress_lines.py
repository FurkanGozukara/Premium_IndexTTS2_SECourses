"""Console lines of a child process as a terminal would show them.

A worker's progress bars (tqdm) redraw one line with carriage returns. Read through a text-mode pipe, every redraw
arrives as a line of its own, and a log fills with hundreds of them ("  4%|4  | 1/25", " 24%|##4 | 6/25", ...).
:func:`collapse_progress_lines` keeps only the state a terminal would leave on screen: a bar's 100 % line, or its last
state when something else is printed first or the stream ends.
"""

from __future__ import annotations

import re
from typing import Iterable, Iterator

# "  4%|4         | 1/25 [...]", "Loading weights:  47%|████  | 146/310 [...]", "Fetching 10 files: 100%|..."
_PROGRESS_LINE = re.compile(r"^\s*(?:[^|\r\n]{0,80}?:\s*)?\d{1,3}%\|")
_FINISHED = re.compile(r"(?:^|\s|:)100%\|")


def is_progress_line(line: str) -> bool:
    return bool(_PROGRESS_LINE.match(line))


def collapse_progress_lines(lines: Iterable[str]) -> Iterator[str]:
    pending: str | None = None
    for line in lines:
        if is_progress_line(line):
            if _FINISHED.search(line):
                pending = None
                yield line
            else:
                pending = line
            continue
        if pending is not None:
            yield pending
            pending = None
        yield line
    if pending is not None:
        yield pending
