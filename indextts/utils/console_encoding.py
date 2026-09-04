"""Console stream setup shared by CLI and subprocess entry points."""

from __future__ import annotations

import sys


def configure_console_output() -> None:
    """Make redirected or legacy-code-page consoles loss-tolerant."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(errors="replace", line_buffering=True, write_through=True)
        except (OSError, TypeError, ValueError):
            try:
                reconfigure(errors="replace")
            except (OSError, TypeError, ValueError):
                pass


__all__ = ["configure_console_output"]
