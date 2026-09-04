"""Console, JSON-file, and Gradio progress reporting."""

from __future__ import annotations

import ctypes
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, TextIO


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    try:
        value = max(0, int(round(float(seconds))))
    except (TypeError, ValueError, OverflowError):
        return "--"
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def format_rate(value: float | None, unit: str = "it/s") -> str:
    if value is None:
        return "--"
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(rate):
        return "--"
    precision = 2 if abs(rate) < 10 else 1
    separator = "" if unit == "x RT" else " "
    return f"{rate:.{precision}f}{separator}{unit}".strip()


def _enable_windows_vt(stream: TextIO) -> None:
    if os.name != "nt" or not getattr(stream, "isatty", lambda: False)():
        return
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


class ProgressReporter:
    def __init__(
        self,
        label: str,
        total: int | None = None,
        progress_file: str | os.PathLike[str] | None = None,
        gr_progress: Callable[..., Any] | None = None,
    ):
        self.label = str(label)
        self.total = max(0, int(total)) if total is not None else None
        self.progress_file = str(progress_file) if progress_file else None
        self.gr_progress = gr_progress
        self.stage = ""
        self.completed = 0
        self.desc = ""
        self.extra: dict[str, Any] = {}
        self.started_at = time.perf_counter()
        self._stream = sys.stdout
        self._tty = bool(getattr(self._stream, "isatty", lambda: False)())
        self._last_console_at = 0.0
        self._last_file_at = 0.0
        self._last_payload: dict[str, Any] | None = None
        self._finished = False
        _enable_windows_vt(self._stream)

    def _payload(self, now: float) -> dict[str, Any]:
        elapsed = max(0.0, now - self.started_at)
        total = self.total
        if total is not None and total > 0:
            fraction = min(1.0, max(0.0, self.completed / total))
            eta = elapsed / self.completed * (total - self.completed) if self.completed > 0 else None
        else:
            fraction = 0.0
            eta = None

        speed_unit = str(self.extra.get("speed_unit") or "it/s")
        explicit_speed = self.extra.get("speed")
        audio_seconds = self.extra.get("audio_seconds")
        if explicit_speed is not None:
            try:
                speed = float(explicit_speed)
            except (TypeError, ValueError):
                speed = None
        elif audio_seconds is not None and elapsed > 0:
            try:
                speed = float(audio_seconds) / elapsed
                speed_unit = "x RT"
            except (TypeError, ValueError):
                speed = None
        else:
            speed = self.completed / elapsed if elapsed > 0 and self.completed > 0 else None

        vram_used, vram_total = 0.0, 0.0
        try:
            import torch

            if torch.cuda.is_available():
                index = torch.cuda.current_device()
                free, total_bytes = torch.cuda.mem_get_info(index)
                vram_used = (total_bytes - free) / 1024**3
                vram_total = total_bytes / 1024**3
        except Exception:
            pass

        return {
            "fraction": fraction,
            "completed": self.completed,
            "total": total,
            "desc": self.desc,
            "stage": self.stage,
            "elapsed_s": elapsed,
            "eta_s": eta,
            "speed": speed,
            "speed_unit": speed_unit,
            "vram_used_gb": vram_used,
            "vram_total_gb": vram_total,
            "updated_at": time.time(),
            "extra": dict(self.extra),
        }

    def _render(self, payload: Mapping[str, Any]) -> str:
        total = payload.get("total")
        completed = int(payload.get("completed") or 0)
        fraction = float(payload.get("fraction") or 0.0)
        count = f"{completed}/{total}" if total is not None else str(completed)
        stage = f"{self.stage}: " if self.stage else ""
        parts = [
            f"[{fraction * 100:5.1f}%] {count} {self.label}",
            f"elapsed {format_duration(payload.get('elapsed_s'))}",
        ]
        if payload.get("eta_s") is not None:
            parts.append(f"ETA {format_duration(payload.get('eta_s'))}")
        if payload.get("speed") is not None:
            parts.append(format_rate(payload.get("speed"), str(payload.get("speed_unit") or "it/s")))
        if self.desc:
            parts.append(f"{stage}{self.desc}")
        elif stage:
            parts.append(stage.rstrip(": "))
        return " | ".join(parts)

    def _write_file(self, payload: Mapping[str, Any]) -> None:
        if not self.progress_file:
            return
        from indextts.utils.atomic_json import write_json_atomic

        write_json_atomic(
            Path(self.progress_file), payload, indent=None, allow_nan=False, trailing_newline=False
        )

    def update(
        self,
        completed: int | float,
        total: int | None = None,
        desc: str = "",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if total is not None:
            self.total = max(0, int(total))
        self.completed = max(0, int(completed))
        if self.total is not None:
            self.completed = min(self.completed, self.total)
        if desc:
            self.desc = str(desc)
        if extra is not None:
            self.extra = dict(extra)

        now = time.perf_counter()
        payload = self._payload(now)
        self._last_payload = payload
        final = bool(self.total is not None and self.completed >= self.total)
        if self._tty:
            print(f"\r\x1b[2K{self._render(payload)}", end="\n" if final else "", file=self._stream, flush=True)
            self._last_console_at = now
        elif final or self._last_console_at == 0.0 or now - self._last_console_at >= 1.0:
            print(self._render(payload), file=self._stream, flush=True)
            self._last_console_at = now

        if self.progress_file and (final or self._last_file_at == 0.0 or now - self._last_file_at >= 0.3):
            try:
                self._write_file(payload)
                self._last_file_at = now
            except OSError:
                pass
        if self.gr_progress is not None:
            try:
                self.gr_progress(payload["fraction"], desc=self.desc or self.stage or self.label)
            except Exception:
                pass
        if final:
            self._finished = True
        return payload

    def set_stage(self, name: str) -> None:
        self.stage = str(name or "")
        now = time.perf_counter()
        payload = self._payload(now)
        self._last_payload = payload
        if self.progress_file and (self._last_file_at == 0.0 or now - self._last_file_at >= 0.3):
            try:
                self._write_file(payload)
                self._last_file_at = now
            except OSError:
                pass

    def log(self, msg: str) -> None:
        if self._tty and self._last_console_at:
            print("\r\x1b[2K", end="", file=self._stream)
        print(str(msg), file=self._stream, flush=True)
        if self._tty and self._last_payload is not None and not self._finished:
            print(f"\r\x1b[2K{self._render(self._last_payload)}", end="", file=self._stream, flush=True)

    def finish(self) -> dict[str, Any]:
        if self._finished and self._last_payload is not None:
            return self._last_payload
        if self.total is None:
            self.total = max(1, self.completed)
        return self.update(self.total, desc=self.desc or "complete", extra=self.extra)


def read_progress_file(path: str | os.PathLike[str] | None) -> dict[str, Any] | None:
    if not path:
        return None
    from indextts.utils.atomic_json import read_json_retry

    value = read_json_retry(path, None, encoding="utf-8")
    return value if isinstance(value, dict) else None


__all__ = ["ProgressReporter", "format_duration", "format_rate", "read_progress_file"]
