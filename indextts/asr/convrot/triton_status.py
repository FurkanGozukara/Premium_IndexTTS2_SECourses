"""Progress reports for the ConvRot engine's Triton kernel cache.

The first ConvRot run on a GPU compiles the Triton kernels and benchmarks the GEMM tile
configurations of every new shape class (autotuning). Triton stores both in its kernel cache
and every later process reuses them, so only the first run of a shape class is slow. The
reports below say so in the console, and a summary tells how many tuned shapes came from the
cache.

Ported from Whisper-WebUI Premium (SECourses), modules/whisper/convrot at revision 613beba (September 2026).
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Callable, Optional

import logging

logger = logging.getLogger("indextts.asr")

_lock = threading.Lock()
_listener: Optional[Callable[[str], None]] = None
_counts = {"reused": 0, "tuned": 0, "tune_seconds": 0.0}


def cache_dir() -> str:
    try:
        from triton import knobs

        return str(knobs.cache.dir)
    except Exception:
        return "the Triton cache"


def report(message: str) -> None:
    """Print ``message`` in the console and pass it to the active listener, if any."""
    print(f">> {message}", flush=True)
    listener = _listener
    if listener is not None:
        try:
            listener(message)
        except Exception:
            pass


@contextmanager
def report_to(listener: Optional[Callable[[str], None]]):
    """Also send kernel cache messages to ``listener`` (for example a progress line)."""
    global _listener
    previous = _listener
    _listener = listener
    try:
        yield
    finally:
        _listener = previous


def counts() -> dict:
    with _lock:
        return dict(_counts)


def summary_since(before: dict) -> Optional[str]:
    """One line describing the kernel cache activity since ``before`` (from ``counts()``), or None."""
    now = counts()
    reused = now["reused"] - before.get("reused", 0)
    tuned = now["tuned"] - before.get("tuned", 0)
    if not reused and not tuned:
        return None
    if not tuned:
        return f"Triton kernel cache reused: {reused} tuned shape(s) loaded from {cache_dir()}"
    seconds = now["tune_seconds"] - before.get("tune_seconds", 0.0)
    return (
        f"Triton kernel cache: {tuned} shape(s) tuned now in {seconds:.1f} s and saved, "
        f"{reused} reused from {cache_dir()}. The next runs reuse them."
    )


def _shape_text(tuning_key) -> str:
    try:
        m_bucket, n, k = (int(value) for value in tuning_key[:3])
        return f"M<={m_bucket} N={n} K={k}"
    except Exception:
        return str(tuning_key)


def instrument_autotuner(autotuner, label: str):
    """Report each new tuning of one ``triton.autotune`` kernel (instance-level, nothing global).

    Triton calls ``check_disk_cache`` for every shape class it has not tuned in this process: it
    loads the stored result, or benchmarks every configuration (``_bench``) and stores the winner.
    Other Triton versions without these methods simply run without reports.
    """
    check_disk_cache = getattr(autotuner, "check_disk_cache", None)
    bench = getattr(autotuner, "_bench", None)
    if check_disk_cache is None or bench is None or getattr(autotuner, "_convrot_reporting", False):
        return autotuner

    progress = {"done": 0, "total": 0, "shape": ""}

    def reporting_check_disk_cache(tuning_key, configs, bench_fn):
        shape = _shape_text(tuning_key)
        progress.update(done=0, total=len(configs), shape=shape)
        started = time.perf_counter()

        def reporting_benchmark():
            with _lock:
                number = _counts["tuned"] + 1
            report(
                f"Triton: tuning #{number}: {label} for {shape} on this GPU, {len(configs)} configurations "
                "(first run only; the result is saved in the kernel cache)..."
            )
            bench_fn()

        reused = check_disk_cache(tuning_key, configs, reporting_benchmark)
        elapsed = time.perf_counter() - started
        with _lock:
            if reused:
                _counts["reused"] += 1
            else:
                _counts["tuned"] += 1
                _counts["tune_seconds"] += elapsed
        if not reused:
            report(f"Triton: {label} for {shape} tuned in {elapsed:.1f} s")
        return reused

    def reporting_bench(*args, config, **meta):
        result = bench(*args, config=config, **meta)
        progress["done"] += 1
        logger.debug("Triton: %s %s configuration %d/%d", label, progress["shape"], progress["done"], progress["total"])
        return result

    autotuner.check_disk_cache = reporting_check_disk_cache
    autotuner._bench = reporting_bench
    autotuner._convrot_reporting = True
    return autotuner
