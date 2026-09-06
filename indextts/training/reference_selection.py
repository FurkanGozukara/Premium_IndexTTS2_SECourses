"""Shared preference for automatically selected training voice references."""
from __future__ import annotations

import math
from typing import Any, Mapping


AUTO_REFERENCE_TARGET_SECONDS = 15.0


def training_reference_priority(record: Mapping[str, Any]) -> tuple[float, bool, float]:
    """Preserve transcript quality, then prefer the nearest known duration."""
    wer = float(record.get("asr_wer", 0) or 0)
    if not math.isfinite(wer) or wer < 0:
        wer = math.inf
    duration = float(record.get("duration_s", 0) or 0)
    distance = (
        abs(duration - AUTO_REFERENCE_TARGET_SECONDS)
        if math.isfinite(duration) and duration > 0 else math.inf
    )
    return wer, not bool(record.get("boundary_words_match", True)), distance
