"""One authoritative checkpoint recommendation for every generation entry point."""
from __future__ import annotations

import json
from pathlib import Path

from .analysis import checkpoint_descriptor, discover_checkpoints, load_training_analysis


def recommended_generation_value(root: str | Path) -> str:
    """Prefer completed speech evaluation, then measured loss; honor explicit Base.

    Missing authoritative files are errors, never silent substitutions. Decoder
    adapters are not GPT checkpoints, including in the unevaluated-run fallback.
    """
    from .checkpoint_eval import load_checkpoint_eval
    from .speech_eval import load_speech_evaluation

    root = Path(root).expanduser().resolve()
    try:
        status = json.loads((root / "status.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        status = {}
    if not isinstance(status, dict):
        status = {}

    def checked(path: str | None, source: str) -> str:
        candidate = Path(path) if path else None
        if candidate is not None and not candidate.is_absolute():
            candidate = root / candidate
        if candidate is None or not candidate.is_file():
            raise ValueError(f"The {source} checkpoint is missing; rerun evaluation or restore that file")
        if candidate.name.lower().endswith(".s2mel.safetensors"):
            raise ValueError(f"The {source} recommendation is a voice decoder, not a GPT checkpoint")
        try:
            descriptor = checkpoint_descriptor(candidate)
        except Exception as exc:
            raise ValueError(f"The {source} checkpoint could not be inspected: {exc}") from exc
        if descriptor["kind"] == "decoder":
            raise ValueError(f"The {source} recommendation is a voice decoder, not a GPT checkpoint")
        return str(candidate.resolve())

    speech = load_speech_evaluation(root)
    if speech is not None and status.get("speech_evaluation_status", "complete") == "complete":
        if speech["recommended_kind"] == "base":
            return ""
        return checked(speech.get("recommended_checkpoint"), "speech-recommended")
    measured = load_checkpoint_eval(root)
    if measured is not None:
        if measured.recommended_kind == "base":
            return ""
        return checked(measured.recommended_checkpoint, "measured")
    analysis = load_training_analysis(root)
    path = analysis.recommended_checkpoint if analysis is not None else status.get("recommended_checkpoint")
    descriptors = discover_checkpoints(root)
    valid = {str(Path(item["path"]).resolve()) for item in descriptors}
    for value in (path, status.get("last_checkpoint")):
        if value:
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = root / candidate
            if str(candidate.resolve()) in valid:
                return str(candidate.resolve())
    if descriptors:
        return max(descriptors, key=lambda item: Path(item["path"]).stat().st_mtime)["path"]
    raise ValueError("No completed checkpoint is available yet")
