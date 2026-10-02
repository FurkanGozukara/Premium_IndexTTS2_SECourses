"""Small, lazy speech-backend boundary; importing it never loads a model."""

MODEL_CHOICES = [("IndexTTS 2.5", "indextts"), ("OmniVoice", "omnivoice"), ("AuK", "auk")]
MODEL_IDS = tuple(key for _, key in MODEL_CHOICES)
MODEL_LABELS = {key: label for label, key in MODEL_CHOICES}


def normalize_model(value):
    value = str(value or "indextts").strip().lower()
    if value not in MODEL_IDS:
        raise ValueError(f"Unknown speech model: {value}")
    return value


def checkpoint_model(metadata):
    """The speech model an adapter or checkpoint was trained for; untagged ones are IndexTTS."""
    base = str((metadata or {}).get("base_model", "")).lower()
    if "omnivoice" in base:
        return "omnivoice"
    if "auk" in base:
        return "auk"
    return "indextts"


def checkpoint_matches_model(metadata, model):
    return checkpoint_model(metadata) == normalize_model(model)
