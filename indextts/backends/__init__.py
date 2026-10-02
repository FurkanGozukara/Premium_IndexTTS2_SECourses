"""Small, lazy speech-backend boundary; importing it never loads a model."""

MODEL_CHOICES = [("IndexTTS 2.5", "indextts"), ("OmniVoice", "omnivoice")]


def normalize_model(value):
    value = str(value or "indextts").strip().lower()
    if value not in {key for _, key in MODEL_CHOICES}:
        raise ValueError(f"Unknown speech model: {value}")
    return value


def checkpoint_matches_model(metadata, model):
    """Legacy checkpoints without a base-model tag belong to IndexTTS."""
    return ("omnivoice" in str(metadata.get("base_model", "")).lower()) == (model == "omnivoice")
