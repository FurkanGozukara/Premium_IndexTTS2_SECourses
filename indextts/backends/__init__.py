"""Small, lazy speech-backend boundary; importing it never loads a model."""

MODEL_CHOICES = [("IndexTTS 2.5", "indextts"), ("OmniVoice", "omnivoice"), ("AuK", "auk")]
MODEL_IDS = tuple(key for _, key in MODEL_CHOICES)
MODEL_LABELS = {key: label for label, key in MODEL_CHOICES}
# Models whose generation settings are their own registry keys ("<model>.*", sent as
# request["<model>"]). Their voice modes other than "clone" need no reference.
SETTINGS_MODELS = ("omnivoice", "auk")
# Voice cloning without a chosen file falls back to the bundled demo voice.
DEFAULT_REFERENCE = "demo_voice.mp3"


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


def model_settings(values, model):
    """A settings model's own generation values without their prefix (its language travels separately)."""
    prefix = f"{model}."
    return {key[len(prefix):]: value for key, value in values.items()
            if key.startswith(prefix) and key != prefix + "language"}


def needs_reference(values):
    model = values.get("app.model") or "indextts"
    return model not in SETTINGS_MODELS or values.get(f"{model}.mode", "clone") == "clone"


def validate_model_settings(model, settings):
    """Reject impossible voice settings before an output folder is created."""
    if model == "omnivoice":
        from .omnivoice import validate_voice_settings
    elif model == "auk":
        from indextts.auk.text import validate_voice_settings
    else:
        return
    validate_voice_settings(settings)
