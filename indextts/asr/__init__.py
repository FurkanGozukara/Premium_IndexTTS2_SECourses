"""Speech recognition built into the app: Whisper large-v3 INT8 ConvRot (see ``recognizer``)."""

from .recognizer import (BEST_QUALITY, BUILTIN_MODEL, FALLBACK_MODEL, Recognition, RecognizedWord, audio_16k,
                         convrot_supported, description, ensure_model, is_builtin_model, model_folder, park, recognize,
                         unload)

__all__ = [
    "BEST_QUALITY", "BUILTIN_MODEL", "FALLBACK_MODEL", "Recognition", "RecognizedWord", "audio_16k", "convrot_supported",
    "description", "ensure_model", "is_builtin_model", "model_folder", "park", "recognize", "unload",
]
