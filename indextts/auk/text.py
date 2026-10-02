"""AuK instructions, settings and the duration model; plain Python, no PyTorch.

The interface imports this module directly, so it must stay import-light.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

# Upstream's zero-shot TTS and instruct TTS templates (docs/COOKBOOK.md, pe.config.yaml), verbatim:
# changing their wording makes the model read the instruction aloud or repeat the reference.
# Zero-shot TTS uses the English template for every language.
CLONE_TEMPLATE = 'Say the following with the same voice: "{text}"'
DESIGN_TEMPLATE = 'Generate speech based on the following description: "{description}". The content to speak is: "{text}".'
DESIGN_TEMPLATE_ZH = '请基于下面的描述: "{description}",生成语音内容"{text}".'
# A trained voice speaks without a reference; its training used this description.
TRAINED_VOICE_DESCRIPTION = "The trained speaker's natural voice, clear studio recording"

GENERATION_DEFAULTS = {
    "mode": "clone", "reference_text": "", "voice_description": "",
    "num_step": 32, "guidance_scale": 2.0, "sway_coef": -1.0, "solver": "euler",
    "max_reference_seconds": 12.0, "trim_reference_silence": True,
    "edge_seconds": 0.0, "match_loudness": True,
}
VOICE_MODES = ("clone", "design", "auto")
LANGUAGES = (("Auto", "AUTO"), ("English", "EN"), ("Chinese", "ZH"))

# Upstream's Prompt Enhancer duration model (pe.config.yaml runtime.duration): seconds of
# ordinary speech per UTF-8 byte of normalized text, by script; text under 10 bytes is
# slowed to 0.3x speed when no reference sets the pace.
SECONDS_PER_BYTE = {"en": 0.0656, "zh": 0.0803}
SHORT_TEXT_BYTES = 10
SHORT_TEXT_SPEED = 0.3

_NORMALIZER = None
_NORMALIZER_LOCK = threading.Lock()


def detect_language(text: str) -> str:
    return "zh" if re.search(r"[一-鿿]", str(text)) else "en"


def normalize_auk_text(text: str, language: str = "auto") -> str:
    """The app's WeText normalization (numbers, versions and units spelled out) for English and Chinese.

    Other languages pass through unchanged; training normalizes transcripts the same way.
    """
    global _NORMALIZER
    language = str(language or "auto").lower()
    if language == "auto":
        language = detect_language(text)
    if language not in {"en", "zh"} or not str(text).strip():
        return str(text)
    with _NORMALIZER_LOCK:
        if _NORMALIZER is None:
            from indextts.utils.front import TextNormalizer

            normalizer = TextNormalizer()
            normalizer.load()
            _NORMALIZER = normalizer
    stripped = text.strip()
    normalized = _NORMALIZER.normalize(stripped, lang=language) or stripped
    return text[:len(text) - len(text.lstrip())] + normalized + text[len(text.rstrip()):]


def text_units(text: str) -> int:
    """UTF-8 byte length of the spoken text."""
    return len(str(text).strip().encode("utf-8"))


def _script(character: str) -> str | None:
    code = ord(character)
    if 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF:
        return "zh"
    if character.isascii() and character.isalpha():
        return "en"
    return None


def f5_seconds(text: str, language: str = "en") -> float:
    """Upstream's language-weighted byte duration: CJK characters count as Chinese, ASCII
    letters as English, and every other character takes the script before it (else the
    one after it, else the text language)."""
    text = str(text).strip()
    scripts = [_script(character) for character in text]
    default = "zh" if str(language).lower() == "zh" else "en"
    seconds, previous = 0.0, None
    for index, character in enumerate(text):
        script = scripts[index] or previous
        if script is None:
            script = next((item for item in scripts[index + 1:] if item), default)
        previous = scripts[index] or previous
        seconds += len(character.encode("utf-8")) * SECONDS_PER_BYTE[script]
    return seconds


def quote_text(text: str) -> str:
    """Text placed inside the template's double quotes; inner quotes become typographic."""
    return re.sub(r'"([^"]*)"', "“\\1”", str(text).strip()).replace('"', "”")


def build_instruction(text: str, mode: str, description: str = "", language: str = "en") -> str:
    if mode == "clone":
        return CLONE_TEMPLATE.format(text=quote_text(text))
    description = str(description or "").strip() or TRAINED_VOICE_DESCRIPTION
    template = DESIGN_TEMPLATE_ZH if str(language).lower() == "zh" else DESIGN_TEMPLATE
    return template.format(description=quote_text(description), text=quote_text(text))


def trained_voice(adapter_path) -> dict:
    """The training run's saved voice record (pace and the description it was trained with)."""
    if not adapter_path:
        return {}
    source = Path(adapter_path)
    run_dir = source.parent.parent if source.parent.name.lower() == "best" else source.parent
    try:
        record = json.loads((run_dir / "auk_voice.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return record if isinstance(record, dict) else {}


def validate_voice_settings(settings):
    mode = settings.get("mode", "clone")
    if mode not in VOICE_MODES:
        raise ValueError("Unknown AuK voice mode")
    if mode == "design" and not str(settings.get("voice_description") or "").strip():
        raise ValueError("Describe the voice for Voice design, for example: a calm middle-aged man with a deep, warm voice.")
