"""Pronunciation dictionary and unknown-word check for generation.

IndexTTS 2.5 reads ``<word|PHONES>`` annotations natively: ARPAbet phones with
stress digits, syllables separated by ``.``, wrapped by the engine into
``<|SPECIAL_TOKEN_1|>`` markers (``indextts.infer_v2_5.apply_pronunciation_annotations``).
This module keeps a user dictionary of such readings (or plain respellings),
finds the words in a text that the selected voice never spoke in training and
that the base model has no reading for, proposes ARPAbet for them from the CMU
Pronouncing Dictionary with compound, CamelCase, acronym and letter-to-sound
fallbacks, and rewrites the text before synthesis. CPU only, no network.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

from indextts.utils.atomic_json import read_json_retry, write_json_atomic


DICTIONARY_VERSION = 1
KIND_PHONEMES = "phonemes"
KIND_RESPELLING = "respelling"
# "unseen": rewrite the word only when the selected voice never spoke it in training, so a
# fine-tuned reading learned from the recordings is never replaced; "always": rewrite every time.
SCOPE_UNSEEN = "unseen"
SCOPE_ALWAYS = "always"
SCOPES = (SCOPE_UNSEEN, SCOPE_ALWAYS)
VOWELS = frozenset({"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"})
CONSONANTS = frozenset({
    "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S", "SH", "T", "TH",
    "V", "W", "Y", "Z", "ZH",
})
_PHONE_RE = re.compile(r"^(?:[A-Z]{1,2}[0-2]?|\.)$")
_ANNOTATION_RE = re.compile(r"<([^|>\n]+)\|([^>\n]+)>")
_WORD_CANDIDATE_RE = re.compile(r"[A-Za-z][A-Za-z0-9'’]*(?:[.-][A-Za-z0-9]+)*")
_PAUSE_TAG_RE = re.compile(r"\[pause:[^\]]*\]|<pause=[^>]*>", re.IGNORECASE)
_CAMEL_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
_LETTER_DIGIT_RE = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)*")
# English syllable onsets used by the maximal-onset syllabifier.
_ONSETS = frozenset(
    {
        (), ("B",), ("CH",), ("D",), ("DH",), ("F",), ("G",), ("HH",), ("JH",), ("K",), ("L",), ("M",), ("N",),
        ("P",), ("R",), ("S",), ("SH",), ("T",), ("TH",), ("V",), ("W",), ("Y",), ("Z",), ("ZH",),
        ("B", "L"), ("B", "R"), ("B", "Y"), ("D", "R"), ("D", "W"), ("D", "Y"), ("F", "L"), ("F", "R"), ("F", "Y"),
        ("G", "L"), ("G", "R"), ("G", "W"), ("HH", "Y"), ("K", "L"), ("K", "R"), ("K", "W"), ("K", "Y"),
        ("M", "Y"), ("N", "Y"), ("P", "L"), ("P", "R"), ("P", "Y"), ("S", "K"), ("S", "L"), ("S", "M"), ("S", "N"),
        ("S", "P"), ("S", "T"), ("S", "W"), ("S", "F"), ("SH", "R"), ("T", "R"), ("T", "W"), ("T", "Y"),
        ("TH", "R"), ("TH", "W"), ("V", "Y"), ("Z", "Y"),
        ("S", "K", "R"), ("S", "K", "W"), ("S", "K", "Y"), ("S", "P", "L"), ("S", "P", "R"), ("S", "P", "Y"),
        ("S", "T", "R"), ("S", "T", "Y"),
    }
)
# How each letter is read on its own, for acronyms (G P U) and letter runs (bf16 -> B F sixteen).
LETTER_PHONES: dict[str, str] = {
    "A": "EY1", "B": "B IY1", "C": "S IY1", "D": "D IY1", "E": "IY1", "F": "EH1 F", "G": "JH IY1",
    "H": "EY1 CH", "I": "AY1", "J": "JH EY1", "K": "K EY1", "L": "EH1 L", "M": "EH1 M", "N": "EH1 N",
    "O": "OW1", "P": "P IY1", "Q": "K Y UW1", "R": "AA1 R", "S": "EH1 S", "T": "T IY1", "U": "Y UW1",
    "V": "V IY1", "W": "D AH1 B AH0 L Y UW0", "X": "EH1 K S", "Y": "W AY1", "Z": "Z IY1",
}
# Letter-to-sound rules for words no dictionary knows (technical names such as "qwen").
# Longest match first; vowels get stress later (primary on the first vowel).
_LTS_RULES: tuple[tuple[str, str], ...] = (
    ("tion", "SH AH N"), ("sion", "ZH AH N"), ("ture", "CH ER"), ("ough", "OW"), ("augh", "AO"),
    ("igh", "AY"), ("eigh", "EY"), ("tch", "CH"), ("dge", "JH"), ("sch", "S K"), ("chr", "K R"),
    ("ch", "CH"), ("sh", "SH"), ("th", "TH"), ("ph", "F"), ("wh", "W"), ("ck", "K"), ("ng", "NG"),
    ("qu", "K W"), ("gh", "G"), ("kn", "N"), ("wr", "R"), ("ps", "S"), ("gn", "N"),
    ("ee", "IY"), ("ea", "IY"), ("ie", "IY"), ("ei", "EY"), ("ey", "EY"), ("ay", "EY"), ("ai", "EY"),
    ("oo", "UW"), ("ou", "AW"), ("ow", "OW"), ("oa", "OW"), ("oe", "OW"), ("oi", "OY"), ("oy", "OY"),
    ("au", "AO"), ("aw", "AO"), ("ue", "UW"), ("ui", "UW"), ("eu", "Y UW"), ("ew", "UW"),
    ("a", "AE"), ("e", "EH"), ("i", "IH"), ("o", "AA"), ("u", "AH"), ("y", "IY"),
    ("b", "B"), ("c", "K"), ("d", "D"), ("f", "F"), ("g", "G"), ("h", "HH"), ("j", "JH"), ("k", "K"),
    ("l", "L"), ("m", "M"), ("n", "N"), ("p", "P"), ("q", "K"), ("r", "R"), ("s", "S"), ("t", "T"),
    ("v", "V"), ("w", "W"), ("x", "K S"), ("z", "Z"),
)
_SOFT_C = frozenset("eiy")
_SILENT_FINAL_E_MIN_LETTERS = 3


@dataclass(frozen=True)
class DictionaryEntry:
    word: str
    pronunciation: str
    kind: str = KIND_PHONEMES
    source: str = "manual"
    note: str = ""
    scope: str = SCOPE_UNSEEN

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def applies_to(self, known_words: Iterable[str] | frozenset[str] | set[str]) -> bool:
        """Whether the entry rewrites text for a voice whose training vocabulary is ``known_words``."""

        if self.scope == SCOPE_ALWAYS:
            return True
        known = known_words if isinstance(known_words, (set, frozenset)) else {str(word).casefold() for word in known_words}
        return not any(form in known for form in _base_forms(self.word))


def is_phoneme_string(value: str) -> bool:
    """True when every token is an ARPAbet phone (optional stress digit) or a syllable dot."""

    tokens = str(value or "").upper().split()
    if not tokens:
        return False
    for token in tokens:
        if token == ".":
            continue
        if not _PHONE_RE.match(token):
            return False
        base = token.rstrip("012")
        if base not in VOWELS and base not in CONSONANTS:
            return False
    return True


def normalize_scope(value: Any) -> str:
    text = str(value or "").strip().lower()
    return SCOPE_ALWAYS if text.startswith("always") else SCOPE_UNSEEN


def normalize_entry(
    word: str,
    pronunciation: str,
    *,
    source: str = "manual",
    note: str = "",
    scope: str = SCOPE_UNSEEN,
) -> DictionaryEntry | None:
    """Validate one dictionary row; phoneme strings are uppercased, respellings kept as typed."""

    clean_word = str(word or "").strip()
    reading = " ".join(str(pronunciation or "").split())
    if not clean_word or not reading or any(char in clean_word for char in "<>|"):
        return None
    if is_phoneme_string(reading):
        return DictionaryEntry(
            clean_word, reading.upper(), KIND_PHONEMES, str(source or "manual"), str(note or ""), normalize_scope(scope)
        )
    if any(char in reading for char in "<>|"):
        return None
    return DictionaryEntry(clean_word, reading, KIND_RESPELLING, str(source or "manual"), str(note or ""), normalize_scope(scope))


def load_dictionary(path: str | os.PathLike[str]) -> list[DictionaryEntry]:
    payload = read_json_retry(Path(path), None)
    rows: Iterable[Any]
    if isinstance(payload, Mapping):
        rows = payload.get("entries") or []
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []
    entries: list[DictionaryEntry] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        entry = normalize_entry(
            str(row.get("word") or ""), str(row.get("pronunciation") or ""),
            source=str(row.get("source") or "manual"), note=str(row.get("note") or ""),
            scope=str(row.get("scope") or SCOPE_UNSEEN),
        )
        if entry is None or entry.word.casefold() in seen:
            continue
        seen.add(entry.word.casefold())
        entries.append(entry)
    return entries


def save_dictionary(path: str | os.PathLike[str], entries: Sequence[DictionaryEntry]) -> Path:
    ordered = sorted({entry.word.casefold(): entry for entry in entries}.values(), key=lambda item: item.word.casefold())
    payload = {"version": DICTIONARY_VERSION, "entries": [entry.to_dict() for entry in ordered]}
    return Path(write_json_atomic(Path(path), payload, indent=2, ensure_ascii=False))


def entries_from_rows(rows: Iterable[Sequence[Any]] | None) -> list[DictionaryEntry]:
    """Dictionary entries from a table of [word, reading, kind, scope, source] rows (blank rows skipped)."""

    entries: list[DictionaryEntry] = []
    seen: set[str] = set()
    for row in rows or []:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        scope = row[3] if len(row) > 3 else SCOPE_UNSEEN
        source = str(row[4] or "manual") if len(row) > 4 and str(row[4] or "").strip() else "manual"
        entry = normalize_entry(str(row[0] or ""), str(row[1] or ""), source=source, scope=str(scope or SCOPE_UNSEEN))
        if entry is None or entry.word.casefold() in seen:
            continue
        seen.add(entry.word.casefold())
        entries.append(entry)
    return entries


# --------------------------------------------------------------------------- #
# CMU dictionary access and reading construction
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def cmu_dictionary() -> dict[str, list[list[str]]]:
    """The CMU Pronouncing Dictionary (empty when the optional ``cmudict`` package is missing)."""

    try:
        import cmudict  # type: ignore

        return dict(cmudict.dict())
    except Exception:
        return {}


def cmu_available() -> bool:
    return bool(cmu_dictionary())


def lookup_word(word: str) -> list[str] | None:
    """Phones of a dictionary word (first pronunciation), or None."""

    variants = cmu_dictionary().get(str(word or "").casefold().replace("’", "'"))
    return list(variants[0]) if variants else None


def _is_vowel(phone: str) -> bool:
    return phone.rstrip("012") in VOWELS


def syllabify(phones: Sequence[str]) -> str:
    """Join ARPAbet phones with ``.`` between syllables (maximal legal onset)."""

    clean = [phone for phone in phones if phone and phone != "."]
    nuclei = [index for index, phone in enumerate(clean) if _is_vowel(phone)]
    if len(nuclei) <= 1:
        return " ".join(clean)
    boundaries: list[int] = []
    for previous, current in zip(nuclei, nuclei[1:]):
        cluster = clean[previous + 1 : current]
        split = len(cluster)
        while split > 0 and tuple(cluster[len(cluster) - split :]) not in _ONSETS:
            split -= 1
        boundaries.append(current - split)
    parts: list[str] = []
    start = 0
    for boundary in boundaries:
        parts.append(" ".join(clean[start:boundary]))
        start = boundary
    parts.append(" ".join(clean[start:]))
    return " . ".join(part for part in parts if part)


def _stress(phones: Sequence[str]) -> list[str]:
    """Primary stress on the first vowel, none on the others (used for rule-built readings)."""

    result: list[str] = []
    stressed = False
    for phone in phones:
        if _is_vowel(phone) and not phone[-1].isdigit():
            result.append(phone + ("1" if not stressed else "0"))
            stressed = True
        else:
            result.append(phone)
    return result


def letters_to_sound(word: str) -> list[str]:
    """Rule-based English letter-to-sound for a word no dictionary knows."""

    text = re.sub(r"[^a-z]", "", str(word or "").lower())
    if not text:
        return []
    silent_e = len(text) >= _SILENT_FINAL_E_MIN_LETTERS and text.endswith("e") and text[-2] not in "aeiou"
    if silent_e:
        text = text[:-1]
    phones: list[str] = []
    position = 0
    while position < len(text):
        for pattern, reading in _LTS_RULES:
            if text.startswith(pattern, position):
                if pattern == "c" and position + 1 < len(text) and text[position + 1] in _SOFT_C:
                    phones.append("S")
                elif pattern == "g" and position + 1 < len(text) and text[position + 1] in "eiy" and len(text) > 2:
                    phones.append("JH")
                elif pattern == "y" and position == 0:
                    phones.append("Y")
                elif pattern == "s" and position == len(text) - 1 and phones and _is_vowel(phones[-1]) is False and phones[-1] in {"B", "D", "G", "V", "Z", "M", "N", "NG", "L", "R"}:
                    phones.append("Z")
                else:
                    phones.extend(reading.split())
                position += len(pattern)
                break
        else:
            position += 1
    if silent_e and phones:
        # A silent final e lengthens the vowel before the last consonant: "kwane" -> K W EY N.
        for index in range(len(phones) - 1, -1, -1):
            if _is_vowel(phones[index]):
                phones[index] = {"AE": "EY", "EH": "IY", "IH": "AY", "AA": "OW", "AH": "UW"}.get(phones[index], phones[index])
                break
    return _stress(phones)


def _acronym_phones(letters: str) -> list[str]:
    phones: list[str] = []
    for letter in letters.upper():
        phones.extend(LETTER_PHONES.get(letter, "").split())
    return phones


def _dictionary_compound(word: str) -> list[list[str]] | None:
    """Split a lowercase letter run into dictionary words (longest first), e.g. runpod -> run + pod."""

    dictionary = cmu_dictionary()
    if not dictionary:
        return None
    lowered = word.lower()

    def solve(start: int) -> list[list[str]] | None:
        if start == len(lowered):
            return []
        for end in range(len(lowered), start + 2, -1):
            piece = lowered[start:end]
            variants = dictionary.get(piece)
            if variants and (end - start >= 3):
                rest = solve(end)
                if rest is not None:
                    return [list(variants[0])] + rest
        remaining = len(lowered) - start
        if 0 < remaining <= 2 and start > 0:
            return [_acronym_phones(lowered[start:])]
        return None

    return solve(0)


@dataclass(frozen=True)
class Suggestion:
    word: str
    pronunciation: str
    kind: str
    method: str
    confidence: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _reading_for_letters(piece: str) -> tuple[list[str], str, str]:
    """Phones for one alphabetic piece plus the method and confidence used."""

    if piece.isupper() and len(piece) <= 3:
        return _acronym_phones(piece), "acronym letters", "medium"
    exact = lookup_word(piece)
    if exact:
        return exact, "dictionary", "high"
    if piece.isupper() and len(piece) <= 5:
        return _acronym_phones(piece), "acronym letters", "medium"
    compound = _dictionary_compound(piece)
    if compound:
        phones: list[str] = []
        for part in compound:
            phones.extend(part)
        return phones, "dictionary compound", "medium"
    if len(piece) <= 2:
        return _acronym_phones(piece), "acronym letters", "medium"
    return letters_to_sound(piece), "letter-to-sound rules", "low"


def suggest_pronunciation(word: str) -> Suggestion | None:
    """Propose a reading for a word: CMU phones when possible, otherwise a respelling."""

    clean = str(word or "").strip().strip(".,;:!?'\"()[]")
    if not clean:
        return None
    if clean.isdigit():
        return None
    if any(char.isdigit() for char in clean):
        # Letters are spelled out, digits (and version numbers such as 2.2) are left for the
        # engine's number normalizer, which reads them as "sixteen" or "two point two".
        pieces = _LETTER_DIGIT_RE.findall(clean)
        spoken: list[str] = []
        for piece in pieces:
            if piece[0].isdigit():
                spoken.append(piece)
            elif lookup_word(piece):
                spoken.append(piece.lower())
            else:
                spoken.append(" ".join(piece.upper()))
        reading = " ".join(spoken)
        return Suggestion(clean, reading, KIND_RESPELLING, "letters and numbers", "medium")
    camel = _CAMEL_RE.findall(clean.replace("-", " ").replace("'", "")) if any(char.isupper() for char in clean[1:]) else [clean]
    pieces = [piece for piece in camel if piece] or [clean]
    phones: list[str] = []
    methods: list[str] = []
    confidences: list[str] = []
    for piece in pieces:
        piece_phones, method, confidence = _reading_for_letters(piece)
        if not piece_phones:
            return None
        phones.extend(piece_phones)
        methods.append(method)
        confidences.append(confidence)
    order = {"low": 0, "medium": 1, "high": 2}
    confidence = min(confidences, key=lambda item: order[item])
    method = " + ".join(dict.fromkeys(methods))
    return Suggestion(clean, syllabify(phones), KIND_PHONEMES, method, confidence)


# --------------------------------------------------------------------------- #
# Text checks and rewriting
# --------------------------------------------------------------------------- #


def candidate_words(text: str) -> list[str]:
    """Distinct words worth checking, in order of appearance (pause tags and annotations skipped)."""

    value = _PAUSE_TAG_RE.sub(" ", str(text or ""))
    value = _ANNOTATION_RE.sub(" ", value)
    seen: set[str] = set()
    words: list[str] = []
    for match in _WORD_CANDIDATE_RE.finditer(value):
        word = match.group(0).rstrip(".-'’")
        if not word or word.casefold() in seen:
            continue
        seen.add(word.casefold())
        words.append(word)
    return words


def _base_forms(word: str) -> list[str]:
    lowered = word.casefold().replace("’", "'")
    forms = [lowered]
    for suffix in ("'s", "s'", "'"):
        if lowered.endswith(suffix) and len(lowered) > len(suffix) + 1:
            forms.append(lowered[: -len(suffix)])
    if lowered.endswith("es") and len(lowered) > 4:
        forms.append(lowered[:-2])
    if lowered.endswith("s") and len(lowered) > 3:
        forms.append(lowered[:-1])
    return forms


def check_text(
    text: str,
    *,
    known_words: Iterable[str] = (),
    entries: Sequence[DictionaryEntry] = (),
    token_len: Callable[[str], int] | None = None,
) -> list[dict[str, Any]]:
    """Words the voice never spoke in training and the base model has no dictionary reading for.

    Each row: word, seen_in_training, in_dictionary, dictionary_reading, fragments
    (tokenizer pieces for the lowercased word), suggestion, method, confidence.
    Only words that are neither in the pronunciation dictionary nor in the CMU
    dictionary nor in the training vocabulary are returned; every other word
    already has a reading the model knows.
    """

    known = {str(word).casefold() for word in known_words}
    lexicon = {entry.word.casefold(): entry for entry in entries}
    rows: list[dict[str, Any]] = []
    for word in candidate_words(text):
        forms = _base_forms(word)
        entry = next((lexicon[form] for form in forms if form in lexicon), None)
        seen = any(form in known for form in forms)
        in_cmu = any(lookup_word(form) for form in forms) if not any(char.isdigit() for char in word) else False
        if entry is not None or seen or in_cmu:
            continue
        fragments = 0
        if token_len is not None:
            try:
                fragments = int(token_len(" " + word.lower()))
            except Exception:
                fragments = 0
        suggestion = suggest_pronunciation(word)
        rows.append(
            {
                "word": word,
                "seen_in_training": seen,
                "in_dictionary": entry is not None,
                "dictionary_reading": entry.pronunciation if entry else "",
                "fragments": fragments,
                "suggestion": suggestion.pronunciation if suggestion else "",
                "kind": suggestion.kind if suggestion else "",
                "method": suggestion.method if suggestion else "",
                "confidence": suggestion.confidence if suggestion else "",
            }
        )
    return rows


def _plural_phones(phones: str) -> str:
    last = phones.split()[-1].rstrip("012") if phones.split() else ""
    if last in {"S", "Z", "SH", "ZH", "CH", "JH"}:
        return phones + " IH0 Z"
    if last in {"P", "T", "K", "F", "TH"}:
        return phones + " S"
    return phones + " Z"


def apply_dictionary(
    text: str,
    entries: Sequence[DictionaryEntry],
    *,
    known_words: Iterable[str] = (),
) -> str:
    """Rewrite whole-word matches: phoneme entries become ``<word|PHONES>``, respellings replace the word.

    Matching is case-insensitive, respects word boundaries, keeps existing
    ``<word|reading>`` annotations and pause tags untouched, and extends to plural
    and possessive forms (``LoRAs``, ``Qwen's``) by appending the fitting phones.
    Entries with the ``unseen`` scope are skipped for words in ``known_words``, the
    vocabulary the selected voice was trained on.
    """

    if not entries or not str(text or ""):
        return str(text or "")
    known = {str(word).casefold() for word in known_words}
    entries = [entry for entry in entries if entry.applies_to(known)]
    if not entries:
        return str(text or "")
    protected: list[str] = []

    def protect(match: re.Match) -> str:
        protected.append(match.group(0))
        return f"\x00{len(protected) - 1}\x00"

    value = _ANNOTATION_RE.sub(protect, str(text))
    value = _PAUSE_TAG_RE.sub(protect, value)
    for entry in sorted(entries, key=lambda item: len(item.word), reverse=True):
        pattern = re.compile(
            r"(?<![\w'’])(" + re.escape(entry.word) + r")(?P<suffix>'s|’s|es|s)?(?![\w'’])",
            re.IGNORECASE,
        )

        def replace(match: re.Match, entry: DictionaryEntry = entry) -> str:
            suffix = match.group("suffix") or ""
            surface = match.group(1) + suffix
            if entry.kind == KIND_PHONEMES:
                phones = _plural_phones(entry.pronunciation) if suffix else entry.pronunciation
                return f"<{surface}|{phones}>"
            reading = entry.pronunciation
            if suffix:
                reading += "'s" if suffix.lower() in {"'s", "’s"} else suffix.lower()
            return reading

        value = pattern.sub(replace, value)
    return re.sub(r"\x00(\d+)\x00", lambda match: protected[int(match.group(1))], value)


def default_dictionary_path(root: str | os.PathLike[str]) -> Path:
    return Path(root).expanduser() / "pronunciations" / "dictionary.json"


def dictionary_rows(entries: Sequence[DictionaryEntry]) -> list[list[str]]:
    return [[entry.word, entry.pronunciation, entry.kind, entry.scope, entry.source] for entry in entries]


# Readings for widely used AI and hardware names the base model has no dictionary entry for. They
# ship with the "unseen" scope, so a voice trained on recordings that contain the word keeps its own
# reading, and every entry can be edited or removed in the generation tab. Words the base model already
# reads well from their spelling (GGUF, RunPod, Musubi in a listening check) are deliberately absent.
BUILTIN_ENTRIES: tuple[tuple[str, str], ...] = (
    ("ComfyUI", "K AH1 M . F IY0 . Y UW1 . AY1"),
    ("SwarmUI", "S W AO1 R M . Y UW1 . AY1"),
    ("Qwen", "K W EH1 N"),
    ("CUDA", "K UW1 . D AH0"),
    ("cuDNN", "K UW1 . D IY1 . EH1 N . EH1 N"),
    ("VRAM", "V IY1 . R AE1 M"),
    ("LoRA", "L AO1 . R AH0"),
    ("DoRA", "D AO1 . R AH0"),
    ("PyTorch", "P AY1 . T AO1 R CH"),
    ("xformers", "EH1 K S . F AO1 R . M ER0 Z"),
    ("SageAttention", "S EY1 JH . AH0 . T EH1 N . SH AH0 N"),
    ("Kohya", "K OW1 . HH Y AH0"),
    ("Nunchaku", "N AH0 N . CH AA1 . K UW0"),
    ("Hunyuan", "HH UW1 N . Y UW0 . AA1 N"),
    ("Krea", "K R IY1 . AH0"),
    ("ControlNet", "K AH0 N . T R OW1 L . N EH1 T"),
    ("NVFP4", "N V F P 4"),
    ("bf16", "B F 16"),
    ("fp16", "F P 16"),
    ("fp8", "F P 8"),
    ("SDXL", "EH1 S . D IY1 . EH1 K S . EH1 L"),
)


def builtin_entries() -> list[DictionaryEntry]:
    entries: list[DictionaryEntry] = []
    for word, reading in BUILTIN_ENTRIES:
        entry = normalize_entry(word, reading, source="builtin")
        if entry is not None:
            entries.append(entry)
    return entries


def merge_entries(existing: Sequence[DictionaryEntry], additions: Iterable[DictionaryEntry]) -> list[DictionaryEntry]:
    merged = {entry.word.casefold(): entry for entry in existing}
    for entry in additions:
        merged[entry.word.casefold()] = entry
    return sorted(merged.values(), key=lambda item: item.word.casefold())


__all__ = [
    "BUILTIN_ENTRIES",
    "DICTIONARY_VERSION",
    "DictionaryEntry",
    "KIND_PHONEMES",
    "KIND_RESPELLING",
    "LETTER_PHONES",
    "SCOPES",
    "SCOPE_ALWAYS",
    "SCOPE_UNSEEN",
    "Suggestion",
    "apply_dictionary",
    "builtin_entries",
    "candidate_words",
    "check_text",
    "cmu_available",
    "default_dictionary_path",
    "dictionary_rows",
    "entries_from_rows",
    "is_phoneme_string",
    "letters_to_sound",
    "load_dictionary",
    "lookup_word",
    "merge_entries",
    "normalize_entry",
    "normalize_scope",
    "save_dictionary",
    "suggest_pronunciation",
    "syllabify",
]
