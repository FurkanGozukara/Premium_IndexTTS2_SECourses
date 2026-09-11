from __future__ import annotations

from pathlib import Path

import pytest

from indextts.utils import pronunciation as pron
from indextts.utils.pronunciation import (
    DictionaryEntry,
    apply_dictionary,
    builtin_entries,
    candidate_words,
    check_text,
    dictionary_rows,
    entries_from_rows,
    is_phoneme_string,
    letters_to_sound,
    load_dictionary,
    merge_entries,
    normalize_entry,
    save_dictionary,
    suggest_pronunciation,
    syllabify,
)


def test_normalize_entry_detects_phonemes_and_respellings() -> None:
    phonemes = normalize_entry(" Qwen ", "k w eh1 n", scope="Always")
    assert phonemes is not None
    assert phonemes.word == "Qwen" and phonemes.pronunciation == "K W EH1 N"
    assert phonemes.kind == "phonemes" and phonemes.scope == "always"
    respelling = normalize_entry("RunPod", "run pod")
    assert respelling is not None and respelling.kind == "respelling" and respelling.scope == "unseen"
    assert normalize_entry("", "K") is None
    assert normalize_entry("bad<word", "K") is None
    assert normalize_entry("word", "<K>") is None
    assert is_phoneme_string("K AH1 M . F IY0") and not is_phoneme_string("comfy") and not is_phoneme_string("")


def test_apply_dictionary_respects_boundaries_scope_and_protected_text() -> None:
    entries = [
        normalize_entry("Qwen", "K W EH1 N"),
        normalize_entry("RunPod", "run pod"),
        normalize_entry("LoRA", "L AO1 . R AH0"),
        normalize_entry("Wan", "W AA1 N", scope="always"),
    ]
    text = "Qwen's LoRAs run on RunPod. <minute|M AY0 . N UW1 T> [pause:500ms] wan and Wan; Qwenx stays."
    result = apply_dictionary(text, entries)
    assert result.startswith("<Qwen's|K W EH1 N Z> <LoRAs|L AO1 . R AH0 Z> run on run pod.")
    assert "<minute|M AY0 . N UW1 T> [pause:500ms]" in result
    assert "<wan|W AA1 N> and <Wan|W AA1 N>;" in result
    assert "Qwenx stays." in result
    # Words the voice spoke in training keep their trained reading unless the scope is "always".
    trained = apply_dictionary(text, entries, known_words={"qwen", "runpod", "lora"})
    assert "<Qwen" not in trained and "run pod" not in trained and "<LoRAs" not in trained
    assert "<Wan|W AA1 N>" in trained
    assert apply_dictionary("", entries) == "" and apply_dictionary("text", []) == "text"


def test_plural_phones_follow_the_final_sound() -> None:
    assert apply_dictionary("boxes", [normalize_entry("box", "B AA1 K S")]) == "<boxes|B AA1 K S IH0 Z>"
    assert apply_dictionary("cats", [normalize_entry("cat", "K AE1 T")]) == "<cats|K AE1 T S>"
    assert apply_dictionary("dogs", [normalize_entry("dog", "D AO1 G")]) == "<dogs|D AO1 G Z>"


def test_syllabify_and_letter_rules() -> None:
    assert syllabify(["K", "AH1", "M", "F", "IY0"]) == "K AH1 M . F IY0"
    assert syllabify(["S", "T", "R", "IY1", "T"]) == "S T R IY1 T"
    assert syllabify(["EH1", "K", "S", "T", "R", "AH0"]) == "EH1 K . S T R AH0"
    assert letters_to_sound("qwen") == ["K", "W", "EH1", "N"]
    assert letters_to_sound("kwane") == ["K", "W", "EY1", "N"]
    assert letters_to_sound("") == []


def test_suggestions_cover_camel_case_acronyms_and_numbers() -> None:
    camel = suggest_pronunciation("ComfyUI")
    assert camel is not None and camel.kind == "phonemes"
    assert camel.pronunciation.endswith("Y UW1 . AY1")
    acronym = suggest_pronunciation("GPU")
    assert acronym is not None and acronym.pronunciation == "JH IY1 . P IY1 . Y UW1"
    numbers = suggest_pronunciation("NVFP4")
    assert numbers is not None and numbers.kind == "respelling" and numbers.pronunciation == "N V F P 4"
    version = suggest_pronunciation("Wan2.2")
    assert version is not None and version.pronunciation == "wan 2.2"
    assert suggest_pronunciation("2509") is None and suggest_pronunciation("") is None
    unknown = suggest_pronunciation("qwen")
    assert unknown is not None and unknown.confidence == "low" and unknown.pronunciation == "K W EH1 N"


def test_check_text_lists_only_uncovered_words() -> None:
    entries = [normalize_entry("Qwen", "K W EH1 N")]
    rows = check_text(
        "Qwen and ComfyUI and Nunchaku run cats. [pause:1s] <SwarmUI|S W AO1 R M> NVFP4",
        known_words={"comfyui"},
        entries=entries,
        token_len=lambda text: len(text.split()) + 2,
    )
    words = [row["word"] for row in rows]
    assert "Qwen" not in words and "ComfyUI" not in words and "cats" not in words and "SwarmUI" not in words
    assert words == ["Nunchaku", "NVFP4"]
    assert rows[0]["suggestion"] and rows[0]["fragments"] == 3
    assert rows[1]["kind"] == "respelling"
    assert candidate_words("Hello, world! <x|Y> [pause:2s] world") == ["Hello", "world"]


def test_dictionary_round_trip_and_rows(tmp_path: Path) -> None:
    path = tmp_path / "pron" / "dictionary.json"
    entries = [normalize_entry("Qwen", "K W EH1 N", source="manual"), normalize_entry("RunPod", "run pod", scope="always")]
    save_dictionary(path, entries)
    loaded = load_dictionary(path)
    assert [entry.word for entry in loaded] == ["Qwen", "RunPod"]
    assert loaded[1].scope == "always" and loaded[1].kind == "respelling"
    rows = dictionary_rows(loaded)
    assert rows[0] == ["Qwen", "K W EH1 N", "phonemes", "unseen", "manual"]
    assert entries_from_rows(rows + [["", "", "", "", ""], ["Dup", "D AH1 P"], ["dup", "D UW1 P"]]) == loaded + [
        DictionaryEntry("Dup", "D AH1 P", "phonemes", "manual", "", "unseen")
    ]
    merged = merge_entries(loaded, [normalize_entry("qwen", "CH W EH1 N", source="suggested")])
    assert [entry.pronunciation for entry in merged] == ["CH W EH1 N", "run pod"]
    assert load_dictionary(tmp_path / "missing.json") == []
    assert all(entry.source == "builtin" and entry.scope == "unseen" for entry in builtin_entries())
    assert len(builtin_entries()) == len(pron.BUILTIN_ENTRIES)


@pytest.mark.skipif(not pron.cmu_available(), reason="cmudict is not installed")
def test_cmu_lookup_is_case_insensitive() -> None:
    assert pron.lookup_word("Comfy") == ["K", "AH1", "M", "F", "IY0"]
    assert pron.lookup_word("qwen") is None
