from __future__ import annotations

from indextts.utils.text_segmentation import (
    DEFAULT_SEGMENTATION_MODE,
    SEGMENTATION_MODES,
    ends_sentence,
    normalize_segmentation_mode,
    pack_sentences,
    segment_token_budget,
    sentence_pieces,
    split_sentences,
    split_text_by_tokens,
)


def words(value: str) -> int:
    """A word-count tokenizer keeps the expected numbers easy to read."""

    return len(value.split())


PREFIX = "<|en|> "
PARAGRAPH = (
    "The model takes two inputs from audio. "
    "The speaker prompt fixes who is talking, and the emotion prompt shapes how the line is delivered. "
    "The app used to feed the same clip to both, so every sentence came out flat. "
    "Now the expressive clip drives the emotion pathway. "
    "This is optional, e.g. you can switch it off. "
    "Dr. Smith agreed. "
    "Try it and listen."
)


def test_modes_are_normalized_with_aliases() -> None:
    assert SEGMENTATION_MODES == ("budget", "sentence", "smart") and DEFAULT_SEGMENTATION_MODE == "smart"
    assert normalize_segmentation_mode("Smart sentences") == "smart"
    assert normalize_segmentation_mode("every sentence") == "sentence"
    assert normalize_segmentation_mode("token budget") == "budget"
    assert normalize_segmentation_mode("nonsense") == DEFAULT_SEGMENTATION_MODE
    assert normalize_segmentation_mode(None) == DEFAULT_SEGMENTATION_MODE


def test_split_sentences_preserves_text_and_skips_abbreviations_annotations_and_cjk() -> None:
    pieces = split_sentences(PARAGRAPH)
    assert "".join(pieces) == PARAGRAPH
    assert len(pieces) == 7
    assert pieces[4] == "This is optional, e.g. you can switch it off. "  # e.g. is not a sentence end
    assert pieces[5] == "Dr. Smith agreed. "  # neither is a title
    assert pieces[-1] == "Try it and listen."
    annotated = "Say <|SPECIAL_TOKEN_1|>K W EH1 N . AH0<|SPECIAL_TOKEN_1|> twice. Then stop."
    assert split_sentences(annotated) == [
        "Say <|SPECIAL_TOKEN_1|>K W EH1 N . AH0<|SPECIAL_TOKEN_1|> twice. ", "Then stop."
    ]
    assert split_sentences("第一句。第二句！第三句？") == ["第一句。", "第二句！", "第三句？"]
    assert split_sentences("line one\nline two") == ["line one\nline two"]
    assert split_sentences('He said "Go." Then left.') == ['He said "Go." ', "Then left."]
    assert split_sentences("Version 2.2 is out. Yes.") == ["Version 2.2 is out. ", "Yes."]
    assert ends_sentence("Done. ") and ends_sentence('Done."') and not ends_sentence("Done, ")


def test_sentence_mode_gives_one_sentence_per_segment_and_cuts_only_oversized_sentences() -> None:
    budget = segment_token_budget(17, 602, PREFIX, words, 1.0)
    assert budget == 16
    segments = split_text_by_tokens(PARAGRAPH, 17, capacity=602, token_len=words, lang_prefix=PREFIX,
                                    segment_budget_scale_non_cjk=1.0, mode="sentence")
    assert "".join(segments) == PARAGRAPH
    sentences = split_sentences(PARAGRAPH)
    # Every sentence within the budget is its own segment; the 18-word sentence is cut at its comma.
    assert segments[0] == sentences[0]
    assert segments[1] == "The speaker prompt fixes who is talking,"
    assert segments[2] == " and the emotion prompt shapes how the line is delivered. "
    assert segments[3:] == sentences[2:]
    assert all(words(segment) <= budget for segment in segments)


def test_smart_mode_packs_whole_sentences_near_the_target_without_orphans() -> None:
    segments = split_text_by_tokens(PARAGRAPH, 31, capacity=602, token_len=words, lang_prefix=PREFIX,
                                    segment_budget_scale_non_cjk=1.0, mode="smart")
    budget = segment_token_budget(31, 602, PREFIX, words, 1.0)
    assert "".join(segments) == PARAGRAPH
    assert all(words(segment) <= budget for segment in segments)
    # No segment ends inside a sentence and the tail is not an orphan.
    assert all(ends_sentence(segment) for segment in segments)
    counts = [words(segment) for segment in segments]
    assert min(counts) >= 0.35 * (budget * 0.85)
    # The greedy budget splitter leaves an orphan on the same text.
    greedy = split_text_by_tokens(PARAGRAPH, 31, capacity=602, token_len=words, lang_prefix=PREFIX,
                                  segment_budget_scale_non_cjk=1.0, mode="budget")
    assert "".join(greedy) == PARAGRAPH
    assert words(greedy[-1]) < min(counts)


def test_smart_mode_aims_at_the_dataset_target_and_respects_the_budget() -> None:
    text = " ".join(f"Sentence number {index} has exactly six words." for index in range(1, 13))
    segments = split_text_by_tokens(text, 61, capacity=602, token_len=words, lang_prefix=PREFIX,
                                    segment_budget_scale_non_cjk=1.0, mode="smart", target_tokens=18)
    assert "".join(segments) == text
    # 12 sentences of 7 words each with an 18-word target: three sentences (21 words) or two (14) per line.
    assert all(words(segment) in {14, 21} for segment in segments)
    assert len(segments) in {4, 5, 6}
    single = split_text_by_tokens("Short text.", 61, capacity=602, token_len=words, lang_prefix=PREFIX,
                                  segment_budget_scale_non_cjk=1.0, mode="smart", target_tokens=18)
    assert single == ["Short text."]


def test_budget_mode_is_unchanged() -> None:
    text = "alpha, beta. gamma delta, epsilon."
    assert split_text_by_tokens(text, 4, capacity=100, token_len=words, mode="budget") == split_text_by_tokens(
        text, 4, capacity=100, token_len=words
    )
    assert split_text_by_tokens("alpha delta", 5, capacity=100, token_len=len) == ["alpha", " ", "delta"]


def test_pieces_and_packer_handle_oversized_words_and_annotations() -> None:
    budget = 6
    text = "Supercalifragilistic is long. <|SPECIAL_TOKEN_1|>ABCDEFGHIJ<|SPECIAL_TOKEN_1|> stays."
    pieces = sentence_pieces(text, budget, len)
    assert "".join(piece for piece, _ in pieces) == text
    assert any("<|SPECIAL_TOKEN_1|>ABCDEFGHIJ<|SPECIAL_TOKEN_1|>" in piece for piece, _ in pieces)
    packed = pack_sentences(pieces, budget, len, target=5)
    assert "".join(packed) == text
    assert pack_sentences([], budget, len) == []
