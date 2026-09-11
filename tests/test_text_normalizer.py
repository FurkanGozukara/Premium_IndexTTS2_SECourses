import pytest

from indextts.utils.front import TextNormalizer


@pytest.fixture(scope="module")
def normalizer():
    instance = TextNormalizer()
    instance.load()
    return instance


def test_wetext_normalizes_english(normalizer):
    normalized = normalizer.normalize("Hello 123 world, it's 5pm")

    assert "123" not in normalized
    assert "one hundred and twenty three" in normalized.lower()
    assert "five" in normalized.lower()


def test_wetext_normalizes_chinese(normalizer):
    normalized = normalizer.normalize("我有123个苹果。")

    assert "123" not in normalized
    assert "一百二十三" in normalized
    assert "苹果" in normalized


@pytest.mark.parametrize(
    "source, expected",
    [("123", "one hundred and twenty three"), ("42", "forty two"), ("3.14", "three point one four")],
)
def test_explicit_english_numeric_text_stays_english(normalizer, source, expected):
    assert normalizer.normalize(source, lang="EN").lower() == expected


def test_explicit_chinese_and_legacy_detection_still_normalize_numbers(normalizer):
    assert normalizer.normalize("123", lang="zh") == "一百二十三"
    assert normalizer.normalize("123") == "一百二十三"


@pytest.mark.parametrize(
    "source",
    [
        "She's been waiting for you.",
        "It's already been completed.",
        "That's been my experience too.",
        "She's happy.",
        "John's car is red.",
        "Where's my coat?",
    ],
)
def test_english_contractions_keep_their_meaning(normalizer, source):
    assert normalizer.normalize(source, lang="en").lower() == source.lower()


def test_explicit_language_keeps_pronunciation_annotations(normalizer):
    result = normalizer.normalize("Read <API|ay pee eye> 42.", lang="en")
    assert "<API|ay pee eye>" in result
    assert "forty two" in result.lower()


@pytest.mark.parametrize("language", ["en", "zh"])
def test_normalizer_preserves_text_when_wetext_rejects_fragment(
    normalizer, monkeypatch, capsys, language
):
    backend = normalizer.en_normalizer if language == "en" else normalizer.zh_normalizer
    monkeypatch.setattr(normalizer, "use_chinese", lambda _text: language == "zh")
    monkeypatch.setattr(backend, "normalize", lambda _text: (_ for _ in ()).throw(AssertionError()))

    assert normalizer.normalize("Fallback text") == "Fallback text"
    output = capsys.readouterr().out
    assert "using the original text" in output
    assert "Traceback" not in output


def test_number_followed_by_plus_is_spoken_as_plus(normalizer, capsys):
    normalized = normalizer.normalize("I am downloading automatically 30+ pre-trained demo voices.", lang="EN")

    assert "thirty plus" in normalized.lower()
    assert "30+" not in normalized
    assert "normalization failed" not in capsys.readouterr().out
    assert "five plus five" in normalizer.normalize("It costs 5+5 dollars.", lang="EN").lower()
