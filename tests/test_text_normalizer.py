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
