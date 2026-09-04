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
