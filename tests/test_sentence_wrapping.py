"""Line wrapping and subtitle cue boundaries must not change speech segmentation."""

from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from indextts.utils.text_segmentation import ends_sentence, split_sentences, split_text_by_tokens


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    "A search can match an idea without matching its words.\n\n"
    "Embeddings turn text into numerical vectors. A trained model can represent related items "
    "close together under a chosen similarity measure, giving search another useful clue.\n\n"
    "Searching for ways to keep a room cool might retrieve a passage about reducing indoor heat, "
    "despite different wording.\n\n"
    "Our floating cards illustrate relationships in a space with many dimensions, not literal geography.\n\n"
    "Similarity depends on the model and task. A nearby result can still be irrelevant or factually "
    "wrong, so check what it says.\n\n"
    "Use similarity to find candidates, then judge their relevance and evidence."
)
WRAPPED_SCRIPT = SCRIPT
for before, after in (
    ("its words", "its\nwords"),
    ("items close", "items\nclose"),
    ("retrieve a passage", "retrieve a\npassage"),
    ("in a space", "in a\nspace"),
    ("can still", "can\nstill"),
    ("judge their", "judge\ntheir"),
):
    WRAPPED_SCRIPT = WRAPPED_SCRIPT.replace(before, after)


def spoken(text):
    return " ".join(text.split())


@pytest.fixture(scope="module")
def tokenizer():
    from indextts.utils.tokenizer import get_tokenizer

    return get_tokenizer(multilingual=True, model_dir=str(ROOT / "models"))


@pytest.mark.parametrize("separator", ["\n", "\r\n", "\r", "\n\n", "\n  \n", "\u2028"])
def test_wrapping_is_not_a_sentence_end(separator):
    text = f"Dr.{separator}Smith agreed.{separator}Then we{separator}left."
    assert split_sentences(text) == [f"Dr.{separator}Smith agreed.{separator}", f"Then we{separator}left."]
    assert not ends_sentence(f"Then we{separator}")
    assert ends_sentence(f"Smith agreed.{separator}")


@pytest.mark.parametrize("mode", ["smart", "sentence"])
@pytest.mark.parametrize("max_tokens,target", [(20, None), (60, None), (66, 40), (120, 30), (220, 92)])
def test_exact_user_example_and_rewrapped_captions_have_identical_segments(tokenizer, mode, max_tokens, target):
    from indextts.utils.text_segmentation import segment_token_budget

    token_len = lambda value: len(tokenizer.encode(value, allowed_special="all"))
    options = dict(capacity=602, token_len=token_len, lang_prefix="<|en|> ", mode=mode, target_tokens=target)
    baseline = split_text_by_tokens(spoken(SCRIPT), max_tokens, **options)
    words = SCRIPT.split()
    cue_wrapped = "\n\n".join(" ".join(words[index:index + 7]) for index in range(0, len(words), 7))
    for text in (SCRIPT, WRAPPED_SCRIPT, WRAPPED_SCRIPT.replace("\n", "\r\n"), cue_wrapped):
        segments = split_text_by_tokens(text, max_tokens, **options)
        assert "".join(segments) == text
        assert [spoken(part) for part in segments] == [spoken(part) for part in baseline]
        budget = segment_token_budget(max_tokens, 602, "<|en|> ", token_len)
        assert all(token_len(re.sub(r"\s+", " ", part)) <= budget for part in segments)
    assert len(split_sentences(WRAPPED_SCRIPT)) == 8


@pytest.mark.parametrize("extension", [".srt", ".vtt"])
def test_subtitle_upload_and_batch_text_keep_whole_sentences(tmp_path, extension, tokenizer):
    from indextts.utils.subtitle_utils import build_subtitle_render_units, parse_subtitle_file, subtitle_cues_to_text
    from ui.batch_tab import _load_batch_item
    from ui.generation_tab import preview_segments

    path = tmp_path / ("wrapped" + extension)
    text = ("WEBVTT\n\n" if extension == ".vtt" else "") + (
        "1\n00:00:01.000 --> 00:00:03.000\nA search can match an\nidea without\n\n"
        "2\n00:00:03.000 --> 00:00:05.000\nmatching its words.\n\n"
        "3\n00:00:06.000 --> 00:00:08.000\nEmbeddings turn text into\nnumerical vectors.\n"
    )
    path.write_text(text, encoding="utf-8")
    cues = parse_subtitle_file(str(path))
    imported = subtitle_cues_to_text(cues)
    batch = _load_batch_item(dict(name="wrapped", path=str(path), subtitle=str(path)))
    assert batch["text"] == imported
    options = dict(segmentation_mode="sentence", model_dir=str(ROOT / "models"))
    rows, _ = preview_segments(imported, "EN", 120, **options)
    baseline, _ = preview_segments(spoken(imported), "EN", 120, **options)
    assert [spoken(row[2]) for row in rows] == [row[2].strip() for row in baseline]
    assert len(rows) == 2
    assert [row[3] for row in rows] == [row[3] for row in baseline]
    timed, note = preview_segments(imported, "EN", 120, True, str(path), **options)
    assert len(timed) == 3 and "3 timing unit(s)" in note
    assert [(unit.start_ms, unit.end_ms) for unit in build_subtitle_render_units(cues)] == [
        (1000, 3000), (3000, 5000), (6000, 8000),
    ]


@pytest.mark.parametrize("mode", ["smart", "sentence"])
@pytest.mark.parametrize("normalize", [False, True])
def test_inference_plan_ignores_wraps_and_preserves_explicit_pauses(tokenizer, mode, normalize):
    from indextts.infer_v2_5 import IndexTTS2
    from indextts.utils.front import TextNormalizer

    engine = IndexTTS2.__new__(IndexTTS2)
    engine.tokenizer = tokenizer
    engine.text_process = TextNormalizer()
    engine.gpt = SimpleNamespace(text_pos_embedding=SimpleNamespace(emb=SimpleNamespace(num_embeddings=602)))
    options = dict(lang="EN", max_tokens=120, text_normalization=normalize, interval_silence=100,
                   enable_pause_tags=True, segment_budget_scale_non_cjk=0.72, segmentation_mode=mode,
                   segment_target_tokens=30, sentence_pause_ms=350)
    plain = spoken(SCRIPT) + " [pause:450ms] Done."
    wrapped = WRAPPED_SCRIPT.replace("\n", "\r\n") + "\n[pause:450ms]\nDone."
    expected = engine._build_text_plan(plain, **options)
    actual = engine._build_text_plan(wrapped, **options)
    assert actual == expected
    assert ("pause", 9922) in actual[1]


def test_upload_event_uses_all_current_preview_settings(tmp_path, monkeypatch):
    from ui.app import build_app
    from ui import generation_tab

    monkeypatch.setattr(generation_tab, "smart_segment_target", lambda _path: 30)
    monkeypatch.setattr(generation_tab, "preview_words_per_second", lambda _path, _rate: 2.5)
    monkeypatch.setattr(generation_tab, "apply_pronunciation_dictionary", lambda text, _path: text.replace("TTS", "speech"))
    demo = build_app(SimpleNamespace(device="cpu", model_dir=str(ROOT / "models")))
    upload = next(fn for fn in demo.fns.values() if fn.name == "load_caption")
    preview = next(fn for fn in demo.fns.values() if fn.name == "update_preview")
    assert upload.inputs == preview.inputs
    path = tmp_path / "wrapped.srt"
    path.write_text("1\n00:00:01,000 --> 00:00:03,000\nTTS reads this\nwhole sentence.\n\n"
                    "2\n00:00:04,000 --> 00:00:06,000\nThen read another one.\n", encoding="utf-8")
    for mode, label in (("smart", "Smart sentences, aiming at 30"), ("sentence", "Every sentence"), ("budget", "Token budget")):
        inputs = ["old script", "EN", 120, False, str(path), True, 0.72, True, "voice", mode, 1.0]
        loaded, status, rows, note = upload.fn(*inputs)
        assert "Loaded 2 SRT cue(s)" in status and label in note
        assert "speech reads" in rows[0][2] and "about" in rows[0][3]
        assert (rows, note) == preview.fn(loaded, *inputs[1:])
        inputs[0], inputs[4] = loaded, None
        cleared = upload.fn(*inputs)
        assert cleared[2:] == (rows, note)
