from types import SimpleNamespace

import gradio as gr

from ui.app import build_app


def test_every_table_with_wide_text_has_one_width_per_column():
    # Tables with paths or transcripts push their last columns out of view and break short headers
    # into pieces ("Epoc h", "Confiden ce") unless column widths are set; a width list of the
    # wrong length would misalign every column after the mismatch.
    args = SimpleNamespace(model_dir="models", device="cpu", verbose=False, no_browser=True,
                           port=7861, host="127.0.0.1", share=False)
    demo = build_app(args)
    tables = [block for block in demo.blocks.values() if isinstance(block, gr.Dataframe)]
    labels = {table.label: table for table in tables}
    for label in ("Live section preview", "Words without a known reading",
                  "Pronunciation dictionary (pronunciations/dictionary.json)", "Recent outputs (last 10)",
                  "Audition results", "Prepared segments", "Checkpoints", "Grid cells", "LoRA / DoRA files"):
        assert label in labels, label
    for table in tables:
        widths = table.column_widths
        if not widths:
            continue
        headers = list(table.headers or [])
        assert len(widths) == len(headers), (table.label, len(widths), len(headers))
        percents = [float(str(width).rstrip("%")) for width in widths if str(width).endswith("%")]
        if len(percents) == len(widths):
            assert 99 <= sum(percents) <= 101, (table.label, sum(percents))
    for label in ("Words without a known reading", "Pronunciation dictionary (pronunciations/dictionary.json)",
                  "Recent outputs (last 10)", "Audition results", "Prepared segments", "Grid cells", "LoRA / DoRA files"):
        assert labels[label].column_widths, label
