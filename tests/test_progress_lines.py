"""Worker logs keep a progress bar's final state only, as a terminal shows it."""

from indextts.utils.progress_lines import collapse_progress_lines, is_progress_line


def test_a_finished_bar_leaves_its_100_percent_line():
    lines = [">> Reference extraction (emotion): 0.055s", "Use the specified emotion vector", "",
             "  0%|          | 0/25 [00:00<?, ?it/s]", "  4%|4         | 1/25 [00:00<00:03,  7.71it/s]",
             " 96%|#########6| 24/25 [00:00<00:00, 30.97it/s]", "100%|##########| 25/25 [00:00<00:00, 31.34it/s]",
             ">> === Generation summary ==="]
    assert list(collapse_progress_lines(lines)) == [
        ">> Reference extraction (emotion): 0.055s", "Use the specified emotion vector", "",
        "100%|##########| 25/25 [00:00<00:00, 31.34it/s]", ">> === Generation summary ==="]


def test_described_bars_and_interrupted_bars():
    lines = ["Loading weights:  24%|███  | 75/310 [00:00<00:00, 708.20it/s]",
             "Loading weights: 100%|█████| 310/310 [00:00<00:00, 1076.67it/s]",
             "Fetching 4 files:  75%|███  | 3/4", ">> download interrupted", " 50%|#####     | 1/2"]
    assert list(collapse_progress_lines(lines)) == [
        "Loading weights: 100%|█████| 310/310 [00:00<00:00, 1076.67it/s]",
        "Fetching 4 files:  75%|███  | 3/4", ">> download interrupted", " 50%|#####     | 1/2"]
    assert not is_progress_line(">> step 140/420 | ep 1/3 | loss 5.757 | 5.49 it/s")
    assert not is_progress_line("[ 50.0%] 1/2 segments | elapsed 7s")
