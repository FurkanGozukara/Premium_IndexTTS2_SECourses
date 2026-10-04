"""Live logs show a progress bar once, in its last state, as a console does."""

from ui.common import tail_text


def test_progress_bar_redraws_collapse_to_their_last_state(tmp_path):
    log = tmp_path / "generation.log"
    bar = "".join(f"\r{percent:3d}%|{'#' * (percent // 10)}| {percent}/100" for percent in range(0, 101, 5))
    log.write_bytes((">> start\r\n" + bar + "\n>> Section 1: take 1 0.0% -> kept take 1\n").encode("utf-8"))
    assert tail_text(log, 60).splitlines() == [">> start", "100%|##########| 100/100", ">> Section 1: take 1 0.0% -> kept take 1"]
    assert tail_text(log, 1) == ">> Section 1: take 1 0.0% -> kept take 1"


def test_only_the_end_of_a_large_log_is_read(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text("".join(f"line {index}\n" for index in range(200_000)), encoding="utf-8")
    assert tail_text(log, 3).splitlines() == ["line 199997", "line 199998", "line 199999"]
    assert tail_text(tmp_path / "missing.log") == "" and tail_text(None) == ""
