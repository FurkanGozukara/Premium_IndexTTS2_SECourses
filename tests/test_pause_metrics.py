"""Internal pause measurement used by the speech comparison reports."""
import numpy as np

from indextts.training.speech_metrics import internal_pause_metrics, summarize


def _tone(seconds: float, sr: int = 16000) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _silence(seconds: float, sr: int = 16000) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


def test_internal_pauses_ignore_edges_and_count_only_sustained_silence():
    audio = np.concatenate([_silence(0.5), _tone(1.0), _silence(0.3), _tone(1.0), _silence(0.05), _tone(0.5), _silence(0.8)])
    metrics = internal_pause_metrics(audio, 16000)
    assert metrics["pause_count"] == 1  # the 50 ms gap is a stop closure, not a pause
    assert 250 <= metrics["longest_pause_ms"] <= 350
    assert abs(metrics["speech_span_s"] - 2.85) < 0.05  # leading and trailing silence are not part of the span
    assert 0.08 < metrics["pause_time_fraction"] < 0.13
    assert metrics["pause_s"] == metrics["longest_pause_ms"] / 1000


def test_quiet_or_empty_audio_is_not_one_long_pause():
    silent = internal_pause_metrics(np.zeros(16000, dtype=np.float32), 16000)
    assert silent["pause_count"] == 0 and silent["pause_time_fraction"] is None
    quiet = internal_pause_metrics(0.001 * np.concatenate([_tone(1.0), _silence(0.3), _tone(1.0)]), 16000)
    assert quiet["pause_count"] == 1  # the relative threshold still finds the gap in a quiet recording
    assert internal_pause_metrics(np.zeros(0, dtype=np.float32), 16000)["pause_count"] == 0


def test_summary_averages_pause_measurements_when_present():
    def row(fraction, ratio):
        return {"error_rate": 0.0, "errors": 0, "units": 10, "speaker_similarity": 0.9, "invalid_audio": False,
                "possible_truncation": False, "possible_repetition": False, "start_matches": True, "end_matches": True,
                "pause_time_fraction": fraction, "pause_ratio_vs_real": ratio}
    summary = summarize([row(0.1, 1.5), row(0.2, 0.5)])
    assert abs(summary["pause_time_fraction"] - 0.15) < 1e-9
    assert abs(summary["pause_ratio_vs_real"] - 1.0) < 1e-9  # mean of per-clip ratios when pause seconds are absent
    assert summarize([{**row(0.1, 1.5), "pause_time_fraction": None, "pause_ratio_vs_real": None}])["pause_ratio_vs_real"] is None
    # With pause seconds, the summary is the ratio of totals: 3.0 s generated over 2.0 s real, not the mean of 1.5 and 0.5.
    corpus = summarize([{**row(0.1, 1.5), "pause_s": 1.5, "real_pause_s": 1.0}, {**row(0.2, 0.5), "pause_s": 1.5, "real_pause_s": 1.0},
                        {**row(0.1, None), "pause_s": 0.4, "real_pause_s": 0.0}])
    assert abs(corpus["pause_ratio_vs_real"] - 1.5) < 1e-9
