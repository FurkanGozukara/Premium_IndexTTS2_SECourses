"""Epoch probe: a small deployment-settings benchmark rendered after an epoch, and the two-signal stop.

Validation token loss says how well a checkpoint predicts held-out audio codes; it does not say how the
voice sounds. After an epoch the trainer renders a few held-out sentences with the settings Voice
Generation applies by default, measures them against the person's real recordings (word error, speaker
similarity, pace, pauses), and scores them against Base with the same deployment score the final
selection uses. The probe keeps the best-scoring epoch as ``best/<name>_probe_best.safetensors`` and
lets the trainer stop only when both the token loss and the probe have stalled, or when the probe shows
the voice getting harder to understand while its score no longer improves.

The probe runs in its own process on the epoch sample's terms: the same free-VRAM gate, the memory tier
fitted to what is free beside the training model, or another GPU when the machine has one with room. The
training process itself never loads a generation or measurement model.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .dataset_manifest import atomic_write_json
from .evaluation_plan import audio_path, record_identity, representative_records
from .speech_metrics import SCORE_PAUSE_WEIGHT, SCORE_WER_WEIGHT, deployment_score, paired_difference, summarize

PROBE_FOLDER = "probe"
PROBE_BEST_SUFFIX = "_probe_best"
PROBE_MIN_WER_TOLERANCE = 0.01
# Automatic interval: a probe may cost at most this share of an epoch before probes are spaced out.
PROBE_MAX_TIME_SHARE = 1.0 / 3.0


def probe_root(run_dir: str | Path) -> Path:
    return Path(run_dir) / "analysis" / PROBE_FOLDER


def probe_best_path(adapter_dir: str | Path, name: str) -> Path:
    from .best_checkpoint import BEST_FOLDER
    return Path(adapter_dir) / BEST_FOLDER / f"{name}{PROBE_BEST_SUFFIX}.safetensors"


def automatic_probe_count(sources: int, available: int) -> int:
    """Three sentences per held-out recording, at least six and at most twelve, never more than exist.

    With two seeds that is 12 to 24 clips per check, enough that one misread word moves the word error by
    well under a point and the mean speaker similarity settles to about a hundredth.
    """

    return max(0, min(int(available), max(6, min(12, 3 * max(1, int(sources))))))


def probe_wer_tolerance(configured: float, real_error_rate: float | None) -> float:
    """The word-error increase that counts as degradation: the configured value, or the measurement's own noise."""

    if configured and configured > 0:
        return float(configured)
    return max(PROBE_MIN_WER_TOLERANCE, float(real_error_rate or 0.0))


def probe_interval(configured: int, probe_elapsed_s: float | None, epoch_elapsed_s: float | None) -> int:
    """Epochs between probes: the configured value, or the smallest spacing that keeps probes under a third of the time."""

    if configured and int(configured) > 0:
        return int(configured)
    if not probe_elapsed_s or not epoch_elapsed_s or epoch_elapsed_s <= 0:
        return 1
    allowed = float(epoch_elapsed_s) * PROBE_MAX_TIME_SHARE
    if float(probe_elapsed_s) <= allowed:
        return 1
    return max(1, int(math.ceil(float(probe_elapsed_s) / allowed)))


def build_probe_plan(config: Any, val_records: Sequence[Mapping[str, Any]], run_dir: str | Path,
                     reference_path: str | Path) -> dict[str, Any] | None:
    """Freeze the probe sentences (balanced across held-out recordings), the reference and the seeds."""

    root = probe_root(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "plan.json"
    reference = Path(reference_path)
    if not reference.is_file():
        return None
    identity = record_identity(list(val_records))
    if destination.is_file():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        if previous.get("dataset_identity") != identity:
            raise ValueError("The saved epoch probe plan belongs to a different dataset; use a new run name")
        return previous
    usable = [row for row in val_records if row.get("audio") and audio_path(config.dataset_dir, row).is_file()]
    if not usable:
        return None
    sources = {str(row.get("source_media") or row["id"]) for row in usable}
    count = int(getattr(config, "probe_prompts", 0) or 0) or automatic_probe_count(len(sources), len(usable))
    selected = representative_records(usable, count, int(config.seed) + 1)
    languages = [str(row.get("language") or "EN").upper() for row in selected]
    language = max(set(languages), key=languages.count) if languages else "EN"
    prompts = []
    for row in selected:
        path = audio_path(config.dataset_dir, row)
        prompts.append({"id": str(row["id"]), "text": str(row["text"]), "audio": str(path),
                        "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "source": str(row.get("source_media") or row["id"]),
                        "duration_s": float(row.get("duration_s", 0) or 0)})
    seeds = [(int(config.seed) + 7919 + 104729 * i) % 2**32 for i in range(max(1, int(getattr(config, "probe_seeds", 1) or 1)))]
    plan = {"version": 1, "dataset_identity": identity, "language": language,
            "speaker": str(selected[0].get("speaker", "")) if selected else "",
            "reference": str(reference.resolve()), "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
            "seeds": seeds, "prompts": prompts, "sources": sorted(sources),
            "prompt_count_mode": "configured" if int(getattr(config, "probe_prompts", 0) or 0) else "automatic",
            "policy": {"score_wer_weight": float(getattr(config, "speech_eval_score_wer_weight", SCORE_WER_WEIGHT)),
                       "score_pause_weight": SCORE_PAUSE_WEIGHT},
            "scope": "epoch probe; deployment settings, measured against the real held-out recordings"}
    atomic_write_json(destination, plan)
    return plan


def load_probe_plan(run_dir: str | Path) -> dict[str, Any] | None:
    try:
        return json.loads((probe_root(run_dir) / "plan.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def score_probe_rows(rows: list[dict[str, Any]], base_rows: list[dict[str, Any]], policy: Mapping[str, Any]) -> dict[str, Any]:
    """Summaries of an epoch's probe clips and their paired deployment score against Base."""

    summary = summarize(rows)
    base = summarize(base_rows)
    speaker_metric = "speaker_similarity_real" if all(
        row.get("speaker_similarity_real") is not None for row in [*rows, *base_rows]) else "speaker_similarity"
    speaker = paired_difference(rows, base_rows, speaker_metric)
    error = paired_difference(rows, base_rows, "error_rate")
    score = deployment_score(summary, base, speaker["mean"], error["mean"],
                             wer_weight=float(policy.get("score_wer_weight", SCORE_WER_WEIGHT)),
                             pause_weight=float(policy.get("score_pause_weight", SCORE_PAUSE_WEIGHT)))
    return {"summary": summary, "base": base, "speaker_metric": speaker_metric, "speaker_gain": speaker,
            "error_delta": error, "deployment_score": score}


@dataclass
class ProbeTracker:
    """Resumable record of the probe's best check, stall and degradation counts, and its pacing."""

    best_score: float | None = None
    best_epoch: int = 0
    best_wer: float | None = None
    best_wer_epoch: int = 0
    stalled_epochs: int = 0
    degraded_epochs: int = 0
    epochs: int = 0
    last_epoch: int = 0
    last_score: float | None = None
    last_wer: float | None = None
    tolerance: float = PROBE_MIN_WER_TOLERANCE
    reason: str = ""
    interval: int = 1
    last_probe_epoch: int = 0
    last_elapsed_s: float | None = None
    epoch_elapsed_s: float | None = None
    skipped: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_state(cls, state: Mapping[str, Any] | None) -> "ProbeTracker":
        values = dict(state or {})
        return cls(**{key: value for key, value in values.items() if key in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def stalled(self, patience: int) -> bool:
        return patience > 0 and self.epochs > 0 and self.stalled_epochs >= patience

    def due(self, epoch: int) -> bool:
        """Probe the first epoch, then every ``interval`` epochs after the last probe attempt."""

        if self.last_probe_epoch <= 0:
            return True
        return epoch - self.last_probe_epoch >= max(1, int(self.interval))

    def plan_next(self, *, epoch: int, configured_interval: int, probe_elapsed_s: float | None,
                  epoch_elapsed_s: float | None) -> int:
        self.last_probe_epoch = int(epoch)
        if probe_elapsed_s is not None:
            self.last_elapsed_s = float(probe_elapsed_s)
        if epoch_elapsed_s is not None:
            self.epoch_elapsed_s = float(epoch_elapsed_s)
        self.interval = probe_interval(configured_interval, self.last_elapsed_s, self.epoch_elapsed_s)
        return self.interval

    def observe(self, *, epoch: int, score: float, wer: float, tolerance: float, min_delta: float,
                patience: int) -> tuple[bool, bool]:
        """Record a probe check. Returns (new best deployment score, stop because intelligibility degraded)."""

        if epoch == self.last_epoch and self.epochs:
            return False, bool(self.reason)
        self.last_epoch = epoch
        self.epochs += 1
        self.last_score, self.last_wer, self.tolerance = float(score), float(wer), float(tolerance)
        improved = self.best_score is None or score > self.best_score + min_delta
        if improved:
            self.best_score, self.best_epoch, self.stalled_epochs = float(score), int(epoch), 0
        else:
            self.stalled_epochs += 1
        if self.best_wer is None or wer < self.best_wer:
            self.best_wer, self.best_wer_epoch = float(wer), int(epoch)
        degraded = self.best_wer is not None and wer > self.best_wer + tolerance
        self.degraded_epochs = self.degraded_epochs + 1 if degraded else 0
        self.history.append({"epoch": int(epoch), "score": float(score), "wer": float(wer), "improved": improved,
                             "degraded": degraded})
        self.reason = ""
        # Degrading intelligibility is overfitting only when the deployment score has also stopped improving;
        # a checkpoint that reads slightly worse but sounds much more like the person is still progress.
        if patience > 0 and self.degraded_epochs >= patience and self.stalled_epochs >= patience:
            self.reason = (f"epoch probe word error {100 * wer:.2f}% stayed more than {100 * tolerance:.2f} points above the "
                           f"best check's ({100 * (self.best_wer or 0):.2f}% at epoch {self.best_wer_epoch}) for "
                           f"{self.degraded_epochs} checks while the deployment score stopped improving; best score "
                           f"{self.best_score:+.4f} at epoch {self.best_epoch}")
        return improved, bool(self.reason)


def decide_stop(loss_stop: bool, tracker: ProbeTracker | None, *, enabled: bool, patience: int) -> tuple[bool, str]:
    """Combine the token-loss verdict with the probe: stop when both stall, or when the probe shows overfitting.

    Without probe results (disabled, not yet run, or failed) the loss verdict stands. Returns the decision and
    a note for the log; the note is empty when nothing changed the loss verdict.
    """

    if not enabled or tracker is None or tracker.epochs == 0:
        return bool(loss_stop), ""
    stalled = tracker.stalled(patience)
    best = f"best {tracker.best_score:+.4f} at epoch {tracker.best_epoch}" if tracker.best_score is not None else "no score yet"
    if tracker.reason and stalled:
        return True, "stopping: " + tracker.reason
    if loss_stop and stalled:
        return True, (f"validation loss and the epoch probe both stalled ({tracker.stalled_epochs} probe checks without a "
                      f"deployment-score gain above the minimum; {best})")
    if loss_stop:
        return False, (f"validation loss stalled, but the epoch probe still improves ({best}; "
                       f"{tracker.stalled_epochs}/{patience} checks without gain); training continues")
    return False, ""


def load_probe_summary(run_dir: str | Path) -> dict[str, Any] | None:
    try:
        return json.loads((probe_root(run_dir) / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def write_probe_summary(run_dir: str | Path, tracker: ProbeTracker, *, best_checkpoint: str, name: str) -> Path:
    path = probe_root(run_dir) / "summary.json"
    payload = {**tracker.to_dict(), "best_checkpoint": best_checkpoint, "name": name,
               "metric": ("deployment score against Base on the probe sentences (speaker gain minus weighted word-error "
                          "increase plus pause term)")}
    atomic_write_json(path, payload)
    return path


def probe_markdown(tracker: ProbeTracker | Mapping[str, Any]) -> str:
    state = tracker if isinstance(tracker, ProbeTracker) else ProbeTracker.from_state(tracker)
    if not state.history:
        return ""
    lines = [f"**Epoch probe:** best deployment score {state.best_score:+.4f} at epoch {state.best_epoch} "
             f"({state.epochs} checks, probe every {max(1, int(state.interval))} epoch(s)).", "",
             "| Epoch | Probe word error | Deployment score vs Base | Note |", "|---:|---:|---:|---|"]
    for item in state.history:
        note = "best score" if item["epoch"] == state.best_epoch else ("word error degraded" if item.get("degraded") else "")
        lines.append(f"| {item['epoch']} | {100 * item['wer']:.2f}% | {item['score']:+.4f} | {note} |")
    return "\n".join(lines)


def group_by_source(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("source") or "")].append(row)
    return dict(grouped)


__all__ = ["PROBE_BEST_SUFFIX", "PROBE_FOLDER", "PROBE_MAX_TIME_SHARE", "ProbeTracker", "automatic_probe_count",
           "build_probe_plan", "decide_stop", "load_probe_plan", "load_probe_summary", "probe_best_path", "probe_interval",
           "probe_markdown", "probe_root", "probe_wer_tolerance", "score_probe_rows", "write_probe_summary"]
