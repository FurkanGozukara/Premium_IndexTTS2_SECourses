from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any

from .dataset_manifest import atomic_write_json
from .dataset_prep import DatasetPrepConfig, run_dataset_prep


class WorkerReporter:
    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.status_path = state_dir / "status.json"
        self.log_path = state_dir / "log.txt"
        self.started = time.monotonic()
        self.phase = "starting"
        self._fraction = 0.0
        self._last = {
            "file_i": 0,
            "file_n": 0,
            "segment_count": 0,
            "total_audio_seconds": 0.0,
            "message": "Starting dataset preparation",
            "progress_completed": 0.0,
            "chunk_i": 0,
            "chunk_n": 0,
        }
        self._write_status()

    @staticmethod
    def _vram() -> tuple[float, float]:
        try:
            import torch

            if not torch.cuda.is_available():
                return 0.0, 0.0
            index = torch.cuda.current_device()
            return (
                torch.cuda.memory_allocated(index) / 1024**3,
                torch.cuda.get_device_properties(index).total_memory / 1024**3,
            )
        except Exception:
            return 0.0, 0.0

    def _write_status(self, **updates: Any) -> None:
        self._last.update(updates)
        elapsed = time.monotonic() - self.started
        file_i = int(self._last.get("file_i", 0) or 0)
        file_n = int(self._last.get("file_n", 0) or 0)
        chunk_i = int(self._last.get("chunk_i", 0) or 0)
        chunk_n = int(self._last.get("chunk_n", 0) or 0)
        if self.phase == "complete":
            fraction = 1.0
        elif chunk_n > 0 and file_n > 0:
            fraction = (max(0, file_i - 1) + min(1.0, chunk_i / chunk_n)) / file_n
        elif file_n > 0:
            fraction = float(self._last.get("progress_completed", 0.0) or 0.0) / file_n
        else:
            fraction = 0.0
        fraction = max(self._fraction, max(0.0, min(1.0, fraction)))
        self._fraction = fraction
        eta = elapsed * (1.0 - fraction) / fraction if 0.0 < fraction < 1.0 else 0.0
        segment_count = int(self._last.get("segment_count", 0) or 0)
        if segment_count > 0 and self.phase in {"segments", "complete", "cancelled"}:
            speed = segment_count / elapsed if elapsed > 0 else 0.0
            speed_unit = "segments/s"
        else:
            speed = fraction * file_n * 60.0 / elapsed if elapsed > 0 and file_n else 0.0
            speed_unit = "files/min"
        vram_used, vram_total = self._vram()
        payload = {
            "phase": self.phase,
            "file_i": file_i,
            "file_n": file_n,
            "segment_count": segment_count,
            "total_audio_seconds": float(self._last.get("total_audio_seconds", 0.0) or 0.0),
            "message": str(self._last.get("message", "")),
            "eta": round(eta, 3),
            "elapsed": round(elapsed, 3),
            "fraction": round(fraction, 6),
            "elapsed_s": round(elapsed, 3),
            "eta_s": round(eta, 3),
            "speed": round(speed, 6),
            "speed_unit": speed_unit,
            "vram_used_gb": round(vram_used, 4),
            "vram_total_gb": round(vram_total, 4),
            "updated_at": time.time(),
            "updated_at_iso": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(self.status_path, payload)

    def set_stage(self, name: str) -> None:
        self.phase = str(name)
        self._write_status(message=str(name).replace("_", " ").capitalize())

    def update(
        self,
        completed: int | float,
        total: int | float | None = None,
        desc: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        values = dict(extra or {})
        if "phase" in values:
            self.phase = str(values.pop("phase"))
        chunk_match = re.search(r"\bchunk\s+(\d+)\s*/\s*(\d+)\b", desc, re.IGNORECASE)
        if chunk_match:
            values["chunk_i"] = int(chunk_match.group(1))
            values["chunk_n"] = int(chunk_match.group(2))
        else:
            values["chunk_i"] = 0
            values["chunk_n"] = 0
            values["progress_completed"] = float(completed)
            values.setdefault("file_i", int(completed))
            if total is not None:
                values.setdefault("file_n", int(total))
        values["message"] = desc
        self._write_status(**values)
        print(f"[{completed}/{total or 0}] {desc}", flush=True)

    def log(self, msg: str) -> None:
        line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
        with self.log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
        print(msg, flush=True)

    def finish(self) -> None:
        self._write_status()

    def mark_finished(self, phase: str, message: str, summary: Any | None = None) -> None:
        self.phase = phase
        updates: dict[str, Any] = {"message": message}
        if summary is not None:
            updates.update(
                segment_count=summary.segment_count,
                total_audio_seconds=summary.total_duration_s,
            )
        self._write_status(**updates)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IndexTTS dataset preparation subprocess worker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "DatasetPrepConfig segmentation_mode values:\n"
            "  auto              sentence_aligned with CUDA, otherwise cue_boundaries\n"
            "  sentence_aligned  caption sentences with cached Whisper word timing\n"
            "  cue_boundaries    legacy sidecar cue timing\n"
            "  whisper_only      Whisper timing and text without sidecars\n\n"
            "align_with_whisper=true remains an alias for sentence_aligned. Configurable quality "
            "fields include min_file_alignment_coverage, min_segment_alignment_coverage, "
            "min_words_per_second, max_words_per_second, min_peak_dbfs, max_clipping_ratio, "
            "and optional max_silence_ratio. Alignment coverage and filter_drop_counts are "
            "written to dataset_info.json."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a DatasetPrepConfig JSON file (including segmentation and quality fields)",
    )
    parser.add_argument("--state-dir", required=True, help="Directory for status, logs, and stop.flag")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    state_dir = Path(args.state_dir).expanduser().resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    reporter = WorkerReporter(state_dir)
    try:
        with Path(args.config).expanduser().open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        config = DatasetPrepConfig.from_dict(payload)
        summary = run_dataset_prep(
            config,
            reporter=reporter,
            cancel_check=lambda: (state_dir / "stop.flag").exists(),
        )
        reporter.mark_finished(
            summary.status,
            f"{summary.status}: {summary.segment_count} segments, "
            f"{summary.total_duration_s / 60.0:.2f} minutes",
            summary,
        )
        elapsed = time.monotonic() - reporter.started
        rate = summary.segment_count / elapsed if elapsed > 0 else 0.0
        print(
            f">> Dataset preparation summary | status={summary.status} | "
            f"items={summary.segment_count} | audio={summary.total_duration_s:.3f}s | "
            f"elapsed={elapsed:.3f}s | {rate:.3f} segments/s | output={summary.output_dir}",
            flush=True,
        )
        return 0
    except Exception:
        detail = traceback.format_exc()
        (state_dir / "error.txt").write_text(detail, encoding="utf-8")
        reporter.log(detail.rstrip())
        reporter.mark_finished("error", "Dataset preparation failed; see error.txt")
        elapsed = time.monotonic() - reporter.started
        segments = int(reporter._last.get("segment_count", 0) or 0)
        audio_seconds = float(reporter._last.get("total_audio_seconds", 0.0) or 0.0)
        print(
            f">> Dataset preparation summary | status=error | items={segments} | "
            f"audio={audio_seconds:.3f}s | elapsed={elapsed:.3f}s | "
            f"{segments / elapsed if elapsed > 0 else 0.0:.3f} segments/s | output=unknown",
            flush=True,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
