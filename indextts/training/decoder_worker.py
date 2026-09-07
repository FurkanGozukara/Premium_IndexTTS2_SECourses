"""Subprocess entry point for voice decoder (s2mel) adaptation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

from indextts.utils.atomic_json import read_json_retry, write_json_atomic
from indextts.utils.console_encoding import configure_console_output

from .decoder_adapter import DecoderAdapterConfig, train_decoder_adapter


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IndexTTS 2.5 voice decoder adaptation worker")
    parser.add_argument("--config", required=True, help="DecoderAdapterConfig JSON file")
    parser.add_argument("--state-dir", required=True, help="Directory for status, log, and stop.flag")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_console_output()
    args = parse_args(argv)
    state_dir = Path(args.state_dir).expanduser().resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        config = DecoderAdapterConfig.from_json(args.config)
        result = train_decoder_adapter(config, state_dir, cancel_callback=lambda: (state_dir / "stop.flag").is_file())
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), flush=True)
        print(f">> Decoder adaptation summary | status={result.status} | steps={result.steps}/{result.total_steps} | "
              f"elapsed={result.elapsed_s:.3f}s | output={result.output_path}", flush=True)
        return 0
    except BaseException as exc:
        detail = traceback.format_exc()
        with (state_dir / "log.txt").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(detail.rstrip() + "\n")
        current = read_json_retry(state_dir / "status.json", {}) or {}
        current.update({"phase": "failed", "message": str(exc), "updated_at": time.time()})
        write_json_atomic(state_dir / "status.json", current, indent=2, ensure_ascii=False)
        traceback.print_exc()
        print(f">> Decoder adaptation summary | status=failed | elapsed={time.perf_counter() - started:.3f}s | output={state_dir}",
              flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
