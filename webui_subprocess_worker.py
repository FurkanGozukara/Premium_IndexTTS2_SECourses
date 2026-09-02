from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

os.environ["PYTHONUNBUFFERED"] = "1"
for stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(line_buffering=True, write_through=True)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
INDEXTTS_DIR = os.path.join(CURRENT_DIR, "indextts")
if INDEXTTS_DIR not in sys.path:
    sys.path.append(INDEXTTS_DIR)

from webui_generation_runner import create_tts, run_generation_request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IndexTTS WebUI subprocess worker")
    parser.add_argument("--request-file", required=True, help="Path to the generation request JSON file")
    parser.add_argument("--result-file", required=True, help="Path to the result JSON file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    with open(args.request_file, "r", encoding="utf-8") as handle:
        request = json.load(handle)

    try:
        tts = create_tts(
            request["runtime"],
            progress_file=request.get("progress_file"),
        )
        result = run_generation_request(request, tts, progress_callback=None)
        payload = {"status": "ok", **result}
        exit_code = 0
    except Exception as exc:
        traceback.print_exc()
        payload = {"status": "error", "error": str(exc)}
        exit_code = 1

    result_dir = os.path.dirname(args.result_file)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
    with open(args.result_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
