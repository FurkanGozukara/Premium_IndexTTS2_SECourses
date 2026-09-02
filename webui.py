"""Thin launcher for the modular IndexTTS 2.5 Premium Gradio interface."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IndexTTS 2.5 Premium SECourses",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=7860, help="HTTP port")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind address")
    parser.add_argument("--share", action="store_true", help="Create a Gradio public share link")
    parser.add_argument("--model_dir", default=str(ROOT / "models"), help="IndexTTS 2.5 model directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose generation logging by default")
    parser.add_argument("--no-browser", dest="no_browser", action="store_true", help="Do not open a browser window")
    parser.add_argument("--device", default="auto", help="Default runtime device, such as auto, cuda:0, or cpu")
    return parser


def configure_environment(args: argparse.Namespace) -> None:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    args.model_dir = str(Path(args.model_dir).expanduser().resolve())


def create_demo(args: argparse.Namespace):
    configure_environment(args)
    from ui.app import build_app

    demo = build_app(args)
    demo.queue(default_concurrency_limit=2)
    return demo


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    demo = create_demo(args)
    from ui.common import FAVICON_PATH

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
        favicon_path=str(FAVICON_PATH),
        show_error=True,
        theme=demo.launch_theme,
        css=demo.launch_css,
        head=demo.launch_head,
        allowed_paths=[
            str(ROOT / "outputs"),
            str(ROOT / "datasets"),
            str(ROOT / "loras"),
            str(ROOT / ".ui_state"),
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
