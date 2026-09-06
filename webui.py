"""Thin launcher for the modular IndexTTS 2.5 Premium Gradio interface."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import webbrowser
from typing import Any

from indextts.utils.console_encoding import configure_console_output


ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IndexTTS 2.5 Premium SECourses",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="HTTP port; omitted by default so Gradio takes the next free port from 7860",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="HTTP bind address; omitted by default so Gradio uses its own default",
    )
    parser.add_argument("--share", action="store_true", help="Create a Gradio public share link")
    parser.add_argument("--model_dir", default=str(ROOT / "models"), help="IndexTTS 2.5 model directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose generation logging by default")
    parser.add_argument("--no-browser", dest="no_browser", action="store_true", help="Do not open a browser window")
    parser.add_argument("--browser", choices=("default", "chrome"), default="default", help="Browser to open after startup")
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


def open_app_browser(url: str, browser: str) -> None:
    if browser == "chrome":
        candidates = [shutil.which("chrome"), shutil.which("google-chrome")]
        if os.name == "nt":
            for variable in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
                folder = os.environ.get(variable)
                if folder:
                    candidates.append(str(Path(folder) / "Google/Chrome/Application/chrome.exe"))
        for path in candidates:
            if path and Path(path).is_file():
                if webbrowser.BackgroundBrowser(path).open_new_tab(url):
                    return
        print(">> Google Chrome was not found or could not open; opening the default browser.", flush=True)
    webbrowser.open_new_tab(url)


def main(argv: list[str] | None = None) -> int:
    configure_console_output()
    args = build_parser().parse_args(argv)
    demo = create_demo(args)
    from ui.common import FAVICON_PATH

    # Only forward an address or port the caller actually asked for: leaving them
    # unset lets Gradio scan upwards from 7860 instead of failing when that port
    # is already taken by another app.
    address: dict[str, Any] = {}
    if args.host:
        address["server_name"] = args.host
    if args.port:
        address["server_port"] = args.port

    demo.launch(
        **address,
        share=args.share,
        inbrowser=False,
        favicon_path=str(FAVICON_PATH),
        show_error=True,
        theme=demo.launch_theme,
        css=demo.launch_css,
        head=demo.launch_head,
        app_kwargs=demo.launch_app_kwargs,
        allowed_paths=[
            str(ROOT / "outputs"),
            str(ROOT / "datasets"),
            str(ROOT / "loras"),
            str(ROOT / "reference_audios"),
            str(ROOT / ".ui_state"),
        ],
        prevent_thread_lock=True,
    )
    # The port is chosen at launch when it was not requested, so repeat it on an
    # unbuffered line that survives redirected output.
    print(f">> IndexTTS 2.5 Premium SECourses is ready at {demo.local_url}", flush=True)
    if not args.no_browser:
        open_app_browser(demo.local_url, args.browser)
    demo.block_thread()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
