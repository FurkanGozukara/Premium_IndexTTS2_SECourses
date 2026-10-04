"""Thin launcher for the modular IndexTTS 2.5 Premium Gradio interface."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import time
import webbrowser
from typing import Any

from indextts.training.best_checkpoint import describe_migration, migrate_legacy_best_checkpoints
from indextts.utils.console_encoding import configure_console_output


ROOT = Path(__file__).resolve().parent
# On first use the CUDA driver prepares each GPU kernel module (it unpacks PyTorch's and the CUDA libraries' kernels
# and compiles cuFFT's for an RTX 5090) and keeps the result in a per-user cache that every CUDA program shares, 1 GB
# by default. One IndexTTS generation alone needs about 1.3 GB, so entries kept being evicted and every new process
# (the first generation, each batch item, every training worker) spent about 15 s preparing them again: the first
# generation after a start took 22 s instead of 8 s. The app keeps its own cache at the driver's 4 GiB maximum.
CUDA_JIT_CACHE_MAX_BYTES = 4 * 1024 ** 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ultimate Text To Speech Generator With Voice Cloning",
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
    parser.add_argument("--model_dir", default=str(ROOT / "models"), help="Shared speech model directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose generation logging by default")
    parser.add_argument("--no-browser", dest="no_browser", action="store_true", help="Do not open a browser window")
    parser.add_argument("--browser", choices=("default", "chrome"), default="default", help="Browser to open after startup")
    parser.add_argument("--device", default="auto", help="Default runtime device, such as auto, cuda:0, or cpu")
    return parser


def configure_cuda_jit_cache(model_dir: Path) -> None:
    """Point the driver's JIT cache at the model folder before CUDA starts; values the user set win."""
    if "CUDA_CACHE_PATH" not in os.environ:
        cache = model_dir / "cuda_jit_cache"
        try:
            cache.mkdir(parents=True, exist_ok=True)
        except OSError:  # an unwritable model folder keeps the driver's shared cache
            return
        os.environ["CUDA_CACHE_PATH"] = str(cache)
    os.environ.setdefault("CUDA_CACHE_MAXSIZE", str(CUDA_JIT_CACHE_MAX_BYTES))


def configure_environment(args: argparse.Namespace) -> None:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    args.model_dir = str(Path(args.model_dir).expanduser().resolve())
    configure_cuda_jit_cache(Path(args.model_dir))


def create_demo(args: argparse.Namespace):
    configure_environment(args)
    started = time.perf_counter()
    print(">> Loading Gradio and application modules...", flush=True)
    from ui.app import build_app
    print(f">> Modules ready in {time.perf_counter() - started:.2f}s; building interface...", flush=True)
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


def install_live_training_route(demo: Any) -> None:
    """Serve the training dashboard's one-second status outside Gradio's event system."""
    from fastapi.responses import JSONResponse
    from ui.training_tab import LIVE_TRAINING_ROUTE, live_training_snapshot

    def live_training_status(model: str | None = None) -> JSONResponse:
        return JSONResponse(live_training_snapshot(model=model), headers={"Cache-Control": "no-store"})

    try:
        demo.server_app.add_api_route(LIVE_TRAINING_ROUTE, live_training_status, methods=["GET"], include_in_schema=False)
    except Exception as exc:  # the dashboard still works through the Gradio timer
        print(f">> live training status route unavailable: {exc}", flush=True)


def main(argv: list[str] | None = None) -> int:
    startup_started = time.perf_counter()
    configure_console_output()
    print(">> Starting Ultimate Text To Speech Generator With Voice Cloning", flush=True)
    args = build_parser().parse_args(argv)
    try:
        for line in describe_migration(migrate_legacy_best_checkpoints(ROOT / "loras")):
            print(f">> best checkpoint naming: {line}", flush=True)
    except Exception as exc:  # never keep the app from starting over a rename
        print(f">> best checkpoint naming migration skipped: {exc}", flush=True)
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
        # Gradio's run history serialises every call's inputs and outputs into the
        # browser's local storage, on the main thread, for every event; with this
        # interface's lists it filled the storage quota and slowed every click.
        run_history=False,
        allowed_paths=[
            str(ROOT / "outputs"),
            str(ROOT / "datasets"),
            str(ROOT / "loras"),
            str(ROOT / "reference_audios"),
            str(ROOT / ".ui_state"),
        ],
        prevent_thread_lock=True,
    )
    install_live_training_route(demo)
    # The port is chosen at launch when it was not requested, so repeat it on an
    # unbuffered line that survives redirected output.
    print(f">> Ultimate Text To Speech Generator With Voice Cloning is ready at {demo.local_url} ({time.perf_counter()-startup_started:.2f}s startup)", flush=True)
    if not args.no_browser:
        open_app_browser(demo.local_url, args.browser)
    demo.block_thread()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
