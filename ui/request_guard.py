"""Reject stale Gradio event maps before they can dispatch to another handler.

Local Gradio apps reuse ports, but event numbers belong to one app instance.
This is a browser compatibility guard, not authentication: native API clients
can still call the documented named endpoints without a browser instance ID.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import TYPE_CHECKING
from uuid import uuid4

from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

if TYPE_CHECKING:
    from gradio import Blocks


INSTANCE_HEADER = "x-indextts-ui-instance"
RELOAD_MESSAGE = (
    "This browser tab belongs to an older session or a different app. "
    "Copy any unsaved text, then reload this page to connect to IndexTTS."
)
EVENT_PATH = re.compile(
    r"/gradio_api/(?:(run|api|call)/(?:v2/)?([^/]+)|(queue/join|cancel))/?$"
)


def _error(message: str, *, stale: bool = False) -> JSONResponse:
    return JSONResponse(
        {
            "error": message,
            "detail": message,
            "code": "stale_ui" if stale else "invalid_ui_event",
        },
        status_code=409 if stale else 422,
        headers={"Cache-Control": "no-store"},
    )


class UIRequestGuard(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, *, demo: Blocks, instance_id: str):
        super().__init__(app)
        self.demo = demo
        self.instance_id = instance_id

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        match = EVENT_PATH.search(request.url.path)
        if request.method != "POST" or match is None:
            response = await call_next(request)
            # A fresh navigation must receive this launch's event map and head.
            if (
                request.url.path.rstrip("/").endswith("/config")
                or "text/html" in response.headers.get("content-type", "")
            ):
                response.headers["Cache-Control"] = "no-store"
            return response

        instance = request.headers.get(INSTANCE_HEADER)
        browser_request = "origin" in request.headers or "sec-fetch-mode" in request.headers
        if (instance is not None or browser_request) and instance != self.instance_id:
            return _error(RELOAD_MESSAGE, stale=True)

        if match[3] == "cancel":
            return await call_next(request)
        try:
            body = await request.json()
        except (ValueError, UnicodeError):
            return _error("Event request must contain valid JSON.")
        if not isinstance(body, dict):
            return _error("Event request must be a JSON object.")

        fns = self.demo.fns
        session = body.get("session_hash")
        # Do not create server sessions just to validate an incoming request.
        if isinstance(session, str) and session in self.demo.state_holder.session_data:
            fns = self.demo.state_holder.session_data[session].blocks_config.fns
        index = body.get("fn_index")
        api_name = match[2]
        if index is not None and match[1] != "call":
            if type(index) is not int:
                return _error("Event index must be an integer.")
            fn = fns.get(index)
        else:
            fn = next((fn for fn in fns.values() if fn.api_name == api_name), None)
        if fn is None:
            return _error("Unknown event. " + RELOAD_MESSAGE, stale=True)
        if api_name and match[1] != "call" and api_name not in ("predict", fn.api_name):
            return _error("Event name and index disagree. " + RELOAD_MESSAGE, stale=True)

        # Gradio's v2 API maps named arguments itself. Legacy UI events send a
        # positional array; validate it before Gradio prints their full values.
        if "/call/v2/" not in request.url.path and not fn.cancels:
            data = body.get("data")
            counts = {len(fn.inputs)}
            if match[1] == "call":
                counts.add(sum(not component.skip_api for component in fn.inputs))
            if not isinstance(data, list) or len(data) not in counts:
                expected = " or ".join(map(str, sorted(counts)))
                return _error(
                    f"Event {fn.api_name!r} expects {expected} input values. "
                    "Reload the page if it is out of date."
                )

        if fn.collects_event_data and any(event == "select" for _, event in fn.targets):
            event = body.get("event_data")
            if not isinstance(event, dict) or not {"index", "value"}.issubset(event):
                return _error(
                    "Selection event requires index and value data. "
                    "Select an item in the current page."
                )
        return await call_next(request)


def configure_request_guard(demo: Blocks) -> None:
    instance_id = uuid4().hex
    script_path = Path(__file__).resolve().parents[1] / "ui_assets" / "request_guard.js"
    script = script_path.read_text(encoding="utf-8")
    demo.ui_instance_id = instance_id
    script = script.replace("__INDEXTTS_INSTANCE_ID__", json.dumps(instance_id))
    demo.launch_head += "\n<script>\n" + script + "\n</script>"
    demo.launch_app_kwargs = {
        "middleware": [Middleware(UIRequestGuard, demo=demo, instance_id=instance_id)]
    }
