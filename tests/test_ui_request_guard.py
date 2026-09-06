from types import SimpleNamespace

import gradio as gr
import pytest
from gradio.routes import App
from starlette.testclient import TestClient

from ui.request_guard import INSTANCE_HEADER, configure_request_guard


@pytest.fixture
def guarded_app():
    calls = []
    with gr.Blocks() as demo:
        text = gr.Textbox()
        output = gr.Textbox()
        table = gr.Dataframe(value=[["one"], ["two"]])

        def preview(*values):
            calls.append(values)
            return "preview ready"

        def select(evt: gr.SelectData):
            calls.append(evt.index)
            return str(evt.value)

        text.change(preview, [text] * 7, output, api_name="update_preview", queue=False)
        text.submit(preview, [text] * 3, output, api_name="cache_features", queue=False)
        table.select(select, outputs=output, api_name="select_segment", queue=False)
    demo.launch_head = ""
    configure_request_guard(demo)
    app = App.create_app(demo, app_kwargs=demo.launch_app_kwargs)
    # launch() normally supplies theme values; this fixture serves in-process.
    demo.config["body_css"] = {}
    with TestClient(app) as client:
        yield demo, client, calls


@pytest.mark.parametrize("route", ["api/predict", "run/predict", "queue/join", "call/update_preview", "cancel"])
@pytest.mark.parametrize("headers", [
    {"Origin": "http://testserver"},
    {"Sec-Fetch-Mode": "cors"},
    {INSTANCE_HEADER: "previous-server-instance"},
])
def test_foreign_or_old_tabs_cannot_dispatch_even_with_correct_input_count(guarded_app, route, headers):
    _, client, calls = guarded_app
    response = client.post(f"/gradio_api/{route}", json={"fn_index": 0, "data": ["text"] * 7}, headers=headers)
    assert response.status_code == 409
    assert response.json()["code"] == "stale_ui"
    assert not calls


@pytest.mark.parametrize("index,data", [(0, [None]), (0, [None, "captioner log"]), (1, ["avocado_int4", "blur"]), (0, [None] * 8)])
@pytest.mark.parametrize("route", ["api/predict", "run/predict", "queue/join"])
def test_bad_positional_inputs_are_rejected_before_gradio_dispatch(guarded_app, index, data, route):
    demo, client, calls = guarded_app
    response = client.post(f"/gradio_api/{route}", json={"fn_index": index, "data": data, "session_hash": "regression"}, headers={INSTANCE_HEADER: demo.ui_instance_id})
    assert response.status_code == 422
    assert response.json()["code"] == "invalid_ui_event"
    assert not calls


@pytest.mark.parametrize("route,body", [
    ("api/predict", {"fn_index": 251, "data": []}),
    ("queue/join", {"fn_index": 251, "data": [], "session_hash": "regression"}),
    ("run/missing", {"data": []}),
    ("call/missing", {"data": []}),
    ("call/v2/missing", {}),
    ("api/cache_features", {"fn_index": 0, "data": [None] * 7}),
])
def test_unknown_or_conflicting_event_identifiers_return_reload_error(guarded_app, route, body):
    _, client, calls = guarded_app
    response = client.post(f"/gradio_api/{route}", json=body)
    assert response.status_code == 409
    assert not calls


@pytest.mark.parametrize("event", [None, {}, {"index": 0}, {"value": "one"}])
def test_selection_requires_event_metadata(guarded_app, event):
    _, client, calls = guarded_app
    response = client.post("/gradio_api/run/select_segment", json={"data": [], "event_data": event})
    assert response.status_code == 422
    assert not calls


def test_current_browser_and_native_api_requests_still_run(guarded_app):
    demo, client, calls = guarded_app
    for headers in ({}, {"Origin": "http://testserver", INSTANCE_HEADER: demo.ui_instance_id}):
        response = client.post("/gradio_api/run/update_preview", json={"data": ["text"] * 7}, headers=headers)
        assert response.status_code == 200
        assert response.json()["data"] == ["preview ready"]
    response = client.post("/gradio_api/run/select_segment", json={"data": [], "event_data": {"index": [0, 0], "value": "one"}})
    assert response.status_code == 200
    assert response.json()["data"] == ["one"]
    assert len(calls) == 3


@pytest.mark.parametrize("body", [None, [], {"fn_index": []}, {"fn_index": True}, {"fn_index": 0, "data": "wrong type"}])
def test_malformed_bodies_do_not_raise_server_errors(guarded_app, body):
    _, client, calls = guarded_app
    response = client.post("/gradio_api/run/predict", json=body)
    assert response.status_code == 422
    assert not calls


def test_html_config_and_assets_remain_accessible(guarded_app):
    _, client, _ = guarded_app
    for path in ("/", "/config"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
    assert client.get("/gradio_api/app_id").status_code == 200


def test_instance_id_changes_per_build():
    first = SimpleNamespace(launch_head="")
    second = SimpleNamespace(launch_head="")
    configure_request_guard(first)
    configure_request_guard(second)
    assert first.ui_instance_id != second.ui_instance_id
    assert first.ui_instance_id in first.launch_head
    assert second.ui_instance_id not in first.launch_head
