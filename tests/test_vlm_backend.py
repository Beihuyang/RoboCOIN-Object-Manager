import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from PIL import Image

import stage3_attribute as stage3
import vlm_backend


def _start_server(behavior):
    class Handler(BaseHTTPRequestHandler):
        requests = []

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length)
            try:
                body = json.loads(raw.decode("utf-8"))
            except Exception:
                body = None
            Handler.requests.append(body)
            status, payload = behavior(body)
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):  # noqa: A003
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, Handler


def _stop_server(server):
    server.shutdown()
    server.server_close()


def _image(size=(1600, 900)):
    return Image.new("RGB", size, (120, 60, 200))


def _success_behavior(body):
    return 200, {"choices": [{"message": {"content": '{"object_name": "cup"}'}}]}


def _echo_prompt_behavior(body):
    content = body["messages"][0]["content"]
    prompt = next(item["text"] for item in content if item.get("type") == "text")
    return 200, {"choices": [{"message": {"content": json.dumps({"prompt": prompt})}}]}


def test_generate_json_posts_openai_vision_request():
    server, handler = _start_server(_success_behavior)
    try:
        backend = vlm_backend.OpenAIChatBackend(
            f"http://127.0.0.1:{server.server_port}/v1", "test-key",
            model="glm-5.3-flash", retry_backoff=0.01,
        )
        answer, raw = backend.generate_json(
            [_image(), _image((300, 200))], "choose object", max_new_tokens=64,
        )
        assert answer == {"object_name": "cup"}
        assert raw == '{"object_name": "cup"}'
        sent = handler.requests[0]
        assert sent["model"] == "glm-5.3-flash"
        # max_new_tokens is scaled up so reasoning tokens fit the same request
        assert sent["max_tokens"] == 192
        assert sent["reasoning_effort"] == "low"
        message = sent["messages"][0]
        parts = message["content"]
        assert [part["type"] for part in parts] == ["image_url", "image_url", "text"]
        assert parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert parts[-1]["text"] == "choose object"
        assert sent["temperature"] == 0.0
    finally:
        _stop_server(server)

def test_image_is_downscaled_to_api_max_edge():
    server, handler = _start_server(_success_behavior)
    try:
        backend = vlm_backend.OpenAIChatBackend(
            f"http://127.0.0.1:{server.server_port}/v1", "k",
            max_image_edge=128, jpeg_quality=80, retry_backoff=0.01,
        )
        backend.generate_json([_image((4000, 2000))], "p")
        url = handler.requests[0]["messages"][0]["content"][0]["image_url"]["url"]
        payload = url.split(",", 1)[1]
        decoded = __import__("base64").b64decode(payload)
        from io import BytesIO

        width, height = Image.open(BytesIO(decoded)).size
        assert max(width, height) <= 128
    finally:
        _stop_server(server)


def test_retries_transient_http_errors_then_succeeds():
    calls = {"count": 0}

    def behavior(body):
        calls["count"] += 1
        if calls["count"] == 1:
            return 503, {"error": "temporary"}
        return 200, {"choices": [{"message": {"content": '{"ok": true}'}}]}

    server, _ = _start_server(behavior)
    try:
        backend = vlm_backend.OpenAIChatBackend(
            f"http://127.0.0.1:{server.server_port}/v1", "k",
            max_retries=3, retry_backoff=0.01,
        )
        answer, _ = backend.generate_json([_image((40, 40))], "p")
        assert answer == {"ok": True}
        assert calls["count"] == 2
    finally:
        _stop_server(server)


def test_token_scale_and_thinking_params_are_sent():
    server, handler = _start_server(_success_behavior)
    try:
        backend = vlm_backend.OpenAIChatBackend(
            f"http://127.0.0.1:{server.server_port}/v1", "k",
            reasoning_effort="low", thinking="enabled", token_scale=2.0,
            retry_backoff=0.01,
        )
        backend.generate_json([_image((20, 20))], "p", max_new_tokens=100)
        sent = handler.requests[0]
        assert sent["max_tokens"] == 200
        assert sent["thinking"] == {"type": "enabled"}
        assert sent["reasoning_effort"] == "low"
    finally:
        _stop_server(server)


def test_non_transient_http_error_raises():
    server, _ = _start_server(lambda body: (400, {"error": "bad request"}))
    try:
        backend = vlm_backend.OpenAIChatBackend(
            f"http://127.0.0.1:{server.server_port}/v1", "k",
            retry_backoff=0.01,
        )
        with pytest.raises(RuntimeError, match="HTTP 400"):
            backend.generate_json([_image((40, 40))], "p")
    finally:
        _stop_server(server)


def test_generate_raw_batch_preserves_order_and_extracts_text():
    server, _ = _start_server(_echo_prompt_behavior)
    try:
        backend = vlm_backend.OpenAIChatBackend(
            f"http://127.0.0.1:{server.server_port}/v1", "k",
            concurrency=4, retry_backoff=0.01,
        )
        groups = [[_image((20, 20))] for _ in range(6)]
        prompts = [f"object-{index}" for index in range(6)]
        raws = backend.generate_raw_batch(groups, prompts, max_new_tokens=48)
        assert len(raws) == 6
        for index, raw in enumerate(raws):
            assert json.loads(raw) == {"prompt": f"object-{index}"}
    finally:
        _stop_server(server)


def test_backend_from_env_requires_base_and_key(monkeypatch):
    monkeypatch.delenv("VLM_API_BASE", raising=False)
    monkeypatch.delenv("VLM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="VLM_API_BASE"):
        vlm_backend.openai_backend_from_env()
    monkeypatch.setenv("VLM_API_BASE", "https://example/v1")
    with pytest.raises(RuntimeError, match="VLM_API_KEY"):
        vlm_backend.openai_backend_from_env()
    monkeypatch.setenv("VLM_API_KEY", "secret")
    monkeypatch.setenv("VLM_MODEL", "glm-5.3-flash")
    backend = vlm_backend.openai_backend_from_env()
    assert backend.model == "glm-5.3-flash"
    assert backend.api_base == "https://example/v1"
    assert backend.reasoning_effort == "low"
    assert backend.token_scale == 3.0


class _StubBackend:
    def __init__(self):
        self.json_calls = []
        self.batch_calls = []

    def generate_json(self, images, prompt, max_new_tokens=384):
        image_count = len(images) if isinstance(images, (list, tuple)) else 1
        self.json_calls.append((image_count, prompt, max_new_tokens))
        return {"object_name": "cup"}, '{"object_name": "cup"}'

    def generate_raw_batch(self, image_groups, prompts, max_new_tokens=384):
        self.batch_calls.append((len(image_groups), max_new_tokens))
        return ['{"object_name": "cup"}'] * len(image_groups)


def test_stage3_generate_functions_delegate_to_api_backend():
    stub = _StubBackend()
    stage3.set_vlm_api_backend(stub)
    try:
        image = Image.new("RGB", (10, 10))
        answer, raw = stage3._generate_json([image, image], "prompt", None, None, "api", 99)
        assert answer == {"object_name": "cup"}
        assert raw == '{"object_name": "cup"}'
        assert stub.json_calls == [(2, "prompt", 99)]

        raws = stage3._generate_raw_batch([[image], [image]], "p", None, None, "api", 7)
        assert raws == ['{"object_name": "cup"}', '{"object_name": "cup"}']
        assert stub.batch_calls == [(2, 7)]
    finally:
        stage3.set_vlm_api_backend(None)
