"""The stdlib HTTP backend against a real local server (no network, no SDK)."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from foveate import Message, Role, errors
from foveate.backends import OpenAIBackend, base, http
from foveate.backends.embeddings import OpenAIEmbedder


class Handler(BaseHTTPRequestHandler):
    script = []  # (status, headers, body) consumed in order
    seen = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        Handler.seen.append((self.path, dict(self.headers), body))
        status, headers, payload = Handler.script.pop(0)
        raw = (
            payload
            if isinstance(payload, bytes)
            else json.dumps(payload).encode()
        )
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


@pytest.fixture()
def server():
    Handler.script, Handler.seen = [], []
    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1"
    httpd.shutdown()
    httpd.server_close()


def chat(content="hello", finish="stop", usage=None):
    return (
        200,
        {},
        {
            "choices": [
                {"message": {"content": content}, "finish_reason": finish}
            ],
            "usage": usage or {"prompt_tokens": 7, "completion_tokens": 3},
        },
    )


def request(**kw):
    return base.Request(messages=(Message(Role.USER, "hi"),), model="m", **kw)


def run(coro):
    return asyncio.run(coro)


def test_chat_completion_roundtrip_sends_key_options_and_reads_usage(server):
    Handler.script.append(
        chat(
            "hello",
            usage={
                "prompt_tokens": 9,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 4},
            },
        )
    )
    backend = OpenAIBackend(base_url=server, api_key="sk-test")
    out = run(
        backend.complete(
            request(max_tokens=50, options={"reasoning_effort": "low"})
        )
    )
    path, headers, body = Handler.seen[0]
    assert path == "/v1/chat/completions"
    assert headers["Authorization"] == "Bearer sk-test"
    assert body["model"] == "m" and body["max_tokens"] == 50
    assert (
        body["reasoning_effort"] == "low"
        and body["messages"][0]["content"] == "hi"
    )
    assert (
        out.text == "hello"
        and out.usage.cached_tokens == 4
        and out.usage.prompt_tokens == 9
    )


def test_reasoning_models_with_null_content_report_empty_truncated_text(server):
    Handler.script.append(chat(None, finish="length"))
    out = run(OpenAIBackend(base_url=server).complete(request()))
    assert out.text == "" and out.truncated


@pytest.mark.parametrize(
    "status,kind",
    [
        (429, errors.RateLimitError),
        (503, errors.TransientBackendError),
        (408, errors.TransientBackendError),
        (400, errors.PermanentBackendError),
        (401, errors.PermanentBackendError),
    ],
)
def test_http_statuses_map_to_the_error_taxonomy(server, status, kind):
    Handler.script.append((status, {"Retry-After": "2"}, {"error": "nope"}))
    with pytest.raises(kind) as caught:
        run(OpenAIBackend(base_url=server).complete(request()))
    if status == 429:
        assert caught.value.retry_after == 2.0
    assert str(status) in str(caught.value)


def test_bad_replies_and_unreachable_servers(server):
    Handler.script.append((200, {}, b"<html>not json</html>"))
    with pytest.raises(errors.ValidationError):
        run(OpenAIBackend(base_url=server).complete(request()))
    Handler.script.append((200, {}, {"choices": []}))
    with pytest.raises(errors.ValidationError):
        run(OpenAIBackend(base_url=server).complete(request()))
    Handler.script.append((200, {}, [1, 2]))
    with pytest.raises(errors.ValidationError):
        run(OpenAIBackend(base_url=server).complete(request()))
    with pytest.raises(errors.TransientBackendError):
        run(
            OpenAIBackend(base_url="http://127.0.0.1:9/v1", timeout=2).complete(
                request()
            )
        )


def test_only_http_urls_are_allowed():
    with pytest.raises(errors.PermanentBackendError):
        http.post_json("file:///etc/passwd", {}, {}, 1)


def test_defaults_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234/v1/")
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    backend = OpenAIBackend()
    assert backend.base_url == "http://localhost:1234/v1"
    assert backend.request_headers()["Authorization"] == "Bearer from-env"
    monkeypatch.delenv("OPENAI_BASE_URL")
    monkeypatch.delenv("OPENAI_API_KEY")
    assert OpenAIBackend().base_url == "https://api.openai.com/v1"
    assert "Authorization" not in OpenAIBackend().request_headers()


def test_embedder_batches_orders_and_sends_input_type(server):
    for _ in range(2):
        Handler.script.append(
            (
                200,
                {},
                {
                    "data": [
                        {"index": 1, "embedding": [0.0, 2.0]},
                        {"index": 0, "embedding": [3.0, 0.0]},
                    ]
                },
            )
        )
    embedder = OpenAIEmbedder(
        "emb",
        base_url=server,
        batch_size=2,
        passage_options={"input_type": "passage"},
    )
    vectors = run(embedder.embed(["a", "b", "c", "d"]))
    assert (
        len(vectors) == 4
        and vectors[0] == [1.0, 0.0]
        and vectors[1] == [0.0, 1.0]
    )
    assert Handler.seen[0][0] == "/v1/embeddings"
    assert (
        Handler.seen[0][2]["input_type"] == "passage" and len(Handler.seen) == 2
    )
    assert Handler.seen[0][2]["input"] == ["a", "b"]


def test_embedder_rejects_malformed_replies_and_bad_batch_size(server):
    Handler.script.append((200, {}, {"data": [{"index": 0}]}))
    with pytest.raises(errors.ValidationError):
        run(OpenAIEmbedder("emb", base_url=server).embed(["a"]))
    Handler.script.append((200, {}, {"data": []}))
    with pytest.raises(errors.ValidationError):
        run(OpenAIEmbedder("emb", base_url=server).embed(["a"]))
    with pytest.raises(errors.ConfigError):
        OpenAIEmbedder("emb", batch_size=0)
