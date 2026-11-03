import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tests import fake_llm

# A developer's .env may switch LangSmith tracing on; tests must never send their prompts anywhere.
os.environ["LANGSMITH_TRACING"] = os.environ["LANGCHAIN_TRACING_V2"] = "false"


@pytest.fixture(autouse=True)
def isolated_from_local_config(settings, monkeypatch):
    """A developer's .env (real keys, neural embeddings) must never leak into the tests."""
    import openai._base_client

    # the OpenAI SDK sleeps between retries; the tests do not need to wait
    monkeypatch.setattr(openai._base_client.BaseClient, "_calculate_retry_timeout", lambda self, *a, **k: 0)
    settings.LLM_BASE_URL = settings.LLM_API_KEY = settings.LLM_MODEL = settings.LLM_FALLBACK_MODELS = ""
    settings.ANTHROPIC_API_KEY = ""
    settings.EMBEDDING_BACKEND = "hash"
    # the hash embedder's scores are tiny, so by default no cutoff; routing tests set their own
    settings.RELEVANCE_THRESHOLD = -1.0
    settings.LLM_REASONING_EFFORT = settings.LLM_REASONING_EFFORT_GENERAL = ""
    settings.WARMUP_EMBEDDINGS = False
    from docqa.services import llm, retrieval

    from django.core.cache import cache

    retrieval.clear_cache()  # every test gets a fresh database whose ids start over, so never reuse a corpus
    llm._COOLING.clear()  # a model rate-limited in one test must not be skipped in the next
    llm._client.cache_clear()
    cache.clear()  # rate-limit counters live in the cache; one test's requests must not throttle the next


@pytest.fixture
def no_throttle(monkeypatch):
    """For tests that deliberately send hundreds of requests."""
    from rest_framework.throttling import SimpleRateThrottle

    monkeypatch.setattr(SimpleRateThrottle, "THROTTLE_RATES", {"user": "1000000/min"})


@pytest.fixture
def fake_server():
    server = fake_llm.start()
    yield server
    server.shutdown()


@pytest.fixture
def llm_settings(settings, fake_server, monkeypatch):
    """Point the app at the fake server; returns it so tests can inspect received requests."""
    from docqa.services import llm

    settings.LLM_BASE_URL = f"http://127.0.0.1:{fake_server.server_port}/v1"
    settings.LLM_API_KEY = "test-key"
    settings.LLM_MODEL = "good"
    settings.LLM_FALLBACK_MODELS = ""
    return fake_server


class _ScriptedHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        server = self.server
        server.requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        reply = server.replies.pop(0)
        if reply == "reset":  # the connection dies without an answer
            self.connection.shutdown(socket.SHUT_RDWR)
            return
        if reply == "hang":  # outlasts the client's read timeout
            time.sleep(1)
            return
        if isinstance(reply, tuple) and reply[0] == "sse":  # ("sse", [lines]): each line is sent as an event of its own
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for line in reply[1]:
                self.wfile.write(f"{line}\n\n".encode())
            return
        if isinstance(reply, tuple) and reply[0] == "raw":  # ("raw", text): exactly this body, with status 200
            raw = reply[1].encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        status, payload = (200, reply) if isinstance(reply, dict) else reply
        raw = json.dumps(payload if isinstance(payload, dict) else {"error": {"message": payload}}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def script(settings):
    """Start a local server that answers each request with the next reply and records what it was sent.

    A reply is a payload dict (HTTP 200), a (status, message) pair, ("sse", lines), ("raw", body text),
    "reset" (drop the connection) or "hang" (outlast the client's read timeout). Calling `script(...)` also points the app at the server.
    """
    servers = []

    def start(*replies):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _ScriptedHandler)
        server.requests, server.replies = [], list(replies)
        server.handle_error = lambda request, client_address: None
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        settings.LLM_BASE_URL = f"http://127.0.0.1:{server.server_port}/v1/"
        return server

    yield start
    for server in servers:
        server.shutdown()
