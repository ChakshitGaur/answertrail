"""A tiny fake OpenAI-compatible /chat/completions server for tests and manual UI checks.

Behaviour is chosen by model name, so no state is needed:
  busy*     -> 503 (overloaded)
  missing*  -> 404 (unknown model)
  denied*   -> 401 (bad key)
  cut*      -> streams two tokens, then the connection dies mid-stream
  empty*    -> streams no text at all
  slow*     -> streams normally but 0.5 s per token (to test the Stop button)
  anything else -> answers "Hello wörld ✓ from <model> [1]." (non-ASCII on purpose)

Run by hand:  python tests/fake_llm.py 8765
"""

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOKENS = ["Hello", " wörld", " ✓", " from {model}", " [1]."]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "FakeLLM"

    def log_message(self, *args):  # keep test output quiet
        pass

    def _json(self, status, payload):
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _chunk(self, data: bytes):
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def do_GET(self):
        self._json(200, {"data": [{"id": "good"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        model = body.get("model", "")
        self.server.requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        if model.startswith("busy"):
            return self._json(503, {"error": {"code": 503, "message": "This model is currently experiencing high demand."}})
        if model.startswith("missing"):
            return self._json(404, {"error": {"message": f"model {model} not found"}})
        if model.startswith("denied"):
            return self._json(401, {"error": {"message": "API key not valid"}})

        tokens = [] if model.startswith("empty") else [t.format(model=model) for t in TOKENS]
        if not body.get("stream"):
            return self._json(200, {"choices": [{"message": {"content": "".join(tokens)}}]})

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")  # deliberately no charset
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for i, token in enumerate(tokens):
            if model.startswith("cut") and i == 2:
                self.close_connection = True
                return  # no terminating chunk: the client sees a broken stream
            frame = {"choices": [{"delta": {"content": token}}]}
            self._chunk(f"data: {json.dumps(frame)}\n\n".encode())
            time.sleep(0.5 if model.startswith("slow") else 0.03)
        self._chunk(b": keep-alive\n\n")  # a comment line the client must ignore
        self._chunk(b"data: [DONE]\n\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


def start(port: int = 0) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.requests = []
    server.handle_error = lambda request, client_address: None  # the "cut" model resets on purpose
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == "__main__":
    srv = start(int(sys.argv[1]) if len(sys.argv) > 1 else 8765)
    print(f"fake LLM on http://127.0.0.1:{srv.server_port}/v1", flush=True)
    threading.Event().wait()
