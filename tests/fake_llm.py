"""A tiny fake OpenAI-compatible /chat/completions server for tests and manual UI checks.

Behaviour is chosen by model name, so no state is needed:
  busy*     -> 503 (overloaded)
  missing*  -> 404 (unknown model)
  denied*   -> 401 (bad key)
  cut*      -> streams two tokens, then the connection dies mid-stream
  empty*    -> streams no text at all
  slow*     -> streams normally but 0.5 s per token (to test the Stop button)
  always400* -> 400 "Please pass a valid API key" (Google reports a bad key as 400)
  noeffort* -> 400 when the request carries reasoning_effort, otherwise normal
  nullcontent* / nochoices* -> 200 with content null / an empty choices list (non-streaming)
  markdown* -> a heading, bullet and numbered lists, bold, inline code and a fenced code block
  special*  -> newlines, quotes, backslashes, emoji, HTML, U+2028 and CJK in the answer
  split*    -> normal answer sent in 3-byte pieces (cuts frames and multi-byte characters)
  decline*  -> when allowed to, answers a grounded question with NOT_IN_DOCUMENTS (split over three
               chunks; "decline2" sends it in one chunk with quotes and a full stop)
  anything else -> answers "Hello wörld ✓ from <model> [1]." (non-ASCII on purpose)

The reply also depends on the system prompt: a "standalone search query" request (follow-up rewrite)
returns "refund policy: <last message>", and a general-assistant request returns
"General answer: <question>".

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
        if model.startswith("always400"):
            return self._json(400, {"error": {"message": "Please pass a valid API key"}})
        if model.startswith("noeffort") and "reasoning_effort" in body:
            return self._json(400, {"error": {"message": "Request contains an invalid argument."}})
        if model.startswith("nullcontent") and not body.get("stream"):
            return self._json(200, {"choices": [{"message": {"role": "assistant", "content": None}, "finish_reason": "length"}]})
        if model.startswith("nochoices") and not body.get("stream"):
            return self._json(200, {"choices": []})

        messages = body.get("messages", [])
        system = messages[0]["content"] if messages else ""
        last_user = messages[-1]["content"] if messages else ""
        if model.startswith("empty"):
            tokens = []
        elif model.startswith("special"):  # awkward characters that must survive JSON, SSE and the DOM
            tokens = ["line1\nline2\n\nline3", ' "quoted" \\ back\\slash ', "\U0001f600 <b>x</b> </script>   é中文"]
        elif model.startswith("markdown"):  # lists, a heading, bold, code and a fenced block, one piece at a time
            tokens = ["## Summary\n\n", "* **First** point [1]\n", "* Second `inline` point\n\n", "```py\nx = items[1]\n```\n", "1. one\n2. two <b>raw</b>"]
        elif model.startswith("decline") and "NOT_IN_DOCUMENTS" in system:  # may decline grounded answers
            tokens = ["`NOT_IN_DOCUMENTS`."] if model == "decline2" else ["NOT_", "IN_DOC", "UMENTS"]
        elif "standalone search query" in system:  # the follow-up rewrite request
            tokens = ["refund policy: " + last_user]
        elif "helpful, accurate assistant" in system:  # a general (not grounded) answer
            tokens = ["General", " answer: ", last_user[:60]]
        else:  # a grounded answer
            tokens = [t.format(model=model) for t in TOKENS]
        if not body.get("stream"):
            return self._json(200, {"choices": [{"message": {"role": "assistant", "content": "".join(tokens)}, "finish_reason": "stop"}]})

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")  # deliberately no charset
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        if model.startswith("split"):  # the whole stream in 3-byte pieces, which cuts frames and characters
            payload = "".join(
                f"data: {json.dumps({'choices': [{'delta': {'content': t}}]})}\n\n" for t in tokens
            ).encode() + b"data: [DONE]\n\n"
            for i in range(0, len(payload), 3):
                self._chunk(payload[i : i + 3])
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            return
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
