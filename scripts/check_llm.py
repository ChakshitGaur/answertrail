"""Diagnose the LLM_* settings: lists models, then makes one tiny chat call and shows the raw reply.

Usage: python scripts/check_llm.py [model]
Reads LLM_BASE_URL / LLM_MODEL / LLM_API_KEY from the environment; prompts (hidden) for the key.
The key is never printed.
"""

import getpass
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.env import load_dotenv  # noqa: E402

load_dotenv(__import__("pathlib").Path(__file__).resolve().parent.parent / ".env")

base =os.environ.get("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/").rstrip("/")
model = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("LLM_MODEL", "gemini-3.8-flash")
key = os.environ.get("LLM_API_KEY") or getpass.getpass("API key (hidden): ")
headers = {"Authorization": f"Bearer {key}"}


def show(resp):
    body = " ".join(resp.text.split())[:400]
    print(f"  -> HTTP {resp.status_code}: {body}")


print(f"base={base}\nmodel={model}\n")

print("1) GET /models")
try:
    r = requests.get(f"{base}/models", headers=headers, timeout=30)
    if r.ok:
        ids = [m["id"].removeprefix("models/") for m in r.json().get("data", [])]
        flash = sorted(i for i in ids if "flash" in i)
        print(f"  -> HTTP 200, {len(ids)} models. Flash models: {', '.join(flash) or 'none'}")
        print(f"  -> '{model}' available: {model in ids}")
    else:
        show(r)
except requests.RequestException as exc:
    print(f"  -> {exc.__class__.__name__}: {exc}")

print("\n2) POST /chat/completions (up to 4 tries)")
payload = {"model": model, "max_tokens": 30, "messages": [{"role": "user", "content": "Reply with the single word: pong"}]}
for attempt in range(1, 5):
    began = time.monotonic()
    try:
        r = requests.post(f"{base}/chat/completions", headers=headers, json=payload, timeout=60)
    except requests.RequestException as exc:
        print(f"  try {attempt} ({time.monotonic() - began:.1f}s): {exc.__class__.__name__}: {exc}")
    else:
        print(f"  try {attempt} ({time.monotonic() - began:.1f}s):", end="")
        show(r)
        if r.ok or r.status_code not in (429, 500, 502, 503, 504):
            break
    time.sleep(3 * attempt)
