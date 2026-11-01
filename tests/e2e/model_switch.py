"""Middleware for the E2E harness: reads tests/e2e/model.txt ("primary|fallback1,fallback2") on every request,
so a browser session can change which fake model the app talks to without restarting the server."""

from pathlib import Path

from django.conf import settings

PATH = Path(__file__).resolve().parent / "model.txt"


class Switch:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if PATH.exists():
            primary, _, fallbacks = PATH.read_text().strip().partition("|")
            settings.LLM_MODEL, settings.LLM_FALLBACK_MODELS = primary, fallbacks
        return self.get_response(request)
