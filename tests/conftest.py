import pytest

from tests import fake_llm


@pytest.fixture(autouse=True)
def isolated_from_local_config(settings):
    """A developer's .env (real keys, neural embeddings) must never leak into the tests."""
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
    llm._NO_EFFORT.clear()
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

    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    settings.LLM_BASE_URL = f"http://127.0.0.1:{fake_server.server_port}/v1"
    settings.LLM_API_KEY = "test-key"
    settings.LLM_MODEL = "good"
    settings.LLM_FALLBACK_MODELS = ""
    return fake_server
