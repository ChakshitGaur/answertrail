import pytest

from tests import fake_llm


@pytest.fixture(autouse=True)
def isolated_from_local_config(settings):
    """A developer's .env (real keys, neural embeddings) must never leak into the tests."""
    settings.LLM_BASE_URL = settings.LLM_API_KEY = settings.LLM_MODEL = settings.LLM_FALLBACK_MODELS = ""
    settings.ANTHROPIC_API_KEY = ""
    settings.EMBEDDING_BACKEND = "hash"


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
