import pytest
import requests

from docqa.services import llm

PASSAGES = ["Refunds are accepted within 30 days."]
OK = {"choices": [{"message": {"content": " 30 days [1]. "}}]}


class FakeResponse:
    def __init__(self, payload=None, status=200, text=""):
        self._payload, self.status_code, self.text = payload or {}, status, text

    @property
    def ok(self):
        return self.status_code < 400

    def close(self):
        pass

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def fast_and_single_model(monkeypatch, settings):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    settings.LLM_MODEL, settings.LLM_FALLBACK_MODELS = "primary", ""


def script(monkeypatch, *responses):
    """Make requests.post return/raise each item in turn; returns the list of calls made."""
    calls, queue = [], list(responses)

    def fake_post(url, headers, json, timeout, stream=False):
        calls.append({"url": url, "headers": headers, "body": json, "stream": stream})
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(llm.requests, "post", fake_post)
    return calls


def test_no_llm_configured_returns_extractive_fallback(settings):
    settings.LLM_BASE_URL, settings.ANTHROPIC_API_KEY = "", ""
    assert "Refunds are accepted" in llm.answer("refund window?", PASSAGES)
    assert llm.answer("anything", []).startswith("I couldn't find")


def test_openai_compatible_request_shape(settings, monkeypatch):
    settings.LLM_BASE_URL = "http://localhost:11434/v1/"
    settings.LLM_MODEL, settings.LLM_API_KEY = "llama3.2", "secret"
    calls = script(monkeypatch, FakeResponse(OK))
    assert llm.answer("refund window?", PASSAGES) == "30 days [1]."
    assert calls[0]["url"] == "http://localhost:11434/v1/chat/completions"
    assert calls[0]["headers"] == {"Authorization": "Bearer secret"}
    assert calls[0]["body"]["model"] == "llama3.2"
    assert "[1] Refunds are accepted" in calls[0]["body"]["messages"][1]["content"]


def test_omits_auth_header_without_key(settings, monkeypatch):
    settings.LLM_BASE_URL, settings.LLM_API_KEY = "http://localhost:11434/v1", ""
    calls = script(monkeypatch, FakeResponse(OK))
    llm.answer("q", PASSAGES)
    assert calls[0]["headers"] == {}


def test_retries_overloaded_server_then_succeeds(settings, monkeypatch):
    settings.LLM_BASE_URL = "http://x/v1"
    calls = script(monkeypatch, FakeResponse(status=503, text="overloaded"), FakeResponse(OK))
    assert llm.answer("q", PASSAGES) == "30 days [1]."
    assert len(calls) == 2


def test_gives_up_after_repeated_503_and_reports_provider_message(settings, monkeypatch):
    settings.LLM_BASE_URL = "http://x/v1"
    calls = script(monkeypatch, *[FakeResponse(status=503, text='{"error": "model is overloaded"}')] * 2)
    with pytest.raises(llm.LLMError) as exc:
        llm.answer("q", PASSAGES)
    assert len(calls) == 2
    assert "503" in str(exc.value) and "model is overloaded" in str(exc.value)


def test_overloaded_primary_falls_back_to_next_model(settings, monkeypatch):
    settings.LLM_BASE_URL = "http://x/v1"
    settings.LLM_FALLBACK_MODELS = "backup-1, backup-2"
    overloaded = FakeResponse(status=503, text="high demand")
    calls = script(monkeypatch, overloaded, overloaded, FakeResponse(OK))
    assert llm.answer("q", PASSAGES) == "30 days [1]."
    assert [c["body"]["model"] for c in calls] == ["primary", "primary", "backup-1"]


def test_unknown_model_skips_straight_to_the_next(settings, monkeypatch):
    settings.LLM_BASE_URL = "http://x/v1"
    settings.LLM_FALLBACK_MODELS = "backup"
    calls = script(monkeypatch, FakeResponse(status=404, text="model not found"), FakeResponse(OK))
    assert llm.answer("q", PASSAGES) == "30 days [1]."
    assert [c["body"]["model"] for c in calls] == ["primary", "backup"]


def test_all_models_failing_lists_every_model_tried(settings, monkeypatch):
    settings.LLM_BASE_URL = "http://x/v1"
    settings.LLM_FALLBACK_MODELS = "backup"
    script(monkeypatch, *[FakeResponse(status=503, text="busy")] * 4)
    with pytest.raises(llm.LLMError, match=r"primary, backup.*busy"):
        llm.answer("q", PASSAGES)


def test_auth_error_does_not_try_other_models(settings, monkeypatch):
    settings.LLM_BASE_URL = "http://x/v1"
    settings.LLM_FALLBACK_MODELS = "backup"
    calls = script(monkeypatch, FakeResponse(status=403, text="forbidden"))
    with pytest.raises(llm.LLMError, match="forbidden"):
        llm.answer("q", PASSAGES)
    assert len(calls) == 1


def test_time_budget_stops_further_attempts(settings, monkeypatch):
    settings.LLM_BASE_URL = "http://x/v1"
    settings.LLM_FALLBACK_MODELS = "backup"
    clock = iter([0, 0, 1000])  # deadline set at t=0; the second attempt sees t=1000
    monkeypatch.setattr(llm.time, "monotonic", lambda: next(clock))
    calls = script(monkeypatch, FakeResponse(status=503, text="busy"))
    with pytest.raises(llm.LLMError, match="ran out of time"):
        llm.answer("q", PASSAGES)
    assert len(calls) == 1


def test_missing_model_name_is_a_clear_error(settings):
    settings.LLM_BASE_URL, settings.LLM_MODEL = "http://x/v1", ""
    with pytest.raises(llm.LLMError, match="LLM_MODEL is empty"):
        llm.answer("q", PASSAGES)


def test_connection_errors_are_retried_then_reported(settings, monkeypatch):
    settings.LLM_BASE_URL = "http://x/v1"
    calls = script(monkeypatch, *[requests.ConnectionError("refused")] * 2)
    with pytest.raises(llm.LLMError, match="ConnectionError"):
        llm.answer("q", PASSAGES)
    assert len(calls) == 2


def test_auth_errors_are_not_retried(settings, monkeypatch):
    settings.LLM_BASE_URL = "http://x/v1"
    calls = script(monkeypatch, FakeResponse(status=401, text="API key not valid"))
    with pytest.raises(llm.LLMError, match="API key not valid"):
        llm.answer("q", PASSAGES)
    assert len(calls) == 1


def test_malformed_success_response_is_an_error(settings, monkeypatch):
    settings.LLM_BASE_URL = "http://x/v1"
    script(monkeypatch, FakeResponse({"unexpected": True}))
    with pytest.raises(llm.LLMError):
        llm.answer("q", PASSAGES)
