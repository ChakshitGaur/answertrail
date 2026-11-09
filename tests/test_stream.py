import json

import pytest

from docqa.services import llm
from tests.test_api import make_client, upload_text

pytestmark = pytest.mark.django_db

REFUNDS = "Refunds are accepted within 30 days of purchase."


def parse_sse(response):
    """Return [(event, data_dict), ...] from a streaming response."""
    raw = b"".join(response.streaming_content).decode("utf-8")
    events = []
    for block in filter(None, raw.split("\n\n")):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def ask_stream(client, question="What is the refund window?", **extra):
    return client.post("/api/ask/stream/", {"question": question, **extra}, format="json")


def text_of(events):
    return "".join(d["t"] for e, d in events if e == "token")


@pytest.fixture
def client():
    c = make_client()
    upload_text(c, "Refunds", REFUNDS)
    return c


def test_headers_and_event_order_without_an_llm(client, settings):
    settings.LLM_BASE_URL, settings.ANTHROPIC_API_KEY = "", ""
    resp = ask_stream(client)
    assert resp.status_code == 200
    assert resp["Content-Type"].startswith("text/event-stream")
    assert resp["Cache-Control"] == "no-cache"
    events = parse_sse(resp)
    assert [e for e, _ in events] == ["sources", "token", "done"]
    assert events[0][1]["sources"][0]["document_title"] == "Refunds"
    assert llm.FALLBACK_PREFIX in text_of(events) and "30 days" in text_of(events)


def test_tokens_stream_in_pieces_and_keep_non_ascii(client, llm_settings):
    events = parse_sse(ask_stream(client))
    names = [e for e, _ in events]
    assert names[0] == "sources" and names[-1] == "done"
    assert names.count("token") == 5  # one SSE event per model delta, keep-alive ignored
    assert text_of(events) == "Hello wörld ✓ from good [1]."
    sent = llm_settings.requests[0]
    assert sent["body"]["stream"] is True and sent["auth"] == "Bearer test-key"


def test_overloaded_primary_fails_over_before_any_output(client, llm_settings, settings):
    settings.LLM_MODEL, settings.LLM_FALLBACK_MODELS = "busy-1", "good"
    events = parse_sse(ask_stream(client))
    assert text_of(events) == "Hello wörld ✓ from good [1]."
    assert [r["body"]["model"] for r in llm_settings.requests] == ["busy-1", "busy-1", "good"]


def test_unknown_model_fails_over_immediately(client, llm_settings, settings):
    settings.LLM_MODEL, settings.LLM_FALLBACK_MODELS = "missing-model", "good"
    events = parse_sse(ask_stream(client))
    assert text_of(events).endswith("from good [1].")
    assert [r["body"]["model"] for r in llm_settings.requests] == ["missing-model", "good"]


def test_every_model_failing_gives_an_error_event_not_a_crash(client, llm_settings, settings):
    settings.LLM_MODEL, settings.LLM_FALLBACK_MODELS = "busy-1", "busy-2"
    events = parse_sse(ask_stream(client))
    assert [e for e, _ in events] == ["sources", "error"]
    detail = events[-1][1]["detail"]
    assert "busy-1, busy-2" in detail and "high demand" in detail


def test_bad_key_is_reported_and_other_models_are_not_tried(client, llm_settings, settings):
    settings.LLM_MODEL, settings.LLM_FALLBACK_MODELS = "denied", "good"
    events = parse_sse(ask_stream(client))
    assert events[-1][0] == "error" and "API key not valid" in events[-1][1]["detail"]
    assert len(llm_settings.requests) == 1


def test_connection_dying_mid_stream_keeps_partial_text_then_errors(client, llm_settings, settings):
    settings.LLM_MODEL = "cut"
    events = parse_sse(ask_stream(client))
    names = [e for e, _ in events]
    assert names == ["sources", "token", "token", "error"]
    assert text_of(events) == "Hello wörld"
    assert "stream interrupted" in events[-1][1]["detail"]


def test_empty_model_answer_is_an_error(client, llm_settings, settings):
    settings.LLM_MODEL = "empty"
    events = parse_sse(ask_stream(client))
    assert events[-1][0] == "error" and "empty answer" in events[-1][1]["detail"]


def test_no_documents_says_so_without_calling_the_model(llm_settings):
    events = parse_sse(ask_stream(make_client("nodocs")))
    assert events[0][1]["sources"] == [] and events[0][1]["mode"] == "not_found"
    assert text_of(events) == llm.NO_RESULTS
    assert llm_settings.requests == []


def test_validation_and_auth_errors_are_plain_json_not_streams(client):
    assert ask_stream(client, question="").status_code == 400
    assert ask_stream(client, top_k=99).status_code == 400
    from rest_framework.test import APIClient

    assert APIClient().post("/api/ask/stream/", {"question": "x"}, format="json").status_code == 401


def test_cannot_stream_from_someone_elses_document(client, llm_settings):
    other = make_client("mallory")
    doc_id = client.get("/api/documents/").data[0]["id"]
    assert ask_stream(other, document_id=doc_id).status_code == 404
    events = parse_sse(ask_stream(other))
    assert events[0][1]["sources"] == []


def test_document_scope_limits_sources(client, llm_settings):
    upload_text(client, "Plants", "Photosynthesis occurs in the chloroplasts of leaves.")
    plants = next(d for d in client.get("/api/documents/").data if d["title"] == "Plants")
    events = parse_sse(ask_stream(client, "refund window", document_id=plants["id"]))
    assert {s["document_title"] for s in events[0][1]["sources"]} == {"Plants"}


def test_non_streaming_endpoint_works_against_the_same_server(client, llm_settings):
    resp = client.post("/api/ask/", {"question": "refund window?"}, format="json")
    assert resp.status_code == 200
    assert resp.data["answer"] == "Hello wörld ✓ from good [1]."
    assert not llm_settings.requests[0]["body"].get("stream")


def test_unexpected_error_mid_stream_sends_an_error_event(monkeypatch, db):
    from django.contrib.auth import get_user_model
    from rest_framework.test import APIClient

    from docqa.services import llm

    def boom(*a, **k):
        raise RuntimeError("kaboom")
        yield  # pragma: no cover

    monkeypatch.setattr(llm, "stream_answer", boom)
    user = get_user_model().objects.create_user("s", password="pw12345678x")
    c = APIClient()
    c.force_authenticate(user)
    c.post("/api/documents/", {"title": "t", "text": "Cats purr when happy. Dogs bark loudly."}, format="json")
    resp = c.post("/api/ask/stream/", {"question": "do cats purr?", "mode": "general"}, format="json")
    assert b"event: error" in b"".join(resp.streaming_content)
