"""Auto mode: the model may say relevant-looking passages don't answer the question."""

import pytest

from docqa.services import llm
from tests.test_api import make_client, upload_text
from tests.test_llm import PASSAGES, FakeResponse, script
from tests.test_modes import REFUNDS, asked, meta
from tests.test_stream import ask_stream, parse_sse, text_of

pytestmark = pytest.mark.django_db


@pytest.fixture
def client():
    c = make_client()
    upload_text(c, "Refunds", REFUNDS)
    return c


# ---- the LLM layer ---------------------------------------------------------------------------


def stream_of(monkeypatch, settings, *pieces):
    """Make the streaming model emit exactly these pieces."""
    settings.LLM_BASE_URL = "http://x/v1"
    monkeypatch.setattr(llm, "_stream", lambda messages, effort="": iter(pieces))


def test_decline_instruction_is_added_only_when_allowed_and_grounded():
    plain = llm.build_messages("q", ["p"], None)
    allowed = llm.build_messages("q", ["p"], None, allow_decline=True)
    assert llm.NOT_IN_DOCUMENTS_SENTINEL not in plain[0]["content"]
    assert llm.NOT_IN_DOCUMENTS_SENTINEL in allowed[0]["content"]
    general = llm.build_messages("q", None, None, allow_decline=True)
    assert llm.NOT_IN_DOCUMENTS_SENTINEL not in general[0]["content"]


@pytest.mark.parametrize(
    "text",
    ["NOT_IN_DOCUMENTS", "not_in_documents.", "`NOT_IN_DOCUMENTS`", '"NOT_IN_DOCUMENTS"', "  \nNOT_IN_DOCUMENTS extra"],
)
def test_declined_matches_the_sentinel_with_decoration(text):
    assert llm.declined(text)


@pytest.mark.parametrize(
    "text", ["Refunds take 30 days.", "I could not find NOT_IN_DOCUMENTS here", "", "Not in the documents, sorry"]
)
def test_declined_ignores_real_answers(text):
    assert not llm.declined(text)


def test_stream_decline_split_across_chunks_raises_before_yielding(settings, monkeypatch):
    stream_of(monkeypatch, settings, "NOT_", "IN_", "DOCUMENTS")
    out = []
    with pytest.raises(llm.NotInDocuments):
        for piece in llm.stream_answer("q", ["p"], allow_decline=True):
            out.append(piece)
    assert out == []  # nothing the user could have seen


def test_stream_real_answers_pass_through_intact_and_in_order(settings, monkeypatch):
    stream_of(monkeypatch, settings, "Ref", "unds are", " accepted", " for 30 days.")
    assert "".join(llm.stream_answer("q", ["p"], allow_decline=True)) == "Refunds are accepted for 30 days."


def test_stream_short_answers_that_end_before_the_buffer_fills_are_not_lost(settings, monkeypatch):
    stream_of(monkeypatch, settings, "Yes.")
    assert "".join(llm.stream_answer("q", ["p"], allow_decline=True)) == "Yes."
    stream_of(monkeypatch, settings)  # an empty stream yields nothing and does not raise
    assert list(llm.stream_answer("q", ["p"], allow_decline=True)) == []


def test_stream_does_not_hold_anything_back_when_declining_is_not_allowed(settings, monkeypatch):
    stream_of(monkeypatch, settings, "NOT_IN_DOCUMENTS")
    assert list(llm.stream_answer("q", ["p"], allow_decline=False)) == ["NOT_IN_DOCUMENTS"]
    assert list(llm.stream_answer("q", None, allow_decline=True)) == ["NOT_IN_DOCUMENTS"]  # general: no decline


def test_non_streaming_answer_raises_when_declined(settings, monkeypatch):
    settings.LLM_BASE_URL, settings.LLM_MODEL = "http://x/v1", "primary"
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    declined_reply = {"choices": [{"message": {"content": "NOT_IN_DOCUMENTS"}}]}
    script(monkeypatch, FakeResponse(declined_reply))
    with pytest.raises(llm.NotInDocuments):
        llm.answer("q", PASSAGES, allow_decline=True)
    script(monkeypatch, FakeResponse(declined_reply))
    assert llm.answer("q", PASSAGES) == "NOT_IN_DOCUMENTS"  # not allowed to decline: returned as text


# ---- the API ---------------------------------------------------------------------------------


def test_auto_switches_to_a_general_answer_when_the_model_declines(client, llm_settings, settings):
    settings.LLM_MODEL, settings.LLM_FALLBACK_MODELS = "decline", ""
    events = parse_sse(ask_stream(client, "What is the refund window?", mode="auto"))
    names = [e for e, _ in events]
    assert names[0] == "sources" and meta(events)["mode"] == "documents"  # first labelled as documents...
    switch = next(d for e, d in events if e == "mode")
    assert switch == {"mode": "general", "sources": []}  # ...then told to switch, before any text
    assert names.index("mode") < names.index("token") and names[-1] == "done"
    assert text_of(events) == "General answer: What is the refund window?"  # nothing of the decline leaks
    assert len(asked(llm_settings, "grounded")) == 1 and len(asked(llm_settings, "general")) == 1


def test_decline_is_recognised_with_quotes_and_punctuation_in_one_chunk(client, llm_settings, settings):
    settings.LLM_MODEL, settings.LLM_FALLBACK_MODELS = "decline2", ""
    events = parse_sse(ask_stream(client, "What is the refund window?", mode="auto"))
    assert text_of(events) == "General answer: What is the refund window?"


def test_the_decline_instruction_is_only_sent_in_auto_mode(client, llm_settings, settings):
    settings.LLM_MODEL, settings.LLM_FALLBACK_MODELS = "decline", ""
    list(ask_stream(client, "What is the refund window?", mode="auto").streaming_content)
    assert "NOT_IN_DOCUMENTS" in asked(llm_settings, "grounded")[0][0]["content"]
    llm_settings.requests.clear()
    events = parse_sse(ask_stream(client, "What is the refund window?", mode="documents"))
    assert "NOT_IN_DOCUMENTS" not in asked(llm_settings, "grounded")[0][0]["content"]
    assert "mode" not in [e for e, _ in events]  # documents-only never switches to general knowledge


def test_a_failure_after_the_switch_is_reported_not_swallowed(client, llm_settings, settings, monkeypatch):
    settings.LLM_MODEL, settings.LLM_FALLBACK_MODELS = "decline", ""
    real = llm.stream_answer
    calls = []

    def flaky(question, passages, history=None, allow_decline=False):
        calls.append("general" if passages is None else "grounded")
        if passages is None:
            raise llm.LLMError("boom")
        yield from real(question, passages, history, allow_decline)

    monkeypatch.setattr(llm, "stream_answer", flaky)
    events = parse_sse(ask_stream(client, "What is the refund window?", mode="auto"))
    assert events[-1][0] == "error" and "boom" in events[-1][1]["detail"]
    assert calls == ["grounded", "general"]


def test_non_streaming_endpoint_also_switches_to_general(client, llm_settings, settings):
    settings.LLM_MODEL, settings.LLM_FALLBACK_MODELS = "decline", ""
    resp = client.post("/api/ask/", {"question": "What is the refund window?", "mode": "auto"}, format="json")
    assert resp.status_code == 200
    assert resp.data["mode"] == "general" and resp.data["sources"] == []
    assert resp.data["answer"] == "General answer: What is the refund window?"
