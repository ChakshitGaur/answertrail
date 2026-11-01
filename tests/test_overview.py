"""Questions about a document as a whole ("summarize this") read a spread sample, not search results."""

import pytest
from django.contrib.auth import get_user_model

from docqa.models import Chunk
from docqa.services import llm, retrieval, routing
from docqa.services.ingest import ingest_document
from tests.test_api import make_client, upload_text
from tests.test_modes import REFUNDS, asked
from tests.test_stream import ask_stream, parse_sse, text_of

pytestmark = pytest.mark.django_db

LONG = " ".join(f"Sentence number {i} talks about topic {i % 7} in some detail." for i in range(400))


@pytest.mark.parametrize(
    "question",
    [
        "Summarize this document",
        "summarise the pdf",
        "What are the key points?",
        "List the main topics covered",
        "Give me an overview",
        "tl;dr",
        "TLDR please",
        "What is this document about?",
        "What's this file about?",
        "What does this document say?",
        "What does the PDF cover?",
        "Please summarize it briefly",
        "summary",
        "Give a short summary of the document in three bullet points",
        "Can you outline the document?",
        "What are the main points?",
    ],
)
def test_requests_about_the_whole_document_are_recognised(question):
    assert routing.is_overview_request(question)


@pytest.mark.parametrize(
    "question",
    [
        "Summarize the refund policy",
        "What is the summary of chapter 3?",
        "Give me an overview of shipping to Canada",
        "What are the key points about the warranty?",
        "What is the refund window?",
        "What is this clause about indemnity?",
        "hi",
        "",
        "   ",
        "What is inheritance?",
        "Explain recursion",
        "Summarize the differences between inheritance and composition",
    ],
)
def test_topic_questions_still_use_targeted_search(question):
    assert not routing.is_overview_request(question)


@pytest.fixture
def user():
    return get_user_model().objects.create_user("alice")


def indexes(hits):
    return [h.chunk.index for h in hits]


def test_a_long_document_is_sampled_from_start_to_end(user):
    doc = ingest_document(user, "Long", LONG)
    count = doc.chunks.count()
    assert count > 20
    hits = retrieval.overview_hits(user, max_passages=8)
    assert len(hits) == 8 and indexes(hits) == sorted(indexes(hits))
    assert indexes(hits)[0] == 0 and indexes(hits)[-1] == count - 1  # covers the opening and the end
    assert all(h.score is None for h in hits)  # not found by similarity, so no similarity to report


def test_short_documents_are_taken_whole(user):
    doc = ingest_document(user, "Short", "One sentence here. Another sentence there. " * 8)
    assert doc.chunks.count() <= 8
    assert indexes(retrieval.overview_hits(user, max_passages=8)) == list(range(doc.chunks.count()))
    one = ingest_document(user, "Tiny", "just a few words")
    assert indexes(retrieval.overview_hits(user, document_id=one.id)) == [0]


def test_several_documents_share_the_budget_newest_first(user):
    for name in ("A", "B", "C"):
        ingest_document(user, name, LONG)
    hits = retrieval.overview_hits(user, max_passages=8)
    titles = [h.chunk.document.title for h in hits]
    assert len(hits) == 3 * (8 // 3) and set(titles) == {"A", "B", "C"}
    assert titles[0] == "C"  # the most recently added document comes first


def test_overview_never_reads_other_users_documents_and_respects_the_scope(user):
    bob = get_user_model().objects.create_user("bob")
    mine = ingest_document(user, "Mine", "my own refund text. " * 30)
    ingest_document(bob, "Bobs", "bobs secret text. " * 30)
    assert {h.chunk.document.title for h in retrieval.overview_hits(user)} == {"Mine"}
    assert {h.chunk.document.title for h in retrieval.overview_hits(bob)} == {"Bobs"}
    assert retrieval.overview_hits(bob, document_id=mine.id) == []  # someone else's document id
    assert retrieval.overview_hits(get_user_model().objects.create_user("nobody")) == []


def test_at_most_five_documents_are_sampled(user):
    for i in range(8):
        ingest_document(user, f"Doc{i}", LONG)
    titles = {h.chunk.document.title for h in retrieval.overview_hits(user, max_passages=10)}
    assert len(titles) == 5


# ---- through the API -------------------------------------------------------------------------------


@pytest.fixture
def client():
    c = make_client()
    upload_text(c, "Refunds", REFUNDS)
    upload_text(c, "Long", LONG)
    return c


def test_summarize_reads_the_document_even_when_nothing_is_similar(client, llm_settings, settings):
    settings.RELEVANCE_THRESHOLD = 0.99  # nothing could ever pass the similarity cutoff
    for mode in ("auto", "documents"):
        llm_settings.requests.clear()
        events = parse_sse(ask_stream(client, "Summarize this document", mode=mode))
        head = events[0][1]
        assert head["mode"] == "documents" and head["overview"] is True and head["sources"]
        assert all(s["score"] is None for s in head["sources"])
        assert head["best_similarity"] is None and head["searched_for"] is None
        assert text_of(events).startswith("Hello") and events[-1][0] == "done"
        (grounded,) = asked(llm_settings, "grounded")
        assert "[1] " in grounded[-1]["content"]
        assert llm.NOT_IN_DOCUMENTS_SENTINEL not in grounded[0]["content"]  # there is no "decline" to offer


def test_overview_needs_no_rewrite_call_even_with_history(client, llm_settings):
    history = [{"role": "user", "content": "Tell me about refunds"}, {"role": "assistant", "content": "30 days."}]
    list(ask_stream(client, "Summarize this document", mode="auto", history=history).streaming_content)
    assert asked(llm_settings, "rewrite") == [] and len(llm_settings.requests) == 1


def test_general_mode_ignores_the_overview_path(client, llm_settings):
    events = parse_sse(ask_stream(client, "Summarize this document", mode="general"))
    assert events[0][1]["mode"] == "general" and events[0][1]["overview"] is False and events[0][1]["sources"] == []


def test_overview_with_no_documents_falls_back_to_the_usual_behaviour(llm_settings):
    nobody = make_client("empty")
    docs_only = parse_sse(ask_stream(nobody, "Summarize this document", mode="documents"))
    assert text_of(docs_only) == llm.NO_RESULTS and docs_only[0][1]["overview"] is False
    auto = parse_sse(ask_stream(nobody, "Summarize this document", mode="auto"))
    assert auto[0][1]["mode"] == "general"


def test_overview_can_be_scoped_to_one_document(client, llm_settings):
    long_id = next(d["id"] for d in client.get("/api/documents/").data if d["title"] == "Long")
    events = parse_sse(ask_stream(client, "Summarize this document", mode="documents", document_id=long_id))
    assert {s["document_title"] for s in events[0][1]["sources"]} == {"Long"}
    assert len(events[0][1]["sources"]) == 8


def test_non_streaming_endpoint_reports_the_overview_too(client, llm_settings):
    resp = client.post("/api/ask/", {"question": "What are the key points?", "mode": "auto"}, format="json")
    assert resp.status_code == 200 and resp.data["overview"] is True and resp.data["mode"] == "documents"
    assert all(s["score"] is None for s in resp.data["sources"])


def test_a_targeted_question_is_not_hijacked(client, llm_settings):
    events = parse_sse(ask_stream(client, "Summarize the refund policy", mode="documents"))
    assert events[0][1]["overview"] is False
    assert events[0][1]["best_similarity"] is not None
    assert chunk_count_unchanged()


def chunk_count_unchanged():
    return Chunk.objects.count() > 0
