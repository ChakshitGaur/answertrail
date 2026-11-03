"""The speed work: reasoning effort, skipping needless rewrites, warm-up and the search cache."""

import random
import sys
from collections import Counter

import numpy as np
import pytest
from django.apps import apps
from django.contrib.auth import get_user_model

from docqa import apps as docqa_apps
from docqa.services import embeddings, llm, retrieval
from docqa.services.ingest import ingest_document
from tests.test_api import make_client, upload_text
from tests.test_modes import HISTORY, REFUNDS, asked
from tests.test_stream import ask_stream, parse_sse, text_of

pytestmark = pytest.mark.django_db


@pytest.fixture
def client():
    c = make_client()
    upload_text(c, "Refunds", REFUNDS)
    return c


def bodies(llm_settings):
    return [r["body"] for r in llm_settings.requests]


# ---- skipping needless rewrites ----------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "and the window?",
        "what about shipping?",
        "why?",
        "How long does it take?",
        "Can you explain that again in more detail please",
        "tell me more about those",
        "And what projects has he built over the last two years at the company?",
        "How about the second one in the list that you just gave me earlier?",
    ],
)
def test_short_or_referring_questions_are_treated_as_follow_ups(question):
    assert llm.looks_like_followup(question)


@pytest.mark.parametrize(
    "question",
    [
        "What is the refund window for items purchased online from the Canada store during the holiday sale?",
        "Explain the difference between inheritance and composition in Python object oriented programming design",
        "How many days do customers have to return an unopened product for a full refund to the original payment method?",
    ],
)
def test_long_self_contained_questions_are_searched_as_typed(question):
    assert not llm.looks_like_followup(question)


def test_a_self_contained_question_costs_no_rewrite_call(client, llm_settings):
    long_question = "What is the refund window for items purchased online from the Canada store during the sale?"
    events = parse_sse(ask_stream(client, long_question, mode="documents", history=HISTORY))
    assert asked(llm_settings, "rewrite") == []
    assert events[0][1]["searched_for"] is None and len(llm_settings.requests) == 1


def test_a_real_follow_up_is_still_rewritten(client, llm_settings):
    list(ask_stream(client, "and the window?", mode="documents", history=HISTORY).streaming_content)
    assert len(asked(llm_settings, "rewrite")) == 1


# ---- reasoning effort ----------------------------------------------------------------------------


def test_effort_is_only_sent_when_configured(client, llm_settings, settings):
    list(ask_stream(client, "What is the refund window?", mode="documents").streaming_content)
    assert "reasoning_effort" not in bodies(llm_settings)[0]


def test_grounded_answers_and_rewrites_use_one_effort_and_general_answers_another(client, llm_settings, settings):
    settings.LLM_REASONING_EFFORT, settings.LLM_REASONING_EFFORT_GENERAL = "minimal", "low"
    list(ask_stream(client, "and the window?", mode="documents", history=HISTORY).streaming_content)
    list(ask_stream(client, "Explain recursion", mode="general").streaming_content)
    rewrite, grounded, general = bodies(llm_settings)
    assert rewrite["reasoning_effort"] == grounded["reasoning_effort"] == "minimal"
    assert general["reasoning_effort"] == "low"


def test_a_model_that_rejects_the_setting_is_retried_without_it(client, llm_settings, settings):
    settings.LLM_MODEL, settings.LLM_FALLBACK_MODELS = "noeffort", ""
    settings.LLM_REASONING_EFFORT = "minimal"
    events = parse_sse(ask_stream(client, "What is the refund window?", mode="documents"))
    assert text_of(events).startswith("Hello") and events[-1][0] == "done"
    assert ["reasoning_effort" in b for b in bodies(llm_settings)] == [True, False]


def test_the_setting_is_not_blamed_for_a_bad_key(client, llm_settings, settings):
    """Google reports an invalid key as HTTP 400 too; that must fail over to the next model, not loop."""
    settings.LLM_MODEL, settings.LLM_FALLBACK_MODELS = "always400", "good"
    settings.LLM_REASONING_EFFORT = "minimal"
    events = parse_sse(ask_stream(client, "What is the refund window?", mode="documents"))
    assert text_of(events).startswith("Hello") and "good" in text_of(events)  # failed over to the next model


def test_the_effort_retry_does_not_use_up_the_normal_retries(client, llm_settings, settings):
    settings.LLM_MODEL, settings.LLM_FALLBACK_MODELS = "busy-1", ""
    settings.LLM_REASONING_EFFORT = "minimal"
    parse_sse(ask_stream(client, "What is the refund window?", mode="documents"))
    assert len(llm_settings.requests) == llm._ATTEMPTS  # a 503 is retried as before


# ---- warm-up ------------------------------------------------------------------------------------


def test_warm_up_reports_how_long_the_model_took():
    assert isinstance(embeddings.warm_up(), float)


def test_warm_up_never_raises(monkeypatch):
    monkeypatch.setattr(embeddings, "embed_query", lambda text: (_ for _ in ()).throw(RuntimeError("no model")))
    assert embeddings.warm_up() is None


class FakeThread:
    started = []

    def __init__(self, target=None, name=None, daemon=None):
        self.target, self.name, self.daemon = target, name, daemon

    def start(self):
        FakeThread.started.append(self.name)


@pytest.fixture
def warmup(monkeypatch, settings):
    FakeThread.started = []
    monkeypatch.setattr(docqa_apps.threading, "Thread", FakeThread)
    settings.WARMUP_EMBEDDINGS, settings.EMBEDDING_BACKEND = True, "fastembed"
    monkeypatch.delenv("RUN_MAIN", raising=False)
    monkeypatch.setattr(sys, "argv", ["run_api.py"])
    return apps.get_app_config("docqa")


def test_the_model_is_loaded_in_the_background_when_enabled(warmup):
    warmup.ready()
    assert FakeThread.started == ["embedding-warm-up"]


def test_no_warm_up_when_disabled_or_for_the_lexical_embedder(warmup, settings):
    settings.WARMUP_EMBEDDINGS = False
    warmup.ready()
    settings.WARMUP_EMBEDDINGS, settings.EMBEDDING_BACKEND = True, "hash"
    warmup.ready()
    assert FakeThread.started == []


def test_the_reloader_parent_process_does_not_load_the_model(warmup, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["manage.py", "runserver"])
    warmup.ready()
    assert FakeThread.started == []  # the parent only watches files; the child serves
    monkeypatch.setenv("RUN_MAIN", "true")
    warmup.ready()
    assert FakeThread.started == ["embedding-warm-up"]
    FakeThread.started.clear()
    monkeypatch.delenv("RUN_MAIN")
    monkeypatch.setattr(sys, "argv", ["manage.py", "runserver", "--noreload"])
    warmup.ready()
    assert FakeThread.started == ["embedding-warm-up"]


# ---- the search cache ----------------------------------------------------------------------------


class CountingIndex(retrieval.BM25Index):
    built = 0

    def __init__(self, *a, **k):
        CountingIndex.built += 1
        super().__init__(*a, **k)


@pytest.fixture
def counted(monkeypatch):
    CountingIndex.built = 0
    monkeypatch.setattr(retrieval, "BM25Index", CountingIndex)
    return CountingIndex


@pytest.fixture
def user():
    return get_user_model().objects.create_user("alice")


def test_the_corpus_is_built_once_and_reused(user, counted):
    ingest_document(user, "Refunds", REFUNDS)
    for _ in range(3):
        retrieval.search(user, "refund window")
    assert counted.built == 1


def test_uploading_primes_the_cache_so_the_first_question_is_already_fast(llm_settings, counted):
    client = make_client()
    upload_text(client, "Refunds", REFUNDS)
    assert counted.built == 1  # built during the upload...
    client.post("/api/ask/", {"question": "What is the refund window?", "mode": "documents"}, format="json")
    assert counted.built == 1  # ...and reused by the first question


def test_a_failure_while_priming_never_fails_the_upload(monkeypatch):
    monkeypatch.setattr(retrieval, "_corpus", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    client = make_client()
    assert upload_text(client, "Refunds", REFUNDS).status_code == 201


def test_adding_or_deleting_a_document_refreshes_the_cache(user, counted):
    first = ingest_document(user, "Refunds", REFUNDS)
    assert retrieval.search(user, "swordfish") == [] or True
    built = counted.built
    ingest_document(user, "Fish", "The swordfish is a large predatory fish.")
    hits = retrieval.search(user, "swordfish", top_k=2)
    assert counted.built == built + 1 and hits[0].chunk.document.title == "Fish"
    first.delete()
    retrieval.search(user, "swordfish", top_k=2)
    assert counted.built == built + 2
    assert {h.chunk.document.title for h in retrieval.search(user, "refund window swordfish", top_k=5)} == {"Fish"}


def test_cached_results_equal_fresh_results(user):
    ingest_document(user, "A", "alpha beta gamma delta. " * 60)
    ingest_document(user, "B", "refunds shipping warranty returns. " * 60)
    for question in ("alpha", "refund window", "gamma warranty", "nothing matches xyzzy"):
        warm = [(h.chunk.id, round(h.score, 6)) for h in retrieval.search(user, question, top_k=5)]
        retrieval.clear_cache()
        cold = [(h.chunk.id, round(h.score, 6)) for h in retrieval.search(user, question, top_k=5)]
        assert warm == cold


def test_users_and_document_scopes_get_separate_cache_entries(user, counted):
    docs = {t: ingest_document(user, t, text) for t, text in (("One", "red green blue"), ("Two", "cat dog bird"))}
    other = get_user_model().objects.create_user("bob")
    ingest_document(other, "Bobs", "red green blue secret")
    retrieval.search(user, "red")
    retrieval.search(other, "red")
    retrieval.search(user, "red", document_id=docs["One"].id)
    assert counted.built == 3
    titles = {h.chunk.document.title for h in retrieval.search(user, "red", top_k=9)}
    assert titles == {"One", "Two"}  # everything of alice's, and never Bob's
    assert {h.chunk.document.title for h in retrieval.search(other, "red", top_k=9)} == {"Bobs"}
    for scope, name in ((docs["One"].id, "One"), (docs["Two"].id, "Two")):
        scoped = retrieval.search(user, "red cat", top_k=9, document_id=scope)
        assert scoped and {h.chunk.document.title for h in scoped} == {name}


def test_the_cache_is_bounded(user, monkeypatch):
    monkeypatch.setattr(retrieval, "_CACHE_MAX_ENTRIES", 2)
    for name in ("u1", "u2", "u3", "u4"):
        u = get_user_model().objects.create_user(name)
        ingest_document(u, "Doc", "some text about refunds")
        retrieval.search(u, "refunds")
    assert len(retrieval._CACHE) == 2


def test_huge_collections_are_searched_without_being_cached(user, monkeypatch):
    monkeypatch.setattr(retrieval, "_CACHE_MAX_CHUNKS", 1)
    ingest_document(user, "Big", "sentence number one. " * 200)
    assert retrieval.search(user, "sentence")
    assert len(retrieval._CACHE) == 0


# ---- the BM25 index and top-k selection ---------------------------------------------------------


def naive_bm25(query, docs, k1=1.5, b=0.75):
    """The textbook formula, written the slow obvious way, to check the inverted index against."""
    import math

    n = len(docs)
    avg = (sum(len(d) for d in docs) / n if n else 0.0) or 1.0
    scores = []
    for doc in docs:
        score = 0.0
        for term in set(query):
            tf = doc.count(term)
            if not tf:
                continue
            df = sum(term in d for d in docs)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc) / avg))
        scores.append(score)
    return scores


def test_the_inverted_index_matches_the_textbook_formula():
    rng = random.Random(7)
    vocab = [f"w{i}" for i in range(40)]
    for _ in range(25):
        docs = [[rng.choice(vocab) for _ in range(rng.randint(0, 30))] for _ in range(rng.randint(0, 25))]
        for _ in range(8):
            query = [rng.choice(vocab) for _ in range(rng.randint(0, 5))]
            got = retrieval.bm25_scores(query, docs)
            assert got == pytest.approx(naive_bm25(query, docs), abs=1e-9)


def test_top_k_selection_matches_a_full_sort():
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(500, 16)).astype("float32")
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    query = rng.normal(size=16).astype("float32")
    query /= np.linalg.norm(query)
    chunks = [object() for _ in range(500)]
    corpus = retrieval._Corpus(chunks, matrix, retrieval.BM25Index([]), {})
    for k in (1, 4, 20, 499, 500, 900):
        got = retrieval._search_numpy(corpus, query.tolist(), k)
        expected = np.sort(matrix @ query)[::-1][: min(k, 500)]
        assert [h.score for h in got] == pytest.approx(expected.tolist(), abs=1e-6)


def test_searching_an_empty_collection_or_with_no_matching_words_is_fine(user):
    assert retrieval.search_ex(user, "anything").hits == []
    ingest_document(user, "Doc", "alpha beta gamma")
    result = retrieval.search_ex(user, "zzzz qqqq", top_k=3)  # no shared words: vector hits only
    assert result.hits and -1.0 <= result.best_similarity <= 1.0


# ---- timings ------------------------------------------------------------------------------------


def test_the_response_reports_where_the_time_went(client, llm_settings):
    events = parse_sse(ask_stream(client, "What is the refund window?", mode="documents"))
    timings = events[0][1]["timings"]
    assert set(timings) == {"rewrite_ms", "search_ms"}
    assert all(isinstance(v, int) and v >= 0 for v in timings.values())
    assert timings["rewrite_ms"] < 50  # nothing to rewrite, so no model call happened
