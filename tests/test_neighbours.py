"""Neighbouring chunks: a section cut across two chunks is not answered from half of it."""

import pytest
from django.contrib.auth import get_user_model
from django.conf import settings as django_settings

from docqa.models import Chunk, Document
from docqa.services import retrieval, routing
from docqa.services.retrieval import Hit, SearchResult, add_neighbours

pytestmark = pytest.mark.django_db


def vec(cosine: float) -> list[float]:
    """A unit vector whose cosine with QUERY is exactly `cosine`."""
    v = [0.0] * django_settings.EMBEDDING_DIM
    v[0], v[1] = cosine, (1 - cosine * cosine) ** 0.5
    return v


QUERY = vec(1.0)


@pytest.fixture
def user():
    return get_user_model().objects.create_user("alice")


def make_doc(owner, title, cosines):
    doc = Document.objects.create(owner=owner, title=title)
    Chunk.objects.bulk_create(
        Chunk(document=doc, index=i, text=f"{title} chunk {i}", embedding=vec(c)) for i, c in enumerate(cosines)
    )
    return doc


def hit(doc, index, score):
    chunk = Chunk.objects.select_related("document").get(document=doc, index=index)
    return Hit(chunk=chunk, score=score)


def result(*hits):
    return SearchResult(list(hits), max(h.score for h in hits), QUERY)


def order(hits):
    return [(h.chunk.document.title, h.chunk.index) for h in hits]


def test_the_chunk_after_a_hit_is_added_beside_it(user):
    doc = make_doc(user, "resume", [0.1, 0.5, 0.7, 0.1])
    out = add_neighbours(result(hit(doc, 1, 0.5)), limit=2)
    assert order(out) == [("resume", 1), ("resume", 2)]
    assert out[1].score == pytest.approx(0.7, abs=1e-4)  # scored on the same cosine scale as the hits


def test_the_chunk_before_a_hit_is_placed_in_front_of_it(user):
    doc = make_doc(user, "resume", [0.1, 0.8, 0.4, 0.1])
    assert order(add_neighbours(result(hit(doc, 2, 0.4)), limit=2)) == [("resume", 1), ("resume", 2)]


def test_a_neighbour_weaker_than_the_weakest_hit_is_filler_and_is_dropped(user):
    doc = make_doc(user, "guide", [0.1, 0.9, 0.3, 0.1])
    out = add_neighbours(result(hit(doc, 1, 0.9)), limit=2)
    assert order(out) == [("guide", 1)]


def test_the_bar_is_the_weakest_hit_not_the_best_one(user):
    guide = make_doc(user, "guide", [0.1, 0.9, 0.1])
    resume = make_doc(user, "resume", [0.1, 0.4, 0.55, 0.1])
    out = add_neighbours(result(hit(guide, 1, 0.9), hit(resume, 1, 0.4)), limit=2)
    assert order(out) == [("guide", 1), ("resume", 1), ("resume", 2)]  # 0.55 beats the weak 0.4 hit


def test_edges_of_a_document_have_only_one_neighbour(user):
    doc = make_doc(user, "d", [0.5, 0.6, 0.6, 0.5])
    first = add_neighbours(result(hit(doc, 0, 0.5)), limit=4)
    last = add_neighbours(result(hit(doc, 3, 0.5)), limit=4)
    assert order(first) == [("d", 0), ("d", 1)]  # nothing before chunk 0, and no chunk -1
    assert order(last) == [("d", 2), ("d", 3)]  # nothing after the last chunk


def test_a_one_chunk_document_has_no_neighbours(user):
    doc = make_doc(user, "tiny", [0.5])
    assert order(add_neighbours(result(hit(doc, 0, 0.5)), limit=4)) == [("tiny", 0)]


def test_a_neighbour_that_is_already_a_hit_is_not_repeated(user):
    doc = make_doc(user, "d", [0.7, 0.5, 0.5, 0.7])
    out = add_neighbours(result(hit(doc, 1, 0.5), hit(doc, 2, 0.5)), limit=4)
    assert order(out) == [("d", 0), ("d", 1), ("d", 2), ("d", 3)]  # 1 and 2 once each


def test_only_the_best_matching_neighbours_are_kept_up_to_the_limit(user):
    doc = make_doc(user, "d", [0.9, 0.5, 0.6, 0.5, 0.9])
    hits = result(hit(doc, 1, 0.5), hit(doc, 3, 0.5))
    assert order(add_neighbours(hits, limit=1)) == [("d", 0), ("d", 1), ("d", 3)]  # 0.9 beats 0.6; ties keep document order
    assert order(add_neighbours(hits, limit=3)) == [("d", 0), ("d", 1), ("d", 2), ("d", 3), ("d", 4)]


def test_a_limit_of_zero_or_no_query_changes_nothing(user):
    doc = make_doc(user, "d", [0.1, 0.5, 0.9])
    base = result(hit(doc, 1, 0.5))
    assert add_neighbours(base, limit=0) is base.hits
    assert add_neighbours(SearchResult(base.hits, 0.5, None), limit=2) is base.hits
    assert add_neighbours(SearchResult([], 0.0, QUERY), limit=2) == []


def test_neighbours_come_from_the_same_document_only(user):
    a = make_doc(user, "a", [0.1, 0.5, 0.1])
    b = make_doc(user, "b", [0.1, 0.5, 0.9])
    out = add_neighbours(result(hit(a, 1, 0.5)), limit=4)
    assert all(h.chunk.document_id == a.id for h in out)  # b's strong chunk 2 is not a neighbour of a's hit
    assert b.chunks.count() == 3


def test_other_users_chunks_are_never_added(user):
    other = get_user_model().objects.create_user("mallory")
    mine = make_doc(user, "mine", [0.1, 0.5, 0.1])
    make_doc(other, "theirs", [0.1, 0.99, 0.99])
    assert order(add_neighbours(result(hit(mine, 1, 0.5)), limit=4)) == [("mine", 1)]


def test_the_original_hits_keep_their_scores_and_order(user):
    doc = make_doc(user, "d", [0.2, 0.6, 0.8, 0.2, 0.7, 0.9])
    hits = [hit(doc, 1, 0.6), hit(doc, 4, 0.7)]
    out = add_neighbours(SearchResult(hits, 0.7, QUERY), limit=2)
    assert [h for h in out if h in hits] == hits  # same objects, same relative order


# ---- in the plan -------------------------------------------------------------------------------------


def seeded(monkeypatch, doc, hits, best):
    monkeypatch.setattr(retrieval, "search_ex", lambda *a, **k: SearchResult(hits, best, QUERY))


def test_the_plan_shows_and_sends_the_neighbouring_passage(user, monkeypatch, settings):
    doc = make_doc(user, "resume", [0.1, 0.5, 0.7, 0.1])
    seeded(monkeypatch, doc, [hit(doc, 1, 0.5)], 0.5)
    settings.NEIGHBOUR_PASSAGES = 2
    plan = routing.plan_answer(user, "education?", "documents", [], 4, None)
    assert [s["chunk_index"] for s in plan.sources] == [1, 2]
    assert plan.passages == ["resume chunk 1", "resume chunk 2"]
    assert plan.best_similarity == 0.5  # the relevance decision is unchanged by neighbours


def test_neighbours_can_be_switched_off(user, monkeypatch, settings):
    doc = make_doc(user, "resume", [0.1, 0.5, 0.7, 0.1])
    seeded(monkeypatch, doc, [hit(doc, 1, 0.5)], 0.5)
    settings.NEIGHBOUR_PASSAGES = 0
    plan = routing.plan_answer(user, "education?", "documents", [], 4, None)
    assert [s["chunk_index"] for s in plan.sources] == [1]


def test_neighbours_do_not_rescue_a_question_the_documents_do_not_cover(user, monkeypatch, settings):
    doc = make_doc(user, "resume", [0.1, 0.5, 0.7, 0.1])
    seeded(monkeypatch, doc, [hit(doc, 1, 0.5)], 0.5)
    settings.RELEVANCE_THRESHOLD = 0.9  # nothing is similar enough
    settings.NEIGHBOUR_PASSAGES = 2
    plan = routing.plan_answer(user, "education?", "documents", [], 4, None)
    assert plan.mode == "not_found" and plan.sources == []


def test_search_results_carry_the_question_vector(user):
    from docqa.services.ingest import ingest_document

    ingest_document(user, "Refunds", "Refunds are accepted within 30 days of purchase.")
    found = retrieval.search_ex(user, "refund window", 4)
    assert found.query is not None and len(found.query) == django_settings.EMBEDDING_DIM
    assert retrieval.search_ex(user, "refund window", 4, hybrid=False).query is not None
