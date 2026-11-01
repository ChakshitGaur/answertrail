import numpy as np
import pytest
from django.contrib.auth import get_user_model

from docqa.models import Chunk
from docqa.services import retrieval
from docqa.services.embeddings import embed_query
from docqa.services.ingest import ingest_document
from docqa.services.retrieval import bm25_scores, search, search_ex

pytestmark = pytest.mark.django_db


@pytest.fixture
def user():
    return get_user_model().objects.create_user("alice")


def make_docs(user, **docs):
    return {title: ingest_document(user, title, text) for title, text in docs.items()}


# ---- BM25 -----------------------------------------------------------------------------------


def test_bm25_rewards_rare_terms_and_ignores_missing_ones():
    docs = [["the", "refund", "policy"], ["error", "zx4417", "occurred"], ["the", "weather", "today"]]
    scores = bm25_scores(["zx4417"], docs)
    assert scores[1] > 0 and scores[0] == scores[2] == 0
    assert bm25_scores([], docs) == [0.0, 0.0, 0.0] and bm25_scores(["x"], []) == []


def test_bm25_prefers_the_shorter_document_for_the_same_term_count():
    scores = bm25_scores(["refund"], [["refund"], ["refund"] + ["filler"] * 30])
    assert scores[0] > scores[1]


def test_rare_terms_outweigh_common_ones():
    docs = [["common", "common2"], ["common", "rare"], ["common", "x"], ["common", "y"]]
    scores = bm25_scores(["common", "rare"], docs)
    assert scores.index(max(scores)) == 1


# ---- fusion ---------------------------------------------------------------------------------


def test_fusion_promotes_items_that_both_rankings_like():
    assert retrieval._fuse([[1, 2, 3], [3, 4, 1]])[:2] == [1, 3]  # 1 and 3 appear in both lists


def test_fusion_keeps_items_found_by_only_one_ranking():
    fused = retrieval._fuse([[1, 2], [9]])
    assert set(fused) == {1, 2, 9}


def test_fusion_of_nothing_is_nothing():
    assert retrieval._fuse([[], []]) == []


# ---- hybrid search --------------------------------------------------------------------------


def test_search_returns_cosine_scores_on_one_scale(user):
    make_docs(user, Refunds="Refunds are accepted within 30 days.", Plants="Photosynthesis happens in leaves.")
    for hybrid in (True, False):
        hits = search(user, "refund window", top_k=2, hybrid=hybrid)
        assert hits and all(-1.0 <= h.score <= 1.0 for h in hits)
        assert hits[0].chunk.document.title == "Refunds"


def test_best_similarity_is_the_closest_chunks_cosine_even_after_fusion(user):
    make_docs(user, Refunds="Refunds are accepted within 30 days.", Plants="Photosynthesis happens in leaves.")
    result = search_ex(user, "refund window", top_k=1, hybrid=True)
    query = np.array(embed_query("refund window"), dtype=np.float32)
    closest = max(float(np.array(c.embedding, dtype=np.float32) @ query) for c in Chunk.objects.all())
    assert result.best_similarity == pytest.approx(closest, abs=1e-5)


def test_keyword_only_hits_are_scored_on_the_cosine_scale(user, monkeypatch):
    """A chunk the vector search missed but BM25 found must still carry a real cosine score."""
    docs = make_docs(user, Codes="Error ZX4417 means the pump stalled.", Other="Nothing about that here at all.")
    other_hit = retrieval.Hit(chunk=docs["Other"].chunks.first(), score=0.3)
    monkeypatch.setattr(retrieval, "_vector_hits", lambda chunks, key, query, k: [other_hit])
    hits = search(user, "ZX4417", top_k=2, hybrid=True)
    codes = next(h for h in hits if h.chunk.document.title == "Codes")
    expected = float(np.array(codes.chunk.embedding, dtype=np.float32) @ np.array(embed_query("ZX4417"), dtype=np.float32))
    assert codes.score == pytest.approx(expected, abs=1e-5)
    # each is ranked first by one of the two searches, so fusion keeps both and does not drop the keyword hit
    assert {h.chunk.document.title for h in hits} == {"Codes", "Other"}


def test_top_k_is_respected_and_empty_corpus_is_handled(user):
    assert search_ex(user, "anything") == retrieval.SearchResult([], 0.0)
    make_docs(user, A="alpha beta gamma. " * 80)  # several chunks
    assert len(search(user, "alpha", top_k=2)) == 2
    assert len(search(user, "alpha", top_k=1)) == 1


def test_search_never_crosses_users_or_document_scope(user):
    bob = get_user_model().objects.create_user("bob")
    docs = make_docs(user, Mine="The secret launch code is swordfish.", Also="Another note about swordfish soup.")
    make_docs(bob, Bobs="Bob's swordfish recipe.")
    titles = {h.chunk.document.title for h in search(user, "swordfish", top_k=10)}
    assert titles == {"Mine", "Also"}
    scoped = search(user, "swordfish", top_k=10, document_id=docs["Also"].id)
    assert {h.chunk.document.title for h in scoped} == {"Also"}
    assert search(bob, "launch code", top_k=10)[0].chunk.document.title == "Bobs"


def test_hybrid_flag_in_settings_controls_the_default(user, settings, monkeypatch):
    make_docs(user, A="alpha beta")
    calls = []
    monkeypatch.setattr(retrieval, "_keyword_hits", lambda *a, **k: calls.append(1) or [])
    settings.HYBRID_SEARCH = False
    search(user, "alpha")
    assert calls == []
    settings.HYBRID_SEARCH = True
    search(user, "alpha")
    assert calls == [1]
