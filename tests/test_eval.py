import json

import pytest
from django.conf import settings
from django.contrib.auth import get_user_model
from django.db import connection

from docqa.services.evaluation import evaluate, routing_report
from docqa.services.ingest import ingest_document

pytestmark = pytest.mark.django_db

DATASET = json.loads((settings.BASE_DIR / "eval" / "dataset.json").read_text(encoding="utf-8"))
KEYWORD = [q for q in DATASET["questions"] if q["kind"] == "keyword"]
PARAPHRASE = [q for q in DATASET["questions"] if q["kind"] == "paraphrase"]


def load_corpus(backend, settings_fixture):
    settings_fixture.EMBEDDING_BACKEND = backend
    user, _ = get_user_model().objects.get_or_create(username="eval-user")
    user.documents.all().delete()
    for doc in DATASET["documents"]:
        ingest_document(user, doc["title"], doc["text"])
    return user


def test_every_evidence_span_exists_in_exactly_one_document():
    """Guards the dataset itself: labels must be answerable and unambiguous."""
    texts = {d["title"]: " ".join(d["text"].lower().split()) for d in DATASET["documents"]}
    for item in DATASET["questions"]:
        owners = [t for t, text in texts.items() if item["evidence"].lower() in text]
        assert len(owners) == 1, (item["q"], owners)


def test_dataset_has_both_question_kinds_and_off_topic_questions():
    assert len(PARAPHRASE) >= 30 and len(KEYWORD) >= 10 and len(DATASET["off_topic"]) >= 10


def test_hash_baseline_does_not_regress(settings):
    user = load_corpus("hash", settings)
    vector = evaluate(user, DATASET["questions"], [1, 3, 5], hybrid=False)
    assert vector.hit_rate[5] >= 0.80  # measured 0.872
    hybrid = evaluate(user, DATASET["questions"], [1, 3, 5], hybrid=True)
    assert hybrid.hit_rate[5] >= 0.95  # measured 0.979


def test_hybrid_search_lifts_the_weak_embedder_a_lot(settings):
    user = load_corpus("hash", settings)
    vector = evaluate(user, DATASET["questions"], [1, 5], hybrid=False)
    hybrid = evaluate(user, DATASET["questions"], [1, 5], hybrid=True)
    assert hybrid.mrr > vector.mrr + 0.1  # measured 0.648 -> 0.869


def test_neural_model_beats_the_lexical_baseline(settings):
    pytest.importorskip("fastembed")
    base_user = load_corpus("hash", settings)
    baseline = evaluate(base_user, DATASET["questions"], [1, 5], hybrid=False)
    neural_user = load_corpus("fastembed", settings)
    neural = evaluate(neural_user, DATASET["questions"], [1, 5], hybrid=False)
    assert neural.hit_rate[5] >= 0.95  # measured 0.979
    assert neural.mrr > baseline.mrr + 0.25  # measured 0.936 vs 0.648


def test_hybrid_fixes_keyword_questions_for_the_neural_model(settings):
    pytest.importorskip("fastembed")
    user = load_corpus("fastembed", settings)
    vector = evaluate(user, KEYWORD, [1, 5], hybrid=False)
    hybrid = evaluate(user, KEYWORD, [1, 5], hybrid=True)
    assert hybrid.mrr > vector.mrr  # measured 0.917 -> 1.000
    assert hybrid.hit_rate[1] >= 0.9
    # and it must not wreck paraphrased questions (measured 0.914 -> 0.886 hit-rate@1)
    assert evaluate(user, PARAPHRASE, [1, 5], hybrid=True).hit_rate[5] >= 0.95


def test_neural_similarity_separates_documents_questions_from_unrelated_ones(settings):
    pytest.importorskip("fastembed")
    from docqa.services.routing import relevance_threshold

    settings.RELEVANCE_THRESHOLD = None  # use the shipped default
    user = load_corpus("fastembed", settings)
    report = routing_report(user, [q["q"] for q in DATASET["questions"]], DATASET["off_topic"])
    assert report.on_topic_recall == 1.0 and report.off_topic_rejection == 1.0  # measured on this set
    assert min(report.on_topic_scores) > max(report.off_topic_scores)
    # the shipped cutoff must sit inside the gap, otherwise it needs recalibrating
    assert max(report.off_topic_scores) < relevance_threshold() < min(report.on_topic_scores)


@pytest.mark.skipif(connection.vendor != "postgresql", reason="pgvector needs PostgreSQL")
def test_pgvector_column_and_hnsw_index_exist():
    with connection.cursor() as cur:
        cur.execute(
            "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
            "WHERE attrelid = 'docqa_chunk'::regclass AND attname = 'embedding'"
        )
        assert cur.fetchone()[0] == f"vector({settings.EMBEDDING_DIM})"
        cur.execute(
            "SELECT indexdef FROM pg_indexes WHERE indexname = 'docqa_chunk_embedding_hnsw'"
        )
        assert "hnsw" in cur.fetchone()[0]
