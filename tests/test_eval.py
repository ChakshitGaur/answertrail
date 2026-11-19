import json

import pytest
from django.conf import settings
from django.contrib.auth import get_user_model
from django.db import connection

from docqa.services.evaluation import evaluate, routing_report
from docqa.services.ingest import ingest_document
from tests.neural import require_neural_model

pytestmark = pytest.mark.django_db

DATASET = json.loads((settings.BASE_DIR / "eval" / "dataset.json").read_text(encoding="utf-8"))
KEYWORD = [q for q in DATASET["questions"] if q["kind"] == "keyword"]
PARAPHRASE = [q for q in DATASET["questions"] if q["kind"] == "paraphrase"]


# A second, deliberately messy set: code-heavy text, a flattened table, a noisy scan (broken hyphenation), a chat
# export, a resume with a section split across passages, and meeting minutes.
MESSY = json.loads((settings.BASE_DIR / "eval" / "messy.json").read_text(encoding="utf-8"))


def load_corpus(backend, settings_fixture, dataset=DATASET):
    settings_fixture.EMBEDDING_BACKEND = backend
    user, _ = get_user_model().objects.get_or_create(username="eval-user")
    user.documents.all().delete()
    for doc in dataset["documents"]:
        ingest_document(user, doc["title"], doc["text"])
    return user


@pytest.mark.parametrize("dataset", [DATASET, MESSY], ids=["clean", "messy"])
def test_every_evidence_span_exists_in_exactly_one_document(dataset):
    """Guards the dataset itself: labels must be answerable and unambiguous."""
    texts = {d["title"]: " ".join(d["text"].lower().split()) for d in dataset["documents"]}
    for item in dataset["questions"]:
        owners = [t for t, text in texts.items() if item["evidence"].lower() in text]
        assert len(owners) == 1, (item["q"], owners)


def test_messy_dataset_is_not_trivially_small():
    kinds = {q["kind"] for q in MESSY["questions"]}
    assert kinds == {"keyword", "paraphrase"} and len(MESSY["questions"]) >= 25 and len(MESSY["off_topic"]) >= 10


def test_messy_documents_are_harder_than_the_clean_ones_for_the_weak_embedder(settings):
    clean = evaluate(load_corpus("hash", settings), DATASET["questions"], [1, 5], hybrid=True)
    messy = evaluate(load_corpus("hash", settings, MESSY), MESSY["questions"], [1, 5], hybrid=True)
    assert messy.hit_rate[1] < clean.hit_rate[1]  # measured 0.621 vs 0.766: the set is doing its job


def test_messy_set_hash_baseline_does_not_regress(settings):
    user = load_corpus("hash", settings, MESSY)
    vector = evaluate(user, MESSY["questions"], [1, 3, 5], hybrid=False)
    hybrid = evaluate(user, MESSY["questions"], [1, 3, 5], hybrid=True)
    assert hybrid.hit_rate[5] >= 0.85  # measured 0.931 (misses: "cat" vs "animals", "discount" vs "free months")
    assert hybrid.mrr > vector.mrr + 0.05  # measured 0.639 -> 0.761


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
    require_neural_model()
    base_user = load_corpus("hash", settings)
    baseline = evaluate(base_user, DATASET["questions"], [1, 5], hybrid=False)
    neural_user = load_corpus("fastembed", settings)
    neural = evaluate(neural_user, DATASET["questions"], [1, 5], hybrid=False)
    assert neural.hit_rate[5] >= 0.95  # measured 0.979
    assert neural.mrr > baseline.mrr + 0.25  # measured 0.936 vs 0.648


def test_hybrid_fixes_keyword_questions_for_the_neural_model(settings):
    require_neural_model()
    user = load_corpus("fastembed", settings)
    vector = evaluate(user, KEYWORD, [1, 5], hybrid=False)
    hybrid = evaluate(user, KEYWORD, [1, 5], hybrid=True)
    assert hybrid.mrr > vector.mrr  # measured 0.917 -> 1.000
    assert hybrid.hit_rate[1] >= 0.9
    # and it must not wreck paraphrased questions (measured 0.914 -> 0.886 hit-rate@1)
    assert evaluate(user, PARAPHRASE, [1, 5], hybrid=True).hit_rate[5] >= 0.95


def test_neural_similarity_separates_documents_questions_from_unrelated_ones(settings):
    require_neural_model()
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
