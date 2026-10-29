import json

import pytest
from django.conf import settings
from django.contrib.auth import get_user_model
from django.db import connection

from docqa.services.evaluation import evaluate
from docqa.services.ingest import ingest_document

pytestmark = pytest.mark.django_db

DATASET = json.loads((settings.BASE_DIR / "eval" / "dataset.json").read_text(encoding="utf-8"))


def run_eval(backend, settings_fixture):
    settings_fixture.EMBEDDING_BACKEND = backend
    user, _ = get_user_model().objects.get_or_create(username="eval-user")
    user.documents.all().delete()
    for doc in DATASET["documents"]:
        ingest_document(user, doc["title"], doc["text"])
    return evaluate(user, DATASET["questions"], [1, 3, 5])


def test_every_evidence_span_exists_in_exactly_one_document():
    """Guards the dataset itself: labels must be answerable and unambiguous."""
    texts = {d["title"]: " ".join(d["text"].lower().split()) for d in DATASET["documents"]}
    for item in DATASET["questions"]:
        owners = [t for t, text in texts.items() if item["evidence"].lower() in text]
        assert len(owners) == 1, (item["q"], owners)


def test_hash_baseline_does_not_regress(settings):
    result = run_eval("hash", settings)
    assert result.hit_rate[5] >= 0.75  # measured 0.829; floor leaves headroom for tweaks


def test_neural_model_beats_baseline_by_a_wide_margin(settings):
    pytest.importorskip("fastembed")
    baseline = run_eval("hash", settings)
    neural = run_eval("fastembed", settings)
    assert neural.hit_rate[5] >= 0.95  # measured 0.971
    assert neural.mrr > baseline.mrr + 0.3


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
