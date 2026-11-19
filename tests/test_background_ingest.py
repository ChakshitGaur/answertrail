import time

import pytest
from django.contrib.auth import get_user_model
from rest_framework.authtoken.models import Token
from rest_framework.test import APIClient

from docqa.models import Chunk, Document
from tests.pdfs import blank_pdf

TEXT = "Refunds are accepted within 30 days of purchase. Shipping takes five business days."


def _client(username="bg"):
    user = get_user_model().objects.create_user(username, password="pw12345678x")
    c = APIClient()
    c.credentials(HTTP_AUTHORIZATION="Token " + Token.objects.create(user=user).key)
    return c, user


def _wait_for(doc_id, status, seconds=10):
    deadline = time.time() + seconds
    while time.time() < deadline:
        doc = Document.objects.get(pk=doc_id)
        if doc.status == status:
            return doc
        time.sleep(0.05)
    raise AssertionError(f"document stayed {Document.objects.get(pk=doc_id).status!r}, expected {status!r}")


@pytest.mark.django_db(transaction=True)
def test_upload_returns_at_once_and_is_searchable_when_ready(settings):
    settings.INGEST_ASYNC = True
    c, _ = _client()
    resp = c.post("/api/documents/", {"title": "Policy", "text": TEXT}, format="json")
    assert resp.status_code == 202 and resp.json()["status"] in ("processing", "ready")
    doc = _wait_for(resp.json()["id"], "ready")
    assert Chunk.objects.filter(document=doc).count() >= 1
    listed = c.get("/api/documents/").json()
    assert listed[0]["status"] == "ready" and listed[0]["chunk_count"] >= 1
    answer = c.post("/api/ask/", {"question": "refunds within 30 days"}).json()
    assert answer["sources"] and answer["sources"][0]["document_title"] == "Policy"


@pytest.mark.django_db(transaction=True)
def test_failed_indexing_is_reported_not_left_processing(settings, monkeypatch):
    from docqa.services import ingest

    settings.INGEST_ASYNC = True

    def boom(texts):
        raise RuntimeError("embedding service down")

    monkeypatch.setattr(ingest, "embed_passages", boom)
    c, _ = _client("bg2")
    resp = c.post("/api/documents/", {"title": "x", "text": TEXT}, format="json")
    doc = _wait_for(resp.json()["id"], "failed")
    assert "embedding service down" in doc.error
    assert Chunk.objects.filter(document=doc).count() == 0


@pytest.mark.django_db(transaction=True)
def test_document_deleted_while_waiting_is_skipped(settings):
    from docqa.services import ingest

    settings.INGEST_ASYNC = True
    c, user = _client("bg3")
    doc = Document.objects.create(owner=user, title="gone", status="processing")
    doc.delete()
    ingest._index_in_background(doc.pk, ["some text here"], user.pk)  # must not raise
    assert not Chunk.objects.exists()


@pytest.mark.django_db
def test_stale_processing_document_is_marked_failed(settings):
    settings.INGEST_TIMEOUT_SECONDS = 60
    c, user = _client("bg4")
    doc = Document.objects.create(owner=user, title="stuck", status="processing")
    Document.objects.filter(pk=doc.pk).update(created_at=doc.created_at - __import__("datetime").timedelta(minutes=5))
    listed = c.get("/api/documents/").json()
    assert listed[0]["status"] == "failed" and "interrupted" in listed[0]["error"].lower()


@pytest.mark.django_db
def test_text_that_cannot_be_indexed_is_refused_before_any_background_work(settings):
    settings.INGEST_ASYNC = True
    c, _ = _client("bg5")
    assert c.post("/api/documents/", {"title": "p", "text": "!!! ???"}, format="json").status_code == 400
    assert not Document.objects.exists()


@pytest.mark.django_db
def test_scanned_pdf_gets_an_explanation():
    c, _ = _client("bg6")
    from django.core.files.uploadedfile import SimpleUploadedFile

    resp = c.post("/api/documents/", {"file": SimpleUploadedFile("scan.pdf", blank_pdf())}, format="multipart")
    assert resp.status_code == 400 and "scan" in resp.json()["detail"].lower()


@pytest.mark.django_db
def test_questions_have_their_own_lower_rate_limit(settings):
    c, _ = _client("bg7")
    codes = [c.post("/api/ask/", {"question": "hello"}, format="json").status_code for _ in range(33)]
    assert 429 in codes and codes[0] != 429
