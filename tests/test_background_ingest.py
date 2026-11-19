import time

import pytest
from django.contrib.auth import get_user_model
from django.db import connection
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
    settings.INGEST_MAX_ATTEMPTS = 1

    def boom(texts):
        raise RuntimeError("embedding service down")

    monkeypatch.setattr(ingest, "embed_passages", boom)
    c, _ = _client("bg2")
    resp = c.post("/api/documents/", {"title": "x", "text": TEXT}, format="json")
    doc = _wait_for(resp.json()["id"], "failed")
    assert "embedding service down" in doc.error
    assert Chunk.objects.filter(document=doc).count() == 0


@pytest.mark.django_db(transaction=True)
def test_a_temporary_failure_is_retried_and_then_succeeds(settings, monkeypatch):
    from docqa.services import ingest

    settings.INGEST_ASYNC = True
    settings.INGEST_EMBEDDED_WORKER = False  # this test is the worker
    monkeypatch.setattr(ingest, "_RETRY_DELAY_SECONDS", 0)  # retry at once
    real, calls = ingest.embed_passages, []

    def flaky(texts):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("temporary hiccup")
        return real(texts)

    monkeypatch.setattr(ingest, "embed_passages", flaky)
    c, user = _client("bg2b")
    doc_id = c.post("/api/documents/", {"title": "x", "text": TEXT}, format="json").json()["id"]
    assert ingest.run_pending() == 2  # first attempt failed, the retry worked
    doc = Document.objects.get(pk=doc_id)
    assert doc.status == "ready" and doc.attempts == 2 and doc.error == ""
    assert Chunk.objects.filter(document=doc).exists()


@pytest.mark.django_db(transaction=True)
def test_document_deleted_while_waiting_is_skipped(settings):
    from docqa.services import ingest

    settings.INGEST_ASYNC = True
    settings.INGEST_EMBEDDED_WORKER = False
    c, user = _client("bg3")
    doc_id = c.post("/api/documents/", {"title": "gone", "text": TEXT}, format="json").json()["id"]
    c.delete(f"/api/documents/{doc_id}/")
    assert ingest.run_pending() == 0  # nothing left to do, and no error
    assert not Chunk.objects.exists()


@pytest.mark.django_db
def test_work_survives_a_restart(settings):
    """A document whose worker died (its claim ran out) is picked up again, with no one having to re-upload it."""
    import datetime

    from django.utils import timezone

    from docqa.services import ingest

    settings.INGEST_LEASE_SECONDS = 60
    c, user = _client("bg4")
    doc = Document.objects.create(
        owner=user, title="orphan", status="processing", source_text=TEXT, reindexable=True,
        processing_since=timezone.now(), claimed_at=timezone.now() - datetime.timedelta(minutes=5), attempts=1,
    )
    assert ingest.run_pending() == 1
    doc.refresh_from_db()
    assert doc.status == "ready" and Chunk.objects.filter(document=doc).exists()


@pytest.mark.django_db
def test_a_document_being_worked_on_is_not_taken_by_a_second_worker(settings):
    from django.utils import timezone

    from docqa.services import ingest

    settings.INGEST_LEASE_SECONDS = 600
    c, user = _client("bg4b")
    Document.objects.create(
        owner=user, title="busy", status="processing", source_text=TEXT, reindexable=True,
        processing_since=timezone.now(), claimed_at=timezone.now(),
    )
    assert ingest.claim_next() is None


@pytest.mark.django_db
def test_waiting_documents_are_taken_oldest_first_and_only_once(settings):
    import datetime

    from django.utils import timezone

    from docqa.services import ingest

    c, user = _client("bg4c")
    now = timezone.now()
    old = Document.objects.create(owner=user, title="old", status="processing", source_text=TEXT,
                                  processing_since=now - datetime.timedelta(minutes=2))
    new = Document.objects.create(owner=user, title="new", status="processing", source_text=TEXT,
                                  processing_since=now)
    first, second, third = ingest.claim_next(), ingest.claim_next(), ingest.claim_next()
    assert (first.pk, second.pk, third) == (old.pk, new.pk, None)


@pytest.mark.django_db
def test_a_document_that_keeps_failing_is_given_up_on(settings):
    from django.utils import timezone

    from docqa.services import ingest

    settings.INGEST_MAX_ATTEMPTS = 3
    settings.INGEST_LEASE_SECONDS = 60
    c, user = _client("bg4d")
    doc = Document.objects.create(
        owner=user, title="poison", status="processing", source_text=TEXT, reindexable=True,
        processing_since=timezone.now(), attempts=3,  # three workers already died on it
    )
    assert ingest.run_pending() == 0
    doc.refresh_from_db()
    assert doc.status == "failed" and "3 attempts" in doc.error


@pytest.mark.skipif(connection.vendor != "postgresql", reason="needs a database that allows concurrent writers")
@pytest.mark.django_db(transaction=True)
def test_two_workers_never_index_the_same_document_twice(settings, monkeypatch):
    import threading

    from docqa.services import ingest

    settings.INGEST_ASYNC = True
    settings.INGEST_EMBEDDED_WORKER = False
    c, user = _client("bg4e")
    for n in range(6):
        c.post("/api/documents/", {"title": f"d{n}", "text": TEXT}, format="json")
    counts = []

    def work():
        counts.append(ingest.run_pending())

    threads = [threading.Thread(target=work) for _ in range(3)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sum(counts) == 6
    assert set(Document.objects.values_list("attempts", flat=True)) == {1}  # each was claimed exactly once
    assert not Document.objects.exclude(status="ready").exists()


@pytest.mark.django_db(transaction=True)
def test_the_embedded_worker_can_be_turned_off(settings):
    from docqa.services import ingest

    settings.INGEST_ASYNC = True
    settings.INGEST_EMBEDDED_WORKER = False
    c, _ = _client("bg4f")
    doc_id = c.post("/api/documents/", {"title": "x", "text": TEXT}, format="json").json()["id"]
    import time

    time.sleep(0.5)
    assert Document.objects.get(pk=doc_id).status == "processing"  # waits for run_ingest_worker
    from django.core.management import call_command

    call_command("run_ingest_worker", "--once")
    assert Document.objects.get(pk=doc_id).status == "ready"
    assert ingest._worker is None


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
