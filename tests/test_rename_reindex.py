import pytest

from docqa.models import Chunk, Document
from tests.test_api import make_client
from tests.test_api import upload_text as _upload


def upload_text(client, title, text):
    resp = _upload(client, title, text)
    assert resp.status_code == 201
    return resp.json()

pytestmark = pytest.mark.django_db

TEXT = "Refunds are accepted within 30 days of purchase. Shipping takes five business days. " * 12


def test_rename_a_document():
    c = make_client()
    doc = upload_text(c, "Old name", TEXT)
    resp = c.patch(f"/api/documents/{doc['id']}/", {"title": "  New name "}, format="json")
    assert resp.status_code == 200 and resp.json()["title"] == "New name"
    assert c.get(f"/api/documents/{doc['id']}/").json()["title"] == "New name"
    assert c.patch(f"/api/documents/{doc['id']}/", {"title": "   "}, format="json").status_code == 400
    assert c.patch(f"/api/documents/{doc['id']}/", {"title": "x" * 256}, format="json").status_code == 400
    assert c.patch(f"/api/documents/{doc['id']}/", {"title": "a\x00b"}, format="json").status_code == 400


def test_nobody_else_can_rename_or_reindex():
    a, b = make_client("alice"), make_client("bob")
    doc = upload_text(a, "Mine", TEXT)
    assert b.patch(f"/api/documents/{doc['id']}/", {"title": "Hacked"}, format="json").status_code == 404
    assert b.post(f"/api/documents/{doc['id']}/reindex/").status_code == 404
    assert Document.objects.get(pk=doc["id"]).title == "Mine"


def test_reindex_uses_the_current_chunk_size(settings):
    settings.CHUNK_SIZE, settings.CHUNK_OVERLAP = 500, 100
    c = make_client()
    doc = upload_text(c, "Policy", TEXT)
    before = c.get(f"/api/documents/{doc['id']}/").json()["chunk_count"]
    old_ids = set(Chunk.objects.filter(document_id=doc["id"]).values_list("id", flat=True))
    settings.CHUNK_SIZE, settings.CHUNK_OVERLAP = 150, 20
    resp = c.post(f"/api/documents/{doc['id']}/reindex/")
    assert resp.status_code == 200 and resp.json()["status"] == "ready"
    after = c.get(f"/api/documents/{doc['id']}/").json()["chunk_count"]
    assert after > before
    assert not old_ids & set(Chunk.objects.filter(document_id=doc["id"]).values_list("id", flat=True))
    answer = c.post("/api/ask/", {"question": "refunds within 30 days"}).json()
    assert answer["sources"]  # still searchable with the new passages


def test_documents_without_stored_text_cannot_be_reindexed():
    c = make_client()
    doc = upload_text(c, "Old", TEXT)
    Document.objects.filter(pk=doc["id"]).update(source_text="", reindexable=False)  # as added before this feature
    assert c.get(f"/api/documents/{doc['id']}/").json()["reindexable"] is False
    resp = c.post(f"/api/documents/{doc['id']}/reindex/")
    assert resp.status_code == 400 and "upload it again" in resp.json()["detail"].lower()
    assert c.patch(f"/api/documents/{doc['id']}/", {"title": "Still renamable"}, format="json").status_code == 200


def test_cannot_reindex_while_processing():
    c = make_client()
    doc = upload_text(c, "Busy", TEXT)
    Document.objects.filter(pk=doc["id"]).update(status="processing")
    assert c.post(f"/api/documents/{doc['id']}/reindex/").status_code == 409


def test_source_text_is_never_returned_by_the_api():
    c = make_client()
    doc = upload_text(c, "Policy", TEXT)
    assert "source_text" not in c.get("/api/documents/").json()[0]
    assert "source_text" not in c.get(f"/api/documents/{doc['id']}/").json()


@pytest.mark.django_db(transaction=True)
def test_background_reindex_keeps_old_passages_until_the_new_ones_are_ready(settings, monkeypatch):
    from docqa.services import ingest

    c = make_client()
    doc = upload_text(c, "Policy", TEXT)
    old = set(Chunk.objects.filter(document_id=doc["id"]).values_list("id", flat=True))
    settings.INGEST_ASYNC = True
    settings.INGEST_MAX_ATTEMPTS = 1

    def boom(texts):
        raise RuntimeError("embedding service down")

    monkeypatch.setattr(ingest, "embed_passages", boom)
    resp = c.post(f"/api/documents/{doc['id']}/reindex/")
    assert resp.status_code == 202 and resp.json()["status"] == "processing"
    import time

    for _ in range(100):
        d = Document.objects.get(pk=doc["id"])
        if d.status != "processing":
            break
        time.sleep(0.05)
    # the failed re-index left the document usable, with the reason attached
    assert d.status == "ready" and "embedding service down" in d.error
    assert set(Chunk.objects.filter(document_id=doc["id"]).values_list("id", flat=True)) == old


def test_reindex_that_keeps_failing_does_not_destroy_a_good_document(settings):
    from django.utils import timezone

    from docqa.services import ingest

    settings.INGEST_MAX_ATTEMPTS = 2
    c = make_client()
    doc = upload_text(c, "Policy", TEXT)
    Document.objects.filter(pk=doc["id"]).update(status="processing", processing_since=timezone.now(), attempts=2)
    ingest.run_pending()  # gives up on it: its old passages are still good
    listed = c.get(f"/api/documents/{doc['id']}/").json()
    assert listed["status"] == "ready" and "2 attempts" in listed["error"] and listed["chunk_count"] >= 1
