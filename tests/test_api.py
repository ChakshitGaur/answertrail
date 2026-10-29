import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework.test import APIClient

pytestmark = pytest.mark.django_db

PASSWORD = "a-Strong-pass-98"


def make_client(username="alice"):
    client = APIClient()
    resp = client.post(
        "/api/auth/register/", {"username": username, "password": PASSWORD}, format="json"
    )
    assert resp.status_code == 201
    client.credentials(HTTP_AUTHORIZATION=f"Token {resp.data['token']}")
    return client


def upload_text(client, title, text):
    return client.post("/api/documents/", {"title": title, "text": text}, format="json")


def test_requires_auth():
    assert APIClient().get("/api/documents/").status_code == 401


def test_register_rejects_weak_password():
    resp = APIClient().post(
        "/api/auth/register/", {"username": "bob", "password": "123"}, format="json"
    )
    assert resp.status_code == 400


def test_upload_text_and_list():
    client = make_client()
    resp = upload_text(client, "Policy", "Refunds are accepted within 30 days. " * 40)
    assert resp.status_code == 201
    assert resp.data["chunk_count"] >= 1
    assert len(client.get("/api/documents/").data) == 1


def test_upload_file_txt():
    client = make_client()
    f = SimpleUploadedFile("notes.txt", b"Django is a Python web framework.")
    resp = client.post("/api/documents/", {"file": f}, format="multipart")
    assert resp.status_code == 201
    assert resp.data["title"] == "notes.txt"


def test_rejects_unsupported_file_and_empty_text():
    client = make_client()
    f = SimpleUploadedFile("evil.exe", b"MZ")
    assert client.post("/api/documents/", {"file": f}, format="multipart").status_code == 400
    assert upload_text(client, "Empty", "   ").status_code == 400


def test_ask_returns_relevant_source_with_fallback_answer(settings):
    settings.ANTHROPIC_API_KEY = ""
    client = make_client()
    upload_text(client, "Refunds", "Refunds are accepted within 30 days of purchase.")
    upload_text(client, "Plants", "Photosynthesis occurs in the chloroplasts of leaves.")
    resp = client.post("/api/ask/", {"question": "What is the refund window?"}, format="json")
    assert resp.status_code == 200
    assert resp.data["sources"][0]["document_title"] == "Refunds"
    assert "30 days" in resp.data["answer"]


def test_ask_returns_502_when_llm_backend_is_down(settings, monkeypatch):
    from docqa.services import llm

    settings.LLM_BASE_URL, settings.LLM_MODEL = "http://localhost:1/v1", "m"
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        llm.requests, "post", lambda *a, **k: (_ for _ in ()).throw(llm.requests.ConnectionError("down"))
    )
    client = make_client()
    upload_text(client, "Refunds", "Refunds are accepted within 30 days of purchase.")
    resp = client.post("/api/ask/", {"question": "refund window?"}, format="json")
    assert resp.status_code == 502
    assert "language model" in resp.data["detail"]


def test_ask_with_no_documents():
    resp = make_client().post("/api/ask/", {"question": "anything?"}, format="json")
    assert resp.status_code == 200
    assert resp.data["sources"] == []


def test_users_cannot_see_or_search_each_others_documents():
    alice, bob = make_client("alice"), make_client("bob")
    doc = upload_text(alice, "Secret", "The launch code is swordfish.").data

    assert bob.get(f"/api/documents/{doc['id']}/").status_code == 404
    assert bob.delete(f"/api/documents/{doc['id']}/").status_code == 404
    assert bob.post("/api/ask/", {"question": "launch code swordfish"}, format="json").data[
        "sources"
    ] == []
    assert (
        bob.post(
            "/api/ask/", {"question": "x", "document_id": doc["id"]}, format="json"
        ).status_code
        == 404
    )


def test_delete_removes_chunks():
    client = make_client()
    doc = upload_text(client, "Temp", "Some text to delete.").data
    assert client.delete(f"/api/documents/{doc['id']}/").status_code == 204
    resp = client.post("/api/ask/", {"question": "text delete"}, format="json")
    assert resp.data["sources"] == []
