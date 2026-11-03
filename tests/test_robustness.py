"""Things that must not break: hostile input, odd files, odd provider replies, concurrent streams."""


import json
import random

import pytest
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from tests.pdfs import blank_pdf, text_pdf
from rest_framework.test import APIClient

from docqa.models import Chunk, Document
from docqa.services import llm
from docqa.services.chunking import chunk_text
from docqa.services.embeddings import embed_passages, embed_query
from docqa.services.ingest import IngestError, extract_text, ingest_document
from tests.test_api import PASSWORD, make_client, upload_text
from tests.test_modes import REFUNDS
from tests.test_stream import ask_stream, parse_sse, text_of

pytestmark = pytest.mark.django_db


# ---- chunking never loses text or crashes -----------------------------------------------------

ALPHABETS = [
    "abcdefghij klmnop. qrst! uvwxyz? ",
    "caf\u00e9 na\u00efve \u00fcber \u00f1and\u00fa. ",
    "\u4e2d\u6587\u6587\u672c\u6ca1\u6709\u7a7a\u683c\u3002\u4e2d\u6587\u6587\u672c ",
    "\U0001f600\U0001f680 \U0001f9ea.  ",
    "e\u0301 a\u0308 o\u0302 ",
    "line one\nline two\r\n\ttabbed\u00a0nbsp\u200bzero\u2028sep ",
    "...!!!???   . . . ",
    "x" * 40 + " ",
]


def random_text(rng):
    parts = []
    for _ in range(rng.randint(0, 40)):
        alphabet = rng.choice(ALPHABETS)
        parts.append("".join(rng.choice(alphabet) for _ in range(rng.randint(1, 120))))
    if rng.random() < 0.2:
        parts.append("Z" * rng.randint(100, 4000))  # one enormous word with no spaces
    return "".join(parts)


def test_chunking_fuzz_never_crashes_never_overflows_and_keeps_every_word():
    rng = random.Random(2026)
    for case in range(400):
        text = random_text(rng)
        size = rng.choice([1, 2, 5, 20, 80, 200, 500, 2000])
        overlap = rng.randint(0, size - 1)
        chunks = chunk_text(text, size, overlap)
        assert all(isinstance(c, str) and c for c in chunks), (case, size, overlap)
        assert all(len(c) <= size for c in chunks), (case, size, overlap, max(map(len, chunks)))
        if text.split():
            assert chunks, (case, text[:50])
        for word in set(text.split()):
            if len(word) <= size:  # a word longer than a chunk has to be cut; anything shorter must survive
                assert any(word in c for c in chunks), (case, word, size)


def test_chunking_pathological_inputs():
    assert chunk_text("a" * 100_000, 500, 100)  # one 100k-character word is cut, not hung on
    assert chunk_text("." * 5000, 50, 10) is not None
    assert chunk_text("\n\n\n   \t\t", 100, 10) == []
    assert chunk_text("word " * 50_000, 500, 100)  # 250k characters stays fast and finite
    assert chunk_text("\u200b\u200b\u200b", 10, 2) is not None


# ---- embeddings accept any text ----------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["", " ", "???", "\U0001f600" * 50, "\u4e2d\u6587", "a" * 10_000, "\x01\x02\x03", "e\u0301" * 100, "\n\t\r"],
)
def test_hash_embeddings_handle_awkward_text(text):
    (vector,) = embed_passages([text])
    norm = sum(x * x for x in vector) ** 0.5
    assert len(vector) == 384 and (abs(norm - 1.0) < 1e-4 or norm == 0.0)
    assert len(embed_query(text)) == 384


def test_neural_embeddings_handle_awkward_text(settings):
    pytest.importorskip("fastembed")
    settings.EMBEDDING_BACKEND = "fastembed"
    for text in ("", "???", "\U0001f600" * 50, "\u4e2d\u6587\u6587\u672c", "a" * 5000):
        assert len(embed_query(text)) == 384


# ---- files and ingestion -------------------------------------------------------------------------

MINI_PDF = text_pdf()


def test_a_real_text_pdf_is_read():
    assert "Hello PDF world" in extract_text("doc.pdf", MINI_PDF)


@pytest.mark.parametrize(
    "data",
    [b"", b"%PDF-", b"not a pdf at all", bytes(range(256)) * 20, MINI_PDF[: len(MINI_PDF) // 2], blank_pdf()],
    ids=["empty", "header-only", "text", "binary", "truncated", "blank-page"],
)
def test_broken_or_textless_pdfs_give_a_clean_error_never_a_crash(data):
    with pytest.raises(IngestError):
        user = get_user_model().objects.create_user("u")
        ingest_document(user, "x", extract_text("scan.pdf", data))


def test_a_password_protected_pdf_is_rejected_cleanly():
    with pytest.raises(IngestError):
        extract_text("secret.pdf", blank_pdf("hunter2"))


def test_text_decoding_never_fails(tmp_path):
    assert "caf" in extract_text("a.txt", "caf\u00e9 refund".encode("latin-1"))  # not UTF-8: replaced, not an error
    assert "refund" in extract_text("a.md", b"\xef\xbb\xbfrefund policy")  # BOM
    assert extract_text("a.TXT", b"upper-case extension")  # extension check is case-insensitive
    with pytest.raises(IngestError):
        extract_text("a.docx", b"PK\x03\x04")
    with pytest.raises(IngestError):
        extract_text("no_extension", b"text")


def test_nul_bytes_are_removed_because_postgres_rejects_them():
    user = get_user_model().objects.create_user("u")
    doc = ingest_document(user, "nul", "refund\x00policy applies\x00 here")
    assert all("\x00" not in c.text for c in doc.chunks.all())
    assert "refundpolicy" in " ".join(c.text for c in doc.chunks.all()).replace(" ", "")


def test_nul_bytes_in_a_titles_are_removed_too():
    user = get_user_model().objects.create_user("u")
    doc = ingest_document(user, "ti\x00tle", "some text")
    assert "\x00" not in doc.title


def test_very_long_file_names_become_valid_titles():
    client = make_client()
    f = SimpleUploadedFile("a" * 400 + ".txt", b"refund policy text")
    resp = client.post("/api/documents/", {"file": f}, format="multipart")
    assert resp.status_code == 201
    assert 0 < len(resp.data["title"]) <= 255 and resp.data["title"].endswith(".txt")


@pytest.mark.parametrize("name", ["../../etc/passwd.txt", "C:\\Windows\\x.txt", "caf\u00e9 \u4e2d\u6587 \U0001f600.md", ".txt", "a.b.c.txt", " .txt"])
def test_odd_file_names_are_accepted_without_touching_the_filesystem(name):
    client = make_client()
    resp = client.post("/api/documents/", {"file": SimpleUploadedFile(name, b"refund policy")}, format="multipart")
    assert resp.status_code == 201, resp.data
    assert resp.data["title"]


def test_upload_size_limit_is_enforced_at_the_boundary(settings):
    settings.MAX_UPLOAD_BYTES = 100
    client = make_client()
    ok = client.post("/api/documents/", {"file": SimpleUploadedFile("a.txt", b"x " * 50)}, format="multipart")
    too_big = client.post("/api/documents/", {"file": SimpleUploadedFile("b.txt", b"x " * 51)}, format="multipart")
    assert ok.status_code == 201 and too_big.status_code == 413


@pytest.mark.parametrize(
    "payload",
    [{}, {"title": "t"}, {"text": "x"}, {"title": "t", "text": "  "}, {"title": "", "text": "x"}, {"title": "t" * 256, "text": "x"}],
)
def test_bad_document_payloads_are_400(payload):
    assert make_client().post("/api/documents/", payload, format="json").status_code == 400


def test_a_request_body_larger_than_django_allows_is_a_client_error_not_a_crash(settings):
    settings.DATA_UPLOAD_MAX_MEMORY_SIZE = 1000
    resp = make_client().post("/api/documents/", {"title": "big", "text": "word " * 1000}, format="json")
    assert 400 <= resp.status_code < 500


def test_both_file_and_text_together_is_rejected():
    f = SimpleUploadedFile("a.txt", b"refund")
    resp = make_client().post("/api/documents/", {"file": f, "text": "also text", "title": "t"}, format="multipart")
    assert resp.status_code == 400


# ---- API fuzzing: nothing a client sends may produce a server error ---------------------------


def junk(rng, depth=0):
    simple = [
        None, True, False, 0, -1, 7, 10**30, 3.14, "", " ", "x" * 3000, "\U0001f600" * 300, "\u2028",
        "ok", "refund window", "<script>alert(1)</script>", "'; DROP TABLE docqa_chunk; --", "auto", "general",
        "documents", "user", "assistant", [], {}, [None], [[]], {"a": 1}, {"role": "user", "content": "hi"},
    ]
    if depth < 2 and rng.random() < 0.25:
        return [junk(rng, depth + 1) for _ in range(rng.randint(0, 4))]
    if depth < 2 and rng.random() < 0.15:
        return {rng.choice(["role", "content", "q", "mode", "x"]): junk(rng, depth + 1) for _ in range(rng.randint(0, 3))}
    return rng.choice(simple)


def junk_body(rng, keys):
    body = {}
    for key in keys:
        if rng.random() < 0.8:
            body[key] = junk(rng)
    if rng.random() < 0.3:
        body[rng.choice(["extra", "__proto__", "mode ", "Question"])] = junk(rng)
    return body


@pytest.mark.parametrize("url", ["/api/ask/", "/api/ask/stream/"])
def test_ask_endpoints_never_return_a_server_error_for_junk(url, no_throttle):
    client = make_client()
    upload_text(client, "Refunds", REFUNDS)
    rng = random.Random(11)
    seen = set()
    for i in range(250):
        body = junk_body(rng, ["question", "mode", "history", "top_k", "document_id"])
        resp = client.post(url, body, format="json")
        seen.add(resp.status_code)
        assert resp.status_code < 500, (i, body, resp.status_code)
        if resp.status_code == 200 and url.endswith("stream/"):
            events = parse_sse(resp)
            assert events[0][0] == "sources" and events[-1][0] in ("done", "error"), (i, body)
    assert 400 in seen and 200 in seen  # the generator really exercised both outcomes


@pytest.mark.parametrize("raw,content_type", [(b"", "application/json"), (b"{not json", "application/json"), (b"[1,2,3]", "application/json"), (b'"str"', "application/json"), (b"null", "application/json"), (b"question=hi", "text/plain"), (b"\xff\xfe\x00", "application/json")])
def test_malformed_request_bodies_are_client_errors(raw, content_type):
    client = make_client()
    for url in ("/api/ask/", "/api/ask/stream/", "/api/documents/"):
        resp = client.post(url, data=raw, content_type=content_type)
        assert 400 <= resp.status_code < 500, (url, raw, resp.status_code)


def test_document_and_register_endpoints_survive_junk(no_throttle):
    rng = random.Random(5)
    client = make_client()
    for _ in range(120):
        assert client.post("/api/documents/", junk_body(rng, ["title", "text", "file"]), format="json").status_code < 500
        anon = APIClient().post("/api/auth/register/", junk_body(rng, ["username", "password"]), format="json")
        assert anon.status_code < 500


@pytest.mark.parametrize(
    "username",
    ["a" * 151, "", "  ", "bad\x00name", "caf\u00e9 \u4e2d\u6587", "has space", "\U0001f600", "O'Brien", "a@b.c", "x" * 150],
)
def test_awkward_usernames_are_accepted_or_rejected_cleanly(username):
    resp = APIClient().post("/api/auth/register/", {"username": username, "password": PASSWORD}, format="json")
    assert resp.status_code in (201, 400), (username, resp.status_code, resp.data)


def test_register_and_login_edge_cases():
    c = APIClient()
    assert c.post("/api/auth/register/", {"username": "twin", "password": PASSWORD}, format="json").status_code == 201
    assert c.post("/api/auth/register/", {"username": "twin", "password": PASSWORD}, format="json").status_code == 400
    assert c.post("/api/auth/register/", {"username": "pw-is-name-1", "password": "pw-is-name-1"}, format="json").status_code in (201, 400)
    assert c.post("/api/auth/token/", {"username": "twin", "password": "wrong"}, format="json").status_code == 400
    assert c.post("/api/auth/token/", {"username": "ghost", "password": PASSWORD}, format="json").status_code == 400
    assert c.post("/api/auth/token/", {}, format="json").status_code == 400
    good = c.post("/api/auth/token/", {"username": "twin", "password": PASSWORD}, format="json")
    assert good.status_code == 200 and good.data["token"]


@pytest.mark.parametrize("header", ["Token", "Token ", "Token a b", "Bearer abc", "token lowercase", "Basic eDp5", "Token \u00e9\u00e8", "Token " + "x" * 5000, ""])
def test_strange_authorization_headers_are_401_not_500(header):
    client = APIClient()
    if header:
        client.credentials(HTTP_AUTHORIZATION=header)
    for url in ("/api/documents/", "/api/ask/"):
        resp = client.get(url) if url.endswith("documents/") else client.post(url, {"question": "q"}, format="json")
        assert resp.status_code in (401, 403), (header[:20], url, resp.status_code)


def test_http_methods_that_do_not_belong_are_405_not_500():
    client = make_client()
    assert client.get("/api/ask/").status_code == 405
    assert client.put("/api/ask/stream/", {}, format="json").status_code == 405
    assert client.delete("/api/documents/").status_code == 405
    assert client.get("/api/documents/not-a-number/").status_code == 404
    assert client.get("/api/documents/99999999999999999999/").status_code in (404, 400)


def test_deleting_a_document_twice_and_asking_about_it_afterwards():
    client = make_client()
    doc = upload_text(client, "Refunds", REFUNDS).data
    assert client.delete(f"/api/documents/{doc['id']}/").status_code == 204
    assert client.delete(f"/api/documents/{doc['id']}/").status_code == 404
    assert ask_stream(client, "refund window", document_id=doc["id"]).status_code == 404
    assert Chunk.objects.count() == 0 and Document.objects.count() == 0


# ---- awkward characters survive the whole stream -------------------------------------------------

SPECIAL = "line1\nline2\n\nline3" + ' "quoted" \\ back\\slash ' + "\U0001f600 <b>x</b> </script> \u2028 \u00e9\u4e2d\u6587"


@pytest.fixture
def client():
    c = make_client()
    upload_text(c, "Refunds", REFUNDS)
    return c


def test_newlines_quotes_backslashes_emoji_and_html_survive_json_and_sse(client, llm_settings, settings):
    settings.LLM_MODEL, settings.LLM_FALLBACK_MODELS = "special", ""
    for mode in ("documents", "general"):
        events = parse_sse(ask_stream(client, "What is the refund window?", mode=mode))
        assert text_of(events) == SPECIAL and events[-1][0] == "done"


def test_a_stream_cut_into_three_byte_pieces_still_decodes_exactly(client, llm_settings, settings):
    """Frames and multi-byte characters are split across network reads."""
    settings.LLM_MODEL, settings.LLM_FALLBACK_MODELS = "split", ""
    events = parse_sse(ask_stream(client, "What is the refund window?", mode="documents"))
    assert text_of(events) == "Hello w\u00f6rld \u2713 from split [1]."
    general = parse_sse(ask_stream(client, "Explain recursion", mode="general"))
    assert text_of(general) == "General answer: Explain recursion"


def test_unicode_questions_and_history_work_end_to_end(client, llm_settings):
    history = [{"role": "user", "content": "\u4e2d\u6587 \U0001f600"}, {"role": "assistant", "content": "caf\u00e9"}]
    events = parse_sse(ask_stream(client, "\u00bfCu\u00e1l es el plazo de reembolso? \U0001f914", mode="auto", history=history))
    assert events[-1][0] == "done" and text_of(events)


def test_two_streams_in_flight_at_once_do_not_mix(client, llm_settings, settings):
    other = make_client("second")
    upload_text(other, "Plants", "Photosynthesis occurs in the chloroplasts of plant leaves.")
    first = ask_stream(client, "What is the refund window?", mode="documents")
    second = ask_stream(other, "photosynthesis chloroplasts", mode="documents")
    second_events = parse_sse(second)  # finish the second before the first has been read at all
    first_events = parse_sse(first)
    assert {s["document_title"] for s in first_events[0][1]["sources"]} == {"Refunds"}
    assert {s["document_title"] for s in second_events[0][1]["sources"]} == {"Plants"}
    assert text_of(first_events) and text_of(second_events)


def test_a_client_that_disconnects_mid_stream_leaves_the_server_healthy(client, llm_settings):
    resp = ask_stream(client, "What is the refund window?", mode="documents")
    stream = iter(resp.streaming_content)
    next(stream)  # read the first event, then walk away
    # (not resp.close(): that fires Django's request_finished and closes the test database connection)
    del stream, resp
    import gc

    gc.collect()  # the abandoned generator is closed with GeneratorExit, as when a client disconnects
    assert ask_stream(client, "What is the refund window?", mode="documents").status_code == 200


# ---- odd replies from the model provider ----------------------------------------------------------


def frame(**delta):
    return "data: " + json.dumps({"choices": [{"delta": delta}]})


def stream_lines(script, settings, lines):
    settings.LLM_MODEL, settings.LLM_FALLBACK_MODELS = "m", ""
    script(("sse", lines))
    return "".join(llm.stream_answer("q", None))


def test_streaming_ignores_frames_it_does_not_understand(script, settings):
    lines = [
        ": keep-alive", 'data: {"choices": []}', 'data: {"usage": {"total_tokens": 3}}',
        frame(role="assistant"), frame(content=None), frame(content=""),
        frame(content="Hel"), frame(content="lo"), "data:[DONE]", frame(content="AFTER DONE"),
    ]
    assert stream_lines(script, settings, lines) == "Hello"


def test_a_frame_that_is_not_json_is_an_llm_error_not_a_crash(script, settings):
    with pytest.raises(llm.LLMError):
        stream_lines(script, settings, [frame(content="Hel"), "data: not json at all", "data: [DONE]"])


def test_streaming_accepts_done_without_a_space_and_carriage_returns(script, settings):
    assert stream_lines(script, settings, [frame(content="ok") + "\r", "data:[DONE]\r"]) == "ok"


@pytest.mark.parametrize("bad", [5, 3.5, True, {"x": 1}])
def test_streaming_never_shows_content_that_is_not_text(script, settings, bad):
    try:
        text = stream_lines(script, settings, [frame(content=bad), frame(content="fine"), "data: [DONE]"])
    except llm.LLMError:
        return  # refusing a malformed reply is fine
    assert text == "fine"  # so is skipping it, but it must never be shown as an answer


def test_a_stream_with_only_an_error_object_is_reported_as_an_llm_error(script, settings):
    with pytest.raises(llm.LLMError, match="quota"):
        stream_lines(script, settings, ['data: {"error": {"message": "quota"}}', "data: [DONE]"])


@pytest.mark.parametrize("model", ["nullcontent", "nochoices"])
def test_a_reply_without_text_is_an_llm_error_not_a_crash(llm_settings, settings, model):
    settings.LLM_MODEL, settings.LLM_FALLBACK_MODELS = model, ""
    with pytest.raises(llm.LLMError):
        llm.answer("q", ["passage"])


@pytest.mark.parametrize(
    "payload",
    [
        {}, [], "text", None, {"choices": None}, {"choices": [None]}, {"choices": [{}]}, {"choices": [{"message": None}]},
        {"choices": [{"message": {"content": 5}}]}, {"choices": [{"message": {"content": ["a"]}}]},
        {"choices": [{"message": {"content": "   "}}]},
    ],
)
def test_non_streaming_replies_with_unexpected_shapes_are_llm_errors(script, settings, payload):
    settings.LLM_MODEL, settings.LLM_FALLBACK_MODELS = "m", ""
    script(("raw", json.dumps(payload)))
    with pytest.raises(llm.LLMError):
        llm.answer("q", None)


def test_a_non_json_reply_is_an_llm_error(script, settings):
    settings.LLM_MODEL, settings.LLM_FALLBACK_MODELS = "m", ""
    script(("raw", "<html>gateway error</html>"))
    with pytest.raises(llm.LLMError):
        llm.answer("q", None)


def test_an_empty_text_from_the_model_is_never_shown_as_a_successful_answer(client, llm_settings, settings):
    settings.LLM_MODEL, settings.LLM_FALLBACK_MODELS = "nullcontent", ""
    resp = client.post("/api/ask/", {"question": "What is the refund window?", "mode": "documents"}, format="json")
    assert resp.status_code == 502 and "language model" in resp.data["detail"]


# ---- the search itself --------------------------------------------------------------------------


@pytest.mark.parametrize("question", ["???", "...", "\U0001f600\U0001f680", "a", "the of and", "\u4e2d\u6587", "x" * 2000, "SELECT * FROM docqa_chunk", "%s %d {0} {{x}}", "\\"])
def test_any_question_can_be_searched(client, llm_settings, question):
    for mode in ("documents", "auto", "general"):
        resp = ask_stream(client, question, mode=mode)
        assert resp.status_code == 200
        events = parse_sse(resp)
        assert events[-1][0] in ("done", "error")


def test_a_document_of_one_enormous_word_can_be_uploaded_and_searched():
    client = make_client()
    assert upload_text(client, "Wall", "Z" * 20_000).status_code == 201
    assert client.post("/api/ask/", {"question": "ZZZ", "mode": "documents"}, format="json").status_code == 200
