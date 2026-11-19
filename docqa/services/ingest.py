import io
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from django.conf import settings
from django.db import connection, transaction
from django.utils import timezone

from docqa.models import Chunk, Document
from docqa.services.chunking import chunk_text
from docqa.services.embeddings import embed_passages

log = logging.getLogger(__name__)


class IngestError(ValueError):
    pass


def _decode(data: bytes) -> str:
    """Text files come in several encodings; guessing wrong would silently store garbage."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")  # Windows "ANSI": covers Latin-1 accents and smart quotes


def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        try:
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            raise IngestError(f"Could not read PDF: {exc}") from exc
        if not text.strip():
            raise IngestError(
                "This PDF has no selectable text; it is probably a scan (an image of pages). "
                "Run it through OCR first, or upload a text version."
            )
        return text
    if name.endswith((".txt", ".md")):
        return _decode(data)
    raise IngestError("Unsupported file type; use .pdf, .txt or .md")


def _clean(value: str) -> str:
    """PostgreSQL text columns cannot hold NUL characters (PDFs and binary-ish files can contain them)."""
    return value.replace(chr(0), "")


def _prepare(title: str, text: str) -> tuple[str, list[str]]:
    """The cheap checks and the split into passages; raises IngestError for a document that cannot be indexed."""
    title = _clean(title).strip()[:255] or "Untitled"
    pieces = chunk_text(_clean(text), settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    if not pieces or not any(ch.isalnum() for ch in "".join(pieces)):
        raise IngestError("Document contains no extractable text")
    return title, pieces


def _store_chunks(doc: Document, pieces: list[str]) -> None:
    """Embed the passages and make them the document's chunks, replacing any it had; then mark it ready.

    The embedding happens first and the swap is one transaction, so a document being re-indexed keeps its old
    passages (and stays searchable) until the new ones are complete.
    """
    vectors = embed_passages(pieces)
    with transaction.atomic():
        doc.chunks.all().delete()
        Chunk.objects.bulk_create(
            Chunk(document=doc, index=i, text=p, embedding=v) for i, (p, v) in enumerate(zip(pieces, vectors))
        )
        Document.objects.filter(pk=doc.pk).update(status="ready", error="")
    doc.status = "ready"


def ingest_document(user, title: str, text: str) -> Document:
    """Index a document inside the caller's request."""
    title, pieces = _prepare(title, text)
    doc = Document.objects.create(owner=user, title=title, source_text=_clean(text), reindexable=True)
    try:
        _store_chunks(doc, pieces)
    except Exception:
        doc.delete()
        raise
    return doc


_executor = None
_executor_lock = threading.Lock()


def _pool() -> ThreadPoolExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:  # one at a time: embedding is CPU-bound, and two big files at once only slow both
            _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")
        return _executor


def _index_in_background(doc_id: int, pieces: list[str], user_id: int) -> None:
    try:
        doc = Document.objects.defer("source_text").filter(pk=doc_id).first()
        if doc is None:  # deleted while it waited
            return
        _store_chunks(doc, pieces)
        from django.contrib.auth import get_user_model

        from docqa.services import retrieval

        retrieval.prime(get_user_model().objects.get(pk=user_id))
    except Exception as exc:
        log.exception("indexing document %s failed", doc_id)
        message = f"Indexing failed: {exc}"[:255]
        # a first indexing that failed leaves nothing to search: "failed". A re-index that failed leaves the old
        # passages in place: the document stays "ready", with the reason attached.
        has_chunks = Chunk.objects.filter(document_id=doc_id).exists()
        Document.objects.filter(pk=doc_id).update(status="ready" if has_chunks else "failed", error=message)
    finally:
        connection.close()  # this thread's own database connection


def start_ingest(user, title: str, text: str) -> Document:
    """Validate and accept a document; with INGEST_ASYNC the embedding then runs in the background.

    Returns at once with status "processing" (the passages appear when it is "ready"), or, with
    INGEST_ASYNC off, after indexing with status "ready".
    """
    if not settings.INGEST_ASYNC:
        return ingest_document(user, title, text)
    title, pieces = _prepare(title, text)
    doc = Document.objects.create(
        owner=user, title=title, status="processing", source_text=_clean(text), reindexable=True,
        processing_since=timezone.now(),
    )
    _pool().submit(_index_in_background, doc.pk, pieces, user.pk)
    return doc


def start_reindex(doc: Document) -> Document:
    """Split and embed a document's stored text again with the current settings (chunk size, embedding model)."""
    if not doc.reindexable or not doc.source_text:
        raise IngestError("This document was added before re-indexing was supported. Upload it again to re-index it.")
    _, pieces = _prepare(doc.title, doc.source_text)
    if not settings.INGEST_ASYNC:
        _store_chunks(doc, pieces)
        return doc
    Document.objects.filter(pk=doc.pk).update(status="processing", error="", processing_since=timezone.now())
    doc.status, doc.error = "processing", ""
    _pool().submit(_index_in_background, doc.pk, pieces, doc.owner_id)
    return doc
