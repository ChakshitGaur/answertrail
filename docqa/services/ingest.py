import io
import logging
import threading

from django.conf import settings
from django.db import close_old_connections, connection, transaction
from django.db.models import F, Q
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


# ---- the indexing queue -----------------------------------------------------------------------
#
# The queue is the Document table: status "processing" means waiting (claimed_at empty) or being worked on
# (claimed_at inside the lease). A worker claims a document with one conditional UPDATE, so two workers never
# take the same one, and a worker that dies simply lets its claim expire. Nothing is held in memory, so a restart
# loses no work.

_RETRY_DELAY_SECONDS = 60  # after a failure that may be temporary, the document is tried again this much later


def _claimable(now):
    lease_ends_before = now - timezone.timedelta(seconds=settings.INGEST_LEASE_SECONDS)
    return Q(status="processing") & (Q(claimed_at__isnull=True) | Q(claimed_at__lt=lease_ends_before))


def _give_up(doc_id: int, message: str) -> None:
    """Stop retrying. A first indexing that failed leaves nothing to search: "failed". A re-index that failed leaves
    the old passages in place: the document stays "ready", with the reason attached."""
    has_chunks = Chunk.objects.filter(document_id=doc_id).exists()
    Document.objects.filter(pk=doc_id).update(
        status="ready" if has_chunks else "failed", error=message[:255], claimed_at=None
    )


def claim_next() -> Document | None:
    """Take the oldest document that is waiting (or whose worker's lease ran out). None if there is none."""
    now = timezone.now()
    for doc in Document.objects.defer("source_text").filter(_claimable(now)).order_by("processing_since", "id")[:10]:
        if doc.attempts >= settings.INGEST_MAX_ATTEMPTS:  # kept killing its workers, or kept failing
            _give_up(doc.pk, f"Indexing failed after {doc.attempts} attempts.")
            continue
        won = Document.objects.filter(_claimable(now), pk=doc.pk).update(
            claimed_at=now, attempts=F("attempts") + 1
        )
        if won:
            doc.attempts += 1
            return doc
    return None


def _index_claimed(doc: Document) -> None:
    from docqa.services import retrieval

    try:
        full = Document.objects.get(pk=doc.pk)  # with its source text
        _, pieces = _prepare(full.title, full.source_text)
        _store_chunks(full, pieces)
        retrieval.prime(full.owner)
    except Document.DoesNotExist:  # deleted while it waited
        return
    except Exception as exc:
        message = f"Indexing failed: {exc}"
        permanent = isinstance(exc, IngestError) or doc.attempts >= settings.INGEST_MAX_ATTEMPTS
        log.exception("indexing document %s failed (attempt %s)", doc.pk, doc.attempts)
        if permanent:
            _give_up(doc.pk, message)
        else:  # try again after a short pause: claim again once the (shortened) lease has run out
            retry_at = timezone.now() - timezone.timedelta(seconds=settings.INGEST_LEASE_SECONDS - _RETRY_DELAY_SECONDS)
            Document.objects.filter(pk=doc.pk).update(claimed_at=retry_at, error=message[:255])


def run_pending(limit: int | None = None) -> int:
    """Index waiting documents, one after another, until there are none (or `limit` were done). Returns how many."""
    done = 0
    while limit is None or done < limit:
        doc = claim_next()
        if doc is None:
            break
        _index_claimed(doc)
        done += 1
    return done


# one worker thread per server process (INGEST_EMBEDDED_WORKER); embedding is CPU-bound, so one at a time
_worker: threading.Thread | None = None
_worker_lock = threading.Lock()
_wake = threading.Event()
_stop = threading.Event()


def _worker_loop() -> None:
    try:
        while not _stop.is_set():
            try:
                worked = run_pending(limit=1)
            except Exception:  # e.g. the database is briefly unavailable: wait and look again
                log.exception("ingest worker error")
                worked = 0
            close_old_connections()
            if not worked:
                _wake.wait(settings.INGEST_POLL_SECONDS)
                _wake.clear()
    finally:
        connection.close()


def ensure_worker() -> None:
    """Start this process's worker thread if it is enabled and not running. Cheap; safe to call often."""
    global _worker
    if not (settings.INGEST_ASYNC and settings.INGEST_EMBEDDED_WORKER):
        return
    with _worker_lock:
        if _worker is None or not _worker.is_alive():
            _stop.clear()
            _worker = threading.Thread(target=_worker_loop, name="ingest-worker", daemon=True)
            _worker.start()


def stop_worker(timeout: float = 10) -> None:
    """Stop this process's worker thread and wait for it (used by tests and clean shutdowns)."""
    global _worker
    with _worker_lock:
        worker, _worker = _worker, None
    if worker is not None:
        _stop.set()
        _wake.set()
        worker.join(timeout)


def _enqueue(doc_id: int) -> None:
    _wake.set()  # a sleeping worker looks now instead of at its next poll
    ensure_worker()


def start_ingest(user, title: str, text: str) -> Document:
    """Validate and accept a document; with INGEST_ASYNC it is queued and indexed by a worker.

    Returns at once with status "processing" (the passages appear when it is "ready"), or, with
    INGEST_ASYNC off, after indexing with status "ready".
    """
    if not settings.INGEST_ASYNC:
        return ingest_document(user, title, text)
    title, _ = _prepare(title, text)  # the cheap checks now, so an unindexable document is refused immediately
    doc = Document.objects.create(
        owner=user, title=title, status="processing", source_text=_clean(text), reindexable=True,
        processing_since=timezone.now(),
    )
    _enqueue(doc.pk)
    return doc


def start_reindex(doc: Document) -> Document:
    """Split and embed a document's stored text again with the current settings (chunk size, embedding model)."""
    if not doc.reindexable or not doc.source_text:
        raise IngestError("This document was added before re-indexing was supported. Upload it again to re-index it.")
    _, pieces = _prepare(doc.title, doc.source_text)
    if not settings.INGEST_ASYNC:
        _store_chunks(doc, pieces)
        return doc
    Document.objects.filter(pk=doc.pk).update(
        status="processing", error="", processing_since=timezone.now(), claimed_at=None, attempts=0
    )
    doc.status, doc.error = "processing", ""
    _enqueue(doc.pk)
    return doc
