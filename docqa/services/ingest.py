import io

from django.conf import settings
from django.db import transaction

from docqa.models import Chunk, Document
from docqa.services.chunking import chunk_text
from docqa.services.embeddings import embed_passages


class IngestError(ValueError):
    pass


def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        try:
            reader = PdfReader(io.BytesIO(data))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            raise IngestError(f"Could not read PDF: {exc}") from exc
    if name.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="replace")
    raise IngestError("Unsupported file type; use .pdf, .txt or .md")


def _clean(value: str) -> str:
    """PostgreSQL text columns cannot hold NUL characters (PDFs and binary-ish files can contain them)."""
    return value.replace(chr(0), "")


@transaction.atomic
def ingest_document(user, title: str, text: str) -> Document:
    title = _clean(title).strip()[:255] or "Untitled"
    pieces = chunk_text(_clean(text), settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    if not pieces:
        raise IngestError("Document contains no extractable text")
    vectors = embed_passages(pieces)
    doc = Document.objects.create(owner=user, title=title)
    Chunk.objects.bulk_create(
        Chunk(document=doc, index=i, text=p, embedding=v)
        for i, (p, v) in enumerate(zip(pieces, vectors))
    )
    return doc
