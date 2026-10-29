import json
from dataclasses import dataclass

import numpy as np
from django.db import connection
from django.db.models import FloatField
from django.db.models.expressions import RawSQL

from docqa.models import Chunk
from docqa.services.embeddings import embed_query


@dataclass
class Hit:
    chunk: Chunk
    score: float


def search(user, question: str, top_k: int = 4, document_id: int | None = None) -> list[Hit]:
    """Top-k chunks by cosine similarity, restricted to the user's own documents."""
    chunks = Chunk.objects.filter(document__owner=user).select_related("document")
    if document_id is not None:
        chunks = chunks.filter(document_id=document_id)
    query = embed_query(question)
    if connection.vendor == "postgresql":
        return _search_pgvector(chunks, query, top_k)
    return _search_numpy(chunks, query, top_k)


def _search_pgvector(chunks, query: list[float], top_k: int) -> list[Hit]:
    # `<=>` is cosine distance; served by the HNSW index when the planner picks it.
    ranked = chunks.annotate(
        distance=RawSQL(
            "docqa_chunk.embedding <=> %s::vector", [json.dumps(query)], output_field=FloatField()
        )
    ).order_by("distance")[:top_k]
    return [Hit(chunk=c, score=1.0 - c.distance) for c in ranked]


def _search_numpy(chunks, query: list[float], top_k: int) -> list[Hit]:
    rows = list(chunks)
    if not rows:
        return []
    matrix = np.array([c.embedding for c in rows], dtype=np.float32)
    scores = matrix @ np.array(query, dtype=np.float32)  # unit vectors: dot == cosine
    order = np.argsort(-scores)[:top_k]
    return [Hit(chunk=rows[i], score=float(scores[i])) for i in order]
