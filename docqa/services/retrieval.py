import json
import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
from django.conf import settings
from django.db import connection
from django.db.models import FloatField
from django.db.models.expressions import RawSQL

from docqa.models import Chunk
from docqa.services.embeddings import embed_query, tokenize

_RRF_K = 60  # reciprocal-rank-fusion constant; 60 is the value from the original paper


@dataclass
class Hit:
    chunk: Chunk
    score: float  # cosine similarity to the question, whichever way the hit was found


@dataclass
class SearchResult:
    hits: list[Hit]
    best_similarity: float  # cosine of the single closest chunk, however the hits were re-ordered


def search(
    user, question: str, top_k: int = 4, document_id: int | None = None, hybrid: bool | None = None
) -> list[Hit]:
    return search_ex(user, question, top_k, document_id, hybrid).hits


def search_ex(
    user, question: str, top_k: int = 4, document_id: int | None = None, hybrid: bool | None = None
) -> SearchResult:
    """Top-k chunks for a question, restricted to the user's own documents.

    Hybrid mode (default) fuses two rankings with reciprocal rank fusion: vector similarity, which
    understands meaning, and BM25, which rewards exact words such as names, numbers and codes.
    `Hit.score` is always the cosine similarity, so callers can judge relevance on one scale, and
    `best_similarity` is the closest chunk's cosine regardless of fusion (used to decide whether
    the documents are relevant to the question at all).
    """
    chunks = Chunk.objects.filter(document__owner=user).select_related("document")
    if document_id is not None:
        chunks = chunks.filter(document_id=document_id)
    query = embed_query(question)
    use_hybrid = settings.HYBRID_SEARCH if hybrid is None else hybrid
    if not use_hybrid:
        hits = _vector_search(chunks, query, top_k)
        return SearchResult(hits, hits[0].score if hits else 0.0)

    pool = max(top_k * 5, 20)
    vector_hits = _vector_search(chunks, query, pool)
    best = vector_hits[0].score if vector_hits else 0.0
    keyword_chunks = _keyword_search(chunks, question, pool)
    fused_ids = _fuse([[h.chunk.id for h in vector_hits], [c.id for c in keyword_chunks]])[:top_k]

    by_id = {h.chunk.id: h for h in vector_hits}
    extra = {c.id: c for c in keyword_chunks if c.id not in by_id}
    if extra:  # found by keywords only: fetch their vectors to put them on the cosine scale
        q = np.array(query, dtype=np.float32)
        for chunk_id, embedding in Chunk.objects.filter(id__in=extra).values_list("id", "embedding"):
            by_id[chunk_id] = Hit(chunk=extra[chunk_id], score=float(np.array(embedding, dtype=np.float32) @ q))
    return SearchResult([by_id[i] for i in fused_ids], best)


def _vector_search(chunks, query: list[float], top_k: int) -> list[Hit]:
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


def _keyword_search(chunks, question: str, limit: int) -> list[Chunk]:
    """Chunks ranked by BM25 against the question; chunks sharing no word with it are left out."""
    rows = list(chunks.defer("embedding"))  # BM25 only needs the text
    scores = bm25_scores(tokenize(question), [tokenize(c.text) for c in rows])
    ranked = sorted(range(len(rows)), key=lambda i: scores[i], reverse=True)
    return [rows[i] for i in ranked[:limit] if scores[i] > 0]


def bm25_scores(query: list[str], docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Okapi BM25 score of every document (a token list) for the query tokens."""
    if not docs:
        return []
    n = len(docs)
    avg_len = sum(len(d) for d in docs) / n or 1.0
    doc_freq = Counter(t for d in docs for t in set(d))
    scores = []
    for doc in docs:
        counts = Counter(doc)
        score = 0.0
        for term in set(query):
            tf = counts.get(term, 0)
            if not tf:
                continue
            idf = math.log(1 + (n - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc) / avg_len))
        scores.append(score)
    return scores


def _fuse(rankings: list[list[int]]) -> list[int]:
    """Reciprocal rank fusion: items ranked high in any list rise; ties keep first-list order."""
    total: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            total[item] = total.get(item, 0.0) + 1.0 / (_RRF_K + rank)
    return sorted(total, key=lambda item: -total[item])
