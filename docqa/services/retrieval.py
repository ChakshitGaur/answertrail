import json
import logging
import math
import operator
import threading
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from functools import reduce

import numpy as np
from django.conf import settings
from django.db import DatabaseError, connection, transaction
from django.db.models import Count, FloatField, Max, Q, Sum
from django.db.models.expressions import RawSQL

from docqa.models import Chunk, Document
from docqa.services.embeddings import embed_query, tokenize

log = logging.getLogger(__name__)

_RRF_K = 60  # reciprocal-rank-fusion constant; 60 is the value from the original paper
_CACHE_MAX_ENTRIES = 4  # users (or user+document scopes) kept in memory
_CACHE_MAX_CHUNKS = 50_000  # larger collections are searched without caching rather than hogging memory


@dataclass
class Hit:
    chunk: Chunk
    score: float | None  # cosine similarity to the question; None for overview samples (not found by similarity)


@dataclass
class SearchResult:
    hits: list[Hit]
    best_similarity: float  # cosine of the single closest chunk, however the hits were re-ordered
    # the question's embedding, so neighbours can be scored without embedding it again (not part of the comparison)
    query: list[float] | None = field(default=None, compare=False, repr=False)


# ---- keyword ranking ------------------------------------------------------------------------


class BM25Index:
    """Okapi BM25 over tokenised documents, with an inverted index so a query only touches the
    documents that contain one of its words instead of scanning every document."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.n = len(docs)
        self.k1, self.b = k1, b
        self.doc_len = [len(d) for d in docs]
        self.avg_len = (sum(self.doc_len) / self.n if self.n else 0.0) or 1.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(docs):
            for term, tf in Counter(doc).items():
                self.postings[term].append((i, tf))

    def scores(self, query: list[str]) -> dict[int, float]:
        """BM25 score of every document sharing at least one word with the query (others score 0)."""
        out: dict[int, float] = {}
        for term in set(query):
            posts = self.postings.get(term)
            if not posts:
                continue
            idf = math.log(1 + (self.n - len(posts) + 0.5) / (len(posts) + 0.5))
            for i, tf in posts:
                norm = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_len)
                out[i] = out.get(i, 0.0) + idf * tf * (self.k1 + 1) / norm
        return out


def bm25_scores(query: list[str], docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Score of every document (a token list) for the query tokens, in document order."""
    found = BM25Index(docs, k1, b).scores(query)
    return [found.get(i, 0.0) for i in range(len(docs))]


def _fuse(rankings: list[list[int]]) -> list[int]:
    """Reciprocal rank fusion: items ranked high in any list rise; ties keep first-list order."""
    total: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            total[item] = total.get(item, 0.0) + 1.0 / (_RRF_K + rank)
    return sorted(total, key=lambda item: -total[item])


# ---- per-user corpus cache ------------------------------------------------------------------


@dataclass
class _Corpus:
    chunks: list[Chunk]
    matrix: np.ndarray | None  # unit vectors, one row per chunk (SQLite path only)
    index: BM25Index
    position: dict[int, int]  # chunk id -> row


_CACHE: "OrderedDict[tuple, tuple[tuple, _Corpus]]" = OrderedDict()
_CACHE_LOCK = threading.Lock()


def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


def _corpus(chunks, key: tuple, with_matrix: bool) -> _Corpus:
    """The caller's chunks, tokenised and (on SQLite) vectorised once and reused until they change.

    Re-reading every embedding and re-tokenising every chunk on every question made search slow
    in proportion to the collection's size. The cache is validated by one cheap aggregate query:
    chunks are never edited, so (count, highest id, sum of ids) changes whenever a document is added
    or removed, even if a database restore or reset were to hand out ids again.
    """
    stats = chunks.aggregate(n=Count("id"), top=Max("id"), total=Sum("id"))
    signature = (stats["n"], stats["top"], stats["total"])
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached and cached[0] == signature and (cached[1].matrix is not None or not with_matrix):
            _CACHE.move_to_end(key)
            return cached[1]

    rows = list(chunks if with_matrix else chunks.defer("embedding"))  # BM25 alone only needs the text
    matrix = np.array([c.embedding for c in rows], dtype=np.float32) if with_matrix and rows else None
    corpus = _Corpus(
        chunks=rows,
        matrix=matrix,
        index=BM25Index([tokenize(c.text) for c in rows]),
        position={c.id: i for i, c in enumerate(rows)},
    )
    if len(rows) <= _CACHE_MAX_CHUNKS:
        with _CACHE_LOCK:
            _CACHE[key] = (signature, corpus)
            _CACHE.move_to_end(key)
            while len(_CACHE) > _CACHE_MAX_ENTRIES:
                _CACHE.popitem(last=False)
    return corpus


# ---- overview -------------------------------------------------------------------------------


def overview_hits(user, document_id: int | None = None, max_passages: int = 8, max_documents: int = 5) -> list[Hit]:
    """An evenly spaced sample of passages, for "summarize this" style questions.

    A generic request like "summarize this document" is not similar to any one passage, so similarity
    search finds nothing useful. Instead take passages spread from the start to the end of each document
    (the newest documents first), so the answer reflects the whole thing and not only its opening.
    """
    docs = Document.objects.filter(owner=user)
    if document_id is not None:
        docs = docs.filter(pk=document_id)
    docs = list(docs.order_by("-created_at", "-id")[:max_documents])
    if not docs:
        return []
    per_doc = max(1, max_passages // len(docs))
    hits: list[Hit] = []
    for doc in docs:
        count = doc.chunks.count()
        if not count:
            continue
        k = min(per_doc, count)
        wanted = sorted({round(i * (count - 1) / (k - 1)) for i in range(k)}) if k > 1 else [0]
        chunks = Chunk.objects.filter(document=doc, index__in=wanted).select_related("document").defer("embedding")
        hits.extend(Hit(chunk=c, score=None) for c in chunks.order_by("index"))
    return hits


def prime(user) -> None:
    """Build the user's search cache now (right after an upload) so the next question does not pay for it."""
    chunks = Chunk.objects.filter(document__owner=user).select_related("document")
    try:
        _corpus(chunks, (user.pk, None), with_matrix=connection.vendor != "postgresql")
    except Exception:  # an optimisation must never fail an upload
        log.exception("could not prime the search cache")


# ---- search ---------------------------------------------------------------------------------


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
    key = (user.pk, document_id)
    use_hybrid = settings.HYBRID_SEARCH if hybrid is None else hybrid

    if not use_hybrid:
        hits = _vector_hits(chunks, key, query, top_k)
        return SearchResult(hits, hits[0].score if hits else 0.0, query)

    pool = max(top_k * 5, 20)
    vector_hits = _vector_hits(chunks, key, query, pool)
    best = vector_hits[0].score if vector_hits else 0.0
    keyword_chunks = _keyword_hits(chunks, key, question, pool)
    fused_ids = _fuse([[h.chunk.id for h in vector_hits], [c.id for c in keyword_chunks]])[:top_k]

    by_id = {h.chunk.id: h for h in vector_hits}
    extra = {c.id: c for c in keyword_chunks if c.id not in by_id}
    if extra:  # found by keywords only: score them on the cosine scale too
        q = np.array(query, dtype=np.float32)
        for chunk_id, embedding in Chunk.objects.filter(id__in=extra).values_list("id", "embedding"):
            by_id[chunk_id] = Hit(chunk=extra[chunk_id], score=float(np.array(embedding, dtype=np.float32) @ q))
    return SearchResult([by_id[i] for i in fused_ids], best, query)


def add_neighbours(result: SearchResult, limit: int) -> list[Hit]:
    """The hits, plus up to `limit` of the chunks right before or after them that best match the question.

    A section of a document (a resume's EDUCATION heading and the schools under it) is often cut across two
    chunks, and only one of them may rank high enough to be picked: the answer then silently omits the rest.
    The candidates are the previous and next chunk of every hit, in the same document. They are ranked by their
    own similarity to the question, and one that scores below the weakest hit is dropped (it would not have been
    found on its own), so filler neighbours are not added to questions that did not need them.
    Each chosen neighbour is placed beside the hit it continues, so the passages read in document order.
    The hits themselves, and `best_similarity`, are never changed.
    """
    hits = result.hits
    if limit <= 0 or not hits or result.query is None:
        return hits
    taken = {(h.chunk.document_id, h.chunk.index) for h in hits}
    wanted = {
        (h.chunk.document_id, i)
        for h in hits
        for i in (h.chunk.index - 1, h.chunk.index + 1)
        if i >= 0 and (h.chunk.document_id, i) not in taken
    }
    if not wanted:
        return hits
    query = np.array(result.query, dtype=np.float32)
    found = Chunk.objects.filter(
        reduce(operator.or_, (Q(document_id=d, index=i) for d, i in wanted))
    ).select_related("document")
    bar = min((h.score for h in hits if h.score is not None), default=-1.0)  # the weakest hit already chosen
    scored = sorted(
        (pair for pair in ((float(np.array(c.embedding, dtype=np.float32) @ query), c) for c in found) if pair[0] >= bar),
        key=lambda pair: (-pair[0], pair[1].document_id, pair[1].index),
    )[:limit]
    before: dict[tuple, Hit] = {}  # hit -> the chosen chunk just before it
    after: dict[tuple, Hit] = {}  # hit -> the chosen chunk just after it
    for score, chunk in scored:
        following = (chunk.document_id, chunk.index + 1)
        if following in taken:
            before[following] = Hit(chunk=chunk, score=score)
        else:
            after[(chunk.document_id, chunk.index - 1)] = Hit(chunk=chunk, score=score)
    ordered: list[Hit] = []
    for hit in hits:
        key = (hit.chunk.document_id, hit.chunk.index)
        if key in before:
            ordered.append(before[key])
        ordered.append(hit)
        if key in after:
            ordered.append(after[key])
    return ordered


def _vector_hits(chunks, key: tuple, query: list[float], top_k: int) -> list[Hit]:
    if connection.vendor == "postgresql":
        return _search_pgvector(chunks, query, top_k)
    return _search_numpy(_corpus(chunks, key, with_matrix=True), query, top_k)


def _enable_iterative_scan() -> None:
    """Index mode: make the HNSW index keep scanning until enough rows survive the owner filter (pgvector 0.8+).

    The index returns its nearest candidates first and the owner/scope filter is applied afterwards, so a
    user whose chunks are not among the globally nearest ones (other users' chunks, entries of deleted
    documents not yet vacuumed) can get fewer results than asked for, even none. The setting lasts until the
    end of the surrounding transaction; an older pgvector that does not know it is ignored (the savepoint
    stops the error from poisoning the transaction).
    """
    try:
        with transaction.atomic(), connection.cursor() as cursor:
            cursor.execute("SET LOCAL hnsw.iterative_scan = 'relaxed_order'")
    except DatabaseError:
        log.info("pgvector without iterative index scans: filtered searches may return fewer results")


def _force_exact_scan() -> None:
    """Default mode: do not use the HNSW index, so the owner filter runs first and the distance ordering
    is exact over just that user's chunks. Every question is filtered to one user's rows, a small set, so this
    is fast and, unlike an approximate index behind a filter, never misses rows."""
    with connection.cursor() as cursor:
        cursor.execute("SET LOCAL enable_indexscan = off")


def _search_pgvector(chunks, query: list[float], top_k: int) -> list[Hit]:
    # `<=>` is cosine distance
    ranked_query = chunks.annotate(
        distance=RawSQL(
            "docqa_chunk.embedding <=> %s::vector", [json.dumps(query)], output_field=FloatField()
        )
    ).order_by("distance")[:top_k]
    with transaction.atomic():  # SET LOCAL needs a transaction to live in
        if settings.PGVECTOR_USE_INDEX:
            _enable_iterative_scan()
        else:
            _force_exact_scan()
        ranked = list(ranked_query)
    ranked.sort(key=lambda c: c.distance)  # relaxed_order (index mode) may return neighbours slightly out of order
    return [Hit(chunk=c, score=1.0 - c.distance) for c in ranked]


def _search_numpy(corpus: _Corpus, query: list[float], top_k: int) -> list[Hit]:
    if corpus.matrix is None or not len(corpus.chunks):
        return []
    scores = corpus.matrix @ np.array(query, dtype=np.float32)  # unit vectors: dot == cosine
    k = min(top_k, len(scores))
    if k < len(scores):  # partial selection, then sort only the winners
        order = np.argpartition(-scores, k - 1)[:k]
        order = order[np.argsort(-scores[order])]
    else:
        order = np.argsort(-scores)
    return [Hit(chunk=corpus.chunks[i], score=float(scores[i])) for i in order]


def _keyword_hits(chunks, key: tuple, question: str, limit: int) -> list[Chunk]:
    """Chunks ranked by BM25 against the question; chunks sharing no word with it are left out."""
    corpus = _corpus(chunks, key, with_matrix=connection.vendor != "postgresql")
    found = corpus.index.scores(tokenize(question))
    ranked = sorted(found, key=lambda i: (-found[i], i))[:limit]  # ties keep document order
    return [corpus.chunks[i] for i in ranked]
