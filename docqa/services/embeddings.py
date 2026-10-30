import hashlib
import re
from functools import lru_cache

import numpy as np
from django.conf import settings

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has in is it its of on or that the this to was what when where which who with".split()
)


def tokenize(text: str) -> list[str]:
    tokens = (t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS)
    # crude plural stripping so "refund" matches "refunds"
    return [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in tokens]


def _hash_embed(text: str) -> list[float]:
    """Lexical feature-hashing embedding (unigrams + bigrams), L2-normalised."""
    dim = settings.EMBEDDING_DIM
    tokens = tokenize(text)
    features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
    vec = np.zeros(dim, dtype=np.float32)
    for feat in features:
        h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "big")
        vec[h % dim] += 1.0 if (h >> 63) & 1 else -1.0
    norm = np.linalg.norm(vec)
    if norm:
        vec /= norm
    return vec.tolist()


@lru_cache(maxsize=1)
def _fastembed_model():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=settings.EMBEDDING_MODEL)


def _check_dim(vectors: list[list[float]]) -> list[list[float]]:
    if vectors and len(vectors[0]) != settings.EMBEDDING_DIM:
        raise RuntimeError(
            f"{settings.EMBEDDING_MODEL} produces {len(vectors[0])}-d vectors but "
            f"EMBEDDING_DIM={settings.EMBEDDING_DIM}; set EMBEDDING_DIM to match "
            "(and re-create the chunk table, since the column width changes)."
        )
    return vectors


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed document chunks. Backend chosen by settings.EMBEDDING_BACKEND."""
    if settings.EMBEDDING_BACKEND == "fastembed":
        return _check_dim([v.tolist() for v in _fastembed_model().passage_embed(texts)])
    return [_hash_embed(t) for t in texts]


def embed_query(text: str) -> list[float]:
    """Embed a search query (some models use a different prefix for queries)."""
    if settings.EMBEDDING_BACKEND == "fastembed":
        return _check_dim([next(iter(_fastembed_model().query_embed(text))).tolist()])[0]
    return _hash_embed(text)
