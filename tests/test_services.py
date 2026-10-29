import numpy as np
import pytest

from docqa.services.chunking import chunk_text
from docqa.services.embeddings import embed_passages, embed_query


def embed(text):
    return embed_query(text)


def test_chunking_overlaps_and_covers_text():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = chunk_text(text, size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    assert chunks[0].startswith("word0")
    assert chunks[-1].endswith("word199")


def test_chunks_never_start_or_end_mid_word():
    words = [f"w{i}" + "x" * (i % 7) for i in range(300)]
    chunks = chunk_text(" ".join(words), size=120, overlap=30)
    vocab = set(words)
    for c in chunks:
        parts = c.split(" ")
        assert parts[0] in vocab and parts[-1] in vocab, c


def test_chunks_keep_whole_sentences_and_respect_size():
    sentences = [f"Sentence number {i} says something useful." for i in range(40)]
    chunks = chunk_text(" ".join(sentences), size=200, overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 200
        assert c.endswith(".") and c.startswith("Sentence")
    assert chunks[0].startswith("Sentence number 0 ")
    assert chunks[-1].endswith("Sentence number 39 says something useful.")
    # every sentence appears intact in at least one chunk
    for s in sentences:
        assert any(s in c for c in chunks)


def test_consecutive_chunks_overlap_by_a_sentence():
    sentences = [f"Fact {i} is stated here." for i in range(30)]
    chunks = chunk_text(" ".join(sentences), size=120, overlap=60)
    last_of_first = chunks[0].split(". ")[-1]
    assert last_of_first.rstrip(".") in chunks[1]


def test_chunking_empty_and_invalid():
    assert chunk_text("   \n ", 100, 10) == []
    with pytest.raises(ValueError):
        chunk_text("abc", 10, 10)


def test_embedding_is_deterministic_and_normalised():
    a, b = embed("the quick brown fox"), embed("the quick brown fox")
    assert a == b
    assert np.isclose(np.linalg.norm(a), 1.0, atol=1e-5)


def test_similar_text_scores_higher_than_unrelated():
    q = np.array(embed("refund policy for returns"))
    related = np.array(embed("our refund policy allows returns within 30 days"))
    unrelated = np.array(embed("photosynthesis occurs in chloroplasts"))
    assert q @ related > q @ unrelated
