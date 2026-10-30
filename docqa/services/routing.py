from dataclasses import dataclass, field

from django.conf import settings

from docqa.services import llm, retrieval

# Calibrated with `manage.py eval_retrieval --routing` on eval/dataset.json (hybrid search, 500/100
# chunks): with the neural model, on-topic questions scored >= 0.587 and off-topic ones <= 0.536.
# The hash embedder cannot separate them well (see README), so Auto mode wants the neural model.
_DEFAULT_THRESHOLDS = {"fastembed": 0.56, "hash": 0.12}

MODES = ("auto", "documents", "general")


def sources_payload(hits) -> list[dict]:
    return [
        {
            "n": i,
            "document_id": h.chunk.document_id,
            "document_title": h.chunk.document.title,
            "chunk_index": h.chunk.index,
            "score": round(h.score, 4),
            "text": h.chunk.text,
        }
        for i, h in enumerate(hits, start=1)
    ]


def relevance_threshold() -> float:
    if settings.RELEVANCE_THRESHOLD is not None:
        return settings.RELEVANCE_THRESHOLD
    return _DEFAULT_THRESHOLDS.get(settings.EMBEDDING_BACKEND, 0.56)


@dataclass
class Plan:
    """What to do with a question: which sources to show and how to answer."""

    mode: str  # "documents" (grounded), "general", or "not_found" (documents-only, nothing relevant)
    sources: list[dict] = field(default_factory=list)
    passages: list[str] | None = None  # grounded when set
    message: str | None = None  # fixed reply instead of calling the model
    searched_for: str | None = None  # the rewritten query, when a follow-up was rewritten
    best_similarity: float | None = None
    # Auto only: the model may decline when the passages turn out not to answer the question, and the
    # caller then switches to a general answer (a similarity score alone can't tell on real documents)
    can_decline: bool = False


def plan_answer(
    user, question: str, mode: str, history: list[dict], top_k: int, document_id: int | None
) -> Plan:
    """Decide how a question is answered.

    - general: skip the documents and answer like a normal assistant.
    - documents: answer only from relevant passages; otherwise say they weren't found.
    - auto: use the documents when something relevant exists, otherwise answer from general knowledge.
    """
    if mode == "general":
        return Plan(mode="general")

    query = llm.condense_question(question, history)
    result = retrieval.search_ex(user, query, top_k, document_id)
    searched_for = query if query != question else None

    if result.hits and result.best_similarity >= relevance_threshold():
        return Plan(
            mode="documents",
            sources=sources_payload(result.hits),
            passages=[h.chunk.text for h in result.hits],
            searched_for=searched_for,
            best_similarity=result.best_similarity,
            can_decline=mode == "auto",
        )
    if mode == "documents":
        message = llm.NOT_IN_DOCUMENTS if result.hits else llm.NO_RESULTS
        return Plan(mode="not_found", message=message, searched_for=searched_for, best_similarity=result.best_similarity)
    return Plan(mode="general", searched_for=searched_for, best_similarity=result.best_similarity)
