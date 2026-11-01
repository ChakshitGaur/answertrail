import re
import time
from dataclasses import dataclass, field

from django.conf import settings

from docqa.services import llm, retrieval

# Calibrated with `manage.py eval_retrieval --routing` on eval/dataset.json (hybrid search, 500/100
# chunks): with the neural model, on-topic questions scored >= 0.587 and off-topic ones <= 0.536.
# The hash embedder cannot separate them well (see README), so Auto mode wants the neural model.
_DEFAULT_THRESHOLDS = {"fastembed": 0.56, "hash": 0.12}

MODES = ("auto", "documents", "general")

# "Summarize this document", "what are the key points?", "main topics covered": requests about the
# document as a whole, which no single passage resembles.
_OVERVIEW_TRIGGER = re.compile(
    r"\b(summari[sz]e|summari[sz]ation|summary|overview|tl;?dr|key points?|main points?|main topics?|main ideas?|"
    r"table of contents|outline|what(?:'s| is| are) (?:this|these|the|my)\b.{0,30}\b(?:about|on)\b|"
    r"what (?:does|do) (?:this|the|my)\b.{0,30}\b(?:say|cover|contain|include)s?)\b",
    re.IGNORECASE,
)
# Words that may accompany an overview request without naming a topic. Anything else ("...the refund
# policy") means the user wants that topic, which similarity search answers better.
_OVERVIEW_FILLER = frozenset(
    "a an and are about all briefly can could covered covers cover contain contains content contents doc docs document "
    "documents does do entire everything file files for from give has have i in include includes is it its key list "
    "main me my of on or outline overview paper papers pdf pdfs please points point quick quickly report reports say "
    "says short shortly simple simply sentences sentence summarise summarize summarised summarized summary summarization "
    "table tell text texts that the these this those to tl tldr topic topics dr what whats s re which whole with would you your "
    "bullet bullets few two three five ten one idea ideas".split()
)


def is_overview_request(question: str) -> bool:
    """True for questions about the document as a whole ("summarize this", "what are the key points?")."""
    text = question.strip().lower()
    if not _OVERVIEW_TRIGGER.search(text):
        return False
    words = re.findall(r"[a-z0-9]+", text)
    return not [w for w in words if w not in _OVERVIEW_FILLER and not w.isdigit()]


def sources_payload(hits) -> list[dict]:
    return [
        {
            "n": i,
            "document_id": h.chunk.document_id,
            "document_title": h.chunk.document.title,
            "chunk_index": h.chunk.index,
            "score": None if h.score is None else round(h.score, 4),
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
    overview: bool = False  # the passages are a spread sample of the document(s), not search results
    timings: dict[str, int] = field(default_factory=dict)  # milliseconds spent rewriting and searching


def plan_answer(
    user, question: str, mode: str, history: list[dict], top_k: int, document_id: int | None
) -> Plan:
    """Decide how a question is answered.

    - general: skip the documents and answer like a normal assistant.
    - documents: answer only from relevant passages; otherwise say they weren't found.
    - auto: use the documents when something relevant exists, otherwise answer from general knowledge.
    A request for a summary or overview reads a spread sample of the documents instead of searching.
    """
    if mode == "general":
        return Plan(mode="general")

    started = time.perf_counter()
    if is_overview_request(question):
        hits = retrieval.overview_hits(user, document_id, max_passages=max(top_k, 8))
        if hits:
            elapsed = round((time.perf_counter() - started) * 1000)
            return Plan(
                mode="documents",
                sources=sources_payload(hits),
                passages=[h.chunk.text for h in hits],
                overview=True,
                timings={"rewrite_ms": 0, "search_ms": elapsed},
            )

    query = llm.condense_question(question, history)
    rewritten = time.perf_counter()
    result = retrieval.search_ex(user, query, top_k, document_id)
    done = time.perf_counter()
    searched_for = query if query != question else None
    timings = {"rewrite_ms": round((rewritten - started) * 1000), "search_ms": round((done - rewritten) * 1000)}

    if result.hits and result.best_similarity >= relevance_threshold():
        return Plan(
            mode="documents",
            sources=sources_payload(result.hits),
            passages=[h.chunk.text for h in result.hits],
            searched_for=searched_for,
            best_similarity=result.best_similarity,
            can_decline=mode == "auto",
            timings=timings,
        )
    if mode == "documents":
        message = llm.NOT_IN_DOCUMENTS if result.hits else llm.NO_RESULTS
        return Plan(
            mode="not_found",
            message=message,
            searched_for=searched_for,
            best_similarity=result.best_similarity,
            timings=timings,
        )
    return Plan(mode="general", searched_for=searched_for, best_similarity=result.best_similarity, timings=timings)
