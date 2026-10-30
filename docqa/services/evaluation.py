from dataclasses import dataclass, field

from docqa.services.retrieval import search, search_ex


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


@dataclass
class EvalResult:
    n: int
    hit_rate: dict[int, float]
    mrr: float
    misses: list[dict] = field(default_factory=list)  # questions with no hit within max(k)


@dataclass
class RoutingReport:
    threshold: float  # best similarity cutoff separating on-topic from off-topic questions
    on_topic_recall: float  # share of on-topic questions at or above the cutoff (answered from documents)
    off_topic_rejection: float  # share of off-topic questions below it (sent to general knowledge)
    on_topic_scores: list[float]
    off_topic_scores: list[float]

    @property
    def balanced_accuracy(self) -> float:
        return (self.on_topic_recall + self.off_topic_rejection) / 2


def best_similarity(user, question: str, top_k: int = 4, hybrid: bool | None = None) -> float:
    return search_ex(user, question, top_k=top_k, hybrid=hybrid).best_similarity


def routing_report(user, on_topic: list[str], off_topic: list[str], hybrid: bool | None = None) -> RoutingReport:
    """Find the similarity cutoff that best tells "answerable from the documents" from "unrelated".

    A question is routed to the documents when its best passage scores at or above the cutoff.
    The returned cutoff maximises balanced accuracy, and sits midway between neighbouring scores.
    """
    on = sorted(best_similarity(user, q, hybrid=hybrid) for q in on_topic)
    off = sorted(best_similarity(user, q, hybrid=hybrid) for q in off_topic)
    values = sorted(set(on + off))
    best = None
    for lo, hi in zip(values, values[1:] + [values[-1] + 0.02]):
        cutoff = (lo + hi) / 2
        recall = sum(s >= cutoff for s in on) / len(on)
        rejection = sum(s < cutoff for s in off) / len(off)
        score = (recall + rejection) / 2
        if best is None or score > best[0]:
            best = (score, cutoff, recall, rejection)
    _, cutoff, recall, rejection = best
    return RoutingReport(cutoff, recall, rejection, on, off)


def evaluate(user, questions: list[dict], ks: list[int], hybrid: bool | None = None) -> EvalResult:
    """Score retrieval: a hit is a retrieved chunk containing the question's evidence span.

    hit_rate[k] = fraction of questions with a hit in the top k;
    MRR = mean of 1/rank of the first hit (0 if none within max(k)).
    """
    ks = sorted(ks)
    hits_at = {k: 0 for k in ks}
    reciprocal_ranks = []
    misses = []
    for item in questions:
        evidence = _norm(item["evidence"])
        ranked = search(user, item["q"], top_k=ks[-1], hybrid=hybrid)
        rank = next(
            (i for i, h in enumerate(ranked, start=1) if evidence in _norm(h.chunk.text)), None
        )
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        for k in ks:
            if rank is not None and rank <= k:
                hits_at[k] += 1
        if rank is None:
            misses.append(
                {"q": item["q"], "evidence": item["evidence"], "top": ranked[0].chunk.text if ranked else ""}
            )
    n = len(questions)
    return EvalResult(
        n=n,
        hit_rate={k: hits_at[k] / n for k in ks},
        mrr=sum(reciprocal_ranks) / n,
        misses=misses,
    )
