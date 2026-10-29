from dataclasses import dataclass, field

from docqa.services.retrieval import search


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


@dataclass
class EvalResult:
    n: int
    hit_rate: dict[int, float]
    mrr: float
    misses: list[dict] = field(default_factory=list)  # questions with no hit within max(k)


def evaluate(user, questions: list[dict], ks: list[int]) -> EvalResult:
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
        ranked = search(user, item["q"], top_k=ks[-1])
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
