import json
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from django.db import transaction

from docqa.services.evaluation import evaluate, routing_report
from docqa.services.ingest import ingest_document


class Command(BaseCommand):
    help = (
        "Measure retrieval quality (hit-rate@k, MRR) and answer-routing on a labelled dataset. "
        "Leaves no data behind."
    )

    def add_arguments(self, parser):
        parser.add_argument("--dataset", default=str(settings.BASE_DIR / "eval" / "dataset.json"))
        parser.add_argument("--backend", choices=["hash", "fastembed"])
        parser.add_argument("--search", choices=["vector", "hybrid"], help="default: settings.HYBRID_SEARCH")
        parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
        parser.add_argument("--chunk-size", type=int, help="default: settings.CHUNK_SIZE")
        parser.add_argument("--chunk-overlap", type=int, help="default: settings.CHUNK_OVERLAP")
        parser.add_argument("--show-misses", action="store_true")
        parser.add_argument("--routing", action="store_true", help="also calibrate the relevance cutoff")

    def handle(self, *args, **opts):
        data = json.loads(Path(opts["dataset"]).read_text(encoding="utf-8"))
        if opts["backend"]:
            settings.EMBEDDING_BACKEND = opts["backend"]
        if opts["chunk_size"]:
            settings.CHUNK_SIZE = opts["chunk_size"]
        if opts["chunk_overlap"] is not None:
            settings.CHUNK_OVERLAP = opts["chunk_overlap"]
        hybrid = None if not opts["search"] else opts["search"] == "hybrid"
        if hybrid is not None:
            settings.HYBRID_SEARCH = hybrid

        report = None
        with transaction.atomic():
            user = get_user_model().objects.create_user("eval-user")
            for doc in data["documents"]:
                ingest_document(user, doc["title"], doc["text"])
            result = evaluate(user, data["questions"], opts["k"])
            kinds = sorted({q.get("kind") for q in data["questions"]} - {None})
            by_kind = (
                {k: evaluate(user, [q for q in data["questions"] if q.get("kind") == k], opts["k"]) for k in kinds}
                if len(kinds) > 1
                else {}
            )
            if opts["routing"] and data.get("off_topic"):
                report = routing_report(user, [q["q"] for q in data["questions"]], data["off_topic"])
            transaction.set_rollback(True)  # evaluation data is never persisted

        mode = "hybrid" if settings.HYBRID_SEARCH else "vector"
        self.stdout.write(
            f"backend={settings.EMBEDDING_BACKEND} search={mode} "
            f"chunk={settings.CHUNK_SIZE}/{settings.CHUNK_OVERLAP} "
            f"docs={len(data['documents'])} questions={result.n}"
        )
        for k, rate in result.hit_rate.items():
            self.stdout.write(f"  hit-rate@{k}: {rate:.3f}")
        self.stdout.write(f"  MRR: {result.mrr:.3f}")
        for kind, part in by_kind.items():
            self.stdout.write(
                f"    {kind} ({part.n}): hit-rate@1 {part.hit_rate[1]:.3f}, "
                f"hit-rate@{max(part.hit_rate)} {part.hit_rate[max(part.hit_rate)]:.3f}, MRR {part.mrr:.3f}"
            )
        if opts["show_misses"]:
            for m in result.misses:
                self.stdout.write(f"  MISS: {m['q']!r} (wanted {m['evidence']!r})")
        if report:
            on, off = report.on_topic_scores, report.off_topic_scores
            self.stdout.write(
                f"  routing: best cutoff {report.threshold:.3f} -> "
                f"{report.on_topic_recall:.0%} of on-topic questions use the documents, "
                f"{report.off_topic_rejection:.0%} of off-topic questions are rejected"
            )
            self.stdout.write(
                f"  best similarity, on-topic: min {on[0]:.3f} median {on[len(on) // 2]:.3f} max {on[-1]:.3f}"
            )
            self.stdout.write(
                f"  best similarity, off-topic: min {off[0]:.3f} median {off[len(off) // 2]:.3f} max {off[-1]:.3f}"
            )
