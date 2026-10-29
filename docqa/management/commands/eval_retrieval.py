import json
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from django.db import transaction

from docqa.services.evaluation import evaluate
from docqa.services.ingest import ingest_document


class Command(BaseCommand):
    help = "Measure retrieval quality (hit-rate@k, MRR) on a labelled dataset. Leaves no data behind."

    def add_arguments(self, parser):
        parser.add_argument("--dataset", default=str(settings.BASE_DIR / "eval" / "dataset.json"))
        parser.add_argument("--backend", choices=["hash", "fastembed"])
        parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
        parser.add_argument("--chunk-size", type=int, help="default: settings.CHUNK_SIZE")
        parser.add_argument("--chunk-overlap", type=int, help="default: settings.CHUNK_OVERLAP")
        parser.add_argument("--show-misses", action="store_true")

    def handle(self, *args, **opts):
        data = json.loads(Path(opts["dataset"]).read_text(encoding="utf-8"))
        if opts["backend"]:
            settings.EMBEDDING_BACKEND = opts["backend"]
        if opts["chunk_size"]:
            settings.CHUNK_SIZE = opts["chunk_size"]
        if opts["chunk_overlap"] is not None:
            settings.CHUNK_OVERLAP = opts["chunk_overlap"]

        with transaction.atomic():
            user = get_user_model().objects.create_user("eval-user")
            for doc in data["documents"]:
                ingest_document(user, doc["title"], doc["text"])
            result = evaluate(user, data["questions"], opts["k"])
            transaction.set_rollback(True)  # evaluation data is never persisted

        self.stdout.write(
            f"backend={settings.EMBEDDING_BACKEND} chunk={settings.CHUNK_SIZE}/{settings.CHUNK_OVERLAP} "
            f"docs={len(data['documents'])} questions={result.n}"
        )
        for k, rate in result.hit_rate.items():
            self.stdout.write(f"  hit-rate@{k}: {rate:.3f}")
        self.stdout.write(f"  MRR: {result.mrr:.3f}")
        if opts["show_misses"]:
            for m in result.misses:
                self.stdout.write(f"  MISS: {m['q']!r} (wanted {m['evidence']!r})")
