import logging
import time

from django.conf import settings
from django.core.management.base import BaseCommand

from docqa.services import ingest


class Command(BaseCommand):
    help = (
        "Index uploaded documents that are waiting in the queue. Run it as its own process when the web servers "
        "have INGEST_EMBEDDED_WORKER=0; several can run at once, and a document is never indexed twice."
    )

    def add_arguments(self, parser):
        parser.add_argument("--once", action="store_true", help="index what is waiting now, then exit")

    def handle(self, *args, **opts):
        logging.getLogger("docqa").setLevel(logging.INFO)
        if opts["once"]:
            self.stdout.write(f"indexed {ingest.run_pending()} document(s)")
            return
        self.stdout.write("waiting for documents to index (Ctrl+C to stop)")
        try:
            while True:
                if not ingest.run_pending():
                    time.sleep(settings.INGEST_POLL_SECONDS)
        except KeyboardInterrupt:
            self.stdout.write("stopped")
