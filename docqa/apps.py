import logging
import os
import sys
import threading

from django.apps import AppConfig
from django.conf import settings

log = logging.getLogger(__name__)


class DocqaConfig(AppConfig):
    name = "docqa"
    default_auto_field = "django.db.models.BigAutoField"

    def ready(self):
        """Optionally load the neural embedding model and the model client in the background at start.

        Without this, the first question after every start waits 10-20 s for the model to load.
        Off by default (tests, migrations and shell sessions should not pay for it); the launch
        scripts and docker-compose turn it on with WARMUP_EMBEDDINGS=1.
        """
        if not settings.WARMUP_EMBEDDINGS or settings.EMBEDDING_BACKEND != "fastembed":
            return
        # `manage.py runserver` with auto-reload runs this twice; only the process that serves needs the model
        reloader_parent = (
            "runserver" in sys.argv and "--noreload" not in sys.argv and os.environ.get("RUN_MAIN") != "true"
        )
        if reloader_parent:
            return
        threading.Thread(target=_warm_up, name="embedding-warm-up", daemon=True).start()


def _warm_up():
    from docqa.services import embeddings, llm

    seconds = embeddings.warm_up()
    if seconds is not None:
        log.info("embedding model ready in %.1fs", seconds)
    seconds = llm.warm_up()
    if seconds is not None:
        log.info("language model client ready in %.1fs", seconds)
