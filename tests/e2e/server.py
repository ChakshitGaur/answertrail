"""Browser end-to-end harness: the real app on a throwaway database, wired to the fake streaming model.

    python tests/e2e/server.py        # then open http://127.0.0.1:8010

Uses the real neural embedder (so relevance routing is genuine) but a fake model server, so no key,
quota or network is needed. Change the fake model at any time by writing "primary|fallback1,fallback2"
to tests/e2e/model.txt (see tests/fake_llm.py for the names: busy, denied, cut, slow, decline, markdown...).
See tests/e2e/README.md for the checklist.
"""

import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
os.chdir(ROOT)
sys.path[:0] = [str(ROOT), str(HERE)]
os.environ["DJANGO_SETTINGS_MODULE"] = "config.settings"
os.environ["EMBEDDING_BACKEND"] = "fastembed"
os.environ["WARMUP_EMBEDDINGS"] = "1"
os.environ["DATABASE_PATH"] = str(Path(tempfile.mkdtemp(prefix="rag-e2e-")) / "db.sqlite3")
os.environ.pop("ANTHROPIC_API_KEY", None)

from tests import fake_llm  # noqa: E402

fake_llm.start(8765)
os.environ.update(
    LLM_BASE_URL="http://127.0.0.1:8765/v1", LLM_API_KEY="e2e-key", LLM_MODEL="good", LLM_FALLBACK_MODELS="",
    LLM_REASONING_EFFORT="minimal", LLM_REASONING_EFFORT_GENERAL="minimal",
)

from django.conf import settings  # noqa: E402

settings.MIDDLEWARE = list(settings.MIDDLEWARE) + ["model_switch.Switch"]

from django.core.management import execute_from_command_line  # noqa: E402

execute_from_command_line(["manage.py", "migrate", "-v0"])
execute_from_command_line(["manage.py", "runserver", "127.0.0.1:8010", "--noreload"])
