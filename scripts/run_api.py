"""Start the Django API with the neural embedding model. Used by the preview/launch config."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
os.environ.setdefault("EMBEDDING_BACKEND", "fastembed")

from django.core.management import execute_from_command_line

execute_from_command_line(["manage.py", "migrate", "-v0"])
port = os.environ.get("PORT", "8000")  # preview tools and hosts can assign a free port
execute_from_command_line(["manage.py", "runserver", f"127.0.0.1:{port}", "--noreload"])
