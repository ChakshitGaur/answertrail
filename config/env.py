import os
from pathlib import Path


def load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ without overriding real variables.

    Precedence is: variables already set in the environment, then .env, then the defaults in settings.
    Supports comments, `export KEY=...`, and single or double quotes. The file is never read by
    anything else and is listed in .gitignore.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key and value:  # an empty value (e.g. `LLM_API_KEY=`) means "not set"
            os.environ.setdefault(key, value)
