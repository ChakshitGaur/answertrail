import os
from functools import lru_cache

import pytest


@lru_cache(maxsize=1)
def _load_error():
    """None if the neural embedding model can be loaded here, else why not (checked once per test run)."""
    try:
        import fastembed  # noqa: F401
    except ImportError:
        return "fastembed is not installed"
    from docqa.services import embeddings

    try:
        embeddings._fastembed_model()
    except Exception as exc:  # no network to download the model, a corrupt cache, ...
        return f"the neural model could not be loaded: {exc}"
    return None


def require_neural_model():
    """Skip a test that needs the neural model when it is unavailable.

    On CI (CI=true) a missing model is a failure, not a skip: the neural tests must really run there.
    """
    problem = _load_error()
    if problem is None:
        return
    if os.environ.get("CI") and "not installed" not in problem:
        pytest.fail(problem)
    pytest.skip(problem)
