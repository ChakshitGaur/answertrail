import pytest

from docqa.models import DailyUsage
from tests.test_api import make_client, upload_text

pytestmark = pytest.mark.django_db

REFUNDS = "Refunds are accepted within 30 days of purchase."


@pytest.fixture
def client():
    c = make_client()
    upload_text(c, "Refunds", REFUNDS)
    return c


def ask(client, url="/api/ask/", **extra):
    return client.post(url, {"question": "What is the refund window?", **extra}, format="json")


def test_questions_over_the_daily_limit_are_refused_with_429(client, llm_settings, settings):
    settings.DAILY_MODEL_QUESTIONS = 2
    assert ask(client).status_code == 200
    assert ask(client).status_code == 200
    resp = ask(client)
    assert resp.status_code == 429 and "daily question limit" in resp.json()["detail"].lower()
    assert DailyUsage.objects.get().questions == 2  # the refused one was not counted


def test_the_streaming_endpoint_shares_the_same_limit(client, llm_settings, settings):
    settings.DAILY_MODEL_QUESTIONS = 1
    first = ask(client, "/api/ask/stream/")
    b"".join(first.streaming_content)
    assert first.status_code == 200
    assert ask(client, "/api/ask/stream/").status_code == 429
    assert ask(client).status_code == 429


def test_questions_that_do_not_call_the_model_are_free(client, llm_settings, settings):
    settings.DAILY_MODEL_QUESTIONS = 1
    settings.RELEVANCE_THRESHOLD = 0.99  # nothing counts as relevant
    for _ in range(3):  # documents-only and nothing relevant: answered without the model
        resp = client.post("/api/ask/", {"question": "zzzz qqqq", "mode": "documents"}, format="json")
        assert resp.status_code == 200
    assert not DailyUsage.objects.exists()


def test_no_model_configured_means_nothing_is_counted(client, settings):
    settings.DAILY_MODEL_QUESTIONS = 1
    for _ in range(3):
        assert ask(client).status_code == 200  # extractive fallback, no model call
    assert not DailyUsage.objects.exists()


def test_zero_means_unlimited(client, llm_settings, settings):
    settings.DAILY_MODEL_QUESTIONS = 0
    for _ in range(4):
        assert ask(client).status_code == 200
    assert not DailyUsage.objects.exists()


def test_each_user_has_their_own_allowance(llm_settings, settings):
    settings.DAILY_MODEL_QUESTIONS = 1
    a, b = make_client("alice"), make_client("bob")
    for c in (a, b):
        upload_text(c, "Refunds", REFUNDS)
    assert ask(a).status_code == 200 and ask(a).status_code == 429
    assert ask(b).status_code == 200


def test_a_new_day_starts_a_new_count(client, llm_settings, settings):
    settings.DAILY_MODEL_QUESTIONS = 1
    assert ask(client).status_code == 200 and ask(client).status_code == 429
    import datetime

    DailyUsage.objects.update(day=datetime.date(2000, 1, 1))  # yesterday's row no longer counts
    assert ask(client).status_code == 200
