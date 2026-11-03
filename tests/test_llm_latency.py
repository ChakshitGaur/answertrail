import io

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError


def run(*args, **kwargs):
    out = io.StringIO()
    call_command("llm_latency", *args, stdout=out, pause=0, **kwargs)
    return out.getvalue()


def test_reports_each_model_and_ranks_them(llm_settings, settings):
    settings.LLM_MODEL, settings.LLM_FALLBACK_MODELS = "good", "split"
    text = run(runs=2)
    assert "good" in text and "split" in text and "first word: median" in text
    assert text.count(" 2  ") >= 2  # both models answered both calls
    assert "fastest first word first:" in text
    assert len([r for r in llm_settings.requests if r["body"].get("stream")]) == 4  # real streaming calls


def test_a_failing_model_is_reported_and_not_asked_again(llm_settings):
    text = run(models="busy-1,good", runs=4)
    busy_calls = [r for r in llm_settings.requests if r["body"]["model"] == "busy-1"]
    assert len(busy_calls) == 2  # two failures and it is dropped: no quota is burned on it
    assert "no successful call" in text and "503" in text
    assert "fastest first word first: good" in text


def test_a_400_comes_with_a_hint(llm_settings):
    text = run(models="always400", runs=2)
    assert "HTTP 400 usually means" in text


def test_every_model_failing_says_what_to_check(llm_settings):
    assert "no model answered" in run(models="denied-1", runs=2)


def test_models_can_be_listed_explicitly(llm_settings):
    run(models="good", runs=1)
    assert {r["body"]["model"] for r in llm_settings.requests} == {"good"}


def test_another_server_can_be_measured_with_a_key_from_the_environment(llm_settings, settings, monkeypatch):
    base = settings.LLM_BASE_URL
    settings.LLM_BASE_URL, settings.LLM_API_KEY = "", ""
    monkeypatch.setenv("OTHER_PROVIDER_KEY", "sekret")
    text = run(models="good", runs=1, base_url=base, key_env="OTHER_PROVIDER_KEY")
    assert llm_settings.requests[0]["auth"] == "Bearer sekret"
    assert "sekret" not in text  # the key is never printed


def test_a_missing_key_variable_is_a_clear_error(llm_settings, monkeypatch):
    monkeypatch.delenv("NOPE_KEY", raising=False)
    with pytest.raises(CommandError, match="NOPE_KEY"):
        run(models="good", base_url="http://x/v1", key_env="NOPE_KEY")


def test_needs_a_server_and_a_model(settings):
    settings.LLM_BASE_URL = ""
    with pytest.raises(CommandError, match="LLM_BASE_URL"):
        run()
    settings.LLM_BASE_URL, settings.LLM_MODEL = "http://x/v1", ""
    with pytest.raises(CommandError, match="no model"):
        run()
    with pytest.raises(CommandError, match="--runs"):
        run(models="good", runs=0)
