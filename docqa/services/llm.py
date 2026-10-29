import json
import time
from collections.abc import Iterator

import requests
from django.conf import settings

_ATTEMPTS = 2  # per model
_BACKOFF_SECONDS = 2
_TIMEOUT = (10, 30)  # (connect, read) seconds per attempt
_BUDGET_SECONDS = 100  # stop trying new attempts after this, so the UI never times out first
_RETRYABLE = {429, 500, 502, 503, 504}
_TRY_NEXT_MODEL = {400, 404}  # unknown model name (Google also uses 400 for an invalid key)

FALLBACK_PREFIX = "[extractive fallback, no LLM configured] "
NO_RESULTS = "I couldn't find anything relevant in your documents."

SYSTEM_PROMPT = (
    "You answer questions using only the numbered context passages provided. "
    "Cite passages inline like [1]. If the context does not contain the answer, "
    "say you don't know."
)


class LLMError(RuntimeError):
    """The language model backend was unreachable or returned an error."""


def _snippet(resp) -> str:
    """The provider's own error text, which usually says exactly what is wrong."""
    try:
        return " ".join(resp.text.split())[:300]
    except Exception:
        return ""


def _build_prompt(question: str, passages: list[str]) -> str:
    context = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, start=1))
    return f"Context:\n{context}\n\nQuestion: {question}"


def _models() -> list[str]:
    """LLM_MODEL first, then LLM_FALLBACK_MODELS (comma separated), without duplicates."""
    names = [settings.LLM_MODEL, *settings.LLM_FALLBACK_MODELS.split(",")]
    return list(dict.fromkeys(n.strip() for n in names if n.strip()))


def _open_chat(question: str, passages: list[str], stream: bool) -> requests.Response:
    """POST /chat/completions on an OpenAI-compatible server (Ollama, Groq, Gemini, OpenRouter...).

    An overloaded model (503/429) is retried briefly, then the next model in the list is tried.
    A model the server doesn't know (404/400) moves on to the next one straight away. Auth
    errors (401/403) stop immediately, since another model will not fix them. Returns the first
    successful response; nothing has been read from its body yet.
    """
    models = _models()
    if not models:
        raise LLMError("LLM_BASE_URL is set but LLM_MODEL is empty")
    headers = {"Authorization": f"Bearer {settings.LLM_API_KEY}"} if settings.LLM_API_KEY else {}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_prompt(question, passages)},
    ]
    url = f"{settings.LLM_BASE_URL.rstrip('/')}/chat/completions"
    deadline = time.monotonic() + _BUDGET_SECONDS
    last_error = "no attempt made"
    tried = []
    for model in models:
        tried.append(model)
        body = {"model": model, "max_tokens": 800, "temperature": 0.2, "messages": messages}
        if stream:
            body["stream"] = True
        for attempt in range(_ATTEMPTS):
            if time.monotonic() > deadline:
                raise LLMError(f"ran out of time. Tried {', '.join(tried)}. Last error: {last_error}")
            if attempt:
                time.sleep(_BACKOFF_SECONDS * attempt)
            try:
                resp = requests.post(url, headers=headers, json=body, timeout=_TIMEOUT, stream=stream)
            except requests.RequestException as exc:
                last_error = f"{model}: {exc.__class__.__name__}: {exc}"
                continue
            if resp.status_code in _RETRYABLE:  # overloaded / rate limited: worth another try
                last_error = f"{model}: HTTP {resp.status_code}: {_snippet(resp)}"
                resp.close()
                continue
            if resp.status_code in _TRY_NEXT_MODEL:
                last_error = f"{model}: HTTP {resp.status_code}: {_snippet(resp)}"
                resp.close()
                break
            if not resp.ok:
                detail = f"{model}: HTTP {resp.status_code}: {_snippet(resp)}"
                resp.close()
                raise LLMError(detail)
            return resp
    raise LLMError(f"all models failed ({', '.join(tried)}). Last error: {last_error}")


def _openai_compatible(question: str, passages: list[str]) -> str:
    resp = _open_chat(question, passages, stream=False)
    try:
        return resp.json()["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise LLMError(f"unexpected response ({exc.__class__.__name__}): {_snippet(resp)}") from exc


def _stream_openai_compatible(question: str, passages: list[str]) -> Iterator[str]:
    resp = _open_chat(question, passages, stream=True)
    resp.encoding = "utf-8"  # event streams carry no charset; the default would garble non-ASCII
    produced = False
    try:
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
            except (ValueError, KeyError, IndexError, TypeError):
                continue  # keep-alives or provider-specific frames
            if delta:
                produced = True
                yield delta
    except requests.RequestException as exc:
        raise LLMError(f"stream interrupted ({exc.__class__.__name__}): {exc}") from exc
    finally:
        resp.close()
    if not produced:
        raise LLMError("the model returned an empty answer")


def _anthropic(question: str, passages: list[str]) -> str:
    return "".join(_stream_anthropic(question, passages))


def _stream_anthropic(question: str, passages: list[str]) -> Iterator[str]:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    try:
        with client.messages.stream(
            model=settings.ANSWER_MODEL,
            max_tokens=800,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_prompt(question, passages)}],
        ) as stream:
            yield from stream.text_stream
    except anthropic.APIError as exc:
        raise LLMError(f"{exc.__class__.__name__}: {exc}") from exc


def answer(question: str, passages: list[str]) -> str:
    """Write an answer from the passages with the configured model.

    Order: an OpenAI-compatible endpoint (LLM_BASE_URL), then Anthropic (ANTHROPIC_API_KEY),
    otherwise return the best passage verbatim so the app still works offline and free.
    """
    if not passages:
        return NO_RESULTS
    if settings.LLM_BASE_URL:
        return _openai_compatible(question, passages)
    if settings.ANTHROPIC_API_KEY:
        return _anthropic(question, passages)
    return FALLBACK_PREFIX + passages[0]


def stream_answer(question: str, passages: list[str]) -> Iterator[str]:
    """Like `answer`, but yields the text in pieces as the model writes it.

    Failures before the first piece (overload, bad key, unknown model) are retried / failed over
    exactly as in `answer`. A failure after output has started raises LLMError mid-iteration.
    """
    if not passages:
        yield NO_RESULTS
    elif settings.LLM_BASE_URL:
        yield from _stream_openai_compatible(question, passages)
    elif settings.ANTHROPIC_API_KEY:
        yield from _stream_anthropic(question, passages)
    else:
        yield FALLBACK_PREFIX + passages[0]
