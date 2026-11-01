import json
import logging
import re
import time
from collections.abc import Iterator

import requests
from django.conf import settings

log = logging.getLogger(__name__)

_ATTEMPTS = 2  # per model
_BACKOFF_SECONDS = 2
_TIMEOUT = (10, 30)  # (connect, read) seconds per attempt
_BUDGET_SECONDS = 100  # stop trying new attempts after this, so the UI never times out first
_RETRYABLE = {429, 500, 502, 503, 504}
_TRY_NEXT_MODEL = {400, 404}  # unknown model name (Google also uses 400 for an invalid key)

FALLBACK_PREFIX = "[extractive fallback, no LLM configured] "
NO_RESULTS = "I couldn't find anything relevant in your documents."
NOT_IN_DOCUMENTS = (
    "I couldn't find that in your documents. Switch the answer mode to Auto or General "
    "if you want an answer from general knowledge."
)
NO_MODEL_FOR_GENERAL = (
    "That doesn't look like it's covered by your documents, and answering from general "
    "knowledge needs a language model, which isn't configured on this server."
)

SYSTEM_PROMPT = (
    "You answer questions using only the numbered context passages provided. "
    "Cite passages inline like [1]. If the context does not contain the answer, "
    "say you don't know."
)
NOT_IN_DOCUMENTS_SENTINEL = "NOT_IN_DOCUMENTS"
DECLINE_INSTRUCTION = (
    f" If the passages do not actually answer the question, reply with exactly "
    f"{NOT_IN_DOCUMENTS_SENTINEL} and nothing else."
)
GENERAL_PROMPT = (
    "You are a helpful, accurate assistant. Answer clearly and concisely. "
    "If you are not sure of something, say so rather than guessing."
)
CONDENSE_PROMPT = (
    "Rewrite the user's last message as a standalone search query that makes sense without the "
    "conversation: resolve pronouns and references like 'it' or 'and the second one'. "
    "Reply with only the query, no explanation. If the message is already standalone, repeat it."
)

HISTORY_MAX_MESSAGES = 8  # the last four exchanges
HISTORY_MAX_CHARS = 1500  # per message


class LLMError(RuntimeError):
    """The language model backend was unreachable or returned an error."""


class NotInDocuments(Exception):
    """The model looked at the passages and said they don't answer the question.

    Only raised when asked to allow it (Auto mode), and before any text has been yielded, so the
    caller can switch to a general answer without the user ever seeing the discarded one.
    """


def declined(text: str) -> bool:
    return re.match(rf"^\W*{NOT_IN_DOCUMENTS_SENTINEL}", text, re.IGNORECASE) is not None


def is_configured() -> bool:
    return bool(settings.LLM_BASE_URL or settings.ANTHROPIC_API_KEY)


def _snippet(resp) -> str:
    """The provider's own error text, which usually says exactly what is wrong."""
    try:
        return " ".join(resp.text.split())[:300]
    except Exception:
        return ""


def _build_prompt(question: str, passages: list[str]) -> str:
    context = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, start=1))
    return f"Context:\n{context}\n\nQuestion: {question}"


def clean_history(history: list[dict] | None) -> list[dict]:
    """Keep the recent conversation in a shape every provider accepts.

    Only user/assistant text is kept, each message is trimmed, consecutive messages from the same
    role are merged, and the list starts with a user message (Anthropic requires alternation).
    """
    cleaned: list[dict] = []
    for item in (history or [])[-HISTORY_MAX_MESSAGES:]:
        role, text = item.get("role"), str(item.get("content", "")).strip()
        if role not in ("user", "assistant") or not text:
            continue
        text = text[:HISTORY_MAX_CHARS]
        if cleaned and cleaned[-1]["role"] == role:
            cleaned[-1]["content"] += "\n" + text
        else:
            cleaned.append({"role": role, "content": text})
    while cleaned and cleaned[0]["role"] != "user":
        cleaned.pop(0)
    return cleaned


def build_messages(
    question: str, passages: list[str] | None, history: list[dict] | None, allow_decline: bool = False
) -> list[dict]:
    """System prompt, recent conversation, then the question (with numbered passages when grounded).

    With `allow_decline`, the model may reply NOT_IN_DOCUMENTS when the passages don't answer it.
    """
    grounded = bool(passages)
    prior = clean_history(history)
    if prior and prior[-1]["role"] == "user":  # keep strict alternation before the new question
        prior.pop()
    system = (SYSTEM_PROMPT + (DECLINE_INSTRUCTION if allow_decline else "")) if grounded else GENERAL_PROMPT
    return [
        {"role": "system", "content": system},
        *prior,
        {"role": "user", "content": _build_prompt(question, passages) if grounded else question},
    ]


def _models() -> list[str]:
    """LLM_MODEL first, then LLM_FALLBACK_MODELS (comma separated), without duplicates."""
    names = [settings.LLM_MODEL, *settings.LLM_FALLBACK_MODELS.split(",")]
    return list(dict.fromkeys(n.strip() for n in names if n.strip()))


_NO_EFFORT: set[str] = set()  # models that rejected reasoning_effort; not asked again this process

# Words that usually mean "this question depends on the conversation so far"
_REFERENCES = frozenset(
    "it its this that these those he she they them his her their him there here also too another other "
    "former latter above same previous earlier more else instead then".split()
)
_FOLLOW_UP_OPENERS = ("and ", "but ", "so ", "what about", "how about", "what else", "why ", "how come")


def looks_like_followup(question: str) -> bool:
    """Cheap check for whether a question needs the chat history to make sense.

    Rewriting costs a model call, so a long, self-contained question is searched as typed. Short
    questions, ones starting like a continuation, or ones with a reference word ("it", "he", "that")
    are rewritten. Errs on the side of rewriting.
    """
    text = question.strip().lower()
    words = re.findall(r"[a-z0-9']+", text)
    return (
        len(words) <= 8
        or text.startswith(_FOLLOW_UP_OPENERS)
        or any(w in _REFERENCES for w in words)
    )


def _effort_for(grounded: bool) -> str:
    return settings.LLM_REASONING_EFFORT if grounded else settings.LLM_REASONING_EFFORT_GENERAL


def _open_chat(messages: list[dict], stream: bool, effort: str = "") -> requests.Response:
    """POST /chat/completions on an OpenAI-compatible server (Ollama, Groq, Gemini, OpenRouter...).

    An overloaded model (503/429) is retried briefly, then the next model in the list is tried.
    A model the server doesn't know (404/400) moves on to the next one straight away, and so does
    a model that doesn't reply in time. Auth errors (401/403) stop immediately, since another model
    will not fix them. Returns the first successful response; nothing has been read from its body.
    """
    models = _models()
    if not models:
        raise LLMError("LLM_BASE_URL is set but LLM_MODEL is empty")
    headers = {"Authorization": f"Bearer {settings.LLM_API_KEY}"} if settings.LLM_API_KEY else {}
    url = f"{settings.LLM_BASE_URL.rstrip('/')}/chat/completions"
    deadline = time.monotonic() + _BUDGET_SECONDS
    last_error = "no attempt made"
    tried = []
    network_failures = other_failures = 0
    for model in models:
        tried.append(model)
        # generous: "thinking" models (Gemini 3.x) spend part of this budget on reasoning first
        body = {"model": model, "max_tokens": 2000, "temperature": 0.2, "messages": messages}
        if stream:
            body["stream"] = True
        sent_effort = bool(effort) and model not in _NO_EFFORT
        if sent_effort:
            body["reasoning_effort"] = effort
        dropped_effort = skip_backoff = False
        attempt = 0
        while attempt < _ATTEMPTS:
            attempt += 1
            if time.monotonic() > deadline:
                raise LLMError(f"ran out of time. Tried {', '.join(tried)}. Last error: {last_error}")
            if attempt > 1 and not skip_backoff:
                time.sleep(_BACKOFF_SECONDS * (attempt - 1))
            skip_backoff = False
            try:
                resp = requests.post(url, headers=headers, json=body, timeout=_TIMEOUT, stream=stream)
            except requests.Timeout as exc:  # a slow model rarely speeds up on retry: move on
                last_error = f"{model}: no reply within {_TIMEOUT[1]}s ({exc.__class__.__name__})"
                other_failures += 1
                break
            except requests.RequestException as exc:
                last_error = f"{model}: {exc.__class__.__name__}: {exc}"
                network_failures += 1
                continue
            if resp.status_code == 400 and sent_effort:
                # this model may not accept reasoning_effort: retry at once without it (not counted)
                resp.close()
                sent_effort, dropped_effort, skip_backoff = False, True, True
                body.pop("reasoning_effort", None)
                attempt -= 1
                continue
            if resp.status_code in _RETRYABLE:  # overloaded / rate limited: worth another try
                last_error = f"{model}: HTTP {resp.status_code}: {_snippet(resp)}"
                other_failures += 1
                resp.close()
                continue
            if resp.status_code in _TRY_NEXT_MODEL:
                last_error = f"{model}: HTTP {resp.status_code}: {_snippet(resp)}"
                other_failures += 1
                resp.close()
                break
            if not resp.ok:
                detail = f"{model}: HTTP {resp.status_code}: {_snippet(resp)}"
                resp.close()
                raise LLMError(detail)
            if dropped_effort:  # it works without the setting, so the setting was the problem
                _NO_EFFORT.add(model)
            return resp
    if network_failures and not other_failures:
        raise LLMError(
            "could not reach the language model service. Check your internet connection and try "
            f"again in a moment. Detail: {last_error[:200]}"
        )
    raise LLMError(f"all models failed ({', '.join(tried)}). Last error: {last_error}")


def _complete_openai(messages: list[dict], effort: str = "") -> str:
    resp = _open_chat(messages, stream=False, effort=effort)
    try:
        choice = resp.json()["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
        raise LLMError(f"unexpected response ({exc.__class__.__name__}): {_snippet(resp)}") from exc
    if not isinstance(content, str) or not content.strip():
        # providers send null or empty text when an answer is cut off or blocked (finish_reason says why)
        reason = choice.get("finish_reason") if isinstance(choice, dict) else None
        raise LLMError("the model returned an empty answer" + (f" (finish reason: {reason})" if reason else ""))
    return content.strip()


def _stream_openai(messages: list[dict], effort: str = "") -> Iterator[str]:
    resp = _open_chat(messages, stream=True, effort=effort)
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
            except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                continue  # keep-alives or provider-specific frames
            if isinstance(delta, str) and delta:  # role-only, null and non-text deltas carry no words
                produced = True
                yield delta
    except requests.RequestException as exc:
        raise LLMError(f"stream interrupted ({exc.__class__.__name__}): {exc}") from exc
    finally:
        resp.close()
    if not produced:
        raise LLMError("the model returned an empty answer")


def _stream_anthropic(messages: list[dict]) -> Iterator[str]:
    import anthropic

    system = messages[0]["content"]
    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    try:
        with client.messages.stream(
            model=settings.ANSWER_MODEL, max_tokens=2000, system=system, messages=messages[1:]
        ) as stream:
            yield from stream.text_stream
    except anthropic.APIError as exc:
        raise LLMError(f"{exc.__class__.__name__}: {exc}") from exc


def _complete(messages: list[dict], effort: str = "") -> str:
    if settings.LLM_BASE_URL:
        return _complete_openai(messages, effort)
    return "".join(_stream_anthropic(messages))


def _stream(messages: list[dict], effort: str = "") -> Iterator[str]:
    if settings.LLM_BASE_URL:
        yield from _stream_openai(messages, effort)
    else:
        yield from _stream_anthropic(messages)


def answer(
    question: str,
    passages: list[str] | None,
    history: list[dict] | None = None,
    allow_decline: bool = False,
) -> str:
    """Write an answer with the configured model.

    With `passages`, the answer is grounded in them and cites them. Without, it is a general answer.
    With `allow_decline` the model may say the passages don't answer it, which raises NotInDocuments.
    Order of providers: an OpenAI-compatible endpoint (LLM_BASE_URL), then Anthropic
    (ANTHROPIC_API_KEY). With neither, grounded questions get the best passage verbatim, so the app
    still works offline and free.
    """
    if not is_configured():
        return FALLBACK_PREFIX + passages[0] if passages else NO_MODEL_FOR_GENERAL
    text = _complete(
        build_messages(question, passages, history, allow_decline and bool(passages)),
        _effort_for(bool(passages)),
    )
    if allow_decline and passages and declined(text):
        raise NotInDocuments
    return text


def stream_answer(
    question: str,
    passages: list[str] | None,
    history: list[dict] | None = None,
    allow_decline: bool = False,
) -> Iterator[str]:
    """Like `answer`, but yields the text in pieces as the model writes it.

    Failures before the first piece (overload, bad key, unknown model) are retried / failed over
    exactly as in `answer`. A failure after output has started raises LLMError mid-iteration.
    With `allow_decline`, the first few characters are held back just long enough to tell whether
    the model is declining; if it is, NotInDocuments is raised before anything has been yielded.
    """
    if not is_configured():
        yield FALLBACK_PREFIX + passages[0] if passages else NO_MODEL_FOR_GENERAL
        return
    stream = _stream(
        build_messages(question, passages, history, allow_decline and bool(passages)),
        _effort_for(bool(passages)),
    )
    if not (allow_decline and passages):
        yield from stream
        return
    held = ""
    for piece in stream:
        if held is not None:
            held += piece
            if len(held) < len(NOT_IN_DOCUMENTS_SENTINEL) + 4:
                continue  # not enough text yet to tell a decline from a real answer
            if declined(held):
                raise NotInDocuments
            piece, held = held, None
        yield piece
    if held:  # a very short answer that ended before the buffer filled
        if declined(held):
            raise NotInDocuments
        yield held


def condense_question(question: str, history: list[dict] | None) -> str:
    """Turn a follow-up ("and his projects?") into a standalone search query using the chat history.

    Never raises: if there is no history, no model, or the model fails, the original question is
    used, so a follow-up simply searches with what the user typed.
    """
    prior = clean_history(history)
    if not prior or not is_configured() or not looks_like_followup(question):
        return question
    messages = [
        {"role": "system", "content": CONDENSE_PROMPT},
        *prior[: -1 if prior[-1]["role"] == "user" else None],
        {"role": "user", "content": question},
    ]
    try:
        rewritten = _complete(messages, settings.LLM_REASONING_EFFORT).strip().strip('"')
    except LLMError as exc:
        log.warning("could not rewrite follow-up question: %s", exc)
        return question
    return rewritten[:500] if rewritten else question
