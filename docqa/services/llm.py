import logging
import re
import time
from collections.abc import Iterator
from functools import lru_cache

from django.conf import settings

log = logging.getLogger(__name__)

_ATTEMPTS = 2  # tries per model when it answers with a server error (5xx)
_TIMEOUT = (10, 30)  # (connect, read) seconds per attempt
_REWRITE_TIMEOUT = (5, 8)  # the follow-up rewrite is optional, so it gets a short leash
_COOLDOWN_SECONDS = 60  # a model that answered 429 (rate limit or quota) is skipped for this long

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


# Words that usually mean "this question depends on the conversation so far"
_REFERENCES = frozenset(
    "it its this that these those he she they them his her their him there here also too another other "
    "former latter above same previous earlier more else instead then".split()
)
_SHORT_QUESTION_WORDS = 5  # "his projects?", "and skills": too short to stand on their own
_FOLLOW_UP_OPENERS = ("and ", "but ", "so ", "what about", "how about", "what else", "why ", "how come")


def looks_like_followup(question: str) -> bool:
    """Cheap check for whether a question needs the chat history to make sense.

    Rewriting costs a model call (seconds, before the search can even start), so a self-contained
    question is searched as typed. Very short questions, ones starting like a continuation, or ones
    with a reference word ("it", "he", "that") are rewritten.
    """
    text = question.strip().lower()
    words = re.findall(r"[a-z0-9']+", text)
    return (
        len(words) <= _SHORT_QUESTION_WORDS
        or text.startswith(_FOLLOW_UP_OPENERS)
        or any(w in _REFERENCES for w in words)
    )


def _effort_for(grounded: bool) -> str:
    return settings.LLM_REASONING_EFFORT if grounded else settings.LLM_REASONING_EFFORT_GENERAL


def _provider_errors() -> tuple[type[Exception], ...]:
    """Everything the model SDKs (and a malformed reply) can raise, to be reported as an LLMError."""
    import httpx
    import openai

    errors = (openai.OpenAIError, httpx.HTTPError, LookupError, ValueError, TypeError, AttributeError)
    if settings.LLM_BASE_URL:  # the Anthropic SDK is only imported when it is the provider
        return errors
    import anthropic

    return (anthropic.APIError, *errors)


@lru_cache(maxsize=16)
def _client(base_url: str, api_key: str, name: str, effort: str, timeout: tuple[float, float]):
    import httpx
    from langchain_openai import ChatOpenAI

    seconds = httpx.Timeout(timeout[1], connect=timeout[0])
    return ChatOpenAI(
        model=name,
        base_url=base_url,
        api_key=api_key or "not-needed",  # local servers want no key, but the client insists on one
        temperature=0.2,
        # generous: "thinking" models (Gemini 3.x) spend part of this budget on reasoning first. Sent as
        # max_tokens through extra_body, which every OpenAI-compatible server accepts (LangChain's own field
        # is sent as max_completion_tokens)
        extra_body={"max_tokens": 2000},
        timeout=seconds,
        max_retries=0,  # retries and failover are explicit, see _openai_runnable
        # keep the connection open between questions: a new TLS handshake costs 0.1-0.5 s, and httpx drops
        # idle connections after 5 s by default
        http_client=httpx.Client(
            timeout=seconds, limits=httpx.Limits(max_keepalive_connections=10, keepalive_expiry=120)
        ),
        **({"reasoning_effort": effort} if effort else {}),
    )


def _chat_model(name: str, effort: str = "", timeout: tuple[float, float] | None = None):
    """A LangChain chat model for an OpenAI-compatible server (Ollama, Groq, Gemini, OpenRouter...).

    Built once per (server, key, model, effort, timeout) and reused together with its HTTP connection.
    """
    return _client(settings.LLM_BASE_URL, settings.LLM_API_KEY, name, effort, timeout or _TIMEOUT)


_COOLING: dict[str, float] = {}  # model name -> time.monotonic() until which it is skipped


def _cooling(name: str) -> bool:
    return _COOLING.get(name, 0.0) > time.monotonic()


@lru_cache(maxsize=1)
def _call_log():
    """The LangChain callback that logs every model attempt and parks a rate-limited model for a while."""
    from langchain_core.callbacks import BaseCallbackHandler

    class CallLog(BaseCallbackHandler):
        def __init__(self):
            self.attempts: dict = {}  # run id -> [model, start, time of first word]

        def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
            name = (kwargs.get("invocation_params") or {}).get("model") or "?"
            self.attempts[run_id] = [name, time.perf_counter(), None]

        def on_llm_new_token(self, token, *, run_id, **kwargs):
            attempt = self.attempts.get(run_id)
            if attempt and attempt[2] is None:
                attempt[2] = time.perf_counter()

        def on_llm_end(self, response, *, run_id, **kwargs):
            attempt = self.attempts.pop(run_id, None)
            if attempt:
                name, started, first = attempt
                now = time.perf_counter()
                log.info("model %s answered: first word %.2fs, total %.2fs", name, (first or now) - started, now - started)

        def on_llm_error(self, error, *, run_id, **kwargs):
            import openai

            attempt = self.attempts.pop(run_id, None)
            if not attempt:
                return
            name, started, _ = attempt
            log.warning("model %s failed after %.2fs: %s", name, time.perf_counter() - started, error.__class__.__name__)
            if isinstance(error, openai.RateLimitError):
                _COOLING[name] = time.monotonic() + _COOLDOWN_SECONDS

    return CallLog()


def _openai_runnable(effort: str, quick: bool = False):
    """LLM_MODEL, then each LLM_FALLBACK_MODELS entry, as one runnable that fails over in that order.

    A server error (5xx) is retried once at once. A rate limit (429), a model the server doesn't know
    (404/400), a timeout or a dropped connection moves on to the next model straight away, and a model
    that answered 429 is skipped for a minute. Auth errors (401/403) stop immediately, since another
    model will not fix them. A model that rejects `reasoning_effort` (400) is retried at once without it.
    With `quick`, just the first usable model with a short timeout and no retries, for optional calls.
    """
    import openai

    names = _models()
    if not names:
        raise LLMError("LLM_BASE_URL is set but LLM_MODEL is empty")
    names = [n for n in names if not _cooling(n)] or names  # all of them cooling: try anyway
    if quick:
        return _chat_model(names[0], effort, _REWRITE_TIMEOUT)

    def attempt(name: str):
        plain = _chat_model(name)
        first = (
            _chat_model(name, effort).with_fallbacks([plain], exceptions_to_handle=(openai.BadRequestError,))
            if effort
            else plain
        )
        return first.with_fallbacks([first], exceptions_to_handle=(openai.InternalServerError,))

    models = [attempt(name) for name in names]
    failover = (
        openai.RateLimitError,
        openai.InternalServerError,
        openai.APIConnectionError,  # includes timeouts
        openai.NotFoundError,
        openai.BadRequestError,  # Google reports an invalid key as 400
    )
    return models[0].with_fallbacks(models[1:], exceptions_to_handle=failover)


def _anthropic_runnable():
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=settings.ANSWER_MODEL, api_key=settings.ANTHROPIC_API_KEY, max_tokens=2000)


def _runnable(effort: str, quick: bool = False):
    return _openai_runnable(effort, quick) if settings.LLM_BASE_URL else _anthropic_runnable()


def warm_up() -> float | None:
    """Import the model SDKs and build the client now, so the first question does not pay about 2 s for it.

    Returns the seconds it took, or None when there is no model or it failed (never raises).
    """
    if not is_configured():
        return None
    started = time.perf_counter()
    try:
        _provider_errors()
        _call_log()
        _runnable(_effort_for(True))
    except Exception:  # an optimisation must never stop the server
        log.exception("model client warm-up failed")
        return None
    return time.perf_counter() - started


def _call_config(run_name: str, *tags: str) -> dict:
    """How the call is labelled in LangSmith (no effect unless tracing is on), and who logs its timing."""
    return {"run_name": run_name, "tags": list(tags), "callbacks": [_call_log()]}


def _llm_error(exc: Exception, produced: bool = False) -> LLMError:
    """Turn what an SDK raised into the one error type the views handle, with the provider's own text."""
    import openai

    detail = " ".join(str(exc).split())[:300]
    if isinstance(exc, ValueError) and not produced and "No generation chunks" in detail:
        return LLMError("the model returned an empty answer")
    if produced:
        return LLMError(f"stream interrupted ({exc.__class__.__name__}): {detail}")
    if isinstance(exc, openai.APIStatusError) and exc.status_code in (401, 403):
        return LLMError(f"HTTP {exc.status_code}: {detail}")
    if isinstance(exc, openai.APITimeoutError):
        return LLMError(f"all models failed ({', '.join(_models())}). First error: no reply within {_TIMEOUT[1]}s")
    if isinstance(exc, openai.APIConnectionError):
        return LLMError(
            "could not reach the language model service. Check your internet connection and try "
            f"again in a moment. Detail: {detail}"
        )
    if isinstance(exc, openai.OpenAIError):
        return LLMError(f"all models failed ({', '.join(_models())}). First error: {detail}")
    if not settings.LLM_BASE_URL:
        import anthropic

        if isinstance(exc, anthropic.APIError):
            return LLMError(f"{exc.__class__.__name__}: {detail}")
    return LLMError(f"unexpected response ({exc.__class__.__name__}): {detail}")


def _run_label(passages: list[str] | None) -> tuple[str, str]:
    """LangSmith run name and tag for an answer: from the documents, or general knowledge."""
    return ("answer_from_documents", "grounded") if passages else ("answer_general", "general")


def _as_messages(messages: list[dict]) -> list[tuple[str, str]]:
    return [(m["role"], m["content"]) for m in messages]


def _complete(
    messages: list[dict], effort: str = "", run_name: str = "answer", *tags: str, quick: bool = False
) -> str:
    try:
        reply = _runnable(effort, quick).invoke(_as_messages(messages), config=_call_config(run_name, *tags))
        text = str(reply.text).strip()
    except _provider_errors() as exc:
        raise _llm_error(exc) from exc
    if not text:
        # providers send null or empty text when an answer is cut off or blocked (finish_reason says why)
        reason = (getattr(reply, "response_metadata", None) or {}).get("finish_reason")
        raise LLMError("the model returned an empty answer" + (f" (finish reason: {reason})" if reason else ""))
    return text


def _stream(messages: list[dict], effort: str = "", run_name: str = "answer", *tags: str) -> Iterator[str]:
    produced = False
    try:
        for chunk in _runnable(effort).stream(_as_messages(messages), config=_call_config(run_name, *tags)):
            text = str(chunk.text)
            if text:  # role-only and non-text chunks carry no words
                produced = True
                yield text
    except _provider_errors() as exc:
        raise _llm_error(exc, produced) from exc
    if not produced:
        raise LLMError("the model returned an empty answer")


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
        *_run_label(passages),
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
        *_run_label(passages),
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
        rewritten = _complete(messages, settings.LLM_REASONING_EFFORT, "rewrite_followup", "rewrite", quick=True).strip().strip('"')
    except LLMError as exc:
        log.warning("could not rewrite follow-up question: %s", exc)
        return question
    return rewritten[:500] if rewritten else question
