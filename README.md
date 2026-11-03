# Answer Trail

Ask questions about your own documents and get answers that cite the passages they came from.

Upload PDF, TXT or Markdown files, then chat with them in the browser. The app finds the most relevant
passages with vector search, sends only those to a language model, and streams back an answer with
clickable citations, so every claim can be checked against its source.

- **Retrieval-augmented generation (RAG)** with sentence-aware chunking, local neural embeddings, and hybrid search (vectors + BM25 keywords, fused with reciprocal rank fusion)
- **Three answer modes**: *Documents only* (cited, or "not found"), *Auto* (documents when they cover the question, otherwise general knowledge, clearly labelled), and *General chat*
- **Conversation memory**: follow-ups like "and the window?" are rewritten into standalone searches
- **Summaries that work**: "summarize this document" and "what are the key points?" read passages spread across the document, since no single passage resembles such a request
- **Fast**: the model's reasoning is turned down (first word in about 1.3 s instead of 3-4 s), the search model loads at startup, and search takes about 10-20 ms per question however large the collection (see Speed)
- **Streaming chat UI** served by Django itself (plain HTML/JS, no build step), with Stop, citations, drag-and-drop upload, dark/light theme and a phone layout
- **Django REST API** with token auth and per-user isolation: users only ever search their own documents
- **Free or paid models** through LangChain: any OpenAI-compatible endpoint (Ollama, Gemini, Groq, OpenRouter) or Anthropic, with automatic retry and model fallback when a provider is overloaded, and optional LangSmith tracing
- **PostgreSQL + pgvector** in production (exact per-user search; HNSW index available for huge collections), SQLite for local use
- **Retrieval eval harness** that measures hit-rate@k and MRR, plus Python and JavaScript tests and CI

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-ml.txt                    # requirements.txt alone = lexical embeddings only
python manage.py migrate
EMBEDDING_BACKEND=fastembed python manage.py runserver
```

Open http://localhost:8000, create an account, upload a document and ask a question. The first
question loads the embedding model (about 10-20 seconds, plus a one-time download); later ones are fast.

Without a language model configured, the app still works and shows the best matching passage instead of
a written answer. To get written answers, pick one option below.

On Windows, `scripts/start.ps1` asks for your key (hidden), starts the server and opens the browser.

## Using it

Upload PDF, TXT or Markdown files, then ask. Click a suggestion, or try: "summarize this document", "what
are the key points?", then a follow-up such as "and the second one?". Click a `[1]` to see the exact passage,
with your question's words highlighted. **Copy** copies an answer, **Try again** re-asks after an error,
**Stop** (or Esc) ends an answer, **New chat** clears the memory. `/` or Ctrl+K focuses the question box.

## Answer modes

Pick one in the sidebar (the API field is `mode`):

| Mode | Behaviour |
|---|---|
| **Auto** (UI default) | If the closest passage is similar enough to the question, answer from the documents with citations. Otherwise answer from the model's general knowledge. |
| **Documents only** (API default) | Answer only from relevant passages. If nothing is relevant, reply "I couldn't find that in your documents" without calling the model. |
| **General chat** | Ignore the documents and answer like a normal assistant. |

Every answer carries a badge: **From your documents**, **General knowledge, not from your files**, or
**Not in your documents**. This matters because a model states invented facts as confidently as quoted ones,
so only the first kind has citations to check. General answers never have citation buttons.

"Similar enough" is a cutoff on the best passage's cosine similarity (`RELEVANCE_THRESHOLD`, default 0.56
with the neural model). It was calibrated with `manage.py eval_retrieval --routing`: on the eval set,
on-topic questions scored 0.587 or higher and 14 unrelated questions ("capital of France", "tell me a joke",
"hi") 0.536 or lower. That margin is small (about 0.025 each side) and was tuned on the same questions, so
recalibrate on your own documents. **Auto mode needs the neural model**: the lexical `hash` embedder cannot
separate the two groups (47% / 86% on the same test).

**A cutoff alone is not enough on real documents.** The cutoff was calibrated on clean policy documents. On
a code-heavy Python guide, "Tell me a joke" scored 0.602 (above the cutoff) because the guide is full of
`Animal`/`speak()` examples, and adding "Answer in one sentence" to a question moved its score from 0.476
to 0.625. So in Auto mode the model is also allowed to reply `NOT_IN_DOCUMENTS` when the passages don't
answer the question. The server holds back the first few characters of the stream just long enough to tell,
then switches: the badge changes, the sources are dropped, and a general answer follows, before any
discarded text could be shown. This costs one extra model call only for such questions (about 12 s against
6 s in a live test with Gemini). In Documents-only mode the model just says it doesn't know.

**Memory.** The browser sends the last few messages with each question. For follow-ups the model first
rewrites the question into a standalone search query (one extra model call, shown to you as "Searched for: ...").
If the rewrite fails the question is searched as typed. Failed, stopped, "not found" and passage-only replies
are not remembered. **New chat** clears it. The server stores no conversations.

## Choosing the answer model

Search and embeddings are free and local. Only the final written answer needs a language model. The
server uses the first option that is configured:

| Option | Settings |
|---|---|
| Any OpenAI-compatible server | `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` (if needed), optional `LLM_FALLBACK_MODELS` |
| Anthropic | `ANTHROPIC_API_KEY` (paid), optional `ANSWER_MODEL` |
| Neither | the best matching passage is returned as-is |

The easiest way to set these is a local `.env` file: copy `.env.example` to `.env` and paste your key
into it. The app reads it on start. `.env` is git-ignored, and real environment variables override it.

Examples for the first option:

- **Ollama** (runs on your PC): `LLM_BASE_URL=http://localhost:11434/v1`, `LLM_MODEL=llama3.2`
- **Google Gemini** (free tier): `LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/`, a model name from AI Studio, and your key
- **Groq, OpenRouter**: base URL, model and key from their docs

Free tiers change their limits and model names, so check the provider's docs.

`LLM_FALLBACK_MODELS` (comma separated) are tried in order when a model is overloaded (503/429) or unknown
(404), times out (30 s) or drops the connection: the next model is tried straight away. A server error (5xx)
is retried once first, and a model that answered 429 (rate limit or quota) is skipped for a minute so the
next questions do not pay for the same refusal. A bad key (401/403) stops
immediately, since another model cannot fix it. If everything fails the UI shows the provider's own error
message, and the retrieved sources are still shown. The model calls, retries and fallbacks are done with
[LangChain](https://python.langchain.com/) chat models (`langchain-openai`, `langchain-anthropic`); retrieval stays our own code.

## Tracing with LangSmith (optional)

Because the model calls go through LangChain, they can be traced in [LangSmith](https://smith.langchain.com/).
Each question shows up as a run named `answer_from_documents`, `answer_general` or `rewrite_followup`, with the
exact prompt (including the retrieved passages), the reply, the time taken and token counts. It is **off by
default**. To switch it on, add these to `.env`:

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=answertrail
```

Privacy: tracing sends your questions, the retrieved passages of your documents and the answers to LangSmith's
servers, so leave it off for confidential documents. The tests always switch it off.

## Speed

Where the time goes for one question, and what was done about it (measured on a laptop with 7.6 GB of RAM,
free-tier Gemini; your numbers will differ):

| Step | Before | Now | How |
|---|---|---|---|
| Model's first word | 3.3-4.3 s | 1.2-1.5 s | `LLM_REASONING_EFFORT=minimal`; answers to the same questions were equally correct on the samples tried |
| First question after a server start | +10-20 s | +0 s | the search model loads in the background at startup (`WARMUP_EMBEDDINGS=1`) |
| Search, 1,655 / 5,655 / 20,655 chunks | 0.46 / 1.5 / 8.4 s | 9 / 11 / 19 ms | per-user cache: embeddings and BM25 inverted index built once, validated by one cheap query |
| First question after an upload | same as above | 0 | the cache is built during the upload |
| Rewriting a follow-up | one model call | skipped unless the question looks like a follow-up | short questions and ones with "it/he/that" are still rewritten |
| An unrelated question in Auto mode | grounded call, decline, then general (about 12 s) | the same, now about 4-5 s | the decline check also runs at minimal reasoning |

Every response reports `timings` (`rewrite_ms`, `search_ms`) so slowness can be traced, and the UI shows how
long an answer took. Notes:
- `minimal` reasoning suits answering from passages. For hard reasoning or coding questions in General mode,
  set `LLM_REASONING_EFFORT_GENERAL=` (empty) to let the model think. I checked two general questions, not a
  benchmark.
- A model that rejects `reasoning_effort` (flash-lite rejected `none`) is retried without it.
- Every model attempt is logged with its time to first word, e.g. `model gemini-3.6-flash answered: first
  word 1.7s, total 2.0s`, and failures as warnings. When answers feel slow, read the server log first: on free
  tiers the delay is usually the provider (the same question took 1.7 s to 30 s on different models and
  minutes), not this app. `DOCQA_LOG_LEVEL=WARNING` hides the timing lines.
- The model client is built and its connection kept open for two minutes (a new TLS handshake cost 0.1-0.5 s
  per question), and its libraries are imported at startup, not on the first question (about 2 s).
- The follow-up rewrite is optional, so it gets one attempt with an 8 s limit and no fallback models; if it
  fails the question is searched as typed. Questions of six or more words with no reference word are not rewritten.
- Ingestion is still about 26 chunks per second on this laptop (a 200 KB document takes about 20 s) and runs
  inside the upload request; a background queue is the next step for large files.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `EMBEDDING_BACKEND` | `hash` | `fastembed` for the neural model (`BAAI/bge-small-en-v1.5`, 384-d) |
| `EMBEDDING_DIM` | `384` | must match the model; changing it means recreating the chunk table |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `100` | characters per chunk / overlap between chunks |
| `HYBRID_SEARCH` | `1` | vectors + BM25 keyword search, fused; `0` for vectors only |
| `LLM_REASONING_EFFORT` | empty | `minimal` recommended: reasoning level for answers from documents and follow-up rewrites |
| `LLM_REASONING_EFFORT_GENERAL` | empty | the same for general-knowledge answers (leave empty for hard questions) |
| `PGVECTOR_USE_INDEX` | `0` | `1` lets PostgreSQL use the approximate HNSW index (only for very large collections) |
| `WARMUP_EMBEDDINGS` | `0` | `1` loads the search model at startup (the launch script and docker-compose set it) |
| `RELEVANCE_THRESHOLD` | per embedder | best-passage similarity needed to answer from documents (see Answer modes) |
| `DATABASE_URL` | SQLite file | `postgres://user:pass@host:5432/db` enables pgvector |
| `MAX_UPLOAD_BYTES` | 5 MB | upload limit (the UI checks the same limit) |
| `DJANGO_DEBUG` | `1` | set `0` in production; then `DJANGO_SECRET_KEY` is required |
| `DJANGO_SECURE` | off | set `1` behind HTTPS: redirects, secure cookies, HSTS |

## Deploying

`docker compose up` starts the app with Postgres + pgvector and the neural model (set `DJANGO_SECRET_KEY`
first). Static files are served by WhiteNoise, so one container serves both the UI and the API.
Put it behind HTTPS and set `DJANGO_SECURE=1`. HSTS preload and include-subdomains are deliberately left off.

## API

| Method | Path | Notes |
|---|---|---|
| POST | `/api/auth/register/` | `{username, password}` -> `{token}` |
| POST | `/api/auth/token/` | login -> `{token}` |
| POST | `/api/documents/` | multipart `file` (.pdf/.txt/.md) or JSON `{title, text}` |
| GET | `/api/documents/`, `/api/documents/<id>/` | own documents only |
| DELETE | `/api/documents/<id>/` | removes its chunks too |
| POST | `/api/ask/` | `{question, mode?, history?, document_id?, top_k?}` -> `{answer, sources[], mode, searched_for, best_similarity}` |
| POST | `/api/ask/stream/` | same input; Server-Sent Events: `sources` (with `mode`, `searched_for`, `best_similarity`), then `token` pieces, then `done` or `error` |

`mode` is `documents` (default), `auto` or `general`; the response `mode` is `documents`, `general` or
`not_found` (what actually happened; in the stream, a `mode` event switches a `documents` answer to
`general` when the model declines). `history` is a list of up to 30 `{role: "user"|"assistant", content}`
messages; only the last 8 are used.

Send `Authorization: Token <token>`. Validation and auth errors are plain JSON with the usual status codes;
model failures on `/api/ask/` return 502, and on the stream arrive as an `error` event after `sources`.

## How it works

```
upload -> extract text -> sentence-aware chunks -> embed -> store (pgvector / SQLite)

question (+ recent chat) -> [follow-up? rewrite into a standalone query]
        -> hybrid search: vector top-N + BM25 top-N -> reciprocal rank fusion -> top-k chunks
        -> best passage similar enough?  yes: grounded answer with citations
                                          no: "not found" (documents only) or general answer (auto)
```

Chunks are whole sentences packed up to the size limit, with trailing sentences repeated as overlap.
Vector search is SQL (`<=>`, exact over the caller's own chunks) on Postgres and a numpy scan on SQLite. BM25 runs in Python
over the caller's chunk text, which is fine for thousands of chunks; move it to Postgres full-text search
for much larger collections. The grounded prompt tells the model to answer only from the numbered passages
and to cite them, and the UI turns `[1]` into a button that opens and highlights that source. Retrieved
text and model output are HTML-escaped before display.

## Retrieval eval

```bash
python manage.py eval_retrieval --backend fastembed --show-misses --routing
python manage.py eval_retrieval --backend hash --search vector      # compare search strategies
```

`eval/dataset.json`: 8 short policy documents, 47 questions (35 paraphrased, 12 keyword-style such as
"Sev1" or "SMS codes") and 14 off-topic questions. Each question has an evidence span; a retrieved chunk is a
hit if it contains the span, so labels survive changes to chunking. The command reports hit-rate@k and MRR,
and with `--routing` the similarity cutoff that best separates on-topic from off-topic questions. It rolls
back all data it creates.

| Embedding | Search | hit-rate@1 | hit-rate@5 | MRR | paraphrase MRR | keyword MRR |
|---|---|---|---|---|---|---|
| hash (lexical baseline) | vector | 0.489 | 0.872 | 0.648 | 0.560 | 0.903 |
| hash | hybrid | 0.766 | 0.979 | 0.869 | 0.838 | 0.958 |
| bge-small-en-v1.5 (neural) | vector | 0.894 | 0.979 | 0.936 | 0.943 | 0.917 |
| bge-small-en-v1.5 (neural) | **hybrid (default)** | **0.915** | 0.979 | **0.943** | 0.924 | **1.000** |

What this shows, and what it doesn't:
- Hybrid search helps most where exact words matter (keyword MRR 0.917 to 1.000 for the neural model) and a
  lot for the weak embedder. For the neural model it cost one paraphrased question its first place
  (paraphrase hit-rate@1 0.914 to 0.886). Overall it is a small net gain on this set.
- The keyword questions were written to test hybrid search and there are only 12, so this is evidence, not
  proof. The dataset is small and written by the project author, and the chunk size and relevance cutoff
  were chosen on it. Treat the numbers as a regression guard rather than a benchmark.
- Not done: a cross-encoder re-ranker, and an evaluation of answer faithfulness.

Chunking mattered: with fixed 800-character windows, the definition of "inheritance" in a 34-page Python
guide ranked 12th, because it sat at the end of a chunk dominated by an unrelated code example. With
sentence-aware chunks it ranks 1st.

## Tests

```bash
pytest -q                                   # about 300 tests: API, modes, retrieval, speed, LLM client, streaming, robustness
node --test "tests/js/**/*.test.js"         # about 40 tests: SSE parsing, safe rendering, history, formatting
python tests/e2e/server.py                  # browser checks, see tests/e2e/README.md
```

What the suites cover beyond the happy path:
- **Fuzzing**: random Unicode text through the chunker (nothing lost, no chunk over the limit), junk JSON
  bodies at every endpoint (never a 5xx), odd authorization headers, malformed and encrypted PDFs, NUL bytes,
  over-long names, and 4,000 random strings through the answer renderer (output is always well-formed and
  never contains an unescaped tag).
- **Odd model replies**: null or empty text, non-text stream frames, an error object mid-stream, a stream cut
  into 3-byte pieces (splitting frames and multi-byte characters), a connection that dies halfway, overload,
  unknown models, a rejected `reasoning_effort`.
- **Concurrency and isolation**: two streams in flight at once, a client that disconnects mid-answer, and
  one user never seeing another's documents through search, overview, scoping or the cache.
- **Mutation checks**: breaking the relevance comparison or the HTML escaping makes tests fail.

The streaming and failover tests run against a real local HTTP server that imitates an OpenAI-style
streaming API (`tests/fake_llm.py`), and it answers differently for grounded, general and rewrite requests so
the tests can check which kind of call the app made. Run it by hand with `python tests/fake_llm.py 8765`.
Tests that need the neural model are skipped when `fastembed` is not installed.

CI (`.github/workflows/ci.yml`) runs the Python and JS tests, a production `collectstatic`, and a second
job against a real `pgvector/pgvector` Postgres container that also runs the retrieval eval.

## Known limits

- Answer quality (faithfulness to the sources) is not evaluated, only retrieval and routing.
- Auto routing starts with a similarity cutoff. An unrelated question that passes it is caught by the model
  check, but a vague on-topic question that falls *below* it never reaches the documents and gets a general
  answer. The badge always says which happened; Documents-only mode avoids it.
- A general-knowledge answer can be wrong or out of date. It is labelled, but not fact-checked.
- Follow-up rewriting costs one extra model call, and a model that rewrites badly can steer the search.
  The "Searched for" line shows what was searched.
- On PostgreSQL, vector search is **exact per user** by default: the owner filter runs first and only that
  user's chunks are ranked. The approximate HNSW index is not used, because behind a filter it can miss a
  user's rows (CI showed searches returning nothing once earlier tests had inserted and rolled back
  thousands of chunks). That is fast for the sizes this app targets (one user's chunks, up to the order of
  100,000). For a very large single collection set `PGVECTOR_USE_INDEX=1`: searches then use the index with
  pgvector 0.8 iterative scans, which reduces but cannot remove the risk of too few results.
- Ingestion is synchronous (about 26 chunks per second on a laptop, so a 5 MB file can take minutes inside one
  request), and scanned (image-only) PDFs are rejected because there is no OCR.
- Summaries read a spread sample (up to 8-10 passages), so details between the samples are not seen. For a
  specific topic, ask about the topic.
- The search cache lives in each server process's memory (up to four users or scopes, 50,000 chunks each), so
  several server workers each keep their own copy.
- Chat history lives in the browser tab only and is lost on reload.
- Free model tiers have quotas. When one is used up the app falls back to the next model and, if all fail,
  says so with the provider's message.
