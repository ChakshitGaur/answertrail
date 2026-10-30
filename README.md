# rag-docqa

Ask questions about your own documents and get answers that cite the passages they came from.

Upload PDF, TXT or Markdown files, then chat with them in the browser. The app finds the most relevant
passages with vector search, sends only those to a language model, and streams back an answer with
clickable citations, so every claim can be checked against its source.

- **Retrieval-augmented generation (RAG)** with sentence-aware chunking, local neural embeddings, and hybrid search (vectors + BM25 keywords, fused with reciprocal rank fusion)
- **Three answer modes**: *Documents only* (cited, or "not found"), *Auto* (documents when they cover the question, otherwise general knowledge, clearly labelled), and *General chat*
- **Conversation memory**: follow-ups like "and the window?" are rewritten into standalone searches
- **Streaming chat UI** served by Django itself (plain HTML/JS, no build step), with Stop, citations, drag-and-drop upload, dark/light theme and a phone layout
- **Django REST API** with token auth and per-user isolation: users only ever search their own documents
- **Free or paid models**: any OpenAI-compatible endpoint (Ollama, Gemini, Groq, OpenRouter) or Anthropic, with automatic retry and model fallback when a provider is overloaded
- **PostgreSQL + pgvector** (HNSW index) in production, SQLite for local use
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
(404). Each model gets 2 attempts and all attempts share a 100-second budget. A bad key (401/403) stops
immediately, since another model cannot fix it. If everything fails the UI shows the provider's own error
message, and the retrieved sources are still shown.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `EMBEDDING_BACKEND` | `hash` | `fastembed` for the neural model (`BAAI/bge-small-en-v1.5`, 384-d) |
| `EMBEDDING_DIM` | `384` | must match the model; changing it means recreating the chunk table |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `100` | characters per chunk / overlap between chunks |
| `HYBRID_SEARCH` | `1` | vectors + BM25 keyword search, fused; `0` for vectors only |
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
`not_found` (what actually happened). `history` is a list of up to 30 `{role: "user"|"assistant", content}`
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
Vector search is SQL (`<=>` with an HNSW index) on Postgres and a numpy scan on SQLite. BM25 runs in Python
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
pytest -q                                   # API, modes, retrieval, chunking, LLM client, streaming, pages
node --test "tests/js/**/*.test.js"         # UI logic: SSE parsing, safe rendering, history, validation
```

The streaming and failover tests run against a real local HTTP server that imitates an OpenAI-style
streaming API (`tests/fake_llm.py`). It can return 503s, unknown models, bad keys, a connection that dies
mid-stream and non-ASCII text, and it answers differently for grounded, general and rewrite requests so the
tests can check which kind of call the app made. Run it by hand with `python tests/fake_llm.py 8765`.
Tests that need the neural model are skipped when `fastembed` is not installed.

CI (`.github/workflows/ci.yml`) runs the Python and JS tests, a production `collectstatic`, and a second
job against a real `pgvector/pgvector` Postgres container that also runs the retrieval eval.

## Known limits

- Answer quality (faithfulness to the sources) is not evaluated, only retrieval and routing.
- Auto routing is a similarity cutoff, so it can misroute: a vague on-topic question may fall below it and
  get a general answer, and an unrelated question that shares vocabulary with your documents may pass it.
  The badge always says which happened.
- A general-knowledge answer can be wrong or out of date. It is labelled, but not fact-checked.
- Follow-up rewriting costs one extra model call, and a model that rewrites badly can steer the search.
  The "Searched for" line shows what was searched.
- Retrieval filters by owner after the HNSW scan, so a user with few chunks among many may get fewer than
  `top_k` results; pgvector 0.8 iterative scans address this.
- Ingestion is synchronous, and scanned (image-only) PDFs are rejected because there is no OCR.
- Chat history lives in the browser tab only and is lost on reload.

## Older Streamlit UI

`ui/app.py` is the first UI, kept as an alternative. It talks to the same API:
`pip install -r requirements-ui.txt && streamlit run ui/app.py` (it uses the non-streaming endpoint).
