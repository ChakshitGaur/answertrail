# rag-docqa

Ask questions about your own documents and get answers that cite the passages they came from.

Upload PDF, TXT or Markdown files, then chat with them in the browser. The app finds the most relevant
passages with vector search, sends only those to a language model, and streams back an answer with
clickable citations, so every claim can be checked against its source.

- **Retrieval-augmented generation (RAG)** with sentence-aware chunking and neural embeddings that run locally
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
| POST | `/api/ask/` | `{question, document_id?, top_k?}` -> `{answer, sources[]}` |
| POST | `/api/ask/stream/` | same input; Server-Sent Events: `sources`, then `token` pieces, then `done` or `error` |

Send `Authorization: Token <token>`. Validation and auth errors are plain JSON with the usual status codes;
model failures on `/api/ask/` return 502, and on the stream arrive as an `error` event after `sources`.

## How it works

```
upload -> extract text -> sentence-aware chunks -> embed -> store (pgvector / SQLite)
question -> embed -> cosine top-k over the caller's chunks -> prompt with numbered passages -> stream answer
```

Chunks are whole sentences packed up to the size limit, with trailing sentences repeated as overlap.
Retrieval is SQL (`<=>` with an HNSW index) on Postgres and a numpy scan on SQLite. The prompt tells the
model to answer only from the numbered passages and to cite them, and the UI turns `[1]` into a button
that opens and highlights that source. Retrieved text and model output are HTML-escaped before display.

## Retrieval eval

```bash
python manage.py eval_retrieval --backend fastembed --show-misses
python manage.py eval_retrieval --backend hash
```

`eval/dataset.json`: 8 short policy documents and 35 paraphrased questions. Each question has an evidence
span; a retrieved chunk is a hit if it contains the span, so labels survive changes to chunking. The
command reports hit-rate@k and MRR, and rolls back all data it creates.

| Embedding | hit-rate@1 | hit-rate@3 | hit-rate@5 | MRR |
|---|---|---|---|---|
| hash (lexical baseline) | 0.371 | 0.771 | 0.829 | 0.560 |
| bge-small-en-v1.5 (fastembed, quantised ONNX) | 0.914 | 0.971 | 0.971 | 0.943 |

The baseline's misses are mostly synonym gaps ("guarantee" vs "warranty"). The neural model misses one
question (the "2 pm Eastern cutoff"). The dataset is small and written by the project author, and the
chunk size was chosen on it, so treat the numbers as a regression guard rather than a benchmark.

Chunking mattered: with fixed 800-character windows, the definition of "inheritance" in a 34-page Python
guide ranked 12th, because it sat at the end of a chunk dominated by an unrelated code example. With
sentence-aware chunks it ranks 1st.

## Tests

```bash
pytest -q                                   # API, retrieval, chunking, LLM client, streaming, pages
node --test "tests/js/**/*.test.js"         # UI logic: SSE parsing, safe rendering, validation
```

The streaming and failover tests run against a real local HTTP server that imitates an OpenAI-style
streaming API (`tests/fake_llm.py`). It can return 503s, unknown models, bad keys, a connection that dies
mid-stream and non-ASCII text. Run it by hand with `python tests/fake_llm.py 8765` to try the UI against it.

CI (`.github/workflows/ci.yml`) runs the Python and JS tests, a production `collectstatic`, and a second
job against a real `pgvector/pgvector` Postgres container that also runs the retrieval eval.

## Known limits

- Answer quality (faithfulness to the sources) is not evaluated, only retrieval.
- Retrieval filters by owner after the HNSW scan, so a user with few chunks among many may get fewer than
  `top_k` results; pgvector 0.8 iterative scans address this.
- Ingestion is synchronous, and scanned (image-only) PDFs are rejected because there is no OCR.
- Chat history lives in the browser tab only; each question is answered on its own, with no memory of earlier turns.

## Older Streamlit UI

`ui/app.py` is the first UI, kept as an alternative. It talks to the same API:
`pip install -r requirements-ui.txt && streamlit run ui/app.py` (it uses the non-streaming endpoint).
