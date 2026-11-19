FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# the pinned versions (requirements-ml.lock) so a rebuild gives the same image; see README to update them
COPY requirements-ml.lock ./
RUN pip install --no-cache-dir -r requirements-ml.lock

# Download the search model while building, so the first start needs no network and is not slow
ENV FASTEMBED_CACHE_PATH=/app/.cache/fastembed
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"

COPY . .
RUN python manage.py collectstatic --noinput \
    && useradd --create-home --uid 10001 app && chown -R app:app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request as u,sys; r=u.Request('http://127.0.0.1:8000/', headers={'X-Forwarded-Proto':'https'}); sys.exit(0 if u.urlopen(r, timeout=4).status == 200 else 1)"
# One process (the embedding model is loaded once) with threads, so a streaming answer does not block other
# users, and a 180 s timeout, because gunicorn's default 30 s would kill long uploads and slow answers.
CMD ["sh", "-c", "python manage.py migrate --noinput && exec gunicorn config.wsgi:application --bind 0.0.0.0:8000 --worker-class gthread --workers ${WEB_CONCURRENCY:-1} --threads ${GUNICORN_THREADS:-8} --timeout ${GUNICORN_TIMEOUT:-180}"]
