FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements-ml.txt

COPY . .
RUN python manage.py collectstatic --noinput
EXPOSE 8000
# One process (the embedding model is loaded once) with threads, so a streaming answer does not block other
# users, and a 180 s timeout, because gunicorn's default 30 s would kill long uploads and slow answers.
CMD ["sh", "-c", "python manage.py migrate --noinput && exec gunicorn config.wsgi:application --bind 0.0.0.0:8000 --worker-class gthread --workers ${WEB_CONCURRENCY:-1} --threads ${GUNICORN_THREADS:-8} --timeout ${GUNICORN_TIMEOUT:-180}"]
