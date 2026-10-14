# ClauseLens API image. Platform-neutral: runs on Hugging Face Spaces (Docker SDK), Railway,
# Render or Fly.io unchanged. The platform injects configuration as environment variables.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

# Tesseract OCR (English) for scanned pages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (cached layer; only rebuilt when the lockfile changes).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Bake the embedding model (~65 MB) into the image: no download on a cold start, and the app
# works even if the model hub is unreachable at runtime.
ENV EMBEDDING_CACHE_DIR=/app/.cache/fastembed
RUN /opt/venv/bin/python -c "from fastembed import TextEmbedding; \
TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/app/.cache/fastembed')"

COPY . .

# Hugging Face Spaces runs containers as uid 1000; doing the same everywhere avoids surprises.
# /app/storage is only used with STORAGE_BACKEND=local (e.g. the docker compose "full" profile).
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/storage && chown -R appuser:appuser /app/storage /app/.cache
USER appuser

ENV PATH="/opt/venv/bin:${PATH}" \
    APP_ENV=production \
    LOG_FORMAT=json \
    RUN_MIGRATIONS_ON_START=true \
    PORT=7860

# 7860 is the Hugging Face Spaces default; Railway/Render override PORT at runtime.
EXPOSE 7860

CMD ["sh", "scripts/start.sh"]
