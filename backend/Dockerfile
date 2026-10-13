# ClauseLens API image. Platform-neutral: runs on Hugging Face Spaces (Docker SDK), Railway,
# Render or Fly.io unchanged. The platform injects configuration as environment variables.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first (cached layer; only rebuilt when the lockfile changes).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .

# Hugging Face Spaces runs containers as uid 1000; doing the same everywhere avoids surprises.
# /app/storage is only used with STORAGE_BACKEND=local (e.g. the docker compose "full" profile).
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/storage && chown appuser:appuser /app/storage
USER appuser

ENV PATH="/opt/venv/bin:${PATH}" \
    APP_ENV=production \
    LOG_FORMAT=json \
    RUN_MIGRATIONS_ON_START=true \
    PORT=7860

# 7860 is the Hugging Face Spaces default; Railway/Render override PORT at runtime.
EXPOSE 7860

CMD ["sh", "scripts/start.sh"]
