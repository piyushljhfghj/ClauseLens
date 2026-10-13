#!/bin/sh
# Container entrypoint: optionally apply database migrations, then start the API.
set -e

if [ "${RUN_MIGRATIONS_ON_START:-false}" = "true" ]; then
  echo "Applying database migrations..."
  alembic upgrade head
fi

# --proxy-headers: hosting platforms terminate TLS in front of us and forward the real scheme/IP.
exec uvicorn app.main:app \
  --host 0.0.0.0 \
  --port "${PORT:-7860}" \
  --proxy-headers \
  --forwarded-allow-ips="*"
