---
title: ClauseLens API
emoji: 📄
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Backend API for ClauseLens (evidence-grounded contract Q&A)
---

# ClauseLens API

FastAPI backend for ClauseLens. The YAML header above configures the Hugging Face Space
this directory is deployed to; it is ignored everywhere else.

See the [project README](../README.md) for setup, and [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md)
for the design.

```powershell
uv sync                      # install dependencies (creates .venv)
uv run alembic upgrade head  # apply database migrations
uv run uvicorn app.main:app --reload --port 8000
uv run pytest                # tests
uv run ruff check .          # lint
```
