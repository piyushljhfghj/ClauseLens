# ClauseLens

**Evidence-grounded contract intelligence.** Upload contracts (PDF/DOCX), ask questions, and get
answers backed by quotations that the backend has located in the document text before they are
shown, so an invented quote is never presented as genuine.

> **Project status: Phase 2 of 15 (document upload & library).** What exists today is listed under
> [What works](#what-works-today). Features described in the architecture doc are planned, not
> implemented, until they appear in that list.

- Design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Stack: Next.js 16 · FastAPI · PostgreSQL 16 + pgvector · SQLAlchemy 2 / Alembic · uv

## What works today

- [x] Monorepo with a Next.js frontend and a FastAPI backend
- [x] PostgreSQL + pgvector via Docker Compose; Alembic migrations (pgvector enabled)
- [x] `GET /api/health` readiness check (database, pgvector, migration version) and `GET /api/health/live`
- [x] Consistent JSON error envelope, request ids, structured logging
- [x] Frontend status page showing live backend/database health (frontend ↔ backend over CORS)
- [x] Backend tests (pytest, including integration tests against real Postgres) and frontend tests (Vitest)
- [x] **Upload** PDF/DOCX (drag & drop, upload progress). File type is decided from the file's contents
      cross-checked with its extension; unsupported, legacy `.doc`, password-protected `.docx`,
      mismatched, empty, oversized and duplicate files are rejected immediately with a specific message
- [x] **Background processing** with live status (stage + progress) via a Postgres-backed job queue:
      retries with backoff, and jobs interrupted by a restart/crash are recovered
- [x] **Inspection**: page count; password-protected, corrupt and empty files fail with a clear reason;
      **scanned PDFs with no text layer fail with an explanation** instead of becoming an empty document;
      partly scanned PDFs are flagged ("3 of 40 pages have no selectable text")
- [x] **Document library**: list, open (PDF shown inline; DOCX downloadable), download, delete
      (removes the database rows and the stored file), retry failed processing
- [ ] Text extraction, OCR, structure, chunking, embeddings — Phase 3
      (until then "Ready" means *stored and inspected*, not yet searchable)
- [ ] Chat, retrieval, verified quotes, highlighting, comparison, research agent — Phases 4–12

## Repository layout

```
backend/    FastAPI app (app/), Alembic migrations (alembic/), tests/, Dockerfile
frontend/   Next.js app (app/, components/, lib/)
docs/       Architecture and submission notes
docker-compose.yml   Local PostgreSQL + pgvector (and optional API container)
```

## Run locally (Windows PowerShell)

Prerequisites: [Docker Desktop](https://www.docker.com/products/docker-desktop/),
[uv](https://docs.astral.sh/uv/), Node.js 22+. uv downloads Python 3.12 automatically.

```powershell
# 1. Database (from the repo root)
docker compose up -d db

# 2. Backend (new terminal)
cd backend
Copy-Item .env.example .env
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --reload --port 8000

# 3. Frontend (another terminal)
cd frontend
Copy-Item .env.example .env.local
npm install
npm run dev
```

Open http://localhost:3000 — the status card should say **All systems operational**.
API docs: http://localhost:8000/docs.

### Tests and checks

```powershell
cd backend;  uv run pytest; uv run ruff check .
cd frontend; npm test; npm run lint; npm run typecheck
```

Database integration tests run against a separate `clauselens_test` database (created and
migrated automatically) when the Docker database is up, and are skipped otherwise.

## Configuration

Backend (`backend/.env`, see `backend/.env.example`):

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Postgres URL (`postgres://` forms are converted automatically) |
| `DB_SSL` | `true` for hosted Postgres (Supabase) |
| `DB_STATEMENT_CACHE_SIZE` | `0` only for transaction-mode poolers (Supabase port 6543) |
| `CORS_ORIGINS` / `CORS_ORIGIN_REGEX` | Browser origins allowed to call the API |
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` | Any OpenAI-compatible provider (from Phase 5) |
| `LOG_LEVEL` / `LOG_FORMAT` | `INFO` … / `text` or `json` |
| `MAX_UPLOAD_MB` / `MAX_PAGES` | Upload size limit (default 25 MB) and page ceiling (default 1000) |
| `STORAGE_BACKEND` | `local` (files under `STORAGE_LOCAL_DIR`, dev) or `supabase` (production) |
| `SUPABASE_URL` / `SUPABASE_SERVICE_KEY` / `SUPABASE_STORAGE_BUCKET` | Supabase Storage (private bucket; server-side key only) |
| `WORKER_ENABLED` / `JOB_MAX_ATTEMPTS` | In-process background worker; retries for unexpected errors |
| `RUN_MIGRATIONS_ON_START` | Container runs `alembic upgrade head` before starting (default `true` in the image) |

Frontend (`frontend/.env.local`): `NEXT_PUBLIC_API_BASE_URL` — baked in at build time.

No secrets are committed: `.env` files are git-ignored; production values live in each platform's secrets settings.

## Deployment

Free tier: **Vercel** (frontend) · **Hugging Face Spaces, Docker SDK** (API) · **Supabase**
(Postgres + pgvector). The API image is platform-neutral (`PORT` env var, migrations on start),
so it can move to Railway without code changes.

**1. Supabase (database)**
1. Create a project and save the database password.
2. *Connect* → **Session pooler** connection string (IPv4-compatible; the direct connection is IPv6-only and unreachable from HF Spaces):
   `postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres`
3. pgvector is enabled by the first migration (or via *Database → Extensions → vector*).
4. *Storage* → create a **private** bucket named `documents`. Copy the project URL and the
   **secret / service_role** key (*Project Settings → API Keys*). This key bypasses row-level
   security: it is only ever set on the backend, never in the frontend.

**2. Hugging Face Space (API)**
1. Create a **public** Space, SDK **Docker**, blank template, CPU basic (free). (A private Space would require auth that the browser can't send.)
2. *Settings → Variables and secrets*:
   - secret `DATABASE_URL` = the Supabase session pooler URL
   - variable `DB_SSL` = `true`
   - variable `CORS_ORIGINS` = your Vercel URL, e.g. `https://clauselens.vercel.app`
   - variable `STORAGE_BACKEND` = `supabase`, variable `SUPABASE_URL` = `https://<ref>.supabase.co`,
     secret `SUPABASE_SERVICE_KEY` = the secret key (container disks on Spaces are wiped on restart,
     so local file storage must not be used in production)
3. Push the `backend/` directory to the Space (it contains the Dockerfile and the Space's README header):
   ```powershell
   git remote add hf https://huggingface.co/spaces/<hf-user>/<space-name>
   git subtree split --prefix backend -b hf-deploy
   git push hf hf-deploy:main --force
   ```
   Use a Hugging Face **write** access token as the password when prompted.
4. API URL: `https://<hf-user>-<space-name>.hf.space` → check `/api/health`.

**3. Vercel (frontend)**
1. Import the GitHub repository; set **Root Directory** to `frontend`.
2. Environment variable `NEXT_PUBLIC_API_BASE_URL` = the HF Space URL (no trailing slash). Deploy.
3. Make sure the final Vercel URL is in the Space's `CORS_ORIGINS`.

**Railway fallback:** New project → deploy from the GitHub repo → root directory `backend`
(Dockerfile detected) → same environment variables. Railway sets `PORT` itself.
