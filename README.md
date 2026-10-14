# ClauseLens

**Evidence-grounded contract intelligence.** Upload contracts (PDF/DOCX), ask questions, and get
answers backed by quotations that the backend has located in the document text before they are
shown, so an invented quote is never presented as genuine.

> **Project status: Phase 5 of 15 (grounded answering with verified quotations).** What exists today is listed under
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
- [x] **Text extraction with positions**: one canonical text per document; every PDF word keeps
      its character range and its box on the displayed page (rotation handled) for later
      citation highlighting. Running headers/footers and page numbers are removed so sentences
      crossing a page break stay contiguous
- [x] **OCR** (Tesseract) for pages without a text layer, per page; each page records how it
      was read (text layer / OCR with confidence / low-quality OCR / not read and why). A PDF
      only fails if *no* page has readable text; pages that could not be read, and OCR pages
      read with low confidence (< 50%, content possibly largely unrecognised), are counted and
      reported separately, never hidden
- [x] **DOCX extraction** including tables, tracked changes (read as accepted) and Word
      auto-numbering ("8.2", "(a)") resolved into the text
- [x] **Section tree** from clause numbering (8 / 8.2 / Section 12.3 / Article IV / Schedule 2)
      and headings (DOCX heading styles; PDF font size, bold, ALL CAPS)
- [x] **Chunks** that cover the whole text with no gaps or overlaps and never span two top-level
      sections; **embeddings** with `BAAI/bge-small-en-v1.5` (local, fastembed/ONNX) in pgvector;
      BM25 tokens stored for Phase 4
- [x] **Index status in the UI**: *Indexed* vs *Not indexed* (stored and checked only, e.g. files
      uploaded before Phase 3) with Reprocess / Reprocess all; coverage summary, per-page
      extraction strip, warnings and detected structure on the document page
- [x] Reprocessing replaces a document's whole index in one transaction (never half-built)
- [x] **Hybrid retrieval** (`POST /api/retrieval/search`): semantic (pgvector) + BM25 (in SQL) +
      exact "phrase" + clause-number lookup, fused with Reciprocal Rank Fusion; scoped to the
      selected document(s); deterministic; near-duplicates removed; off-topic or empty queries
      return nothing; each passage carries its exact source location and OCR quality
- [x] **Grounded answers** (`POST /api/answers`): only strong evidence is given to the LLM (weak
      retrieval tails are filtered; token budget; each document represented), every quotation
      is verified against the stored canonical text before it is shown (exact character and page
      range, OCR quality), and an answer with no verified quotation is replaced by an explicit
      "insufficient support" statement. Works with any OpenAI-compatible provider; no reranker
- [ ] Highlighting, streaming chat, comparison, research agent, full verification — Phases 6–12

### Known limitations (extraction)

- PDF reading order follows the PDF's own text order; complex multi-column layouts may interleave.
- DOCX text boxes, footnotes and headers/footers are not extracted.
- OCR is English-only by default (`OCR_LANGUAGE`); low-confidence OCR pages are flagged.
- Heading detection is heuristic: unusual numbering schemes may be missed (the text is still
  fully indexed as passages).

## Repository layout

```
backend/    FastAPI app (app/), Alembic migrations (alembic/), tests/, Dockerfile
frontend/   Next.js app (app/, components/, lib/)
docs/       Architecture and submission notes
docker-compose.yml   Local PostgreSQL + pgvector (and optional API container)
```

## Run locally (Windows PowerShell)

Prerequisites: [Docker Desktop](https://www.docker.com/products/docker-desktop/),
[uv](https://docs.astral.sh/uv/), Node.js 22+, and [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
(for scanned PDFs; if it is not on PATH set `TESSERACT_CMD`). uv downloads Python 3.12
automatically; the embedding model (~65 MB) downloads on first use into `backend/.cache/`.

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
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` | Any OpenAI-compatible provider; blank disables answering (503) |
| `LLM_TIMEOUT_S` / `LLM_MAX_OUTPUT_TOKENS` / `LLM_TEMPERATURE` / `LLM_SEED` | Request limits; temperature 0 for repeatable answers |
| `ANSWER_*` | Evidence selection: candidates (20), max passages (8), token budget (3000), weak-match thresholds |
| `LOG_LEVEL` / `LOG_FORMAT` | `INFO` … / `text` or `json` |
| `MAX_UPLOAD_MB` / `MAX_PAGES` | Upload size limit (default 25 MB) and page ceiling (default 1000) |
| `STORAGE_BACKEND` | `local` (files under `STORAGE_LOCAL_DIR`, dev) or `supabase` (production) |
| `SUPABASE_URL` / `SUPABASE_SERVICE_KEY` / `SUPABASE_STORAGE_BUCKET` | Supabase Storage (private bucket; server-side key only) |
| `OCR_ENABLED` / `OCR_LANGUAGE` / `OCR_DPI` / `TESSERACT_CMD` | OCR of scanned pages (Tesseract) |
| `EMBEDDING_MODEL` / `EMBEDDING_CACHE_DIR` / `EMBEDDING_BATCH_SIZE` | Local embedding model (only `BAAI/bge-small-en-v1.5`, 384-dim, is supported) |
| `WORKER_ENABLED` / `JOB_MAX_ATTEMPTS` | In-process background worker; retries for unexpected errors |
| `RETRIEVAL_*` | Candidates per retriever (30), default/max `top_k` (10/50), RRF k (60), minimum semantic similarity (0.55), BM25 k1/b, query length and document limits |
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
