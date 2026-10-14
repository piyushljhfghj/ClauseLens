# ClauseLens — Architecture

> Status: design frozen at the start of the project; sections are marked with the phase that
> implements them. Anything labelled **(Rec)** is an engineering decision, not an assignment
> requirement. Kept in sync with the code: if they disagree, the code is wrong or this file is stale.

## 1. Goal and non-negotiables

A user uploads contracts (PDF/DOCX) and asks questions. Every factual answer is backed by
quotations that **our code** has located in the extracted document text before they are shown.

1. The LLM is never trusted for quote validity, page numbers, offsets or locations.
2. A quote that cannot be located is never presented as genuine (removed or clearly unverified).
3. If the answer is not in the documents, say so; never fall back on general legal knowledge.
4. If only part of a document was read, the answer must not imply the whole was read.

## 2. System overview

```
Browser (Next.js on Vercel)
  │  REST + Server-Sent Events, called directly (no Vercel proxy: serverless timeouts would
  │  cut long streams)
  ▼
FastAPI (Docker: Hugging Face Spaces, or Railway)
  api/          routers, SSE, error envelope, request ids
  ingestion/    detect → extract (PyMuPDF | python-docx | Tesseract OCR) → furniture removal
                → canonical text → structure (sections) → chunking → embeddings
  retrieval/    query analysis → BM25 ∥ dense → RRF → cross-encoder rerank → context + coverage
  generation/   OpenAI-compatible client → prompt → streamed answer + citation lines
  verification/ normalise → match → disambiguate → locate        (never imports generation/)
  agent/        tool registry → bounded tool loop → evidence ledger
  comparison/   segment → align → deterministic signals → LLM explanation
  jobs/         DB-backed job queue, restart recovery
  ▼
PostgreSQL 16 + pgvector (Docker locally, Supabase in production) · file storage (local disk /
Supabase Storage)
```

| Concern | Choice | Reason |
|---|---|---|
| Frontend | Next.js 16 (App Router) + TypeScript + Tailwind | Assignment prefers Next.js |
| Backend | FastAPI, SQLAlchemy 2 (async, asyncpg), Alembic, Pydantic v2, uv | Typed; Python owns document/ML tooling |
| Vectors | pgvector, **exact scan, no ANN index (Rec)** | Queries are scoped to 1–5 documents (~600 chunks each): exact is milliseconds, and filtered HNSW can silently drop results |
| Keyword | **In-process BM25 (Rec)** over stored tokens | Real BM25 (Postgres `ts_rank` is not); legal-aware tokenizer keeps `8.2`, `AED`, `100,000` |
| Embeddings | `BAAI/bge-small-en-v1.5` via fastembed (ONNX, CPU) **(Rec)** | Free, no PyTorch, ~20–30 s for a 150-page contract |
| Reranker | `Xenova/ms-marco-MiniLM-L-6-v2` via fastembed, env-toggled | ~0.3 s for 40 candidates on CPU |
| LLM | Any OpenAI-compatible endpoint: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | Provider-agnostic as required; must support tool calling (Part C). Concrete model chosen in Phase 5 after a price/quality comparison |
| Viewer | react-pdf (pdf.js) + **our own highlight overlay** from stored word boxes | Avoids matching against pdf.js's differently laid-out text layer |
| Hosting **(Rec)** | Vercel (web) · HF Spaces Docker (API, 2 vCPU/16 GB free) · Supabase (Postgres + pgvector + Storage) | Free; Render's free 512 MB is too small for ONNX + OCR. API image is platform-neutral so Railway is a drop-in fallback |

## 3. Data model (Phase 2+; one migration per phase)

Central idea: **one canonical text per document**; every location (page, block, section, chunk,
citation) is a `[char_start, char_end)` range into it.

| Table | Key columns |
|---|---|
| `documents` | id, filename, file_type, sha256, storage_key, status (queued/processing/ready/failed), stage, progress, error_code, error_message, page_count, ocr_page_count |
| `document_texts` | document_id, canonical_text |
| `document_pages` (PDF) | page_number, width, height, rotation, char_start/end, extraction (text/ocr), `words` jsonb `[[cs,ce,x0,y0,x1,y1],…]` |
| `blocks` | ordinal, kind (paragraph/heading/table_row/list_item), section_id, char/page ranges, docx_para_index |
| `sections` | parent_id, level, number (`8.2(a)`), number_normalized, title, char/page ranges |
| `chunks` | section_id, section_path, char/page ranges, text, context_text, lexical_tokens, `embedding vector(384)` |
| `chats`, `chat_documents` | a chat is bound to 1..n documents (history per document / document set) |
| `messages` | role, mode (rag/agent), content, status (streaming/complete/stopped/error), grounding, coverage, evidence, agent_trace |
| `citations` | claimed_quote, claimed_evidence_id, claimed_page (audit only), status (verified/unverified/ambiguous), match_type, matched_text, char/page ranges, occurrence_count, rects, failure_reason |
| `comparisons`, `comparison_items` | change_type, aligned sections, word_diff, signals, category, significance, explanation |
| `jobs` | kind (ingest/compare), status, attempts, heartbeat_at — stale jobs are re-queued on startup |

## 4. API (all errors: `{"error": {code, message, details, request_id}}`)

| Endpoint | Phase |
|---|---|
| `GET /api/health` (readiness: DB, pgvector, migrations; 503 when degraded) · `GET /api/health/live` | 1 ✅ |
| `POST /api/documents` (multipart, magic-byte validation) · `GET /api/documents[/{id}]` · `DELETE` · `GET /{id}/file` · `POST /{id}/reprocess` | 2 ✅ |
| `GET /api/documents/{id}/outline` (section tree) · `/{id}/pages` (per-page extraction status) | 3 ✅ |
| `GET /api/documents/{id}/layout` (word boxes / blocks for the viewer) | 7 |
| `POST /api/chats` · `GET /api/chats?document_id=` · `GET/DELETE /api/chats/{id}` | 8 |
| `POST /api/chats/{id}/messages` → SSE · `POST /api/chats/{id}/messages/{mid}/stop` | 5–8 |
| `POST /api/comparisons` · `GET /api/comparisons/{id}?type=&significance=&sort=` | 11 |

SSE events: `message_start`, `status`, `agent_step`, `evidence`, `token`, `citation`, `message_end`, `error`.
TypeScript types are generated from the OpenAPI schema (`npm run gen:api`).

## 5. Ingestion pipeline (Phases 2–3)

**Implemented in Phase 2 ✅**

- *Upload validation* (synchronous, before anything is stored): type from magic bytes
  cross-checked with the extension (`%PDF-`; ZIP containing `word/document.xml`; OLE2 means
  legacy `.doc` or a password-protected `.docx`); size limit enforced from `Content-Length` by
  middleware and again while streaming; DOCX zip-bomb guard; SHA-256 duplicate check (failed
  documents may be re-uploaded). The file is written to storage *before* the DB transaction
  commits, so a row never points at a missing file.
- *Job queue*: `jobs` table claimed with `UPDATE … WHERE id = (SELECT … FOR UPDATE SKIP LOCKED)`;
  an in-process asyncio worker (woken immediately on upload, otherwise polling) runs blocking
  work in threads. A heartbeat (every 2 s) also copies stage/progress to the document for the UI.
  `ProcessingFailure` (problem with the file) → document failed, not retried; any other
  exception → retried with exponential backoff up to `JOB_MAX_ATTEMPTS`, then failed with a
  generic message (internals only in logs); stale heartbeat (process killed) → re-queued, or
  failed as `PROCESSING_INTERRUPTED` when out of attempts. Deleting a document mid-job is safe.
- *Inspection* (the only processing stage so far): page count, page limit, password-protected
  / corrupt / empty detection, pages without a text layer. All pages without text →
  `SCANNED_NO_TEXT` failure. Owner-password-only PDFs (print/copy restrictions) are readable.
  MuPDF repairs damaged files; unrecoverable pages show up as pages without text.
- *Storage*: `Storage` protocol with `LocalStorage` (dev) and `SupabaseStorage` (REST, private
  bucket, server-side key; files are always streamed through the API).
- In Phase 2 `ready` meant *stored and inspected*. Phase 3 added the stages below; documents
  processed before that have `index_version = NULL` and are reported as `needs_reprocessing`.

**Implemented in Phase 3 ✅** — `download → inspect → extract (+OCR) → structure → chunk → embed → save`

1. **Extract words with boxes** (`pdf_extract.py`): PyMuPDF `rawdict` (per-character boxes; ligatures
   expanded), grouped into words on whitespace. Boxes are transformed with the page's rotation
   matrix into *displayed-page* coordinates (PDF points, origin top-left, page size = `page.rect`),
   so the viewer only scales. MuPDF blocks are split wherever a line starts a numbered clause.
2. **OCR** (`ocr.py`): a page is OCR'd when (a) it has < 20 text-layer characters but visible
   content (dark pixels in a 30-DPI render: scans, handwriting, vector-drawn text), or (b) it has
   < 200 characters and images cover ≥ 50% of it (a scan carrying a typed stamp / Bates number);
   in case (b) the OCR text replaces the text layer only if it adds ≥ 50 characters. Rendering at
   `OCR_DPI` (grey) and Tesseract `image_to_data` give word boxes in the same coordinate space,
   mean confidence stored per page. Page outcome: `text` | `ocr` | `none` with note `blank`
   (no visible content) | `ocr_unavailable` | `ocr_failed` | `ocr_no_text` (visible content, no
   recognisable text). `unreadable_page_count` counts those unreadable notes (content may exist
   but was not read) and is surfaced in the API/UI with explicit warnings. An OCR page whose mean
   word confidence is below 50% keeps its text and stays indexed, but gets note
   `ocr_low_quality`: counted separately in `low_quality_page_count`, warned about ("content may be
   largely unrecognised; answers based on these pages may be incomplete") and shown as a distinct
   striped amber cell in the coverage strip — partially read, neither "read" nor "unreadable".
   A PDF fails (`SCANNED_NO_TEXT`) only if no page yields text.
3. **Furniture** (`furniture.py`): lines in the top/bottom 10% of the page whose digit-normalised
   text repeats on ≥ 50% of pages (min. 2), or that are bare page numbers, are dropped.
4. **DOCX** (`docx_extract.py`, `docx_numbering.py`): body walked in order (paragraphs, tables,
   content controls); run text read directly so tracked insertions count, deletions do not, and
   tab-stop definitions are not text; Word auto-numbering resolved from `numbering.xml`
   (levels, formats, restarts, start overrides, style-inherited numbering) and prefixed.
   No pages. Not extracted: text boxes, footnotes, headers/footers.
5. **Canonical text** (`model.py`): words joined by spaces, lines by `\n`, blocks/pages by
   `\n\n` (all whitespace, so verification can collapse them). Page and block ranges recorded.
6. **Structure** (`structure.py`): numbered patterns (`8.`, `8.2`, `Section 12.3`, `Article IV`,
   `Schedule 2`; guards against "30 days", "100,000", "2026.") + emphasis (DOCX heading styles /
   outline level / all-runs-bold paragraphs; PDF font ≥ 1.15× body size, bold, ALL CAPS) →
   section tree; each section spans until the next sibling-or-higher section. Emphasis rules are
   not applied to OCR text (no reliable typography; recognition noise is often short and
   upper-case). A run of consecutive unnumbered headings is a title block: only the last line
   before body text becomes a section (numbered and Word-styled headings are never demoted).
   Titles: short heading text, a short leading phrase ending in "." or "?" ("8.2 Termination for
   Convenience. …"), or a single-line numbered question; lead-ins ending in ":" are not titles.
   Untitled sections get a display-only `preview` (opening words) in the outline API.
   `INDEX_VERSION` history: 2 = Phase 3 audit fixes; 3 = low-quality OCR classification
   (migration `0004` adds `low_quality_page_count`). Older indexes are reported as needing
   reprocessing.
7. **Chunking** (`chunking.py`): whole blocks packed to ~1,200 chars (max 1,800; oversized
   paragraphs split at sentence boundaries); never across top-level sections; headings stay with
   their text; **no overlap and no gaps** (tested invariant). `context_text = section path +
   index-normalised text` is embedded; BM25 tokens (`text/tokenize.py`) stored per chunk.
8. **Embeddings** (`embeddings.py`): fastembed `BAAI/bge-small-en-v1.5`, 384-d, L2-normalised,
   lazy-loaded once per process, run in a thread with per-batch progress. Baked into the Docker
   image. Tests inject a deterministic fake; one test exercises the real model.
9. **Save** (`index_writer.py`): document row locked, old index rows deleted and new ones inserted,
   document set `ready` with `index_version`, counts and warnings, all in **one transaction**.
   A failed re-index leaves the previous complete index untouched.

## 6. Retrieval and generation (Phases 4–5, 9–10)

question → deterministic query analysis (section references, quoted phrases, amounts, absence /
exhaustive intent) → per document BM25 top-30 ∥ dense top-30 (+ direct section fetch) → RRF
(k=60) → cross-encoder rerank of top 40 → 8–12 passages under a token budget, section-diverse,
**per-document quotas** for multi-document questions → evidence labelled `[E1]…` → LLM.

Output contract (line-oriented, robust for cheap models):
```
<answer> … claim [1] … </answer>
<citations>
{"n":1,"evidence":"E4","quote":"exact words from the passage"}
</citations>
```
Only `<answer>` text is streamed; each citation line is verified as soon as it completes.
Insufficient evidence → the fixed sentence *"I couldn't find sufficient support for that answer in the selected document(s)."*

**Large-document honesty**: retrieval always runs over every chunk of the document; the model
reads only K passages and the response says so via `coverage` (pages total/indexed, passages
read, which pages). The prompt forbids asserting absence; absence/exhaustive questions widen
retrieval; extraction gaps are reported.

## 7. Citation verification (Phase 6) — independent of the LLM

1. **Normalise with an offset map**: NFKC (ligatures), unify quotes/dashes, drop soft hyphens /
   zero-width chars, join end-of-line hyphenation, collapse whitespace, case-fold. Punctuation kept.
2. **Tiered match** over the whole document's normalised text: exact/normalised → whitespace-insensitive
   (≥ 25 non-space chars) → elided (`…` segments in order within 600 chars) → otherwise
   **unverified** (fuzzy search only produces a diagnostic "closest passage"; never verified,
   never silently substituted). Quotes under 5 words are unverified as too short.
3. **Duplicates**: prefer the occurrence inside the cited evidence chunk → any retrieved chunk →
   LLM page claim as tie-breaker only → else `ambiguous` with an occurrence count.
4. **Locate**: canonical range → pages → overlapping word boxes → per-line rectangles (multiple
   pages for page-spanning quotes). DOCX: block + in-block ranges. The UI displays `matched_text`
   (the document's own words), not the model's string.
5. **Multi-document**: each quote is verified only against its own evidence's document.
6. **Grounding**: grounded / partially grounded / ungrounded (warning, never shown as supported) / not found.

## 8. Agent — Part C, Option 2 (Phase 12)

Tools (validated by Pydantic): `search_document(query, document_id?, top_k≤8)`,
`get_section(section_number, document_id?)`, `list_clauses(document_id?)`. Loop: up to
**6 rounds**, 4 calls per round, 16 calls total, 10 s per tool, 90 s overall. Unknown tools, bad
JSON, missing/invalid args are returned to the model as structured errors (3 consecutive errors
end research). Each step emits an `agent_step` event with a label derived from the real call.
The final answer is a separate streamed call over the evidence ledger only, then the same verifier.

## 9. Comparison (Phase 11)

Segment both versions into clauses → align (number/title anchors, then embedding + fuzzy
similarity with Hungarian assignment; handles renumbering and moves) → unchanged / modified /
added / removed → deterministic signals (amounts, currencies, durations, dates, modal verbs,
negation, key terms; formatting-only changes classified without the LLM) → LLM explains only
substantive candidates, with significance = max(rule floor, LLM) → UI filters and sorts by significance.

## 10. Cross-cutting foundations (Phase 1 ✅)

- **Config**: `pydantic-settings`; no module reads `os.environ` directly; secrets as `SecretStr`.
- **Errors**: `AppError` subclasses with stable codes → one JSON envelope; unhandled exceptions are logged with traceback and returned as a generic 500 (no internals leaked).
- **Logging**: stdlib logging, text or JSON format, request id on every line, `X-Request-ID` header echoed (accepted from proxies if safe).
- **Middleware**: pure ASGI (not `BaseHTTPMiddleware`, which interferes with streaming and disconnect detection needed for SSE stop).
- **Health**: `/api/health/live` (process only, for platform restarts) vs `/api/health` (DB, pgvector, migration head; 503 when degraded).
- **Migrations**: Alembic, async engine, deterministic constraint naming; migrations run on container start (`RUN_MIGRATIONS_ON_START`).

## 11. Testing strategy

pytest with a scripted `FakeLLM` (no API key needed) and programmatically generated fixture
contracts (PyMuPDF / python-docx). Required cases: correct quote; hallucinated quote; whitespace
differences; line/page-break spanning; duplicates; no-answer; 150-page retrieval; multi-document
attribution; wrong LLM page number; malformed tool call; max-rounds cap; comparison
added/removed/modified. Plus a small golden-question eval reporting retrieval recall, verification
rate and not-found accuracy. Frontend: Vitest for logic, Playwright smoke test later.
