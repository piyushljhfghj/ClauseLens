"""Hybrid retrieval against real PostgreSQL + pgvector, through the HTTP API.

Embeddings come from `ConceptEmbedder` (deterministic synonyms -> shared dimensions), so semantic
paraphrase behaviour is exact and repeatable. One test at the end uses the real model.
"""

import re
import uuid

import httpx
import pymupdf
import pytest
from fastapi import FastAPI
from sqlalchemy import delete, select, update

from app.core.config import Settings
from app.db.models import Document, DocumentChunk, DocumentPage, DocumentText
from app.db.session import get_sessionmaker
from app.ingestion import pdf_extract
from app.ingestion.model import SourceBlock, SourceLine, SourceWord
from app.ingestion.ocr import OcrResult
from app.main import create_app
from tests.conftest import truncate_all
from tests.fakes import ConceptEmbedder
from tests.fixtures import factory

SEARCH = "/api/retrieval/search"

CONTRACT_A = [
    "1. Definitions. Services means the consulting services described in the Order Form.\n\n"
    "2. Term. This Agreement starts on the Effective Date and continues for 24 months.",
    "8. Termination\n\n8.1 The Company may terminate immediately for material breach.\n\n"
    "8.2 Either party may terminate this Agreement by giving thirty (30) days' written notice "
    "to the other party.",
    "12. Liability\n\n12.1 The total aggregate liability of the Supplier shall not exceed "
    "AED 100,000. Purchase order reference ACME-0000123.",
    "15. Governing Law. This Agreement is governed by the laws of the United Arab Emirates.",
    "Schedule 1 - Pricing\n\nPricing applies as set out in the Order Form agreed by the parties.",
    # The previous page again, e.g. a page duplicated when the contract was scanned/assembled.
    "Schedule 1 - Pricing\n\nPricing applies as set out in the Order Form agreed by the parties.",
]
CONTRACT_B = [
    "12. Liability\n\n12.1 The total aggregate liability of the Supplier shall not exceed "
    "AED 1,000,000.",
    "15. Governing Law. This Agreement is governed by the laws of England and Wales.",
]


def contract_pdf(pages: list[str]) -> bytes:
    document = pymupdf.open()
    for text in pages:
        document.new_page().insert_textbox(pymupdf.Rect(72, 72, 520, 760), text, fontsize=11)
    return document.tobytes()


@pytest.fixture
def fake_embedder() -> ConceptEmbedder:  # overrides the conftest fixture used by live_app
    return ConceptEmbedder()


async def upload(api: httpx.AsyncClient, app: FastAPI, name: str, data: bytes) -> str:
    response = await api.post("/api/documents", files={"file": (name, data)})
    assert response.status_code == 202, response.text
    while await app.state.worker.run_once():
        pass
    document = (await api.get(f"/api/documents/{response.json()['id']}")).json()
    assert document["indexed"], document
    return document["id"]


async def search(api: httpx.AsyncClient, query: str, *ids: str, **extra) -> dict:  # type: ignore[no-untyped-def,type-arg]
    response = await api.post(SEARCH, json={"query": query, "document_ids": list(ids), **extra})
    assert response.status_code == 200, response.text
    return response.json()


def collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text)


@pytest.fixture
async def contract_a(api: httpx.AsyncClient, live_app: FastAPI) -> str:
    return await upload(api, live_app, "contract-a.pdf", contract_pdf(CONTRACT_A))


# --- lexical, phrase, clause, semantic, hybrid --------------------------------------------------


async def test_identifier_is_found_lexically(api: httpx.AsyncClient, contract_a: str) -> None:
    result = await search(api, "ACME-0000123", contract_a)
    top = result["passages"][0]
    assert "ACME-0000123" in top["text"]
    assert top["signals"]["bm25_rank"] == 1
    assert result["no_match"] is False


async def test_clause_number_uses_the_section_lookup(
    api: httpx.AsyncClient, contract_a: str
) -> None:
    result = await search(api, "clause 8.2", contract_a)
    top = result["passages"][0]
    assert "8.2 Either party may terminate" in collapse(top["text"])
    assert top["signals"]["section_rank"] is not None
    assert result["analysis"]["section_refs"] == ["8.2"]


async def test_exact_phrase(api: httpx.AsyncClient, contract_a: str) -> None:
    result = await search(api, '"thirty (30) days"', contract_a)
    top = result["passages"][0]
    assert "thirty (30) days" in collapse(top["text"])
    assert top["signals"]["phrase_rank"] == 1


async def test_amount_with_thousands_separator(api: httpx.AsyncClient, contract_a: str) -> None:
    result = await search(api, "AED 100,000", contract_a)
    assert "AED 100,000" in result["passages"][0]["text"]


async def test_semantic_paraphrase_without_shared_keywords(
    api: httpx.AsyncClient, contract_a: str
) -> None:
    result = await search(api, "how can it be ended early?", contract_a)
    top = result["passages"][0]
    assert "terminate" in top["text"]
    # Semantic-only: no query term occurs in the passage.
    assert top["signals"]["bm25_rank"] is None and top["signals"]["dense_rank"] == 1


async def test_hybrid_agreement_ranks_first(api: httpx.AsyncClient, contract_a: str) -> None:
    result = await search(api, "liability cap", contract_a)
    top = result["passages"][0]
    assert "liability" in top["text"]
    assert top["signals"]["dense_rank"] is not None and top["signals"]["bm25_rank"] is not None


# --- no match, empty, malformed -----------------------------------------------------------------


async def test_off_topic_query_returns_nothing(api: httpx.AsyncClient, contract_a: str) -> None:
    result = await search(api, "chocolate cake recipe", contract_a)
    assert result["passages"] == [] and result["no_match"] is True


async def test_stopword_only_query_returns_nothing_with_a_note(
    api: httpx.AsyncClient, contract_a: str
) -> None:
    result = await search(api, "what is the", contract_a)
    assert result["no_match"] is True
    assert "no searchable terms" in " ".join(result["notes"])


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({"query": "   ", "document_ids": [str(uuid.uuid4())]}, "VALIDATION_ERROR"),
        ({"query": "", "document_ids": [str(uuid.uuid4())]}, "VALIDATION_ERROR"),
        ({"query": "fee", "document_ids": []}, "VALIDATION_ERROR"),
        ({"query": "fee", "document_ids": ["not-a-uuid"]}, "VALIDATION_ERROR"),
        ({"query": "x" * 1001, "document_ids": [str(uuid.uuid4())]}, "QUERY_TOO_LONG"),
        ({"query": "fee", "document_ids": [str(uuid.uuid4())], "top_k": 0}, "VALIDATION_ERROR"),
        ({"query": "fee", "document_ids": [str(uuid.uuid4())], "top_k": 51}, "TOP_K_TOO_LARGE"),
        (
            {"query": "fee", "document_ids": [str(uuid.uuid4()) for _ in range(11)]},
            "TOO_MANY_DOCUMENTS",
        ),
    ],
)
async def test_malformed_requests(api: httpx.AsyncClient, body: dict, code: str) -> None:  # type: ignore[type-arg]
    response = await api.post(SEARCH, json=body)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == code


async def test_empty_index_returns_nothing(api: httpx.AsyncClient, contract_a: str) -> None:
    async with get_sessionmaker()() as session:
        await session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == contract_a))
        await session.commit()
    result = await search(api, "liability", contract_a)
    assert result["no_match"] is True
    assert result["coverage"][0]["chunks_searched"] == 0


# --- scope: filtering, multi-document, deleted, stale -------------------------------------------


async def test_document_filter_and_multi_document_scope(
    api: httpx.AsyncClient, live_app: FastAPI, contract_a: str
) -> None:
    contract_b = await upload(api, live_app, "contract-b.pdf", contract_pdf(CONTRACT_B))

    only_b = await search(api, "liability", contract_b)
    assert {p["document_id"] for p in only_b["passages"]} == {contract_b}

    both = await search(api, "liability cap", contract_a, contract_b)
    by_doc = {p["document_id"]: p for p in both["passages"] if "liability" in p["text"]}
    assert set(by_doc) == {contract_a, contract_b}  # each document's own clause is found
    assert "AED 100,000" in by_doc[contract_a]["text"]
    assert "AED 1,000,000" in by_doc[contract_b]["text"]
    assert {c["document_id"] for c in both["coverage"]} == {contract_a, contract_b}


async def test_deleted_document_is_not_found(api: httpx.AsyncClient, contract_a: str) -> None:
    assert (await api.delete(f"/api/documents/{contract_a}")).status_code == 204
    response = await api.post(SEARCH, json={"query": "liability", "document_ids": [contract_a]})
    assert response.status_code == 404
    assert response.json()["error"]["details"]["document_ids"] == [contract_a]


@pytest.mark.parametrize(
    "change",
    [
        {"index_version": 2},  # indexed by an older pipeline
        {"index_version": None},  # stored but never indexed (Phase 2)
        {"embedding_model": "some-other-model"},  # vectors from a different embedding space
        {"status": "processing"},  # being re-indexed right now
    ],
)
async def test_stale_or_unindexed_documents_are_refused(
    api: httpx.AsyncClient,
    contract_a: str,
    change: dict,  # type: ignore[type-arg]
) -> None:
    async with get_sessionmaker()() as session:
        await session.execute(update(Document).where(Document.id == contract_a).values(**change))
        await session.commit()
    response = await api.post(SEARCH, json={"query": "liability", "document_ids": [contract_a]})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "DOCUMENT_NOT_INDEXED"


async def test_reindexed_document_returns_only_current_chunks(
    api: httpx.AsyncClient, live_app: FastAPI, contract_a: str
) -> None:
    before = {p["chunk_id"] for p in (await search(api, "liability", contract_a))["passages"]}
    await api.post(f"/api/documents/{contract_a}/reprocess")
    while await live_app.state.worker.run_once():
        pass
    after = await search(api, "liability", contract_a)
    async with get_sessionmaker()() as session:
        current = set(
            (
                await session.execute(
                    select(DocumentChunk.id).where(DocumentChunk.document_id == contract_a)
                )
            ).scalars()
        )
    assert {p["chunk_id"] for p in after["passages"]} <= current
    assert before.isdisjoint(current)  # the old index is gone and never returned


# --- determinism, duplicates --------------------------------------------------------------------


async def test_repeated_queries_give_identical_results(
    api: httpx.AsyncClient, contract_a: str
) -> None:
    first = await search(api, "agreement parties order form liability", contract_a)
    for _ in range(3):
        assert await search(api, "agreement parties order form liability", contract_a) == first


async def test_near_duplicate_passages_are_collapsed(
    api: httpx.AsyncClient, contract_a: str
) -> None:
    result = await search(api, "pricing order form", contract_a)
    pricing = [p for p in result["passages"] if "Pricing applies" in p["text"]]
    assert len(pricing) == 1
    assert any("near-duplicate" in n for n in result["notes"])


# --- evidence integrity -------------------------------------------------------------------------


async def test_passages_carry_exact_canonical_source_locations(
    api: httpx.AsyncClient, contract_a: str
) -> None:
    result = await search(api, "termination notice liability law", contract_a, top_k=20)
    async with get_sessionmaker()() as session:
        canonical = (await session.get(DocumentText, uuid.UUID(contract_a))).canonical_text  # type: ignore[union-attr]
        pages = (
            (
                await session.execute(
                    select(DocumentPage).where(DocumentPage.document_id == uuid.UUID(contract_a))
                )
            )
            .scalars()
            .all()
        )
    page_ranges = {p.page_number: (p.char_start, p.char_end) for p in pages}
    assert result["passages"]
    for passage in result["passages"]:
        assert passage["text"] == canonical[passage["char_start"] : passage["char_end"]]
        start_page, end_page = passage["page_start"], passage["page_end"]
        assert page_ranges[start_page][0] <= passage["char_start"] < page_ranges[start_page][1]
        assert page_ranges[end_page][0] < passage["char_end"] <= page_ranges[end_page][1]
        assert passage["quality"]["level"] == "text"
        assert passage["document_filename"] == "contract-a.pdf"
        assert passage["section_path"] is not None


async def test_low_quality_ocr_stays_visible_to_consumers(
    api: httpx.AsyncClient, live_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    def low_confidence_ocr(page, *, dpi, language, page_index):  # type: ignore[no-untyped-def]
        words = [SourceWord(w, 72 + 70 * i, 100, 130 + 70 * i, 112) for i, w in
                 enumerate("indemnity obligations survive termination".split())]  # fmt: skip
        return OcrResult([SourceBlock([SourceLine(words)], page_index=page_index)], 0.31)

    monkeypatch.setattr(pdf_extract, "ocr_page", low_confidence_ocr)
    monkeypatch.setattr(pdf_extract, "tesseract_version", lambda cmd=None: "5.0")
    doc_id = await upload(api, live_app, "scan.pdf", factory.scanned_pdf(pages=1))

    result = await search(api, "indemnity obligations", doc_id)
    quality = result["passages"][0]["quality"]
    assert quality == {
        "level": "low_quality_ocr",
        "ocr_pages": [1],
        "low_quality_pages": [1],
        "min_ocr_confidence": 0.31,
    }
    assert any("low-quality OCR" in n for n in result["notes"])
    assert result["coverage"][0]["low_quality_page_count"] == 1


async def test_docx_passages_have_no_pages(api: httpx.AsyncClient, live_app: FastAPI) -> None:
    doc_id = await upload(api, live_app, "nda.docx", factory.numbered_docx())
    passage = (await search(api, "liability", doc_id))["passages"][0]
    assert passage["page_start"] is None and passage["quality"]["level"] == "not_applicable"
    assert passage["section_path"].startswith("3 LIABILITY")


# --- real embedding model ----------------------------------------------------------------------


async def test_real_model_paraphrase_retrieval(db_settings: Settings) -> None:
    """End to end with bge-small-en-v1.5 (cached locally). Fails, not skips, if it cannot load:
    the model is a hard dependency of the product."""
    from app.embeddings import FastEmbedEmbedder

    embedder = FastEmbedEmbedder(db_settings.embedding_model, cache_dir=".cache/fastembed")
    app = create_app(db_settings, embedder=embedder)
    async with app.router.lifespan_context(app):
        await truncate_all()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as api:
            doc_id = await upload(api, app, "contract.pdf", contract_pdf(CONTRACT_A))
            ended = await search(api, "how can the contract be ended early?", doc_id)
            law = await search(api, "which country's law applies", doc_id)
            cake = await search(api, "recipe for chocolate cake", doc_id)
    assert "terminate" in ended["passages"][0]["text"]
    assert "governed by the laws" in law["passages"][0]["text"]
    assert cake["no_match"] is True


# --- Phase 4 review regressions -----------------------------------------------------------------

LONG_BODY = " ".join(
    f"Sentence {i} about notices and the obligations of the parties under this part."
    for i in range(60)
)


async def test_phrase_in_section_heading_only_is_not_an_exact_match(
    api: httpx.AsyncClient, live_app: FastAPI
) -> None:
    """Regression: chunks of a long section matched a quoted phrase that appeared only in
    their section path, not in their text."""
    doc = await upload(
        api,
        live_app,
        "long.pdf",
        contract_pdf(["9. Indemnification\n\n" + LONG_BODY[:2500], LONG_BODY[2500:]]),
    )
    result = await search(api, '"Indemnification"', doc, top_k=20)
    phrase_hits = [p for p in result["passages"] if p["signals"]["phrase_rank"] is not None]
    assert phrase_hits, "the chunk containing the heading must still match"
    for passage in phrase_hits:
        assert "Indemnification" in passage["text"]


async def test_quantity_in_query_does_not_pull_in_a_same_numbered_clause(
    api: httpx.AsyncClient, live_app: FastAPI
) -> None:
    doc = await upload(
        api,
        live_app,
        "numbered.pdf",
        contract_pdf(
            [
                "2. Fees\n\n2.5 Late invoices accrue interest monthly.",
                "7. Insurance\n\nThe Supplier carries cover of USD 2.5 million per claim.",
            ]
        ),
    )
    quantity = await search(api, "insurance cover of 2.5 million", doc)
    assert all(p["signals"]["section_rank"] is None for p in quantity["passages"])
    assert "2.5 million" in collapse(quantity["passages"][0]["text"])
    explicit = await search(api, "clause 2.5", doc)
    assert explicit["passages"][0]["signals"]["section_rank"] == 1
    assert "2.5 Late invoices" in collapse(explicit["passages"][0]["text"])


async def test_sql_statements_per_search_are_bounded(
    api: httpx.AsyncClient, live_app: FastAPI, contract_a: str
) -> None:
    """No N+1: the statement count depends on the number of documents and phrases, never on the
    number of chunks or passages."""
    from sqlalchemy import event

    from app.db.session import get_engine

    contract_b = await upload(api, live_app, "contract-b.pdf", contract_pdf(CONTRACT_B))
    statements: list[str] = []

    def count(conn, cursor, statement, parameters, context, executemany):  # type: ignore[no-untyped-def]
        statements.append(statement)

    sync_engine = get_engine().sync_engine
    event.listen(sync_engine, "before_cursor_execute", count)
    try:
        query = 'liability "order form" "governing law" termination clause 8.2'
        await search(api, query, contract_a, contract_b, top_k=20)
    finally:
        event.remove(sync_engine, "before_cursor_execute", count)
    # scope + coverage + dense + bm25 + section + 2 phrases + chunk fetch + 2 page lookups (one per
    # document) + one headings-only check for all passages = 11, plus transaction set-up
    # statements issued by the driver/SQLAlchemy.
    selects = [s for s in statements if s.lstrip().upper().startswith(("SELECT", "WITH"))]
    assert len(selects) <= 11, selects


async def test_internal_errors_do_not_leak_details(
    api: httpx.AsyncClient, live_app: FastAPI, contract_a: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(text: str):  # type: ignore[no-untyped-def]
        raise RuntimeError("onnxruntime secret path C:/internal/model.onnx")

    monkeypatch.setattr(live_app.state.embedder, "embed_query", broken)
    response = await api.post(SEARCH, json={"query": "liability", "document_ids": [contract_a]})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL_ERROR"
    assert "onnx" not in response.text and "C:/" not in response.text


async def test_no_match_on_low_quality_pages_is_not_presented_as_absence(
    api: httpx.AsyncClient, live_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a no-match result on a document with low-quality OCR pages gave no hint that
    the miss may be an OCR failure rather than the content being absent."""

    def low_confidence_ocr(page, *, dpi, language, page_index):  # type: ignore[no-untyped-def]
        words = [
            SourceWord(w, 72 + 70 * i, 100, 130 + 70 * i, 112) for i, w in enumerate(["xq", "zz"])
        ]
        return OcrResult([SourceBlock([SourceLine(words)], page_index=page_index)], 0.25)

    monkeypatch.setattr(pdf_extract, "ocr_page", low_confidence_ocr)
    monkeypatch.setattr(pdf_extract, "tesseract_version", lambda cmd=None: "5.0")
    doc_id = await upload(api, live_app, "scan.pdf", factory.scanned_pdf(pages=2))

    result = await search(api, "network routing protocols", doc_id)
    assert result["no_match"] is True
    assert any("2 page(s) were read with low OCR quality" in n for n in result["notes"])


# --- P1 Step 1a: passages without substantive content ------------------------------------------


async def test_headings_only_passages_are_flagged_from_stored_block_kinds(
    api: httpx.AsyncClient, live_app: FastAPI
) -> None:
    from tests.fixtures.journal import journal_docx

    journal = await upload(api, live_app, "journal.docx", journal_docx())
    contract = await upload(api, live_app, "contract.pdf", contract_pdf(CONTRACT_A))
    found = await search(api, "about yourself learning experiences", journal, top_k=50)
    flagged = {p["text"].strip() for p in found["passages"] if p["headings_only"]}
    assert flagged == {"About Yourself", "My Learning Experiences"}
    # Content passages, including ones that start with a heading, are not flagged.
    assert any("Strengths:" in p["text"] for p in found["passages"] if not p["headings_only"])
    pdf = await search(api, "terminate this agreement", contract)
    assert pdf["passages"] and not any(p["headings_only"] for p in pdf["passages"])


async def test_numbered_clause_with_inline_title_is_content_not_a_heading(
    api: httpx.AsyncClient, live_app: FastAPI, contract_a: str
) -> None:
    # Regression (found while implementing Step 1a): "15. Governing Law. This Agreement is
    # governed by ..." is stored as a heading block, yet carries the clause's body. Block kind
    # alone wrongly made it "headings only" and removed it from answer evidence.
    found = await search(api, "governing law", contract_a, top_k=20)
    clause = [p for p in found["passages"] if "governed by the laws" in p["text"]]
    assert clause and not any(p["headings_only"] for p in clause)
