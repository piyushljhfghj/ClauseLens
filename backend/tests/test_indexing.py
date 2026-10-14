"""End-to-end indexing against real Postgres: upload -> worker -> stored index -> API."""

import re

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from sqlalchemy import func, select, text, update

from app.core.config import Settings
from app.db.models import (
    Document,
    DocumentBlock,
    DocumentChunk,
    DocumentPage,
    DocumentSection,
    DocumentText,
)
from app.db.session import get_sessionmaker
from app.ingestion.ocr import tesseract_version
from app.main import create_app
from tests.conftest import truncate_all
from tests.fakes import FakeEmbedder
from tests.fixtures import factory

needs_tesseract = pytest.mark.skipif(tesseract_version() is None, reason="Tesseract not installed")
INDEX_MODELS = (DocumentText, DocumentPage, DocumentSection, DocumentBlock, DocumentChunk)


async def upload_and_process(api: httpx.AsyncClient, app: FastAPI, name: str, data: bytes) -> dict:
    response = await api.post("/api/documents", files={"file": (name, data)})
    assert response.status_code == 202, response.text
    while await app.state.worker.run_once():
        pass
    return (await api.get(f"/api/documents/{response.json()['id']}")).json()


async def count_rows(model: type, document_id: str | None = None) -> int:
    async with get_sessionmaker()() as session:
        query = select(func.count()).select_from(model)
        if document_id:
            query = query.where(model.document_id == document_id)
        return (await session.execute(query)).scalar_one()


async def test_text_pdf_is_fully_indexed(api: httpx.AsyncClient, live_app: FastAPI) -> None:
    document = await upload_and_process(api, live_app, "msa.pdf", factory.text_pdf(pages=3))
    assert document["status"] == "ready"
    assert document["indexed"] is True and document["needs_reprocessing"] is False
    assert document["page_count"] == 3 and document["ocr_page_count"] == 0
    assert document["unreadable_page_count"] == 0 and document["extraction_warnings"] == []
    assert document["chunk_count"] > 0 and document["section_count"] == 3
    doc_id = document["id"]

    async with get_sessionmaker()() as session:
        canonical = (await session.get(DocumentText, doc_id)).canonical_text  # type: ignore[union-attr]
        chunks = (
            (
                await session.execute(
                    select(DocumentChunk)
                    .where(DocumentChunk.document_id == doc_id)
                    .order_by(DocumentChunk.ordinal)
                )
            )
            .scalars()
            .all()
        )
        pages = (
            (await session.execute(select(DocumentPage).where(DocumentPage.document_id == doc_id)))
            .scalars()
            .all()
        )
    assert len(canonical) == document["char_count"]
    assert len(chunks) == document["chunk_count"] and len(pages) == 3
    for chunk in chunks:
        assert chunk.text == canonical[chunk.char_start : chunk.char_end]
        vector = np.asarray(chunk.embedding)
        assert vector.shape == (384,) and abs(np.linalg.norm(vector) - 1) < 1e-4
        assert chunk.lexical_tokens  # stored for BM25 (Phase 4)
    for page in pages:
        for start, end, *_ in page.words:
            assert canonical[int(start) : int(end)].strip()

    outline = (await api.get(f"/api/documents/{doc_id}/outline")).json()
    assert [s["number"] for s in outline["sections"]] == ["1", "2", "3"]
    page_info = (await api.get(f"/api/documents/{doc_id}/pages")).json()
    assert [p["extraction"] for p in page_info["pages"]] == ["text", "text", "text"]
    assert all(p["char_count"] > 0 for p in page_info["pages"])


async def test_docx_is_indexed_with_numbered_sections(
    api: httpx.AsyncClient, live_app: FastAPI
) -> None:
    document = await upload_and_process(api, live_app, "nda.docx", factory.numbered_docx())
    assert document["indexed"] is True
    assert document["page_count"] is None and document["ocr_page_count"] is None
    outline = (await api.get(f"/api/documents/{document['id']}/outline")).json()
    assert [s["number"] for s in outline["sections"]] == [
        None,
        "1",
        "1.1",
        "2",
        "2.1",
        "2.2",
        "3",
        "3.1",
    ]
    assert (await api.get(f"/api/documents/{document['id']}/pages")).json()["pages"] == []


@needs_tesseract
async def test_scanned_pdf_is_indexed_via_ocr(api: httpx.AsyncClient, live_app: FastAPI) -> None:
    document = await upload_and_process(api, live_app, "scan.pdf", factory.scanned_pdf(pages=2))
    assert document["status"] == "ready" and document["indexed"] is True
    assert document["ocr_page_count"] == 2 and document["unreadable_page_count"] == 0
    pages = (await api.get(f"/api/documents/{document['id']}/pages")).json()["pages"]
    assert [p["extraction"] for p in pages] == ["ocr", "ocr"]
    assert all(p["ocr_confidence"] > 0.8 for p in pages)


@needs_tesseract
async def test_mixed_pdf_reports_per_page_method(api: httpx.AsyncClient, live_app: FastAPI) -> None:
    document = await upload_and_process(api, live_app, "mixed.pdf", factory.mixed_pdf())
    assert (document["ocr_page_count"], document["unreadable_page_count"]) == (1, 0)
    pages = (await api.get(f"/api/documents/{document['id']}/pages")).json()["pages"]
    assert [p["extraction"] for p in pages] == ["text", "ocr"]


async def test_partially_unreadable_pdf_is_ready_but_says_so(
    db_settings: Settings, fake_embedder: FakeEmbedder
) -> None:
    """OCR off + mixed PDF: indexed from the readable page, with the gap reported, not hidden."""
    app = create_app(db_settings.model_copy(update={"ocr_enabled": False}), embedder=fake_embedder)
    async with app.router.lifespan_context(app):
        await truncate_all()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as api:
            document = await upload_and_process(api, app, "mixed.pdf", factory.mixed_pdf())
            pages = (await api.get(f"/api/documents/{document['id']}/pages")).json()["pages"]
    assert document["status"] == "ready" and document["indexed"] is True
    assert document["unreadable_page_count"] == 1
    assert any("could not be read" in w for w in document["extraction_warnings"])
    assert [(p["extraction"], p["note"]) for p in pages] == [
        ("text", None),
        ("none", "ocr_unavailable"),
    ]


async def test_large_document_page_mapping(api: httpx.AsyncClient, live_app: FastAPI) -> None:
    document = await upload_and_process(api, live_app, "big.pdf", factory.large_pdf_with_needle())
    assert document["page_count"] == 150 and document["indexed"] is True
    async with get_sessionmaker()() as session:
        chunks = (
            (
                await session.execute(
                    select(DocumentChunk).where(DocumentChunk.document_id == document["id"])
                )
            )
            .scalars()
            .all()
        )
        page_137 = await session.get(DocumentPage, (document["id"], 137))
    needle = [c for c in chunks if factory.NEEDLE in re.sub(r"\s+", " ", c.text)]
    assert len(needle) == 1 and (needle[0].page_start, needle[0].page_end) == (137, 137)
    assert page_137 is not None
    assert page_137.char_start <= needle[0].char_start < page_137.char_end
    assert min(c.page_start for c in chunks) == 1 and max(c.page_end for c in chunks) == 150


async def test_phase2_document_needs_reprocessing_then_is_indexed(
    api: httpx.AsyncClient, live_app: FastAPI
) -> None:
    document = await upload_and_process(api, live_app, "old.pdf", factory.text_pdf(pages=2))
    doc_id = document["id"]
    # Simulate a document stored and inspected by Phase 2: ready, but no index.
    async with get_sessionmaker()() as session:
        for model in INDEX_MODELS:
            await session.execute(model.__table__.delete().where(model.document_id == doc_id))
        await session.execute(
            update(Document).where(Document.id == doc_id).values(index_version=None)
        )
        await session.commit()

    stale = (await api.get(f"/api/documents/{doc_id}")).json()
    assert (stale["status"], stale["indexed"], stale["needs_reprocessing"]) == (
        "ready",
        False,
        True,
    )
    not_indexed = await api.get(f"/api/documents/{doc_id}/outline")
    assert not_indexed.status_code == 409
    assert not_indexed.json()["error"]["code"] == "DOCUMENT_NOT_INDEXED"

    assert (await api.post(f"/api/documents/{doc_id}/reprocess")).status_code == 202
    while await live_app.state.worker.run_once():
        pass
    fresh = (await api.get(f"/api/documents/{doc_id}")).json()
    assert (fresh["indexed"], fresh["needs_reprocessing"]) == (True, False)
    assert await count_rows(DocumentChunk, doc_id) == fresh["chunk_count"]


async def test_reprocessing_replaces_the_index_without_duplicates(
    api: httpx.AsyncClient, live_app: FastAPI
) -> None:
    document = await upload_and_process(api, live_app, "a.pdf", factory.text_pdf(pages=4))
    doc_id = document["id"]
    async with get_sessionmaker()() as session:
        before = set(
            (
                await session.execute(
                    select(DocumentChunk.id).where(DocumentChunk.document_id == doc_id)
                )
            ).scalars()
        )

    await api.post(f"/api/documents/{doc_id}/reprocess")
    while await live_app.state.worker.run_once():
        pass
    async with get_sessionmaker()() as session:
        after = set(
            (
                await session.execute(
                    select(DocumentChunk.id).where(DocumentChunk.document_id == doc_id)
                )
            ).scalars()
        )
    assert len(after) == len(before) == document["chunk_count"]
    assert before.isdisjoint(after)  # old rows gone, new rows in place
    for model in (DocumentPage, DocumentSection, DocumentBlock):
        assert await count_rows(model, doc_id) > 0
    assert await count_rows(DocumentPage, doc_id) == 4


async def test_failed_reindex_keeps_previous_index_intact(
    api: httpx.AsyncClient, live_app: FastAPI, fake_embedder: FakeEmbedder
) -> None:
    document = await upload_and_process(api, live_app, "a.pdf", factory.text_pdf(pages=2))
    doc_id, chunk_count = document["id"], document["chunk_count"]

    calls_before = fake_embedder.calls
    fake_embedder.fail_with = RuntimeError("embedding service crashed")
    await api.post(f"/api/documents/{doc_id}/reprocess")
    for _ in range(3):  # job_max_attempts = 3; skip the backoff delays
        assert await live_app.state.worker.run_once()
        async with get_sessionmaker()() as session:
            await session.execute(text("UPDATE jobs SET run_after = now()"))
            await session.commit()

    failed = (await api.get(f"/api/documents/{doc_id}")).json()
    assert (failed["status"], failed["error_code"], failed["indexed"]) == (
        "failed",
        "PROCESSING_ERROR",
        False,
    )
    # Nothing half-written: the previous complete index is still there, untouched.
    assert await count_rows(DocumentChunk, doc_id) == chunk_count
    assert fake_embedder.calls - calls_before == 3  # every attempt reached embedding

    fake_embedder.fail_with = None
    await api.post(f"/api/documents/{doc_id}/reprocess")
    while await live_app.state.worker.run_once():
        pass
    assert (await api.get(f"/api/documents/{doc_id}")).json()["indexed"] is True


async def test_delete_removes_the_whole_index(api: httpx.AsyncClient, live_app: FastAPI) -> None:
    document = await upload_and_process(api, live_app, "a.pdf", factory.text_pdf(pages=2))
    assert await count_rows(DocumentChunk) > 0
    assert (await api.delete(f"/api/documents/{document['id']}")).status_code == 204
    for model in INDEX_MODELS:
        assert await count_rows(model) == 0, model.__tablename__


async def test_index_endpoints_404_for_unknown_document(api: httpx.AsyncClient) -> None:
    import uuid

    for suffix in ("outline", "pages"):
        response = await api.get(f"/api/documents/{uuid.uuid4()}/{suffix}")
        assert response.status_code == 404


async def test_concurrent_reprocess_requests_queue_only_one_job(
    api: httpx.AsyncClient, live_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simultaneous 'Reprocess' clicks must not queue several jobs for the same document."""
    import asyncio

    from app.db.models import Job
    from app.jobs import queue

    document = await upload_and_process(api, live_app, "a.pdf", factory.text_pdf(pages=1))
    original_enqueue = queue.enqueue

    async def slow_enqueue(*args, **kwargs):  # type: ignore[no-untyped-def]
        await asyncio.sleep(0.2)  # widen the window between the status check and the commit
        return await original_enqueue(*args, **kwargs)

    monkeypatch.setattr(queue, "enqueue", slow_enqueue)
    responses = await asyncio.gather(
        *(api.post(f"/api/documents/{document['id']}/reprocess") for _ in range(5))
    )
    assert sorted(r.status_code for r in responses) == [202, 409, 409, 409, 409]
    async with get_sessionmaker()() as session:
        queued = (
            await session.execute(
                select(func.count())
                .select_from(Job)
                .where(Job.document_id == document["id"], Job.status == "queued")
            )
        ).scalar_one()
    assert queued == 1


async def test_outline_includes_previews_for_untitled_sections(
    api: httpx.AsyncClient, live_app: FastAPI
) -> None:
    document = await upload_and_process(api, live_app, "nda.docx", factory.numbered_docx())
    sections = (await api.get(f"/api/documents/{document['id']}/outline")).json()["sections"]
    untitled = [s for s in sections if s["title"] is None]
    assert untitled, "fixture has untitled numbered clauses"
    assert all(s["preview"] for s in untitled)  # never a bare number in the outline
    by_number = {s["number"]: s for s in sections}
    assert by_number["2.1"]["preview"].startswith("Either party may terminate")


@needs_tesseract
async def test_page_with_no_recognisable_text_is_reported_through_the_api(
    api: httpx.AsyncClient, live_app: FastAPI
) -> None:
    document = await upload_and_process(api, live_app, "diagram.pdf", factory.picture_only_pdf())
    assert document["status"] == "ready" and document["unreadable_page_count"] == 1
    assert any("No text could be recognised" in w for w in document["extraction_warnings"])
    pages = (await api.get(f"/api/documents/{document['id']}/pages")).json()["pages"]
    assert [(p["extraction"], p["note"]) for p in pages] == [
        ("text", None),
        ("none", "ocr_no_text"),
    ]
