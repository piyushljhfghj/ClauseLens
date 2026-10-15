"""GET /api/documents/{id}/layout (Phase 7, task 1): stored blocks and their text."""

import uuid

import httpx
from fastapi import FastAPI
from sqlalchemy import update

from app.db.models import Document, DocumentBlock, DocumentText
from app.db.session import get_sessionmaker
from tests.fixtures import factory
from tests.test_indexing import INDEX_MODELS, upload_and_process


async def stored(document_id: str) -> tuple[str, list[DocumentBlock]]:
    async with get_sessionmaker()() as session:
        text = await session.get(DocumentText, uuid.UUID(document_id))
        assert text is not None
        rows = (
            await session.execute(
                DocumentBlock.__table__.select()
                .where(DocumentBlock.document_id == uuid.UUID(document_id))
                .order_by(DocumentBlock.ordinal)
            )
        ).all()
        return text.canonical_text, rows  # type: ignore[return-value]


async def test_docx_layout_returns_every_block_in_order_with_its_text(
    api: httpx.AsyncClient, live_app: FastAPI
) -> None:
    document = await upload_and_process(api, live_app, "msa.docx", factory.numbered_docx())
    response = await api.get(f"/api/documents/{document['id']}/layout")
    assert response.status_code == 200
    layout = response.json()
    assert layout["document_id"] == document["id"]
    assert layout["file_type"] == "docx"
    assert layout["processed_at"] == document["processed_at"]

    canonical, rows = await stored(document["id"])
    blocks = layout["blocks"]
    assert [b["ordinal"] for b in blocks] == list(range(len(rows)))
    for block, row in zip(blocks, rows, strict=True):
        assert (block["kind"], block["char_start"], block["char_end"]) == (
            row.kind,
            row.char_start,
            row.char_end,
        )
        assert block["text"] == canonical[row.char_start : row.char_end]
    kinds = {b["kind"] for b in blocks}
    assert {"heading", "paragraph", "table_row"} <= kinds
    assert "2.1 Either party may terminate" in "".join(b["text"] for b in blocks)
    # Read-only and compact: no word boxes or other index internals.
    assert set(blocks[0]) == {"ordinal", "kind", "char_start", "char_end", "text"}


async def test_layout_text_offsets_count_code_points(
    api: httpx.AsyncClient, live_app: FastAPI
) -> None:
    emoji = chr(0x1F600)  # outside the Basic Multilingual Plane: 1 code point, 2 UTF-16 units
    paragraphs = [f"Smile {emoji} clause one applies.", f"Second {emoji}{emoji} clause applies."]
    document = await upload_and_process(api, live_app, "emoji.docx", factory.docx_file(paragraphs))
    blocks = (await api.get(f"/api/documents/{document['id']}/layout")).json()["blocks"]
    assert [b["text"] for b in blocks] == paragraphs
    for block, text in zip(blocks, paragraphs, strict=True):
        assert block["char_end"] - block["char_start"] == len(text)  # Python len: code points
    assert blocks[1]["char_start"] == len(paragraphs[0]) + 2  # after the blank-line separator


async def test_manual_line_break_is_part_of_the_block_text(
    api: httpx.AsyncClient, live_app: FastAPI
) -> None:
    paragraphs = ["Notices must be in writing.\nEmail counts as writing."]
    document = await upload_and_process(api, live_app, "br.docx", factory.docx_file(paragraphs))
    (block,) = (await api.get(f"/api/documents/{document['id']}/layout")).json()["blocks"]
    assert block["text"] == "Notices must be in writing.\nEmail counts as writing."


async def test_pdf_layout_also_returns_its_blocks(
    api: httpx.AsyncClient, live_app: FastAPI
) -> None:
    document = await upload_and_process(api, live_app, "a.pdf", factory.text_pdf(pages=2))
    layout = (await api.get(f"/api/documents/{document['id']}/layout")).json()
    canonical, rows = await stored(document["id"])
    assert layout["file_type"] == "pdf" and len(layout["blocks"]) == len(rows) > 0
    assert all(b["text"] == canonical[b["char_start"] : b["char_end"]] for b in layout["blocks"])


async def test_layout_of_a_document_without_an_index_is_409(
    api: httpx.AsyncClient, live_app: FastAPI
) -> None:
    document = await upload_and_process(api, live_app, "old.docx", factory.docx_file())
    async with get_sessionmaker()() as session:
        for model in INDEX_MODELS:
            await session.execute(
                model.__table__.delete().where(model.document_id == document["id"])
            )
        await session.execute(
            update(Document).where(Document.id == document["id"]).values(index_version=None)
        )
        await session.commit()
    response = await api.get(f"/api/documents/{document['id']}/layout")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "DOCUMENT_NOT_INDEXED"


async def test_layout_of_an_unknown_document_is_404(api: httpx.AsyncClient) -> None:
    response = await api.get(f"/api/documents/{uuid.uuid4()}/layout")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"
    assert (await api.get("/api/documents/not-a-uuid/layout")).status_code == 422


async def test_layout_is_read_only(api: httpx.AsyncClient, live_app: FastAPI) -> None:
    document = await upload_and_process(api, live_app, "ro.docx", factory.docx_file())
    before = (await api.get(f"/api/documents/{document['id']}")).json()
    for method in ("post", "put", "patch", "delete"):
        response = await getattr(api, method)(f"/api/documents/{document['id']}/layout")
        assert response.status_code == 405
    after = (await api.get(f"/api/documents/{document['id']}")).json()
    assert after["updated_at"] == before["updated_at"]
