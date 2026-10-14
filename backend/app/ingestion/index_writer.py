"""Persist a document's index atomically.

Old rows are deleted and new rows inserted in ONE transaction, together with the document's
status change to `ready`. Readers therefore see either the previous complete index or the new
complete index, never a mixture or a partial one.
"""

import uuid
from collections.abc import Sequence

import numpy as np
from sqlalchemy import delete, func, insert, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Document,
    DocumentBlock,
    DocumentChunk,
    DocumentPage,
    DocumentSection,
    DocumentText,
)
from app.ingestion.chunking import Chunk
from app.ingestion.model import ExtractedDocument
from app.ingestion.structure import Section
from app.text.tokenize import lexical_tokens


async def _allocate_ids(session: AsyncSession, table: str, count: int) -> list[int]:
    """Reserve `count` ids from the table's sequence, so parent/child rows can reference each
    other before insertion (sections -> parent sections, blocks/chunks -> sections)."""
    if count == 0:
        return []
    rows = await session.execute(
        text(
            f"SELECT nextval(pg_get_serial_sequence('{table}', 'id')) FROM generate_series(1, :n)"
        ),
        {"n": count},
    )
    return [row[0] for row in rows]


async def replace_index(
    session: AsyncSession,
    document_id: uuid.UUID,
    *,
    extracted: ExtractedDocument,
    sections: Sequence[Section],
    chunks: Sequence[Chunk],
    embeddings: np.ndarray,
    embedding_model: str,
    index_version: int,
    page_count: int | None,
) -> bool:
    """Returns False (and writes nothing) if the document was deleted meanwhile."""
    locked = await session.execute(
        select(Document.id).where(Document.id == document_id).with_for_update()
    )
    if locked.scalar_one_or_none() is None:
        return False

    for model in (DocumentChunk, DocumentBlock, DocumentSection, DocumentPage, DocumentText):
        await session.execute(delete(model).where(model.document_id == document_id))

    await session.execute(
        insert(DocumentText).values(
            document_id=document_id, canonical_text=extracted.canonical_text
        )
    )
    if extracted.pages:
        await session.execute(
            insert(DocumentPage),
            [
                {
                    "document_id": document_id,
                    "page_number": p.page_number,
                    "width": p.width,
                    "height": p.height,
                    "rotation": p.rotation,
                    "extraction": p.extraction,
                    "note": p.note,
                    "ocr_confidence": p.ocr_confidence,
                    "char_start": p.char_start,
                    "char_end": p.char_end,
                    "words": p.words,
                }
                for p in extracted.pages
            ],
        )

    section_ids = await _allocate_ids(session, "sections", len(sections))
    if sections:
        await session.execute(
            insert(DocumentSection),
            [
                {
                    "id": section_ids[s.index],
                    "document_id": document_id,
                    "parent_id": section_ids[s.parent_index]
                    if s.parent_index is not None
                    else None,
                    "ordinal": s.index,
                    "level": s.level,
                    "kind": s.kind,
                    "number": s.number,
                    "number_normalized": s.number_normalized,
                    "title": s.title,
                    "char_start": s.char_start,
                    "char_end": s.char_end,
                    "page_start": s.page_start,
                    "page_end": s.page_end,
                }
                for s in sections
            ],
        )

    def section_id(index: int | None) -> int | None:
        return section_ids[index] if index is not None else None

    if extracted.blocks:
        await session.execute(
            insert(DocumentBlock),
            [
                {
                    "document_id": document_id,
                    "ordinal": b.ordinal,
                    "kind": b.kind,
                    "section_id": section_id(b.section_index),
                    "char_start": b.char_start,
                    "char_end": b.char_end,
                    "page_start": b.page_start,
                    "page_end": b.page_end,
                }
                for b in extracted.blocks
            ],
        )
    if chunks:
        await session.execute(
            insert(DocumentChunk),
            [
                {
                    "document_id": document_id,
                    "ordinal": c.ordinal,
                    "section_id": section_id(c.section_index),
                    "section_path": c.section_path,
                    "char_start": c.char_start,
                    "char_end": c.char_end,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "block_start": c.block_start,
                    "block_end": c.block_end,
                    "text": c.text,
                    "context_text": c.context_text,
                    "lexical_tokens": lexical_tokens(c.context_text),
                    "embedding": embeddings[i],
                }
                for i, c in enumerate(chunks)
            ],
        )

    await session.execute(
        update(Document)
        .where(Document.id == document_id)
        .values(
            status="ready",
            stage=None,
            stage_detail=None,
            progress=100,
            error_code=None,
            error_message=None,
            page_count=page_count,
            pages_without_text=extracted.pages_without_text_layer if extracted.pages else None,
            char_count=len(extracted.canonical_text),
            section_count=len(sections),
            chunk_count=len(chunks),
            ocr_page_count=extracted.ocr_page_count if extracted.pages else None,
            unreadable_page_count=len(extracted.unreadable_pages) if extracted.pages else None,
            low_quality_page_count=len(extracted.low_quality_pages) if extracted.pages else None,
            extraction_warnings=extracted.warnings,
            index_version=index_version,
            embedding_model=embedding_model,
            processed_at=func.now(),
        )
    )
    return True
