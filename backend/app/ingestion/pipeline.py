"""Document ingestion pipeline: the work an `ingest` job performs.

  download -> inspect (validate) -> extract (text layer + OCR) -> structure -> chunk -> embed
  -> save (atomic swap of the whole index, document becomes `ready`)

Progress is reported through `ProgressReporter`, which may be called from worker threads; the
job worker periodically copies it to the database so the UI shows live status.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db.models import Document
from app.embeddings import Embedder
from app.ingestion.chunking import build_chunks
from app.ingestion.docx_extract import extract_docx
from app.ingestion.index_writer import replace_index
from app.ingestion.inspect import ProcessingFailure, ProgressCallback, inspect
from app.ingestion.model import ExtractedDocument
from app.ingestion.pdf_extract import OcrSettings, extract_pdf
from app.ingestion.structure import detect_sections
from app.storage import ObjectNotFoundError, Storage

logger = logging.getLogger(__name__)

# Bump whenever extraction/chunking/embedding changes in a way that makes old indexes stale.
# Documents indexed by an older version are reported as needing reprocessing.
# v2 (Phase 3 audit): DOCX bold headings, title-block rule, OCR of stamped scans and reporting
# of image pages with no recognisable text. Documents indexed by v1 must be reprocessed.
# v3: low-quality OCR pages (confidence < 50%) are flagged and counted separately.
INDEX_VERSION = 3


@dataclass
class ProgressReporter:
    """Latest stage/progress of the running job. Plain attribute writes are atomic in CPython, so
    a worker thread can update it while the event loop reads it."""

    stage: str = "starting"
    progress: int = 0
    detail: str | None = None

    def set(self, stage: str, progress: float, detail: str | None = None) -> None:
        self.stage = stage
        self.progress = max(0, min(100, round(progress)))
        self.detail = detail

    def span(self, stage: str, start: float, end: float) -> ProgressCallback:
        """A callback mapping a step's own 0.0-1.0 progress onto [start, end] overall."""
        self.set(stage, start)
        return lambda fraction: self.set(stage, start + (end - start) * fraction)


def _extract(
    file_type: str, data: bytes, settings: Settings, progress: ProgressReporter
) -> ExtractedDocument:
    """Runs in a worker thread. Overall progress 5-50%."""
    if file_type == "docx":
        return extract_docx(data, on_progress=progress.span("extracting", 5, 50))

    def on_pdf_progress(fraction: float, detail: str, is_ocr: bool) -> None:
        progress.set("ocr" if is_ocr else "extracting", 5 + 45 * fraction, detail)

    return extract_pdf(
        data,
        ocr=OcrSettings(
            enabled=settings.ocr_enabled,
            language=settings.ocr_language,
            dpi=settings.ocr_dpi,
            tesseract_cmd=settings.tesseract_cmd,
        ),
        on_progress=on_pdf_progress,
    )


def _no_text_failure(file_type: str, extracted: ExtractedDocument) -> ProcessingFailure:
    if file_type == "docx":
        return ProcessingFailure("EMPTY_DOCUMENT", "This Word document contains no text.")
    if extracted.unreadable_pages:
        return ProcessingFailure(
            "SCANNED_NO_TEXT",
            "This PDF has no selectable text (it appears to be scanned) and text recognition "
            "(OCR) could not be run, so its contents cannot be read.",
        )
    return ProcessingFailure(
        "SCANNED_NO_TEXT",
        "No readable text was found in this PDF, even after text recognition (OCR).",
    )


async def ingest_document(
    document_id: uuid.UUID,
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    storage: Storage,
    settings: Settings,
    embedder: Embedder,
    progress: ProgressReporter,
) -> None:
    async with sessionmaker() as session:
        document = (
            await session.execute(select(Document).where(Document.id == document_id))
        ).scalar_one_or_none()
        if document is None:
            logger.info(
                "document deleted before processing", extra={"document_id": str(document_id)}
            )
            return
        file_type, storage_key = document.file_type, document.storage_key
        document.status = "processing"
        document.stage = "downloading"
        document.stage_detail = None
        document.progress = 0
        document.error_code = document.error_message = None
        await session.commit()

    progress.set("downloading", 1)
    try:
        data = await storage.get(storage_key)
    except ObjectNotFoundError as exc:
        raise ProcessingFailure(
            "FILE_MISSING", "The uploaded file could not be found in storage. Upload it again."
        ) from exc

    # Validation: password-protected, corrupt, empty, too many pages.
    inspection = await asyncio.to_thread(
        inspect,
        file_type,
        data,
        max_pages=settings.max_pages,
        on_progress=progress.span("inspecting", 2, 5),
    )

    extracted = await asyncio.to_thread(_extract, file_type, data, settings, progress)
    if not extracted.canonical_text.strip():
        raise _no_text_failure(file_type, extracted)

    progress.set("structuring", 50, "Detecting sections and splitting into passages")
    sections = detect_sections(extracted.blocks, extracted.canonical_text)
    chunks = build_chunks(extracted, sections)

    def on_embed_progress(done: int, total: int) -> None:
        progress.set(
            "embedding", 55 + 40 * done / max(total, 1), f"Embedded {done} of {total} passages"
        )

    progress.set("embedding", 55, f"Embedding {len(chunks)} passages")
    embeddings = await asyncio.to_thread(
        embedder.embed_passages, [c.context_text for c in chunks], on_embed_progress
    )

    progress.set("saving", 96, "Saving index")
    async with sessionmaker() as session:
        written = await replace_index(
            session,
            document_id,
            extracted=extracted,
            sections=sections,
            chunks=chunks,
            embeddings=embeddings,
            embedding_model=embedder.model_name,
            index_version=INDEX_VERSION,
            page_count=inspection.page_count,
        )
        await session.commit()
    if not written:
        logger.info("document deleted during processing", extra={"document_id": str(document_id)})
        return
    logger.info(
        "document indexed",
        extra={
            "document_id": str(document_id),
            "pages": inspection.page_count,
            "chars": len(extracted.canonical_text),
            "sections": len(sections),
            "chunks": len(chunks),
            "ocr_pages": extracted.ocr_page_count,
            "unreadable_pages": len(extracted.unreadable_pages),
            "low_quality_pages": len(extracted.low_quality_pages),
        },
    )
