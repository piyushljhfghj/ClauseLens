"""Document ingestion pipeline: the work an `ingest` job performs.

Phase 2 stages: download the stored file -> inspect it. Phase 3 appends extraction, OCR,
structure, chunking and embedding stages before the document is marked ready.

Progress is reported through `ProgressReporter`, which may be called from a worker thread; the
job worker periodically copies it to the database so the UI can show live status.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db.models import Document
from app.ingestion.inspect import ProcessingFailure, ProgressCallback, inspect
from app.storage import ObjectNotFoundError, Storage

logger = logging.getLogger(__name__)


@dataclass
class ProgressReporter:
    """Latest stage/progress of the running job. Plain attribute writes are atomic in CPython, so
    a worker thread can update it while the event loop reads it."""

    stage: str = "starting"
    progress: int = 0

    def set(self, stage: str, progress: float) -> None:
        self.stage = stage
        self.progress = max(0, min(100, round(progress)))

    def span(self, stage: str, start: float, end: float) -> ProgressCallback:
        """A callback mapping a step's own 0.0-1.0 progress onto [start, end] overall."""
        self.set(stage, start)
        return lambda fraction: self.set(stage, start + (end - start) * fraction)


async def ingest_document(
    document_id: uuid.UUID,
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    storage: Storage,
    settings: Settings,
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
        document.progress = 0
        document.error_code = document.error_message = None
        await session.commit()

    progress.set("downloading", 2)
    try:
        data = await storage.get(storage_key)
    except ObjectNotFoundError as exc:
        raise ProcessingFailure(
            "FILE_MISSING", "The uploaded file could not be found in storage. Upload it again."
        ) from exc

    result = await asyncio.to_thread(
        inspect,
        file_type,
        data,
        max_pages=settings.max_pages,
        on_progress=progress.span("inspecting", 5, 95),
    )

    async with sessionmaker() as session:
        await session.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(
                status="ready",
                stage=None,
                progress=100,
                page_count=result.page_count,
                pages_without_text=result.pages_without_text,
                processed_at=func.now(),
            )
        )
        await session.commit()
    logger.info(
        "document processed",
        extra={"document_id": str(document_id), "pages": result.page_count},
    )
