"""Document library operations: upload, list, fetch, delete, reprocess."""

import hashlib
import logging
import uuid
from collections.abc import Callable
from pathlib import PurePath

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.errors import AppError, ConflictError, NotFoundError, ServiceUnavailableError
from app.db.models import Document
from app.ingestion.detect import detect_file_type, sanitize_filename
from app.jobs import queue
from app.storage import Storage, StorageError, document_key

logger = logging.getLogger(__name__)

STORAGE_UNAVAILABLE = "File storage is temporarily unavailable. Please try again."


class DocumentService:
    def __init__(
        self,
        session: AsyncSession,
        storage: Storage,
        settings: Settings,
        on_job_enqueued: Callable[[], None] = lambda: None,
    ) -> None:
        self._session = session
        self._storage = storage
        self._settings = settings
        self._on_job_enqueued = on_job_enqueued

    async def upload(self, raw_filename: str | None, data: bytes) -> Document:
        if len(data) > self._settings.max_upload_bytes:
            raise AppError(
                f"The file is larger than the {self._settings.max_upload_mb} MB limit.",
                code="FILE_TOO_LARGE",
                status_code=413,
            )
        detected = detect_file_type(raw_filename or "", data)
        filename = sanitize_filename(raw_filename, detected.file_type)
        if PurePath(filename).suffix.lower() != f".{detected.file_type}":
            filename = f"{filename}.{detected.file_type}"
        sha256 = hashlib.sha256(data).hexdigest()

        # Re-uploading a failed document is allowed (it may have been fixed); anything else
        # identical is already in the library.
        existing = (
            (
                await self._session.execute(
                    select(Document).where(Document.sha256 == sha256, Document.status != "failed")
                )
            )
            .scalars()
            .first()
        )
        if existing is not None:
            raise ConflictError(
                f"This file is already in your library as '{existing.filename}'.",
                code="DUPLICATE_DOCUMENT",
                details={"document_id": str(existing.id)},
            )

        document_id = uuid.uuid4()
        document = Document(
            id=document_id,
            filename=filename,
            file_type=detected.file_type,
            mime_type=detected.mime_type,
            size_bytes=len(data),
            sha256=sha256,
            storage_key=document_key(str(document_id), detected.file_type),
            status="queued",
            stage="waiting",
            progress=0,
        )
        self._session.add(document)
        await queue.enqueue(
            self._session,
            kind="ingest",
            document_id=document_id,
            max_attempts=self._settings.job_max_attempts,
        )
        await self._session.flush()

        # Store the file before committing: if storage fails, the transaction rolls back and no
        # document row points at a missing file.
        try:
            await self._storage.put(document.storage_key, data, detected.mime_type)
        except StorageError as exc:
            logger.error("storage upload failed", extra={"error": str(exc)})
            await self._session.rollback()
            raise ServiceUnavailableError(STORAGE_UNAVAILABLE, code="STORAGE_UNAVAILABLE") from exc
        try:
            await self._session.commit()
        except Exception:
            await self._delete_file_quietly(document.storage_key)
            raise

        await self._session.refresh(document)
        logger.info(
            "document uploaded",
            extra={"document_id": str(document_id), "type": detected.file_type, "bytes": len(data)},
        )
        self._on_job_enqueued()
        return document

    async def list(self, *, limit: int, offset: int) -> tuple[list[Document], int]:
        total = (await self._session.execute(select(func.count(Document.id)))).scalar_one()
        rows = await self._session.execute(
            select(Document).order_by(Document.created_at.desc()).limit(limit).offset(offset)
        )
        return list(rows.scalars()), total

    async def get(self, document_id: uuid.UUID) -> Document:
        document = await self._session.get(Document, document_id)
        if document is None:
            raise NotFoundError("Document not found.", code="DOCUMENT_NOT_FOUND")
        return document

    async def get_file(self, document_id: uuid.UUID) -> tuple[Document, bytes]:
        document = await self.get(document_id)
        try:
            return document, await self._storage.get(document.storage_key)
        except StorageError as exc:
            raise ServiceUnavailableError(
                "The original file could not be retrieved from storage.",
                code="FILE_UNAVAILABLE",
            ) from exc

    async def delete(self, document_id: uuid.UUID) -> None:
        document = await self.get(document_id)
        key = document.storage_key
        await self._session.delete(document)  # jobs cascade in the database
        await self._session.commit()
        # The row is gone, so the document has disappeared for the user. A leftover file is
        # only wasted space; log it rather than failing the request.
        await self._delete_file_quietly(key)
        logger.info("document deleted", extra={"document_id": str(document_id)})

    async def reprocess(self, document_id: uuid.UUID) -> Document:
        document = await self.get(document_id)
        if document.status in ("queued", "processing"):
            raise ConflictError("This document is already being processed.", code="DOCUMENT_BUSY")
        document.status = "queued"
        document.stage = "waiting"
        document.progress = 0
        document.error_code = document.error_message = None
        await queue.enqueue(
            self._session,
            kind="ingest",
            document_id=document.id,
            max_attempts=self._settings.job_max_attempts,
        )
        await self._session.commit()
        await self._session.refresh(document)
        self._on_job_enqueued()
        return document

    async def _delete_file_quietly(self, key: str) -> None:
        try:
            await self._storage.delete(key)
        except StorageError as exc:
            logger.warning("orphaned file left in storage", extra={"key": key, "error": str(exc)})
