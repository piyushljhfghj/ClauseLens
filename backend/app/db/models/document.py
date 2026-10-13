import uuid
from datetime import datetime
from typing import Literal, get_args

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Integer,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, sql_in

DocumentStatus = Literal["queued", "processing", "ready", "failed"]
FileType = Literal["pdf", "docx"]


class Document(Base):
    """An uploaded contract. Its lifecycle is driven by `status`; `stage`/`progress` describe
    what processing is doing right now, so the UI never shows a bare spinner."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename: Mapped[str] = mapped_column(String(255))
    file_type: Mapped[str] = mapped_column(String(8))
    mime_type: Mapped[str] = mapped_column(String(128))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    storage_key: Mapped[str] = mapped_column(String(512))

    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    stage: Mapped[str | None] = mapped_column(String(32))
    progress: Mapped[int] = mapped_column(SmallInteger, default=0)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)

    # Filled in by processing.
    page_count: Mapped[int | None] = mapped_column(Integer)
    # PDF pages with no extractable text layer (scanned). OCR candidates for Phase 3.
    pages_without_text: Mapped[int | None] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint(sql_in("status", get_args(DocumentStatus)), name="status"),
        CheckConstraint(sql_in("file_type", get_args(FileType)), name="file_type"),
        CheckConstraint("progress BETWEEN 0 AND 100", name="progress"),
    )
