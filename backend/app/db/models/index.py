"""The extracted, structured and embedded form of a document (Phase 3).

All rows hang off `documents` with ON DELETE CASCADE, and are replaced together in one
transaction on (re)processing, so a document never has a half-built index.

Location model: `document_texts.canonical_text` is the single source of truth. Pages, blocks,
sections and chunks are `[char_start, char_end)` ranges into it; `document_pages.words` maps
character ranges to boxes on the displayed page for highlighting.
"""

import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, sql_in
from app.embeddings import EMBEDDING_DIMENSION


def _document_fk() -> ForeignKey:
    return ForeignKey("documents.id", ondelete="CASCADE")


class DocumentText(Base):
    __tablename__ = "document_texts"

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), _document_fk(), primary_key=True
    )
    canonical_text: Mapped[str] = mapped_column(Text)


class DocumentPage(Base):
    __tablename__ = "document_pages"

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), _document_fk(), primary_key=True
    )
    page_number: Mapped[int] = mapped_column(Integer, primary_key=True)
    width: Mapped[float] = mapped_column(Float)
    height: Mapped[float] = mapped_column(Float)
    rotation: Mapped[int] = mapped_column(SmallInteger)
    extraction: Mapped[str] = mapped_column(String(8))
    note: Mapped[str | None] = mapped_column(String(32))
    ocr_confidence: Mapped[float | None] = mapped_column(Float)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    # [[char_start, char_end, x0, y0, x1, y1], ...] in displayed-page PDF points.
    words: Mapped[list[list[float]]] = mapped_column(JSONB)

    __table_args__ = (
        CheckConstraint(sql_in("extraction", ("text", "ocr", "none")), name="extraction"),
    )


class DocumentSection(Base):
    __tablename__ = "sections"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), _document_fk())
    parent_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("sections.id", ondelete="CASCADE")
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    level: Mapped[int] = mapped_column(SmallInteger)
    kind: Mapped[str] = mapped_column(String(16))
    number: Mapped[str | None] = mapped_column(String(64))
    number_normalized: Mapped[str | None] = mapped_column(String(64))
    title: Mapped[str | None] = mapped_column(Text)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (
        Index("ix_sections_document_ordinal", "document_id", "ordinal", unique=True),
        Index("ix_sections_document_number", "document_id", "number_normalized"),
    )


class DocumentBlock(Base):
    __tablename__ = "blocks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), _document_fk())
    ordinal: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(16))
    section_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("sections.id", ondelete="SET NULL")
    )
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (
        CheckConstraint(sql_in("kind", ("paragraph", "heading", "table_row")), name="kind"),
        Index("ix_blocks_document_ordinal", "document_id", "ordinal", unique=True),
    )


class DocumentChunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), _document_fk())
    ordinal: Mapped[int] = mapped_column(Integer)
    section_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("sections.id", ondelete="SET NULL")
    )
    section_path: Mapped[str] = mapped_column(Text)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    block_start: Mapped[int] = mapped_column(Integer)
    block_end: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    context_text: Mapped[str] = mapped_column(Text)
    lexical_tokens: Mapped[list[str]] = mapped_column(ARRAY(Text))
    # No ANN index: queries are scoped to a few documents, where an exact scan is fast and,
    # unlike a filtered HNSW search, never silently drops results.
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSION))

    __table_args__ = (Index("ix_chunks_document_ordinal", "document_id", "ordinal", unique=True),)
