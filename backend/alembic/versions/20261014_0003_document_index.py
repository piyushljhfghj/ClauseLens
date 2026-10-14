"""Document index: canonical text, pages with word boxes, sections, blocks, chunks (Phase 3).

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-14

Existing documents keep index_version NULL: they were stored and inspected by Phase 2 but never
indexed, and the API reports them as needing reprocessing.
"""

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "document_pages",
        sa.Column("document_id", sa.UUID(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("width", sa.Float(), nullable=False),
        sa.Column("height", sa.Float(), nullable=False),
        sa.Column("rotation", sa.SmallInteger(), nullable=False),
        sa.Column("extraction", sa.String(length=8), nullable=False),
        sa.Column("note", sa.String(length=32), nullable=True),
        sa.Column("ocr_confidence", sa.Float(), nullable=True),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("words", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.CheckConstraint(
            "extraction IN ('text', 'ocr', 'none')", name=op.f("ck_document_pages_extraction")
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_document_pages_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("document_id", "page_number", name=op.f("pk_document_pages")),
    )
    op.create_table(
        "document_texts",
        sa.Column("document_id", sa.UUID(), nullable=False),
        sa.Column("canonical_text", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_document_texts_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("document_id", name=op.f("pk_document_texts")),
    )
    op.create_table(
        "sections",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("document_id", sa.UUID(), nullable=False),
        sa.Column("parent_id", sa.BigInteger(), nullable=True),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("level", sa.SmallInteger(), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("number", sa.String(length=64), nullable=True),
        sa.Column("number_normalized", sa.String(length=64), nullable=True),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_sections_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["parent_id"],
            ["sections.id"],
            name=op.f("fk_sections_parent_id_sections"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_sections")),
    )
    op.create_index(
        "ix_sections_document_number",
        "sections",
        ["document_id", "number_normalized"],
        unique=False,
    )
    op.create_index(
        "ix_sections_document_ordinal", "sections", ["document_id", "ordinal"], unique=True
    )
    op.create_table(
        "blocks",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("document_id", sa.UUID(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("section_id", sa.BigInteger(), nullable=True),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.CheckConstraint(
            "kind IN ('paragraph', 'heading', 'table_row')", name=op.f("ck_blocks_kind")
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_blocks_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["section_id"],
            ["sections.id"],
            name=op.f("fk_blocks_section_id_sections"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_blocks")),
    )
    op.create_index("ix_blocks_document_ordinal", "blocks", ["document_id", "ordinal"], unique=True)
    op.create_table(
        "chunks",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("document_id", sa.UUID(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("section_id", sa.BigInteger(), nullable=True),
        sa.Column("section_path", sa.Text(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column("block_start", sa.Integer(), nullable=False),
        sa.Column("block_end", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("context_text", sa.Text(), nullable=False),
        sa.Column("lexical_tokens", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.vector.VECTOR(dim=384), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_chunks_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["section_id"],
            ["sections.id"],
            name=op.f("fk_chunks_section_id_sections"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chunks")),
    )
    op.create_index("ix_chunks_document_ordinal", "chunks", ["document_id", "ordinal"], unique=True)
    op.add_column("documents", sa.Column("stage_detail", sa.String(length=120), nullable=True))
    op.add_column("documents", sa.Column("index_version", sa.SmallInteger(), nullable=True))
    op.add_column("documents", sa.Column("embedding_model", sa.String(length=128), nullable=True))
    op.add_column("documents", sa.Column("char_count", sa.Integer(), nullable=True))
    op.add_column("documents", sa.Column("section_count", sa.Integer(), nullable=True))
    op.add_column("documents", sa.Column("chunk_count", sa.Integer(), nullable=True))
    op.add_column("documents", sa.Column("ocr_page_count", sa.Integer(), nullable=True))
    op.add_column("documents", sa.Column("unreadable_page_count", sa.Integer(), nullable=True))
    op.add_column(
        "documents",
        sa.Column(
            "extraction_warnings",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("documents", "extraction_warnings")
    op.drop_column("documents", "unreadable_page_count")
    op.drop_column("documents", "ocr_page_count")
    op.drop_column("documents", "chunk_count")
    op.drop_column("documents", "section_count")
    op.drop_column("documents", "char_count")
    op.drop_column("documents", "embedding_model")
    op.drop_column("documents", "index_version")
    op.drop_column("documents", "stage_detail")
    op.drop_index("ix_chunks_document_ordinal", table_name="chunks")
    op.drop_table("chunks")
    op.drop_index("ix_blocks_document_ordinal", table_name="blocks")
    op.drop_table("blocks")
    op.drop_index("ix_sections_document_ordinal", table_name="sections")
    op.drop_index("ix_sections_document_number", table_name="sections")
    op.drop_table("sections")
    op.drop_table("document_texts")
    op.drop_table("document_pages")
