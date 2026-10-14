"""Count of low-quality OCR pages per document (Phase 3 audit follow-up).

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-14

Nullable: existing documents get NULL until reprocessed (INDEX_VERSION 3 marks them stale).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("low_quality_page_count", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("documents", "low_quality_page_count")
