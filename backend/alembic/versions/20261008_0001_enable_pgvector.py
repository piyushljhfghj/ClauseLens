"""Enable the pgvector extension.

Revision ID: 0001
Revises:
Create Date: 2026-10-08

Tables are added by later migrations, one per phase. On Supabase, pgvector can also be enabled
from the dashboard (Database -> Extensions); IF NOT EXISTS makes this migration a no-op then.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS vector")
