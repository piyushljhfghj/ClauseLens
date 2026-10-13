"""ORM models.

Import every model module here so `Base.metadata` is complete when Alembic autogenerates
migrations. Tables arrive phase by phase (documents in Phase 2, chunks in Phase 3, ...).
"""

from app.db.base import Base
from app.db.models.document import Document
from app.db.models.job import Job

__all__ = ["Base", "Document", "Job"]
