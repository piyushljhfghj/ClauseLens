import uuid
from datetime import datetime
from typing import Literal, get_args

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, sql_in

JobKind = Literal["ingest"]
JobStatus = Literal["queued", "running", "succeeded", "failed"]


class Job(Base):
    """A unit of background work, persisted so it survives restarts.

    Claimed with `FOR UPDATE SKIP LOCKED`; a running job refreshes `heartbeat_at`, and a job
    whose heartbeat goes stale (process killed mid-job) is re-queued or, once `attempts` reaches
    `max_attempts`, failed.
    """

    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    kind: Mapped[str] = mapped_column(String(16))
    # Deleting a document deletes its jobs.
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[str] = mapped_column(String(16), default="queued")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    run_after: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint(sql_in("kind", get_args(JobKind)), name="kind"),
        CheckConstraint(sql_in("status", get_args(JobStatus)), name="status"),
        # The worker's claim query scans queued jobs in order.
        Index("ix_jobs_claim", "status", "run_after", "created_at"),
    )
