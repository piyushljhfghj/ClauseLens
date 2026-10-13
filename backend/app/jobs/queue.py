"""Postgres-backed job queue.

Why not Celery/Redis: one more service to host and pay for. Postgres' `FOR UPDATE SKIP LOCKED`
gives safe concurrent claiming, and rows survive restarts, which is exactly what we need.
Each function runs in the caller's session; the caller commits.
"""

import uuid
from dataclasses import dataclass

from sqlalchemy import text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, Job


@dataclass(frozen=True)
class ClaimedJob:
    id: uuid.UUID
    kind: str
    document_id: uuid.UUID | None
    attempts: int
    max_attempts: int


@dataclass(frozen=True)
class StaleJob:
    id: uuid.UUID
    document_id: uuid.UUID | None
    requeued: bool


async def enqueue(
    session: AsyncSession, *, kind: str, document_id: uuid.UUID, max_attempts: int
) -> Job:
    job = Job(kind=kind, document_id=document_id, status="queued", max_attempts=max_attempts)
    session.add(job)
    await session.flush()
    return job


_CLAIM_SQL = text(
    """
    UPDATE jobs
       SET status = 'running', attempts = attempts + 1,
           locked_at = now(), heartbeat_at = now(), last_error = NULL
     WHERE id = (
            SELECT id FROM jobs
             WHERE status = 'queued' AND run_after <= now()
             ORDER BY created_at
             FOR UPDATE SKIP LOCKED
             LIMIT 1)
    RETURNING id, kind, document_id, attempts, max_attempts
    """
)


async def claim_next(session: AsyncSession) -> ClaimedJob | None:
    row = (await session.execute(_CLAIM_SQL)).mappings().first()
    return ClaimedJob(**row) if row else None


async def heartbeat(session: AsyncSession, job_id: uuid.UUID) -> bool:
    """Returns False if the job no longer exists (its document was deleted)."""
    result = await session.execute(
        update(Job)
        .where(Job.id == job_id, Job.status == "running")
        .values(heartbeat_at=text("now()"))
    )
    return result.rowcount > 0  # type: ignore[attr-defined]


async def mark_succeeded(session: AsyncSession, job_id: uuid.UUID) -> None:
    await session.execute(
        update(Job).where(Job.id == job_id).values(status="succeeded", finished_at=text("now()"))
    )


async def mark_failed(session: AsyncSession, job_id: uuid.UUID, error: str) -> None:
    await session.execute(
        update(Job)
        .where(Job.id == job_id)
        .values(status="failed", last_error=error[:2000], finished_at=text("now()"))
    )


async def schedule_retry(
    session: AsyncSession, job_id: uuid.UUID, error: str, delay_s: float
) -> None:
    await session.execute(
        update(Job)
        .where(Job.id == job_id)
        .values(
            status="queued",
            last_error=error[:2000],
            locked_at=None,
            heartbeat_at=None,
            run_after=text("now() + make_interval(secs => :delay)").bindparams(
                delay=float(delay_s)
            ),
        )
    )


_STALE_SQL = text(
    """
    UPDATE jobs
       SET status = CASE WHEN attempts < max_attempts THEN 'queued' ELSE 'failed' END,
           finished_at = CASE WHEN attempts < max_attempts THEN NULL ELSE now() END,
           last_error = 'worker stopped responding (process restarted or crashed)',
           locked_at = NULL, heartbeat_at = NULL, run_after = now()
     WHERE status = 'running'
       AND heartbeat_at < now() - make_interval(secs => :stale_after)
    RETURNING id, document_id, (status = 'queued') AS requeued
    """
)


async def recover_stale(session: AsyncSession, *, stale_after_s: float) -> list[StaleJob]:
    """Re-queue (or fail, if out of attempts) jobs whose worker died mid-job, and reflect that on
    their documents so the UI does not show "processing" forever."""
    rows = (
        (await session.execute(_STALE_SQL, {"stale_after": float(stale_after_s)})).mappings().all()
    )
    stale = [StaleJob(**row) for row in rows]
    for job in stale:
        if job.document_id is None:
            continue
        values = (
            {"status": "queued", "stage": "retrying", "progress": 0}
            if job.requeued
            else {
                "status": "failed",
                "stage": None,
                "error_code": "PROCESSING_INTERRUPTED",
                "error_message": "Processing was interrupted repeatedly and has been stopped. "
                "Use 'Retry processing' to try again.",
            }
        )
        await session.execute(
            update(Document).where(Document.id == job.document_id).values(**values)
        )
    return stale
