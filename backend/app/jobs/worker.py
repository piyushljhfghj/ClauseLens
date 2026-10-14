"""In-process background worker.

Runs as an asyncio task inside the API process (one container on a free tier, so no separate
worker service). CPU-heavy steps run in threads so the API stays responsive.

Failure policy:
- `ProcessingFailure` (a problem with the file itself) -> document failed, no retry.
- Any other exception (bug, storage outage, ...) -> retried with backoff up to `max_attempts`,
  then the document is failed with a generic message.
- Process killed mid-job -> heartbeat goes stale -> `recover_stale` re-queues it.
"""

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db.models import Document
from app.embeddings import Embedder
from app.ingestion.inspect import ProcessingFailure
from app.ingestion.pipeline import ProgressReporter, ingest_document
from app.jobs import queue
from app.storage import Storage

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_S = 2.0
STALE_SWEEP_INTERVAL_S = 30.0
RETRY_BASE_DELAY_S = 5.0

JobHandler = Callable[[queue.ClaimedJob, ProgressReporter], Awaitable[None]]


class JobWorker:
    def __init__(
        self,
        sessionmaker: async_sessionmaker[AsyncSession],
        storage: Storage,
        settings: Settings,
        embedder: Embedder,
    ) -> None:
        self._sessionmaker = sessionmaker
        self._storage = storage
        self._settings = settings
        self._embedder = embedder
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._handlers: dict[str, JobHandler] = {"ingest": self._handle_ingest}

    # --- lifecycle -----------------------------------------------------------------------------

    def start(self) -> None:
        self._task = asyncio.create_task(self._run_forever(), name="job-worker")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    def notify(self) -> None:
        """Wake the worker immediately (called after a job is enqueued)."""
        self._wake.set()

    async def _run_forever(self) -> None:
        logger.info("job worker started")
        last_sweep = 0.0
        while True:
            try:
                if time.monotonic() - last_sweep > STALE_SWEEP_INTERVAL_S:
                    await self.recover_stale()
                    last_sweep = time.monotonic()
                if await self.run_once():
                    continue  # there may be more work queued
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("job worker loop error")  # e.g. database briefly unreachable
            self._wake.clear()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), self._settings.worker_poll_interval_s)

    # --- one job ---------------------------------------------------------------------------------

    async def recover_stale(self) -> list[queue.StaleJob]:
        async with self._sessionmaker() as session:
            stale = await queue.recover_stale(
                session, stale_after_s=self._settings.job_stale_after_s
            )
            await session.commit()
        for job in stale:
            logger.warning(
                "recovered stale job",
                extra={"job_id": str(job.id), "requeued": job.requeued},
            )
        return stale

    async def run_once(self) -> bool:
        """Claim and run one job. Returns False if there was nothing to do."""
        async with self._sessionmaker() as session:
            job = await queue.claim_next(session)
            await session.commit()
        if job is None:
            return False

        log_extra = {"job_id": str(job.id), "kind": job.kind, "attempt": job.attempts}
        logger.info("job started", extra=log_extra)
        progress = ProgressReporter()
        heartbeat = asyncio.create_task(self._heartbeat(job, progress))
        try:
            handler = self._handlers.get(job.kind)
            if handler is None:
                raise ProcessingFailure("UNKNOWN_JOB", f"no handler for job kind {job.kind!r}")
            await handler(job, progress)
        except ProcessingFailure as failure:
            logger.info("job failed permanently", extra={**log_extra, "code": failure.code})
            await self._fail(job, failure.code, failure.message, failure.message)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if job.attempts < job.max_attempts:
                delay = RETRY_BASE_DELAY_S * 2 ** (job.attempts - 1)
                logger.warning(
                    "job error, will retry", extra={**log_extra, "error": error, "delay_s": delay}
                )
                await self._retry(job, error, delay)
            else:
                logger.exception("job error, giving up", extra=log_extra)
                await self._fail(
                    job,
                    "PROCESSING_ERROR",
                    "Something went wrong while processing this document. "
                    "Use 'Retry processing' to try again.",
                    error,
                )
        else:
            async with self._sessionmaker() as session:
                await queue.mark_succeeded(session, job.id)
                await session.commit()
            logger.info("job succeeded", extra=log_extra)
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
        return True

    async def _heartbeat(self, job: queue.ClaimedJob, progress: ProgressReporter) -> None:
        last_reported: tuple[str, int, str | None] | None = None
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL_S)
            try:
                async with self._sessionmaker() as session:
                    await queue.heartbeat(session, job.id)
                    current = (progress.stage, progress.progress, progress.detail)
                    if job.document_id and current != last_reported:
                        await session.execute(
                            update(Document)
                            .where(Document.id == job.document_id, Document.status == "processing")
                            .values(stage=current[0], progress=current[1], stage_detail=current[2])
                        )
                        last_reported = current
                    await session.commit()
            except Exception:
                logger.warning("heartbeat failed", exc_info=True)

    async def _fail(
        self, job: queue.ClaimedJob, code: str, user_message: str, internal_error: str
    ) -> None:
        async with self._sessionmaker() as session:
            await queue.mark_failed(session, job.id, internal_error)
            if job.document_id:
                await session.execute(
                    update(Document)
                    .where(Document.id == job.document_id)
                    .values(
                        status="failed",
                        stage=None,
                        stage_detail=None,
                        error_code=code,
                        error_message=user_message,
                    )
                )
            await session.commit()

    async def _retry(self, job: queue.ClaimedJob, error: str, delay_s: float) -> None:
        async with self._sessionmaker() as session:
            await queue.schedule_retry(session, job.id, error, delay_s)
            if job.document_id:
                await session.execute(
                    update(Document)
                    .where(Document.id == job.document_id)
                    .values(status="queued", stage="retrying", stage_detail=None, progress=0)
                )
            await session.commit()

    # --- handlers --------------------------------------------------------------------------------

    async def _handle_ingest(self, job: queue.ClaimedJob, progress: ProgressReporter) -> None:
        assert job.document_id is not None
        await ingest_document(
            job.document_id,
            sessionmaker=self._sessionmaker,
            storage=self._storage,
            settings=self._settings,
            embedder=self._embedder,
            progress=progress,
        )
