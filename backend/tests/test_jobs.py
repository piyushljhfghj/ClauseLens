"""Background job behaviour: retries, crash recovery, deletion mid-job, live progress, and the
real worker loop."""

import asyncio
import threading

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import select, text

from app.db.models import Document, Job
from app.db.session import get_sessionmaker
from app.ingestion import pipeline
from app.ingestion.inspect import InspectionResult
from app.jobs import queue
from app.jobs import worker as worker_module
from tests.fixtures import factory


async def upload_pdf(api: httpx.AsyncClient, pages: int = 1) -> str:
    response = await api.post(
        "/api/documents", files={"file": ("a.pdf", factory.text_pdf(pages=pages))}
    )
    assert response.status_code == 202
    return response.json()["id"]


async def load(document_id: str) -> tuple[Document, Job | None]:
    async with get_sessionmaker()() as session:
        document = await session.get(Document, document_id)
        job = (
            await session.execute(select(Job).where(Job.document_id == document_id))
        ).scalar_one_or_none()
        return document, job  # type: ignore[return-value]


async def make_job_runnable_now() -> None:
    async with get_sessionmaker()() as session:
        await session.execute(text("UPDATE jobs SET run_after = now()"))
        await session.commit()


async def test_unexpected_errors_are_retried_then_fail(
    api: httpx.AsyncClient, live_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_inspect(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("simulated bug")

    monkeypatch.setattr(pipeline, "inspect", broken_inspect)
    document_id = await upload_pdf(api)
    worker = live_app.state.worker

    await worker.run_once()
    document, job = await load(document_id)
    assert (document.status, document.stage) == ("queued", "retrying")
    assert job is not None and (job.status, job.attempts) == ("queued", 1)
    assert "simulated bug" in (job.last_error or "")
    assert not await worker.run_once(), "retry must wait for its backoff delay"

    for _ in range(2):  # attempts 2 and 3 (max_attempts = 3)
        await make_job_runnable_now()
        assert await worker.run_once()

    document, job = await load(document_id)
    assert document.status == "failed"
    assert document.error_code == "PROCESSING_ERROR"
    assert "simulated bug" not in (document.error_message or ""), "internals must not leak"
    assert job is not None and (job.status, job.attempts) == ("failed", 3)


async def simulate_crash_mid_job(document_id: str, *, attempts: int | None = None) -> None:
    """Claim the job as a worker would, then 'die': the heartbeat stops and goes stale."""
    async with get_sessionmaker()() as session:
        await queue.claim_next(session)
        await session.execute(
            text("UPDATE documents SET status = 'processing', stage = 'inspecting' WHERE id = :id"),
            {"id": document_id},
        )
        stale = "UPDATE jobs SET heartbeat_at = now() - interval '1 hour'"
        if attempts is not None:
            stale += f", attempts = {attempts}"
        await session.execute(text(stale))
        await session.commit()


async def test_crashed_job_is_recovered_and_completes(
    api: httpx.AsyncClient, live_app: FastAPI
) -> None:
    document_id = await upload_pdf(api)
    await simulate_crash_mid_job(document_id)

    recovered = await live_app.state.worker.recover_stale()
    assert [job.requeued for job in recovered] == [True]
    document, _ = await load(document_id)
    assert (document.status, document.stage) == ("queued", "retrying")

    assert await live_app.state.worker.run_once()
    document, job = await load(document_id)
    assert document.status == "ready"
    assert job is not None and (job.status, job.attempts) == ("succeeded", 2)


async def test_crash_with_no_attempts_left_fails_the_document(
    api: httpx.AsyncClient, live_app: FastAPI
) -> None:
    document_id = await upload_pdf(api)
    await simulate_crash_mid_job(document_id, attempts=3)

    recovered = await live_app.state.worker.recover_stale()
    assert [job.requeued for job in recovered] == [False]
    document, job = await load(document_id)
    assert (document.status, document.error_code) == ("failed", "PROCESSING_INTERRUPTED")
    assert job is not None and job.status == "failed"


async def test_live_jobs_are_not_treated_as_stale(
    api: httpx.AsyncClient, live_app: FastAPI
) -> None:
    await upload_pdf(api)
    async with get_sessionmaker()() as session:
        await queue.claim_next(session)  # fresh heartbeat
        await session.commit()
    assert await live_app.state.worker.recover_stale() == []


class BlockingInspect:
    """Stands in for inspection: reports 50% progress, then waits until released."""

    def __init__(self) -> None:
        self.reached = threading.Event()
        self.release = threading.Event()

    def __call__(self, file_type, data, *, max_pages, on_progress=None):  # type: ignore[no-untyped-def]
        if on_progress:
            on_progress(0.5)
        self.reached.set()
        self.release.wait(timeout=10)
        return InspectionResult(page_count=1, pages_without_text=0)


async def test_progress_is_visible_while_processing(
    api: httpx.AsyncClient, live_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = BlockingInspect()
    monkeypatch.setattr(pipeline, "inspect", blocker)
    monkeypatch.setattr(worker_module, "HEARTBEAT_INTERVAL_S", 0.05)
    document_id = await upload_pdf(api)

    run = asyncio.create_task(live_app.state.worker.run_once())
    await asyncio.to_thread(blocker.reached.wait, 5)
    await asyncio.sleep(0.3)  # let a few heartbeats land

    seen = (await api.get(f"/api/documents/{document_id}")).json()
    assert seen["status"] == "processing"
    assert seen["stage"] == "inspecting"
    assert seen["progress"] == 50  # 5 + (95 - 5) * 0.5

    blocker.release.set()
    await run
    assert (await api.get(f"/api/documents/{document_id}")).json()["status"] == "ready"


async def test_document_deleted_mid_processing(
    api: httpx.AsyncClient, live_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = BlockingInspect()
    monkeypatch.setattr(pipeline, "inspect", blocker)
    document_id = await upload_pdf(api)

    run = asyncio.create_task(live_app.state.worker.run_once())
    await asyncio.to_thread(blocker.reached.wait, 5)
    assert (await api.delete(f"/api/documents/{document_id}")).status_code == 204
    blocker.release.set()

    assert await run  # finishes without raising
    assert (await api.get(f"/api/documents/{document_id}")).status_code == 404
    assert (await api.get("/api/documents")).json()["total"] == 0


async def test_running_worker_picks_up_uploads_immediately(
    api: httpx.AsyncClient, live_app: FastAPI
) -> None:
    worker = live_app.state.worker
    worker.start()
    try:
        document_id = await upload_pdf(api, pages=2)
        for _ in range(100):
            status = (await api.get(f"/api/documents/{document_id}")).json()["status"]
            if status in ("ready", "failed"):
                break
            await asyncio.sleep(0.05)
        assert status == "ready"
    finally:
        await worker.stop()
