"""End-to-end document library tests: real Postgres, real storage (temp dir), real worker code."""

import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI
from sqlalchemy import func, select

from app.core.config import Settings
from app.db.models import Document, Job
from app.db.session import get_sessionmaker
from app.main import create_app
from tests.conftest import truncate_all
from tests.fakes import FakeEmbedder
from tests.fixtures import factory


async def upload(api: httpx.AsyncClient, name: str, data: bytes) -> httpx.Response:
    return await api.post("/api/documents", files={"file": (name, data)})


async def process_all(app: FastAPI) -> None:
    while await app.state.worker.run_once():
        pass


async def count(model: type) -> int:
    async with get_sessionmaker()() as session:
        return (await session.execute(select(func.count()).select_from(model))).scalar_one()


def stored_files(settings_dir: Path) -> list[Path]:
    return [p for p in settings_dir.rglob("*") if p.is_file()] if settings_dir.exists() else []


# --- happy paths ---------------------------------------------------------------------------------


async def test_pdf_upload_is_queued_then_processed(
    api: httpx.AsyncClient, live_app: FastAPI
) -> None:
    pdf = factory.text_pdf(pages=3)
    response = await upload(api, "Master Services Agreement.pdf", pdf)
    assert response.status_code == 202
    created = response.json()
    assert created["status"] == "queued"
    assert created["filename"] == "Master Services Agreement.pdf"
    assert created["file_type"] == "pdf"
    assert created["size_bytes"] == len(pdf)

    await process_all(live_app)

    document = (await api.get(f"/api/documents/{created['id']}")).json()
    assert document["status"] == "ready"
    assert document["progress"] == 100
    assert document["stage"] is None
    assert document["page_count"] == 3
    assert document["pages_without_text"] == 0
    assert document["processed_at"] is not None
    assert document["error_code"] is None


async def test_docx_upload_is_processed(api: httpx.AsyncClient, live_app: FastAPI) -> None:
    created = (await upload(api, "nda.docx", factory.docx_file())).json()
    await process_all(live_app)
    document = (await api.get(f"/api/documents/{created['id']}")).json()
    assert document["status"] == "ready"
    assert document["page_count"] is None  # Word documents have no fixed pages


async def test_list_is_newest_first(api: httpx.AsyncClient) -> None:
    first = (await upload(api, "a.pdf", factory.text_pdf(1, "first contract text " * 3))).json()
    second = (await upload(api, "b.pdf", factory.text_pdf(1, "second contract text " * 3))).json()
    listing = (await api.get("/api/documents")).json()
    assert listing["total"] == 2
    assert [d["id"] for d in listing["items"]] == [second["id"], first["id"]]


async def test_file_download_returns_original_bytes(api: httpx.AsyncClient) -> None:
    pdf = factory.text_pdf(pages=1)
    created = (await upload(api, "Vertrag über Lieferung.pdf", pdf)).json()

    inline = await api.get(f"/api/documents/{created['id']}/file")
    assert inline.status_code == 200
    assert inline.content == pdf
    assert inline.headers["content-type"] == "application/pdf"
    disposition = inline.headers["content-disposition"]
    assert disposition.startswith("inline;")
    assert "filename*=UTF-8''Vertrag%20%C3%BCber%20Lieferung.pdf" in disposition

    attachment = await api.get(f"/api/documents/{created['id']}/file", params={"download": True})
    assert attachment.headers["content-disposition"].startswith("attachment;")


# --- rejected uploads: nothing may be stored -----------------------------------------------------


async def assert_nothing_stored(db_settings_dir: Path) -> None:
    assert await count(Document) == 0
    assert await count(Job) == 0
    assert stored_files(db_settings_dir) == []


async def test_unsupported_type_is_rejected_and_nothing_stored(
    api: httpx.AsyncClient, db_settings: Settings
) -> None:
    response = await upload(api, "notes.txt", b"some text")
    assert response.status_code == 415
    error = response.json()["error"]
    assert error["code"] == "UNSUPPORTED_FILE_TYPE"
    assert "PDF" in error["message"] and "docx" in error["message"].lower()
    await assert_nothing_stored(Path(db_settings.storage_local_dir))


async def test_mismatched_and_legacy_files_are_rejected(api: httpx.AsyncClient) -> None:
    assert (await upload(api, "x.pdf", b"not a pdf")).json()["error"][
        "code"
    ] == "FILE_TYPE_MISMATCH"
    legacy = await upload(api, "old.doc", factory.OLE_HEADER)
    assert legacy.json()["error"]["code"] == "LEGACY_DOC_UNSUPPORTED"


async def test_empty_file_is_rejected(api: httpx.AsyncClient) -> None:
    response = await upload(api, "empty.pdf", b"")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "EMPTY_FILE"


async def test_oversized_file_is_rejected_before_reading(
    api: httpx.AsyncClient, db_settings: Settings
) -> None:
    too_big = b"%PDF-1.7\n" + b"0" * (3 * 1024 * 1024)  # limit is 2 MB in tests
    response = await upload(api, "big.pdf", too_big)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "FILE_TOO_LARGE"
    await assert_nothing_stored(Path(db_settings.storage_local_dir))


async def test_missing_file_field_is_a_validation_error(api: httpx.AsyncClient) -> None:
    response = await api.post("/api/documents", data={"other": "x"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


async def test_duplicate_upload_is_rejected_with_existing_id(api: httpx.AsyncClient) -> None:
    pdf = factory.text_pdf(pages=1)
    first = (await upload(api, "a.pdf", pdf)).json()
    duplicate = await upload(api, "renamed copy.pdf", pdf)
    assert duplicate.status_code == 409
    error = duplicate.json()["error"]
    assert error["code"] == "DUPLICATE_DOCUMENT"
    assert error["details"]["document_id"] == first["id"]


async def test_reupload_of_a_failed_document_is_allowed(
    api: httpx.AsyncClient, live_app: FastAPI
) -> None:
    locked = factory.encrypted_pdf()
    await upload(api, "locked.pdf", locked)
    await process_all(live_app)
    assert (await upload(api, "locked.pdf", locked)).status_code == 202


# --- processing failures: never "ready" ----------------------------------------------------------


async def test_scanned_pdf_without_ocr_fails_with_clear_reason(
    db_settings: Settings, fake_embedder: FakeEmbedder
) -> None:
    # With OCR switched off, a fully scanned PDF must fail, never become an empty "ready" doc.
    app = create_app(db_settings.model_copy(update={"ocr_enabled": False}), embedder=fake_embedder)
    async with app.router.lifespan_context(app):
        await truncate_all()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as api:
            response = await upload(api, "scan.pdf", factory.scanned_pdf())
            assert response.status_code == 202, response.text
            created = response.json()
            await process_all(app)
            document = (await api.get(f"/api/documents/{created['id']}")).json()
    assert document["status"] == "failed"
    assert document["error_code"] == "SCANNED_NO_TEXT"
    assert "scanned" in document["error_message"]


async def test_password_protected_pdf_fails(api: httpx.AsyncClient, live_app: FastAPI) -> None:
    created = (await upload(api, "locked.pdf", factory.encrypted_pdf())).json()
    await process_all(live_app)
    document = (await api.get(f"/api/documents/{created['id']}")).json()
    assert (document["status"], document["error_code"]) == ("failed", "ENCRYPTED_PDF")


async def test_permanent_failures_are_not_retried(
    api: httpx.AsyncClient, live_app: FastAPI
) -> None:
    await upload(api, "locked.pdf", factory.encrypted_pdf())
    await process_all(live_app)
    async with get_sessionmaker()() as session:
        job = (await session.execute(select(Job))).scalar_one()
    assert (job.status, job.attempts) == ("failed", 1)


# --- delete / not found / reprocess --------------------------------------------------------------


async def test_delete_removes_row_jobs_and_file(
    api: httpx.AsyncClient, live_app: FastAPI, db_settings: Settings
) -> None:
    created = (await upload(api, "a.pdf", factory.text_pdf(1))).json()
    await process_all(live_app)
    assert len(stored_files(Path(db_settings.storage_local_dir))) == 1

    response = await api.delete(f"/api/documents/{created['id']}")
    assert response.status_code == 204
    assert (await api.get(f"/api/documents/{created['id']}")).status_code == 404
    await assert_nothing_stored(Path(db_settings.storage_local_dir))


async def test_unknown_and_malformed_ids(api: httpx.AsyncClient) -> None:
    missing = await api.get(f"/api/documents/{uuid.uuid4()}")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"
    assert (await api.delete(f"/api/documents/{uuid.uuid4()}")).status_code == 404
    assert (await api.get("/api/documents/not-a-uuid")).status_code == 422


async def test_reprocess(api: httpx.AsyncClient, live_app: FastAPI) -> None:
    created = (await upload(api, "a.pdf", factory.text_pdf(1))).json()

    busy = await api.post(f"/api/documents/{created['id']}/reprocess")
    assert busy.status_code == 409
    assert busy.json()["error"]["code"] == "DOCUMENT_BUSY"

    await process_all(live_app)
    again = await api.post(f"/api/documents/{created['id']}/reprocess")
    assert again.status_code == 202
    assert again.json()["status"] == "queued"
    await process_all(live_app)
    assert (await api.get(f"/api/documents/{created['id']}")).json()["status"] == "ready"
