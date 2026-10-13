import uuid
from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Query, Request, Response, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.db.session import get_session
from app.schemas.documents import DocumentList, DocumentOut
from app.services.documents import DocumentService

router = APIRouter(prefix="/api/documents", tags=["documents"])

READ_CHUNK_BYTES = 1024 * 1024


def get_document_service(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> DocumentService:
    worker = getattr(request.app.state, "worker", None)
    return DocumentService(
        session,
        request.app.state.storage,
        settings,
        on_job_enqueued=worker.notify if worker else (lambda: None),
    )


Service = Annotated[DocumentService, Depends(get_document_service)]


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DocumentOut,
    summary="Upload a PDF or DOCX; processing continues in the background",
    responses={
        400: {"description": "Empty file"},
        409: {"description": "Duplicate of a document already in the library"},
        413: {"description": "File too large"},
        415: {"description": "Unsupported, mismatched, legacy or encrypted file type"},
    },
)
async def upload_document(
    file: Annotated[UploadFile, File(description="The contract (.pdf or .docx)")],
    service: Service,
    settings: Annotated[Settings, Depends(get_settings)],
) -> DocumentOut:
    # Read in chunks and stop as soon as the limit is exceeded, so an oversized upload is not
    # loaded into memory (the middleware already rejects most of these by Content-Length).
    limit = settings.max_upload_bytes
    chunks: list[bytes] = []
    size = 0
    while chunk := await file.read(READ_CHUNK_BYTES):
        size += len(chunk)
        if size > limit:
            raise AppError(
                f"The file is larger than the {settings.max_upload_mb} MB limit.",
                code="FILE_TOO_LARGE",
                status_code=413,
            )
        chunks.append(chunk)
    document = await service.upload(file.filename, b"".join(chunks))
    return DocumentOut.model_validate(document)


@router.get("", response_model=DocumentList, summary="List documents, newest first")
async def list_documents(
    service: Service,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentList:
    items, total = await service.list(limit=limit, offset=offset)
    return DocumentList(items=[DocumentOut.model_validate(d) for d in items], total=total)


@router.get("/{document_id}", response_model=DocumentOut, summary="Get one document")
async def get_document(document_id: uuid.UUID, service: Service) -> DocumentOut:
    return DocumentOut.model_validate(await service.get(document_id))


@router.get(
    "/{document_id}/file",
    response_class=Response,
    summary="Download the original file",
    responses={200: {"content": {"application/pdf": {}, "application/octet-stream": {}}}},
)
async def get_document_file(
    document_id: uuid.UUID,
    service: Service,
    download: Annotated[bool, Query(description="Force a download instead of inline view")] = False,
) -> Response:
    document, data = await service.get_file(document_id)
    disposition = "attachment" if download else "inline"
    # RFC 6266/5987: an ASCII fallback plus the exact UTF-8 name.
    ascii_name = document.filename.encode("ascii", "replace").decode().replace('"', "'")
    return Response(
        content=data,
        media_type=document.mime_type,
        headers={
            "Content-Disposition": (
                f'{disposition}; filename="{ascii_name}"; '
                f"filename*=UTF-8''{quote(document.filename)}"
            ),
            "Cache-Control": "private, max-age=3600",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete a document and its stored file",
)
async def delete_document(document_id: uuid.UUID, service: Service) -> Response:
    await service.delete(document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{document_id}/reprocess",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DocumentOut,
    summary="Run processing again (e.g. after a failure)",
)
async def reprocess_document(document_id: uuid.UUID, service: Service) -> DocumentOut:
    return DocumentOut.model_validate(await service.reprocess(document_id))
