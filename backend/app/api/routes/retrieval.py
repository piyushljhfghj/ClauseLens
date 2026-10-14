from typing import Annotated

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db.session import get_session
from app.retrieval.service import RetrievalService
from app.schemas.retrieval import RetrievalRequest, RetrievalResponse

router = APIRouter(prefix="/api/retrieval", tags=["retrieval"])


def get_retrieval_service(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> RetrievalService:
    return RetrievalService(session, request.app.state.embedder, settings)


@router.post(
    "/search",
    response_model=RetrievalResponse,
    summary="Hybrid search (semantic + BM25 + phrase + clause reference) over selected documents",
    responses={
        404: {"description": "A selected document does not exist"},
        409: {"description": "A selected document is not indexed with the current pipeline"},
        422: {"description": "Invalid query, too many documents or top_k too large"},
    },
)
async def search(
    body: RetrievalRequest,
    service: Annotated[RetrievalService, Depends(get_retrieval_service)],
) -> RetrievalResponse:
    return await service.search(body.query, body.document_ids, body.top_k)
