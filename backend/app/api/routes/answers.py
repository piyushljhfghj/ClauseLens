from typing import Annotated

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.answering.service import AnswerService
from app.core.config import Settings, get_settings
from app.db.session import get_session
from app.schemas.answers import AnswerRequest, AnswerResponse

router = APIRouter(prefix="/api/answers", tags=["answers"])


def get_answer_service(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AnswerService:
    state = request.app.state
    return AnswerService(session, state.embedder, getattr(state, "llm", None), settings)


@router.post(
    "",
    response_model=AnswerResponse,
    summary="Answer a question from the selected documents, with verified quotations",
    responses={
        404: {"description": "A selected document does not exist"},
        409: {"description": "A selected document is not indexed with the current pipeline"},
        422: {"description": "Invalid question or document selection"},
        502: {"description": "The LLM provider failed"},
        503: {"description": "No LLM provider is configured"},
    },
)
async def answer(
    body: AnswerRequest,
    service: Annotated[AnswerService, Depends(get_answer_service)],
) -> AnswerResponse:
    return await service.answer(body.question, body.document_ids)
