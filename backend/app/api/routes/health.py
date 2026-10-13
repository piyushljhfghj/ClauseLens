from typing import Annotated

from fastapi import APIRouter, Depends, Response

from app.core.config import Settings, get_settings
from app.db.session import get_engine
from app.schemas.health import HealthResponse, LivenessResponse
from app.services.health import HealthService

router = APIRouter(prefix="/api/health", tags=["health"])


def get_health_service(settings: Annotated[Settings, Depends(get_settings)]) -> HealthService:
    return HealthService(get_engine(), settings)


@router.get(
    "",
    response_model=HealthResponse,
    summary="Readiness: API, database, pgvector and migrations",
    responses={503: {"model": HealthResponse, "description": "One or more checks failed"}},
)
async def health(
    response: Response, service: Annotated[HealthService, Depends(get_health_service)]
) -> HealthResponse:
    result = await service.check()
    if result.status != "ok":
        response.status_code = 503
    return result


@router.get("/live", response_model=LivenessResponse, summary="Liveness: the process is up")
async def live() -> LivenessResponse:
    # Deliberately touches nothing external: hosting platforms use this to decide whether to
    # restart the container, and a database outage should not cause restart loops.
    return LivenessResponse()
