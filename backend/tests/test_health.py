import httpx
from fastapi import FastAPI

from app.api.routes.health import get_health_service
from app.core.config import Settings
from app.db.session import create_engine
from app.schemas.health import ComponentCheck, HealthChecks, HealthResponse
from app.services.health import HealthService, expected_migration_heads


class StubHealthService:
    def __init__(self, ok: bool) -> None:
        self.ok = ok

    async def check(self) -> HealthResponse:
        check = ComponentCheck(status="ok" if self.ok else "error")
        return HealthResponse(
            status="ok" if self.ok else "degraded",
            version="test",
            environment="test",
            llm_configured=False,
            checks=HealthChecks(database=check, pgvector=check, migrations=check),
        )


async def test_health_ok_returns_200(app: FastAPI, client: httpx.AsyncClient) -> None:
    app.dependency_overrides[get_health_service] = lambda: StubHealthService(ok=True)
    response = await client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "X-Request-ID" in response.headers


async def test_health_degraded_returns_503(app: FastAPI, client: httpx.AsyncClient) -> None:
    app.dependency_overrides[get_health_service] = lambda: StubHealthService(ok=False)
    response = await client.get("/api/health")
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"


async def test_unreachable_database_is_reported_not_raised(settings: Settings) -> None:
    # Port 1 on localhost refuses connections immediately: a real driver-level failure.
    broken = settings.model_copy(
        update={"database_url": "postgresql+asyncpg://u:p@127.0.0.1:1/none"}
    )
    engine = create_engine(broken)
    try:
        result = await HealthService(engine, broken).check()
    finally:
        await engine.dispose()
    assert result.status == "degraded"
    assert result.checks.database.status == "error"
    assert result.checks.database.detail
    assert result.checks.pgvector.detail == "database unreachable"


async def test_liveness_never_touches_database(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_single_migration_head() -> None:
    # Two heads means two migrations branched from the same parent and must be merged.
    assert len(expected_migration_heads()) == 1
