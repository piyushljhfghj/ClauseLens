"""Integration test against a real Postgres (the Docker Compose database).

Skipped automatically when the database is not reachable, so `uv run pytest` still works
without Docker. Run migrations first: `uv run alembic upgrade head`.
"""

import pytest

from app.core.config import Settings, get_settings
from app.db.session import create_engine
from app.services.health import HealthService


@pytest.fixture
def real_settings() -> Settings:
    return get_settings()


async def test_health_against_real_database(real_settings: Settings) -> None:
    engine = create_engine(real_settings)
    try:
        result = await HealthService(engine, real_settings).check()
    finally:
        await engine.dispose()

    if result.checks.database.status != "ok":
        pytest.skip(f"database not reachable: {result.checks.database.detail}")

    assert result.checks.pgvector.status == "ok", result.checks.pgvector.detail
    assert result.checks.migrations.status == "ok", result.checks.migrations.detail
    assert result.status == "ok"
