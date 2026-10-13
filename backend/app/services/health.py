"""Readiness checks: can we reach Postgres, is pgvector installed, are migrations applied?"""

import asyncio
import logging
import time
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from app.core.config import APP_VERSION, Settings
from app.schemas.health import ComponentCheck, HealthChecks, HealthResponse

logger = logging.getLogger(__name__)

CHECK_TIMEOUT_S = 5.0
_ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic.ini"


def expected_migration_heads() -> set[str]:
    config = Config(str(_ALEMBIC_INI))
    return set(ScriptDirectory.from_config(config).get_heads())


def _describe(exc: BaseException) -> str:
    if isinstance(exc, TimeoutError):
        return f"timed out after {CHECK_TIMEOUT_S:.0f}s"
    return f"{type(exc).__name__}: {exc}"[:300]


class HealthService:
    def __init__(self, engine: AsyncEngine, settings: Settings) -> None:
        self._engine = engine
        self._settings = settings

    async def check(self) -> HealthResponse:
        start = time.perf_counter()
        try:
            async with asyncio.timeout(CHECK_TIMEOUT_S):
                async with self._engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))
                    db_check = ComponentCheck(
                        status="ok", latency_ms=round((time.perf_counter() - start) * 1000, 1)
                    )
                    pgvector_check = await self._check_pgvector(conn)
                    migrations_check = await self._check_migrations(conn)
        except Exception as exc:  # any failure to reach the DB makes every DB check fail
            logger.warning("database health check failed", extra={"error": _describe(exc)})
            db_check = ComponentCheck(status="error", detail=_describe(exc))
            pgvector_check = ComponentCheck(status="error", detail="database unreachable")
            migrations_check = ComponentCheck(status="error", detail="database unreachable")

        checks = HealthChecks(
            database=db_check, pgvector=pgvector_check, migrations=migrations_check
        )
        all_ok = all(c.status == "ok" for c in (db_check, pgvector_check, migrations_check))
        return HealthResponse(
            status="ok" if all_ok else "degraded",
            version=APP_VERSION,
            environment=self._settings.app_env,
            llm_configured=self._settings.llm_configured,
            checks=checks,
        )

    @staticmethod
    async def _check_pgvector(conn: AsyncConnection) -> ComponentCheck:
        version = (
            await conn.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'"))
        ).scalar_one_or_none()
        if version is None:
            return ComponentCheck(
                status="error", detail="pgvector extension is not installed (run migrations)"
            )
        return ComponentCheck(status="ok", detail=f"pgvector {version}")

    @staticmethod
    async def _check_migrations(conn: AsyncConnection) -> ComponentCheck:
        expected = expected_migration_heads()
        exists = (
            await conn.execute(text("SELECT to_regclass('public.alembic_version') IS NOT NULL"))
        ).scalar_one()
        current = (
            set((await conn.execute(text("SELECT version_num FROM alembic_version"))).scalars())
            if exists
            else set()
        )
        if current == expected:
            return ComponentCheck(status="ok", detail=f"at {', '.join(sorted(current))}")
        return ComponentCheck(
            status="error",
            detail=(
                f"database at {sorted(current) or 'no revision'}, code expects {sorted(expected)}"
                " - run `uv run alembic upgrade head`"
            ),
        )
