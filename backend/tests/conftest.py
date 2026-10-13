import os

# Tests must never pick up a developer's real backend/.env (or real LLM keys).
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("LOG_LEVEL", "WARNING")

from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI

from app.core.config import Settings
from app.main import create_app


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, app_env="test", log_level="WARNING")  # type: ignore[call-arg]


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings)


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    # raise_app_exceptions=False lets us assert on the 500 response instead of the raw exception.
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# --- Database integration fixtures ---------------------------------------------------------------
# Integration tests run against a separate `<dev db name>_test` database on the same server,
# created and migrated automatically. They are skipped (not failed) when Postgres is not running.

import asyncio
from pathlib import Path

import asyncpg
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.engine import make_url

from app.core.config import get_settings

BACKEND_DIR = Path(__file__).resolve().parents[1]


async def _ensure_database(url: str) -> None:
    target = make_url(url)
    conn = await asyncpg.connect(
        host=target.host,
        port=target.port,
        user=target.username,
        password=target.password,
        database="postgres",
        timeout=5,
    )
    try:
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", target.database
        )
        if not exists:
            await conn.execute(f'CREATE DATABASE "{target.database}"')
    finally:
        await conn.close()


@pytest.fixture(scope="session")
def test_database_url() -> str:
    explicit = os.environ.get("TEST_DATABASE_URL")
    if explicit:
        url = Settings(_env_file=None, database_url=explicit).database_url  # type: ignore[call-arg]
    else:
        dev = make_url(get_settings().database_url)
        url = dev.set(database=f"{dev.database}_test").render_as_string(hide_password=False)
    try:
        asyncio.run(_ensure_database(url))
    except (OSError, asyncpg.PostgresError, TimeoutError) as exc:
        pytest.skip(f"Postgres not reachable for integration tests: {exc}")

    previous = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = url
    get_settings.cache_clear()
    try:
        command.upgrade(Config(str(BACKEND_DIR / "alembic.ini")), "head")
    finally:
        if previous is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous
        get_settings.cache_clear()
    return url


@pytest.fixture
def db_settings(test_database_url: str, tmp_path: Path) -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        app_env="test",
        log_level="WARNING",
        database_url=test_database_url,
        storage_backend="local",
        storage_local_dir=str(tmp_path / "storage"),
        worker_enabled=False,  # tests drive the worker explicitly
        max_upload_mb=2,
        job_max_attempts=3,
    )


@pytest.fixture
async def live_app(db_settings: Settings) -> AsyncIterator[FastAPI]:
    """The real app with its lifespan running (engine, storage, worker object) on a clean DB."""
    app = create_app(db_settings)
    async with app.router.lifespan_context(app):
        from app.db.session import get_engine

        async with get_engine().begin() as conn:
            await conn.execute(text("TRUNCATE documents, jobs CASCADE"))
        yield app


@pytest.fixture
async def api(live_app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=live_app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
