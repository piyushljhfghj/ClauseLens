"""Async engine and session management.

The engine is created once per process (in the app lifespan) and disposed on shutdown.
Request handlers get a session through the `get_session` dependency.
"""

from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import Settings

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def build_connect_args(settings: Settings) -> dict[str, Any]:
    connect_args: dict[str, Any] = {
        "statement_cache_size": settings.db_statement_cache_size,
        "timeout": 10,  # seconds to establish a connection
        "server_settings": {"application_name": "clauselens-api"},
    }
    if settings.db_ssl:
        connect_args["ssl"] = "require"
    return connect_args


def create_engine(settings: Settings) -> AsyncEngine:
    return create_async_engine(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_pre_ping=True,  # transparently replace connections dropped by the server/pooler
        pool_recycle=1800,
        connect_args=build_connect_args(settings),
    )


def init_engine(settings: Settings) -> AsyncEngine:
    global _engine, _sessionmaker
    _engine = create_engine(settings)
    _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


async def dispose_engine() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine, _sessionmaker = None, None


def get_engine() -> AsyncEngine:
    if _engine is None:
        raise RuntimeError("Database engine is not initialised (app lifespan has not run).")
    return _engine


async def get_session() -> AsyncIterator[AsyncSession]:
    if _sessionmaker is None:
        raise RuntimeError("Database engine is not initialised (app lifespan has not run).")
    async with _sessionmaker() as session:
        yield session


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """For code outside a request (the background worker)."""
    if _sessionmaker is None:
        raise RuntimeError("Database engine is not initialised (app lifespan has not run).")
    return _sessionmaker
