"""FastAPI application factory."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import documents, health
from app.core.config import APP_VERSION, Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import RequestContextMiddleware, UploadSizeLimitMiddleware
from app.db.session import dispose_engine, get_sessionmaker, init_engine
from app.jobs.worker import JobWorker
from app.storage import build_storage

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        init_engine(settings)
        storage = build_storage(settings)
        app.state.storage = storage
        worker = JobWorker(get_sessionmaker(), storage, settings)
        app.state.worker = worker
        if settings.worker_enabled:
            worker.start()
        logger.info(
            "ClauseLens API starting",
            extra={
                "version": APP_VERSION,
                "env": settings.app_env,
                "cors_origins": settings.cors_origin_list,
                "llm_configured": settings.llm_configured,
                "storage": settings.storage_backend,
                "worker": settings.worker_enabled,
            },
        )
        yield
        await worker.stop()
        if hasattr(storage, "aclose"):
            await storage.aclose()
        await dispose_engine()
        logger.info("ClauseLens API stopped")

    app = FastAPI(
        title="ClauseLens API",
        version=APP_VERSION,
        description="Evidence-grounded contract intelligence.",
        lifespan=lifespan,
    )

    # Middleware order: the last one added runs first. CORS must wrap everything (including
    # error responses) so the browser can read error bodies.
    app.add_middleware(UploadSizeLimitMiddleware, max_bytes=settings.max_upload_bytes)
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_origin_regex=settings.cors_origin_regex,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "Content-Disposition"],
    )

    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(documents.router)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {"name": "ClauseLens API", "version": APP_VERSION, "docs": "/docs"}

    return app


app = create_app()
