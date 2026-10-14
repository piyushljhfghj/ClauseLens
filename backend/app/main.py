"""FastAPI application factory."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import answers, documents, health, retrieval
from app.core.config import APP_VERSION, Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import RequestContextMiddleware, UploadSizeLimitMiddleware
from app.db.session import dispose_engine, get_sessionmaker, init_engine
from app.embeddings import Embedder, FastEmbedEmbedder
from app.ingestion.ocr import tesseract_version
from app.jobs.worker import JobWorker
from app.llm import LLMClient, OpenAICompatibleClient
from app.storage import build_storage

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    embedder: Embedder | None = None,
    llm: LLMClient | None = None,
) -> FastAPI:
    """`embedder` and `llm` can be injected (tests use deterministic fakes)."""
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        init_engine(settings)
        storage = build_storage(settings)
        app.state.storage = storage
        app.state.embedder = embedder or FastEmbedEmbedder(
            settings.embedding_model,
            cache_dir=settings.embedding_cache_dir,
            batch_size=settings.embedding_batch_size,
        )
        worker = JobWorker(get_sessionmaker(), storage, settings, app.state.embedder)
        # None when no provider is configured: answering then returns 503, everything else works.
        app.state.llm = llm or (
            OpenAICompatibleClient(settings) if settings.llm_configured else None
        )
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
                "ocr": (
                    (tesseract_version(settings.tesseract_cmd) or "unavailable")
                    if settings.ocr_enabled
                    else "disabled"
                ),
                "embedding_model": settings.embedding_model,
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

    # Request handlers get settings via Depends(get_settings); make that resolve to the settings
    # this app was created with, so lifespan, middleware and routes can never disagree.
    app.dependency_overrides[get_settings] = lambda: settings
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(retrieval.router)
    app.include_router(answers.router)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {"name": "ClauseLens API", "version": APP_VERSION, "docs": "/docs"}

    return app


app = create_app()
