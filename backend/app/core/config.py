"""Application settings, loaded from environment variables (and `backend/.env` in development).

Every tunable lives here so that no other module reads `os.environ` directly.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_VERSION = "0.1.0"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Runtime ---------------------------------------------------------------------------
    app_env: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # "text" is readable in a terminal; "json" is what hosted log viewers handle best.
    log_format: Literal["text", "json"] = "text"

    # --- Database --------------------------------------------------------------------------
    # Accepts the usual `postgres://` / `postgresql://` forms (as copied from Supabase or
    # Railway) and converts them to the async driver URL SQLAlchemy needs.
    database_url: str = "postgresql+asyncpg://clauselens:clauselens@localhost:5432/clauselens"
    # Hosted Postgres (Supabase) requires TLS; the local Docker database does not.
    db_ssl: bool = False
    # asyncpg caches prepared statements. Poolers in *transaction* mode (e.g. Supabase port
    # 6543) do not support that, so set this to 0 there. Session mode (port 5432) is fine.
    db_statement_cache_size: int = Field(default=100, ge=0)
    db_pool_size: int = Field(default=5, ge=1)
    db_max_overflow: int = Field(default=5, ge=0)
    # When true, the container runs `alembic upgrade head` before starting the API.
    run_migrations_on_start: bool = False

    # --- HTTP ------------------------------------------------------------------------------
    # Comma-separated list of browser origins allowed to call the API.
    cors_origins: str = "http://localhost:3000"
    # Optional regex for origins, e.g. Vercel preview deployments: https://clauselens-.*\.vercel\.app
    cors_origin_regex: str | None = None

    # --- Uploads & documents ---------------------------------------------------------------
    max_upload_mb: int = Field(default=25, ge=1, le=200)
    # Hard ceiling on pages, so one pathological file cannot monopolise the worker.
    max_pages: int = Field(default=1000, ge=1)

    # --- File storage ------------------------------------------------------------------------
    # "local" keeps files on disk (development). "supabase" uses Supabase Storage (production:
    # container disks on Hugging Face Spaces are wiped on restart).
    storage_backend: Literal["local", "supabase"] = "local"
    storage_local_dir: str = "storage"
    supabase_url: str | None = None
    supabase_service_key: SecretStr | None = None
    supabase_storage_bucket: str = "documents"

    # --- Extraction & OCR -----------------------------------------------------------------------
    # Pages without a usable text layer are OCR'd with Tesseract. Disable only if Tesseract is
    # not installed; scanned pages are then reported as unreadable rather than silently skipped.
    ocr_enabled: bool = True
    ocr_language: str = "eng"
    ocr_dpi: int = Field(default=300, ge=100, le=600)
    # Full path to tesseract(.exe) if it is not on PATH.
    tesseract_cmd: str | None = None

    # --- Embeddings ------------------------------------------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Where fastembed caches the downloaded ONNX model (~65 MB).
    embedding_cache_dir: str = ".cache/fastembed"
    embedding_batch_size: int = Field(default=32, ge=1, le=256)

    # --- Retrieval ------------------------------------------------------------------------------
    # Candidates taken from each retriever (dense, BM25, phrase, section) per document.
    retrieval_candidates_per_list: int = Field(default=30, ge=1, le=200)
    retrieval_default_top_k: int = Field(default=10, ge=1, le=50)
    retrieval_max_top_k: int = Field(default=50, ge=1, le=200)
    retrieval_max_documents: int = Field(default=10, ge=1, le=50)
    retrieval_max_query_chars: int = Field(default=1000, ge=10, le=10000)
    # Reciprocal Rank Fusion constant (Cormack et al. 2009): 60 is the standard choice.
    retrieval_rrf_k: int = Field(default=60, ge=1)
    # Semantic-only hits below this cosine similarity are dropped, so an off-topic query returns
    # nothing instead of the "least unrelated" chunks. Calibrated for bge-small-en-v1.5: related
    # query/passage pairs measured 0.62-0.74, unrelated 0.38-0.51.
    retrieval_min_dense_similarity: float = Field(default=0.55, ge=-1.0, le=1.0)
    # BM25 parameters (standard values).
    retrieval_bm25_k1: float = Field(default=1.2, gt=0)
    retrieval_bm25_b: float = Field(default=0.75, ge=0, le=1)

    # --- Background jobs -----------------------------------------------------------------------
    # The worker runs inside the API process. Tests disable it and drive jobs explicitly.
    worker_enabled: bool = True
    worker_poll_interval_s: float = Field(default=2.0, gt=0)
    # A running job whose heartbeat is older than this is assumed dead and re-queued.
    job_stale_after_s: int = Field(default=90, ge=10)
    job_max_attempts: int = Field(default=3, ge=1)

    # --- LLM (used from Phase 5; any OpenAI-compatible provider) ---------------------------
    llm_base_url: str | None = None
    llm_api_key: SecretStr | None = None
    llm_model: str | None = None

    @field_validator("database_url")
    @classmethod
    def _normalise_database_url(cls, value: str) -> str:
        for prefix in ("postgres://", "postgresql://"):
            if value.startswith(prefix):
                return "postgresql+asyncpg://" + value[len(prefix) :]
        return value

    @field_validator(
        "llm_base_url",
        "llm_model",
        "cors_origin_regex",
        "supabase_url",
        "tesseract_cmd",
        mode="before",
    )
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("llm_api_key", "supabase_service_key", mode="before")
    @classmethod
    def _blank_key_to_none(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @property
    def cors_origin_list(self) -> list[str]:
        return [
            origin.strip().rstrip("/") for origin in self.cors_origins.split(",") if origin.strip()
        ]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @model_validator(mode="after")
    def _check_storage(self) -> "Settings":
        if self.storage_backend == "supabase" and not (
            self.supabase_url and self.supabase_service_key
        ):
            raise ValueError(
                "STORAGE_BACKEND=supabase requires SUPABASE_URL and SUPABASE_SERVICE_KEY"
            )
        return self

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_base_url and self.llm_api_key and self.llm_model)

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
