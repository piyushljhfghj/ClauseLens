from typing import Literal

from pydantic import BaseModel, Field

# "disabled": intentionally switched off by configuration; not counted as a failure.
CheckStatus = Literal["ok", "error", "disabled"]


class ComponentCheck(BaseModel):
    status: CheckStatus
    latency_ms: float | None = None
    detail: str | None = Field(
        default=None, description="Version info, or the reason a check failed."
    )


class HealthChecks(BaseModel):
    database: ComponentCheck
    pgvector: ComponentCheck
    migrations: ComponentCheck
    ocr: ComponentCheck
    embeddings: ComponentCheck


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    environment: str
    llm_configured: bool = Field(
        description="Whether LLM_BASE_URL, LLM_API_KEY and LLM_MODEL are set."
    )
    checks: HealthChecks


class LivenessResponse(BaseModel):
    status: Literal["ok"] = "ok"
