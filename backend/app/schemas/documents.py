import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DocumentStatus = Literal["queued", "processing", "ready", "failed"]


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    file_type: Literal["pdf", "docx"]
    size_bytes: int
    status: DocumentStatus
    stage: str | None = Field(
        description="What processing is doing right now (e.g. 'inspecting'); null when idle."
    )
    progress: int = Field(ge=0, le=100)
    error_code: str | None = Field(description="Stable machine-readable reason when failed.")
    error_message: str | None = Field(description="Human-readable reason when failed.")
    page_count: int | None = Field(description="PDF page count; null for DOCX or until processed.")
    pages_without_text: int | None = Field(
        description="PDF pages with no selectable text (scanned)."
    )
    created_at: datetime
    updated_at: datetime
    processed_at: datetime | None


class DocumentList(BaseModel):
    items: list[DocumentOut]
    total: int
