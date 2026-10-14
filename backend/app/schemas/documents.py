import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

from app.ingestion.pipeline import INDEX_VERSION

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
    stage_detail: str | None = Field(default=None, description="e.g. 'OCR page 3 of 40'.")

    # Index summary; null until the document has been fully indexed.
    index_version: int | None = None
    char_count: int | None = None
    section_count: int | None = None
    chunk_count: int | None = None
    ocr_page_count: int | None = Field(default=None, description="Pages read with OCR.")
    unreadable_page_count: int | None = Field(
        default=None,
        description="Pages that may contain text that could not be read (OCR unavailable/failed).",
    )
    low_quality_page_count: int | None = Field(
        default=None,
        description=(
            "OCR pages read with low confidence: indexed, but possibly largely unrecognised."
        ),
    )
    extraction_warnings: list[str] = Field(default_factory=list)

    created_at: datetime
    updated_at: datetime
    processed_at: datetime | None

    @computed_field(description="Fully extracted, chunked and embedded with the current pipeline.")  # type: ignore[prop-decorator]
    @property
    def indexed(self) -> bool:
        return self.status == "ready" and self.index_version == INDEX_VERSION

    @computed_field(  # type: ignore[prop-decorator]
        description="Stored and inspected but not indexed (or indexed by an older pipeline)."
    )
    @property
    def needs_reprocessing(self) -> bool:
        return self.status == "ready" and self.index_version != INDEX_VERSION


class SectionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    parent_id: int | None
    ordinal: int
    level: int
    kind: str
    number: str | None
    title: str | None = Field(description="Heading text, when the section has one.")
    preview: str = Field(
        default="", description="Opening words of the section, for display when there is no title."
    )
    char_start: int
    char_end: int
    page_start: int | None
    page_end: int | None


class DocumentOutline(BaseModel):
    document_id: uuid.UUID
    sections: list[SectionOut]


class PageOut(BaseModel):
    page_number: int
    width: float
    height: float
    extraction: Literal["text", "ocr", "none"]
    note: str | None = Field(
        description=(
            "Why content may be missing: blank | ocr_unavailable | ocr_failed | ocr_no_text, "
            "or ocr_low_quality (text kept, low OCR confidence)."
        )
    )
    ocr_confidence: float | None
    char_count: int


class DocumentPages(BaseModel):
    document_id: uuid.UUID
    pages: list[PageOut]


class DocumentList(BaseModel):
    items: list[DocumentOut]
    total: int
