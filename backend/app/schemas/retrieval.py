import uuid
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1, description="Natural-language question or keywords.")
    document_ids: list[uuid.UUID] = Field(
        min_length=1, description="Documents to search (the selected document or documents)."
    )
    top_k: int | None = Field(default=None, ge=1, description="Passages to return.")

    @field_validator("query")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Query must not be blank.")
        return value

    @field_validator("document_ids")
    @classmethod
    def _unique(cls, value: list[uuid.UUID]) -> list[uuid.UUID]:
        return list(dict.fromkeys(value))  # de-duplicate, keep order


class RetrievalSignals(BaseModel):
    """Why a passage was retrieved. Ranks are 1-based within (retriever, document)."""

    dense_rank: int | None = None
    dense_similarity: float | None = Field(default=None, description="Cosine similarity.")
    bm25_rank: int | None = None
    bm25_score: float | None = None
    phrase_rank: int | None = None
    section_rank: int | None = None


class PassageQuality(BaseModel):
    """How the passage's pages were read. Low-quality OCR text is retrievable but unreliable;
    downstream answers and citations must treat it accordingly."""

    level: Literal["text", "ocr", "low_quality_ocr", "not_applicable"] = Field(
        description="Worst extraction quality among the passage's pages; not_applicable for DOCX."
    )
    ocr_pages: list[int] = Field(default_factory=list)
    low_quality_pages: list[int] = Field(default_factory=list)
    min_ocr_confidence: float | None = None


class RetrievedPassage(BaseModel):
    rank: int
    score: float = Field(description="Reciprocal Rank Fusion score (relative, not a probability).")
    document_id: uuid.UUID
    document_filename: str
    chunk_id: int
    chunk_ordinal: int
    section_id: int | None
    section_path: str
    # Canonical source location: text == canonical_text[char_start:char_end]. Word boxes for
    # these characters live on the document's pages (for highlighting in a later phase).
    char_start: int
    char_end: int
    page_start: int | None
    page_end: int | None
    text: str = Field(description="Exact canonical text of the passage (never modified).")
    signals: RetrievalSignals
    quality: PassageQuality


class DocumentCoverage(BaseModel):
    """What the search covered in one document, including what extraction could not read."""

    document_id: uuid.UUID
    filename: str
    chunks_searched: int
    page_count: int | None
    unreadable_page_count: int | None
    low_quality_page_count: int | None


class QueryAnalysisOut(BaseModel):
    terms: list[str]
    phrases: list[str]
    section_refs: list[str]


class RetrievalResponse(BaseModel):
    query: str
    passages: list[RetrievedPassage]
    no_match: bool = Field(description="True when nothing in the documents matched the query.")
    coverage: list[DocumentCoverage]
    analysis: QueryAnalysisOut
    notes: list[str] = Field(
        default_factory=list,
        description="Caveats, e.g. no searchable terms, duplicates removed, unreadable pages.",
    )
