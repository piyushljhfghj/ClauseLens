import uuid
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.schemas.retrieval import DocumentCoverage, PassageQuality

Grounding = Literal["grounded", "partially_grounded", "ungrounded", "not_found"]


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, description="The question about the selected documents.")
    document_ids: list[uuid.UUID] = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question must not be blank.")
        return value

    @field_validator("document_ids")
    @classmethod
    def _unique(cls, value: list[uuid.UUID]) -> list[uuid.UUID]:
        return list(dict.fromkeys(value))


class EvidenceOut(BaseModel):
    """A passage the model was given, labelled as it saw it (E1, E2, ...)."""

    evidence_id: str
    document_id: uuid.UUID
    document_filename: str
    chunk_id: int
    section_path: str
    char_start: int
    char_end: int
    page_start: int | None
    page_end: int | None
    text: str
    retrieval_rank: int
    quality: PassageQuality


class HighlightRect(BaseModel):
    """One highlight rectangle: a line of a verified quote on a PDF page."""

    page: int = Field(description="1-based page number, as in `page_start` / `page_end`.")
    page_width: float = Field(description="Displayed page width in PDF points.")
    page_height: float = Field(description="Displayed page height in PDF points.")
    x0: float = Field(description="Left, in PDF points from the page's left edge.")
    y0: float = Field(description="Top, in PDF points from the page's top edge (y grows down).")
    x1: float
    y1: float


class BlockRange(BaseModel):
    """The part of a verified quote inside one block of a Word document's extracted text."""

    block: int = Field(description="0-based block number in document order, as stored.")
    kind: str = Field(description="paragraph, heading or table_row.")
    block_char_start: int = Field(description="The block's start in the canonical text.")
    block_char_end: int = Field(description="The block's end in the canonical text.")
    start: int = Field(description="Where the quote starts within the block's text.")
    end: int = Field(description="Where the quote ends (exclusive) within the block's text.")


class CitationOut(BaseModel):
    n: int = Field(description="Marker number used in the answer text, e.g. [1].")
    status: Literal["verified", "unverified"]
    proposed_quote: str = Field(description="What the model claimed to quote (untrusted).")
    cited_evidence_id: str
    # Present only when verified: computed from stored canonical text, never from the model.
    evidence_id: str | None = None
    document_id: uuid.UUID | None = None
    document_filename: str | None = None
    quote: str | None = Field(default=None, description="Exact canonical text that was matched.")
    char_start: int | None = None
    char_end: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    quality: PassageQuality | None = None
    highlights: list[HighlightRect] = Field(
        default_factory=list,
        description="Verified PDF quotes only: one rectangle per text line, on every page the "
        "quote spans, in displayed-page coordinates (rotation applied, origin top-left). Empty "
        "for unverified quotes, Word documents, or when the stored layout cannot place the quote "
        "reliably (then `reason` says so).",
    )
    block_ranges: list[BlockRange] = Field(
        default_factory=list,
        description="Verified Word (DOCX) quotes only: the quote's range in each extracted block "
        "it spans (block text = canonical text [block_char_start, block_char_end); the blank "
        "line between blocks belongs to neither). Offsets count Unicode code points, like every "
        "character offset in this API; a JavaScript string index counts UTF-16 units, which "
        "differ only for characters outside the Basic Multilingual Plane. Empty for unverified "
        "quotes, PDFs, or when the stored structure cannot place the quote (then `reason` says "
        "so).",
    )
    occurrence_count: int | None = Field(
        default=None,
        description="How many times the quote occurs (as whole words) in its document.",
    )
    location_ambiguous: bool = Field(
        default=False,
        description="The quote occurs more than once where it was found (the cited passage, the "
        "other passages given to the model, or the rest of the document): the location is the "
        "first of several equally plausible ones, not a definitive one.",
    )
    reason: str | None = Field(
        default=None,
        description="Why the citation is unverified; for a verified one, where it was found when "
        "that is not simply the cited passage.",
    )


class ExcludedPassage(BaseModel):
    document_id: uuid.UUID
    chunk_id: int
    retrieval_rank: int
    reason: str


class AnswerResponse(BaseModel):
    question: str
    answer: str = Field(description="Answer text with [n] markers; see `grounding` first.")
    grounding: Grounding
    citations: list[CitationOut]
    evidence: list[EvidenceOut]
    excluded: list[ExcludedPassage] = Field(
        description="Retrieved passages not shown to the model, and why (e.g. weak match)."
    )
    coverage: list[DocumentCoverage]
    notes: list[str]
    unsupported_draft: str | None = Field(
        default=None,
        description="The model's answer when nothing in it could be verified (shown for "
        "transparency only; the `answer` field then states that support was insufficient).",
    )
    model: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
