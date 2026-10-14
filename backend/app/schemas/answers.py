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
    reason: str | None = Field(default=None, description="Why the citation is unverified.")


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
