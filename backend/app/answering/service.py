"""Grounded question answering.

  retrieve (Phase 4) -> select strong evidence within a token budget -> LLM with evidence only
  -> parse proposed citations -> verify every quote against stored canonical text
  -> grounding verdict -> answer, or an explicit "insufficient support" statement

Nothing the model says about sources is trusted. An answer with no verified quotation is never
presented as an answer: the response states that support was insufficient and keeps the
model's draft separately for transparency.
"""

import asyncio
import bisect
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.answering.context import select_context
from app.answering.integrity import corruption, degenerate_repetition, uncited_statements
from app.answering.parser import ProposedCitation, parse_output
from app.answering.prompt import INSUFFICIENT_ANSWER, build_messages
from app.core.config import Settings
from app.db.models import DocumentPage, DocumentText
from app.embeddings import Embedder
from app.ingestion.model import LOW_QUALITY_NOTE
from app.llm import INCOMPLETE_OUTPUT, LLMClient, LLMNotConfiguredError, LLMOutputInvalidError
from app.retrieval.service import RetrievalService
from app.schemas.answers import (
    AnswerResponse,
    CitationOut,
    EvidenceOut,
    ExcludedPassage,
    Grounding,
)
from app.schemas.retrieval import DocumentCoverage, PassageQuality, RetrievedPassage
from app.verification.normalize import normalize
from app.verification.quotes import EvidenceText, QuoteCheck, SourceDocument, verify_quote

logger = logging.getLogger(__name__)

MALFORMED_OUTPUT = (
    "The answering service returned a malformed answer, so it is not shown. Please try again."
)


@dataclass(frozen=True)
class PageRecord:
    """A stored page's text range and how it was read (no word boxes: those are not needed)."""

    char_start: int
    char_end: int
    page_number: int
    extraction: str
    note: str | None
    ocr_confidence: float | None


PageRecords = dict[uuid.UUID, list[PageRecord]]  # document -> its pages, in page order


class AnswerService:
    def __init__(
        self,
        session: AsyncSession,
        embedder: Embedder,
        llm: LLMClient | None,
        settings: Settings,
    ) -> None:
        self._session = session
        self._embedder = embedder
        self._llm = llm
        self._settings = settings

    async def answer(self, question: str, document_ids: list[uuid.UUID]) -> AnswerResponse:
        if self._llm is None:
            raise LLMNotConfiguredError(
                "Question answering is not configured: set LLM_BASE_URL, LLM_API_KEY and "
                "LLM_MODEL on the server."
            )
        settings = self._settings
        retrieval = await RetrievalService(self._session, self._embedder, settings).search(
            question, document_ids, top_k=settings.answer_candidate_top_k
        )
        selection = select_context(retrieval.passages, settings, tuple(retrieval.analysis.terms))
        # Verification searches the whole text of each document that evidence came from. It is
        # read in the same snapshot as the passages, so their offsets agree.
        sources, pages = await self._load_documents(selection.selected)
        # End the read-only snapshot now: the LLM call can take seconds and must not hold a
        # database connection. Everything verification needs is already in memory.
        await self._session.rollback()

        notes = list(retrieval.notes)
        excluded = [
            ExcludedPassage(
                document_id=p.document_id,
                chunk_id=p.chunk_id,
                retrieval_rank=p.rank,
                reason=reason,
            )
            for p, reason in selection.excluded
        ]
        weak = sum(1 for _, reason in selection.excluded if reason.startswith("weak"))
        if weak:
            notes.append(f"{weak} weakly related passage(s) were not used as evidence.")

        if not selection.selected:
            notes.append(
                "No passage in the selected document(s) was relevant enough to answer from, so "
                "the model was not asked."
            )
            return AnswerResponse(
                question=question,
                answer=INSUFFICIENT_ANSWER,
                grounding="not_found",
                citations=[],
                evidence=[],
                excluded=excluded,
                coverage=retrieval.coverage,
                notes=notes,
            )

        evidence = {f"E{i}": p for i, p in enumerate(selection.selected, start=1)}
        result = await self._llm.complete(build_messages(retrieval.query, evidence))
        parsed = parse_output(result.text)
        # The exact "insufficient" reply is complete even if its closing tag is missing.
        truncated = parsed.truncated and not _is_insufficient(parsed.answer)
        _check_output(parsed.answer, truncated, [p.text for p in evidence.values()])
        notes.extend(parsed.problems)
        notes.append(_reading_note(evidence, retrieval.coverage))
        if any(p.quality.level == "low_quality_ocr" for p in evidence.values()):
            notes.append(
                "Some evidence comes from low-quality OCR pages: it may be misrecognised or "
                "incomplete, and relevant content on those pages may have been missed."
            )

        evidence_out = [
            EvidenceOut(
                evidence_id=evidence_id,
                document_id=p.document_id,
                document_filename=p.document_filename,
                chunk_id=p.chunk_id,
                section_path=p.section_path,
                char_start=p.char_start,
                char_end=p.char_end,
                page_start=p.page_start,
                page_end=p.page_end,
                text=p.text,
                retrieval_rank=p.rank,
                quality=p.quality,
            )
            for evidence_id, p in evidence.items()
        ]
        common = {
            "question": question,
            "evidence": evidence_out,
            "excluded": excluded,
            "coverage": retrieval.coverage,
            "model": result.model,
            "usage": result.usage,
        }

        if _is_insufficient(parsed.answer):
            return AnswerResponse(
                answer=INSUFFICIENT_ANSWER,
                grounding="not_found",
                citations=[],
                notes=notes,
                **common,
            )

        # CPU-bound (normalising a 150-page document takes ~0.2 s): keep it off the event loop.
        citations = await asyncio.to_thread(
            lambda: [self._verify(c, evidence, sources, pages) for c in parsed.citations]
        )
        grounding, grounding_notes = _grounding(
            parsed.markers, citations, uncited_statements(parsed.answer)
        )
        notes.extend(grounding_notes)
        for citation in citations:
            if citation.quality and citation.quality.level == "low_quality_ocr":
                notes.append(
                    f"Quote [{citation.n}] comes from a low-quality OCR page; check it against "
                    "the original document."
                )
        if grounding == "ungrounded":
            return AnswerResponse(
                answer=INSUFFICIENT_ANSWER,
                grounding="ungrounded",
                citations=citations,
                notes=notes,
                unsupported_draft=parsed.answer,
                **common,
            )
        return AnswerResponse(
            answer=parsed.answer, grounding=grounding, citations=citations, notes=notes, **common
        )

    # --- helpers --------------------------------------------------------------------------------

    async def _load_documents(
        self, passages: list[RetrievedPassage]
    ) -> tuple[dict[uuid.UUID, SourceDocument], PageRecords]:
        """Canonical text and page records of every document the evidence comes from."""
        document_ids = sorted({p.document_id for p in passages}, key=str)
        if not document_ids:
            return {}, {}
        texts = await self._session.execute(
            select(DocumentText.document_id, DocumentText.canonical_text).where(
                DocumentText.document_id.in_(document_ids)
            )
        )
        sources = {
            row.document_id: SourceDocument(row.document_id, row.canonical_text) for row in texts
        }
        rows = await self._session.execute(
            select(
                DocumentPage.document_id,
                DocumentPage.char_start,
                DocumentPage.char_end,
                DocumentPage.page_number,
                DocumentPage.extraction,
                DocumentPage.note,
                DocumentPage.ocr_confidence,
            )
            .where(DocumentPage.document_id.in_(document_ids))
            .order_by(DocumentPage.document_id, DocumentPage.page_number)
        )
        pages: PageRecords = defaultdict(list)
        for row in rows:
            pages[row.document_id].append(
                PageRecord(
                    row.char_start,
                    row.char_end,
                    row.page_number,
                    row.extraction,
                    row.note,
                    row.ocr_confidence,
                )
            )
        return sources, dict(pages)

    def _verify(
        self,
        citation: ProposedCitation,
        evidence: dict[str, RetrievedPassage],
        sources: dict[uuid.UUID, SourceDocument],
        pages: PageRecords,
    ) -> CitationOut:
        texts = {
            evidence_id: EvidenceText(evidence_id, p.document_id, p.char_start, p.text)
            for evidence_id, p in evidence.items()
        }
        check = verify_quote(citation.quote, citation.evidence_id, texts, sources)
        base = {
            "n": citation.n,
            "proposed_quote": citation.quote,
            "cited_evidence_id": citation.evidence_id,
        }
        if not check.verified:
            return CitationOut(status="unverified", reason=check.reason, **base)
        assert check.char_start is not None and check.char_end is not None
        cited = evidence[citation.evidence_id]  # a verified quote is always in its document
        records = pages.get(cited.document_id, [])
        page_start = _page_at(records, check.char_start)
        page_end = _page_at(records, check.char_end - 1)
        reason = _location_reason(check, citation.evidence_id, cited.document_filename)
        return CitationOut(
            status="verified",
            evidence_id=check.evidence_id,
            document_id=cited.document_id,
            document_filename=cited.document_filename,
            quote=check.matched_text,
            char_start=check.char_start,
            char_end=check.char_end,
            page_start=page_start,
            page_end=page_end,
            quality=_quote_quality(records, page_start, page_end),
            occurrence_count=check.occurrences,
            location_ambiguous=check.ambiguous,
            reason=reason,
            **base,
        )


def _location_reason(check: QuoteCheck, cited_evidence_id: str, filename: str) -> str | None:
    """Where a verified quote was found, when that is not simply (once) in the cited passage."""
    if check.tier == "document":
        reason = (
            f"Found elsewhere in {filename}, outside the passages the model was given (it cited "
            f"{cited_evidence_id})."
        )
        where = "outside those passages"
    elif check.tier == "other_evidence":
        reason = f"Found in {check.evidence_id}, not in the cited {cited_evidence_id}."
        where = "in the other passages the model was given"
    else:
        reason = None
        where = f"in {cited_evidence_id}"
    if check.ambiguous:
        repeat = (
            f"It occurs {check.tier_occurrences} times {where}; the location shown is the first, "
            "not necessarily the one meant."
        )
        reason = f"{reason} {repeat}" if reason else repeat
    return reason


def _page_at(records: list[PageRecord], position: int) -> int | None:
    """Page whose text range contains `position` (canonical offset), from stored page ranges."""
    if not records:
        return None
    index = bisect.bisect_right([r.char_start for r in records], position) - 1
    if index < 0:
        return None
    record = records[index]
    return record.page_number if record.char_start <= position < record.char_end else None


def _quote_quality(
    records: list[PageRecord], page_start: int | None, page_end: int | None
) -> PassageQuality:
    """How the pages the quote itself is on were read (a passage may span better pages)."""
    if page_start is None or page_end is None:
        return PassageQuality(level="not_applicable")  # DOCX: no pages
    own = [r for r in records if page_start <= r.page_number <= page_end]
    ocr = [r for r in own if r.extraction == "ocr"]
    low = [r.page_number for r in own if r.note == LOW_QUALITY_NOTE]
    confidences = [r.ocr_confidence for r in ocr if r.ocr_confidence is not None]
    return PassageQuality(
        level="low_quality_ocr" if low else "ocr" if ocr else "text",
        ocr_pages=[r.page_number for r in ocr],
        low_quality_pages=low,
        min_ocr_confidence=min(confidences) if confidences else None,
    )


def _is_insufficient(answer: str) -> bool:
    return normalize(answer).text == normalize(INSUFFICIENT_ANSWER).text


def _check_output(answer: str, truncated: bool, evidence_texts: list[str]) -> None:
    """Refuse model output that is incomplete or visibly corrupted: an honest error is better
    than presenting it, even if some of its quotes verify. Only the kind of problem is logged."""
    if truncated:
        logger.warning("LLM output rejected", extra={"problem": "truncated"})
        raise LLMOutputInvalidError(INCOMPLETE_OUTPUT)
    found = corruption(answer, evidence_texts)
    if found is not None:
        logger.warning(
            "LLM output rejected", extra={"problem": "mojibake", "sequence": ascii(found)}
        )
        raise LLMOutputInvalidError(MALFORMED_OUTPUT)
    fragment = degenerate_repetition(answer, evidence_texts)
    if fragment is not None:
        logger.warning(
            "LLM output rejected", extra={"problem": "repetition", "fragment": ascii(fragment[:60])}
        )
        raise LLMOutputInvalidError(MALFORMED_OUTPUT)


def _grounding(
    markers: list[int], citations: list[CitationOut], uncited: list[str] | None = None
) -> tuple[Grounding, list[str]]:
    verified = {c.n for c in citations if c.status == "verified"}
    if not verified:
        return "ungrounded", [
            "No quotation in the model's answer could be verified against the documents, so the "
            "answer is not presented."
        ]
    notes: list[str] = []
    unverified = sorted(c.n for c in citations if c.status != "verified")
    unsupported = sorted(set(markers) - verified)
    if unverified:
        notes.append(
            f"Citation(s) {', '.join(f'[{n}]' for n in unverified)} could not be verified; "
            "statements relying on them are not supported."
        )
    missing = sorted(set(unsupported) - set(unverified))
    if missing:
        notes.append(
            f"Marker(s) {', '.join(f'[{n}]' for n in missing)} have no citation; those "
            "statements are not supported."
        )
    if not markers:
        notes.append("The answer does not mark which statements its quotations support.")
    elif uncited:
        notes.append(
            f"{len(uncited)} statement(s) in the answer have no citation and are not supported: "
            + " | ".join(f'"{statement[:80]}"' for statement in uncited)
        )
    return ("grounded" if not notes else "partially_grounded"), notes


def _reading_note(evidence: dict[str, RetrievedPassage], coverage: list[DocumentCoverage]) -> str:
    searched = sum(c.chunks_searched for c in coverage)
    by_document: dict[str, set[int]] = defaultdict(set)
    for p in evidence.values():
        if p.page_start is not None and p.page_end is not None:
            by_document[p.document_filename].update(range(p.page_start, p.page_end + 1))
    pages = "; ".join(
        f"{name}: page(s) {', '.join(map(str, sorted(nums)))}"
        for name, nums in sorted(by_document.items())
    )
    return (
        f"The answer is based on {len(evidence)} excerpt(s) selected from {searched} searched "
        f"passage(s), not on a full reading of the document(s)" + (f" ({pages})." if pages else ".")
    )
