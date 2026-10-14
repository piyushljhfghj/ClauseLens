"""Grounded question answering.

  retrieve (Phase 4) -> select strong evidence within a token budget -> LLM with evidence only
  -> parse proposed citations -> verify every quote against stored canonical text
  -> grounding verdict -> answer, or an explicit "insufficient support" statement

Nothing the model says about sources is trusted. An answer with no verified quotation is never
presented as an answer: the response states that support was insufficient and keeps the
model's draft separately for transparency.
"""

import bisect
import uuid
from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.answering.context import select_context
from app.answering.parser import ProposedCitation, parse_output
from app.answering.prompt import INSUFFICIENT_ANSWER, build_messages
from app.core.config import Settings
from app.db.models import DocumentPage
from app.embeddings import Embedder
from app.llm import LLMClient, LLMNotConfiguredError
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
from app.verification.quotes import EvidenceText, verify_quote

PageRanges = dict[uuid.UUID, list[tuple[int, int, int]]]  # doc -> [(char_start, char_end, page)]


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
        pages = await self._page_ranges(selection.selected)
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

        citations = [self._verify(c, evidence, pages) for c in parsed.citations]
        grounding, grounding_notes = _grounding(parsed.markers, citations)
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

    async def _page_ranges(self, passages: list[RetrievedPassage]) -> PageRanges:
        wanted: dict[uuid.UUID, set[int]] = defaultdict(set)
        for p in passages:
            if p.page_start is not None and p.page_end is not None:
                wanted[p.document_id].update(range(p.page_start, p.page_end + 1))
        ranges: PageRanges = {}
        for document_id, numbers in wanted.items():
            rows = await self._session.execute(
                select(DocumentPage.char_start, DocumentPage.char_end, DocumentPage.page_number)
                .where(
                    DocumentPage.document_id == document_id,
                    DocumentPage.page_number.in_(sorted(numbers)),
                )
                .order_by(DocumentPage.page_number)
            )
            ranges[document_id] = [(r.char_start, r.char_end, r.page_number) for r in rows]
        return ranges

    def _verify(
        self,
        citation: ProposedCitation,
        evidence: dict[str, RetrievedPassage],
        pages: PageRanges,
    ) -> CitationOut:
        texts = {
            evidence_id: EvidenceText(evidence_id, p.document_id, p.char_start, p.text)
            for evidence_id, p in evidence.items()
        }
        check = verify_quote(citation.quote, citation.evidence_id, texts)
        base = {
            "n": citation.n,
            "proposed_quote": citation.quote,
            "cited_evidence_id": citation.evidence_id,
        }
        if not check.verified:
            return CitationOut(status="unverified", reason=check.reason, **base)
        assert check.evidence_id and check.char_start is not None and check.char_end is not None
        passage = evidence[check.evidence_id]
        page_start = _page_at(pages.get(passage.document_id, []), check.char_start)
        page_end = _page_at(pages.get(passage.document_id, []), check.char_end - 1)
        return CitationOut(
            status="verified",
            evidence_id=check.evidence_id,
            document_id=passage.document_id,
            document_filename=passage.document_filename,
            quote=check.matched_text,
            char_start=check.char_start,
            char_end=check.char_end,
            page_start=page_start,
            page_end=page_end,
            quality=_quote_quality(passage.quality, page_start, page_end),
            reason=(
                f"Found in {check.evidence_id}, not in the cited {citation.evidence_id}."
                if check.evidence_id != citation.evidence_id
                else None
            ),
            **base,
        )


def _page_at(ranges: list[tuple[int, int, int]], position: int) -> int | None:
    """Page whose text range contains `position` (canonical offset), from stored page ranges."""
    if not ranges:
        return None
    starts = [r[0] for r in ranges]
    index = bisect.bisect_right(starts, position) - 1
    if index < 0:
        return None
    start, end, page = ranges[index]
    return page if start <= position < end else None


def _quote_quality(
    passage_quality: PassageQuality, page_start: int | None, page_end: int | None
) -> PassageQuality:
    """The quality of the pages the quote itself is on (a passage may span better pages)."""
    if page_start is None or page_end is None:
        return passage_quality
    on = set(range(page_start, page_end + 1))
    low = [p for p in passage_quality.low_quality_pages if p in on]
    ocr = [p for p in passage_quality.ocr_pages if p in on]
    level = "low_quality_ocr" if low else "ocr" if ocr else "text"
    return PassageQuality(
        level=level,
        ocr_pages=ocr,
        low_quality_pages=low,
        min_ocr_confidence=passage_quality.min_ocr_confidence if ocr else None,
    )


def _is_insufficient(answer: str) -> bool:
    return normalize(answer).text == normalize(INSUFFICIENT_ANSWER).text


def _grounding(markers: list[int], citations: list[CitationOut]) -> tuple[Grounding, list[str]]:
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
