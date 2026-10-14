"""Verify a quotation proposed by the LLM against the canonical text of the evidence passages.

The LLM is trusted for nothing: not the quote, not which passage it came from, not positions.
A quote is VERIFIED only if its normalised form occurs verbatim in the normalised canonical text
of an evidence passage *from the same document as the passage it cites*; the reported location
is then computed from our stored text.

Phase 5 scope: search within the passages that were given to the model. Searching the whole
document, elided quotes ("..."), near-miss diagnostics and highlight rectangles are Phase 6.
"""

import re
import uuid
from dataclasses import dataclass

from app.verification.normalize import normalize

MIN_QUOTE_CHARS = 8
MIN_QUOTE_WORDS = 2
_ELLIPSIS = re.compile(r"\.\.\.|\u2026")
_EDGE = " \t\r\n\"'`\u2018\u2019\u201c\u201d"


@dataclass(frozen=True)
class EvidenceText:
    evidence_id: str
    document_id: uuid.UUID
    char_start: int  # absolute canonical offset of this passage
    text: str  # == canonical_text[char_start:char_end]


@dataclass(frozen=True)
class QuoteCheck:
    verified: bool
    reason: str | None = None  # why not verified
    evidence_id: str | None = None  # passage where the quote was actually found
    char_start: int | None = None  # absolute canonical range of the matched text
    char_end: int | None = None
    matched_text: str | None = None  # exact canonical text (what the UI shows)
    occurrences: int = 0  # occurrences within that passage


def clean_quote(quote: str) -> tuple[str, str | None]:
    """Strip wrapping quotes/whitespace and leading/trailing ellipses (models often mark a quote
    that starts mid-sentence that way; the rest must still be verbatim and contiguous)."""
    text = quote.strip(_EDGE)
    text = re.sub(r"^(\.\.\.|\u2026)\s*", "", text)
    text = re.sub(r"\s*(\.\.\.|\u2026)$", "", text).strip(_EDGE)
    if _ELLIPSIS.search(text):
        return text, "The quote omits words ('...'); only contiguous verbatim quotes are verified."
    normalised = normalize(text).text
    if len(normalised) < MIN_QUOTE_CHARS or len(normalised.split()) < MIN_QUOTE_WORDS:
        return text, "The quote is too short to identify a passage reliably."
    return text, None


def _find_all(haystack: str, needle: str) -> list[int]:
    found, start = [], haystack.find(needle)
    while start != -1:
        found.append(start)
        start = haystack.find(needle, start + 1)
    return found


def verify_quote(
    quote: str, cited_evidence_id: str, evidence: dict[str, EvidenceText]
) -> QuoteCheck:
    cited = evidence.get(cited_evidence_id)
    if cited is None:
        return QuoteCheck(False, f"Evidence '{cited_evidence_id}' was not provided to the model.")
    text, problem = clean_quote(quote)
    if problem:
        return QuoteCheck(False, problem)
    needle = normalize(text).text

    # The cited passage first, then the other passages of the SAME document (the model may cite
    # the wrong label). Never another document: each quote is verified against its own document.
    candidates = [cited] + [
        e for e in evidence.values() if e.document_id == cited.document_id and e is not cited
    ]
    for passage in candidates:
        normalised = normalize(passage.text)
        hits = _find_all(normalised.text, needle)
        if hits:
            start, end = normalised.original_span(hits[0], hits[0] + len(needle))
            return QuoteCheck(
                True,
                evidence_id=passage.evidence_id,
                char_start=passage.char_start + start,
                char_end=passage.char_start + end,
                matched_text=passage.text[start:end],
                occurrences=len(hits),
            )
    return QuoteCheck(
        False, "The quoted text was not found in the cited document's evidence passages."
    )
