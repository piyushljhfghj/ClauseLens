"""Verify a quotation proposed by the LLM against the canonical text of the evidence passages.

The LLM is trusted for nothing: not the quote, not which passage it came from, not positions.
A quote is VERIFIED only if its normalised form occurs verbatim, as whole words, in the
normalised canonical text of *the document of the passage it cites* (never another selected
document); the reported location is then computed from our stored text.

- Whole document: the cited document's full canonical text is searched, not only the passages
  the model was given. An occurrence inside the cited passage is preferred, then inside another
  passage of that document, then anywhere else; a match outside every passage carries no
  evidence id, so it is never presented as coming from the model's evidence.
- Whole words: a match may not start or end inside a word or number ("s liability" inside
  "Supplier's liability", "AED 100" inside "AED 100,000", "S. law" inside "U.S. law").
- Repeated text: when the tier the match was found in holds several occurrences, the location
  is flagged ambiguous with the document-wide occurrence count.

Not yet (later Phase 6 tasks): whitespace-insensitive matching, highlight rectangles. Elided
quotes ("...") stay unverified.
"""

import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass

from app.verification.normalize import Normalized, normalize

MIN_QUOTE_CHARS = 8
MIN_QUOTE_WORDS = 2
_ELLIPSIS = re.compile(r"\.\.\.|\u2026")
_HYPHENS = {"-", chr(0x2010), chr(0x2011)}  # not dashes: those separate words
_EDGE = " \t\r\n\"'`\u2018\u2019\u201c\u201d"


@dataclass(frozen=True)
class EvidenceText:
    evidence_id: str
    document_id: uuid.UUID
    char_start: int  # absolute canonical offset of this passage
    text: str  # == canonical_text[char_start:char_end]


class SourceDocument:
    """A document's full canonical text; normalised once, on first use."""

    def __init__(self, document_id: uuid.UUID, text: str) -> None:
        self.document_id = document_id
        self.text = text
        self._normalized: Normalized | None = None

    @property
    def normalized(self) -> Normalized:
        if self._normalized is None:
            self._normalized = normalize(self.text)
        return self._normalized


@dataclass(frozen=True)
class QuoteCheck:
    verified: bool
    reason: str | None = None  # why not verified
    evidence_id: str | None = None  # passage containing the match; None if outside all evidence
    char_start: int | None = None  # absolute canonical range of the matched text
    char_end: int | None = None
    matched_text: str | None = None  # exact canonical text (what the UI shows)
    occurrences: int = 0  # whole-word occurrences in the cited document
    # Where it was found (the cited passage, the other passages given to the model, or the rest
    # of the document) it occurs more than once: the location is the first of several equally
    # plausible ones, not a definitive one.
    ambiguous: bool = False
    tier: str | None = None  # "cited" | "other_evidence" | "document"
    tier_occurrences: int = 0  # occurrences in that tier


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


def _joiner(norm: Normalized, original: str, k: int) -> bool:
    """Whether normalised character k joins the characters on either side into one word:
    an apostrophe ("Supplier's"), a hyphen but not a dash ("non-compete" is one word; an em
    dash between two words is not; both normalise to "-", so the original character decides),
    a thousands/decimal separator between digits ("100,000", "8.2"), or a full stop between
    two letters with no space, as in an abbreviation ("U.S.", "A.P.J."); a sentence end is
    followed by whitespace, so it still separates words."""
    t = norm.text
    if not 0 < k < len(t) - 1:
        return False
    before, ch, after = t[k - 1], t[k], t[k + 1]
    if ch == "'":
        return before.isalnum() and after.isalnum()
    if ch == "-":
        return before.isalnum() and after.isalnum() and original[norm.offsets[k]] in _HYPHENS
    if ch == ".":
        return (before.isdigit() and after.isdigit()) or (before.isalpha() and after.isalpha())
    if ch == ",":
        return before.isdigit() and after.isdigit()
    return False


def _inside_word(norm: Normalized, original: str, i: int) -> bool:
    """Whether the boundary before normalised position i falls inside a word or number.
    Line-break hyphenation is already joined by normalisation ("termi-\\nnation")."""
    t = norm.text
    if i <= 0 or i >= len(t):
        return False
    if t[i - 1].isalnum() and t[i].isalnum():
        return True
    return _joiner(norm, original, i - 1) or _joiner(norm, original, i)


def verify_quote(
    quote: str,
    cited_evidence_id: str,
    evidence: dict[str, EvidenceText],
    documents: Mapping[uuid.UUID, SourceDocument],
) -> QuoteCheck:
    cited = evidence.get(cited_evidence_id)
    if cited is None:
        return QuoteCheck(False, f"Evidence '{cited_evidence_id}' was not provided to the model.")
    text, problem = clean_quote(quote)
    if problem:
        return QuoteCheck(False, problem)
    # Only the cited passage's own document is searched: never another selected document.
    source = documents.get(cited.document_id)
    if source is None:
        return QuoteCheck(False, "The cited document's text is not available for verification.")
    norm = source.normalized
    # Quotation marks at the edges may be the model's wrapping or part of the source text
    # (a quoted title). Try the quote with them first, so a match covers the source's own marks
    # and its offsets are exact, then without them.
    needles = list(dict.fromkeys([normalize(quote.strip()).text, normalize(text).text]))

    # Every whole-word occurrence in the whole document, as canonical ranges.
    spans: list[list[tuple[int, int]]] = []
    partial_word = False
    for needle in needles:
        found = []
        for hit in _find_all(norm.text, needle):
            end = hit + len(needle)
            if _inside_word(norm, source.text, hit) or _inside_word(norm, source.text, end):
                partial_word = True
                continue
            found.append(norm.original_span(hit, end))
        spans.append(found)
    if not any(spans):
        if partial_word:
            return QuoteCheck(
                False,
                "The quoted text only occurs as part of longer words in the cited document "
                "(it starts or ends in the middle of a word).",
            )
        return QuoteCheck(False, "The quoted text was not found in the cited document.")

    # Tiers: the cited passage, then the other passages of the same document the model was
    # given, then anywhere else in that document. Within a tier the quote with its edge quotation
    # marks beats the quote without them. A tier holding several occurrences is ambiguous: the
    # first is reported (passage order, then document order) and flagged, never as definitive.
    passages = [cited] + [
        e for e in evidence.values() if e.document_id == cited.document_id and e is not cited
    ]

    def placement(start: int, end: int) -> tuple[int, int]:
        """(tier, passage index) of a match lying wholly inside a passage, else the document."""
        for index, passage in enumerate(passages):
            if passage.char_start <= start and end <= passage.char_start + len(passage.text):
                return (0 if index == 0 else 1), index
        return 2, len(passages)

    candidates = list(
        (*placement(start, end), needle_index, start, end)
        for needle_index, found in enumerate(spans)
        for start, end in found
    )
    best_tier, passage_index, needle_index, start, end = min(
        candidates, key=lambda c: (c[0], c[2], c[1], c[3])
    )
    rivals = [c for c in candidates if c[0] == best_tier and c[2] == needle_index]
    return QuoteCheck(
        True,
        evidence_id=passages[passage_index].evidence_id if best_tier < 2 else None,
        char_start=start,
        char_end=end,
        matched_text=source.text[start:end],
        occurrences=len(spans[needle_index]),
        ambiguous=len(rivals) > 1,
        tier=("cited", "other_evidence", "document")[best_tier],
        tier_occurrences=len(rivals),
    )
