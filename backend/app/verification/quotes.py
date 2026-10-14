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
- Extraction differences, tried only when the exact form occurs nowhere in the document: a
  hyphen at a line break kept as a real hyphen ("non-compete" for "non-" + line break +
  "compete"), then spacing differences, conservatively (see `_spacing_spans`).

Not yet (later Phase 6 tasks): highlight rectangles. Elided quotes ("...") stay unverified.
"""

import re
import uuid
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import pairwise

from app.verification.normalize import Normalized, normalize

MIN_QUOTE_CHARS = 8
MIN_QUOTE_WORDS = 2
MIN_SPACING_CHARS = 25  # non-space characters, for spacing-tolerant matching
MAX_WORD_SPLITS = 2
# Short common words: a split that leaves one of these as a piece reads as real separate words
# ("the rapist" / "therapist", "not able" / "notable"), so it is never treated as a split word.
_FUNCTION_WORDS = frozenset(
    "a an and any are as at be by for from he her his i if in into is it its no nor not of off "
    "on or our out so than that the their them they this to up us was we were who with you".split()
)
_WORD = re.compile(r"\w+")
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
    """A document's full canonical text; each derived form is computed once, on first use."""

    def __init__(self, document_id: uuid.UUID, text: str) -> None:
        self.document_id = document_id
        self.text = text
        self._normalized: Normalized | None = None
        self._hyphens_kept: Normalized | None = None
        self._hyphens_checked = False
        self._compact: tuple[str, list[int]] | None = None
        self._tokens: Counter[str] | None = None

    @property
    def normalized(self) -> Normalized:
        if self._normalized is None:
            self._normalized = normalize(self.text)
        return self._normalized

    @property
    def line_hyphens_kept(self) -> Normalized | None:
        """Normalised with hyphens at line breaks kept ("non-compete"); None if that changes
        nothing (the document has no hyphen at a line break)."""
        if not self._hyphens_checked:
            variant = normalize(self.text, keep_line_hyphens=True)
            self._hyphens_kept = variant if variant.text != self.normalized.text else None
            self._hyphens_checked = True
        return self._hyphens_kept

    @property
    def compact(self) -> tuple[str, list[int]]:
        """The normalised text without spaces, and each character's normalised index."""
        if self._compact is None:
            self._compact = _compact(self.normalized.text)
        return self._compact

    @property
    def tokens(self) -> Counter[str]:
        """How often each word (normalised) occurs in the document."""
        if self._tokens is None:
            self._tokens = Counter(_WORD.findall(self.normalized.text))
        return self._tokens


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
    # How it matched: "exact" (after normalisation), "line_hyphen" (a hyphen at a line break kept
    # as a real hyphen), or "spacing" (extraction spacing differences tolerated). Counts and
    # ambiguity refer to occurrences found the same way.
    match_type: str | None = None


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


def _whole_word_spans(
    norm: Normalized, original: str, needle: str
) -> tuple[list[tuple[int, int]], bool]:
    """Canonical ranges of `needle` in `norm` that start and end on word boundaries, and whether
    any occurrence was rejected for starting or ending inside a word."""
    found, partial = [], False
    for hit in _find_all(norm.text, needle):
        end = hit + len(needle)
        if _inside_word(norm, original, hit) or _inside_word(norm, original, end):
            partial = True
            continue
        found.append(norm.original_span(hit, end))
    return found, partial


def _compact(text: str) -> tuple[str, list[int]]:
    """`text` (normalised: single spaces only) without spaces, and each character's index."""
    positions = [i for i, ch in enumerate(text) if ch != " "]
    return "".join(text[i] for i in positions), positions


def _spacing_spans(source: SourceDocument, needle: str) -> tuple[list[tuple[int, int]], bool]:
    """Occurrences that differ from `needle` only in spacing the extraction plausibly caused.

    Conservative by design. The quote must have at least MIN_SPACING_CHARS non-space characters.
    Spacing next to punctuation may differ ("Agreement ." / "Agreement."), except around a
    separator inside a number ("8.2" / "8. 2"). Between two letters or digits, the only accepted
    difference is a space the source has inside a quoted word (a split word, "Agree ment"), at
    most MAX_WORD_SPLITS of them, each passing `_plausible_split`. Never accepted: a split or
    join between two digits ("8 2" / "82"), or source text that runs two quoted words together
    (that could equally be different words).
    """
    compact_quote, quote_map = _compact(needle)
    if len(compact_quote) < MIN_SPACING_CHARS:
        return [], False
    norm = source.normalized
    compact_text, text_map = source.compact
    found, partial = [], False
    for hit in _find_all(compact_text, compact_quote):
        if not _spacing_differences_allowed(source, needle, quote_map, text_map, hit):
            continue
        start, end = text_map[hit], text_map[hit + len(compact_quote) - 1] + 1
        if _inside_word(norm, source.text, start) or _inside_word(norm, source.text, end):
            partial = True
            continue
        found.append(norm.original_span(start, end))
    return found, partial


def _spacing_differences_allowed(
    source: SourceDocument, needle: str, quote_map: list[int], text_map: list[int], hit: int
) -> bool:
    splits: list[int] = []  # positions in `needle` where the source has a space and it does not
    for k in range(1, len(quote_map)):
        quote_space = quote_map[k] - quote_map[k - 1] > 1
        source_space = text_map[hit + k] - text_map[hit + k - 1] > 1
        if quote_space == source_space:
            continue
        before, after = needle[quote_map[k - 1]], needle[quote_map[k]]
        if _number_separator(needle, quote_map, k - 1) or _number_separator(needle, quote_map, k):
            return False  # "8.2" vs "8. 2", "100,000" vs "100, 000": could be different numbers
        if not (before.isalnum() and after.isalnum()):
            continue  # spacing around punctuation does not change the words
        if before.isdigit() and after.isdigit():
            return False  # never split or join a number
        if quote_space:
            return False  # the source runs two quoted words together
        splits.append(quote_map[k])
    if len(splits) > MAX_WORD_SPLITS:
        return False
    words: dict[tuple[int, int], list[int]] = {}
    for position in splits:
        start = position
        while start > 0 and needle[start - 1].isalnum():
            start -= 1
        end = position
        while end < len(needle) and needle[end].isalnum():
            end += 1
        words.setdefault((start, end), []).append(position)
    return all(
        _plausible_split(source.tokens, needle[start:end], [p - start for p in cuts])
        for (start, end), cuts in words.items()
    )


def _number_separator(needle: str, quote_map: list[int], k: int) -> bool:
    """Whether compact character k is a "." or "," between two digits."""
    if not 0 < k < len(quote_map) - 1:
        return False
    return (
        needle[quote_map[k]] in ".,"
        and needle[quote_map[k - 1]].isdigit()
        and needle[quote_map[k + 1]].isdigit()
    )


def _plausible_split(tokens: Counter[str], word: str, cuts: list[int]) -> bool:
    """Whether a quoted `word` that the source has in pieces is an extraction split rather than
    genuinely different words. All of: the document uses the word unsplit elsewhere; at least
    one piece never occurs on its own as a word in the document ("Agree ment": "ment"); and no
    piece is a common short word ("the rapist" is two words, not a split "therapist")."""
    bounds = [0, *cuts, len(word)]
    pieces = [word[a:b] for a, b in pairwise(bounds)]
    return (
        tokens[word] >= 1
        and any(tokens[piece] <= 1 for piece in pieces)
        and not any(piece in _FUNCTION_WORDS for piece in pieces)
    )


def _locate(
    source: SourceDocument, needles: list[str]
) -> tuple[list[list[tuple[int, int]]], bool, str | None]:
    """Whole-word occurrences of each needle, from the first strategy that finds any: exact
    (after normalisation), then with line-break hyphens kept, then tolerant of extraction
    spacing. A later strategy never overrides an earlier one's match anywhere in the document.
    Returns the spans per needle, whether a partial-word match was rejected, and the strategy."""
    strategies: list[tuple[str, Callable[[str], tuple[list[tuple[int, int]], bool]]]] = [
        ("exact", lambda n: _whole_word_spans(source.normalized, source.text, n))
    ]
    hyphens_kept = source.line_hyphens_kept
    if hyphens_kept is not None:
        strategies.append(
            ("line_hyphen", lambda n: _whole_word_spans(hyphens_kept, source.text, n))
        )
    strategies.append(("spacing", lambda n: _spacing_spans(source, n)))
    partial = False
    for kind, find in strategies:
        results = [find(needle) for needle in needles]
        partial = partial or any(p for _, p in results)
        spans = [found for found, _ in results]
        if any(spans):
            return spans, partial, kind
    return [[] for _ in needles], partial, None


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
    # Quotation marks at the edges may be the model's wrapping or part of the source text
    # (a quoted title). Try the quote with them first, so a match covers the source's own marks
    # and its offsets are exact, then without them.
    needles = list(dict.fromkeys([normalize(quote.strip()).text, normalize(text).text]))

    spans, partial_word, match_type = _locate(source, needles)
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
        match_type=match_type,
    )
