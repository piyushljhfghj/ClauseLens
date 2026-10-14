"""Deterministic query analysis (no LLM): what to look for, lexically and structurally.

- `terms`: BM25 terms, produced by the same tokenizer as the index, minus stopwords, so that
  "what is the governing law" searches for "governing" and "law", not for "the".
- `phrases`: text in double quotes, matched as an exact (whitespace-insensitive) phrase.
- `section_refs`: clause references ("clause 8.2", "Article IV", "Schedule 2", bare "8.2") in the
  same normalised form as `sections.number_normalized`, for a direct section lookup.
"""

import re
from dataclasses import dataclass, field

from app.ingestion.structure import roman_to_int
from app.text.tokenize import lexical_tokens, prepare_for_indexing

# Function words and question words that carry no topical signal. Deliberately short: legal
# words such as "shall", "may", "not", "any" are kept because they distinguish clauses.
STOPWORDS = frozenset(
    """
    a an and are as at be been being but by can could did do does doing for from had has have
    having he her hers him his how i if in into is it its itself me my of on or our ours she so
    than that the their theirs them then there these they this those to too us was we were what
    when where which while who whom why will with would you your yours about tell show find
    give explain describe please document contract agreement say says said mention mentions
    regarding state states stated
    """.split()
)

_QUOTED = re.compile(r"[\"“”]([^\"“”]{2,200})[\"“”]")
_KEYWORD_REF = re.compile(
    r"\b(section|clause|article|schedule|annex|annexure|exhibit|appendix|attachment)\s+"
    r"(\d{1,3}(?:\.\d{1,3})*|[ivxlcdm]{1,7}|[a-z])\b",
    re.I,
)
_DOTTED_NUMBER = re.compile(r"(?<![\w.,])(\d{1,3}(?:\.\d{1,3})+)(?![\w,]|\.\d)")
_ANNEX_WORDS = {"schedule", "annex", "annexure", "exhibit", "appendix", "attachment"}


@dataclass(frozen=True)
class AnalyzedQuery:
    text: str
    terms: tuple[str, ...]
    phrases: tuple[tuple[str, ...], ...]  # each phrase as its token sequence
    section_refs: tuple[str, ...]
    has_content: bool  # False for "???" or only stopwords: nothing meaningful to search for
    notes: list[str] = field(default_factory=list)


def _section_ref(keyword: str, value: str) -> str | None:
    keyword, value = keyword.lower(), value.lower()
    if keyword == "article":
        number = int(value) if value.isdigit() else roman_to_int(value)
        return f"article {number}" if number else None
    if keyword in _ANNEX_WORDS:
        return f"{keyword} {value}"
    # section / clause: only numeric references map to clause numbers
    return value if value[0].isdigit() else None


MAX_PHRASES = 5

# A bare dotted number is read as a clause reference ("what does 8.2 say") unless context shows
# it is a quantity or a version. Explicit references ("clause 2.5") are always clause numbers.
_UNIT_AFTER = re.compile(
    r"\s*(%|per\s*cent\b|percent\b|million\b|billion\b|thousand\b|bn\b|m\b|k\b|x\b|times\b|"
    r"days?\b|weeks?\b|months?\b|years?\b|hours?\b|kg\b|km\b|metres?\b|meters?\b)",
    re.I,
)
_CONTEXT_BEFORE = re.compile(
    r"(\b(aed|usd|eur|gbp|inr|sar|version|v|rate|ratio|times)|[$€£])\s*$", re.I
)


def _is_quantity(text: str, match: re.Match[str]) -> bool:
    return bool(
        _UNIT_AFTER.match(text, match.end()) or _CONTEXT_BEFORE.search(text[: match.start()])
    )


def analyze_query(text: str) -> AnalyzedQuery:
    cleaned = " ".join(text.split())
    prepared = prepare_for_indexing(cleaned)

    notes: list[str] = []
    phrases: list[tuple[str, ...]] = []
    for match in _QUOTED.finditer(prepared):
        tokens = tuple(lexical_tokens(match.group(1)))
        if tokens and tokens not in phrases:
            phrases.append(tokens)
    if len(phrases) > MAX_PHRASES:
        # Each phrase is one SQL scan: bound the work a single request can cause.
        notes.append(f"Only the first {MAX_PHRASES} quoted phrases were matched exactly.")
        phrases = phrases[:MAX_PHRASES]

    refs: list[str] = []
    for match in _KEYWORD_REF.finditer(prepared):
        ref = _section_ref(match.group(1), match.group(2))
        if ref and ref not in refs:
            refs.append(ref)
    for match in _DOTTED_NUMBER.finditer(prepared):
        if _is_quantity(prepared, match):
            continue  # "2.5 million", "3.14 percent", "version 1.5": a number, not a clause
        if match.group(1) not in refs:
            refs.append(match.group(1))

    terms: list[str] = []
    for token in lexical_tokens(prepared):
        if token not in STOPWORDS and token not in terms:
            terms.append(token)

    has_content = bool(terms or phrases or refs)
    if not has_content:
        notes.append("The query contains no searchable terms.")
    return AnalyzedQuery(
        text=cleaned,
        terms=tuple(terms),
        phrases=tuple(phrases),
        section_refs=tuple(refs),
        has_content=has_content,
        notes=notes,
    )
