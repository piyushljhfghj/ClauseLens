"""Section detection: turn a flat list of blocks into a clause tree.

Signals, strongest first:
1. Explicit numbering at the start of a block: "8.", "8.2", "Section 8.2", "Article IV",
   "Schedule 2" (for DOCX, Word's auto-numbering has already been resolved into the text).
2. Heading styles (DOCX "Heading N") or visual emphasis (PDF: larger font, bold, ALL CAPS) on a
   short line without terminal punctuation.

Deliberately not sections: "(a)"/"(i)" sub-paragraphs (too granular), and numbers that are part
of a sentence ("30 days' notice", "100,000").
"""

import re
import statistics
from dataclasses import dataclass, field

from app.ingestion.model import Block

_ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
_DASHES = r"[\s.:\-\u2013\u2014]*"  # spaces, dots, colons, hyphens, en/em dashes

_ARTICLE = re.compile(rf"^(article)\s+([ivxlcdm]+|\d+)\b{_DASHES}(.*)$", re.I | re.S)
_ANNEX = re.compile(
    rf"^(schedule|annex|annexure|exhibit|appendix|attachment)\s+([a-z]|\d+|[ivx]+)\b{_DASHES}(.*)$",
    re.I | re.S,
)
_SECTION_WORD = re.compile(
    rf"^(section|clause)\s+(\d{{1,3}}(?:\.\d{{1,3}})*)\b{_DASHES}(.*)$", re.I | re.S
)
# "8." / "8)" / "8.2" / "8.2.1." then text starting with a capital, straight/curly quote or "(".
_NUMBERED = re.compile(
    r"^(\d{1,3}(?:\.\d{1,3})*)(\.|\))?[ \t]+([A-Z\"'(\u201c\u2018].*)$",
    re.S,
)
_SHORT_TITLE_PREFIX = re.compile(r"^([^.;:?]{3,80}?)([.?])\s+[A-Z(]")
_TERMINAL_PUNCTUATION = re.compile(r"[.;,:]$")  # headings do not end like sentences or lead-ins

MAX_HEADING_WORDS = 12
MAX_TITLE_PREFIX_WORDS = 8
MAX_QUESTION_TITLE_WORDS = 25
HEADING_FONT_RATIO = 1.15


@dataclass
class HeadingMatch:
    number: str | None
    number_normalized: str | None
    title: str | None
    level: int
    kind: str  # article | section | schedule | heading


@dataclass
class Section:
    index: int
    parent_index: int | None
    level: int
    kind: str
    number: str | None
    number_normalized: str | None
    title: str | None
    char_start: int
    char_end: int
    page_start: int | None
    page_end: int | None
    children: list[int] = field(default_factory=list)

    @property
    def label(self) -> str:
        parts = [p for p in (self.number, self.title) if p]
        return " ".join(parts) if parts else "(untitled)"


def roman_to_int(value: str) -> int | None:
    value = value.lower()
    if not value or any(ch not in _ROMAN for ch in value):
        return None
    total = 0
    for i, ch in enumerate(value):
        current = _ROMAN[ch]
        nxt = _ROMAN[value[i + 1]] if i + 1 < len(value) else 0
        total += -current if current < nxt else current
    return total


def _first_line(text: str) -> str:
    return text.split("\n", 1)[0].strip()


def _title_from_rest(rest: str, *, whole_block_is_line: bool) -> str | None:
    """'TERMINATION' -> title; 'Termination for Convenience. The Company may...' -> its first
    phrase; a long clause sentence -> no title."""
    rest = rest.strip()
    if not rest:
        return None
    prefix = _SHORT_TITLE_PREFIX.match(rest)
    if prefix and len(prefix.group(1).split()) <= MAX_TITLE_PREFIX_WORDS:
        # Keep a question mark ("How did this course help you grow?"), drop a full stop.
        return prefix.group(1).strip() + ("?" if prefix.group(2) == "?" else "")
    words = len(rest.split())
    if (
        whole_block_is_line
        and words <= MAX_HEADING_WORDS
        and not _TERMINAL_PUNCTUATION.search(rest)
    ):
        return rest
    if whole_block_is_line and words <= MAX_QUESTION_TITLE_WORDS and rest.endswith("?"):
        return rest  # a numbered question on its own line is its heading
    return None


def match_numbered_heading(text: str) -> HeadingMatch | None:
    """Explicit clause numbering at the start of `text` (first line considered)."""
    line = _first_line(text)
    single_line = "\n" not in text.strip()

    if m := _ARTICLE.match(line):
        raw = m.group(2)
        value = int(raw) if raw.isdigit() else roman_to_int(raw)
        if value is None:
            return None
        return HeadingMatch(
            number=f"{m.group(1).capitalize()} {raw.upper() if not raw.isdigit() else raw}",
            number_normalized=f"article {value}",
            title=_title_from_rest(m.group(3), whole_block_is_line=single_line),
            level=1,
            kind="article",
        )
    if m := _ANNEX.match(line):
        word, raw = m.group(1).capitalize(), m.group(2)
        if len(raw) == 1 and raw.isalpha() and raw.islower():
            return None  # "Exhibit a copy of ..." is a sentence, not "Exhibit A"
        return HeadingMatch(
            number=f"{word} {raw.upper()}",
            number_normalized=f"{word.lower()} {raw.lower()}",
            title=_title_from_rest(m.group(3), whole_block_is_line=single_line),
            level=1,
            kind="schedule",
        )
    if m := _SECTION_WORD.match(line):
        number = m.group(2)
        return HeadingMatch(
            number=number,
            number_normalized=number,
            title=_title_from_rest(m.group(3), whole_block_is_line=single_line),
            level=number.count(".") + 1,
            kind="section",
        )
    if m := _NUMBERED.match(line):
        number, terminator = m.group(1), m.group(2)
        parts = number.split(".")
        # A bare integer needs a terminator ("8." / "8)"): "30 Days" alone is too ambiguous.
        if len(parts) == 1 and not terminator:
            return None
        if int(parts[0]) == 0 or int(parts[0]) > 200:
            return None
        return HeadingMatch(
            number=number,
            number_normalized=number,
            title=_title_from_rest(m.group(3), whole_block_is_line=single_line),
            level=len(parts),
            kind="section",
        )
    return None


def looks_like_clause_start(line: str) -> bool:
    return match_numbered_heading(line) is not None


def _is_all_caps_heading(text: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    return len(letters) >= 4 and all(ch.isupper() for ch in letters) and len(text.split()) <= 8


def match_unnumbered_heading(block: Block, text: str, body_font_size: float) -> HeadingMatch | None:
    stripped = text.strip()
    if "\n" in stripped or not stripped:
        return None
    words = stripped.split()
    if len(words) > MAX_HEADING_WORDS or _TERMINAL_PUNCTUATION.search(stripped):
        return None
    if not any(ch.isalpha() for ch in stripped):
        return None
    if block.heading_level is not None:
        return HeadingMatch(None, None, stripped, block.heading_level, "heading")
    if block.ocr:
        # OCR output has no font information, and recognition noise is often short and
        # upper-case ("AAA CY"); only explicit numbering is trusted to start a section there.
        return None
    emphasised = (
        block.bold
        or (body_font_size > 0 and block.font_size >= body_font_size * HEADING_FONT_RATIO)
        or _is_all_caps_heading(stripped)
    )
    return HeadingMatch(None, None, stripped, 1, "heading") if emphasised else None


def body_font_size(blocks: list[Block]) -> float:
    """Median font size weighted by text length: the size most of the text is set in."""
    weighted = [
        b.font_size
        for b in blocks
        if b.font_size > 0
        for _ in range(max(1, (b.char_end - b.char_start) // 50))
    ]
    return statistics.median(weighted) if weighted else 0.0


def _classify(blocks: list[Block], canonical_text: str) -> list[HeadingMatch | None]:
    """A heading match (or None) per block.

    A run of consecutive unnumbered headings with no body text between them is a title block
    (cover page, letterhead), not a series of empty sections. Within such a run, lines that are
    headings only because of emphasis (bold / caps / font size) are not sections; Word heading
    styles and explicit numbering are always kept.
    """
    body_size = body_font_size(blocks)
    matches: list[HeadingMatch | None] = []
    for block in blocks:
        if block.kind == "table_row":
            matches.append(None)
            continue
        text = canonical_text[block.char_start : block.char_end]
        matches.append(
            match_numbered_heading(text) or match_unnumbered_heading(block, text, body_size)
        )

    def unnumbered(i: int) -> bool:
        return 0 <= i < len(matches) and matches[i] is not None and matches[i].kind == "heading"  # type: ignore[union-attr]

    for i, block in enumerate(blocks):
        emphasis_only = unnumbered(i) and block.heading_level is None
        if emphasis_only and unnumbered(i + 1):
            matches[i] = None
    return matches


def detect_sections(blocks: list[Block], canonical_text: str) -> list[Section]:
    """Classify heading blocks and build the section tree. Mutates blocks: heading blocks get
    kind="heading", and every block gets `section_index` (deepest enclosing section)."""
    sections: list[Section] = []
    stack: list[int] = []

    for block, match in zip(blocks, _classify(blocks, canonical_text), strict=True):
        if match is not None:
            # A block is a heading if it is only a heading line: an unnumbered heading, or a
            # single-line numbered heading with a title ("8. TERMINATION"). A numbered clause
            # with body text ("8.2 The Company may...") opens a section but stays a paragraph.
            block_text = canonical_text[block.char_start : block.char_end].strip()
            single_line = len(block_text.splitlines()) == 1
            if match.kind == "heading" or (single_line and match.title):
                block.kind = "heading"
            while stack and sections[stack[-1]].level >= match.level:
                stack.pop()
            parent = stack[-1] if stack else None
            section = Section(
                index=len(sections),
                parent_index=parent,
                level=match.level,
                kind=match.kind,
                number=match.number,
                number_normalized=match.number_normalized,
                title=match.title,
                char_start=block.char_start,
                char_end=block.char_end,
                page_start=block.page_start,
                page_end=block.page_end,
            )
            if parent is not None:
                sections[parent].children.append(section.index)
            sections.append(section)
            stack.append(section.index)
        block.section_index = stack[-1] if stack else None
        # Every open section extends to cover this block.
        for open_index in stack:
            sections[open_index].char_end = block.char_end
            if block.page_end is not None:
                sections[open_index].page_end = block.page_end
    return sections


def section_path(sections: list[Section], index: int | None) -> str:
    """'8 Termination > 8.2 Termination for Convenience'."""
    labels: list[str] = []
    while index is not None:
        labels.append(sections[index].label)
        index = sections[index].parent_index
    return " > ".join(reversed(labels))


def root_index(sections: list[Section], index: int | None) -> int | None:
    while index is not None and sections[index].parent_index is not None:
        index = sections[index].parent_index
    return index


_LEADING_LABEL = re.compile(
    r"^(?:(?:article|section|clause|schedule|annex|annexure|exhibit|appendix|attachment)\s+)?"
    # the label itself, then an optional separator: "8.2 ", "Article IV - ", "Section 12.3: "
    r"[\w.]+[.)]*\s*(?:[:\-" + chr(0x2013) + chr(0x2014) + r"]\s*)?",
    re.I,
)
PREVIEW_CHARS = 80
_ELLIPSIS = chr(0x2026)
_NEWLINE = chr(10)


def section_preview(section_text: str, *, has_number: bool) -> str:
    """The opening words of a section, for display when it has no title: '5' alone says nothing,
    '5 Enlist your semester achievements and...' does. Never stored as a title."""
    line = section_text.strip().split(_NEWLINE, 1)[0].strip()
    if has_number:
        line = _LEADING_LABEL.sub("", line, count=1)
    if len(line) <= PREVIEW_CHARS:
        return line
    cut = line.rfind(" ", 0, PREVIEW_CHARS)
    return line[: cut if cut > 20 else PREVIEW_CHARS].rstrip(" ,;:") + _ELLIPSIS
