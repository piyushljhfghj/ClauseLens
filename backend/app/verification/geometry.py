"""Highlight rectangles for a verified quotation in a PDF.

Geometry comes only from extraction: each stored page lists its words as
`[char_start, char_end, x0, y0, x1, y1]` (canonical offsets; displayed-page PDF points, origin
top-left, y downwards; OCR words use the same space). A verified canonical range is mapped to the
words it overlaps, and words on the same text line are merged into one rectangle per line, on
each page the quote touches. Nothing the model said about pages or positions is used.

Lines are read from the canonical text itself: words on one line are joined by a space, lines
by a line break. A whole word's box is used even when the quote covers only part of it (an
opening quotation mark, trailing punctuation), so a rectangle may be slightly wider than the
quote, never narrower.

If the range cannot be mapped reliably (a visible character with no stored word box, a word
outside its page's range or outside the page, a malformed box) the result is None: no
rectangles at all rather than misleading ones.
"""

import math
from dataclasses import dataclass

# Boxes may extend marginally past the page edge (glyph overhang, rounding).
_PAGE_TOLERANCE = 1.0


@dataclass(frozen=True)
class PageGeometry:
    page_number: int  # 1-based, as stored
    width: float  # displayed page size in PDF points
    height: float
    char_start: int  # the page's canonical text range
    char_end: int
    words: list[list[float]]  # [char_start, char_end, x0, y0, x1, y1], in reading order


@dataclass(frozen=True)
class HighlightBox:
    page_number: int
    page_width: float
    page_height: float
    x0: float
    y0: float
    x1: float
    y1: float


def highlight_boxes(
    text: str, start: int, end: int, pages: list[PageGeometry]
) -> list[HighlightBox] | None:
    """Line rectangles covering canonical `text[start:end]`, or None if it cannot be mapped."""
    if not 0 <= start < end <= len(text):
        return None
    boxes: list[HighlightBox] = []
    covered: list[tuple[int, int]] = []
    for page in sorted(pages, key=lambda p: p.page_number):
        if page.char_end <= start or end <= page.char_start:
            continue
        line: list[float] | None = None  # running union [x0, y0, x1, y1]
        previous_end: int | None = None
        for word in sorted(page.words, key=lambda w: w[0]):
            if len(word) != 6:
                return None
            word_start, word_end = int(word[0]), int(word[1])
            if word_end <= start or end <= word_start:
                continue
            if not page.char_start <= word_start < word_end <= page.char_end:
                return None  # the stored word does not belong to this page's text
            box = [float(v) for v in word[2:]]
            if not _valid(box, page):
                return None
            same_line = previous_end is not None and _same_line(text, previous_end, word_start)
            if line is not None and same_line:
                line = [
                    min(line[0], box[0]),
                    min(line[1], box[1]),
                    max(line[2], box[2]),
                    max(line[3], box[3]),
                ]
            else:
                if line is not None:
                    boxes.append(_box(page, line))
                line = box
            previous_end = word_end
            covered.append((word_start, word_end))
        if line is not None:
            boxes.append(_box(page, line))
    if not boxes or not _covers(text, start, end, covered):
        return None
    return boxes


def _same_line(text: str, previous_end: int, next_start: int) -> bool:
    """Consecutive words are on one line when only word spaces separate them."""
    gap = text[previous_end:next_start]
    return previous_end <= next_start and gap.strip(" ") == ""


def _valid(box: list[float], page: PageGeometry) -> bool:
    x0, y0, x1, y1 = box
    if not all(math.isfinite(v) for v in box) or x1 < x0 or y1 < y0:
        return False
    t = _PAGE_TOLERANCE
    return -t <= x0 and -t <= y0 and x1 <= page.width + t and y1 <= page.height + t


def _covers(text: str, start: int, end: int, covered: list[tuple[int, int]]) -> bool:
    """Every visible character of the range lies inside some stored word."""
    position = start
    for word_start, word_end in sorted(covered):
        while position < min(word_start, end):
            if not text[position].isspace():
                return False
            position += 1
        position = max(position, word_end)
        if position >= end:
            return True
    return all(ch.isspace() for ch in text[position:end])


def _box(page: PageGeometry, line: list[float]) -> HighlightBox:
    return HighlightBox(
        page_number=page.page_number,
        page_width=page.width,
        page_height=page.height,
        x0=round(line[0], 2),
        y0=round(line[1], 2),
        x1=round(line[2], 2),
        y1=round(line[3], 2),
    )
