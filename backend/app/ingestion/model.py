"""In-memory representation of an extracted document.

Everything downstream (structure, chunks, citations, highlighting) refers to positions in ONE
string: the document's canonical text. Every word records its `[char_start, char_end)` range in
that string and, for PDFs, its box on the displayed page, so a character range can always be
mapped back to page rectangles.
"""

from dataclasses import dataclass, field
from typing import Literal

PageExtraction = Literal["text", "ocr", "none"]
# Why a page has extraction "none". "blank" means nothing to read; the others mean content may
# have been missed and must be reported.
UNREADABLE_NOTES = ("ocr_unavailable", "ocr_failed", "ocr_no_text")
# Text was recognised, but with low confidence: partially read, not unreadable.
LOW_QUALITY_NOTE = "ocr_low_quality"
BlockKind = Literal["paragraph", "heading", "table_row"]

# Separators used when assembling canonical text. All whitespace, so quote verification (which
# collapses whitespace) treats line, block and page boundaries as ordinary spaces.
WORD_SEP = " "
LINE_SEP = "\n"
BLOCK_SEP = "\n\n"


@dataclass
class SourceWord:
    """A word as extracted, before it is placed in the canonical text."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class SourceLine:
    words: list[SourceWord]
    font_size: float = 0.0
    bold: bool = False

    @property
    def text(self) -> str:
        return WORD_SEP.join(w.text for w in self.words)

    @property
    def y0(self) -> float:
        return min(w.y0 for w in self.words)

    @property
    def y1(self) -> float:
        return max(w.y1 for w in self.words)


@dataclass
class SourceBlock:
    """A paragraph-like group of lines from one page (PDF) or one paragraph/table row (DOCX)."""

    lines: list[SourceLine]
    page_index: int | None = None  # 0-based; None for DOCX
    kind: BlockKind = "paragraph"
    # DOCX hints for structure detection.
    heading_level: int | None = None
    numbering_label: str | None = None


@dataclass
class Block:
    ordinal: int
    kind: BlockKind
    char_start: int
    char_end: int
    page_start: int | None  # 1-based page numbers; None for DOCX
    page_end: int | None
    font_size: float = 0.0
    bold: bool = False
    heading_level: int | None = None
    numbering_label: str | None = None
    section_index: int | None = None  # index into the structure's section list
    ocr: bool = False  # text came from OCR: no reliable typography (font size, bold, case)


@dataclass
class ExtractedPage:
    page_number: int  # 1-based
    width: float  # displayed (rotation applied) size in PDF points
    height: float
    rotation: int
    extraction: PageExtraction
    char_start: int = 0
    char_end: int = 0
    # Why content may be missing: blank | ocr_unavailable | ocr_failed | ocr_no_text |
    # ocr_low_quality (extraction "ocr", text kept but low confidence). With
    # extraction "text", ocr_unavailable marks a stamped scan whose image could not be OCR'd.
    note: str | None = None
    ocr_confidence: float | None = None  # 0..1 mean word confidence, OCR pages only
    # Compact word boxes: [char_start, char_end, x0, y0, x1, y1] in displayed page coordinates.
    words: list[list[float]] = field(default_factory=list)


@dataclass
class ExtractedDocument:
    canonical_text: str
    blocks: list[Block]
    pages: list[ExtractedPage]  # empty for DOCX
    warnings: list[str] = field(default_factory=list)

    @property
    def ocr_page_count(self) -> int:
        return sum(1 for p in self.pages if p.extraction == "ocr")

    @property
    def unreadable_pages(self) -> list[int]:
        """Pages that may contain text we could not read (not merely blank pages)."""
        return [p.page_number for p in self.pages if p.note in UNREADABLE_NOTES]

    @property
    def low_quality_pages(self) -> list[int]:
        """OCR pages read with low confidence: indexed, but probably only partially recognised."""
        return [p.page_number for p in self.pages if p.note == LOW_QUALITY_NOTE]

    @property
    def pages_without_text_layer(self) -> int:
        return sum(1 for p in self.pages if p.extraction != "text")


class CanonicalTextBuilder:
    """Appends blocks of words, recording the exact character range of every word and block."""

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._length = 0
        self.blocks: list[Block] = []

    @property
    def length(self) -> int:
        return self._length

    def _append(self, text: str) -> int:
        start = self._length
        self._parts.append(text)
        self._length += len(text)
        return start

    def add_block(
        self,
        source: SourceBlock,
        page: ExtractedPage | None = None,
    ) -> Block | None:
        lines = [line for line in source.lines if line.words]
        if not lines:
            return None
        if self.blocks:
            self._append(BLOCK_SEP)
        block_start = self._length
        for line_index, line in enumerate(lines):
            if line_index:
                self._append(LINE_SEP)
            for word_index, word in enumerate(line.words):
                if word_index:
                    self._append(WORD_SEP)
                start = self._append(word.text)
                if page is not None:
                    page.words.append([start, start + len(word.text), *_round_box(word)])
        page_number = page.page_number if page is not None else None
        block = Block(
            ordinal=len(self.blocks),
            kind=source.kind,
            char_start=block_start,
            char_end=self._length,
            page_start=page_number,
            page_end=page_number,
            font_size=max((line.font_size for line in lines), default=0.0),
            bold=all(line.bold for line in lines),
            heading_level=source.heading_level,
            numbering_label=source.numbering_label,
            ocr=page is not None and page.extraction == "ocr",
        )
        self.blocks.append(block)
        return block

    def build(self) -> str:
        return "".join(self._parts)


def _round_box(word: SourceWord) -> list[float]:
    return [round(word.x0, 2), round(word.y0, 2), round(word.x1, 2), round(word.y1, 2)]
