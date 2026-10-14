"""Structure-aware chunking.

Invariants (tested):
- Chunks are contiguous, ordered, non-overlapping ranges of the canonical text, and together they
  cover every block. No extracted text is left out of the index, which is what lets the system
  say truthfully that it searched the whole document.
- A chunk never spans two top-level sections (e.g. Article 7 and Article 8).
- A chunk is at most MAX_CHARS long; an oversized paragraph is split at sentence boundaries.
- A heading is kept with the text that follows it.
No overlap between chunks: retrieval adds neighbouring chunks when it needs more context.
"""

import bisect
import re
from dataclasses import dataclass

from app.ingestion.model import Block, ExtractedDocument
from app.ingestion.structure import Section, root_index, section_path
from app.text.tokenize import prepare_for_indexing

TARGET_CHARS = 1200  # ~300 tokens: a clause or two
MAX_CHARS = 1800  # stays under the embedding model's 512-token window with the section path
MIN_CHARS = 300  # below this, a chunk absorbs the next section heading rather than closing

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.;:])\s+(?=[A-Z(\"'\u201c\u2018\d])")


@dataclass
class Chunk:
    ordinal: int
    char_start: int
    char_end: int
    page_start: int | None
    page_end: int | None
    block_start: int
    block_end: int
    section_index: int | None
    section_path: str
    text: str
    context_text: str


@dataclass
class _Piece:
    start: int
    end: int
    block: Block


def _split_block(block: Block, text: str) -> list[_Piece]:
    """Split an oversized block at sentence boundaries, packing sentences up to TARGET_CHARS; a
    single sentence longer than MAX_CHARS is cut at the last whitespace before the limit."""
    if block.char_end - block.char_start <= MAX_CHARS:
        return [_Piece(block.char_start, block.char_end, block)]

    body = text[block.char_start : block.char_end]
    cuts = [m.start() for m in _SENTENCE_BOUNDARY.finditer(body)] + [len(body)]
    pieces: list[_Piece] = []
    start = 0
    last_ok = 0
    for cut in cuts:
        if cut - start > TARGET_CHARS and last_ok > start:
            pieces.append(_Piece(block.char_start + start, block.char_start + last_ok, block))
            start = _skip_space(body, last_ok)
        last_ok = cut
        while last_ok - start > MAX_CHARS:  # one enormous sentence
            hard = body.rfind(" ", start, start + MAX_CHARS)
            hard = hard if hard > start else start + MAX_CHARS
            pieces.append(_Piece(block.char_start + start, block.char_start + hard, block))
            start = _skip_space(body, hard)
    if start < len(body):
        pieces.append(_Piece(block.char_start + start, block.char_end, block))
    return pieces


def _skip_space(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


class _PageLocator:
    def __init__(self, document: ExtractedDocument) -> None:
        pages = [p for p in document.pages if p.char_end > p.char_start]
        self._starts = [p.char_start for p in pages]
        self._numbers = [p.page_number for p in pages]

    def page_at(self, position: int) -> int | None:
        if not self._starts:
            return None
        index = bisect.bisect_right(self._starts, position) - 1
        return self._numbers[max(index, 0)]


def build_chunks(document: ExtractedDocument, sections: list[Section]) -> list[Chunk]:
    text = document.canonical_text
    pieces = [piece for block in document.blocks for piece in _split_block(block, text)]
    locate = _PageLocator(document)
    chunks: list[Chunk] = []
    current: list[_Piece] = []

    def size(group: list[_Piece]) -> int:
        return group[-1].end - group[0].start if group else 0

    def only_headings(group: list[_Piece]) -> bool:
        return all(p.block.kind == "heading" for p in group)

    def flush() -> None:
        if not current:
            return
        start, end = current[0].start, current[-1].end
        section_index = current[0].block.section_index
        path = section_path(sections, section_index)
        body = text[start:end]
        chunks.append(
            Chunk(
                ordinal=len(chunks),
                char_start=start,
                char_end=end,
                page_start=locate.page_at(start),
                page_end=locate.page_at(end - 1),
                block_start=current[0].block.ordinal,
                block_end=current[-1].block.ordinal,
                section_index=section_index,
                section_path=path,
                text=body,
                context_text=f"{path}\n{prepare_for_indexing(body)}"
                if path
                else prepare_for_indexing(body),
            )
        )
        current.clear()

    for piece in pieces:
        if current:
            different_root = root_index(sections, piece.block.section_index) != root_index(
                sections, current[0].block.section_index
            )
            starts_section = (
                piece.start == piece.block.char_start
                and piece.block.section_index is not None
                and sections[piece.block.section_index].char_start == piece.block.char_start
            )
            grown = piece.end - current[0].start
            if (
                different_root
                or (starts_section and size(current) >= MIN_CHARS)
                or (grown > TARGET_CHARS and not only_headings(current))
                or grown > MAX_CHARS
            ):
                flush()
        current.append(piece)
    flush()
    return chunks
