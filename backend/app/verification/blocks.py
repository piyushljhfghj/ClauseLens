"""Where a verified quotation lies in a Word document's stored blocks.

Word documents have no fixed pages or positions, so a citation is located by text: the viewer
shows the extracted blocks (paragraphs, headings, table rows) exactly as stored, and highlights
`[start, end)` within each block's text. Each block is `canonical_text[char_start:char_end]`;
blocks are separated by a blank line, which is not part of any block, while a manual line break
inside a paragraph (or between the cells of a table row) is part of its block.

Offsets count Unicode code points, as Python strings and every other canonical offset do. (A
JavaScript string index counts UTF-16 units instead, which differ only for characters outside
the Basic Multilingual Plane, such as emoji.)

If the range cannot be mapped reliably (an invalid range, a visible character outside every
block, blocks that overlap, are out of order or lie outside the text) the result is None: no
ranges at all rather than misleading ones.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class StoredBlock:
    ordinal: int  # 0-based, in document order, as stored
    kind: str  # paragraph | heading | table_row
    char_start: int  # the block's canonical text range
    char_end: int


@dataclass(frozen=True)
class BlockSpan:
    block: int  # the block's ordinal
    kind: str
    block_char_start: int
    block_char_end: int
    start: int  # the quote's range within the block's text
    end: int


def block_ranges(
    text: str, start: int, end: int, blocks: list[StoredBlock]
) -> list[BlockSpan] | None:
    """The parts of canonical `text[start:end]` in each block it touches, or None if it cannot
    be mapped. `blocks` may hold any stored blocks of the document; only overlapping ones are
    used, but all are checked for consistency."""
    if not 0 <= start < end <= len(text):
        return None
    ordered = sorted(blocks, key=lambda b: b.ordinal)
    previous: StoredBlock | None = None
    for block in ordered:
        if not 0 <= block.char_start < block.char_end <= len(text):
            return None  # empty, or outside the document text
        if previous is not None and (
            block.ordinal == previous.ordinal or block.char_start < previous.char_end
        ):
            return None  # duplicated, overlapping, or out of document order
        previous = block

    spans: list[BlockSpan] = []
    position = start  # everything before `position` is inside a block or whitespace
    for block in ordered:
        overlap_start, overlap_end = max(start, block.char_start), min(end, block.char_end)
        if overlap_start >= overlap_end:
            continue
        if text[position:overlap_start].strip():
            return None  # visible text between blocks: not covered by any block
        spans.append(
            BlockSpan(
                block=block.ordinal,
                kind=block.kind,
                block_char_start=block.char_start,
                block_char_end=block.char_end,
                start=overlap_start - block.char_start,
                end=overlap_end - block.char_start,
            )
        )
        position = overlap_end
    if not spans or text[position:end].strip():
        return None
    return spans
