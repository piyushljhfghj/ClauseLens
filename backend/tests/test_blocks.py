"""Word citation block ranges (Phase 6, task 5): pure mapping, no database."""

from app.verification.blocks import BlockSpan, StoredBlock, block_ranges

NL = chr(10)


def document(*blocks: tuple[str, str]) -> tuple[str, list[StoredBlock]]:
    """Canonical text and stored blocks the way DOCX extraction builds them: blocks separated
    by a blank line; a line break inside a block is part of it."""
    text, stored = "", []
    for ordinal, (kind, block_text) in enumerate(blocks):
        if ordinal:
            text += NL + NL
        start = len(text)
        text += block_text
        stored.append(StoredBlock(ordinal, kind, start, len(text)))
    return text, stored


def span(text: str, quote: str, occurrence: int = 0) -> tuple[int, int]:
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(quote, start + 1)
    return start, start + len(quote)


TEXT, BLOCKS = document(
    ("heading", "MASTER SERVICES AGREEMENT"),
    ("paragraph", "1.2 Either party may terminate this Agreement on thirty (30) days notice."),
    ("paragraph", "Notices must be in writing." + NL + "Email counts as writing."),
    ("table_row", "Fee" + NL + "AED 100,000"),
    ("paragraph", "Pricing follows the Order Form."),
    ("paragraph", "Pricing follows the Order Form."),
)


def check(quote: str, occurrence: int = 0) -> list[BlockSpan] | None:
    return block_ranges(TEXT, *span(TEXT, quote, occurrence), BLOCKS)


def in_block(s: BlockSpan) -> str:
    return TEXT[s.block_char_start : s.block_char_end][s.start : s.end]


def test_quote_inside_one_paragraph() -> None:
    quote = "Either party may terminate this Agreement"
    ranges = check(quote)
    assert ranges is not None and len(ranges) == 1
    (r,) = ranges
    assert (r.block, r.kind) == (1, "paragraph")
    assert (r.block_char_start, r.block_char_end) == (BLOCKS[1].char_start, BLOCKS[1].char_end)
    assert (r.start, r.end) == (4, 4 + len(quote)) and in_block(r) == quote


def test_numbered_clause_label_is_part_of_its_block() -> None:
    (r,) = check("1.2 Either party may terminate") or []
    assert (r.block, r.start) == (1, 0) and in_block(r) == "1.2 Either party may terminate"


def test_quote_across_two_paragraphs_excludes_the_separator() -> None:
    quote = "(30) days notice." + NL + NL + "Notices must be in writing."
    ranges = check(quote)
    assert ranges is not None and [r.block for r in ranges] == [1, 2]
    assert [in_block(r) for r in ranges] == ["(30) days notice.", "Notices must be in writing."]
    assert ranges[0].end == BLOCKS[1].char_end - BLOCKS[1].char_start  # to the block's end
    assert ranges[1].start == 0


def test_manual_line_break_stays_inside_the_block() -> None:
    quote = "in writing." + NL + "Email counts"
    (r,) = check(quote) or []
    assert r.block == 2 and in_block(r) == quote


def test_quote_across_cells_of_a_table_row() -> None:
    (r,) = check("Fee" + NL + "AED 100,000") or []
    assert (r.block, r.kind, r.start) == (3, "table_row", 0) and in_block(r) == "Fee\nAED 100,000"


def test_heading_block() -> None:
    (r,) = check("SERVICES AGREEMENT") or []
    assert (r.block, r.kind) == (0, "heading")


def test_repeated_text_maps_to_the_given_occurrence() -> None:
    first = check("Pricing follows the Order Form.", 0)
    second = check("Pricing follows the Order Form.", 1)
    assert first and second and (first[0].block, second[0].block) == (4, 5)


def test_only_overlapping_blocks_are_needed() -> None:
    start, end = span(TEXT, "Email counts as writing.")
    ranges = block_ranges(TEXT, start, end, [BLOCKS[2]])
    assert ranges and ranges[0].block == 2


def test_invalid_ranges_give_nothing() -> None:
    assert block_ranges(TEXT, 5, 5, BLOCKS) is None
    assert block_ranges(TEXT, -1, 5, BLOCKS) is None
    assert block_ranges(TEXT, 0, len(TEXT) + 1, BLOCKS) is None


def test_missing_blocks_give_nothing() -> None:
    start, end = span(TEXT, "(30) days notice." + NL + NL + "Notices must")
    assert block_ranges(TEXT, start, end, [BLOCKS[1]]) is None  # second block not supplied
    assert block_ranges(TEXT, start, end, []) is None


def test_visible_text_outside_every_block_gives_nothing() -> None:
    # A block whose stored range stops short of its text leaves part of the quote uncovered.
    short = StoredBlock(1, "paragraph", BLOCKS[1].char_start, BLOCKS[1].char_end - 8)
    start, end = span(TEXT, "thirty (30) days notice.")
    assert block_ranges(TEXT, start, end, [short]) is None


def test_inconsistent_block_metadata_gives_nothing() -> None:
    start, end = span(TEXT, "Either party may terminate")
    b0, b1, b2 = BLOCKS[0], BLOCKS[1], BLOCKS[2]
    overlapping = StoredBlock(2, "paragraph", b1.char_end - 5, b2.char_end)
    out_of_order = StoredBlock(0, "heading", b2.char_start, b2.char_end)  # ordinal 0 after 1
    duplicate = StoredBlock(1, "paragraph", b2.char_start, b2.char_end)
    outside = StoredBlock(9, "paragraph", len(TEXT) - 3, len(TEXT) + 10)
    empty = StoredBlock(8, "paragraph", b2.char_start, b2.char_start)
    for broken in (
        [b0, b1, overlapping],
        [out_of_order, b1],
        [b1, duplicate],
        [b1, outside],
        [b1, empty],
    ):
        assert block_ranges(TEXT, start, end, broken) is None
