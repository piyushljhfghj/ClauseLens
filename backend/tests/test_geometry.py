"""PDF highlight rectangles from stored word boxes (Phase 6, task 4): pure geometry, no database."""

from app.verification.geometry import HighlightBox, PageGeometry, highlight_boxes

LINE_HEIGHT = 14.0
CHAR_WIDTH = 6.0


def layout(pages: list[list[list[str]]], width: float = 595.0, height: float = 842.0):  # type: ignore[no-untyped-def]
    """Build canonical text and per-page word boxes the way extraction does: words joined by a
    space, lines by a line break, pages by a blank line. Each word gets a box from its position."""
    text = ""
    geometry = []
    for page_index, lines in enumerate(pages):
        if page_index:
            text += "\n\n"
        page_start = len(text)
        words = []
        for line_index, line in enumerate(lines):
            if line_index:
                text += "\n"
            x = 72.0
            y = 100.0 + line_index * LINE_HEIGHT
            for word_index, word in enumerate(line.split(" ")):
                if word_index:
                    text += " "
                    x += CHAR_WIDTH
                start = len(text)
                text += word
                x1 = x + CHAR_WIDTH * len(word)
                words.append([start, start + len(word), x, y, x1, y + 10.0])
                x = x1
        geometry.append(PageGeometry(page_index + 1, width, height, page_start, len(text), words))
    return text, geometry


def span(text: str, quote: str, occurrence: int = 0) -> tuple[int, int]:
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(quote, start + 1)
    return start, start + len(quote)


def words_of(page: PageGeometry, start: int, end: int) -> list[list[float]]:
    return [w for w in page.words if w[0] < end and start < w[1]]


def test_single_line_quote_is_one_rectangle_over_its_words() -> None:
    text, pages = layout([["1. The Supplier may terminate this Agreement by notice."]])
    start, end = span(text, "terminate this Agreement")
    boxes = highlight_boxes(text, start, end, pages)
    covered = words_of(pages[0], start, end)
    assert boxes == [
        HighlightBox(
            1,
            595.0,
            842.0,
            covered[0][2],
            covered[0][3],
            covered[-1][4],
            covered[-1][5],
        )
    ]
    assert len(covered) == 3  # terminate, this, Agreement: nothing else is included


def test_quote_over_several_lines_gives_one_rectangle_per_line() -> None:
    text, pages = layout(
        [["8.2 Either party may terminate", "this Agreement by giving thirty", "days notice."]]
    )
    start, end = span(text, "may terminate\nthis Agreement by giving thirty\ndays")
    boxes = highlight_boxes(text, start, end, pages)
    assert boxes is not None and len(boxes) == 3
    assert [b.y0 for b in boxes] == [100.0, 114.0, 128.0]
    assert boxes[0].x0 == words_of(pages[0], start, end)[0][2]  # starts at "may", not the line
    assert boxes[2].x1 == 72.0 + CHAR_WIDTH * len("days")  # ends after "days", not "notice."


def test_quote_across_a_page_break_has_rectangles_on_both_pages() -> None:
    text, pages = layout(
        [
            ["Clause 4. The Supplier shall deliver the goods"],
            ["within ten (10) business days of the order."],
        ],
    )
    start, end = span(text, "deliver the goods\n\nwithin ten (10) business days")
    boxes = highlight_boxes(text, start, end, pages)
    assert boxes is not None and [b.page_number for b in boxes] == [1, 2]
    assert all((b.page_width, b.page_height) == (595.0, 842.0) for b in boxes)
    assert boxes[1].x0 == 72.0  # "within" starts the second page's line


def test_each_occurrence_maps_to_its_own_position() -> None:
    sentence = "Pricing follows the Order Form."
    text, pages = layout([[f"A. {sentence}", "B. other words here", f"C. {sentence}"]])
    first = highlight_boxes(text, *span(text, sentence, 0), pages)
    second = highlight_boxes(text, *span(text, sentence, 1), pages)
    assert first and second
    assert first[0].y0 == 100.0 and second[0].y0 == 128.0  # line A versus line C


def test_partial_word_coverage_uses_the_whole_word_box() -> None:
    text, pages = layout([["The title is (Wings of Fire) here."]])
    start, end = span(text, "Wings of Fire")  # inside the words "(Wings" and "Fire)"
    boxes = highlight_boxes(text, start, end, pages)
    covered = words_of(pages[0], start, end)
    assert boxes and boxes[0].x0 == covered[0][2] and boxes[0].x1 == covered[-1][4]


def test_missing_page_geometry_gives_no_rectangles() -> None:
    text, pages = layout([["The Supplier shall deliver the goods"], ["within ten days."]])
    start, end = span(text, "deliver the goods\n\nwithin ten days")
    assert highlight_boxes(text, start, end, pages[:1]) is None  # page 2 not supplied
    assert highlight_boxes(text, start, end, []) is None


def test_words_missing_from_stored_geometry_give_no_rectangles() -> None:
    text, pages = layout([["The Supplier shall deliver the goods today."]])
    start, end = span(text, "shall deliver the goods")
    page = pages[0]
    without_deliver = [w for w in page.words if text[int(w[0]) : int(w[1])] != "deliver"]
    damaged = PageGeometry(1, 595.0, 842.0, page.char_start, page.char_end, without_deliver)
    assert highlight_boxes(text, start, end, [damaged]) is None
    empty = PageGeometry(1, 595.0, 842.0, page.char_start, page.char_end, [])
    assert highlight_boxes(text, start, end, [empty]) is None


def test_inconsistent_page_metadata_gives_no_rectangles() -> None:
    text, pages = layout([["The Supplier shall deliver the goods today."]])
    start, end = span(text, "shall deliver the goods")
    page = pages[0]
    # The page's recorded text range does not contain its own words.
    shifted = PageGeometry(1, 595.0, 842.0, page.char_start + 30, page.char_end, page.words)
    assert highlight_boxes(text, start, end, [shifted]) is None
    # A page smaller than its word boxes (wrong size, or boxes from another page).
    tiny = PageGeometry(1, 100.0, 100.0, page.char_start, page.char_end, page.words)
    assert highlight_boxes(text, start, end, [tiny]) is None


def test_malformed_boxes_give_no_rectangles() -> None:
    text, pages = layout([["The Supplier shall deliver the goods today."]])
    start, end = span(text, "shall deliver the goods")
    page = pages[0]
    for bad in (
        [w[:5] if text[int(w[0]) : int(w[1])] == "deliver" else w for w in page.words],
        [
            [w[0], w[1], w[4], w[3], w[2], w[5]] if text[int(w[0]) : int(w[1])] == "the" else w
            for w in page.words
        ],  # x1 < x0
        [
            [w[0], w[1], float("nan"), *w[3:]] if text[int(w[0]) : int(w[1])] == "goods" else w
            for w in page.words
        ],
    ):
        broken = PageGeometry(1, 595.0, 842.0, page.char_start, page.char_end, bad)
        assert highlight_boxes(text, start, end, [broken]) is None


def test_invalid_ranges_give_no_rectangles() -> None:
    text, pages = layout([["The Supplier shall deliver the goods today."]])
    assert highlight_boxes(text, 5, 5, pages) is None
    assert highlight_boxes(text, -1, 5, pages) is None
    assert highlight_boxes(text, 0, len(text) + 1, pages) is None


def test_ocr_style_page_uses_the_same_convention() -> None:
    # OCR boxes are stored in the same displayed-page points; a rotated (landscape) page has
    # its displayed size, so boxes beyond 595 points wide are valid on it, and not on portrait.
    text, pages = layout([["SCANNED AGREEMENT clause one applies"]], width=842.0, height=595.0)
    page = pages[0]
    wide = [[w[0], w[1], w[2] + 400.0, w[3], w[4] + 400.0, w[5]] for w in page.words]
    start, end = span(text, "clause one applies")
    landscape = PageGeometry(1, 842.0, 595.0, page.char_start, page.char_end, wide)
    boxes = highlight_boxes(text, start, end, [landscape])
    assert boxes and 595.0 < boxes[0].x1 <= 842.0
    assert (boxes[0].page_width, boxes[0].page_height) == (842.0, 595.0)
    portrait = PageGeometry(1, 595.0, 842.0, page.char_start, page.char_end, wide)
    assert highlight_boxes(text, start, end, [portrait]) is None
