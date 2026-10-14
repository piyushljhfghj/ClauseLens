"""Section detection."""

import pytest

from app.ingestion.docx_extract import extract_docx
from app.ingestion.model import CanonicalTextBuilder, SourceBlock, SourceLine, SourceWord
from app.ingestion.pdf_extract import OcrSettings, extract_pdf
from app.ingestion.structure import (
    detect_sections,
    match_numbered_heading,
    roman_to_int,
    section_path,
)
from tests.fixtures import factory


@pytest.mark.parametrize(
    ("text", "number", "title", "level"),
    [
        ("8. TERMINATION", "8", "TERMINATION", 1),
        ("8.2 The Company may terminate this Agreement upon notice.", "8.2", None, 2),
        (
            "8.2 Termination for Convenience. The Company may terminate.",
            "8.2",
            "Termination for Convenience",
            2,
        ),
        ("12.3.1 Notices shall be in writing.", "12.3.1", None, 3),
        ("Section 12.3: Governing Law", "12.3", "Governing Law", 2),
        ("Article IV \u2013 Payment", "Article IV", "Payment", 1),
        ("ARTICLE 7 CONFIDENTIALITY", "Article 7", "CONFIDENTIALITY", 1),
        ("Schedule 2 \u2014 Pricing", "Schedule 2", "Pricing", 1),
        ("Exhibit A", "Exhibit A", None, 1),
    ],
)
def test_numbered_headings(text: str, number: str, title: str | None, level: int) -> None:
    match = match_numbered_heading(text)
    assert match is not None
    assert (match.number, match.title, match.level) == (number, title, level)


@pytest.mark.parametrize(
    "text",
    [
        "30 days written notice is required.",
        "100,000 AED is payable on signature.",
        "2026. The parties agree.",
        "8.2 the lowercase continuation of a sentence",
        "Exhibit a copy of the notice to the Supplier.",
        "Article in the newspaper",
        "(a) the Supplier breaches",
    ],
)
def test_not_headings(text: str) -> None:
    assert match_numbered_heading(text) is None


def test_roman_numerals() -> None:
    assert [roman_to_int(v) for v in ["i", "iv", "IX", "xiv", "MMXXVI", "abc"]] == [
        1,
        4,
        9,
        14,
        2026,
        None,
    ]


def _document(*blocks: tuple[str, bool]):  # type: ignore[no-untyped-def]
    builder = CanonicalTextBuilder()
    for text, bold in blocks:
        lines = [
            SourceLine([SourceWord(w, 0, 0, 1, 1) for w in line.split()], font_size=10, bold=bold)
            for line in text.split("\n")
        ]
        builder.add_block(SourceBlock(lines=lines))
    return builder.build(), builder.blocks


def test_section_tree_nesting_and_ranges() -> None:
    text, blocks = _document(
        ("MASTER SUPPLY AGREEMENT", True),
        ("This Agreement is made on 1 January 2026.", False),
        ("1. Definitions", False),
        ("1.1 \u201cGoods\u201d means the products listed in Schedule 1.", False),
        ("2. Term and Termination", False),
        ("2.1 Either party may terminate on 30 days notice.", False),
        ("30 days notice must be given in writing.", False),
        ("Schedule 1 \u2013 Goods", False),
        ("Widgets and gadgets.", False),
    )
    sections = detect_sections(blocks, text)
    labels = [(s.number, s.title, s.level, s.parent_index) for s in sections]
    assert labels == [
        (None, "MASTER SUPPLY AGREEMENT", 1, None),
        ("1", "Definitions", 1, None),
        ("1.1", None, 2, 1),
        ("2", "Term and Termination", 1, None),
        ("2.1", None, 2, 3),
        ("Schedule 1", "Goods", 1, None),
    ]
    # Children lie inside parents; a parent runs until the next sibling-or-higher section.
    two, two_one = sections[3], sections[4]
    assert two.char_start <= two_one.char_start < two_one.char_end <= two.char_end
    assert "30 days notice must be given" in text[two_one.char_start : two_one.char_end]
    assert sections[5].char_start > two.char_end
    # Blocks know their innermost section; heading blocks are marked.
    assert blocks[6].section_index == 4
    assert [b.kind for b in blocks[:3]] == ["heading", "paragraph", "heading"]
    assert section_path(sections, 4) == "2 Term and Termination > 2.1"


def test_unemphasised_short_line_is_not_a_heading() -> None:
    text, blocks = _document(("Recitals", False), ("Some text follows here.", False))
    assert detect_sections(blocks, text) == []


def test_larger_font_heading_in_pdf() -> None:
    doc = extract_pdf(factory.headings_pdf(), ocr=OcrSettings(False, "eng", 300))
    sections = detect_sections(doc.blocks, doc.canonical_text)
    assert [s.title for s in sections] == ["Payment Terms", "Governing Law"]


def test_docx_numbered_structure() -> None:
    doc = extract_docx(factory.numbered_docx())
    sections = detect_sections(doc.blocks, doc.canonical_text)
    numbers = [s.number for s in sections]
    assert numbers == [None, "1", "1.1", "2", "2.1", "2.2", "3", "3.1"]
    by_number = {s.number: s for s in sections}
    assert by_number["2.2"].title is None  # "...if the Supplier:" is a lead-in, not a title
    assert by_number["2.1"].parent_index == by_number["2"].index
