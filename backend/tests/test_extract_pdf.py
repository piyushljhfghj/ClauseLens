"""PDF extraction: canonical text, word positions, page mapping, furniture, OCR."""

import re

import pytest

from app.ingestion import pdf_extract
from app.ingestion.model import ExtractedDocument
from app.ingestion.ocr import tesseract_version
from app.ingestion.pdf_extract import OcrSettings, extract_pdf
from tests.fixtures import factory

OCR_ON = OcrSettings(enabled=True, language="eng", dpi=300)
OCR_OFF = OcrSettings(enabled=False, language="eng", dpi=300)
needs_tesseract = pytest.mark.skipif(
    tesseract_version() is None, reason="Tesseract is not installed"
)


def collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def assert_positions_consistent(doc: ExtractedDocument) -> None:
    """Every stored word box points at exactly that word in the canonical text, inside its page;
    page ranges are ordered and non-overlapping."""
    text = doc.canonical_text
    previous_end = 0
    for page in doc.pages:
        assert previous_end <= page.char_start <= page.char_end <= len(text)
        previous_end = page.char_end
        for start, end, x0, y0, x1, y1 in page.words:
            word = text[int(start) : int(end)]
            assert word and not any(ch.isspace() for ch in word)
            assert page.char_start <= start < end <= page.char_end
            assert -1 <= x0 < x1 <= page.width + 1
            assert -1 <= y0 < y1 <= page.height + 1


def test_text_pdf_words_map_back_to_canonical_text() -> None:
    doc = extract_pdf(factory.text_pdf(pages=3), ocr=OCR_ON)
    assert_positions_consistent(doc)
    assert [p.extraction for p in doc.pages] == ["text", "text", "text"]
    assert "terminate this Agreement upon thirty (30) days" in collapse(doc.canonical_text)
    # Every page's text is attributed to that page and nowhere else.
    for page in doc.pages:
        assert doc.canonical_text[page.char_start : page.char_end].startswith(
            f"{page.page_number}."
        )


def test_page_ranges_on_a_150_page_document() -> None:
    doc = extract_pdf(factory.large_pdf_with_needle(), ocr=OCR_ON)
    assert len(doc.pages) == 150
    assert_positions_consistent(doc)
    offset = doc.canonical_text.index("Escrow")  # the needle sentence wraps across lines
    owner = [p.page_number for p in doc.pages if p.char_start <= offset < p.char_end]
    assert owner == [137]


def test_headers_footers_removed_and_page_breaks_are_contiguous() -> None:
    doc = extract_pdf(factory.furniture_pdf(pages=4), ocr=OCR_ON)
    assert "CONFIDENTIAL" not in doc.canonical_text
    assert "Page 1 of 4" not in doc.canonical_text
    # The sentence continues across the page break with only whitespace in between, so a quote
    # spanning the break can be verified later.
    assert "deliver the goods within ten (10) business days" in collapse(doc.canonical_text)
    assert_positions_consistent(doc)


def test_rotated_page_boxes_are_in_displayed_coordinates() -> None:
    import pymupdf

    source = pymupdf.open()
    page = source.new_page(width=595, height=842)
    page.insert_text((72, 100), "Rotated clause text appears on this page for testing")
    page.set_rotation(90)
    doc = extract_pdf(source.tobytes(), ocr=OCR_ON)
    extracted = doc.pages[0]
    assert (extracted.width, extracted.height, extracted.rotation) == (842, 595, 90)
    assert_positions_consistent(doc)
    _, _, x0, y0, x1, y1 = extracted.words[0]
    assert (y1 - y0) > (x1 - x0)  # horizontal text on a page rotated 90 degrees reads vertically


def test_merged_clauses_are_split_into_separate_blocks() -> None:
    doc = extract_pdf(factory.merged_clauses_pdf(), ocr=OCR_ON)
    texts = [doc.canonical_text[b.char_start : b.char_end] for b in doc.blocks]
    assert [t[:3] for t in texts] == ["8.1", "8.2"]


def test_blank_page_is_not_reported_as_unreadable() -> None:
    doc = extract_pdf(factory.pdf_with_blank_page(), ocr=OCR_ON)
    assert [(p.extraction, p.note) for p in doc.pages] == [("text", None), ("none", "blank")]
    assert doc.unreadable_pages == []
    assert doc.warnings == []


@needs_tesseract
def test_scanned_pdf_is_read_with_ocr() -> None:
    doc = extract_pdf(factory.scanned_pdf(pages=2), ocr=OCR_ON)
    assert [p.extraction for p in doc.pages] == ["ocr", "ocr"]
    assert all(p.ocr_confidence is not None and p.ocr_confidence > 0.8 for p in doc.pages)
    assert "terminate this Agreement" in collapse(doc.canonical_text)
    assert_positions_consistent(doc)
    assert doc.ocr_page_count == 2 and doc.unreadable_pages == []


@needs_tesseract
def test_mixed_pdf_uses_text_layer_and_ocr_per_page() -> None:
    doc = extract_pdf(factory.mixed_pdf(), ocr=OCR_ON)
    assert [p.extraction for p in doc.pages] == ["text", "ocr"]
    assert_positions_consistent(doc)


def test_without_ocr_scanned_pages_are_reported_not_hidden() -> None:
    doc = extract_pdf(factory.mixed_pdf(), ocr=OCR_OFF)
    assert [(p.extraction, p.note) for p in doc.pages] == [
        ("text", None),
        ("none", "ocr_unavailable"),
    ]
    assert doc.unreadable_pages == [2]
    assert any("could not be read" in w and "pages 2" in w for w in doc.warnings)


def test_ocr_failure_on_a_page_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("tesseract crashed")

    monkeypatch.setattr(pdf_extract, "ocr_page", broken)
    monkeypatch.setattr(pdf_extract, "tesseract_version", lambda cmd=None: "5.0")
    doc = extract_pdf(factory.mixed_pdf(), ocr=OCR_ON)
    assert doc.pages[1].note == "ocr_failed"
    assert doc.unreadable_pages == [2]
    assert any("OCR failed" in w for w in doc.warnings)


def test_progress_reports_ocr_separately() -> None:
    events: list[tuple[float, str, bool]] = []
    extract_pdf(factory.text_pdf(pages=2), ocr=OCR_ON, on_progress=lambda *e: events.append(e))
    assert events and events[-1][0] == pytest.approx(1.0)
    assert not any(is_ocr for _, _, is_ocr in events)
