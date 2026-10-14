"""Low-quality OCR pages: kept and indexed, but counted and warned about separately.

OCR confidences are injected through a fake `ocr_page`, so these tests exercise the
classification logic without depending on what Tesseract happens to read from an image.
"""

import httpx
import pymupdf
import pytest
from fastapi import FastAPI

from app.ingestion import pdf_extract
from app.ingestion.model import SourceBlock, SourceLine, SourceWord
from app.ingestion.ocr import OcrResult, tesseract_version
from app.ingestion.pdf_extract import LOW_QUALITY_OCR_CONFIDENCE, OcrSettings, extract_pdf
from tests.fixtures import factory

OCR_ON = OcrSettings(enabled=True, language="eng", dpi=300)
needs_tesseract = pytest.mark.skipif(tesseract_version() is None, reason="Tesseract not installed")

LOW_QUALITY_WARNING = (
    "content may be largely unrecognised; answers based on these pages may be incomplete"
)


def fake_ocr(confidences: dict[int, float | None]):  # type: ignore[no-untyped-def]
    """An `ocr_page` stand-in: page index -> confidence (None = no text recognised)."""

    def run(page, *, dpi, language, page_index):  # type: ignore[no-untyped-def]
        confidence = confidences[page_index]
        if confidence is None:
            return OcrResult(blocks=[], confidence=None)
        words = [
            SourceWord(f"page{page_index}word{i}", 72 + 60 * i, 100, 120 + 60 * i, 112)
            for i in range(5)
        ]
        return OcrResult(
            blocks=[SourceBlock(lines=[SourceLine(words=words)], page_index=page_index)],
            confidence=confidence,
        )

    return run


@pytest.fixture
def patch_ocr(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    def apply(confidences: dict[int, float | None]) -> None:
        monkeypatch.setattr(pdf_extract, "ocr_page", fake_ocr(confidences))
        monkeypatch.setattr(pdf_extract, "tesseract_version", lambda cmd=None: "5.0")

    return apply


def mixed_quality_pdf() -> bytes:
    """Page 1 text layer; pages 2-5 scanned (OCR needed)."""
    document = pymupdf.open(stream=factory.text_pdf(pages=1), filetype="pdf")
    document.insert_pdf(pymupdf.open(stream=factory.scanned_pdf(pages=4), filetype="pdf"))
    return document.tobytes()


def test_low_confidence_page_keeps_its_text_but_is_flagged(patch_ocr) -> None:  # type: ignore[no-untyped-def]
    patch_ocr({0: 0.31})
    doc = extract_pdf(factory.scanned_pdf(pages=1), ocr=OCR_ON)
    page = doc.pages[0]
    assert (page.extraction, page.note, page.ocr_confidence) == ("ocr", "ocr_low_quality", 0.31)
    assert "page0word0" in doc.canonical_text  # text preserved and therefore indexed
    assert page.words  # positions preserved for later highlighting
    assert doc.low_quality_pages == [1]
    assert doc.unreadable_pages == []  # low quality is not "unreadable"
    assert any(LOW_QUALITY_WARNING in w and "page(s) 1" in w for w in doc.warnings)


def test_high_confidence_page_is_not_flagged(patch_ocr) -> None:  # type: ignore[no-untyped-def]
    patch_ocr({0: 0.93})
    doc = extract_pdf(factory.scanned_pdf(pages=1), ocr=OCR_ON)
    assert (doc.pages[0].extraction, doc.pages[0].note) == ("ocr", None)
    assert doc.low_quality_pages == [] and doc.warnings == []


def test_threshold_is_strictly_below_fifty_percent(patch_ocr) -> None:  # type: ignore[no-untyped-def]
    patch_ocr({0: LOW_QUALITY_OCR_CONFIDENCE, 1: LOW_QUALITY_OCR_CONFIDENCE - 0.001})
    doc = extract_pdf(factory.scanned_pdf(pages=2), ocr=OCR_ON)
    assert [p.note for p in doc.pages] == [None, "ocr_low_quality"]


def test_unreadable_page_is_still_unreadable_not_low_quality(patch_ocr) -> None:  # type: ignore[no-untyped-def]
    patch_ocr({0: None})
    doc = extract_pdf(factory.scanned_pdf(pages=1), ocr=OCR_ON)
    assert (doc.pages[0].extraction, doc.pages[0].note) == ("none", "ocr_no_text")
    assert doc.unreadable_pages == [1] and doc.low_quality_pages == []


def test_mixed_quality_document_counts_each_category_separately(patch_ocr) -> None:  # type: ignore[no-untyped-def]
    # Page indexes 1-4 are scanned: good, low quality, nothing recognised, exactly at threshold.
    patch_ocr({1: 0.92, 2: 0.30, 3: None, 4: 0.50})
    doc = extract_pdf(mixed_quality_pdf(), ocr=OCR_ON)
    assert [(p.extraction, p.note) for p in doc.pages] == [
        ("text", None),
        ("ocr", None),
        ("ocr", "ocr_low_quality"),
        ("none", "ocr_no_text"),
        ("ocr", None),
    ]
    assert doc.ocr_page_count == 3
    assert doc.low_quality_pages == [3]
    assert doc.unreadable_pages == [4]
    assert sum(LOW_QUALITY_WARNING in w for w in doc.warnings) == 1
    assert sum("No text could be recognised" in w for w in doc.warnings) == 1
    # Text from every page that yielded any is in the canonical text.
    assert all(f"page{i}word0" in doc.canonical_text for i in (1, 2, 4))


@needs_tesseract
def test_real_ocr_of_a_clean_scan_is_not_low_quality() -> None:
    """Real Tesseract on a clean generated scan: robust check (confidence is well above 50%)."""
    doc = extract_pdf(factory.scanned_pdf(pages=1), ocr=OCR_ON)
    assert doc.pages[0].extraction == "ocr"
    assert doc.pages[0].note is None and doc.low_quality_pages == []


async def test_low_quality_count_and_warning_reach_the_api(
    api: httpx.AsyncClient, live_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pdf_extract, "ocr_page", fake_ocr({1: 0.92, 2: 0.30, 3: None, 4: 0.50}))
    monkeypatch.setattr(pdf_extract, "tesseract_version", lambda cmd=None: "5.0")
    response = await api.post("/api/documents", files={"file": ("mixed.pdf", mixed_quality_pdf())})
    assert response.status_code == 202
    while await live_app.state.worker.run_once():
        pass

    document = (await api.get(f"/api/documents/{response.json()['id']}")).json()
    assert document["indexed"] is True and document["index_version"] == 3
    assert (document["ocr_page_count"], document["low_quality_page_count"]) == (3, 1)
    assert document["unreadable_page_count"] == 1
    assert any(LOW_QUALITY_WARNING in w for w in document["extraction_warnings"])

    pages = (await api.get(f"/api/documents/{document['id']}/pages")).json()["pages"]
    assert [p["note"] for p in pages] == [None, None, "ocr_low_quality", "ocr_no_text", None]
    assert pages[2]["ocr_confidence"] == 0.3 and pages[2]["char_count"] > 0


async def test_documents_indexed_by_an_older_pipeline_need_reprocessing(
    api: httpx.AsyncClient, live_app: FastAPI
) -> None:
    from sqlalchemy import update

    from app.db.models import Document
    from app.db.session import get_sessionmaker

    response = await api.post("/api/documents", files={"file": ("a.pdf", factory.text_pdf(1))})
    while await live_app.state.worker.run_once():
        pass
    async with get_sessionmaker()() as session:
        await session.execute(update(Document).values(index_version=2))  # pre-low-quality index
        await session.commit()
    document = (await api.get(f"/api/documents/{response.json()['id']}")).json()
    assert (document["indexed"], document["needs_reprocessing"]) == (False, True)
