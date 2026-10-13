import pytest

from app.ingestion.inspect import ProcessingFailure, inspect, inspect_docx, inspect_pdf
from app.ingestion.pipeline import ProgressReporter
from tests.fixtures import factory


def failure_code(fn, *args, **kwargs) -> str:  # type: ignore[no-untyped-def]
    with pytest.raises(ProcessingFailure) as info:
        fn(*args, **kwargs)
    assert info.value.message
    return info.value.code


def test_text_pdf() -> None:
    result = inspect_pdf(factory.text_pdf(pages=3), max_pages=100)
    assert (result.page_count, result.pages_without_text) == (3, 0)


def test_mixed_pdf_counts_pages_without_text() -> None:
    result = inspect_pdf(factory.mixed_pdf(), max_pages=100)
    assert (result.page_count, result.pages_without_text) == (2, 1)


def test_scanned_pdf_fails_instead_of_becoming_an_empty_document() -> None:
    assert failure_code(inspect_pdf, factory.scanned_pdf(), max_pages=100) == "SCANNED_NO_TEXT"


def test_password_protected_pdf() -> None:
    assert failure_code(inspect_pdf, factory.encrypted_pdf(), max_pages=100) == "ENCRYPTED_PDF"


def test_owner_restricted_pdf_is_readable() -> None:
    assert inspect_pdf(factory.owner_restricted_pdf(), max_pages=100).page_count == 1


def test_corrupt_pdf() -> None:
    assert failure_code(inspect_pdf, factory.corrupt_pdf(), max_pages=100) == "CORRUPT_FILE"


def test_damaged_pdf_is_repaired_and_lost_content_is_counted() -> None:
    # The unreadable page shows up as a page without text rather than silently vanishing.
    result = inspect_pdf(factory.damaged_pdf(), max_pages=100)
    assert result.page_count == 2
    assert result.pages_without_text == 1


def test_page_limit() -> None:
    assert failure_code(inspect_pdf, factory.text_pdf(pages=5), max_pages=4) == "TOO_MANY_PAGES"


def test_pdf_progress_reaches_completion() -> None:
    seen: list[float] = []
    inspect_pdf(factory.text_pdf(pages=4), max_pages=100, on_progress=seen.append)
    assert seen == [0.25, 0.5, 0.75, 1.0]


def test_docx_with_text() -> None:
    result = inspect_docx(factory.docx_file())
    assert (result.page_count, result.pages_without_text) == (None, None)


def test_docx_with_text_only_in_a_table() -> None:
    assert inspect_docx(factory.docx_file(paragraphs=[], table=[["Fee", "AED 100,000"]]))


def test_empty_docx() -> None:
    assert failure_code(inspect_docx, factory.docx_file(paragraphs=["", "  "])) == "EMPTY_DOCUMENT"


def test_corrupt_docx() -> None:
    assert failure_code(inspect_docx, factory.corrupt_docx()) == "CORRUPT_FILE"


def test_dispatch_rejects_unknown_type() -> None:
    with pytest.raises(ValueError):
        inspect("txt", b"x", max_pages=1)


def test_progress_span_maps_onto_overall_range() -> None:
    reporter = ProgressReporter()
    callback = reporter.span("inspecting", 10, 90)
    assert (reporter.stage, reporter.progress) == ("inspecting", 10)
    callback(0.5)
    assert reporter.progress == 50
    callback(1.0)
    assert reporter.progress == 90
