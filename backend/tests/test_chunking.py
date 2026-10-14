"""Chunking invariants: full coverage, no overlap, bounded size, section boundaries, pages."""

import itertools
import re

import pytest

from app.ingestion.chunking import MAX_CHARS, Chunk, build_chunks
from app.ingestion.docx_extract import extract_docx
from app.ingestion.model import (
    CanonicalTextBuilder,
    ExtractedDocument,
    SourceBlock,
    SourceLine,
    SourceWord,
)
from app.ingestion.pdf_extract import OcrSettings, extract_pdf
from app.ingestion.structure import Section, detect_sections, root_index
from tests.fixtures import factory

OCR_OFF = OcrSettings(False, "eng", 300)


def _build(doc: ExtractedDocument) -> tuple[list[Section], list[Chunk]]:
    sections = detect_sections(doc.blocks, doc.canonical_text)
    return sections, build_chunks(doc, sections)


def _synthetic(*paragraphs: str) -> ExtractedDocument:
    builder = CanonicalTextBuilder()
    for text in paragraphs:
        builder.add_block(
            SourceBlock(lines=[SourceLine([SourceWord(w, 0, 0, 1, 1) for w in text.split()])])
        )
    return ExtractedDocument(builder.build(), builder.blocks, pages=[])


SENTENCE = "The Supplier shall comply with all applicable laws and regulations at its own cost. "


def documents() -> dict[str, ExtractedDocument]:
    return {
        "150-page pdf": extract_pdf(factory.large_pdf_with_needle(), ocr=OCR_OFF),
        "furniture pdf": extract_pdf(factory.furniture_pdf(), ocr=OCR_OFF),
        "numbered docx": extract_docx(factory.numbered_docx()),
        "long paragraphs": _synthetic(
            "1. Compliance",
            SENTENCE * 60,  # ~5,000 chars: must be split
            "2. Audit",
            "The Company may audit the Supplier once per year.",
            "x" * 4000,  # a "sentence" with no boundaries at all
        ),
    }


@pytest.mark.parametrize("name", list(documents()))
def test_chunk_invariants(name: str) -> None:
    doc = documents()[name]
    sections, chunks = _build(doc)
    text = doc.canonical_text
    assert chunks

    # Ordered, non-overlapping, and only whitespace (block/page separators) between neighbours.
    for a, b in itertools.pairwise(chunks):
        assert a.char_end <= b.char_start
        assert text[a.char_end : b.char_start].strip() == ""
    # Together they cover every block: nothing extracted is left out of the index.
    covered = [(c.char_start, c.char_end) for c in chunks]
    for block in doc.blocks:
        for position in (block.char_start, block.char_end - 1):
            assert any(s <= position < e for s, e in covered), (name, block)
    for chunk in chunks:
        assert chunk.text == text[chunk.char_start : chunk.char_end]
        assert chunk.char_end - chunk.char_start <= MAX_CHARS
        assert [c.ordinal for c in chunks] == list(range(len(chunks)))
        # Never spans two top-level sections.
        roots = {
            root_index(sections, b.section_index)
            for b in doc.blocks
            if chunk.char_start <= b.char_start < chunk.char_end
        }
        assert len(roots) <= 1, (name, chunk.section_path)


def test_oversized_paragraph_is_split_at_sentence_boundaries() -> None:
    _, chunks = _build(_synthetic("1. Compliance", SENTENCE * 60))
    body = [c for c in chunks if "Supplier" in c.text]
    assert len(body) > 1
    assert all(c.text.rstrip().endswith(".") for c in body)


def test_heading_is_kept_with_following_text() -> None:
    _, chunks = _build(_synthetic("1. Compliance", "The Supplier shall comply with all laws."))
    assert len(chunks) == 1
    assert chunks[0].text.startswith("1. Compliance")


def test_section_path_in_context_text() -> None:
    _, chunks = _build(extract_docx(factory.numbered_docx()))
    liability = next(c for c in chunks if "capped at AED 100,000" in c.text)
    assert liability.section_path.startswith("3 LIABILITY")
    assert liability.context_text.startswith(liability.section_path + "\n")


def test_chunk_pages_on_large_document() -> None:
    doc = extract_pdf(factory.large_pdf_with_needle(), ocr=OCR_OFF)
    _, chunks = _build(doc)
    needle = next(c for c in chunks if factory.NEEDLE in re.sub(r"\s+", " ", c.text))
    assert (needle.page_start, needle.page_end) == (137, 137)
    assert chunks[0].page_start == 1 and chunks[-1].page_end == 150


def test_chunk_spanning_a_page_break_reports_both_pages() -> None:
    doc = extract_pdf(factory.furniture_pdf(pages=2), ocr=OCR_OFF)
    _, chunks = _build(doc)
    spanning = [c for c in chunks if c.page_start != c.page_end]
    assert spanning and spanning[0].page_start == 1 and spanning[0].page_end == 2
