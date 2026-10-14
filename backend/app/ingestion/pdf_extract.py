"""PDF text extraction with word positions, OCR fallback and furniture removal.

Coordinates: PyMuPDF reports text in *unrotated* page space. We transform every box with the
page's rotation matrix into *displayed* page space (origin top-left, PDF points, size =
`page.rect`), which is what a viewer renders, so highlighting only needs scaling.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import pymupdf

from app.ingestion.furniture import find_furniture
from app.ingestion.model import (
    CanonicalTextBuilder,
    ExtractedDocument,
    ExtractedPage,
    SourceBlock,
    SourceLine,
    SourceWord,
)
from app.ingestion.ocr import ocr_page, tesseract_version
from app.ingestion.structure import looks_like_clause_start

logger = logging.getLogger(__name__)

# A page with fewer extractable characters than this is treated as having no text layer.
MIN_TEXT_CHARS_PER_PAGE = 20
# A page mostly covered by images with little text is a scan carrying a typed stamp (Bates
# number, "Confidential", page footer): its text layer does not represent its content.
STAMPED_SCAN_MAX_CHARS = 200
STAMPED_SCAN_IMAGE_COVERAGE = 0.5
# OCR text replaces such a page's text layer only if it adds substantially more text.
OCR_SUPPLEMENT_MIN_GAIN = 50
# "Visible content" test: fraction of dark pixels in a low-resolution greyscale render.
INK_DPI = 30
INK_DARK_LEVEL = 160
MIN_INK_FRACTION = 0.003
# Below this mean word confidence an OCR page is *low quality*: its text is kept and indexed,
# but much of the page was probably not recognised, so it is counted and warned about
# separately (it is neither "fully read" nor "unreadable").
LOW_QUALITY_OCR_CONFIDENCE = 0.5
# Relative cost of OCR vs reading a text page, for progress reporting.
OCR_PAGE_WEIGHT = 15
TEXT_FLAGS = pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_MEDIABOX_CLIP  # ligatures expanded

# progress(fraction 0..1, detail, is_ocr)
ExtractProgress = Callable[[float, str, bool], None]


@dataclass(frozen=True)
class OcrSettings:
    enabled: bool
    language: str
    dpi: int
    tesseract_cmd: str | None = None


def _is_bold(span: dict) -> bool:  # type: ignore[type-arg]
    return bool(span.get("flags", 0) & 16) or "bold" in span.get("font", "").lower()


def _line_from_raw(raw_line: dict, matrix: pymupdf.Matrix) -> SourceLine | None:  # type: ignore[type-arg]
    """Split a rawdict line into words (whitespace-separated runs of characters). A word's box
    is the union of its characters' boxes, transformed into displayed-page coordinates."""
    words: list[SourceWord] = []
    sizes: list[float] = []
    bold_spans: list[bool] = []
    chars: list[str] = []
    box: pymupdf.Rect | None = None

    def emit(chars: list[str], box: pymupdf.Rect | None) -> None:
        if chars and box is not None:
            shown = box * matrix  # unrotated -> displayed coordinates
            shown.normalize()
            words.append(SourceWord("".join(chars), shown.x0, shown.y0, shown.x1, shown.y1))

    for span in raw_line["spans"]:
        if any(not c["c"].isspace() for c in span["chars"]):
            sizes.append(span["size"])
            bold_spans.append(_is_bold(span))
        for char in span["chars"]:
            if char["c"].isspace():
                emit(chars, box)
                chars, box = [], None
                continue
            chars.append(char["c"])
            char_box = pymupdf.Rect(char["bbox"])
            box = char_box if box is None else box | char_box
    emit(chars, box)
    if not words:
        return None
    return SourceLine(
        words=words, font_size=max(sizes, default=0.0), bold=bool(bold_spans) and all(bold_spans)
    )


def _text_layer_blocks(page: pymupdf.Page, page_index: int) -> list[SourceBlock]:
    matrix = page.rotation_matrix
    raw = page.get_text("rawdict", flags=TEXT_FLAGS)
    blocks: list[SourceBlock] = []
    for raw_block in raw["blocks"]:
        if raw_block.get("type") != 0:  # image block
            continue
        lines = [
            line
            for raw_line in raw_block["lines"]
            if (line := _line_from_raw(raw_line, matrix)) is not None
        ]
        if lines:
            blocks.extend(_split_at_clause_starts(lines, page_index))
    return blocks


def _split_at_clause_starts(lines: list[SourceLine], page_index: int) -> list[SourceBlock]:
    """MuPDF sometimes merges consecutive clauses into one block; start a new block at any line
    that begins with clause numbering ("8.2 ...") so each clause is its own block."""
    blocks: list[SourceBlock] = []
    current: list[SourceLine] = []
    for line in lines:
        if current and looks_like_clause_start(line.text):
            blocks.append(SourceBlock(lines=current, page_index=page_index))
            current = []
        current.append(line)
    if current:
        blocks.append(SourceBlock(lines=current, page_index=page_index))
    return blocks


def ink_fraction(page: pymupdf.Page) -> float:
    """Share of dark pixels when the page is rendered: > 0 for anything visible (scanned images,
    handwriting, vector-drawn text), ~0 for a genuinely blank page."""
    pixmap = page.get_pixmap(dpi=INK_DPI, colorspace=pymupdf.csGRAY)
    samples = pixmap.samples
    return sum(1 for value in samples if value < INK_DARK_LEVEL) / max(len(samples), 1)


def image_coverage(page: pymupdf.Page) -> float:
    """Fraction of the page area covered by images (including images inside form XObjects)."""
    area = page.rect.get_area() or 1.0
    covered = sum(
        (pymupdf.Rect(info["bbox"]) & page.rect).get_area() for info in page.get_image_info()
    )
    return min(covered / area, 1.0)


def _char_count(blocks: list[SourceBlock]) -> int:
    return sum(len(w.text) for b in blocks for line in b.lines for w in line.words)


def extract_pdf(
    data: bytes, *, ocr: OcrSettings, on_progress: ExtractProgress | None = None
) -> ExtractedDocument:
    document = pymupdf.open(stream=data, filetype="pdf")
    warnings: list[str] = []
    with document:
        if document.is_repaired:
            warnings.append(
                "The PDF file was damaged and has been repaired; some content may be missing."
            )
        page_count = document.page_count
        ocr_available = ocr.enabled and tesseract_version(ocr.tesseract_cmd) is not None

        # Pass 1: text layer for every page, and which pages need OCR:
        # - "replace": (almost) no text layer but visible content -> the page is a scan;
        # - "supplement": a little text (a stamp) over a page-sized image -> OCR, and keep
        #   whichever reading has more text.
        page_blocks: list[list[SourceBlock]] = []
        pages: list[ExtractedPage] = []
        needs_ocr: list[int] = []
        supplement: set[int] = set()
        for index, page in enumerate(document):
            blocks = _text_layer_blocks(page, index)
            rect = page.rect
            extracted = ExtractedPage(
                page_number=index + 1,
                width=round(rect.width, 2),
                height=round(rect.height, 2),
                rotation=page.rotation,
                extraction="text",
            )
            chars = _char_count(blocks)
            if chars < MIN_TEXT_CHARS_PER_PAGE:
                blocks = []
                extracted.extraction = "none"
                if ink_fraction(page) >= MIN_INK_FRACTION:
                    needs_ocr.append(index)
                    extracted.note = "ocr_pending"
                else:
                    extracted.note = "blank"
            elif (
                chars < STAMPED_SCAN_MAX_CHARS
                and image_coverage(page) >= STAMPED_SCAN_IMAGE_COVERAGE
            ):
                needs_ocr.append(index)
                supplement.add(index)
            page_blocks.append(blocks)
            pages.append(extracted)

        # Pass 2: OCR. Progress is weighted: one OCR page costs as much as many text pages.
        total_work = page_count + OCR_PAGE_WEIGHT * len(needs_ocr) if ocr_available else page_count
        done = page_count
        if on_progress:
            on_progress(done / max(total_work, 1), f"Read {page_count} pages", False)
        for n, index in enumerate(needs_ocr, start=1):
            extracted = pages[index]
            if not ocr_available:
                # A stamped scan keeps its stamp text but its real content was not read.
                extracted.note = "ocr_unavailable"
                continue
            if on_progress:
                on_progress(done / total_work, f"OCR page {n} of {len(needs_ocr)}", True)
            try:
                result = ocr_page(
                    document[index], dpi=ocr.dpi, language=ocr.language, page_index=index
                )
            except Exception as exc:
                logger.warning("OCR failed", extra={"page": index + 1, "error": str(exc)})
                extracted.note = "ocr_failed"
            else:
                ocr_chars = _char_count(result.blocks)
                layer_chars = _char_count(page_blocks[index])
                if ocr_chars > 0 and ocr_chars >= layer_chars + OCR_SUPPLEMENT_MIN_GAIN * (
                    index in supplement
                ):
                    page_blocks[index] = result.blocks
                    extracted.extraction = "ocr"
                    extracted.note = None
                    extracted.ocr_confidence = (
                        round(result.confidence, 3) if result.confidence is not None else None
                    )
                    if (
                        extracted.ocr_confidence is not None
                        and extracted.ocr_confidence < LOW_QUALITY_OCR_CONFIDENCE
                    ):
                        extracted.note = "ocr_low_quality"
                elif index not in supplement:
                    # Visible content but no recognisable text: handwriting, a diagram, a poor
                    # scan. Reported as unreadable rather than passed off as a blank page.
                    extracted.note = "ocr_no_text"
            done += OCR_PAGE_WEIGHT

        # Pass 3: drop running headers/footers, then assemble the canonical text.
        furniture = find_furniture(page_blocks, [p.height for p in pages])
        builder = CanonicalTextBuilder()
        for index, blocks in enumerate(page_blocks):
            page = pages[index]
            for b, block in enumerate(blocks):
                kept = [
                    line for i, line in enumerate(block.lines) if (index, b, i) not in furniture
                ]
                if kept:
                    builder.add_block(SourceBlock(lines=kept, page_index=index), page)
            # The page's range runs from its first word to its last; separators before the first
            # block belong to no page. A page without text gets an empty range at its position.
            page.char_end = builder.length
            page.char_start = int(page.words[0][0]) if page.words else builder.length

    if not ocr_available and needs_ocr:
        reason = "OCR is disabled" if not ocr.enabled else "Tesseract OCR is not installed"
        warnings.append(
            f"{len(needs_ocr)} scanned page(s) could not be read because {reason}: "
            f"pages {_page_list([i + 1 for i in needs_ocr])}."
        )
    no_text = [p.page_number for p in pages if p.note == "ocr_no_text"]
    if no_text:
        warnings.append(
            f"No text could be recognised on page(s) {_page_list(no_text)}, although they are "
            "not blank (handwriting, diagrams or a poor-quality scan). Their content is not "
            "searchable."
        )
    failed = [p.page_number for p in pages if p.note == "ocr_failed"]
    if failed:
        warnings.append(f"OCR failed on page(s) {_page_list(failed)}.")
    low_quality = [p.page_number for p in pages if p.note == "ocr_low_quality"]
    if low_quality:
        warnings.append(
            f"Low OCR quality on page(s) {_page_list(low_quality)}: content may be largely "
            "unrecognised; answers based on these pages may be incomplete."
        )

    return ExtractedDocument(
        canonical_text=builder.build(), blocks=builder.blocks, pages=pages, warnings=warnings
    )


def _page_list(numbers: list[int], limit: int = 12) -> str:
    shown = ", ".join(str(n) for n in numbers[:limit])
    return shown + (f" and {len(numbers) - limit} more" if len(numbers) > limit else "")
