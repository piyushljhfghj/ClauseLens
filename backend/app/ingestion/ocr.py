"""Tesseract OCR for pages without a usable text layer.

The page is rendered as displayed (rotation applied), so OCR word boxes come out in the same
displayed-page coordinate space as text-layer words.
"""

import functools
import logging
from collections import defaultdict
from dataclasses import dataclass

import pymupdf
import pytesseract
from PIL import Image

from app.ingestion.model import SourceBlock, SourceLine, SourceWord

logger = logging.getLogger(__name__)

OCR_PAGE_TIMEOUT_S = 180


@dataclass
class OcrResult:
    blocks: list[SourceBlock]
    confidence: float | None  # mean word confidence 0..1; None when no words were found


@functools.cache
def tesseract_version(cmd: str | None = None) -> str | None:
    """The installed Tesseract version, or None if it cannot be run. Cached per command."""
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    try:
        return str(pytesseract.get_tesseract_version())
    except Exception as exc:  # TesseractNotFoundError, permission errors, ...
        logger.warning("tesseract unavailable", extra={"error": str(exc)})
        return None


def ocr_page(page: pymupdf.Page, *, dpi: int, language: str, page_index: int) -> OcrResult:
    pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    image = Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)
    data = pytesseract.image_to_data(
        image,
        lang=language,
        config="--psm 3",
        output_type=pytesseract.Output.DICT,
        timeout=OCR_PAGE_TIMEOUT_S,
    )
    scale = 72.0 / dpi  # pixels -> PDF points

    # Tesseract hierarchy: block > paragraph > line > word. Paragraphs become our blocks.
    lines: dict[tuple[int, int, int], list[SourceWord]] = defaultdict(list)
    confidences: list[float] = []
    for i, raw in enumerate(data["text"]):
        text = (raw or "").strip()
        confidence = float(data["conf"][i])
        if not text or confidence < 0:
            continue
        left, top = data["left"][i] * scale, data["top"][i] * scale
        width, height = data["width"][i] * scale, data["height"][i] * scale
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines[key].append(SourceWord(text, left, top, left + width, top + height))
        confidences.append(confidence)

    paragraphs: dict[tuple[int, int], list[SourceLine]] = defaultdict(list)
    for (block_num, par_num, _), words in sorted(lines.items()):
        paragraphs[(block_num, par_num)].append(SourceLine(words=words))

    blocks = [
        SourceBlock(lines=par_lines, page_index=page_index)
        for _, par_lines in sorted(paragraphs.items())
    ]
    confidence = sum(confidences) / len(confidences) / 100 if confidences else None
    return OcrResult(blocks=blocks, confidence=confidence)
