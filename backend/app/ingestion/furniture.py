"""Detect page furniture: running headers, footers and page numbers.

They are removed from the canonical text so that (a) a sentence continuing across a page break
is contiguous, which lets quotes spanning the break be verified, and (b) "Confidential - Page 3
of 150" does not pollute retrieval. Detection is deliberately conservative: only lines in the
top/bottom margin that repeat on many pages, or that are bare page numbers.
"""

import math
import re
from collections import Counter
from collections.abc import Sequence

from app.ingestion.model import SourceBlock

MARGIN_FRACTION = 0.10
MIN_REPEAT_PAGES = 2  # a margin line repeated on every page of a 2-page document counts
REPEAT_PAGE_FRACTION = 0.5

_DIGITS = re.compile(r"\d+")
_SPACES = re.compile(r"\s+")
_PAGE_NUMBER = re.compile(
    r"^(page\s*)?#(\s*(of|/)\s*#)?$"  # "#", "page #", "page # of #", "# / #"
    r"|^[-\u2013\u2014]\s*#\s*[-\u2013\u2014]$"  # "- # -" (hyphen, en or em dash)
    r"|^#\s*\|.*|.*\|\s*page\s*#$"  # "# | Confidential", "Acme | Page #"
)


def _signature(text: str) -> str:
    """Normalise so 'Page 3 of 150' and 'Page 4 of 150' compare equal."""
    return _SPACES.sub(" ", _DIGITS.sub("#", text.lower())).strip()


def _in_margin(y0: float, y1: float, page_height: float) -> bool:
    return y1 <= page_height * MARGIN_FRACTION or y0 >= page_height * (1 - MARGIN_FRACTION)


def find_furniture(
    pages: Sequence[Sequence[SourceBlock]], page_heights: Sequence[float]
) -> set[tuple[int, int, int]]:
    """Return (page_index, block_index, line_index) keys of lines that are page furniture."""
    candidates: list[tuple[tuple[int, int, int], str]] = []
    for p, blocks in enumerate(pages):
        for b, block in enumerate(blocks):
            for l_, line in enumerate(block.lines):
                if line.words and _in_margin(line.y0, line.y1, page_heights[p]):
                    candidates.append(((p, b, l_), _signature(line.text)))

    pages_with: Counter[str] = Counter()
    for signature in {(key[0], sig) for key, sig in candidates}:
        pages_with[signature[1]] += 1

    threshold = max(MIN_REPEAT_PAGES, math.ceil(len(pages) * REPEAT_PAGE_FRACTION))
    furniture: set[tuple[int, int, int]] = set()
    for key, signature in candidates:
        if _PAGE_NUMBER.match(signature) or (
            len(pages) >= MIN_REPEAT_PAGES and pages_with[signature] >= threshold
        ):
            furniture.add(key)
    return furniture
