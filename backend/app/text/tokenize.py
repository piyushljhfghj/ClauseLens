"""Text preparation for indexing (embeddings and BM25).

This never changes the canonical text, which stays exactly as extracted: citations are
verified and located against it. These functions only produce derived *index* text.
"""

import re
import unicodedata

_LINE_HYPHENATION = re.compile(r"(\w)-\n(\w)")
_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})
# Numbers (with thousands separators/decimals, e.g. 100,000 / 8.2) or words (with apostrophes).
_TOKEN = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+(?:'[^\W\d_]+)*")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")


def prepare_for_indexing(text: str) -> str:
    """NFKC (ligatures, full-width forms), straight quotes, and line-break hyphenation joined
    ('termi-' + newline + 'nation' -> 'termination')."""
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES)
    return _LINE_HYPHENATION.sub(r"\1\2", text)


def lexical_tokens(text: str) -> list[str]:
    """Lowercased tokens for BM25. '100,000' and '100000' produce the same token; clause numbers
    like '8.2' stay intact so 'clause 8.2' can be matched exactly."""
    prepared = prepare_for_indexing(text).lower()
    return [_THOUSANDS.sub("", token) for token in _TOKEN.findall(prepared)]
