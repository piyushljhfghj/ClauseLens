"""Checks on the model's answer text itself (citations are verified separately).

- Corruption: text decoded with the wrong encoding somewhere ("mojibake", e.g. a UTF-8 curly
  quote shown as "â€œ"), C1 control characters or U+FFFD replacement characters.
  Legitimate Unicode (curly quotes, dashes, accents, symbols) is never flagged.
- Degenerate repetition: the same fragment (>= 10 characters, with letters) three or more times
  back to back, as a looping model produces ("Wings of FireWings of FireWings of Fire").
  Ordinary repeated words, a sentence repeated twice, identical table cells, or repetition that
  is in the evidence itself are not flagged.
- Uncited statements: sentences of the answer that carry no [n] marker, so nothing supports them.
"""

import re

from app.answering.parser import MARKER

# Built from code points so the source stays ASCII (and linters cannot rewrite the ranges).
_MOJIBAKE = re.compile(
    "|".join(
        [
            chr(0xC3) + f"[{chr(0x80)}-{chr(0xBF)}]",  # UTF-8 2-byte char read as Latin-1
            chr(0xE2) + chr(0x20AC),  # UTF-8 E2 80 xx (curly quotes, dashes) read as cp1252
            chr(0xC2) + f"[{chr(0xA0)}-{chr(0xBF)}]",  # UTF-8 C2 xx (nbsp, §) read as Latin-1
            f"[{chr(0x80)}-{chr(0x9F)}]",  # C1 controls: UTF-8 continuation bytes read as Latin-1
            chr(0xFFFD),  # replacement character: bytes invalid in the decoding used
        ]
    )
)
_REPEATED = re.compile(r"(.{10,200}?)\1{2,}", re.S)
_MIN_LETTERS = 3
MIN_STATEMENT_WORDS = 4
# A period after these does not end a sentence ("Dr. A.P.J. Abdul Kalam", "Sec. 4", "e.g.").
_ABBREVIATION = re.compile(
    r"(?:\b[A-Za-z]|\b(?:Dr|Mr|Mrs|Ms|Prof|St|No|Nos|Sec|Secs|Art|Cl|Para|Sch|vs|etc|e\.g|i\.e"
    r"|Inc|Ltd|Co|Corp|approx|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec))\.$",
    re.I,
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
_LEADING_MARKERS = re.compile(r"^((?:\s*\[\d{1,3}\])+)\s*")


def corruption(answer: str, evidence_texts: list[str]) -> str | None:
    """The first mojibake sequence in the answer that is not in the evidence itself."""
    for match in _MOJIBAKE.finditer(answer):
        if not any(match.group(0) in text for text in evidence_texts):
            return match.group(0)
    return None


def degenerate_repetition(answer: str, evidence_texts: list[str]) -> str | None:
    """The first fragment repeated back to back 3+ times that is not so in the evidence."""
    for match in _REPEATED.finditer(answer):
        fragment = match.group(1)
        if sum(ch.isalpha() for ch in fragment) < _MIN_LETTERS:
            continue  # "----------", "1 1 1 1 1 1"
        if "|" in fragment:
            continue  # table cells: "| Not specified | Not specified | Not specified |"
        if any(match.group(0) in text for text in evidence_texts):
            continue
        return fragment
    return None


def statements(answer: str) -> list[str]:
    """Split the answer into sentences/lines, keeping each [n] with the statement it follows."""
    pieces = [p.strip() for p in _SENTENCE_END.split(answer) if p and p.strip()]
    merged: list[str] = []
    for piece in pieces:
        if merged and _ABBREVIATION.search(merged[-1]) and not merged[-1].endswith(("!", "?")):
            merged[-1] = f"{merged[-1]} {piece}"
            continue
        leading = _LEADING_MARKERS.match(piece)
        if leading and merged:  # "... patience. [2]" -> the marker belongs to that sentence
            merged[-1] = f"{merged[-1]} {leading.group(1).strip()}"
            piece = piece[leading.end() :]
            if not piece:
                continue
        merged.append(piece)
    return merged


def uncited_statements(answer: str) -> list[str]:
    """Substantive statements with no citation marker. Headings and lead-ins ending in ':' and
    fragments under four words are not statements of fact."""
    uncited = []
    for statement in statements(answer):
        if MARKER.search(statement) or statement.endswith(":"):
            continue
        if len(re.findall(r"\w+", statement)) >= MIN_STATEMENT_WORDS:
            uncited.append(statement)
    return uncited
