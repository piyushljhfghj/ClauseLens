"""Checks on the model's answer text itself (citations are verified separately).

- Corruption: text decoded with the wrong encoding somewhere ("mojibake", e.g. a UTF-8 curly
  quote shown as "â€œ"), C1 control characters or U+FFFD replacement characters.
  Legitimate Unicode (curly quotes, dashes, accents, symbols) is never flagged.
- Degenerate repetition: the same fragment (>= 10 characters, with letters) three or more times
  back to back, as a looping model produces ("Wings of FireWings of FireWings of Fire").
  Ordinary repeated words, a sentence repeated twice, identical table cells, or repetition that
  is in the evidence itself are not flagged.
- Uncited statements: sentences of the answer that carry no [n] marker, so nothing supports them.
  A full stop inside a double-quoted passage does not end the sentence.
- Misattributed quotations: text the answer puts in double quotation marks that none of the
  statement's own verified quotations contain. This only checks quoted text; an unsupported
  paraphrase is not detected here.
- Absence statements: a sentence that only says the passages do not state something ("The
  passages provided do not state the CGPA."). No quotation can show that something is missing,
  so such a sentence needs no marker; one that carries a marker is reported, since that
  quotation cannot support it. Only plain sentences qualify: one that also says something else
  ("..., but they mention X") is an ordinary statement.
- Unverified figures: numbers in a cited statement (amounts, dates, clause or section numbers)
  that none of its own verified quotations contain.
"""

import re
from collections.abc import Mapping

from app.answering.parser import MARKER, marker_numbers
from app.verification.normalize import normalize

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
# A sentence ends at ".", "!" or "?", possibly followed by closing quotation marks; the break is
# the whitespace after them. Citation markers directly adjacent to that punctuation, with no space
# ('...communication."[1]', '...required.[1][2]', '..."[1, 2]'), are matched by the regex and stay
# in the sentence. Markers after whitespace ('...communication." [1]') are not matched: they begin
# the next piece, and statements() reattaches those leading markers to the preceding sentence.
_SENTENCE_BREAK = re.compile(r"[.!?][\"\u201d\u2019]*(?:\[\d{1,3}(?:\s*,\s*\d{1,3})*\])*(\s+)")
_LINE_BREAK = re.compile(r"\n+")
_LEADING_MARKERS = re.compile(r"^((?:\s*\[\d{1,3}(?:\s*,\s*\d{1,3})*\])+)\s*")
_QUOTED_SPAN = re.compile(r'"([^"\n]+)"|\u201c([^\u201c\u201d\n]+)\u201d')
# An omission in quoted text: "...", "\u2026", or either in square brackets ("[...]", "[\u2026]").
_ELLIPSIS = re.compile(r"\[\s*(?:\.{3}|\u2026)\s*\]|\.{3}|\u2026")
MIN_QUOTED_WORDS = 3  # shorter quoted text ("the term 'Services'") is not checked
# Markers right after a quotation: only spaces and punctuation before each ('"A" [1]', '", [1][2]').
_OWN_MARKERS = re.compile(rf"(?:[^\w\[]*{MARKER.pattern})+")
# "The passages provided do not state ...", optionally after a lead-in ("Regarding X, ...").
_ABSENCE = re.compile(
    r"(?:(?:regarding|as for|as to|on|for|about|concerning|in terms of|with regard to"
    r"|with respect to)\b[^,]{1,200},\s*)?"
    r"(?:the\s+|these\s+)?(?:provided\s+)?(?:passages?|excerpts?|evidence)\s+(?:provided\s+)?"
    r"(?:do|does)\s+not\s+(?:state|show|say|mention|specify|provide|support|indicate|contain"
    r"|include|establish|identify|give|list|describe|explain|confirm|rank)\b",
    re.I,
)
# Anything that would add a claim of its own to an absence statement.
_OTHER_CLAUSE = re.compile(
    r"\b(?:but|however|though|although|instead|while|whereas|except|rather|only|yet|besides"
    r"|beyond|unless|other than|apart from|aside from|other|beside)\b"
    r"|;|:|,\s*(?:and|so|which|as|because|since)\b",
    re.I,
)
# A number: digits, with "," or "." only between digits ("100,000.50", "8.2"); not part of a word
# or code ("E3", "1st").
_FIGURE = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)*(?!\w)")
_UNITS = {
    word: value
    for value, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen".split()
    )
} | {
    word: 10 * value
    for value, word in enumerate(
        "twenty thirty forty fifty sixty seventy eighty ninety".split(), start=2
    )
}
_SCALES = {"thousand": 1_000, "million": 1_000_000}
_NUMBER_WORDS = "|".join(sorted([*_UNITS, "hundred", *_SCALES], key=len, reverse=True))
# A run of number words: "thirty", "twenty-one", "one hundred and twenty", "five thousand".
_WORD_NUMBER = re.compile(
    rf"\b(?:{_NUMBER_WORDS})(?:[\s-]+(?:and[\s-]+)?(?:{_NUMBER_WORDS}))*\b", re.I
)


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


def _inside_quotes(line: str, position: int) -> bool:
    before = line[:position]
    return before.count('"') % 2 == 1 or before.count("\u201c") > before.count("\u201d")


def _sentence_pieces(answer: str) -> list[str]:
    """Lines, then sentences, without breaking inside a double-quoted passage. A line whose
    quotation marks do not pair up is split as if there were none (never merged blindly)."""
    pieces: list[str] = []
    for line in _LINE_BREAK.split(answer):
        balanced = line.count('"') % 2 == 0 and line.count("\u201c") == line.count("\u201d")
        start = 0
        for match in _SENTENCE_BREAK.finditer(line):
            if balanced and _inside_quotes(line, match.start(1)):
                continue
            pieces.append(line[start : match.start(1)])
            start = match.end(1)
        pieces.append(line[start:])
    return [p.strip() for p in pieces if p.strip()]


def statements(answer: str) -> list[str]:
    """Split the answer into sentences/lines, keeping each [n] with the statement it follows."""
    pieces = _sentence_pieces(answer)
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
    """Substantive statements with no citation marker. Headings and lead-ins ending in ':',
    fragments under four words and absence statements are not statements of fact."""
    uncited = []
    for statement in statements(answer):
        if MARKER.search(statement) or statement.endswith(":") or is_absence(statement):
            continue
        if len(re.findall(r"\w+", statement)) >= MIN_STATEMENT_WORDS:
            uncited.append(statement)
    return uncited


def is_absence(statement: str) -> bool:
    """Whether the statement only says that the passages do not state something: it opens with
    that (after an optional "Regarding X," lead-in), quotes nothing, and adds no other clause."""
    text = MARKER.sub("", statement).strip()
    return (
        _ABSENCE.match(text) is not None
        and _QUOTED_SPAN.search(text) is None
        and _OTHER_CLAUSE.search(text) is None
    )


def only_absence(answer: str) -> bool:
    """Whether every sentence of the answer is an absence statement with no marker: a reply
    that only says what the passages do not state, and claims nothing that needs a quotation."""
    pieces = statements(answer)
    return bool(pieces) and all(is_absence(s) and not MARKER.search(s) for s in pieces)


def cited_absence_statements(answer: str) -> list[str]:
    """Absence statements that carry a citation marker: the quotation cannot support them."""
    return [s for s in statements(answer) if MARKER.search(s) and is_absence(s)]


def unverified_figures(answer: str, quotes: Mapping[int, list[str]]) -> list[str]:
    """Cited statements with a number that none of their own verified quotations contain
    (`quotes` as for `misattributed_statements`). Numbers compare without thousands separators
    or leading zeros ("100,000" = "100000", "01" = "1"), and a number the quotation writes in
    words matches its digits ("thirty" = "30", "one hundred and twenty" = "120"). Only digits in
    the statement are checked: words such as "three strongest qualities" are not figures.
    Absence statements, and statements whose markers have no verified quotation, are left to
    the other checks."""
    flagged: list[str] = []
    for statement in statements(answer):
        cited = [q for n in marker_numbers(statement) for q in quotes.get(n, [])]
        if not cited or is_absence(statement):
            continue
        known = {figure for quote in cited for figure in _figures(quote) | _word_figures(quote)}
        if any(figure not in known for figure in _figures(MARKER.sub("", statement))):
            flagged.append(statement)
    return flagged


def _figures(text: str) -> set[str]:
    found = set()
    for match in _FIGURE.finditer(text):
        figure = match.group(0).replace(",", "")
        found.add(figure if "." in figure else str(int(figure)))
    return found


def _word_figures(text: str) -> set[str]:
    """Numbers written in words ("thirty", "twenty-one", "one hundred and twenty")."""
    found = set()
    for match in _WORD_NUMBER.finditer(text):
        total = current = 0
        for word in re.split(r"[\s-]+", match.group(0).lower()):
            if word in _UNITS:
                current += _UNITS[word]
            elif word == "hundred":
                current = max(current, 1) * 100
            elif word in _SCALES:
                total += max(current, 1) * _SCALES[word]
                current = 0
        found.add(str(total + current))
    return found


def misattributed_statements(answer: str, quotes: Mapping[int, list[str]]) -> list[str]:
    """Statements that put text in double quotation marks which none of their own cited
    quotations contain (`quotes`: citation number -> its verified quotation texts). Quoted text
    of at least MIN_QUOTED_WORDS words is checked, and so is any quoted text with an omission
    ("...", "[...]"), whose kept pieces must all occur, in order, in one quotation; statements
    whose markers have no verified quotation are left to the other checks.

    When every checked quoted span has markers of its own ('"A" [1] and "B" [2]'), each span must
    be in its own markers' quotations, so swapped markers are caught. Otherwise ('"A" and "B"
    [1][2]') which marker belongs to which span is not stated, and each span is checked against
    all the statement's quotations."""
    flagged: list[str] = []
    for statement in statements(answer):
        targets = _targets(marker_numbers(statement), quotes)
        if not targets:
            continue
        if any(
            _not_quoted_in(span, own or targets) for span, own in _quoted_spans(statement, quotes)
        ):
            flagged.append(statement)
    return flagged


def _targets(numbers: list[int], quotes: Mapping[int, list[str]]) -> list[str]:
    return [normalize(q).text for n in numbers for q in quotes.get(n, [])]


def _quoted_spans(statement: str, quotes: Mapping[int, list[str]]) -> list[tuple[str, list[str]]]:
    """Each checked quoted span with the verified quotations of its own markers: those right
    after its closing quotation mark, with only spaces and punctuation between ('"A" [1]',
    '"A", [1][2]'). A marker after other words ('"A" and so [1]'), or right before the next
    quotation ('"A", and [1] "B"'), is no span's own. Short quoted terms that are not checked
    ('the "Supplier"') are left out. If any checked span has no marker of its own, no span is
    given its own (empty lists); a span whose own markers have no verified quotation also gets an
    empty list. An empty list means "check against all the statement's quotations"."""
    matches = list(_QUOTED_SPAN.finditer(statement))
    ends = [m.start() for m in matches[1:]] + [len(statement)] if matches else []
    spans: list[tuple[str, list[int]]] = []
    for m, end in zip(matches, ends, strict=True):
        span = m.group(1) or m.group(2)
        if _checked(span):
            own = _OWN_MARKERS.match(statement, m.end(), end)
            spans.append((span, marker_numbers(own.group(0)) if own else []))
    if not all(numbers for _, numbers in spans):
        spans = [(span, []) for span, _ in spans]
    return [(span, _targets(numbers, quotes)) for span, numbers in spans]


def _checked(span: str) -> bool:
    """Whether `_not_quoted_in` checks this quoted text at all."""
    words = len(re.findall(r"\w+", span))
    return words >= MIN_QUOTED_WORDS or (words > 0 and _ELLIPSIS.search(span) is not None)


def _not_quoted_in(span: str, targets: list[str]) -> bool:
    pieces = _ELLIPSIS.split(span)
    if len(pieces) == 1:
        if len(re.findall(r"\w+", span)) < MIN_QUOTED_WORDS:
            return False  # a short quoted term ("the 'Services'") is not checked
        needle = normalize(span).text.strip(" .,;:!?")
        return bool(needle) and not any(needle in target for target in targets)
    # An omission: every kept piece, however short, must occur in order in one quotation, as
    # whole words, so nothing beside the omission (an invented "never") goes unchecked.
    kept = [p for p in (normalize(piece).text.strip(" .,;:!?") for piece in pieces) if p]
    return bool(kept) and not any(_in_order(kept, target) for target in targets)


def _in_order(pieces: list[str], target: str) -> bool:
    position = 0
    for piece in pieces:
        found = re.compile(rf"(?<!\w){re.escape(piece)}(?!\w)").search(target, position)
        if found is None:
            return False
        position = found.end()
    return True
