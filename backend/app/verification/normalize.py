"""Text normalisation with an offset map back to the original string.

Quote verification compares a normalised quote with normalised source text, then uses the map
to recover the exact *original* character range, so what is shown and highlighted is always
the stored canonical text, never the model's version of it.

Normalisation (each step only removes differences that do not change the words):
- Unicode NFKC (ligatures such as "fi", full-width forms)
- curly quotes / primes -> straight quotes; hyphen and dash variants -> "-"
- soft hyphens and zero-width characters removed
- end-of-line hyphenation joined: "termi-" + newline + "nation" -> "termination"
- every whitespace run -> one space; leading/trailing whitespace dropped
- case-folded
Punctuation is kept: dropping it could merge or alter words.
"""

import unicodedata
from dataclasses import dataclass

_TRANSLATE = {
    **dict.fromkeys(map(chr, (0x2018, 0x2019, 0x201A, 0x201B, 0x2032)), "'"),
    **dict.fromkeys(map(chr, (0x201C, 0x201D, 0x201E, 0x201F, 0x2033)), '"'),
    **dict.fromkeys(map(chr, (0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, 0x2212)), "-"),
}
_REMOVE = set(map(chr, (0x00AD, 0x200B, 0x200C, 0x200D, 0xFEFF)))


@dataclass(frozen=True)
class Normalized:
    text: str
    # offsets[i] = index in the original string of normalised character i
    offsets: tuple[int, ...]

    def original_span(self, start: int, end: int) -> tuple[int, int]:
        """Original [start, end) covering normalised characters [start, end)."""
        return self.offsets[start], self.offsets[end - 1] + 1


def normalize(text: str, *, keep_line_hyphens: bool = False) -> Normalized:
    r"""Normalise `text`. With `keep_line_hyphens`, a hyphen at a line break is kept and only the
    break is dropped ("non-\ncompete" -> "non-compete") instead of joining the parts
    ("noncompete"): extraction cannot tell a real compound hyphen from word hyphenation."""
    # 1. Character-level mapping (NFKC can expand one character into several).
    chars: list[tuple[str, int]] = []
    for index, ch in enumerate(text):
        if ch in _REMOVE:
            continue
        for out in unicodedata.normalize("NFKC", ch):
            out = _TRANSLATE.get(out, out)
            for lowered in out.casefold():
                chars.append((lowered, index))

    # 2. Join end-of-line hyphenation, then collapse whitespace.
    result: list[tuple[str, int]] = []
    i = 0
    while i < len(chars):
        ch, index = chars[i]
        if ch == "-" and result and result[-1][0].isalpha():
            j = i + 1
            newline = False
            while j < len(chars) and chars[j][0].isspace():
                newline = newline or chars[j][0] in "\n\r"
                j += 1
            if newline and j < len(chars) and chars[j][0].isalpha():
                if keep_line_hyphens:
                    result.append((ch, index))  # keep the hyphen, drop only the line break
                i = j  # drop the hyphen and the line break; continue with the next word part
                continue
        if ch.isspace():
            if result and result[-1][0] != " ":
                result.append((" ", index))
            i += 1
            continue
        result.append((ch, index))
        i += 1
    if result and result[-1][0] == " ":
        result.pop()
    return Normalized("".join(c for c, _ in result), tuple(i for _, i in result))
