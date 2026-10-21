"""Parse the model's output into answer text and proposed citations.

Tolerant of common formatting slips (a missing closing tag, code fences, trailing commas, an
<answer> wrapped in another <answer>); a citation line that is not valid JSON is dropped and
reported. The output format's own tags never reach the answer text or the notes. Nothing the
model claims is trusted here: citations are only *proposed*, and are verified afterwards.

Markers may combine numbers ("[3, 9]", "[3,9]"): each number is a marker of its own. A citation
number used for two different quotations is a conflict: both quotations are kept (never silently
dropped), and which statement each supports is unknown, so the statements marked with that number
are treated as unsupported downstream. An identical repeat (same passage and quote) adds nothing
and is collapsed, with a note.
"""

import json
import re
from dataclasses import dataclass, field

_ANSWER = re.compile(r"<answer>\s*(.*?)\s*(?:</answer>|<citations>|$)", re.S | re.I)
_CITATIONS = re.compile(r"<citations>\s*(.*?)\s*(?:</citations>|$)", re.S | re.I)
_FENCE = re.compile(r"^```[a-z]*\s*|\s*```$", re.I | re.M)
_EVIDENCE_ID = re.compile(r"^E\d{1,3}$")
# A citation marker: "[3]", or several numbers in one bracket, "[3, 9]" / "[3,9]".
MARKER = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]")


def marker_numbers(text: str) -> list[int]:
    """Every citation number in `text`, in order, combined markers expanded."""
    return [int(number) for group in MARKER.findall(text) for number in group.split(",")]


# The prompt's protocol tags (and evidence labels a model may echo). Only these: document text
# can legitimately contain "<" ("< 5 days"), so arbitrary markup is left to the renderer.
_PROTOCOL_TAG = re.compile(r"</?\s*(?:answer|citations|evidence)\b[^<>]{0,80}>", re.I)


def strip_protocol_tags(text: str) -> str:
    """`text` without the output format's tags, e.g. a nested `<answer>` or a stray
    `</citations>`; whitespace left at the edges is trimmed."""
    return _PROTOCOL_TAG.sub("", text).strip()


@dataclass(frozen=True)
class ProposedCitation:
    n: int
    evidence_id: str
    quote: str


@dataclass
class ParsedOutput:
    answer: str
    citations: list[ProposedCitation] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    truncated: bool = False  # the output stops mid-answer or mid-citation
    # Citation numbers used for more than one different quotation.
    conflicts: set[int] = field(default_factory=set)

    @property
    def markers(self) -> list[int]:
        return sorted(set(marker_numbers(self.answer)))


def parse_output(raw: str) -> ParsedOutput:
    text = raw.strip()
    problems: list[str] = []
    answer_match = _ANSWER.search(text)
    if answer_match:
        answer = answer_match.group(1)
    else:
        problems.append("The model did not use the answer format.")
        answer = re.split(r"<citations>", text, flags=re.I)[0]
    # "<answer><answer>X</answer></answer>" captures "<answer>X": unwrap any nesting.
    answer = strip_protocol_tags(answer)

    # Cut off inside the answer: it opened and nothing that ends it follows.
    truncated = bool(answer_match) and not re.search(r"</answer>|<citations>", text, re.I)

    citations: list[ProposedCitation] = []
    conflicts: set[int] = set()
    block = _CITATIONS.search(text)
    if block:
        body = _FENCE.sub("", block.group(1))
        closed = re.search(r"</citations>", text, re.I) is not None
        lines = [line.strip().rstrip(",") for line in body.splitlines() if line.strip()]
        first: dict[int, ProposedCitation] = {}
        for index, line in enumerate(lines):
            try:
                item = json.loads(line)
                n, evidence_id, quote = int(item["n"]), str(item["evidence"]), str(item["quote"])
            except (ValueError, KeyError, TypeError):
                if not closed and index == len(lines) - 1 and line.startswith("{"):
                    # An unfinished last JSON object with no closing tag: the output was cut off.
                    truncated = True
                problems.append(
                    f"Ignored a malformed citation line: {strip_protocol_tags(line)[:80]}"
                )
                continue
            if not _EVIDENCE_ID.match(evidence_id) or not quote.strip():
                problems.append(f"Ignored citation [{n}]: invalid evidence id or empty quote.")
                continue
            proposed = ProposedCitation(n, evidence_id, quote)
            if n in first:
                earlier = first[n]
                if earlier.evidence_id == evidence_id and _squash(earlier.quote) == _squash(quote):
                    problems.append(
                        f"Citation [{n}] was repeated identically; the repeat was merged."
                    )
                    continue
                conflicts.add(n)  # kept below: never silently dropped
            else:
                first[n] = proposed
            citations.append(proposed)
    return ParsedOutput(
        answer=answer,
        citations=citations,
        problems=problems,
        truncated=truncated,
        conflicts=conflicts,
    )


def _squash(text: str) -> str:
    return " ".join(text.split())
