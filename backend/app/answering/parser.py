"""Parse the model's output into answer text and proposed citations.

Tolerant of common formatting slips (a missing closing tag, code fences, trailing commas); a
citation line that is not valid JSON is dropped and reported. Nothing the model claims is
trusted here: citations are only *proposed*, and are verified afterwards.
"""

import json
import re
from dataclasses import dataclass, field

_ANSWER = re.compile(r"<answer>\s*(.*?)\s*(?:</answer>|<citations>|$)", re.S | re.I)
_CITATIONS = re.compile(r"<citations>\s*(.*?)\s*(?:</citations>|$)", re.S | re.I)
_FENCE = re.compile(r"^```[a-z]*\s*|\s*```$", re.I | re.M)
_EVIDENCE_ID = re.compile(r"^E\d{1,3}$")
MARKER = re.compile(r"\[(\d{1,3})\]")


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

    @property
    def markers(self) -> list[int]:
        return sorted({int(m) for m in MARKER.findall(self.answer)})


def parse_output(raw: str) -> ParsedOutput:
    text = raw.strip()
    problems: list[str] = []
    answer_match = _ANSWER.search(text)
    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        problems.append("The model did not use the <answer> format.")
        answer = re.split(r"<citations>", text, flags=re.I)[0].strip()

    # Cut off inside the answer: it opened and nothing that ends it follows.
    truncated = bool(answer_match) and not re.search(r"</answer>|<citations>", text, re.I)

    citations: list[ProposedCitation] = []
    block = _CITATIONS.search(text)
    if block:
        body = _FENCE.sub("", block.group(1))
        closed = re.search(r"</citations>", text, re.I) is not None
        lines = [line.strip().rstrip(",") for line in body.splitlines() if line.strip()]
        seen: set[int] = set()
        for index, line in enumerate(lines):
            try:
                item = json.loads(line)
                n, evidence_id, quote = int(item["n"]), str(item["evidence"]), str(item["quote"])
            except (ValueError, KeyError, TypeError):
                if not closed and index == len(lines) - 1 and line.startswith("{"):
                    # An unfinished last JSON object with no closing tag: the output was cut off.
                    truncated = True
                problems.append(f"Ignored a malformed citation line: {line[:80]}")
                continue
            if not _EVIDENCE_ID.match(evidence_id) or not quote.strip():
                problems.append(f"Ignored citation [{n}]: invalid evidence id or empty quote.")
                continue
            if n in seen:
                problems.append(f"Ignored a duplicate citation number [{n}].")
                continue
            seen.add(n)
            citations.append(ProposedCitation(n, evidence_id, quote))
    return ParsedOutput(answer=answer, citations=citations, problems=problems, truncated=truncated)
