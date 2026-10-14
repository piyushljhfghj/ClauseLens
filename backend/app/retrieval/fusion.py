"""Reciprocal Rank Fusion and de-duplication: pure functions, no database.

Why RRF: BM25 scores (unbounded, corpus-dependent) and cosine similarities (bounded, model-
dependent) are not on comparable scales, so adding them would need calibration we do not have.
RRF uses only ranks: score(c) = sum over retrievers of 1 / (k + rank_r(c)). A chunk ranked well
by several retrievers rises; one strong signal alone still counts. k = 60 is the standard value
from Cormack, Clarke & Buettcher (2009).

Ranks are per (retriever, document), so with several documents each document's best chunks
compete on equal terms instead of a long document crowding out a short one.
"""

import re
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from app.retrieval.candidates import Candidate


@dataclass
class FusedCandidate:
    chunk_id: int
    document_id: uuid.UUID
    ordinal: int
    score: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)  # retriever -> rank
    scores: dict[str, float] = field(default_factory=dict)  # retriever -> raw score

    @property
    def best_rank(self) -> int:
        return min(self.ranks.values())


def reciprocal_rank_fusion(
    lists: Mapping[str, Iterable[Candidate]], *, k: int
) -> list[FusedCandidate]:
    fused: dict[int, FusedCandidate] = {}
    for retriever in sorted(lists):  # sorted: iteration order never affects the result
        for candidate in lists[retriever]:
            entry = fused.setdefault(
                candidate.chunk_id,
                FusedCandidate(candidate.chunk_id, candidate.document_id, candidate.ordinal),
            )
            if retriever in entry.ranks:  # a retriever lists a chunk once; keep its best rank
                continue
            entry.ranks[retriever] = candidate.rank
            entry.scores[retriever] = candidate.score
            entry.score += 1.0 / (k + candidate.rank)
    # Deterministic order: fused score, then agreement (more retrievers), then best single rank,
    # then position in the document. Document id breaks any remaining tie.
    return sorted(
        fused.values(),
        key=lambda c: (-c.score, -len(c.ranks), c.best_rank, str(c.document_id), c.ordinal),
    )


_WS = re.compile(r"\s+")
_WORD = re.compile(r"\w+")
NEAR_DUPLICATE_JACCARD = 0.9


def _signature(text: str) -> tuple[str, frozenset[str]]:
    normalised = _WS.sub(" ", text).strip().lower()
    return normalised, frozenset(_WORD.findall(normalised))


def remove_near_duplicates(
    ranked: list[FusedCandidate], texts: Mapping[int, str]
) -> tuple[list[FusedCandidate], int]:
    """Drop lower-ranked chunks that repeat a higher-ranked chunk *of the same document*
    (identical or >= 90% word overlap, e.g. boilerplate repeated on several pages). Matching
    chunks in *different* documents are kept: in a multi-document question, two contract versions
    containing the same clause is information, not noise. Returns (kept, removed_count)."""
    kept: list[FusedCandidate] = []
    seen: dict[uuid.UUID, list[tuple[str, frozenset[str]]]] = {}
    removed = 0
    for candidate in ranked:
        normalised, words = _signature(texts[candidate.chunk_id])
        previous = seen.setdefault(candidate.document_id, [])
        duplicate = any(
            normalised == other
            or (
                words
                and other_words
                and len(words & other_words) / len(words | other_words) >= NEAR_DUPLICATE_JACCARD
            )
            for other, other_words in previous
        )
        if duplicate:
            removed += 1
            continue
        previous.append((normalised, words))
        kept.append(candidate)
    return kept, removed
