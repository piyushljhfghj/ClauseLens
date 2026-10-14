"""Evidence selection: which retrieved passages the model may read.

Retrieval returns a ranked list that always fills top-k; the tail is often only weakly related
(one retriever, low similarity). Sending it to the model invites answers built on irrelevant
text, so a passage is kept only on strong signals:

- an exact phrase or the named clause matched, or
- the semantic and keyword retrievers agree (both ranked it), or
- semantic-only: similarity >= max(floor, best similarity in the result - margin), or
- keyword-only: BM25 score >= ratio x the best BM25 score in the same document AND the passage
  contains at least two distinct search terms of the question (one if the question has only
  one): a single shared word ("student") is not enough, even when it is the only keyword hit.
  A count, not a fraction, so each part of a multi-part question can still be answered.

Kept passages are then packed into a token budget: each document's best passage first (so a
multi-document question sees every relevant document), then the rest in rank order.
No reranker is used: ranking is Phase 4's Reciprocal Rank Fusion.
"""

import uuid
from dataclasses import dataclass

from app.core.config import Settings
from app.schemas.retrieval import RetrievedPassage
from app.text.tokenize import lexical_tokens

CHARS_PER_TOKEN = 4
MIN_MATCHED_TERMS = 2
HEADER_TOKENS = 30  # evidence label, document name, section, pages, quality line


@dataclass(frozen=True)
class Selection:
    selected: list[RetrievedPassage]  # in the order they are shown to the model (E1, E2, ...)
    excluded: list[tuple[RetrievedPassage, str]]  # passage, reason
    tokens_used: int


def estimate_tokens(passage: RetrievedPassage) -> int:
    return len(passage.text) // CHARS_PER_TOKEN + HEADER_TOKENS


def strength_reason(
    passage: RetrievedPassage,
    *,
    best_similarity: float | None,
    best_bm25_by_document: dict[uuid.UUID, float],
    query_terms: tuple[str, ...],
    settings: Settings,
) -> str | None:
    """None if the passage is strong enough; otherwise why it is excluded."""
    s = passage.signals
    if s.phrase_rank is not None or s.section_rank is not None:
        return None
    if s.dense_rank is not None and s.bm25_rank is not None:
        return None
    if s.dense_similarity is not None:
        floor = settings.answer_min_dense_similarity
        relative = (best_similarity or 0.0) - settings.answer_dense_margin
        if s.dense_similarity >= max(floor, relative):
            return None
        return (
            f"weak semantic-only match (similarity {s.dense_similarity:.3f} below "
            f"{max(floor, relative):.3f})"
        )
    if s.bm25_score is not None:
        best = best_bm25_by_document.get(passage.document_id) or 0.0
        threshold = settings.answer_bm25_relative * best
        if s.bm25_score < threshold:
            return f"weak keyword-only match (BM25 {s.bm25_score:.2f} below {threshold:.2f})"
        matched = matched_terms(passage, query_terms)
        required = min(MIN_MATCHED_TERMS, len(query_terms))
        if matched < required:
            return f"weak keyword-only match ({matched} of the question's {len(query_terms)} terms)"
        return None
    return "no retrieval signal"


def matched_terms(passage: RetrievedPassage, query_terms: tuple[str, ...]) -> int:
    """How many distinct search terms of the question occur in the passage (or its section
    path, which BM25 also indexes)."""
    tokens = set(lexical_tokens(f"{passage.section_path} {passage.text}"))
    return sum(term in tokens for term in query_terms)


def select_context(
    passages: list[RetrievedPassage], settings: Settings, query_terms: tuple[str, ...] = ()
) -> Selection:
    similarities = [p.signals.dense_similarity for p in passages if p.signals.dense_similarity]
    best_similarity = max(similarities) if similarities else None
    best_bm25: dict[uuid.UUID, float] = {}
    for p in passages:
        if p.signals.bm25_score is not None:
            best_bm25[p.document_id] = max(best_bm25.get(p.document_id, 0.0), p.signals.bm25_score)

    strong: list[RetrievedPassage] = []
    excluded: list[tuple[RetrievedPassage, str]] = []
    for passage in passages:  # already in rank order
        reason = strength_reason(
            passage,
            best_similarity=best_similarity,
            best_bm25_by_document=best_bm25,
            query_terms=query_terms,
            settings=settings,
        )
        if reason is None:
            strong.append(passage)
        else:
            excluded.append((passage, reason))

    # Each document's best passage first, then everything else, both in rank order.
    firsts: dict[uuid.UUID, RetrievedPassage] = {}
    for passage in strong:
        firsts.setdefault(passage.document_id, passage)
    ordered = list(firsts.values()) + [p for p in strong if p not in firsts.values()]

    selected: list[RetrievedPassage] = []
    used = 0
    for passage in ordered:
        cost = estimate_tokens(passage)
        if len(selected) >= settings.answer_max_passages:
            excluded.append((passage, "passage limit reached"))
        elif used + cost > settings.answer_context_token_budget:
            excluded.append((passage, "context budget reached"))
        else:
            selected.append(passage)
            used += cost
    return Selection(selected=selected, excluded=excluded, tokens_used=used)
