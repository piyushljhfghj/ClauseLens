"""Candidate generators. Each returns, per document, a ranked list of chunk ids.

All work happens in PostgreSQL over the selected documents only; nothing is loaded into Python
beyond the ranked ids and scores. Every ORDER BY ends with the chunk ordinal so equal scores
always rank the same way (deterministic results for an unchanged index).
"""

import re
import uuid
from dataclasses import dataclass

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class Candidate:
    chunk_id: int
    document_id: uuid.UUID
    ordinal: int
    rank: int  # 1-based, within (retriever, document)
    score: float  # retriever-specific; never compared across retrievers


_DENSE_SQL = text(
    """
    WITH scored AS (
        SELECT id, document_id, ordinal, 1 - (embedding <=> CAST(:query AS vector)) AS similarity
          FROM chunks
         WHERE document_id = ANY(:document_ids)
    ), ranked AS (
        SELECT *, row_number() OVER (
                   PARTITION BY document_id ORDER BY similarity DESC, ordinal) AS rank
          FROM scored
         WHERE similarity >= :min_similarity
    )
    SELECT id, document_id, ordinal, rank, similarity
      FROM ranked WHERE rank <= :limit
     ORDER BY document_id, rank
    """
)


async def dense_candidates(
    session: AsyncSession,
    document_ids: list[uuid.UUID],
    query_vector: np.ndarray,
    *,
    limit: int,
    min_similarity: float,
) -> list[Candidate]:
    """Cosine similarity via pgvector, exact scan (no ANN index: queries are scoped to a few
    documents, and a filtered approximate search could silently miss results)."""
    vector = "[" + ",".join(f"{float(v):.7f}" for v in query_vector) + "]"
    rows = await session.execute(
        _DENSE_SQL,
        {
            "query": vector,
            "document_ids": document_ids,
            "min_similarity": min_similarity,
            "limit": limit,
        },
    )
    return [Candidate(r.id, r.document_id, r.ordinal, r.rank, float(r.similarity)) for r in rows]


# BM25 (Robertson/Sparck Jones, Lucene-style IDF which is always positive), computed per document:
# N and avgdl are that document's chunk count and mean token count, df counts its chunks that
# contain the term. Only chunks containing at least one query term are unnested.
_BM25_SQL = text(
    """
    WITH scope AS (
        SELECT id, document_id, ordinal, lexical_tokens,
               cardinality(lexical_tokens) AS dl
          FROM chunks
         WHERE document_id = ANY(:document_ids)
    ), stats AS (
        SELECT document_id, count(*)::float AS n, greatest(avg(dl), 1)::float AS avgdl
          FROM scope GROUP BY document_id
    ), hits AS (
        SELECT s.id, s.document_id, s.ordinal, s.dl, t.term, count(*)::float AS tf
          FROM scope s, unnest(s.lexical_tokens) AS t(term)
         WHERE s.lexical_tokens && CAST(:terms AS text[]) AND t.term = ANY(CAST(:terms AS text[]))
         GROUP BY s.id, s.document_id, s.ordinal, s.dl, t.term
    ), df AS (
        SELECT document_id, term, count(*)::float AS df FROM hits GROUP BY document_id, term
    ), scored AS (
        SELECT h.id, h.document_id, h.ordinal,
               sum(ln(1 + (st.n - df.df + 0.5) / (df.df + 0.5))
                   * h.tf * (:k1 + 1) / (h.tf + :k1 * (1 - :b + :b * h.dl / st.avgdl))) AS score
          FROM hits h
          JOIN df ON df.document_id = h.document_id AND df.term = h.term
          JOIN stats st ON st.document_id = h.document_id
         GROUP BY h.id, h.document_id, h.ordinal
    ), ranked AS (
        SELECT *, row_number() OVER (PARTITION BY document_id ORDER BY score DESC, ordinal) AS rank
          FROM scored
    )
    SELECT id, document_id, ordinal, rank, score
      FROM ranked WHERE rank <= :limit
     ORDER BY document_id, rank
    """
)


async def bm25_candidates(
    session: AsyncSession,
    document_ids: list[uuid.UUID],
    terms: tuple[str, ...],
    *,
    limit: int,
    k1: float,
    b: float,
) -> list[Candidate]:
    if not terms:
        return []
    rows = await session.execute(
        _BM25_SQL,
        {"document_ids": document_ids, "terms": list(terms), "limit": limit, "k1": k1, "b": b},
    )
    return [Candidate(r.id, r.document_id, r.ordinal, r.rank, float(r.score)) for r in rows]


def phrase_regex(tokens: tuple[str, ...]) -> str:
    """A PostgreSQL regex matching the tokens in order, separated by any non-word characters
    (spaces, line breaks, punctuation), on word boundaries. Case-insensitive via ~*."""
    return r"\m" + r"\W+".join(_token_pattern(token) for token in tokens) + r"\M"


def _token_pattern(token: str) -> str:
    # Query tokens drop thousands separators ("100,000" -> "100000"); the text keeps them.
    if token.isdigit() and len(token) > 3:
        return ",?".join(token)
    return re.escape(token)


_PHRASE_SQL = text(
    """
    WITH body AS (
        -- context_text is the section path, a newline, then the passage (or just the passage
        -- when there is no path). Match the passage only: a phrase found only in the section
        -- heading path is not in the passage text and must not count as an exact-phrase match.
        SELECT id, document_id, ordinal,
               CASE WHEN section_path = '' THEN context_text
                    ELSE substr(context_text, char_length(section_path) + 2) END AS passage
          FROM chunks
         WHERE document_id = ANY(:document_ids)
    ), matches AS (
        SELECT id, document_id, ordinal,
               (SELECT count(*) FROM regexp_matches(passage, :pattern, 'gi'))::float AS hits
          FROM body
         WHERE passage ~* :pattern
    ), ranked AS (
        SELECT *, row_number() OVER (PARTITION BY document_id ORDER BY hits DESC, ordinal) AS rank
          FROM matches
    )
    SELECT id, document_id, ordinal, rank, hits FROM ranked WHERE rank <= :limit
     ORDER BY document_id, rank
    """
)


async def phrase_candidates(
    session: AsyncSession,
    document_ids: list[uuid.UUID],
    phrase: tuple[str, ...],
    *,
    limit: int,
) -> list[Candidate]:
    """Chunks whose passage text contains the exact phrase. The passage part of `context_text`
    is used (not the raw text) because it has ligatures expanded and line-break hyphenation
    joined, like the query tokens; the section path prefix is excluded."""
    rows = await session.execute(
        _PHRASE_SQL,
        {"document_ids": document_ids, "pattern": phrase_regex(phrase), "limit": limit},
    )
    return [Candidate(r.id, r.document_id, r.ordinal, r.rank, float(r.hits)) for r in rows]


_SECTION_SQL = text(
    """
    WITH target AS (
        SELECT document_id, char_start, char_end, ordinal AS section_ordinal
          FROM sections
         WHERE document_id = ANY(:document_ids) AND number_normalized = ANY(:refs)
    ), matches AS (
        SELECT DISTINCT ON (c.id) c.id, c.document_id, c.ordinal, t.section_ordinal
          FROM chunks c
          JOIN target t ON t.document_id = c.document_id
                       AND c.char_start < t.char_end AND t.char_start < c.char_end
         ORDER BY c.id, t.section_ordinal
    ), ranked AS (
        SELECT *, row_number() OVER (PARTITION BY document_id ORDER BY section_ordinal, ordinal)
                  AS rank
          FROM matches
    )
    SELECT id, document_id, ordinal, rank FROM ranked WHERE rank <= :limit
     ORDER BY document_id, rank
    """
)


async def section_candidates(
    session: AsyncSession,
    document_ids: list[uuid.UUID],
    refs: tuple[str, ...],
    *,
    limit: int,
) -> list[Candidate]:
    """Chunks overlapping a section whose number the query names ("clause 8.2"), in reading
    order: the section's own text first, then its subsections."""
    if not refs:
        return []
    rows = await session.execute(
        _SECTION_SQL, {"document_ids": document_ids, "refs": list(refs), "limit": limit}
    )
    return [Candidate(r.id, r.document_id, r.ordinal, r.rank, 1.0) for r in rows]
