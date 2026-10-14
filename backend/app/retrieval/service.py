"""Hybrid retrieval over the selected documents.

  query analysis -> (per document) dense | BM25 | exact phrase | clause reference candidates
  -> Reciprocal Rank Fusion -> near-duplicate removal -> top-k passages with source metadata

Retrieval only finds passages that are *likely relevant*. It does not establish that a passage
supports any claim; that is the job of answer generation plus independent citation
verification in later phases.
"""

import asyncio
import uuid
from collections import defaultdict

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.errors import AppError, ConflictError, NotFoundError
from app.db.models import Document, DocumentChunk, DocumentPage
from app.embeddings import Embedder
from app.ingestion.model import LOW_QUALITY_NOTE
from app.ingestion.pipeline import INDEX_VERSION
from app.retrieval.candidates import (
    Candidate,
    bm25_candidates,
    dense_candidates,
    phrase_candidates,
    section_candidates,
)
from app.retrieval.fusion import FusedCandidate, reciprocal_rank_fusion, remove_near_duplicates
from app.retrieval.query import AnalyzedQuery, analyze_query
from app.schemas.retrieval import (
    DocumentCoverage,
    PassageQuality,
    QueryAnalysisOut,
    RetrievalResponse,
    RetrievalSignals,
    RetrievedPassage,
)

# Fetch more fused candidates than requested, so near-duplicate removal can still fill top_k.
DEDUP_HEADROOM = 3


class RetrievalService:
    def __init__(self, session: AsyncSession, embedder: Embedder, settings: Settings) -> None:
        self._session = session
        self._embedder = embedder
        self._settings = settings

    async def search(
        self, query: str, document_ids: list[uuid.UUID], top_k: int | None = None
    ) -> RetrievalResponse:
        settings = self._settings
        if len(query) > settings.retrieval_max_query_chars:
            raise AppError(
                f"The query is too long (maximum {settings.retrieval_max_query_chars} characters).",
                code="QUERY_TOO_LONG",
                status_code=422,
            )
        if len(document_ids) > settings.retrieval_max_documents:
            raise AppError(
                f"Select at most {settings.retrieval_max_documents} documents.",
                code="TOO_MANY_DOCUMENTS",
                status_code=422,
            )
        top_k = top_k or settings.retrieval_default_top_k
        if top_k > settings.retrieval_max_top_k:
            raise AppError(
                f"top_k may be at most {settings.retrieval_max_top_k}.",
                code="TOP_K_TOO_LARGE",
                status_code=422,
            )

        # One consistent snapshot for every read below: a re-index committing mid-search can
        # neither remove a chunk between "rank it" and "fetch it" nor mix two index versions.
        await self._session.connection(
            execution_options={"isolation_level": "REPEATABLE READ", "postgresql_readonly": True}
        )
        documents = await self._load_scope(document_ids)
        coverage = await self._coverage(documents)
        analysis = analyze_query(query)
        notes = list(analysis.notes)

        passages: list[RetrievedPassage] = []
        if analysis.has_content:
            passages, removed = await self._retrieve(analysis, documents, top_k)
            if removed:
                notes.append(f"{removed} near-duplicate passage(s) were omitted.")

        for doc in coverage:
            if doc.unreadable_page_count:
                notes.append(
                    f"'{doc.filename}': {doc.unreadable_page_count} page(s) could not be read and "
                    "were not searched."
                )
            if doc.low_quality_page_count:
                # Also (especially) when nothing matched: a miss may mean OCR did not capture
                # the text, not that the document does not contain it.
                notes.append(
                    f"'{doc.filename}': {doc.low_quality_page_count} page(s) were read with low "
                    "OCR quality; relevant content on them may be missing from these results."
                )
        if any(p.quality.level == "low_quality_ocr" for p in passages):
            notes.append(
                "Some passages come from low-quality OCR pages: their text may be largely "
                "unrecognised and must be checked against the original."
            )
        return RetrievalResponse(
            query=analysis.text,
            passages=passages,
            no_match=not passages,
            coverage=coverage,
            analysis=QueryAnalysisOut(
                terms=list(analysis.terms),
                phrases=[" ".join(p) for p in analysis.phrases],
                section_refs=list(analysis.section_refs),
            ),
            notes=notes,
        )

    # --- scope ----------------------------------------------------------------------------------

    async def _load_scope(self, document_ids: list[uuid.UUID]) -> dict[uuid.UUID, Document]:
        rows = await self._session.execute(select(Document).where(Document.id.in_(document_ids)))
        documents = {d.id: d for d in rows.scalars()}
        missing = [str(i) for i in document_ids if i not in documents]
        if missing:
            raise NotFoundError(
                "Document not found.", code="DOCUMENT_NOT_FOUND", details={"document_ids": missing}
            )
        stale = [
            str(d.id)
            for d in documents.values()
            if d.status != "ready"
            or d.index_version != INDEX_VERSION
            or d.embedding_model != self._embedder.model_name
        ]
        if stale:
            raise ConflictError(
                "Some selected documents are not indexed with the current pipeline. Reprocess "
                "them before searching.",
                code="DOCUMENT_NOT_INDEXED",
                details={"document_ids": stale},
            )
        # In the caller's order, so coverage is reported in the order the documents were selected.
        return {i: documents[i] for i in document_ids}

    async def _coverage(self, documents: dict[uuid.UUID, Document]) -> list[DocumentCoverage]:
        counts = dict(
            (
                await self._session.execute(
                    select(DocumentChunk.document_id, func.count())
                    .where(DocumentChunk.document_id.in_(list(documents)))
                    .group_by(DocumentChunk.document_id)
                )
            ).all()
        )
        return [
            DocumentCoverage(
                document_id=d.id,
                filename=d.filename,
                chunks_searched=counts.get(d.id, 0),
                page_count=d.page_count,
                unreadable_page_count=d.unreadable_page_count,
                low_quality_page_count=d.low_quality_page_count,
            )
            for d in documents.values()
        ]

    # --- retrieval ------------------------------------------------------------------------------

    async def _retrieve(
        self, analysis: AnalyzedQuery, documents: dict[uuid.UUID, Document], top_k: int
    ) -> tuple[list[RetrievedPassage], int]:
        settings = self._settings
        ids = list(documents)
        limit = settings.retrieval_candidates_per_list

        vector = await asyncio.to_thread(self._embedder.embed_query, analysis.text)
        lists: dict[str, list[Candidate]] = {
            "dense": await dense_candidates(
                self._session,
                ids,
                vector,
                limit=limit,
                min_similarity=settings.retrieval_min_dense_similarity,
            ),
            "bm25": await bm25_candidates(
                self._session,
                ids,
                analysis.terms,
                limit=limit,
                k1=settings.retrieval_bm25_k1,
                b=settings.retrieval_bm25_b,
            ),
            "section": await section_candidates(
                self._session, ids, analysis.section_refs, limit=limit
            ),
        }
        for i, phrase in enumerate(analysis.phrases):
            lists[f"phrase{i}"] = await phrase_candidates(self._session, ids, phrase, limit=limit)

        fused = reciprocal_rank_fusion(lists, k=settings.retrieval_rrf_k)
        if not fused:
            return [], 0
        head = fused[: top_k * DEDUP_HEADROOM]
        chunks = {
            c.id: c
            for c in (
                await self._session.execute(
                    select(DocumentChunk).where(DocumentChunk.id.in_([f.chunk_id for f in head]))
                )
            ).scalars()
        }
        head = [f for f in head if f.chunk_id in chunks]  # defensive; same snapshot
        kept, removed = remove_near_duplicates(head, {i: c.text for i, c in chunks.items()})
        kept = kept[:top_k]
        quality = await self._quality([chunks[f.chunk_id] for f in kept])
        return [
            self._passage(rank, f, chunks[f.chunk_id], documents, quality)
            for rank, f in enumerate(kept, start=1)
        ], removed

    async def _quality(self, chunks: list[DocumentChunk]) -> dict[int, PassageQuality]:
        """Extraction quality of each passage's pages, from the stored per-page records."""
        wanted: dict[uuid.UUID, set[int]] = defaultdict(set)
        for chunk in chunks:
            if chunk.page_start is not None and chunk.page_end is not None:
                wanted[chunk.document_id].update(range(chunk.page_start, chunk.page_end + 1))
        pages: dict[tuple[uuid.UUID, int], DocumentPage] = {}
        for document_id, numbers in wanted.items():
            rows = await self._session.execute(
                select(DocumentPage).where(
                    DocumentPage.document_id == document_id,
                    DocumentPage.page_number.in_(sorted(numbers)),
                )
            )
            pages.update({(document_id, p.page_number): p for p in rows.scalars()})

        result: dict[int, PassageQuality] = {}
        for chunk in chunks:
            if chunk.page_start is None or chunk.page_end is None:
                result[chunk.id] = PassageQuality(level="not_applicable")
                continue
            own = [
                pages[(chunk.document_id, n)]
                for n in range(chunk.page_start, chunk.page_end + 1)
                if (chunk.document_id, n) in pages
            ]
            ocr = [p for p in own if p.extraction == "ocr"]
            low = [p.page_number for p in own if p.note == LOW_QUALITY_NOTE]
            confidences = [p.ocr_confidence for p in ocr if p.ocr_confidence is not None]
            result[chunk.id] = PassageQuality(
                level="low_quality_ocr" if low else "ocr" if ocr else "text",
                ocr_pages=[p.page_number for p in ocr],
                low_quality_pages=low,
                min_ocr_confidence=min(confidences) if confidences else None,
            )
        return result

    @staticmethod
    def _passage(
        rank: int,
        fused: FusedCandidate,
        chunk: DocumentChunk,
        documents: dict[uuid.UUID, Document],
        quality: dict[int, PassageQuality],
    ) -> RetrievedPassage:
        phrase_ranks = [r for name, r in fused.ranks.items() if name.startswith("phrase")]
        return RetrievedPassage(
            rank=rank,
            score=round(fused.score, 6),
            document_id=chunk.document_id,
            document_filename=documents[chunk.document_id].filename,
            chunk_id=chunk.id,
            chunk_ordinal=chunk.ordinal,
            section_id=chunk.section_id,
            section_path=chunk.section_path,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            text=chunk.text,
            signals=RetrievalSignals(
                dense_rank=fused.ranks.get("dense"),
                dense_similarity=(
                    round(fused.scores["dense"], 4) if "dense" in fused.scores else None
                ),
                bm25_rank=fused.ranks.get("bm25"),
                bm25_score=round(fused.scores["bm25"], 4) if "bm25" in fused.scores else None,
                phrase_rank=min(phrase_ranks) if phrase_ranks else None,
                section_rank=fused.ranks.get("section"),
            ),
            quality=quality[chunk.id],
        )
