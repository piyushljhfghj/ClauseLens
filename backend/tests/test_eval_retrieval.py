"""Deterministic retrieval evaluation for answer quality (no model calls).

Runs every case of tests/eval/cases.py, and the frozen blind paraphrases of each, against the
synthetic journal with the real embedding model, through exactly the calls the answer service
makes (retrieval with the answer's candidate count, then evidence selection). Candidate
retrieval and the final selection are measured separately. A report (JSON and Markdown) is
written outside the repository: $CLAUSELENS_EVAL_DIR, or <temp dir>/clauselens-eval.

The test locks the measured results. Any change, better or worse, fails it until the expected
values below are updated with a new, reviewed measurement, so a retrieval change can never
alter the outcome silently. A passing test does not mean the system is accurate.
"""

import json
import os
import re
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from sqlalchemy import select

from app.answering.context import estimate_tokens
from app.answering.service import AnswerService
from app.core.config import Settings
from app.db.models import DocumentBlock, DocumentChunk
from app.db.session import get_sessionmaker
from app.embeddings import FastEmbedEmbedder
from app.main import create_app
from app.schemas.retrieval import RetrievedPassage
from tests.conftest import truncate_all
from tests.eval.cases import CASES, PARAPHRASES
from tests.fixtures.journal import journal_docx
from tests.test_retrieval import upload

# Measured results, recorded facts rather than targets.
# Step 1b (small-document mode): the compact journal (about 1,292 tokens) is given whole for
# all 12 original questions and 39 of 40 paraphrases; the probe's evidence losses fall from 19
# to 1 (C08.p2: the search returns nothing, so the model is not asked). The padded journal
# (about 5,181 tokens) never qualifies and is unchanged.
# Step 1a (passages made only of headings excluded from evidence): heading-only / no-content
# selected passages 6 -> 0 (compact) and 5 -> 0 (padded); evidence recall and the probe's
# losses unchanged. Step 0 baseline (before any change) was:
# - original cases whose evidence is not all selected: none (the synthetic journal did not
#   reproduce the real journal's lost evidence; its strengths passage scored 0.619 vs 0.60);
# - selected passages whose text is just their own heading (text rule): 6 compact, 5 padded;
# - selected passages made only of heading blocks (stored block kinds): 6 compact, 5 padded;
# - paraphrases whose evidence is not all selected (blind probe baseline).
# 19 of the 40 blind paraphrases lose evidence at the baseline, in both variants.
BASELINE_PROBE_LOST = {
    "C01.p2",
    "C01.p3",
    "C01.p4",
    "C02.p2",
    "C02.p3",
    "C03.p2",
    "C05.p4",
    "C06.p2",
    "C06.p3",
    "C06.p4",
    "C08.p2",
    "C08.p4",
    "C09.p1",
    "C09.p2",
    "C09.p3",
    "C09.p4",
    "C12.p1",
    "C12.p2",
    "C12.p3",
}
EXPECTED = {
    "compact": {
        "whole_document_questions": (12, 39),
        "not_fully_selected": set[str](),
        "selected_heading_only": 0,
        "selected_no_content": 0,
        "probe_not_fully_selected": {"C08.p2"},
    },
    "padded": {
        "whole_document_questions": (0, 0),
        "not_fully_selected": set[str](),
        "selected_heading_only": 0,
        "selected_no_content": 0,
        "probe_not_fully_selected": BASELINE_PROBE_LOST,
    },
}


def _squash(text: str) -> str:
    return " ".join(text.split())


def _alnum(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", text.lower())


def heading_only(passage: RetrievedPassage) -> bool:
    """The passage's text is nothing but its own section heading (the Step 0 text rule)."""
    label = passage.section_path.split(" > ")[-1]
    return bool(label) and _alnum(passage.text) == _alnum(label)


@dataclass
class PassageRecord:
    rank: int
    chunk_ordinal: int
    section: str
    chars: int
    heading_only: bool
    no_content: bool  # every stored block of the passage is a heading
    dense_similarity: float | None
    bm25_score: float | None
    exact_signal: bool
    selected: bool
    exclusion: str | None


@dataclass
class EvidenceRecord:
    sentence: str
    candidate_rank: int | None  # rank of the retrieved passage containing it
    selected: bool
    lost_at: str | None  # None, "not retrieved", or the selection's exclusion reason


@dataclass
class QuestionResult:
    label: str  # "C02" or "C02.p3" for a paraphrase
    case_id: str
    category: str
    expect: str
    question: str
    search_terms: list[str]
    whole_documents: bool = False  # small-document mode: every meaningful passage selected
    selected_ordinals: list[int] = field(default_factory=list)  # what the model is given
    selected_no_content: int = 0
    candidates: list[PassageRecord] = field(default_factory=list)
    evidence: list[EvidenceRecord] = field(default_factory=list)

    @property
    def fully_selected(self) -> bool | None:
        return all(e.selected for e in self.evidence) if self.evidence else None


def _reason_kind(reason: str) -> str:
    if reason.startswith("weak semantic"):
        return "weak semantic-only"
    if reason.startswith("weak keyword"):
        return "weak keyword-only"
    return reason  # e.g. "passage limit reached", "context budget reached"


@dataclass(frozen=True)
class StoredDocument:
    id: uuid.UUID
    texts: list[str]
    ordinals: dict[int, int]  # chunk id -> chunk ordinal
    no_content: set[int]  # chunk ids made only of heading blocks


async def _stored(document_id: uuid.UUID) -> StoredDocument:
    async with get_sessionmaker()() as session:
        chunks = (
            await session.execute(
                select(
                    DocumentChunk.id,
                    DocumentChunk.ordinal,
                    DocumentChunk.block_start,
                    DocumentChunk.block_end,
                    DocumentChunk.text,
                ).where(DocumentChunk.document_id == document_id)
            )
        ).all()
        kinds = dict(
            (
                await session.execute(
                    select(DocumentBlock.ordinal, DocumentBlock.kind).where(
                        DocumentBlock.document_id == document_id
                    )
                )
            ).all()
        )
        await session.rollback()
    no_content = {
        c.id
        for c in chunks
        if all(kinds.get(o) == "heading" for o in range(c.block_start, c.block_end + 1))
    }
    return StoredDocument(
        document_id, [c.text for c in chunks], {c.id: c.ordinal for c in chunks}, no_content
    )


async def _measure(
    label: str,
    case_id: str,
    question: str,
    stored: StoredDocument,
    embedder: FastEmbedEmbedder,
    settings: Settings,
) -> QuestionResult:
    case = next(c for c in CASES if c.id == case_id)
    # The answer service's own evidence selection, exactly as it runs before calling the model.
    async with get_sessionmaker()() as session:
        service = AnswerService(session, embedder, None, settings)
        retrieval, selection = await service.select_evidence(question, [stored.id])
        await session.rollback()
    chosen = {p.chunk_id for p in selection.selected}
    reasons = {p.chunk_id: reason for p, reason in selection.excluded}

    result = QuestionResult(
        label, case.id, case.category, case.expect, question, list(retrieval.analysis.terms)
    )
    result.whole_documents = selection.whole_documents
    result.selected_ordinals = [stored.ordinals[p.chunk_id] for p in selection.selected]
    result.selected_no_content = sum(p.chunk_id in stored.no_content for p in selection.selected)
    for p in retrieval.passages:
        s = p.signals
        result.candidates.append(
            PassageRecord(
                rank=p.rank,
                chunk_ordinal=stored.ordinals[p.chunk_id],
                section=p.section_path,
                chars=len(p.text),
                heading_only=heading_only(p),
                no_content=p.chunk_id in stored.no_content,
                dense_similarity=round(s.dense_similarity, 3) if s.dense_similarity else None,
                bm25_score=round(s.bm25_score, 2) if s.bm25_score else None,
                exact_signal=s.phrase_rank is not None or s.section_rank is not None,
                selected=p.chunk_id in chosen,
                exclusion=reasons.get(p.chunk_id),
            )
        )
    for sentence in case.evidence:
        # Candidate retrieval and the final selection, measured separately: in small-document
        # mode the model is also given passages the search did not return.
        holder = next((p for p in retrieval.passages if _squash(sentence) in _squash(p.text)), None)
        selected = any(_squash(sentence) in _squash(p.text) for p in selection.selected)
        if selected:
            lost = None
        elif holder is None:
            lost = "not retrieved"
        else:
            lost = _reason_kind(reasons.get(holder.chunk_id, "not selected"))
        rank = holder.rank if holder else None
        result.evidence.append(EvidenceRecord(sentence, rank, selected, lost))
    return result


def _aggregate(results: list[QuestionResult]) -> dict[str, object]:
    pairs = [e for r in results for e in r.evidence]
    with_evidence = [r for r in results if r.evidence]
    candidates = [c for r in results for c in r.candidates]
    exclusions: dict[str, int] = {}
    for c in candidates:
        if c.exclusion:
            kind = _reason_kind(c.exclusion)
            exclusions[kind] = exclusions.get(kind, 0) + 1
    lost: dict[str, int] = {}
    for e in pairs:
        if e.lost_at:
            lost[e.lost_at] = lost.get(e.lost_at, 0) + 1
    return {
        "questions": len(results),
        "evidence_sentences": len(pairs),
        "candidate_recall": round(sum(e.candidate_rank is not None for e in pairs) / len(pairs), 3),
        "selection_recall": round(sum(e.selected for e in pairs) / len(pairs), 3),
        "questions_with_evidence": len(with_evidence),
        "questions_fully_selected": sum(bool(r.fully_selected) for r in with_evidence),
        "not_fully_selected": sorted(r.label for r in with_evidence if not r.fully_selected),
        "evidence_lost_at": lost,
        "whole_document_questions": sum(r.whole_documents for r in results),
        "selected_passages": sum(len(r.selected_ordinals) for r in results),
        "selected_heading_only": sum(c.heading_only for c in candidates if c.selected),
        "selected_no_content": sum(r.selected_no_content for r in results),
        "candidate_no_content": sum(c.no_content for c in candidates),
        "exclusions_by_reason": exclusions,
    }


def _markdown(title: str, summary: dict[str, object], results: list[QuestionResult]) -> str:
    lines = [
        f"# {title}",
        "",
        "| Question | Expect | Evidence in candidates | Evidence selected | Lost at "
        "| Selected (no content) |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        found = sum(e.candidate_rank is not None for e in r.evidence)
        chosen = sum(e.selected for e in r.evidence)
        lost = "; ".join(sorted({e.lost_at for e in r.evidence if e.lost_at})) or "-"
        n = len(r.evidence)
        mode = " whole" if r.whole_documents else ""
        lines.append(
            f"| {r.label} {r.category} | {r.expect} | {f'{found}/{n}' if n else 'n/a'} | "
            f"{f'{chosen}/{n}' if n else 'n/a'} | {lost} | "
            f"{len(r.selected_ordinals)} ({r.selected_no_content}){mode} |"
        )
    lines += ["", "```json", json.dumps(summary, indent=2), "```", ""]
    return "\n".join(lines)


def _report_dir() -> Path:
    base = os.environ.get("CLAUSELENS_EVAL_DIR") or str(
        Path(tempfile.gettempdir()) / "clauselens-eval"
    )
    path = Path(base)
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.mark.parametrize("variant", ["compact", "padded"])
async def test_retrieval_evaluation(db_settings: Settings, variant: str) -> None:
    embedder = FastEmbedEmbedder(db_settings.embedding_model, cache_dir=".cache/fastembed")
    app = create_app(db_settings, embedder=embedder)
    async with app.router.lifespan_context(app):
        await truncate_all()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as api:
            doc_id = await upload(
                api, app, "journal.docx", journal_docx(padded=variant == "padded")
            )
        stored = await _stored(uuid.UUID(doc_id))
        # The fixture is valid only if every evidence sentence lies inside one stored passage.
        for case in CASES:
            for sentence in case.evidence:
                holders = [t for t in stored.texts if _squash(sentence) in _squash(t)]
                assert len(holders) == 1, f"{case.id}: evidence not inside exactly one passage"

        originals = [
            await _measure(c.id, c.id, c.question, stored, embedder, db_settings) for c in CASES
        ]
        probe = [
            await _measure(f"{case_id}.p{i}", case_id, question, stored, embedder, db_settings)
            for case_id, questions in PARAPHRASES.items()
            for i, question in enumerate(questions, start=1)
        ]

    tokens = sum(len(text) // 4 + 30 for text in stored.texts)  # as estimate_tokens
    summary = {
        "variant": variant,
        "document_tokens_estimate": tokens,
        "context_budget_tokens": db_settings.answer_context_token_budget,
        "original_cases": _aggregate(originals),
        "paraphrase_probe": _aggregate(probe),
    }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = _report_dir()
    payload = {
        "summary": summary,
        "original_cases": [asdict(r) for r in originals],
        "paraphrase_probe": [asdict(r) for r in probe],
    }
    (out / f"retrieval-{variant}-{stamp}.json").write_text(json.dumps(payload, indent=2), "utf-8")
    report = _markdown(f"Original cases: {variant} synthetic journal", summary, originals)
    report += "\n" + _markdown(f"Paraphrase probe: {variant}", summary["paraphrase_probe"], probe)
    (out / f"retrieval-{variant}-{stamp}.md").write_text(report, "utf-8")
    print(f"\nreport: {out / f'retrieval-{variant}-{stamp}.md'}")

    o, p = summary["original_cases"], summary["paraphrase_probe"]
    measured = {
        "whole_document_questions": (o["whole_document_questions"], p["whole_document_questions"]),
        "not_fully_selected": set(o["not_fully_selected"]),
        "selected_heading_only": o["selected_heading_only"],
        "selected_no_content": o["selected_no_content"],
        "probe_not_fully_selected": set(p["not_fully_selected"]),
    }
    assert measured == EXPECTED[variant], (
        "The retrieval evaluation changed; update EXPECTED only with a measured, reviewed "
        f"result. Now: {measured}"
    )


def test_estimate_matches_the_selection_budget_estimate() -> None:
    # The report's document size uses the same estimate as evidence selection.
    passage = type("P", (), {"text": "x" * 400})()
    assert estimate_tokens(passage) == 400 // 4 + 30  # type: ignore[arg-type]
