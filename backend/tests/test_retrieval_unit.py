"""Retrieval building blocks that need no database: query analysis, RRF, de-duplication."""

import itertools
import re
import uuid

import pytest

from app.retrieval.candidates import Candidate, phrase_regex
from app.retrieval.fusion import reciprocal_rank_fusion, remove_near_duplicates
from app.retrieval.query import analyze_query

DOC_A = uuid.UUID("00000000-0000-0000-0000-00000000000a")
DOC_B = uuid.UUID("00000000-0000-0000-0000-00000000000b")
LDQ, RDQ = chr(0x201C), chr(0x201D)  # curly double quotes


def cand(chunk_id: int, rank: int, document: uuid.UUID = DOC_A, ordinal: int | None = None):  # type: ignore[no-untyped-def]
    return Candidate(chunk_id, document, chunk_id if ordinal is None else ordinal, rank, 0.0)


# --- query analysis -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "terms", "refs"),
    [
        ("What is the governing law?", ("governing", "law"), ()),
        ("clause 8.2 termination", ("clause", "8.2", "termination"), ("8.2",)),
        ("What does Article IV say about payment?", ("article", "iv", "payment"), ("article 4",)),
        ("Schedule 2", ("schedule", "2"), ("schedule 2",)),
        ("liability cap AED 100,000", ("liability", "cap", "aed", "100000"), ()),
        ("ACME-0000123", ("acme", "0000123"), ()),
        ("Section 12.3.1 and 8.2", ("section", "12.3.1", "8.2"), ("12.3.1", "8.2")),
    ],
)
def test_query_terms_and_section_refs(query: str, terms: tuple, refs: tuple) -> None:  # type: ignore[type-arg]
    analysis = analyze_query(query)
    assert analysis.terms == terms
    assert analysis.section_refs == refs
    assert analysis.has_content


@pytest.mark.parametrize("query", ["the of a", "???", "what is it", "   ...   "])
def test_queries_without_searchable_content(query: str) -> None:
    analysis = analyze_query(query)
    assert not analysis.has_content
    assert analysis.notes == ["The query contains no searchable terms."]


def test_quoted_phrases_straight_and_curly() -> None:
    analysis = analyze_query(
        f'Is there a "material adverse change" or {LDQ}force majeure{RDQ} clause?'
    )
    assert analysis.phrases == (("material", "adverse", "change"), ("force", "majeure"))


def test_terms_are_deduplicated_and_order_preserved() -> None:
    assert analyze_query("fee fee late fee").terms == ("fee", "late")


# --- phrase pattern -----------------------------------------------------------------------------


def _pg_to_python(pattern: str) -> str:
    return pattern.replace(r"\m", r"\b").replace(r"\M", r"\b")


@pytest.mark.parametrize(
    ("tokens", "text", "matches"),
    [
        (("thirty", "30", "days"), "within thirty (30)\ndays of", True),
        (("aed", "100000"), "capped at AED 100,000.", True),
        (("aed", "100000"), "capped at AED 1,000,000.", False),
        (("force", "majeure"), "Force\n\nMajeure event", True),
        (("adverse", "change"), "adverse changes", False),  # word boundary
    ],
)
def test_phrase_regex(tokens: tuple[str, ...], text: str, matches: bool) -> None:
    found = re.search(_pg_to_python(phrase_regex(tokens)), text, re.I) is not None
    assert found is matches


# --- reciprocal rank fusion ---------------------------------------------------------------------


def test_rrf_scores_are_sums_of_reciprocal_ranks() -> None:
    fused = reciprocal_rank_fusion(
        {"dense": [cand(1, 1), cand(2, 2)], "bm25": [cand(2, 1), cand(3, 2)]}, k=60
    )
    scores = {f.chunk_id: f.score for f in fused}
    assert scores[2] == pytest.approx(1 / 62 + 1 / 61)
    assert scores[1] == pytest.approx(1 / 61)
    assert scores[3] == pytest.approx(1 / 62)
    assert [f.chunk_id for f in fused] == [2, 1, 3]  # agreement across retrievers wins
    assert fused[0].ranks == {"dense": 2, "bm25": 1}


def test_rrf_does_not_use_raw_scores() -> None:
    # A huge BM25 score cannot dominate: only its rank counts.
    huge = Candidate(1, DOC_A, 1, 1, 9_999.0)
    tiny = Candidate(2, DOC_A, 2, 1, 0.01)
    fused = reciprocal_rank_fusion({"bm25": [huge], "dense": [tiny]}, k=60)
    assert fused[0].score == fused[1].score


def test_rrf_is_deterministic_regardless_of_input_order() -> None:
    lists = {
        "dense": [cand(5, 1), cand(3, 2), cand(9, 3)],
        "bm25": [cand(9, 1), cand(5, 2)],
        "section": [cand(3, 1)],
    }
    expected = [f.chunk_id for f in reciprocal_rank_fusion(lists, k=60)]
    for order in itertools.permutations(lists):
        shuffled = {name: list(reversed(lists[name])) for name in order}
        assert [f.chunk_id for f in reciprocal_rank_fusion(shuffled, k=60)] == expected


def test_rrf_ties_break_by_agreement_then_rank_then_position() -> None:
    # Chunks 7 and 4 have identical scores (rank 1 in one list each): earlier ordinal wins.
    fused = reciprocal_rank_fusion({"dense": [cand(7, 1)], "bm25": [cand(4, 1)]}, k=60)
    assert [f.chunk_id for f in fused] == [4, 7]
    # Across documents with equal scores, document id decides (stable, not arbitrary).
    fused = reciprocal_rank_fusion(
        {"dense": [cand(1, 1, DOC_B, ordinal=0), cand(2, 1, DOC_A, ordinal=5)]}, k=60
    )
    assert [f.document_id for f in fused] == [DOC_A, DOC_B]


def test_rrf_ignores_a_repeated_chunk_within_one_list() -> None:
    fused = reciprocal_rank_fusion({"dense": [cand(1, 1), cand(1, 3)]}, k=60)
    assert fused[0].score == pytest.approx(1 / 61)


def test_rrf_of_nothing_is_nothing() -> None:
    assert reciprocal_rank_fusion({"dense": [], "bm25": []}, k=60) == []


# --- near-duplicate removal ---------------------------------------------------------------------


def test_near_duplicates_within_a_document_are_removed() -> None:
    ranked = reciprocal_rank_fusion({"dense": [cand(1, 1), cand(2, 2), cand(3, 3)]}, k=60)
    texts = {
        1: "Pricing applies as set out in the Order Form.",
        2: "Pricing  applies as set out in the\nOrder Form.",  # same words, other whitespace
        3: "The Supplier shall deliver the goods.",
    }
    kept, removed = remove_near_duplicates(ranked, texts)
    assert [k.chunk_id for k in kept] == [1, 3] and removed == 1


def test_identical_text_in_different_documents_is_kept() -> None:
    ranked = reciprocal_rank_fusion({"dense": [cand(1, 1, DOC_A), cand(2, 1, DOC_B)]}, k=60)
    texts = {1: "Liability is capped at AED 100,000.", 2: "Liability is capped at AED 100,000."}
    kept, removed = remove_near_duplicates(ranked, texts)
    assert len(kept) == 2 and removed == 0


def test_similar_but_different_clauses_are_kept() -> None:
    ranked = reciprocal_rank_fusion({"dense": [cand(1, 1), cand(2, 2)]}, k=60)
    texts = {
        1: "Liability of the Supplier is capped at AED 100,000 per claim.",
        2: "Liability of the Customer is capped at AED 1,000,000 in aggregate.",
    }
    assert remove_near_duplicates(ranked, texts)[1] == 0


# --- Phase 4 review regressions -----------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "what is the 3.14 percent fee",
        "version 1.5 of the API",
        "rate of 2.5 million",
        "AED 1.5 million cap",
        "$2.5 bn",
        "within 1.5 years",
    ],
)
def test_quantities_and_versions_are_not_clause_references(query: str) -> None:
    assert analyze_query(query).section_refs == ()


@pytest.mark.parametrize(
    ("query", "refs"),
    [
        ("clause 2.5", ("2.5",)),
        ("What does 8.2 say?", ("8.2",)),
        ("see 12.3.1 and 8.2", ("12.3.1", "8.2")),
        ("clause 2.5 about the 2.5 million cap", ("2.5",)),
    ],
)
def test_genuine_clause_references_are_kept(query: str, refs: tuple[str, ...]) -> None:
    assert analyze_query(query).section_refs == refs


def test_number_of_exact_phrases_is_capped() -> None:
    analysis = analyze_query(" ".join(f'"term{i} word{i}"' for i in range(8)))
    assert len(analysis.phrases) == 5
    assert analysis.notes == ["Only the first 5 quoted phrases were matched exactly."]
