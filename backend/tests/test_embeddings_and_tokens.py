"""Tokenizer and the real embedding model."""

import numpy as np
import pytest

from app.embeddings import FastEmbedEmbedder
from app.text.tokenize import lexical_tokens, prepare_for_indexing


def test_lexical_tokens() -> None:
    tokens = lexical_tokens("Clause 8.2: liability capped at AED 100,000 (the “Cap”).")
    assert tokens == ["clause", "8.2", "liability", "capped", "at", "aed", "100000", "the", "cap"]


def test_index_text_joins_line_break_hyphenation_and_ligatures() -> None:
    assert prepare_for_indexing("termi-\nnation of the ﬁrst Agreement") == (
        "termination of the first Agreement"
    )
    assert lexical_tokens("termi-\nnation") == ["termination"]


def test_unsupported_model_is_rejected() -> None:
    with pytest.raises(ValueError):
        FastEmbedEmbedder("some/other-model", cache_dir=".cache/fastembed")


@pytest.fixture(scope="module")
def real_embedder() -> FastEmbedEmbedder:
    embedder = FastEmbedEmbedder("BAAI/bge-small-en-v1.5", cache_dir=".cache/fastembed")
    try:
        embedder.embed_passages(["warm up"])
    except Exception as exc:  # no network on first run, etc.
        pytest.skip(f"embedding model unavailable: {exc}")
    return embedder


def test_real_model_embeddings(real_embedder: FastEmbedEmbedder) -> None:
    progress: list[tuple[int, int]] = []
    vectors = real_embedder.embed_passages(
        [
            "The Company may terminate this Agreement on thirty days written notice.",
            "Either party can end the contract by giving 30 days notice.",
            "This Agreement is governed by the laws of the United Arab Emirates.",
        ],
        on_progress=lambda done, total: progress.append((done, total)),
    )
    assert vectors.shape == (3, 384)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
    assert vectors[0] @ vectors[1] > vectors[0] @ vectors[2]  # paraphrase beats unrelated clause
    assert progress[-1] == (3, 3)

    query = real_embedder.embed_query("notice period for termination")
    assert query.shape == (384,)
    assert query @ vectors[0] > query @ vectors[2]


def test_real_model_handles_empty_input(real_embedder: FastEmbedEmbedder) -> None:
    assert real_embedder.embed_passages([]).shape == (0, 384)
