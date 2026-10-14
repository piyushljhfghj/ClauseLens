"""Test doubles."""

import hashlib
from collections.abc import Sequence

import numpy as np

from app.embeddings import EMBEDDING_DIMENSION, EmbedProgress


class FakeEmbedder:
    """Deterministic embeddings from a hash of the text: fast, no model download. Identical texts
    get identical vectors; vectors are unit length like the real model's."""

    model_name = "fake-embedder"
    dimension = EMBEDDING_DIMENSION
    loaded = True

    def __init__(self) -> None:
        self.calls = 0
        self.fail_with: Exception | None = None

    def _vector(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
        vector = np.random.default_rng(seed).standard_normal(self.dimension).astype(np.float32)
        return vector / np.linalg.norm(vector)

    def embed_passages(
        self, texts: Sequence[str], on_progress: EmbedProgress | None = None
    ) -> np.ndarray:
        self.calls += 1
        if self.fail_with is not None:
            raise self.fail_with
        if on_progress:
            on_progress(len(texts), len(texts))
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.stack([self._vector(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


# Synonym groups standing in for meaning: texts sharing a concept get similar vectors, texts
# sharing none are near-orthogonal. Deterministic, so paraphrase retrieval can be tested exactly.
CONCEPTS: tuple[frozenset[str], ...] = (
    frozenset({"terminate", "terminated", "termination", "end", "ended", "ending", "cancel",
               "cancelled", "cancellation", "early", "exit"}),
    frozenset({"liability", "liable", "damages", "cap", "capped", "exceed", "losses", "owe"}),
    frozenset({"law", "laws", "governed", "governing", "jurisdiction", "country", "courts"}),
    frozenset({"confidential", "confidentiality", "secret", "secrets", "disclose", "share"}),
    frozenset({"payment", "payable", "pay", "invoice", "invoices", "bills", "fees", "fee"}),
)  # fmt: skip
_CONCEPT_WEIGHT = 1.0
_WORD_WEIGHT = 0.08  # every other word adds a little hashed noise, so no vector is all-zero


class ConceptEmbedder(FakeEmbedder):
    model_name = "concept-embedder"

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for word in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split():
            hit = False
            for index, concept in enumerate(CONCEPTS):
                if word in concept:
                    vector[index] += _CONCEPT_WEIGHT
                    hit = True
            if not hit:
                slot = len(CONCEPTS) + int(hashlib.sha256(word.encode()).hexdigest(), 16) % (
                    self.dimension - len(CONCEPTS)
                )
                vector[slot] += _WORD_WEIGHT
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector
