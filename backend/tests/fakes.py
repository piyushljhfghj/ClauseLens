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
